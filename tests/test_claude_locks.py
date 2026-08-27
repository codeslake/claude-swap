"""Tests for the proper-lockfile-compatible Claude Code lock helpers."""

from __future__ import annotations

import errno
import logging
import os
import threading
import time
from pathlib import Path

import pytest

from claude_swap import claude_locks
from claude_swap.claude_locks import (
    claude_config_lock,
    claude_credentials_lock,
    config_lock_dir,
    credentials_lock_dir,
    proper_lockfile,
)
from claude_swap.exceptions import ClaudeCodeLockTimeout


@pytest.fixture
def lock_dir(tmp_path: Path) -> Path:
    return tmp_path / "target.lock"


class TestProperLockfile:
    def test_acquire_creates_and_release_removes(self, lock_dir):
        with proper_lockfile(lock_dir):
            assert lock_dir.is_dir()
        assert not lock_dir.exists()

    def test_reacquire_after_release(self, lock_dir):
        with proper_lockfile(lock_dir):
            pass
        with proper_lockfile(lock_dir):
            assert lock_dir.is_dir()

    def test_contention_times_out(self, lock_dir):
        lock_dir.mkdir()  # fresh mtime = live holder
        start = time.monotonic()
        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(lock_dir, timeout=0.5):
                pass
        assert time.monotonic() - start < 5.0
        assert lock_dir.is_dir()  # the holder's lock is left alone

    def test_stale_lock_is_taken_over(self, lock_dir):
        lock_dir.mkdir()
        past = time.time() - 30
        os.utime(lock_dir, (past, past))
        with proper_lockfile(lock_dir, timeout=2.0):
            assert lock_dir.is_dir()
            # We own it now: mtime is fresh, not the 30s-old corpse.
            assert time.time() - lock_dir.stat().st_mtime < 5.0
        assert not lock_dir.exists()

    def test_release_tolerates_stolen_lock(self, lock_dir):
        with proper_lockfile(lock_dir):
            os.rmdir(lock_dir)  # simulate a stale-takeover by another process
        # No exception; nothing left behind.
        assert not lock_dir.exists()

    def test_toucher_keeps_mtime_fresh(self, lock_dir, monkeypatch):
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.1)
        with proper_lockfile(lock_dir):
            past = time.time() - 30
            os.utime(lock_dir, (past, past))
            time.sleep(0.4)
            assert time.time() - lock_dir.stat().st_mtime < 10.0

    def _count_touches(self, monkeypatch, lock_dir, fail_first=None, fail_all=None):
        """Patch os.utime and count only the calls aimed at OUR lock.

        Keyed on the path, not on a global call counter: the module does
        `import os`, so patching `claude_locks.os.utime` patches the global
        one and any other caller in the process consumes an injected failure.
        """
        real = os.utime
        state = {"n": 0}

        def counting(path, *a, **k):
            if not isinstance(path, int) and os.fspath(path) == os.fspath(lock_dir):
                state["n"] += 1
                if fail_all is not None:
                    raise fail_all
                if fail_first is not None and state["n"] == 1:
                    raise fail_first
            return real(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "utime", counting)
        return state, real

    def test_a_transient_utime_failure_does_not_disarm_the_heartbeat(
            self, lock_dir, monkeypatch):
        """One failed touch must not stop the touching.

        The toucher returned on any OSError, so a single transient failure
        killed the heartbeat while the lock was still held — and after
        CONFIG_STALENESS_S any waiter legally steals it.
        """
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.05)
        touches, real = self._count_touches(
            monkeypatch, lock_dir, fail_first=OSError("transient"))
        with proper_lockfile(lock_dir):
            past = time.time() - 30
            real(lock_dir, (past, past))
            time.sleep(0.4)
            assert touches["n"] > 1, "the toucher stopped after one failure"
            assert time.time() - lock_dir.stat().st_mtime < 10.0, (
                "the lock went stale while still held, so a waiter may steal it"
            )

    def test_a_transient_failure_does_not_claim_the_lock_is_about_to_be_stolen(
            self, lock_dir, monkeypatch, caplog):
        """The sentence is only true of a failure that outlives the window.

        "its mtime stops advancing, so a waiter may take it over as stale" is
        the warning's own justification, and the arm fired on the FIRST
        failure regardless — so a single hiccup that clears on the next tick
        printed it while the mtime went on advancing and nothing was ever at
        risk. Measured before this: 7 touches, a lock fresh throughout, one
        warning saying the opposite.

        Its sibling covers the persistent case with `fail_all`, so nothing
        stood between the two.
        """
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        touches, real = self._count_touches(
            monkeypatch, lock_dir, fail_first=OSError("transient"))
        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            with proper_lockfile(lock_dir):
                # REWOUND FIRST, like the sibling. Without it the mtime only
                # ever moves FORWARD from the mkdir, so `age` is bounded by
                # the hold (0.3s) whatever the toucher does and the premise
                # below cannot fail. Measured: with a toucher that calls
                # `utime` but never advances the mtime, both siblings go red
                # and this one stayed green.
                past = time.time() - 30
                real(lock_dir, (past, past))
                time.sleep(0.3)
                # INSIDE THE HOLD: the release removes the directory, so the
                # freshness this is about is unobservable afterwards.
                age = time.time() - lock_dir.stat().st_mtime

        assert touches["n"] > 2, "control: the toucher must have kept trying"
        assert age < 5.0, (
            "premise: the mtime kept advancing, so a theft warning here would "
            "have been FALSE -- if it had frozen, the silence below is correct "
            "for the wrong reason"
        )
        said = [r.getMessage() for r in caplog.records if "refresh" in r.getMessage()]
        assert said == [], (
            f"a hiccup that cleared was reported as imminent theft: {said}"
        )

    def test_a_hiccup_after_a_long_healthy_hold_is_still_silent(
            self, lock_dir, monkeypatch, caplog):
        """`last_ok` is what makes the persistence gate mean anything, and
        deleting that one line left the whole suite green.

        Every other case here holds the lock for LESS than `staleness`, so
        `time.time() - last_ok > staleness` is false whether or not `last_ok`
        ever advances -- verified where it cannot fail. This one holds it
        WELL past the window with every touch succeeding, then fails one:
        with `last_ok` advancing that is a hiccup and says nothing; without
        it, the gate compares against acquisition time and the cries-wolf
        warning comes straight back.
        """
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        real = os.utime
        state = {"n": 0}

        def fail_the_tenth(path, *a, **k):
            if os.fspath(path) == os.fspath(lock_dir):
                state["n"] += 1
                if state["n"] == 10:
                    raise PermissionError("injected: one hiccup")
            return real(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "utime", fail_the_tenth)
        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            # Below the hold, so the gate is genuinely reached -- but not so
            # close to TOUCH_INTERVAL_S that one scheduler stall reads as a
            # frozen `last_ok`. At 0.05 a single 30ms hiccup between two ticks
            # cries wolf on correct code; 0.15 leaves 130ms and still catches
            # the `last_ok` deletion, because the hold is 0.4s either way.
            with proper_lockfile(lock_dir, staleness=0.15):
                time.sleep(0.4)

        assert state["n"] > 10, (
            f"premise: only {state['n']} touch(es) ran, so the hiccup never "
            "landed inside a hold longer than the staleness window"
        )
        said = [r.getMessage() for r in caplog.records if "refresh" in r.getMessage()]
        assert said == [], (
            f"one hiccup after a long healthy hold was reported as imminent "
            f"theft: {said}"
        )

    def test_a_persistent_touch_failure_is_reported_once(
            self, lock_dir, monkeypatch, caplog):
        """A failure that never clears leaves the takeover unexplainable.

        Staying armed is right — the errno in hand is what separates transient
        from gone — but a failure that outlives the staleness window stops the
        mtime advancing, and a waiter then legitimately steals a lock that is
        still held, with nothing anywhere recording why. One warning costs a
        bool; one per attempt would bury it.

        OUTLIVING THE WINDOW is what the warning claims, so the window is
        shortened here rather than waiting `CONFIG_STALENESS_S` for it. With
        the default the failure never persists long enough and this case
        stopped discriminating the moment the arm learned to ask.
        """
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        touches, _ = self._count_touches(
            monkeypatch, lock_dir, fail_all=PermissionError("injected"))
        with caplog.at_level(logging.WARNING, logger="claude-swap"):
            with proper_lockfile(lock_dir, staleness=0.05):
                time.sleep(0.3)

        assert touches["n"] > 1, "control: the toucher must have kept trying"
        said = [r for r in caplog.records if "refresh" in r.getMessage()]
        assert len(said) == 1, f"expected exactly one warning, got {len(said)}"

    @pytest.mark.parametrize("errno_", [errno.ESTALE, errno.EACCES, errno.EIO])
    def test_a_failure_that_is_not_absence_keeps_the_heartbeat(
            self, lock_dir, monkeypatch, errno_):
        """Only absence may stop the toucher; every other errno is transient.

        `os.stat` fails too, because that is what these errnos mean on a real
        filesystem — a stale NFS handle or an unreadable directory does not
        answer `stat` either. Injecting only a `utime` failure leaves the
        directory genuinely present, so any implementation that asks the
        filesystem gets a truthful "still there" and the test passes without
        exercising anything.

        With `stat` failing too, asking `lock_dir.exists()` is wrong on every
        supported Python and for two different reasons: 3.12 re-raises these
        errnos out of the thread as does 3.13, 3.14+ swallows them and answers False, which
        reads as "gone" for a lock we still hold. requires-python is >=3.12
        and CI runs 3.12, so neither half can be dismissed as theoretical.
        """
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.05)
        real_utime, real_stat = os.utime, os.stat
        touches = {"n": 0}
        broken = {"on": False}

        def failing_utime(path, *a, **k):
            if not isinstance(path, int) and os.fspath(path) == os.fspath(lock_dir):
                touches["n"] += 1
                if not broken["on"]:
                    broken["on"] = True
                    raise OSError(errno_, "not absence")
            return real_utime(path, *a, **k)

        def failing_stat(path, *a, **k):
            if (broken["on"] and not isinstance(path, int)
                    and os.fspath(path) == os.fspath(lock_dir)):
                raise OSError(errno_, "not absence")
            return real_stat(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "utime", failing_utime)
        monkeypatch.setattr(claude_locks.os, "stat", failing_stat)
        with proper_lockfile(lock_dir):
            time.sleep(0.4)
            settled = touches["n"]
        assert broken["on"], (
            "the injected failure never fired — the instrument, not the code"
        )
        assert settled > 1, (
            f"errno {errno_} stopped the toucher after {settled} touch(es); "
            "the lock is still held and only absence may stop it"
        )

    def test_a_vanished_lock_still_stops_the_toucher(self, lock_dir, monkeypatch):
        """The control. Retrying forever on a lock that is GONE would be the
        opposite defect, and the original `return` existed for this case.

        Asserted on the TOUCHER, not on the directory: `os.utime` cannot
        create one, so `assert not lock_dir.exists()` passes for a toucher
        that never stops at all.
        """
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.05)
        # COUNTED ON THE TICK, not on `os.utime`. WHERE a tick notices absence
        # is an implementation choice -- the leading stat is one syscall
        # earlier than the refresh -- and a utime counter reads zero for a
        # toucher that ran and correctly stopped, failing its own premise
        # about code that is right. What this case is about is that the LOOP
        # ends, so count the loop.
        # BOTH SYSCALLS, because either one can be the tick's first. A
        # heartbeat that verifies identity before refreshing notices absence
        # at the STAT and never reaches `utime`; one that refreshes straight
        # away notices it at the utime. Counting only the second reads zero
        # for a toucher that ran and stopped correctly.
        real = {"stat": os.stat, "utime": os.utime}
        ticks = {"n": 0}

        def counting(name):
            def call(path, *a, **k):
                if (not isinstance(path, int)
                        and os.fspath(path) == os.fspath(lock_dir)):
                    ticks["n"] += 1
                return real[name](path, *a, **k)
            return call

        with proper_lockfile(lock_dir):
            # Armed INSIDE the hold, so the acquire's own calls are not ticks.
            monkeypatch.setattr(claude_locks.os, "stat", counting("stat"))
            monkeypatch.setattr(claude_locks.os, "utime", counting("utime"))
            os.rmdir(lock_dir)
            time.sleep(0.15)
            settled = ticks["n"]
            assert settled >= 1, (
                "the toucher never ran in the window — the instrument, not "
                "the code (raise the sleep or lower TOUCH_INTERVAL_S)"
            )
            time.sleep(0.3)
            assert ticks["n"] == settled, (
                f"the toucher kept going on a dead lock "
                f"({ticks['n'] - settled} more attempts)"
            )

    def test_creates_missing_parent(self, tmp_path):
        nested = tmp_path / "a" / "b" / "target.lock"
        with proper_lockfile(nested):
            assert nested.is_dir()

    def test_a_small_timeout_is_not_overshot_by_the_retry_sleep(self, lock_dir):
        """`timeout` must bound the call, sleeps included.

        The unclamped retry sleeps a full jittered 0.25-0.5s whatever the
        budget, so a sub-sleep timeout never times out anywhere near when it
        says.
        """
        lock_dir.mkdir()  # fresh mtime -> contended, not stale
        start = time.monotonic()
        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(lock_dir, timeout=0.01):
                pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.15, (
            f"a 0.01s timeout overshot to {elapsed:.3f}s — the retry sleep "
            "ignored the remaining budget"
        )

    def test_the_rmdir_branch_sleeps_only_what_is_left(
        self, lock_dir, monkeypatch
    ):
        """A SCRIPTED CLOCK: the remaining budget at each sleep is CHOSEN.

        The two ~90-line cases this replaces raced for it. They anchored on
        the code's own `monotonic` and then asserted the run had ENTERED the
        region where a flat sleep is observable -- and `min(lefts)` is
        `budget mod flat`, so losing one loop iteration to latency raises it
        by a whole `flat` and the assertion can never be satisfied. Measured
        on pristine source with per-iteration latency injected: FAIL at 7ms,
        8ms, 25ms, 30ms, 80ms; PASS at 0, 6, 10, 12, 40. Under ordinary
        fair-share contention, 3 of 60 runs red.
        
        Deleting that assertion is not the fix either: without it a flat 0.05
        SURVIVES at 25ms of latency. The case was caught between passing the
        defect and failing on correct code. With the clock scripted there is
        no race to lose: every sleep is measured against a remainder this
        test decided, and the flat 0.005 the body called uncatchable dies too.
        """
        lock_dir.mkdir()
        os.utime(lock_dir, (0, 0))                    # ancient -> stale
        budget, clock, slept = 0.175, [0.0], []
        real_rmdir = os.rmdir

        def refuse(path, *a, **k):
            if os.fspath(path) == os.fspath(lock_dir):
                clock[0] += 0.001                     # one iteration of work
                raise OSError(errno.EACCES, "cannot remove")
            return real_rmdir(path, *a, **k)

        mine = threading.get_ident()
        real_sleep = claude_locks.time.sleep

        def fake_sleep(seconds):
            # SCOPED TO THIS THREAD, or a leaked one spins through a sleep
            # that never sleeps and writes its own calls into this budget --
            # red on correct code, measured.
            if threading.get_ident() != mine:
                return real_sleep(seconds)
            slept.append((round(budget - clock[0], 3), round(seconds, 3)))
            clock[0] += seconds

        monkeypatch.setattr(claude_locks.os, "rmdir", refuse)
        monkeypatch.setattr(claude_locks.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(claude_locks.time, "sleep", fake_sleep)

        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(lock_dir, timeout=budget):
                pass

        assert len(slept) >= 3, f"the instrument, not the code: {slept}"
        for left, seconds in slept:
            assert seconds <= max(left, 0.0), (
                f"slept {seconds}s with {left}s left — the clamp used "
                f"`timeout`, not what remains of it (all sleeps: {slept})"
            )
        assert min(l for l, _ in slept) < 0.05, (
            f"the run must reach a remainder under the flat 0.05: {slept}"
        )

    def test_the_jitter_branch_sleeps_only_what_is_left(
        self, lock_dir, monkeypatch
    ):
        """THE SITE THIS PR IS NAMED AFTER, and it had no flat-sleep guard.

        Mutating only the jitter clamp to a flat sleep: 0.05 passed, 0.10
        passed, 0.14 passed — a 14x overshoot of the 0.01s timeout the PR's
        own opening table calls the bug — and only 0.16 failed. Every earlier
        flat-sleep row was measured with both sites mutated together, so all
        the signal came from the rmdir site.

        A HELD, FRESH lock takes this branch: the staleness test reads
        `time.time()`, which is left alone, and only `monotonic` is scripted.
        """
        lock_dir.mkdir()                              # held, and fresh
        # THE JITTER IS SCRIPTED TOO. It is randomness in the CODE, not in the
        # clock, so leaving it live makes the tail remainder vary per run and
        # the region assert below flaky -- measured 27 of 60 red before this.
        # Pinned to 0, the draw is a flat 0.25 and the budget is chosen so
        # `budget mod (0.25 + 0.001)` is 0.003: small enough that any flat
        # sleep overshoots it.
        monkeypatch.setattr(claude_locks.random, "random", lambda: 0.0)
        budget, clock, slept = 0.756, [0.0], []

        mine = threading.get_ident()
        real_sleep = claude_locks.time.sleep

        def fake_sleep(seconds):
            # SCOPED, like its siblings: this patch is process-global, so an
            # unscoped recorder writes every other thread's sleeps into this
            # budget -- red on correct code.
            if threading.get_ident() != mine:
                return real_sleep(seconds)
            slept.append((round(budget - clock[0], 3), round(seconds, 3)))
            # ONE ITERATION OF WORK on top of the sleep. Nothing else advances
            # a scripted clock, so a clamped sleep of 0.0 would leave the
            # deadline check reading the same instant for ever.
            clock[0] += seconds + 0.001

        monkeypatch.setattr(claude_locks.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(claude_locks.time, "sleep", fake_sleep)

        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(lock_dir, timeout=budget):
                pass

        assert len(slept) >= 2, f"the instrument, not the code: {slept}"
        for left, seconds in slept:
            assert seconds <= max(left, 0.0), (
                f"slept {seconds}s with {left}s left — the jitter clamp used "
                f"`timeout`, not what remains of it (all sleeps: {slept})"
            )
        assert min(l for l, _ in slept) < 0.005, (
            f"the run must reach a remainder small enough for any flat sleep "
            f"to overshoot it: {slept}"
        )


class TestLockPaths:
    def test_default_paths(self, temp_home, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        assert credentials_lock_dir() == temp_home / ".claude.lock"
        assert config_lock_dir() == temp_home / ".claude.json.lock"

    def test_claude_config_dir_is_honored(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom-claude"
        custom.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        assert credentials_lock_dir() == tmp_path / "custom-claude.lock"
        # ~/.claude.json resolves relative to CLAUDE_CONFIG_DIR too.
        assert config_lock_dir() == custom / ".claude.json.lock"

    def test_named_helpers_lock_their_dirs(self, temp_home, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        with claude_credentials_lock():
            assert (temp_home / ".claude.lock").is_dir()
            with claude_config_lock():
                assert (temp_home / ".claude.json.lock").is_dir()
        assert not (temp_home / ".claude.lock").exists()
        assert not (temp_home / ".claude.json.lock").exists()


class TestCcRefreshLockProtocol:
    """Claude Code 2.1.218 guards its OAuth refresh with TWO locks —
    ``<config-home>/.oauth_refresh.lock`` (primary) then the legacy
    ``<config-home>.lock`` — both at a 60s staleness. cswap must follow the
    same protocol or mutual exclusion silently fails (extracted from the
    2.1.218 bundle: ``uKi``/``CKi``, ``stale: 60000, update: 5000``)."""

    def test_oauth_refresh_lock_dir_default(self, temp_home, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        assert (
            claude_locks.oauth_refresh_lock_dir()
            == temp_home / ".claude" / ".oauth_refresh.lock"
        )

    def test_oauth_refresh_lock_dir_honors_claude_config_dir(
        self, tmp_path, monkeypatch
    ):
        custom = tmp_path / "custom-claude"
        custom.mkdir()
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(custom))
        assert claude_locks.oauth_refresh_lock_dir() == custom / ".oauth_refresh.lock"

    def test_credentials_lock_takes_both_locks(self, temp_home, monkeypatch):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        new = temp_home / ".claude" / ".oauth_refresh.lock"
        legacy = temp_home / ".claude.lock"
        with claude_credentials_lock():
            assert new.is_dir(), "primary .oauth_refresh.lock not held"
            assert legacy.is_dir(), "legacy .claude.lock not held"
        assert not new.exists()
        assert not legacy.exists()

    def test_primary_contention_never_touches_legacy(self, temp_home, monkeypatch):
        """CC's order: primary first. If the primary is held we must time out
        without ever creating the legacy lock."""
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        new = temp_home / ".claude" / ".oauth_refresh.lock"
        new.mkdir(parents=True)  # fresh mtime = live CC holding its refresh lock
        with pytest.raises(ClaudeCodeLockTimeout):
            with claude_credentials_lock(timeout=0.5):
                pass
        assert not (temp_home / ".claude.lock").exists()
        assert new.is_dir()  # holder's lock untouched

    def test_legacy_contention_releases_primary(self, temp_home, monkeypatch):
        """If the legacy lock is contended after the primary was acquired,
        the primary must not be left behind (CC releases its new lock on
        legacy ELOCKED)."""
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        legacy = temp_home / ".claude.lock"
        legacy.mkdir()  # fresh = held
        with pytest.raises(ClaudeCodeLockTimeout):
            with claude_credentials_lock(timeout=0.5):
                pass
        assert not (temp_home / ".claude" / ".oauth_refresh.lock").exists()
        assert legacy.is_dir()

    def test_credentials_staleness_is_60s_not_10s(self, temp_home, monkeypatch):
        """A 30s-old credential lock belongs to a live CC (its budget is 60s)
        and must NOT be stolen — the old 10s staleness stole it."""
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        new = temp_home / ".claude" / ".oauth_refresh.lock"
        new.mkdir(parents=True)
        past = time.time() - 30
        os.utime(new, (past, past))
        with pytest.raises(ClaudeCodeLockTimeout):
            with claude_credentials_lock(timeout=0.5):
                pass
        assert new.is_dir()

    def test_credentials_lock_stale_past_60s_is_taken_over(
        self, temp_home, monkeypatch
    ):
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        new = temp_home / ".claude" / ".oauth_refresh.lock"
        legacy = temp_home / ".claude.lock"
        new.mkdir(parents=True)
        legacy.mkdir()
        past = time.time() - 70
        os.utime(new, (past, past))
        os.utime(legacy, (past, past))
        with claude_credentials_lock(timeout=2.0):
            assert new.is_dir()
            assert legacy.is_dir()
        assert not new.exists()
        assert not legacy.exists()

    def test_config_lock_staleness_stays_10s(self, temp_home, monkeypatch):
        """The config lock's CC-side defaults are unchanged — a 30s-old
        config lock is still stale and taken over."""
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
        cfg = temp_home / ".claude.json.lock"
        cfg.mkdir()
        past = time.time() - 30
        os.utime(cfg, (past, past))
        with claude_config_lock(timeout=2.0):
            assert cfg.is_dir()
        assert not cfg.exists()


class TestTheClampsSurviveWeakeningNotOnlyDeletion:
    """`min(sleep, timeout)` is a no-op once most of the budget is spent.

    The two cases that guard these clamps both use a timeout SMALLER than the
    flat sleep (0.01 and 0.001), where `min(sleep, timeout)` and
    `min(sleep, remaining)` are indistinguishable. Measured: weakening both
    clamps to the whole timeout leaves the suite at 36/36 while the acquire
    path overshoots by up to 0.5s, and at the production default the mutation
    is a total no-op -- `proper_lockfile` is unbounded again.

    The jitter is pinned, or the weakened form's overshoot is a random draw
    that can land inside any fixed margin.
    """

    def test_a_timeout_above_the_sleep_still_bounds_the_call(
        self, lock_dir, monkeypatch
    ):
        lock_dir.mkdir()  # fresh mtime -> contended, not stale
        monkeypatch.setattr(claude_locks.random, "random", lambda: 1.0)
        start = time.monotonic()
        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(lock_dir, timeout=0.6):
                pass
        elapsed = time.monotonic() - start
        # Correct: 0.5 then 0.1 -> 0.6. Weakened: 0.5 twice -> 1.0.
        assert elapsed < 0.8, (
            f"a 0.6s budget took {elapsed:.2f}s — the retry sleep clamped to "
            "`timeout` rather than to what is left of it"
        )

class TestEveryArmOfTheLoopBacksOff:
    """The bounding pass clamped the two sleeping arms and skipped one.

    A lock PATH that is a dangling symlink answers `FileExistsError` to
    `mkdir` and `FileNotFoundError` to `stat`, so the retry took the one arm
    with no sleep in it, for the whole budget. Measured before this: 109,000
    mkdir attempts per second -- four times the spin this branch was written
    to bound, and reached with no race at all.
    """

    def test_a_dangling_symlink_does_not_spin(self, tmp_path, monkeypatch):
        target = tmp_path / "target.lock"
        target.symlink_to(tmp_path / "nothing-here")
        assert not target.exists(), "premise: the symlink must dangle"

        real_mkdir = os.mkdir
        tries = {"n": 0}

        def counting(path, *a, **k):
            if os.fspath(path) == os.fspath(target):
                tries["n"] += 1
            return real_mkdir(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "mkdir", counting)
        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(target, timeout=0.3):
                pass

        assert tries["n"] > 1, "premise: the loop must have retried at all"
        assert tries["n"] < 40, (
            f"{tries['n']} mkdir attempts in a 0.3s budget — the arm that "
            "retries a vanished name never sleeps, so it pins a core"
        )

    def test_the_dangling_symlink_arm_sleeps_only_what_is_left(
        self, tmp_path, monkeypatch
    ):
        """A SCRIPTED CLOCK, like its two siblings.

        NAMED FOR THE ARM IT REACHES. A dangling symlink answers
        FileExistsError to `mkdir`, so this drives the arm where the READ-BACK
        stat raises ENOENT -- not a name swept between `mkdir` and `stat`.
        The two are one `except` on this branch and two on the merged tree,
        where the swept-name arm has its own back-off that this case never
        touches.

        The wall-clock form this replaces could only see a FLATTENING. With a
        budget under the flat constant, `min(0.05, timeout)` and
        `min(0.05, remaining)` are the same number -- which is exactly the
        weakening this branch's clamps exist against. Measured on that form:
        clamping to `timeout` instead of the remainder left the suite at 43
        passed, and its lower bound could not fail at all, because the raise
        it waits for is itself gated on the deadline having passed.
        """
        target = tmp_path / "target.lock"
        target.symlink_to(tmp_path / "nothing-here")
        assert not target.exists(), "premise: the symlink must dangle"

        budget, clock, slept = 0.175, [0.0], []
        real_mkdir = os.mkdir
        mine = threading.get_ident()
        real_sleep = claude_locks.time.sleep

        def counting(path, *a, **k):
            if os.fspath(path) == os.fspath(target):
                clock[0] += 0.001                     # one iteration of work
            return real_mkdir(path, *a, **k)

        def fake_sleep(seconds):
            # SCOPED TO THIS THREAD. `claude_locks.time` IS the `time` module,
            # so an unscoped patch freezes the clock process-wide and records
            # every other thread's sleeps into this budget.
            if threading.get_ident() != mine:
                return real_sleep(seconds)
            slept.append((round(budget - clock[0], 3), round(seconds, 3)))
            clock[0] += seconds

        monkeypatch.setattr(claude_locks.os, "mkdir", counting)
        monkeypatch.setattr(claude_locks.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(claude_locks.time, "sleep", fake_sleep)

        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(target, timeout=budget):
                pass

        assert len(slept) >= 3, f"the instrument, not the code: {slept}"
        for left, seconds in slept:
            assert seconds <= max(left, 0.0), (
                f"slept {seconds}s with {left}s left — the clamp used "
                f"`timeout`, not what remains of it (all sleeps: {slept})"
            )
        assert min(l for l, _ in slept) < 0.05, (
            f"the run must reach a remainder under the flat 0.05: {slept}"
        )
