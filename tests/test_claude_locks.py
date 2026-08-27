"""Tests for the proper-lockfile-compatible Claude Code lock helpers."""

from __future__ import annotations

import errno
import logging
import os
import shutil
import threading
import time
from pathlib import Path

import pytest
from unittest.mock import patch

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

    def test_a_lock_swept_between_mkdir_and_stat_stays_bounded(
        self, lock_dir, monkeypatch
    ):
        """The stamp read-back has to sit inside the retry loop.

        A waiter that judged this lock stale does `stat` then `rmdir`; the
        holder can release and we can take the name in that gap, and its rmdir
        then removes OUR fresh directory. A read-back outside the loop raises
        `FileNotFoundError` out of a function documented to raise only
        `ClaudeCodeLockTimeout`.

        Falling through to the deadline rather than restarting the loop above
        it: a name swept on every attempt must still end at the budget, not
        spin until the sweeper stops.
        """
        real_mkdir = os.mkdir
        started = time.monotonic()

        # THE BOUND IS ASSERTED WHERE IT IS REACHABLE. Both assertions below
        # run only after `proper_lockfile` returns, so a loop that never exits
        # -- `continue` instead of falling through to the deadline -- HANGS
        # rather than failing: measured exit 124 with no test name printed,
        # and under `-n auto` one spinning worker takes the whole run with it.
        # Nothing catches that: `pytest-timeout` is not a dependency and
        # neither CI job that runs this file sets `timeout-minutes`.
        def swept(path, *args, **kwargs):
            # FILTERED. Unfiltered this rmdir'd every directory anything in
            # the process created while it ran; `vanished` and `slow_utime`
            # in this file both scope to the lock and this one did not.
            if os.fspath(path) != os.fspath(lock_dir):
                return real_mkdir(path, *args, **kwargs)
            assert time.monotonic() - started < 1.5, (
                "the loop never reached its deadline"
            )
            real_mkdir(path, *args, **kwargs)
            os.rmdir(path)

        monkeypatch.setattr(claude_locks.os, "mkdir", swept)
        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(lock_dir, timeout=0.3):
                pass
        elapsed = time.monotonic() - started
        assert elapsed < 1.5, f"a 0.3s budget took {elapsed:.2f}s"

    def test_a_swept_name_does_not_pin_a_core_for_the_whole_budget(
        self, lock_dir, monkeypatch
    ):
        """Ending at the deadline is not the same as waiting for it.

        The swept arm falls through to the deadline, then stats a name that is
        gone, then `continue`s -- so it never reaches the jittered sleep at the
        bottom of the loop and retries as fast as the kernel will take it.
        Measured on the branch before this: 25,184 mkdir attempts in 1.0s at
        99% of one core, and `claude_credentials_lock` takes two locks, so a
        9s default is ~18s of a pinned core. The sibling case above times the
        budget and cannot see it: a hot spin and a patient wait both end at the
        deadline.
        """
        real_mkdir = os.mkdir
        attempts: list[int] = []
        started = time.monotonic()

        def swept(path, *args, **kwargs):
            if os.fspath(path) != os.fspath(lock_dir):
                return real_mkdir(path, *args, **kwargs)
            # THE BOUND ASSERTED WHERE IT IS REACHABLE, like the sibling case.
            # Both assertions below run only after `proper_lockfile` returns,
            # so a loop that never exits -- `continue` instead of falling
            # through to the deadline -- HANGS rather than failing, and under
            # `-n auto` one spinning worker takes the whole run with it.
            # Measured on that mutant: this case rc=124 at the 90s wrapper
            # timeout with no output, the sibling `rc=1` naming the failure.
            assert time.monotonic() - started < 1.5, (
                "the loop never reached its deadline"
            )
            attempts.append(1)
            real_mkdir(path, *args, **kwargs)
            os.rmdir(path)

        monkeypatch.setattr(claude_locks.os, "mkdir", swept)
        with pytest.raises(ClaudeCodeLockTimeout):
            with proper_lockfile(lock_dir, timeout=0.3):
                pass
        assert attempts, "premise: the swept arm was never entered"
        # A back-off of any sane size keeps a 0.3s budget in single digits;
        # the unbounded spin was four orders of magnitude above this.
        assert len(attempts) <= 30, (
            f"{len(attempts)} attempts in a 0.3s budget — the swept arm is "
            "spinning, not waiting"
        )

    def test_one_transient_stat_error_does_not_end_the_heartbeat(
        self, lock_dir, monkeypatch
    ):
        """Absence is terminal; every other errno is transient.

        One `except OSError: return` over the tick's three syscalls meant a
        single EIO or ESTALE -- the ordinary errnos on a network `~/.claude` --
        froze the mtime for the rest of the hold, and a lock whose mtime stops
        advancing is one a waiter may take over as stale, mid-swap.
        """
        import errno as _errno

        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        real_stat = os.stat
        # ARMED ONLY ONCE THE LOCK IS HELD. `os.stat` is patched on the module,
        # and the ACQUIRE path stats too (its staleness check) -- an injection
        # armed from the start fires there, where no handler covers a bare
        # OSError, and the case then measures the acquire instead of the tick.
        state = {"armed": False, "fired": 0}

        def one_bad_stat(path, *a, **kw):
            if (state["armed"] and not state["fired"]
                    and os.fspath(path) == os.fspath(lock_dir)):
                state["fired"] = 1
                raise OSError(_errno.EIO, "injected: one transient read")
            return real_stat(path, *a, **kw)

        monkeypatch.setattr(claude_locks.os, "stat", one_bad_stat)
        seen = set()
        with proper_lockfile(lock_dir):
            state["armed"] = True
            for _ in range(12):
                time.sleep(0.02)
                seen.add(real_stat(lock_dir).st_mtime_ns)

        assert state["fired"] == 1, "premise: the injected error never fired"
        assert len(seen) > 1, (
            "the mtime never advanced after one transient stat error, so the "
            "heartbeat is dead and the lock is stealable while still held"
        )

    def test_release_leaves_a_lock_that_was_taken_over(self, lock_dir):
        # Control: test_acquire_creates_and_release_removes above — a release
        # that quietly stopped removing anything passes here and fails there.
        stolen_mtime = 1_000_000.0
        with proper_lockfile(lock_dir):
            # Deemed stale, removed, and re-created by somebody else.
            os.rmdir(lock_dir)
            os.mkdir(lock_dir)
            os.utime(lock_dir, (stolen_mtime, stolen_mtime))

        assert lock_dir.is_dir()
        assert os.stat(lock_dir).st_mtime == stolen_mtime

    def test_release_leaves_a_lock_a_running_toucher_saw(self, lock_dir, monkeypatch):
        # A live toucher must not adopt the successor's stamp as its own.
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.05)
        with proper_lockfile(lock_dir):
            os.rmdir(lock_dir)
            os.mkdir(lock_dir)
            # mtimes are ms-coarse, so a bare mkdir can land on our own
            # acquire stamp and the case would then measure nothing. A whole
            # second AHEAD both round-trips distinctly and is the direction a
            # real takeover moves: the successor's mkdir stamps it now, which
            # is later than ours. A second BACK also round-trips, but it asks
            # the toucher to read a REWIND as somebody else's lock -- and the
            # transient-failure case one screen up rewinds our own by 30s and
            # requires the heartbeat to survive it. Only one of those two can
            # hold on an mtime, and inode identity cannot separate them either
            # (measured here: 200 rmdir+mkdir cycles, inode reused 200 times).
            fresh = float(int(time.time()) + 1)
            os.utime(lock_dir, (fresh, fresh))
            time.sleep(0.25)  # at least one toucher tick

        assert lock_dir.is_dir()
        assert lock_dir.stat().st_mtime == fresh  # nor did we refresh theirs

    def test_release_removes_a_slowly_touched_lock(self, lock_dir, monkeypatch):
        # Seen half-done, our own refresh reads to the release as a takeover.
        # A 1.2s tick outlives `_RELEASE_WAIT_S`, so the release reaches its
        # wait-or-refuse arm rather than deciding on a half-written stamp.
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.05)
        real_utime = os.utime

        def slow_utime(path, *args, **kwargs):
            real_utime(path, *args, **kwargs)
            if path == lock_dir:
                time.sleep(1.2)

        monkeypatch.setattr(claude_locks.os, "utime", slow_utime)
        with proper_lockfile(lock_dir):
            time.sleep(0.15)  # a tick starts and stalls mid-refresh

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
        """KEEPS, not "advanced once".

        A `_touch` that ticks once and returns satisfies a single-advance
        assertion and lets a credential lock go stale mid-hold anyway --
        `CREDENTIALS_STALENESS_S` is 60s and the interval is 3s, so one tick
        buys nothing. Measured: that mutant passed all 2098.

        So this samples repeatedly and requires the mtime to keep MOVING, not
        merely to differ from where it started.
        """
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.05)
        seen: list[int] = []
        with proper_lockfile(lock_dir):
            for _ in range(8):
                time.sleep(0.05)
                seen.append(lock_dir.stat().st_mtime_ns)
        assert seen[-1] > seen[0], "the mtime never advanced at all"
        # DISTINCT VALUES, which is what "keeps beating" means. A single tick
        # gives one step and then a flat line; this filesystem's granularity
        # is ~1ms, far below the 50ms interval, so several are reachable.
        assert len(set(seen)) >= 3, (
            f"the heartbeat advanced the mtime {len(set(seen))} time(s) over "
            "8 samples — one tick and then silence is what a lock taken over "
            "mid-hold looks like"
        )
        assert not lock_dir.exists()  # a refreshed lock is still ours to remove

    def test_creates_missing_parent(self, tmp_path):
        nested = tmp_path / "a" / "b" / "target.lock"
        with proper_lockfile(nested):
            assert nested.is_dir()


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


class TestAPermanentENOENTIsNotARetry:
    """Only a SWEPT name wants the fall-through.

    A missing parent (`~/.claude` removed or replaced between the
    `parents=True` mkdir and `os.mkdir`) raises the same `FileNotFoundError`
    and can never succeed, so the widened handler turned it into a full-budget
    100%-CPU spin that then blamed Claude Code. Measured before the fix:
    ~95,000 mkdir attempts per second, and at the production default that is
    two locks x 9s of a pinned core.
    """

    def test_a_missing_parent_raises_at_once(self, tmp_path, monkeypatch):
        gone = tmp_path / "gone"
        gone.mkdir()
        target = gone / "target.lock"

        # THE WINDOW: the parent disappears AFTER `parents=True` ran, so the
        # `os.mkdir` inside the loop can never succeed. `parents=True` is what
        # makes a plain rmtree before the call insufficient — it just puts the
        # directory back, and the lock is taken.
        real_mkdir = os.mkdir
        attempts = {"n": 0}

        def vanished(path, *a, **k):
            if os.fspath(path) == os.fspath(target):
                attempts["n"] += 1
                # REALLY GONE, not just the errno: the retry's own
                # `parents=True` already ran, and whether the parent is there
                # is the only thing that separates this from a swept name.
                shutil.rmtree(gone, ignore_errors=True)
            return real_mkdir(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "mkdir", vanished)
        start = time.monotonic()
        with pytest.raises(FileNotFoundError):
            with proper_lockfile(target, timeout=1.0):
                pass
        elapsed = time.monotonic() - start
        assert elapsed < 0.5, (
            f"a permanent ENOENT burned {elapsed:.2f}s of the budget spinning "
            f"over {attempts['n']} attempts instead of surfacing — and the "
            "timeout then blames Claude Code"
        )
        assert attempts["n"] <= 2, (
            f"{attempts['n']} mkdir attempts for a parent that cannot come "
            "back: that is a pinned core, not a retry"
        )


class TestTheAcquireAndReleaseAreBounded:
    """Two things the identity rewrite took out and had to put back."""

    def test_the_release_closes_the_descriptor_AFTER_its_identity_stat(
        self, tmp_path, monkeypatch
    ):
        """Ordering, and it is the only thing pinning the inode.

        The release stats the lock to prove it is still ours, then removes it
        by NAME. While the descriptor is open the inode cannot be recycled, so
        the stat is answering about the directory we actually hold. Close
        first and the inode is free the instant the last reference goes: this
        file's own sibling case measures ext4 handing the same number back
        200/200 times unheld, so the stat can then describe a stranger's
        directory and the rmdir removes it.

        The comment calls the ordering load-bearing; moving the close above
        the stat was measured green across the whole suite.
        """
        if not claude_locks._CAN_PIN_A_DIRECTORY:
            pytest.skip("no descriptor is held on this platform")

        lock = tmp_path / "target.lock"
        order = []
        real_stat, real_close = os.stat, os.close

        def tracking_stat(path, *a, **k):
            if not isinstance(path, int) and os.fspath(path) == os.fspath(lock):
                order.append("stat")
            return real_stat(path, *a, **k)

        def tracking_close(fd, *a, **k):
            order.append("close")
            return real_close(fd, *a, **k)

        with proper_lockfile(lock, timeout=1.0):
            # Installed HERE so only the RELEASE's calls are the subject;
            # nothing has been appended yet, so there is nothing to clear.
            monkeypatch.setattr(claude_locks.os, "stat", tracking_stat)
            monkeypatch.setattr(claude_locks.os, "close", tracking_close)
        monkeypatch.undo()

        assert "close" in order, (
            f"premise: the release never closed a descriptor: {order}"
        )
        assert "stat" in order, (
            f"premise: the release never stat'd the lock, so there is no "
            f"ordering to judge: {order}"
        )
        assert order.index("stat") < order.index("close"), (
            "the release closed the descriptor BEFORE its identity stat, so "
            "the inode was free to be recycled and the stat may describe a "
            f"stranger's directory: {order}"
        )

    def test_a_failed_identity_read_does_not_leak_the_descriptor(
        self, tmp_path, monkeypatch
    ):
        """The descriptor is ours the instant `os.open` returns.

        The `finally` that closes it is not reached until the body runs, so a
        raise between the open and the yield strands it -- on an inode the
        acquire's own error arm then unlinks, which pins it for the life of
        the process.
        """
        if not claude_locks._CAN_PIN_A_DIRECTORY:
            pytest.skip("no descriptor is held on this platform")
        if not os.path.isdir("/proc/self/fd"):
            pytest.skip("no /proc to count descriptors with")

        lock = tmp_path / "target.lock"

        def refuse(fd, *a, **k):
            raise OSError(errno.EIO, "I/O error")

        before = len(os.listdir("/proc/self/fd"))
        monkeypatch.setattr(claude_locks.os, "fstat", refuse)
        with pytest.raises(OSError):
            with proper_lockfile(lock, timeout=0.2):
                pass
        monkeypatch.undo()

        after = len(os.listdir("/proc/self/fd"))
        assert after <= before, (
            f"the acquire leaked {after - before} descriptor(s) on a failed "
            "identity read, pinned to an inode it then unlinked"
        )
        # AND THE DIRECTORY IS GONE. The arm makes the lock and then cannot
        # read it back; leaving it behind is a lock with no holder and no
        # heartbeat, which every waiter must sit out for the whole staleness
        # window. The fd count alone does not see that.
        assert not lock.exists(), (
            "the acquire left the lock directory behind after failing to "
            "identify it — an orphan no process holds and none will refresh"
        )

    def _successor_at(self, lock, mtime_ns):
        """Replace the lock with a fresh directory carrying `mtime_ns`."""
        os.rmdir(lock)
        os.mkdir(lock)
        os.utime(lock, ns=(mtime_ns, mtime_ns))
        return os.stat(lock).st_ino

    @pytest.mark.parametrize(
        "offset_ns,label",
        # BOTH REACH THE SAME GUARD, and the labels used to claim otherwise.
        # Instrumented, each param fires only the read-back-mismatch site --
        # neither reaches `_ours`'s rewind branch nor the adopt branch, which
        # has its own case. What varies is the SUCCESSOR's stamp relative to
        # ours, and the guard must refuse to remove in both directions.
        [(-2_000_000_000, "successor stamped EARLIER than our last write"),
         (2_000_000_000, "successor stamped LATER than our last write")],
    )
    def test_an_unproven_tick_forbids_the_release_from_removing(
        self, tmp_path, monkeypatch, caplog, offset_ns, label
    ):
        """UNPINNED. A separate READING cannot make the release independent.

        `os.utime` writes to the NAME and the read-back reads from the NAME,
        so one tick that proceeds without exact equality makes `last_stamp`
        equal whatever is at that path -- and a strict comparison then matches
        a SUCCESSOR's directory. Both readings share the value, so the only
        thing that can gate the removal is never having accepted an inexact
        one.
        """
        monkeypatch.setattr(claude_locks, "_CAN_PIN_A_DIRECTORY", False)
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)

        lock = tmp_path / "target.lock"
        swapped = threading.Event()
        successor = {}

        real_utime = os.utime

        def take_over_once(path, *a, **k):
            out = real_utime(path, *a, **k)
            if (not swapped.is_set() and not isinstance(path, int)
                    and os.fspath(path) == os.fspath(lock)):
                swapped.set()
                successor["ino"] = self._successor_at(
                    lock, os.stat(lock).st_mtime_ns + offset_ns)
            return out

        monkeypatch.setattr(claude_locks.os, "utime", take_over_once)
        caplog.set_level(logging.WARNING, logger="claude-swap")
        with proper_lockfile(lock, timeout=2.0, staleness=60.0):
            for _ in range(150):
                if swapped.is_set():
                    break
                time.sleep(0.02)
            time.sleep(0.15)   # ticks that could adopt the successor

        assert swapped.is_set(), "premise: the takeover never fired"
        assert lock.is_dir(), (
            f"the release removed a SUCCESSOR's lock ({label}) — its holder "
            "is now running unprotected and the name is free for a third "
            "waiter"
        )
        assert os.stat(lock).st_ino == successor["ino"], (
            "premise: the directory on disk is no longer the successor's, so "
            "this asserts nothing about whose lock survived"
        )
        # WHICH REFUSAL, not merely that one happened. A strict mismatch
        # leaves the lock too, so "it survived" passes even when the tick
        # never noticed it had adopted a stranger's stamp — the read-back's
        # own check is what this distinguishes.
        assert any("unproven stamp" in r.getMessage() for r in caplog.records), (
            "the lock survived, but by the strict comparison rather than "
            "because the tick recorded that it could not prove ownership: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    def test_a_toucher_that_cannot_start_leaves_nothing_behind(
        self, tmp_path, monkeypatch
    ):
        """The window between the acquire's `break` and the body's `try`.

        `RuntimeError: can't start new thread` there left the descriptor open
        AND the directory on disk with nobody holding it. On a credentials
        lock that blocks Claude Code's own refresh for the whole staleness
        window, and the next switch then blames Claude Code for a lock this
        process abandoned.
        """
        lock = tmp_path / "target.lock"
        have_proc = os.path.isdir("/proc/self/fd")
        before = len(os.listdir("/proc/self/fd")) if have_proc else None

        def refuse(self):
            raise RuntimeError("can't start new thread")

        monkeypatch.setattr(threading.Thread, "start", refuse)
        with pytest.raises(RuntimeError):
            with proper_lockfile(lock, timeout=0.5):
                pass
        monkeypatch.undo()

        assert not lock.exists(), (
            "the lock directory was left on disk with no heartbeat and no "
            "holder, so every waiter is blocked for the staleness window"
        )
        if have_proc:
            assert len(os.listdir("/proc/self/fd")) <= before, (
                "the descriptor was stranded on an inode nothing will unlink"
            )

    def test_an_external_rewind_is_not_read_as_a_takeover(
        self, tmp_path, monkeypatch
    ):
        """UNPINNED ONLY, where the mtime is a second witness at all.

        A takeover stamps NOW, so only a LATER mtime can be a successor's.
        One that moved BACKWARDS -- `rsync --times`, a restore, a clock step
        on a shared home -- is an external writer touching a lock we still
        hold, and equality reads it as theft: the heartbeat ends, the lock
        then really does go stale, and a waiter takes it mid-swap.
        """
        monkeypatch.setattr(claude_locks, "_CAN_PIN_A_DIRECTORY", False)
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)

        lock = tmp_path / "target.lock"
        real_utime = os.utime
        touches = {"n": 0}

        def counting(path, *a, **k):
            if (not isinstance(path, int)
                    and os.fspath(path) == os.fspath(lock)):
                touches["n"] += 1
            return real_utime(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "utime", counting)

        # THE REWIND LANDS IMMEDIATELY BEFORE A READ OF THE LOCK. Written as
        # a bare `utime` from this thread it races the tick: one that lands
        # after `_ours` has already read is overwritten by that same tick's
        # own refresh, whose read-back then matches, and the rewind is never
        # observed at all. Measured on APFS, 1 run in 3.
        armed = threading.Event()
        rewound = threading.Event()
        real_stat = os.stat

        def rewinding(path, *a, **k):
            if (armed.is_set() and not rewound.is_set()
                    and not isinstance(path, int)
                    and os.fspath(path) == os.fspath(lock)):
                rewound.set()
                real_utime(lock, (time.time() - 30,) * 2)
            return real_stat(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "stat", rewinding)
        with proper_lockfile(lock):
            time.sleep(0.1)
            before = touches["n"]
            assert before >= 1, "premise: the heartbeat never ran at all"
            armed.set()
            assert rewound.wait(2.0), (
                "premise: no tick read the lock after arming, so no rewind "
                "was ever placed for one to see"
            )
            time.sleep(0.15)
            after = touches["n"]

        assert after > before, (
            f"the heartbeat stopped after an external REWIND ({before} -> "
            f"{after}) — an mtime that moved backwards was read as a takeover"
        )
        # THE LOCK IS LEFT, AND THAT IS THE TRADE. A rewind is one of the
        # states no reading can tell from a successor stamped in the past, so
        # the heartbeat keeps beating (this case's subject) and the release
        # refuses to remove what it cannot prove. The stale sweep recovers a
        # lock left behind; nothing recovers a successor's lock removed
        # inside its critical section.
        assert lock.exists(), (
            "the release removed a lock a rewind had made unprovable"
        )

    def test_a_failed_read_back_is_ADOPTED_so_the_heartbeat_survives(
        self, tmp_path, monkeypatch
    ):
        """The unpinned path's whole reason for `adopt_stamp`, and nothing ran it.

        The tick writes a stamp it chose, then reads it back. If THAT read
        fails, `last_stamp` never advances, and the next tick's `_ours` sees a
        mtime it cannot match. Refusing there returns out of the heartbeat, the
        mtime stops advancing, and a waiter takes the lock over as stale --
        mid-hold, on a lock nobody actually took. Adopting the observed stamp
        keeps the beat alive, and `unproven` stops the release acting on it.

        Measured before this case existed: `if adopt_stamp:` -> `if False:`
        left the whole suite green.
        """
        monkeypatch.setattr(claude_locks, "_CAN_PIN_A_DIRECTORY", False)
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)

        lock = tmp_path / "target.lock"
        real_stat, real_utime = os.stat, os.utime
        state = {"ticks": 0, "injected": False, "after": 0, "tid": None}

        def counting_utime(path, *a, **k):
            if not isinstance(path, int) and os.fspath(path) == os.fspath(lock):
                state["tid"] = threading.get_ident()
                state["ticks"] += 1
                if state["injected"]:
                    state["after"] += 1
            return real_utime(path, *a, **k)

        def failing_stat(path, *a, **k):
            # THE READ-BACK ONLY: the toucher's own thread, this lock, once.
            if (not isinstance(path, int)
                    and os.fspath(path) == os.fspath(lock)
                    and threading.get_ident() == state["tid"]
                    and state["ticks"] >= 1
                    and not state["injected"]):
                state["injected"] = True
                raise OSError(errno.EIO, "injected read-back failure")
            return real_stat(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "utime", counting_utime)
        monkeypatch.setattr(claude_locks.os, "stat", failing_stat)
        with proper_lockfile(lock, timeout=2.0):
            time.sleep(0.4)
        monkeypatch.undo()

        assert state["injected"], (
            f"premise: the read-back never failed ({state['ticks']} tick(s)), "
            "so this says nothing about what happens when it does"
        )
        assert state["after"] >= 2, (
            f"the heartbeat stopped after the failed read-back "
            f"({state['after']} tick(s) followed it) — the lock's mtime now "
            "freezes and a waiter may take it over as stale, mid-hold"
        )
        assert lock.exists(), (
            "the release removed a lock whose stamp it could not prove, "
            "which is the other direction of the same fault"
        )

    def test_the_release_does_not_wait_out_a_stalled_tick(
        self, tmp_path, monkeypatch
    ):
        """UNPINNED ONLY, where the release takes the stamp mutex at all.

        `_touch` holds it across three syscalls, so an unbounded acquire
        hands the release however long the filesystem stalls -- on a
        `finally` that a single switch reaches three times.
        """
        monkeypatch.setattr(claude_locks, "_CAN_PIN_A_DIRECTORY", False)
        monkeypatch.setattr(claude_locks, "_RELEASE_WAIT_S", 0.2)
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)

        lock = tmp_path / "target.lock"
        real_utime = os.utime
        entered = threading.Event()

        def stalling(path, *a, **k):
            if (not isinstance(path, int)
                    and os.fspath(path) == os.fspath(lock)):
                entered.set()
                time.sleep(2.0)
            return real_utime(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "utime", stalling)
        start = time.monotonic()
        with proper_lockfile(lock, timeout=1.0):
            assert entered.wait(1.0), "premise: no tick ever entered the stall"
        elapsed = time.monotonic() - start

        assert elapsed < 1.5, (
            f"the release waited {elapsed:.2f}s on a tick stalled 2.0s — the "
            "acquire of the stamp mutex is unbounded, so the filesystem sets "
            "the bound"
        )


class TestIdentityNotAStamp:
    """A stamp is a value we write; a successor's directory can carry it."""

    @pytest.mark.skipif(
        not claude_locks._CAN_PIN_A_DIRECTORY,
        reason="no way to hold a directory open, so the hole cannot be closed",
    )
    def test_the_release_does_not_remove_a_successors_lock(self, tmp_path):
        """The window is between the heartbeat's utime and anything after it.

        A stalled holder is judged stale, and the takeover lands while our
        own refresh is in flight. Under an mtime stamp the holder read the
        successor's directory back as its own and removed it on exit --
        inside the successor's critical section, freeing the name for a
        third waiter. Identity cannot be adopted that way.
        """
        lock = tmp_path / "the.lock"
        real_utime = os.utime
        took_over = threading.Event()
        successor = {}

        def utime_then_takeover(path, *a, **kw):
            real_utime(path, *a, **kw)
            if not took_over.is_set() and str(path) == str(lock):
                took_over.set()
                os.rmdir(lock)
                os.mkdir(lock)
                successor["ino"] = os.stat(lock).st_ino

        with patch.object(claude_locks, "TOUCH_INTERVAL_S", 0.02), \
                patch.object(os, "utime", utime_then_takeover):
            with claude_locks.proper_lockfile(lock, timeout=2.0, staleness=60.0):
                for _ in range(200):
                    if took_over.is_set():
                        break
                    time.sleep(0.02)
                time.sleep(0.15)  # ticks that would adopt the successor

        assert took_over.is_set(), "premise: the takeover never fired"
        assert lock.is_dir(), (
            "our release removed the SUCCESSOR's lock, so its holder is "
            "running unprotected and the name is free for a third waiter"
        )
        assert os.stat(lock).st_ino == successor["ino"], (
            "premise: the directory on disk is no longer the successor's, "
            "so this asserts nothing about whose lock survived"
        )

    @pytest.mark.skipif(
        not claude_locks._CAN_PIN_A_DIRECTORY,
        reason="asserts the POSIX pin this platform does not have",
    )
    def test_an_unheld_inode_number_comes_straight_back(self, tmp_path):
        """Why identity is taken from a HELD descriptor and not a plain stat.

        Without the open, `(st_dev, st_ino)` is no better than the mtime it
        replaced: the next `mkdir` gets the same number. The open pins the
        inode in the orphan list, which is what makes the comparison mean
        "the same directory" rather than "the same slot".
        """
        lock = tmp_path / "L"
        reused_unheld = reused_held = 0
        for hold in (False, True):
            for _ in range(50):
                os.mkdir(lock)
                fd = os.open(lock, os.O_RDONLY) if hold else None
                first = os.stat(lock).st_ino
                os.rmdir(lock)
                os.mkdir(lock)
                same = os.stat(lock).st_ino == first
                os.rmdir(lock)
                if fd is not None:
                    os.close(fd)
                if hold:
                    reused_held += same
                else:
                    reused_unheld += same

        if reused_unheld == 0:
            # NOT A FAILURE: APFS mints a new inode number every time
            # (measured 0/50), so nothing here can distinguish a pin that
            # works from one that does nothing, whatever `reused_held` says.
            # ext4 recycles on every iteration (200/200), which is where this
            # case has something to prove.
            pytest.skip(
                "this filesystem never reuses an inode number even unheld "
                f"({reused_unheld}/50), so the held result proves nothing"
            )
        assert reused_held == 0, (
            f"a held descriptor did not pin the inode ({reused_held}/50), so "
            "identity can be adopted the same way a stamp was"
        )


class TestATransientErrnoIsNotFatalToTheHold:
    """Two arms reached by an ordinary errno on a network `~/.claude`, and
    neither is reachable from any case that asserts a return value.

    The heartbeat runs on a daemon thread, so anything it raises is swallowed
    by `threading.excepthook` and the suite stays green while the mtime stops
    advancing -- which is the stale lock this module exists to prevent.
    """

    def _run(self, tmp_path, monkeypatch, *, syscall, errno_):
        """-> (lock, raised, after) where `after` counts calls to `syscall` on
        the lock AFTER the injected one.

        `after` is the half that matters. "The thread did not raise" is not
        the arm's contract -- the arm promises the heartbeat STAYS ARMED, and
        `except OSError: return` satisfies the first while breaking the
        second. Measured: that one-token mutant, which is the original defect
        the arm's comment describes, left the whole file at 36 passed.
        """
        lock = tmp_path / "target.lock"
        real = getattr(os, syscall)
        seen = {"before": 0, "after": 0}
        raised: list[str] = []

        def failing(path, *a, **k):
            if not isinstance(path, int) and os.fspath(path) == os.fspath(lock):
                if not seen["before"]:
                    seen["before"] = 1
                    raise OSError(errno_, "injected")
                seen["after"] += 1
            return real(path, *a, **k)

        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        monkeypatch.setattr(claude_locks.os, syscall, failing)
        monkeypatch.setattr(
            threading, "excepthook",
            lambda a: raised.append(f"{a.exc_type.__name__}: {a.exc_value}"),
        )
        with proper_lockfile(lock, timeout=2.0):
            time.sleep(0.2)
            in_body = seen["after"]
        assert seen["before"], "premise: the injected error never fired"
        return lock, raised, in_body

    def test_a_utime_error_that_is_not_absence_does_not_kill_the_toucher(
        self, tmp_path, monkeypatch
    ):
        """The arm's own comment promises the heartbeat stays armed on EIO or
        ESTALE. It called a name this module does not define, so the thread
        died on the first one and the lock's mtime froze."""
        _lock, raised, after = self._run(
            tmp_path, monkeypatch, syscall="utime", errno_=errno.EIO
        )
        assert not raised, (
            f"the heartbeat thread died on a transient errno: {raised} — the "
            "mtime stops advancing and a waiter takes the lock as stale"
        )
        # AND IT KEPT BEATING. Not raising is only half of what the arm
        # promises, and `return` here is the defect its own comment describes.
        assert after > 0, (
            "the heartbeat made no further refresh after ONE transient errno "
            "— it did not raise, it stopped, and a mtime that stops advancing "
            "is a lock a waiter takes over mid-swap"
        )

    def test_a_stat_error_does_not_forfeit_a_pinned_release(
        self, tmp_path, monkeypatch
    ):
        """`unproven` is a fact about the STAMP, and a pinned platform never
        reads one -- identity comes from the held descriptor. Recording it
        anyway made one transient errno leave the credentials lock on disk for
        the whole staleness window, blocking Claude Code's own refresh."""
        if not claude_locks._CAN_PIN_A_DIRECTORY:
            pytest.skip("no held descriptor here, so the stamp really is the witness")
        lock, raised, after = self._run(
            tmp_path, monkeypatch, syscall="stat", errno_=errno.EIO
        )
        assert not raised, f"the heartbeat thread died: {raised}"
        assert after > 0, "premise: the heartbeat stopped, so no stamp was owed"
        assert not lock.exists(), (
            "the release left the lock behind over an unprovable STAMP on a "
            "platform that decides identity by descriptor and never reads one"
        )

    def test_the_quantum_probe_is_not_in_the_unprotected_gap(
        self, tmp_path, monkeypatch
    ):
        """Nothing removes the lock between the acquire's `break` and the
        `try` whose `finally` does.

        One hunk lower this file already argues it: "a `RuntimeError: can't
        start new thread` here left the descriptor open AND the lock directory
        on disk, with nobody holding it". The quantum probe then put up to
        fourteen syscalls back into that same gap. A `KeyboardInterrupt` there
        -- not an `OSError`, so the probe's own handler does not see it -- is
        the realistic trigger, and on a credentials lock the directory blocks
        Claude Code's own refresh for the full staleness window.
        """
        monkeypatch.setattr(claude_locks, "_CAN_PIN_A_DIRECTORY", False)

        def interrupted(_directory):
            raise KeyboardInterrupt("injected inside the probe")

        monkeypatch.setattr(claude_locks, "_mtime_quantum_ns", interrupted)
        lock = tmp_path / "target.lock"
        try:
            with claude_locks.proper_lockfile(lock, timeout=2):
                pass
        except KeyboardInterrupt:
            pass
        assert not lock.exists(), (
            "the acquire raised out of the quantum probe and left the lock "
            "directory on disk with nobody holding it"
        )

    @pytest.mark.parametrize(
        "fs_quantum", [1, 100, 1_000_000, 1_000_000_000])
    def test_the_probe_answers_the_fine_end_too(
        self, tmp_path, monkeypatch, fs_quantum
    ):
        """Its coarse sibling is satisfied by `return 2_000_000_000`.

        That constant passes the 2s case, so the candidate list, the loop and
        the ordering have no witness at all -- and this function runs on the
        only platform that cannot pin a directory, where coarsening every
        stamp widens the window in which a successor's mkdir mtime collides
        with ours and the release removes THEIR lock.
        """
        real_utime = os.utime

        def at_granularity(path, *a, **k):
            ns = k.get("ns")
            if isinstance(ns, tuple) and len(ns) == 2:
                k["ns"] = tuple(
                    (int(v) // fs_quantum) * fs_quantum for v in ns)
            return real_utime(path, *a, **k)

        # THE FAKE IS A FLOOR, NOT A SETTING. It can only coarsen, so on a
        # host that already keeps less than `fs_quantum` the honest answer is
        # the host's own -- asserting `fs_quantum` there demands a precision
        # the filesystem does not have. Measured on the Windows runner: NTFS
        # keeps 100ns and the case for 1ns failed on the real granularity.
        #
        # MEASURED WITHOUT THE SUBJECT. Taking this baseline from
        # `_mtime_quantum_ns` itself makes the expectation move with the
        # answer: a body replaced by `return 2_000_000_000` sets both sides to
        # 2e9 and the case passes -- the control satisfied by the mutation it
        # exists to catch. The duplicated round-trip below is the price of a
        # baseline the subject cannot influence.
        def host_keeps(q):
            probe = tmp_path / ".host-probe"
            probe.touch()
            v = ((time.time_ns() // (2 * q)) * 2 + 1) * q
            os.utime(probe, ns=(v, v))
            return os.stat(probe).st_mtime_ns == v

        host = next(q for q in (1, 100, 1_000, 1_000_000,
                                1_000_000_000, 2_000_000_000)
                    if host_keeps(q))
        monkeypatch.setattr(claude_locks.os, "utime", at_granularity)
        got = claude_locks._mtime_quantum_ns(tmp_path)
        want = max(fs_quantum, host)
        assert got == want, (
            f"the probe measured {got}ns on a filesystem that keeps {want}ns "
            f"(fake floor {fs_quantum}ns over a host that keeps {host}ns)"
        )

    @pytest.mark.parametrize("offset_ns", [0, 1_000_000_000])
    def test_the_quantum_probe_is_not_fooled_by_an_aligned_candidate(
        self, tmp_path, monkeypatch, offset_ns
    ):
        """A candidate must be REJECTED by a filesystem coarser than itself.

        The probe writes a value derived from the clock and keeps the first
        candidate that round-trips. Derived as `(t // q) * q`, that value is
        ALREADY a multiple of any coarser quantum whenever the clock happens to
        sit on one -- so on a two-second mount the one-second candidate
        round-trips for every even second and the quantum comes back half what
        it is. The stamp is then written finer than the filesystem keeps, the
        read-back differs, `unproven` latches, and the release leaves the lock:
        exactly the defect the quantisation exists to remove, on half of holds.

        Both parities are exercised, because the wrong answer is the one that
        only appears on one of them.
        """
        quantum = 2_000_000_000
        real_utime = os.utime

        def coarse_utime(path, *a, **k):
            ns = k.get("ns")
            if isinstance(ns, tuple) and len(ns) == 2:
                k["ns"] = tuple((int(v) // quantum) * quantum for v in ns)
            return real_utime(path, *a, **k)

        base = (time.time_ns() // (2 * quantum)) * 2 * quantum + offset_ns
        monkeypatch.setattr(claude_locks.os, "utime", coarse_utime)
        monkeypatch.setattr(claude_locks.time, "time_ns", lambda: base)

        got = claude_locks._mtime_quantum_ns(tmp_path)
        assert got == quantum, (
            f"the probe measured {got}ns on a filesystem that keeps {quantum}ns "
            "— a stamp written at that granularity cannot round-trip, so every "
            "tick latches `unproven` and the release removes nothing"
        )

    def test_a_coarse_mtime_still_lets_the_release_remove_its_own_lock(
        self, tmp_path, monkeypatch
    ):
        """UNPINNED, undisturbed, nobody stealing -- and the lock survived.

        The heartbeat proves a hold by reading back the stamp it wrote. A
        filesystem that keeps a coarser mtime truncates it, so EVERY read-back
        differs, `unproven` latches on the first tick, and the release then
        removes nothing. The next acquire on that name waits out the whole
        staleness window blaming Claude Code for a lock this process
        abandoned, and a fresh orphan appears for every hold.
        """
        monkeypatch.setattr(claude_locks, "_CAN_PIN_A_DIRECTORY", False)
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        lock = tmp_path / "target.lock"
        real_utime = os.utime
        quantum = 1_000_000_000  # a one-second mount

        def coarse_utime(path, *a, **k):
            ns = k.get("ns")
            if isinstance(ns, tuple) and len(ns) == 2:
                k["ns"] = tuple((int(v) // quantum) * quantum for v in ns)
            return real_utime(path, *a, **k)

        monkeypatch.setattr(claude_locks.os, "utime", coarse_utime)

        with proper_lockfile(lock, timeout=2.0):
            time.sleep(0.2)  # several ticks at the interval above

        assert not lock.exists(), (
            "the release left an undisturbed lock behind: the read-back never "
            "matched because the filesystem coarsened it, not because anybody "
            "else wrote"
        )

    def test_an_unprovable_stamp_does_not_let_the_release_take_a_successor(
        self, tmp_path, monkeypatch, caplog
    ):
        """The UNPINNED half of the arm above, and the half that had no case.

        Pinned, `unproven` is a fact the release never consults. Unpinned it
        is the only thing between a transient errno and rmdir-ing a stranger's
        lock: the failed identity read leaves a window, a successor takes the
        name inside it, and the refresh then writes OUR chosen stamp onto
        THEIR directory. The read-back matches, because we wrote it. Without
        this flag the release sees a reused inode carrying its own stamp,
        calls the successor's lock its own, and removes it mid-hold.
        """
        monkeypatch.setattr(claude_locks, "_CAN_PIN_A_DIRECTORY", False)
        lock = tmp_path / "target.lock"
        real_stat = os.stat
        state = {"fired": False}

        def failing_stat(path, *a, **k):
            if (
                not isinstance(path, int)
                and os.fspath(path) == os.fspath(lock)
                and not state["fired"]
                # THE HEARTBEAT'S READ, not the acquire's. `proper_lockfile`
                # stats the same name on the way in, and firing there raises
                # out of the `with` instead of reaching the arm under test.
                and threading.current_thread() is not threading.main_thread()
            ):
                state["fired"] = True
                # THE TAKEOVER LANDS IN THE WINDOW the failed read opens.
                os.rmdir(lock)
                os.mkdir(lock)
                raise OSError(errno.EIO, "injected")
            return real_stat(path, *a, **k)

        raised: list[str] = []
        caplog.set_level(logging.WARNING, logger="claude-swap")
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        monkeypatch.setattr(claude_locks.os, "stat", failing_stat)
        monkeypatch.setattr(
            threading, "excepthook",
            lambda a: raised.append(f"{a.exc_type.__name__}: {a.exc_value}"),
        )
        with proper_lockfile(lock, timeout=2.0):
            time.sleep(0.2)

        assert state["fired"], "premise: the injected error never fired"
        assert not raised, f"the heartbeat thread died: {raised}"
        assert lock.exists(), (
            "the release removed a SUCCESSOR's lock — the tick could not "
            "prove whose directory it stamped, and its own stamp read back "
            "as proof of ownership"
        )
        # WHICH REFUSAL, not merely that it survived. The strict check
        # refuses on IDENTITY too, so on a filesystem that mints a fresh inode
        # (APFS always, ext4 once the number is consumed) the lock survives for
        # a reason unrelated to the flag under test and the mutation ships green.
        assert any("unproven stamp" in r.getMessage() for r in caplog.records), (
            "the lock survived, but by the strict identity comparison rather "
            "than because the tick recorded it could not prove ownership: "
            f"{[r.getMessage() for r in caplog.records]}"
        )

    def test_an_undisturbed_hold_is_removed_and_keeps_beating(
        self, tmp_path, monkeypatch
    ):
        """THE BASELINE for the two cases above: with nothing injected the
        heartbeat ticks and the release removes.

        It is NOT a control against an always-removing release -- six other
        cases in this file fail that mutation and this one does not, so
        claiming it here would name a power it has not got.
        """
        lock = tmp_path / "target.lock"
        ticks = {"n": 0}
        real_utime = os.utime

        def counting(path, *a, **k):
            if not isinstance(path, int) and os.fspath(path) == os.fspath(lock):
                ticks["n"] += 1
            return real_utime(path, *a, **k)

        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.02)
        monkeypatch.setattr(claude_locks.os, "utime", counting)
        with proper_lockfile(lock, timeout=2.0):
            time.sleep(0.2)
            assert lock.exists(), "premise: the lock was never taken"
        assert ticks["n"] > 1, (
            f"{ticks['n']} refresh(es) in 0.2s at a 0.02s interval — the "
            "instrument, not the code"
        )
        assert not lock.exists()
