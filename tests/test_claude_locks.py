"""Tests for the proper-lockfile-compatible Claude Code lock helpers."""

from __future__ import annotations

import errno
import os
import shutil
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

        def swept(path, *args, **kwargs):
            if os.fspath(path) != os.fspath(lock_dir):
                return real_mkdir(path, *args, **kwargs)
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
        # 1.2s outlives the join, so the release must wait rather than guess.
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

    def test_release_removes_a_lock_a_late_tick_refreshes(self, lock_dir, monkeypatch):
        # A tick past its deadline but not yet at the stamp is in flight too:
        # read it and let go, and the stat sees a directory that tick refreshed.
        monkeypatch.setattr(claude_locks, "TOUCH_INTERVAL_S", 0.05)
        main = threading.current_thread()
        releasing = threading.Event()
        real_wait, real_stat = threading.Event.wait, os.stat

        def late_wait(self, timeout=None):
            woke = real_wait(self, timeout)
            if not woke and threading.current_thread() is not main:
                time.sleep(1.2)  # past the release's join, still before the lock
            return woke

        def gap_before_stat(path, *args, **kwargs):
            if releasing.is_set() and threading.current_thread() is main:
                releasing.clear()
                time.sleep(0.3)  # the late tick lands in here
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(threading.Event, "wait", late_wait)
        monkeypatch.setattr(claude_locks.os, "stat", gap_before_stat)
        with proper_lockfile(lock_dir):
            time.sleep(0.1)
            releasing.set()

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
            acquired_ns = lock_dir.stat().st_mtime_ns
            time.sleep(0.4)
            assert lock_dir.stat().st_mtime_ns > acquired_ns
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


class TestTheBackstopIsReachableOnTheDeclaredFloor:
    """An ini option the declared pytest cannot parse is not a backstop.

    `faulthandler_exit_on_timeout` landed in pytest 9.0.0. Under 8.x it is an
    unknown key: a `PytestConfigWarning` nobody reads, no abort, and the hang
    the option exists to end. CI installs from `uv.lock` and is safe; an
    editable install anywhere inside the declared range was not, and nothing
    said so.
    """

    # option -> the first pytest that understands it
    _INI_FLOORS = {"faulthandler_exit_on_timeout": (9, 0)}

    @staticmethod
    def _pyproject():
        import tomllib
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        with open(root / "pyproject.toml", "rb") as fh:
            return tomllib.load(fh), root

    def test_the_declared_pytest_floor_understands_every_ini_option_used(self):
        cfg, root = self._pyproject()
        ini = cfg["tool"]["pytest"]["ini_options"]
        used = {k: v for k, v in self._INI_FLOORS.items() if k in ini}
        # THE DENOMINATOR. With no version-gated option declared there is no
        # requirement, and an empty loop below would pass without looking.
        assert used, (
            "no version-gated ini option is declared — this guard has no "
            f"subject; known ones: {sorted(self._INI_FLOORS)}"
        )

        specs = [d for d in cfg["dependency-groups"]["dev"]
                 if d.split(">=")[0].strip() == "pytest"]
        assert len(specs) == 1, f"expected one pytest dev spec, got {specs}"
        declared = tuple(int(x) for x in specs[0].split(">=")[1].split(".")[:2])
        for option, floor in used.items():
            assert declared >= floor, (
                f"`{option}` needs pytest >= {floor[0]}.{floor[1]} and the dev "
                f"group declares >= {declared[0]}.{declared[1]}, so inside the "
                "declared range the option is an unknown key and the backstop "
                "is inert"
            )

    def test_the_lockfile_carries_the_same_floor(self):
        """`uv sync --locked` fails on a specifier the lock disagrees with, so
        a dependency edit is a two-file commit and a green suite never opens
        the second one."""
        cfg, root = self._pyproject()
        spec = next(d for d in cfg["dependency-groups"]["dev"]
                    if d.split(">=")[0].strip() == "pytest")
        lock = (root / "uv.lock").read_text(encoding="utf-8")
        needle = f'{{ name = "pytest", specifier = ">={spec.split(">=")[1]}" }}'
        assert needle in lock, (
            f"uv.lock does not carry {spec!r}; `uv sync --locked` refuses a "
            "pyproject the lock was not regenerated for"
        )
