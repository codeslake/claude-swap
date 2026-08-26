"""Tests for file locking mechanism."""

from __future__ import annotations

import multiprocessing
import threading
import time
from pathlib import Path

import pytest

from claude_swap import locking
from claude_swap.exceptions import LockError
from claude_swap.locking import FileLock


class TestFileLock:
    """Test FileLock class."""

    def test_acquire_and_release(self, tmp_path: Path):
        """Test basic lock acquire and release."""
        lock_path = tmp_path / ".lock"
        lock = FileLock(lock_path)

        assert lock.acquire(timeout=1.0) is True
        assert lock._locked is True
        lock.release()
        assert lock._locked is False

    def test_context_manager(self, tmp_path: Path):
        """Test using lock as context manager."""
        lock_path = tmp_path / ".lock"

        with FileLock(lock_path) as lock:
            assert lock._locked is True

        assert lock._locked is False

    def test_context_manager_creates_parent_dirs(self, tmp_path: Path):
        """Test that lock creates parent directories."""
        lock_path = tmp_path / "nested" / "dir" / ".lock"

        with FileLock(lock_path):
            assert lock_path.parent.exists()

    def test_lock_timeout(self, tmp_path: Path):
        """Test that lock times out when already held."""
        lock_path = tmp_path / ".lock"

        # Acquire first lock
        lock1 = FileLock(lock_path)
        assert lock1.acquire(timeout=1.0) is True

        # Try to acquire second lock - should timeout
        lock2 = FileLock(lock_path)
        assert lock2.acquire(timeout=0.5) is False

        lock1.release()

    def test_a_small_timeout_is_not_overshot_by_the_retry_sleep(
            self, tmp_path: Path, monkeypatch):
        """`timeout` must bound acquire(), the retry sleep included.

        The deadline is checked before the sleep, so a flat sleep runs to
        completion past a timeout shorter than itself and the check cannot
        fire until it returns.

        Asserted on the sleep ARGUMENT rather than on elapsed time alone: a
        wall-clock ceiling only says "overshoot smaller than N", so shrinking
        the flat sleep below N keeps it green while `timeout` is still
        unbounded for every value under the new constant. It also makes the
        test independent of clock granularity, which differs by platform.
        """
        lock_path = tmp_path / "overshoot.lock"
        holder = FileLock(lock_path)
        assert holder.acquire(timeout=1.0), "the fixture failed to take the lock"
        waiter = FileLock(lock_path)
        try:
            slept = []
            real_sleep = time.sleep
            mine = threading.get_ident()

            def recording(seconds):
                # SCOPED. `locking.time` IS the `time` module, so this patch is
                # process-global: an unscoped recorder collects every other
                # thread's sleeps too. conftest documents that this suite
                # leaves threads running, and `proper_lockfile`'s own retry
                # loop sleeps 0.25-0.5s — a foreign 0.5 landing here fails the
                # per-sleep invariant below on a correct implementation. The
                # sibling in test_claude_locks.py carries this guard and its
                # comment records that exact miss being measured.
                if threading.get_ident() != mine:
                    return real_sleep(seconds)      # not ours: leave it alone
                slept.append((time.monotonic(), seconds))
                return real_sleep(seconds)

            monkeypatch.setattr(locking.time, "sleep", recording)
            # SMALL ON PURPOSE. The invariant only discriminates against a
            # flat sleep LARGER than the budget: with a generous budget a
            # 0.1s sleep fits inside what is left and the assert passes on
            # the very bug this exists for (measured: budget=0.5 lets flat
            # 0.1 through).
            budget = 0.01
            begin = time.monotonic()
            got = waiter.acquire(timeout=budget)
            elapsed = time.monotonic() - begin

            assert not got, "the lock was held; acquire must not succeed"
            assert slept, "no retry sleep happened — the instrument, not the code"
            # `begin` is anchored here, but `acquire` starts its own deadline
            # AFTER `mkdir(parents=...)` + `open()`, so a correct clamp can
            # sleep slightly past THIS deadline. Measured worst case for that
            # skew: 0.9ms, which left only 12% headroom under a 1ms tolerance
            # -- real flakiness on a slow first `open()`. Widened to `budget`,
            # which is the most a correct clamp can produce: its own cap is
            # the remaining internal budget, never more than `timeout`.
            #
            # NOT wider. At 0.05 a flat 0.02 and a flat 0.04 both PASS
            # (measured), so a regression reintroducing a small flat sleep
            # would ship green. A ceiling you can tune a flat sleep under is
            # not an invariant, which is the whole point of the docstring
            # above.
            deadline = begin + budget
            for at, seconds in slept:
                left = deadline - at
                assert seconds <= max(0.0, left) + budget, (
                    f"slept {seconds:.3f}s with {left:.3f}s of budget left — "
                    "the retry sleep ignored the deadline"
                )
                # Anchor-free, so no skew to absorb: the clamp is capped by
                # the remaining budget, which never exceeds the whole timeout.
                assert seconds <= budget + 1e-6, (
                    f"a single retry sleep of {seconds:.3f}s exceeds the "
                    f"whole {budget}s timeout"
                )
            # NO SUM CEILING. It claimed to catch "many small sleeps, each
            # legal, together overrunning", but the clamp caps every sleep at
            # the budget the code itself has left, so a correct implementation
            # reaches that shape only when `monotonic()` does not advance
            # across a sleep -- it then legitimately sleeps the whole remaining
            # budget again. Measured on a Windows shard, where the clock ticks
            # in ~15.6ms steps: two 0.01s sleeps against a 0.01s budget, on
            # code with no defect. The flat sleep this test exists for is
            # caught by the per-sleep assert above (measured: same failure,
            # same line, with and without a sum ceiling), so the ceiling cost
            # a red CI and bought nothing.
        finally:
            waiter.release()
            holder.release()

    def test_lock_acquired_after_release(self, tmp_path: Path):
        """Test that lock can be acquired after previous holder releases."""
        lock_path = tmp_path / ".lock"

        lock1 = FileLock(lock_path)
        lock1.acquire(timeout=1.0)
        lock1.release()

        lock2 = FileLock(lock_path)
        assert lock2.acquire(timeout=1.0) is True
        lock2.release()

    def test_context_manager_raises_on_timeout(self, tmp_path: Path):
        """Test that context manager raises LockError on timeout."""
        lock_path = tmp_path / ".lock"

        # Hold the lock
        holder = FileLock(lock_path)
        holder.acquire(timeout=1.0)

        # Try to acquire with context manager
        with pytest.raises(LockError):
            # Create a lock with very short timeout
            lock = FileLock(lock_path)
            lock.acquire = lambda timeout=10.0: False  # Force failure
            with lock:
                pass

        holder.release()

    def test_double_release_safe(self, tmp_path: Path):
        """Test that releasing twice doesn't raise."""
        lock_path = tmp_path / ".lock"
        lock = FileLock(lock_path)

        lock.acquire(timeout=1.0)
        lock.release()
        lock.release()  # Should not raise


def _hold_lock_process(lock_path: str, duration: float, ready_event, done_event):
    """Helper function to hold a lock in a subprocess."""
    lock = FileLock(Path(lock_path))
    if lock.acquire(timeout=5.0):
        ready_event.set()  # Signal that lock is held
        time.sleep(duration)
        lock.release()
    done_event.set()


class TestFileLockConcurrency:
    """Test concurrent access to file locks."""

    def test_concurrent_access_blocked(self, tmp_path: Path):
        """Test that concurrent processes are blocked."""
        lock_path = tmp_path / ".lock"

        ready_event = multiprocessing.Event()
        done_event = multiprocessing.Event()

        # Start process that holds the lock
        p = multiprocessing.Process(
            target=_hold_lock_process,
            args=(str(lock_path), 2.0, ready_event, done_event),
        )
        p.start()

        # Wait for the subprocess to acquire the lock
        ready_event.wait(timeout=5.0)

        # Now try to acquire - should fail fast
        lock = FileLock(lock_path)
        result = lock.acquire(timeout=0.5)

        assert result is False

        # Clean up
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()

    def test_lock_acquired_after_process_exits(self, tmp_path: Path):
        """Test that lock can be acquired after holding process exits."""
        lock_path = tmp_path / ".lock"

        ready_event = multiprocessing.Event()
        done_event = multiprocessing.Event()

        # Start process that holds the lock briefly
        p = multiprocessing.Process(
            target=_hold_lock_process,
            args=(str(lock_path), 0.5, ready_event, done_event),
        )
        p.start()

        # Wait for subprocess to finish
        done_event.wait(timeout=5.0)
        p.join(timeout=5.0)

        # Now we should be able to acquire
        lock = FileLock(lock_path)
        result = lock.acquire(timeout=1.0)

        assert result is True
        lock.release()


class TestTheClampSurvivesWeakeningNotOnlyDeletion:
    """`min(sleep, timeout)` is a no-op once most of the budget is spent.

    The case above uses a budget SMALLER than the flat sleep (0.01 vs 0.1),
    where `min(0.1, timeout)` and `min(0.1, remaining)` are the same number.
    Measured: weakening this clamp to the whole timeout leaves the two lock
    suites at 38 passed, while DELETING it is caught — so the file
    discriminates against deletion and is blind to weakening. That is the
    identical hole `test_claude_locks.py` grew a class to close, left open on
    the other file the same change touched.

    Per sleep against the budget left at that moment, never a wall-clock total
    or the shape of the last draw: both depend on when the loop happens to
    arrive, and on a saturated CI core they report a correct clamp as broken.
    """

    def test_no_retry_sleep_outlives_the_budget_it_was_given(self, tmp_path):
        target = tmp_path / "held.json"
        holder = FileLock(target, timeout=5)
        assert holder.acquire(), "premise: the lock must be held to contend"
        try:
            # NOT A MULTIPLE OF THE FLAT SLEEP. With 0.5 the weakened form
            # divides evenly, every sleep lands exactly on the remainder, and
            # the mutant survives -- measured, it did. 0.45 leaves 0.05 when
            # the last 0.1 starts, which is the overshoot.
            budget = 0.45
            over: list[tuple[float, float]] = []
            lefts: list[float] = []
            real_sleep = locking.time.sleep
            real_monotonic = locking.time.monotonic
            mine = threading.get_ident()
            # THE CODE'S OWN ANCHOR. `start` taken out here sits before the
            # `mkdir` AND the `open` inside `acquire`, and this window is the
            # wider of the two: measured, 3ms of injected skew already failed
            # this case on correct code. The first `monotonic()` inside the
            # call is the deadline the clamp is read against.
            anchor: list[float] = []

            def anchoring():
                t = real_monotonic()
                if threading.get_ident() == mine and not anchor:
                    anchor.append(t)
                return t

            def recording(seconds):
                if threading.get_ident() != mine:
                    return real_sleep(seconds)
                if not anchor:
                    # Nothing has read the clock inside the call yet, so
                    # there is no deadline to measure against.
                    return real_sleep(seconds)
                left = budget - (real_monotonic() - anchor[0])
                lefts.append(left)
                # A tick of slack for the instructions between the clamp
                # reading the clock and us reading it.
                if seconds > max(left, 0.0) + 0.005:
                    over.append((seconds, left))
                return real_sleep(seconds)

            locking.time.monotonic = anchoring
            locking.time.sleep = recording
            try:
                waiter = FileLock(target, timeout=budget)
                # CLEARED HERE, not at the patch: `locking.time` IS the `time`
                # module, so anything reading the clock in between would
                # otherwise become the anchor.
                anchor.clear()
                assert not waiter.acquire(), "the held lock was handed over"
            finally:
                locking.time.sleep = real_sleep
                locking.time.monotonic = real_monotonic
            assert len(lefts) >= 2, (
                f"only {len(lefts)} sleep(s) — the instrument, not the code"
            )
            # THE DISCRIMINATING REGION. A clamp is observable only once the
            # remainder falls below the sleep the code would otherwise take;
            # loop latency can step over that window and an unclamped flat
            # 0.1 then passes with nothing saying the run had no power.
            assert min(lefts) < 0.1, (
                f"the smallest remaining budget at a sleep was "
                f"{min(lefts):.3f}s, never below the flat 0.1 this separates "
                "— the run never entered the region where the clamp shows"
            )
            assert not over, (
                f"{len(over)} sleep(s) ran past the deadline; worst slept "
                f"{max(o for o, _ in over):.3f}s with "
                f"{min(l for _, l in over):.3f}s left — the clamp used "
                "`timeout`, not what remains of it"
            )
        finally:
            holder.release()
