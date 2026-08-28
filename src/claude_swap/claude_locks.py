"""Cooperate with Claude Code's own advisory locks while mutating its files.

Claude Code guards its OAuth token refresh with the npm ``proper-lockfile``
package, and its ``~/.claude.json`` writes with the same mechanism on the
config file. The protocol (verified against the 2.1.218 bundle):

- The lock artifact is a **directory**; ``mkdir`` atomicity is the mutex.
- The refresh path takes **two** locks, in order: the primary
  ``<config-home>/.oauth_refresh.lock``, then the legacy
  ``<config-home>.lock`` (``~/.claude.lock``) kept for compatibility with
  external tools. Both run ``stale: 60000, update: 5000`` — a credential
  lock is stale only past **60s**, and live holders touch every 5s. On a
  contended legacy lock Claude Code releases the primary and retries.
- The config lock (``~/.claude.json.lock``) keeps the older defaults:
  stale after 10s, touched every 5s.
- Claude Code retries a held credentials lock 5 times with 1-2s jittered
  sleeps before giving up, so briefly holding it is fully cooperative.

Holding these locks while swapping credentials closes the one real race with a
running Claude Code: its refresh reads credentials, refreshes over the network,
and saves — all under both credential locks — so a swap landing inside that
window would be overwritten by the refreshed old-account token (and the
just-taken backup would keep a pre-rotation refresh token). Under the lock,
Claude Code's own double-checked re-read sees the swapped (non-expired)
credential and aborts the refresh instead.

References (claude-code 2.1.218 bundle): the ``uKi`` lock-options helper
(``lockfilePath: join(dir, ".oauth_refresh.lock"), stale: 60000, update:
5000``) and ``CKi`` (dual acquisition, legacy released-on-contention with
``tengu_oauth_refresh_legacy_lock_contended`` telemetry).
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from contextlib import contextmanager

from claude_swap.exceptions import LockError
from claude_swap.locking import FileLock
from pathlib import Path

from claude_swap.exceptions import ClaudeCodeLockTimeout
from claude_swap.paths import get_claude_config_home, get_global_config_path

# Claude Code's credential-refresh locks run ``stale: 60000, update: 5000``
# (2.1.218 ``uKi``): a lock younger than 60s belongs to a live holder and
# must never be stolen — the holder's toucher may stall well past 10s
# (suspend, blocked event loop) while it still legitimately owns the lock.
CREDENTIALS_STALENESS_S = 60.0
# The config lock (~/.claude.json.lock) keeps the older proper-lockfile
# defaults: stale after 10s, touched every 5s.
CONFIG_STALENESS_S = 10.0
# We touch a little faster than CC's 5s for margin.
TOUCH_INTERVAL_S = 3.0
# Claude Code holds the credentials lock for one token-endpoint round trip
# (sub-second to a few seconds); its config lock for a local RMW. 9s of
# bounded waiting comfortably outlasts both without stalling the CLI forever.
# Note this is a PER-LOCK budget: claude_credentials_lock acquires two locks
# sequentially, so its worst case is ~2x this value.
DEFAULT_TIMEOUT_S = 9.0

_logger = logging.getLogger("claude-swap")


def credentials_lock_dir() -> Path:
    """Legacy credential lock (``~/.claude.lock``) — CC still takes it for
    compatibility; external exclusion today rests on this one."""
    home = get_claude_config_home()
    return home.parent / (home.name + ".lock")


def oauth_refresh_lock_dir() -> Path:
    """Claude Code's primary OAuth refresh lock
    (``<config-home>/.oauth_refresh.lock``, 2.1.218+)."""
    return get_claude_config_home() / ".oauth_refresh.lock"


def config_lock_dir() -> Path:
    """Lock directory guarding the global config file (``~/.claude.json.lock``)."""
    path = get_global_config_path()
    return path.parent / (path.name + ".lock")


# A CAP on the guard wait; the caller's remaining budget is the bound, so the
# product tolerates any value. Its contended-guard test does not -- small
# enough, the clamped and unclamped forms stop separating by more than the
# margin that test allows, and its premise refuses rather than passing blind.
_TAKEOVER_GUARD_S = 0.5


def _take_over_stale(
    lock_dir, staleness: float, budget: float = _TAKEOVER_GUARD_S
) -> bool:
    """Remove a lock whose holder is gone, but never a successor's.

    `os.stat` decides and `os.rmdir` acts, and between them another waiter
    can win the same race and create ITS lock at this name -- which the
    rmdir then removes, putting two processes inside the critical section
    at once. Measured on this tree: 7 of 320 holds saw a co-holder with a
    stale lock present, 0 of 320 without one, and widening only that gap in
    one waiter made it dominant (30 of 160 at 50 ms).

    The window is serialized on an flock -- the one primitive here a peer
    cannot steal -- and the staleness is re-read while holding it, so a
    directory a peer already retook is left alone. Claude Code performs the
    same takeover and takes no lock of ours, so this closes the race
    between cswap processes and narrows, but cannot close, the
    cross-implementation one.

    `budget` is the caller's remaining time; the guard waits the smaller of it
    and `_TAKEOVER_GUARD_S`, so a contended guard cannot push the caller past
    its own deadline. Returns whether the corpse was removed -- False also
    covers a peer having retaken it, a refused rmdir, and an exhausted budget,
    which mean the same thing here: back off and retry.
    """
    guard = lock_dir.parent / f"{lock_dir.name}.takeover"
    try:
        with FileLock(guard, timeout=max(0.0, min(_TAKEOVER_GUARD_S, budget))):
            try:
                if time.time() - os.stat(lock_dir).st_mtime <= staleness:
                    return False  # a peer retook it; it is not ours to remove
            except FileNotFoundError:
                return False  # already gone; the mkdir above will race for it
            os.rmdir(lock_dir)
            return True
    except (LockError, OSError):
        return False


@contextmanager
def proper_lockfile(
    lock_dir: Path,
    *,
    timeout: float | None = None,
    staleness: float = CONFIG_STALENESS_S,
):
    """Acquire a proper-lockfile-compatible directory lock.

    Blocks up to ``timeout`` seconds (default ``DEFAULT_TIMEOUT_S``, resolved
    at call time so tests can shorten it), taking over locks whose mtime is
    older than ``staleness``, touches the directory mtime while held so other
    holders don't deem us stale, and removes it on exit.

    Raises:
        ClaudeCodeLockTimeout: The lock stayed held past ``timeout``.
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_S
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start + timeout
    while True:
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            pass
        if time.monotonic() - start > timeout:
            raise ClaudeCodeLockTimeout(
                f"Could not acquire {lock_dir.name} — Claude Code appears "
                "to be refreshing credentials. Retry in a few seconds."
            )
        try:
            held_mtime = os.stat(lock_dir).st_mtime
        except FileNotFoundError:
            # BACK OFF HERE TOO. A dangling symlink at the lock path answers
            # FileExistsError to mkdir and FileNotFoundError to stat, so this
            # arm can repeat for the whole budget. No claim here about which
            # OTHER arms sleep: a sibling adds one above, and that sentence
            # goes false on the merged tree.
            time.sleep(max(0.0, min(0.05, deadline - time.monotonic())))
            continue
        if time.time() - held_mtime > staleness:
            # Dead holder per the protocol: remove and retake. Declining --
            # a peer retook it, or the corpse could not be removed -- must
            # not spin hot, and must not sleep past the deadline.
            if not _take_over_stale(
                lock_dir, staleness, budget=deadline - time.monotonic()
            ):
                time.sleep(max(0.0, min(0.05, deadline - time.monotonic())))
            continue
        # Clamped to the remaining budget, so `timeout` bounds the whole
        # call: the deadline check above cannot fire while a full-length
        # sleep is still running past it. With budget to spare the clamp is a
        # no-op and the jitter still spreads waiters apart.
        time.sleep(max(0.0, min(0.25 + random.random() * 0.25, deadline - time.monotonic())))

    stop_touching = threading.Event()
    warned = False
    last_ok = time.time()

    def _touch() -> None:
        nonlocal warned, last_ok
        while not stop_touching.wait(TOUCH_INTERVAL_S):
            try:
                os.utime(lock_dir)
                # THE LATCH IS PER FREEZE, NOT PER HOLD. A refresh that lands
                # ends the episode the warning describes, so the next one that
                # outlives `staleness` is a new fact and the takeover it
                # precedes would otherwise arrive unexplained.
                last_ok, warned = time.time(), False
                continue
            except FileNotFoundError:
                return  # gone; nothing left to keep alive
            except OSError as e:
                # Transient, so stay armed: absence is terminal and every
                # other errno is not. Re-checking the path to tell them apart
                # is a second syscall that can fail the same way, and from
                # 3.14 one that reads absence out of a permission error.
                #
                # WARN ONLY ONCE THE SENTENCE IS TRUE. It describes a freeze
                # that outlives `staleness`; firing on the first failure
                # announced imminent theft over a lock that stayed fresh.
                if not warned and time.time() - last_ok > staleness:
                    warned = True
                    _logger.warning(
                        "Could not refresh %s (%s); its mtime stops advancing, "
                        "so a waiter may take it over as stale",
                        lock_dir,
                        e,
                    )

    toucher = threading.Thread(target=_touch, daemon=True)
    toucher.start()
    try:
        yield
    finally:
        stop_touching.set()
        toucher.join(timeout=1.0)
        try:
            os.rmdir(lock_dir)
        except FileNotFoundError:
            _logger.warning(
                "Lock %s vanished while held (taken over as stale?)", lock_dir
            )
        except OSError as e:
            _logger.warning("Failed to release lock %s: %s", lock_dir, e)


@contextmanager
def claude_credentials_lock(*, timeout: float | None = None):
    """Hold Claude Code's credential-refresh locks, in CC's own order.

    2.1.218 takes ``<config-home>/.oauth_refresh.lock`` first, then the
    legacy ``~/.claude.lock``; on legacy contention it releases the primary
    before retrying. Mirroring both the pair and the order means a waiting
    cswap and a waiting Claude Code can never deadlock against each other,
    and exclusion holds even after CC drops the legacy lock. Both use CC's
    60s staleness — never steal a lock a live CC may still hold.
    """
    with (
        proper_lockfile(
            oauth_refresh_lock_dir(),
            timeout=timeout,
            staleness=CREDENTIALS_STALENESS_S,
        ),
        proper_lockfile(
            credentials_lock_dir(),
            timeout=timeout,
            staleness=CREDENTIALS_STALENESS_S,
        ),
    ):
        yield


@contextmanager
def claude_config_lock(*, timeout: float | None = None):
    """Hold Claude Code's global-config write lock (``~/.claude.json.lock``)."""
    with proper_lockfile(config_lock_dir(), timeout=timeout):
        yield
