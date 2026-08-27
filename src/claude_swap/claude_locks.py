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
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
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
# WINDOWS CANNOT HOLD A DIRECTORY OPEN: `os.open` on one raises EACCES, and
# the stdlib offers no other way. Identity there is the file index from a
# plain stat -- still not a value we write, so still not adoptable the way an
# mtime stamp was, but not pinned, so an index a takeover frees can come back.
_CAN_PIN_A_DIRECTORY = sys.platform != "win32"
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
    holders don't deem us stale, and removes it on exit — unless it was taken
    over meanwhile, in which case the successor's lock is left in place.

    Raises:
        ClaudeCodeLockTimeout: The lock stayed held past ``timeout``.
        FileNotFoundError: The lock's PARENT is gone. Retrying cannot make a
            directory under a directory nobody has, so that errno is the one
            case this re-raises instead of waiting out the budget.
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT_S
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    held_fd = -1
    start = time.monotonic()
    while True:
        try:
            os.mkdir(lock_dir)
            # HOLD A DESCRIPTOR ON THE DIRECTORY WE MADE, and take identity
            # from it. An mtime is a value we WRITE, so a successor's
            # directory can come to carry ours; (st_dev, st_ino) is the object
            # itself. The open is what makes it decisive -- an unheld inode
            # number is reused by the next mkdir, while an open fd pins it in
            # the orphan list so mkdir cannot get it back.
            #
            # Open HERE, not after the loop. A waiter that judged this lock
            # stale does stat-then-rmdir, and the holder can release and we
            # can take the name in that gap -- its rmdir then removes the
            # directory we just made. Outside the loop that raises
            # FileNotFoundError out of a call documented to raise only
            # ClaudeCodeLockTimeout.
            if _CAN_PIN_A_DIRECTORY:
                held_fd = os.open(lock_dir, os.O_RDONLY)
                st = os.fstat(held_fd)
            else:
                st = os.stat(lock_dir)
            ident = (st.st_dev, st.st_ino)
            last_stamp = st.st_mtime_ns
            break
        except FileExistsError:
            pass
        except FileNotFoundError:
            # ONLY A SWEPT NAME RETRIES. The same errno arrives when the
            # PARENT is gone -- removed or replaced after the `parents=True`
            # above -- and that can never succeed, so the fall-through became a
            # full-budget 100%-CPU spin (measured ~95,000 mkdir/s) ending in a
            # timeout that blames Claude Code for a directory nobody has.
            # The parent existing is what separates them, and it is one stat.
            if not lock_dir.parent.is_dir():
                raise
            # Swept between the two calls. Fall THROUGH to the deadline below,
            # never back to the top: a name swept on every attempt has to end
            # at the budget rather than spin until the sweeper stops.
            #
            # AND WAIT ON THE WAY. The fall-through reaches the deadline, then
            # stats a name that is gone and `continue`s, so it never reaches
            # the jittered sleep at the bottom -- ending at the budget while
            # pinning a core for all of it. The ordinary "holder released
            # between mkdir and stat" retry stays instant; only a name being
            # swept out from under us backs off.
            time.sleep(0.05)
        except OSError:
            # `mkdir` made the directory and the open could not read it back,
            # so nobody holds it and every waiter is blocked for the full
            # staleness window. Retrying cannot clear this errno; releasing
            # the name can.
            try:
                os.rmdir(lock_dir)
            except OSError:
                pass
            raise
        if time.monotonic() - start > timeout:
            raise ClaudeCodeLockTimeout(
                f"Could not acquire {lock_dir.name} — Claude Code appears "
                "to be refreshing credentials. Retry in a few seconds."
            )
        try:
            held_mtime = os.stat(lock_dir).st_mtime
        except FileNotFoundError:
            continue  # holder released between mkdir and stat; retry now
        if time.time() - held_mtime > staleness:
            # Dead holder per the protocol: remove and retake. Losing the
            # rmdir/mkdir race to another waiter just means looping again.
            try:
                os.rmdir(lock_dir)
            except OSError:
                time.sleep(0.05)  # can't remove it either; don't spin hot
            continue
        time.sleep(0.25 + random.random() * 0.25)

    stop_touching = threading.Event()
    # UNPINNED ONLY. Where the inode can be handed back, `last_stamp` is a
    # read-modify-write the release also reads, and half-done -- utime landed,
    # read-back has not -- our own lock reads foreign. Pinned, identity is
    # immutable and there is nothing to serialise, so this costs nothing.
    stamping = threading.Lock() if not _CAN_PIN_A_DIRECTORY else nullcontext()

    def _ours() -> bool:
        """Is the directory at this path still the one we created?

        Raises whatever the stat raises; each caller decides what an errno
        means for it.

        The mtime is a SECOND witness, and only where the first one is not
        decisive. Unpinned, a takeover's `mkdir` can be handed the inode
        number back, and then the index alone says ours about a successor's
        directory. Both readers treat a mismatch as "not ours", whose cost is
        a lock left for the stale sweep -- so a stamp read mid-refresh is
        safe, and there is nothing here for a mutex to protect.
        """
        st = os.stat(lock_dir)
        if (st.st_dev, st.st_ino) != ident:
            return False
        return _CAN_PIN_A_DIRECTORY or st.st_mtime_ns == last_stamp

    # WINDOWS HAS NO FIX HERE, and this says so rather than implying one. The
    # stamp above narrows the window; it cannot close it, because the tick's
    # own read-back adopts whatever the path now carries.

    def _touch() -> None:
        nonlocal last_stamp
        # ABSENCE IS TERMINAL; EVERY OTHER ERRNO IS TRANSIENT. One `except
        # OSError: return` over both syscalls meant a single EIO or ESTALE --
        # the ordinary errnos on a network `~/.claude` -- ended the heartbeat
        # for the rest of the hold, and a mtime that stops advancing is a lock
        # a waiter may take over as stale, mid-swap.
        while not stop_touching.wait(TOUCH_INTERVAL_S):
            with stamping:
                try:
                    if not _ours():
                        return  # taken over; refreshing it keeps THEIR lock alive
                except FileNotFoundError:
                    return  # gone; nothing left to keep alive
                except OSError:
                    continue  # unreadable this tick, not stolen
                try:
                    os.utime(lock_dir)
                except FileNotFoundError:
                    return
                except OSError:
                    continue
                if not _CAN_PIN_A_DIRECTORY:
                    # A read-back that fails leaves the stamp stale, so the next
                    # tick and the release both read "not ours" and leave the
                    # lock. That is the safe direction, which is why no errno
                    # here needs its own arm.
                    try:
                        last_stamp = os.stat(lock_dir).st_mtime_ns
                    except OSError:
                        pass

    toucher = threading.Thread(target=_touch, daemon=True)
    toucher.start()
    try:
        yield
    finally:
        stop_touching.set()
        # NO WAIT HERE. Pinned, identity is immutable, so the release needs
        # nothing from the heartbeat: `stamping` is a nullcontext and a tick
        # landing inside this block can at worst refresh a successor's lock
        # once. It can never make one look like ours, which is what a stamp
        # could and did. Unpinned, the mutex is what keeps the release from
        # reading `last_stamp` half-written -- one wait, and only there.
        #
        # NOT `return` ON ANY ARM: this whole block is the context manager's
        # `finally`, and a `return` there discards an exception the body
        # raised.
        try:
            with stamping:
                if _ours():
                    os.rmdir(lock_dir)
                else:
                    # A successor's critical section would be left with nothing on
                    # disk, free for a third waiter to take.
                    _logger.warning(
                        "Lock %s was taken over while held; leaving it", lock_dir
                    )
        except FileNotFoundError:
            _logger.warning(
                "Lock %s vanished while held (taken over as stale?)", lock_dir
            )
        except OSError as e:
            _logger.warning("Failed to release lock %s: %s", lock_dir, e)
        finally:
            # LAST, so the inode stays pinned across the decision above. Close
            # it first and a takeover between the stat and the rmdir could
            # reuse the number, which is the whole hole this replaced.
            if held_fd >= 0:
                os.close(held_fd)

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
