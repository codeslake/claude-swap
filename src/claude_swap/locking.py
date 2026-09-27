"""File locking for concurrent access protection."""

from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path
from typing import IO, Iterator

# Platform-specific imports for file locking
if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from claude_swap.exceptions import LockError

#: How long `add`, `add-token`, `remove`, `import` and `export` wait for the
#: account store lock (the switcher's ``lock_file``). Every other holder
#: (switch, swap, move, the usage-refresh persists, the consume gate) holds it
#: for local I/O or one bounded profile fetch, so a wait this long times out
#: only on a stuck holder, and a user's `add` never fails because a switch was
#: running.
STORE_LOCK_WAIT_S = 60.0


def lock_holders(lock_path: Path) -> list[str]:
    """Name the processes holding a ``flock`` on ``lock_path``.

    Each entry reads ``pid N (command name)``. The answer comes from
    ``/proc/locks``, so it exists on Linux only; elsewhere, or when the lock
    file or ``/proc/locks`` cannot be read, the list is empty and the caller
    says the holder is unidentified. Waiters (the ``->`` rows) are not
    holders and are left out.
    """
    if not sys.platform.startswith("linux"):
        return []
    try:
        st = lock_path.stat()
        rows = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return []
    inode = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    holders = []
    for row in rows:
        fields = row.split()
        if len(fields) < 6 or fields[1] != "FLOCK" or fields[5] != inode:
            continue
        pid = fields[4]
        try:
            name = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            name = "exited"
        holders.append(f"pid {pid} ({name})")
    return holders


class FileLock:
    """Cross-process file lock using platform-specific APIs."""

    def __init__(self, lock_path: Path, timeout: float = 10.0):
        self.lock_path = lock_path
        self.timeout = timeout
        self._lock_file: IO | None = None
        self._locked = False

    def acquire(self, timeout: float | None = None) -> bool:
        """Acquire exclusive lock with timeout.

        Args:
            timeout: Maximum seconds to wait for lock. Defaults to the
                timeout given at construction.

        Returns:
            True if lock acquired, False if timeout.
        """
        if timeout is None:
            timeout = self.timeout
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_file = open(self.lock_path, "w")

        start = time.monotonic()
        while True:
            try:
                if sys.platform == "win32":
                    # Windows: use msvcrt for file locking
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    # POSIX: use fcntl for file locking
                    fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._locked = True
                return True
            except (BlockingIOError, OSError):
                if time.monotonic() - start > timeout:
                    self._lock_file.close()
                    self._lock_file = None
                    return False
                time.sleep(0.1)

    def release(self) -> None:
        """Release the lock."""
        if self._lock_file and self._locked:
            if sys.platform == "win32":
                # Windows: unlock using msvcrt
                try:
                    msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass  # File may already be unlocked
            else:
                # POSIX: unlock using fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
            self._locked = False

    def __enter__(self) -> FileLock:
        if not self.acquire():
            raise LockError("Failed to acquire lock - another instance may be running")
        return self

    def __exit__(self, *args) -> None:
        self.release()

    @contextlib.contextmanager
    def held_for(self, action: str) -> Iterator[FileLock]:
        """Hold the lock for ``action``, waiting up to ``self.timeout``.

        A timeout raises ``LockError`` before ``action`` has changed anything,
        and the message says so, naming the lock file and the process holding
        it, so whoever ran ``action`` can tell a slow holder from a stuck one.
        """
        if not self.acquire():
            holders = lock_holders(self.lock_path)
            held_by = ", ".join(holders) if holders else "an unidentified process"
            raise LockError(
                f"{action}: waited {self.timeout:g}s for the account store lock "
                f"{self.lock_path}, held by {held_by}; nothing was changed. "
                f"Retry once it is released."
            )
        try:
            yield self
        finally:
            self.release()
