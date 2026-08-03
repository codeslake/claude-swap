"""A test process must be structurally unable to write the REAL account
store — not merely warned not to, because a guard that unwinds (any
``patch``/``monkeypatch`` fixture) is gone by the time a background thread
that outlived its own test's teardown gets around to writing.

Measured incident: ``sequence.json``/``credentials/*.enc`` under the real
``~/.local/share/claude-swap`` were overwritten at 03:26 with the exact
``a@example.com``/``b@example.com`` pair ``EngineHarness.seed`` (this repo,
``tests/test_autoswitch.py``) writes. ``tests/conftest.py``'s ``temp_home``
and ``_isolate_real_home`` fixtures use ``patch.dict``/``monkeypatch`` as
context managers, which unwind at teardown; a thread started inside a test
that survives past that teardown sees the REAL ``$HOME`` (``pathlib.Path.home``
is process-global, not thread-local), because the patch is gone by the time
it runs.

The fix under test: ``conftest.py`` installs a process-global
``sys.addaudithook`` (module import time, no removal API — it cannot be
unwound the way a fixture patch can) that refuses any WRITE-mode ``open``/
``os.rename``/``os.mkdir``/``os.remove``/``os.rmdir`` whose target resolves,
AT THE MOMENT OF THE CALL, under the REAL (currently-computed, not cached)
``claude_swap.paths`` roots — regardless of which thread performs it.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from claude_swap import paths


def test_control_a_tmp_path_write_is_allowed(tmp_path: Path):
    """CONTROL A: the guard must not block everything — a legitimate write
    to pytest's own isolated tmp_path must still succeed."""
    target = tmp_path / "control-a-allowed.txt"
    target.write_text("ok", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "ok"


def test_control_b_and_c_real_store_write_is_refused(monkeypatch):
    """CONTROL B (main thread) and CONTROL C (a thread that outlives its
    own test's isolation, the case that actually matters) both attempt a
    write under the REAL ``claude_swap.paths.get_backup_root()`` and must
    both be refused — never silently succeed, never silently no-op.

    ``monkeypatch.undo()`` reverses the autouse ``_isolate_real_home``
    fixture's patches (it shares this test's ``monkeypatch`` instance —
    the same mechanism ``test_move_strict_clear_fails_closed_on_locked_
    keychain`` already uses elsewhere in this suite), exposing the TRUE,
    unpatched ``$HOME``/``Path.home()`` for the rest of this test body —
    exactly the state a thread sees after its own test's teardown has run.
    """
    from claude_swap.exceptions import ClaudeSwitchError  # noqa: F401  (sanity import only)

    marker_name = ".cswap-test-real-store-guard-probe-DELETE-ME"

    monkeypatch.undo()  # expose the REAL, unpatched HOME from here on

    real_backup_root = paths.get_backup_root()
    real_marker = real_backup_root / marker_name
    if real_marker.exists():
        real_marker.unlink()  # defensive: a prior failed run left one behind

    # -- CONTROL B: main thread --------------------------------------
    outcome_main: dict = {}
    try:
        real_marker.write_text("probe\n", encoding="utf-8")
        outcome_main["wrote"] = True
    except PermissionError as e:
        outcome_main["wrote"] = False
        outcome_main["error"] = e

    try:
        assert outcome_main["wrote"] is False, (
            "CONTROL B FAILED: a main-thread write to the REAL backup root "
            f"({real_marker}) was not refused"
        )
        assert not real_marker.exists(), (
            "the write was reported as refused but the file exists anyway"
        )
    finally:
        if real_marker.exists():
            real_marker.unlink()  # never leave real-store litter, pass or fail

    # -- CONTROL C: a thread that outlives its own test's teardown ---
    # (the case the incident actually was: a thread started while isolation
    # was active, but not joined before the isolating patches unwound).
    release = threading.Event()
    outcome_thread: dict = {}

    def leaked_write():
        release.wait(timeout=5)
        target = paths.get_backup_root() / marker_name
        try:
            target.write_text("probe\n", encoding="utf-8")
            outcome_thread["wrote"] = True
        except PermissionError as e:
            outcome_thread["wrote"] = False
            outcome_thread["error"] = e

    t = threading.Thread(target=leaked_write, daemon=True)
    t.start()
    release.set()
    t.join(timeout=5)

    try:
        assert not t.is_alive(), "the probe thread did not finish in time"
        assert "wrote" in outcome_thread, "the probe thread never reached its write"
        assert outcome_thread["wrote"] is False, (
            "CONTROL C FAILED (the case that matters): a background thread's "
            f"write to the REAL backup root ({real_marker}) was not refused"
        )
        assert not real_marker.exists(), (
            "the thread's write was reported as refused but the file exists anyway"
        )
    finally:
        if real_marker.exists():
            real_marker.unlink()
