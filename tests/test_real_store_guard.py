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

import os
import shutil
import threading
from pathlib import Path

import pytest

from claude_swap import paths
from claude_swap.models import Platform
from tests import conftest


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


def test_rmtree_of_a_protected_root_is_refused_before_any_child_is_removed(
    tmp_path: Path, monkeypatch
):
    """C-1: ``shutil.rmtree`` walks via ``os.scandir``/``dir_fd`` and unlinks
    children by RELATIVE name (``os.remove('seq.json', dir_fd=...)``) — only
    the outermost ``os.rmdir(path)`` carries an absolute path, so hooking the
    per-child ``os.remove``/``os.rmdir`` events (as every other write shape
    in this guard does) lets every child vanish before the guard ever fires;
    the ``RealStoreWriteBlocked`` it eventually raises reports a refusal it
    did not perform. A stand-in root is registered into ``_REAL_STORE_SPECS``
    (never the real store) so this test is safe against the developer's
    actual account data regardless of isolation state.
    """
    stand_in_root = tmp_path / "claude-swap"
    stand_in_root.mkdir()
    (stand_in_root / "configs").mkdir()
    (stand_in_root / "configs" / ".claude-config-1-a@example.com.json").write_text("{}")
    (stand_in_root / "credentials").mkdir()
    (stand_in_root / "credentials" / ".creds-1-a@example.com.enc").write_text("x")
    (stand_in_root / "sequence.json").write_text("{}")

    monkeypatch.setattr(conftest, "_REAL_STORE_SPECS", ((stand_in_root, True),))

    entries_before = sorted(
        str(p.relative_to(stand_in_root)) for p in stand_in_root.rglob("*")
    )
    assert len(entries_before) == 5, entries_before

    with pytest.raises(conftest.RealStoreWriteBlocked):
        shutil.rmtree(stand_in_root)

    entries_after = (
        sorted(str(p.relative_to(stand_in_root)) for p in stand_in_root.rglob("*"))
        if stand_in_root.exists()
        else []
    )
    assert entries_after == entries_before, (
        f"guard raised but data was already gone: before={entries_before} "
        f"after={entries_after}"
    )


# -- m-1: five previously-untested guard shapes (M7/M8/M9/M11/M12) --------
#
# `test_control_b_and_c_real_store_write_is_refused` exercises exactly one
# shape (a pathlib mode-string ``write_text`` into the recursive backup
# root) — it never touches ``os.mkdir``/``os.remove``/``os.rmdir`` directly,
# never an ``os.open`` flags-only call, never a non-recursive root, and
# never the env-neutralization. Each mutation below survived a full run for
# exactly that reason.


def test_os_mkdir_and_os_remove_into_protected_root_are_refused(
    tmp_path: Path, monkeypatch
):
    """M7: narrowing ``_WRITE_EVENTS`` to ``{"open"}`` survived because
    nothing exercised ``os.mkdir``/``os.remove`` directly (only through
    ``pathlib``'s own ``open``-backed ``write_text``/``unlink``)."""
    stand_in_root = tmp_path / "claude-swap"
    stand_in_root.mkdir()
    target = stand_in_root / "sequence.json"
    target.write_text("{}")  # seeded before the guard is armed on this root

    monkeypatch.setattr(conftest, "_REAL_STORE_SPECS", ((stand_in_root, True),))

    new_dir = stand_in_root / "new_subdir"
    with pytest.raises(conftest.RealStoreWriteBlocked):
        os.mkdir(new_dir)
    assert not new_dir.exists()

    with pytest.raises(conftest.RealStoreWriteBlocked):
        os.remove(target)
    assert target.exists()


def test_os_open_flags_only_write_into_protected_root_is_refused(
    tmp_path: Path, monkeypatch
):
    """M8: the ``os.open`` flags-only branch of ``_is_write_open`` (no
    ``mode`` string — only ``flags`` says WRITE) returning ``False``
    survived because every write in the suite goes through ``pathlib``,
    which always supplies a ``mode`` string."""
    stand_in_root = tmp_path / "claude-swap"
    stand_in_root.mkdir()
    monkeypatch.setattr(conftest, "_REAL_STORE_SPECS", ((stand_in_root, True),))

    target = stand_in_root / "sequence.json"
    fd = None
    try:
        with pytest.raises(conftest.RealStoreWriteBlocked):
            fd = os.open(target, os.O_WRONLY | os.O_CREAT)
    finally:
        if fd is not None:
            os.close(fd)
    assert not target.exists()


def test_non_recursive_root_protects_only_direct_children(
    tmp_path: Path, monkeypatch
):
    """M9: collapsing the recursive/non-recursive split to always-recursive
    survived because the suite's one guard test only ever writes into a
    RECURSIVE root — a non-recursive root (``~/.claude``) must still permit
    a deeply nested write (a job worktree under ``~/.claude/jobs/...``)
    while refusing a direct child (``.credentials.json``)."""
    non_recursive_root = tmp_path / ".claude"
    deep_dir = non_recursive_root / "jobs" / "abc" / "tmp"
    deep_dir.mkdir(parents=True)  # created before the guard is armed on it

    monkeypatch.setattr(conftest, "_REAL_STORE_SPECS", ((non_recursive_root, False),))

    deep_target = deep_dir / "somefile.json"
    deep_target.write_text("ok", encoding="utf-8")  # must NOT raise
    assert deep_target.exists()

    direct_target = non_recursive_root / ".credentials.json"
    with pytest.raises(conftest.RealStoreWriteBlocked):
        direct_target.write_text("ok", encoding="utf-8")
    assert not direct_target.exists()


def test_frozen_specs_include_the_two_non_recursive_roots(monkeypatch, tmp_path):
    """M11: dropping the four non-recursive-root entries survived because
    no test inspects ``_REAL_STORE_SPECS`` directly — these two roots
    (``~/.claude``, ``$HOME``) are the ONLY protection for
    ``~/.claude/.credentials.json`` and ``~/.claude.json``."""
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    for var in ("CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)

    specs = conftest._freeze_real_store_specs()
    non_recursive_roots = {root for root, recursive in specs if not recursive}

    assert home / ".claude" in non_recursive_roots
    assert home in non_recursive_roots


def test_frozen_specs_ignore_a_developer_exported_claude_config_dir(
    monkeypatch, tmp_path
):
    """M12: removing the env-neutralization inside ``_freeze_real_store_specs``
    survived because no test exports ``CLAUDE_CONFIG_DIR`` around the call —
    a developer with it set in their normal shell would otherwise get ONLY the
    override path protected, silently dropping real ``~/.claude`` protection
    (the env-neutralization is what produces the separate DEFAULT snapshot at
    all). Both roots must be protected now: the override IS also a real
    account-store location for a developer who has it exported (the same
    both-must-be-protected reasoning the XDG_DATA_HOME fix applies) — this
    updated assertion reflects that; only the default-snapshot regression
    (dropping ``~/.claude``) is what the mutation below still needs to kill.
    """
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude").mkdir()
    elsewhere = home / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(elsewhere))

    specs = conftest._freeze_real_store_specs()
    non_recursive_roots = {root for root, recursive in specs if not recursive}

    assert home / ".claude" in non_recursive_roots, (
        "the genuinely-default ~/.claude must still be protected even with "
        "CLAUDE_CONFIG_DIR exported"
    )
    assert elsewhere in non_recursive_roots, (
        "the override path must ALSO be protected — a developer with "
        "CLAUDE_CONFIG_DIR exported has their real config home there"
    )


def test_frozen_specs_include_the_ambient_xdg_override_backup_root(
    monkeypatch, tmp_path
):
    """`_freeze_real_store_specs` clears XDG_DATA_HOME before resolving, so on
    a machine where it's exported OUTSIDE $HOME, the real account store lives
    at the override path and this snapshot never included it — the defaults
    snapshot alone is not enough. The frozen set must ALSO contain the root
    `claude_swap.paths` resolves to under the environment as it actually is."""
    home = tmp_path / "home"
    home.mkdir()
    xdg = tmp_path / "xdg-outside-home"  # deliberately NOT under `home`
    xdg.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.LINUX))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    specs = conftest._freeze_real_store_specs()
    recursive_roots = {root for root, recursive in specs if recursive}

    assert xdg / "claude-swap" in recursive_roots, (
        "the XDG-override backup root must be protected, not only the "
        "cleared-env default ~/.local/share/claude-swap"
    )
    assert home / ".local" / "share" / "claude-swap" in recursive_roots, (
        "the genuinely-default root must still be protected too"
    )


def test_layout_a_runtime_real_store_is_refused_and_unrelated_tmp_still_writes(
    monkeypatch, tmp_path
):
    """The mandatory YES/NO probe from the finding: under layout A (XDG_DATA_HOME
    exported outside $HOME), a write to the RUNTIME real store (what
    `paths.get_backup_root()` actually resolves to under this environment)
    must be refused (YES-arm) while a write to an unrelated tmp path must
    still succeed (NO-arm) — a guard that refuses everything is not a fix.
    """
    home = tmp_path / "home"
    home.mkdir()
    xdg = tmp_path / "xdg-outside-home"
    xdg.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    monkeypatch.setattr(Platform, "detect", staticmethod(lambda: Platform.LINUX))
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    # Install the specs a real conftest import would freeze under THIS
    # (simulated) ambient environment — this is what the fix under test
    # changes; the live audit hook reads `conftest._REAL_STORE_SPECS` by
    # module-global lookup at call time, so this patch governs its behavior.
    monkeypatch.setattr(conftest, "_REAL_STORE_SPECS", conftest._freeze_real_store_specs())

    runtime_real_store = paths.get_backup_root()
    assert runtime_real_store == xdg / "claude-swap"  # sanity: the hole's target

    # YES-arm: the runtime real store must be refused.
    yes_target = runtime_real_store / "sequence.json"
    with pytest.raises(conftest.RealStoreWriteBlocked):
        yes_target.parent.mkdir(parents=True, exist_ok=True)
        yes_target.write_text("{}", encoding="utf-8")
    assert not yes_target.exists()

    # NO-arm: an unrelated tmp path must still succeed (the guard isn't a
    # blanket refuse-everything).
    no_target = tmp_path / "unrelated" / "file.txt"
    no_target.parent.mkdir(parents=True, exist_ok=True)
    no_target.write_text("ok", encoding="utf-8")
    assert no_target.read_text(encoding="utf-8") == "ok"
