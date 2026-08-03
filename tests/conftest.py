"""Pytest fixtures for Claude Switch tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import macos_keychain as _macos_keychain
from claude_swap import paths as _paths


class RealStoreWriteBlocked(PermissionError):
    """A test process tried to write the REAL account store.

    Raised from a process-global ``sys.addaudithook`` — installed once,
    below, with no removal API (by CPython design: an audit hook cannot be
    uninstalled short of process exit). That permanence is the point: the
    fixture-based isolation (``_isolate_real_home``, ``temp_home``) uses
    ``patch.dict``/``monkeypatch`` as context managers, which unwind at test
    teardown. A thread started inside a test that outlives its own test's
    teardown (this repo's tests spawn threads — the auto-switch loop, the
    consume-gate lock toucher) sees the REAL, unpatched ``$HOME`` by the
    time it gets around to writing, because ``pathlib.Path.home`` and
    ``os.environ`` are process-global, not thread-local, and the patch that
    protected it is already gone. A guard that itself unwinds cannot stop
    that; this one cannot unwind.

    Measured incident: the real ``~/.local/share/claude-swap/sequence.json``
    and ``credentials/*.enc`` were overwritten with the exact
    ``a@example.com``/``b@example.com`` pair ``EngineHarness.seed`` (this
    repo, ``tests/test_autoswitch.py``) writes.
    """


def _freeze_real_store_specs() -> tuple[tuple[Path, bool], ...]:
    """Snapshot the REAL (non-test) account-store roots EXACTLY ONCE, here,
    at conftest import time — before any fixture has ever touched
    ``$HOME``/``CLAUDE_CONFIG_DIR``/``XDG_DATA_HOME``.

    This snapshot, not a live re-resolution, is what the audit hook always
    compares candidate write targets against. Re-resolving ``claude_swap.
    paths`` at CHECK time (this guard's first draft) is tautological and
    wrong: during an isolated test those functions correctly resolve to the
    test's OWN tmp directory, so "does this write's target match what
    paths.* resolves to right now" is true for every legitimate isolated
    write too — measured directly, it blocked ``mock_claude_config``'s own
    fixture write to its own ``temp_home``. A FROZEN reference is what makes
    the exemptions (``temp_home``, ``CLAUDE_CONFIG_DIR``/``XDG_DATA_HOME``
    overrides) fall out for free: their resolved paths are tmp paths that
    simply never equal the frozen real ones, no special-casing needed. Only
    once a test's isolation has unwound — or the rare test that constructs a
    switcher without ``temp_home`` at all — does ``paths.*`` resolve back to
    something matching this frozen set, which is exactly the moment a write
    must be refused.

    Mirrors ``_isolate_real_home``'s notion of "real": ``CLAUDE_CONFIG_DIR``/
    ``CLAUDE_SECURESTORAGE_CONFIG_DIR``/``XDG_DATA_HOME`` are cleared for this
    one resolution (then restored), so a developer who happens to have one of
    those exported in their normal shell still gets the genuinely-default
    roots protected — the same three vars that fixture neutralizes for every
    test, regardless of ``temp_home``.

    ``recursive=True`` roots (the cswap backup root, current-XDG and legacy)
    are exclusively cswap's own data — everything beneath them is protected,
    any depth.

    ``recursive=False`` roots are directories cswap shares with unrelated
    machinery — notably ``~/.claude``, which also holds Claude Code CLI's
    OWN job/worktree/project state (this worktree itself lives under
    ``~/.claude/jobs/...``, several directories deep) and can contain a
    ``.pytest_cache``. Only DIRECT children are protected — exactly the
    files ``claude_swap.paths``/``claude_locks`` ever place directly inside
    these dirs (``.credentials.json``, ``.config.json``,
    ``.oauth_refresh.lock``, sibling atomic-write tempfiles, and — via
    ``get_global_config_path().parent``, i.e. ``$HOME`` itself when no
    ``CLAUDE_CONFIG_DIR`` override is set — ``~/.claude.json``,
    ``~/.claude.json.lock``, ``~/.claude.lock``). A deeply nested path (a
    job's worktree several directories under ``~/.claude/jobs/...``) is
    correctly left alone.
    """
    saved_env = {
        k: os.environ.get(k)
        for k in ("CLAUDE_CONFIG_DIR", "CLAUDE_SECURESTORAGE_CONFIG_DIR", "XDG_DATA_HOME")
    }
    for k in saved_env:
        os.environ.pop(k, None)
    try:
        return (
            (_paths.get_backup_root(), True),
            (_paths.get_legacy_backup_root(), True),
            (_paths.get_claude_config_home(), False),
            (_paths.get_default_claude_config_home(), False),
            (_paths.get_global_config_path().parent, False),
            (_paths.get_default_global_config_path().parent, False),
        )
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_REAL_STORE_SPECS = _freeze_real_store_specs()


# Cheap pre-filter substrings: every protected root's basename contains one
# of these, OR the target is a direct child of $HOME (recursive=False roots
# resolve to $HOME itself when CLAUDE_CONFIG_DIR is unset — ~/.claude.json
# lives there). Checked before any Path resolution, so the overwhelming
# majority of audit events (Python's own imports, pytest's internals,
# unrelated stdlib file activity — the guard fires on EVERY "open"
# system-wide for the whole test run) are rejected with a plain substring
# scan, not a function call into claude_swap.paths.
_REAL_STORE_HINTS = (".claude", "claude-swap")

_WRITE_EVENTS = frozenset({"open", "os.rename", "os.mkdir", "os.remove", "os.rmdir"})


def _is_write_open(mode: str | None, flags: int) -> bool:
    """Whether an ``open`` audit event represents a WRITE (not a plain read).

    ``mode`` is set for the ``io.open``/``pathlib`` path (e.g. ``"w"``,
    ``"r+"``); ``os.open`` calls pass ``mode=None`` and only ``flags``, so
    both must be checked — mirrors the exact two shapes measured via
    ``sys.addaudithook`` probing (``('path', 'w', flags)`` vs.
    ``('path', None, flags)``).
    """
    if mode is not None:
        return any(c in mode for c in ("w", "a", "x", "+"))
    # `os.O_ACCMODE` is POSIX-only — it does not exist on Windows, where this
    # raised AttributeError from inside the hook and took pytest down at
    # collection with an INTERNALERROR (measured: CI test-windows on a5c4b61,
    # while the same tree was 1845-green on Linux). Derive the mask from
    # O_WRONLY|O_RDWR, both of which every platform defines, instead of
    # importing a constant only some platforms have.
    _accmode = os.O_WRONLY | os.O_RDWR
    if (flags & _accmode) != os.O_RDONLY:
        return True
    return bool(flags & (os.O_CREAT | os.O_TRUNC | os.O_APPEND))


def _real_store_audit_hook(event: str, args: tuple) -> None:
    if event not in _WRITE_EVENTS:
        return
    if event == "open":
        path, mode, flags = args
        if isinstance(path, int) or not _is_write_open(mode, flags):
            return  # an already-open fd, or a plain read
        candidates = (path,)
    elif event == "os.rename":
        # Covers os.replace too (same underlying audit event) — args are
        # (src, dst, src_dir_fd, dst_dir_fd); only the DESTINATION matters,
        # a rename FROM inside a protected root but landing outside it is
        # not a write into the store.
        candidates = (args[1],)
    else:  # os.mkdir / os.remove / os.rmdir
        candidates = (args[0],)

    for candidate in candidates:
        if isinstance(candidate, (bytes, int)) or candidate is None:
            continue
        if not any(hint in candidate for hint in _REAL_STORE_HINTS):
            continue  # cheap reject — the common case
        target = Path(candidate)
        if not target.is_absolute():
            continue  # relative paths under a protected root never occur here
        for root, recursive in _REAL_STORE_SPECS:
            hit = (
                target == root or root in target.parents
                if recursive
                else target == root or target.parent == root
            )
            if hit:
                raise RealStoreWriteBlocked(
                    f"{event} refused: {target} is under the REAL account "
                    f"store ({root}), not an isolated test tmp_path. If this "
                    "test needs the real store, it doesn't — fix the "
                    "isolation instead of the guard."
                )


sys.addaudithook(_real_store_audit_hook)


class _KeychainStore:
    """In-memory ``(service, account) -> secret`` map standing in for the real
    macOS Keychain so unit tests never shell out to ``security`` or ``keyring``."""

    def __init__(self) -> None:
        self.data: dict[tuple[str, str], str] = {}

    # Mirrors the ``macos_keychain`` (security CLI) contract.
    def get_password(self, service: str, account: str) -> str | None:
        return self.data.get((service, account))

    def item_exists(self, service: str, account: str) -> bool:
        return (service, account) in self.data

    def set_password(self, service: str, account: str, password: str) -> None:
        self.data[(service, account)] = password

    def delete_password(self, service: str, account: str) -> None:
        self.data.pop((service, account), None)  # absent = no-op (rc 44)


def _make_fake_keyring() -> types.ModuleType:
    """Build an in-memory stand-in for the ``keyring`` module (which would hit the
    real Keychain on macOS) for code paths that lazily ``import keyring``."""

    class _Errors:
        class PasswordDeleteError(Exception):
            pass

        class PasswordSetError(Exception):
            pass

        class KeyringError(Exception):
            pass

    store: dict[tuple[str, str], str] = {}
    mod = types.ModuleType("keyring")
    mod.errors = _Errors  # type: ignore[attr-defined]

    def get_password(service: str, username: str):
        return store.get((service, username))

    def set_password(service: str, username: str, password: str) -> None:
        store[(service, username)] = password

    def delete_password(service: str, username: str) -> None:
        if (service, username) not in store:
            raise _Errors.PasswordDeleteError("not found")
        del store[(service, username)]

    mod.get_password = get_password  # type: ignore[attr-defined]
    mod.set_password = set_password  # type: ignore[attr-defined]
    mod.delete_password = delete_password  # type: ignore[attr-defined]
    return mod


@pytest.fixture(autouse=True)
def _isolate_real_home(request, tmp_path_factory, monkeypatch):
    """Safety net: no test may read or write the developer's real ``$HOME``.

    Some tests (CLI/TUI argument tests that call ``main()``, etc.) construct a real
    ``ClaudeAccountSwitcher`` without the ``temp_home`` fixture. Without isolation
    that switcher resolves to the real ``~/.claude-swap-backup`` — writing logs,
    running data migrations, and reading the real account list. Redirect ``$HOME``
    to a throwaway dir unless the test already uses ``temp_home`` (which sets its
    own). Runs first (autouse, before the keychain guard and other fixtures).

    Exempt the ``tmp_keychain`` fixture too: the macOS-CI integration tests that
    use it drive the real ``security`` CLI (``default-keychain`` /
    ``list-keychains``), which needs the real ``$HOME`` to locate
    ``~/Library/Keychains``. An isolated ``$HOME`` makes those commands fail. The
    fixture itself swaps the default keychain to a throwaway one and restores it.

    Always neutralize ``CLAUDE_CONFIG_DIR`` and ``XDG_DATA_HOME`` (even for
    ``temp_home`` tests): both bypass ``$HOME`` in path resolution
    (``paths.get_global_config_path``/``get_backup_root``), so a developer with
    either exported could otherwise have tests read/write real Claude config or
    backup paths — and on macOS that leads back to the real Keychain. Tests that
    exercise those vars set them explicitly, overriding this.
    """
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("CLAUDE_SECURESTORAGE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    if "temp_home" in request.fixturenames:
        return  # temp_home provides its own isolated home
    if "tmp_keychain" in request.fixturenames:
        return  # real-keychain integration tests need the real $HOME
    safe_home = tmp_path_factory.mktemp("isolated_home")
    (safe_home / ".claude").mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(safe_home))
    monkeypatch.setenv("USERPROFILE", str(safe_home))
    monkeypatch.setattr("pathlib.Path.home", lambda: safe_home)


@pytest.fixture(autouse=True)
def block_real_keychain(request, monkeypatch):
    """Safety net: no test may touch the real macOS Keychain.

    Replaces the ``security``-CLI wrapper (``claude_swap.macos_keychain``) with an
    in-memory fake and injects a fake ``keyring`` module (for the lazy
    ``import keyring`` paths in purge/migrations). Tests marked
    ``@pytest.mark.no_keychain_fake`` opt out — either because they mock
    ``subprocess`` themselves (the wrapper's own unit tests) or because they run
    against a temporary keychain on GitHub Actions.

    Yields the in-memory :class:`_KeychainStore` so tests can seed/inspect it.
    """
    if request.node.get_closest_marker("no_keychain_fake"):
        yield None
        return
    store = _KeychainStore()
    monkeypatch.setattr(_macos_keychain, "get_password", store.get_password)
    monkeypatch.setattr(_macos_keychain, "item_exists", store.item_exists)
    monkeypatch.setattr(_macos_keychain, "set_password", store.set_password)
    monkeypatch.setattr(_macos_keychain, "delete_password", store.delete_password)
    monkeypatch.setitem(sys.modules, "keyring", _make_fake_keyring())
    yield store


@pytest.fixture
def temp_home(tmp_path: Path):
    """Create a temporary home directory for testing."""
    home = tmp_path / "home"
    home.mkdir()

    # Create .claude directory structure
    claude_dir = home / ".claude"
    claude_dir.mkdir()

    # Patch HOME environment variable (and USERPROFILE for Windows)
    env_patch = {"HOME": str(home), "USERPROFILE": str(home)}
    with patch.dict(os.environ, env_patch):
        # Also patch Path.home() directly for cross-platform compatibility
        with patch("pathlib.Path.home", return_value=home):
            yield home


@pytest.fixture
def mock_claude_config(temp_home: Path):
    """Create a mock Claude configuration file."""
    config = {
        "oauthAccount": {
            "emailAddress": "test@example.com",
            "accountUuid": "test-uuid-1234",
        }
    }
    config_path = temp_home / ".claude.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_credentials_file(temp_home: Path):
    """Create a mock credentials file for Linux/WSL."""
    creds = {"accessToken": "test-token", "refreshToken": "test-refresh"}
    cred_path = temp_home / ".claude" / ".credentials.json"
    cred_path.write_text(json.dumps(creds))
    return cred_path


@pytest.fixture
def sample_sequence_data():
    """Sample sequence.json data."""
    return {
        "activeAccountNumber": 1,
        "lastUpdated": "2024-01-01T00:00:00Z",
        "sequence": [1, 2],
        "accounts": {
            "1": {
                "email": "account1@example.com",
                "uuid": "uuid-1",
                "added": "2024-01-01T00:00:00Z",
            },
            "2": {
                "email": "account2@example.com",
                "uuid": "uuid-2",
                "added": "2024-01-02T00:00:00Z",
            },
        },
    }


@pytest.fixture
def mock_org_claude_config(temp_home: Path):
    """Claude config file with an active organization account."""
    config = {
        "oauthAccount": {
            "emailAddress": "user@example.com",
            "accountUuid": "user-uuid-1234",
            "organizationUuid": "org-uuid-5678",
            "organizationName": "Acme Corp",
            "organizationRole": "primary_owner",
            "displayName": "Test User",
        }
    }
    config_path = temp_home / ".claude.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def mock_personal_claude_config(temp_home: Path):
    """Claude config file with a personal account (no organizationUuid)."""
    config = {
        "oauthAccount": {
            "emailAddress": "user@example.com",
            "accountUuid": "user-uuid-1234",
        }
    }
    config_path = temp_home / ".claude.json"
    config_path.write_text(json.dumps(config))
    return config_path


@pytest.fixture
def sample_sequence_data_pre_v06():
    """Pre-v0.6.0 sequence.json data without organizationUuid/Name fields."""
    return {
        "activeAccountNumber": 1,
        "lastUpdated": "2024-01-01T00:00:00Z",
        "sequence": [1, 2],
        "accounts": {
            "1": {
                "email": "user@example.com",
                "uuid": "user-uuid-1234",
                "added": "2024-01-01T00:00:00Z",
            },
            "2": {
                "email": "other@example.com",
                "uuid": "other-uuid-5678",
                "added": "2024-01-02T00:00:00Z",
            },
        },
    }


@pytest.fixture
def sample_sequence_data_with_org():
    """sequence.json data with mixed organization and personal accounts."""
    return {
        "activeAccountNumber": 1,
        "lastUpdated": "2024-01-01T00:00:00Z",
        "sequence": [1, 2],
        "accounts": {
            "1": {
                "email": "user@example.com",
                "uuid": "user-uuid",
                "organizationUuid": "org-uuid-5678",
                "organizationName": "Acme Corp",
                "added": "2024-01-01T00:00:00Z",
            },
            "2": {
                "email": "user@example.com",
                "uuid": "user-uuid",
                "organizationUuid": "",
                "organizationName": "",
                "added": "2024-01-02T00:00:00Z",
            },
        },
    }


@pytest.fixture(autouse=True)
def _deterministic_poll_jitter(monkeypatch):
    """Zero the poll-plan jitter so cadence tests are clock-exact; the jitter
    itself is exercised in test_poll_policy via an injected rng."""
    monkeypatch.setattr("claude_swap.poll_policy.JITTER_FRAC", 0.0)
