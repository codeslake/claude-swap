"""macOS Keychain contract tests.

Two layers of coverage:

1. **Mocked tests** (run on every PR, every platform): assert that the macOS
   backup-credentials path passes the correct `(service, account)` tuple to the
   `macos_keychain` security wrapper, under the new `claude-swap` service. This
   guards the multi-account backup namespace on every CI run.

2. **Real-keychain integration tests** (GHA macOS only): exercise
   `_read_credentials` / `_write_credentials` end-to-end against a temporary
   keychain, comparing token values rather than argv shape.

The Layer 2 gate (`GITHUB_ACTIONS=true AND sys.platform=="darwin"`, plus the
`no_keychain_fake` marker) is deliberate: no local opt-in, so a developer cannot
accidentally swap their default keychain by running pytest.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import call, patch

import pytest

from claude_swap import macos_keychain
from claude_swap.exceptions import SwitchError
from claude_swap.models import Platform
from claude_swap.json_output import (
    USAGE_KEYCHAIN_UNAVAILABLE,
    USAGE_NO_CREDENTIALS,
)
from claude_swap.switcher import ClaudeAccountSwitcher


# ---------------------------------------------------------------------------
# Mocked keyring tests — backup-credentials path. Run everywhere.
# ---------------------------------------------------------------------------


@pytest.fixture
def macos_switcher(temp_home: Path) -> ClaudeAccountSwitcher:
    """Switcher with platform forced to MACOS regardless of host OS."""
    switcher = ClaudeAccountSwitcher()
    switcher.platform = Platform.MACOS
    return switcher


class TestBackupCredentialsSecurity:
    """Mocked tests for the macOS backup-creds path: assert the correct
    (service, account) tuple flows to the ``macos_keychain`` security wrapper.

    The autouse ``block_real_keychain`` guard already prevents any real Keychain
    access; here we install a MagicMock to assert the exact call shape. The
    per-account backup service is the new ``claude-swap`` (not the old keyring
    ``claude-code``).
    """

    def test_read_account_credentials_uses_security_service(
        self, macos_switcher: ClaudeAccountSwitcher
    ):
        with patch("claude_swap.credentials.macos_keychain") as mock_kc:
            mock_kc.get_password.return_value = "fake-token"

            result = macos_switcher._read_account_credentials("1", "user@example.com")

            mock_kc.get_password.assert_called_once_with(
                "claude-swap", "account-1-user@example.com"
            )
            assert result == "fake-token"

    def test_write_account_credentials_uses_security_service(
        self, macos_switcher: ClaudeAccountSwitcher
    ):
        with patch("claude_swap.credentials.macos_keychain") as mock_kc:
            # No existing backup → the .prev retention step has nothing to
            # keep and the write stays a single Keychain call.
            mock_kc.get_password.return_value = None
            macos_switcher._write_account_credentials(
                "2", "alice@example.com", "secret-token"
            )

            mock_kc.set_password.assert_called_once_with(
                "claude-swap", "account-2-alice@example.com", "secret-token"
            )

    def test_write_retains_prev_generation_in_keychain_not_a_file(
        self, macos_switcher: ClaudeAccountSwitcher
    ):
        """Retention must not weaken storage posture: on a Keychain-backed
        Mac the previous generation goes to the Keychain, never a file."""
        with patch("claude_swap.credentials.macos_keychain") as mock_kc:
            mock_kc.get_password.return_value = "old-generation"
            macos_switcher._write_account_credentials(
                "2", "alice@example.com", "secret-token"
            )

            mock_kc.set_password.assert_has_calls([
                call("claude-swap", "account-2-alice@example.com.prev",
                     "old-generation"),
                call("claude-swap", "account-2-alice@example.com",
                     "secret-token"),
            ])
        prev_file = macos_switcher._store._prev_backup_path(
            "2", "alice@example.com"
        )
        assert not prev_file.exists()

    def test_delete_account_credentials_uses_security_service(
        self, macos_switcher: ClaudeAccountSwitcher
    ):
        with patch("claude_swap.credentials.macos_keychain") as mock_kc:
            macos_switcher._delete_account_credentials("3", "bob@example.com")

            mock_kc.delete_password.assert_has_calls([
                call("claude-swap", "account-3-bob@example.com"),
                call("claude-swap", "account-3-bob@example.com.prev"),
                call("claude-swap", "account-None-bob@example.com"),
                call("claude-swap", "account-None-bob@example.com.prev"),
            ])


# ---------------------------------------------------------------------------
# Real-keychain integration tests. macOS GHA only.
# ---------------------------------------------------------------------------

mac_ci_only = pytest.mark.skipif(
    os.environ.get("GITHUB_ACTIONS") != "true" or sys.platform != "darwin",
    reason="Modifies default Keychain — runs on GitHub Actions macOS only",
)


@pytest.fixture
def tmp_keychain(tmp_path: Path):
    """Create a temporary keychain, swap it in as default + sole user search-list
    entry, and restore both on teardown.

    `default-keychain` controls where new items go; `list-keychains -d user`
    controls what `find-generic-password` searches. These are independent — both
    must be redirected for `_read_credentials` (which doesn't pass `-k`) to find
    the seeded entry.

    The try/finally is the safety-critical part: a crash mid-test must still
    restore the user's original keychain config. CI doesn't care, but the safe
    shape is kept so the same code is risk-free if anyone copies it.
    """
    test_keychain = str(tmp_path / "test.keychain")
    subprocess.run(
        ["security", "create-keychain", "-p", "", test_keychain], check=True
    )
    subprocess.run(
        ["security", "unlock-keychain", "-p", "", test_keychain], check=True
    )

    # CI runners don't reliably have a default keychain configured (rc 1,
    # "A default keychain could not be found") — and an earlier swap/restore
    # cycle in this job may have cleared it. Capture it only if present and
    # skip the restore otherwise, rather than failing setup.
    default_proc = subprocess.run(
        ["security", "default-keychain"],
        capture_output=True,
        text=True,
    )
    original_default = (
        default_proc.stdout.strip().strip('"')
        if default_proc.returncode == 0
        else None
    )
    list_proc = subprocess.run(
        ["security", "list-keychains", "-d", "user"],
        capture_output=True,
        text=True,
    )
    original_list = [
        line.strip().strip('"')
        for line in (list_proc.stdout if list_proc.returncode == 0 else "").splitlines()
        if line.strip()
    ]

    try:
        subprocess.run(
            ["security", "default-keychain", "-s", test_keychain], check=True
        )
        subprocess.run(
            ["security", "list-keychains", "-d", "user", "-s", test_keychain],
            check=True,
        )
        # Harden against an invisible SecurityAgent dialog hanging the job: a
        # `security` call against a (re-)locked keychain blocks forever on a
        # headless runner waiting for an unlock prompt nobody can click. Remove
        # the auto-lock timeout and unlock *after* the default/search-list swap
        # (the order fastlane's setup_ci uses).
        subprocess.run(
            ["security", "set-keychain-settings", test_keychain], check=True
        )
        subprocess.run(
            ["security", "unlock-keychain", "-p", "", test_keychain], check=True
        )
        yield test_keychain
    finally:
        # Restore the search list BEFORE the default: macOS won't report a
        # default keychain that isn't in the search list, so the reverse order
        # leaves the default dangling for whatever runs next in this job.
        if original_list:
            subprocess.run(
                ["security", "list-keychains", "-d", "user", "-s", *original_list],
                check=False,
            )
        if original_default:
            subprocess.run(
                ["security", "default-keychain", "-s", original_default], check=False
            )
        subprocess.run(["security", "delete-keychain", test_keychain], check=False)


@pytest.mark.no_keychain_fake
@mac_ci_only
def test_read_credentials_finds_claude_code_seeded_entry(tmp_keychain: str):
    username = os.environ["USER"]
    subprocess.run(
        [
            "security",
            "add-generic-password",
            "-a",
            username,
            "-s",
            "Claude Code-credentials",
            "-w",
            "fake-token-read",
            "-A",
            tmp_keychain,
        ],
        check=True,
    )

    switcher = ClaudeAccountSwitcher()
    switcher.platform = Platform.MACOS
    assert switcher._read_credentials() == "fake-token-read"


@pytest.mark.no_keychain_fake
@mac_ci_only
def test_write_credentials_creates_user_scoped_entry(tmp_keychain: str):
    # If _write_credentials ever stores the entry under a hardcoded account name
    # (or any value other than $USER), the verification lookup below — which
    # mirrors Claude Code's own read shape — returns 44 and the test fails.
    switcher = ClaudeAccountSwitcher()
    switcher.platform = Platform.MACOS
    switcher._write_credentials("fake-token-write")

    username = os.environ["USER"]
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            username,
            "-s",
            "Claude Code-credentials",
            "-w",
            tmp_keychain,
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"security find-generic-password failed: {result.stderr}"
    )
    assert result.stdout.strip() == "fake-token-write"


@pytest.mark.no_keychain_fake
@mac_ci_only
def test_wrapper_roundtrip_real_keychain(tmp_keychain: str):
    """set → get → delete through the real wrapper against the temp keychain.

    Covers the full production read/write/delete path the other Layer-2 tests
    only half-exercise: a wrapper-created item (no ``-A`` any-app access) read
    back via the keychain *search list* (no explicit keychain argument), then
    deleted, with the rc-44 "not found" contract checked at the end.
    """
    macos_keychain.set_password("claude-swap-test", "acct-1", "round-trip-token")
    assert macos_keychain.get_password("claude-swap-test", "acct-1") == "round-trip-token"
    macos_keychain.delete_password("claude-swap-test", "acct-1")
    assert macos_keychain.get_password("claude-swap-test", "acct-1") is None


class TestOurOwnFileModeIsNotAKeychainFailure:
    """A file mode WE pinned must not read as "the Keychain is unreadable".

    ``_pin_file_mode`` sets the capability cache False deliberately and
    permanently — nothing failed, we wrote the credential to the file, and that
    file is what Claude Code reads too. ``_read_active_credentials`` already
    excludes that case via ``_file_mode_is_ours``; two sibling sites read the
    raw flag instead, so ONE deliberate file-mode write made every genuinely
    empty backup report "keychain unavailable" for the rest of the process —
    and made the consume gate answer ``transient`` for a slot with no backup
    at all.
    """

    def test_an_empty_backup_stays_empty_after_our_own_file_mode_write(
        self, macos_switcher
    ):
        store = macos_switcher._store
        store._pin_file_mode()  # OUR write; nothing failed
        _value, unreadable = store._read_account_credentials_ex("9", "x@e.com")
        assert unreadable is False

    def test_a_write_fallback_does_not_certify_an_unread_backup(
        self, macos_switcher, monkeypatch
    ):
        """The pin says the ACTIVE credential is in the file. Not the backup.

        On macOS the pin is only reachable THROUGH a Keychain op that just
        failed — both call sites sit in the fallback branch of a write that
        raised. So it arrives carrying the opposite of what it asserts: it
        clears the failure verdict AND the re-probe deadline, and every guard
        keyed on ``_keychain_unreadable`` goes quiet for the rest of the
        process while the Keychain is still locked.

        The backup is the part that is not covered. It lives Keychain-only by
        design (``_reconcile_enc_after_keychain_write`` deletes the ``.enc``
        after a successful Keychain write), so a write fallback for the ACTIVE
        credential says nothing about whether that backup could be read — and
        an empty read of it still proves nothing.

        Measured before the fix, identical world, one fallback write apart::

            state A   _read_account_credentials_ex('3') -> ('', True)
            state B   _read_account_credentials_ex('3') -> ('', False)

        The consume gate reads that second answer as "the slot is genuinely
        empty", POSTs the caller's possibly-superseded snapshot, takes
        ``invalid_grant``, and quarantines a slot whose live refresh token is
        sitting unread in the Keychain.
        """
        from claude_swap import macos_keychain as _kc

        store = macos_switcher._store
        store._keychain_usable_cache = True

        def locked(*_a, **_kw):
            raise _kc.KeychainError("locked")

        for fn in ("set_password", "get_password", "delete_password"):
            monkeypatch.setattr(_kc, fn, locked)

        assert store._read_account_credentials_ex("3", "c@e.com") == ("", True), (
            "premise: a locked Keychain makes an empty backup read unprovable"
        )

        # One active-credential write falls back to the file and pins.
        store._write_oauth_credentials('{"claudeAiOauth": {"accessToken": "sk-fb"}}')
        assert store._file_mode_is_ours is True, "premise: the write pinned"

        _value, unreadable = store._read_account_credentials_ex("3", "c@e.com")
        assert unreadable is True, (
            "the Keychain is still locked and the backup still unread — a "
            "write fallback for the ACTIVE credential does not certify it empty"
        )

    def test_a_write_fallback_does_not_certify_the_ACTIVE_read_either(
        self, macos_switcher, monkeypatch
    ):
        """The same argument, at the two sites the backup fix scoped out.

        `_read_active_credentials` asks `_keychain_unreadable` only on the
        `_use_keychain()` False branch — where it never attempted a read, so
        it has to reconstruct the verdict. `_pin_file_mode` clears exactly
        that verdict, and on macOS it is reachable ONLY through a Keychain op
        that just failed. So one active-credential write fallback flips both:

            state A   degraded=True   sentinel='keychain unavailable'
            state B   degraded=False  sentinel='no credentials'

        `degraded=False` disarms `_refuse_degraded_capture`, and the sentinel
        sends the user to `cswap --add-account`, the one remedy that cannot
        work while the real credential sits unread in the Keychain. The pin's
        best-effort `_delete_active_keychain_entry()` also failed, so a
        residual survives and Claude Code reads Keychain-first — our file is
        the superseded generation, POSTed with the guard disarmed.
        """
        from claude_swap import macos_keychain as _kc

        store = macos_switcher._store
        store._keychain_usable_cache = True

        def locked(*_a, **_kw):
            raise _kc.KeychainError("locked")

        for fn in ("set_password", "get_password", "delete_password"):
            monkeypatch.setattr(_kc, fn, locked)

        assert store._read_active_credentials().degraded is True, "premise"
        store._write_oauth_credentials('{"claudeAiOauth": {"accessToken": "x"}}')
        assert store._file_mode_is_ours is True, "premise: the write pinned"

        assert store._read_active_credentials().degraded is True, (
            "the Keychain is still locked; a write fallback for the active "
            "credential does not prove the slot is genuinely empty"
        )
        verdict = macos_switcher._static_usage_sentinel(
            ("2", "b@example.com", None, None, False, "", None)
        )
        assert verdict == USAGE_KEYCHAIN_UNAVAILABLE, (
            f"sentinel says {verdict!r} — sends the user to re-add a slot "
            f"whose backup is alive but unread"
        )

    def test_the_default_capture_path_refuses_a_degraded_read(
        self, macos_switcher, monkeypatch
    ):
        """`_refuse_degraded_capture` on the path that actually reaches it.

        Mutation-checked: neutering the guard left all 1783 green. Every other
        test of this area drives the two env-var branches, which pass
        `strict_keychain=True` and raise inside `read_config_dir_credentials`
        instead — so the guard the PR body calls the fix for "the most-used
        path" had nothing pinning it.

        With the Keychain locked, the only readable credential is the plaintext
        fallback, which on macOS may be the consumed predecessor because Claude
        Code rotates keychain-only. Capturing it files a spent refresh token
        against the slot and `add_account` then clears the dead-token strike.
        """
        import json

        from claude_swap import macos_keychain as _kc
        from claude_swap.exceptions import CredentialReadError
        from claude_swap.paths import get_credentials_path

        store = macos_switcher._store
        store._keychain_usable_cache = True
        monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

        def locked(*_a, **_kw):
            raise _kc.KeychainError("locked")

        for fn in ("get_password", "set_password", "delete_password"):
            monkeypatch.setattr(_kc, fn, locked)

        # A plaintext fallback exists and is readable — the trap.
        get_credentials_path().parent.mkdir(parents=True, exist_ok=True)
        get_credentials_path().write_text(
            json.dumps({"claudeAiOauth": {"accessToken": "sk-maybe-spent"}})
        )

        with pytest.raises(CredentialReadError) as exc:
            macos_switcher._read_capture_credentials()
        assert "superseded generation" in str(exc.value), exc.value

    def test_a_real_keychain_failure_still_reports_unreadable(
        self, macos_switcher, monkeypatch
    ):
        """A Keychain that actually raises. Not a leftover flag.

        This used to set ``_keychain_usable_cache = False`` and call that "a
        read FAILED" — but that flag is a record of some EARLIER op, and the
        backup read does not consult it (``_kc_read_backup`` goes straight
        through ``_kc_call``). So the setup described a stale flag over a
        working Keychain, and the assertion passed because the answer was
        reconstructed from the flag instead of from the read.

        The read is now the witness, so the failure has to be real for the
        verdict to be real — which is the guarantee the name claims.
        """
        from claude_swap import macos_keychain as _kc

        store = macos_switcher._store

        def locked(*_a, **_kw):
            raise _kc.KeychainError("locked")

        monkeypatch.setattr(_kc, "get_password", locked)
        _value, unreadable = store._read_account_credentials_ex("9", "x@e.com")
        assert unreadable is True

    def test_a_stale_failure_flag_does_not_condemn_a_working_read(
        self, macos_switcher
    ):
        """The other half: a flag from an earlier op must not outvote the read.

        An op failed a moment ago and the Keychain has since recovered. The
        backup read reaches it and answers cleanly that slot 9 has no backup.
        Reporting "unreadable" there sends the user to the one remedy that
        cannot work ("retry from a GUI terminal; do not re-add") and hides the
        one that can.
        """
        store = macos_switcher._store
        store._keychain_usable_cache = False  # an earlier op failed
        _value, unreadable = store._read_account_credentials_ex("9", "x@e.com")
        assert unreadable is False

    def test_a_recovered_keychain_is_not_condemned_by_the_previous_read(
        self, macos_switcher, monkeypatch
    ):
        """The verdict is per-read, so it must be CLEARED per read.

        Two reads in one process: the first cannot reach the Keychain, the
        second can. Without the reset the first read's True is still set when
        the second returns, so a recovered Keychain keeps reporting
        "unreadable" and the consume gate defers forever.

        This is reachable in ordinary use, not only across a recovery: many
        tests and callers substitute ``_read_account_credentials``, which never
        reaches the line that sets the flag — so whatever the last real read
        left behind would be reported as if it were this read's answer.
        """
        from claude_swap import macos_keychain as _kc

        store = macos_switcher._store
        real_get = _kc.get_password

        def locked(*_a, **_kw):
            raise _kc.KeychainError("locked")

        monkeypatch.setattr(_kc, "get_password", locked)
        assert store._read_account_credentials_ex("9", "x@e.com") == ("", True)

        monkeypatch.setattr(_kc, "get_password", real_get)   # recovered
        _value, unreadable = store._read_account_credentials_ex("9", "x@e.com")
        assert unreadable is False, "the previous read's failure outlived it"

    def test_a_recovered_keychain_does_not_latch_through_a_later_pin(
        self, macos_switcher, monkeypatch
    ):
        """An observation must be UPDATED by a later observation, not frozen.

        `_keychain_op_failed` records that an op failed, and nothing clears it.
        The cooldown re-probe in `_use_keychain` normally masks a stale one —
        but `_pin_file_mode` zeroes `_keychain_disabled_until`, so once a write
        falls back the mask is gone for the rest of the process.

        Measured timeline: a read times out (t0); the Keychain recovers and the
        cooldown lapses, verified `unreadable is False` (t1); a later write
        takes the file branch purely because ROUTING still says file, no op
        raises, and its `_delete_active_keychain_entry()` SUCCEEDS so there is
        no residual and the file genuinely is the authority (t2). From t2 on,
        forever: `degraded=True`, `_fetch_active_usage` reports
        "keychain unavailable" every pass, `_refuse_degraded_capture` blocks
        `cswap add`, `_resync_rotated_backup` never runs.

        That violates the same self-heal this predicate's own docstring
        promises: one transient failure must not be permanent for the process.
        A SUCCESSFUL op is evidence too, and it is the newer one.
        """
        import time

        from claude_swap import macos_keychain as _kc

        store = macos_switcher._store
        store._keychain_usable_cache = True

        def locked(*_a, **_kw):
            raise _kc.KeychainError("locked")

        # Save the conftest fake rather than using monkeypatch.undo(), which
        # would also unwind the keychain stub and let the call reach the real
        # `security` binary.
        healthy = _kc.get_password
        monkeypatch.setattr(_kc, "get_password", locked)
        with pytest.raises(_kc.KeychainError):
            store._kc_call(_kc.get_password, "svc", "acct")
        assert store._keychain_unreadable is True, "premise: inside the cooldown"

        monkeypatch.setattr(_kc, "get_password", healthy)
        store._keychain_disabled_until = time.monotonic() - 1
        assert store._keychain_unreadable is False, "premise: cooldown lapsed"

        # A real op now succeeds — the Keychain is demonstrably back.
        store._kc_call(_kc.get_password, "svc", "acct")

        # ...and later a write falls back and pins, zeroing the re-probe.
        store._pin_file_mode()

        assert store._keychain_unreadable is False, (
            "a Keychain that answered successfully after the failure is still "
            "reported unreadable — the failure latched through the pin"
        )

    def test_an_idle_backup_read_does_not_erase_the_active_verdict(
        self, macos_switcher, monkeypatch
    ):
        """A success on ONE item is not evidence about ANOTHER.

        `_kc_call` clears `_keychain_op_failed` on any successful op, and
        backup reads go straight through it (`_read_account_credentials` never
        consults `_use_keychain`). So one idle slot's readable backup erased
        the verdict recorded when the ACTIVE OAuth read failed, and
        `_read_active_credentials().degraded` flipped to False while the
        plaintext file still held the superseded generation.

        Measured, same fixture, only the slot ORDER changed:

            [2,3]  degraded=True   sentinel='keychain unavailable'  POSTed: []
            [3,2]  degraded=False  error='invalid_grant'            POSTed: ['rt-SPENT']

        and it does not self-heal — the strike binds to the backup's
        fingerprint, the backup still holds that generation, so a brand-new
        process with a healthy Keychain keeps reporting "re-login needed".

        The clear was right about a success being a newer observation; it was
        wrong about WHAT it observes. "The Keychain answers" and "this active
        read succeeded" are different facts, and `degraded` needs the second.
        """
        from claude_swap import macos_keychain as _kc

        store = macos_switcher._store
        store._keychain_usable_cache = True
        real_get = _kc.get_password

        def only_the_active_item_fails(service, account):
            if "credentials" in service:
                raise _kc.KeychainError("locked")
            return real_get(service, account)

        monkeypatch.setattr(_kc, "get_password", only_the_active_item_fails)

        assert store._read_active_credentials().degraded is True, "premise"
        store._read_account_credentials("3", "c@example.com")  # idle, succeeds
        assert store._read_active_credentials().degraded is True, (
            "an unrelated slot's readable backup erased the active read's own "
            "failure — the consume gate now POSTs a possibly-spent generation"
        )

    def test_a_lapsed_cooldown_clears_the_unreadable_verdict(self, macos_switcher):
        """One transient failure must not be permanent for the process.

        The cooldown re-probe lives in ``_use_keychain``, and the BACKUP read
        path never calls it — ``_read_account_credentials`` goes straight to
        ``_kc_read_backup``. Reading the cache raw therefore made a single
        hiccup stick: a genuinely empty slot kept answering "unreadable, do
        not re-add" (advice with no remedy) and the consume gate kept
        deferring long after the Keychain answered again.
        """
        import time
        from claude_swap import macos_keychain as _kc

        store = macos_switcher._store
        with pytest.raises(_kc.KeychainError):
            store._kc_call(lambda: (_ for _ in ()).throw(_kc.KeychainError("denied")))
        assert store._keychain_unreadable is True, "inside the cooldown"

        store._keychain_disabled_until = time.monotonic() - 1
        assert store._keychain_unreadable is False, "cooldown lapsed — re-probe"

    def test_off_macos_there_is_no_keychain_to_be_unreadable(self, temp_home: Path):
        """The predicate carries its own platform check.

        Every path that can flip the cache is macOS-gated today, so the cache
        stays ``None`` off macOS — but that is an invariant the CALLERS hold,
        not this predicate. It is the single source of truth for three sites
        now; it keeps the check itself.
        """
        s = ClaudeAccountSwitcher()
        s.platform = Platform.LINUX
        store = s._store
        store._keychain_usable_cache = False
        assert store._keychain_unreadable is False

    def test_a_missing_slot_is_no_credentials_not_keychain_unavailable(
        self, macos_switcher
    ):
        """Same conflation on the display path."""
        macos_switcher._store._pin_file_mode()
        verdict = macos_switcher._static_usage_sentinel(
            ("9", "x@e.com", None, None, False, "", None)
        )
        assert verdict == USAGE_NO_CREDENTIALS, verdict

    def test_switch_to_an_empty_slot_says_re_add_after_our_file_mode_write(
        self, macos_switcher, temp_home: Path
    ):
        """The third sibling: ``_perform_switch``'s ``backup_unreadable``.

        Reading the raw flag there sends the user to the one remedy that
        cannot work — "retry from a GUI terminal; do not re-add" for a slot
        that has no backup at all — and hides the one that can.
        """
        s = macos_switcher
        s._setup_directories()
        s._init_sequence_file()
        s._write_json(
            s.sequence_file,
            {
                "activeAccountNumber": None,
                "lastUpdated": "",
                "sequence": [2],
                "accounts": {
                    "2": {
                        "email": "b@example.com",
                        "uuid": "uuid-2",
                        "organizationUuid": "",
                        "organizationName": "",
                        "added": "2024-01-01T00:00:00Z",
                    }
                },
            },
        )
        s._store._pin_file_mode()  # OUR write; nothing failed

        with pytest.raises(SwitchError) as exc:
            s._perform_switch("2", emit_output=False, force_activate=True)
        assert "has no stored credentials" in str(exc.value), exc.value
