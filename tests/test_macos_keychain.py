"""Unit tests for the macOS ``security``-CLI wrapper (claude_swap.macos_keychain).

These mock ``subprocess.run`` so they exercise the wrapper's argv/stdin shaping,
hex encoding, and return-code handling without ever invoking the real
``security`` binary. (The autouse ``block_real_keychain`` guard replaces the
module's functions for *other* tests; here we patch ``subprocess`` so the real
function bodies run against a fake process.)
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from claude_swap import credentials, macos_keychain, oauth
from claude_swap.json_output import USAGE_KEYCHAIN_UNAVAILABLE
from claude_swap.models import Platform
from claude_swap.switcher import ClaudeAccountSwitcher

# Every test here drives the *real* wrapper bodies (mocking subprocess) or runs
# against a temp keychain on CI, so opt the whole module out of the in-memory
# Keychain guard that replaces these functions for other tests.
pytestmark = pytest.mark.no_keychain_fake


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["security"], returncode=returncode, stdout=stdout, stderr=stderr
    )


# ---------------------------------------------------------------------------
# get_password
# ---------------------------------------------------------------------------


def test_get_password_returns_value_on_rc0():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(0, stdout="the-secret\n")
        assert macos_keychain.get_password("svc", "acct") == "the-secret"
        args = run.call_args.args[0]
        assert args[:2] == ["/usr/bin/security", "find-generic-password"]
        assert "-a" in args and "acct" in args and "svc" in args


def test_get_password_returns_none_only_on_rc44():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(44)
        assert macos_keychain.get_password("svc", "acct") is None


def test_get_password_raises_on_other_nonzero():
    # e.g. locked / denied / unavailable — must NOT be masked as "not found".
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(51, stderr="boom")
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.get_password("svc", "acct")


# ---------------------------------------------------------------------------
# item_exists
# ---------------------------------------------------------------------------


def test_item_exists_true_on_rc0_and_never_requests_secret():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(0)
        assert macos_keychain.item_exists("svc", "acct") is True
        args = run.call_args.args[0]
        # Attribute-only lookup: must never pass -w (decrypting could prompt).
        assert "-w" not in args


def test_item_exists_false_on_rc44_and_errors():
    for rc in (44, 51):
        with patch("claude_swap.macos_keychain.subprocess.run") as run:
            run.return_value = _completed(rc)
            assert macos_keychain.item_exists("svc", "acct") is False


# ---------------------------------------------------------------------------
# set_password — stdin (security -i) vs argv fallback
# ---------------------------------------------------------------------------


def test_set_password_small_payload_uses_security_i_stdin():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(0)
        macos_keychain.set_password("svc", "acct", "short-secret")

        args = run.call_args.args[0]
        kwargs = run.call_args.kwargs
        assert args == ["/usr/bin/security", "-i"]  # stdin path
        # Secret is NOT in argv; it rides in on stdin as a hex `-X` value.
        assert "short-secret" not in args
        stdin = kwargs["input"]
        assert stdin.startswith("add-generic-password -U")
        assert "-X " + "short-secret".encode().hex() in stdin
        # -a/-s are quoted in the stdin command line.
        assert '-a "acct"' in stdin and '-s "svc"' in stdin


def test_set_password_large_payload_falls_back_to_argv():
    big = "x" * macos_keychain.SECURITY_STDIN_LINE_LIMIT  # hex doubles the length
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(0)
        macos_keychain.set_password("svc", "acct", big)

        args = run.call_args.args[0]
        assert args[:3] == ["/usr/bin/security", "add-generic-password", "-U"]  # argv path
        assert "input" not in run.call_args.kwargs  # not via stdin
        # Hex value passed as a raw list element (no shell, no quoting).
        assert big.encode().hex() in args
        assert "acct" in args and "svc" in args


def test_set_password_raises_on_nonzero():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(45, stderr="nope")
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.set_password("svc", "acct", "secret")


def test_set_get_roundtrip_hex_is_decodable():
    # The hex written on set must decode back to the original UTF-8 secret.
    secret = 'token-with "quotes" and \\ backslash and é'
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _completed(0)

    with patch("claude_swap.macos_keychain.subprocess.run", side_effect=fake_run):
        macos_keychain.set_password("svc", "acct", secret)
    stdin = captured["kwargs"]["input"]
    hex_token = stdin.split("-X ", 1)[1].strip()
    assert bytes.fromhex(hex_token).decode("utf-8") == secret


# ---------------------------------------------------------------------------
# delete_password
# ---------------------------------------------------------------------------


def test_delete_password_rc0_and_rc44_are_success():
    for rc in (0, 44):
        with patch("claude_swap.macos_keychain.subprocess.run") as run:
            run.return_value = _completed(rc)
            macos_keychain.delete_password("svc", "acct")  # no raise


def test_delete_password_raises_on_other_nonzero():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(51, stderr="locked")
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.delete_password("svc", "acct")


# ---------------------------------------------------------------------------
# timeouts — a wedged Keychain must surface as KeychainError, never a hang
# ---------------------------------------------------------------------------


def test_calls_pass_timeout_to_subprocess():
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(0, stdout="x\n")
        macos_keychain.get_password("svc", "acct")
        assert run.call_args.kwargs.get("timeout") == macos_keychain._TIMEOUT


@pytest.mark.parametrize("fn,args", [
    ("get_password", ("svc", "acct")),
    ("set_password", ("svc", "acct", "secret")),
    ("delete_password", ("svc", "acct")),
])
def test_timeout_becomes_keychain_error(fn, args):
    timeout = subprocess.TimeoutExpired(cmd="security", timeout=5)
    with patch("claude_swap.macos_keychain.subprocess.run", side_effect=timeout):
        with pytest.raises(macos_keychain.KeychainError):
            getattr(macos_keychain, fn)(*args)


def test_item_exists_stays_false_on_timeout_and_missing_binary():
    # item_exists must never raise (it feeds cleanup, not the capability cache).
    timeout = subprocess.TimeoutExpired(cmd="security", timeout=5)
    with patch("claude_swap.macos_keychain.subprocess.run", side_effect=timeout):
        assert macos_keychain.item_exists("svc", "acct") is False
    with patch("claude_swap.macos_keychain.subprocess.run", side_effect=FileNotFoundError):
        assert macos_keychain.item_exists("svc", "acct") is False


# ---------------------------------------------------------------------------
# keychain_account_name — mirror Claude Code's getUsername()
# ---------------------------------------------------------------------------


def test_keychain_account_name_prefers_user_env(monkeypatch):
    monkeypatch.setenv("USER", "alice")
    assert macos_keychain.keychain_account_name() == "alice"


def test_keychain_account_name_no_user_env_avoids_legacy_default(monkeypatch):
    # The old active-store default was the bare string "user", which mismatches
    # Claude Code's OS-username on headless hosts ($USER unset). The shared helper
    # must fall back to the OS username / "claude-code-user", never "user".
    monkeypatch.delenv("USER", raising=False)
    name = macos_keychain.keychain_account_name()
    assert name and name != "user"


# ---------------------------------------------------------------------------
# item_modified_at
# ---------------------------------------------------------------------------


def test_item_modified_at_parses_a_captured_attribute_block():
    """T1312 [m]: parsed against a REALISTIC captured
    ``security find-generic-password`` attribute block (no ``-w``), not a
    hand-built string that happens to satisfy the regex."""
    stdout = (
        'keychain: "/Users/x/Library/Keychains/login.keychain-db"\n'
        'version: 512\n'
        'class: "genp"\n'
        'attributes:\n'
        '    0x00000007 <blob>="Claude Code-credentials"\n'
        '    0x00000008 <blob>=<NULL>\n'
        '    "acct"<blob>="x"\n'
        '    "cdat"<timedate>=0x32303236303931383132333435365A00  '
        '"20260918123456Z"\n'
        '    "crtr"<uint32>=<NULL>\n'
        '    "cusi"<sint32>=<NULL>\n'
        '    "invi"<sint32>=0x0\n'
        '    "mdat"<timedate>=0x32303236303932343130313533305A00  '
        '"20260924101530Z"\n'
        '    "prot"<blob>=<NULL>\n'
        '    "scrp"<sint32>=<NULL>\n'
        '    "svce"<blob>="Claude Code-credentials"\n'
        '    "type"<uint32>=<NULL>\n'
    )
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(0, stdout=stdout)
        got = macos_keychain.item_modified_at("Claude Code-credentials", "x")
    expected = datetime.strptime(
        "20260924101530", "%Y%m%d%H%M%S"
    ).replace(tzinfo=timezone.utc).timestamp()
    assert got == expected
    args = run.call_args.args[0]
    assert "-w" not in args, "the mdat read must be attribute-only, no -w"


@pytest.mark.parametrize("returncode, stdout", [
    (44, ""),                    # absent item (rc-44)
    (0, "attributes:\n"),        # no "mdat" attribute in the output
])
def test_item_modified_at_none_when_unavailable(returncode, stdout):
    with patch("claude_swap.macos_keychain.subprocess.run") as run:
        run.return_value = _completed(returncode, stdout=stdout)
        assert macos_keychain.item_modified_at("svc", "acct") is None


# The real-Keychain round-trip test lives in test_macos_keychain_contract.py,
# next to the `tmp_keychain` fixture it depends on.


# ---------------------------------------------------------------------------
# memo -- one ``security`` exec per item until the login keychain file changes
# ---------------------------------------------------------------------------

_KEY = ("svc", "acct")


class _FakeSecurity:
    """Stands in for ``subprocess.run`` against ``security``: a dict of items,
    a forced rc for reads (36 = locked) and for writes, the services whose
    reads are denied, and a count of read execs."""

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}
        self.rc: int | None = None
        self.write_rc: int | None = None
        self.denied: set[str] = set()
        self.execs = 0

    def __call__(self, args, **kwargs):
        if "find-generic-password" not in args:  # add / delete
            return _completed(self.write_rc or 0, stderr="locked")
        self.execs += 1
        key = (args[args.index("-s") + 1], args[args.index("-a") + 1])
        rc = 36 if key[0] in self.denied else self.rc
        if rc is not None:
            return _completed(rc, stderr="locked")
        if key not in self.items:
            return _completed(44)
        if "-w" in args:
            return _completed(0, stdout=self.items[key] + "\n")
        return _completed(0, stdout='    "mdat"<timedate>=0x00  "20260924101530Z"\n')


@pytest.fixture
def fake_security():
    fake = _FakeSecurity()
    with patch("claude_swap.macos_keychain.subprocess.run", side_effect=fake):
        yield fake


def _make_kc_file(home):
    """The login keychain file under the scratch HOME, whose stat the memo checks."""
    path = home / "Library" / "Keychains" / "login.keychain-db"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"kc")
    return path


@pytest.fixture
def kc_file(temp_home):
    return _make_kc_file(temp_home)


def _touch(path):
    """A keychain write as the memo sees it: a later mtime, same size and inode."""
    ns = path.stat().st_mtime_ns + 1_000_000
    os.utime(path, ns=(ns, ns))


@pytest.mark.parametrize("stored, has_file, execs", [
    ("secret", True, 1),   # found: memoized
    (None, True, 1),       # rc 44 is a definite answer too
    ("secret", False, 5),  # no keychain file to stat: never memoized
])
def test_get_password_memo_execs_once_while_the_stat_is_unchanged(
    fake_security, temp_home, stored, has_file, execs
):
    if has_file:
        _make_kc_file(temp_home)
    if stored is not None:
        fake_security.items[_KEY] = stored
    for _ in range(5):
        assert macos_keychain.get_password(*_KEY) == stored
    assert fake_security.execs == execs


def test_get_password_reads_again_when_the_keychain_file_changes(
    fake_security, kc_file
):
    fake_security.items[_KEY] = "old"
    assert macos_keychain.get_password(*_KEY) == "old"
    fake_security.items[_KEY] = "new"
    _touch(kc_file)
    assert macos_keychain.get_password(*_KEY) == "new"
    assert fake_security.execs == 2


@pytest.mark.parametrize("write", [
    lambda: macos_keychain.set_password(*_KEY, "new"),
    lambda: macos_keychain.delete_password(*_KEY),
])
def test_a_write_drops_only_its_own_key(fake_security, kc_file, write):
    fake_security.items[_KEY] = "old"
    fake_security.items[("svc", "other")] = "kept"
    macos_keychain.get_password(*_KEY)
    macos_keychain.get_password("svc", "other")
    # Same stamp: only the write's own invalidation can make the next read exec.
    fake_security.items[_KEY] = "new"
    write()
    assert macos_keychain.get_password("svc", "other") == "kept"
    assert fake_security.execs == 2
    assert macos_keychain.get_password(*_KEY) == "new"
    assert fake_security.execs == 3


def test_an_error_is_never_memoized(fake_security, kc_file):
    fake_security.rc = 36
    for _ in range(2):
        with pytest.raises(macos_keychain.KeychainError):
            macos_keychain.get_password(*_KEY)
    assert fake_security.execs == 2
    fake_security.rc = None
    fake_security.items[_KEY] = "unlocked"
    assert macos_keychain.get_password(*_KEY) == "unlocked"


def test_item_modified_at_memoizes_a_definite_answer_only(fake_security, kc_file):
    assert macos_keychain.item_modified_at(*_KEY) is None  # absent: not memoized
    assert macos_keychain.item_modified_at(*_KEY) is None
    assert fake_security.execs == 2
    fake_security.items[_KEY] = "x"
    first = macos_keychain.item_modified_at(*_KEY)
    assert first is not None and macos_keychain.item_modified_at(*_KEY) == first
    assert fake_security.execs == 3
    _touch(kc_file)
    macos_keychain.item_modified_at(*_KEY)
    assert fake_security.execs == 4


@pytest.mark.parametrize("op, forced, raises", [
    (lambda: macos_keychain.get_password(*_KEY), "rc", True),
    (lambda: macos_keychain.item_modified_at(*_KEY), "rc", False),
    (lambda: macos_keychain.set_password(*_KEY, "v"), "write_rc", True),
    (lambda: macos_keychain.delete_password(*_KEY), "write_rc", True),
])
def test_a_failed_exec_forgets_every_memoized_read(
    fake_security, kc_file, op, forced, raises
):
    fake_security.items[("svc", "other")] = "kept"
    macos_keychain.get_password("svc", "other")
    setattr(fake_security, forced, 36)
    if raises:
        with pytest.raises(macos_keychain.KeychainError):
            op()
    else:
        op()
    setattr(fake_security, forced, None)
    before = fake_security.execs
    assert macos_keychain.get_password("svc", "other") == "kept"
    assert fake_security.execs == before + 1


def test_any_keychain_file_moving_drops_the_whole_memo(fake_security, kc_file):
    work = kc_file.with_name("work.keychain-db")  # a non-login default keychain
    work.write_bytes(b"kc")
    for account in "abc":
        fake_security.items[("svc", account)] = account
        macos_keychain.get_password("svc", account)
    _touch(work)
    macos_keychain.get_password("svc", "a")
    assert fake_security.execs == 4
    assert len(macos_keychain._memo) == 1  # one keychain generation at a time


# The memo serves display and snapshot reads only. A read that leads to a grant
# POST (the consume gate, the active refresh) is made on fresh reads, so it sees a
# locked Keychain as locked, and a memo hit is never read as the Keychain answering.

_EMAIL = "user@example.com"


def _creds(refresh: str, expires_at: int) -> str:
    return json.dumps({"claudeAiOauth": {
        "accessToken": "at",
        "refreshToken": refresh,
        "expiresAt": expires_at,
        "refreshTokenExpiresAt": 1_900_000_000_000,
    }})


@pytest.fixture
def mac_switcher(temp_home, fake_security, kc_file, monkeypatch):
    monkeypatch.setenv("USER", "alice")
    switcher = ClaudeAccountSwitcher()
    switcher.platform = Platform.MACOS
    switcher._setup_directories()
    return switcher


def test_the_consume_gate_reads_the_backup_past_the_memo(mac_switcher, fake_security):
    key = ("claude-swap", f"account-1-{_EMAIL}")
    fake_security.items[key] = _creds("spent", 1_000)
    mac_switcher._read_account_credentials_ex("1", _EMAIL)  # memoized
    # Written under the SAME stat: only an exec can see it.
    fake_security.items[key] = _creds("current", 1_000)
    with patch(
        "claude_swap.oauth.try_refresh_oauth_credentials",
        return_value=oauth.RefreshOutcome(None, "transient"),
    ) as post:
        mac_switcher.consume_backup_grant("1", _EMAIL, _creds("spent", 1_000))
    assert post.call_args.args[0] == _creds("current", 1_000)


_ACTIVE = ("Claude Code-credentials", "alice")


def _post_replaces(fake_security, rc):
    """A refresh POST during which the Keychain locks (`rc` 36) or stays as is."""
    def post(*args, **kwargs):
        fake_security.rc = rc
        return oauth.RefreshOutcome(_creds("next", 2_000), None)
    return patch("claude_swap.oauth.try_refresh_oauth_credentials", side_effect=post)


def test_the_gate_cas_reads_past_the_memo(mac_switcher, fake_security):
    fake_security.items[("claude-swap", f"account-1-{_EMAIL}")] = _creds("spent", 1_000)
    with _post_replaces(fake_security, 36):  # screen lock, mid-POST
        outcome = mac_switcher.consume_backup_grant(
            "1", _EMAIL, _creds("spent", 1_000)
        )
    # Unreadable, not "unchanged": the successor is stashed and nothing rewritten.
    assert (outcome.error, outcome.stashed) == ("transient", True)


def test_the_gate_reads_the_live_store_past_the_memo(mac_switcher, fake_security):
    fake_security.items[("claude-swap", f"account-1-{_EMAIL}")] = _creds("spent", 1_000)
    fake_security.items[_ACTIVE] = _creds("other", 5_000)
    mac_switcher._read_active_credentials()
    mac_switcher._read_account_credentials_ex("1", _EMAIL)
    fake_security.denied.add(_ACTIVE[0])  # only the live item refuses
    with _post_replaces(fake_security, None) as post:
        outcome = mac_switcher.consume_backup_grant(
            "1", _EMAIL, _creds("spent", 1_000)
        )
    post.assert_not_called()
    assert outcome.error == "live-store-unreadable"


@pytest.mark.parametrize("plaintext", [False, True])
def test_an_expired_active_token_is_not_posted_once_the_keychain_locks(
    mac_switcher, fake_security, temp_home, mock_claude_config,
    sample_sequence_data, monkeypatch, plaintext,
):
    monkeypatch.setattr(credentials, "_ACTIVE_READ_RETRY_DELAY", 0)
    email = "test@example.com"  # the identity mock_claude_config names as live
    sample_sequence_data["accounts"]["1"]["email"] = email
    mac_switcher._write_json(mac_switcher.sequence_file, sample_sequence_data)
    expired = _creds("rt", 1_000)
    fake_security.items[_ACTIVE] = expired
    fake_security.items[("claude-swap", f"account-1-{email}")] = expired
    if plaintext:
        (temp_home / ".claude" / ".credentials.json").write_text(expired)
    seen = mac_switcher._read_active_credentials()
    mac_switcher._read_account_credentials_ex("1", email)
    assert not seen.degraded  # the collect pass, served from the memo
    fake_security.rc = 36  # locked; the keychain file is untouched
    with _post_replaces(fake_security, 36) as post:
        record = mac_switcher._fetch_active_usage("1", email, seen.value)
    post.assert_not_called()
    assert record.sentinel == USAGE_KEYCHAIN_UNAVAILABLE


def test_a_display_read_beside_a_plaintext_file_is_memoized(
    mac_switcher, fake_security, temp_home
):
    fake_security.items[_ACTIVE] = _creds("rt", 1_000)
    (temp_home / ".claude" / ".credentials.json").write_text(_creds("rt", 1_000))
    reads = [mac_switcher._read_active_credentials() for _ in range(3)]
    assert reads[0] == reads[2]
    assert fake_security.execs == 1


def test_a_memo_hit_is_not_the_keychain_answering(
    mac_switcher, fake_security, monkeypatch
):
    monkeypatch.setattr(credentials, "_ACTIVE_READ_RETRY_DELAY", 0)
    fake_security.items[_ACTIVE] = _creds("rt", 1_000)
    fake_security.items[("claude-swap", f"account-1-{_EMAIL}")] = _creds("rt", 1_000)
    store = mac_switcher._store
    store._read_active_credentials()
    store._read_account_credentials("1", _EMAIL)
    fake_security.rc = 36
    with macos_keychain.fresh_reads():
        store._read_active_credentials()  # a fresh read finds it locked
    assert store._keychain_unreadable
    store._read_account_credentials("1", _EMAIL)  # an idle slot's backup read
    assert store._keychain_unreadable
