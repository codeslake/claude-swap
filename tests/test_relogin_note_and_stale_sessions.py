"""Two user-facing claims this PR measured false.

1. ``SENTINEL_NOTES[USAGE_RELOGIN_REQUIRED]`` said "refresh token dead" of the
   ACCOUNT. ``_entry_token_dead`` confirms dead off whichever stored source
   still matches the struck generation, so an active slot whose LIVE
   credential has already rotated past it is condemned on its BACKUP alone —
   deliberately (the strike must survive a live-only compare), but the account
   authenticates fine and the note said otherwise.

2. ``_print_switch_followup``'s "no restart needed". A session already running
   when the credential is replaced can end up rendering "Not logged in ·
   Please run /login", which 2.1.251 raises as ``InvalidRequestHeaderValueError``
   ("request header value rejected before send") — no HTTP status, so none of
   the retry paths that rebuild the client fire, and no rewrite cswap makes can
   reach it. A restart is the only thing that clears it.
"""
import json

import pytest
from unittest.mock import patch

from claude_swap import oauth
from claude_swap.credentials import ActiveCredentials
from claude_swap.json_output import USAGE_RELOGIN_REQUIRED
from claude_swap.switcher import SENTINEL_NOTES, ClaudeAccountSwitcher
from claude_swap.usage_store import FetchRecord

OLD = json.dumps({"claudeAiOauth": {"accessToken": "sk-old",
                                    "refreshToken": "rt-old", "expiresAt": 1000}})
NEW = json.dumps({"claudeAiOauth": {"accessToken": "sk-new", "refreshToken": "rt-new",
                                    "expiresAt": 99999999999000}})


def test_the_relogin_note_does_not_condemn_a_live_credential_that_moved_on(
    temp_home, mock_claude_config, sample_sequence_data, monkeypatch,
):
    """The note renders in a state where the account is NOT dead.

    The premise assert is the load-bearing half: it drives the real collector,
    so it fails if the backup-confirms branch ever stops being reachable and
    the wording assert below stops standing on anything.
    """
    sample_sequence_data["accounts"]["2"]["email"] = "b@example.com"
    s = ClaudeAccountSwitcher()
    s._setup_directories()
    s._write_json(s.sequence_file, sample_sequence_data)
    idents = {"2": ("b@example.com", "")}
    # Struck on the generation the BACKUP still holds.
    s._usage_store.record(
        {"2": FetchRecord(error="invalid_grant",
                          struck_fp=oauth.credential_fingerprint(OLD))},
        idents,
    )
    s._write_account_credentials("2", "b@example.com", OLD)
    # The LIVE credential rotated past it — a healthy, non-degraded read.
    monkeypatch.setattr(s, "_read_active_credentials",
                        lambda: ActiveCredentials(NEW, False, False))
    monkeypatch.setattr(s, "_get_current_account", lambda: ("b@example.com", ""))

    with patch.object(s, "current_account_number", return_value="2"):
        entries = s._collect_usage_entries(s._build_accounts_info(), fetch=set())

    assert entries["2"].sentinel == USAGE_RELOGIN_REQUIRED, (
        "PREMISE BROKEN: the backup-confirms branch did not fire, so this test "
        "no longer measures the state the note is wrong about"
    )
    note = SENTINEL_NOTES[USAGE_RELOGIN_REQUIRED]
    assert "refresh token dead" not in note, (
        f"{note!r} states the account's refresh token is dead; here the live "
        "credential authenticates and only a stored copy was rejected"
    )


@pytest.mark.parametrize("backend", ["keychain", "file"])
@pytest.mark.parametrize("running", [0, 2])
def test_switch_followup_names_the_sessions_that_predate_the_swap(
    backend, running, monkeypatch, capsys,
):
    """Every live session predates a switch that has just committed, so the
    caveat needs no timestamp — only whether any exist."""
    s = ClaudeAccountSwitcher.__new__(ClaudeAccountSwitcher)
    monkeypatch.setattr(
        ClaudeAccountSwitcher, "_last_active_credentials_backend", backend,
    )
    monkeypatch.setattr("claude_swap.switcher.get_running_instances",
                        lambda: ([object()] * running, []))

    s._print_switch_followup()
    out = capsys.readouterr().out

    assert ("already running" in out) is bool(running), (
        f"backend={backend} running={running}: got {out!r}"
    )
    if running:
        assert "Not logged in" in out and str(running) in out, (
            f"the caveat must name the count and the symptom: {out!r}"
        )
