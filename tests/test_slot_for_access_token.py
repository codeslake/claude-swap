"""``ClaudeAccountSwitcher.slot_for_access_token``: the owner proxy names the
account a usage request's bearer belongs to, so a session still on an
account cswap switched away from charges that account's hourly budget."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from claude_swap.exceptions import ClaudeSwitchError, CredentialReadError
from claude_swap.switcher import ClaudeAccountSwitcher

ROSTER = {
    "activeAccountNumber": 1,
    "lastUpdated": "2026-10-03T00:00:00Z",
    "sequence": [1, 2, 3],
    "accounts": {
        "1": {"email": "one@example.com", "uuid": "u1", "added": "x"},
        "2": {"email": "two@example.com", "uuid": "u2", "added": "x"},
        "3": {"email": "three@example.com", "uuid": "u3", "added": "x"},
    },
}


def _creds(token: str) -> str:
    return json.dumps(
        {"claudeAiOauth": {"accessToken": token, "refreshToken": f"r-{token}"}}
    )


@pytest.fixture
def switcher(temp_home: Path, monkeypatch) -> ClaudeAccountSwitcher:
    sw = ClaudeAccountSwitcher()
    sw._setup_directories()
    sw._write_json(sw.sequence_file, ROSTER)
    sw._store._write_backup_enc("1", "one@example.com", _creds("tok-1-stale"))
    sw._store._write_backup_enc("2", "two@example.com", _creds("tok-2-now"))
    sw._store._prev_backup_path("2", "two@example.com").write_text(
        base64.b64encode(_creds("tok-2-before").encode()).decode()
    )
    sw._store._write_backup_enc("3", "three@example.com", _creds("tok-3-now"))
    monkeypatch.setattr(sw, "current_account_number", lambda: "1")
    monkeypatch.setattr(sw, "_read_credentials", lambda: _creds("tok-1-live"))
    return sw


class TestSlotForAccessToken:
    def test_live_login_token_names_the_live_slot(self, switcher):
        assert switcher.slot_for_access_token("tok-1-live") == "1"

    def test_live_slot_backup_token_names_the_live_slot(self, switcher):
        # The live slot's backup can trail the live login; a session that
        # read it is still on slot 1.
        assert switcher.slot_for_access_token("tok-1-stale") == "1"

    def test_other_slot_current_backup_names_that_slot(self, switcher):
        assert switcher.slot_for_access_token("tok-2-now") == "2"
        assert switcher.slot_for_access_token("tok-3-now") == "3"

    def test_other_slot_previous_generation_names_that_slot(self, switcher):
        assert switcher.slot_for_access_token("tok-2-before") == "2"

    def test_unknown_token_is_none(self, switcher):
        assert switcher.slot_for_access_token("tok-nobody") is None
        assert switcher.slot_for_access_token("") is None

    def test_prefix_of_a_stored_token_does_not_match(self, switcher):
        assert switcher.slot_for_access_token("tok-2") is None

    def test_token_under_two_slots_raises(self, switcher):
        switcher._store._write_backup_enc(
            "3", "three@example.com", _creds("tok-2-now")
        )
        with pytest.raises(ClaudeSwitchError, match=r"\['2', '3'\]"):
            switcher.slot_for_access_token("tok-2-now")

    def test_unreadable_backup_with_no_match_raises(self, switcher, monkeypatch):
        real = switcher._store._read_account_credentials_direct

        def direct(num, email, failed=None):
            if num == "3":
                if failed is not None:
                    failed.append(True)
                return ""
            return real(num, email, failed)

        monkeypatch.setattr(
            switcher._store, "_read_account_credentials_direct", direct
        )
        with pytest.raises(CredentialReadError, match=r"\['3'\]"):
            switcher.slot_for_access_token("tok-nobody")
        # A match elsewhere is still an answer: slot 3 cannot hold slot 2's
        # token without the two-slot case above.
        assert switcher.slot_for_access_token("tok-2-now") == "2"

    def test_reads_write_nothing(self, switcher):
        creds_dir = switcher._store._backup_enc_path("1", "one@example.com").parent
        before = sorted(
            (p.name, p.read_bytes()) for p in creds_dir.iterdir() if p.is_file()
        )
        for token in ("tok-1-live", "tok-2-before", "tok-nobody"):
            switcher.slot_for_access_token(token)
        after = sorted(
            (p.name, p.read_bytes()) for p in creds_dir.iterdir() if p.is_file()
        )
        assert after == before
