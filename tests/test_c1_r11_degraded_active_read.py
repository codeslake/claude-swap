"""R11 CRITICAL: is the collector's own ACTIVE read a third collapse site?

`_build_accounts_info:3188` does `creds = active.value or ""` and records
`_active_read_degraded`, but the quarantine scan passes those bytes to
`_entry_token_dead` and `_static_usage_sentinel:3905` has only a
`keychain_unavailable` arm. A DEGRADED read returns BYTES, so every
"empty read" guard is bypassed.

The probe is INSTRUMENTED: it prints the strike state it actually built, so a
`sentinel=None` from never reaching the path cannot be mistaken for a fix.
"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.credentials import ActiveCredentials
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.json_output import USAGE_RELOGIN_REQUIRED
from claude_swap.models import Platform
from claude_swap.usage_store import FetchRecord as FR

OLD = json.dumps({"claudeAiOauth": {"accessToken": "sk-old",
                                    "refreshToken": "rt-old", "expiresAt": 1000}})
NEW = json.dumps({"claudeAiOauth": {"accessToken": "sk-new",
                                    "refreshToken": "rt-new", "expiresAt": 99999999999000}})


@pytest.mark.parametrize("degraded", [False, True], ids=["CONTROL-healthy", "PROBE-degraded"])
def test_degraded_active_read_must_not_condemn_a_healed_slot(
    degraded, temp_home: Path, mock_claude_config: Path,
    sample_sequence_data: dict, monkeypatch,
):
    sample_sequence_data["accounts"]["2"]["email"] = "b@example.com"
    s = ClaudeAccountSwitcher()
    s.platform = Platform.MACOS
    s._setup_directories()
    s._write_json(s.sequence_file, sample_sequence_data)
    idents = {"2": ("b@example.com", "")}
    s._usage_store.record(
        {"2": FR(error="invalid_grant", struck_fp=oauth.credential_fingerprint(OLD))},
        idents,
    )
    # HEALED: backup now holds the new generation; the struck fp matches nothing.
    s._write_account_credentials("2", "b@example.com", NEW)

    pre = s._usage_store.entries(idents, [])["2"]
    print(f"\n  [{'degraded' if degraded else 'healthy '}] PREMISE "
          f"strikes={pre.auth_dead_strikes} token_dead={pre.token_dead()}")

    # Patch the object _build_accounts_info ACTUALLY calls (self, not _store).
    # A degraded read serves the STALE generation — that is what "degraded"
    # means: the Keychain read failed and a plaintext fallback covered it,
    # so the bytes may be the superseded ones. Serving NEW here (my first
    # cut) can never condemn anything, which is why both rows passed.
    served = OLD if degraded else NEW
    monkeypatch.setattr(s, "_read_active_credentials",
                        lambda: ActiveCredentials(served, False, degraded))
    # `_build_accounts_info` derives active_num from _get_current_account()
    # (the live IDENTITY), NOT from current_account_number(). Patching the
    # latter left every row is_active=False, so the branch under test never
    # ran and both rows "passed" for the wrong reason.
    monkeypatch.setattr(s, "_get_current_account",
                        lambda: ("b@example.com", ""))
    with patch.object(s, "current_account_number", return_value="2"):
        info = s._build_accounts_info()
        print("  info rows:", [(r[0], r[4], (r[5] or '')[:12]) for r in info])
        entries = s._collect_usage_entries(info, fetch=set())

    e = entries["2"]
    print(f"  [{'degraded' if degraded else 'healthy '}] RESULT  "
          f"sentinel={e.sentinel!r}  _active_read_degraded="
          f"{getattr(s, '_active_read_degraded', '<absent>')}")
    assert e.sentinel != USAGE_RELOGIN_REQUIRED, (
        "an already-healed active slot was condemned on a degraded read"
    )
