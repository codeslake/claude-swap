"""Tests for the "login <countdown>" token: the formatter and its alignment
across every view that lists accounts (CLI list/status, TUI dashboard mini
rows, TUI auto-view Next-best rows)."""

from __future__ import annotations

import dataclasses
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_swap import oauth
from claude_swap.models import AccountSnapshot
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.tui import widgets
from claude_swap.tui.theme import Palette
from claude_swap.usage_store import UsageEntry

from tests.test_tui import FakeSwitcher, fake_engine, make_account, make_app, settle


# ---------------------------------------------------------------------------
# format_login_expiry
# ---------------------------------------------------------------------------


class TestFormatLoginExpiry:
    def test_days_and_hours(self):
        now = 1_000_000.0
        expires_at = now + 23 * 86400 + 4 * 3600
        assert oauth.format_login_expiry(expires_at, False, now) == "23d04h"

    def test_hours_and_minutes(self):
        now = 1_000_000.0
        expires_at = now + 5 * 3600 + 7 * 60
        assert oauth.format_login_expiry(expires_at, False, now) == "5h07m "

    def test_under_an_hour(self):
        now = 1_000_000.0
        expires_at = now + 45 * 60
        assert oauth.format_login_expiry(expires_at, False, now) == "0h45m "

    def test_expired(self):
        now = 1_000_000.0
        assert oauth.format_login_expiry(now - 10, False, now) == "needed"
        assert oauth.format_login_expiry(now, False, now) == "needed"

    def test_quarantined_wins_over_a_future_expiry(self):
        now = 1_000_000.0
        assert oauth.format_login_expiry(now + 999_999, True, now) == "needed"

    def test_unknown(self):
        assert oauth.format_login_expiry(None, False) == "?     "

    def test_every_shape_is_the_same_width(self):
        now = 1_000_000.0
        shapes = [
            oauth.format_login_expiry(None, False),
            oauth.format_login_expiry(now - 1, False, now),
            oauth.format_login_expiry(now + 999_999, True, now),
            oauth.format_login_expiry(now + 23 * 86400 + 4 * 3600, False, now),
            oauth.format_login_expiry(now + 5 * 3600 + 7 * 60, False, now),
            oauth.format_login_expiry(now + 45 * 60, False, now),
        ]
        widths = {len(s) for s in shapes}
        assert widths == {6}


# ---------------------------------------------------------------------------
# CLI list block: `switcher._usage_entry_lines`, via `list_accounts`
# ---------------------------------------------------------------------------


def _iso_in(seconds: float) -> str:
    return (
        (datetime.now(timezone.utc) + timedelta(seconds=seconds))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


class TestCliListAlignment:
    def test_login_column_starts_where_in_countdown_starts(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys
    ):
        sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
        active_creds = json.dumps({"claudeAiOauth": {"accessToken": "sk-active"}})
        # `list_accounts` sources the non-active slot's creds through
        # `_read_account_credentials` (patched below) -- that account is the
        # one that gets real 5h/7d rows to compare against, so its login
        # expiry is the one under test.
        expires_ms = int((time.time() + 23 * 86400 + 4 * 3600) * 1000)
        backup_creds = json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "sk-backup",
                    "refreshTokenExpiresAt": expires_ms,
                }
            }
        )

        usage_response = {
            "five_hour": {"utilization": 10.0, "resets_at": _iso_in(7200)},
            "seven_day": {"utilization": 50.0, "resets_at": _iso_in(3 * 86400)},
        }
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(usage_response).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, sample_sequence_data)

        with patch.object(switcher, "_read_credentials", return_value=active_creds), \
             patch.object(switcher, "_read_account_credentials", return_value=backup_creds), \
             patch("claude_swap.oauth.urllib.request.urlopen", return_value=mock_response):
            switcher.list_accounts()

        output = capsys.readouterr().out
        lines = output.splitlines()
        line_7d = next(line for line in lines if "7d:" in line)
        # The active slot's creds carry no refreshTokenExpiresAt (its login
        # line reads "?"); pick the row with the real value under test.
        line_login = next(line for line in lines if "login:" in line and "?" not in line)

        in_at = line_7d.index("in ") + len("in ")
        # The login row's value is left-padded with spaces up to the same
        # column, then the countdown text itself — find the first non-space
        # character after the "login:" label.
        after_label = line_login.index("login:") + len("login:")
        value_at = after_label + len(line_login[after_label:]) - len(
            line_login[after_label:].lstrip(" ")
        )
        assert value_at == in_at
        assert line_login[value_at:].strip().endswith("h")  # "23d0Nh" shape


# ---------------------------------------------------------------------------
# TUI dashboard inactive rows: `tui.widgets.mini_account_text`
# ---------------------------------------------------------------------------


def _snapshot(number: int, email: str, login_expires_at: float | None) -> AccountSnapshot:
    return AccountSnapshot(
        number=str(number),
        email=email,
        org_name="",
        org_uuid="",
        is_active=False,
        kind="oauth",
        switchable=True,
        usage=UsageEntry(),
        login_expires_at=login_expires_at,
    )


class TestDashboardMiniRowAlignment:
    def test_login_column_starts_at_the_same_offset(self):
        now = time.time()
        accounts = [
            _snapshot(1, "a@example.com", now + 23 * 86400 + 4 * 3600),
            _snapshot(2, "a-much-longer-email@example.com", None),
            _snapshot(3, "mid@example.com", now + 5 * 3600 + 7 * 60),
        ]
        email_width = max(len(a.email) for a in accounts)
        offsets = []
        for acc in accounts:
            text = widgets.mini_account_text(
                acc, now, email_width=email_width, palette=Palette.DARK
            )
            offsets.append(text.plain.index("login"))
        assert len(set(offsets)) == 1


# ---------------------------------------------------------------------------
# TUI auto-view Next-best rows: `AutoScreen._candidates_text`
# ---------------------------------------------------------------------------


def _replace_login(acc: AccountSnapshot, login_expires_at: float | None) -> AccountSnapshot:
    return dataclasses.replace(acc, login_expires_at=login_expires_at)


@pytest.mark.asyncio
async def test_next_best_login_column_alignment(tmp_path, fake_engine):
    now = time.time()
    accounts = [
        make_account(1, active=True, email="active@example.com"),
        _replace_login(make_account(2, email="a@example.com"), now + 23 * 86400 + 4 * 3600),
        _replace_login(
            make_account(3, email="a-much-longer-email@example.com"), None
        ),
        _replace_login(make_account(4, email="mid@example.com"), now + 5 * 3600 + 7 * 60),
    ]
    fake = FakeSwitcher(accounts, tmp_path)
    app = make_app(fake)
    async with app.run_test(size=(100, 40)) as pilot:
        await settle(pilot)
        await pilot.press("g")
        await pilot.pause()
        from textual.widgets import Static

        plain = app.screen.query_one("#candidates", Static).render().plain
        lines = [line for line in plain.splitlines() if "login" in line]
        assert len(lines) == 3
        offsets = {line.index("login") for line in lines}
        assert len(offsets) == 1
