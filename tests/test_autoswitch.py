"""Tests for the auto-switch engine (autoswitch.py)."""

from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth, poll_policy
from claude_swap.autoswitch import (
    IDLE_HOLD_MAX_S,
    NO_RESET_FALLBACK_S,
    AllExhaustedEvent,
    AutoSwitchEngine,
    ConfigWarningEvent,
    ErrorEvent,
    NoSwitchEvent,
    PollEvent,
    QuarantineEvent,
    SwitchEvent,
    TickOutcome,
    UnquarantineEvent,
    pct_label,
)
from claude_swap.json_output import USAGE_FOREIGN_CREDENTIAL, USAGE_TOKEN_EXPIRED
from claude_swap.usage_store import FetchRecord, UsageEntry
from claude_swap.models import Platform
from claude_swap.settings import AutoSwitchSettings
from claude_swap.switcher import ClaudeAccountSwitcher


class FakeClock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _usage(pct: float, resets_at: str | None = None) -> dict:
    window: dict = {"pct": pct}
    if resets_at:
        window["resets_at"] = resets_at
    return {"five_hour": window, "seven_day": {"pct": 0.0}}


def _entry_for(value: dict | str | None, now: float) -> UsageEntry:
    """Synthesize the store entry a live fetch would have produced."""
    if isinstance(value, dict):
        return UsageEntry(last_good=value, fetched_at=now, age_s=0.0)
    if isinstance(value, str):
        return UsageEntry(sentinel=value)
    return UsageEntry()


class EngineHarness:
    """Seeded switcher + engine + captured events, on the Linux file backend."""

    def __init__(self, temp_home: Path, **settings_kwargs):
        self.temp_home = temp_home
        self.switcher = ClaudeAccountSwitcher()
        self.switcher.platform = Platform.LINUX
        self.switcher._setup_directories()
        self.switcher._init_sequence_file()
        self.settings = AutoSwitchSettings(**settings_kwargs)
        self.events: list = []
        self.clock = FakeClock()
        # Keep the usage store on the same fake clock as the engine so
        # freshness/claims/poll scheduling are deterministic in tests.
        self.switcher._usage_store.clock = self.clock
        self.engine = self._make_engine()

    def _make_engine(self, **kwargs) -> AutoSwitchEngine:
        return AutoSwitchEngine(
            self.switcher,
            self.settings,
            self.events.append,
            clock=self.clock,
            **kwargs,
        )

    def seed(self, num: int, email: str, *, expires_at: int | None = None) -> None:
        oauth_blob: dict = {
            "accessToken": f"sk-{num}",
            "refreshToken": f"rt-{num}",
        }
        if expires_at is not None:
            oauth_blob["expiresAt"] = expires_at
        self.switcher._write_account_credentials(
            str(num), email, json.dumps({"claudeAiOauth": oauth_blob})
        )
        self.switcher._write_account_config(
            str(num),
            email,
            json.dumps({
                "oauthAccount": {"emailAddress": email, "accountUuid": f"uuid-{num}"},
            }),
        )
        data = self.switcher._get_sequence_data()
        data["accounts"][str(num)] = {
            "email": email,
            "uuid": f"uuid-{num}",
            "organizationUuid": "",
            "organizationName": "",
            "added": "2024-01-01T00:00:00Z",
        }
        if num not in data["sequence"]:
            data["sequence"].append(num)
            data["sequence"].sort()
        if data["activeAccountNumber"] is None:
            data["activeAccountNumber"] = num
        self.switcher._write_json(self.switcher.sequence_file, data)

    def make_live(self, email: str, num: int) -> None:
        (self.temp_home / ".claude" / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {"accessToken": "sk-live", "refreshToken": "rt-live"},
        }))
        (self.temp_home / ".claude.json").write_text(json.dumps({
            "oauthAccount": {"emailAddress": email, "accountUuid": f"uuid-{num}"},
        }))

    def tick_with_usage(self, usage: dict) -> TickOutcome:
        entries = {
            num: _entry_for(value, self.clock.now) for num, value in usage.items()
        }
        return self.tick_with_entries(entries)

    def tick_with_entries(self, entries: dict[str, UsageEntry]) -> TickOutcome:
        with patch.object(
            self.switcher, "usage_entries_by_account", return_value=entries
        ):
            return self.engine.tick()

    def active_number(self) -> int | None:
        return self.switcher._get_sequence_data()["activeAccountNumber"]

    def kinds(self) -> list[str]:
        return [e.kind for e in self.events]

    def state(self) -> dict:
        path = self.switcher.backup_dir / "autoswitch_state.json"
        if not path.exists():
            return {}
        return json.loads(path.read_text())


@pytest.fixture
def harness(temp_home: Path) -> EngineHarness:
    h = EngineHarness(temp_home)
    h.seed(1, "a@example.com")
    h.seed(2, "b@example.com")
    h.seed(3, "c@example.com")
    h.make_live("a@example.com", 1)
    return h


class TestDecisionTable:
    def test_below_threshold_is_no_action(self, harness):
        outcome = harness.tick_with_usage({
            "1": _usage(50), "2": _usage(10), "3": _usage(10),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert harness.active_number() == 1
        reasons = [e.reason for e in harness.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["below-threshold"]

    def test_over_threshold_switches_to_max_headroom(self, harness):
        outcome = harness.tick_with_usage({
            "1": _usage(95), "2": _usage(40), "3": _usage(20),
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "proactive"
        assert switch.to_ref == {"number": 3, "email": "c@example.com"}
        assert harness.state()["lastSwitchTo"] == "3"

    def test_no_active_account(self, temp_home):
        h = EngineHarness(temp_home)
        assert h.engine.tick() is TickOutcome.NO_ACTION
        assert [e.reason for e in h.events if isinstance(e, NoSwitchEvent)] == [
            "no-active-account"
        ]

    def test_hysteresis_margin_blocks_marginal_candidates(self, harness):
        # threshold 90, hysteresis 10 → a candidate must beat the active
        # account's utilization by >= 10 points; 95→86 is only 9 better.
        # Failing the margin is NOT exhaustion: no all-exhausted event, no
        # reset-sleep — the next tick must stay at normal cadence so the
        # at-limit escape isn't missed when the active account tops out.
        outcome = harness.tick_with_usage({
            "1": _usage(95), "2": _usage(86), "3": _usage(88),
        })
        assert outcome is TickOutcome.BLOCKED
        assert harness.active_number() == 1
        assert not any(isinstance(e, AllExhaustedEvent) for e in harness.events)
        reasons = [e.reason for e in harness.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["no-qualifying-candidate"]
        assert harness.engine._sleep_until_ts is None
        delay = harness.engine._next_delay(outcome)
        assert delay <= 1.1 * harness.settings.interval_seconds

    def test_issue_115_strictly_better_candidate_switches(self, harness):
        # Regression for #115: active bound by 5h (99%), candidate bound by
        # 7d (89%). The old absolute bar (<= 80% used) vetoed the candidate;
        # the relative gate takes it: 89 < 90 and 99 - 89 >= 10.
        outcome = harness.tick_with_usage({
            "1": {"five_hour": {"pct": 99.0}, "seven_day": {"pct": 24.0}},
            "2": {"five_hour": {"pct": 3.0}, "seven_day": {"pct": 89.0}},
            "3": {"five_hour": {"pct": 95.0}, "seven_day": {"pct": 10.0}},
        })
        assert outcome is TickOutcome.SWITCHED
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "proactive"
        assert harness.active_number() == 2

    def test_proactive_never_lands_at_or_over_threshold(self, temp_home):
        # threshold 80, hysteresis 5: the candidate at 85% is five points
        # better than the active 90%, but it already sits over the threshold
        # and would re-trigger on the very next tick — blocked.
        h = EngineHarness(temp_home, threshold=80.0, hysteresis_pct=5.0)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)
        outcome = h.tick_with_usage({"1": _usage(90), "2": _usage(85)})
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["no-qualifying-candidate"]

    def test_stable_landing_does_not_switch_back(self, temp_home):
        # Cooldown disabled so only the gate itself prevents flapping: after
        # 99→89 the roles reverse, and the old account (99%) can never beat
        # the new active (89%) — the move is one-way.
        h = EngineHarness(temp_home, cooldown_seconds=0.0)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)
        usage = {
            "1": {"five_hour": {"pct": 99.0}, "seven_day": {"pct": 24.0}},
            "2": {"five_hour": {"pct": 3.0}, "seven_day": {"pct": 89.0}},
        }
        assert h.tick_with_usage(usage) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.clock.advance(60)
        assert h.tick_with_usage(usage) is TickOutcome.NO_ACTION
        assert h.active_number() == 2
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["below-threshold"]

    def test_mixed_unknown_and_exhausted_is_not_all_exhausted(self, harness):
        # One candidate at its limit, the other unreadable this tick: usage
        # could recover any moment, so no long reset-sleep.
        outcome = harness.tick_with_usage({
            "1": _usage(95),
            "2": _usage(100, "2026-07-03T12:00:00Z"),
            "3": None,
        })
        assert outcome is TickOutcome.BLOCKED
        assert not any(isinstance(e, AllExhaustedEvent) for e in harness.events)
        reasons = [e.reason for e in harness.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["no-qualifying-candidate"]
        assert harness.engine._sleep_until_ts is None
        delay = harness.engine._next_delay(outcome)
        assert delay <= 1.1 * harness.settings.interval_seconds

    def test_stale_beyond_trust_blocks_all_exhausted(self, harness):
        # One candidate exhausted on trusted-stale data, the other's data aged
        # past every trust window (no failures, no plan — just overdue): the
        # unknown candidate could be viable, so no long reset-sleep.
        now = harness.clock.now
        reset = "2026-07-05T12:00:00Z"
        outcome = harness.tick_with_entries({
            "1": UsageEntry(last_good=_usage(95), fetched_at=now, age_s=0.0),
            "2": UsageEntry(
                last_good=_usage(100, reset), fetched_at=now - 400, age_s=400.0,
                consecutive_failures=1, trust_extended=True,
            ),
            "3": UsageEntry(last_good=_usage(10), fetched_at=now - 400, age_s=400.0),
        })
        assert outcome is TickOutcome.BLOCKED
        assert not any(isinstance(e, AllExhaustedEvent) for e in harness.events)
        reasons = [e.reason for e in harness.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["no-qualifying-candidate"]

    def test_trusted_stale_exhausted_set_still_fires_all_exhausted(self, harness):
        # Every candidate at its limit, known only through trusted-stale data
        # (in failure state) — that is still "known and exhausted".
        now = harness.clock.now
        reset = "2026-07-05T12:00:00Z"
        stale_exhausted = UsageEntry(
            last_good=_usage(100, reset), fetched_at=now - 400, age_s=400.0,
            consecutive_failures=1, trust_extended=True,
        )
        outcome = harness.tick_with_entries({
            "1": UsageEntry(last_good=_usage(95), fetched_at=now, age_s=0.0),
            "2": stale_exhausted,
            "3": stale_exhausted,
        })
        assert outcome is TickOutcome.BLOCKED
        exhausted = next(
            e for e in harness.events if isinstance(e, AllExhaustedEvent)
        )
        assert exhausted.earliest_reset_at == reset

    def test_cooldown_suppresses_proactive(self, harness):
        harness.engine._mutate_state(
            lambda s: s.update(lastSwitchAt=harness.clock() - 10)
        )
        outcome = harness.tick_with_usage({
            "1": _usage(95), "2": _usage(10), "3": _usage(10),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert [e.reason for e in harness.events if isinstance(e, NoSwitchEvent)] == [
            "cooldown"
        ]

    def test_at_limit_bypasses_cooldown(self, harness):
        harness.engine._mutate_state(
            lambda s: s.update(lastSwitchAt=harness.clock() - 10)
        )
        outcome = harness.tick_with_usage({
            "1": _usage(100), "2": _usage(10), "3": _usage(50),
        })
        assert outcome is TickOutcome.SWITCHED
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "at-limit"
        assert harness.active_number() == 2

    def test_cooldown_expires(self, harness):
        harness.engine._mutate_state(
            lambda s: s.update(lastSwitchAt=harness.clock())
        )
        harness.clock.advance(400)  # past the 300s default cooldown
        outcome = harness.tick_with_usage({
            "1": _usage(95), "2": _usage(10), "3": _usage(50),
        })
        assert outcome is TickOutcome.SWITCHED

    def test_unknown_active_usage_waits_then_fails_over(self, harness):
        usage = {"1": None, "2": _usage(10), "3": _usage(50)}
        assert harness.tick_with_usage(usage) is TickOutcome.NO_ACTION
        assert harness.tick_with_usage(usage) is TickOutcome.NO_ACTION
        assert harness.tick_with_usage(usage) is TickOutcome.SWITCHED
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "failover"
        assert harness.active_number() == 2

    def test_known_active_usage_resets_unhealthy_counter(self, harness):
        unknown = {"1": None, "2": _usage(10), "3": _usage(10)}
        healthy = {"1": _usage(50), "2": _usage(10), "3": _usage(10)}
        harness.tick_with_usage(unknown)
        harness.tick_with_usage(unknown)
        harness.tick_with_usage(healthy)  # resets the counter
        assert harness.tick_with_usage(unknown) is TickOutcome.NO_ACTION
        assert harness.active_number() == 1

    def test_all_candidates_unknown_is_no_comparison(self, harness):
        outcome = harness.tick_with_usage({
            "1": _usage(95), "2": None, "3": None,
        })
        assert outcome is TickOutcome.BLOCKED
        assert [e.reason for e in harness.events if isinstance(e, NoSwitchEvent)] == [
            "no-comparison"
        ]

    def test_tie_resolves_to_earliest_slot(self, harness):
        outcome = harness.tick_with_usage({
            "1": _usage(95), "2": _usage(30), "3": _usage(30),
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2

    def test_candidate_not_better_than_active_is_skipped(self, harness):
        # Active 91% used (9 headroom); candidates worse or equal → exhausted.
        outcome = harness.tick_with_usage({
            "1": _usage(91), "2": _usage(95), "3": _usage(99),
        })
        assert outcome is TickOutcome.BLOCKED
        assert harness.active_number() == 1

    def test_at_limit_escapes_hysteresis_bar(self, harness):
        # Active hard at 100%; the only room anywhere is a candidate at 85%,
        # which the proactive hysteresis bar (<=80%) would reject. At-limit is
        # an escape: any account with real headroom beats a blocked one.
        outcome = harness.tick_with_usage({
            "1": _usage(100), "2": _usage(85), "3": _usage(97),
        })
        assert outcome is TickOutcome.SWITCHED
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "at-limit"
        assert harness.active_number() == 2

    def test_at_limit_never_targets_another_at_limit_account(self, harness):
        outcome = harness.tick_with_usage({
            "1": _usage(100), "2": _usage(100), "3": _usage(100),
        })
        assert outcome is TickOutcome.BLOCKED
        assert harness.active_number() == 1

    def test_failover_ignores_hysteresis_bar(self, harness):
        # Active usage unreadable (auth likely dead); the only candidate with
        # room sits above the hysteresis bar — failover takes it anyway.
        usage = {"1": None, "2": _usage(85), "3": _usage(100)}
        harness.tick_with_usage(usage)
        harness.tick_with_usage(usage)
        outcome = harness.tick_with_usage(usage)
        assert outcome is TickOutcome.SWITCHED
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "failover"
        assert harness.active_number() == 2

    def test_unmanaged_live_login_is_never_touched(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        # The user logged in with an account cswap doesn't manage.
        h.make_live("stranger@example.com", 9)
        live_before = (temp_home / ".claude" / ".credentials.json").read_text()
        outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})
        assert outcome is TickOutcome.NO_ACTION
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["unmanaged-active-account"]
        assert (temp_home / ".claude" / ".credentials.json").read_text() == live_before

    def test_all_exhausted_carries_earliest_reset(self, harness):
        outcome = harness.tick_with_usage({
            "1": _usage(100, "2026-07-03T12:00:00Z"),
            "2": _usage(100, "2026-07-03T10:30:00Z"),
            "3": _usage(100, "2026-07-03T11:00:00Z"),
        })
        assert outcome is TickOutcome.BLOCKED
        event = next(e for e in harness.events if isinstance(e, AllExhaustedEvent))
        assert event.earliest_reset_at == "2026-07-03T10:30:00Z"
        assert harness.engine._sleep_until_ts is not None

    @pytest.mark.parametrize("offset", [-60.0, 0.0])
    def test_all_exhausted_ignores_non_future_reset(self, harness, offset):
        from datetime import datetime, timezone

        reset = (
            datetime.fromtimestamp(harness.clock.now + offset, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        outcome = harness.tick_with_usage({
            "1": _usage(100, reset),
            "2": _usage(100, reset),
            "3": _usage(100, reset),
        })
        assert outcome is TickOutcome.BLOCKED
        event = next(e for e in harness.events if isinstance(e, AllExhaustedEvent))
        assert event.earliest_reset_at is None
        assert harness.engine._sleep_until_ts is None
        assert harness.engine._next_delay(outcome) == NO_RESET_FALLBACK_S


class TestIdleHold:
    """Active token expired while Claude Code owns it → hold, don't fail over."""

    _HELD = {"1": USAGE_TOKEN_EXPIRED, "2": _usage(10), "3": _usage(20)}

    def test_token_expired_holds_instead_of_failover(self, harness):
        for _ in range(6):  # far past unhealthy_ticks (3)
            assert harness.tick_with_usage(self._HELD) is TickOutcome.NO_ACTION
            harness.clock.advance(60)
        assert harness.active_number() == 1
        assert not any(isinstance(e, SwitchEvent) for e in harness.events)
        reasons = {e.reason for e in harness.events if isinstance(e, NoSwitchEvent)}
        assert reasons == {"active-idle"}
        assert harness.engine._unhealthy_ticks == 0

    def test_idle_hold_slows_cadence(self, harness):
        outcome = harness.tick_with_usage(self._HELD)
        assert outcome is TickOutcome.NO_ACTION
        assert harness.engine._next_delay(outcome) >= NO_RESET_FALLBACK_S

    def test_idle_hold_cap_escalates_to_failover(self, harness):
        assert harness.tick_with_usage(self._HELD) is TickOutcome.NO_ACTION
        harness.clock.advance(IDLE_HOLD_MAX_S + 1)
        # Past the cap the sentinel counts as unhealthy again → failover after
        # unhealthy_ticks (3) consecutive ticks.
        assert harness.tick_with_usage(self._HELD) is TickOutcome.NO_ACTION
        assert harness.tick_with_usage(self._HELD) is TickOutcome.NO_ACTION
        assert harness.tick_with_usage(self._HELD) is TickOutcome.SWITCHED
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "failover"

    def test_recovery_resets_the_hold_clock(self, harness):
        healthy = {"1": _usage(50), "2": _usage(10), "3": _usage(20)}
        harness.tick_with_usage(self._HELD)
        harness.clock.advance(IDLE_HOLD_MAX_S - 60)
        harness.tick_with_usage(healthy)  # user came back; token refreshed
        harness.clock.advance(120)
        # New expiry long after: the hold clock restarted, so still held.
        assert harness.tick_with_usage(self._HELD) is TickOutcome.NO_ACTION
        assert harness.engine._unhealthy_ticks == 0
        assert harness.active_number() == 1

    def test_plain_fetch_failure_still_counts_unhealthy(self, harness):
        # A None (network failure / dead creds) is NOT the idle sentinel:
        # unhealthy counting and the hold clock reset both apply.
        harness.tick_with_usage(self._HELD)
        unknown = {"1": None, "2": _usage(10), "3": _usage(20)}
        assert harness.tick_with_usage(unknown) is TickOutcome.NO_ACTION
        assert harness.engine._unhealthy_ticks == 1
        assert harness.engine._idle_hold_since is None

    def test_foreign_credential_sentinel_fails_over_instead_of_holding(
        self, harness
    ):
        """The foreign sentinel (live credential proven to be another
        account's) must NOT idle-hold like TOKEN_EXPIRED: holding preserves
        the drift, while the failover switch stashes the foreign credential
        and restores the slot's backup — the switch IS the repair."""
        foreign = {
            "1": USAGE_FOREIGN_CREDENTIAL, "2": _usage(10), "3": _usage(20),
        }
        assert harness.tick_with_usage(foreign) is TickOutcome.NO_ACTION
        assert harness.tick_with_usage(foreign) is TickOutcome.NO_ACTION
        assert harness.tick_with_usage(foreign) is TickOutcome.SWITCHED
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert switch.trigger == "failover"
        assert harness.engine._idle_hold_since is None


class TestAdaptiveScheduler:
    """End-to-end through the real store: O(1) baseline, escalations,
    skip-to-reset, movement-based cadence."""

    @pytest.fixture(autouse=True)
    def _no_profile_probe(self):
        """Collect passes whose active credential drifted from the slot
        backup probe the profile oracle before resyncing — unpatched, a real
        HTTP call. "Probe failed" (resync skipped) is inert for scheduler
        behavior."""
        with patch(
            "claude_swap.oauth.fetch_oauth_profile", return_value=None
        ):
            yield

    def _harness(self, temp_home, monkeypatch, accounts=3, **settings_kwargs):
        monkeypatch.setattr("claude_swap.switcher._FETCH_STAGGER_S", 0)
        h = EngineHarness(temp_home, **settings_kwargs)
        emails = ["a@example.com", "b@example.com", "c@example.com"]
        for num in range(1, accounts + 1):
            h.seed(num, emails[num - 1])
        h.make_live("a@example.com", 1)
        monkeypatch.setattr(h.switcher, "_live_session_pids", lambda *a: [])
        return h

    @staticmethod
    def _counting_fetch(counts, usage_by_num, errors_by_num=None):
        def fake(num, email, creds, is_active=False, persist_credentials=None):
            counts[num] = counts.get(num, 0) + 1
            error = (errors_by_num or {}).get(num)
            if error:
                return oauth.UsageOutcome(None, error=error)
            value = usage_by_num.get(num)
            return oauth.UsageOutcome(dict(value) if value else None)
        return fake

    def _tick(self, h, counts, usage_by_num, errors_by_num=None):
        with patch(
            "claude_swap.oauth.try_fetch_usage_for_account",
            side_effect=self._counting_fetch(counts, usage_by_num, errors_by_num),
        ):
            return h.engine.tick()

    def test_baseline_fetches_active_plus_one_candidate(self, temp_home, monkeypatch):
        h = self._harness(temp_home, monkeypatch)
        usage = {"1": _usage(50), "2": _usage(10), "3": _usage(20)}
        counts: dict[str, int] = {}
        # t0: active (never fetched) + the stalest candidate.
        self._tick(h, counts, usage)
        assert counts == {"1": 1, "2": 1}
        # t60: active planned MIN_INTERVAL_S out; the never-fetched candidate
        # is the due one.
        h.clock.advance(60)
        self._tick(h, counts, usage)
        assert counts == {"1": 1, "2": 1, "3": 1}
        # t120: nobody due — everyone served from the store.
        h.clock.advance(60)
        self._tick(h, counts, usage)
        assert counts == {"1": 1, "2": 1, "3": 1}
        # t180: the active account's plan comes due.
        h.clock.advance(60)
        self._tick(h, counts, usage)
        assert counts == {"1": 2, "2": 1, "3": 1}

    def test_near_threshold_escalates_to_full_refresh(self, temp_home, monkeypatch):
        # threshold 90, margin 15 → active at 80% is within the escalation band.
        h = self._harness(temp_home, monkeypatch)
        counts: dict[str, int] = {}
        outcome = self._tick(
            h, counts, {"1": _usage(80), "2": _usage(10), "3": _usage(20)}
        )
        assert outcome is TickOutcome.NO_ACTION  # still below the threshold
        assert counts == {"1": 1, "2": 1, "3": 1}  # but everyone got refreshed

    def test_active_unknown_escalates_before_failover(self, temp_home, monkeypatch):
        h = self._harness(temp_home, monkeypatch, unhealthy_ticks=1)
        counts: dict[str, int] = {}
        outcome = self._tick(
            h, counts,
            {"2": _usage(10), "3": _usage(50)},
            errors_by_num={"1": "timeout"},
        )
        # Candidate data was refreshed in the same tick the failover ran on.
        assert counts == {"1": 1, "2": 1, "3": 1}
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_active_cadence_floor_and_decay(self, temp_home, monkeypatch):
        # The active account polls at MIN_INTERVAL_S first; unmoved usage
        # decays the interval ×1.5 toward ACTIVE_MAX_INTERVAL_S.
        h = self._harness(temp_home, monkeypatch, accounts=2)
        usage = {"1": _usage(10), "2": _usage(20)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)  # never-fetched → fetched
        assert counts["1"] == 1
        for _ in range(2):  # ages 60s and 120s — inside the 180s floor
            h.clock.advance(60)
            self._tick(h, counts, usage)
        assert counts["1"] == 1
        h.clock.advance(60)  # age 180s → due again
        self._tick(h, counts, usage)
        assert counts["1"] == 2
        # Unmoved → interval decayed to 270s: not due at +240, due at +300.
        h.clock.advance(240)
        self._tick(h, counts, usage)
        assert counts["1"] == 2
        h.clock.advance(60)
        self._tick(h, counts, usage)
        assert counts["1"] == 3

    def test_urgent_cadence_when_burning_near_the_band(self, temp_home, monkeypatch):
        # Active moving inside the escalation band → 60s urgent cadence, so
        # a threshold crossing is seen within a minute of the previous poll.
        h = self._harness(temp_home, monkeypatch, accounts=2)
        usage = {"1": _usage(70), "2": _usage(10)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        usage["1"] = _usage(80)  # burning: +10 pts, now inside the band
        h.clock.advance(180)
        self._tick(h, counts, usage)  # movement + in band → urgent plan
        assert counts["1"] == 2
        usage["1"] = _usage(84)
        h.clock.advance(60)
        self._tick(h, counts, usage)  # urgent plan due after only 60s
        assert counts["1"] == 3

    def test_in_band_without_movement_keeps_the_floor(self, temp_home, monkeypatch):
        # In the escalation band but not burning: no urgency — the normal
        # 180s floor applies (escalation keeps candidates fresh; it must not
        # re-fetch a fresh, unmoving active every tick).
        h = self._harness(temp_home, monkeypatch, accounts=2)
        usage = {"1": _usage(80), "2": _usage(10)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        for _ in range(2):
            h.clock.advance(60)
            self._tick(h, counts, usage)
        assert counts["1"] == 1  # not due inside the floor
        h.clock.advance(60)
        self._tick(h, counts, usage)
        assert counts["1"] == 2

    def test_urgent_band_follows_the_threshold(self, temp_home, monkeypatch):
        # The urgent band is distance-to-threshold, not absolute pct: with
        # threshold 50 (band edge 35), movement at 40% engages the urgent
        # cadence that the default threshold would ignore.
        h = self._harness(temp_home, monkeypatch, accounts=2, threshold=50)
        usage = {"1": _usage(30), "2": _usage(10)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        usage["1"] = _usage(40)
        h.clock.advance(180)
        self._tick(h, counts, usage)  # movement inside the 35..50 band
        assert counts["1"] == 2
        usage["1"] = _usage(44)
        h.clock.advance(60)
        self._tick(h, counts, usage)  # urgent plan due after only 60s
        assert counts["1"] == 3

    def test_stale_candidate_plan_never_gates_the_active(
        self, temp_home, monkeypatch
    ):
        # Role change outside a cswap switch (e.g. manual login): the active
        # slot can carry a plan written while it was an idle candidate, up to
        # 600s out. The ACTIVE_MAX_INTERVAL_S age cap overrides it.
        h = self._harness(temp_home, monkeypatch, accounts=2)
        usage = {"1": _usage(50), "2": _usage(20)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        h.switcher._usage_store.set_poll_plan(
            {"1": (h.clock.now + 600.0, 600.0)}, {"1": ("a@example.com", "")}
        )
        h.clock.advance(240)  # inside the bogus plan, under the age cap
        self._tick(h, counts, usage)
        assert counts["1"] == 1
        h.clock.advance(120)  # age 360 ≥ ACTIVE_MAX_INTERVAL_S
        self._tick(h, counts, usage)
        assert counts["1"] == 2

    def test_exhausted_active_is_rechecked_before_its_reset(
        self, temp_home, monkeypatch
    ):
        from datetime import datetime, timezone

        h = self._harness(temp_home, monkeypatch, accounts=1)
        reset_ts = h.clock.now + 7200.0
        reset_iso = (
            datetime.fromtimestamp(reset_ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        usage = {"1": _usage(100, reset_iso)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        assert counts["1"] == 1
        for _ in range(3):
            h.clock.advance(400)
            self._tick(h, counts, usage)
        assert counts["1"] == 2

    def test_engine_repairs_legacy_reset_parked_active_plan(
        self, temp_home, monkeypatch
    ):
        from datetime import datetime, timezone

        h = self._harness(temp_home, monkeypatch, accounts=1)
        reset_ts = h.clock.now + 86_400.0
        reset_iso = (
            datetime.fromtimestamp(reset_ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        usage = {"1": _usage(100, reset_iso)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        h.switcher._usage_store.set_poll_plan(
            {"1": (reset_ts, 300.0)}, {"1": ("a@example.com", "")}
        )

        h.clock.advance(400)
        self._tick(h, counts, usage)
        assert counts["1"] == 2
        entry = h.switcher._usage_store.entries(
            {"1": ("a@example.com", "")}
        )["1"]
        assert entry.next_poll_at is not None
        assert entry.next_poll_at < reset_ts

    def test_band_jump_is_seen_at_most_one_poll_late(
        self, temp_home, monkeypatch
    ):
        # Active at 40% jumps into the band between polls: the jump is picked
        # up on the next planned poll, escalates the same tick, and the
        # movement flips the active onto the urgent cadence.
        h = self._harness(temp_home, monkeypatch, accounts=2)
        usage = {"1": _usage(40), "2": _usage(20)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        usage["1"] = _usage(80)
        h.clock.advance(60)
        self._tick(h, counts, usage)  # plan-skipped: still believed at 40%
        assert counts["1"] == 1
        h.clock.advance(120)
        self._tick(h, counts, usage)  # planned poll sees 80% → escalate-all
        assert counts["1"] == 2
        assert counts["2"] == 1  # at the TTL edge: still served, not refetched
        h.clock.advance(60)
        self._tick(h, counts, usage)  # movement in band → urgent cadence
        assert counts["1"] == 3
        assert counts["2"] == 2  # now stale → the escalation refreshes it

    def test_active_in_backoff_keeps_trusted_headroom(self, temp_home, monkeypatch):
        # The active account's fetches are being refused (429 with a long
        # Retry-After). Its last-good data ages past STALE_OK_S, but the
        # staleness is deliberate: headroom stays known, so no unhealthy
        # ticks and no escalate-all burst while the server is rate limiting.
        h = self._harness(temp_home, monkeypatch)
        usage = {"1": _usage(50), "2": _usage(10), "3": _usage(20)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        h.clock.advance(60)
        self._tick(h, counts, usage)
        h.switcher._usage_store.record(
            {"1": FetchRecord(error="http-429", retry_after_s=600.0)},
            {"1": ("a@example.com", "")},
        )
        h.clock.advance(400)  # active data now well past STALE_OK_S, in backoff
        counts.clear()
        outcome = self._tick(h, counts, usage)
        assert outcome is TickOutcome.NO_ACTION
        assert h.engine._unhealthy_ticks == 0
        assert "1" not in counts  # backoff respected
        assert sum(counts.values()) == 1  # baseline slot only, no escalate-all

    def test_all_exhausted_escalation_preserves_wider_plan(
        self, temp_home, monkeypatch
    ):
        h = self._harness(temp_home, monkeypatch)
        usage = {num: _usage(100) for num in ("1", "2", "3")}
        counts: dict[str, int] = {}
        assert self._tick(h, counts, usage) is TickOutcome.BLOCKED
        assert counts == {"1": 1, "2": 1, "3": 1}

        # Simulate the wider plan learned after repeated 429s. The next
        # all-exhausted wake may refresh other stale rows, but escalation must
        # not defeat this token's congestion-control interval.
        h.switcher._usage_store.set_poll_plan(
            {"2": (h.clock.now + 1800.0, 1800.0)},
            {"2": ("b@example.com", "")},
        )
        h.clock.advance(NO_RESET_FALLBACK_S)
        assert self._tick(h, counts, usage) is TickOutcome.BLOCKED
        assert counts["2"] == 1

    def test_exhausted_candidate_keeps_a_bounded_poll_plan(
        self, temp_home, monkeypatch
    ):
        h = self._harness(temp_home, monkeypatch)
        reset_iso = "2026-07-05T12:00:00Z"
        usage = {"1": _usage(50), "2": _usage(100, reset_iso), "3": _usage(20)}
        counts: dict[str, int] = {}
        for _ in range(3):
            self._tick(h, counts, usage)
            h.clock.advance(60)
        assert counts["2"] == 1
        entry = h.switcher._usage_store.entries(
            {"2": ("b@example.com", "")}
        )["2"]
        assert entry.poll_interval_s == poll_policy.EXHAUSTED_INTERVAL_S
        assert entry.next_poll_at is not None
        assert entry.next_poll_at <= (
            entry.fetched_at
            + poll_policy.EXHAUSTED_INTERVAL_S * (1 + poll_policy.JITTER_FRAC)
        )

    def test_poll_never_scheduled_past_a_window_reset(self, temp_home, monkeypatch):
        from datetime import datetime, timezone

        from claude_swap.autoswitch import RESET_SLACK_S

        # The candidate's default interval is 300s, but its 5h window resets
        # in 90s — its stored 40% is obsolete at the rollover, so the next
        # poll must be clamped to reset + slack rather than waiting it out.
        h = self._harness(temp_home, monkeypatch, accounts=2)
        reset_ts = h.clock.now + 90.0
        reset_iso = (
            datetime.fromtimestamp(reset_ts, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        usage = {"1": _usage(50), "2": _usage(40, reset_iso)}
        counts: dict[str, int] = {}
        self._tick(h, counts, usage)
        entry = h.switcher._usage_store.entries(
            {"2": ("b@example.com", "")}
        )["2"]
        assert entry.next_poll_at == pytest.approx(reset_ts + RESET_SLACK_S)
        # Learned cadence untouched by the clamp.
        assert entry.poll_interval_s == poll_policy.CANDIDATE_DEFAULT_INTERVAL_S

    def test_movement_adapts_poll_interval(self, temp_home, monkeypatch):
        h = self._harness(temp_home, monkeypatch, accounts=2)
        usage = {"1": _usage(50), "2": _usage(10)}
        counts: dict[str, int] = {}

        def interval() -> float | None:
            return h.switcher._usage_store.entries(
                {"2": ("b@example.com", "")}
            )["2"].poll_interval_s

        self._tick(h, counts, usage)          # first data point → base interval
        assert interval() == poll_policy.CANDIDATE_DEFAULT_INTERVAL_S  # 300s
        h.clock.advance(180)
        self._tick(h, counts, usage)          # not due yet (300s interval)
        assert counts["2"] == 1
        h.clock.advance(120)
        self._tick(h, counts, usage)          # unmoved → backs off ×1.5
        assert counts["2"] == 2
        assert interval() == 450.0
        h.clock.advance(450)
        usage["2"] = _usage(20)               # moved 10 pts on another machine
        self._tick(h, counts, usage)
        assert counts["2"] == 3
        assert interval() == 225.0            # halved: polled closer while moving

    def test_idle_hold_skips_candidate_polling(self, temp_home, monkeypatch):
        h = self._harness(temp_home, monkeypatch)
        # Active token locally expired. The first tick now ATTEMPTS the
        # locked refresh (the fix's whole point); when it fails transiently
        # (network down), the row enters a failure backoff and subsequent
        # ticks surface the expired sentinel statically → idle-hold, with no
        # candidate slot spent.
        (h.temp_home / ".claude" / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-live", "refreshToken": "rt-live",
                "expiresAt": 1000,
            },
        }))
        # The slot backup must be expired too — a non-expired backup would be
        # restored without any POST (no failure, no backoff, no hold).
        h.seed(1, "a@example.com", expires_at=1000)
        usage = {"2": _usage(10), "3": _usage(20)}
        counts: dict[str, int] = {}
        with patch(
            "claude_swap.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(None, "network"),
        ):
            assert self._tick(h, counts, usage) is TickOutcome.NO_ACTION
            h.clock.advance(10)  # still inside the 30s failure backoff
            counts.clear()
            # Backoff established → the next tick polls nothing at all: the
            # active row is gated, the sentinel surfaces statically, and no
            # candidate slot is spent.
            assert self._tick(h, counts, usage) is TickOutcome.NO_ACTION
        assert counts == {}
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons[-1] == "active-idle"

    def test_poll_event_carries_fetch_errors(self, temp_home, monkeypatch):
        h = self._harness(temp_home, monkeypatch, accounts=2, unhealthy_ticks=3)
        counts: dict[str, int] = {}
        self._tick(
            h, counts, {"2": _usage(10)}, errors_by_num={"1": "http-429"}
        )
        poll = next(e for e in h.events if isinstance(e, PollEvent))
        assert poll.fetch_errors.get("1") == "http-429"
        assert "http-429" in poll.human()
        assert poll.to_json()["fetchErrors"] == {"1": "http-429"}

    def test_quarantined_candidate_never_consumes_the_poll_slot(
        self, temp_home, monkeypatch
    ):
        h = self._harness(temp_home, monkeypatch)
        h.engine._quarantine("2", "b@example.com", "invalid_grant")
        usage = {"1": _usage(50), "2": _usage(10), "3": _usage(20)}
        counts: dict[str, int] = {}
        for _ in range(3):
            self._tick(h, counts, usage)
            h.clock.advance(60)
        # The alternate slot always went to account 3; 2 is dead weight.
        assert "2" not in counts
        assert counts["3"] >= 1

    def test_expired_active_enters_idle_hold_even_during_backoff(
        self, temp_home, monkeypatch
    ):
        """Finding-2 regression: the owned+expired sentinel must not be hidden
        by the active row's failure backoff (e.g. a Retry-After window), or
        the engine would count unhealthy ticks toward a spurious failover."""
        from claude_swap.usage_store import FetchRecord

        h = self._harness(temp_home, monkeypatch)
        # Active token locally expired while an owner is present.
        (h.temp_home / ".claude" / ".credentials.json").write_text(json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-live", "refreshToken": "rt-live",
                "expiresAt": 1000,
            },
        }))
        # Active row sits in a long failure backoff → the fetch path (and its
        # own expired short-circuit) is unreachable this tick.
        h.switcher._usage_store.record(
            {"1": FetchRecord(error="http-429", retry_after_s=600.0)},
            {"1": ("a@example.com", "")},
        )
        counts: dict[str, int] = {}
        outcome = self._tick(h, counts, {"2": _usage(10), "3": _usage(20)})
        assert outcome is TickOutcome.NO_ACTION
        assert h.engine._unhealthy_ticks == 0
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["active-idle"]

    def test_consume_first_hold_never_escalates_below_threshold(
        self, temp_home, monkeypatch
    ):
        """Flat-traffic guard: a below-threshold consume-first tick that ends
        in a hold (no switch would fire) keeps the O(1) baseline — the
        phase-2 escalation is reserved for ticks that would actually switch.
        The fetch-set spy also catches an accidental all-candidates request
        that reserve() would have served from the store without HTTP."""
        h = self._harness(temp_home, monkeypatch, strategy="consume-first")
        # Active resets soonest -> every tick holds already-consuming-soonest.
        # five_hour 50 mirrors the baseline-cadence test's active plan.
        usage = {
            "1": _usage7(50, 20, _R_SOON),
            "2": _usage7(10, 10, _R_LATER),
            "3": _usage7(10, 10, _R_LATEST),
        }
        counts: dict[str, int] = {}
        fetch_sets: list[set] = []
        real_collect = h.switcher.usage_entries_by_account

        def spying_collect(*args, **kwargs):
            fetch_sets.append(set(kwargs.get("fetch") or ()))
            return real_collect(*args, **kwargs)

        with patch.object(
            h.switcher, "usage_entries_by_account", side_effect=spying_collect
        ):
            for _ in range(4):  # t0, t60, t120, t180
                outcome = self._tick(h, counts, usage)
                assert outcome is TickOutcome.NO_ACTION
                h.clock.advance(60)
        # (a) HTTP volume identical to the baseline cadence under `best`.
        assert counts == {"1": 2, "2": 1, "3": 1}
        # (b) no collection ever requested the all-candidates escalation set.
        assert {"1", "2", "3"} not in fetch_sets

    def test_consume_first_stale_target_holds_then_switches(
        self, temp_home, monkeypatch
    ):
        """Stale-after-escalation: when the phase-2 refetch cannot freshen the
        chosen target (Retry-After backoff), the freshness gate holds with
        stale-usage instead of switching on old data; once the backoff lapses
        a later tick freshens the target and the switch lands."""
        h = self._harness(temp_home, monkeypatch, strategy="consume-first")
        counts: dict[str, int] = {}
        # Populate the store while the active account resets soonest (holds).
        view_a = {
            "1": _usage7(50, 20, _R_SOON),
            "2": _usage7(10, 10, _R_LATER),
            "3": _usage7(10, 10, _R_LATEST),
        }
        self._tick(h, counts, view_a)          # t0: fetches 1, 2
        h.clock.advance(60)
        self._tick(h, counts, view_a)          # t60: fetches 3
        assert counts == {"1": 1, "2": 1, "3": 1}
        # #2 enters a Retry-After backoff; its stored entry ages past the
        # serve TTL (180s) while staying inside decision trust (300s).
        h.switcher._usage_store.record(
            {"2": FetchRecord(error="http-429", retry_after_s=600.0)},
            {"2": ("b@example.com", "")},
        )
        h.clock.advance(181)                   # t241
        h.events.clear()
        # The active refetch now reports the LATEST reset, so stored #2
        # (age 241: decision-trusted, no longer fresh) is the provisional
        # pick — but phase 2 cannot freshen it through the backoff.
        view_b = {
            "1": _usage7(50, 20, _R_LATEST),
            "2": _usage7(10, 10, _R_LATER),
            "3": _usage7(10, 10, _R_LATEST),
        }
        outcome = self._tick(h, counts, view_b)
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert "stale-usage" in reasons
        assert counts["2"] == 1  # the backoff kept every refetch off #2
        # Backoff lapses -> a later tick freshens #2 and the switch lands.
        h.events.clear()
        h.clock.advance(700)
        outcome = self._tick(h, counts, view_b)
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        sw = next(e for e in h.events if isinstance(e, SwitchEvent))
        assert sw.trigger == "consume-first"


class TestApiKeyAccounts:
    def _mark_api_key(self, harness, num: int) -> None:
        data = harness.switcher._get_sequence_data()
        data["accounts"][str(num)]["kind"] = "api_key"
        harness.switcher._write_json(harness.switcher.sequence_file, data)

    def test_api_key_candidate_excluded_by_default(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "key@token.local")
        h.make_live("a@example.com", 1)
        self._mark_api_key(h, 2)
        outcome = h.tick_with_usage({"1": _usage(95), "2": "api key"})
        assert outcome is TickOutcome.BLOCKED
        assert h.active_number() == 1

    def test_api_key_is_last_resort_when_included(self, temp_home):
        h = EngineHarness(temp_home, include_api_key_accounts=True)
        h.seed(1, "a@example.com")
        h.seed(2, "key@token.local")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)
        self._mark_api_key(h, 2)
        # A qualifying OAuth candidate wins over the API key...
        outcome = h.tick_with_usage({
            "1": _usage(95), "2": "api key", "3": _usage(10),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_api_key_used_when_oauth_exhausted(self, temp_home):
        h = EngineHarness(temp_home, include_api_key_accounts=True)
        h.seed(1, "a@example.com")
        h.seed(2, "key@token.local")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)
        self._mark_api_key(h, 2)
        outcome = h.tick_with_usage({
            "1": _usage(100), "2": "api key", "3": _usage(100),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_active_api_key_idles_engine(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "key@token.local")
        h.seed(2, "b@example.com")
        h.make_live("key@token.local", 1)
        self._mark_api_key(h, 1)
        outcome = h.tick_with_usage({"1": "api key", "2": _usage(10)})
        assert outcome is TickOutcome.NO_ACTION
        assert [e.reason for e in h.events if isinstance(e, NoSwitchEvent)] == [
            "active-api-key"
        ]


class TestFreshening:
    def test_near_expiry_target_is_refreshed_and_persisted(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com", expires_at=int(h.clock() * 1000) + 60_000)
        h.make_live("a@example.com", 1)

        rotated = json.dumps({
            "claudeAiOauth": {
                "accessToken": "sk-2-new",
                "refreshToken": "rt-2-new",
                "expiresAt": int(h.clock() * 1000) + 3_600_000,
            }
        })
        live_creds_path = temp_home / ".claude" / ".credentials.json"
        live_before = live_creds_path.read_text()
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(rotated, None),
        ) as mock_refresh:
            outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})

        assert outcome is TickOutcome.SWITCHED
        mock_refresh.assert_called_once()
        # Freshening itself never touched the active store (the switch did,
        # afterwards, via _perform_switch): the rotated token must have gone
        # through the backup, and now be live.
        assert "sk-2-new" in live_creds_path.read_text()
        assert live_creds_path.read_text() != live_before

    def test_fresh_target_is_not_refreshed(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com", expires_at=int(h.clock() * 1000) + 3_600_000)
        h.make_live("a@example.com", 1)
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials"
        ) as mock_refresh:
            outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})
        assert outcome is TickOutcome.SWITCHED
        mock_refresh.assert_not_called()

    def test_invalid_grant_quarantines_and_tries_next(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com", expires_at=1)  # long expired
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(None, "invalid_grant"),
        ):
            outcome = h.tick_with_usage({
                "1": _usage(95), "2": _usage(10), "3": _usage(20),
            })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3  # next candidate after 2 was quarantined
        q = next(e for e in h.events if isinstance(e, QuarantineEvent))
        assert (q.number, q.reason) == ("2", "invalid_grant")
        assert "2" in h.state()["quarantine"]

    def test_transient_failure_skips_without_quarantine(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com", expires_at=1)
        h.make_live("a@example.com", 1)
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(None, "transient"),
        ):
            outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})
        assert outcome is TickOutcome.ERROR
        assert h.active_number() == 1
        assert not h.state().get("quarantine")
        assert any(isinstance(e, ErrorEvent) for e in h.events)

    def test_live_session_target_is_skipped_even_with_fresh_token(self, temp_home):
        # Auto never activates an account that has a live `cswap run` session:
        # dual refresh-token ownership with nobody reading the warning.
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com", expires_at=int(h.clock() * 1000) + 3_600_000)
        h.make_live("a@example.com", 1)
        with patch.object(
            h.switcher, "live_session_pids_for", return_value=[4242]
        ), patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials"
        ) as mock_refresh:
            outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})
        assert outcome is TickOutcome.BLOCKED
        mock_refresh.assert_not_called()
        assert h.active_number() == 1

    def test_live_session_near_expiry_is_skipped(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com", expires_at=1)  # long expired
        h.make_live("a@example.com", 1)
        with patch.object(
            h.switcher, "live_session_pids_for", return_value=[4242]
        ), patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials"
        ) as mock_refresh:
            outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})
        assert outcome is TickOutcome.BLOCKED
        mock_refresh.assert_not_called()
        assert h.active_number() == 1


class TestQuarantineLifecycle:
    def test_quarantine_persists_across_engine_instances(self, harness):
        harness.engine._quarantine("2", "b@example.com", "invalid_grant")
        harness.events.clear()
        fresh_engine = harness._make_engine()
        usage = {"1": _usage(95), "2": _usage(0), "3": _usage(50)}
        with patch.object(
            harness.switcher,
            "usage_entries_by_account",
            return_value={
                num: _entry_for(value, harness.clock.now)
                for num, value in usage.items()
            },
        ):
            outcome = fresh_engine.tick()
        # 2 has the most headroom but is quarantined → 3 wins.
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_replaced_credentials_lift_quarantine(self, harness):
        harness.engine._quarantine("2", "b@example.com", "invalid_grant")
        # User re-logged in and re-captured the slot: new refresh token.
        harness.switcher._write_account_credentials(
            "2",
            "b@example.com",
            json.dumps({
                "claudeAiOauth": {"accessToken": "sk-2b", "refreshToken": "rt-2b"},
            }),
        )
        harness.events.clear()
        outcome = harness.tick_with_usage({
            "1": _usage(95), "2": _usage(0), "3": _usage(50),
        })
        assert any(isinstance(e, UnquarantineEvent) for e in harness.events)
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2
        assert "2" not in (harness.state().get("quarantine") or {})

    def test_state_lock_preserves_concurrent_writes(self, harness):
        # Simulate another engine writing between our read and our write: the
        # RMW under the state lock must preserve its quarantine entry.
        harness.engine._mutate_state(
            lambda s: s.setdefault("quarantine", {}).update(
                {"3": {"email": "c@example.com", "reason": "invalid_grant",
                       "at": "x", "refreshTokenFingerprint": None}}
            )
        )
        harness.engine._mutate_state(lambda s: s.update(lastSwitchAt=123.0))
        state = harness.state()
        assert state["lastSwitchAt"] == 123.0
        assert "3" in state["quarantine"]


class TestDryRunAndNoOp:
    def test_dry_run_mutates_nothing(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)
        h.engine = h._make_engine(dry_run=True)
        live_before = (temp_home / ".claude" / ".credentials.json").read_text()

        outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})

        assert outcome is TickOutcome.SWITCHED
        switch = next(e for e in h.events if isinstance(e, SwitchEvent))
        assert switch.dry_run is True
        assert h.active_number() == 1  # unchanged
        assert (temp_home / ".claude" / ".credentials.json").read_text() == live_before
        assert h.state() == {}  # no lastSwitchAt recorded

    def test_dry_run_never_freshens_or_quarantines(self, temp_home):
        # A near-expiry target would normally be refreshed (a real token
        # rotation) and a dead one quarantined (a state write). Dry-run must
        # stop at the decision: no network, no writes of any kind.
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com", expires_at=1)  # long expired
        h.make_live("a@example.com", 1)
        h.engine = h._make_engine(dry_run=True)
        backup_before = h.switcher.read_account_credentials("2", "b@example.com")

        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials"
        ) as mock_refresh:
            outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})

        assert outcome is TickOutcome.SWITCHED  # reported the would-switch
        mock_refresh.assert_not_called()
        assert h.switcher.read_account_credentials("2", "b@example.com") == backup_before
        assert h.state() == {}  # no quarantine, no lastSwitchAt

    def test_dry_run_does_not_release_quarantines(self, temp_home):
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)
        h.engine._quarantine("2", "b@example.com", "invalid_grant")
        # Replace the credential — a real tick would lift the quarantine.
        h.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {"accessToken": "n", "refreshToken": "n"}}),
        )
        h.events.clear()
        h.engine = h._make_engine(dry_run=True)
        state_before = h.state()

        outcome = h.tick_with_usage({"1": _usage(95), "2": _usage(10)})

        assert not any(isinstance(e, UnquarantineEvent) for e in h.events)
        assert h.state() == state_before  # state file untouched
        # And the still-recorded quarantine keeps 2 out of the dry-run plan.
        assert outcome is TickOutcome.BLOCKED

    def test_already_active_result_is_noop(self, harness):
        with patch.object(
            harness.switcher,
            "switch_to",
            return_value={"switched": False, "reason": "already-active"},
        ):
            outcome = harness.tick_with_usage({
                "1": _usage(95), "2": _usage(10), "3": _usage(50),
            })
        assert outcome is TickOutcome.NO_ACTION
        assert "lastSwitchAt" not in harness.state()


class TestEventsShape:
    def test_every_event_has_envelope(self, harness):
        harness.tick_with_usage({"1": _usage(95), "2": _usage(10), "3": _usage(50)})
        assert harness.events
        for event in harness.events:
            payload = event.to_json()
            assert payload["schemaVersion"] == 1
            assert payload["event"] == event.kind
            assert payload["ts"].endswith("Z")

    def test_switch_event_refs_match_account_ref_shape(self, harness):
        harness.tick_with_usage({"1": _usage(95), "2": _usage(10), "3": _usage(50)})
        switch = next(e for e in harness.events if isinstance(e, SwitchEvent))
        payload = switch.to_json()
        assert payload["from"] == {"number": 1, "email": "a@example.com"}
        assert payload["to"] == {"number": 2, "email": "b@example.com"}

    def test_poll_event_human_line(self, harness):
        harness.tick_with_usage({"1": _usage(42), "2": _usage(10), "3": None})
        poll = next(e for e in harness.events if isinstance(e, PollEvent))
        line = poll.human()
        assert "Account-1" in line and "42% used" in line
        # Others show per-window pcts, not just the ambiguous binding pct.
        assert "#2: 5h 10% · 7d 0%" in line
        assert "#3: ?" in line

    def test_poll_event_windows_match_the_decision_set(self, temp_home):
        # Scoped windows appear only when configured: rendering an ignored
        # Fable 100% next to a switch onto that account would read as a bug.
        usage = {
            "1": _usage(42),
            "2": {
                "five_hour": {"pct": 3.0},
                "seven_day": {"pct": 89.0},
                "scoped": [{"name": "Fable", "pct": 21.0}],
            },
        }

        def build(**kw):
            h = EngineHarness(temp_home, **kw)
            h.seed(1, "a@example.com")
            h.seed(2, "b@example.com")
            h.make_live("a@example.com", 1)
            return h

        plain = build()
        plain.tick_with_usage(usage)
        poll = next(e for e in plain.events if isinstance(e, PollEvent))
        assert "#2: 5h 3% · 7d 89%" in poll.human()
        assert "Fable" not in poll.human()
        assert poll.to_json()["windowsPct"]["2"] == {"5h": 3.0, "7d": 89.0}

        modeled = build(model="Fable")
        modeled.tick_with_usage(usage)
        poll = next(e for e in modeled.events if isinstance(e, PollEvent))
        assert "#2: 5h 3% · 7d 89% · Fable 21%" in poll.human()
        assert poll.to_json()["windowsPct"]["2"] == {
            "5h": 3.0, "7d": 89.0, "Fable": 21.0,
        }


class TestRunLoop:
    def test_loop_ticks_until_stopped(self, harness):
        ticks = []

        def fake_tick():
            ticks.append(1)
            if len(ticks) >= 2:
                harness.engine.stop()
            return TickOutcome.NO_ACTION

        with patch.object(harness.engine, "tick", side_effect=fake_tick), \
             patch.object(harness.engine._wake, "wait", return_value=None):
            assert harness.engine.run_loop() == 0
        assert len(ticks) == 2

    def test_loop_survives_raising_tick(self, harness):
        calls = []

        def raising_inner():
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            harness.engine.stop()
            return TickOutcome.NO_ACTION

        with patch.object(
            harness.engine, "_tick_inner", side_effect=raising_inner
        ), patch.object(harness.engine._wake, "wait", return_value=None):
            harness.engine.run_loop()
        assert len(calls) == 2
        assert any(isinstance(e, ErrorEvent) for e in harness.events)

    def test_stop_before_start_is_not_lost(self, harness):
        # A stop() issued before the worker thread enters run_loop must not
        # be cleared away: the loop exits without a single tick.
        harness.engine.stop()
        with patch.object(harness.engine, "tick") as tick:
            assert harness.engine.run_loop() == 0
        tick.assert_not_called()

    def test_wake_during_tick_cuts_the_following_sleep_short(self, harness):
        # No wait patching on purpose: if the clear-at-top ordering were
        # wrong (wake cleared after the wait), the wake fired during tick 1
        # would be lost and the loop would block on the real 60s sleep —
        # caught by the join timeout instead of hanging the suite.
        ticks: list[int] = []

        def fake_tick():
            ticks.append(1)
            if len(ticks) == 1:
                harness.engine.wake()  # e.g. apply_threshold landed mid-tick
            else:
                harness.engine.stop()
            return TickOutcome.NO_ACTION

        with patch.object(harness.engine, "tick", side_effect=fake_tick):
            worker = threading.Thread(target=harness.engine.run_loop)
            worker.start()
            worker.join(timeout=10)
            finished = not worker.is_alive()
            harness.engine.stop()  # unblock a failing loop before asserting
            worker.join(timeout=5)
        assert finished
        assert len(ticks) == 2

    def test_blocked_with_reset_rechecks_at_exhausted_cadence(self, harness):
        harness.engine._sleep_until_ts = harness.clock() + 1800
        delay = harness.engine._next_delay(TickOutcome.BLOCKED)
        assert delay == poll_policy.EXHAUSTED_INTERVAL_S

    def test_blocked_exhausted_without_reset_uses_fallback(self, harness):
        harness.engine._sleep_until_ts = None
        harness.engine._blocked_wait_long = True
        assert harness.engine._next_delay(TickOutcome.BLOCKED) == 300.0

    def test_blocked_on_resolvable_condition_keeps_normal_cadence(self, harness):
        harness.engine._sleep_until_ts = None
        harness.engine._blocked_wait_long = False
        delay = harness.engine._next_delay(TickOutcome.BLOCKED)
        assert 0.9 * 60 <= delay <= 1.1 * 60

    def test_normal_delay_is_jittered_interval(self, harness):
        delay = harness.engine._next_delay(TickOutcome.NO_ACTION)
        assert 0.9 * 60 <= delay <= 1.1 * 60

    def test_sleep_cap(self, harness):
        harness.engine._sleep_until_ts = harness.clock() + 50 * 3600
        assert (
            harness.engine._next_delay(TickOutcome.BLOCKED)
            == poll_policy.EXHAUSTED_INTERVAL_S
        )


class TestLoopObeysThePollPlan:
    """The loop must not oversleep the plan the planner wrote.

    When the active account burns near the threshold the planner tightens its
    row to URGENT_INTERVAL_S so the crossing is caught quickly. The loop used
    to sleep ``interval_seconds`` regardless, so on any machine configured
    slower than the plan (360s here, the default) that plan could not be
    honoured: measured on the linux box mid-episode, the active row asked to
    be polled 112s ago while the engine still had minutes of sleep left, and
    the account sat over the threshold until the engine was restarted by hand.
    """

    def _plan(self, harness, *, due_in: float) -> None:
        num = harness.engine.switcher.current_account_number()
        real = harness.engine.switcher.usage_entries_by_account

        def patched(fetch=frozenset(), **kw):
            entries = dict(real(fetch=fetch, **kw))
            entries[num] = replace(
                entries[num], next_poll_at=harness.clock() + due_in
            )
            return entries

        harness.engine.switcher.usage_entries_by_account = patched

    def test_sleep_is_cut_to_the_rows_next_poll(self, harness):
        harness.engine.settings = replace(
            harness.engine.settings, interval_seconds=360.0
        )
        self._plan(harness, due_in=60.0)
        # Pre-fix this returned ~360s and the 60s plan silently ran late.
        assert harness.engine._next_delay(TickOutcome.NO_ACTION) == 60.0

    def test_never_sleeps_below_the_planners_own_floor(self, harness):
        """A row already overdue must not spin: the floor is the rate budget."""
        harness.engine.settings = replace(
            harness.engine.settings, interval_seconds=360.0
        )
        self._plan(harness, due_in=-500.0)
        assert (
            harness.engine._next_delay(TickOutcome.NO_ACTION)
            == poll_policy.URGENT_INTERVAL_S
        )

    def test_a_relaxed_plan_never_lengthens_the_sleep(self, harness):
        """Only ever shortens — a distant plan must not stretch the cadence
        past what the user configured."""
        harness.engine.settings = replace(
            harness.engine.settings, interval_seconds=60.0
        )
        self._plan(harness, due_in=3600.0)
        assert harness.engine._next_delay(TickOutcome.NO_ACTION) <= 1.1 * 60

    def test_a_store_failure_leaves_the_cadence_alone(self, harness):
        def boom(*a, **k):
            raise RuntimeError("store unreadable")

        harness.engine.switcher.usage_entries_by_account = boom
        delay = harness.engine._next_delay(TickOutcome.NO_ACTION)
        assert 0.9 * 60 <= delay <= 1.1 * 60


class TestSessionThreshold:
    """apply_threshold(): the TUI's session-only, mid-run override."""

    def test_apply_threshold_retargets_trigger_and_poll_pin(self, harness):
        harness.engine.apply_threshold(72.0)
        assert harness.engine.settings.threshold == 72.0
        # Poll-cadence planning follows the new value immediately.
        assert harness.switcher._poll_inputs_override == (72.0, ())
        # And the very next tick decides with it: 80% ≥ 72 switches, where
        # the constructed 90 would not have.
        outcome = harness.tick_with_usage({
            "1": _usage(80), "2": _usage(10), "3": _usage(10),
        })
        assert outcome is TickOutcome.SWITCHED

    def test_clear_poll_policy_inputs_unpins(self, harness):
        harness.engine.apply_threshold(72.0)
        harness.switcher.clear_poll_policy_inputs()
        assert harness.switcher._poll_inputs_override is None

    def _collect_fetch_sets(self, harness, threshold: float) -> list:
        entries = {
            n: _entry_for(_usage(80.0 if n == "1" else 10.0), harness.clock.now)
            for n in ("1", "2", "3")
        }
        with patch.object(
            harness.switcher, "usage_entries_by_account", return_value=entries
        ) as collect:
            harness.engine._collect_scheduled_usage("1", threshold=threshold)
        return [c.kwargs.get("fetch") for c in collect.call_args_list]

    def test_collect_escalates_on_the_tick_snapshot_threshold(self, harness):
        # Escalation must key on the threshold captured by the tick, not a
        # re-read of self.settings (engine settings stay at 90 throughout).
        # Active at 80%: within ESCALATION_MARGIN_PCT of 90 → full refresh...
        assert {"1", "2", "3"} in self._collect_fetch_sets(harness, 90.0)
        # ...but not of 99.9 → baseline fetching only.
        assert {"1", "2", "3"} not in self._collect_fetch_sets(harness, 99.9)


class TestPctLabel:
    def test_whole_numbers_drop_the_decimal(self):
        assert pct_label(90.0) == "90"

    def test_fractional_threshold_keeps_one_decimal(self):
        # .0f would render the valid maximum 99.9 as a lying "100".
        assert pct_label(99.9) == "99.9"

    def test_configured_precision_is_preserved(self):
        # settings.json accepts arbitrary floats; display must not round.
        assert pct_label(85.55) == "85.55"
        assert pct_label(85.555555) == "85.555555"

    def test_float_noise_is_absorbed(self):
        assert pct_label(100.0 - 37.4) == "62.6"
        assert pct_label(99.85000000000001) == "99.85"

    def test_poll_event_shows_fractional_threshold(self):
        poll = PollEvent(
            active={"number": 1, "email": "a@example.com"},
            headroom={"1": 40.0},
            threshold=99.9,
        )
        assert "switch at 99.9%" in poll.human()

    def test_below_threshold_detail_shows_fractional_threshold(self, temp_home):
        h = EngineHarness(temp_home, threshold=99.9)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)
        h.tick_with_usage({"1": _usage(50), "2": _usage(10)})
        details = [
            e.detail for e in h.events if isinstance(e, NoSwitchEvent)
        ]
        assert details == ["50% < 99.9%"]

    def test_below_threshold_detail_never_shows_impossible_comparison(
        self, temp_home
    ):
        # utilization 99.85 with threshold 99.9: .0f on the left side used
        # to render the logically impossible "100% < 99.9%".
        h = EngineHarness(temp_home, threshold=99.9)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)
        h.tick_with_usage({"1": _usage(99.85), "2": _usage(10)})
        details = [
            e.detail for e in h.events if isinstance(e, NoSwitchEvent)
        ]
        assert details == ["99.85% < 99.9%"]


class TestTokenIdentity:
    """The token endpoint's free identity data: uuid backfill and the
    identity-conflict detector (the zero-request check that catches a
    poisoned slot the moment auto freshens it)."""

    def test_uuid_backfill_from_token_account_on_freshen(self, harness):
        data = harness.switcher._get_sequence_data()
        data["accounts"]["2"]["uuid"] = ""
        harness.switcher._write_json(harness.switcher.sequence_file, data)
        # Slot 2 near expiry → freshen path runs.
        harness.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2", "refreshToken": "rt-2", "expiresAt": 0,
            }}),
        )
        fresh = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2f", "refreshToken": "rt-2f",
            "expiresAt": 99_999_999_999_000,
        }})
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(
                fresh, None,
                {"uuid": "uuid-2-real", "email": "b@example.com",
                 "organizationUuid": ""},
            ),
        ):
            status = harness.engine._freshen_target("2", "b@example.com")
        assert status == "ok"
        assert harness.switcher._get_sequence_data()["accounts"]["2"]["uuid"] == (
            "uuid-2-real"
        )

    def test_conflicting_token_identity_returns_identity_conflict(self, harness):
        """A slot whose credential authenticates as a different account is not
        a viable target — but the rotated generation is still persisted (the
        grant consumed its predecessor)."""
        harness.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2", "refreshToken": "rt-2", "expiresAt": 0,
            }}),
        )
        fresh = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2f", "refreshToken": "rt-2f",
            "expiresAt": 99_999_999_999_000,
        }})
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(
                fresh, None,
                {"uuid": "uuid-somebody-else", "email": "z@example.com",
                 "organizationUuid": ""},
            ),
        ):
            status = harness.engine._freshen_target("2", "b@example.com")
        assert status == "identity-conflict"
        # The consumed generation's successor was persisted regardless.
        assert harness.switcher.read_account_credentials(
            "2", "b@example.com"
        ) == fresh

    def test_identity_conflict_quarantines_instead_of_activating(self, harness):
        """Tick path: the conflicted slot is quarantined (wrong-account switch
        prevented); rotation falls through to the next candidate."""
        harness.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2", "refreshToken": "rt-2", "expiresAt": 0,
            }}),
        )
        fresh = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2f", "refreshToken": "rt-2f",
            "expiresAt": 99_999_999_999_000,
        }})

        def refresh(creds):
            data = json.loads(creds)["claudeAiOauth"]
            if data["refreshToken"] == "rt-2":
                return oauth.RefreshOutcome(
                    fresh, None,
                    {"uuid": "uuid-somebody-else", "email": "z@example.com",
                     "organizationUuid": ""},
                )
            return oauth.RefreshOutcome(creds, None)

        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            side_effect=refresh,
        ):
            outcome = harness.tick_with_usage({
                "1": _usage(95), "2": _usage(10), "3": _usage(80),
            })
        # Account 2 had the most headroom but is conflicted → quarantined,
        # and the switch landed elsewhere.
        assert "account-quarantined" in harness.kinds()
        q = harness.state().get("quarantine", {})
        assert q.get("2", {}).get("reason") == "identity-conflict"
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_dead_slot_quarantined_even_with_safety_copy_present(self, harness):
        """No automatic promotion (fail-open rework of the issue #117 guard):
        a dead slot is quarantined outright; safety copies are forensic
        material, and recovery is the documented /login + cswap add."""
        harness.switcher._store._write_unclaimed_credential(
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2-successor",
                "refreshToken": "rt-2-successor",
                "expiresAt": 99_999_999_999_000,
            }}),
            {"resolvedIdentity": {
                "uuid": "uuid-2", "email": "b@example.com",
                "organizationUuid": "",
            }},
        )
        harness.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2-dead", "refreshToken": "rt-2-dead",
                "expiresAt": 0,
            }}),
        )

        def refresh(creds):
            data = json.loads(creds)["claudeAiOauth"]
            if data["refreshToken"] == "rt-2-dead":
                return oauth.RefreshOutcome(None, "invalid_grant")
            return oauth.RefreshOutcome(creds, None)

        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            side_effect=refresh,
        ):
            outcome = harness.tick_with_usage({
                "1": _usage(95), "2": _usage(10), "3": _usage(80),
            })
        q = harness.state().get("quarantine", {})
        assert q.get("2", {}).get("reason") == "invalid_grant"
        # The safety copy was not consumed, and the switch landed elsewhere.
        assert len(harness.switcher.list_unclaimed_credentials()) == 1
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_same_uuid_different_org_is_identity_conflict(self, harness):
        """Organization is part of account identity everywhere else in the
        codebase: the same account uuid under a different org is a conflict
        (org compared only when both sides record one)."""
        data = harness.switcher._get_sequence_data()
        data["accounts"]["2"]["organizationUuid"] = "org-2"
        harness.switcher._write_json(harness.switcher.sequence_file, data)
        harness.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2", "refreshToken": "rt-2", "expiresAt": 0,
            }}),
        )
        fresh = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2f", "refreshToken": "rt-2f",
            "expiresAt": 99_999_999_999_000,
        }})
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(
                fresh, None,
                {"uuid": "uuid-2", "email": "b@example.com",
                 "organizationUuid": "org-other"},
            ),
        ):
            status = harness.engine._freshen_target("2", "b@example.com")
        assert status == "identity-conflict"

    def test_malformed_token_identity_never_breaks_freshen(self, harness):
        """A schema change feeding a non-string uuid must be ignored, not
        raise — by this point the refreshed credential is already persisted,
        and a crash here would skip the persist bookkeeping and error the
        tick."""
        harness.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2", "refreshToken": "rt-2", "expiresAt": 0,
            }}),
        )
        fresh = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2f", "refreshToken": "rt-2f",
            "expiresAt": 99_999_999_999_000,
        }})
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(
                fresh, None, {"uuid": 12345, "email": ["weird"]},
            ),
        ):
            status = harness.engine._freshen_target("2", "b@example.com")
        assert status == "ok"
        assert harness.switcher.read_account_credentials(
            "2", "b@example.com"
        ) == fresh

    def test_blank_uuid_slot_with_org_conflict_quarantines_not_backfills(
        self, harness,
    ):
        """Org conflict must be checked before the blank-uuid backfill: a
        wrong-org credential is evidence the slot holds the wrong account,
        and backfilling its uuid would stick a foreign identity onto the
        slot (backfill never rewrites a non-empty uuid). Blank-uuid slots
        with a recorded org are what accounts added by older versions look
        like."""
        data = harness.switcher._get_sequence_data()
        data["accounts"]["2"]["uuid"] = ""
        data["accounts"]["2"]["organizationUuid"] = "org-A"
        harness.switcher._write_json(harness.switcher.sequence_file, data)
        harness.switcher._write_account_credentials(
            "2", "b@example.com",
            json.dumps({"claudeAiOauth": {
                "accessToken": "sk-2", "refreshToken": "rt-2", "expiresAt": 0,
            }}),
        )
        fresh = json.dumps({"claudeAiOauth": {
            "accessToken": "sk-2f", "refreshToken": "rt-2f",
            "expiresAt": 99_999_999_999_000,
        }})
        with patch(
            "claude_swap.autoswitch.oauth.try_refresh_oauth_credentials",
            return_value=oauth.RefreshOutcome(
                fresh, None,
                {"uuid": "uuid-real", "email": "z@example.com",
                 "organizationUuid": "org-B"},
            ),
        ):
            status = harness.engine._freshen_target("2", "b@example.com")
        assert status == "identity-conflict"
        # The foreign uuid was NOT backfilled onto the slot.
        assert harness.switcher._get_sequence_data()["accounts"]["2"]["uuid"] == ""
        # The successor generation was still persisted (grant consumed it).
        assert harness.switcher.read_account_credentials(
            "2", "b@example.com"
        ) == fresh


def _model_usage(five_h: float, fable: float) -> dict:
    """Usage with a low 5h/7d but a per-model (Fable) weekly window."""
    return {
        "five_hour": {"pct": five_h},
        "seven_day": {"pct": 0.0},
        "scoped": [{"name": "Fable", "pct": fable}],
    }


class TestModelAwareSwitch:
    """`autoswitch.model` folds a per-model weekly limit into the decision."""

    def _seed(self, temp_home: Path, **kw) -> EngineHarness:
        h = EngineHarness(temp_home, **kw)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)
        return h

    def test_model_maxed_switches_despite_session_headroom(self, temp_home):
        # Active #1: 5h only 5% used, but Fable is maxed → must leave.
        h = self._seed(temp_home, model="Fable")
        outcome = h.tick_with_usage({
            "1": _model_usage(5, 100),
            "2": _model_usage(5, 30),
            "3": _model_usage(5, 60),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2  # most Fable headroom
        switch = next(e for e in h.events if isinstance(e, SwitchEvent))
        assert switch.to_ref == {"number": 2, "email": "b@example.com"}

    def test_without_model_setting_the_same_usage_holds(self, temp_home):
        # Default engine ignores scoped windows → #1 reads 5% used, no switch.
        h = self._seed(temp_home)
        outcome = h.tick_with_usage({
            "1": _model_usage(5, 100),
            "2": _model_usage(5, 30),
            "3": _model_usage(5, 60),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["below-threshold"]

    def test_model_headroom_still_gated_by_session_window(self, temp_home):
        # Fable has room on every account, but #1's 5h is maxed → still leaves.
        h = self._seed(temp_home, model="Fable")
        outcome = h.tick_with_usage({
            "1": _model_usage(100, 40),
            "2": _model_usage(10, 40),
            "3": _model_usage(20, 40),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2  # lowest binding (max of 5h, Fable)

    def test_comma_separated_models_switch_on_any(self, temp_home):
        # Configured for "Fable,Opus"; active #1 is fine on Fable but maxed on
        # Opus → must leave. Candidate scoped windows carry both models.
        h = self._seed(temp_home, model="Fable,Opus")

        def usage(five_h, fable, opus):
            return {
                "five_hour": {"pct": five_h},
                "seven_day": {"pct": 0.0},
                "scoped": [
                    {"name": "Fable", "pct": fable},
                    {"name": "Opus", "pct": opus},
                ],
            }

        outcome = h.tick_with_usage({
            "1": usage(5, 20, 100),   # Opus maxed
            "2": usage(5, 20, 30),    # most headroom
            "3": usage(5, 20, 70),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_all_sentinel_binds_every_scoped_window(self, temp_home):
        # "all" needs no names: each account's own scoped windows bind,
        # whatever they're called.
        h = self._seed(temp_home, model="all")
        outcome = h.tick_with_usage({
            "1": {"five_hour": {"pct": 5.0}, "seven_day": {"pct": 0.0},
                  "scoped": [{"name": "Sonnet", "pct": 100.0}]},
            "2": {"five_hour": {"pct": 5.0}, "seven_day": {"pct": 0.0},
                  "scoped": [{"name": "Sonnet", "pct": 20.0}]},
            "3": {"five_hour": {"pct": 5.0}, "seven_day": {"pct": 0.0},
                  "scoped": [{"name": "Opus", "pct": 60.0}]},
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_dual_exhausted_candidate_recovers_at_its_later_reset(self, temp_home):
        # #2 is blocked on both its 5h (resets 12:00) and Fable (15:00): it's
        # only usable again at the LATER one. #3 recovers later still (20:00),
        # so the all-exhausted wake is #2's Fable reset — which the old
        # earliest-of-any-window scan (12:00) would have jumped early for.
        h = self._seed(temp_home, model="Fable")
        fable_reset = "2026-07-05T15:00:00Z"
        outcome = h.tick_with_usage({
            "1": _model_usage(95, 10),
            "2": {
                "five_hour": {"pct": 100.0, "resets_at": "2026-07-05T12:00:00Z"},
                "seven_day": {"pct": 0.0},
                "scoped": [
                    {"name": "Fable", "pct": 100.0, "resets_at": fable_reset},
                ],
            },
            "3": {
                "five_hour": {"pct": 100.0, "resets_at": "2026-07-05T20:00:00Z"},
                "seven_day": {"pct": 0.0},
            },
        })
        assert outcome is TickOutcome.BLOCKED
        exhausted = next(e for e in h.events if isinstance(e, AllExhaustedEvent))
        assert exhausted.earliest_reset_at == fable_reset

    def test_unknown_recovery_falls_back_instead_of_oversleeping(self, temp_home):
        # #2 is exhausted with NO reset timestamp — it could recover any
        # moment. Sleeping toward #3's known 20:00 reset would suppress
        # checks for hours, so the wake time must be unprovable (bounded
        # blocked-cadence fallback instead of a reset sleep).
        h = self._seed(temp_home, model="Fable")
        outcome = h.tick_with_usage({
            "1": _model_usage(95, 10),
            "2": {
                "five_hour": {"pct": 0.0},
                "seven_day": {"pct": 0.0},
                "scoped": [{"name": "Fable", "pct": 100.0}],  # no resets_at
            },
            "3": {
                "five_hour": {"pct": 100.0, "resets_at": "2026-07-05T20:00:00Z"},
                "seven_day": {"pct": 0.0},
            },
        })
        assert outcome is TickOutcome.BLOCKED
        exhausted = next(e for e in h.events if isinstance(e, AllExhaustedEvent))
        assert exhausted.earliest_reset_at is None
        assert h.engine._sleep_until_ts is None
        assert h.engine._next_delay(outcome) == NO_RESET_FALLBACK_S

    def test_scoped_only_exhaustion_drives_the_wake_time(self, temp_home):
        # Candidates blocked ONLY by Fable: the wake must come from the scoped
        # reset — the 5h/7d-only scan would find no ≥100% window at all.
        h = self._seed(temp_home, model="Fable")
        fable_reset = "2026-07-06T09:00:00Z"
        blocked = {
            "five_hour": {"pct": 3.0, "resets_at": "2026-07-05T12:00:00Z"},
            "seven_day": {"pct": 0.0},
            "scoped": [{"name": "Fable", "pct": 100.0, "resets_at": fable_reset}],
        }
        outcome = h.tick_with_usage({
            "1": _model_usage(95, 10), "2": blocked, "3": blocked,
        })
        assert outcome is TickOutcome.BLOCKED
        exhausted = next(e for e in h.events if isinstance(e, AllExhaustedEvent))
        assert exhausted.earliest_reset_at == fable_reset

    def test_scoped_binding_window_keeps_active_cadence_tight(self, temp_home):
        # Fable moving at 88% is inside the escalation band: with the model
        # configured the urgent cadence engages, while the 5%-used 5h window
        # alone would just decay the interval.
        kwargs = dict(
            prev_interval_s=poll_policy.MIN_INTERVAL_S,
            prev_usage=_model_usage(5, 84),
            new_usage=_model_usage(5, 88),
            is_active=True,
            threshold=90.0,
            recent_429=False,
            now=1000.0,
            rng=lambda: 0.5,
        )
        _, scoped = poll_policy.plan_after_fetch(models=("Fable",), **kwargs)
        assert scoped == poll_policy.URGENT_INTERVAL_S
        _, unscoped = poll_policy.plan_after_fetch(models=(), **kwargs)
        assert unscoped > poll_policy.MIN_INTERVAL_S  # plain decay

    def test_unmatched_model_name_warns_once(self, temp_home):
        h = self._seed(temp_home, model="Fabel")  # deliberate typo
        usage = {
            "1": _model_usage(5, 10),
            "2": _model_usage(5, 10),
            "3": _model_usage(5, 10),
        }
        h.tick_with_usage(usage)
        warnings = [e for e in h.events if isinstance(e, ConfigWarningEvent)]
        assert len(warnings) == 1
        assert "Fabel" in warnings[0].message
        assert warnings[0].to_json()["event"] == "config-warning"
        h.tick_with_usage(usage)
        warnings = [e for e in h.events if isinstance(e, ConfigWarningEvent)]
        assert len(warnings) == 1  # once per run, not per tick

    def test_no_false_warning_while_an_account_is_unreadable(self, temp_home):
        h = self._seed(temp_home, model="Fabel")
        h.tick_with_usage({
            "1": _model_usage(5, 10), "2": _model_usage(5, 10), "3": None,
        })
        assert not any(isinstance(e, ConfigWarningEvent) for e in h.events)
        # Once every account reports, the check completes and warns.
        h.tick_with_usage({
            "1": _model_usage(5, 10),
            "2": _model_usage(5, 10),
            "3": _model_usage(5, 10),
        })
        assert any(isinstance(e, ConfigWarningEvent) for e in h.events)

    def test_matching_name_never_warns(self, temp_home):
        h = self._seed(temp_home, model="Fable")
        h.tick_with_usage({
            "1": _model_usage(5, 10),
            "2": _model_usage(5, 10),
            "3": _model_usage(5, 10),
        })
        assert not any(isinstance(e, ConfigWarningEvent) for e in h.events)


# --- consume-first strategy ----------------------------------------------------

# Weekly-reset instants in ascending order (all valid ISO-8601, absolute).
# The 2024 dates are all far in the FUTURE relative to FakeClock's epoch
# (1_000_000.0 ≈ 1970-01-12); _R_PAST is before it.
_R_PAST = "1970-01-10T00:00:00Z"
_R_SOON = "2024-01-05T00:00:00Z"
_R_LATER = "2024-01-08T00:00:00Z"
_R_LATEST = "2024-01-10T00:00:00Z"


def _usage7(pct5: float, pct7: float, reset7: str | None = None) -> dict:
    """Usage with an explicit 7-day window (utilization + optional reset)."""
    seven: dict = {"pct": pct7}
    if reset7:
        seven["resets_at"] = reset7
    return {"five_hour": {"pct": pct5}, "seven_day": seven}


class TestConsumeFirstStrategy:
    def _harness(self, temp_home: Path) -> EngineHarness:
        h = EngineHarness(temp_home, strategy="consume-first")
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)
        return h

    def test_below_threshold_switches_to_soonest_weekly_reset(self, temp_home):
        h = self._harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),    # active resets later
            "2": _usage7(10, 10, _R_SOON),     # soonest -> consume first
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        sw = next(e for e in h.events if isinstance(e, SwitchEvent))
        assert sw.trigger == "consume-first"
        assert sw.to_ref == {"number": 2, "email": "b@example.com"}

    def test_stays_when_active_already_resets_soonest(self, temp_home):
        h = self._harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_SOON),     # active is soonest -> stay
            "2": _usage7(10, 10, _R_LATER),
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["already-consuming-soonest"]

    def test_over_threshold_prefers_soonest_reset_over_max_headroom(self, temp_home):
        h = self._harness(temp_home)
        # Active over threshold -> must move. #2 has LESS headroom but resets
        # sooner; #3 has more headroom but resets latest. consume-first -> #2.
        outcome = h.tick_with_usage({
            "1": _usage7(95, 20, _R_LATER),
            "2": _usage7(50, 40, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2

    def test_a_consume_first_target_must_still_be_healthy(self, temp_home):
        """The threshold landing gate has no cover on the consume-first path.

        `if (100.0 - h) >= settings.threshold and not all_above: continue` ->
        `if False` survives the whole suite. On the `best` path the hysteresis
        gate below masks it; consume-first has no headroom test at all — its
        `elif` compares weekly resets only, so with the gate gone a 96%-used
        account whose weekly window resets sooner is a valid target. Measured:

            active 1: 60 pts (util 40%), weekly reset 500h
            peer   2:  4 pts (util 96%), weekly reset  10h
            ORIGINAL ranking=[]      tick -> NO_ACTION
            MUTANT   ranking=['2']   tick -> SWITCHED to 2

        Landing there re-triggers on the very next tick, which is the harm the
        comment on that gate describes.

        TWO accounts on purpose: a third healthy peer would win the sort and
        hide the defect behind a correct answer.
        """
        h = EngineHarness(temp_home, strategy="consume-first")
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        outcome = h.tick_with_usage({
            "1": _usage7(40, 40, _R_LATEST),   # active, 60 pts, resets LAST
            "2": _usage7(96, 96, _R_SOON),     # 4 pts: sooner, but spent
        })
        assert outcome is not TickOutcome.SWITCHED, (
            "consume-first moved onto an account at 96% utilization because "
            "its weekly window resets sooner — it re-triggers next tick"
        )
        assert h.active_number() == 1

    def test_respects_cooldown(self, temp_home):
        h = self._harness(temp_home)  # default cooldown 300s
        h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert h.active_number() == 2  # switched to soonest
        h.events.clear()
        # Now a sooner account (#3) appears, but we're within cooldown.
        outcome = h.tick_with_usage({
            "2": _usage7(20, 20, _R_LATER),
            "1": _usage7(10, 10, _R_LATEST),
            "3": _usage7(10, 10, _R_SOON),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2
        assert "cooldown" in [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]

    def test_locked_recheck_stops_concurrent_engine(self, temp_home):
        """The under-lock cooldown recheck in _perform must cover consume-first.

        The tick-level gate reads state *before* the lock, so an engine that
        read state before another engine's switch passes it on a stale
        snapshot; only the recheck inside _perform serializes the two. Drive a
        loser engine through _perform with a stale pre-lock read and a usage
        view that ranks a different target, and assert it backs off instead of
        double-switching inside the cooldown window.
        """
        h = self._harness(temp_home)  # default cooldown 300s
        loser = h._make_engine()
        # Winner: 1 -> 2 (soonest reset), records lastSwitchAt.
        h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert h.active_number() == 2
        h.events.clear()
        # Loser's first (pre-lock) state read predates the winner's write; its
        # usage view ranks #3 soonest, so it reaches _perform for a different
        # target and only the locked recheck can stop it.
        real_read = loser._read_state
        calls: list[bool] = []

        def racing_read() -> dict:
            calls.append(True)
            return {} if len(calls) == 1 else real_read()

        entries = {
            num: _entry_for(value, h.clock.now)
            for num, value in {
                "2": _usage7(20, 20, _R_LATER),
                "1": _usage7(10, 10, _R_LATEST),
                "3": _usage7(10, 10, _R_SOON),
            }.items()
        }
        with patch.object(loser, "_read_state", side_effect=racing_read):
            with patch.object(
                h.switcher, "usage_entries_by_account", return_value=entries
            ):
                outcome = loser.tick()
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 2  # no double-switch
        assert "cooldown" in [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]

    def test_reset_unknown_when_active_reset_missing(self, temp_home):
        # Active has no seven_day.resets_at: the strictly-sooner filter skips
        # every candidate, so the strategy is inert — say so, instead of the
        # false "already consuming soonest".
        h = self._harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20),              # no reset timestamp
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["reset-unknown"]

    def test_unreadable_candidates_stay_no_comparison(self, temp_home):
        # Every candidate unreadable this tick is a BLOCKED no-comparison for
        # any strategy — consume-first must not relabel it as a healthy hold.
        h = self._harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": None,
            "3": None,
        })
        assert outcome is TickOutcome.BLOCKED
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["no-comparison"]

    def test_exhausted_candidates_hold_without_false_reset_claim(self, temp_home):
        # All candidates at their limit while the active account is healthy:
        # staying put is right, but the detail must not claim the active
        # account resets first.
        h = self._harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(100, 100, _R_SOON),
            "3": _usage7(100, 100, _R_LATEST),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        holds = [e for e in h.events if isinstance(e, NoSwitchEvent)]
        assert [e.reason for e in holds] == ["already-consuming-soonest"]
        assert holds[0].detail == "no sooner-resetting account with room to spare"

    def test_single_account_below_threshold_is_no_action(self, temp_home):
        # Exit-code parity with `best`: a healthy below-threshold tick with
        # zero candidates is NO_ACTION/below-threshold, not BLOCKED/
        # no-candidates — cron wrappers key on the documented exit codes.
        h = EngineHarness(temp_home, strategy="consume-first")
        h.seed(1, "a@example.com")
        h.make_live("a@example.com", 1)
        outcome = h.tick_with_usage({"1": _usage7(20, 20, _R_SOON)})
        assert outcome is TickOutcome.NO_ACTION
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["below-threshold"]

    def test_api_key_only_peers_below_threshold_is_no_action(self, temp_home):
        # Same exit-code parity when the only alternatives are included
        # API-key accounts: they're never consume-first targets (no weekly
        # window), so a healthy below-threshold tick must stay
        # NO_ACTION/below-threshold — not fall through to a false
        # BLOCKED/no-comparison from the empty OAuth ranking.
        h = EngineHarness(
            temp_home, strategy="consume-first", include_api_key_accounts=True
        )
        h.seed(1, "a@example.com")
        h.seed(2, "key@token.local")
        h.make_live("a@example.com", 1)
        data = h.switcher._get_sequence_data()
        data["accounts"]["2"]["kind"] = "api_key"
        h.switcher._write_json(h.switcher.sequence_file, data)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_SOON),
            "2": "api key",
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["below-threshold"]

    def test_skips_sooner_account_that_is_exhausted(self, temp_home):
        h = self._harness(temp_home)
        # #2 resets soonest but is itself at its limit (no headroom) -> ignored;
        # #3 resets later but has room and is sooner than active -> switch there.
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATEST),   # active resets latest
            "2": _usage7(100, 100, _R_SOON),   # soonest but exhausted
            "3": _usage7(10, 10, _R_LATER),    # sooner than active, has room
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_best_strategy_unaffected_below_threshold(self, temp_home):
        # Regression: default (best) still holds below threshold even when a
        # peer resets sooner — consume-first behavior must be opt-in.
        h = EngineHarness(temp_home)  # strategy defaults to "best"
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        assert [e.reason for e in h.events if isinstance(e, NoSwitchEvent)] == [
            "below-threshold"
        ]

    def test_candidate_with_past_reset_is_not_selected(self, temp_home):
        # A stale snapshot whose resets_at has already elapsed means the
        # weekly window just rolled over — the LEAST perishable quota. It
        # must rank as unknown, never as "soonest".
        h = self._harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_PAST),     # inverted pick pre-fix
            "3": _usage7(10, 10, _R_SOON),
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3
        sw = next(e for e in h.events if isinstance(e, SwitchEvent))
        assert sw.to_ref == {"number": 3, "email": "c@example.com"}

    def test_active_past_reset_holds_reset_unknown(self, temp_home):
        # The active account's own reset can be stale too: past == unknown,
        # which lands on the existing reset-unknown hold.
        h = self._harness(temp_home)
        outcome = h.tick_with_usage({
            "1": _usage7(20, 20, _R_PAST),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATER),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["reset-unknown"]

    def _two_phase_tick(
        self, h: EngineHarness, stored: dict, fresh: dict
    ) -> tuple[TickOutcome, list[set]]:
        """Drive one tick where stored-snapshot collections serve ``stored``
        and the all-candidates escalation serves ``fresh``.

        These ticks run outside the escalation band (utilization far below
        threshold - ESCALATION_MARGIN_PCT), so the collector never escalates
        on its own and the only all-candidates call a tick can make is the
        consume-first phase-2 refetch — the returned fetch sets prove whether
        it happened.
        """
        fetch_sets: list[set] = []

        def collect(fetch=None, **_kwargs):
            requested = set(fetch or ())
            fetch_sets.append(requested)
            view = fresh if requested == {"1", "2", "3"} else stored
            return {
                num: _entry_for(value, h.clock.now)
                for num, value in view.items()
            }

        with patch.object(
            h.switcher, "usage_entries_by_account", side_effect=collect
        ):
            outcome = h.engine.tick()
        return outcome, fetch_sets

    def test_two_phase_refetch_disqualifies_stale_pick(self, temp_home):
        # The stored snapshot ranks #2; the phase-2 refetch shows it
        # exhausted. The tick must re-decide on the fresh data and hold.
        h = self._harness(temp_home)
        stored = {
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        }
        fresh = {
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(100, 100, _R_SOON),   # burned out since the snapshot
            "3": _usage7(10, 10, _R_LATEST),
        }
        outcome, fetch_sets = self._two_phase_tick(h, stored, fresh)
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["already-consuming-soonest"]
        assert fetch_sets.count({"1", "2", "3"}) == 1  # phase 2 fired once

    def test_two_phase_refetch_confirms_switch(self, temp_home):
        # Fresh data agrees with the stored pick: the switch proceeds through
        # the freshness gate (entries served by phase 2 are age-0).
        h = self._harness(temp_home)
        view = {
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        }
        outcome, fetch_sets = self._two_phase_tick(h, view, view)
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert {"1", "2", "3"} in fetch_sets

    def test_two_phase_refetch_reranks_to_fresh_best(self, temp_home):
        # Phase 2 is a full re-rank, not a yes/no check on the provisional
        # target: #2 stays eligible on fresh data, but #3 now resets sooner
        # and must win.
        h = self._harness(temp_home)
        stored = {
            "1": _usage7(20, 20, _R_LATEST),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATER),
        }
        fresh = {
            "1": _usage7(20, 20, _R_LATEST),
            "2": _usage7(10, 10, _R_LATER),    # still sooner than active
            "3": _usage7(10, 10, _R_SOON),     # but #3 is now soonest
        }
        outcome, _ = self._two_phase_tick(h, stored, fresh)
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3

    def test_threshold_crossed_in_phase_two_holds_then_escapes_next_tick(
        self, temp_home
    ):
        # Deliberate design pin: phase 2 never re-classifies the trigger
        # mid-tick. When the fresh active is over the threshold with no
        # strictly-sooner candidate, the tick holds; the NEXT tick classifies
        # at-limit and escapes normally (no freshness gate on escapes).
        h = self._harness(temp_home)
        stored = {
            "1": _usage7(20, 20, _R_LATER),
            "2": _usage7(10, 10, _R_SOON),
            "3": _usage7(10, 10, _R_LATEST),
        }
        fresh = {
            "1": _usage7(100, 20, _R_LATER),   # crossed while the snapshot aged
            "2": _usage7(10, 10, _R_LATEST),   # no longer strictly sooner
            "3": _usage7(10, 10, _R_LATEST),
        }
        outcome, fetch_sets = self._two_phase_tick(h, stored, fresh)
        assert outcome is TickOutcome.NO_ACTION
        assert h.active_number() == 1
        assert not any(isinstance(e, SwitchEvent) for e in h.events)
        assert {"1", "2", "3"} in fetch_sets
        reasons = [e.reason for e in h.events if isinstance(e, NoSwitchEvent)]
        assert reasons == ["already-consuming-soonest"]
        h.events.clear()
        outcome = h.tick_with_usage(fresh)
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 2
        sw = next(e for e in h.events if isinstance(e, SwitchEvent))
        assert sw.trigger == "at-limit"


class TestEveryAccountAboveThreshold:
    """With nothing below the threshold, go to whatever comes back soonest.

    The state that motivated this was measured, not imagined: all three
    accounts' 5-hour windows at 100/99/95%, threshold 90. Every candidate
    failed the "landing must be healthy" gate, so the engine sat still while
    the active account burned to 100% and Claude Code took a hard session
    limit — with a peer whose window reset in 8 minutes never tried. Claude
    Code's own retry timer is driven by the rate-limit headers it already
    received, so once that limit lands no credential swap can shorten it; the
    only cure is not to arrive there.

    Below the threshold nothing changes: a single healthy peer still wins the
    normal way, and the hysteresis margin still keeps two near-line accounts
    from ping-ponging.
    """

    def _at(self, harness, seconds: float) -> str:
        from datetime import datetime, timezone

        return (
            datetime.fromtimestamp(harness.clock.now + seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def test_moves_to_the_soonest_recovering_account(self, harness):
        """The measured shape: active 99, peers 100 and 95. Account 3 is the
        only one both viable and soon, and it is where we must land."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 3600 * 2)),   # active, back in 2h
            "2": _usage(100, self._at(harness, 600)),       # at limit — never a target
            "3": _usage(95, self._at(harness, 480)),        # back in 8 minutes
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_soonest_wins_over_most_headroom(self, harness):
        """Ranking flips in this state: the usual "most headroom" pick is the
        wrong one when every account is nearly spent — what matters is which
        one can work again first."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 3600)),
            "2": _usage(91, self._at(harness, 3600 * 3)),  # most headroom, latest back
            "3": _usage(97, self._at(harness, 300)),       # least headroom, soonest back
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_a_single_healthy_peer_still_wins_normally(self, harness):
        """The escape must not fire while an ordinary target exists."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 3600)),
            "2": _usage(95, self._at(harness, 60)),   # soonest, but still spent
            "3": _usage(20, self._at(harness, 3600 * 5)),  # healthy
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_below_threshold_is_untouched(self, harness):
        """Nothing about the ordinary below-threshold path changes."""
        outcome = harness.tick_with_usage({
            "1": _usage(50), "2": _usage(10), "3": _usage(10),
        })
        assert outcome is TickOutcome.NO_ACTION
        assert harness.active_number() == 1

    def test_all_at_limit_still_reports_exhausted(self, harness):
        """h <= 0 is still never a target: with everything truly maxed there
        is nowhere to go and the exhausted path must still own that case."""
        outcome = harness.tick_with_usage({
            "1": _usage(100, self._at(harness, 600)),
            "2": _usage(100, self._at(harness, 300)),
            "3": _usage(100, self._at(harness, 900)),
        })
        assert outcome is TickOutcome.BLOCKED
        assert harness.active_number() == 1

    def test_unknown_reset_sorts_last_not_first(self, harness):
        """A candidate whose reset nobody knows must not masquerade as
        'back immediately' and beat a measured, genuinely imminent one."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 3600)),
            "2": _usage(95),                          # no resets_at at all
            "3": _usage(97, self._at(harness, 600)),  # known, soon
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_does_not_flap_between_two_near_equal_accounts(self, harness):
        """The escape relaxes the percentage-point hysteresis, so it owes the
        anti-flap guarantee on its own axis: two accounts whose windows roll
        over at nearly the same time must not trade places forever."""
        a = self._at(harness, 600)
        b = self._at(harness, 660)  # 60s apart — inside RECOVERY_HYSTERESIS_S
        first = harness.tick_with_usage({
            "1": _usage(99, a), "2": _usage(98, b), "3": _usage(100, a),
        })
        assert first is TickOutcome.BLOCKED, "60s sooner is not worth a switch"
        assert harness.active_number() == 1

    def test_a_meaningfully_sooner_account_still_wins(self, harness):
        """The margin must not be so wide it swallows the real case."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 3600)),
            "2": _usage(98, self._at(harness, 600)),  # an hour sooner
            "3": _usage(100, self._at(harness, 60)),  # at limit — not a target
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2

    def test_consume_first_gets_the_same_anti_flap_guard(self, temp_home):
        """The escape must not depend on which strategy is configured.

        `if consume_first:` used to catch first, so a consume-first user
        reached the ranking (soonest binding recovery) without ever passing
        the recovery-hysteresis gate — filtering on one axis while sorting on
        another. Two accounts whose windows roll over a minute apart could
        then trade places forever.
        """
        h = EngineHarness(temp_home, strategy="consume-first")
        h.seed(1, "a@example.com"); h.seed(2, "b@example.com")
        h.seed(3, "c@example.com"); h.make_live("a@example.com", 1)
        a = self._at(h, 600)
        b = self._at(h, 660)  # 60s apart — inside RECOVERY_HYSTERESIS_S
        outcome = h.tick_with_usage({
            "1": _usage(99, a), "2": _usage(98, b), "3": _usage(100, a),
        })
        assert outcome is TickOutcome.BLOCKED, (
            "consume-first skipped the recovery hysteresis"
        )
        assert h.active_number() == 1

    def test_at_limit_trigger_still_ignores_the_landing_rule(self, harness):
        """at-limit and failover skip the whole proactive block. The escape
        must not have made the active account's 100% case *narrower* — an
        account with real headroom still wins there regardless of resets."""
        outcome = harness.tick_with_usage({
            "1": _usage(100, self._at(harness, 60)),   # active, at limit
            "2": _usage(30, self._at(harness, 86400)),  # healthy but far reset
            "3": _usage(95, self._at(harness, 120)),    # soon but spent
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            "at-limit must still take the account with real headroom"
        )
        sw = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert sw.trigger == "at-limit"


class TestRecoveryHorizon:
    """The recovery escape must not spend real headroom on a distant reset.

    #202's rule ("go where quota returns first") was measured on minutes-scale
    resets and shipped with no upper bound, so it applied identically days
    out — measured live, 9 points of headroom traded for 2 on a reset nobody
    reaches today. Past the horizon, ranking returns to headroom.
    """

    def _at(self, harness, seconds: float) -> str:
        from datetime import datetime, timezone

        return (
            datetime.fromtimestamp(harness.clock.now + seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def test_a_minutes_away_reset_still_wins(self, harness):
        """The #202 design case is unchanged: an 8-minute wait is worth 9
        points of headroom."""
        outcome = harness.tick_with_usage({
            "1": _usage(91, self._at(harness, 7200)),   # active, 9 left, back in 2h
            "2": _usage(94, self._at(harness, 1800)),
            "3": _usage(98, self._at(harness, 480)),    # back in 8 minutes
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3

    def test_a_days_away_reset_does_not_buy_headroom(self, harness):
        """The measured live shape. Every reset is days out, so ranking falls
        back to headroom and the account with 9 points left keeps the work."""
        outcome = harness.tick_with_usage({
            "1": _usage(91, self._at(harness, 109 * 3600)),  # active, 9 left
            "2": _usage(94, self._at(harness, 80 * 3600)),
            "3": _usage(98, self._at(harness, 50 * 3600)),   # 2 left, soonest
        })
        assert harness.active_number() == 1, (
            "traded 9 points of headroom for 2 on a reset nobody reaches today"
        )

    def test_an_unreadable_peer_does_not_veto_the_spent_check(
        self, temp_home
    ):
        """The spent check ranks the CANDIDATES, not every account in `usage`.

        `headroom` is keyed off `usage`, which carries a row for accounts the
        loop can never pick — a sentinel (unreadable credential, keychain
        locked) yields `None` headroom. Testing `headroom.values()` let one
        such row make `all(...)` False forever, so the spent escape could not
        fire: measured, three accounts at 99% days out and the engine parked
        on the one resetting LAST. That is the bug SPENT_HEADROOM_PCT exists
        to prevent, reintroduced through the iteration set.
        """
        h = EngineHarness(temp_home)
        for n, e in ((1, "a@example.com"), (2, "b@example.com"),
                     (3, "c@example.com"), (4, "d@example.com")):
            h.seed(n, e)
        h.make_live("a@example.com", 1)

        outcome = h.tick_with_usage({
            "1": _usage(99, self._at(h, 109 * 3600)),  # active, resets LAST
            "2": _usage(99, self._at(h, 80 * 3600)),
            "3": _usage(99, self._at(h, 50 * 3600)),   # soonest
            "4": USAGE_TOKEN_EXPIRED,                  # sentinel: headroom None
        })
        assert outcome is TickOutcome.SWITCHED
        assert h.active_number() == 3, (
            "an unreadable peer vetoed the spent check and parked the engine "
            "on the account resetting last"
        )

    def test_a_weekly_bound_active_does_not_refuse_a_peer_back_in_minutes(
        self, harness
    ):
        """The horizon is asked PER CANDIDATE, not once on the active.

        An active bound by its WEEKLY window sits days out while a peer's
        five-hour window returns in minutes. Asking the active refused that
        peer — the #202 case this horizon is supposed to preserve. The
        existing tests never caught it because they populate only a 5h
        window, so the active's reset and the candidates' always moved
        together.
        """
        outcome = harness.tick_with_usage({
            # active: 5h fine, WEEKLY at 96% resetting 109h out — days.
            "1": _usage7(10, 96, self._at(harness, 109 * 3600)),
            # peer: binding 5h window back in 8 minutes.
            "2": _usage(98, self._at(harness, 480)),
            "3": _usage(99, self._at(harness, 90 * 3600)),
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            "refused a peer returning in 8 minutes because the ACTIVE was "
            "weekly-bound"
        )

    def test_an_unknown_active_reset_keeps_the_headroom(self, harness):
        """`inf` means unknown OR already elapsed — not 'rank by reset'.

        It used to keep the recovery axis, which re-armed the exact trade the
        horizon forbids: measured, an active with 9 points and no `resets_at`
        moved to a peer with 1 point resetting 50h out. No evidence that a
        sooner reset helps is a reason to keep the headroom, not spend it.
        """
        outcome = harness.tick_with_usage({
            "1": _usage(91),                                # active, no reset
            "2": _usage(98, self._at(harness, 80 * 3600)),
            "3": _usage(99, self._at(harness, 50 * 3600)),  # soonest, 1 left
        })
        assert harness.active_number() == 1, (
            "traded 9 points for 1 because the active's reset was unknown"
        )
        assert outcome is not TickOutcome.SWITCHED

    def test_a_peer_with_real_headroom_still_wins_past_the_horizon(self, harness):
        """Falling back to headroom is not "never move": a peer holding
        materially more quota is still the right landing, days-away or not."""
        outcome = harness.tick_with_usage({
            "1": _usage(97, self._at(harness, 50 * 3600)),   # active, 3 left
            "2": _usage(91, self._at(harness, 109 * 3600)),  # 9 left
            "3": _usage(98, self._at(harness, 60 * 3600)),
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2


class TestTheHorizonDoesNotDiscardWhatItAlreadyKnows:
    """Two regressions from carrying the horizon into the ranking.

    Both are cases where the PR had the right answer in hand and dropped it —
    base 9f35426 gets them right, so neither is inherited.
    """

    def _at(self, harness, seconds: float) -> str:
        from datetime import datetime, timezone

        return (
            datetime.fromtimestamp(harness.clock.now + seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def test_equal_headroom_past_the_horizon_takes_the_sooner_reset(self, harness):
        """The tier-1 key hard-coded ``0.0`` where ``recovery_ts`` belongs.

        Two peers with IDENTICAL headroom, both past the horizon, one returning
        in 5h and one in 500h. With the second slot zeroed the tie falls
        through to sequence order, so which account is chosen depends on which
        slot number it happens to occupy:

            near is acct 3  ->  base picks 3 (5h),  head picked 2 (500h)

        `recovery_ts` is already computed at that point and is strictly better
        than list order at zero cost. Headroom still outranks it — the tier
        byte separates the two axes, and `-h` still comes first within tier 1.
        """
        out = harness.tick_with_usage({
            "1": _usage(96, self._at(harness, 300 * 3600)),   # active, 4 pts
            "2": _usage(92, self._at(harness, 500 * 3600)),   # 8 pts, LAST
            "3": _usage(92, self._at(harness, 5 * 3600)),     # 8 pts, soonest
        })
        assert out is TickOutcome.SWITCHED
        assert harness.active_number() == 3, (
            "equal headroom, and the 5h reset lost to the 500h one on slot order"
        )

    def test_a_peer_worth_having_is_not_filtered_out_of_the_worth_having_check(
        self, harness
    ):
        """The floor must not exclude a peer that plainly has quota.

        `best_candidate_headroom` was scoped by
        `active_headroom x HORIZON_HEADROOM_RATIO` — but that constant is an
        ANTI-FLAP MARGIN, not a "worth having" cutoff. A peer at
        2x-minus-epsilon the active's headroom is very much worth having; it
        merely fails this tick's margin.

        With every candidate below the floor, `default=0.0` makes
        `best_candidate_headroom` 0.0, which SATISFIES the spent clause — so
        the clause fires for everybody, the horizon check is never reached, and
        ranking falls to soonest reset regardless of headroom. The engine then
        takes a nearly-empty account over one holding 60x more:

            active   3.00 pts, 500h     peerA 5.99 pts, 400h
            peerB    0.10 pts, 200h  <- chosen

        Three-way: base 9f35426 also lands on peerB, but 0457cb0 (this PR
        before the veto-scope fix) holds the active. So the fix reintroduced
        base's answer in a band the commit before it had already made safe.

        Asserted as "does not take the nearly-empty account". Whether it takes
        peerA or holds is the ANTI-FLAP margin's call, and at 5.99 against a
        6.00 margin holding is correct — that is a separate question from
        whether peerA counts as quota existing, which is what this pins.
        """
        out = harness.tick_with_usage({
            "1": _usage(97.00, self._at(harness, 500 * 3600)),  # active, 3.00
            "2": _usage(94.01, self._at(harness, 400 * 3600)),  # 5.99
            "3": _usage(99.90, self._at(harness, 200 * 3600)),  # 0.10
        })
        assert harness.active_number() != 3, (
            "took the 0.10-point account over one holding 5.99 — the floor "
            "excluded the peer that made the spent clause false, and an empty "
            "max reads as 'nothing is worth having'"
        )
        assert out is not TickOutcome.SWITCHED or harness.active_number() == 2

    def test_an_unchoosable_peer_does_not_veto_the_reset_ranking(self, harness):
        """``best_candidate_headroom`` counted a candidate the ranking cannot pick.

        The spent check asks "is anything worth having?" of the BEST candidate.
        A peer holding 3.05 points is above ``SPENT_HEADROOM_PCT``, so the
        answer is no for everybody — yet that peer cannot itself be chosen,
        because 3.05 < 3.0 x HORIZON_HEADROOM_RATIO fails the ratio gate.
        Nothing qualifies, and the engine parks on the account that returns
        LAST:

            active   3.00 pts, resets in 200h   <- stays here
            peer     3.00 pts, resets in  10h   <- 190h sooner, refused
            vetoer   3.05 pts, resets in 500h   <- unchoosable, decides

        Measured against base: base switches at 3.05 and this branch blocks.
        The veto band is (SPENT_HEADROOM_PCT, active x RATIO], up to 3 points
        wide, so it is not an edge case in the endgame this code is for.

        ``test_an_unreadable_peer_does_not_veto_the_spent_check`` pins the same
        shape for an UNREADABLE peer; a readable one 0.05 points over the line
        does the same damage.
        """
        out = harness.tick_with_usage({
            "1": _usage(97.0, self._at(harness, 200 * 3600)),   # active, 3 pts
            "2": _usage(97.0, self._at(harness, 10 * 3600)),    # 3 pts, sooner
            "3": _usage(96.95, self._at(harness, 500 * 3600)),  # 3.05 pts
        })
        assert out is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            "a peer that cannot be chosen vetoed the ranking for everyone"
        )


class TestHorizonAxisDoesNotFlap:
    """Past the horizon the headroom axis needs its own anti-flap margin.

    The first cut required only *strictly more* headroom, which is no margin
    at all: one point is enough to move, and the account we move to burns that
    point back within a poll or two. Measured live on 2026-07-30, four switches
    in 35 minutes, each buying one point and each costing a credential rewrite:

        17:28  acct 1 (5% left) -> acct 2 (6%)
        17:49  acct 2 (4% left) -> acct 1 (5%)
        17:54  acct 1           -> acct 2
        18:03  acct 2 (2% left) -> acct 1 (3%)

    The ordinary path uses ``hysteresis_pct`` (10 points), but that is
    unmeetable here by construction — everything is within a few points of its
    limit, so requiring ten would park the engine and let it ride into the
    wall, which is the failure #202 exists to prevent.

    A RATIO is the right unit in the endgame: with two points left, what
    matters is how many times more runway the target has, not how many points.
    Requiring the target to hold ``HORIZON_HEADROOM_RATIO`` times the active
    account's headroom makes the move one-way by construction — the reverse
    would need the new active to fall to a quarter of what it just beat.
    """

    def _at(self, harness, seconds: float) -> str:
        from datetime import datetime, timezone

        return (
            datetime.fromtimestamp(harness.clock.now + seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def _days_out(self, harness, hours):
        return self._at(harness, hours * 3600)

    def test_one_point_of_headroom_does_not_move(self, harness):
        """The measured flap: 95% active against a 94% peer, both days out."""
        outcome = harness.tick_with_usage({
            "1": _usage(95, self._days_out(harness, 109)),   # active, 5 left
            "2": _usage(94, self._days_out(harness, 80)),    # 6 left
            "3": _usage(99, self._days_out(harness, 50)),
        })
        assert harness.active_number() == 1, (
            "moved for one point of headroom; the target burns it back and the "
            "engine ping-pongs (measured: 4 switches in 35 minutes)"
        )
        assert outcome is not TickOutcome.SWITCHED

    def test_the_return_leg_is_blocked_too(self, harness):
        """Same shape with the roles reversed — symmetric, so neither leg runs."""
        outcome = harness.tick_with_usage({
            "1": _usage(96, self._days_out(harness, 109)),   # active, 4 left
            "2": _usage(95, self._days_out(harness, 80)),    # 5 left
            "3": _usage(99, self._days_out(harness, 50)),
        })
        assert harness.active_number() == 1
        assert outcome is not TickOutcome.SWITCHED

    def test_a_pair_straddling_the_horizon_does_not_ping_pong(self, harness):
        """Each guard is one-way on ITS OWN axis — but the axis itself flips.

        ``_recovery_is_useful`` reads the ACTIVE account's headroom and the
        CANDIDATE's reset, and a switch swaps both operands. So a pair that
        straddles the horizon takes the recovery gate going out and the
        headroom gate coming back, and neither guard ever sees the other leg:

            acct 1   8 points, reset 109h out   (past the horizon)
            acct 2   3 points, reset 3.5h out   (inside it)

            active=1 -> candidate 2 is inside  -> recovery axis  -> 3.5h < 109h
            active=2 -> candidate 1 is outside -> headroom axis  -> 8 >= 3*2

        Both legs qualify on frozen inputs, so the engine rewrites credentials
        every cooldown until the sooner reset actually lands. Every other test
        in this class ticks ONCE, which is why the pair went unseen.
        """
        r_far = self._days_out(harness, 109)
        r_near = self._days_out(harness, 3.5)
        seen = []
        for _ in range(6):
            harness.tick_with_usage({
                "1": _usage(92, r_far),    # 8 points, returns days out
                "2": _usage(97, r_near),   # 3 points, returns inside 4h
            })
            seen.append(harness.active_number())
            harness.clock.advance(301.0)   # past the 300s cooldown
        assert len(set(seen)) == 1, (
            f"cross-axis oscillation: active trace {seen} — each leg passes "
            "the guard belonging to the OTHER leg's axis"
        )

    def test_the_spent_fallback_needs_a_meaningfully_sooner_reset(self, harness):
        """The fallback is bounded by the SAME hysteresis the recovery axis uses.

        Its other two guards are pinned by the tests above (dropping the spent
        gate reddens `test_one_point_of_headroom_does_not_move` and
        `test_the_return_leg_is_blocked_too`; dropping `h >= active` reddens
        `test_a_peer_worth_having_is_not_filtered_out_of_the_worth_having_check`).
        The margin was the one nothing killed — measured, replacing
        `< active_recovery_ts - RECOVERY_HYSTERESIS_S` with a bare
        `< active_recovery_ts` left all 168 tests in this file green.

        Exhaustive 2- and 3-account sweep over headroom x reset, 16200 shapes:
        exactly 42 change answer, all of the shape below.

            acct 1   2.5 points, reset 500.02h out   (active)
            acct 2   4.0 points, reset 500.00h out   (72s sooner)

            with the margin     no move, both legs
            without it          active=1 moves to 2; active=2 holds

        One-way, so not a flap — a credential rewrite bought with 72 seconds
        of earlier return, on a pair that both come back in three weeks. The
        margin is what makes "sooner" mean sooner enough to be worth the write.
        """
        outcome = harness.tick_with_usage({
            "1": _usage(97.5, self._days_out(harness, 500.02)),  # active, 2.5 left
            "2": _usage(96.0, self._days_out(harness, 500.0)),   # 4 left, 72s sooner
        })
        assert harness.active_number() == 1, (
            "moved for a 72-second-sooner reset three weeks out — inside "
            "RECOVERY_HYSTERESIS_S, which is what bounds the write rate"
        )
        assert outcome is not TickOutcome.SWITCHED

    def test_the_tier_byte_puts_a_returning_peer_ahead_of_a_distant_one(
        self, harness
    ):
        """`(0, ...)` before `(1, ...)` — the tier prefix itself, not its tail.

        Both existing tier tests compare candidates WITHIN one tier, so the
        byte cancels and neither pins it. Collapsing it to a flat key left the
        suite green.

        A candidate returning inside the horizon beats one that does not,
        whatever its headroom: acct 2 is nearly spent but works again in an
        hour; acct 3 has nine points that never return this session.
        """
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._days_out(harness, 300)),    # active, 1 left
            "2": _usage(98.5, self._days_out(harness, 1)),    # 1.5 left, back in 1h
            "3": _usage(91, self._days_out(harness, 400)),    # 9 left, never
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            f"landed on {harness.active_number()} — took headroom that never "
            "returns over a peer that works again in an hour"
        )

    def test_the_fallback_breaks_a_reset_tie_by_headroom(self, harness):
        """The fallback key's THIRD slot: `(0, recovery_ts, -h)`.

        `test_the_fallback_ranks_by_reset_not_by_headroom` pins the second
        slot (reset leads). The third was untested — `-h` to `h` left the suite
        green. It needs an actual tie in `recovery_ts` AND both peers routed
        through the fallback, which requires the active to sit exactly at
        SPENT_HEADROOM_PCT so neither peer meets the ratio.
        """
        same = self._days_out(harness, 10)
        outcome = harness.tick_with_usage({
            "1": _usage(97, self._days_out(harness, 300)),   # active, 3.0 left
            "2": _usage(97, same),                            # 3.00 left
            "3": _usage(96.95, same),                         # 3.05 left
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3, (
            f"landed on {harness.active_number()} — at an equal reset the "
            "fallback took the smaller headroom"
        )

    def test_past_the_horizon_headroom_decides_before_the_reset(self, harness):
        """Tier 1 is `(1, -h, recovery_ts)` — headroom leads, reset breaks ties.

        Past the horizon the reset is days out either way, so it cannot be the
        thing that decides; the headroom is the only resource that still does
        work this session. The reset stays in the key so two equal-headroom
        peers do not tie into sequence order.

        Nothing pinned the ORDER: swapping to `(1, recovery_ts, -h)` left the
        whole suite green. The one test that touches the tier uses EQUAL
        headroom (92/92), where both orderings agree — it kills the
        hard-coded-`0.0` mutant and not this one.

        Sweep over 23328 three-account shapes: 2040 change answer. The shape
        below is one, and the trade the reset-first key makes is 2 points of
        headroom for a reset 10 hours sooner, on a pair that both return
        within a day.
        """
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._days_out(harness, 20)),     # active, 1 left
            "2": _usage(98, self._days_out(harness, 10)),     # 2 left, sooner
            "3": _usage(96, self._days_out(harness, 20)),     # 4 left
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3, (
            f"landed on {harness.active_number()} — the reset outranked twice "
            "the headroom, past a horizon where neither reset is near"
        )

    def test_a_burn_walk_settles_instead_of_oscillating(self, harness):
        """A-B-A under burn is the fleet changing regime, not a gate leaking.

        The reviewed concern was that the outbound gate is RELATIVE
        (`h >= active x HORIZON_HEADROOM_RATIO`) while the fallback's is
        ABSOLUTE (`active <= SPENT_HEADROOM_PCT`), so a pair could take one
        gate out and the other back. Measured at the moment of each move,
        only the active burning, both resets past the horizon:

            out    active 2.0 / peer 4.0   headroom axis, 4.0 >= 2.0x2
            back   active 3.0 / peer 2.0   recovery  axis, 10h vs 80h

        Both legs are legitimate on the axis their own state selects: at the
        first the fleet still held real headroom, by the second every account
        is spent, which is the regime the reset axis exists for. Base never
        makes that transition because it refuses the outbound leg too.

        Two candidate constraints were measured and BOTH changed the count by
        zero — requiring the fallback's candidate to be spent, and taking
        `max`/`min` over the pair in `_recovery_is_useful`. The transition is
        in the data, not in the gates, so neither is shipped.

        What must hold is that a walk SETTLES. This one does: two moves in 24
        ticks, then stationary.
        """
        seen = []
        pct = {"1": 96.0, "2": 92.0}          # 4.0 and 8.0 points
        for _ in range(24):
            harness.tick_with_usage({
                "1": _usage(pct["1"], self._days_out(harness, 20)),
                "2": _usage(pct["2"], self._days_out(harness, 80)),
            })
            active = harness.active_number()
            seen.append(active)
            pct[str(active)] = min(99.95, pct[str(active)] + 0.25)   # burn
            harness.clock.advance(301.0)

        moves = [n for i, n in enumerate(seen) if i == 0 or n != seen[i - 1]]
        assert len(moves) <= 2, (
            f"move sequence {moves} — a burn walk that keeps moving is a flap, "
            "whatever axis each leg took"
        )
        assert seen[-4:] == [seen[-1]] * 4, (
            f"active trace {seen} — the walk never settled"
        )

    def test_the_no_return_filter_does_not_block_the_at_limit_escape(
        self, harness
    ):
        """Every sibling anti-flap gate is scoped to the proactive triggers.

        `at-limit` and `failover` skip them by design — there we are escaping a
        dead account, not optimising a return time. The no-return filter ran in
        `_tick_inner` BEFORE the trigger is consulted, so it stripped the
        candidate from those escapes too.

        Measured, 2 accounts, the active exhausted and the peer at full quota:

            base 9f35426   switches on tick 1
            here           switches=0, "no-candidates" every tick, 720 ticks

        Nothing releases it: `lastSwitchFrom` is only rewritten by a successful
        switch, and the field is what prevents the switch. On a 3-account fleet
        it also emits AllExhaustedEvent — a false claim that reaches the user
        as a macOS notification and a critical TUI row while a peer sits at 0%.
        """
        harness.engine._mutate_state(
            lambda st: st.__setitem__("lastSwitchFrom", "2")
        )
        outcome = harness.tick_with_usage({
            "1": _usage(100),      # active, exhausted -> at-limit
            "2": _usage(0),        # the account we left, now at full quota
        })
        assert outcome is TickOutcome.SWITCHED, (
            "the at-limit escape was refused because we had left that account "
            "once — the engine sits on an exhausted account with a peer at 0%"
        )
        assert harness.active_number() == 2

    def test_a_burn_walk_never_returns_to_what_it_left(self, harness):
        """The axis can flip more than once, and nothing bounded how often.

        A previous round measured A-B-A under burn and dismissed it: each leg
        IS legitimate on the axis its own state selects, and base shows 0 only
        because it refuses the outbound leg too. That reasoning holds. The
        conclusion did not — it rested on the walk settling in at most three
        moves, which is a property of the one shape that was measured.

        Measured, only the active burning at 0.5 pts/tick, both resets past
        the horizon, 24 ticks:

            pcts (96,92) resets (20h, 80h)    moves [2, 1]        settles
            pcts (92,92) resets (500h,400h)   moves [1, 2, 1, 2]  does not

        Base on both: a single move. Traced at each leg of the second shape —

            t8   1->2  headroom axis   active 4.0 / best 8.0
            t20  2->1  headroom axis   active 2.0 / best 4.0
            t22  1->2  recovery axis   active 3.0 / best 2.0

        The ratio gate is RELATIVE (`h >= active x 2`) and the spent gate
        ABSOLUTE (`active <= 3.0`), so burn walks the pair across the boundary
        repeatedly and each crossing re-opens a move. Extending to 120 ticks
        stops only because both accounts hit the 99.95 burn cap, so the fourth
        move is not a transient.

        Refusing the account we most recently left bounds it — but identity
        alone has no release, and on a 2-account fleet that is a permanent
        proactive lockout (see the sibling test). Released by asking the
        ranking, the walk is still BOUNDED: it ends, because each return has to
        clear the margin and burn makes that harder every time.

        A trace of the exact moves used to sit here. It has been re-taken three
        times and come back different every time — the walk depends on the
        release, and the release has changed in every round that quoted it. The
        assertion is on SETTLING for the same reason.

        So this asserts a BOUND, not zero returns. Zero was a property of the
        release-less filter, and that property is what made the lockout
        permanent. A walk that ends is the real requirement; the live incident
        this class documents was four moves in 35 minutes and still climbing.
        """
        pct = {"1": 92.0, "2": 92.0}
        seen = []
        # 60, not 24: the settling point moves with fleet size — a longer
        # ring walks further before it comes back — and 24 ticks caught this
        # shape mid-walk.
        for _ in range(60):
            harness.tick_with_usage({
                "1": _usage(pct["1"], self._days_out(harness, 500)),
                "2": _usage(pct["2"], self._days_out(harness, 400)),
            })
            active = harness.active_number()
            seen.append(active)
            pct[str(active)] = min(99.95, pct[str(active)] + 0.5)
            harness.clock.advance(301.0)

        moves = [n for i, n in enumerate(seen) if i == 0 or n != seen[i - 1]]
        # SETTLING is the property, not a move count. `len(moves) <= 4` was
        # true of this 2-account shape and false of every other — the bar
        # refuses only the ONE account left last, so a longer ring walks
        # further before it comes back, and every fleet size settles.
        #
        # The per-size counts that used to be quoted here did not re-measure
        # after the release changed. A count that holds for one fleet size and
        # one release reads as a bound and is neither. What the walk has to do
        # is END.
        assert len(set(seen[-8:])) == 1, (
            f"move sequence {moves} — the walk was still moving in the last "
            "eight ticks, so it does not settle at all"
        )

    def test_a_proactive_move_does_not_lock_out_the_next_one(self, harness):
        """The no-return filter has no release condition on a 2-account fleet.

        `lastSwitchFrom` is written only by a SUCCESSFUL switch, and on two
        accounts the filter removes the only candidate — so the switch that
        would rewrite it can never happen. Self-perpetuating, and persisted:
        it survives a restart and a week of wall clock.

        Reached by ONE ordinary proactive move, no seeded state. After it, the
        peer resets to full and the active keeps burning; every proactive tick
        answers "no-candidates" while a 0% account sits there. The engine
        escapes only at a hard 100%, which is the feature turned off — the
        user hits the limit they were supposed to be switched away from.

        Asserts on the SWITCH, not on the state field: the field being clear
        proves nothing about whether a move can happen, and the scoped filter
        deliberately leaves the field set on the at-limit path.
        """
        assert harness.tick_with_usage({
            "1": _usage(92, self._days_out(harness, 500)),
            "2": _usage(10, self._days_out(harness, 400)),
        }) is TickOutcome.SWITCHED
        assert harness.active_number() == 2
        harness.clock.advance(301.0)

        # The account we left is now fully reset; the new active burns on.
        outcomes = []
        for _ in range(20):
            outcomes.append(harness.tick_with_usage({
                "1": _usage(0, self._days_out(harness, 400)),
                "2": _usage(97, self._days_out(harness, 500)),
            }))
            harness.clock.advance(1801.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"20 ticks / 10h of outcomes {[o.name for o in outcomes]} — the "
            "peer was at 0% the whole time and the active burned to 97%. The "
            "one proactive move disabled proactive switching permanently."
        )

    def test_a_filtered_candidate_does_not_forge_an_all_exhausted_claim(
        self, temp_home
    ):
        """The filter runs BEFORE `truly_exhausted`, so it hides the evidence.

        A healthy peer removed from `oauth_candidates` cannot make the `all()`
        False, and the engine then claims every account is exhausted. The user
        gets a macOS notification and a critical TUI row while that peer sits
        at 0%, and `_blocked_wait_long` stretches the poll interval, so the
        recovery it is wrong about arrives slower too.

        Needs a genuinely spent THIRD account: with only the filtered peer the
        list goes empty and the tick exits at `no-candidates` first — measured,
        3 ticks of `no-candidates` and no event. The false claim needs the
        remaining candidates to be real and all spent, which is the shape a
        user on three accounts actually hits.

        Reached on the proactive path, which the at-limit escape test does not
        cover. Asserts on the EVENT the user sees, not on the candidate list.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
            "3": _usage(100, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(301.0)

        h.events.clear()
        h.tick_with_usage({
            "1": _usage(0, self._days_out(h, 400)),    # the one we left: FULL
            "2": _usage(97, self._days_out(h, 500)),
            "3": _usage(100, self._days_out(h, 300)),  # genuinely spent
        })
        assert not any(isinstance(e, AllExhaustedEvent) for e in h.events), (
            f"events {[type(e).__name__ for e in h.events]} — account 1 is at "
            "0%; the fleet is not exhausted, the filter hid the one peer that "
            "disproves it"
        )

    def test_the_bar_does_not_hide_the_account_from_the_census(
        self, temp_home
    ):
        """Barring a candidate must not make it cease to EXIST.

        Removing it from `oauth_candidates` fed eight consumers a list with a
        healthy account missing. `truly_exhausted` is the loudest: measured,
        peer 1 holding 15 points while the engine emitted AllExhaustedEvent —
        a macOS notification and a critical TUI row — because `all()` over the
        shortened list was vacuously true.

        DRIVES THE BARRED BRANCH, which is the part the previous tests missed.
        Peer 15 pts against active 10 pts does NOT satisfy `left >= active x
        2`, so the release does not fire and the bar is genuinely in effect.
        Both earlier tests used a 0% peer against a 97% active, where the
        release always fired — measured, disabling the filter outright left
        the whole suite green.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
            "3": _usage(100, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(301.0)

        h.events.clear()
        h.tick_with_usage({
            "1": _usage(85, self._days_out(h, 400)),   # left; 15 pts, BARRED
            "2": _usage(90, self._days_out(h, 500)),   # active; 10 pts
            "3": _usage(100, self._days_out(h, 300)),  # genuinely spent
        })
        assert not any(isinstance(e, AllExhaustedEvent) for e in h.events), (
            f"events {[type(e).__name__ for e in h.events]} — account 1 holds "
            "15 points; barring it from the CHOICE must not erase it from the "
            "fleet"
        )

    def test_the_bar_lifts_when_it_would_leave_nothing(self, temp_home):
        """Identity has no release of its own, and the ratio cannot cover it.

        `lastSwitchFrom` is rewritten only by a successful switch — the one
        the bar prevents — so on two accounts it was permanent. The ratio
        release does not reach it either: `left >= active x 2` is unsatisfiable
        for any active headroom above 50, and consume-first fires exactly
        there. Measured before this: active 70 pts against a peer at 100 pts,
        20 ticks answering below-threshold, still locked after seven days.

        A bar that leaves the engine nothing to choose is a stall, not
        anti-flap.
        """
        h = EngineHarness(temp_home, strategy="consume-first")
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage7(95, 95, self._days_out(h, 500)),
            "2": _usage7(5, 5, self._days_out(h, 400)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(301.0)

        outcomes = []
        for _ in range(20):
            outcomes.append(h.tick_with_usage({
                # left; weekly window resets SOONEST, which is what
                # consume-first ranks on
                "1": _usage7(0, 0, self._days_out(h, 10)),
                "2": _usage7(30, 30, self._days_out(h, 500)),   # active, 70 pts
            }))
            h.clock.advance(1801.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"20 ticks of {[o.name for o in outcomes]} — the only peer was "
            "barred with no way to lift it, so consume-first is off for good"
        )

    def test_the_bar_never_applies_to_an_escape(self, harness):
        """`at-limit` and `failover` skip every anti-flap gate by design.

        There we are escaping a dead or unreadable active, not optimising a
        return time — and the account we left may be the only place to go.
        Measured: dropping the trigger check left the full suite green, so
        nothing pinned it. The at-limit half is defended for the wrong reason
        (at-limit implies `active_headroom <= 0`, so the ratio release fires
        anyway); failover has no such accident, because an unreadable active
        gives `active_headroom is None` and the release cannot fire.

        Asserts on the BAR, not on a tick outcome: the ratio release makes the
        at-limit case pass either way, which is what hid this.
        """
        state = {"lastSwitchFrom": 1}
        # 15 pts against an active on 10: does NOT clear `left >= active x 2`,
        # so the ratio release cannot fire and only the trigger check can
        # answer. A peer far ahead would pass for the wrong reason.
        headroom = {"1": 15.0, "2": 10.0, "3": 1.0}
        for trigger in ("at-limit", "failover"):
            for active in (10.0, None):
                assert harness.engine._no_return_account(
                    trigger, state, headroom, active, ["1", "3"]
                ) is None, (
                    f"trigger={trigger} active_headroom={active} barred the "
                    "account we left; an escape must reach every candidate"
                )
        # The control: the SAME state bars on a proactive tick.
        assert harness.engine._no_return_account(
            "proactive", state, headroom, 10.0, ["1", "3"]
        ) == "1", "premise: these inputs are barred when the trigger allows it"

    def test_the_bar_actually_removes_the_account_from_the_ranking(
        self, harness
    ):
        """The bar has to BAR something, and nothing pinned that.

        Measured: disabling it outright — `if num == no_return` -> `if False`
        — left the whole suite green, this class included. Both sibling tests
        assert what happens when the bar is LIFTED (the census stays intact,
        the lockout ends), so neither notices when it never engages.

        Drives `_rank_candidates` directly. Through `tick()` the cooldown and
        the hysteresis gates decide these inputs first, so the same pair moves
        identically with the bar on and off — measured across 12 peer/active
        combinations, every one identical. The ranking is the only place the
        bar's effect is observable in isolation.
        """
        from claude_swap.settings import AutoSwitchSettings

        args = dict(
            trigger="proactive",
            consume_first=False,
            oauth_candidates=["1", "3"],
            usage={"1": _usage(40), "2": _usage(96), "3": _usage(99)},
            headroom={"1": 60.0, "2": 4.0, "3": 1.0},
            current="2",
            active_headroom=4.0,
            settings=AutoSwitchSettings(),
            now=harness.clock.now,
        )
        unbarred, _, _ = harness.engine._rank_candidates(no_return=None, **args)
        barred, _, _ = harness.engine._rank_candidates(no_return="1", **args)

        assert list(unbarred) == ["1"], (
            f"premise: account 1 holds 60 points against an active on 4 and "
            f"is the pick when nothing bars it — got {list(unbarred)}"
        )
        assert list(barred) == [], (
            f"the bar did not remove account 1 from the ranking: {list(barred)}"
        )

    def test_the_bar_lifts_when_the_only_alternative_cannot_be_chosen(
        self, temp_home
    ):
        """Existing is not the same as being an alternative.

        The leaves-nothing release asked whether any OTHER account exists. A
        third account that exists but can never qualify — at its limit, or with
        unreadable headroom — answered yes while offering the ranking nothing,
        so the release never fired and the n=2 stall simply moved to n>=3.

        Measured before this, one ordinary proactive move and no seeded state:
        30 ticks / 30h all BLOCKED with the active on 2 points and the barred
        peer on 3, while the same fleet with the bar cleared switches on the
        first tick.

        THIRD ACCOUNT AT ITS LIMIT on purpose: with a healthy third the bar is
        correct and the sibling tests cover it. The defect needs an alternative
        that the ranking loop would skip anyway.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
            "3": _usage(100, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(301.0)

        outcomes = []
        for _ in range(30):
            outcomes.append(h.tick_with_usage({
                "1": _usage(97, self._days_out(h, 10)),    # barred, 3 pts
                "2": _usage(98, self._days_out(h, 500)),   # active, 2 pts
                "3": _usage(100, self._days_out(h, 300)),  # exists, spent
            }))
            h.clock.advance(3601.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"30 ticks of {[o.name for o in outcomes[:6]]}… — the only "
            "choosable peer was barred and the third account is at its limit, "
            "so the bar left the engine nothing"
        )

    def test_the_bar_lifts_for_an_alternative_the_ranking_would_reject(
        self, temp_home
    ):
        """Not-at-its-limit is not the same as rankable.

        The predicate above was `(headroom.get(n) or 0.0) > 0.0` — "has any
        points left". That is only the FIRST of the gates a candidate must
        clear: past the horizon it also needs `h >= active x HORIZON_HEADROOM_
        RATIO`, or the spent fallback's `h >= active` with a meaningfully
        sooner reset. A third account holding ONE point clears `> 0.0` and
        clears nothing else, so the release stayed shut and the n>=3 stall the
        release above was written for came straight back one point up.

        Measured, one ordinary proactive move and no seeded state: barred peer
        3.5 pts / back in 10h, active 2 pts / 500h out, third 1 pt — 30 ticks
        all BLOCKED. The control below is the same fleet with `lastSwitchFrom`
        popped and switches on the first tick, so the bar is the cause.

        This is the third time this release has been fixed one step short of
        the gate that actually decides (present -> not-at-limit -> rankable),
        which is why the fix is no longer a predicate that PREDICTS the
        ranking: `_tick_inner` now asks the ranking itself and re-ranks unbarred
        when the bar empties the list.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
            "3": _usage(99, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(301.0)

        outcomes = []
        for _ in range(30):
            outcomes.append(h.tick_with_usage({
                "1": _usage(96.5, self._days_out(h, 10)),   # barred, 3.5 pts
                "2": _usage(98, self._days_out(h, 500)),    # active, 2 pts
                "3": _usage(99, self._days_out(h, 300)),    # 1 pt: > 0, unrankable
            }))
            h.clock.advance(3601.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"30 ticks of {[o.name for o in outcomes[:6]]}… — the third "
            "account holds one point, which passes `> 0.0` and no ranking "
            "gate, so the bar left the engine nothing"
        )

    def test_an_unreadable_barred_account_does_not_crash_the_tick(
        self, harness
    ):
        """`headroom.get(barred)` is None when that slot's usage is unreadable.

        The ratio release compares it against `active_headroom * RATIO`, so
        without the None check the tick raises `TypeError: '>=' not supported
        between instances of 'NoneType' and 'float'` — inside `_tick_inner`,
        on an ordinary proactive tick, whenever the account we just left has
        no readable usage. Measured: dropping `left_headroom is not None` left
        the whole suite green, so nothing pinned it.

        Asserts the CALL returns rather than the tick outcome: which account
        wins is the ranking's business, and a crash is the defect.
        """
        state = {"lastSwitchFrom": 1}
        headroom = {"2": 10.0, "3": 40.0}       # slot 1 unreadable — absent
        assert harness.engine._no_return_account(
            "proactive", state, headroom, 10.0, ["1", "3"]
        ) == "1", (
            "an unreadable barred account must still bar — unknown headroom "
            "is not evidence it beats us"
        )

    def test_the_bar_lifts_for_a_peer_returning_inside_the_horizon(
        self, temp_home
    ):
        """The release had no condition on the RECOVERY axis at all.

        Its only release was `left >= active x HORIZON_HEADROOM_RATIO`, a pure
        headroom test — while `_recovery_is_useful` deliberately ranks by RESET
        when a candidate returns inside the horizon, and this module's own
        docstring names that case as the one the horizon exists to preserve:
        a weekly-bound active days out against a peer back in minutes.

        A barred peer in exactly that state was refused, because 4 points
        against an active on 3 misses `4 >= 3 x 2`. Measured on the predicate
        form: barred peer back in 1h, active 200h out — 10 ticks all BLOCKED.

        Asking the ranking covers it without a third predicate: barring the
        only account the reset axis would pick empties the list, so the retry
        opens. That is the point of not predicting — the release now follows
        every axis the ranking has, including ones added later.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(301.0)

        outcomes = []
        for _ in range(10):
            outcomes.append(h.tick_with_usage({
                "1": _usage(96, self._at(h, 3600)),      # barred, back in 1h
                "2": _usage(97, self._days_out(h, 200)),  # active, 200h out
            }))
            h.clock.advance(1801.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"{[o.name for o in outcomes]} — the bar refused the only peer "
            "returning inside the horizon, so the engine holds an account 200h "
            "out while a peer is back in one"
        )

    def test_the_same_fleet_moves_with_the_bar_cleared(self, temp_home):
        """The control for the test above: identical state, no bar.

        Without this, a stall could be the fleet's own numbers rather than the
        bar, and the assertion above would be measuring nothing. Same seeds,
        same usage, same clock — only `lastSwitchFrom` is popped.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
            "3": _usage(99, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(301.0)
        h.engine._mutate_state(lambda st: st.pop("lastSwitchFrom", None))

        assert h.tick_with_usage({
            "1": _usage(96.5, self._days_out(h, 10)),
            "2": _usage(98, self._days_out(h, 500)),
            "3": _usage(99, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED, (
            "the control blocked too — the stall above is the fleet's numbers, "
            "not the bar, and that assertion is measuring nothing"
        )

    def test_the_bar_reaches_the_ranking_through_tick(self, temp_home):
        """The bar's production WIRING, which nothing pinned.

        `_no_return_account` is computed inside `_rank` and threaded into
        `_rank_candidates`. Measured: replacing that computation with
        `no_return = None` — the whole feature off in production — left the
        full suite green. The only test of the bar's effect drives
        `_rank_candidates` directly and passes `no_return` by hand, so the unit
        was pinned and the integration was not: any refactor that drops the
        kwarg reverts the anti-flap bound silently.

        Asserts a DIFFERENT DESTINATION, not a block: the leaves-nothing
        release is now answered by the ranking itself, so a bar that empties
        the list re-ranks unbarred and the engine moves anyway. A fleet where
        the bar blocks therefore proves nothing about the wiring — the only
        observable left is the engine landing somewhere else.

        ON THE RECOVERY AXIS, which is the only axis where the bar can change
        an answer at all. Past the horizon the release (`left >= active x
        RATIO` -> not barred) and the ranking gate (`h >= active x RATIO` ->
        qualifies) are the SAME inequality, so anything the bar could remove
        the loop had already dropped. Inside the horizon the ranking sorts by
        reset time instead, the two stop agreeing, and the bar bites. Both
        peers are back within the hour here, which is what puts the tick on
        that axis.

        ONE fleet, ticked twice: `temp_home` is a single home and a second
        `EngineHarness` over it inherits the first run's roster, so the control
        comes from popping `lastSwitchFrom`, not from a fresh box.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)
        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
            "3": _usage(50, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.clock.advance(301.0)
        assert h.engine._read_state().get("lastSwitchFrom") is not None, (
            "premise: the move recorded what it left"
        )

        second = {
            "1": _usage(99, self._at(h, 1800)),      # left; back in 30 min
            "2": _usage(99, self._at(h, 7200)),      # active, spent, back in 2h
            "3": _usage(99, self._at(h, 3600)),      # back in 1h
        }
        assert h.tick_with_usage(second) is TickOutcome.SWITCHED
        assert h.active_number() == 3, (
            "the bar never reached the ranking through tick() — the engine "
            "went back to the account it had just left, which returns sooner "
            "than the peer it should have taken"
        )

        # Control: same numbers, the bar removed — the soonest return wins.
        # The clock advances past the post-switch cooldown, and `second`'s
        # resets are relative to the ORIGINAL now, so both peers are still
        # ahead of the active by the same margins.
        h.make_live("b@example.com", 2)
        h.clock.advance(301.0)
        h.engine._mutate_state(lambda st: st.pop("lastSwitchFrom", None))
        assert h.tick_with_usage(second) is TickOutcome.SWITCHED
        assert h.active_number() == 1, (
            "premise: unbarred, the account we left returns soonest and IS the "
            "pick — without this the assertion above would pass on a fleet "
            "where 3 wins for its own reasons"
        )

    def test_the_release_needs_the_barred_account_to_have_improved(
        self, temp_home
    ):
        """An empty barred ranking is a reason to ASK, not a reason to release.

        On two accounts the barred ranking is ALWAYS empty — barring the only
        candidate necessarily empties the list — so a release keyed on
        emptiness alone is a no-op at n=2, which is the fleet size the flap was
        reported on. Measured on the emptiness-only release, sweeping active x
        barred headroom x both reset shapes through `_rank_candidates(
        no_return="1", oauth_candidates=["1"])`:

            n=2 barred-rank EMPTY=320 NONEMPTY=0

        So the retry fired every time and the bar never applied. The cited flap
        reproduced unchanged: pcts 92/92, resets 500h/400h, 60 ticks gave
        `[1, 2, 1, 2]` with the bar ON and `[1, 2, 1, 2]` with `lastSwitchFrom`
        popped every tick — identical, and worse than base's single move.

        WHAT SEPARATES THE TWO STATES is not the ranking, which only sees the
        present. It is whether the barred account is a different proposition
        from the one we left. At each leg of that walk it was not:

            t8   1->2   left 1 holding 4.0 pts, 500h out
            t20  2->1   account 1 holds 4.0 pts, 500h out   <- nothing changed
            t22  1->2   account 2 holds 2.0 pts, 400h out   <- nothing changed

        Every return won because the ACTIVE burned down, never because the
        target recovered. That is the flap, exactly.

        Both legs of the test below are the flap shape: the barred account is
        no better than we left it on either axis. The bar must hold even
        though the ranking is empty and the tick therefore does nothing.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(96, self._days_out(h, 500)),   # 4 pts
            "2": _usage(92, self._days_out(h, 400)),   # 8 pts
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.clock.advance(3612.0)

        outcomes = []
        for _ in range(10):
            outcomes.append(h.tick_with_usage({
                # unchanged since we left it: same headroom, same reset
                "1": _usage(96, self._days_out(h, 500)),
                "2": _usage(98, self._days_out(h, 400)),   # active, burnt down
            }))
            h.clock.advance(301.0)

        assert TickOutcome.SWITCHED not in outcomes, (
            f"{[o.name for o in outcomes]} — the engine went back to an "
            "account that is exactly as we left it. The ranking flipped "
            "because the active burned, not because the target recovered; "
            "that is the flap this bar exists for."
        )

    def test_a_departure_at_full_quota_is_immediately_eligible(self, temp_home):
        """`left_headroom == 100.0` must not be a permanent lockout.

        `h >= left_headroom + SPENT_HEADROOM_PCT` is `h >= 103.0` when the
        departure was recorded at a full 100.0 points — unsatisfiable forever,
        because `oauth.account_headroom` caps `h` at 100.0
        (`100 - max(pct)`, and pct cannot go negative). consume-first departs
        BELOW the threshold, so this is the routine case, not a corner one: a
        fresh/full account handed off to a sooner-resetting peer records
        exactly this.

        The account below holds the SAME 100.0 points at every check (never
        spent anything) and the SAME resets_at (no recovery-axis movement
        either) — the only way this switches is if a departure at the cap is
        treated as needing no recovery on the headroom axis.
        """
        h = EngineHarness(temp_home, strategy="consume-first", threshold=90.0)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage7(0.0, 0.0, self._days_out(h, 500)),
            "2": _usage7(0.0, 0.0, self._days_out(h, 100)),
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        assert h.engine._read_state().get("leftHeadroom") == 100.0, (
            "premise: consume-first recorded a full-quota departure"
        )
        h.clock.advance(301.0)

        outcomes = []
        for _ in range(10):
            outcomes.append(h.tick_with_usage({
                # unchanged since departure on BOTH axes
                "1": _usage7(0.0, 0.0, self._days_out(h, 500)),
                "2": _usage7(90.0, 0.0, self._days_out(h, 100)),
            }))
            h.clock.advance(301.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"{[o.name for o in outcomes]} — account 1 never dropped below a "
            "full 100.0 points and is the only peer; refusing it is the "
            "unsatisfiable-above-97 lockout, not anti-flap"
        )

    def test_a_reset_that_crept_nearer_is_not_a_recovery(self, temp_home):
        """The recovery leg carries `RECOVERY_HYSTERESIS_S`, and it must.

        Without a margin (`< was - 0.0`) any reset that moved a second nearer
        counts as the barred account "recovering", and a `resets_at` that
        drifts — a refetch landing a slightly different estimate, or simply a
        nearer window starting to bind — hands the flap a release for free.
        That is the same shape as the ratio gate before it was gated: a
        threshold burn crosses on its own.

        Measured with the margin removed: the walk below returns to account 1
        because its binding reset reads 60s nearer than the value recorded at
        departure, while its headroom is unchanged.

        `RECOVERY_HYSTERESIS_S` is the margin the recovery AXIS already ranks
        by one gate later, so the release and the ranking agree about what
        "meaningfully sooner" means rather than being two numbers to reason
        about separately.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        depart = self._at(h, 500 * 3600)
        assert h.tick_with_usage({
            "1": _usage(96, depart),                    # 4 pts
            "2": _usage(92, self._days_out(h, 400)),    # 8 pts
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.clock.advance(3612.0)

        outcomes = []
        for _ in range(10):
            outcomes.append(h.tick_with_usage({
                # same headroom as at departure; the reset crept 60s nearer,
                # which is well inside RECOVERY_HYSTERESIS_S
                "1": _usage(96, self._at(h, 500 * 3600 - 3612 - 60)),
                "2": _usage(98, self._days_out(h, 400)),
            }))
            h.clock.advance(301.0)

        assert TickOutcome.SWITCHED not in outcomes, (
            f"{[o.name for o in outcomes]} — a reset one minute nearer is not "
            "the barred account recovering; without the margin any drift in "
            "`resets_at` releases the bar"
        )

    def test_an_unschedulable_account_that_gained_a_reset_has_recovered(
        self, temp_home
    ):
        """`inf` is the right departure value for an unknown reset, not zero.

        `_binding_recovery_ts` returns `inf` for a binding window with no
        usable `resets_at` — an account nobody can schedule around — and
        `_perform` stores that as JSON `null`. Reading it back as `0.0` makes
        the recovery leg unsatisfiable, because no real timestamp is below
        `0 - RECOVERY_HYSTERESIS_S`, so an account that gained a reset while we
        were away is refused forever on that axis.

        That IS an improvement, and it is one the headroom leg cannot see: the
        account below holds the same 4 points it had at departure, so only the
        reset changed. Measured with the default flipped to `0.0`: 10 ticks
        BLOCKED with the peer back in an hour.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(96),                            # 4 pts, NO reset known
            "2": _usage(92, self._days_out(h, 400)),
        }) is TickOutcome.SWITCHED
        assert h.engine._read_state().get("leftRecoveryAt") is None, (
            "premise: the departure reset was unknown and stored as null"
        )
        h.clock.advance(3612.0)

        outcomes = []
        for _ in range(10):
            outcomes.append(h.tick_with_usage({
                # SAME 4 points; only the reset is now known, and it is near
                "1": _usage(96, self._at(h, 3600)),
                "2": _usage(98, self._days_out(h, 400)),
            }))
            h.clock.advance(301.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"{[o.name for o in outcomes]} — the barred account went from "
            "unschedulable to back-in-an-hour, which the headroom leg cannot "
            "see; reading the stored null as 0.0 makes that unreachable"
        )

    def test_a_switch_that_recorded_no_snapshot_still_releases(
        self, temp_home
    ):
        """No departure snapshot means release, and that direction is chosen.

        State written before `leftHeadroom`/`leftRecoveryAt` existed — an
        upgrade in place, with the file persisted across restarts — names a
        barred account and carries no evidence about it either way. The two
        failure modes are not symmetric: barring on absent evidence is the
        permanent proactive lockout this branch has already fixed twice, and it
        survives a restart and a week of wall clock, while releasing costs at
        most one extra move that the next switch then records properly.

        Measured with the default flipped to `return False`: the shape below
        answers BLOCKED for 20 ticks with a peer at full quota.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        # A pre-upgrade record: the bar is named, the snapshot is not there.
        h.engine._mutate_state(lambda st: st.update({"lastSwitchFrom": "2"}))
        h.engine._mutate_state(lambda st: st.pop("leftHeadroom", None))
        h.engine._mutate_state(lambda st: st.pop("leftRecoveryAt", None))
        assert "leftHeadroom" not in h.engine._read_state(), (
            "premise: the state carries no departure snapshot"
        )

        outcomes = []
        for _ in range(20):
            outcomes.append(h.tick_with_usage({
                "1": _usage(97, self._days_out(h, 500)),   # active, 3 pts
                "2": _usage(0, self._days_out(h, 400)),    # barred, FULL quota
            }))
            h.clock.advance(1801.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"20 ticks of {[o.name for o in outcomes[:6]]}… — state written "
            "before the snapshot field existed barred the only peer forever, "
            "which is the persisted lockout, not anti-flap"
        )

    def test_the_release_fires_when_the_barred_account_recovered(
        self, temp_home
    ):
        """The control: same fleet, same bar, the barred account IS better.

        Without this the assertion above would pass on a bar that never
        releases at all, which is the permanent 2-account lockout this branch
        already fixed twice. Only account 1's numbers differ — it reset to full
        quota — and that must move the engine on the recovery axis the
        emptiness retry was reaching for.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(96, self._days_out(h, 500)),
            "2": _usage(92, self._days_out(h, 400)),
        }) is TickOutcome.SWITCHED
        h.clock.advance(3612.0)

        outcomes = []
        for _ in range(10):
            outcomes.append(h.tick_with_usage({
                "1": _usage(0, self._days_out(h, 500)),    # reset to FULL
                "2": _usage(98, self._days_out(h, 400)),
            }))
            h.clock.advance(301.0)

        assert TickOutcome.SWITCHED in outcomes, (
            f"{[o.name for o in outcomes]} — account 1 came back to full "
            "quota and is the only peer; refusing it is the permanent "
            "2-account lockout, not anti-flap"
        )

    def test_the_ratio_release_changes_where_the_engine_lands(self, temp_home):
        """`left >= active x HORIZON_HEADROOM_RATIO` — worth 50 points, unpinned.

        Measured: replacing the whole condition with `False` left the full
        suite green. It is not equivalent. `test_the_bar_never_applies_to_an_
        escape` deliberately uses 15 against 10 so the ratio CANNOT fire, and
        every other bar test releases through the emptiness path instead, so
        nothing observed the release doing its job.

        End-to-end after a 1->2 move, active 2 on 10 pts, the barred 1 on 80,
        a third peer on 30:

            release ON   -> SWITCHED to account 1 (80 pts)
            release OFF  -> SWITCHED to account 3 (30 pts)

        Asserts the DESTINATION: the tick switches either way, so an outcome
        assertion would pass with the release gone.
        """
        h = EngineHarness(temp_home)
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage(92, self._days_out(h, 500)),
            "2": _usage(10, self._days_out(h, 400)),
            "3": _usage(70, self._days_out(h, 300)),
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.clock.advance(301.0)

        assert h.tick_with_usage({
            "1": _usage(20, self._days_out(h, 500)),   # barred, 80 pts
            "2": _usage(90, self._days_out(h, 400)),   # active, 10 pts
            "3": _usage(70, self._days_out(h, 300)),   # peer, 30 pts
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 1, (
            f"landed on {h.active_number()} — the account we left now holds "
            "8x the active's headroom, which is a move the outbound leg would "
            "have made on its own merits, not the flip the bar refuses"
        )

    def test_the_bar_is_recomputed_on_the_phase_two_snapshot(self, temp_home):
        """Consume-first refetches, and the bar has to be re-asked.

        The two-phase commit replaces `usage`, `headroom` and `active_headroom`
        with an escalated refetch, then re-ranks — but the bar was computed
        once, before phase 1, from the STALE snapshot. `_no_return_account`'s
        ratio release consumes exactly the two values phase 2 replaces, so the
        bar is decided on data the ranking has already thrown away:

            no_return(stale: left=20, active=30) = '1'    (barred)
            no_return(fresh: left=90, active=15) = None   (released)

        Drives a real consume-first tick and swaps the snapshot underneath it:
        phase A serves the stale numbers, the phase-2 escalation (the only
        fetch that asks for every account) serves the fresh ones. On the fresh
        numbers the account we left holds 6x the active's headroom and its
        weekly window resets soonest, so it is the pick — unless the bar is
        still answering from the stale snapshot, where it lost by well under
        the ratio and stayed barred.
        """
        h = EngineHarness(temp_home, strategy="consume-first")
        h.seed(1, "a@example.com")
        h.seed(2, "b@example.com")
        h.seed(3, "c@example.com")
        h.make_live("a@example.com", 1)

        assert h.tick_with_usage({
            "1": _usage7(20, 20, self._days_out(h, 500)),   # active, LAST
            "2": _usage7(5, 5, self._days_out(h, 10)),      # SOONEST
            "3": _usage7(5, 5, self._days_out(h, 400)),
        }) is TickOutcome.SWITCHED
        assert h.active_number() == 2
        h.clock.advance(301.0)

        stale = {
            "1": _usage7(80, 80, self._days_out(h, 10)),    # left; 20 pts
            "2": _usage7(70, 70, self._days_out(h, 500)),   # active; 30 pts
            "3": _usage7(60, 60, self._days_out(h, 400)),   # 40 pts
        }
        fresh = {
            "1": _usage7(10, 10, self._days_out(h, 10)),    # left; 90 pts NOW
            "2": _usage7(85, 85, self._days_out(h, 500)),   # active; 15 pts
            "3": _usage7(60, 60, self._days_out(h, 400)),   # 40 pts
        }

        def _serve(fetch=frozenset(), **kw):
            # The phase-2 escalation is the only call that asks for the whole
            # fleet; everything before it is the stale baseline.
            snap = fresh if len(fetch) >= 3 else stale
            return {n: _entry_for(v, h.clock.now) for n, v in snap.items()}

        with patch.object(
            h.switcher, "usage_entries_by_account", side_effect=_serve
        ):
            assert h.engine.tick() is TickOutcome.SWITCHED
        assert h.active_number() == 1, (
            f"landed on {h.active_number()} — on the FRESH snapshot the "
            "account we left holds 6x the active's headroom and its weekly "
            "window resets soonest, so the release fires; the bar was still "
            "answering from the stale snapshot the ranking had replaced"
        )

    def test_the_fallback_never_outranks_a_real_qualifier(self, harness):
        """It runs only when nothing else qualifies, and the key is why.

        The fallback's key is tier 0; every ordinary candidate is tier 1. So
        if a fallback entry ever reached the same list as a qualifier it would
        sort FIRST regardless of headroom. `qualifying or fallback` is the
        only thing preventing that, and nothing tested it — measured,
        replacing it with `qualifying + fallback` left the suite green while
        flipping this scenario 0/3 -> 3/3 in the fallback's favour.

        Active is spent (3 pts). One peer qualifies outright on headroom; one
        margin-failure peer resets sooner. The qualifier must win.
        """
        outcome = harness.tick_with_usage({
            "1": _usage(97, self._days_out(harness, 500)),   # active, 3 left
            "2": _usage(94, self._days_out(harness, 400)),   # 6 left: qualifies
            "3": _usage(96.4, self._days_out(harness, 100)), # 3.6 left, sooner
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            f"landed on {harness.active_number()} — the tier-0 fallback key "
            "outranked a candidate with twice the headroom"
        )

    def test_the_fallback_ranks_by_reset_not_by_headroom(self, harness):
        """`(0, recovery_ts, -h)` — the reset leads, and that is deliberate.

        Every account in the fallback is spent, and below SPENT_HEADROOM_PCT a
        headroom edge is under two poll intervals of work. The only real
        question is which account can work again first, which is the same
        judgement `_recovery_is_useful` makes one gate earlier.

        Nothing tested it: swapping to `(0, -h, recovery_ts)` left the suite
        green. Exhaustive sweep over 42336 three-account shapes, 558 change
        answer, all this shape —

            active 2.0 pts / 300h
            acct 2  2.0 pts /  10h   (spent, back soonest)
            acct 3  3.1 pts /  50h   (a point more, back 40h later)

            reset key      -> 2 first
            headroom key   -> 3 first

        Taking acct 3 buys 1.1 points, worth minutes, at the cost of 40 hours
        of waiting.
        """
        outcome = harness.tick_with_usage({
            "1": _usage(98, self._days_out(harness, 300)),    # active, 2 left
            "2": _usage(98, self._days_out(harness, 10)),     # 2 left, soonest
            "3": _usage(96.9, self._days_out(harness, 50)),   # 3.1 left, later
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            f"landed on {harness.active_number()} — took a point of spent "
            "headroom over a reset 40 hours sooner"
        )

    def test_a_materially_better_peer_still_wins(self, harness):
        """The escape must survive: 2 points left against 10 is a real move."""
        outcome = harness.tick_with_usage({
            "1": _usage(98, self._days_out(harness, 109)),   # active, 2 left
            "2": _usage(90, self._days_out(harness, 80)),    # 10 left — 5x
            "3": _usage(99, self._days_out(harness, 50)),
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2

    def test_a_minutes_away_reset_is_unaffected(self, harness):
        """Inside the horizon the recovery axis still decides, ratio or not."""
        outcome = harness.tick_with_usage({
            "1": _usage(91, self._at(harness, 7200)),
            "2": _usage(94, self._at(harness, 1800)),
            "3": _usage(98, self._at(harness, 480)),         # back in 8 min
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3


class TestAllSpentGoesToTheSoonestReset:
    """When every account is spent, sit where the quota comes back first.

    Headroom decides while there is headroom worth comparing. Once everyone is
    down to a point or two, headroom says nothing — a one-point edge is under
    ten minutes of work at the burn rates measured on 2026-07-30 — and the only
    thing that still matters is who returns first, so the reset finds us
    already on it.

    The horizon rule alone got this wrong: past four hours it always ranked by
    headroom, so three accounts at 99% had no qualifying candidate and the
    engine parked on whichever one it happened to be on — including the one
    resetting LAST, 109h out against a peer 50h out.
    """

    def _at(self, harness, seconds: float) -> str:
        from datetime import datetime, timezone

        return (
            datetime.fromtimestamp(harness.clock.now + seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def test_all_spent_moves_to_the_soonest_reset(self, harness):
        """The reported shape: 99/99/99, days out, active resets last."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 109 * 3600)),  # active, LAST
            "2": _usage(99, self._at(harness, 80 * 3600)),
            "3": _usage(99, self._at(harness, 50 * 3600)),   # SOONEST
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 3, (
            "parked on the account that returns last while a peer comes back "
            "59h sooner"
        )

    def test_already_on_the_soonest_stays_put(self, harness):
        """No move when we are already where the quota returns first."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 50 * 3600)),   # active, SOONEST
            "2": _usage(99, self._at(harness, 80 * 3600)),
            "3": _usage(99, self._at(harness, 109 * 3600)),
        })
        assert outcome is not TickOutcome.SWITCHED
        assert harness.active_number() == 1

    def test_real_headroom_still_beats_a_sooner_reset(self, harness):
        """Above the spent band the headroom axis still rules: a peer holding
        ten points wins even though a spent one resets sooner."""
        outcome = harness.tick_with_usage({
            "1": _usage(98, self._at(harness, 109 * 3600)),  # active, 2 left
            "2": _usage(90, self._at(harness, 80 * 3600)),   # 10 left
            "3": _usage(99, self._at(harness, 50 * 3600)),   # 1 left, soonest
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2

    def test_a_spent_fleet_takes_the_soonest_reset_over_the_most_headroom(
        self, harness
    ):
        """`_recovery_is_useful`'s spent clause, as a whole, was unpinned.

        Its two legs are individually killed
        (`test_a_peer_with_real_headroom_still_wins_past_the_horizon`,
        `test_an_unknown_active_reset_keeps_the_headroom`), but removing the
        entire `if` — the clause's whole stated purpose — left the full suite
        green. Reachable through `tick()`:

            active 1: 0.5 pts, 300h out
            peer   2: 0.5 pts, back in 10h
            peer   3: 1.0 pt,  500h out

            ORIGINAL -> SWITCHED to 2 (the 10h account)
            MUTANT   -> SWITCHED to 3 (the 500h account)

        Every account is under SPENT_HEADROOM_PCT, which is exactly the regime
        the clause exists for: at half a point a headroom edge is minutes of
        work, so the only real question is who returns first. Without the
        clause the axis falls back to headroom, and one extra point buys a
        490-hour wait.

        Asserts the DESTINATION — both answers are a switch.
        """
        outcome = harness.tick_with_usage({
            "1": _usage(99.5, self._at(harness, 300 * 3600)),  # active, 0.5 pt
            "2": _usage(99.5, self._at(harness, 10 * 3600)),   # 0.5 pt, SOON
            "3": _usage(99.0, self._at(harness, 500 * 3600)),  # 1.0 pt, LAST
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            f"landed on {harness.active_number()} — every account is spent, "
            "so half a point of extra headroom bought a 490-hour wait over an "
            "account back in ten"
        )

    def test_the_flap_guard_survives_in_the_spent_band(self, harness):
        """Ranking by reset must not reintroduce ping-pong: an account whose
        reset is barely sooner does not qualify."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 50 * 3600)),        # active
            "2": _usage(99, self._at(harness, 50 * 3600 - 60)),   # 60s sooner
            "3": _usage(99, self._at(harness, 80 * 3600)),
        })
        assert outcome is not TickOutcome.SWITCHED


class TestEscapeBeforeTheLimitLands:
    """At the brink the ordinary proactive path already leaves — verified.

    I assumed ``at-limit`` firing only at exactly 0% meant an account rode to
    100% before escaping, and set out to move the trigger a point earlier.
    Measuring it refuted that: at 99% with a peer that has real headroom, the
    engine switches on the ORDINARY proactive path, because 99% is above the
    threshold and the peer clears the hysteresis margin easily.

    What actually happened in the 18:50 observation that prompted this: the
    only peers were 99% (one point) and 100% (never a target), so there was
    nowhere better and holding was correct. The spent check already covers that
    case by ranking on the soonest reset.

    Kept as a regression pin: moving the at-limit trigger earlier looks
    appealing and is wrong — it hijacks the recovery ranking (#202) and the
    spent-band ranking, both of which belong to `proactive`.
    """

    def _at(self, harness, seconds: float) -> str:
        from datetime import datetime, timezone

        return (
            datetime.fromtimestamp(harness.clock.now + seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def test_at_99_the_proactive_path_already_escapes(self, harness):
        """No special trigger needed: 99% is over the threshold and a healthy
        peer clears the margin."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 109 * 3600)),  # active, 1 left
            "2": _usage(70, self._at(harness, 80 * 3600)),   # 30 left
            "3": _usage(100, self._at(harness, 50 * 3600)),
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2
        sw = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert sw.trigger == "proactive", (
            "at-limit must stay bound to headroom <= 0: it skips the recovery "
            "and spent-band rankings that proactive owns"
        )

    def test_at_99_with_only_spent_peers_it_holds(self, harness):
        """The 18:50 shape: nowhere better, so staying is right. The spent-band
        rule decides where to sit, not an early escape."""
        outcome = harness.tick_with_usage({
            "1": _usage(99, self._at(harness, 109 * 3600)),
            "2": _usage(100, self._at(harness, 80 * 3600)),
            "3": _usage(100, self._at(harness, 50 * 3600)),
        })
        assert outcome is not TickOutcome.SWITCHED

    def test_below_the_brink_the_ordinary_rules_still_decide(self, harness):
        """A comfortable account is untouched: the hysteresis margin applies."""
        outcome = harness.tick_with_usage({
            "1": _usage(50, self._at(harness, 109 * 3600)),
            "2": _usage(45, self._at(harness, 80 * 3600)),
            "3": _usage(40, self._at(harness, 50 * 3600)),
        })
        assert outcome is not TickOutcome.SWITCHED


class TestReviewFindings202:
    """Three defects found reviewing #202, each reproduced before fixing.

    All three shared a cause worth naming: the code was written against the
    interval I happened to run (360s) and the account shapes I happened to
    test, not against the configurable range or the trigger matrix.
    """

    def _at(self, harness, seconds: float) -> str:
        from datetime import datetime, timezone

        return (
            datetime.fromtimestamp(harness.clock.now + seconds, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )

    def test_a_short_interval_is_never_lengthened(self, harness):
        """The default is 60s and the floor is 15s, not the 360s I developed
        against. max(min(delay, due_in), URGENT) RAISES a delay already below
        URGENT, so a 15s interval slept 60s — the exact opposite of the
        'only ever shortens' invariant this function claims."""
        num = harness.engine.switcher.current_account_number()
        real = harness.engine.switcher.usage_entries_by_account

        def patched(fetch=frozenset(), **kw):
            entries = dict(real(fetch=fetch, **kw))
            entries[num] = replace(entries[num], next_poll_at=harness.clock() + 5.0)
            return entries

        harness.engine.switcher.usage_entries_by_account = patched
        harness.engine.settings = replace(
            harness.engine.settings, interval_seconds=15.0
        )
        delay = harness.engine._next_delay(TickOutcome.NO_ACTION)
        assert delay <= 15.0 * 1.1, f"a 15s interval slept {delay:.1f}s"

    def test_recovery_reads_the_binding_windows_reset(self, harness):
        """Filtering unusable resets BEFORE taking the max let a lower window
        answer for the account: 7d at 95% with no reset and 5h at 40% resetting
        in an hour reported 'back in an hour', which is not what binds."""
        from claude_swap.autoswitch import _binding_recovery_ts

        now = harness.clock()
        usage = {
            "five_hour": {"pct": 40.0, "resets_at": self._at(harness, 3600)},
            "seven_day": {"pct": 95.0},  # binding, and no reset we can use
        }
        assert _binding_recovery_ts(usage, (), now) == float("inf")

    def test_at_limit_still_ranks_by_headroom_when_all_are_above(self, harness):
        """The gate was scoped to proactive/consume-first; the KEY was not, so
        at-limit silently re-ranked by soonest-recovery. My earlier at-limit
        test missed it because its healthy candidate made all_above False —
        this one keeps every account above the line, which is the combination
        that reaches the key."""
        outcome = harness.tick_with_usage({
            "1": _usage(100, self._at(harness, 60)),    # active, at its limit
            "2": _usage(91, self._at(harness, 86400)),  # most headroom, far reset
            "3": _usage(97, self._at(harness, 120)),    # soonest back, less room
        })
        assert outcome is TickOutcome.SWITCHED
        assert harness.active_number() == 2, (
            "at-limit must take the most headroom, not the soonest recovery"
        )
        sw = next(e for e in harness.events if isinstance(e, SwitchEvent))
        assert sw.trigger == "at-limit"


