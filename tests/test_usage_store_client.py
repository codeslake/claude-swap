"""Tests for usage requests cswap did not make: the owner proxy's entry.

Claude Code sends ``GET /api/oauth/usage`` itself, through the owner proxy,
on the account cswap has live. The store answers it from Anthropic's body as
received when that body is fresh, counts it in the hourly ``attempts``
ledger when it goes upstream, and holds it when the cap is spent or a 429
block is running.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.poll_policy import ATTEMPT_WINDOW_S, ATTEMPTS_PER_HOUR_MAX
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.usage_store import (
    CLIENT_FORWARD,
    CLIENT_HOLD,
    CLIENT_SERVE,
    HOLD_REASON_BACKOFF,
    HOLD_REASON_CAP,
    SERVE_TTL_S,
    USAGE_VARIANT_AT_WALL,
    USAGE_VARIANT_CEDAR_EMBER,
    USAGE_VARIANT_PLAIN,
    USAGE_VARIANT_UNKNOWN,
    FetchRecord,
    UsageStore,
    usage_variant,
)

IDENT = {"1": ("a@x.com", ""), "2": ("b@x.com", "org-2")}
# A plain body carries more than build_usage_result keeps (seven_day_opus,
# a weekly_scoped list): the served answer must be the body, not the trim.
BODY = {
    "five_hour": {"utilization": 25.0, "resets_at": "2099-01-01T00:00:00Z"},
    "seven_day": {"utilization": 10.0, "resets_at": "2099-01-05T00:00:00Z"},
    "seven_day_opus": {"utilization": 3.0, "resets_at": None},
    "extra_usage": {"is_enabled": False},
    "weekly_scoped": [{"kind": "weekly_scoped", "percent": 4}],
}
WALL_BODY = {"five_hour": {"utilization": 99.0}, "juniper_tide": {"eligible": True}}


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock):
    return UsageStore(tmp_path / "cache", clock=clock)


def _row(store: UsageStore, num: str = "1") -> dict:
    return json.loads(store.path.read_text(encoding="utf-8"))["accounts"][num]


class TestUsageVariant:
    def test_forms_claude_code_sends(self):
        """Asserts: the three query forms Claude Code 2.1.287 sends map to
        their variants in any parameter order, and anything else is
        unknown."""
        assert usage_variant("") == USAGE_VARIANT_PLAIN
        assert usage_variant("at_wall=1&skip_spend=1") == USAGE_VARIANT_AT_WALL
        assert usage_variant("skip_spend=1&at_wall=1") == USAGE_VARIANT_AT_WALL
        assert (
            usage_variant("cedar_ember=1&skip_spend=1") == USAGE_VARIANT_CEDAR_EMBER
        )
        assert usage_variant("skip_spend=1") == USAGE_VARIANT_UNKNOWN
        assert usage_variant("at_wall=1") == USAGE_VARIANT_UNKNOWN


class TestOwnFetchKeepsTheBody:
    def test_record_stores_the_body_beside_last_good(self, store, clock):
        """Asserts: cswap's own successful fetch keeps the body as received
        under its plain form with its own read time, and lastGood stays the
        trimmed reading."""
        usage = oauth.build_usage_result(BODY)
        store.record({"1": FetchRecord(usage=usage, body=BODY)}, IDENT)
        row = _row(store)
        assert row["lastGood"] == usage
        assert row["usageBodies"] == {
            USAGE_VARIANT_PLAIN: {"body": BODY, "fetchedAt": clock.now}
        }

    def test_a_failure_leaves_the_body(self, store, clock):
        """Asserts: a failed fetch never touches the stored body."""
        store.record({"1": FetchRecord(usage={}, body=BODY)}, IDENT)
        clock.advance(10)
        store.record({"1": FetchRecord(error="timeout")}, IDENT)
        assert _row(store)["usageBodies"][USAGE_VARIANT_PLAIN]["body"] == BODY

    def test_a_header_reading_does_not_refresh_the_body(self, store, clock):
        """Asserts: a /v1/messages header reading moves fetchedAt but not the
        body's own time, so the body ages out of SERVE_TTL_S on schedule."""
        store.record({"1": FetchRecord(usage={}, body=BODY)}, IDENT)
        read_at = clock.now
        clock.advance(SERVE_TTL_S + 1)
        headers = {"anthropic-ratelimit-unified-5h-utilization": "0.3"}
        assert store.record_header_reading("1", IDENT, headers) is True
        row = _row(store)
        assert row["fetchedAt"] == clock.now
        assert row["usageBodies"][USAGE_VARIANT_PLAIN]["fetchedAt"] == read_at
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_FORWARD

    def test_try_fetch_carries_the_raw_body(self):
        """Asserts: a successful usage fetch returns the body it parsed
        alongside the trimmed reading."""
        creds = json.dumps({"claudeAiOauth": {"accessToken": "t"}})
        with patch("claude_swap.oauth.request_usage_data", return_value=BODY):
            out = oauth.try_fetch_usage_for_account(
                "1", "a@x.com", creds, is_active=True
            )
        assert out.error is None
        assert out.body == BODY
        assert out.usage == oauth.build_usage_result(BODY)


class TestServe:
    def test_fresh_plain_body_is_served_uncounted(self, store, clock):
        """Asserts: a plain body under SERVE_TTL_S answers a plain request
        with that exact body and stamps no attempt."""
        store.record({"1": FetchRecord(usage={}, body=BODY)}, IDENT)
        clock.advance(SERVE_TTL_S - 1)
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_SERVE
        assert answer.body == BODY
        assert answer.body_age_s == pytest.approx(SERVE_TTL_S - 1)
        assert "attempts" not in _row(store)

    def test_stale_body_goes_upstream(self, store, clock):
        """Asserts: a plain body older than SERVE_TTL_S is not served."""
        store.record({"1": FetchRecord(usage={}, body=BODY)}, IDENT)
        clock.advance(SERVE_TTL_S + 1)
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_FORWARD

    @pytest.mark.parametrize(
        "variant",
        [USAGE_VARIANT_AT_WALL, USAGE_VARIANT_CEDAR_EMBER, USAGE_VARIANT_UNKNOWN],
    )
    def test_no_other_form_is_served(self, store, clock, variant):
        """Asserts: a reset-offer or unknown form always goes upstream, even
        with a fresh plain body and a fresh body of its own form."""
        store.record({"1": FetchRecord(usage={}, body=BODY)}, IDENT)
        store.record_client_usage("1", IDENT, variant, status=200, body=WALL_BODY)
        answer = store.answer_client_usage("1", IDENT, variant)
        assert answer.action == CLIENT_FORWARD

    def test_another_accounts_body_is_never_served(self, store, clock):
        """Asserts: a body stored for a slot's previous account is invisible
        once the slot maps to a different identity."""
        store.record({"1": FetchRecord(usage={}, body=BODY)}, IDENT)
        other = {"1": ("c@x.com", "")}
        answer = store.answer_client_usage("1", other, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_FORWARD
        assert answer.body is None


class TestForwardCounts:
    def test_forward_stamps_the_shared_ledger(self, store, clock):
        """Asserts: a forwarded request lands in the same attempts ledger
        reserve() reads, so the hourly cap counts it."""
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_FORWARD
        assert answer.attempts_in_window == 1
        assert _row(store)["attempts"] == [clock.now]
        assert store.entries(IDENT)["1"].attempts_in_window == 1

    def test_forward_ignores_freshness_plans_and_leases(self, store, clock):
        """Asserts: unlike reserve(), a client's request is not refused by a
        just-read row, a future poll plan or a live fetch lease."""
        store.record(
            {"1": FetchRecord(usage={})}, IDENT, plans={"1": (clock.now + 900, 900)}
        )
        store.claim(["1"], IDENT)
        assert store.reserve(["1"], IDENT, respect_plans=True) == {}
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_AT_WALL)
        assert answer.action == CLIENT_FORWARD
        assert _row(store)["claimId"] is not None  # cswap's lease untouched

    def test_forward_ignores_a_non_429_backoff(self, store, clock):
        """Asserts: a timeout's backoff does not hold a client request; only
        a 429 block does."""
        store.record({"1": FetchRecord(error="timeout")}, IDENT)
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_FORWARD


class TestHold:
    def test_cap_holds_with_the_last_body(self, store, clock):
        """Asserts: at ATTEMPTS_PER_HOUR_MAX attempts in the window the
        request is held, answered with the last same-form body, its age,
        and when the oldest attempt ages out; nothing more is stamped."""
        store.record({"1": FetchRecord(usage={}, body=BODY)}, IDENT)
        clock.advance(SERVE_TTL_S + 1)
        first = clock.now
        for _ in range(ATTEMPTS_PER_HOUR_MAX):
            assert (
                store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN).action
                == CLIENT_FORWARD
            )
            clock.advance(1)
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_HOLD
        assert answer.reason == HOLD_REASON_CAP
        assert answer.body == BODY
        assert answer.attempts_in_window == ATTEMPTS_PER_HOUR_MAX
        assert answer.retry_after_s == pytest.approx(
            first + ATTEMPT_WINDOW_S - clock.now
        )
        assert len(_row(store)["attempts"]) == ATTEMPTS_PER_HOUR_MAX

    def test_cswap_attempts_count_toward_the_client_cap(self, store, clock):
        """Asserts: attempts cswap's own reserve() stamped hold a client
        request, the other direction of one shared budget."""
        for _ in range(ATTEMPTS_PER_HOUR_MAX):
            won = store.reserve(["1"], IDENT, respect_plans=False)
            store.record({"1": FetchRecord(error="timeout")}, IDENT, claims=won)
            clock.advance(1)
            row = _row(store)
            row["backoffUntil"] = None
            rows = json.loads(store.path.read_text(encoding="utf-8"))
            rows["accounts"]["1"] = row
            store.path.write_text(json.dumps(rows), encoding="utf-8")
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_HOLD
        assert answer.reason == HOLD_REASON_CAP
        assert answer.body is None

    def test_a_429_block_holds(self, store, clock):
        """Asserts: while an http-429 backoff runs the request is held with
        the backoff's remaining time, so nothing re-arms the block."""
        store.record(
            {"1": FetchRecord(error="http-429", retry_after_s=3600.0)}, IDENT
        )
        until = _row(store)["backoffUntil"]
        clock.advance(60)
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_CEDAR_EMBER)
        assert answer.action == CLIENT_HOLD
        assert answer.reason == HOLD_REASON_BACKOFF
        assert answer.retry_after_s == pytest.approx(until - clock.now)
        assert answer.body is None
        assert "attempts" not in _row(store)


class TestRecordClientUsage:
    def test_plain_success_is_a_full_reading(self, store, clock):
        """Asserts: a forwarded plain 200 stores the body and writes lastGood
        through build_usage_result, clearing failure fields, as cswap's own
        fetch does."""
        store.record({"1": FetchRecord(error="timeout")}, IDENT)
        clock.advance(5)
        assert store.record_client_usage(
            "1", IDENT, USAGE_VARIANT_PLAIN, status=200, body=BODY
        )
        row = _row(store)
        assert row["lastGood"] == oauth.build_usage_result(BODY)
        assert row["fetchedAt"] == clock.now
        assert row["consecutiveFailures"] == 0
        assert row["backoffUntil"] is None
        assert row["usageBodies"][USAGE_VARIANT_PLAIN]["body"] == BODY
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.action == CLIENT_SERVE

    def test_reset_offer_success_keeps_last_good(self, store, clock):
        """Asserts: a skip_spend body is stored for its own form and never
        replaces lastGood, which needs the spend block it lacks."""
        store.record({"1": FetchRecord(usage={"five_hour": {"pct": 1.0}})}, IDENT)
        before = _row(store)
        clock.advance(5)
        assert store.record_client_usage(
            "1", IDENT, USAGE_VARIANT_AT_WALL, status=200, body=WALL_BODY
        )
        row = _row(store)
        assert row["lastGood"] == before["lastGood"]
        assert row["fetchedAt"] == before["fetchedAt"]
        assert row["usageBodies"][USAGE_VARIANT_AT_WALL]["body"] == WALL_BODY

    def test_429_records_the_block(self, store, clock):
        """Asserts: a forwarded 429 records http-429 with its Retry-After the
        way cswap's own 429 is recorded, so the next request is held."""
        assert store.record_client_usage(
            "1", IDENT, USAGE_VARIANT_PLAIN, status=429, retry_after_s=3600.0
        )
        twin = UsageStore(store.path.parent / "twin", clock=clock)
        twin.record({"1": FetchRecord(error="http-429", retry_after_s=3600.0)}, IDENT)
        row = _row(store)
        assert row["lastError"] == "http-429"
        assert row["last429At"] == clock.now
        assert row["backoffUntil"] == _row(twin)["backoffUntil"]
        answer = store.answer_client_usage("1", IDENT, USAGE_VARIANT_PLAIN)
        assert answer.reason == HOLD_REASON_BACKOFF

    @pytest.mark.parametrize("status", [401, 403, 500])
    def test_other_statuses_record_nothing(self, store, status):
        """Asserts: a status that says nothing about the account's budget
        writes nothing."""
        assert not store.record_client_usage(
            "1", IDENT, USAGE_VARIANT_PLAIN, status=status
        )
        assert not store.path.exists()

    def test_unknown_form_body_is_not_kept(self, store):
        """Asserts: an unrecognised form's body is never stored, so it can
        never answer anything."""
        assert store.record_client_usage(
            "1", IDENT, USAGE_VARIANT_UNKNOWN, status=200, body=BODY
        )
        assert "usageBodies" not in _row(store)

    def test_a_200_without_an_object_body_records_nothing(self, store):
        """Asserts: a 200 whose body did not parse to a JSON object is not
        recorded as a reading."""
        assert not store.record_client_usage(
            "1", IDENT, USAGE_VARIANT_PLAIN, status=200, body=None
        )


class TestSwitcherEntry:
    """The proxy-facing entry points: a slot number, no identity passed."""

    def test_round_trip_through_the_switcher(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        """Asserts: the switcher resolves the slot's identity, forwards on an
        empty row, records the reply, and serves it next."""
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, sample_sequence_data)
        first = switcher.answer_client_usage("1", USAGE_VARIANT_PLAIN)
        assert first is not None and first.action == CLIENT_FORWARD
        assert switcher.record_client_usage(
            "1", USAGE_VARIANT_PLAIN, status=200, body=BODY
        )
        second = switcher.answer_client_usage("1", USAGE_VARIANT_PLAIN)
        assert second is not None and second.action == CLIENT_SERVE
        assert second.body == BODY

    def test_unknown_slot_decides_nothing(
        self, temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict
    ):
        """Asserts: a slot the roster does not hold gets None and records
        nothing."""
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, sample_sequence_data)
        assert switcher.answer_client_usage("9", USAGE_VARIANT_PLAIN) is None
        assert not switcher.record_client_usage(
            "9", USAGE_VARIANT_PLAIN, status=200, body=BODY
        )
