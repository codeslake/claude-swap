"""The 5h rule applies only while the API reports a LIVE 5h window.

Live means ``five_hour`` carries a utilization above 0 and a ``resets_at``.
Everything else (idle ``{0, null}``, a rolling ``{0, +5h}``, a null or an
absent key) is "no 5h limit right now" and must neither bind a decision, be
drawn as a bar, nor be synthesized as full by a 429 wall.

Shapes ``live`` and ``idle`` are the two measured on real accounts (the numbers
are one active account's, 40% / 67%). ``absent``, ``null`` and ``rolling`` are
HYPOTHESES for what a removed 5h window looks like, until one is captured.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from claude_swap import menubar, oauth, usage_store
from claude_swap.autoswitch import PollEvent, _window_pcts, consume_first_rank_key
from claude_swap.lines import binding_window
from claude_swap.models import AccountSnapshot
from claude_swap.switcher import ClaudeAccountSwitcher, _format_usage_lines
from claude_swap.tui import data as tui_data, widgets
from claude_swap.usage_store import WALL_FALLBACK_S, FetchRecord, UsageEntry, UsageStore

IDENT = {"1": ("a@x.com", "")}
ABSENT = object()
NOT_LIVE = ["idle", "absent", "null", "rolling"]


def _iso(seconds: float) -> str:
    when = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return when.isoformat(timespec="seconds").replace("+00:00", "Z")


def _api(shape: str) -> dict:
    """A raw usage-endpoint body with the named ``five_hour`` shape."""
    five_hour = {
        "live": {"utilization": 40.0, "resets_at": _iso(3 * 3600)},
        "idle": {"utilization": 0.0, "resets_at": None},
        "absent": ABSENT,
        "null": None,
        "rolling": {"utilization": 0.0, "resets_at": _iso(5 * 3600)},
    }[shape]
    body: dict = {"seven_day": {"utilization": 67.0, "resets_at": _iso(5 * 86400)}}
    if five_hour is not ABSENT:
        body["five_hour"] = five_hour
    return body


def _usage(shape: str) -> dict:
    usage = oauth.build_usage_result(_api(shape))
    assert usage is not None
    return usage


def _snapshot(usage: dict, plan: str | None = None) -> AccountSnapshot:
    entry = UsageEntry(last_good=usage, fetched_at=time.time(), age_s=1.0, plan=plan)
    return AccountSnapshot(
        number="2", email="b@x.com", org_name="", org_uuid="", is_active=False,
        kind="oauth", switchable=True, usage=entry,
    )


@pytest.fixture
def store(tmp_path):
    return UsageStore(tmp_path / "cache")


# -- the predicate and the plan label --------------------------------------


@pytest.mark.parametrize(
    "shape,live",
    [("live", True), ("idle", False), ("absent", False), ("null", False), ("rolling", False)],
)
def test_only_a_live_window_is_live(shape, live):
    assert oauth.five_hour_live(_usage(shape)) is live


@pytest.mark.parametrize(
    "usage",
    [None, "token expired", {}, {"five_hour": {"pct": 9.0}}, {"five_hour": "x"}],
    ids=["none", "sentinel", "empty", "no-reset", "not-a-window"],
)
def test_a_reading_without_a_proper_5h_window_is_not_live(usage):
    assert oauth.five_hour_live(usage) is False


@pytest.mark.parametrize(
    "subscription,tier,label",
    [
        ("max", "default_claude_max_20x", "Ultra"),
        ("max", "default_claude_ultra", "Ultra"),
        ("max", "default_claude_max_5x", "Max 5x"),
        ("max", "default_claude_unheard_of", "Max"),
        ("pro", None, "Pro"),
        ("team", "default_raven", "Team"),
        ("enterprise", "default_claude_zero", "Enterprise"),
        (None, "default_claude_new_tier", "default_claude_new_tier"),
        ("weird", None, "weird"),
        ("weird", "", "weird"),
        (None, None, ""),
        (5, {}, ""),
    ],
)
def test_plan_label(subscription, tier, label):
    assert oauth.plan_label(subscription, tier) == label


# -- ranking and rendering ---------------------------------------------------


@pytest.mark.parametrize("shape", NOT_LIVE)
def test_non_live_shapes_rank_and_render_like_an_absent_window(shape):
    ref, got, now = _usage("absent"), _usage(shape), time.time()
    assert oauth.account_headroom(got) == oauth.account_headroom(ref)
    assert consume_first_rank_key(got, 90.0, now) == consume_first_rank_key(ref, 90.0, now)
    assert _format_usage_lines(got) == _format_usage_lines(ref)
    assert widgets.usage_rows(got, now) == widgets.usage_rows(ref, now)
    assert menubar.usage_summary(got, now) == menubar.usage_summary(ref, now)
    both = menubar.MenuBarSettings(title_pct="both")
    assert menubar.format_title("a@x.com", got, both, now) == menubar.format_title(
        "a@x.com", ref, both, now
    )
    assert (
        widgets.mini_account_text(_snapshot(got), now).plain
        == widgets.mini_account_text(_snapshot(ref), now).plain
    )
    assert "5h" not in widgets.account_card_text(_snapshot(got), 80, now=now).plain
    assert tui_data.window_pct(got, "five_hour") is None  # no Next-best chip


def test_a_live_window_still_renders():
    got, now = _usage("live"), time.time()
    assert [label for label, *_ in widgets.usage_rows(got, now)] == ["5h", "7d"]
    assert _format_usage_lines(got)[0].startswith("5h:")
    assert menubar.usage_summary(got, now).startswith("5h 40%")
    assert ":40%" in widgets.mini_account_text(_snapshot(got), now).plain
    assert tui_data.window_pct(got, "five_hour") == 40.0
    assert "5h 40% · 7d 67%" in menubar.format_title(
        "a@x.com", got, menubar.MenuBarSettings(title_pct="both"), now
    )


def test_the_both_title_labels_its_numbers():
    both = menubar.MenuBarSettings(title_pct="both", show_account_name=False)
    # A 7d figure must never sit where the 5h one used to.
    assert menubar.format_title("a@x.com", _usage("idle"), both) == f"{menubar.ICON} 7d 67%"
    seven = menubar.MenuBarSettings(title_pct="7d", show_account_name=False)
    assert menubar.format_title("a@x.com", _usage("live"), seven) == f"{menubar.ICON} 67%"


def test_a_missing_5h_line_is_not_a_preference():
    # Same 7d: one account with a live low 5h, one without. Same rank key.
    seven = {"pct": 67.0, "resets_at": _iso(5 * 86400)}
    live_low = {"five_hour": {"pct": 10.0, "resets_at": _iso(3600)}, "seven_day": seven}
    now = time.time()
    assert consume_first_rank_key(live_low, 90.0, now) == consume_first_rank_key(
        {"seven_day": seven}, 90.0, now
    )


@pytest.mark.parametrize("shape", NOT_LIVE)
def test_a_threshold_with_no_weekly_line_still_leaves_a_no_5h_account_at_its_7d(shape):
    usage = _usage(shape)
    for pct, over in ((97.0, True), (96.0, False)):
        usage["seven_day"]["pct"] = pct
        binding = binding_window(oauth.relevant_windows(usage), 97.0, None)
        assert (binding.label, binding.over, binding.line) == ("7d", over, 97.0)


# -- the 429 wall --------------------------------------------------------------


def _describe(value: dict) -> str:
    event = PollEvent(
        active=None, headroom={}, threshold=90.0,
        windows={"1": _window_pcts(value, ("all",))},
    )
    return event._describe("1")


@pytest.mark.parametrize("shape", NOT_LIVE)
def test_a_429_on_a_no_5h_account_marks_its_binding_window(store, shape):
    usage = _usage(shape)
    usage["seven_day"]["pct"] = 80.0
    store.record({"1": FetchRecord(usage=usage)}, IDENT)
    store.mark_at_limit("1", IDENT)

    value = store.entries(IDENT)["1"].decision_value()
    assert "five_hour" not in value
    assert value["seven_day"]["pct"] == 100.0
    assert value["seven_day"]["resets_at"] == usage["seven_day"]["resets_at"]
    assert _describe(value) == "7d 100%"
    assert oauth.binding_window_label(value, ("all",)) == "7d"


def test_a_429_marks_the_scoped_window_when_that_one_binds(store):
    usage = _usage("idle")
    usage["seven_day"]["pct"] = 40.0
    usage["scoped"] = [{"name": "Fable", "pct": 90.0, "resets_at": _iso(2 * 86400)}]
    store.record({"1": FetchRecord(usage=usage)}, IDENT)
    store.mark_at_limit("1", IDENT, ("all",))

    value = store.entries(IDENT)["1"].decision_value()
    assert (value["seven_day"]["pct"], value["scoped"][0]["pct"]) == (40.0, 100.0)
    assert oauth.binding_window_label(value, ("all",)) == "Fable"


def test_a_429_on_a_live_5h_account_still_reads_5h_full(store):
    store.record({"1": FetchRecord(usage=_usage("live"))}, IDENT)
    store.mark_at_limit("1", IDENT)

    entry = store.entries(IDENT)["1"]
    value = entry.decision_value()
    assert value["five_hour"] == {
        "pct": 100.0, "resets_at": usage_store._reset_ts_to_resets_at(entry.walled_until),
    }
    assert value["seven_day"]["pct"] == 67.0


def test_a_rolling_phantom_reset_does_not_set_the_wall_span(store):
    usage = _usage("rolling")
    usage["five_hour"]["resets_at"] = _iso(3600)  # the phantom rolls in an hour
    usage["seven_day"] = {"pct": 100.0, "resets_at": _iso(3 * 86400)}
    store.record({"1": FetchRecord(usage=usage)}, IDENT)
    store.mark_at_limit("1", IDENT)

    assert store.entries(IDENT)["1"].walled_until == pytest.approx(
        time.time() + WALL_FALLBACK_S, abs=5
    )


# -- header readings -------------------------------------------------------------


def test_a_reply_without_a_5h_header_keeps_the_7d_reading(store):
    scoped = [{"name": "Fable", "pct": 33.0}]
    store.record({"1": FetchRecord(usage={**_usage("live"), "scoped": scoped})}, IDENT)

    assert store.record_header_reading(
        "1", IDENT, {usage_store.USAGE_HEADER_7D_PCT: "0.55"}
    ) is True
    last_good = store.entries(IDENT)["1"].last_good
    assert last_good["seven_day"]["pct"] == pytest.approx(55.0)
    assert last_good["five_hour"]["pct"] == 40.0  # left as polled until the next poll
    assert last_good["scoped"] == scoped


# -- the plan reaches the screens ----------------------------------------------


def test_a_record_stamps_the_plan_beside_lastgood(store):
    store.record({"1": FetchRecord(usage=_usage("live"), plan="Ultra")}, IDENT)
    row = json.loads(store.path.read_text(encoding="utf-8"))["accounts"]["1"]
    assert row["plan"] == "Ultra"
    assert "plan" not in row["lastGood"]
    assert store.entries(IDENT)["1"].plan == "Ultra"
    # A record that carries no plan leaves the stamped one alone.
    store.record({"1": FetchRecord(usage=_usage("live"))}, IDENT)
    assert store.entries(IDENT)["1"].plan == "Ultra"


PLAN_BLOBS = [
    ({"subscriptionType": "max", "rateLimitTier": "default_claude_max_20x"}, "Ultra"),
    ({"rateLimitTier": "default_claude_new_tier"}, "default_claude_new_tier"),
    ({}, None),
]


def _fetch_with(blob: dict, sample_sequence_data: dict, then):
    """Run ``then(switcher)`` against a switcher whose slots hold ``blob``'s
    credential and whose usage endpoint answers a live 5h window."""
    sample_sequence_data["accounts"]["1"]["email"] = "test@example.com"
    active, backup = (
        json.dumps({"claudeAiOauth": {"accessToken": token, **blob}})
        for token in ("sk-active", "sk-backup")
    )
    response = MagicMock()
    response.read.return_value = json.dumps(_api("live")).encode()
    response.__enter__ = lambda s: s
    response.__exit__ = MagicMock(return_value=False)
    switcher = ClaudeAccountSwitcher()
    switcher._setup_directories()
    switcher._write_json(switcher.sequence_file, sample_sequence_data)
    with patch.object(switcher, "_read_credentials", return_value=active), \
         patch.object(switcher, "_read_account_credentials", return_value=backup), \
         patch("claude_swap.oauth.urllib.request.urlopen", return_value=response):
        return then(switcher)


@pytest.mark.parametrize("blob,plan", PLAN_BLOBS)
def test_a_fetch_stamps_the_plan_from_the_slots_credential(
    temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, blob, plan
):
    entries = _fetch_with(
        blob, sample_sequence_data,
        lambda s: s._collect_usage_entries(s._build_accounts_info()),
    )
    assert entries["2"].plan == plan


@pytest.mark.parametrize("blob,plan", PLAN_BLOBS)
def test_list_shows_the_plan_after_the_tag(
    temp_home: Path, mock_claude_config: Path, sample_sequence_data: dict, capsys, blob, plan
):
    _fetch_with(blob, sample_sequence_data, lambda s: s.list_accounts())
    header = next(
        line for line in capsys.readouterr().out.splitlines() if "account2@example.com" in line
    )
    assert header.rstrip().endswith(f"[personal] {plan}" if plan else "[personal]")


@pytest.mark.parametrize("plan", ["Ultra", "default_claude_new_tier"])
def test_the_tui_row_and_card_show_the_plan(plan):
    now = time.time()
    snap, bare = _snapshot(_usage("live"), plan), _snapshot(_usage("live"))
    assert f"[personal]  {plan}" in widgets.mini_account_text(snap, now).plain
    assert f"[personal]  {plan}" in widgets.account_card_text(snap, 80, now=now).plain
    assert plan not in widgets.mini_account_text(bare, now).plain
    assert plan not in widgets.account_card_text(bare, 80, now=now).plain


def test_the_menubar_row_shows_the_plan():
    row = menubar.format_account_label("2", "b@x.com", _usage("live"), plan="Ultra")
    assert row.startswith("2  b@x.com  Ultra  5h 40%")
    assert "Ultra" not in menubar.format_account_label("2", "b@x.com", _usage("live"))
