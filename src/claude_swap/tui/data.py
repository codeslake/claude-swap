"""Data service for the TUI: snapshots, blocking actions, display helpers.

The TUI never parses printed CLI output — it consumes
``ClaudeAccountSwitcher.accounts_snapshot`` (one collect pass, see
switcher.py) and renders structured data. Fetch pacing lives in
``claude_swap.snapshot_source.SnapshotSource`` (shared with any GUI shell);
this module re-exports it for the TUI's use.

Everything here is blocking (file locks, keychain subprocesses, network) and
must be called from a thread worker, never the UI event loop.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from claude_swap import oauth, printer, usage_store
from claude_swap.autoswitch import (
    CONSUME_FIRST_STRATEGIES,
    _classify_dynamic_trigger,
    _dynamic_active_headroom,
    _headroom_by_account,
    _model_window_binds_everywhere,
    rank_candidates_pass,
)
from claude_swap.exceptions import ClaudeSwitchError
from claude_swap.models import AccountsSnapshot
from claude_swap.poll_policy import binding_pct
from claude_swap.settings import parse_model_names
from claude_swap.snapshot_source import SnapshotSource
from claude_swap.switcher import SENTINEL_NOTES, last_seen_note

if TYPE_CHECKING:
    from claude_swap.settings import AutoSwitchSettings

# Trigger names where the ranking pass never runs a tick at all -- shared
# with the auto view's `_UNMODELED_TEXT`, same three keys.
_UNMODELED_TRIGGERS = frozenset(
    {"dynamic-unmodeled", "below-threshold", "unreadable-active"}
)


# ---------------------------------------------------------------------------
# Blocking actions (switch/add/remove) run captured, off the UI thread
# ---------------------------------------------------------------------------


@dataclass
class ActionResult:
    """Outcome of a captured switcher action."""

    ok: bool
    output: str  # captured stdout+stderr, ANSI-colored (render with Text.from_ansi)
    payload: dict | None = None  # structured result for json-capable actions

    @property
    def first_line(self) -> str:
        """First non-empty output line, ANSI-stripped — notification material."""
        from rich.text import Text

        for line in self.output.splitlines():
            plain = Text.from_ansi(line).plain.strip()
            if plain:
                return plain
        return ""


def run_action(fn: Callable[[], dict | None]) -> ActionResult:
    """Run a switcher action capturing stdout+stderr (color forced on).

    ``sys.stdin`` is swapped for an empty stream so an unexpected ``input()``
    raises ``EOFError`` instead of freezing the app (in-scope actions never
    prompt once ``assume_yes``/explicit identifiers are used; this is
    defensive). The redirect is process-global for the duration — fine here
    because the TUI owns the terminal and nothing else prints while it runs.
    """
    buf = io.StringIO()
    payload: dict | None = None
    saved_stdin = sys.stdin
    sys.stdin = io.StringIO()
    try:
        with printer.force_color(), contextlib.redirect_stdout(
            buf
        ), contextlib.redirect_stderr(buf):
            try:
                payload = fn()
            except ClaudeSwitchError as e:
                print(f"Error: {e}")
                return ActionResult(False, buf.getvalue())
            except EOFError:
                print("Error: interactive input is not available here.")
                return ActionResult(False, buf.getvalue())
    finally:
        sys.stdin = saved_stdin
    return ActionResult(
        True, buf.getvalue(), payload if isinstance(payload, dict) else None
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def sentinel_label(sentinel: str) -> str:
    """The same wording ``cswap list`` prints for this sentinel state."""
    return SENTINEL_NOTES.get(sentinel, sentinel)


def window_pct(last_good: dict | None, key: str) -> float | None:
    """Utilization pct of one window ("five_hour"/"seven_day"), if known."""
    if not isinstance(last_good, dict):
        return None
    window = last_good.get(key)
    if not isinstance(window, dict):
        return None
    pct = window.get("pct")
    return float(pct) if isinstance(pct, (int, float)) else None


def reset_text(window: dict | None, now: float) -> str | None:
    """Live countdown to one window's reset ("resets 2h 13m"), if known.

    Computed from ``resets_at`` at render time — the countdown the API sent
    was correct at *fetch* time and drifts as the measurement ages.
    """
    if not isinstance(window, dict):
        return None
    resets_at = window.get("resets_at")
    if not resets_at:
        return None
    try:
        ts = datetime.fromisoformat(str(resets_at).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None
    remaining = ts - now
    if remaining <= 0:
        return "resets now"
    return f"resets {format_duration(remaining)}"


def reset_clock(window: dict | None, now: float) -> str | None:
    """Absolute local reset time ("20:39" / "Jul 14 09:00"), if known.

    None once the reset has elapsed — "resets now" needs no clock.
    """
    if not isinstance(window, dict):
        return None
    resets_at = window.get("resets_at")
    if not resets_at:
        return None
    try:
        reset_utc = datetime.fromisoformat(str(resets_at).replace("Z", "+00:00"))
    except ValueError:
        return None
    if reset_utc.timestamp() - now <= 0:
        return None
    return oauth.reset_clock_string(
        reset_utc, datetime.fromtimestamp(now, tz=timezone.utc)
    )


def window_reset_text(last_good: dict | None, key: str, now: float) -> str | None:
    """`reset_text` for one of the top-level 5h/7d windows."""
    if not isinstance(last_good, dict):
        return None
    return reset_text(last_good.get(key), now)


def chip_label(label: str, reset: str | None) -> str:
    """The reading for one window, without its percentage: ``5h(⟳2h28m)``.

    THE one place that decides how a window reads — the dashboard's inactive
    rows and the auto view's Next-best rows both draw it, so one account
    cannot read two ways on two screens. The caller appends the pct so it can
    colour it by severity. The countdown shows whenever it is known, not only
    at 100%: a saturated candidate's worth IS when it comes back.
    """
    if not reset:
        return f"{label}:"
    return f"{label}(⟳{reset.removeprefix('resets ').replace(' ', '')}):"


def window_chip_label(last_good: dict | None, key: str, label: str, now: float) -> str:
    """`chip_label` for one of the top-level 5h/7d windows."""
    return chip_label(label, window_reset_text(last_good, key, now))


def format_duration(seconds: float) -> str:
    """Compact duration: "45s", "12m", "2h 13m", "3d 4h"."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m"
    if s < 86400:
        h, m = divmod(s // 60, 60)
        return f"{h}h {m}m" if m else f"{h}h"
    d, h = divmod(s // 3600, 24)
    return f"{d}d {h}h" if h else f"{d}d"


def format_age(age_s: float | None) -> str | None:
    """Measurement age note ("· 2m ago"); None while comfortably fresh."""
    if age_s is None or age_s < usage_store.SERVE_TTL_S:
        return None
    return f"· {format_duration(age_s)} ago"


def clock_stamp() -> str:
    """HH:MM:SS local-time stamp for the event log."""
    return time.strftime("%H:%M:%S")


def rank_switch_candidates(
    snap: AccountsSnapshot,
    settings: "AutoSwitchSettings",
    now: float,
    active_number: str | None,
) -> tuple[list[str], str | None, str, bool]:
    """(ordered, rank_axis, trigger, unmodeled): the engine's own admission
    and order. THE shared computation -- ``ordered_accounts`` and the auto
    view's "Next best" panel both read off this, never a pass of their own.
    """
    models = parse_model_names(settings.model)
    consume_first = settings.strategy in CONSUME_FIRST_STRATEGIES
    usage = {acc.number: acc.usage.decision_value() for acc in snap.accounts}
    oauth_candidates = [
        acc.number
        for acc in snap.accounts
        if acc.number != active_number
        and acc.switchable
        and not acc.disabled
        and acc.kind != "api_key"
    ]
    api_key_candidates = (
        [
            acc.number
            for acc in snap.accounts
            if acc.number != active_number
            and acc.switchable
            and not acc.disabled
            and acc.kind == "api_key"
        ]
        if settings.include_api_key_accounts
        else []
    )

    def _trigger_for(active_headroom: float | None, active_disabled: bool) -> str:
        if active_disabled:
            return "disabled-active"
        if active_headroom is None:
            return "unreadable-active"
        if settings.strategy == "dynamic":
            kind = _classify_dynamic_trigger(active_headroom)
            return "at-limit" if kind == "at-limit" else "dynamic-unmodeled"
        if (100.0 - active_headroom) < settings.threshold:
            return (
                settings.strategy
                if settings.strategy in CONSUME_FIRST_STRATEGIES
                else "below-threshold"
            )
        return "at-limit" if active_headroom <= 0 else "proactive"

    def _rank_on(axis: tuple[str, ...], trigger: str) -> tuple[list[str], str | None]:
        if trigger in _UNMODELED_TRIGGERS:
            return [], None
        headroom = _headroom_by_account(usage, axis)
        ordered, _any_known, _reset_ts, _waiting, rank_axis = rank_candidates_pass(
            models=axis,
            trigger=trigger,
            consume_first=consume_first,
            oauth_candidates=oauth_candidates,
            no_return=None,
            usage=usage,
            headroom=headroom,
            current=active_number,
            active_headroom=headroom.get(active_number),
            settings=settings,
            now=now,
        )
        return ordered, rank_axis

    model_headroom = _headroom_by_account(usage, models)
    active_account = next(
        (acc for acc in snap.accounts if acc.number == active_number), None
    )
    active_disabled = active_account.disabled if active_account is not None else False
    trigger = _trigger_for(
        _dynamic_active_headroom(
            settings, models, usage, active_number, model_headroom.get(active_number)
        ),
        active_disabled,
    )
    unmodeled = trigger in _UNMODELED_TRIGGERS
    ordered, rank_axis = _rank_on(models, trigger)
    if (
        not ordered
        and models
        and settings.strategy == "dynamic"
        and _model_window_binds_everywhere(usage, models, settings.threshold)
    ):
        ordered, rank_axis = _rank_on((), trigger)
    if (
        not ordered
        and api_key_candidates
        and not unmodeled
        and trigger not in CONSUME_FIRST_STRATEGIES
    ):
        ordered, rank_axis = api_key_candidates, None
    return ordered, rank_axis, trigger, unmodeled


def ordered_accounts(
    snap: AccountsSnapshot, settings: "AutoSwitchSettings", now: float
) -> list[str]:
    """Every account number, active first, then the rest as the engine's own
    pass would rank them: ranked-and-open, usable-but-refused, sentinel-
    blocked, spend-only, unswitchable last. THE one order every screen
    renders in -- a screen keeping slot order says so at its own call site.
    """
    active_number = snap.active_number
    others = [acc for acc in snap.accounts if acc.number != active_number]
    ordered, _axis, _trigger, _unmodeled = rank_switch_candidates(
        snap, settings, now, active_number
    )
    ordered_rank = {num: i for i, num in enumerate(ordered)}
    models = parse_model_names(settings.model)

    def bucket(acc) -> tuple:
        if not acc.switchable:
            return (4,)
        if acc.number in ordered_rank:
            return (0, ordered_rank[acc.number])
        if acc.usage.sentinel is not None:
            return (2,)
        if binding_pct(acc.usage.last_good, models) is None:
            return (3,)
        return (1,)

    # `(bucket, number)`: same tie-break as the auto view's own `sorted`.
    numbers = [
        acc.number for acc in sorted(others, key=lambda a: (bucket(a), a.number))
    ]
    return ([active_number] if active_number is not None else []) + numbers


__all__ = [
    "ActionResult",
    "SnapshotSource",
    "format_age",
    "format_duration",
    "last_seen_note",
    "ordered_accounts",
    "rank_switch_candidates",
    "reset_clock",
    "reset_text",
    "run_action",
    "sentinel_label",
    "clock_stamp",
    "window_pct",
    "window_reset_text",
]
