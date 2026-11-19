"""Live auto-switch screen: the real engine, visualized.

Runs :class:`AutoSwitchEngine` in a thread worker and renders its typed
events. Opens in **dry-run** — opening a view must never start switching
accounts on its own; going live is an explicit, confirmed action. The
engine's own state file semantics (shared cooldown, quarantine list, state
lock) make it safe to run alongside an external ``cswap auto``.

The active account's full card sits on top (same widget as the dashboard's
panel, with the threshold tick); this screen adds the engine badge, the
ranked switch candidates, and the decision log. While it is up, the app's
snapshot poller runs store-only: the engine is the only fetcher.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from rich.cells import cell_len
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from claude_swap import oauth
from claude_swap.autoswitch import (
    AutoSwitchEngine,
    AutoSwitchEvent,
    PollEvent,
    _binding_at,
    binding_pct,
    classify_candidate_block,
    pct_label,
    proactive_switch_bar_pct,
)
from claude_swap import pin
from claude_swap.json_output import USAGE_API_KEY, USAGE_NO_CREDENTIALS
from claude_swap.lines import BindingWindow, weekly_capped, window_line
from claude_swap.models import AccountsSnapshot
from claude_swap.settings import (
    SETTING_SPECS,
    AutoSwitchSettings,
    load_settings,
    parse_model_names,
    set_setting,
)
from claude_swap.tui import data
from claude_swap.tui.modals import ConfirmModal
from claude_swap.tui.theme import Palette
from claude_swap.tui.widgets import AccountsPanel, login_segment, spend_row, usage_rows

if TYPE_CHECKING:
    from claude_swap.tui.app import CswapApp

_EVENT_ROLES = {
    "switch": "accent",
    "error": "sev_warn",
    "account-quarantined": "sev_warn",
    "all-exhausted": "sev_crit",
}
_QUIET_KINDS = {"poll", "no-switch", "sleep", "account-unquarantined"}


def event_text(event: AutoSwitchEvent, *, palette: Palette = Palette.DARK) -> Text:
    """Log line for one engine event, styled like the CLI's human renderer."""
    role = _EVENT_ROLES.get(event.kind)
    if role == "sev_crit" and getattr(event, "deliberate_wait", False):
        # The map keys on the KIND and this kind carries two states; the
        # critical colour overstates a hold whose gate proves every candidate
        # was READ and one still holds quota.
        role = "sev_warn"
    if role is not None:
        style = getattr(palette, role)
    else:
        style = palette.muted if event.kind in _QUIET_KINDS else palette.foreground
    text = Text()
    text.append(f"{data.clock_stamp()}  ", style=palette.muted)
    text.append(event.human(), style=style)
    return text


_STRATEGY_CYCLE = ("best", "consume-first", "dynamic")

# Trigger names keyed to WHY the ranking pass never ran -- never "no
# candidate qualifies", the claim only a real, empty pass earns.
_UNMODELED_TEXT = {
    "below-threshold": "not previewed (active below threshold)",
    "unreadable-active": "not previewed (active status unknown)",
}

_NO_PLAN_TEXT = "no active plan"

# The engine's refusal reasons (`reasons` of `rank_switch_candidates`) in the panel's plain words: (long, short), one entry for every literal the engine gives
# but `over the switch threshold`, which is the at-line piece. `_unreadable_reason`'s texts (a poll that failed, a reading overdue) are told by how it opens them: short `unreadable`.
# Any other text keeps itself, long and short, so a reason the engine words later is never called unreadable.
_REFUSAL_WORDS = {
    "weekly reset not sooner": ("weekly reset comes later", "later reset"),
    "weekly reset unknown": ("weekly reset unknown", "reset unknown"),
    "recovers no sooner": ("recovers no sooner", "no sooner"),
    "headroom margin too small": ("too little headroom", "low headroom"),
    "at its limit": ("at its limit", "at limit"),
    "just left": ("just switched away", "just left"),
    "no headroom left": ("no headroom left", "no headroom"),
}
_UNREADABLE_OPENINGS = ("usage unreadable", "usage overdue", "last read ", "last fetch failed", "rate-limited until", "no poll before")


def _refusal_words(why: str | None) -> tuple[str, str]:
    """The long and the short words of the engine's reason for refusing a row."""
    if why is None:
        return "not picked", "skipped"
    if why in _REFUSAL_WORDS:
        return _REFUSAL_WORDS[why]
    return why, "unreadable" if why.startswith(_UNREADABLE_OPENINGS) or ", retry at " in why else why


# The panel's words for each refusal the tick reaches around its pass, one
# key per literal `no_peer_refusal` and `empty_pass_refusal` return (the
# engine's own reading of the state, shared with the tick). A hold the engine
# calls healthy (NO_ACTION) is worded as a hold, never as a block. The
# "below-threshold" here is the consume-first hold with no OAuth peer, not
# `best` below its line (`_UNMODELED_TEXT`).
_REFUSAL_TEXT = {
    "below-threshold": "holding (below its line, no other OAuth account to compare)",
    "no-candidates": "no account to switch to",
    "no-comparison": "no candidate has readable usage",
    "reset-unknown": "holding (the active account's weekly reset is unknown)",
    "already-consuming-soonest":
        "holding (no sooner-resetting account with room to spare)",
    "no-qualifying-candidate": "no candidate qualifies",
    "deliberate-wait": "holding for a nearer reset than any peer offers",
    "all-exhausted": "all accounts exhausted",
}


def _line_note(binding: BindingWindow | None, label: str, noted: bool) -> str:
    """The line's pct ("97%") for the "limit 97%" status piece, once a weekly
    line or the account's own threshold makes the lines differ: the chip's raw
    pct stays, and the line it is judged against is named only where it is not
    the one number in the header. Empty unless `label` is the BINDING window's."""
    if not noted or binding is None or label != binding.label:
        return ""
    return f"{pct_label(binding.line)}%"


# Every per-account reason a Next best row is not a plain candidate, in the one
# order they read: the limits reached, the other reasons, auto-swap off. A new
# marker registers by adding its key here; only "block" and "threshold" have a
# producer on this branch.
STATUS_ORDER = (
    "capped", "at-line", "block", "walled", "threshold", "stale", "no-plan",
    "not-candidate", "disabled",
)

EMAIL_FLOOR = 12  # display columns the email slot is cut to, at the least


@dataclass
class CandidateRow:
    """One Next best row as named slots. A producer fills a slot and never
    pads it: `render_candidate_rows` owns every column. `status` and `where`
    render as ONE trailing note, `where` last; the `login` note is the last slot."""

    number: Text
    email: Text
    usage: Text  # the chip block, the spend text, or the sentinel note
    status: dict[str, Text] = field(default_factory=dict)  # keys of STATUS_ORDER
    # The short form of a status piece, for a narrow panel; a key without one
    # keeps its long text.
    short: dict[str, Text] = field(default_factory=dict)
    where: Text = field(default_factory=Text)  # where-in-use markers, joined by " · "
    # The `re-login in 5d 12h` note that ends the row, empty unless the login is
    # due, and its short form for a narrow panel (a row without one keeps the long).
    login: Text = field(default_factory=Text)
    login_short: Text = field(default_factory=Text)
    note: bool = False  # usage is free text (a sentinel note): it sizes no column

    def __post_init__(self) -> None:
        for pieces in (self.status, self.short):
            if unknown := pieces.keys() - set(STATUS_ORDER):
                raise ValueError(f"status key not in STATUS_ORDER: {sorted(unknown)}")
        # A marker never pads itself; usage is exempt, its chip blanks pad on purpose.
        for cell in (self.login, self.login_short, self.where, *self.status.values(), *self.short.values()):
            if cell.plain != cell.plain.strip():
                raise ValueError(
                    f"marker has leading or trailing whitespace: {cell.plain!r}"
                )
        if orphan := self.short.keys() - self.status.keys():
            raise ValueError(f"short form without a status entry: {sorted(orphan)}")


def _cut(cell: Text, width: int) -> Text:
    """`cell` in at most `width` display columns, ending in `…` where it is cut."""
    cell = cell.copy()
    cell.truncate(width, overflow="ellipsis")
    return cell


def render_candidate_rows(
    rows: list[CandidateRow], palette: Palette, width: int | None = None
) -> Text:
    """The block's rows, each led by a newline, every slot starting in one
    display column on all of them. A slot is as wide as its widest cell over
    every row that fills it; a free-text usage note sizes nothing, and a slot
    empty on every row takes no column at all. The status pieces and `where`
    are one note cell, joined by a dot, `where` last; the login note ends the row.

    Over `width`, the block takes ONE form so its columns stay aligned. In
    this order it gives up: the login, to its short form; the email slot, cut
    to a shared width no narrower than EMAIL_FLOOR; the long status texts for
    their short ones; the login; last, the note itself, cut to the room left.
    A form given up stays given up at every narrower width: the email slot
    keeps the width its cut left it and takes back no room a later step frees.
    The usage slot is never cut.
    A free-text note row counts in the fit unless it overflows even with no
    login and the email at its floor: that one never fits, so it is left out
    and carries no login."""
    sep = Text(" \u00b7 ", style=palette.muted)

    def note(row: CandidateRow, short: bool) -> Text:
        pieces = [
            (row.short.get(key) if short else None) or row.status[key]
            for key in STATUS_ORDER if key in row.status
        ]
        if row.where:
            # Each marker's leading glyph (`○ cloud` -> `○`).
            pieces.append(Text(" ").join(
                marker[: len(marker.plain.partition(" ")[0])]
                for marker in row.where.split(" \u00b7 ")
            ) if short else row.where)
        out = Text()
        for piece in pieces:
            if out:
                out.append_text(sep)
            out.append_text(piece)
        return out

    def login_in(row: CandidateRow, login: str | None) -> Text:
        """The row's login in the `login` form: `"full"`, `"short"`, or none."""
        if login == "full":
            return row.login
        return (row.login_short or row.login) if login else Text()

    def lines(
        email: int, short: bool, login: str | None = None, room: int | None = None
    ) -> list[Text]:
        """Every row at an email slot of at most `email` columns, its login in
        the `login` form; with `room`, a note is cut to end by that column."""
        grid = [
            [row.number, _cut(row.email, email), row.usage, note(row, short),
             login_in(row, login if ok else None)]
            for row, ok in zip(rows, fits)
        ]
        usage, notes = 2, 3
        widths = [2, 0, 0, 0, 0]  # the number is right-aligned, two wide at least
        used = [False] * len(widths)
        for row, cells in zip(rows, grid):
            for i, cell in enumerate(cells):
                used[i] |= bool(cell)
                if not (i == usage and row.note):
                    widths[i] = max(widths[i], cell.cell_len)
        out = []
        for cells in grid:
            line = Text()
            for i, cell in enumerate(cells):
                if i and not used[i]:
                    continue
                if room is not None and i == notes:
                    if (left := room - line.cell_len - 2) < 2:
                        continue
                    cell = _cut(cell, left)
                else:
                    cell = cell.copy()
                # A note wider than usage overflows on its own row; the next slot follows it.
                pad = max(0, widths[i] - cell.cell_len)
                if i:
                    cell.pad_right(pad)
                else:
                    cell.pad_left(pad)
                line.append("  ")  # the row's indent, then the gutter between slots
                line.append_text(cell)
            line.rstrip()
            out.append(line)
        return out

    limit = 10**9 if width is None else width

    widest = max((row.email.cell_len for row in rows), default=0)
    # A free-text note row that overflows without its login, even at the email
    # floor, is left out of the fit: its text is never cut, and it gives up its login.
    fits = [True] * len(rows)  # `lines` reads it; the probe below carries no login
    fits[:] = [
        not row.note or line.cell_len <= limit
        for row, line in zip(rows, lines(EMAIL_FLOOR, True))
    ]

    def over(block: list[Text]) -> int:
        return max((line.cell_len for ok, line in zip(fits, block) if ok), default=0) - limit

    # (login form, short status texts, email cut): each step keeps what the one
    # before gave up, the email slot at the width the last cut left it.
    email = widest
    for login, short, cut in (
        ("full", False, False),
        ("short", False, False),
        ("short", False, True),
        ("short", True, True),
        (None, True, True),
    ):
        shown = lines(email, short, login)
        if cut and over(shown) > 0:
            email = max(EMAIL_FLOOR, email - over(shown))
            shown = lines(email, short, login)
        if over(shown) <= 0:
            break
    else:
        shown = lines(EMAIL_FLOOR, True, room=limit)
    out = Text()
    for line in shown:
        out.append("\n")
        out.append_text(line)
    return out


class AutoScreen(Screen):
    BINDINGS = [
        Binding("l", "toggle_live", "Go live / dry-run"),
        Binding("t", "adjust_threshold", "Threshold"),
        Binding("s", "cycle_strategy", "Strategy"),
        Binding("c", "toggle_cache_aware", "Cache-aware"),
        Binding("left", "threshold_step(-1)", "-1%"),
        Binding("right", "threshold_step(1)", "+1%"),
        Binding("enter", "adjust_done", "Done"),
        Binding("d", "toggle_desktop_follows", "Desktop follows"),
        Binding("o", "toggle_connectors_follow_pin", "Connectors follow pin"),
        Binding("escape,q", "back", "Back"),
    ]

    app: "CswapApp"

    def __init__(self, *, start_live: bool = False) -> None:
        super().__init__()
        self._engine: AutoSwitchEngine | None = None
        self._settings = None
        # `cswap tui --auto` only. The engine starts LIVE without the modal
        # because the flag IS the consent, for that launch alone.
        self._start_live = start_live
        # Session-only threshold adjustment (t, then arrows). Never written
        # to settings.json — same memory-only precedent as the dry-run
        # toggle. ``_configured_threshold`` is the mount-time file value the
        # screen reverts to on exit; ``_entry_threshold`` is the value when
        # adjust mode was entered (wake/log only on a net change).
        self._adjusting = False
        self._configured_threshold: float | None = None
        self._entry_threshold: float | None = None
        # Session-only strategy override (s cycles best -> consume-first ->
        # dynamic). Same precedent as the threshold above: never written to
        # settings.json. ``_configured_strategy`` is the mount-time file
        # value the screen reverts to on exit, and the strategy the pin arms
        # cache-aware on: `c` and the summary's cache-aware segment follow
        # it, not the session copy.
        self._configured_strategy: str | None = None
        # The bar the engine's last poll leaves the ACTIVE at (the primary's 99
        # under `dynamic`, not 97); dropped on a session setting change.
        self._switch_bar: float | None = None

    def compose(self) -> ComposeResult:
        yield AccountsPanel(show_minis=False, id="auto-active-panel")
        with Vertical(id="auto-top"):
            with Horizontal(id="auto-title-row"):
                yield Static(" DRY-RUN ", id="mode-badge", classes="dry")
                yield Static("", id="auto-summary")
            yield Static("", id="candidates")
        yield RichLog(id="event-log", highlight=False, markup=False, wrap=True)
        yield Footer()

    # -- lifecycle ----------------------------------------------------------

    def on_mount(self) -> None:
        self.app.set_store_only(True)
        self._settings = load_settings(self.app.switcher.backup_dir)
        # threshold_pct AND auto_settings are loaded once at app startup;
        # sync both to the fresh file value (unmount reverts only the
        # session threshold adjustment, never this correction).
        self._configured_threshold = self._settings.threshold
        self.app.threshold_pct = proactive_switch_bar_pct(
            self._settings.strategy, self._settings.threshold
        )
        self.app.auto_settings = self._settings
        self._configured_strategy = self._settings.strategy
        # The weekly line is not session-adjustable, so this is only the same
        # sync-to-the-file correction; nothing to restore on unmount.
        self.app.weekly_threshold_pct = self._settings.weekly_threshold
        self._update_summary()
        self.watch(self.app, "snapshot", self._on_snapshot)
        self.watch(self.app, "refresh_status", self._on_refresh_status)
        self.watch(self.app, "theme", self._on_theme_change)
        # ONLY `cswap tui --auto` starts LIVE. Entering the view from the
        # menu always starts dry-run, because opening a view must never
        # begin switching accounts — and a persisted "yes" is not consent
        # for a launch nobody asked to be live. A setting used to be read
        # here too, so one confirmed "Go live" made every later menu visit
        # switch accounts unasked, on every machine sharing settings.json.
        self._start_engine(dry_run=not self._start_live)

    def on_unmount(self) -> None:
        if self._engine is not None:
            self._engine.stop("view closed")
        # A session threshold must not outlive the engine it steered: unpin
        # the poll planner and put the bar tick back on the file value.
        self.app.switcher.clear_poll_policy_inputs()
        if self._configured_threshold is not None:
            self.app.threshold_pct = proactive_switch_bar_pct(
                self._configured_strategy, self._configured_threshold
            )
        self.app.set_store_only(False)

    def _on_theme_change(self, _theme: str) -> None:
        self._update_summary()
        self._update_badge()
        snap = self.app.snapshot
        if snap is not None:
            self._on_snapshot(snap)

    def action_back(self) -> None:
        if self._adjusting:
            self._end_adjust()
            return
        self.app.pop_screen()

    # -- threshold adjust mode ------------------------------------------------

    def check_action(self, action: str, parameters: tuple) -> bool | None:
        if action in ("threshold_step", "adjust_done") and not self._adjusting:
            return False  # hidden and inert until adjust mode is armed
        if action == "toggle_cache_aware":
            return self._configured_strategy == "dynamic"
        return True

    def action_adjust_threshold(self) -> None:
        if self._adjusting:
            self._end_adjust()
            return
        self._adjusting = True
        self._entry_threshold = self._settings.threshold
        self._update_summary()
        self.refresh_bindings()

    def action_adjust_done(self) -> None:
        if self._adjusting:
            self._end_adjust()

    def action_threshold_step(self, delta: float) -> None:
        if not self._adjusting:
            return
        spec = SETTING_SPECS["autoswitch.threshold"]
        value = min(spec.hi, max(spec.lo, self._settings.threshold + delta))
        self._set_threshold(value)

    def action_toggle_desktop_follows(self) -> None:
        self._toggle_pin_switch("desktop_follows", "pin.desktopFollows", "desktop follows")

    def action_toggle_connectors_follow_pin(self) -> None:
        self._toggle_pin_switch(
            "connectors_follow_pin", "pin.connectorsFollowPin", "connectors follow pin"
        )

    def _toggle_pin_switch(self, field: str, key: str, label: str) -> None:
        # Persisted, unlike `t`: the pin is another process and reads the saved setting.
        # Flips the file's value now, not what this screen last saw.
        from claude_swap.settings import load_pin_settings, set_setting

        backup_dir = self.app.switcher.backup_dir
        value = not getattr(load_pin_settings(backup_dir), field)
        try:
            set_setting(backup_dir, key, "true" if value else "false")
        except Exception as exc:  # best-effort, like the theme; never crash the UI
            self.notify(f"Could not save {label}: {exc}", severity="warning")
            return
        self._update_summary()
        self.query_one("#event-log", RichLog).write(
            Text(
                f"{label} {'on' if value else 'off'}",
                style=Palette.from_theme(self.app.current_theme).muted,
            )
        )

    def _end_adjust(self) -> None:
        self._adjusting = False
        self._update_summary()
        self.refresh_bindings()
        if self._settings.threshold == self._entry_threshold:
            return  # no net change: nothing to announce, no tick to force
        if self._engine is not None:
            self._engine.wake()  # show a decision at the new value now
        self.query_one("#event-log", RichLog).write(
            Text(
                f"— threshold set to {pct_label(self._settings.threshold)}% "
                "for this session —",
                style=Palette.from_theme(self.app.current_theme).muted,
            )
        )

    def _set_threshold(self, value: float) -> None:
        if value == self._settings.threshold:
            return
        if self._switch_bar == self._settings.threshold:
            self._switch_bar = None  # it was tracking the threshold; next poll resets
        self._settings = replace(self._settings, threshold=value)
        if self._engine is not None:
            self._engine.apply_threshold(value)
        self.app.threshold_pct = proactive_switch_bar_pct(
            self._settings.strategy, value
        )
        self.query_one("#auto-active-panel", AccountsPanel).refresh()
        self._update_summary()
        # The Next best list ranks by the lines in force, so it must follow
        # the session value now rather than at the next snapshot: under a
        # weekly line the order and the 5h rows' "threshold N%" both move.
        self._on_snapshot(self.app.snapshot)

    def action_cycle_strategy(self) -> None:
        # Session-only, exactly like `t`/threshold above: never written to
        # settings.json, reverted to the file value on unmount.
        # A configured strategy the cycle does not offer (priority-consume-first)
        # starts it over.
        strategy = self._settings.strategy
        current = _STRATEGY_CYCLE.index(strategy) if strategy in _STRATEGY_CYCLE else -1
        value = _STRATEGY_CYCLE[(current + 1) % len(_STRATEGY_CYCLE)]
        self._settings = replace(self._settings, strategy=value)
        self._switch_bar = None
        # A binding may be gated on strategy; the footer reads check_action
        # only on refresh.
        self.refresh_bindings()
        self.app.threshold_pct = proactive_switch_bar_pct(
            value, self._settings.threshold
        )
        if self._engine is not None:
            self._engine.apply_strategy(value)
            self._engine.wake()  # show a decision under the new strategy now
        self.query_one("#auto-active-panel", AccountsPanel).refresh()
        self._update_summary()
        self.query_one("#event-log", RichLog).write(
            Text(
                f"— strategy set to {value} for this session —",
                style=Palette.from_theme(self.app.current_theme).muted,
            )
        )

    def action_toggle_cache_aware(self) -> None:
        # Persisted, unlike `t`: the pin is another process and reads the
        # file. Nothing for the engine to apply (it never reads this).
        value = not load_settings(self.app.switcher.backup_dir).cache_aware
        try:
            set_setting(
                self.app.switcher.backup_dir, "autoswitch.cacheAware",
                "true" if value else "false",
            )
        except Exception as exc:  # best-effort, like the theme; never crash the UI
            self.notify(f"Could not save cache-aware: {exc}", severity="warning")
            return
        self._settings = replace(self._settings, cache_aware=value)
        self._update_summary()
        self.query_one("#event-log", RichLog).write(
            Text(
                f"— cache-aware {'on' if value else 'off'} —",
                style=Palette.from_theme(self.app.current_theme).muted,
            )
        )

    def _update_summary(self) -> None:
        palette = Palette.from_theme(self.app.current_theme)
        text = Text()
        # Leading: #auto-title-row is one line and clips the end of the text.
        if self.app.refresh_status:
            text.append(f"{self.app.refresh_status} · ", style=palette.sev_warn)
        text.append("auto-switch · ")
        bar = proactive_switch_bar_pct(
            self._settings.strategy, self._settings.threshold
        )
        # The bar shown is the engine's (`_switch_bar`) once a poll has set it.
        shown = bar if self._switch_bar is None else self._switch_bar
        # The field is the bar in force: it prints only when the threshold
        # IS the bar (`shown == threshold`), plus while it is being adjusted
        # (`t` + arrows) -- that is the number being changed.
        show_threshold = self._adjusting or shown == self._settings.threshold
        if show_threshold:
            text.append(
                f"threshold {pct_label(self._settings.threshold)}%",
                style=palette.accent if self._adjusting else "",
            )
            if self._settings.threshold != self._configured_threshold:
                text.append(" (session)", style=palette.muted)
        if shown != self._settings.threshold:
            text.append(f"{' · ' if show_threshold else ''}switch at {pct_label(shown)}%")
        text.append(f" · {self._settings.strategy}")
        if self._settings.strategy != self._configured_strategy:
            text.append(" (session)", style=palette.muted)
        if self._configured_strategy == "dynamic":
            text.append(f" · cache-aware {'on' if self._settings.cache_aware else 'off'}")
        if self._settings.weekly_threshold is not None:
            text.append(
                f" · weekly {pct_label(self._settings.weekly_threshold)}%"
            )
        text.append(f" · poll every {self._settings.interval_seconds:.0f}s")
        from claude_swap.settings import load_pin_settings

        follows = load_pin_settings(self.app.switcher.backup_dir)
        text.append(f" · desktop follows {'on' if follows.desktop_follows else 'off'}")
        text.append(f" · connectors follow pin {'on' if follows.connectors_follow_pin else 'off'}")
        if self._adjusting:
            text.append("   ← → adjust · enter done", style=palette.muted)
        self.query_one("#auto-summary", Static).update(text)

    # -- engine -------------------------------------------------------------

    def _start_engine(self, *, dry_run: bool) -> None:
        engine = AutoSwitchEngine(
            self.app.switcher,
            self._settings,
            self._emit_from_thread,
            dry_run=dry_run,
        )
        self._engine = engine
        # A LIVE request the engine could not honor: another LIVE engine holds
        # the machine's lock. Report what actually started, not what was asked
        # for — the badge reads engine.dry_run, so it is already right.
        dry_run = engine.dry_run
        self.run_worker(
            engine.run_loop,
            thread=True,
            group="engine",
            exit_on_error=False,
            name=f"auto-engine-{'dry' if dry_run else 'live'}",
        )
        self._update_badge()
        log = self.query_one("#event-log", RichLog)
        mode = "DRY-RUN (watching only)" if dry_run else "LIVE (will switch accounts)"
        log.write(
            Text(
                f"— engine started: {mode} —",
                style=Palette.from_theme(self.app.current_theme).muted,
            )
        )

    def _emit_from_thread(self, event: AutoSwitchEvent) -> None:
        """Engine ``on_event`` callback — runs on the worker thread."""
        try:
            self.app.call_from_thread(self._on_engine_event, event)
        except Exception:
            # App/screen tearing down mid-tick; the event has nowhere to go.
            pass

    def _on_engine_event(self, event: AutoSwitchEvent) -> None:
        if not self.is_attached:
            return
        palette = Palette.from_theme(self.app.current_theme)
        self.query_one("#event-log", RichLog).write(event_text(event, palette=palette))
        # The engine can PROMOTE itself mid-run: a demotion is a contention
        # answer, and the holder eventually exits. Nothing else re-reads
        # `dry_run` after mount, so the badge would keep saying DRY-RUN over a
        # live engine — worse than the stuck-dry-run it fixes, because now the
        # display disagrees with what is actually switching accounts.
        self._update_badge()
        if isinstance(event, PollEvent) and event.strategy == self._settings.strategy:
            self._switch_bar = event.switch_bar
            self._update_summary()
        if event.kind == "switch":
            self._switch_bar = None  # the old active's bar; the next poll sets it
            self._update_summary()
            self.app.request_refresh()

    def action_toggle_live(self) -> None:
        if self._engine is None:
            return
        if self._engine.dry_run:
            self.app.push_screen(
                ConfirmModal(
                    "Go live? claude-swap will switch your active account "
                    "automatically when the switch point is reached.\n\n"
                    "(Same behavior as running `cswap auto` in a terminal.)",
                    title="Go live",
                    yes_label="Go live",
                ),
                self._on_live_confirm,
            )
        else:
            self._restart_engine(dry_run=True)

    def _on_live_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            self._restart_engine(dry_run=False)

    def _restart_engine(self, *, dry_run: bool) -> None:
        if self._engine is not None:
            self._engine.stop("switched to dry-run" if dry_run else "switched to LIVE")
        self._start_engine(dry_run=dry_run)

    def _update_badge(self) -> None:
        badge = self.query_one("#mode-badge", Static)
        if self._engine is not None and not self._engine.dry_run:
            badge.update(" LIVE ")
            badge.set_classes("live")
        else:
            badge.update(" DRY-RUN ")
            badge.set_classes("dry")

    # -- candidates -----------------------------------------------------------

    def _on_refresh_status(self, _status: str) -> None:
        self._update_summary()

    def _on_snapshot(self, snap: AccountsSnapshot | None) -> None:
        if snap is None:
            return
        self._last_active_at = data.read_last_active_at(self.app.switcher.backup_dir)
        self.query_one("#candidates", Static).update(
            self._candidates_text(snap, active_number=snap.active_number)
        )

    def _candidates_text(
        self, snap: AccountsSnapshot, active_number: str | None
    ) -> Text:
        """Switch targets ranked the way the engine's strategy would rank them,
        by headroom against each window's own switch line."""
        # Same window set AND the same lines as the engine (autoswitch.model,
        # each account's own line (its policy threshold, else the strategy's
        # bar), the weekly line, and that bar as the ruler every account is
        # scored on) -- for the ranking KEY.
        # Not a guarantee the order agrees end to end: see below, the
        # engine's own tick can rank on a different axis.
        palette = Palette.from_theme(self.app.current_theme)
        models = parse_model_names(self._settings.model) if self._settings else ()
        settings = self._settings or AutoSwitchSettings()
        weekly = settings.weekly_threshold

        def noted(acc) -> bool:
            return weekly is not None or acc.policy.threshold is not None

        # THE STRATEGY'S BAR, EVERY ROW'S FALLBACK (#321; see `row_bar`): under
        # `dynamic` this is `proactive_switch_bar_pct`'s 97, not the raw
        # `settings.threshold` — a candidate with hours left on its reset
        # is headroom `dynamic` exists to spend, not a blocked one. Every
        # other strategy gets `settings.threshold` back unchanged, so this
        # is a no-op for them. Off `settings` (never `self._settings`
        # directly, which can be `None` before `on_mount` loads it) so the
        # bar can never read a DIFFERENT strategy than the ranking below.
        bar = proactive_switch_bar_pct(settings.strategy, settings.threshold)
        bars = data.switch_bars(snap)
        ranked: list[tuple[tuple, str]] = []  # (sort key, number)
        rows: dict[str, CandidateRow] = {}
        # The badge rides on that account's own row rather than the summary
        # line: naming the pin separately makes you match an email against the
        # list directly below it instead of just reading the list.
        pinned_identity = pin.pinned_identity(self.app.switcher)
        # ONCE PER RENDER, not once per row: this reads the daemon's record off
        # disk, and the badge below is drawn inside the account loop. The same
        # mistake was fixed for the pin lookup itself — see the test that pins
        # its call count.
        pin_applying = pin.pin_is_applying(self.app.switcher) if pinned_identity else None
        now = time.time()
        # THE engine's own admission and order -- `ordered_accounts` (data.py)
        # reads off this exact same call for every other listing screen.
        # `getattr` (never plain `self._last_active_at`): a test instance
        # built via `AutoScreen.__new__` skips `__init__`/`_on_snapshot`.
        last_active_at = getattr(self, "_last_active_at", None) or {}
        reasons: dict[str, str] = {}   # the engine's own cause per refused row
        engine = getattr(self, "_engine", None)
        hold: dict[str, str] = {}   # the engine's own refusal when it picks nobody (`hold_out`)
        ordered, rank_axis, trigger, unmodeled = data.rank_switch_candidates(
            snap, settings, now, active_number,
            probe_cooldown=engine._last_probe_cooldown if engine is not None else None,
            last_active_at=last_active_at,
            reasons_out=reasons,
            # The engine's own verdict, a 5xx burst read off the pin's trace that no snapshot carries (T3433)
            overloaded=engine is not None and active_number is not None
            and getattr(engine, "overloaded_account", None) == active_number,
            hold_out=hold,
        )
        ordered_rank = {num: i for i, num in enumerate(ordered)}
        # Captured before the loop rebinds `now` below (per-row, for the
        # window chips) -- the fallback key must agree with the SAME `now`
        # the admission pass above just ran on, not a fresh read.
        admission_now = now
        # Chip columns are keyed by WINDOW NAME, never by position: two rows
        # can have different-length window lists built from that account's
        # own payload (`relevant_windows`), so "chip 1" is not the same
        # window across rows. Computed once, before any row is drawn, over
        # exactly the accounts that will reach the chip branch below
        # (switchable, no sentinel, a known binding pct).
        row_windows: dict[str, list[tuple[str, float, str | None]]] = {}
        for acc in snap.accounts:
            if (
                acc.number == active_number
                or not acc.switchable
                or acc.usage.sentinel is not None
                or binding_pct(acc.usage.last_good, models) is None
                # an unranked no-plan row draws no chips (below), so it sizes no column
                or (data.is_no_plan(acc) and acc.number not in ordered_rank)
            ):
                continue
            row_windows[acc.number] = oauth.relevant_windows(oauth.display_usage(acc.usage.last_good), models)  # #199: no chip for a 5h window that is not running
        # The column set is the union of window names over every chip row, in
        # first-seen order; a row without a window gets a blank cell, so no
        # later column moves. (label text, pct text) widths per column; the
        # threshold note is the row's last cell, so it takes no column.
        chip_width: dict[str, tuple[int, int]] = {}
        for acc in snap.accounts:
            windows = row_windows.get(acc.number)
            if windows is None:
                continue
            for label, wpct, resets_at in windows:
                label_w, pct_w = chip_width.get(label, (0, 0))
                chip_width[label] = (
                    max(label_w, cell_len(data.chip_label(
                        label,
                        data.reset_text(
                            {"resets_at": resets_at}, now, acc.usage.fetched_at,
                            entry=acc.usage,
                        ),
                        wpct,
                    ))),
                    max(pct_w, cell_len(f"{wpct:.0f}%")),
                )
        # 5h and 7d lead whatever row came first (sorted is stable, so the
        # rest keep first-seen order).
        chip_width = dict(sorted(
            chip_width.items(), key=lambda kv: {"5h": 0, "7d": 1}.get(kv[0], 2)
        ))

        def key_of(acc) -> tuple:
            """The engine's own rank when it picked the row, else the tier every
            other switchable row sorts in (`ordered_accounts` reads the same key)."""
            if acc.number in ordered_rank:
                return (0, ordered_rank[acc.number])
            return data.non_candidate_key(acc, admission_now)

        for acc in snap.accounts:
            if acc.number == active_number:
                continue
            # A slot with no stored login is still a place you can GO — that
            # is now how you fill one. It used to be dropped from this list
            # entirely, so a machine with the roster but not the credentials
            # showed two accounts here and five in the engine's own log. A row
            # that says why it cannot be picked beats a row that isn't there.
            if not acc.switchable:
                # From SENTINEL_NOTES, not written here: an API-key slot has no
                # login to restore, and the switch screen reads the same table,
                # so both surfaces must describe a slot identically.
                # The slot's OWN sentinel first: unswitchable is not always
                # "nothing stored". A backup that exists but could not be READ
                # reads USAGE_KEYCHAIN_UNAVAILABLE, and sending that slot to
                # `cswap add` overwrites a working stored grant. `kind` still
                # wins for api_key — the sentinel diverges from it behind a
                # locked keychain.
                why = data.sentinel_label(
                    USAGE_API_KEY if acc.kind == "api_key"
                    else acc.usage.sentinel or USAGE_NO_CREDENTIALS
                )
                rows[acc.number] = CandidateRow(
                    Text(acc.number, style=palette.muted),
                    Text(acc.email, style=palette.muted),
                    Text(why, style=palette.sev_warn),
                    note=True,
                )
                ranked.append(((1000.0,), acc.number))   # last: never a target
                continue
            no_plan = data.is_no_plan(acc)
            if no_plan and acc.number not in ordered_rank:
                # No subscription behind this login and the pass did not rank
                # it: chips would read as a candidate, so the reason replaces
                # them and the row goes behind even the unswitchable ones. A
                # RANKED one (`decision_value()` serves `last_good` for a
                # while after polls fail) is the engine's pick and keeps its
                # place below, the reason a marker after the chips.
                rows[acc.number] = CandidateRow(
                    Text(acc.number, style=palette.muted),
                    Text(acc.email, style=palette.muted),
                    Text(),
                    status={"no-plan": Text(_NO_PLAN_TEXT, style=palette.sev_warn)},
                    short={"no-plan": Text("no plan", style=palette.sev_warn)},
                )
                ranked.append(((1001.0,), acc.number))
                continue
            # THIS row's bar: its own threshold when it has one, as the engine
            # lands on it (`_rank_candidates_pass`), else the strategy's.
            row_bar = bars.get(acc.number, bar)
            binding = _binding_at(acc.usage.last_good, settings, models, row_bar)
            usage = Text()
            status: dict[str, Text] = {}
            short: dict[str, Text] = {}
            free_text = False
            if acc.usage.sentinel is not None:
                free_text = True
                usage.append(
                    data.sentinel_label(acc.usage.sentinel), style=palette.muted
                )
                # `ordered_rank` first: the API-key last resort (data.rank_
                # switch_candidates) can name THIS row as the engine's actual
                # next pick, and it must not sort behind every refused row.
                ranked.append((key_of(acc), acc.number))
            elif binding is None:
                # An extra-usage (pay-as-you-go) account has no 5h/7d window,
                # so binding_window answers None — but it is not unknown, it has
                # a SPEND budget, and the watch screen already renders it.
                # `relevant_windows` excludes spend on purpose (a separate
                # axis from a rate-limit window), so this row was the only
                # place the same account read two different ways.
                spend = spend_row(
                    usage_rows(
                        acc.usage.last_good, now, acc.usage.fetched_at,
                        entry=acc.usage,
                    )
                )
                if spend is not None:
                    _label, spend_pct, spend_suffix, _full = spend
                    usage.append("$$ ", style=palette.muted)
                    usage.append(f"{spend_pct:.0f}%",
                                 style=palette.severity(spend_pct))
                    usage.append(f" · {spend_suffix}", style=palette.muted)
                else:
                    free_text = True
                    usage.append("usage unknown", style=palette.muted)
                # A spend-axis account is never a ranking target regardless
                # of `acc.disabled` (`relevant_windows` excludes spend, so
                # `_rank` can't see it) -- but every OTHER row here says why
                # it is never chosen, and this was the one silent exception.
                if acc.disabled:
                    status["disabled"] = Text("auto-swap off", style=palette.muted)
                    short["disabled"] = Text("off", style=palette.muted)
                # RANKED LAST, UNLESS THE API-KEY LAST RESORT NAMED IT:
                # spend is not headroom, so folding it into the sort key
                # would change which account the engine picks -- but when
                # `ordered_rank` already names this row (never derived here),
                # it IS the engine's pick and must not sort behind a row the
                # engine refused.
                ranked.append((key_of(acc), acc.number))
            else:
                # Per-window chips, from the same helper the dashboard uses
                # (data.chip_label) so one account cannot read two ways. The
                # SAME relevant_windows call feeds the label below, so the
                # chips and the label can never disagree on which windows
                # exist for this account.
                windows = row_windows[acc.number]
                fetched_at = acc.usage.fetched_at
                # Computed once per window so the chip and the block label
                # below read the exact same reset, never two separate calls
                # that could drift.
                chips = [
                    (
                        label, wpct,
                        data.reset_text(
                            {"resets_at": resets_at}, now, fetched_at,
                            entry=acc.usage,
                        ),
                    )
                    for label, wpct, resets_at in windows
                ]
                have = {label: (wpct, reset) for label, wpct, reset in chips}
                shown = False
                for i, (label, (label_w, pct_w)) in enumerate(
                    chip_width.items() if windows else ()
                ):
                    sep = "" if i == 0 else " · " if shown else "   "
                    if label not in have:
                        usage.append(" " * (len(sep) + label_w + pct_w))
                        continue
                    wpct, reset = have[label]
                    label_text = data.chip_label(label, reset, wpct)
                    pct_text = f"{wpct:.0f}%"
                    usage.append(sep, style=palette.muted)
                    # Label right-aligned so the `:` shares a column (owner,
                    # 2026-09-15); pct right-aligned so the `%` does.
                    usage.append(
                        " " * (label_w - cell_len(label_text)) + label_text
                        + " " * (pct_w - cell_len(pct_text)),
                        style=palette.muted,
                    )
                    usage.append(pct_text, style=palette.severity(wpct))
                    shown = True
                # A candidate whose LAST poll failed (an active backoff, a
                # run of failures; a success clears both) must not present
                # its cached figures as live. Not `fresh()`'s 180 s TTL: a
                # healthy row's `fetched_at` is older than that for most of
                # every poll cycle, and the engine lands on it happily.
                if acc.usage.in_backoff(now) or acc.usage.consecutive_failures:
                    # A lapsed plan is a permanent 403, not a poll gap: the
                    # no-plan marker below speaks for it, never "stale".
                    if not no_plan:
                        status["stale"] = Text("usage out of date", style=palette.sev_warn)
                        short["stale"] = Text("stale", style=palette.sev_warn)
                if no_plan:
                    status["no-plan"] = Text(_NO_PLAN_TEXT, style=palette.sev_warn)
                    short["no-plan"] = Text("no plan", style=palette.sev_warn)
                # WHAT blocks this candidate, not just the raw chips: a 5h/7d
                # window (no model choice escapes it) reads differently from
                # a model-only block (the engine's fallback ranks around it).
                # The CLASSIFICATION (open/model/full) reads the same way
                # here as in the decision log — same helper,
                # `classify_candidate_block`, the same row bar the engine's
                # own landing gate reads (#321). It still does
                # not fold in the engine's other, separate, stricter landing
                # floor -- `cold_switch_cost_pct` in autoswitch.py -- a
                # candidate clearing this label can still fail that one. The
                # two print `kind == "full"` in different words on purpose
                # (the elif below): the log always says "full", this panel
                # reserves "full" for a window actually at or over 100 and
                # says nothing otherwise. Always on `models`, the full pinned
                # set: this label explains why the row is not simply "open"
                # on the criteria the user actually configured, independent
                # of whether the pass above retried on the 5h/7d-only axis
                # for ORDERING purposes.
                kind, blocked_model = "open", None
                if self._settings:
                    # A window whose chip reads data.REFETCHING, or one of
                    # the retry/backoff markers reset_text names for the
                    # same provably-stale case (#325 follow-up — the chip
                    # no longer says REFETCHING forever, but the pct beside
                    # it is exactly as stale either way), has no opinion to
                    # contribute -- so it is dropped rather than zeroed: a
                    # zeroed pct would still be a fabricated measurement,
                    # never one the window actually reported (#325). Every
                    # OTHER reading `reset_text` returns is prefixed
                    # "resets " (a duration, "now") or is None (unknown) --
                    # the stale-placeholder family is the one shape that is
                    # neither, so that is what is excluded here rather than
                    # naming each marker.
                    kind, blocked_model = classify_candidate_block(
                        (
                            (label, p) for label, p, reset in chips
                            if reset is None or reset.startswith("resets ")
                        ),
                        row_bar,
                        weekly,
                    )
                    if kind == "model":
                        status["walled"] = Text(f"{blocked_model} limit reached", style=palette.muted)
                        short["walled"] = Text(f"{blocked_model} full", style=palette.muted)
                    elif kind == "full":
                        # A window at or over 100 is `<w> limit reached` (short `<w> full`); a window at or over its switch line but under
                        # 100 reads `limit <line>% reached` (short `<line>% reached`) when the pass above left the row out, the line the
                        # window is judged against (the strategy's bar, or the weekly line). The line alone refuses on the proactive path:
                        # at-limit, failover and a capped active skip the landing gate, and a fleet with every account over its line admits
                        # them, so a picked row reads nothing. Never on an unmodeled pass, which ranked nobody (T3328).
                        window_pct = next(
                            p for label, p, _ in windows if label == blocked_model
                        )
                        if window_pct >= 100.0:
                            status["block"] = Text(f"{blocked_model} limit reached", style=palette.muted)
                            short["block"] = Text(f"{blocked_model} full", style=palette.muted)
                        elif acc.number not in ordered_rank and not unmodeled:
                            line_pct = pct_label(window_line(blocked_model, row_bar, weekly))
                            status["at-line"] = Text(f"limit {line_pct}% reached", style=palette.muted)
                            short["at-line"] = Text(f"{line_pct}% reached", style=palette.muted)
                # Only when the binding window's chip was drawn above.
                if binding.label in have:
                    pct = _line_note(binding, binding.label, noted(acc))
                    if pct:
                        status["threshold"] = Text(f"limit {pct}", style=palette.muted)
                        short["threshold"] = Text(f"lim {pct}", style=palette.muted)
                capped = weekly_capped(acc.policy.weekly_cap, acc.usage.last_good)
                if acc.disabled:
                    status["disabled"] = Text("auto-swap off", style=palette.muted)
                    short["disabled"] = Text("off", style=palette.muted)
                elif (
                    acc.number not in ordered_rank
                    and kind == "open"
                    and not unmodeled
                    and not capped
                ):
                    # "open": nothing per-window blocks it, yet the engine's
                    # own pass still refused it -- a weekly reset later than
                    # the active's own, losing the hysteresis margin to a
                    # healthier peer, or ranking behind a sooner recovery.
                    # Every other excluded row already has a reason above,
                    # a capped one its "capped" label below.
                    # Never when `unmodeled`: this pass never ran, so
                    # "refused" is not a claim this row can support.
                    why = reasons.get(acc.number)
                    if why == "over the switch threshold":
                        # The engine's own words for a row over its line: the same piece, in the same words, as a window at its line above.
                        line_pct = pct_label(binding.line)
                        status["at-line"] = Text(f"limit {line_pct}% reached", style=palette.muted)
                        short["at-line"] = Text(f"{line_pct}% reached", style=palette.muted)
                    else:
                        words, short_words = _refusal_words(why)
                        status["not-candidate"] = Text(words, style=palette.muted)
                        short["not-candidate"] = Text(short_words, style=palette.muted)
                # Position from `ordered_rank` (the engine's own pass, called once
                # above), never a locally re-derived key; a row the pass never
                # ranked sorts by `key_of`'s tier.
                if capped:
                    # A veto, not a rank: the engine never lands here. A weekly cap vetoes the row, so it reads in the status as `limit <cap>% reached`
                    # (short `<cap>% reached`, the cap being the account's own `weekly_cap`), ONE reason: it takes the place of the threshold note and of
                    # the label that names the same window (a window block, the line, or a model wall: the stronger claim, not a second one beside it).
                    for name in ("threshold", *(("block", "at-line", "walled") if blocked_model == capped[0] else ())):
                        status.pop(name, None)
                        short.pop(name, None)
                    cap_pct = pct_label(acc.policy.weekly_cap)
                    status["capped"] = Text(f"limit {cap_pct}% reached", style=palette.sev_warn)
                    short["capped"] = Text(f"{cap_pct}% reached", style=palette.sev_warn)
                # ONE reason: a row at its switch line reads `limit <line>% reached` (either producer above), so the threshold note, the same number, goes.
                if "at-line" in status:
                    status.pop("threshold", None)
                    short.pop("threshold", None)
                ranked.append((key_of(acc), acc.number))
            # Outside the usage branches on purpose: an account whose usage is
            # unknown still owns the claude.ai side, so the badge must not hang
            # off whichever branch happened to run.
            where = Text()
            if pin.account_is_pinned(pinned_identity, acc.email, acc.org_uuid):
                # SET IS NOT APPLYING. This badge used to be lit by
                # the pin's presence alone, so it stayed green while the daemon
                # could not mint the pinned token and every request went out
                # unpinned, with the statusline and the coherence check
                # agreeing with it. `False` is the
                # only value worth shouting about; `None` means "could not
                # tell" and must read as healthy here, same rule as
                # `pin_is_broken`.
                if pin_applying is False:
                    where.append("⚠ cloud UNPINNED", style=f"bold {palette.sev_crit}")
                else:
                    where.append("○ cloud", style=f"bold {palette.sev_warn}")
            rows[acc.number] = CandidateRow(
                Text(acc.number, style=palette.foreground),
                Text(acc.email, style=palette.foreground),
                usage,
                status=status,
                short=short,
                where=where,
                login=Text(login_segment(acc, "full", gap=""), style=palette.muted),
                login_short=Text(login_segment(acc, "short", gap=""), style=palette.muted),
                note=free_text,
            )

        text = Text()
        # `rank_axis`: the engine's own report, never re-derived here.
        if rank_axis is not None:
            text.append(f"Next best ({rank_axis})", style=palette.muted)
        else:
            text.append("Next best", style=palette.muted)
        if not ranked:
            # Reached only when this is the sole account. Slots that cannot be
            # switched to are listed above with the reason, so "no other
            # accounts" is now literal rather than a filter's side effect.
            text.append("\n  no other accounts", style=palette.muted)
            return text
        if not ordered:
            reason = _UNMODELED_TEXT.get(trigger)
            if reason is not None:
                # The honest claim for WHY the pass never ran -- never
                # "no candidate qualifies", which claims one did.
                text.append(f"\n  {reason}", style=palette.muted)
            else:
                # DIFFERENT from "no other accounts": rows are still listed
                # below, each naming why -- but none is something a tick would
                # switch to, and ranking one anyway would name a top row the
                # engine could never pick.
                text.append(
                    f"\n  {_REFUSAL_TEXT.get(hold.get('refusal'), 'no candidate qualifies')}", style=palette.muted
                )
        # The box has no width before its first layout (the app's less its
        # padding), and a screen built with `__new__` has no `_nodes` to query.
        try:
            box = self.query_one("#candidates", Static)
        except AttributeError:
            width = None
        else:
            width = box.size.width or self.app.size.width - 4
        text.append_text(render_candidate_rows(
            [rows[number] for _key, number in sorted(ranked)], palette, width
        ))
        return text
