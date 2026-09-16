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
from dataclasses import replace
from typing import TYPE_CHECKING

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Footer, RichLog, Static

from claude_swap import oauth
from claude_swap.autoswitch import (
    CONSUME_FIRST_STRATEGIES,
    AutoSwitchEngine,
    AutoSwitchEvent,
    _classify_dynamic_trigger,
    _dynamic_active_headroom,
    _headroom_by_account,
    _model_window_binds_everywhere,
    binding_pct,
    classify_candidate_block,
    model_block_label,
    pct_label,
    proactive_switch_bar_pct,
    rank_candidates_pass,
)
from claude_swap import pin
from claude_swap.json_output import USAGE_API_KEY, USAGE_NO_CREDENTIALS
from claude_swap.models import AccountsSnapshot
from claude_swap.settings import (
    SETTING_SPECS,
    AutoSwitchSettings,
    load_settings,
    parse_model_names,
)
from claude_swap.tui import data
from claude_swap.tui.modals import ConfirmModal
from claude_swap.tui.theme import Palette
from claude_swap.tui.widgets import AccountsPanel, spend_row, usage_rows

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


class AutoScreen(Screen):
    BINDINGS = [
        Binding("l", "toggle_live", "Go live / dry-run"),
        Binding("t", "adjust_threshold", "Threshold"),
        Binding("s", "cycle_strategy", "Strategy"),
        Binding("left", "threshold_step(-1)", "-1%"),
        Binding("right", "threshold_step(1)", "+1%"),
        Binding("enter", "adjust_done", "Done"),
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
        # value the screen reverts to on exit.
        self._configured_strategy: str | None = None

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
        # The bar tick everywhere reads app.threshold_pct, loaded once at app
        # startup — sync it to the fresh file value so bars and engine agree,
        # and remember that value: unmount restores it (only the session
        # adjustment reverts, not this correction).
        self._configured_threshold = self._settings.threshold
        self.app.threshold_pct = proactive_switch_bar_pct(
            self._settings.strategy, self._settings.threshold
        )
        self._configured_strategy = self._settings.strategy
        self._update_summary()
        self.watch(self.app, "snapshot", self._on_snapshot)
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
            self._engine.stop()
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
        self._settings = replace(self._settings, threshold=value)
        if self._engine is not None:
            self._engine.apply_threshold(value)
        self.app.threshold_pct = proactive_switch_bar_pct(
            self._settings.strategy, value
        )
        self.query_one("#auto-active-panel", AccountsPanel).refresh()
        self._update_summary()

    def action_cycle_strategy(self) -> None:
        # Session-only, exactly like `t`/threshold above: never written to
        # settings.json, reverted to the file value on unmount.
        current = _STRATEGY_CYCLE.index(self._settings.strategy)
        value = _STRATEGY_CYCLE[(current + 1) % len(_STRATEGY_CYCLE)]
        self._settings = replace(self._settings, strategy=value)
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

    def _update_summary(self) -> None:
        palette = Palette.from_theme(self.app.current_theme)
        text = Text()
        text.append("auto-switch · ")
        text.append(
            f"threshold {pct_label(self._settings.threshold)}%",
            style=palette.accent if self._adjusting else "",
        )
        if self._settings.threshold != self._configured_threshold:
            text.append(" (session)", style=palette.muted)
        bar = proactive_switch_bar_pct(
            self._settings.strategy, self._settings.threshold
        )
        if bar != self._settings.threshold:
            text.append(f" · switch at {pct_label(bar)}%")
        text.append(f" · {self._settings.strategy}")
        if self._settings.strategy != self._configured_strategy:
            text.append(" (session)", style=palette.muted)
        text.append(f" · poll every {self._settings.interval_seconds:.0f}s")
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
        if event.kind == "switch":
            self.app.request_refresh()

    def action_toggle_live(self) -> None:
        if self._engine is None:
            return
        if self._engine.dry_run:
            self.app.push_screen(
                ConfirmModal(
                    "Go live? claude-swap will switch your active account "
                    "automatically when the threshold is reached.\n\n"
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
            self._engine.stop()
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

    def _on_snapshot(self, snap: AccountsSnapshot | None) -> None:
        if snap is None:
            return
        self.query_one("#candidates", Static).update(
            self._candidates_text(snap, active_number=snap.active_number)
        )

    def _candidates_text(
        self, snap: AccountsSnapshot, active_number: str | None
    ) -> Text:
        """Switch targets ranked the way the engine's strategy would rank them."""
        # Same window set as the engine (autoswitch.model included), so the
        # displayed ranking can never disagree with the account it picks.
        palette = Palette.from_theme(self.app.current_theme)
        models = parse_model_names(self._settings.model) if self._settings else ()
        settings = self._settings or AutoSwitchSettings()
        # Same strategy the engine ticks on, so the panel's order can never
        # disagree with the account a tick would actually switch to. Off
        # `settings` (never `self._settings` directly, which can be `None`
        # before `on_mount` loads it) so this can never read a DIFFERENT
        # strategy than the one `_trigger_for`/`_rank_on` below rank on.
        consume_first = settings.strategy in CONSUME_FIRST_STRATEGIES
        ranked: list[tuple[tuple, str]] = []  # (sort key, number)
        lines: dict[str, Text] = {}
        # The badge rides on that account's own row rather than the summary
        # line: naming the pin separately makes you match an email against the
        # list directly below it instead of just reading the list.
        pinned_identity = pin.pinned_identity(self.app.switcher)
        # ONCE PER RENDER, not once per row: this reads the daemon's record off
        # disk, and the badge below is drawn inside the account loop. The same
        # mistake was fixed for the pin lookup itself — see the test that pins
        # its call count.
        pin_applying = pin.pin_is_applying(self.app.switcher) if pinned_identity else None
        # Padded to the widest email among the rows THIS block renders, so
        # every row's chips start in the same column — computed from exactly
        # the accounts the loop below iterates (all but the active one).
        email_width = max(
            (len(acc.email) for acc in snap.accounts if acc.number != active_number),
            default=0,
        )
        now = time.time()
        # `switchable_account_numbers()` (switcher.py) drops `disabled` too.
        oauth_candidates = [
            acc.number
            for acc in snap.accounts
            if acc.number != active_number
            and acc.switchable
            and not acc.disabled
            and acc.kind != "api_key"
        ]

        def _trigger_for(
            active_headroom: float | None, active_disabled: bool
        ) -> str | None:
            # Mirrors `_tick_inner`'s own classification closely enough for
            # `_rank_candidates_pass`'s gates to engage the way they would
            # on a real tick -- an unrecognized literal (e.g. the bare
            # strategy name "best") skips every gate in the pass and admits
            # whatever is readable, unranked, which is the defect class
            # this refactor exists to close (#199). `None` means no trigger
            # this panel can derive reaches this pass on a real tick at
            # all; the caller ranks nothing rather than guess.
            #
            # FIRST AND UNCONDITIONAL, before any headroom reading --
            # `_tick_inner` checks `is_account_disabled(current)` ahead of
            # everything else (autoswitch.py:2290), so a disabled active
            # must win here even when its own headroom reads healthy.
            # Its own trigger name, not "at-limit"/"proactive": both
            # landing-health gates (3952, 4048) key on those literals, and
            # a disabled active is withdrawn from rotation, not merely
            # blocked -- any readable candidate should rank, not just a
            # healthy one.
            if active_disabled:
                return "disabled-active"
            if active_headroom is None:
                # `_tick_inner`'s idle-hold/eventual-failover branch, taken
                # regardless of strategy. The pass ranks fine with
                # `active_headroom=None` under "failover" -- only the
                # hysteresis-margin leg (`elif active_headroom is not
                # None`) is unreachable, not the landing gate itself
                # (`_every_account_above_threshold` already returns False
                # on an unknown active). `unhealthy_ticks` gates WHEN a
                # real tick acts on this, not what it would rank.
                return "failover"
            if settings.strategy == "dynamic":
                kind = _classify_dynamic_trigger(active_headroom)
                if kind != "at-limit":
                    # `_tick_inner` never reaches `_rank_candidates_pass`
                    # for "proactive"/"dynamic-healthy" under `dynamic` --
                    # both set `dynamic_ordered` via the separate warm/cold
                    # -tiered `_rank_dynamic_candidates` (needs
                    # `last_active_at` engine state this panel does not
                    # track). Only "at-limit" reaches this pass for
                    # `dynamic`; a wrong ranking for the other two is worse
                    # than none.
                    return None
                return "at-limit"
            if (
                settings.strategy in CONSUME_FIRST_STRATEGIES
                and (100.0 - active_headroom) < settings.threshold
            ):
                return settings.strategy
            # Not a real trigger -- nothing this panel can derive reaches
            # the pass under it on a real tick: a non-consume-first
            # strategy below the threshold never assigns a trigger at all
            # in the engine (it emits `below-threshold` and returns
            # `TickOutcome.NO_ACTION`, autoswitch.py:2314). "proactive" is
            # this panel's own answer to "what would it pick if it had
            # to", which is the panel's whole job -- there is simply no
            # engine trigger to mirror here.
            return "at-limit" if active_headroom <= 0 else "proactive"

        def _rank_on(
            axis: tuple[str, ...], trigger: str | None
        ) -> tuple[list[str], dict]:
            usage = {
                acc.number: acc.usage.decision_value(axis) for acc in snap.accounts
            }
            headroom = _headroom_by_account(usage, axis)
            active_headroom = headroom.get(active_number)
            if trigger is None:
                return [], usage
            ordered, *_rest = rank_candidates_pass(
                models=axis, trigger=trigger,
                consume_first=consume_first, oauth_candidates=oauth_candidates,
                no_return=None, usage=usage, headroom=headroom,
                current=active_number, active_headroom=active_headroom,
                settings=settings, now=now, entries=None,
                probe_cooldown=getattr(
                    getattr(self, "_engine", None), "_last_probe_cooldown", None
                ),
            )
            return ordered, usage

        # THE ENGINE'S OWN ADMISSION, not a re-derived predicate (#199): a
        # static pass call, needing no engine instance, for which candidates
        # a tick would consider and in what order.
        #
        # ONE TRIGGER, CLASSIFIED ONCE -- `_tick_inner` decides `trigger` at
        # a single site (autoswitch.py:2290-2368) and carries that one
        # literal into both its own `_rank_candidates_pass` calls through
        # the `kw` dict (3760, used at 3782 and 3803). Calling `_trigger_for`
        # separately per axis let the model-gated pass and the 5h/7d retry
        # disagree on what a real tick would classify as one decision.
        model_usage = {
            acc.number: acc.usage.decision_value(models) for acc in snap.accounts
        }
        model_headroom = _headroom_by_account(model_usage, models)
        active_account = next(
            (acc for acc in snap.accounts if acc.number == active_number), None
        )
        active_disabled = (
            active_account.disabled if active_account is not None else False
        )
        # Widened exactly like the engine's own `active_headroom` ahead of
        # classification (autoswitch.py:2246) -- `_rank_on` below keeps its
        # own per-axis, UNWIDENED read for `active_headroom=` (autoswitch.py
        # deliberately re-derives that per axis, 3771-3781: passing the
        # widened value through mixes a margin between two different axes).
        # Only the trigger classification widens.
        trigger = _trigger_for(
            _dynamic_active_headroom(
                settings, models, model_usage, active_number,
                model_headroom.get(active_number),
            ),
            active_disabled,
        )
        ordered, usage_by_account = _rank_on(models, trigger)
        # The SAME retry `_rank_candidates` makes (`dynamic` only): drop the
        # model set and re-rank on 5h/7d alone, but only when the model
        # window is what is actually blocking every account, active
        # included -- never a per-account "any row reads open" heuristic,
        # which ran for every strategy and could retry when the active
        # itself still had real model headroom (#199 review).
        if (
            not ordered and models and settings.strategy == "dynamic"
            and _model_window_binds_everywhere(
                usage_by_account, models, settings.threshold
            )
        ):
            ordered, usage_by_account = _rank_on((), trigger)
        ordered_rank = {num: i for i, num in enumerate(ordered)}
        # Chip columns are keyed by WINDOW NAME, never by position: two rows
        # can have different-length window lists built from that account's
        # own payload (`relevant_windows`), so "chip 1" is not the same
        # window across rows. Computed once, before any row is drawn, over
        # exactly the accounts that will reach the chip branch below
        # (switchable, no sentinel, a known binding pct).
        row_windows: dict[str, list[tuple[str, float, str | None]]] = {}
        chip_width: dict[str, int] = {}
        for acc in snap.accounts:
            if (
                acc.number == active_number
                or not acc.switchable
                or acc.usage.sentinel is not None
                or binding_pct(acc.usage.last_good, models) is None
            ):
                continue
            windows = oauth.relevant_windows(acc.usage.last_good, models)
            row_windows[acc.number] = windows
            for label, wpct, resets_at in windows:
                width = len(
                    data.chip_label(
                        label,
                        data.reset_text(
                            {"resets_at": resets_at}, now, acc.usage.fetched_at
                        ),
                        wpct,
                    )
                ) + len(f"{wpct:.0f}%")
                chip_width[label] = max(chip_width.get(label, 0), width)
        for acc in snap.accounts:
            if acc.number == active_number:
                continue
            # A slot with no stored login is still a place you can GO — that
            # is now how you fill one. It used to be dropped from this list
            # entirely, so a machine with the roster but not the credentials
            # showed two accounts here and five in the engine's own log. A row
            # that says why it cannot be picked beats a row that isn't there.
            if not acc.switchable:
                entry = Text()
                entry.append(f"\n  {acc.number:>2}  ", style=palette.muted)
                entry.append(f"{acc.email:<{email_width}}", style=palette.muted)
                # From SENTINEL_NOTES, not written here: an API-key slot has no
                # login to restore, and the switch screen reads the same table,
                # so both surfaces must describe a slot identically.
                # The slot's OWN sentinel first: unswitchable is not always
                # "nothing stored". A backup that exists but could not be READ
                # reads USAGE_KEYCHAIN_UNAVAILABLE, and sending that slot to
                # `cswap add` overwrites a working stored grant. `kind` still
                # wins for api_key — the sentinel diverges from it behind a
                # locked keychain.
                note = data.sentinel_label(
                    USAGE_API_KEY if acc.kind == "api_key"
                    else acc.usage.sentinel or USAGE_NO_CREDENTIALS
                )
                entry.append(f"  {note}", style=palette.sev_warn)
                lines[acc.number] = entry
                ranked.append(((1000.0,), acc.number))   # last: never a target
                continue
            pct = binding_pct(acc.usage.last_good, models)
            entry = Text()
            entry.append(f"\n  {acc.number:>2}  ", style=palette.foreground)
            entry.append(f"{acc.email:<{email_width}}", style=palette.foreground)
            if acc.usage.sentinel is not None:
                entry.append(
                    f"  {data.sentinel_label(acc.usage.sentinel)}", style=palette.muted
                )
                ranked.append(((998.0,), acc.number))
            elif pct is None:
                # An extra-usage (pay-as-you-go) account has no 5h/7d window,
                # so binding_pct answers None — but it is not unknown, it has
                # a SPEND budget, and the watch screen already renders it.
                # `relevant_windows` excludes spend on purpose (a separate
                # axis from a rate-limit window), so this row was the only
                # place the same account read two different ways.
                spend = spend_row(
                    usage_rows(acc.usage.last_good, now, acc.usage.fetched_at)
                )
                if spend is not None:
                    _label, spend_pct, spend_suffix, _full = spend
                    entry.append("  $$ ", style=palette.muted)
                    entry.append(f"{spend_pct:.0f}%",
                                 style=palette.severity(spend_pct))
                    entry.append(f" · {spend_suffix}", style=palette.muted)
                else:
                    entry.append("  usage unknown", style=palette.muted)
                # A spend-axis account is never a ranking target regardless
                # of `acc.disabled` (`relevant_windows` excludes spend, so
                # `_rank` can't see it) -- but every OTHER row here says why
                # it is never chosen, and this was the one silent exception.
                if acc.disabled:
                    entry.append("  auto-swap disabled", style=palette.muted)
                # RANKED LAST EITHER WAY. Spend is not headroom: folding it
                # into the sort key would change which account the engine
                # picks, and the ranking axis is not this row's to move.
                ranked.append(((999.0,), acc.number))
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
                    (label, wpct, data.reset_text({"resets_at": resets_at}, now, fetched_at))
                    for label, wpct, resets_at in windows
                ]
                for i, (label, wpct, reset) in enumerate(chips):
                    entry.append("  " if i == 0 else " · ", style=palette.muted)
                    label_text = data.chip_label(label, reset, wpct)
                    entry.append(label_text, style=palette.muted)
                    pct_text = f"{wpct:.0f}%"
                    entry.append(pct_text, style=palette.severity(wpct))
                    # Never on the LAST chip: nothing after it needs
                    # aligning, and padding it would leave trailing
                    # whitespace before end of line, the `-only`/`full`
                    # suffix or the pin badge.
                    if i < len(windows) - 1:
                        pad = chip_width[label] - len(label_text) - len(pct_text)
                        entry.append(" " * pad, style=palette.muted)
                if not windows:  # no window data at all — keep the old reading
                    entry.append(f"  {pct:3.0f}% used", style=palette.severity(pct))
                # A candidate whose LAST poll failed (an active backoff, a
                # run of failures; a success clears both) must not present
                # its cached figures as live. Not `fresh()`'s 180 s TTL: a
                # healthy row's `fetched_at` is older than that for most of
                # every poll cycle, and the engine lands on it happily.
                if acc.usage.in_backoff(now) or acc.usage.consecutive_failures:
                    entry.append("  stale", style=palette.sev_warn)
                # WHAT blocks this candidate, not just the raw chips: a 5h/7d
                # window (no model choice escapes it) reads differently from
                # a model-only block (the engine's fallback ranks around it),
                # and the two must read the same way here as in the decision
                # log — same helper, `classify_candidate_block`. Always on
                # `models`, the full pinned set: this label explains why the
                # row is not simply "open" on the criteria the user actually
                # configured, independent of whether `_rank_on` above has
                # retried on the 5h/7d-only axis for ORDERING purposes.
                kind = "open"
                if self._settings:
                    # A window whose chip just read data.REFETCHING has no
                    # opinion to contribute -- its pct provably predates its
                    # own reset -- so it is dropped rather than zeroed: a
                    # zeroed pct would still be a fabricated measurement,
                    # never one the window actually reported (#325).
                    kind, blocked_model = classify_candidate_block(
                        (
                            (label, p) for label, p, reset in chips
                            if reset != data.REFETCHING
                        ),
                        self._settings.threshold,
                    )
                    if kind == "model":
                        entry.append(
                            f"  {model_block_label(blocked_model)}",
                            style=palette.muted,
                        )
                    elif kind == "full":
                        entry.append(f"  {blocked_model} full", style=palette.muted)
                if acc.disabled:
                    entry.append("  auto-swap disabled", style=palette.muted)
                elif acc.number not in ordered_rank and kind == "open":
                    # "open": nothing per-window blocks it, yet the pass
                    # still dropped it -- a weekly reset later than the
                    # active's own (consume-first/dynamic), losing the
                    # hysteresis margin to a healthier peer (best), or
                    # ranking behind a sooner recovery when every account
                    # is at/over the threshold. Every other excluded row
                    # already has a reason from the block label above.
                    entry.append("  not a candidate", style=palette.muted)
                # Position from `ordered_rank` (the pass, called once
                # above), never a locally re-derived key.
                key = (
                    (0, ordered_rank[acc.number])
                    if acc.number in ordered_rank
                    else (1,)
                )
                ranked.append((key, acc.number))
            # Outside the usage branches on purpose: an account whose usage is
            # unknown still owns the claude.ai side, so the badge must not hang
            # off whichever branch happened to run.
            if pin.account_is_pinned(pinned_identity, acc.email, acc.org_uuid):
                entry.append("  · ", style=palette.muted)
                # SET IS NOT APPLYING. This badge used to be lit by
                # the pin's presence alone, so it stayed green while the daemon
                # could not mint the pinned token and every request went out
                # unpinned, with the statusline and the coherence check
                # agreeing with it. `False` is the
                # only value worth shouting about; `None` means "could not
                # tell" and must read as healthy here, same rule as
                # `pin_is_broken`.
                if pin_applying is False:
                    entry.append("⚠ cloud UNPINNED", style=f"bold {palette.sev_crit}")
                else:
                    entry.append("○ cloud", style=f"bold {palette.sev_warn}")
            lines[acc.number] = entry

        text = Text()
        # The order below is the engine's own admission, not a re-derived
        # read of the raw window pcts.
        text.append("Next best (engine order)", style=palette.muted)
        if not ranked:
            # Reached only when this is the sole account. Slots that cannot be
            # switched to are listed above with the reason, so "no other
            # accounts" is now literal rather than a filter's side effect.
            text.append("\n  no other accounts", style=palette.muted)
            return text
        if not ordered:
            # DIFFERENT from "no other accounts": rows are still listed
            # below, each naming why -- but none is something a tick would
            # switch to, and ranking one anyway would name a top row the
            # engine could never pick.
            text.append("\n  no candidate qualifies", style=palette.muted)
        for _key, number in sorted(ranked):
            text.append(lines[number])
        return text
