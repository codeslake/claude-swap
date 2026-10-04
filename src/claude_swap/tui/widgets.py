"""Shared render widgets: usage bars, account cards, and the accounts panel.

``bar_cells``/``usage_bar`` are custom renderers rather than Textual's
``ProgressBar`` because the design needs three things the stock widget
doesn't do: a severity color ramp, an optional threshold tick mark (the
auto-switch trigger line), and stale-measurement dimming.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Iterable, NamedTuple

from rich.text import Text
from textual.widgets import ListItem, Static

from claude_swap import oauth, pace, pin
from claude_swap.json_output import (
    USAGE_API_KEY,
    USAGE_FOREIGN_CREDENTIAL,
    USAGE_NO_CREDENTIALS,
    USAGE_RELOGIN_REQUIRED,
)
from claude_swap.models import AccountSnapshot
from claude_swap.switcher import ERROR_NOTES
from claude_swap.usage_store import STALE_OK_S, UsageEntry
from claude_swap.tui import data
from claude_swap.tui.theme import Palette

if TYPE_CHECKING:
    from claude_swap.tui.app import CswapApp

_BAR_FILLED = "━"
_BAR_HALF = "╸"
_BAR_EMPTY = "─"
_BAR_TICK = "┃"


def bar_cells(
    pct: float | None,
    width: int,
    *,
    stale: bool = False,
    threshold: float | None = None,
    palette: Palette = Palette.DARK,
) -> Text:
    """Just the bar glyphs: severity-colored fill, track, optional tick."""
    text = Text()
    if pct is None:
        text.append(_BAR_EMPTY * width, style=palette.track)
        return text
    frac = min(max(pct, 0.0), 100.0) / 100.0
    cells = frac * width
    full = int(cells)
    half = (cells - full) >= 0.5 and full < width
    tick_at: int | None = None
    if threshold is not None:
        tick_at = min(width - 1, max(0, round(threshold / 100.0 * width)))
    color = palette.severity(pct)
    fill_style = f"{color} dim" if stale else color
    for i in range(width):
        if tick_at is not None and i == tick_at:
            text.append(_BAR_TICK, style=palette.sev_warn)
        elif i < full:
            text.append(_BAR_FILLED, style=fill_style)
        elif i == full and half:
            text.append(_BAR_HALF, style=fill_style)
        else:
            text.append(_BAR_EMPTY, style=palette.track)
    return text


def usage_bar(
    label: str,
    pct: float | None,
    suffix: str | None,
    width: int,
    *,
    stale: bool = False,
    threshold: float | None = None,
    palette: Palette = Palette.DARK,
) -> Text:
    """One full bar line: ``5h ━━━━╸────┃──  47%  resets 2h 13m · 20:39``."""
    text = Text()
    text.append(f"{label} ", style=palette.muted)
    text.append(bar_cells(pct, width, stale=stale, threshold=threshold, palette=palette))
    if pct is None:
        text.append("  usage unknown", style=palette.muted)
    else:
        color = palette.severity(pct)
        text.append(f" {pct:3.0f}%", style=f"{color} dim" if stale else color)
    if suffix:
        text.append(f"  {suffix}", style=palette.muted)
    return text


def _reset_parts(
    window: dict, now: float, fetched_at: float | None = None,
    entry: UsageEntry | None = None,
) -> tuple[str, str]:
    """Countdown suffix and its clock-extended variant for one window.

    ``("resets 2h 13m", "resets 2h 13m · 20:39")`` — the second form is what
    a row shows when it has the width for it. Equal when no clock is known.

    An unknown reset used to return ``(None, None)`` and this row's suffix
    then went blank — the same "nothing to report" reading that hid it in
    the chips, on the account's OWN detail card this time. Named instead, so
    the reset column never disappears merely because it is unmeasured.

    ``entry``, when given, is what tells a rolled reset's "refetching" from
    its retry/backoff naming (see ``data.reset_text``) — this card's own
    copy of the same #325 follow-up the chips carry.
    """
    reset = data.reset_text(window, now, fetched_at, entry=entry)
    if not reset:
        return "reset unknown", "reset unknown"
    clock = data.reset_clock(window, now)
    return reset, f"{reset} · {clock}" if clock else reset


def _pace_suffix(window: dict, fetched_at: float | None) -> str:
    """"(ahead of pace)" when a weekly window is meaningfully ahead, else ""."""
    result = pace.compute_pace(window, fetched_at=fetched_at)
    return "(ahead of pace)" if result and result.ahead else ""


def usage_rows(
    last_good: dict | None, now: float, fetched_at: float | None = None,
    entry: UsageEntry | None = None,
) -> list[tuple[str, float, str, str]]:
    """(label, pct, suffix, suffix_full) rows mirroring the CLI's
    ``_format_usage_lines``.

    ``suffix_full`` extends the reset countdown with the absolute clock time
    (``resets 2h 13m · 20:39``) for rows that have room; otherwise it equals
    ``suffix``. Only windows the account actually has produce a row — an
    annual plan without a 7-day window simply has no 7d line. Order matches
    the CLI: spend, 5h, 7d, then per-model scoped windows (e.g. "Fable"),
    the latter marked ``(!)`` at/over their limit. The weekly (7d) and scoped
    rows also carry a "(ahead of pace)" marker when meaningfully ahead of the
    week's expected usage (issue #125) — never the 5h row.

    ``entry``, when given, is threaded to every ``_reset_parts`` call so a
    rolled reset names its retry/backoff wait instead of resting on
    "refetching" (#325 follow-up) — the same object as ``fetched_at`` came
    from, passed alongside it rather than in place of it.
    """
    if not isinstance(last_good, dict):
        return []
    rows: list[tuple[str, float, str, str]] = []
    spend = last_good.get("spend")
    if spend:
        amounts = f"${spend['used']:,.2f} / ${spend['limit']:,.2f}"
        # A monthly budget the server never reported a reset for has no
        # usage-window reset to name at all -- unlike 5h/7d/scoped, this
        # is not a gap in a real countdown, so it reads its own truth
        # (the amounts alone) instead of borrowing "reset unknown".
        suffix = suffix_full = amounts
        if spend.get("resets_at"):
            reset, reset_full = _reset_parts(spend, now, fetched_at, entry=entry)
            suffix, suffix_full = f"{reset}  {amounts}", f"{reset_full}  {amounts}"
        rows.append((SPEND_LABEL, float(spend["pct"]), suffix, suffix_full))
    for key, label in (("five_hour", "5h"), ("seven_day", "7d")):
        window = last_good.get(key)
        if window:
            reset, reset_full = _reset_parts(window, now, fetched_at, entry=entry)
            # A lapsed 5h window (no reported reset, no usage) has nothing
            # withheld -- "reset unknown" would assert a gap that isn't one.
            if key == "five_hour" and not window.get("resets_at") and window["pct"] == 0:
                reset = reset_full = "resets 5h"
            suffix, suffix_full = reset or "", reset_full or ""
            if key == "seven_day":
                marker = _pace_suffix(window, fetched_at)
                if marker:
                    suffix = f"{suffix}  {marker}" if suffix else marker
                    suffix_full = f"{suffix_full}  {marker}" if suffix_full else marker
            rows.append((label, float(window["pct"]), suffix, suffix_full))
    for window in last_good.get("scoped") or []:
        pct = float(window["pct"])
        suffix, suffix_full = _reset_parts(window, now, fetched_at, entry=entry)
        suffix, suffix_full = suffix or "", suffix_full or ""
        if pct >= 100:
            suffix = f"{suffix}  (!)" if suffix else "(!)"
            suffix_full = f"{suffix_full}  (!)" if suffix_full else "(!)"
        else:
            marker = _pace_suffix(window, fetched_at)
            if marker:
                suffix = f"{suffix}  {marker}" if suffix else marker
                suffix_full = f"{suffix_full}  {marker}" if suffix_full else marker
        rows.append((window["name"], pct, suffix, suffix_full))
    return rows


def pin_is_broken(acc: AccountSnapshot) -> bool:
    """Whether pinning to ``acc`` currently cannot produce a bearer.

    Only the states where the pinned account genuinely has no usable
    credential count. ``token expired`` deliberately does not: the proxy
    refreshes that itself, and flagging it would cry wolf on the normal case.
    ``keychain unavailable`` is a read problem on THIS process, not evidence
    about the credential, so it is left alone too — a warning that fires on
    "I could not look" teaches people to ignore warnings.
    """
    # An API-key account can never produce one: `sk-ant-api…` is not OAuth
    # JSON, so the provider returns None for every request and each one fails
    # open. `kind` rather than the sentinel — the sentinel is derived and reads
    # something else entirely for an unreadable backup blob or a locked macOS
    # keychain, while `kind` is the same fact set_pin refuses on.
    if getattr(acc, "kind", None) == "api_key":
        return True
    return acc.usage.sentinel in (
        USAGE_NO_CREDENTIALS,      # nothing stored for the slot
        USAGE_RELOGIN_REQUIRED,    # refresh lineage dead; only a human fixes it
        USAGE_FOREIGN_CREDENTIAL,  # the stored credential is another account's
    )


_LOGIN_LABEL = "login"


def _append_login_line(
    text: Text, acc: AccountSnapshot, label_width: int, now: float, palette: Palette
) -> None:
    """One "login <countdown>" row, label-padded like the card's other rows."""
    quarantined = acc.usage.sentinel == USAGE_RELOGIN_REQUIRED
    value = oauth.format_login_expiry(acc.login_expires_at, quarantined, now)
    text.append("\n    ")
    text.append(f"{_LOGIN_LABEL:<{label_width}} ", style=palette.muted)
    text.append(value, style=palette.muted)


def account_card_text(
    acc: AccountSnapshot,
    width: int,
    *,
    threshold: float | None = None,
    now: float | None = None,
    palette: Palette = Palette.DARK,
    cloud_pinned: bool = False,
    show_tag: bool = True,
) -> Text:
    """The full account card: header line + per-window bar rows.

    ``cloud_pinned`` marks the account that owns the claude.ai-side assets
    (Remote Control sessions, Artifacts). It is independent of ``is_active``
    — inference follows the active account while those stay pinned — so both
    badges can appear, on different accounts or the same one.
    """
    now = now if now is not None else time.time()

    text = Text()
    text.append(f"{acc.number:>2}  ", style=f"bold {palette.foreground}")
    if acc.alias:
        text.append(acc.alias, style=f"bold {palette.accent}")
        text.append(f" ({acc.email})", style=palette.foreground)
    else:
        text.append(acc.email, style=palette.foreground)
    if show_tag:
        text.append(f"  [{acc.display_tag}]", style=palette.muted)
    if acc.is_active:
        text.append("   ● active", style=f"bold {palette.accent}")
    if cloud_pinned:
        # Same marker shape as "● active" — the two are sibling states of one
        # account, and a lone glyph read as decoration next to the usage
        # figures rather than as a label.
        text.append("   ○ cloud", style=f"bold {palette.sev_warn}")
        if pin_is_broken(acc):
            # The pin is FAIL-OPEN: an account that cannot mint a bearer sends
            # RC and Artifacts back to whichever account is active, silently.
            # The account's own row already says "re-login needed", but the
            # cloud marker looked healthy right next to it — so the one place
            # that claims "your claude.ai side lives here" was the one place
            # not admitting it no longer does.
            text.append(" (not applying)", style=f"bold {palette.sev_crit}")
    if acc.disabled:
        text.append("   (disabled)", style=palette.muted)
    age = data.format_age(acc.usage.age_s)
    if age:
        text.append(f"   {age}", style=palette.muted)

    sentinel = acc.usage.sentinel
    if sentinel is not None:
        text.append("\n    ")
        style = palette.muted if sentinel == USAGE_API_KEY else palette.sev_warn
        marker = "·" if sentinel == USAGE_API_KEY else "⚠"
        text.append(f"{marker} {data.sentinel_label(sentinel)}", style=style)
        # Same supplementary line `cswap list` prints: the last good
        # measurement behind the sentinel (API-key accounts have no quota to
        # have "seen").
        if sentinel != USAGE_API_KEY:
            last_seen = data.last_seen_note(acc.usage)
            if last_seen is not None:
                text.append("\n    ")
                text.append(f"└ {last_seen}", style=palette.muted)
        _append_login_line(text, acc, len(_LOGIN_LABEL), now, palette)
        return text

    rows = usage_rows(acc.usage.last_good, now, acc.usage.fetched_at, entry=acc.usage)
    if not rows:
        text.append("\n    ")
        text.append("usage unavailable", style=palette.muted)
        if acc.usage.last_error:
            # Same wording as the CLI detail line: error KINDS with a
            # friendly note render it, so both surfaces describe the state
            # identically.
            note = ERROR_NOTES.get(acc.usage.last_error, acc.usage.last_error)
            text.append(f" · {note}", style=palette.muted)
        _append_login_line(text, acc, len(_LOGIN_LABEL), now, palette)
        return text

    stale = acc.usage.age_s is not None and acc.usage.age_s > STALE_OK_S
    label_width = max(
        [len(label) for label, _pct, _suffix, _full in rows] + [len(_LOGIN_LABEL)]
    )
    bar_width = max(12, min(30, width - 42 - label_width))
    # everything on a row except the suffix: indent, label, bar, " NNN%", gap
    row_overhead = 4 + label_width + 1 + bar_width + 5 + 2
    for label, pct, suffix, suffix_full in rows:
        # per-row: show the absolute clock only where it fits, so a long
        # spend row degrading doesn't cost the 5h/7d rows their clocks
        if suffix_full != suffix and row_overhead + len(suffix_full) <= width:
            suffix = suffix_full
        text.append("\n    ")
        text.append(
            usage_bar(
                f"{label:<{label_width}}",
                pct,
                suffix or None,
                bar_width,
                stale=stale,
                threshold=threshold,
                palette=palette,
            )
        )
    _append_login_line(text, acc, label_width, now, palette)
    return text


SPEND_LABEL = "$$"


def spend_row(rows: list[tuple]) -> tuple | None:
    """The pay-as-you-go spend row out of :func:`usage_rows`, or ``None``.

    Both compact surfaces render spend, so the label lives here rather than
    as a literal in each of them — the same reason `data.chip_label`
    exists for a window.
    """
    return next((r for r in rows if r[0] == SPEND_LABEL), None)


_MINI_TAG_CAP = 24  # widest "[tag]" a compact row shows, brackets included
_MINI_NOTE_CAP = 30  # widest note that widens the body column; longer is ellipsized
_MINI_GAP = "  "


class MiniWidths(NamedTuple):
    """Column widths the compact rows share, from :func:`mini_widths`."""

    name: int
    tag: int
    cells: dict[str, int]  # window label -> cell width, in display order
    note: int  # widest sentinel / "usage unknown" note shown, capped
    marker: int  # widest marker field shown, whichever kinds the rows carry
    chip: dict[str, int]  # window label -> widest chip label, so its pct starts at one column


def _fit(text: Text, width: int) -> Text:
    text.truncate(width, overflow="ellipsis", pad=True)
    return text


def _mini_name(acc: AccountSnapshot, palette: Palette) -> Text:
    name = Text()
    if acc.alias:
        name.append(acc.alias, style=f"bold {palette.accent}")
        name.append(f" ({acc.email})", style=palette.foreground)
    else:
        name.append(acc.email, style=palette.foreground)
    return name


def _mini_tag(acc: AccountSnapshot, palette: Palette) -> Text:
    tag = Text(acc.display_tag)
    tag.truncate(_MINI_TAG_CAP - 2, overflow="ellipsis")
    return Text.assemble("[", tag, "]", style=palette.muted)


def _mini_marks(
    acc: AccountSnapshot, ahead: bool, cloud_pinned: bool, palette: Palette
) -> Text:
    """The row's one trailing marker field: ``(ahead)``, ``○ cloud``,
    ``(not applying)``, ``(disabled)``, whichever apply, in that order."""
    marks = [Text("(ahead)", style=palette.sev_warn)] if ahead else []
    if cloud_pinned:
        # Labelled, like the full card: a bare glyph sitting between the
        # org tag and the usage figures read as decoration, not as a state.
        marks.append(Text("○ cloud", style=f"bold {palette.sev_warn}"))
        if pin_is_broken(acc):
            marks.append(Text("(not applying)", style=f"bold {palette.sev_crit}"))
    if acc.disabled:
        marks.append(Text("(disabled)", style=palette.muted))
    return Text(" ").join(marks)


def _mini_body(
    acc: AccountSnapshot, now: float, palette: Palette
) -> tuple[dict[str, Text], Text | None, bool]:
    """A compact row's window cells (label -> cell), or the note that stands in
    for them when it has no window to show, and whether the weekly window is
    ahead of pace (a row state, shown in the marker field, not in a cell)."""
    sentinel = acc.usage.sentinel
    if sentinel is not None:
        style = palette.muted if sentinel == USAGE_API_KEY else palette.sev_warn
        return {}, Text(data.sentinel_label(sentinel), style=style), False
    last_good = acc.usage.last_good
    if not isinstance(last_good, dict):
        last_good = {}
    fetched_at = acc.usage.fetched_at
    entry = acc.usage
    stale = acc.usage.age_s is not None and acc.usage.age_s > STALE_OK_S
    cells: dict[str, Text] = {}
    ahead = False
    for key, label in (("five_hour", "5h"), ("seven_day", "7d")):
        window = last_good.get(key)
        if not window:
            continue
        pct = float(window["pct"])
        color = palette.severity(pct)
        # Same chip the auto view's Next-best rows draw, from the same
        # helper — one account must not read two ways on two screens.
        cell = Text(
            data.chip_label(
                label, data.reset_text(window, now, fetched_at, entry=entry),
                pct,
            ),
            style=palette.muted,
        )
        cell.append(f"{pct:3.0f}%", style=f"{color} dim" if stale else color)
        if key == "seven_day":
            result = pace.compute_pace(window, fetched_at=fetched_at)
            ahead = bool(result and result.ahead)
        cells[label] = cell
    for window in last_good.get("scoped") or []:
        pct = float(window["pct"])
        color = palette.severity(pct)
        # Same chip helper the 5h/7d loop above uses — a scoped window reads
        # the same way whether it is the account's only window or sits
        # beside 5h/7d.
        cell = Text(
            data.chip_label(
                window["name"], data.reset_text(window, now, fetched_at, entry=entry)
            ),
            style=palette.muted,
        )
        cell.append(f"{pct:3.0f}%", style=f"{color} dim" if stale else color)
        if pct >= 100:
            cell.append(" (!)", style=palette.sev_crit)
        cells[window["name"]] = cell
    # Spend is a separate axis from a rate-limit window (never enters the
    # ranking — see oauth.relevant_windows) so it must show whether or not a
    # 5h/7d window already rendered above, not only as a last-resort fallback
    # when nothing else was shown; a budget can be 95% spent behind a window
    # that still reads perfectly healthy. From `usage_rows`, not a third
    # spelling of the same amounts.
    spend = spend_row(usage_rows(last_good, now, fetched_at, entry=entry))
    if spend is not None:
        _label, pct, suffix, _full = spend
        color = palette.severity(pct)
        cell = Text("$$ ", style=palette.muted)
        cell.append(f"{pct:.0f}%", style=f"{color} dim" if stale else color)
        cell.append(f" · {suffix}", style=palette.muted)
        cells[SPEND_LABEL] = cell
    # Nothing above rendered — every source `usage_rows` draws from
    # (spend, 5h, 7d, scoped) uses the same truthiness test as the loops
    # above, so `usage_rows` is provably empty here too.
    return cells, None if cells else Text("usage unknown", style=palette.muted), ahead


def _chip_len(cell: Text) -> int:
    """Where a window cell's pct starts: just past its chip's ``):`` (the end of
    every ``data.chip_label``), or 0 for a cell with no chip, like the spend one."""
    chip, sep, _pct = cell.plain.partition("):")
    return len(chip) + 2 if sep else 0


def _chip_pad(cell: Text, chip: int) -> Text:
    """``cell`` with blanks after its chip, so its pct starts ``chip`` cells in."""
    at = _chip_len(cell)
    if not at or at >= chip:
        return cell
    head, tail = cell.divide([at])
    return head.append(" " * (chip - at)).append(tail)


def mini_widths(
    accs: Iterable[AccountSnapshot],
    now: float,
    pinned_identity: tuple[str, str] | None = None,
) -> MiniWidths:
    """Column widths that fit every one of ``accs``, the compact rows shown.

    A window's cell is its widest chip (``chip``) plus the widest of what
    follows it, so every pct of that window starts at one column, a back-off
    chip (``7d(⟳429 30m):``) included. ``pinned_identity``
    (``pin.pinned_identity``) says which row carries the ``○ cloud`` marker,
    which the marker field's width counts.
    """
    palette = Palette.DARK  # widths do not depend on style
    name = tag = note = marker = 0
    chip = {"5h": 0, "7d": 0}  # fixed order; scoped windows follow as seen
    rest = dict(chip)
    for acc in accs:
        name = max(name, _mini_name(acc, palette).cell_len)
        tag = max(tag, _mini_tag(acc, palette).cell_len)
        row_cells, row_note, ahead = _mini_body(acc, now, palette)
        cloud = pin.account_is_pinned(pinned_identity, acc.email, acc.org_uuid)
        marker = max(marker, _mini_marks(acc, ahead, cloud, palette).cell_len)
        for label, cell in row_cells.items():
            at = _chip_len(cell)
            chip[label] = max(chip.get(label, 0), at)
            rest[label] = max(rest.get(label, 0), cell.cell_len - at)
        if row_note:
            note = max(note, min(row_note.cell_len, _MINI_NOTE_CAP))
    # spend is its own cell after every window, wherever it was first seen
    order = sorted(rest, key=lambda label: label == SPEND_LABEL)
    cells = {label: chip[label] + rest[label] for label in order if rest[label]}
    return MiniWidths(name, tag, cells, note, marker, chip)


def mini_account_text(
    acc: AccountSnapshot,
    now: float,
    *,
    palette: Palette = Palette.DARK,
    cloud_pinned: bool = False,
    widths: MiniWidths | None = None,
    width: int | None = None,
) -> Text:
    """One minimized line for an inactive account, in fixed columns.

    ``2  work@acme.dev  login 5h07m  [personal]  5h(⟳2h28m): 92%  7d(⟳3d04h): 63%  Fable(⟳?):100% (!)  $$ 40% · $4.00 / $10.00  (ahead) ○ cloud (disabled)``
    — slot, name, login, tag, one cell per window, the spend cell, then ONE
    trailing field of markers (``(ahead)``, ``○ cloud``, ``(not applying)``,
    ``(disabled)``, whichever apply, in that order). Pcts only, severity
    colored; each window reads as the same chip the auto view draws, padded
    so its pct starts at one column, and a maxed per-model window carries
    ``(!)``. A window the account lacks is a blank cell, and a sentinel
    state shows its label in place of the cells (ellipsized past
    ``_MINI_NOTE_CAP`` and the cells' width; the expanded card and the CLI
    carry the full text). Every row's body is as wide as the wider of the
    cells and the longest such note, so the marker field keeps one column.
    ``widths`` (from :func:`mini_widths` over the rows shown) lines the
    columns up across rows; alone, a row fits itself. A row wider than
    ``width`` is ellipsized before the marker field, which stays whole and
    starts where the widest field shown would, so the markers still share
    one column.
    """
    widths = widths or mini_widths([acc], now)
    cells, note, ahead = _mini_body(acc, now, palette)
    marks = _mini_marks(acc, ahead, cloud_pinned, palette)
    grid = Text(_MINI_GAP).join(
        _fit(_chip_pad(cells.get(label) or Text(), widths.chip[label]), width)
        for label, width in widths.cells.items()
    )
    quarantined = acc.usage.sentinel == USAGE_RELOGIN_REQUIRED
    login_value = oauth.format_login_expiry(acc.login_expires_at, quarantined, now)
    fields = [
        _fit(_mini_name(acc, palette), widths.name),
        # `format_login_expiry` already pads to a fixed width: one column
        Text(f"{_LOGIN_LABEL} {login_value}", style=palette.muted),
        _fit(_mini_tag(acc, palette), widths.tag),
        _fit(note or grid, max(grid.cell_len, widths.note)),
    ]
    text = Text(no_wrap=True, overflow="ellipsis")
    text.append(f"{acc.number:>2}  ", style=f"bold {palette.muted}")
    text.append(Text(_MINI_GAP).join(fields))
    tail = Text.assemble(_MINI_GAP, marks) if marks else Text()
    if not tail:
        text.rstrip()
    # a cut row reserves the widest marker field shown, so cut markers align too
    reserve = len(_MINI_GAP) + max(widths.marker, marks.cell_len) if marks else 0
    if width is not None and text.cell_len + reserve > width:
        room = width - reserve
        if room < 1:  # no room for the marker: cut the whole row
            tail, room = Text(), width
        # blank padding is no content: the ellipsis marks a real cut only
        text.rstrip()
        text.truncate(room, overflow="ellipsis", pad=True)
    return text.append(tail)


class AccountsPanel(Static):
    """Static account overview: the active full-size and pinned first,
    others as minis in the shared ranked order (``data.ordered_accounts``).
    The dashboard's — and with ``show_minis=False`` the auto screen's."""

    def __init__(self, *, show_minis: bool = True, id: str | None = None) -> None:
        super().__init__(id=id)
        self._show_minis = show_minis
        self._pinned_identity: "tuple[str, str] | None" = None

    def on_mount(self) -> None:
        self.watch(self.app, "snapshot", lambda _snap: self._resolve_and_refresh())
        # THEME ONLY REPAINTS. The pin did not move, so re-resolving it here
        # would put the package lookup back on a path that is not a snapshot.
        self.watch(self.app, "theme", lambda _t: self.refresh(layout=True))

    def _resolve_and_refresh(self) -> None:
        """Ask the pin ONCE per snapshot, like AccountCard's owner does.

        `render()` used to ask it. That is one call per repaint rather than
        the N `AccountCard` was making, so the arithmetic is milder — but the
        argument in that class's docstring is about WHEN `render()` fires, not
        how many widgets fire it: resize and reflow, not the 3s poll. This
        widget already watched `snapshot`, so the answer had a place to live
        and simply was not put there.
        """
        self._pinned_identity = pin.pinned_identity(self.app.switcher)
        self.refresh(layout=True)

    def render(self) -> Text:
        app: "CswapApp" = self.app  # type: ignore[assignment]
        palette = Palette.from_theme(app.current_theme)
        snap = app.snapshot
        if snap is None:
            return Text("loading…", style=palette.muted)
        if not snap.accounts:
            return Text(
                "No managed accounts yet.\n"
                "Use the menu below: Add account — from your current "
                "Claude Code login, or from a setup-token / API key.",
                style=palette.muted,
            )
        now = time.time()
        width = (self.size.width or 80) - 2
        minis = [a for a in snap.accounts if not a.is_active and self._show_minis]
        widths = mini_widths(minis, now, self._pinned_identity)
        blocks: list[Text] = []
        pinned_identity = self._pinned_identity
        by_number = {acc.number: acc for acc in snap.accounts}
        order = data.ordered_accounts(
            snap, app.auto_settings, now,
            data.read_last_active_at(app.switcher.backup_dir),
        )
        for number in order:
            acc = by_number[number]
            pinned = pin.account_is_pinned(pinned_identity, acc.email, acc.org_uuid)
            if acc.is_active:
                blocks.append(
                    account_card_text(
                        acc, width, threshold=app.threshold_pct, now=now,
                        palette=palette, cloud_pinned=pinned,
                        show_tag=app.show_org_tag,
                    )
                )
            elif self._show_minis:
                row = mini_account_text(
                    acc, now, palette=palette, widths=widths, width=width,
                    cloud_pinned=pinned, show_tag=app.show_org_tag,
                )  # never wraps
                blocks.append(row)
        if not blocks:
            return Text("no active managed login", style=palette.muted)
        text = Text()
        previous_multiline = False
        for i, block in enumerate(blocks):
            multiline = "\n" in block.plain
            if i:
                # breathe around the expanded active card
                text.append("\n\n" if (multiline or previous_multiline) else "\n")
            text.append(block)
            previous_multiline = multiline
        return text


class AccountCard(Static):
    """One account rendered full-size (used by the switch screen's list).

    ``cloud_pinned`` IS HANDED IN, not asked for. This used to call
    ``pin.pinned_identity`` from ``render()``, which is per-widget and not on the
    poll: it fires on every repaint, resize and reflow, so N accounts cost N
    package resolutions per frame (measured 450us each with the extra absent,
    the majority case). Steady state that is 0.15% of a 3s tick and would not
    be worth a line, but a held arrow key repaints at key-repeat rate and the
    same work becomes ~13% of a core. The panel beside this already resolves
    it once per render and passes it down; this is the same fact, resolved
    once per SNAPSHOT one frame further up.
    """

    def __init__(
        self, acc: AccountSnapshot, *, threshold: float | None = None,
        cloud_pinned: bool = False,
    ) -> None:
        super().__init__()
        self._acc = acc
        self._threshold = threshold
        self._cloud_pinned = cloud_pinned

    def set_account(self, acc: AccountSnapshot, *, cloud_pinned: bool = False) -> None:
        self._acc = acc
        self._cloud_pinned = cloud_pinned
        self.refresh(layout=True)

    def render(self) -> Text:
        return account_card_text(
            self._acc, self.size.width or 80, threshold=self._threshold,
            palette=Palette.from_theme(self.app.current_theme),
            cloud_pinned=self._cloud_pinned,
            show_tag=self.app.tag_shown(self._acc),
        )


class AccountItem(ListItem):
    """ListView row wrapping an :class:`AccountCard`; remembers its slot."""

    def __init__(self, acc: AccountSnapshot, *, cloud_pinned: bool = False) -> None:
        super().__init__(AccountCard(acc, cloud_pinned=cloud_pinned))
        self.number = acc.number
        self.email = acc.email

    def set_account(self, acc: AccountSnapshot, *, cloud_pinned: bool = False) -> None:
        self.number = acc.number
        self.email = acc.email
        self.query_one(AccountCard).set_account(acc, cloud_pinned=cloud_pinned)


class MenuItem(ListItem):
    """One menu row: a label plus an action id the screen dispatches on."""

    def __init__(self, label: str, action_id: str, *, muted: bool = False) -> None:
        item = Static(label, markup=False)
        if muted:
            item.add_class("menu-item-muted")
        super().__init__(item)
        self.action_id = action_id
