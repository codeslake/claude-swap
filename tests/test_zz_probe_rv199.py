"""TEMP review probe - renders through the real widgets, prints a table."""
import time

from tests.test_tui import make_account, make_entry, _iso_in
from claude_swap.tui.widgets import mini_account_text, account_card_text, usage_rows


CASES = [
    ("A window-only 5h+7d", dict(pct5=25.0, pct7=10.0)),
    ("B 5h+7d+spend", dict(pct5=25.0, pct7=10.0,
        spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
    ("C spend-only", dict(pct5=None, pct7=None,
        spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
    ("D maxedscoped+spend", dict(pct5=None, pct7=None, scoped=[("Fable", 100.0)],
        spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
    ("E scoped-only 99", dict(pct5=None, pct7=None, scoped=[("Fable", 99.0)])),
    ("F all 5h+7d+maxed+spend", dict(pct5=92.0, pct7=63.0, scoped=[("Fable", 100.0)],
        spend={"used": 1029.50, "limit": 2000.0, "pct": 51.45, "currency": "USD"})),
    ("G two scoped below cap", dict(pct5=None, pct7=None,
        scoped=[("Fable", 99.0), ("Opus", 40.0)])),
    ("H scoped maxed + below", dict(pct5=None, pct7=None,
        scoped=[("Fable", 100.0), ("Opus", 40.0)])),
    ("I 5h + scoped below", dict(pct5=10.0, pct7=None, scoped=[("Opus", 40.0)])),
    ("J nothing", dict(pct5=None, pct7=None)),
]


def test_probe_mini_widths():
    now = time.time()
    print("\n\n=== MINI LINE  (short email, no alias) ===")
    for tag, kw in CASES:
        acc = make_account(1, entry=make_entry(**kw))
        t = mini_account_text(acc, now)
        print(f"{tag:<26} len={len(t.plain):>4}  {t.plain!r}")

    print("\n=== MINI LINE  (realistic long email + alias) ===")
    for tag, kw in CASES:
        acc = make_account(3, entry=make_entry(**kw), alias="work",
                           email="junyong.lee@samsung-research.example")
        t = mini_account_text(acc, now)
        print(f"{tag:<26} len={len(t.plain):>4}  {t.plain!r}")

    print("\n=== CARD at width 100, same entries ===")
    for tag, kw in CASES:
        acc = make_account(1, entry=make_entry(**kw))
        print(f"{tag:<26} {account_card_text(acc, 100, now=now).plain!r}")


def test_probe_panel_render():
    from rich.text import Text
    from rich.console import Console

    now = time.time()
    entry = make_entry(pct5=92.0, pct7=63.0,
                       spend={"used": 1029.50, "limit": 2000.0, "pct": 95.0,
                              "currency": "USD"})
    mini = mini_account_text(
        make_account(3, entry=entry, alias="work",
                     email="junyong.lee@samsung-research.example"), now)
    print(f"\nmini.no_wrap={mini.no_wrap!r} mini.overflow={mini.overflow!r}")
    agg = Text()
    agg.append(mini)
    print(f"agg .no_wrap={agg.no_wrap!r} agg .overflow={agg.overflow!r}")

    con = Console(width=80, no_color=True, legacy_windows=False)
    for tag, obj in (("mini alone w80", mini), ("via aggregate w80", agg)):
        with con.capture() as cap:
            con.print(obj)
        print(f"--- {tag} ---\n{cap.get()!r}")

    entry2 = make_entry(pct5=92.0, pct7=63.0)
    mini2 = mini_account_text(
        make_account(3, entry=entry2, alias="work",
                     email="junyong.lee@samsung-research.example"), now)
    agg2 = Text()
    agg2.append(mini2)
    with con.capture() as cap:
        con.print(agg2)
    print(f"--- CONTROL window-only via aggregate w80 ---\n{cap.get()!r}")


def test_probe_spend_reset_suffix():
    now = time.time()
    for tag, spend in [
        ("no resets_at", {"used": 10.29, "limit": 20.0, "pct": 51.45,
                          "currency": "USD"}),
        ("with resets_at", {"used": 10.29, "limit": 20.0, "pct": 51.45,
                            "currency": "USD", "resets_at": _iso_in(86400 * 12)}),
    ]:
        rows = usage_rows({"spend": spend}, now, now - 5)
        print(f"\n{tag}: rows={rows!r}")
        acc = make_account(1, entry=make_entry(pct5=25.0, pct7=10.0, spend=spend))
        t = mini_account_text(acc, now)
        print(f"  mini len={len(t.plain)}: {t.plain!r}")
