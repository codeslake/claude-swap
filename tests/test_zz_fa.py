import time
from claude_swap.tui.widgets import usage_rows
from tests.test_tui import _iso_in

NOW = 1785000000.0


def test_fetched_at_effect_on_dollar_row():
    lg = {
        "spend": {"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"},
        "seven_day": {"pct": 90.0, "resets_at": _iso_in(86400 * 6)},
        "scoped": [{"name": "Fable", "pct": 90.0, "resets_at": _iso_in(86400 * 6)}],
    }
    for tag, fa in (("omitted (autoview)", None), ("passed (widgets)", NOW - 5)):
        rows = usage_rows(lg, NOW, fa)
        d = [r for r in rows if r[0] == "$$"]
        other = [r for r in rows if r[0] != "$$"]
        print(f"\n{tag}: $$ row = {d!r}")
        print(f"   other rows = {other!r}")
