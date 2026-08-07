import time
from rich.console import Console, ConsoleOptions
from rich.text import Text
from tests.test_tui import make_account, make_entry
from claude_swap.tui.widgets import mini_account_text

NOW = 1785000000.0
EMAIL = "junyong.lee@samsung-research.example"


def lines(con, obj, w):
    opts = con.options.update(width=w)
    out = []
    for line in con.render_lines(obj, opts, pad=False):
        out.append("".join(s.text for s in line))
    return out


def show(tag, entry_kw, w):
    con = Console(width=w, no_color=True, legacy_windows=False)
    mini = mini_account_text(make_account(3, entry=make_entry(**entry_kw),
                                          alias="work", email=EMAIL), NOW)
    agg = Text()
    agg.append(mini)
    a = lines(con, mini, w)
    b = lines(con, agg, w)
    print(f"\n[{tag}] plain len={len(mini.plain)}  width={w}")
    print(f"  mini alone   -> {len(a)} line(s): {a!r}")
    print(f"  via panel agg-> {len(b)} line(s): {b!r}")


def test_width():
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"\n\n##### WIDTH REV {sha} #####")
    ordinary = dict(pct5=25.0, pct7=10.0)
    with_spend = dict(pct5=25.0, pct7=10.0,
                      spend={"used": 10.29, "limit": 20.0, "pct": 51.45,
                             "currency": "USD"})
    for w in (80, 100, 120):
        show("CONTROL ordinary (no spend)", ordinary, w)
        show("PROBE   ordinary + spend", with_spend, w)
