"""Cross-revision probe: identical account -> mini line, printed with length.
Imports ONLY mini_account_text + the suite's own builders, so it runs on
merge-base, parent and head unchanged."""
import time

from tests.test_tui import make_account, make_entry
from claude_swap.tui.widgets import mini_account_text

CASES = [
    ("A window-only 5h+7d", dict(pct5=25.0, pct7=10.0)),
    ("B 5h+7d+spend", dict(pct5=25.0, pct7=10.0,
        spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
    ("C spend-only", dict(pct5=None, pct7=None,
        spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
    ("D maxedscoped+spend", dict(pct5=None, pct7=None, scoped=[("Fable", 100.0)],
        spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
    ("E scoped-only 99", dict(pct5=None, pct7=None, scoped=[("Fable", 99.0)])),
    ("F all", dict(pct5=92.0, pct7=63.0, scoped=[("Fable", 100.0)],
        spend={"used": 1029.50, "limit": 2000.0, "pct": 51.45, "currency": "USD"})),
    ("G two scoped below cap", dict(pct5=None, pct7=None,
        scoped=[("Fable", 99.0), ("Opus", 40.0)])),
    ("H scoped maxed + below", dict(pct5=None, pct7=None,
        scoped=[("Fable", 100.0), ("Opus", 40.0)])),
    ("I 5h + scoped below", dict(pct5=10.0, pct7=None, scoped=[("Opus", 40.0)])),
    ("J nothing", dict(pct5=None, pct7=None)),
]


def test_xrev():
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    now = 1785000000.0   # FIXED clock -> countdown strings identical per rev
    print(f"\n\n##### REV {sha} #####")
    for tag, kw in CASES:
        acc = make_account(1, entry=make_entry(**kw))
        try:
            t = mini_account_text(acc, now)
            print(f"{tag:<24} len={len(t.plain):>4}  {t.plain!r}")
        except Exception as e:
            print(f"{tag:<24} RAISED {type(e).__name__}: {e}")
    # width behaviour of the resulting Text object itself
    t = mini_account_text(make_account(1, entry=make_entry(**CASES[1][1])), now)
    print(f"no_wrap={t.no_wrap!r} overflow={t.overflow!r}")
