"""Separator ambiguity: is the ' . ' BETWEEN rows distinguishable from the
' . ' INSIDE the spend row?"""
import subprocess, time
from tests.test_tui import make_account, make_entry
from claude_swap.tui.widgets import mini_account_text

NOW = 1785000000.0


def test_sep():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"\n\n##### SEP REV {sha} #####")
    for tag, kw in [
        ("PROBE  5h+7d+spend", dict(pct5=25.0, pct7=10.0,
            spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
        ("PROBE  maxed+spend", dict(pct5=None, pct7=None, scoped=[("Fable", 100.0)],
            spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
        ("CONTROL 5h+7d only", dict(pct5=25.0, pct7=10.0)),
        ("CONTROL spend only", dict(pct5=None, pct7=None,
            spend={"used": 10.29, "limit": 20.0, "pct": 51.45, "currency": "USD"})),
    ]:
        t = mini_account_text(make_account(1, entry=make_entry(**kw)), NOW)
        seps = [(s.start, s.end, str(s.style)) for s in t.spans
                if t.plain[s.start:s.end] == " · "]
        print(f"\n{tag}: {t.plain!r}")
        print(f"   ' . ' occurrences in plain = {t.plain.count(' · ')}")
        for st, en, sty in seps:
            print(f"   span[{st}:{en}] style={sty!r}")
