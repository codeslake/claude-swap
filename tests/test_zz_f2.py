import subprocess, time
from tests.test_tui import make_account, make_entry
from claude_swap.tui.widgets import mini_account_text, account_card_text

NOW = 1785000000.0


def test_nonmonotonic():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"\n\n##### F2 REV {sha} #####")
    print("\n--- NON-MONOTONIC: adding a maxed window HIDES the one already shown ---")
    a = make_entry(pct5=None, pct7=None, scoped=[("Opus", 40.0)])
    b = make_entry(pct5=None, pct7=None, scoped=[("Opus", 40.0), ("Fable", 100.0)])
    ta = mini_account_text(make_account(1, entry=a), NOW).plain
    tb = mini_account_text(make_account(1, entry=b), NOW).plain
    print(f"  PROBE   Opus 40 only            -> {ta!r}")
    print(f"  PROBE   Opus 40 + Fable 100     -> {tb!r}")
    print(f"  'Opus' present in the 1-window render? {'Opus' in ta}")
    print(f"  'Opus' present after ADDING data?      {'Opus' in tb}")
    print(f"  CONTROL card, same 2-window entry:")
    print(f"    {account_card_text(make_account(1, entry=b), 100, now=NOW).plain!r}")
    print("  CONTROL both-below-cap (no maxed) keeps both:")
    c = make_entry(pct5=None, pct7=None, scoped=[("Opus", 40.0), ("Fable", 99.0)])
    print(f"    {mini_account_text(make_account(1, entry=c), NOW).plain!r}")

    print("\n--- FALLBACK DROPS suffix: countdown + '(ahead of pace)' ---")
    e = make_entry(pct5=None, pct7=None, scoped=[("Fable", 99.0)])
    print(f"  PROBE   mini scoped-only  -> {mini_account_text(make_account(1, entry=e), NOW).plain!r}")
    print(f"  CONTROL card, same entry  -> {account_card_text(make_account(1, entry=e), 100, now=NOW).plain!r}")
    f = make_entry(pct5=10.0, pct7=None)
    print(f"  CONTROL mini 5h-only KEEPS its countdown -> {mini_account_text(make_account(1, entry=f), NOW).plain!r}")
