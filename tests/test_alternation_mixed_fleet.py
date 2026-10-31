"""T3167: the percent alternation step on a mixed fleet, from the pairwise model
`tests/pict/autoswitch_mixed_fleet.pict`.

One due active (5h 40%, a step of 20 spent) and two candidates. Candidate 2 takes
the row's windows (5h, 7d, a pinned model window) and its Plan; candidate 3 is
always cold on the plan PartnerPlan, with 5h 20% and 7d 60% (40 points of room).
Each row below is a generated case with the account the rules take (1 = hold): a
candidate is taken iff it keeps `step + SPENT_HEADROOM_PCT` = 23 points of room on
its binding window, a cold one after its re-write is charged to that window (5h
points where it has a 5h window, 7d points where it has not, a model window the
user pins counted in the room), and of the candidates that do the one with the
most headroom first.

`coldSwitchCostPct` is the Ultra (Max 20x) figure and the engine scales it x4 on
a Max 5x account (`UsageEntry.plan`, on #199). One live session's prefix costs
`tokens x 1.9 units / (2.08M units per 5h point x plan scale)` points
(alternation-replay.md, calibration; Max 20x scale 1, Max 5x 0.25), and the
setting prices a whole live fleet landing together, about 19 sessions
(settings.py): 100k tokens on 20x = 1.7 points, 400k = 6.9, and 4x those on 5x.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claude_swap.usage_store import UsageEntry
from tests.test_alternation_step import _harness, _iso_at, _reasons, _stamp, _tick, _track, _u

HAS_PLAN = "plan" in UsageEntry.__dataclass_fields__
COST = {"small": 2.0, "large": 7.0}  # the Ultra figure
PLAN = {"max5x": "Max 5x", "max20x": "Ultra"}
SEVEN = {"low": 20.0, "mid": 50.0, "tight": 70.0, "near_cap": 90.0, "reset_from_full": 0.0}

# plan, partner plan, 5h shape, 7d level, partner, context, pinned model window -> the account taken
CASES = [
    ("max5x", "max5x", "running", "mid", "cold", "large", "none", 1),         # 2: 50 - 28 = 22 < 23; 3: 40 - 28 = 12
    ("max5x", "max5x", "removed", "mid", "cold", "large", "open", 2),         # 28 / 4.4 = 6.4 on the 7d: 43.6
    ("max5x", "max5x", "idle", "mid", "warm", "large", "none", 2),            # warm: 50
    ("max5x", "max5x", "removed", "tight", "cold", "large", "none", 2),       # 30 - 6.4 = 23.6: only the /4.4 admits it
    ("max20x", "max20x", "idle", "reset_from_full", "cold", "small", "none", 2),  # both viable, 2 has 100 against 40
    ("max5x", "max20x", "running", "mid", "cold", "large", "none", 3),        # 2: 50 - 28 = 22; 3: 40 - 7 = 33
    ("max20x", "max5x", "running", "mid", "cold", "large", "none", 2),        # 2: 50 - 7 = 43; 3: 40 - 28 = 12
    ("max20x", "max20x", "running", "tight", "warm", "small", "full", 3),     # 2's pinned window is full; 3: 40 - 2
    ("max5x", "max5x", "removed", "mid", "cold", "small", "full", 3),         # 2 full; 3: 40 - 8 = 32
    ("max20x", "max20x", "removed", "near_cap", "warm", "large", "full", 3),  # 2 full; 3: 40 - 7
    ("max20x", "max5x", "removed", "reset_from_full", "warm", "large", "open", 2),  # 2: 90; 3: 40 - 28 = 12
    ("max5x", "max5x", "running", "near_cap", "cold", "small", "none", 3),    # 2: 10 - 8; 3: 40 - 8 = 32
    ("max20x", "max20x", "idle", "near_cap", "warm", "small", "open", 3),     # 2: 10 points of room; 3: 38
    ("max5x", "max20x", "removed", "low", "warm", "large", "none", 2),        # both viable, 2 has 80 against 40
    ("max20x", "max5x", "idle", "low", "cold", "small", "full", 3),           # 2 full; 3: 40 - 8 = 32
    ("max5x", "max5x", "idle", "reset_from_full", "cold", "small", "full", 3),  # 2 full; 3: 40 - 8 = 32
    ("max5x", "max20x", "running", "low", "cold", "small", "open", 2),        # both viable, 2 has 80 against 40
    ("max20x", "max20x", "idle", "tight", "cold", "small", "open", 3),        # 2: 30 - 2 = 28; 3 has 40 against 30
]


def test_the_cases_are_the_generated_ones():
    cases = Path(__file__).parent / "pict" / "autoswitch_mixed_fleet.cases"
    rows = [line.split("\t") for line in cases.read_text().splitlines()[1:]]
    assert rows == [list(case[:7]) for case in CASES]


@pytest.mark.parametrize("case", CASES, ids=lambda c: "-".join(c[:7]))
def test_the_step_on_a_mixed_fleet(temp_home, case):
    plan, partner_plan, five, seven, partner, context, pinned, taken = case
    if not HAS_PLAN and "max5x" in (plan, partner_plan):
        pytest.skip("UsageEntry.plan is on #199: the cold cost cannot read a plan on this branch")
    h = _harness(
        temp_home,
        cold_switch_cost_pct=COST[context],
        model=None if pinned == "none" else "Fable",
    )
    b = _u({"running": 20.0, "idle": 0.0, "removed": None}[five], SEVEN[seven])
    if five == "running":
        b["five_hour"]["resets_at"] = _iso_at(h.clock.now + 7200.0)
    if seven != "reset_from_full":
        b["seven_day"]["resets_at"] = _iso_at(h.clock.now + 3 * 86400.0)
    if pinned != "none":
        b["scoped"] = [{"name": "Fable", "pct": 10.0 if pinned == "open" else 100.0}]
    c = _u(20.0, 60.0, reset7=h.clock.now + 3 * 86400.0)
    c["five_hour"]["resets_at"] = _iso_at(h.clock.now + 7200.0)
    plans = {"1": "Ultra", "2": PLAN[plan], "3": PLAN[partner_plan]} if HAS_PLAN else None
    _stamp(h, **{"1": 60.0, **({"2": 120.0} if partner == "warm" else {})})
    _track(h, anchor=0.0)
    outcome = _tick(h, {"1": _u(40.0), "2": b, "3": c}, plans=plans)
    assert h.active_number() == taken, (outcome, _reasons(h))
