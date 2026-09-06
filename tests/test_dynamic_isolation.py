"""Regression guard for adr/0009: `dynamic`'s own admission bar must not move
`best` or `consume-first` at all — every behaviour it adds is gated on
`settings.strategy == "dynamic"`. Hashes a fixed-seed fleet across 8 real
`engine.tick()`s, per strategy and with/without a pinned model, and pins the
digest; a mutant control proves the hash is sensitive rather than a no-op.
Not a simulator — a small, deterministic fixture with a fixed answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from tests.test_autoswitch import EngineHarness, _iso_at

# This PR's own base — before DRAIN STATE — for the cross-revision check
# below. Not "integration"/trunk; this branch's history.
_BASE_REV = "e9afe401"

# Captured on this branch (seed picked so `#1` starts ABOVE `threshold` —
# below it, `best`/`consume-first` never reach `_rank_candidates` at all
# and 8x NO_ACTION would pin nothing; `test_traces_actually_rank_not_just_
# hold` guards against that regressing silently), with `best`/`consume-
# first` untouched by the dynamic-only admission bar
# (adr/0009-a-model-window-is-not-a-blackout.md).
_SEED = 2
_GOLDEN = {
    ("best", ""): "0ba84e943f68f81a84f95aea6437eb497d65ad6da9e1494b4e5cd51c3e540b46",
    ("best", "Fable"): "0a1c94aba28954d3f487cd320ca78f7dbc81584ebcfb01acad1332b6cccf64e5",
    ("consume-first", ""): "4a4bc2fffb074934516f616dd246906b98960da4e06b8550cfaceaf0af8751d7",
    ("consume-first", "Fable"): "4a4bc2fffb074934516f616dd246906b98960da4e06b8550cfaceaf0af8751d7",
}


def _random_fleet(seed: int, now: float, n: int = 5) -> dict:
    rng = random.Random(seed)
    fleet = {}
    for i in range(1, n + 1):
        fleet[str(i)] = {
            "five_hour": {"pct": rng.uniform(0, 100)},
            "seven_day": {
                "pct": rng.uniform(0, 100),
                "resets_at": _iso_at(now + rng.uniform(3600, 30 * 86400)),
            },
            "scoped": [{"name": "Fable", "pct": rng.uniform(0, 100)}],
        }
    return fleet


def _run_trace(home, strategy: str, model: str, seed: int) -> list:
    """One fresh ``home`` per call — a shared root across combos leaves the
    prior combo's switched-to account and backups behind, aliasing the next
    one's decisions onto stale state (measured: reusing one ``temp_home``
    across the four combos below moved a digest that a fresh root does not).

    ``Path.home()`` stays patched for the whole call, not only construction:
    ``make_live()`` and ``tick()`` both resolve the active account through
    ``paths.*``, which reads ``Path.home()`` live — patched only around
    ``EngineHarness.__init__`` (as the conftest ``temp_home`` fixture does
    NOT do here) reads every tick's active account off the real ambient
    home instead, and every tick reads `no-active-account` (measured).
    """
    home.mkdir()
    (home / ".claude").mkdir()
    with (
        patch("pathlib.Path.home", return_value=home),
        patch.dict(os.environ, {"HOME": str(home), "USERPROFILE": str(home)}),
    ):
        h = EngineHarness(home, model=model, threshold=90.0, strategy=strategy)
        fleet = _random_fleet(seed, h.clock.now)
        for num in fleet:
            h.seed(int(num), f"acc{num}@example.com")
        h.make_live("acc1@example.com", 1)
        trace = []
        for _ in range(8):
            n0 = len(h.events)
            outcome = h.tick_with_usage(fleet)
            events = [(e.kind, getattr(e, "reason", None)) for e in h.events[n0:]]
            trace.append([outcome.name, h.active_number(), events])
            h.clock.advance(300.0)
    return trace


def _digest(home, strategy: str, model: str, seed: int) -> str:
    trace = _run_trace(home, strategy, model, seed)
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()


class TestDynamicNeverMovesBestOrConsumeFirst:
    def test_outcome_digests_are_pinned(self, tmp_path):
        for i, ((strategy, model), golden) in enumerate(_GOLDEN.items()):
            got = _digest(tmp_path / f"h{i}", strategy, model, seed=_SEED)
            assert got == golden, (
                f"{strategy}/model={model or '(none)'}: digest moved to "
                f"{got} — dynamic's admission bar must not reach this "
                "strategy at all"
            )

    def test_traces_actually_rank_not_just_hold(self, tmp_path):
        """A digest pin over an all-``NO_ACTION`` trace proves the fleet
        never entered `_rank_candidates` at all — identical `best`/
        `consume-first` digests for model on/off would most likely mean
        this, not that dynamic correctly left them alone (with 5 accounts
        and a uniform Fable pct, the model window binds ~99.6% of draws).
        The seed is chosen so `#1` starts ABOVE `threshold`; this pins that
        choice so a future reseed cannot regress back to a vacuous guard
        silently."""
        for i, (strategy, model) in enumerate(_GOLDEN):
            trace = _run_trace(tmp_path / f"t{i}", strategy, model, seed=_SEED)
            outcomes = {tick[0] for tick in trace}
            assert outcomes != {"NO_ACTION"}, (
                f"{strategy}/model={model or '(none)'}: every tick was "
                f"NO_ACTION ({trace}) — the fleet never reached the "
                "ranking path this guard exists to protect"
            )

    def test_mutant_control_moves_every_digest(self, tmp_path, monkeypatch):
        """Not a vacuous pin: a real change to shared ranking code must move
        every one of the four digests above. Inverts every account's
        headroom (``100 - h``) — flips who looks healthy vs. blocked, which
        a single anti-flap constant does not reliably do when a fixed fleet
        only ever switches once."""
        from claude_swap import oauth

        real_headroom = oauth.account_headroom
        monkeypatch.setattr(
            oauth,
            "account_headroom",
            lambda usage, models: (
                None if (h := real_headroom(usage, models)) is None else 100.0 - h
            ),
        )
        for i, ((strategy, model), golden) in enumerate(_GOLDEN.items()):
            got = _digest(tmp_path / f"m{i}", strategy, model, seed=_SEED)
            assert got != golden, (
                f"mutant control did not move {strategy}/model={model or '(none)'}"
            )


class TestDynamicLeavesTheBaseRevisionAlone:
    """The actual adr/0009 measurement: golden values above pin FUTURE
    drift on this branch, which is not the same claim as "this round left
    `best`/`consume-first` alone". That claim needs a SECOND revision to
    compare against — `_BASE_REV`, this PR's own base before DRAIN STATE.
    `tests/test_autoswitch.py` (the `EngineHarness` this module imports)
    and `src/claude_swap` are archived from that revision into a scratch
    tree and run there, unchanged, in a subprocess (a different
    `claude_swap` package must not share `sys.modules` with this process)."""

    def test_e9afe401_produces_the_same_four_digests(self, tmp_path):
        repo_root = Path(__file__).resolve().parents[1]
        old_root = tmp_path / "base"
        old_root.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(repo_root), "archive", _BASE_REV, "src", "tests"],
            capture_output=True, check=True,
        )
        subprocess.run(["tar", "-x", "-C", str(old_root)], input=archive.stdout, check=True)
        driver = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(old_root)!r})\n"
            f"sys.path.insert(0, {str(old_root / 'src')!r})\n"
            f"sys.path.insert(0, {str(tmp_path)!r})\n"  # this module's own dir, for _run_trace
            "from test_dynamic_isolation import _run_trace, _GOLDEN\n"
            "from pathlib import Path\n"
            "import hashlib, json as _json\n"
            "out = {}\n"
            "for i, (strategy, model) in enumerate(_GOLDEN):\n"
            "    trace = _run_trace(Path(sys.argv[1]) / f'b{i}', strategy, model, int(sys.argv[2]))\n"
            "    out[f'{strategy}|{model}'] = hashlib.sha256("
            "_json.dumps(trace, sort_keys=True).encode()).hexdigest()\n"
            "print(_json.dumps(out))\n"
        )
        (tmp_path / "test_dynamic_isolation.py").write_text(
            Path(__file__).read_text()
        )
        driver_path = tmp_path / "_zz_cross_rev_driver.py"
        driver_path.write_text(driver)
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        result = subprocess.run(
            [sys.executable, str(driver_path), str(runs_dir), str(_SEED)],
            capture_output=True, text=True, cwd=str(old_root),
        )
        assert result.returncode == 0, (
            f"driver failed: rc={result.returncode}\nSTDOUT={result.stdout}\n"
            f"STDERR={result.stderr}"
        )
        got = json.loads(result.stdout.strip().splitlines()[-1])
        for (strategy, model), golden in _GOLDEN.items():
            key = f"{strategy}|{model}"
            assert got[key] == golden, (
                f"{strategy}/model={model or '(none)'}: {_BASE_REV} gives "
                f"{got[key]}, HEAD gives {golden} — this round moved "
                "behaviour this strategy never authorized"
            )
