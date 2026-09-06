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
from unittest.mock import patch

from tests.test_autoswitch import EngineHarness, _iso_at

# Captured on this branch, with `best`/`consume-first` untouched by the
# dynamic-only admission bar (adr/0009-a-model-window-is-not-a-blackout.md).
_GOLDEN = {
    ("best", ""): "563e692d603dde66034c0d7f10689f3cf70c599879cac4a14604f23d9373c2fc",
    ("best", "Fable"): "563e692d603dde66034c0d7f10689f3cf70c599879cac4a14604f23d9373c2fc",
    ("consume-first", ""): "b7067f4d5ab8f01d10455fcf8a06a358b5b91846cf03bf5803be153e30f5e9ed",
    ("consume-first", "Fable"): "b7067f4d5ab8f01d10455fcf8a06a358b5b91846cf03bf5803be153e30f5e9ed",
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


def _digest(home, strategy: str, model: str, seed: int) -> str:
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
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()


class TestDynamicNeverMovesBestOrConsumeFirst:
    def test_outcome_digests_are_pinned(self, tmp_path):
        for i, ((strategy, model), golden) in enumerate(_GOLDEN.items()):
            got = _digest(tmp_path / f"h{i}", strategy, model, seed=1234)
            assert got == golden, (
                f"{strategy}/model={model or '(none)'}: digest moved to "
                f"{got} — dynamic's admission bar must not reach this "
                "strategy at all"
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
            got = _digest(tmp_path / f"m{i}", strategy, model, seed=1234)
            assert got != golden, (
                f"mutant control did not move {strategy}/model={model or '(none)'}"
            )
