"""A quarantine with no exit, enterable by a race.

`_row_eligible` refuses a struck row and `record` clears the strike only on a
success, so a quarantined slot can never reach the fetch that would prove it
alive. The strike can be a race artifact: not from the consume gate (both
POST sites hold `.consume-N.lock` now), but from a concurrent Claude Code or a
sibling machine rotating the same lineage, outside every lock cswap holds. The dead direction is pinned first and deliberately: this guard
trades a false alarm for a silent failure if it is one line too wide.
"""
from __future__ import annotations

import pytest

from claude_swap.usage_store import (
    AUTH_DEAD_STRIKES,
    RACE_WINDOW_S,
    FetchRecord,
    UsageEntry,
    UsageStore,
    _row_eligible,
)

IDENT = {"1": ("a@x.com", "")}


class FakeClock:
    def __init__(self, start: float = 1_000_000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def store(tmp_path, clock):
    return UsageStore(tmp_path / "cache", clock=clock)


def _row(fetched_at, last_attempt_at, strikes=AUTH_DEAD_STRIKES):
    return {"authDeadStrikes": strikes, "fetchedAt": fetched_at,
            "lastAttemptAt": last_attempt_at}


# --- the dead direction: a real quarantine must survive all of this ---

@pytest.mark.parametrize("fetched,attempt,why", [
    (None, 2_000.0, "never succeeded, so nothing says it was ever alive"),
    (2_000.0, None, "no attempt recorded, so no gap can be computed"),
    (1_000.0, 1_000.0 + RACE_WINDOW_S + 1, "the success is outside the window"),
    (1_000.0, 1_000.0 + 900, "a full poll interval later is not a race"),
    (2_000.0, 1_000.0, "attempt BEFORE the success: a negative gap is not a race"),
])
def test_a_genuine_quarantine_is_not_released(fetched, attempt, why):
    """No success inside the window means no evidence the lineage answered."""
    entry = UsageEntry(auth_dead_strikes=AUTH_DEAD_STRIKES,
                       fetched_at=fetched, last_attempt_at=attempt)
    assert entry.token_dead(), f"a dead token was released: {why}"
    assert not _row_eligible(_row(fetched, attempt), now=9_999.0,
                             respect_plans=False), (
        f"a dead token was let back onto the fetch path: {why}"
    )


def test_an_unstruck_row_is_unaffected_by_the_window():
    """CONTROL: the guard must not be what makes a healthy row healthy."""
    entry = UsageEntry(auth_dead_strikes=0, fetched_at=1_000.0,
                       last_attempt_at=1_310.0)
    assert not entry.token_dead()
    # `token_dead` returns before the guard when unstruck, so that assert
    # alone survives every mutation of it. This call does traverse it.
    assert _row_eligible(_row(1_000.0, 1_310.0, strikes=0), now=9_999.0,
                         respect_plans=False)


# --- the race direction: the shape measured on a live host ---

def test_a_strike_moments_after_a_success_is_not_read_as_dead():
    """310 s inside a 600 s window, the gap seen in the field."""
    entry = UsageEntry(auth_dead_strikes=AUTH_DEAD_STRIKES,
                       fetched_at=1_000.0, last_attempt_at=1_310.0)
    assert not entry.token_dead(), (
        "a lineage that answered 310s before it was struck was condemned"
    )


def test_a_suspected_race_is_fetched_again():
    """The escape. Display alone would leave the row quarantined forever —
    only eligibility lets the success that clears the strike happen."""
    assert _row_eligible(_row(1_000.0, 1_310.0), now=9_999.0,
                         respect_plans=False), (
        "a suspected race stayed locked out of the fetch that would clear it"
    )


def test_the_retry_that_fails_again_makes_the_strike_stick(store, clock):
    """The guard must not loop. `fetchedAt` only advances on a success and
    `lastAttemptAt` on every attempt, so a second failure widens the gap past
    the window on its own — end to end through the store, not by hand."""
    store.record({"1": FetchRecord(usage={"five_hour": {"utilization": 1}})},
                 IDENT)
    clock.advance(310)
    store.record({"1": FetchRecord(error="invalid_grant", struck_fp="sha256:a")},
                 IDENT)
    assert not store.entries(IDENT)["1"].token_dead(), (
        "PREMISE: the first strike must land inside the window"
    )

    clock.advance(RACE_WINDOW_S + 1)
    store.record({"1": FetchRecord(error="invalid_grant", struck_fp="sha256:a")},
                 IDENT)
    entry = store.entries(IDENT)["1"]
    assert entry.token_dead(), (
        f"a lineage that failed twice, the second time {RACE_WINDOW_S + 311:.0f}s "
        f"after its last success, was still excused as a race: "
        f"fetched_at={entry.fetched_at} last_attempt_at={entry.last_attempt_at}"
    )


@pytest.mark.parametrize("gap,relogin", [(310, False), (RACE_WINDOW_S + 1, True)])
def test_the_sentinel_tracks_the_window(
    gap, relogin, temp_home, mock_claude_config, sample_sequence_data, monkeypatch,
):
    """What the owner actually sees. The guard sits before `token_dead`'s
    fingerprint compare, so `_entry_token_dead` answers False and the collector
    sets no sentinel — but that is a chain of four calls, and reasoning it
    through is not the same as running it."""
    import json
    from unittest.mock import patch

    from claude_swap import oauth
    from claude_swap.credentials import ActiveCredentials
    from claude_swap.json_output import USAGE_RELOGIN_REQUIRED
    from claude_swap.switcher import ClaudeAccountSwitcher

    creds = json.dumps({"claudeAiOauth": {
        "accessToken": "sk-a", "refreshToken": "rt-a", "expiresAt": 99999999999000}})
    sample_sequence_data["accounts"]["2"]["email"] = "b@example.com"
    s = ClaudeAccountSwitcher()
    s._setup_directories()
    s._write_json(s.sequence_file, sample_sequence_data)
    idents = {"2": ("b@example.com", "")}

    # A success, then a strike 310 s later on the SAME stored bytes.
    s._usage_store.record({"2": FetchRecord(usage={"five_hour": {"utilization": 5}})},
                          idents)
    row = s._usage_store._read_rows()["2"]
    row["lastAttemptAt"] = row["fetchedAt"] + gap
    row["authDeadStrikes"] = AUTH_DEAD_STRIKES
    row["struckFingerprint"] = oauth.credential_fingerprint(creds)
    s._usage_store._write_rows({"2": row})

    s._write_account_credentials("2", "b@example.com", creds)
    monkeypatch.setattr(s, "_read_active_credentials",
                        lambda: ActiveCredentials(creds, False, False))
    monkeypatch.setattr(s, "_get_current_account", lambda: ("b@example.com", ""))
    with patch.object(s, "current_account_number", return_value="2"):
        entries = s._collect_usage_entries(s._build_accounts_info(), fetch=set())

    # `is None`, not `!=`: a negative over a six-way sentinel enum keeps
    # passing if the row starts hitting a DIFFERENT sentinel for an unrelated
    # reason. The positive row is the control that the branch is reached.
    expected = USAGE_RELOGIN_REQUIRED if relogin else None
    assert entries["2"].sentinel == expected, (
        f"gap={gap:.0f}s rendered {entries['2'].sentinel!r}"
    )


def test_only_the_FIRST_strike_is_ever_doubted(store, clock):
    """The bound that caps the cost at one extra POST. A second rejection is
    dead even with a success still inside the window -- otherwise a dead grant
    could be re-POSTed for as long as the window allowed."""
    store.record({"1": FetchRecord(usage={"five_hour": {"utilization": 1}})}, IDENT)
    clock.advance(10)
    for _ in range(2):
        store.record({"1": FetchRecord(error="invalid_grant", struck_fp="sha256:a")},
                     IDENT)
    entry = store.entries(IDENT)["1"]
    gap = entry.last_attempt_at - entry.fetched_at
    assert gap <= RACE_WINDOW_S, (
        f"PREMISE: the success must still be inside the window (gap {gap:.0f}s)"
    )
    assert entry.token_dead(), (
        "a second rejection was excused, so a dead grant keeps being POSTed "
        "for as long as the window lasts"
    )
