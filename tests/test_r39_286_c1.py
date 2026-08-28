"""A failed park must not make a foreign `.swapping` leftover read as account A."""
from pathlib import Path

import pytest

from claude_swap.session import is_session_stale, mark_session_stale
from claude_swap.switcher import ClaudeAccountSwitcher

EA, EB = "aa@example.com", "bb@example.com"


def _prep(s):
    # `_swap_session_dirs` reads the session dirs and nothing else -- no roster,
    # no sequence file. Seeding either would advertise a dependency it lacks.
    s._setup_directories()
    a, b = s._session_dir("1", EA), s._session_dir("2", EB)
    for d, tag in ((a, "A-HISTORY"), (b, "B-HISTORY")):
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker").write_text(tag)
    return a, b


def test_a_failed_park_leaves_As_flag_on_As_own_profile(temp_home: Path):
    s = ClaudeAccountSwitcher()
    dir_a, _ = _prep(s)
    mark_session_stale(dir_a)
    assert is_session_stale(dir_a), "premise: A starts stale"

    # A leftover from an interrupted earlier swap. Nothing in the codebase
    # removes one, so a real os.replace raises ENOTEMPTY on the park.
    leftover = dir_a.with_name(dir_a.name + ".swapping")
    leftover.mkdir(parents=True)
    (leftover / "leftover").write_text("FOREIGN-LEFTOVER")

    moved: list[Path] = []
    s._swap_session_dirs("1", EA, "2", EB, moved)

    assert dir_a.exists() and (dir_a / "marker").read_text() == "A-HISTORY", (
        "premise: the park must have FAILED with A still at dir_a"
    )
    assert moved == [], f"premise: nothing should have landed, got {moved}"
    assert is_session_stale(dir_a), (
        "DEFECT: A's stale flag was taken off its own profile and written beside "
        f"an unrelated leftover; leftover_stale={is_session_stale(leftover)}"
    )


def test_control_the_flag_tracks_A_when_the_park_succeeds(temp_home: Path):
    """CONTROL: the same check in a case where it MUST report presence."""
    s = ClaudeAccountSwitcher()
    dir_a, _ = _prep(s)
    mark_session_stale(dir_a)
    moved: list[Path] = []
    s._swap_session_dirs("1", EA, "2", EB, moved)
    assert is_session_stale(s._session_dir("2", EA)), (
        "CONTROL FAILED: the instrument cannot report True"
    )


def test_an_interrupt_after_the_park_lands_still_recovers_A(temp_home: Path):
    """The park's rename and its record must not be two statements.

    A signal between them leaves A under `<slot>-<slug>.swapping` with the
    strand recovery disarmed, and that leftover is exactly what makes the next
    swap's park fail the way the test above describes.
    """
    import os

    s = ClaudeAccountSwitcher()
    dir_a, _ = _prep(s)
    real_replace = os.replace

    def interrupt_once_the_park_has_landed(src, dst):
        out = real_replace(src, dst)
        if str(dst).endswith(".swapping"):
            raise KeyboardInterrupt
        return out

    moved: list[Path] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os, "replace", interrupt_once_the_park_has_landed)
        with pytest.raises(KeyboardInterrupt):
            s._swap_session_dirs("1", EA, "2", EB, moved)

    stranded = dir_a.with_name(dir_a.name + ".swapping")
    assert not stranded.exists(), (
        f"DEFECT: A is stranded under the staging name; it holds "
        f"{sorted(p.name for p in stranded.iterdir())}"
    )
    assert dir_a.exists() and (dir_a / "marker").read_text() == "A-HISTORY", (
        "DEFECT: A did not come back to its own name"
    )
