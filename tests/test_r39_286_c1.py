"""A failed park must not make a foreign `.swapping` leftover read as account A."""
from pathlib import Path

from claude_swap.session import is_session_stale, mark_session_stale
from claude_swap.switcher import ClaudeAccountSwitcher

EA, EB = "aa@example.com", "bb@example.com"


def _prep(s, data):
    s._setup_directories()
    s._write_json(s.sequence_file, data)
    a, b = s._session_dir("1", EA), s._session_dir("2", EB)
    for d, tag in ((a, "A-HISTORY"), (b, "B-HISTORY")):
        d.mkdir(parents=True, exist_ok=True)
        (d / "marker").write_text(tag)
    return a, b


def test_a_failed_park_leaves_As_flag_on_As_own_profile(
    temp_home: Path, sample_sequence_data_with_org: dict
):
    s = ClaudeAccountSwitcher()
    dir_a, _ = _prep(s, sample_sequence_data_with_org)
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


def test_control_the_flag_tracks_A_when_the_park_succeeds(
    temp_home: Path, sample_sequence_data_with_org: dict
):
    """CONTROL: the same check in a case where it MUST report presence."""
    s = ClaudeAccountSwitcher()
    dir_a, _ = _prep(s, sample_sequence_data_with_org)
    mark_session_stale(dir_a)
    moved: list[Path] = []
    s._swap_session_dirs("1", EA, "2", EB, moved)
    assert is_session_stale(s._session_dir("2", EA)), (
        "CONTROL FAILED: the instrument cannot report True"
    )
