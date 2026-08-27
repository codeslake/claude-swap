"""The CI shards must cover every test file, exactly once.

Sharding the Windows job splits the suite across parallel jobs by PATH, and a
path list in a workflow is a list that goes stale: add `tests/test_foo.py`,
name it in no shard, and it silently stops running in CI while the suite still
passes locally and every job stays green. That failure is invisible in exactly
the place a test suite is supposed to be loud.

So the split is data, read back out of the workflow and checked here. The
shards are parsed from the matrix rather than restated, so this cannot drift
from what CI actually runs the way a hand-copied list would.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_WORKFLOW = _ROOT / ".github" / "workflows" / "ci.yml"
_TESTS = _ROOT / "tests"


def _windows_shards() -> list[dict[str, str]]:
    """The matrix entries, parsed without PyYAML.

    PyYAML is not a project dependency, and `importorskip` would make this a
    guard that SKIPS — which is the same silent hole it exists to close, just
    one level up. The shape being read is two keys of a `- name:` list, so a
    small reader beats adding a dependency to the host project for one test.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    block = re.search(
        r"\n  test-windows:.*?\n        include:\n(.*?)\n    name:", text, re.S
    )
    assert block, "the test-windows matrix is gone or was restructured"
    shards, current = [], None
    for raw in block.group(1).splitlines():
        line = raw.strip()
        if line.startswith("#") or not line:
            continue
        if line.startswith("- name:"):
            current = {"name": line.split(":", 1)[1].strip(), "paths": ""}
            shards.append(current)
        elif line.startswith("paths:"):
            rest = line.split(":", 1)[1].strip()
            current["paths"] = "" if rest == ">-" else rest
        elif current is not None:
            current["paths"] += " " + line
    assert shards, "no shards parsed out of the matrix"
    return shards


def _split(paths: str) -> tuple[set[str], set[str]]:
    """(files this shard RUNS by name, files it IGNORES)."""
    runs, ignores = set(), set()
    for word in paths.split():
        if word.startswith("--ignore="):
            ignores.add(word.split("=", 1)[1])
        else:
            runs.add(word)
    return runs, ignores


def test_every_test_file_runs_in_exactly_one_windows_shard():
    """Each test file must run in one shard — not zero, not two.

    Zero is a file that quietly left CI. Two is the same file paid for twice
    on the slowest platform, which is the cost this sharding exists to cut.
    """
    on_disk = {f"tests/{p.name}" for p in _TESTS.glob("test_*.py")}
    assert on_disk, "no test files found — the glob or the layout moved"

    runs_per_file: dict[str, list[str]] = {p: [] for p in on_disk}
    for shard in _windows_shards():
        named, ignored = _split(shard["paths"])
        stale = (named | ignored) - on_disk
        assert not stale, (
            f"shard {shard['name']!r} names {sorted(stale)}, which do not "
            f"exist — a rename left the workflow behind"
        )
        # A shard with no named paths is the catch-all: everything except
        # what it ignores.
        covered = named if named else on_disk - ignored
        for path in covered:
            runs_per_file[path].append(shard["name"])

    missing = sorted(p for p, s in runs_per_file.items() if not s)
    assert not missing, f"{missing} run in NO Windows shard — they left CI silently"

    doubled = sorted((p, s) for p, s in runs_per_file.items() if len(s) > 1)
    assert not doubled, f"these run in more than one Windows shard: {doubled}"


def _macos_pytest_run(text: str) -> str:
    """The macOS job's pytest command, found by JOB ID, never by step name.

    The step name is prose and has already been rewritten once. The job id is
    what branch protection matches on, so it is the only key here that cannot
    move without somebody noticing.
    """
    block = re.search(r"\n  macos-keychain:\n(.*?)(?=\n  \w[\w-]*:\n|\Z)", text, re.S)
    assert block, "the macos-keychain job is gone or was renamed"
    runs = [m.group(1) for m in re.finditer(r"\n\s+run: (.*)", block.group(1))]
    pytest_runs = [r for r in runs if "pytest" in r]
    assert len(pytest_runs) == 1, (
        f"the macOS job has {len(pytest_runs)} pytest step(s), and this reads "
        f"one: {runs}"
    )
    return pytest_runs[0]


def test_the_macos_job_runs_something_that_exists():
    """Whatever the macOS job NAMES must exist; naming nothing is allowed only
    when the reason is that it runs everything.

    A path list in a workflow goes stale exactly the way the Windows shards
    do. But a job that names no path is not that failure when the command is
    a bare `pytest`: that IS every file, which is strictly more than the two
    it used to name. Requiring a name made the guard fire on a job that had
    gained coverage.
    """
    run = _macos_pytest_run(_WORKFLOW.read_text(encoding="utf-8"))
    for path in re.findall(r"(tests/test_\w+\.py)", run):
        assert (_ROOT / path).exists(), f"the macOS job names {path}, which does not exist"
    if not re.search(r"tests/test_\w+\.py", run):
        assert not re.search(r"(^| )-[km]( |=)", run), (
            f"the macOS job names no file and filters with {run!r} — that can "
            "select nothing, which is the silent hole this guards"
        )


def test_the_reader_accepts_both_shapes_and_still_refuses_a_deletion():
    """THE TABLE, because the file on disk can only ever be ONE shape.

    Measured: the disk case alone read "runs the whole suite" as "stopped
    naming any test file" and failed a workflow that had gained coverage —
    and no branch could see it, because the guard and the widened job live on
    different ones. The reader is exercised on both here, and on a workflow
    that has lost the job entirely.
    """
    head = (
        "\n  macos-keychain:\n    runs-on: macos-latest\n    steps:\n"
        "      - name: Install dependencies\n        run: uv sync --locked\n"
    )
    tail = "\n  test-windows:\n    runs-on: windows-latest\n"
    named = head + "      - name: X\n        run: uv run pytest tests/test_x.py -v\n" + tail
    whole = head + "      - name: X\n        run: uv run pytest -o faulthandler_timeout=60\n" + tail

    assert _macos_pytest_run(named) == "uv run pytest tests/test_x.py -v"
    assert _macos_pytest_run(whole) == "uv run pytest -o faulthandler_timeout=60"
    with pytest.raises(AssertionError, match="gone or was renamed"):
        _macos_pytest_run("\n  test:\n    runs-on: ubuntu-latest\n")
    with pytest.raises(AssertionError, match="pytest step"):
        _macos_pytest_run(head + tail)
