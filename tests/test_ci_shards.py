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



def _macos_job(workflow: Path) -> str:
    """The `macos-keychain` job's block, with comments stripped.

    Comments go because the question below is whether the job INVOKES pytest,
    and a `#` line that merely mentions it answers yes for the wrong reason.
    """
    text = workflow.read_text(encoding="utf-8")
    block = re.search(r"\n  macos-keychain:\n(.*?)(?=\n  \S|\Z)", text, re.S)
    assert block, "the macos-keychain job is gone or was renamed"
    return "\n".join(re.sub(r"(?<!\S)#.*", "", ln) for ln in block.group(1).splitlines())


def _assert_macos_job_is_intact(workflow: Path) -> None:
    job = _macos_job(workflow)
    assert re.search(r"(?<![\w-])pytest(?![\w-])", job), (
        "the macOS job no longer invokes pytest — it would stay green testing nothing"
    )
    for path in re.findall(r"tests/test_\w+\.py", job):
        assert (_ROOT / path).exists(), f"the macOS job names {path}, which does not exist"


def test_the_macos_job_runs_pytest_on_paths_that_exist():
    """Two ways the macOS job goes green testing nothing: the invocation is
    deleted, or a rename leaves it naming a file that no longer exists."""
    _assert_macos_job_is_intact(_WORKFLOW)


def _mutate_macos_run(text: str, repl: str) -> str:
    """`text` with the macOS job's pytest `run:` line rewritten by `repl`.

    SCOPED TO THAT JOB, and keyed on the invocation rather than on any path it
    names: a sibling PR replaces the two named files with a bare `pytest`, so a
    substitution keyed on a filename matches nothing once merged and the case
    that used it silently stops testing.
    """
    block = re.search(r"\n  macos-keychain:\n(.*?)(?=\n  \S|\Z)", text, re.S)
    assert block, "the macos-keychain job is gone or was renamed"
    body, n = re.subn(
        r"(\n +run: [^\n]*(?<![\w-])pytest(?![\w-])[^\n]*)", repl,
        block.group(1), count=1,
    )
    assert n == 1, "the macOS job has no pytest invocation to mutate"
    return text[: block.start(1)] + body + text[block.end(1) :]


def test_a_job_that_stopped_running_pytest_is_refused(tmp_path):
    """The deletion above, on a real copy of the real workflow."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    gutted = _mutate_macos_run(text, "")
    assert gutted != text, "nothing was mutated — this check lost its subject"
    fake = tmp_path / "ci.yml"
    fake.write_text(gutted, encoding="utf-8")
    with pytest.raises(AssertionError, match="no longer invokes pytest"):
        _assert_macos_job_is_intact(fake)


def test_a_job_naming_a_deleted_file_is_refused(tmp_path):
    """The other half: the invocation survives, a path it names does not.

    The decoy is APPENDED, so this holds whether the job names two files or
    none -- which is exactly what a sibling PR changes about this step.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    assert not (_ROOT / "tests/test_gone.py").exists(), "the decoy path exists"
    stale = _mutate_macos_run(text, r"\1 tests/test_gone.py")
    assert stale != text, "nothing was mutated — this check lost its subject"
    fake = tmp_path / "ci.yml"
    fake.write_text(stale, encoding="utf-8")
    with pytest.raises(AssertionError, match="names tests/test_gone.py"):
        _assert_macos_job_is_intact(fake)


def test_a_commented_out_invocation_does_not_count(tmp_path):
    """A step someone commented out to debug and never restored.

    The word `pytest` is still in the file, so a reader that keeps comments
    passes here while the job runs nothing. This is what the stripping in
    `_macos_job` buys, and without this case nothing witnessed it.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    disabled = _mutate_macos_run(text, lambda m: m.group(1).replace("run:", "# run:"))
    assert disabled != text, "nothing was commented out — lost its subject"
    assert "pytest" in disabled, "the decoy is gone; the case no longer discriminates"
    fake = tmp_path / "ci.yml"
    fake.write_text(disabled, encoding="utf-8")
    with pytest.raises(AssertionError, match="no longer invokes pytest"):
        _assert_macos_job_is_intact(fake)
