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
    # ONLY `run:` VALUES, and INDENTATION is what says which lines are one.
    # `run:` takes an inline string OR a block scalar, so the walk has to
    # follow the scalar -- but stripping indentation to do that admits every
    # line in the job, and then `- name: Run pytest` (the most ordinary step
    # name in Actions), a `with:` value, and an install line all count as
    # pytest commands. A line belongs to a scalar only while it is indented
    # DEEPER than the `run:` that opened it.
    lines = []
    scalar_at = None
    for raw in block.group(1).splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        if scalar_at is not None:
            if indent > scalar_at:
                lines.append(raw.strip())
                continue
            scalar_at = None  # dedented back out of the block
        run = re.match(r"\s*(?:- )?run:\s*(.*)$", raw)
        if not run:
            continue
        value = run.group(1).strip()
        if value in ("|", "|-", "|+", ">", ">-", ">+"):
            scalar_at = indent
        elif value:
            lines.append(value)
    # A CONTINUED COMMAND IS ONE COMMAND. `uv run pytest \` + an indented
    # path list is the ordinary way to write a long invocation, and reading
    # only the first physical line drops every path after it -- silently, so
    # the existence check below skips exactly the names it exists to check.
    folded: list[str] = []
    for line in lines:
        if folded and folded[-1].endswith("\\"):
            folded[-1] = folded[-1][:-1].rstrip() + " " + line
        else:
            folded.append(line)
    lines = folded
    pytest_runs = [c for c in lines if re.search(r"(^|\s)pytest(\s|$)", c)]
    assert len(pytest_runs) == 1, (
        f"the macOS job has {len(pytest_runs)} pytest command(s), and this "
        f"reads one: {lines}"
    )
    return pytest_runs[0]


def test_the_macos_job_runs_something_that_exists():
    """Whatever the macOS job NAMES must exist.

    A path list in a workflow goes stale exactly the way the Windows shards
    do. A job that names NO path is not that failure -- a bare `pytest` is
    every file, strictly more than the two it used to name -- so requiring a
    name made the guard fire on a job that had gained coverage.

    WHAT THIS DOES NOT SAY is that naming nothing is fine ONLY because the
    command runs everything. Nothing here checks that, and the arm that tried
    to was wrong in both directions: `python -m pytest` runs the whole suite
    and was refused, while `-kfoo`, `--deselect` and `--collect-only` select
    nothing and passed. Judging a pytest command line from a regex needs
    pytest's own parser.

    So the loop below is EMPTY on the merged tree, where the job is a bare
    `uv run pytest` -- vacuously true, and correct, because a command that
    names nothing cannot name something stale. What guards the merged tree is
    `_macos_pytest_run`'s own "exactly one pytest command" assertion, which
    runs there whatever the command says.
    """
    run = _macos_pytest_run(_WORKFLOW.read_text(encoding="utf-8"))
    named = re.findall(r"(tests/test_\w+\.py)", run)
    for path in named:
        assert (_ROOT / path).exists(), f"the macOS job names {path}, which does not exist"


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

    # A `|` BLOCK SCALAR IS THE ORDINARY WAY TO ADD A SECOND COMMAND, and
    # reading only `run: (.*)` captures the bare `|` — measured, that reported
    # "0 pytest step(s)" for a step spelled the normal way.
    block = (
        head
        + "      - name: X\n        run: |\n"
        + "          echo starting\n"
        + "          uv run pytest -o faulthandler_timeout=60\n"
        + tail
    )
    # A COMMENT IS NOT A STEP. Without dropping them, the paragraph above the
    # step — which names pytest — counts as a second command.
    commented = (
        head
        + "      - name: X\n        # two pytest shards were tried here\n"
        + "        run: uv run pytest -o faulthandler_timeout=60\n"
        + tail
    )

    # A STEP NAME IS NOT A COMMAND. `- name: Run pytest` is the most
    # idiomatic step name in Actions, and a walk that strips indentation
    # counts it as a second pytest command -- reddening CI while naming a
    # cause that is not the one. The reader stopped keying on the step name
    # in the first place because it is prose that gets rewritten.
    step_named = (
        head
        + "      - name: Run pytest\n"
        + "        run: uv run pytest -o faulthandler_timeout=60\n"
        + tail
    )
    # NEITHER IS AN INSTALL, and this row is what makes the word boundary in
    # the predicate load-bearing: widen it to `"pytest" in c` and this fails.
    installs = (
        head
        + "      - name: Deps\n        run: uv pip install pytest-xdist\n"
        + "      - name: X\n        run: uv run pytest -o faulthandler_timeout=60\n"
        + tail
    )
    # A SHELL COMMENT INSIDE THE SCALAR is the only place the comment skip
    # still does anything: outside one, nothing but a `run:` value is
    # admitted at all. Without this row that skip is unwitnessed.
    scalar_comment = (
        head
        + "      - name: X\n        run: |\n"
        + "          # two pytest shards were tried here\n"
        + "          uv run pytest -o faulthandler_timeout=60\n"
        + tail
    )
    # A BACKSLASH CONTINUATION, which is how a long path list is written.
    # Read one physical line at a time it returns "uv run pytest \\" and the
    # existence check sees NO paths -- the silent hole, not a loud one.
    continued = (
        head
        + "      - name: X\n        run: |\n"
        + "          uv run pytest \\\n"
        + "            tests/test_x.py \\\n"
        + "            tests/test_y.py\n"
        + tail
    )
    # NOR A `with:` MAPPING VALUE -- a step shape the real job carries and
    # no row here had, which is why the table missed all three of these.
    with_value = (
        head
        + "      - uses: actions/upload-artifact@v4\n"
        + "        with:\n          name: pytest results\n"
        + "      - name: X\n        run: uv run pytest -o faulthandler_timeout=60\n"
        + tail
    )

    assert _macos_pytest_run(named) == "uv run pytest tests/test_x.py -v"
    assert _macos_pytest_run(whole) == "uv run pytest -o faulthandler_timeout=60"
    assert _macos_pytest_run(block) == "uv run pytest -o faulthandler_timeout=60"
    assert _macos_pytest_run(commented) == "uv run pytest -o faulthandler_timeout=60"
    assert _macos_pytest_run(step_named) == "uv run pytest -o faulthandler_timeout=60"
    assert _macos_pytest_run(installs) == "uv run pytest -o faulthandler_timeout=60"
    assert _macos_pytest_run(with_value) == "uv run pytest -o faulthandler_timeout=60"
    assert _macos_pytest_run(scalar_comment) == "uv run pytest -o faulthandler_timeout=60"
    assert _macos_pytest_run(continued) == (
        "uv run pytest tests/test_x.py tests/test_y.py")
    with pytest.raises(AssertionError, match="gone or was renamed"):
        _macos_pytest_run("\n  test:\n    runs-on: ubuntu-latest\n")
    with pytest.raises(AssertionError, match="pytest command"):
        _macos_pytest_run(head + tail)
