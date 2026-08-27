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
            current["paths"] = "" if rest in (">-", ">", "|", "|-", ">+", "|+") else rest
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



def _assert_windows_job_consumes_the_matrix(workflow: Path) -> None:
    """The shards are data ONLY if the job actually consumes them.

    Everything above reads the matrix and nothing read its consumer, so three
    edits that collapse sharding entirely were invisible: dropping
    `${{ matrix.paths }}` makes the matrix dead data and every `--ignore`
    inert; dropping `-o testpaths=` lets pyproject's own `testpaths` override
    the positional paths so each shard silently runs the whole suite; deleting
    the run line stops the job testing anything. All three left the suite green.
    """
    text = workflow.read_text(encoding="utf-8")
    block = re.search(r"\n  test-windows:\n(.*?)(?=\n  \S|\Z)", text, re.S)
    assert block, "the test-windows job is gone or was renamed"
    runs = _pytest_run_lines(block.group(1))
    assert runs, "the Windows job no longer invokes pytest — the shards run nowhere"
    # ONE INVOCATION, NOT THEIR CONCATENATION. Joined, a second pytest step
    # anywhere in the job satisfies both markers below for the real shard
    # command -- the exact collapse this case exists to forbid. It also pins
    # the count `_mutate_windows_run` already assumes.
    assert len(runs) == 1, (
        f"the Windows job invokes pytest {len(runs)} times; the shard markers "
        f"below would be satisfied across them rather than by one line: {runs!r}"
    )
    line = runs[0]
    assert "${{ matrix.paths }}" in line, (
        "the Windows pytest command does not consume `matrix.paths`, so the "
        f"shard matrix is dead data and every shard runs the same thing: {line!r}"
    )
    assert "-o testpaths=" in line, (
        "the Windows pytest command does not clear `testpaths`, so pyproject's "
        f"own value overrides the shard paths and each shard runs it all: {line!r}"
    )


def test_a_commented_out_windows_command_is_refused(tmp_path):
    """The Windows reader is handed the RAW job; the macOS one is stripped.

    Folding a block scalar joins its lines, so a `#` that comments the real
    command out ends up INSIDE the folded value carrying `pytest`,
    `${{ matrix.paths }}` and `-o testpaths=` — every marker the guard looks
    for, on a job that runs `echo`. The asymmetry is what makes it a false
    CLEAN on one job and a correct refusal on the other.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    block = re.search(r"\n  test-windows:\n(.*?)(?=\n  \S|\Z)", text, re.S)
    assert block, "the test-windows job is gone or was renamed"
    real = re.search(r"\n( +)run: ([^\n]*(?<![\w-])pytest(?![\w-])[^\n]*)",
                     block.group(1))
    assert real, "premise: no inline pytest run line to comment out"
    indent, cmd = real.group(1), real.group(2)
    disabled = text.replace(
        f"\n{indent}run: {cmd}",
        f"\n{indent}run: |\n{indent}  # {cmd}\n{indent}  echo \"disabled\"",
        1,
    )
    assert disabled != text, "nothing was mutated — this check lost its subject"
    fake = tmp_path / "ci.yml"
    fake.write_text(disabled, encoding="utf-8")
    with pytest.raises(AssertionError, match="no longer invokes pytest"):
        _assert_windows_job_consumes_the_matrix(fake)


def test_the_windows_job_consumes_the_shard_matrix():
    _assert_windows_job_consumes_the_matrix(_WORKFLOW)


def test_a_step_named_for_pytest_that_runs_nothing_is_refused(tmp_path):
    """The scoping fix had no witness: every case passed with it reverted.

    Keying anywhere in the job rather than on the `run:` line accepts a step
    whose NAME says pytest while its command does something else -- and
    "Run pytest on macOS" is the most ordinary step name there is. Measured:
    with the reader loosened back to a bare search over the block, this is the
    only case that fails.
    """
    text = _WORKFLOW.read_text(encoding="utf-8")
    renamed = _mutate_macos_run(text, "\n        name: Run pytest on macOS")
    assert renamed != text, "nothing was mutated — this check lost its subject"
    fake = tmp_path / "ci.yml"
    fake.write_text(renamed, encoding="utf-8")
    with pytest.raises(AssertionError, match="no longer invokes pytest"):
        _assert_macos_job_is_intact(fake)


def _mutate_windows_run(text: str, repl: str) -> str:
    block = re.search(r"\n  test-windows:\n(.*?)(?=\n  \S|\Z)", text, re.S)
    assert block, "the test-windows job is gone or was renamed"
    body, n = re.subn(
        r"(\n +run: [^\n]*(?<![\w-])pytest(?![\w-])[^\n]*)", repl,
        block.group(1), count=1,
    )
    assert n == 1, "the Windows job has no pytest invocation to mutate"
    return text[: block.start(1)] + body + text[block.end(1) :]


@pytest.mark.parametrize(
    "repl, expected",
    [
        ("", "no longer invokes pytest"),
        (lambda m: m.group(1).replace(" ${{ matrix.paths }}", ""), "dead data"),
        (lambda m: m.group(1).replace(" -o testpaths=", ""), "overrides the shard paths"),
    ],
    ids=["invocation-deleted", "matrix-not-consumed", "testpaths-not-cleared"],
)
def test_each_way_the_windows_shards_collapse_is_refused(tmp_path, repl, expected):
    """The three edits that were measured green before this existed."""
    text = _WORKFLOW.read_text(encoding="utf-8")
    broken = _mutate_windows_run(text, repl)
    assert broken != text, "nothing was mutated — this check lost its subject"
    fake = tmp_path / "ci.yml"
    fake.write_text(broken, encoding="utf-8")
    with pytest.raises(AssertionError, match=expected):
        _assert_windows_job_consumes_the_matrix(fake)


def _macos_job(workflow: Path) -> str:
    """The `macos-keychain` job's block, with comments stripped.

    Comments go because the question below is whether the job INVOKES pytest,
    and a `#` line that merely mentions it answers yes for the wrong reason.
    """
    text = workflow.read_text(encoding="utf-8")
    block = re.search(r"\n  macos-keychain:\n(.*?)(?=\n  \S|\Z)", text, re.S)
    assert block, "the macos-keychain job is gone or was renamed"
    return "\n".join(re.sub(r"(?<!\S)#.*", "", ln) for ln in block.group(1).splitlines())


def _pytest_run_lines(job: str) -> list[str]:
    """The job's `run:` lines that invoke pytest.

    ON THE `run:` LINE, not anywhere in the block. Keyed loosely, the word in a
    step's `name:` answers yes while the command does nothing -- and `name: Run
    pytest on macOS` is the most ordinary step name there is, so the loose form
    is both a false clean and a false alarm.

    Block scalars are folded first, because `run: |` is a legal spelling of the
    same command and the header alone carries no `pytest`: read literally it
    reports a job that tests everything as testing nothing.

    WHAT THIS STILL CANNOT DO, stated rather than discovered: the two mutation
    helpers below select their target with a line regex and do NOT fold. On a
    workflow written with block scalars they raise "no pytest invocation to
    mutate" -- a loud refusal, which is the safe direction, but the cases then
    test their own scaffolding rather than the workflow.
    """
    # STRIPPED HERE, NOT PER CALLER. Folding joins a block scalar's lines, so
    # a `#` that comments the real command out lands INSIDE the folded value
    # and carries every marker with it. One caller stripped its job first and
    # the other did not, which made that a correct refusal on one and a false
    # CLEAN on the other.
    lines = [re.sub(r"(?<!\S)#.*", "", ln) for ln in job.splitlines()]
    folded, i = [], 0
    while i < len(lines):
        head = re.match(r"(\s*)run: *[|>][-+]?\s*$", lines[i])
        if not head:
            folded.append(lines[i])
            i += 1
            continue
        # A BLOCK SCALAR IS THE SAME COMMAND. `run: |` puts the invocation on
        # the following, deeper-indented lines; read literally the header has
        # no `pytest` in it and the job reads as running nothing.
        indent, body = len(head.group(1)), []
        i += 1
        while i < len(lines) and (
            not lines[i].strip()
            or len(lines[i]) - len(lines[i].lstrip()) > indent
        ):
            body.append(lines[i].strip())
            i += 1
        folded.append(head.group(1) + "run: " + " ".join(x for x in body if x))
    return [
        ln for ln in folded
        if re.match(r"\s*run: ", ln) and re.search(r"(?<![\w-])pytest(?![\w-])", ln)
    ]


def _assert_macos_job_is_intact(workflow: Path) -> None:
    job = _macos_job(workflow)
    assert _pytest_run_lines(job), (
        "the macOS job no longer invokes pytest — it would stay green testing nothing"
    )
    for path in re.findall(r"tests/test_\w+\.py", job):
        assert (_ROOT / path).exists(), f"the macOS job names {path}, which does not exist"
    # A STEP THAT CANNOT FAIL RUNS NOTHING, as far as CI is concerned. LAST,
    # because it is the broadest: raised ahead of the path check it masked
    # `test_a_job_naming_a_deleted_file_is_refused`, which then failed with
    # "regex did not match" about a workflow whose real defect was elsewhere.
    assert not re.search(r"^\s*continue-on-error:\s*true", job, re.M), (
        "the macOS job's step is continue-on-error, so a failure there is green"
    )


def test_the_macos_job_runs_pytest_on_paths_that_exist():
    """Ways the macOS job goes green testing nothing that THIS can see: the
    invocation deleted, the word present only in a step name, the step made
    continue-on-error, or a rename leaving it naming a file that is gone.

    NOT AN EXHAUSTIVE LIST, and deliberately not. `echo "uv run pytest ..."`,
    `|| true`, `--collect-only` and `-k nothing` all leave a real pytest
    invocation on a real `run:` line while selecting or running nothing.
    Judging that needs pytest's own parser; the reader that tried it was 250
    lines and got 16 of 31 shapes wrong, which is why it was deleted.
    """
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
