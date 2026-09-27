"""``tests/conftest.py``'s ``pytest_sessionfinish`` self-deletes THIS
process's own ``pytest-N`` basetemp on a green run, so a LATER session never
becomes the one to evict it (pytest's own ``cleanup_numbered_dir``, ``keep=3``
default, scans ``/tmp/pytest-of-<user>`` and ``shutil.rmtree``s whatever
falls off the back). Measured: this suite's ~45k-entry basetemp (the
autouse ``isolated_home`` fixture alone makes ~1,987 ``mktemp`` dirs per run)
costs ~3.8s to delete serially, and the bimodal ``-n 10`` wall (~19s/~23s)
was that cost landing on whichever session happened to trigger it.

These tests call the hook directly with stand-in ``session``/``config``
objects over a real temp directory — no ``pytester`` subprocess, since the
hook is a handful of attribute reads and one ``shutil.rmtree``.
"""
from __future__ import annotations

import pytest

from tests import conftest


class _FactoryStub:
    def __init__(self, basetemp, given_basetemp):
        self._basetemp = basetemp
        self._given_basetemp = given_basetemp


class _ConfigStub:
    def __init__(self, factory, is_worker):
        self._tmp_path_factory = factory
        if is_worker:
            self.workerinput = {"workerid": "gw0"}


class _SessionStub:
    def __init__(self, config):
        self.config = config


def _make_basetemp(tmp_path):
    """A throwaway directory standing in for a session's real basetemp,
    with one nested entry so a no-op deletion can't pass by accident."""
    basetemp = tmp_path / "pytest-7"
    (basetemp / "popen-gw0" / "isolated_home-3").mkdir(parents=True)
    return basetemp


@pytest.mark.parametrize(
    "exitstatus, is_worker, user_given, expect_deleted",
    [
        (0, True, True, True),  # green worker: cleans its own share regardless
        (1, True, True, False),  # red: kept for debugging
        (0, False, True, False),  # green controller, user's own --basetemp: left alone
        (0, False, False, True),  # green controller, no user basetemp: cleans its own
    ],
    ids=["green-worker", "red-worker", "green-controller-user-given", "green-controller"],
)
def test_pytest_sessionfinish(tmp_path, exitstatus, is_worker, user_given, expect_deleted):
    basetemp = _make_basetemp(tmp_path)
    factory = _FactoryStub(basetemp, basetemp if user_given else None)
    session = _SessionStub(_ConfigStub(factory, is_worker))

    conftest.pytest_sessionfinish(session, exitstatus=exitstatus)

    assert basetemp.exists() != expect_deleted


def test_nothing_made_is_a_no_op(tmp_path):
    session = _SessionStub(_ConfigStub(_FactoryStub(None, None), is_worker=False))

    conftest.pytest_sessionfinish(session, exitstatus=0)  # must not raise
