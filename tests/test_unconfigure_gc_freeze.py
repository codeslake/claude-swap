"""The test session ends without a long exit-time garbage collection.

Under ``-n N`` the controller waits for every worker interpreter to exit. A
worker that finishes with a large heap of gc-tracked objects spends seconds
walking it during finalization; ``gc.freeze()`` in a last ``pytest_unconfigure``
moves that heap out of the collector's view, after pytest's own end-of-run
sweep for unraisable exceptions has collected and reported what it found.
"""
from __future__ import annotations

import gc

from tests import conftest


def test_unconfigure_freezes_the_heap():
    before = gc.get_freeze_count()
    try:
        conftest.pytest_unconfigure()
        assert gc.get_freeze_count() > before
    finally:
        if before == 0:
            gc.unfreeze()


def test_freeze_runs_after_pytests_unraisable_sweep(request):
    impls = request.config.pluginmanager.hook.pytest_unconfigure.get_hookimpls()
    # pluggy calls the last-listed implementation first.
    called = [impl.function for impl in reversed(impls)]
    sweep = next(i.function for i in impls if i.plugin_name == "unraisableexception")
    assert called.index(conftest.pytest_unconfigure) > called.index(sweep)
