"""The test session ends without a long exit-time garbage collection.

Under ``-n N`` the controller waits for every worker interpreter to exit. A
worker that finishes with a large heap of gc-tracked objects spends seconds
walking it during finalization; ``gc.freeze()`` in ``pytest_sessionfinish``
moves that heap out of the collector's view first.
"""
from __future__ import annotations

import gc

from tests import conftest


def test_sessionfinish_freezes_the_heap():
    before = gc.get_freeze_count()
    try:
        conftest.pytest_sessionfinish()
        assert gc.get_freeze_count() > before
    finally:
        gc.unfreeze()
