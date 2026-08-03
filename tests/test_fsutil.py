"""Tests for claude_swap.fsutil filesystem primitives."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from claude_swap.fsutil import read_text_with_retry, replace_with_retry


class TestReplaceWithRetry:
    """os.replace is not reliably available on Windows: antivirus and the
    search indexer open freshly-written files opportunistically, so a replace
    onto a just-created target fails with ERROR_ACCESS_DENIED/SHARING_VIOLATION
    for a few milliseconds. Measured at ~44% of replaces on a Defender-scanned
    temp dir, which silently broke credential and usage-store writes.
    """

    def _win_oserror(self, winerror: int) -> OSError:
        e = OSError(13, "Access is denied")
        e.winerror = winerror
        return e

    @pytest.mark.parametrize("winerror", [5, 32, 33])
    def test_retries_transient_windows_errors(self, tmp_path, monkeypatch, winerror):
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "win32")
        calls = []
        real_replace = os.replace

        def flaky(src, dst):
            calls.append(1)
            if len(calls) < 4:
                raise self._win_oserror(winerror)
            return real_replace(src, dst)

        monkeypatch.setattr("claude_swap.fsutil.os.replace", flaky)
        src = tmp_path / "tmp.tmp"
        src.write_text("payload")
        dst = tmp_path / "target.json"

        replace_with_retry(src, dst)

        assert len(calls) == 4
        assert dst.read_text() == "payload"

    def test_gives_up_and_raises_after_attempts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "win32")

        def always_fail(src, dst):
            raise self._win_oserror(5)

        monkeypatch.setattr("claude_swap.fsutil.os.replace", always_fail)
        with pytest.raises(OSError):
            replace_with_retry(tmp_path / "a", tmp_path / "b", attempts=3)

    def test_does_not_retry_real_errors(self, tmp_path, monkeypatch):
        """A genuine failure (e.g. missing source) must surface immediately."""
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "win32")
        calls = []

        def missing(src, dst):
            calls.append(1)
            raise self._win_oserror(2)  # ERROR_FILE_NOT_FOUND

        monkeypatch.setattr("claude_swap.fsutil.os.replace", missing)
        with pytest.raises(OSError):
            replace_with_retry(tmp_path / "a", tmp_path / "b")
        assert len(calls) == 1

    def test_posix_never_retries(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "linux")
        calls = []

        def fail(src, dst):
            calls.append(1)
            raise self._win_oserror(5)

        monkeypatch.setattr("claude_swap.fsutil.os.replace", fail)
        with pytest.raises(OSError):
            replace_with_retry(tmp_path / "a", tmp_path / "b")
        assert len(calls) == 1

    def test_rejects_nonpositive_attempts(self, tmp_path):
        """attempts < 1 would silently skip the replace; refuse it instead."""
        src = tmp_path / "tmp.tmp"
        src.write_text("payload")
        with pytest.raises(ValueError):
            replace_with_retry(src, tmp_path / "target.json", attempts=0)
        assert src.exists()


class TestReadTextWithRetry:
    """The read side of the same Windows window. `_write_json` publishes the
    roster by renaming onto `sequence.json`, so the file is freshly-modified
    exactly when AV/the indexer opens it — and a strict roster read raises
    ConfigError at ~59 call sites, including SwitchTransaction.rollback."""

    def _win_oserror(self, winerror: int) -> OSError:
        e = PermissionError(13, "Access is denied")
        e.winerror = winerror
        return e

    @pytest.mark.parametrize("winerror", [5, 32, 33])
    def test_retries_transient_windows_errors(self, tmp_path, monkeypatch, winerror):
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "win32")
        target = tmp_path / "sequence.json"
        target.write_text("payload")
        calls = []
        real_read = Path.read_text
        err = self._win_oserror(winerror)

        def flaky(path_self, *a, **kw):
            calls.append(1)
            if len(calls) < 4:
                raise err
            return real_read(path_self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", flaky)

        assert read_text_with_retry(target) == "payload"
        assert len(calls) == 4

    def test_gives_up_and_raises_after_attempts(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "win32")
        err = self._win_oserror(32)

        def always_fail(self, *a, **kw):
            raise err

        monkeypatch.setattr(Path, "read_text", always_fail)
        with pytest.raises(OSError):
            read_text_with_retry(tmp_path / "a", attempts=3)

    def test_posix_eacces_surfaces_immediately(self, tmp_path, monkeypatch):
        """A POSIX EACCES is a real, persistent permission problem: raise at
        once rather than stalling ~0.75s on a condition that will not clear."""
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "linux")
        calls = []
        err = self._win_oserror(5)

        def fail(self, *a, **kw):
            calls.append(1)
            raise err

        monkeypatch.setattr(Path, "read_text", fail)
        with pytest.raises(OSError):
            read_text_with_retry(tmp_path / "a")
        assert len(calls) == 1

    def test_does_not_retry_a_non_contention_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_swap.fsutil.sys.platform", "win32")
        calls = []
        err = self._win_oserror(2)  # ERROR_FILE_NOT_FOUND

        def missing(self, *a, **kw):
            calls.append(1)
            raise err

        monkeypatch.setattr(Path, "read_text", missing)
        with pytest.raises(OSError):
            read_text_with_retry(tmp_path / "a")
        assert len(calls) == 1

    def test_rejects_nonpositive_attempts(self, tmp_path):
        target = tmp_path / "a.json"
        target.write_text("x")
        with pytest.raises(ValueError):
            read_text_with_retry(target, attempts=0)


class TestStrictRosterReadBranches:
    """`_read_json(strict=True)`'s OSError and non-dict branches were both
    reachable and both unkilled. `_get_sequence_data` is strict because ~59
    call sites read it and 27 write the result back through `or {}` — a torn
    or unreadable roster read as "no accounts" rebuilds it from nothing."""

    def _switcher(self, tmp_path):
        import logging

        from claude_swap.switcher import ClaudeAccountSwitcher

        s = ClaudeAccountSwitcher.__new__(ClaudeAccountSwitcher)
        s._logger = logging.getLogger("test")
        return s

    def test_a_roster_holding_a_list_is_refused_not_dereferenced(self, tmp_path):
        """Without the guard the caller gets AttributeError: 'list' object has
        no attribute 'get' — uncaught by cli.py's `except ClaudeSwitchError`,
        so `--json` emits no envelope at all."""
        from claude_swap.exceptions import ConfigError

        p = tmp_path / "sequence.json"
        p.write_text("[1, 2, 3]")
        with pytest.raises(ConfigError, match="not a JSON object"):
            self._switcher(tmp_path)._read_json(p, strict=True)

    # ONE condition, and the POSIX check must be INSIDE it. `skipif`'s argument
    # is evaluated at COLLECTION, on every platform, so `os.geteuid()` in a
    # second decorator runs before the win32 skip above it can apply — and
    # Windows has no `geteuid`, so the module fails to import and the WHOLE
    # suite is interrupted (1819 collected / 1 error), not just this test.
    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0,
        reason="POSIX mode semantics; root reads through a 0000 mode",
    )
    def test_an_unreadable_roster_is_refused_not_read_as_empty(self, tmp_path):
        """The distinction the strict reader exists for: `None` here would be
        spliced back as "no accounts" and destroy the roster on the next
        write."""
        from claude_swap.exceptions import ConfigError

        p = tmp_path / "sequence.json"
        p.write_text('{"sequence": [1], "accounts": {"1": {}}}')
        os.chmod(p, 0o000)
        try:
            with pytest.raises(ConfigError, match="could not be read"):
                self._switcher(tmp_path)._read_json(p, strict=True)
        finally:
            os.chmod(p, 0o600)

    def test_non_strict_keeps_the_soft_none(self, tmp_path):
        """Read-only callers that pass no `strict` still get `None`, so this
        change cannot turn an advisory read into a hard failure."""
        p = tmp_path / "config.json"
        p.write_text("[1, 2, 3]")
        assert self._switcher(tmp_path)._read_json(p) is None
