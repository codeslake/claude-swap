"""The pin is an optional extra: cswap must work fully without it.

Mirrors how menubar is treated — the module is import-safe without the
dependency, and the missing extra surfaces as a ClaudeSwitchError naming the
install, not as a traceback.
"""

import json
import sys

import pytest

from claude_swap.exceptions import ClaudeSwitchError


class TestImportSafeWithoutTheExtra:
    def test_the_module_imports(self):
        """A top-level import of the optional dependency would make cswap
        refuse to start at all — no switching, no TUI — over a feature the
        user opted out of."""
        import importlib

        importlib.import_module("claude_swap.pin")

    def test_nothing_imports_cswap_pin_at_module_scope(self):
        """Behavioural tests pass right up until someone adds a top-level
        import; assert the seam itself. Walks the whole tree, not just
        module-level statements, so a conditional import at module scope
        (`if sys.platform != "win32": import cswap_pin`) is caught too."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src" / "claude_swap"
        offenders = []
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            nested = set()
            for fn in ast.walk(tree):
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    nested.update(id(c) for c in ast.walk(fn))
            for node in ast.walk(tree):
                if id(node) in nested:
                    continue
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                if any(n.startswith("cswap_pin") for n in names):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, f"module-scope import of cswap_pin: {offenders}"

    def test_the_whole_package_imports_without_it(self):
        """Runs in a subprocess with cswap_pin blocked at sys.meta_path —
        importlib.import_module does not go through builtins.__import__, so a
        __import__ patch would miss that form."""
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        code = textwrap.dedent(
            f"""
            import pkgutil, sys
            sys.path.insert(0, {src!r})
            class Block:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "cswap_pin":
                        raise ImportError("blocked", name=name)
                    return None
            sys.meta_path.insert(0, Block())
            import claude_swap
            for m in pkgutil.walk_packages(claude_swap.__path__, "claude_swap."):
                __import__(m.name)
            """
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-900:]


class TestTheMissingExtraIsReported:
    def test_impl_raises_the_install_hint(self, monkeypatch):
        import importlib.util

        from claude_swap import pin

        monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: None)
        with pytest.raises(ClaudeSwitchError, match=r"claude-swap\[pin\]"):
            pin._impl()

    def test_a_broken_dependency_is_not_reported_as_missing(self, monkeypatch):
        """The package is THERE and its own import fails (a missing
        cryptography). That must surface, not be rewritten into 'install the
        pin extra' — advice that would be wrong."""
        import importlib
        import importlib.util

        from claude_swap import pin

        monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: object())

        def boom(name, package=None):
            raise ImportError("No module named 'cryptography'", name="cryptography")

        monkeypatch.setattr(importlib, "import_module", boom)
        with pytest.raises(ImportError, match="cryptography"):
            pin._impl()


class TestLaunchIsNeverBlocked:
    def test_wire_launch_env_passes_through_without_the_extra(self, monkeypatch):
        from claude_swap import pin

        monkeypatch.setattr(pin, "_impl", lambda: (_ for _ in ()).throw(
            ClaudeSwitchError("nope")))
        monkeypatch.setattr(pin, "clear_wiring", lambda *a, **k: False)
        env = {"A": "1"}
        assert pin.wire_launch_env(object(), env) == env

    def test_a_failing_pin_does_not_block_the_launch(self, monkeypatch):
        """An optional feature must never be able to stop claude from starting."""
        import types

        from claude_swap import pin

        impl = types.SimpleNamespace(
            ensure_proxy=lambda sw: (_ for _ in ()).throw(RuntimeError("boom"))
        )
        monkeypatch.setattr(pin, "_impl", lambda: impl)
        env = {"A": "1"}
        assert pin.wire_launch_env(object(), env) == env


class TestTheWiringCanAlwaysBeRemoved:
    """`.claude.json` names the pin's port, and Claude Code applies that env
    block at boot. If only the optional package could remove it, uninstalling
    the pin would strand every launch dialling a dead port."""

    def _wired(self, tmp_path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        cfg = tmp_path / ".claude.json"
        cfg.write_text(
            json.dumps(
                {
                    "env": {
                        "HTTPS_PROXY": "http://127.0.0.1:36301",
                        "CSWAP_PIN_PORT": "36301",
                        "UNRELATED": "keep me",
                    },
                    "_cswapPinWiredKeys": ["HTTPS_PROXY", "CSWAP_PIN_PORT"],
                    "_cswapPinWiredKeysSaved": {"HTTPS_PROXY": "http://127.0.0.1:9901"},
                }
            )
        )
        return cfg

    def test_clear_wiring_works_without_the_extra(self, tmp_path, monkeypatch):
        import claude_swap.paths as paths
        from claude_swap.pin import clear_wiring
        from claude_swap.switcher import ClaudeAccountSwitcher

        cfg = self._wired(tmp_path)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        assert clear_wiring(ClaudeAccountSwitcher()) is True
        env = json.loads(cfg.read_text())["env"]
        assert "CSWAP_PIN_PORT" not in env
        assert env["UNRELATED"] == "keep me"
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:9901"  # displaced value back

    def test_clearing_an_unwired_config_is_a_no_op(self, tmp_path, monkeypatch):
        import claude_swap.paths as paths
        from claude_swap.pin import clear_wiring
        from claude_swap.switcher import ClaudeAccountSwitcher

        cfg = tmp_path / ".claude.json"
        cfg.write_text(json.dumps({"env": {"UNRELATED": "keep me"}}))
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        assert clear_wiring(ClaudeAccountSwitcher()) is False
        assert json.loads(cfg.read_text())["env"] == {"UNRELATED": "keep me"}

    def test_clear_also_reaches_the_default_profile_from_inside_a_session(
        self, tmp_path, monkeypatch
    ):
        """`cswap run` sets CLAUDE_CONFIG_DIR in the CHILD's env, so a launch
        from a normal terminal wires ~/.claude.json while one from inside a
        session terminal wires the session's copy. Resolving one path clears
        whichever the caller happens to sit in and reports success over a
        config that still names a dead port."""
        import claude_swap.paths as paths
        from claude_swap.pin import clear_wiring
        from claude_swap.switcher import ClaudeAccountSwitcher

        session = self._wired(tmp_path / "session")
        default = self._wired(tmp_path / "home")
        monkeypatch.setattr(paths, "get_global_config_path", lambda: session)
        monkeypatch.setattr(paths, "get_default_global_config_path", lambda: default)

        assert clear_wiring(ClaudeAccountSwitcher()) is True
        for cfg in (session, default):
            assert "_cswapPinWiredKeys" not in json.loads(cfg.read_text()), (
                f"{cfg.parent.name} left wired — its sessions still dial a dead port"
            )

    def test_the_launch_path_does_not_wait_on_the_config_lock(
        self, tmp_path, monkeypatch
    ):
        """clear_wiring takes Claude Code's config lock, whose default wait is
        9s, and the launch path calls it on EVERY `cswap run` for users who
        will never install the pin. Claude Code itself holds that lock while
        refreshing credentials, so an unbounded wait stalls the launch."""
        import time

        import claude_swap.paths as paths
        from claude_swap import claude_locks, pin

        cfg = self._wired(tmp_path)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        monkeypatch.setattr(paths, "get_default_global_config_path", lambda: cfg)
        monkeypatch.setattr(
            pin, "_impl", lambda: (_ for _ in ()).throw(ClaudeSwitchError("absent"))
        )

        # Derived here rather than via config_lock_dir() so the test does not
        # depend on how that resolves the config path.
        held = cfg.parent / (cfg.name + ".lock")
        monkeypatch.setattr(claude_locks, "config_lock_dir", lambda: held)
        # A live holder: fresh mtime, so the staleness takeover does not fire.
        held.mkdir()
        try:
            start = time.monotonic()
            env = pin.wire_launch_env(object(), {"A": "1"})
            waited = time.monotonic() - start
        finally:
            held.rmdir()

        assert env == {"A": "1"}
        assert waited < 3.0, f"a contended launch blocked for {waited:.2f}s"

    def test_the_config_is_not_left_world_readable(self, tmp_path, monkeypatch):
        """It can hold primaryApiKey and inline MCP credentials; a plain write
        takes the umask and rename publishes that mode."""
        import os
        import stat as _stat

        import claude_swap.paths as paths
        from claude_swap.pin import clear_wiring
        from claude_swap.switcher import ClaudeAccountSwitcher

        cfg = self._wired(tmp_path)
        os.chmod(cfg, 0o644)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        old = os.umask(0o022)
        try:
            assert clear_wiring(ClaudeAccountSwitcher()) is True
        finally:
            os.umask(old)
        assert not _stat.S_IMODE(cfg.stat().st_mode) & 0o077


class TestTheLaunchPathIsWired:
    """`cswap run` must route its child through the proxy, not only
    hand-launched sessions that read .claude.json.

    wire_launch_env existed with zero production callers — session.py never
    mentioned the pin — so `cswap run 2` launched unpinned while `cswap pin`
    reported success. Found by a complexity review flagging it as dead code;
    it was a missing call, not spare code.
    """

    def _manager(self, temp_home):
        # The real switcher, not a stub: _exec touches enough of it that a
        # SimpleNamespace only proves the stub matches itself.
        from claude_swap.session import SessionManager
        from claude_swap.switcher import ClaudeAccountSwitcher

        return SessionManager(ClaudeAccountSwitcher())

    def test_exec_routes_the_child_through_the_pin(self, temp_home, monkeypatch):
        from claude_swap import pin as pin_mod
        from claude_swap import session as session_mod

        monkeypatch.setattr(
            pin_mod,
            "wire_launch_env",
            lambda sw, env: {**env, "HTTPS_PROXY": "http://127.0.0.1:9955"},
        )
        captured = {}

        def fake_execvpe(binary, argv, env):
            captured["env"] = env
            raise SystemExit(0)

        monkeypatch.setattr(session_mod.os, "execvpe", fake_execvpe)
        with pytest.raises(SystemExit):
            self._manager(temp_home)._exec("/bin/claude", [], env={"A": "1"})
        assert captured["env"].get("HTTPS_PROXY") == "http://127.0.0.1:9955", (
            "the launch path does not wire the pin — `cswap run` goes out unpinned"
        )

    def test_a_pin_failure_still_launches(self, temp_home, monkeypatch):
        from claude_swap import pin as pin_mod
        from claude_swap import session as session_mod

        def boom(sw, env):
            raise RuntimeError("pin exploded")

        monkeypatch.setattr(pin_mod, "wire_launch_env", boom)
        monkeypatch.setattr(
            session_mod.os, "execvpe", lambda b, a, e: (_ for _ in ()).throw(SystemExit(0))
        )
        with pytest.raises((SystemExit, RuntimeError)) as exc:
            self._manager(temp_home)._exec("/bin/claude", [], env={"A": "1"})
        assert exc.type is SystemExit, "a pin failure blocked the launch"


class TestWindowsIsRejectedCleanly:
    """cswap advertises Windows support; the pin cannot honour it.

    The proxy takes its daemon lock with fcntl.flock and refcounts sessions
    through os.mkfifo. Without this guard a Windows user gets a
    ModuleNotFoundError from inside the dependency instead of a sentence they
    can act on — and only at first use, after `pip install claude-swap[pin]`
    appeared to succeed.
    """

    def test_it_says_so_rather_than_failing_inside_the_dependency(self, monkeypatch):
        import importlib.util
        import sys as _sys

        from claude_swap import pin

        monkeypatch.setattr(_sys, "platform", "win32")
        # Even with the package apparently installed, it must refuse.
        monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: object())
        with pytest.raises(ClaudeSwitchError, match="Windows"):
            pin._impl()


class TestClearRunsWithTheExtraGone:
    """`cswap pin --clear` is priority 1 and the whole reason clear_wiring
    lives in cswap rather than the optional package.

    An AST scan proves nothing about it: a review defeated the import-time
    guards by adding a runtime `from cswap_pin.proxy import ...` inside
    pin.run(), and every test stayed green while `--clear` died with a
    ModuleNotFoundError. Drive the real command with cswap_pin blocked at
    sys.meta_path — the one form that also stops importlib.import_module.
    """

    def test_the_marker_still_matches_the_package_that_writes_it(self):
        """cswap READS a key cswap-pin WRITES, and the two version
        independently. Agreeing on a magic string by convention is this seam's
        one silent-drift risk: rename it there and `--clear` stops finding
        wirings while still reporting 'No cloud account pinned'.

        Skipped without the extra because there is nothing to compare against
        — the assertion is about two installed packages agreeing.
        """
        proxy = pytest.importorskip("cswap_pin.proxy")

        from claude_swap.pin import _WIRE_MARK

        assert proxy._WIRE_MARK == _WIRE_MARK

    def test_clear_removes_the_wiring_with_cswap_pin_blocked(self, tmp_path):
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        cfg = tmp_path / ".claude.json"
        cfg.write_text(
            json.dumps(
                {
                    "env": {"HTTPS_PROXY": "http://127.0.0.1:36301", "K": "v"},
                    "_cswapPinWiredKeys": ["HTTPS_PROXY"],
                }
            )
        )
        code = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {src!r})
            class Block:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "cswap_pin":
                        raise ImportError("blocked", name=name)
                    return None
            sys.meta_path.insert(0, Block())
            from pathlib import Path
            import claude_swap.paths as paths
            cfg = Path({str(cfg)!r})
            paths.get_global_config_path = lambda: cfg
            paths.get_default_global_config_path = lambda: cfg
            from claude_swap import pin
            from claude_swap.switcher import ClaudeAccountSwitcher
            sys.exit(pin.run(ClaudeAccountSwitcher(), None, clear=True))
            """
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-900:]
        raw = json.loads(cfg.read_text())
        assert "_cswapPinWiredKeys" not in raw
        assert raw["env"] == {"K": "v"}, "the wiring outlived the uninstall"
