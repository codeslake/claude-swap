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
        monkeypatch.setattr(pin, "clear_wiring", lambda: False)
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

        cfg = self._wired(tmp_path)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        assert clear_wiring() is True
        env = json.loads(cfg.read_text())["env"]
        assert "CSWAP_PIN_PORT" not in env
        assert env["UNRELATED"] == "keep me"
        assert env["HTTPS_PROXY"] == "http://127.0.0.1:9901"  # displaced value back

    def test_clearing_an_unwired_config_is_a_no_op(self, tmp_path, monkeypatch):
        import claude_swap.paths as paths
        from claude_swap.pin import clear_wiring

        cfg = tmp_path / ".claude.json"
        cfg.write_text(json.dumps({"env": {"UNRELATED": "keep me"}}))
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        assert clear_wiring() is False
        assert json.loads(cfg.read_text())["env"] == {"UNRELATED": "keep me"}

    def test_the_config_is_not_left_world_readable(self, tmp_path, monkeypatch):
        """It can hold primaryApiKey and inline MCP credentials; a plain write
        takes the umask and rename publishes that mode."""
        import os
        import stat as _stat

        import claude_swap.paths as paths
        from claude_swap.pin import clear_wiring

        cfg = self._wired(tmp_path)
        os.chmod(cfg, 0o644)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        old = os.umask(0o022)
        try:
            assert clear_wiring() is True
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
