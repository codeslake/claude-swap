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


@pytest.fixture
def posix(monkeypatch):
    """Force the POSIX branch of _impl.

    Everything in this class tests how the optional dependency is RESOLVED,
    which is platform-independent — but on Windows _impl refuses before it
    gets there, so the resolution logic would go untested on exactly one CI
    runner. Pinning the platform tests the logic on all three rather than
    skipping it where it happens not to run.
    """
    import sys as _sys

    monkeypatch.setattr(_sys, "platform", "linux")


class TestTheMissingExtraIsReported:
    def test_impl_raises_the_install_hint(self, posix, monkeypatch):
        import importlib.util

        from claude_swap import pin

        monkeypatch.setattr(importlib.util, "find_spec", lambda *a, **k: None)
        with pytest.raises(ClaudeSwitchError, match=r"claude-swap\[pin\]"):
            pin._impl()

    def test_a_broken_package_ROOT_is_not_reported_as_missing(
        self, posix, tmp_path, monkeypatch
    ):
        """The breakage that actually happens is in the package ROOT, not the
        submodule: cswap_pin/__init__.py imports cryptography.

        find_spec has to IMPORT the parent to read its __path__, so that error
        comes out of find_spec — not out of the import_module below it. The
        sibling test stubs find_spec to succeed, so it proves nothing about
        this path, and the bug it missed told users to install a package they
        already had.

        Real files, no stubs: stubbing find_spec is exactly what hid this.

        In-process, not a subprocess: a subprocess would have to fake the
        platform to get past the Windows refusal, and faking it before the
        imports run makes claude_swap.locking pick the POSIX branch and die
        on `import fcntl`. The `posix` fixture reaches _impl without that.
        """
        import importlib
        import importlib.util

        pkg = tmp_path / "cswap_pin"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            "raise ImportError(\"No module named 'cryptography'\", "
            "name='cryptography')"
        )
        (pkg / "proxy.py").write_text("")

        from claude_swap import pin

        monkeypatch.syspath_prepend(str(tmp_path))
        for name in [m for m in sys.modules if m.split(".")[0] == "cswap_pin"]:
            monkeypatch.delitem(sys.modules, name)
        importlib.invalidate_caches()

        with pytest.raises(ImportError) as exc:
            pin._impl()
        assert exc.value.name == "cryptography", (
            f"reported as {exc.value!r} — a broken package root must not be "
            "rewritten into 'install the extra'"
        )

    def test_a_broken_dependency_is_not_reported_as_missing(self, posix, monkeypatch):
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
        from claude_swap import pin

        cfg = self._wired(tmp_path)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        monkeypatch.setattr(paths, "get_default_global_config_path", lambda: cfg)
        monkeypatch.setattr(
            pin, "_impl", lambda: (_ for _ in ()).throw(ClaudeSwitchError("absent"))
        )

        # The same name clear_wiring derives, held for real. Nothing is
        # patched here: a monkeypatch of config_lock_dir used to sit in this
        # test and was dead — clear_wiring stopped calling it, and the test
        # stayed green with that function rigged to raise on call.
        held = cfg.parent / (cfg.name + ".lock")
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

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="NTFS has no POSIX mode bits; switcher._write_json skips the "
        "chmod on win32 for the same reason",
    )
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

        # _exec forks two ways — execvpe on POSIX, subprocess.run on Windows —
        # and the pin has to be wired on BOTH. Stub whichever this platform
        # actually takes, rather than skipping the Windows runner and leaving
        # that branch unasserted.
        monkeypatch.setattr(session_mod.os, "execvpe", fake_execvpe)
        monkeypatch.setattr(
            session_mod.subprocess,
            "run",
            lambda argv, env=None, **kw: fake_execvpe(argv[0], argv, env),
        )
        with pytest.raises(SystemExit):
            self._manager(temp_home)._exec("claude", [], env={"A": "1"})
        assert captured["env"].get("HTTPS_PROXY") == "http://127.0.0.1:9955", (
            "the launch path does not wire the pin — `cswap run` goes out unpinned"
        )

    def test_a_pin_failure_still_launches(self, temp_home, monkeypatch):
        from claude_swap import pin as pin_mod
        from claude_swap import session as session_mod

        def boom(sw, env):
            raise RuntimeError("pin exploded")

        monkeypatch.setattr(pin_mod, "wire_launch_env", boom)

        def launched(*a, **kw):
            raise SystemExit(0)

        monkeypatch.setattr(session_mod.os, "execvpe", launched)
        monkeypatch.setattr(session_mod.subprocess, "run", launched)
        with pytest.raises((SystemExit, RuntimeError)) as exc:
            self._manager(temp_home)._exec("claude", [], env={"A": "1"})
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


class TestClearReachesBothConfigsWithTheExtraINSTALLED:
    """The two-path clear must hold for users who HAVE the pin.

    clear_wiring only ran in the except branch, so on the happy path the
    unwiring was done entirely by the package's own single-path resolver:
    `cswap pin --clear` from inside a session terminal cleared that session's
    config and left ~/.claude.json naming a dead port, while printing
    "Unpinned". Everyone has the extra at the moment they unpin, so the
    guarantee held for exactly nobody.
    """

    def test_the_default_profile_is_cleared_too(self, tmp_path, monkeypatch):
        import types

        import claude_swap.paths as paths
        from claude_swap import pin
        from claude_swap.switcher import ClaudeAccountSwitcher

        def wired(where):
            where.mkdir(parents=True, exist_ok=True)
            cfg = where / ".claude.json"
            cfg.write_text(
                json.dumps(
                    {
                        "env": {"HTTPS_PROXY": "http://127.0.0.1:44444"},
                        "_cswapPinWiredKeys": ["HTTPS_PROXY"],
                    }
                )
            )
            return cfg

        session = wired(tmp_path / "session")
        default = wired(tmp_path / "home")
        monkeypatch.setattr(paths, "get_global_config_path", lambda: session)
        monkeypatch.setattr(paths, "get_default_global_config_path", lambda: default)

        # The extra IS installed and apply_pin succeeds — the path that had no
        # clear_wiring call at all. It unwires only what its own resolver sees.
        def apply_pin(switcher, email, org):
            session.write_text(json.dumps({"env": {}}))
            return False

        monkeypatch.setattr(
            pin,
            "_impl",
            lambda: types.SimpleNamespace(
                apply_pin=apply_pin, load_pin=lambda d: ("a@b.c", None)
            ),
        )
        pin.run(ClaudeAccountSwitcher(), None, clear=True)
        assert "_cswapPinWiredKeys" not in json.loads(default.read_text()), (
            "the default profile stayed wired to a dead port while --clear "
            "reported success"
        )


class TestTheTuiSurfaceSurvivesTheSplit:
    """The pin has a TUI half, and the split dropped it once already.

    The old in-tree pin wired three files — dashboard.py (menu row + submenu +
    action), widgets.py (the ○ cloud badge), autoview.py (the same badge on
    the auto view). The first cut of this seam carried the CLI and launch
    paths and none of that, and every check stayed green: the CLI probe
    passed, the daemon answered, and the running TUIs were serving code they
    had exec'd 16 hours before the cutover, so the badge was still on screen.
    A human looking at the screen is what caught it.

    These assert the surface exists at all. A check that exercises only one
    surface reports the other as healthy.
    """

    def test_no_extra_means_no_pin_row(self, monkeypatch):
        """A user who never asked for the pin must not see a row for it."""
        import types

        from claude_swap.tui import dashboard

        monkeypatch.setattr(dashboard.pin, "is_available", lambda: False)
        monkeypatch.setattr(
            dashboard.DashboardScreen,
            "app",
            property(lambda self: types.SimpleNamespace(switcher=object(), snapshot=None)),
            raising=False,
        )
        screen = object.__new__(dashboard.DashboardScreen)
        ids = [a for _l, a in screen._root_entries()]
        assert "pin-menu" not in ids, f"pin offered without the extra: {ids}"

        monkeypatch.setattr(dashboard.pin, "is_available", lambda: True)
        monkeypatch.setattr(dashboard.pin, "pinned_email", lambda sw: None)
        assert "pin-menu" in [a for _l, a in screen._root_entries()]

    def test_installing_the_extra_is_seen_without_a_restart(self, tmp_path):
        """A TUI open across an install must start offering the pin.

        A long-lived process caches each sys.path directory by mtime, so a
        package installed after start can stay invisible — measured, usually
        visible but not when the install lands inside the same mtime tick.
        That is the "I installed it and the menu is still missing" report,
        and invalidate_caches is what closes it.
        """
        import subprocess
        import textwrap
        from pathlib import Path

        pkg = tmp_path / "late" / "cswap_pin"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "proxy.py").write_text("def load_pin(d):\n    return None\n")
        src = str(Path(__file__).resolve().parent.parent / "src")
        code = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {src!r})
            from claude_swap import pin
            # _impl refuses on win32 BEFORE it looks for the package, so
            # is_available is False there no matter what gets installed —
            # a cross-platform claim cannot be tested through a
            # platform-gated function. Assert on the resolution step, which
            # is what invalidate_caches actually affects. (Measured: this is
            # why the Windows runner failed while linux and macos passed.)
            import importlib, importlib.util
            def resolvable():
                importlib.invalidate_caches()
                try:
                    return importlib.util.find_spec("cswap_pin.proxy") is not None
                except ImportError:
                    return False
            assert resolvable() is False, "saw a package that is not there"
            # The install: a path entry that did not exist when we started.
            sys.path.insert(0, {str(tmp_path / "late")!r})
            assert resolvable() is True, "a restart should not be required"
            # And the seam honours it on any platform that has the pin at all.
            if sys.platform != "win32":
                assert pin.is_available() is True
            sys.exit(0)
            """
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-400:]

    def test_the_menu_rebuilds_on_the_poll(self):
        """The root menu was built once at mount, so a row that appears on
        install could not appear until a restart."""
        import inspect

        from claude_swap.tui import dashboard

        assert hasattr(dashboard.DashboardScreen, "refresh_root_menu")
        assert "refresh_root_menu" in inspect.getsource(
            dashboard.DashboardScreen.on_mount
        ) or "_refresh_menu_on_snapshot" in inspect.getsource(
            dashboard.DashboardScreen.on_mount
        ), "nothing rebuilds the root menu after mount"

    def test_opening_the_pin_submenu_lists_the_accounts(self, monkeypatch):
        """The row existing is not the same as the row WORKING.

        Every other guard here asserts the pin surface is present; none of
        them called it. A stray `del impl` on the last line shipped an
        UnboundLocalError into the one function the menu row opens — the
        submenu raised on every successful call, and the whole suite was
        green. Call it.
        """
        import types

        from claude_swap.tui import dashboard

        acc = types.SimpleNamespace(number="2", email="a@b.c", alias=None)
        monkeypatch.setattr(dashboard.pin, "_impl", lambda: types.SimpleNamespace())
        monkeypatch.setattr(dashboard.pin, "pinned_email", lambda sw: "a@b.c")
        monkeypatch.setattr(
            dashboard.DashboardScreen,
            "app",
            property(lambda self: types.SimpleNamespace(
                switcher=object(), snapshot=types.SimpleNamespace(accounts=[acc]))),
            raising=False,
        )
        screen = object.__new__(dashboard.DashboardScreen)
        entries = screen._pin_entries()
        labels = [label for label, _a in entries]
        actions = [a for _l, a in entries]
        assert "pin:2" in actions, f"the account is not pinnable: {actions}"
        assert any("○ cloud" in label for label in labels), "the pinned account is unmarked"
        assert "pin:clear" in actions, "no way to unpin from the TUI"

    def test_the_badge_helper_is_reachable_and_fails_open(self, monkeypatch):
        """pinned_email answers the TUI's one question and never raises: no
        extra, no pin, and a malformed pin file all render as no badge."""
        from claude_swap import pin

        monkeypatch.setattr(
            pin, "_impl", lambda: (_ for _ in ()).throw(ClaudeSwitchError("absent"))
        )
        assert pin.pinned_email(object()) is None

        monkeypatch.setattr(
            pin, "_impl", lambda: (_ for _ in ()).throw(RuntimeError("broken"))
        )
        assert pin.pinned_email(object()) is None

    def test_the_dashboard_root_menu_still_offers_the_pin(self, monkeypatch):
        """The ROOT MENU ROW, not merely the handler behind it.

        Grepping the module for "pin-menu" passes with the row deleted — the
        action handler still contains the string — so the row can vanish while
        the check stays green. Measured. Call _root_entries and look at the
        ids it actually returns.
        """
        import types

        from claude_swap.tui import dashboard

        assert "none" in dashboard.cloud_menu_label(None)
        assert "a@b.c" in dashboard.cloud_menu_label("a@b.c")

        # `app` is a read-only Textual property, so patch it on the class.
        monkeypatch.setattr(
            dashboard.DashboardScreen,
            "app",
            property(lambda self: types.SimpleNamespace(switcher=object(), snapshot=None)),
            raising=False,
        )
        # With the extra present — its absence hiding the row is the sibling
        # test; this one is about the row existing at all when it should.
        monkeypatch.setattr(dashboard.pin, "is_available", lambda: True)
        monkeypatch.setattr(dashboard.pin, "pinned_email", lambda sw: None)
        screen = object.__new__(dashboard.DashboardScreen)
        ids = [action for _label, action in screen._root_entries()]
        assert "pin-menu" in ids, (
            f"the cloud pin is unreachable from the dashboard menu: {ids}"
        )
        labels = [label for label, action in screen._root_entries()
                  if action == "pin-menu"]
        assert "Cloud account" in labels[0]

    def test_both_account_renderers_take_the_badge(self):
        """The badge rides on the account rows, and there are two renderers —
        the full card and the minimised line. Losing it from one is the half
        that reads as healthy."""
        import inspect

        from claude_swap.tui import widgets

        for fn in (widgets.account_card_text, widgets.mini_account_text):
            assert "cloud_pinned" in inspect.signature(fn).parameters, (
                f"{fn.__name__} cannot render the cloud badge"
            )
        assert "○ cloud" in inspect.getsource(widgets)
        assert "○ cloud" in inspect.getsource(
            __import__("claude_swap.tui.autoview", fromlist=["x"])
        ), "the auto-switch view lost the cloud badge"

    def test_a_pinned_account_actually_renders_the_badge(self):
        """Not just the parameter — the glyph has to reach the text."""
        from claude_swap.tui.widgets import account_card_text
        from tests.test_tui import make_account

        acc = make_account(1, active=True)
        plain = account_card_text(acc, 80, cloud_pinned=True).plain
        assert "○ cloud" in plain
        assert "○ cloud" not in account_card_text(acc, 80).plain


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


class TestAFailedClearIsNotReportedAsSuccess:
    """`--clear` must not say "Unpinned" while the pin survives.

    The failure is silent by construction: the success message was gated on
    clear_wiring(), which reports on the .claude.json wiring and never on the
    pin itself, so every way of failing printed "Unpinned" over a live pin.

    The pin record is cswap's OWN file, so these drive it through that file
    rather than through a stubbed package. That is the point of the fix: an
    earlier version asked the package "is it still pinned", which cannot answer
    when the package is the thing that is broken.
    """

    def _run(self, tmp_path, impl_src):
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        cfg = tmp_path / ".claude.json"
        cfg.write_text(json.dumps({"env": {}}))
        backup = tmp_path / "backup"
        backup.mkdir()
        # A real pin record, written the way cswap writes one.
        (backup / "settings.json").write_text(
            json.dumps({"remoteControl": {"pinnedEmail": "cloud@example.com"}}, indent=2)
        )
        code = (
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {src!r})
                from pathlib import Path
                import claude_swap.paths as paths
                cfg = Path({str(cfg)!r})
                paths.get_global_config_path = lambda: cfg
                paths.get_default_global_config_path = lambda: cfg
                from claude_swap import pin
                """
            )
            + impl_src
            + textwrap.dedent(
                f"""
                pin._impl = _impl_factory
                from claude_swap.switcher import ClaudeAccountSwitcher
                sw = ClaudeAccountSwitcher()
                sw.backup_dir = Path({str(backup)!r})
                sys.exit(pin.run(sw, None, clear=True))
                """
            )
        )
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )

    def test_a_broken_package_cannot_report_success(self, tmp_path):
        """_impl() ITSELF raising is the case the guard has to cover.

        A broken `cryptography` does not make apply_pin raise -- cswap_pin's
        __init__ imports nothing from proxy, so find_spec succeeds and
        import_module("cswap_pin.proxy") is what raises, one line before
        apply_pin is ever reached.
        """
        impl = (
            "def _impl_factory():\n"
            "    raise ImportError('cryptography')\n"
        )
        r = self._run(tmp_path, impl)
        assert "Unpinned" not in r.stdout, (
            "reported success while the pin survived -- the user stops looking"
        )
        assert "Could not remove" in r.stdout, r.stdout + r.stderr[-400:]
        assert r.returncode == 1, "a failed clear must not exit 0"

    def test_a_failing_apply_pin_cannot_report_success(self, tmp_path):
        impl = (
            "class _I:\n"
            "    def apply_pin(self, *a): raise OSError('disk full')\n"
            "def _impl_factory(): return _I()\n"
        )
        r = self._run(tmp_path, impl)
        assert "Unpinned" not in r.stdout
        assert "Could not remove" in r.stdout, r.stdout + r.stderr[-400:]
        assert r.returncode == 1

    def test_a_real_clear_still_reports_success(self, tmp_path):
        """The control: the message must be right exactly when it worked."""
        impl = (
            "import json as _j\n"
            "class _I:\n"
            "    def apply_pin(self, sw, *a):\n"
            "        (sw.backup_dir / 'settings.json').write_text(_j.dumps({}))\n"
            "def _impl_factory(): return _I()\n"
        )
        r = self._run(tmp_path, impl)
        assert "Unpinned" in r.stdout, r.stdout + r.stderr[-400:]
        assert r.returncode == 0


class TestTheSetPathIsAsHonestAsTheClearPath:
    """`cswap pin NUM` must not report a pin that is not in effect.

    apply_pin writes the record BEFORE it starts the proxy, so both failures
    here leave a pin that `cswap pin` and the TUI badge report as live while
    nothing serves it.
    """

    def _run(self, tmp_path, impl_src):
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        backup = tmp_path / "backup"
        backup.mkdir()
        code = (
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {src!r})
                from pathlib import Path
                from claude_swap import pin
                """
            )
            + impl_src
            + textwrap.dedent(
                f"""
                pin._impl = _impl_factory
                class _SW:
                    backup_dir = Path({str(backup)!r})
                    def resolve_account(self, a):
                        return (2, "user2@example.com", "org-uuid")
                sys.exit(pin.run(_SW(), "2"))
                """
            )
        )
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )

    def test_a_raising_apply_pin_is_not_a_traceback(self, tmp_path):
        # Real trigger, no injection: <backup>/pin-proxy as a plain FILE makes
        # ensure_proxy's certdir.mkdir raise FileExistsError, which is not a
        # ClaudeSwitchError and so reached the user as a traceback.
        impl = (
            "class _I:\n"
            "    def apply_pin(self, *a): raise FileExistsError('pin-proxy')\n"
            "def _impl_factory(): return _I()\n"
        )
        r = self._run(tmp_path, impl)
        assert "Traceback" not in r.stderr, r.stderr[-400:]
        assert "Pinned" not in r.stdout, "reported a pin that did not happen"
        assert "Could not pin" in r.stdout, r.stdout + r.stderr[-300:]
        assert r.returncode == 1

    def test_no_proxy_serving_is_not_unqualified_success(self, tmp_path):
        # apply_pin returning False means no proxy is serving. Suppressing the
        # follow-up note was the only signal; the word "Pinned" still went out.
        impl = (
            "class _I:\n"
            "    def apply_pin(self, *a): return False\n"
            "def _impl_factory(): return _I()\n"
        )
        r = self._run(tmp_path, impl)
        assert "nothing is pinned yet" in r.stdout, r.stdout + r.stderr[-300:]
        assert r.returncode == 1, "a pin nothing serves must not exit 0"


class TestTheRuntimeVersionFloor:
    """pyproject's floor binds a fresh resolve only.

    Anyone who installed cswap-pin before 0.1.1 and later upgrades claude-swap
    WITHOUT the extra keeps 0.1.0 — the release where a refused swap is handed
    to the client and ends that process's Remote Control.

    WINDOWS REFUSES BEFORE ANY VERSION IS READ. `_impl` raises on win32 first
    (POSIX locks and FIFOs), so on Windows every assertion below about the
    version is unreachable — and the failure that taught this was the quiet
    kind: `test_an_old_pin_is_refused_by_version` PASSED its `"REFUSED" in out`
    line there, for the platform, not for the version. A version check that did
    nothing at all would have passed it too.

    So Windows is not skipped, it is asserted separately: the platform refusal
    is itself a property worth holding, and a bare skip would leave the OS
    where the seam is most likely to drift with no coverage of it.
    """

    WIN = sys.platform == "win32"

    def _probe(self, version_literal):
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        code = textwrap.dedent(
            f"""
            import sys, types
            sys.path.insert(0, {src!r})
            pkg = types.ModuleType("cswap_pin")
            pkg.__path__ = []
            {version_literal}
            proxy = types.ModuleType("cswap_pin.proxy")
            sys.modules["cswap_pin"] = pkg
            sys.modules["cswap_pin.proxy"] = proxy
            import importlib.util
            real = importlib.util.find_spec
            importlib.util.find_spec = lambda n, *a, **k: (
                object() if n.startswith("cswap_pin") else real(n, *a, **k))
            import importlib
            importlib.import_module = lambda n, *a, **k: sys.modules[n]
            from claude_swap import pin
            try:
                pin._impl()
                print("ACCEPTED")
            except Exception as e:
                print("REFUSED:", e)
            """
        )
        return subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        ).stdout

    def test_an_old_pin_is_refused_by_version(self):
        out = self._probe('pkg.__version__ = "0.1.0"')
        if self.WIN:
            assert "not available on Windows" in out, out
            return
        # Never a bare `"REFUSED" in out`: that is what passed on Windows for
        # the wrong reason. The MESSAGE is the assertion.
        assert "0.1.0 is too old" in out, f"0.1.0 was accepted at runtime: {out}"

    def test_the_current_version_is_accepted(self):
        out = self._probe('pkg.__version__ = "0.1.1"')
        assert ("not available on Windows" if self.WIN else "ACCEPTED") in out, out

    def test_an_absent_version_is_not_treated_as_old(self):
        # A dev checkout must not be refused over a guess.
        out = self._probe("pass")
        assert ("not available on Windows" if self.WIN else "ACCEPTED") in out, out
        # And never for the version, on either platform: "too old" here would
        # mean an absent version was read as 0.
        assert "too old" not in out, out


class TestARound2Regressions:
    """The seven from review round 2. Each drives the seam, not a stub.

    The shape they share: an action reported as done while the state it claims
    to have changed is unchanged. A return value cannot carry that — "nothing
    to do" and "could not do it" collapse into the same False — so each of
    these re-reads the thing it just claimed.
    """

    def _cli(self, tmp_path, impl_src, argv_account=None, clear=False, wired=True):
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        cfg = tmp_path / ".claude.json"
        cfg.write_text(
            json.dumps(
                {
                    "env": {"HTTPS_PROXY": "http://127.0.0.1:36301"},
                    "_cswapPinWiredKeys": ["HTTPS_PROXY"],
                }
                if wired
                else {"env": {}}
            )
        )
        backup = tmp_path / "backup"
        backup.mkdir()
        (backup / "settings.json").write_text(
            json.dumps({"remoteControl": {"pinnedEmail": "cloud@example.com"}}, indent=2)
        )
        code = (
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {src!r})
                from pathlib import Path
                import claude_swap.paths as paths
                cfg = Path({str(cfg)!r})
                paths.get_global_config_path = lambda: cfg
                paths.get_default_global_config_path = lambda: cfg
                from claude_swap import pin
                """
            )
            + impl_src
            + textwrap.dedent(
                f"""
                pin._impl = _impl_factory
                class _SW:
                    backup_dir = Path({str(backup)!r})
                    def resolve_account(self, a):
                        return (2, "user2@example.com", "org-uuid")
                    def _account_kind(self, n):
                        return "oauth"
                sys.exit(pin.run(_SW(), {argv_account!r}, clear={clear!r}))
                """
            )
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        return r, cfg, backup

    def test_a_clear_that_cannot_remove_the_wiring_is_a_failure(self, tmp_path):
        # apply_pin succeeds (the pin goes), but the wiring cannot be removed.
        # Reported as success, the daemon idles out and every hand-launched
        # claude dials a dead port — the stranding clear_wiring exists to stop.
        impl = (
            "import json as _j\n"
            "class _I:\n"
            "    def apply_pin(self, sw, *a):\n"
            "        (sw.backup_dir / 'settings.json').write_text(_j.dumps({}))\n"
            "def _impl_factory(): return _I()\n"
        )
        # Make clear_wiring a no-op so the wiring survives, as a held lock does.
        impl += "pin.clear_wiring = lambda *a, **k: False\n"
        r, cfg, _ = self._cli(tmp_path, impl, clear=True)
        assert "Unpinned" not in r.stdout, r.stdout
        assert "Could not remove" in r.stdout, r.stdout + r.stderr[-300:]
        assert r.returncode == 1
        assert "_cswapPinWiredKeys" in cfg.read_text(), "fixture no longer valid"

    def test_a_failed_set_rolls_the_record_back(self, tmp_path):
        # apply_pin writes the record before starting the proxy, so reporting
        # the failure is not enough: `cswap pin` reads it back and calls it
        # live, and the TUI badge agrees.
        impl = (
            "import json as _j\n"
            "class _I:\n"
            "    calls = []\n"
            "    def apply_pin(self, sw, email, org):\n"
            "        _I.calls.append(email)\n"
            "        if email is not None and len(_I.calls) == 1:\n"
            "            (sw.backup_dir / 'settings.json').write_text(\n"
            "                _j.dumps({'remoteControl': {'pinnedEmail': email}}))\n"
            "            raise FileExistsError('pin-proxy')\n"
            "        (sw.backup_dir / 'settings.json').write_text(_j.dumps({}))\n"
            "def _impl_factory(): return _I()\n"
        )
        r, _, backup = self._cli(tmp_path, impl, argv_account="2")
        assert "Could not pin" in r.stdout, r.stdout + r.stderr[-300:]
        assert r.returncode == 1
        raw = json.loads((backup / "settings.json").read_text())
        assert not raw.get("remoteControl", {}).get("pinnedEmail"), (
            "the failed pin stayed in the record; `cswap pin` would call it live"
        )

    def test_an_api_key_account_is_refused(self, tmp_path):
        impl = (
            "class _I:\n"
            "    def apply_pin(self, *a): raise AssertionError('must not be reached')\n"
            "def _impl_factory(): return _I()\n"
        )
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        backup = tmp_path / "b"
        backup.mkdir()
        (backup / "settings.json").write_text("{}")
        code = (
            textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {src!r})
                from pathlib import Path
                from claude_swap import pin
                from claude_swap.exceptions import ClaudeSwitchError
                """
            )
            + impl
            + textwrap.dedent(
                f"""
                pin._impl = _impl_factory
                class _SW:
                    backup_dir = Path({str(backup)!r})
                    def resolve_account(self, a):
                        return (3, "key@example.com", "org")
                    def _account_kind(self, n):
                        return "api_key"
                try:
                    pin.run(_SW(), "3")
                    print("ACCEPTED")
                except ClaudeSwitchError as e:
                    print("REFUSED:", e)
                """
            )
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        ).stdout
        assert "REFUSED" in out, f"an API-key account was pinned: {out}"
        assert "API-key account" in out, out

    def test_one_place_decides_the_install_command(self):
        """A second hardcoded hint diverged from the derived one on pipx."""
        from pathlib import Path

        import ast

        src = Path(
            str(Path(__file__).resolve().parent.parent / "src")
        ).joinpath("claude_swap/pin.py")
        tree = ast.parse(src.read_text(encoding="utf-8"))

        # STRING CONSTANTS ONLY. The prose in docstrings names these commands
        # while explaining why there is one decider, so counting raw text
        # would forbid documenting the rule. What must not repeat is a literal
        # the code can PRINT.
        allowed = set()
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_install_how":
                allowed = {id(n) for n in ast.walk(node)}
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in allowed or ast.get_docstring(tree) == node.value:
                continue
            if any(
                f in node.value
                for f in ("uv tool install", "pipx install", "pip install")
            ):
                # A docstring is prose, not something the code emits.
                offenders.append(node.lineno)
        # Drop the lines that ARE docstrings.
        docs = {
            n.body[0].lineno
            for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module))
            and n.body
            and isinstance(n.body[0], ast.Expr)
            and isinstance(n.body[0].value, ast.Constant)
            and isinstance(n.body[0].value.value, str)
        }
        offenders = [ln for ln in offenders if ln not in docs]
        assert not offenders, (
            f"install command literal outside _install_how() at line(s) {offenders} "
            "— two places decide it, and they diverged on pipx once already"
        )


class TestTheVerdictIsSharedNotDuplicated:
    """clear_pin/set_pin are the one place the outcome is decided.

    Three review rounds found the same shape: a fix landed on the CLI and the
    TUI's sibling call site kept the old behaviour. These assert the shared
    functions themselves, so a future divergence needs someone to write a
    second copy rather than to forget a line.
    """

    def _sw(self, tmp_path, pinned="cloud@example.com", wired=True):
        import types

        backup = tmp_path / "b"
        backup.mkdir()
        (backup / "settings.json").write_text(
            json.dumps({"remoteControl": {"pinnedEmail": pinned}} if pinned else {})
        )
        cfg = tmp_path / ".claude.json"
        cfg.write_text(
            json.dumps(
                {"env": {"HTTPS_PROXY": "x"}, "_cswapPinWiredKeys": ["HTTPS_PROXY"]}
                if wired
                else {"env": {}}
            )
        )
        return types.SimpleNamespace(backup_dir=backup), cfg

    def test_clear_pin_fails_when_the_wiring_survives(self, tmp_path, monkeypatch):
        import claude_swap.paths as paths

        from claude_swap import pin

        sw, cfg = self._sw(tmp_path)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        monkeypatch.setattr(paths, "get_default_global_config_path", lambda: cfg)

        class _I:
            def apply_pin(self, s, *a):
                (s.backup_dir / "settings.json").write_text("{}")

        monkeypatch.setattr(pin, "_impl", lambda: _I())
        # The lock is contended, so clear_wiring skips the path and returns
        # False — indistinguishable from "nothing to remove" by its return.
        monkeypatch.setattr(pin, "clear_wiring", lambda *a, **k: False)
        ok, msg = pin.clear_pin(sw)
        assert not ok, msg
        assert "wiring" in msg, msg

    def test_clear_pin_succeeds_when_both_are_gone(self, tmp_path, monkeypatch):
        import claude_swap.paths as paths

        from claude_swap import pin

        sw, cfg = self._sw(tmp_path, wired=False)
        monkeypatch.setattr(paths, "get_global_config_path", lambda: cfg)
        monkeypatch.setattr(paths, "get_default_global_config_path", lambda: cfg)

        class _I:
            def apply_pin(self, s, *a):
                (s.backup_dir / "settings.json").write_text("{}")

        monkeypatch.setattr(pin, "_impl", lambda: _I())
        ok, msg = pin.clear_pin(sw)
        assert ok and "Unpinned" in msg, msg

    def test_set_pin_rolls_back_on_failure(self, tmp_path, monkeypatch):
        from claude_swap import pin

        sw, _ = self._sw(tmp_path, pinned=None)

        class _I:
            n = 0

            def apply_pin(self, s, email, org):
                _I.n += 1
                if _I.n == 1:
                    (s.backup_dir / "settings.json").write_text(
                        json.dumps({"remoteControl": {"pinnedEmail": email}})
                    )
                    raise FileExistsError("pin-proxy")
                (s.backup_dir / "settings.json").write_text("{}")

        monkeypatch.setattr(pin, "_impl", lambda: _I())
        ok, msg = pin.set_pin(sw, "user2@example.com", "org")
        assert not ok, msg
        assert pin._pinned_email_now(sw) is None, (
            "the failed pin stayed recorded; every read-back would call it live"
        )


class TestTheUncoveredRound2Fixes:
    """Two round-2 fixes survived reversion green. A guard nothing asserts is
    a guard someone deletes."""

    def test_the_cli_renders_a_broken_package_instead_of_a_traceback(self, tmp_path):
        import subprocess
        import textwrap
        from pathlib import Path

        src = str(Path(__file__).resolve().parent.parent / "src")
        code = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {src!r})
            from claude_swap import pin
            # A broken package ROOT: _impl re-raises the underlying ImportError
            # on purpose, and nothing between there and the shell rendered it.
            pin.run = lambda *a, **k: (_ for _ in ()).throw(
                ImportError("No module named 'cryptography'", name="cryptography"))
            from claude_swap import cli
            sys.argv = ["cswap", "pin", "2"]
            try:
                cli._pin_command(["2"])
            except SystemExit as e:
                print("EXIT", e.code)
            """
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        combined = r.stdout + r.stderr
        assert "Traceback" not in combined, combined[-500:]
        assert "not usable" in combined, combined[-500:]
        assert "EXIT 1" in combined, combined[-300:]

    def test_the_launch_unwire_is_bounded_by_the_budget(self, monkeypatch):
        """The unwire took the package's own 5s lock, unbounded by the budget
        the no-package branch gets. Assert the PROBE gates it."""
        import types

        from claude_swap import pin

        calls = []
        impl = types.SimpleNamespace(
            ensure_proxy=lambda sw: None,
            unwire_if_dead=lambda p: calls.append("unwired"),
        )
        monkeypatch.setattr(pin, "_impl", lambda: impl)
        monkeypatch.setattr(pin, "_config_lock_is_free", lambda b: False)
        sw = types.SimpleNamespace(backup_dir=__import__("pathlib").Path("/tmp"))
        assert pin.wire_launch_env(sw, {"A": "1"}) == {"A": "1"}
        assert calls == [], "the unwire ran while the lock was held"

        monkeypatch.setattr(pin, "_config_lock_is_free", lambda b: True)
        pin.wire_launch_env(sw, {"A": "1"})
        assert calls == ["unwired"], "the unwire never runs, even when free"
