"""cswap must work with the pin package absent.

The cloud pin ships as a separate distribution (realiti4/claude-swap#198): the
maintainer does not want a MITM proxy inside claude-swap itself, so it becomes
`claude-swap[pin]` depending on a companion package. These tests are the
boundary — they fail the moment cswap starts REQUIRING the optional half.
"""

import builtins
import importlib
import sys

import pytest


@pytest.fixture
def no_pin_package(monkeypatch):
    """Make `import claude_swap.pin_proxy` fail, as if it were not installed."""
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "claude_swap.pin_proxy" or (
            name == "claude_swap" and fromlist and "pin_proxy" in fromlist
        ):
            raise ImportError("No module named 'claude_swap.pin_proxy'")
        return real_import(name, globals, locals, fromlist, level)

    # Evict the real modules so an import is actually attempted, and PUT THEM
    # BACK afterwards. monkeypatch.delitem does not restore a deletion, so the
    # eviction outlived the test: every later test that imported claude_swap
    # got a fresh module object while its own patches pointed at the old one.
    # Measured — 8 failures in test_pin_proxy.py, none real, all from this
    # fixture. A test that breaks its neighbours is worse than no test.
    saved = {
        m: sys.modules[m]
        for m in list(sys.modules)
        if "pin_proxy" in m or "pin_optional" in m
    }
    for m in saved:
        del sys.modules[m]
    monkeypatch.setattr(builtins, "__import__", fake_import)
    # The absent-flag is sticky by design (see load()); clear it so a test that
    # runs after a successful import still exercises the missing-package path.
    import claude_swap.pin_optional as _po
    monkeypatch.setattr(_po, "_absent", False, raising=False)
    try:
        yield
    finally:
        for m in [m for m in list(sys.modules) if "pin_proxy" in m or "pin_optional" in m]:
            del sys.modules[m]
        sys.modules.update(saved)


class TestCswapRunsWithoutThePin:
    def test_pin_optional_reports_absent_instead_of_raising(self, no_pin_package):
        # import_module, not reload: the fixture evicts claude_swap.pin_optional
        # from sys.modules, and reload() REQUIRES the module to still be there
        # ("not in sys.modules"). Whether it was cached depends on what ran
        # before this file, so reload passed alone and failed inside the full
        # suite — a test that only works in one ordering.
        pin_optional = importlib.import_module("claude_swap.pin_optional")
        assert pin_optional.load() is None
        assert pin_optional.available() is False

    @pytest.mark.parametrize("module", ["claude_swap.session", "claude_swap.cli"])
    def test_the_entry_points_still_import(self, no_pin_package, module):
        """The one that actually broke cswap: session.py imported pin_proxy at
        module scope, so without the package `import claude_swap.session`
        raises and NOTHING works — not switching, not the TUI — over an
        optional feature.

        Runs in a SUBPROCESS. Clearing claude_swap out of sys.modules in-process
        and re-importing it left every later test looking at freshly-imported
        module objects while their own monkeypatches pointed at the old ones —
        measured: 132 failures across test_update_check.py and others, none of
        them real, all caused by this test. A subprocess cannot poison a cache
        it does not share.
        """
        import subprocess
        import textwrap
        from pathlib import Path

        # Point the child at THIS checkout. cswap is installed editable from
        # the integration checkout, so a bare subprocess imports the DEPLOYED
        # tree and silently tests someone else's code — it reported
        # session.py:51 from cswap_fork/src while this worktree had already
        # fixed it.
        src = str(Path(__file__).resolve().parent.parent / "src")

        code = textwrap.dedent(
            f"""
            import builtins, sys
            sys.path.insert(0, {src!r})
            real = builtins.__import__
            def fake(name, g=None, l=None, fromlist=(), level=0):
                if name == "claude_swap.pin_proxy" or (
                    name == "claude_swap" and fromlist and "pin_proxy" in fromlist
                ):
                    raise ImportError("No module named 'claude_swap.pin_proxy'")
                return real(name, g, l, fromlist, level)
            builtins.__import__ = fake
            import {module}
            """
        )
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert r.returncode == 0, (
            f"{module} does not import without the pin package:\n{r.stderr[-800:]}"
        )


class TestTheBoundaryIsNotBypassed:
    """A single top-level `import pin_proxy` anywhere re-creates the hard
    dependency, and it would pass every behavioural test above until the day
    the package is actually split out. Assert the seam itself."""

    def _sources(self):
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src" / "claude_swap"
        for path in root.rglob("*.py"):
            if path.name in ("pin_proxy.py", "pin_optional.py"):
                continue
            yield path

    def test_no_module_scope_import_of_pin_proxy(self):
        import ast

        offenders = []
        for path in self._sources():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module scope ONLY — nested is the fix
                names = []
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [f"{node.module}.{a.name}" for a in node.names or ()]
                if any("pin_proxy" in n for n in names):
                    offenders.append(f"{path.name}:{node.lineno}")
        assert not offenders, (
            "module-scope import of pin_proxy makes the optional package "
            f"mandatory: {offenders}"
        )
