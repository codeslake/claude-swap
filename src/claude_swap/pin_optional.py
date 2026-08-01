"""The one place cswap reaches for the cloud pin, and the only one allowed to.

The pin (Remote Control / Artifacts held on one account while inference
follows the swap) is implemented as a local MITM proxy. Upstream's maintainer
does not want that shipped inside claude-swap itself and asked for a separate
package instead — ``claude-swap[pin]`` depending on a companion distribution
(realiti4/claude-swap#198). This module is the seam that makes that split a
packaging change rather than a refactor.

Two rules keep the boundary real:

1. **Nothing outside this module imports ``pin_proxy`` at module scope.** A
   top-level import makes the pin a hard dependency: without it installed,
   ``import claude_swap.session`` raises and cswap does not start at all —
   swapping accounts, the TUI, everything — over an optional feature. Import
   inside the function instead, or come through here.
2. **A missing pin is not an error.** ``load()`` returns ``None`` when the
   package is absent, and every caller must already handle "no pin" because
   that is also what an unpinned machine looks like. The absent-package path
   and the not-pinned path are deliberately the same path, so the common case
   keeps the uncommon one working.

Deliberately no shim classes and no re-exported functions: callers take the
module and use it directly, so this file does not have to grow an entry per
call site and cannot drift from the real API.
"""

from __future__ import annotations

from types import ModuleType

_absent = False


def load() -> ModuleType | None:
    """The pin implementation, or ``None`` when it is not installed.

    Only the NEGATIVE answer is cached. A successful import is already cached
    by ``sys.modules``, and holding our own reference to the module object on
    top of that makes this function lie about identity: after anything
    replaces ``claude_swap.pin_proxy`` in ``sys.modules``, callers coming
    through here keep getting the module that was imported first.

    That is not a test-only concern, though tests are where it showed up —
    eight in test_pin_proxy.py went red because they patched the freshly
    imported module while ``session.py`` still held the original. Any code
    that reloads or substitutes the module (a plugin, a reload-on-upgrade
    path) would see the same split. ``sys.modules`` is the one cache allowed
    to answer "which module object is current".

    The negative answer is worth keeping: a package that is not installed
    will not appear mid-process, and this runs on the launch path where the
    failed module search would otherwise repeat per exec.
    """
    global _absent
    if _absent:
        return None
    try:
        from claude_swap import pin_proxy
    except ImportError:
        _absent = True
        return None
    return pin_proxy


def available() -> bool:
    """Whether the pin can be used at all. For callers that only want to know
    whether to OFFER it — a TUI row, a CLI subcommand — rather than use it."""
    return load() is not None
