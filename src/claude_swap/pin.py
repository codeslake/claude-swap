"""Cloud pin: keep Remote Control and Artifacts on one account.

cswap swaps the on-disk credential, so *everything* follows the swap —
including two things that are not inference and that you usually want to stay
put:

- **Remote Control** — a session's owner is fixed at creation by whichever
  bearer created it. Swap accounts and the phone/web loses the session.
- **Artifacts** — owned by the publishing bearer. After a swap a republish
  403s and the artifact "disappears" from the account you are logged into.

Claude Code resolves all of these through one credential accessor and has no
per-operation token selector, so splitting auth per operation inside a single
session means intercepting the requests. That interception lives in a separate
package (``cswap-pin``, installed via the ``pin`` extra) rather than here.

Built on ``cswap-pin`` (an optional extra). This module is import-safe without
it by design — the helpers below are pure and unit-testable in CI, and the
dependency is imported lazily inside the entry points, exactly as
:mod:`claude_swap.menubar` does with ``rumps``.
"""

from __future__ import annotations

import json
from types import ModuleType

from claude_swap.exceptions import ClaudeSwitchError, ConfigError

_INSTALL_HINT = (
    "The cloud pin requires 'cswap-pin'. "
    "Install with: pip install 'claude-swap[pin]'"
)


def _impl() -> ModuleType:
    """The pin implementation, or a clean error naming the fix.

    Raises the type the CLI already renders rather than letting an
    ``ImportError`` traceback out of a command the user typed.

    "Not installed" and "installed but broken" are separated by ``find_spec``
    instead of by catching ``ImportError``. Catching cannot tell them apart,
    and conflating them tells the user to install a package they already have
    when the real cause is, say, a missing ``cryptography``. A failure raised
    from inside a module that IS present propagates unchanged.
    """
    import importlib.util
    import sys

    # POSIX only, the same way the menu bar is macOS only. The proxy holds its
    # daemon lock with fcntl.flock and refcounts sessions through a FIFO
    # (os.mkfifo); neither exists on Windows, so an install there would fail at
    # first use with a ModuleNotFoundError from inside the dependency rather
    # than a sentence the user can act on. cswap itself advertises Windows
    # support (pyproject classifiers), so this has to be said, not assumed.
    if sys.platform == "win32":
        raise ClaudeSwitchError(
            "The cloud pin is not available on Windows: it needs POSIX file "
            "locks and FIFOs."
        )

    try:
        found = importlib.util.find_spec("cswap_pin.proxy") is not None
    except ImportError as exc:
        # find_spec has to IMPORT the parent package to read its __path__, so a
        # cswap_pin/__init__.py that raises surfaces here rather than below —
        # and swallowing it is what turns "your cryptography is broken" into
        # "install the package you already have". Measured: a package root
        # raising ImportError("No module named 'cryptography'") propagates out
        # of find_spec, not out of import_module.
        #
        # e.name is what tells them apart: absent -> 'cswap_pin', broken root
        # -> whatever the package failed to import.
        if exc.name and not exc.name.startswith("cswap_pin"):
            raise
        found = False
    except ValueError:
        found = False
    if not found:
        raise ClaudeSwitchError(_INSTALL_HINT)
    return importlib.import_module("cswap_pin.proxy")


def _live_impl() -> ModuleType | None:
    """The implementation if it is usable RIGHT NOW, else None. Never raises.

    Both display helpers below need the same thing: resolve the package, and
    treat every failure as "no pin" rather than an error. Callers that ACT on
    the pin use :func:`_impl` instead and report what it raises — hiding a
    broken install is right for a badge and wrong for a command.

    ``invalidate_caches`` because a long-lived process caches each sys.path
    directory by mtime, so a package installed after start can stay invisible.
    Measured: usually visible immediately, but an install landing inside the
    same mtime tick is not — which is exactly the "I installed it and the menu
    is still missing" report.
    """
    import importlib

    importlib.invalidate_caches()
    try:
        return _impl()
    except Exception:  # noqa: BLE001
        return None


def is_available() -> bool:
    """Whether a pin surface should be shown at all."""
    return _live_impl() is not None


def pinned_email(switcher) -> str | None:
    """The pinned account's email, or None.

    The TUI's one question about the pin is "which account is it on", and
    None is the honest answer in every failure: no extra, no pin, a malformed
    pin file. With no extra there IS no pin, so a notice would be a permanent
    banner on machines that deliberately run without it, and an always-on
    warning is one people stop reading.
    """
    impl = _live_impl()
    if impl is None:
        return None
    try:
        pin = impl.load_pin(switcher.backup_dir)
    except Exception:  # noqa: BLE001 — a badge must not take the view down
        return None
    return pin[0] if pin else None


# -- launch integration ------------------------------------------------------


def wire_launch_env(switcher, env: dict[str, str]) -> dict[str, str]:
    """Route a child Claude Code through the pin proxy, if one is pinned.

    Returns ``env`` unchanged when there is no pin, when the extra is not
    installed, or when the proxy cannot be started: an optional feature must
    never be able to block a launch.
    """
    # ONE guard around everything, including _impl(). A split try left the
    # resolution step uncovered, so anything raised there — a broken
    # cryptography, a corrupt install — propagated out of a launch. Measured:
    # the launch died instead of starting unpinned.
    try:
        pin = _impl()
    except Exception:  # noqa: BLE001 — never block the launch
        # No pin this launch, whatever the reason: not installed, or installed
        # and broken. A wiring a previous install left behind would otherwise
        # outlive it and point every session at a dead port — see clear_wiring.
        try:
            clear_wiring(switcher, timeout=_LAUNCH_LOCK_BUDGET_S)
        except Exception:  # noqa: BLE001
            pass
        return env
    try:
        pinned = pin.ensure_proxy(switcher)
        if pinned:
            port, ca_path = pinned
            return pin.wire_env(env, port, ca_path)
    except Exception:  # noqa: BLE001 — never block the launch
        pass
    # No proxy this launch, whether ensure_proxy said so or died saying it.
    # .claude.json's env block is applied at boot, so a wiring a previous
    # launch left behind would send this child at a port nothing answers.
    # ONE tail, not one per branch: duplicating it ran the unwire twice when
    # the None path's own unwire raised.
    try:
        pin.unwire_if_dead(switcher.backup_dir / "pin-proxy")
    except Exception:  # noqa: BLE001
        pass
    return env


# -- wiring removal ----------------------------------------------------------
#
# This half deliberately does NOT live in the optional package.

_WIRE_MARK = "_cswapPinWiredKeys"

# The launch path calls this on every `cswap run`, so its lock wait has to be
# bounded by something far below the 9s default: a user who never installed
# the pin must not wait on Claude Code's config lock at all, and one who did
# must not wait long. Nothing is lost by giving up — an unremoved wiring is
# retried on the next launch, and the caller fails open either way.
_LAUNCH_LOCK_BUDGET_S = 0.5


def clear_wiring(switcher, timeout: float | None = None) -> bool:
    """Remove a pin wiring from the global config. True when it removed one.

    The pin writes its proxy address into ``.claude.json``'s env block and
    records which keys it wrote in ``_cswapPinWiredKeys``; this reads that
    marker and puts the file back. It touches no proxy, no daemon and no
    credential — only a record cswap left.

    It has to be here rather than in the optional package because the failure
    it prevents is caused by that package being GONE. Claude Code applies the
    env block at boot, so a wiring naming a port nothing listens on makes
    every hand-launched ``claude`` dial a dead proxy and retry forever. If the
    only code able to remove it shipped in the pin package, uninstalling the
    pin — the very thing an optional extra invites — would strand the wiring
    permanently, with hand-editing ``.claude.json`` the sole cure.

    Only keys the pin recorded are touched, and anything it displaced is
    restored, so a proxy the user or their launcher set beforehand comes back
    rather than being lost with ours.
    """
    from claude_swap.claude_locks import proper_lockfile
    from claude_swap.paths import (
        get_default_global_config_path,
        get_global_config_path,
    )

    # BOTH configs, because the writing side resolves the same way this does:
    # `CLAUDE_CONFIG_DIR` is set in the *child's* env dict, not the process's,
    # so a `cswap run` from a normal terminal wires ~/.claude.json while one
    # from inside a session terminal wires that session's copy. Clearing only
    # the resolved path leaves the other wired, and `cswap pin --clear` then
    # prints "No cloud account pinned" over a config that still names a dead
    # port — the exact stranding this function exists to prevent. Measured:
    # the two paths diverge as soon as CLAUDE_CONFIG_DIR is set.
    paths = [get_global_config_path()]
    default = get_default_global_config_path()
    if default != paths[0]:
        paths.append(default)

    # ONE LOCK PER PATH. The shared config lock derives its directory from
    # get_global_config_path(), so a single lock around the loop guards one
    # file and leaves the other rewritten unprotected — racing `cswap switch`
    # and Claude Code, the whole-file clobber the lock exists to prevent.
    changed = False
    for path in paths:
        try:
            with proper_lockfile(
                path.parent / (path.name + ".lock"), timeout=timeout
            ):
                if _clear_wiring_locked(switcher, path):
                    changed = True
        except Exception:  # noqa: BLE001
            # A lock we cannot take is a reason to skip THIS file, not to
            # abandon the other one — and on the launch path (sub-second
            # budget) a contended config must not fail the clear outright.
            continue
    return changed


def _clear_wiring_locked(switcher, path) -> bool:
    """The read-modify-write of :func:`clear_wiring`, under its lock."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if not isinstance(raw, dict):
        return False

    ours = raw.get(_WIRE_MARK)
    if not isinstance(ours, list) or not ours:
        return False  # nothing of ours in there

    env = raw.get("env")
    env = dict(env) if isinstance(env, dict) else {}
    saved = raw.get(f"{_WIRE_MARK}Saved")
    saved = dict(saved) if isinstance(saved, dict) else {}
    for key in ours:
        env.pop(key, None)
    env.update(saved)

    raw.pop(_WIRE_MARK, None)
    raw.pop(f"{_WIRE_MARK}Saved", None)
    if env:
        raw["env"] = env
    else:
        raw.pop("env", None)

    try:
        # The switcher's own writer, not a second one: it already validates the
        # JSON it produced and chmods the TEMP file so the rename is the atomic
        # commit. This file can hold ``primaryApiKey`` and inline MCP
        # credentials, so a hand-rolled write here would be a second place for
        # that 0600 to drift out of agreement with switcher.py.
        switcher._write_json(path, raw)
    except (OSError, ConfigError):
        return False
    return True


def clear_pin(switcher) -> bool:
    """Unpin the cloud account AND remove the launch wiring. True when there
    was something to clear.

    The one place either surface unpins. The CLI and the TUI each spelled the
    sequence out, and the TUI's copy was missing the :func:`clear_wiring`
    half — it cleared the pin and left ``~/.claude.json`` naming a dead port,
    so every hand-launched ``claude`` dialled it and retried forever, while
    the TUI said "Cloud pin cleared". A second call site that can drift IS
    the defect, so there is no longer a second one.

    Works WITHOUT the package on purpose: unpinning is what a user reaches
    for precisely when they have uninstalled the pin, and the wiring is
    cswap's own record (see :func:`clear_wiring`). Any failure falls back,
    not just a missing package — "installed but unusable" (a broken
    cryptography) is the other way a user ends up here, and a traceback is
    the worst possible outcome for the one action whose job is to work when
    the pin does not.
    """
    cleared_pin = False
    try:
        impl = _impl()
        had_pin = impl.load_pin(switcher.backup_dir) is not None
        impl.apply_pin(switcher, None, None)
        cleared_pin = had_pin  # credit it only once apply_pin has returned
    except Exception:  # noqa: BLE001
        pass
    # ALWAYS, not only when apply_pin failed. The package unwires through its
    # own single-path resolver, so with the extra installed a clear from
    # inside a session terminal cleared that session's config and left
    # ~/.claude.json naming a dead port — while reporting success. The
    # both-paths guarantee has to hold for the users who HAVE the pin, which
    # is all of them at the moment they unpin.
    #
    # apply_pin cannot answer "was there anything to clear": it returns
    # whether a proxy is now serving, which on this path is always False.
    return clear_wiring(switcher) or cleared_pin


# -- command -----------------------------------------------------------------


def run(switcher, account: str | None, clear: bool = False) -> int:
    """Entry point for ``cswap pin``. Mirrors :func:`claude_swap.menubar.run`:
    the optional dependency is resolved here, at call time, not at import."""
    from claude_swap.printer import accent, dimmed

    if clear:
        if not clear_pin(switcher):
            print(dimmed("No cloud account pinned"))
            return 0
        print(f"{accent('Unpinned')} the cloud account")
        return 0

    pin = _impl()  # raises ClaudeSwitchError with the install hint

    if account is None:
        current = pin.load_pin(switcher.backup_dir)
        if current:
            print(f"Cloud account (RC/artifacts): {current[0]}")
        else:
            print(dimmed("No cloud account pinned"))
        return 0

    account_num, email, org_uuid = switcher.resolve_account(account)
    started = pin.apply_pin(switcher, email, org_uuid)
    print(
        f"{accent('Pinned')} the cloud account (RC/artifacts) to "
        f"Account-{account_num} ({email})"
    )
    if not started:
        return 0

    # A re-pin takes effect under the live proxy: the pinned account is re-read
    # per request, so nothing has to restart. The one thing it cannot move is a
    # Remote Control session that is ALREADY open — the server fixed its owner
    # at creation, so reconnecting inside it is what mints a new one under the
    # new pin. Name those sessions instead of telling everyone to restart.
    open_rc = pin.live_remote_control_sessions()
    if open_rc:
        which = ", ".join(open_rc[:3])
        if len(open_rc) > 3:
            which += f", +{len(open_rc) - 3} more"
        print(
            dimmed(
                f"Remote Control is open on: {which}. Those stay on the "
                "previous account until you reconnect them "
                "(/rc -> Disconnect this session -> /rc)."
            )
        )
    else:
        print(dimmed("New sessions pick this up."))
    return 0
