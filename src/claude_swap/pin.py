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

def _install_hint() -> str:
    """How to install the extra, in a form that reaches THIS install.

    Not a constant, because `pip install` is wrong for the install method most
    users have. Under a uv tool install, pip puts a second copy in whatever pip
    is on PATH and the extra never reaches the tool's environment — the user
    follows the instruction, it succeeds, and the pin is still missing.
    `cswap upgrade` already solves this; reuse its detector rather than
    re-deriving it.
    """
    from claude_swap.update_check import _detect_install_method

    how = {
        "uv": "uv tool install 'claude-swap[pin]'",
        "pipx": "pipx install 'claude-swap[pin]'",
    }.get(_detect_install_method() or "", "pip install 'claude-swap[pin]'")
    return f"The cloud pin requires 'cswap-pin'. Install with: {how}"


# The FLOOR is a correctness bound, not packaging hygiene. 0.1.0 is the release
# that hands a REFUSED swap to the client instead of retrying it unswapped, and
# Claude Code treats 401/403/404 as terminal (SSETransport sets state="closed"
# and never reconnects), so one misrouted request ends that session's Remote
# Control for the life of the process.
#
# pyproject's `pin = ["cswap-pin>=0.1.1"]` binds a FRESH resolve only. Anyone
# who installed cswap-pin before 0.1.1 and later upgrades claude-swap WITHOUT
# the extra keeps 0.1.0 — and nothing here noticed, because the seam only ever
# asked "does it import".
_MIN_PIN_VERSION = (0, 1, 1)


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
        raise ClaudeSwitchError(_install_hint())
    mod = importlib.import_module("cswap_pin.proxy")

    # A RUNTIME floor, because pyproject's only binds a fresh resolve. Refusing
    # is right rather than warning: below the floor a refused swap reaches the
    # client and ends that session's Remote Control permanently, which is worse
    # than not pinning at all. An unparseable or absent version is NOT treated
    # as too old — that would break a dev checkout over a guess.
    #
    # Read off the PACKAGE, not `proxy`: measured against the installed 0.1.1,
    # cswap_pin.__version__ is "0.1.1" and cswap_pin.proxy.__version__ does not
    # exist. Reading the wrong one makes this check dead code that passes for
    # every version, which is the failure it exists to prevent.
    raw = getattr(sys.modules.get("cswap_pin"), "__version__", "")
    try:
        got = tuple(int(p) for p in str(raw).split(".")[:3])
    except ValueError:
        got = ()
    if got and got < _MIN_PIN_VERSION:
        want = ".".join(str(p) for p in _MIN_PIN_VERSION)
        raise ClaudeSwitchError(
            f"cswap-pin {raw} is too old (need >= {want}): it hands a refused "
            f"swap to the client, which ends that session's Remote Control. "
            f"{_install_hint()}"
        )
    return mod


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


# -- command -----------------------------------------------------------------


def run(switcher, account: str | None, clear: bool = False) -> int:
    """Entry point for ``cswap pin``. Mirrors :func:`claude_swap.menubar.run`:
    the optional dependency is resolved here, at call time, not at import."""
    from claude_swap.printer import accent, dimmed, warning

    if clear:
        # Works WITHOUT the package on purpose: ``--clear`` is what a user
        # reaches for precisely when they have uninstalled the pin, and the
        # wiring is cswap's own record (see clear_wiring).
        #
        # Any failure falls back, not just a missing package: "installed but
        # unusable" (a broken cryptography) is the other way a user ends up
        # here, and a traceback is the worst possible outcome for the one
        # command whose job is to work when the pin does not.
        # READ THE RECORD OURSELVES, not through the package.
        #
        # An earlier fix re-read the pin AFTER apply_pin, which covered
        # "apply_pin raises" and missed the case its own comment names.
        # A broken `cryptography` does not make apply_pin raise: cswap_pin's
        # __init__ imports nothing from proxy, so find_spec succeeds and it is
        # `import_module("cswap_pin.proxy")` inside _impl() that raises —
        # one line EARLIER than that guard. `had_pin` was then never measured,
        # the except set it False, and "Unpinned" printed over a live pin.
        #
        # The pin record is cswap's OWN file (settings.json -> remoteControl),
        # so the one command whose job is to work when the pin does not should
        # not need the pin to answer "is it still there".
        def _pinned_now() -> bool:
            from claude_swap import settings as _s

            section = _s._read_raw(_s.settings_path(switcher.backup_dir)).get(
                "remoteControl"
            )
            return isinstance(section, dict) and bool(section.get("pinnedEmail"))

        try:
            had_pin = _pinned_now()
        except Exception:  # noqa: BLE001 — an unreadable file is not a pin
            had_pin = False
        still_pinned = had_pin
        try:
            impl = _impl()
            impl.apply_pin(switcher, None, None)
        except Exception:  # noqa: BLE001
            pass
        # Re-read either way. The success message was gated on clear_wiring(),
        # which reports on the .claude.json wiring and never on the pin itself,
        # so every way of failing above printed "Unpinned" over a live pin.
        # Reading here covers all of them at once — a raising _impl(), a
        # raising apply_pin, and an apply_pin that returns having done nothing.
        try:
            still_pinned = _pinned_now()
        except Exception:  # noqa: BLE001
            still_pinned = had_pin
        # ALWAYS, not only when apply_pin failed. The package unwires through
        # its own single-path resolver, so with the extra installed a --clear
        # run from inside a session terminal cleared that session's config and
        # left ~/.claude.json naming a dead port — while printing "Unpinned".
        # The both-paths guarantee has to hold for the users who HAVE the pin,
        # which is all of them at the moment they unpin.
        #
        # apply_pin cannot answer "was there anything to clear": it returns
        # whether a proxy is now serving, which on this path is always False.
        cleared_wiring = clear_wiring(switcher)
        if still_pinned:
            # The wiring may well be gone; the PIN is not. Saying "Unpinned"
            # here is the failure, not the pin surviving: a user who is told
            # they are unpinned stops looking.
            warning("Could not remove the cloud pin")
            print(dimmed("  the pin package is installed but not usable here"))
            print(dimmed("  reinstall it:  uv tool install 'claude-swap[pin]'"))
            return 1
        if not cleared_wiring and not had_pin:
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
    # The SET path needs the same honesty the clear path was given, and for the
    # same reason: apply_pin writes the record BEFORE it starts the proxy, so
    # a failure here leaves a pin that `cswap pin` and the TUI badge both
    # report as live while nothing serves it.
    #
    # _pin_command catches ClaudeSwitchError only, and the failures that reach
    # here are not that: with <backup>/pin-proxy a plain file (a leftover, a
    # restore-from-backup), ensure_proxy's certdir.mkdir raises FileExistsError
    # and the user gets a traceback plus a recorded pin.
    try:
        started = pin.apply_pin(switcher, email, org_uuid)
    except Exception as exc:  # noqa: BLE001 — a traceback tells a user nothing
        warning(f"Could not pin the cloud account: {exc}")
        print(dimmed("  nothing is wired; run `cswap pin` to see the current state"))
        return 1
    print(
        f"{accent('Pinned')} the cloud account (RC/artifacts) to "
        f"Account-{account_num} ({email})"
    )

    if not started:
        # False means NO PROXY IS SERVING. The record is written, so the pin
        # takes effect on the next launch that can start one — but nothing is
        # pinned right now, and "Pinned" alone reads as if it were. Suppressing
        # the follow-up note was the only signal this failure had.
        print(dimmed("  but no proxy is running, so nothing is pinned yet"))
        print(dimmed("  the daemon log says why: <backup>/pin-proxy/daemon.log"))
        return 1

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
