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

def _install_how() -> str:
    """The install COMMAND for this install method, on its own.

    Split out so there is exactly one place that decides it. A second
    hardcoded `uv tool install ...` survived beside the derived hint and
    diverged from it on a pipx machine — one screen apart, both wrong for
    someone.
    """
    from claude_swap.update_check import _detect_install_method

    return {
        "uv": "uv tool install 'claude-swap[pin]'",
        "pipx": "pipx install 'claude-swap[pin]'",
    }.get(_detect_install_method() or "", "pip install 'claude-swap[pin]'")


def _install_hint() -> str:
    """How to install the extra, in a form that reaches THIS install.

    Not a constant, because `pip install` is wrong for the install method most
    users have. Under a uv tool install, pip puts a second copy in whatever pip
    is on PATH and the extra never reaches the tool's environment — the user
    follows the instruction, it succeeds, and the pin is still missing.
    `cswap upgrade` already solves this; reuse its detector rather than
    re-deriving it.
    """
    return f"The cloud pin requires 'cswap-pin'. Install with: {_install_how()}"


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
    #
    # BOUNDED, like the no-package branch above. `unwire_if_dead` takes no
    # timeout and uses the package's own claude_config_lock(timeout=5), so a
    # held .claude.json.lock made every `cswap run` wait 5.3s (measured, 5.19s
    # of it inside the unwire) before returning the env unchanged — and Claude
    # Code holds that lock routinely while refreshing credentials. The budget
    # this path was given was only ever applied to the branch where the package
    # is absent.
    #
    # If the lock is not free right now, SKIP: the wiring is stale but the next
    # launch heals it, and a launch that blocks is worse than a launch that is
    # briefly unpinned — the whole reason this path fails open.
    try:
        if _config_lock_is_free(_LAUNCH_LOCK_BUDGET_S):
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


def _config_lock_is_free(budget: float) -> bool:
    """Can the config lock be taken within ``budget`` seconds?

    A probe, not a hold — the caller re-locks immediately after. That race is
    deliberate: losing it costs one skipped unwire (the next launch heals it),
    while the alternative is the launch itself waiting on the package's own
    5-second lock timeout, which it has no way to shorten.
    """
    from claude_swap.claude_locks import proper_lockfile
    from claude_swap.paths import get_global_config_path

    path = get_global_config_path()
    try:
        with proper_lockfile(path.parent / (path.name + ".lock"), timeout=budget):
            return True
    except Exception:  # noqa: BLE001
        return False


def _pinned_email_now(switcher) -> tuple[str, str] | None:
    """The pin record as cswap's OWN file has it, or None. Never the package.

    Both the clear and set paths need this and neither can ask ``cswap_pin``
    for it: the package is precisely what may be broken, and on the set path
    ``apply_pin`` has already written the record by the time a failure is
    known. ``settings.json -> remoteControl`` is cswap's file, so read it.
    """
    from claude_swap import settings as _s

    section = _s._read_raw(_s.settings_path(switcher.backup_dir)).get("remoteControl")
    if not isinstance(section, dict):
        return None
    email = section.get("pinnedEmail")
    if not email:
        return None
    return email, section.get("pinnedOrganizationUuid")


def _clear_pin_record(switcher) -> None:
    """Drop ``remoteControl`` from settings.json. Never raises.

    Only for the path where the package cannot do it — normally ``apply_pin``
    owns this file's pin section, and going around it would race the daemon's
    own writes. Here there IS no package, so nothing else can.
    """
    from claude_swap import settings as _s

    try:
        path = _s.settings_path(switcher.backup_dir)
        raw = _s._read_raw_for_write(path)
        if raw.pop("remoteControl", None) is not None:
            _s.atomic_write_json(path, raw)
    except Exception:  # noqa: BLE001 — the caller re-reads and reports
        pass


def _wiring_present(switcher) -> bool:
    """Does either config still carry a pin wiring?

    The companion to :func:`clear_wiring`'s return value, which cannot answer
    this: it returns False both for "there was nothing to remove" and for "the
    lock was contended so this path was skipped", and only the second is a
    failure. Read without a lock — a stale read here costs a re-run, while
    waiting on the same lock that just failed costs the command.
    """
    # Imported here, as clear_wiring does: paths.py reads CLAUDE_CONFIG_DIR at
    # CALL time, and a module-scope import would freeze the resolution for a
    # process whose env changes (the `cswap run` case both functions exist for).
    from claude_swap.paths import (
        get_default_global_config_path,
        get_global_config_path,
    )

    seen = set()
    for path in (get_global_config_path(), get_default_global_config_path()):
        if path in seen:
            continue
        seen.add(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — unreadable/absent is not "wired"
            continue
        if isinstance(raw, dict) and raw.get(_WIRE_MARK):
            return True
    return False


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


# -- the operations, shared by the CLI and the TUI ---------------------------
#
# THE VERDICT LIVES HERE, NOT AT EACH CALL SITE. Three review rounds found the
# same shape: a fix landed on the CLI and its sibling in tui/dashboard.py kept
# the old behaviour — first the missing clear_wiring, then the discarded
# apply_pin return, then the rollback and the API-key refusal. Each was a
# separate bug report for one decision implemented twice.
#
# So these return a (ok, message) the caller only has to render. A fourth
# divergence now needs someone to write a second copy of the logic rather than
# to forget a line.


def clear_pin(switcher) -> tuple[bool, str]:
    """Remove the pin AND its wiring. ``(ok, message)``.

    Both halves are re-read afterwards rather than inferred: ``apply_pin``
    cannot report on the wiring, and ``clear_wiring``'s bool is False both for
    "nothing to remove" and for "the lock was contended so this path was
    skipped" — only the second is a failure, and the skip is deliberate.
    """
    had_pin = _pinned_email_now(switcher) is not None
    try:
        impl = _impl()
        impl.apply_pin(switcher, None, None)
    except Exception:  # noqa: BLE001 — this command must work when the pin does not
        # THE RECORD IS CSWAP'S OWN FILE, so clear it here rather than
        # reporting that the package could not. With the extra uninstalled,
        # leaving it meant `--clear` failed, told the user to REINSTALL the
        # package they had just removed (advice that inverts their intent and
        # never converges — run 2 is identical), and then re-pinned the old
        # account the moment anything reinstalled it, live, with no user
        # action. Same for a too-old or broken-root package.
        #
        # This is the stranding clear_wiring was moved into this repo to
        # prevent, one level up: the wiring is cswap's file and gets cleared,
        # and settings.json -> remoteControl is equally cswap's file.
        _clear_pin_record(switcher)
    cleared = clear_wiring(switcher)
    still_pinned = _pinned_email_now(switcher) is not None
    still_wired = _wiring_present(switcher)
    if still_pinned or still_wired:
        what = " and ".join(
            w for w, on in (("the pin", still_pinned), ("the wiring", still_wired)) if on
        )
        return False, f"Could not remove {what} — re-run once it frees up"
    if not cleared and not had_pin:
        return True, "No cloud account pinned"
    return True, "Unpinned the cloud account"


def set_pin(switcher, email: str, org_uuid: str | None) -> tuple[bool, str]:
    """Pin the cloud surface to ``email``. ``(ok, message)``.

    A failure ROLLS THE RECORD BACK: ``apply_pin`` writes ``remoteControl``
    before it starts the proxy, so reporting the failure while leaving it makes
    every read-back — ``cswap pin``, the TUI badge — contradict the message.
    """
    # REFUSED HERE, not at the call sites. An API-key account can never be
    # pinned — `sk-ant-api…` is not OAuth JSON, so the provider returns None
    # for every request and each one fails open: daemon spawned, badge lit,
    # nothing pinned, ever. The TUI's row filter is a courtesy, not the
    # enforcement: refresh_root_menu returns early below depth 1, so an open
    # submenu is never rebuilt while the snapshot keeps updating, and a row
    # that was OAuth when the menu was drawn pins an API-key account when it
    # is selected.
    try:
        num, _e, _o = switcher.resolve_account(email)
        if switcher._account_kind(num) == "api_key":
            return False, (
                f"{email} is an API-key account, which the cloud pin cannot "
                "use: Remote Control and Artifacts need an OAuth bearer"
            )
    except Exception:  # noqa: BLE001 — an unresolvable kind is not a refusal
        pass
    before = _pinned_email_now(switcher)
    try:
        started = _impl().apply_pin(switcher, email, org_uuid)
    except Exception as exc:  # noqa: BLE001 — a traceback tells a user nothing
        try:
            _impl().apply_pin(switcher, *(before or (None, None)))
        except Exception:  # noqa: BLE001
            return False, (
                f"Could not pin the cloud account: {exc} — the record may still "
                f"name {email}, check with `cswap pin`"
            )
        return False, (
            f"Could not pin the cloud account: {exc} — "
            + ("the previous pin is unchanged" if before else "nothing is pinned")
        )
    if not started:
        return False, (
            f"Cloud pin recorded for {email}, but no proxy is running — "
            "nothing is pinned yet"
        )
    return True, f"Pinned the cloud account (RC/artifacts) to {email}"


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
        # THE SAME clear_pin THE TUI CALLS. This branch used to carry its own
        # copy of the logic, which is how the API-key refusal ended up in one
        # front end and not the other for three review rounds. One decision,
        # one implementation, two renderings.
        ok, msg = clear_pin(switcher)
        if not ok:
            warning(msg)
            return 1
        print(msg if msg.startswith("No ") else f"{accent('Unpinned')} the cloud account")
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
    # THE SAME set_pin THE TUI CALLS. This branch carried its own copy of the
    # refusal, the rollback and the no-proxy verdict — and the API-key refusal
    # is the divergence that survived in it after the shared pair was added.
    ok, msg = set_pin(switcher, email, org_uuid)
    if not ok:
        warning(msg)
        if "no proxy is running" in msg:
            print(dimmed("  the daemon log says why: <backup>/pin-proxy/daemon.log"))
        return 1
    print(
        f"{accent('Pinned')} the cloud account (RC/artifacts) to "
        f"Account-{account_num} ({email})"
    )

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
