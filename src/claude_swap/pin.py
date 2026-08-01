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
import os
import stat
from types import ModuleType

from claude_swap.exceptions import ClaudeSwitchError

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
    import importlib
    import importlib.util

    try:
        found = importlib.util.find_spec("cswap_pin.proxy") is not None
    except (ImportError, ValueError):
        found = False
    if not found:
        raise ClaudeSwitchError(_INSTALL_HINT)
    return importlib.import_module("cswap_pin.proxy")


def is_available() -> bool:
    """Whether the pin can be used. For callers deciding whether to OFFER it
    (a TUI row, a status line) rather than to use it."""
    try:
        _impl()
    except ClaudeSwitchError:
        return False
    return True


# -- launch integration ------------------------------------------------------


def wire_launch_env(switcher, env: dict[str, str]) -> dict[str, str]:
    """Route a child Claude Code through the pin proxy, if one is pinned.

    Returns ``env`` unchanged when there is no pin, when the extra is not
    installed, or when the proxy cannot be started: an optional feature must
    never be able to block a launch.
    """
    try:
        pin = _impl()
    except ClaudeSwitchError:
        # Not installed. A wiring a previous install left behind would
        # otherwise outlive it — see clear_wiring.
        clear_wiring()
        return env
    try:
        pinned = pin.ensure_proxy(switcher)
        if not pinned:
            # No pin this launch. Clear a wiring left pointing at a proxy that
            # is no longer there, or the session dials a dead port.
            pin.unwire_if_dead(switcher.backup_dir / "pin-proxy")
            return env
        port, ca_path = pinned
        return pin.wire_env(env, port, ca_path)
    except Exception:  # noqa: BLE001 — never block the launch
        return env


# -- wiring removal ----------------------------------------------------------
#
# This half deliberately does NOT live in the optional package.

_WIRE_MARK = "_cswapPinWiredKeys"


def clear_wiring() -> bool:
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
    from claude_swap.paths import get_global_config_path

    try:
        path = get_global_config_path()
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
        # 0600 from creation, never write-then-chmod: this file can hold
        # ``primaryApiKey`` and inline MCP credentials, and a plain write takes
        # the umask (022 on a default install), which rename then publishes.
        tmp = path.with_suffix(".cswap-tmp")
        mode = 0o600
        try:
            current = stat.S_IMODE(path.stat().st_mode)
            if not current & 0o077:
                mode = current  # already stricter — do not relax it
        except OSError:
            pass
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, json.dumps(raw, indent=2).encode("utf-8"))
        finally:
            os.close(fd)
        if mode != 0o600:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    except OSError:
        return False
    return True


# -- command -----------------------------------------------------------------


def run(switcher, account: str | None, clear: bool = False) -> int:
    """Entry point for ``cswap pin``. Mirrors :func:`claude_swap.menubar.run`:
    the optional dependency is resolved here, at call time, not at import."""
    from claude_swap.printer import accent, dimmed

    if clear:
        # Works WITHOUT the package on purpose: ``--clear`` is what a user
        # reaches for precisely when they have uninstalled the pin, and the
        # wiring is cswap's own record (see clear_wiring).
        try:
            _impl().apply_pin(switcher, None, None)
        except ClaudeSwitchError:
            if not clear_wiring():
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
