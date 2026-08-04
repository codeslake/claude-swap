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
import logging
from types import ModuleType

from claude_swap.exceptions import ClaudeSwitchError, ConfigError

_logger = logging.getLogger("claude-swap")

def _log_unresolvable(get, exc: BaseException, level: int = logging.DEBUG) -> None:
    """Record a path getter's raise, every time it happens. DEBUG by default.

    THE LEVEL IS THE CALLER'S, and only `clear_wiring` passes WARNING. No cap.
    Both a cap and a blanket WARNING were wrong, and measured wrong through the
    real CLI rather than reasoned about.

    A once-per-PROCESS cap cannot suppress anything here: the statusline hook
    spawns `cswap pin --heal` fresh on a ~2s cadence (`pin-ensure`), `heal`'s
    only caller is `pin.run(..., heal_only=True)` from `cli.py`, and there is
    no long-lived daemon. The cap's lifetime IS one tick.

    Worse, adding the warning to `_wiring_present` and `_wired_ports` — which
    `heal` calls on EVERY tick regardless of wiring state — inverted the thing
    it claimed to fix. Counting lines in the real rotating log with an
    unreadable `~/.claude`:

        tick   before        after adding it
        1      0             1
        6      0             6

    Before, the warning was reachable only from `clear_wiring`, gated behind
    `_wiring_is_stale`, so it logged once and went quiet. After, ~4.2MB/day,
    overwriting the whole 4MB history every ~22.7h — the damage the code
    claimed to prevent, created by the code that claimed it.

    So the default is DEBUG, for the two getters `heal` calls unconditionally.
    It keeps the observability that was genuinely missing (both swallowed the
    raise silently) without paying for it every tick: the rotating handler does
    not record DEBUG by default, so the record is there for anyone who turns
    the level up and costs nothing when nobody has. `clear_wiring` overrides to
    WARNING because it is gated (see its call site), and because a config that
    could not be LOCATED is the one fact its return value cannot carry: the
    bool is a claim about every path it REACHED. Naming why a wiring could not
    be REMOVED is a different record — the lock WARNING at the bottom of
    `clear_wiring`, which is the site that fires on the stuck shape this one
    cannot reach.
    """
    # `stacklevel=2` ATTRIBUTES THE RECORD TO THE CALLER. Without it all three
    # call sites' records are identical in origin — same `funcName`, same
    # `pathname`, same `lineno`, this line — so nothing downstream can tell the
    # per-tick getters from the gated one. That is not academic: the guard on
    # this split could only key on LEVEL, which made it pass on a fixture that
    # never reached `clear_wiring` at all, and fail on the correct WARNING as
    # soon as the fixture was hardened. With it, `record.funcName` is
    # `_wiring_present` / `_wired_ports` / `clear_wiring`.
    #
    # Production output is UNCHANGED: `logging_config` formats
    # "%(asctime)s - %(levelname)s - %(message)s" and never renders funcName,
    # filename or lineno.
    _logger.log(level, "%s could not be resolved: %s", get.__name__, exc, stacklevel=2)


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
    # NO RUNTIME VERSION FLOOR. The extra's floor lives in ONE place, the
    # `pin = ["cswap-pin>=X"]` requirement, exactly as the menubar extra
    # declares `rumps>=0.4.0` and then only asks whether the import works.
    #
    # A hardcoded tuple here was the alternative, and it does not survive
    # contact with the release cycle: cswap-pin ships on its own schedule, so
    # every release of it would need a matching pull request against THIS
    # project just to raise a constant. A gate whose maintenance depends on
    # someone else's release cadence is a gate that goes stale, and a stale
    # floor is worse than none — it refuses a package the installer has just
    # chosen, with a message blaming the user's version.
    #
    # Keeping a released version out is an INSTALL-time job (the requirement,
    # and whatever provisioning runs it), not something the seam re-litigates
    # on every call.
    return importlib.import_module("cswap_pin.proxy")


# Both display helpers (is_available/pinned_email) are called on every TUI
# RENDER — AccountsPanel.render, AccountCard.render, and twice per
# dashboard._root_entries — not just on the poll. Measured: 0.168ms/call for
# _live_impl's invalidate_caches()+find_spec with the extra absent and a
# 6-entry sys.path, scaling with sys.path length. A TTL well under the TUI's
# poll cadence (POLL_INTERVAL_S = 3.0 in tui/app.py) removes that from every
# render while still noticing a mid-session install: dashboard.refresh_root_menu
# re-renders on every poll tick, so a cache younger than one poll interval is
# stale for at most one render, never for the rest of the session — no
# restart required. Tests must reset this between runs (see conftest.py); it
# is bare module state so nothing else has to plumb a cache handle through.
_LIVE_IMPL_CACHE_TTL_S = 1.0
_live_impl_cache: tuple[float, ModuleType | None] = (float("-inf"), None)


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

    Cached for ``_LIVE_IMPL_CACHE_TTL_S`` (see the module-level comment) so a
    render burst pays for the resolution once, not once per widget.
    """
    import importlib
    import time as _time

    global _live_impl_cache
    cached_at, cached = _live_impl_cache
    now = _time.monotonic()
    if now - cached_at < _LIVE_IMPL_CACHE_TTL_S:
        return cached

    importlib.invalidate_caches()
    try:
        resolved = _impl()
    except Exception:  # noqa: BLE001
        resolved = None
    _live_impl_cache = (now, resolved)
    return resolved


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
        #
        # ASK FIRST, LOCK ONLY IF THERE IS WORK. The budget is per PATH and
        # clear_wiring takes one lock per config, so a user who never installed
        # the pin — the case this budget exists for — paid it twice: measured
        # 1.37-1.64s with Claude Code holding the lock, against a documented
        # cap of 0.5s. _wiring_present is lock-free and answers in ~1.5ms, and
        # for that user the answer is always "nothing to remove".
        #
        # AND NOT SERVING. `_impl()` raising says nothing about the daemon: a
        # broken cryptography, a half-finished reinstall, an import error in a
        # new release all land here while the proxy on the port keeps answering
        # every session already wired to it. This branch used to unwire on
        # presence alone. Measured: with `_impl` raising and the port serving,
        # one `cswap run` stripped the env block and unpinned a healthy pin —
        # the same damage heal's own guard exists to prevent, at the other
        # call site.
        #
        # The probe is bounded well under the launch budget rather than given
        # the default 2s: a black-holed port must not turn a launch-path guard
        # into the stall it was written to avoid.
        #
        # `clear_wiring`'s WARNINGs — there are TWO of them, the getter one at
        # the top and the lock one at the bottom — are self-limiting HERE only
        # in the same sense they are inside `heal` (see those call sites): the
        # gate goes false when the removal succeeds, not otherwise. So an
        # unremovable wiring logs per LAUNCH, and on the shape where both fire
        # it logs TWICE per launch. Measured, 6 launches through this branch,
        # counting only what the INFO-level logger actually writes:
        #
        #                                       259f598      HEAD
        #   no HOME, unremovable (ro dir)      6 lines     12 lines  {getter 6, lock 6}
        #   no HOME, REMOVABLE                 1 line       1 line
        #   nothing wired                      0 lines      0 lines
        #   HOME fine, ro dir, wired           0 lines      6 lines  {lock 6}
        #
        # THE BOTTOM ROW IS THE FIX WORKING, not a regression. That shape — a
        # config dir this process cannot write, HOME perfectly resolvable — is
        # the flagship stranding, and before the lock WARNING existed it went
        # to the user's screen on every launch with ZERO records at any level
        # naming which config or why. The getter WARNING cannot cover it: on
        # this shape nothing raises and it never fires (row 4, left column).
        #
        # At human cadence six launches is negligible either way, which is why
        # the churn arithmetic lives at the statusline call site and not here.
        try:
            if _wiring_is_stale(switcher, connect_timeout=_LAUNCH_PROBE_S):
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

# The same reasoning for the SERVING probe on that path. A refused connect on
# loopback comes back in microseconds, so this only ever bites on a port that
# accepts nothing and answers nothing — a firewall rule, a half-dead daemon —
# and there a 2s default would blow the launch budget four times over on the
# probe alone. Guessing "not serving" after 0.2s costs at worst one unwire the
# next launch redoes; guessing wrong the other way costs a stalled launch.
_LAUNCH_PROBE_S = 0.2


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
    #
    # EACH GETTER CAN RAISE (see the same guard on `_wired_ports` and
    # `_wiring_present`): `get_default_global_config_path` calls `Path.home()`,
    # which raises `RuntimeError` with no HOME and no `/etc/passwd` entry. A
    # config this call cannot even locate has nothing to clear there — that
    # is a fact about ONE config, not a reason to abandon the other.
    #
    # LOGGED, not just skipped: a config that could not be RESOLVED and one
    # that resolved with nothing wired both leave this loop silently short a
    # path, and `clear_wiring`'s bool is a claim about every path it reached
    # — not a claim that every path was reachable. Without a record, "the
    # default profile was never attempted because HOME could not be found"
    # and "the default profile was attempted and had nothing wired" are the
    # same silence from the outside.
    paths = []
    for get in (get_global_config_path, get_default_global_config_path):
        try:
            path = get()
        except Exception as exc:  # noqa: BLE001 — unresolvable: no opinion
            # WARNING HERE ONLY. `heal` reaches `clear_wiring` through
            # `_wiring_is_stale`, which goes false ONCE THE REMOVAL SUCCEEDS
            # — so this logs once and goes quiet, which is what `cae7cfa` did
            # correctly before I broke it. The two getters `heal` calls
            # UNCONDITIONALLY stay at DEBUG; putting WARNING there is what
            # produced 12 lines per 6 ticks.
            #
            # IT IS NOT THE RECORD THAT EXPLAINS AN UNREMOVABLE WIRING. Four
            # rounds of this comment claimed it was, and the claim is false in
            # both directions. Measured, 10 `heal` ticks, read-only config dir
            # with HOME perfectly resolvable — the flagship shape those rounds
            # named:
            #
            #   heal   : "…could not be removed (the config is locked) —
            #             re-run `cswap pin --heal`", every tick
            #   THIS site: never fires (nothing raises; both paths resolve)
            #
            # Getting it to fire needs `Path.home()` to raise as well, and
            # then it names `get_default_global_config_path` while the STUCK
            # config is the one `get_global_config_path` resolved fine. Put
            # the wiring in the raising getter's own config instead and
            # `_wiring_present` cannot see it either, so `heal` answers
            # "Nothing to heal" and never reaches this function at all
            # (measured, both shapes). There is no reachable shape where this
            # WARNING names an unremovable wiring's cause.
            #
            # What does name it is the lock-failure WARNING at the bottom of
            # this function, added for exactly that gap. This record's job is
            # the narrower one it can actually do: a config that could not be
            # LOCATED is missing from `paths`, and `clear_wiring`'s bool is a
            # claim about every path it REACHED.
            #
            # SO THE COST HERE IS ONE LINE, not the per-tick churn the old
            # table charged it. Measured through the real `setup_logging`
            # handler, 10 ticks, `Path.home()` raising and the removal
            # succeeding: 1 line, 98 B — `_wiring_is_stale` goes false and
            # this is never reached again. The 10-line column belongs to the
            # bottom WARNING and is accounted for there.
            _log_unresolvable(get, exc, logging.WARNING)
            continue
        if path not in paths:
            paths.append(path)

    # ONE LOCK PER PATH. The shared config lock derives its directory from
    # get_global_config_path(), so a single lock around the loop guards one
    # file and leaves the other rewritten unprotected — racing `cswap switch`
    # and Claude Code, the whole-file clobber the lock exists to prevent.
    #
    # ``timeout`` is a TOTAL, not a per-file allowance. Passing it to each
    # acquisition made the real worst case a multiple of the number of
    # configs, so the launch path's sub-second cap silently became ~2x that
    # (measured: 1.37-1.64s against a documented 0.5s).
    import time as _time

    # An UNTIMED call still gets a total. Leaving `None` meant each config
    # independently waited the lock's own default, so `cswap pin --clear`
    # with both locks held froze for 2x that (measured: 18.18s against a 9s
    # default) — the same multiple-of-the-configs bug this deadline was added
    # to fix, on the branch that did not pass a timeout.
    if timeout is None:
        from claude_swap.claude_locks import DEFAULT_TIMEOUT_S

        timeout = DEFAULT_TIMEOUT_S
    deadline = _time.monotonic() + timeout
    changed = False
    for i, path in enumerate(paths):
        left = deadline - _time.monotonic()
        if left <= 0:
            continue  # budget spent; the next launch heals what is left
        # FAIR SHARE of what remains, not "however much is left". Handing the
        # first path the whole remaining budget let a config that stayed
        # contended for the entire call consume it all, so a SECOND path
        # whose lock was completely free was skipped by the `left <= 0` check
        # above without ever being tried. Measured: session lock held for the
        # full 0.5s budget, `clear_wiring` returned False with BOTH configs
        # still wired.
        #
        # Dividing by how many paths are still untried gives each one at
        # least an equal slice of whatever time remains when its turn comes,
        # while the running total can still never exceed `timeout` — each
        # share is carved out of `left`, never added to it.
        share = left / (len(paths) - i)
        try:
            with proper_lockfile(
                path.parent / (path.name + ".lock"), timeout=share
            ):
                if _clear_wiring_locked(switcher, path):
                    changed = True
        except Exception as exc:  # noqa: BLE001
            # A lock we cannot take is a reason to skip THIS file, not to
            # abandon the other one — and on the launch path (sub-second
            # budget) a contended config must not fail the clear outright.
            #
            # BUT SAY WHICH FILE AND WHY. Skipping silently is what left the
            # flagship failure with no record anywhere: measured on a
            # read-only config dir with HOME perfectly resolvable, five `heal`
            # ticks each told the user "could not be removed (the config is
            # locked) — re-run `cswap pin --heal`" and produced ZERO log
            # records at any level, while this line swallowed
            # `PermissionError: [Errno 13] ... '<session>/.claude.json.lock'`
            # — the one fact naming the cause. The getter WARNING above cannot
            # cover it (see its comment): on this shape it does not fire.
            #
            # THE CHURN IS HERE, then. Measured through the real
            # `setup_logging` handler, 10 `heal` ticks each:
            #
            #   nothing wired                       0 lines,  unwired
            #   stale, removed on tick 1            0 lines,  unwired
            #   stale, unremovable (read-only dir) 10 lines,  still wired
            #
            # A LINE'S SIZE DEPENDS ON WHICH EXCEPTION THIS CAUGHT, and there
            # are two reachable kinds. `%s` renders the path once bare, and
            # then whatever the exception's own message carries. Measured
            # through the real formatter, then scaled to
            # `/home/j.lee8/.claude.json` (25 chars):
            #
            #                          path appears   per line    active file
            #   PermissionError          2x           147 B       wraps 3.96 h
            #   ClaudeCodeLockTimeout    1x           198 B       wraps 2.94 h
            #
            # `PermissionError` renders the LOCK path (the config path plus
            # `.lock`), so the config path appears twice and the old
            # `97 B + 2*len(path)` model is exact for it — at every length
            # measured, 22 through 140 chars, diff 0.
            #
            # `ClaudeCodeLockTimeout` names only `lock_dir.name`, so the path
            # appears ONCE, and the model does not fit it AT ALL — the miss is
            # not a constant. Its fixed sentence ("Could not acquire … Claude
            # Code appears to be refreshing credentials. Retry in a few
            # seconds.") is paid whatever the path costs, so the difference
            # between the two kinds is `76 - len(path)`: measured exact at
            # every length tested.
            #
            #   len(path)   PermErr    Timeout    gap    76-len
            #          25       147        198     51        51   <- documented
            #          71       239        244      5         5
            #          78       253        251     -2        -2
            #         140       377        313    -64       -64
            #
            # The first row is the table above, measured at exactly the
            # documented path rather than extrapolated: the gap is 51 B, NOT
            # the 32 the prose used to claim (32 corresponds to a 44-char
            # path). It DECAYS with path length, and past 76 chars it INVERTS
            # — the timeout becomes the cheaper line. At 43200 ticks: 6.06 MiB/day
            # and 8.16 MiB/day respectively, against `logging_config.py`'s
            # `maxBytes=1024 * 1024`, so a `tail` three hours late shows
            # none of it; the 4 MiB across all `backupCount=3` rotations
            # lasts 15.9 h / 11.8 h.
            #
            # THE WORST CASE IS THE TIMEOUT, and it is also the one that never
            # stops. An ORPHANED lock dir (a holder killed -9) inside a config
            # dir this process cannot write is permanent: the takeover path
            # `rmdir`s the stale dir, that `rmdir` needs write permission on
            # the parent, and it never gets it. Measured, 10 ticks: 10 lines,
            # still wired, every tick identical.
            #
            # KEPT AT WARNING, BOTH KINDS, and this is the site the "a silent
            # log is the worse failure" argument was always about: it is the
            # only record naming which config and which cause, and every one
            # of those ticks is already printing an unexplained failure to the
            # user's status line.
            #
            # SPLITTING `ClaudeCodeLockTimeout` DOWN TO DEBUG WAS PROPOSED AND
            # REFUSED — ON COST, which is the only ground that holds. The type
            # does not separate the two cases: a live competitor raises
            # `ClaudeCodeLockTimeout` and so does the permanently-orphaned lock
            # dir above, so a type-keyed split silences the stuck machine this
            # WARNING exists for and keeps only `PermissionError`, the kind
            # that already names its own errno.
            #
            # A DISCRIMINATOR DOES EXIST, though earlier versions of this
            # comment said none did. The lock dir's mtime age tells them apart
            # at the moment of the raise, and `proper_lockfile` already does
            # both: `os.stat(lock_dir).st_mtime`, then `> staleness` against
            # `CONFIG_STALENESS_S`. Measured, ~4 lines, and `os.stat` works
            # fine on a `0o500` dir:
            #
            #   LIVE competitor (fresh mtime)  Timeout | age    0.3s TRANSIENT
            #   ORPHAN + read-only parent      Timeout | age 3600.3s PERMANENT
            #   LIVE competitor + ro parent    Timeout | age    0.3s TRANSIENT
            #   CC holder, mtime 4s old        Timeout | age    4.3s TRANSIENT
            #   no lock dir + ro parent        PermErr | age   None  PERMANENT
            #
            # Right on all five shapes. So the refusal cannot rest on "nothing
            # can tell them apart". It rests on the transient case being too
            # cheap to be worth the code.
            #
            # AND THE OLD "10 LINES" FIGURE MIXED TWO CADENCES. It came from a
            # tight loop of 10 `heal()` calls back to back, while the permanent
            # case was priced at the real ~2s statusline cadence (43200
            # ticks/day) — the comparison ran in the direction that made its
            # own refusal look weaker. Re-measured, a 3s competitor, both
            # cadences on the same fixture:
            #
            #   tight loop (no sleep)   11 unwire lines, cleared on tick 12
            #   real ~2s statusline      2 unwire lines, cleared on tick 3
            #
            # THE 2 IS A FUNCTION OF THE COMPETITOR'S 3s, not a property of
            # the transient case: the line count is however many ticks fit
            # inside the hold. Same 2s cadence, competitor length moved:
            #
            #    3s competitor (above)         2 lines, cleared on tick 3
            #   30s competitor                14 lines, cleared on tick 15
            #   30s NON-TOUCHING holder        5 lines, cleared on tick 6
            #
            # The non-touching row caps at 5 because `CONFIG_STALENESS_S`
            # (10.0) lets `proper_lockfile` take the lock over, so the
            # unbounded shape is a LIVE holder that keeps touching — which is
            # the first two rows, and is what a Claude Code credential
            # refresh actually is.
            #
            # TWO LINES, once, against 43200/day forever. That is what four
            # lines of mtime arithmetic would buy, and it is not worth it —
            # 1:21600 at the documented 3s, still ~1:3000 at 30s. The
            # transient case is self-limiting by construction: the competitor
            # lets go and the very next free tick unwires the config, while
            # the orphan is still on line 43200 with nothing changed.

            _logger.warning("%s could not be unwired: %s", path, exc)
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
    # `or ""` to match the WRITER. cswap_pin.save_pin always writes
    # `org_uuid or ""`, so a record with no org key read back as None here
    # while the same record after a rollback read as "" — unequal, and
    # _restore_pin then reported a successful rollback as a failure. The
    # package's own load_pin already normalizes; this reader had diverged.
    return email, section.get("pinnedOrganizationUuid") or ""


def _safe(exc: object) -> str:
    """An exception rendered for display, with URL userinfo removed.

    Every failure renderer here interpolates ``str(exc)`` — into CLI output,
    into a TUI modal, into a MENU LABEL — and the text comes out of an
    optional third-party package the seam does not control. A proxy URL
    carrying ``user:secret@host`` in a message would reach the screen
    verbatim. No path in this PR builds one; the scrub is here because the
    seam has no way to promise none ever will.

    USERINFO ONLY. This is not general redaction: a bearer token in a header
    dump or a ``password 'x'`` embedded in a package's own message passes
    through untouched. Only ``scheme://user:pass@host`` is recognized and
    scrubbed.
    """
    import re

    # scheme://userinfo@host  ->  scheme://***@host. Anchored on "://" so an
    # ordinary email address in a message is left alone.
    return re.sub(r"(?<=://)[^/\s@]+@", "***@", str(exc))


def _rollback_tail(rolled: bool, before, email: str) -> str:
    """How a failed set_pin ended, in the record's own terms.

    Both failure branches say the same three things and said them twice.
    """
    if not rolled:
        return f"and the record may still name {email}, check with `cswap pin`"
    return "the previous pin is unchanged" if before else "nothing is pinned"


def _restore_pin(switcher, before: tuple[str, str] | None) -> bool:
    """Put the record back the way ``before`` had it. True when it IS back.

    The verdict is MEASURED, never inferred from the restore call: when
    ``_impl()`` itself is what raised, ``apply_pin`` never ran and the record
    was never touched, so a message claiming "may still name <email>" was
    telling the user to go check a state the code could already disprove.
    """
    try:
        _impl().apply_pin(switcher, *(before or (None, None)))
    except Exception:  # noqa: BLE001 — the re-read below is the verdict
        pass
    return _pinned_email_now(switcher) == before


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


def _wire_mark_of(raw: object) -> list | None:
    """The marker THIS module wrote, or None. The single reader.

    ``_wiring_present`` and ``_clear_wiring_locked`` both answer "is it
    wired", and they disagreed: one accepted any truthy marker, the other
    required a non-empty list. A malformed marker (a hand-edit, a format
    change in a future cswap-pin) therefore satisfied the first and not the
    second, so `--clear` reported "could not remove the wiring — re-run once
    it frees up" forever: nothing was contended and nothing ever converged.
    """
    if not isinstance(raw, dict):
        return None
    ours = raw.get(_WIRE_MARK)
    return ours if isinstance(ours, list) and ours else None


def _wiring_present(_switcher) -> bool:
    """Does either config still carry a pin wiring?

    The companion to :func:`clear_wiring`'s return value, which cannot answer
    this: it returns False both for "there was nothing to remove" and for "the
    lock was contended so this path was skipped", and only the second is a
    failure. Read without a lock — a stale read here costs a re-run, while
    waiting on the same lock that just failed costs the command.

    ``_switcher`` is unused: both configs resolve from ``claude_swap.paths``,
    not from the switcher instance. Kept (and underscore-prefixed rather than
    dropped) so every call site stays symmetric with :func:`clear_wiring` and
    :func:`_wired_port_is_serving`, which the pin CLI/TUI pass a switcher to
    interchangeably — dropping it here alone would make this one predicate
    look different from its siblings for no reason a caller could see.
    """
    # Imported here, as clear_wiring does: paths.py reads CLAUDE_CONFIG_DIR at
    # CALL time, and a module-scope import would freeze the resolution for a
    # process whose env changes (the `cswap run` case both functions exist for).
    from claude_swap.paths import (
        get_default_global_config_path,
        get_global_config_path,
    )

    seen = set()
    for get in (get_global_config_path, get_default_global_config_path):
        # THE GETTER ITSELF CAN RAISE (see the same guard on `_wired_ports`,
        # Task 1 of the prior round): `get_default_global_config_path` calls
        # `Path.home()`, which raises `RuntimeError` with no HOME and no
        # `/etc/passwd` entry. `heal` survives that today only because this
        # function's own raise happens to land inside `heal`'s bottom `try`
        # — a refactor moving the call above it would reintroduce the
        # traceback `heal` documents as never happening. Measured with no
        # HOME: `_wired_ports()` guarded this and returned `[]`;
        # `_wiring_present` had no such guard and raised.
        try:
            path = get()
        except Exception as exc:  # noqa: BLE001 — unresolvable: no opinion
            _log_unresolvable(get, exc)
            continue
        if path in seen:
            continue
        seen.add(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — unreadable/absent is not "wired"
            continue
        if _wire_mark_of(raw) is not None:
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

    ours = _wire_mark_of(raw)
    if ours is None:
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


def set_pin(
    switcher, email: str, org_uuid: str | None, num: str | None = None
) -> tuple[bool, str]:
    """Pin the cloud surface to ``email``. ``(ok, message)``.

    A failure ROLLS THE RECORD BACK: ``apply_pin`` writes ``remoteControl``
    before it starts the proxy, so reporting the failure while leaving it makes
    every read-back — ``cswap pin``, the TUI badge — contradict the message.

    ``num`` is the slot both call sites ALREADY resolved. Re-deriving it from
    the email here was a real bypass, not a tidiness point: cswap's own
    documented personal+org pattern gives one address two slots, so
    ``resolve_account(email)`` raises ``ConfigError`` and the API-key refusal
    below was skipped entirely — accepting exactly the account it exists to
    reject.
    """
    # REFUSED HERE, not at the call sites. An API-key account can never be
    # pinned — `sk-ant-api…` is not OAuth JSON, so the provider returns None
    # for every request and each one fails open: daemon spawned, badge lit,
    # nothing pinned, ever. The TUI's row filter is a courtesy, not the
    # enforcement: refresh_root_menu returns early below depth 1, so an open
    # submenu is never rebuilt while the snapshot keeps updating, and a row
    # that was OAuth when the menu was drawn pins an API-key account when it
    # is selected.
    #
    # A kind we cannot READ is not permission to proceed. Swallowing the
    # lookup turned an unreadable sequence.json into a silent skip of the
    # refusal — the failure mode is identical to having no refusal at all,
    # and it is invisible. Refuse loudly instead; the user can fix the store.
    if num is None:
        try:
            num = switcher.resolve_account(email)[0]
        except Exception as exc:  # noqa: BLE001
            return False, (
                f"Could not resolve {email} to one account ({_safe(exc)}), so the "
                "cloud pin cannot check it is not an API-key account"
            )
    try:
        kind = switcher._account_kind(num)
    except Exception as exc:  # noqa: BLE001
        return False, (
            f"Could not read what kind of account {email} is ({_safe(exc)}); the "
            "cloud pin needs an OAuth account and will not guess"
        )
    if kind == "api_key":
        return False, (
            f"{email} is an API-key account, which the cloud pin cannot "
            "use: Remote Control and Artifacts need an OAuth bearer"
        )
    before = _pinned_email_now(switcher)
    try:
        started = _impl().apply_pin(switcher, email, org_uuid)
    except Exception as exc:  # noqa: BLE001 — a traceback tells a user nothing
        rolled = _restore_pin(switcher, before)
        return False, (
            f"Could not pin the cloud account: {_safe(exc)} — "
            + _rollback_tail(rolled, before, email)
        )
    if not started:
        # SAME DEFECT AS THE RAISE PATH, sibling branch. apply_pin writes the
        # record before starting the proxy, so leaving it here made the two
        # commands contradict each other: `cswap pin 2` said "nothing is
        # pinned yet" and exited 1 while `cswap pin` then printed the address
        # and exited 0, with the ○ cloud badge lit. Roll back to whatever was
        # pinned before, exactly as a raise does.
        rolled = _restore_pin(switcher, before)
        return False, (
            f"Could not pin the cloud account to {email}: no proxy is running, "
            "so nothing is pinned yet — " + _rollback_tail(rolled, before, email)
        )
    return True, f"Pinned the cloud account (RC/artifacts) to {email}"


def _wiring_is_stale(_switcher, connect_timeout: float = 2.0) -> bool:
    """Should this wiring be removed? Present AND not serving.

    THE VERDICT LIVES HERE, NOT AT EACH CALL SITE — the rule this file already
    states for the pin record, applied to the wiring. It was not, and the two
    places that forgot the serving half both tore down a working pin:

      * ``heal`` had the guard.
      * ``wire_launch_env`` did not. Measured: with ``_impl()`` raising for a
        reason unrelated to the daemon (a broken ``cryptography`` after an
        unrelated upgrade — precisely the case ``_impl`` re-raises separately),
        one ``cswap run`` unwired a pin whose port was answering, and every
        session on the box lost it.

    ``connect_timeout`` exists because the launch path has a sub-second budget
    and a black-holed port would otherwise blow it on the probe alone.

    ``_switcher`` is unused (see :func:`_wiring_present`) but kept, and
    underscore-prefixed, so this predicate stays call-compatible with its
    sibling and every call site can keep passing the switcher it already has
    on hand without checking which predicate needs it.
    """
    # THIS GUARD DOES A SECOND JOB the comment below (added when the
    # per-config verdict became machine-wide) does not mention: it is also
    # the ONLY thing left checking the `_cswapPinWiredKeys` MARKER before a
    # port is treated as cswap's to condemn. The short-circuit this replaced
    # read only the session config's own port (a now-deleted per-config
    # helper — see Task 3 of this file's history) and so incidentally never
    # reached the serving probe for a config cswap never wired — the marker
    # was never checked explicitly, but the narrow per-config read meant it
    # didn't have to be. `_wired_ports()` below reads BOTH configs' ports
    # with NO marker check at all, so without this line a foreign
    # `CSWAP_PIN_PORT` (no marker, e.g. a future `cswap-pin` release that
    # stops writing it, or an unrelated var of the same name) sitting in
    # either config and pointing at a dead port makes `_wiring_is_stale`
    # True with nothing of cswap's actually wired.
    #
    # Measured with this line deleted: `_wiring_present=False`,
    # `_wired_ports=[<dead>]`, `_wiring_is_stale=True`, and `heal()` reports
    # "Removed a cloud pin wiring…" while the config is byte-for-byte
    # unchanged — a false removal claim in the machine-readable channel the
    # status line polls on a timer. `_clear_wiring_locked` itself still
    # refuses to touch a markerless file, so nothing is ever mutated; the
    # bug is entirely in the VERDICT this guard exists to keep honest.
    if not _wiring_present(_switcher):
        return False
    # "I CANNOT TELL" IS NOT "IT IS DEAD". `_wiring_present` keys on the
    # marker; the serving probe reads CSWAP_PIN_PORT. A config carrying the
    # marker and no port satisfied both "wired" and "not serving" at once, so
    # the launch path tore it down — against a proxy that may be perfectly
    # live.
    #
    # Today's writer always emits the port, so this is not reachable through
    # it. But the seam's stated threat model is that the package is a PEER on
    # an independent release schedule, and refusing to trust its return value
    # while trusting its file FORMAT with the destructive operation is the same
    # inference this module keeps being burned by.
    #
    # MACHINE-WIDE, not per-config: this guards a WHOLE-MACHINE action
    # (`clear_wiring` clears every wired config), so "I cannot tell" has to
    # mean nothing ON THE MACHINE names a readable port — not merely that
    # THIS config alone does not. The shipped deployment shape makes the
    # narrower reading the COMMON case, not a corner one: `cswap run` wires
    # ~/.claude.json and launches a child whose OWN config is seeded with no
    # wiring at all, and the status line hook inside that child is what calls
    # `heal` on a timer. So the process that heals is normally the one whose
    # own config has no port to name — and a dead port sitting in the OTHER
    # config must still be reachable. Measured (real path getters, package
    # uninstalled): own config unwired, ~/.claude.json wired to a dead port —
    # the per-config read (own config only) was None, so this used to return
    # False and `heal()` answered "Nothing to heal" over a dead port that
    # survived.
    if not _wired_ports():
        return False
    return not _wired_port_is_serving(_switcher, connect_timeout=connect_timeout)


def _port_of_config(path) -> int | None:
    """The pin port ONE config file names, or None when it names none, is
    unreadable, or malformed. The single-file read :func:`_wired_ports`
    builds on, so a config's own answer is asked once.

    RANGE-CHECKED HERE, at the read, not at the probe. A value outside
    0-65535 is not a port at all — `int()` accepts it happily, but
    `socket.connect` raises `OverflowError` for it, a type
    `_wired_port_is_serving` never catches (it only catches `OSError`), and
    both its call sites inside `heal` sit OUTSIDE its bottom `try`. That
    turned a malformed `CSWAP_PIN_PORT` (any hand-edit or future writer bug,
    e.g. 99999, 70000, -1, 4294967296) into a traceback out of `cswap pin
    --heal` — called from the status line on a timer, where `heal`
    documents "never raises". Treating it as "no opinion" here, at the
    source, means every downstream consumer inherits the fix for free.
    """
    import json as _json

    try:
        env = _json.loads(path.read_text(encoding="utf-8")).get("env") or {}
        port = int(env.get("CSWAP_PIN_PORT") or 0)
    except Exception:  # noqa: BLE001 — unreadable/unwired: no opinion
        return None
    return port if 0 < port <= 65535 else None


def _wired_ports() -> list[int]:
    """Every pin port the configs name, in read order. Unreadable ones are
    absent rather than zero — "no opinion" and "port 0" are different facts.

    For "is ANYTHING wired at all" questions (``_wiring_present``,
    ``clear_wiring``, the every-config-must-serve probe below) where both
    configs' opinions genuinely apply at once.
    """
    from claude_swap.paths import (
        get_default_global_config_path,
        get_global_config_path,
    )

    ports, seen = [], set()
    for get in (get_global_config_path, get_default_global_config_path):
        # THE GETTER ITSELF CAN RAISE. `get()` is not just "resolve a path and
        # a set membership test" — a claim this file's own history made once
        # and measured wrong: `get_default_global_config_path` calls
        # `Path.home()`, which raises RuntimeError when HOME is unset and the
        # uid has no /etc/passwd entry (the standard rootless-container
        # shape). `heal`'s docstring promises "never raises" because the
        # status line calls it on a timer, and this function sits on the path
        # from `heal` through `_wired_port_is_serving` with no guard above it
        # — a raise here reached the status line's caller directly. Measured:
        # `pin.heal(sw)` raised RuntimeError instead of returning
        # ``(False, 'Could not heal…')``.
        try:
            path = get()
        except Exception as exc:  # noqa: BLE001 — unreadable/unresolvable: no opinion
            _log_unresolvable(get, exc)
            continue
        if path in seen:
            continue
        seen.add(path)
        port = _port_of_config(path)
        if port:
            ports.append(port)
    return ports


def _wired_port_is_serving(_switcher, connect_timeout: float = 2.0) -> bool:
    """Is the port the CONFIG names actually answering?

    Asks the thing that is about to be removed, rather than any state file.
    ``proxy.json`` is unlinked at the START of a respawn, so its absence is not
    proof of death while the original daemon is still serving — deciding from
    the record alone has already unwired a live pin once.

    Works with the extra absent or broken: it is a loopback connect, not an
    import. That matters because the uninstalled case is exactly when a user
    can least afford a wrong answer in either direction.

    False when nothing is wired, when the port is unreadable, or when it
    refuses — all of which mean "healing is allowed to proceed".

    ``_switcher`` is unused (see :func:`_wiring_present`) but kept, and
    underscore-prefixed, for the same call-compatibility reason.
    """
    import socket

    # EVERY WIRED CONFIG MUST SERVE, not merely one of them.
    #
    # This used to return True on the first config that answered, and the two
    # are written asymmetrically: `cswap_pin.wire_global_config` writes only
    # the session config, while this reads both — the same asymmetry
    # `clear_wiring` documents as its reason for clearing both. So a live
    # session config masked a DEAD default config, and a user launching plain
    # `claude` from a terminal booted against the dead one while `--heal`
    # answered "Nothing to heal" every tick. Measured:
    #
    #     session cfg -> 42967 (LIVE)   default cfg -> 39967 (DEAD)
    #     _wired_port_is_serving : True      <- OR over both
    #     heal()                 : (False, "Nothing to heal")
    #     default cfg still names the dead port: True
    #
    # An unwired config is not a counter-example — it sends nobody anywhere.
    # Only a config that NAMES a port has an opinion, and every such opinion
    # has to be right for the pin to be serving.
    ports = _wired_ports()
    for port in ports:
        sock = socket.socket()
        sock.settimeout(connect_timeout)
        try:
            sock.connect(("127.0.0.1", port))
        except OSError:
            return False  # a config names a port nothing serves
        finally:
            sock.close()
    return bool(ports)


def heal(switcher) -> tuple[bool, str]:
    """Make the pin serving again, or make it harmless. ``(changed, message)``.

    A DEAD PIN MUST NOT TAKE THE SESSION WITH IT. Everything else here reacts
    to a launch, so when the daemon dies while sessions are up nothing brings
    it back — and the stale wiring in ``.claude.json`` is applied at BOOT, so
    new sessions inherit the dead port too and cannot start either. Measured on
    this machine: every session on the box showed ``Unable to connect to API
    (ConnectionRefused) · attempt 6/300`` for hours, while the proxies behind
    the pin (CCF on 9901, privoxy on 8118) were healthy the whole time. A human
    had to re-pin by hand.

    Two outcomes, in order of preference:

    1. Restart the daemon on the SAME port. Live sessions are already wired to
       that address and their env is fixed at exec, so a daemon returning to it
       is picked up with no restart and nothing to reconnect.
    2. Failing that, REMOVE THE WIRING. Unpinned is a working session; wired to
       a dead port is not. The fallback the shell provides (the corporate proxy,
       or nothing) is what the user had before they ever pinned.

    Never raises: this is called from the status line every few seconds, and a
    health check that can break the prompt is worse than the fault it reports.
    """
    # A SERVING PIN IS NEVER **TORN DOWN**. That is what the guard protects,
    # and the destructive operation is `clear_wiring` at the bottom — not the
    # restart. This used to return here on `serving`, before `impl.heal()` ran
    # at all, and that made a whole class of repair unreachable:
    #
    #   a daemon SERVING its wired port while running code we no longer ship
    #
    # is exactly the state an upgrade leaves behind, and it answered "Nothing
    # to heal" forever. Measured across three machines after installing a new
    # release: two had daemons serving their own wired port, 24h old, running
    # the previous version, and every tick declined to touch them. The third
    # recycled only because its wiring named a DEAD port — the right outcome
    # for the wrong reason.
    #
    # So the restart runs FIRST and the serving check gates only the unwire.
    # `impl.heal` is safe to call in the serving case by construction: it
    # returns False for "serving, wired, and current" and recycles only when
    # the fingerprint says the daemon predates the installed code — rebinding
    # the SAME port, so live sessions never see the swap.
    impl = _live_impl()
    if impl is not None:
        try:
            # Covers THREE halves now: restart a daemon that died, re-wire a
            # daemon that is serving while the config names nothing, and
            # recycle one that is serving but obsolete. The second is the state
            # a recovery leaves behind; the third is the state an upgrade does.
            # RE-READ THE TRUE AS WELL AS THE FALSE. The branch below already
            # refuses to infer an outage from a False; trusting a True was the
            # same mistake pointing the other way, and this function's whole
            # thesis is that a verdict comes from the state, not from a call.
            #
            # It matters because the package is a PEER on its own release
            # schedule (see _impl): the seam cannot promise what a future
            # version returns. Measured with an impl that returns True while
            # binding nothing —
            #     heal() -> (True, "Restored the cloud pin")
            #     the wired port actually serving? False
            # and the status line, which calls this on a timer, would show
            # healthy while every session still dialled a dead port. That is
            # the failure this file names as its signature defect.
            if impl.heal(switcher.backup_dir) and _wired_port_is_serving(switcher):
                return True, "Restored the cloud pin"
        except Exception:  # noqa: BLE001 — fall through to the safe outcome
            pass
        # The restart may have succeeded while returning False (it also uses
        # False for "already serving"). Re-READ rather than infer: unwiring a
        # pin that just came back is the same damage as unwiring a live one.
        if _wired_port_is_serving(switcher):
            return False, "Nothing to heal"
    elif _wired_port_is_serving(switcher):
        # No package, so nothing can restart OR recycle — but a serving pin is
        # still a working one, and removing its wiring would unpin a healthy
        # session. The guard has to survive the package being absent, which is
        # exactly when a user can least afford a wrong answer.
        #
        # The port the WIRING names is the right question, not any state file:
        # `_spawn_daemon` unlinks proxy.json as its first act, so a missing
        # record is not proof of death while the original daemon still serves.
        return False, "Nothing to heal"
    # No package, or the restart failed. Either way the wiring must not outlive
    # the daemon it points at. clear_wiring works WITHOUT the package on
    # purpose — the wiring is cswap's own record, and the case where the extra
    # is broken is exactly when a user cannot afford to be stranded.
    #
    # AND SAY WHICH OF THE TWO HAPPENED. `present and clear_wiring(...)`
    # collapsed "there was nothing to remove" into "I could not remove it", and
    # fell through to the healthy verdict for both. The second is reachable and
    # routine: the budget here is 0.5s and Claude Code holds the config lock
    # during a credential refresh. Measured with the lock held — wiring
    # present, port dead — `heal` returned (False, "Nothing to heal") over an
    # outage in progress, and the wiring survived.
    #
    # That is this file's signature defect, in the channel that matters most:
    # the status line calls `heal` on a timer, so during the exact failure it
    # exists to report, the user's only signal said everything was fine.
    #
    # RE-READ AFTER CLEAR_WIRING, exactly as clear_pin already does — its
    # bool is True when ANY of the two configs changed, not when BOTH did.
    # Measured with the session config's lock held and the default config
    # free: clear_wiring cleared the default, returned True for that one
    # change, and `heal` reported "Removed a cloud pin wiring" while the
    # session config still named the dead port — every new session from that
    # terminal kept booting against it.
    #
    # THE SAME QUESTION `_wiring_is_stale` ASKS, not `_wiring_present` alone.
    # `_wiring_present` keys on the marker only, so a config carrying the
    # marker with no readable CSWAP_PIN_PORT satisfied it and got torn down
    # here — the exact shape `_wiring_is_stale`'s own guard (see its
    # docstring) declares must not be read as "the proxy is dead". `heal` is
    # the worse of the two call sites to leave unguarded: the status line
    # calls it on a timer, unattended, while the launch path runs once.
    try:
        if _wiring_is_stale(switcher):
            clear_wiring(switcher, timeout=_LAUNCH_LOCK_BUDGET_S)
            if not _wiring_present(switcher):
                return True, (
                    "Removed a cloud pin wiring whose proxy was gone — "
                    "sessions fall back to the proxy they had before the pin"
                )
            return False, (
                "A cloud pin wiring points at a proxy that is gone, and it "
                "could not be removed (the config is locked) — re-run "
                "`cswap pin --heal`"
            )
    except Exception as exc:  # noqa: BLE001
        return False, f"Could not heal the cloud pin ({_safe(exc)})"
    return False, "Nothing to heal"


def run(switcher, account: str | None, clear: bool = False, heal_only: bool = False) -> int:
    """Entry point for ``cswap pin``. Mirrors :func:`claude_swap.menubar.run`:
    the optional dependency is resolved here, at call time, not at import."""
    from claude_swap.printer import accent, dimmed, warning

    if heal_only:
        # Deliberately BEFORE _impl(): healing must work when the package is
        # missing or broken, because removing a stale wiring is the half that
        # matters most then. Exit 0 either way — the status line calls this on
        # a timer and a non-zero exit for "nothing was wrong" is noise.
        changed, msg = heal(switcher)
        print(msg if changed else dimmed(msg))
        return 0

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
        # Same rule for the read-only path: a malformed pin file is "no pin I
        # can read", not "the package is broken". The TUI badge already answers
        # None in this exact state, so reporting an error here made the two
        # front ends disagree about one file.
        try:
            current = pin.load_pin(switcher.backup_dir)
        except Exception:  # noqa: BLE001
            current = None
        if current:
            print(f"Cloud account (RC/artifacts): {current[0]}")
        else:
            print(dimmed("No cloud account pinned"))
        return 0

    account_num, email, org_uuid = switcher.resolve_account(account)
    # THE SAME set_pin THE TUI CALLS. This branch carried its own copy of the
    # refusal, the rollback and the no-proxy verdict — and the API-key refusal
    # is the divergence that survived in it after the shared pair was added.
    # num is passed, not re-derived: a duplicate email resolves ambiguously
    # and would skip the API-key refusal (see set_pin).
    ok, msg = set_pin(switcher, email, org_uuid, num=account_num)
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
    # A NOTE MUST NOT FAIL THE ACTION. The pin is already applied and
    # "Pinned…" has already printed; everything below is advice about which
    # sessions need reconnecting. This was the one call into the optional
    # package in `run()` that no `try` covered, so a raise here — from a peer
    # on its own release schedule — turned a SUCCEEDED pin into:
    #
    #     Error: the cloud pin is installed but not usable: …
    #       `cswap pin --clear` still works, and removes the wiring …
    #     Pinned the cloud account (RC/artifacts) to Account-2 (…)
    #     exit 1
    #
    # Exit 1 and advice to `--clear` over a pin that is on disk and working —
    # a user following it destroys it. The TUI's sibling call already guards
    # this (dashboard.py, "a note must not fail the action"); the two front
    # ends disagreeing is the defect this module's header names as its own.
    try:
        open_rc = pin.live_remote_control_sessions()
    except Exception:  # noqa: BLE001 — advice is not the operation
        open_rc = None
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
