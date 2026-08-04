#!/bin/bash
# Named liveness probes for the cswap deploy, run on each host after deploy.sh.
# Format, one per line:  <name>\tOK|FAIL\t<detail>
# Exit non-zero if any FAIL.
#
# A check that cannot RUN is a FAIL, never a skip. A probe that quietly reports
# nothing on the host where it does not apply is how a broken machine passes:
# `/proc/net/tcp` returns zero on both macs, so a socket count that "found no
# problems" there had simply never looked.
set -u
CS="$HOME/.local/bin/cswap"
fail=0
say() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; [ "$2" = FAIL ] && fail=1; return 0; }

# 1. The console script runs at all. This is the check that catches a deploy
#    that landed a tree the entry point cannot even import.
if out=$("$CS" --version 2>&1); then say version OK "$out"
else say version FAIL "${out:-no output}"; fi

# 2. The tree the editable install actually serves is the tree we deployed.
#    A matching SHA in the checkout proves the file arrived, not that the
#    installed package points at it — the .pth has pointed at a stale worktree
#    before, and every process check stayed green.
py="$HOME/.local/share/uv/tools/claude-swap/bin/python"
if [ -x "$py" ]; then
  src=$("$py" -c 'import claude_swap,os;print(os.path.dirname(os.path.dirname(claude_swap.__file__)))' 2>&1)
  want="$HOME/workspace/cswap/cswap_fork/src"
  [ "$src" = "$want" ] && say editable-src OK "$src" || say editable-src FAIL "serves $src, deployed $want"
else
  say editable-src FAIL "no interpreter at $py"
fi

# 3. Exactly one auto-switch engine. Two LIVE engines double-fetch and race
#    each other's switch decisions, and the second one is invisible to whoever
#    is watching the first.
#
#    Counted with `pgrep -f | wc -l`, NOT `pgrep -fc`: BSD pgrep has no -c, so
#    on both macs that spelling exits 2 and a `|| echo 0` turns "this check
#    could not run" into "OK, 0 running". Measured: it reported 0 on a host
#    with a live TUI, i.e. the split-brain detector could never fire there.
#    rc 0 = matches, rc 1 = none (both fine), anything else = cannot run.
#
#    AND THEN DROP THE SHELLS. `pgrep -f` matches the WHOLE command line, so a
#    wrapper whose argv merely MENTIONS the TUI counts as an engine — including
#    the shell that invoked this script. Measured on lambda-docker,
#    deterministic across three runs, with exactly one live TUI:
#
#      matched 2711454  comm cswap   argv `… cswap tui --auto`      ENGINE
#      matched 2793975  comm zsh     argv `… bash .claude/verify.sh …`
#      verdict "2 engines — split brain", truth 1
#
#    The instrument was counting its own caller, and that is worse than a wrong
#    number: a FAIL nobody can act on trains everyone to ignore the line, and a
#    REAL split brain (two engines racing the state lock) prints exactly those
#    same words. `comm` separates them on every host we run — the engine reads
#    `cswap` on Linux and `…/bin/python` on the macs, never a shell name.
pids=$(pgrep -f 'cswap (tui|auto)' 2>/dev/null); rc=$?
if [ "$rc" -gt 1 ]; then
  say tui-single FAIL "pgrep unusable here (rc=$rc) — cannot count engines"
else
  n=0
  for p in $pids; do
    # basename, and strip the leading `-` a login shell carries.
    c=$(ps -o comm= -p "$p" 2>/dev/null); c=${c##*/}; c=${c#-}
    case "$c" in (sh|bash|zsh|dash|ksh|fish|csh|tcsh) continue ;; esac
    n=$((n + 1))
  done
  [ "$n" -le 1 ] && say tui-single OK "$n running" || say tui-single FAIL "$n engines — split brain"
fi

# 4. The pin, only where one is set. `cswap pin` with no argument reports the
#    current pin and must not traceback even when the optional package is
#    absent — that is the one command whose job is to work when the pin does
#    not. Do NOT import cswap_pin.proxy to test this: it calls require("oauth")
#    at module scope and raises by design.
if out=$("$CS" pin 2>&1 | head -3 | tr '\n' ' '); then say pin OK "$out"
else say pin FAIL "$out"; fi

# 4b. THE PIN IS RECORDED vs THE PIN IS SERVING. Check 4 reads the pin FILE and
#     nothing else, so its line is byte-identical on a machine whose proxy is
#     running and on one whose proxy is dead — measured on all three hosts at
#     once, the same "Cloud account (RC/artifacts): <email>" while one of them
#     had an EMPTY env block and no daemon. Sessions there go out UNPINNED,
#     which is the designed degrade and not an outage, but a deploy report that
#     cannot say it happened is the problem: three identical OK lines described
#     two different worlds.
#
#     Asks the wiring, which is what a session actually reads: `env` in
#     ~/.claude.json plus the `_cswapPinWiredKeys` receipt cswap writes beside
#     it. Both must agree. A recorded pin with no wiring is REPORTED, not
#     failed — an unpinned machine still runs, and a check that fails here
#     would block every deploy to a host whose daemon is being repaired.
wire=$("$py" - <<'PYEOF' 2>&1
import json, os
try:
    d = json.load(open(os.path.expanduser("~/.claude.json")))
except Exception as e:
    print("UNREADABLE", e); raise SystemExit
env = d.get("env") or {}
keys = d.get("_cswapPinWiredKeys")
proxy = env.get("HTTPS_PROXY", "")
port = ""
if "127.0.0.1:" in proxy:
    port = proxy.rsplit(":", 1)[-1]
if keys and proxy:
    print("WIRED", port)
elif keys or proxy:
    # One half without the other: a teardown that stopped halfway leaves the
    # config naming a dead port, which is worse than unpinned because every
    # request dials it.
    print("HALF", "keys=%s proxy=%s" % (bool(keys), bool(proxy)))
else:
    print("UNWIRED")
PYEOF
)
case "$wire" in
  WIRED*)      say pin-wired OK "serving on 127.0.0.1:${wire#WIRED }" ;;
  UNWIRED)     say pin-wired OK "UNPINNED — no wiring, no daemon; sessions go out direct" ;;
  HALF*)       say pin-wired FAIL "half-wired, requests dial a port nothing serves — ${wire#HALF }" ;;
  UNREADABLE*) say pin-wired FAIL "~/.claude.json ${wire#UNREADABLE }" ;;
  *)           say pin-wired FAIL "could not determine wiring (got: ${wire:-no output})" ;;
esac

# 5. The TUI's pin surface, not just the CLI's. Check 4 probes `cswap pin` and
#    the daemon; BOTH stayed green through a cutover that deleted every pin
#    feature from the TUI (measured: tui/ went from 29 cloud-refs and 8 pin
#    symbols to zero, on all three machines, and nothing here noticed). A check
#    that exercises one surface reports the other as healthy.
#
#    Asserted against the IMPORTED modules, not the checkout, because the
#    editable install can serve a different tree than `git` describes.
#    PRESENT and WORKING are different claims. This check used to count the
#    substring "cloud" in the sources — which goes green over a function that
#    raises the moment it is entered. Measured on the pin branch: a stray
#    `del impl` left _pin_entries raising UnboundLocalError on every successful
#    call, while every guard asserting the surface EXISTS stayed green. So it
#    now CALLS the pin surface and reads what comes back.
tui=$("$py" - <<'PYEOF' 2>&1
import importlib, inspect
n = 0
for m in ("dashboard", "widgets", "autoview"):
    try:
        mod = importlib.import_module("claude_swap.tui." + m)
    except Exception as exc:
        print("IMPORTFAIL %s: %s" % (m, exc)); raise SystemExit
    n += open(mod.__file__).read().lower().count("cloud")
if n == 0:
    print(0); raise SystemExit
# Call it. Bound methods need an instance we cannot build here, so the callable
# is exercised through its underlying function with a stand-in self — enough to
# execute the body, which is where UnboundLocalError lives.
dash = importlib.import_module("claude_swap.tui.dashboard")
fn = None
for cls in vars(dash).values():
    f = getattr(cls, "_pin_entries", None)
    if callable(f):
        fn = f; break
if fn is None:
    print("NOCALLABLE _pin_entries"); raise SystemExit
# The stub must let the body RUN, not die on the first attribute. A stub that
# raises AttributeError early masks exactly the bug this check exists for:
# measured, `del impl` injected into _pin_entries was swallowed as "other error"
# by a lambda-returning stub, so the check passed over the real defect.
class _Any:
    app = None
    def __getattr__(self, k): return _Any()
    def __call__(self, *a, **kw): return None
    def __iter__(self): return iter(())
    def __bool__(self): return False
stub = _Any(); stub.app = _Any()
try:
    fn(stub)
except (NameError, UnboundLocalError) as exc:
    print("RAISES %s: %s" % (type(exc).__name__, exc)); raise SystemExit
except Exception:
    pass  # anything else is the stub's shape, not the code under test
print(n)
PYEOF
)
case "$tui" in
  IMPORTFAIL*)   say tui-pin FAIL "$tui" ;;
  RAISES*)       say tui-pin FAIL "the pin surface EXISTS but RAISES when called — $tui" ;;
  NOCALLABLE*)   say tui-pin FAIL "$tui" ;;
  ''|*[!0-9]*)   say tui-pin FAIL "could not count (got: ${tui:-no output})" ;;
  0)             say tui-pin FAIL "the TUI has NO pin surface — CLI works, dashboard/badge gone" ;;
  *)             say tui-pin OK "$tui cloud refs across dashboard/widgets/autoview" ;;
esac

# 6. Is the RUNNING TUI on the code that is on disk, or on whatever it exec-ed?
#    An editable install means the pull IS the deploy for the FILES and is NOT
#    the deploy for any running process. Measured: after the pin cutover all
#    three TUIs were still serving code from 16 hours before the commit, the
#    operator could see a badge the deployed tree no longer contains, and every
#    other check here went green. The screen was the only honest witness.
#
#    The verdict comes from tui-age.sh, shared with deploy.sh. It used to be
#    implemented here AND there, and the two copies drifted apart — same
#    machine, same second, "INCOMPLETE: restart required" from one and
#    "tui-fresh OK" from the other.
#
#    Resolve the repo, do not infer it from $0: this script gets copied
#    elsewhere to run (measured: from /tmp, dirname $0/.. was "/" and the check
#    reported "cannot read the deployed commit date" on all three machines).
D="${CSWAP_REPO:-$HOME/workspace/cswap/cswap_fork}"
. "$(dirname "${BASH_SOURCE[0]}")/tui-age.sh"
verdict=$(tui_age_verdict "$D"); detail="${verdict#* }"
case "${verdict%% *}" in
  fresh)   say tui-fresh OK   "$detail" ;;
  none)    say tui-fresh OK   "$detail" ;;
  stale)   say tui-fresh FAIL "$detail — it serves pre-deploy code; RESTART REQUIRED" ;;
  # A check that cannot RUN is a FAIL, never a skip.
  *)       say tui-fresh FAIL "$detail" ;;
esac

exit $fail
