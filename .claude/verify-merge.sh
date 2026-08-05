#!/usr/bin/env bash
# Post-rebuild check for the merge results git resolves CLEANLY but WRONGLY.
#
#   ./verify-merge.sh [<repo-or-worktree>]     # default: $CSWAP_REPO or the cwd's repo
#
# One line per check: <name>\tOK|FAIL\t<detail>. Exit = number of FAILs.
#
# WHY THIS EXISTS, and why it is not a lint.
#
# Every site below is a place where BOTH sides of a merge touched DIFFERENT
# LINES of the same region, so git emitted no conflict marker, called the merge
# resolved, and produced a tree in which one side's semantic tail is gone. The
# branches are each green alone; only the integration build is broken. Measured
# three times in one session, and all three RECURRED on the very next rebuild —
# which is the whole argument for a check rather than a one-time fix.
#
# EVERY CHECK CARRIES ITS OWN CONTROL. A grep that cannot fail is not a check:
# it reads as protection while proving nothing. So each check below is run
# twice — once against the tree, once against a deliberately broken copy that
# it MUST reject. If the control passes, the check is reported BROKEN, not OK,
# because a check that cannot say NO says nothing when it says YES.
set -u

R="${1:-${CSWAP_REPO:-$(git rev-parse --show-toplevel 2>/dev/null)}}"
[ -n "$R" ] && [ -d "$R" ] || { echo "usage: verify-merge.sh <repo>" >&2; exit 2; }

FAIL=0
row() { printf '%s\t%s\t%s\n' "$1" "$2" "$3"; [ "$2" = FAIL ] && FAIL=$((FAIL + 1)); return 0; }

SW="$R/src/claude_swap/switcher.py"
TA="$R/tests/test_autoswitch.py"
CF="$R/tests/conftest.py"

# A check is a predicate over a FILE, so the same code can be pointed at the
# real tree and at a broken copy. `$1` is the path; prints nothing, returns 0/1.
# The BEHAVIOUR, not the helper. This first named `_read_target_credentials`'s
# `return ""` tail, because that is where #196 put the read. #199 deleted the
# helper and inlined the read at both `_perform_switch` sites, so the shape was
# gone from every branch while the semantics were intact -- and the check then
# reported the regression it exists to catch on a tree that has none.
#
# What must survive is the LANDING: an empty target slot reaches
# `_switch_to_empty_slot` rather than raising. A roster import syncs the account
# LIST and not the credentials, so an empty slot is a destination, and the only
# way to fill one is to be ON it and log in. Mutation-checked: making both arms
# raise instead kills 6 tests; a whitespace NOOP on the same file kills none.
absent_tail()  { python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
n = len(re.findall(
    r"if not target_creds:\n\s+return self\._switch_to_empty_slot\(", s))
sys.exit(0 if n >= 2 else 1)          # both _perform_switch read sites
PY
}
perform_arity() { grep -q 'def perform_after_stop(\*args' "$1"; }
pin_fixture()   { python3 - "$1" <<'PY'
import re, sys
L = open(sys.argv[1], encoding="utf-8").read().splitlines()
for i, l in enumerate(L):
    if re.match(r"def _reset_pin_live_impl_cache\b", l):
        sys.exit(0 if i and "fixture" in L[i - 1] else 1)
sys.exit(1)                           # fixture absent entirely
PY
}
# The CALL, not the word: 3 of the 4 occurrences in switcher.py are comments
# ABOUT it, so a bare `grep -c clear_wiring` still reads >= 1 on a tree where
# only the prose survives and the call is gone -- the exact shape of failure
# this file exists to catch.
clear_wiring()  { grep -qE '_pin[.]clear_wiring[(]' "$1"; }

# Run a predicate against the real file AND against a copy with the very line
# it looks for removed. Both directions, every time.
check() {
  local name="$1" fn="$2" file="$3" break_expr="$4" why="$5"
  if [ ! -f "$file" ]; then
    row "$name" FAIL "missing file: $file"; return
  fi
  local tmp; tmp=$(mktemp); trap 'rm -f "$tmp"' RETURN
  python3 - "$file" "$tmp" "$break_expr" <<'PY'
import re, sys
src, dst, pat = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(src, encoding="utf-8").read()
# Remove EVERY occurrence. `count=1` removed the FIRST match in the file,
# which for a pattern appearing more than once is not the one the check
# reads -- the check then still passed on the "broken" copy and correctly
# reported ITSELF broken. Removing all of them is the honest breaker.
out, n = re.subn(pat, "", text)
if n == 0:
    sys.exit(3)   # the breaker matched nothing: it can prove nothing
open(dst, "w", encoding="utf-8").write(out)
PY
  local breaker_rc=$?
  # ORDER MATTERS. Ask the REAL tree first, then the control.
  #
  # When the thing is already MISSING the breaker has nothing left to remove
  # and exits 3 -- which is not a broken check, it is the regression itself.
  # Reporting "CHECK IS BROKEN" there hides the very failure the check exists
  # to name, and the two read identically to whoever is looking. So: a tree
  # that fails the predicate is a FAIL with its own reason, always; the
  # control only decides whether a PASS can be trusted.
  if ! "$fn" "$file"; then
    row "$name" FAIL "$why"
    return
  fi
  if [ "$breaker_rc" -eq 3 ]; then
    row "$name" FAIL "CHECK IS UNVERIFIABLE: tree passes but the breaker matched nothing, so the pass is unproven"
    return
  fi
  if "$fn" "$tmp"; then
    row "$name" FAIL "CHECK IS BROKEN: it accepts a tree with the thing removed"
    return
  fi
  row "$name" OK "present, and the check rejects its absence"
}

check merge-absent-tail absent_tail "$SW" 'return self\._switch_to_empty_slot\(' \
  'the empty-slot landing is gone: an ABSENT target slot raises instead of reaching _switch_to_empty_slot, so a roster-imported slot is unreachable and the only way to fill it (be on it and log in) is closed (PR #199)'

check merge-perform-arity perform_arity "$TA" 'def perform_after_stop\(\*args' \
  'perform_after_stop takes #199'"'"'s 3-arg signature while the merged _perform takes #204'"'"'s 4 — TypeError at the stop-race test'

check merge-pin-fixture pin_fixture "$CF" '@pytest\.fixture\(autouse=True\)\n(?=def _reset_pin_live_impl_cache)' \
  '_reset_pin_live_impl_cache lost its @pytest.fixture(autouse=True): pin._live_impl_cache is never reset and one test'"'"'s monkeypatched impl leaks into the next (PR #210)'

check merge-clear-wiring clear_wiring "$SW" '_pin[.]clear_wiring[(]' \
  'purge lost its clear_wiring call: purge deletes backup_dir with the pin record and cert dir while .claude.json'"'"'s env block survives pointing every hand-launched claude at a dead port (PR #210)'

exit "$FAIL"
