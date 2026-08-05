#!/bin/bash
# Harness for tui-age.sh, the shared "is the running TUI serving the code on
# disk" verdict — and for the two callers agreeing about it.
#
#   ./test_tui_age.sh <dir containing tui-age.sh, deploy.sh, verify.sh>
#
# The two clocks the check confuses (commit date vs file mtime) are set
# independently here, which is the whole point.
set -u
DIR="${1:?usage: test_tui_age.sh <dir>}"
[ -f "$DIR/tui-age.sh" ] || { echo "no tui-age.sh in $DIR"; exit 3; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/bin" "$W/repo/src/claude_swap/tui"

cat > "$W/bin/pgrep" <<'EOF'
#!/bin/bash
rc=$(cat "$FAKE/pgrep_rc" 2>/dev/null || echo 1)
[ "$rc" = 0 ] || exit "$rc"
cat "$FAKE/pid"
EOF
cat > "$W/bin/ps" <<'EOF'
#!/bin/bash
date -d "@$(cat "$FAKE/pid_epoch")" '+%a %b %d %T %Y'
EOF
chmod +x "$W/bin/pgrep" "$W/bin/ps"
export PATH="$W/bin:$PATH" FAKE="$W"

git -C "$W/repo" init -q -b integration 2>/dev/null
git -C "$W/repo" symbolic-ref HEAD refs/heads/integration 2>/dev/null
# deploy.sh fetches `origin` before it ever reaches the TUI question, so a
# scratch repo with no remote dies at rc 128 and the agreement cases silently
# measure nothing. Point origin at the repo itself.
git -C "$W/repo" remote add origin "$W/repo" 2>/dev/null
: > "$W/repo/src/claude_swap/switcher.py"
: > "$W/repo/src/claude_swap/tui/dashboard.py"
git -C "$W/repo" add -A 2>/dev/null

setup() {  # setup <commit-epoch> <newest-src-mtime> <pid-epoch|"">
  GIT_AUTHOR_DATE="@$1 +0000" GIT_COMMITTER_DATE="@$1 +0000" \
    git -C "$W/repo" commit -q --allow-empty -m build 2>/dev/null
  touch -d "@$2" "$W/repo/src/claude_swap/tui/dashboard.py" 2>/dev/null
  touch -d "@$(( $2 - 100 ))" "$W/repo/src/claude_swap/switcher.py" 2>/dev/null
  if [ -n "$3" ]; then echo 0 > "$W/pgrep_rc"; echo 4242 > "$W/pid"; echo "$3" > "$W/pid_epoch"
  else echo 1 > "$W/pgrep_rc"; : > "$W/pid"; echo 0 > "$W/pid_epoch"; fi
}

verdict() { bash -c '. "'"$DIR/tui-age.sh"'"; tui_age_verdict "'"$W/repo"'"'; }

pass=0; fail=0
check() {  # check <name> <want-word> <commit> <mtime> <pid>
  local name="$1" want="$2"; shift 2
  setup "$@"
  local out; out=$(verdict); local got="${out%% *}"
  if [ "$got" = "$want" ]; then echo "  PASS  $name"; pass=$((pass+1))
  else echo "  FAIL  $name — want '$want', got '$out'"; fail=$((fail+1)); fi
}

T=1785000000
echo "== $DIR/tui-age.sh =="
# THE DEFECT this replaced. A rebuild that changed only .claude/ still mints a
# fresh merge commit, so HEAD's date jumps while every src/ file keeps its old
# mtime. The TUI booted after those files were written and is serving exactly
# them; a commit-date comparison called it stale and failed the deploy.
# Measured on a real rebuild: commit at +537, dashboard.py at 0, TUI at +12.
check "config-only rebuild leaves a current TUI fresh"  fresh   $((T+537)) $T        $((T+12))
check "src rewritten under a running TUI is stale"      stale   $((T+537)) $((T+400)) $((T+12))
check "TUI started after the newest src write is fresh" fresh   $((T+537)) $((T+400)) $((T+500))
check "no TUI running is 'none', not stale"             none    $((T+537)) $T        ""
# ps reports whole seconds, so same-second cannot be shown to have imported the
# new bytes. Ambiguous is stale.
check "same-second start is stale, not a coin flip"     stale   $((T+537)) $((T+400)) $((T+400))
# A question that cannot be answered is never a pass.
setup $((T+537)) $((T+400)) $((T+12)); echo 2 > "$W/pgrep_rc"
out=$(verdict); [ "${out%% *}" = unknown ] \
  && { echo "  PASS  pgrep that cannot run is 'unknown', not 'none'"; pass=$((pass+1)); } \
  || { echo "  FAIL  pgrep that cannot run — got '$out'"; fail=$((fail+1)); }
mv "$W/repo/src" "$W/repo/src.hidden"
check "an unreadable src/ is 'unknown', not fresh"      unknown $((T+537)) $((T+400)) $((T+12))
mv "$W/repo/src.hidden" "$W/repo/src"

# BOTH mtime spellings, because the two macs can only reach the second one.
# `find -printf` is GNU; measured on via-work-mac and via-personal-mac it exits
# "unknown primary or operator". A harness that runs only on the linux host
# proves nothing about two thirds of the fleet.
cat > "$W/bin/find" <<'EOF'
#!/bin/bash
for a in "$@"; do [ "$a" = "-printf" ] && { echo "find: -printf: unknown primary or operator" >&2; exit 1; }; done
exec /usr/bin/find "$@"
EOF
cat > "$W/bin/stat" <<'EOF'
#!/bin/bash
[ "${1:-}" = "-f" ] && [ "${2:-}" = "%m" ] || exec /usr/bin/stat "$@"
shift 2
for f in "$@"; do /usr/bin/stat -c %Y "$f"; done
EOF
chmod +x "$W/bin/find" "$W/bin/stat"
echo "  -- the macs' path: GNU find -printf unavailable --"
check "BSD stat: config-only rebuild stays fresh"       fresh   $((T+537)) $T        $((T+12))
check "BSD stat: rewritten src is stale"                stale   $((T+537)) $((T+400)) $((T+12))
check "BSD stat: later start is fresh"                  fresh   $((T+537)) $((T+400)) $((T+500))
# GNU stat reads -f as "filesystem info" and prints "Inodes: Total: ..." with
# rc 0. Believed, it lands in `[ 500 -le Inodes: ... ]`, which errors and
# evaluates TRUE — every process fresh forever. The ANSWER must be validated,
# not the exit status.
rm -f "$W/bin/stat"
check "a non-numeric stat answer is refused"            unknown $((T+537)) $((T+400)) $((T+500))
# Same guard on the FIRST path, which is the one all three machines take today.
cat > "$W/bin/find" <<'EOF'
#!/bin/bash
for a in "$@"; do [ "$a" = "-printf" ] && { echo "not-an-epoch"; exit 0; }; done
exec /usr/bin/find "$@"
EOF
cat > "$W/bin/stat" <<'EOF'
#!/bin/bash
exit 1
EOF
chmod +x "$W/bin/find" "$W/bin/stat"
check "a non-numeric find answer is refused"            unknown $((T+537)) $((T+400)) $((T+500))
rm -f "$W/bin/find" "$W/bin/stat"

# THE REASON THIS FILE IS SHARED. deploy.sh and verify.sh each had their own
# copy and they drifted: verify.sh was fixed to age against src/ mtime while
# deploy.sh still compared against HEAD's commit date, so the same machine in
# the same second reported "INCOMPLETE: restart required" and "tui-fresh OK".
# An operator handed two answers takes the one that lets them stop.
if [ -f "$DIR/deploy.sh" ] && [ -f "$DIR/verify.sh" ]; then
  echo "  -- deploy.sh and verify.sh must not disagree --"
  for c in "fresh $((T+537)) $T $((T+12))" "stale $((T+537)) $((T+400)) $((T+12))"; do
    set -- $c; want=$1; setup "$2" "$3" "$4"
    d_out=$(cd "$W/repo" && CSWAP_REPO="$W/repo" bash "$DIR/deploy.sh" 2>&1); d_rc=$?
    v_out=$(CSWAP_REPO="$W/repo" bash "$DIR/verify.sh" 2>&1)
    v_line=$(printf '%s' "$v_out" | grep '^tui-fresh')
    # deploy: rc 2 == INCOMPLETE.  verify: the tui-fresh line says OK or FAIL.
    d_says=fresh; [ "$d_rc" = 2 ] && d_says=stale
    v_says=fresh; printf '%s' "$v_line" | grep -q 'FAIL' && v_says=stale
    # deploy.sh REFUSES on a non-integration branch before it ever reaches the
    # TUI question, so a scratch repo can only exercise this when it gets that
    # far. Skip loudly rather than pretend the case ran.
    if printf '%s' "$d_out" | grep -q '^REFUSE'; then
      echo "  SKIP  agreement on '$want' — deploy.sh refused first: $(printf '%s' "$d_out" | head -1)"
      continue
    fi
    if [ "$d_says" = "$want" ] && [ "$v_says" = "$want" ]; then
      echo "  PASS  both callers say '$want'"; pass=$((pass+1))
    else
      echo "  FAIL  callers disagree on '$want' — deploy says '$d_says' (rc=$d_rc), verify says '$v_says'"
      fail=$((fail+1))
    fi
  done
fi

echo "  $pass passed, $fail failed"
[ "$fail" = 0 ]
