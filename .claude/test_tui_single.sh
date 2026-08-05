#!/usr/bin/env bash
# Harness for verify.sh's tui-single check. Fakes pgrep/ps on PATH, the same
# convention .claude/test_tui_age.sh uses, so every case is exact.
#
#   ./test_tui_single.sh <verify.sh to test>
set -u
V="${1:?usage: test_tui_single.sh <verify.sh>}"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
mkdir -p "$W/bin"; export FAKE="$W"

# pgrep prints the pids in $FAKE/pids, exits 1 when empty (rc 1 = no matches).
cat > "$W/bin/pgrep" <<'P'
#!/bin/bash
[ -s "$FAKE/pids" ] || exit 1
cat "$FAKE/pids"
P
# ps -o comm= -p <pid> looks the pid up in $FAKE/comm.<pid>.
cat > "$W/bin/ps" <<'P'
#!/bin/bash
for a in "$@"; do case "$prev" in -p) pid=$a ;; esac; prev=$a; done
cat "$FAKE/comm.$pid" 2>/dev/null
P
chmod +x "$W/bin/pgrep" "$W/bin/ps"

pass=0; fail=0
ok() { printf '  \033[32m✓\033[0m %s\n' "$*"; pass=$((pass+1)); }
ng() { printf '  \033[31m✗\033[0m %s\n' "$*"; fail=$((fail+1)); }

# case <name> <expect OK|FAIL> <expect-n> <pid:comm> ...
case_() {
  local name="$1" want="$2" wantn="$3"; shift 3
  : > "$W/pids"
  for spec in "$@"; do
    printf '%s\n' "${spec%%:*}" >> "$W/pids"
    printf '%s\n' "${spec#*:}" > "$W/comm.${spec%%:*}"
  done
  local line
  line=$(PATH="$W/bin:$PATH" bash "$V" 2>/dev/null | grep '^tui-single')
  local got_v got_d
  got_v=$(printf '%s' "$line" | cut -f2)
  got_d=$(printf '%s' "$line" | cut -f3)
  if [ "$got_v" = "$want" ] && printf '%s' "$got_d" | grep -q "^$wantn "; then
    ok "$name -> $got_v ($got_d)"
  else
    ng "$name -> got '$got_v' '$got_d', want $want with $wantn"
  fi
}

echo "── the instrument must be able to say BOTH ──"
case_ "one engine, no wrapper noise"        OK   1 "100:cswap"
case_ "TWO engines — a REAL split brain"    FAIL 2 "100:cswap" "200:cswap"

echo "── and must not count what is not an engine ──"
case_ "engine + the zsh that invoked us"    OK   1 "100:cswap" "200:zsh"
case_ "engine + bash wrapper"               OK   1 "100:cswap" "300:bash"
case_ "engine + LOGIN shell (-zsh)"         OK   1 "100:cswap" "400:-zsh"
case_ "mac spelling: full python path"      OK   1 "100:/Users/x/.local/share/uv/tools/claude-swap/bin/python"
case_ "real split brain BEHIND wrappers"    FAIL 2 "100:cswap" "200:zsh" "300:cswap" "400:bash"
case_ "no engine at all"                    OK   0

printf '\n  passed=%d failed=%d\n' "$pass" "$fail"
[ "$fail" -eq 0 ]
