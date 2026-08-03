#!/bin/bash
# Harness for rebuild.sh's deploy-section rc arithmetic.
#
# Runs the REAL lines, extracted from the file between two stable markers, on
# top of stubbed hosts. Extracting rather than copying is the point: a copy
# would keep passing after someone edits the script it claims to cover.
#
#   ./test_rc.sh <path-to-rebuild.sh>
set -u
SRC="${1:?usage: test_rc.sh <rebuild.sh>}"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT

# The section under test: from the restart decision (or the deploy banner on
# the old layout) through the final verdict.
start=$(grep -n '^# A running process keeps the code\|^# Deploy. Each host runs' "$SRC" | head -1 | cut -d: -f1)
end=$(grep -n '^exit \$rc' "$SRC" | head -1 | cut -d: -f1)
[ -n "$start" ] && [ -n "$end" ] || { echo "EXTRACT FAILED — markers moved in $SRC"; exit 3; }
sed -n "${start},${end}p" "$SRC" > "$W/section.sh"

# The fake host's HOME is laid out so the script's own remote command string
# — `$HOME/workspace/cswap/cswap_fork/.claude/...` — resolves without the stub
# having to know what that string is.
export FAKE_HOME="$W/home"
export FAKE_REPO="$FAKE_HOME/workspace/cswap/cswap_fork"
mkdir -p "$W/bin" "$FAKE_REPO/.claude"

cat > "$W/bin/hostname" <<'EOF'
#!/bin/bash
echo "${FAKE_HOST:-localbox}"
EOF
# ssh: answer `hostname -s` as the host asked for, and EXECUTE anything else
# verbatim with HOME pointed at the fake tree. Running the real command string
# is the whole point — a stub that pattern-matches for "deploy.sh" silently
# drops the `&&` that decides whether verify runs at all, which is one of the
# defects under test.
cat > "$W/bin/ssh" <<'EOF'
#!/bin/bash
h=""; cmd=""
for a in "$@"; do
  case "$a" in
    -o|-*) continue ;;
    BatchMode=*|ConnectTimeout=*) continue ;;
  esac
  if [ -z "$h" ]; then h="$a"; else cmd="$a"; fi
done
case "$cmd" in
  "hostname -s") echo "$h"; exit 0 ;;
esac
HOME="$FAKE_HOME" bash -c "$cmd"
EOF
chmod +x "$W/bin/hostname" "$W/bin/ssh"
export PATH="$W/bin:$PATH"

# Stubs, driven by files so a run can change its mind between passes.
cat > "$FAKE_REPO/.claude/deploy.sh" <<'EOF'
#!/bin/bash
echo "  (deploy ran)"; exit "$(cat "$FAKE_REPO/deploy.rc")"
EOF
# Which answer verify gives is keyed on whether the RESTART has happened, not
# on a call counter: with several hosts the counter made host B's first pass
# answer as though it were host A's second.
cat > "$FAKE_REPO/.claude/verify.sh" <<'EOF'
#!/bin/bash
if [ -e "$FAKE_REPO/restarted" ]; then
  echo "tui-fresh	$(cat "$FAKE_REPO/verify2.word")	after restart"; exit "$(cat "$FAKE_REPO/verify2.rc")"
else
  echo "tui-fresh	$(cat "$FAKE_REPO/verify1.word")	first pass"; exit "$(cat "$FAKE_REPO/verify1.rc")"
fi
EOF
cat > "$FAKE_REPO/.claude/restart-tui.sh" <<'EOF'
#!/bin/bash
echo "  (restart ran)"; touch "$FAKE_REPO/restarted"
exit "$(cat "$FAKE_REPO/restart.rc")"
EOF
chmod +x "$FAKE_REPO/.claude"/*.sh

run() {  # run <deploy.rc> <v1.rc> <v2.rc> <restart.rc> <touched-src?>
  echo "$1" > "$FAKE_REPO/deploy.rc"
  echo "$2" > "$FAKE_REPO/verify1.rc"; echo "$3" > "$FAKE_REPO/verify2.rc"
  echo "$4" > "$FAKE_REPO/restart.rc"
  [ "$2" = 0 ] && echo OK > "$FAKE_REPO/verify1.word" || echo FAIL > "$FAKE_REPO/verify1.word"
  [ "$3" = 0 ] && echo OK > "$FAKE_REPO/verify2.word" || echo FAIL > "$FAKE_REPO/verify2.word"
  rm -f "$FAKE_REPO/restarted"
  # One remote host and one that IS this box, so both branches of the
  # local-vs-ssh fork run in every case.
  TOUCHED="$5" FAKE_HOST=localbox \
    bash -c '
      R="'"$FAKE_REPO"'"; DEPLOY_HOSTS="hostA localbox"; INTEGRATED_BRANCH=integration
      # `g` is the script'"'"'s git wrapper; only the diff call matters here.
      g() { case "$1" in diff) [ "$TOUCHED" = 1 ] && echo src/claude_swap/switcher.py;; esac; }
      . "'"$W/section.sh"'"
    ' 2>&1
  echo "rc=$?"
}

pass=0; fail=0
check() {  # check <name> <expected-rc> <expected-substring> <run-args...>
  local name="$1" want_rc="$2" want_txt="$3"; shift 3
  local out; out=$(run "$@")
  local got_rc; got_rc=$(printf '%s' "$out" | grep -o 'rc=[0-9]*$' | tail -1 | cut -d= -f2)
  if [ "$got_rc" = "$want_rc" ] && printf '%s' "$out" | grep -q "$want_txt"; then
    echo "  PASS  $name"; pass=$((pass+1))
  else
    echo "  FAIL  $name — want rc=$want_rc + '$want_txt', got rc=$got_rc"
    printf '%s\n' "$out" | sed 's/^/          /'
    fail=$((fail+1))
  fi
}

echo "== $SRC =="
# THE DEFECT: deploy exits 2 (files landed, process stale) and the first verify
# FAILS on tui-fresh for that same reason. The restart repairs both and the
# second verify is green. That is a SUCCESSFUL deploy and must exit 0.
check "successful deploy with a restart reports success" 0 "all machines deployed and verified" \
      2 1 0 0 1
# A restart that does not fix it is a real failure.
check "restart that leaves verify red still fails" 1 "SOME MACHINE FAILED" \
      2 1 1 0 1
# No src/ change: no restart, so the first verify is authoritative.
check "no product change, verify red, fails" 1 "SOME MACHINE FAILED" \
      0 1 1 0 0
check "no product change, verify green, succeeds" 0 "all machines deployed and verified" \
      0 0 0 0 0
# A deploy that REFUSES (rc 1) is never routine.
check "deploy refusing (rc 1) fails" 1 "SOME MACHINE FAILED" \
      1 0 0 0 1
# The restart itself failing must not be swallowed by a green re-verify.
check "restart-tui.sh failing fails" 1 "SOME MACHINE FAILED" \
      2 1 0 1 1

echo "  $pass passed, $fail failed"
[ "$fail" = 0 ]
