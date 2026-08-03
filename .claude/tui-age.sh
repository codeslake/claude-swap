#!/bin/bash
# Is the running TUI serving the code that is on disk?
#
# ONE implementation, sourced by both deploy.sh and verify.sh. It was written
# twice before, and the two copies disagreed: verify.sh was fixed to age the
# process against src/ mtime while deploy.sh still compared against HEAD's
# commit date, so the same machine in the same second reported "INCOMPLETE,
# restart required" and "tui-fresh OK" — and an operator reading two answers
# picks the one that lets them stop.
#
#   . <dir>/tui-age.sh
#   tui_age_verdict "$D"   ->  prints "fresh|stale|none|unknown <detail>"
#
# rc is not the answer; the first word is. A caller that needs an exit status
# should branch on the word, because "cannot tell" and "stale" want different
# handling and one rc cannot carry both.

# Newest mtime under a source tree, in whole seconds.
#
# mtime, not HEAD's commit date. The integration branch is REBUILT from
# upstream/main on every run, so it mints a new merge commit with a new date
# each time — even when the rebuild changed only .claude/ and `reset --hard`
# therefore rewrote no source file at all. Measured: a config-only rebuild
# produced HEAD dated 537s after every src/ file's mtime while the TUI had
# started 12s after those files were written and was serving exactly them; the
# commit-date comparison called all three machines stale and failed the deploy.
#
# mtime is the clock the import actually depends on: `git reset --hard` rewrites
# a file only when its content differs, so an untouched module keeps its mtime
# through any number of rebuilds.
#
# Both spellings are tried and the ANSWER is validated, never the exit status.
# `find -printf` is GNU-only — measured on via-work-mac and via-personal-mac it
# exits "unknown primary or operator" — and `stat -f %m` is what answers there.
# GNU stat, meanwhile, reads -f as "filesystem info" and prints
# "Inodes: Total: ..." with rc 0, which then lands in an arithmetic comparison
# that ERRORS and evaluates TRUE, reporting every process fresh forever. So a
# candidate must be all-digits to be accepted.
tui_age_newest_mtime() {
  local out
  out=$(find "$1" -name '*.py' -printf '%T@\n' 2>/dev/null | cut -d. -f1 | sort -rn | head -1)
  case "$out" in ''|*[!0-9]*) ;; *) printf '%s' "$out"; return 0;; esac
  out=$(find "$1" -name '*.py' -exec stat -f '%m' {} + 2>/dev/null | sort -rn | head -1)
  case "$out" in ''|*[!0-9]*) return 1;; *) printf '%s' "$out"; return 0;; esac
}

tui_age_verdict() {
  local D="${1:-$HOME/workspace/cswap/cswap_fork}" pid started pid_epoch src_epoch age rc
  # `pgrep | head` would report HEAD's status, which is 0 whatever pgrep did —
  # so a pgrep that could not run at all read as "no match". Capture pgrep's own
  # rc first, then take the first line. rc 0 = matched, 1 = no match (both real
  # answers); anything else means it could not run, and a check that cannot run
  # is never a pass. BSD pgrep has no -c, and a spelling that exits 2 once
  # turned "cannot count" into "0 running" on both macs.
  pid=$(pgrep -f 'cswap (tui|auto)' 2>/dev/null); rc=$?
  pid=$(printf '%s\n' "$pid" | head -1)
  [ "$rc" -gt 1 ] && { echo "unknown pgrep unusable here (rc=$rc)"; return 0; }
  [ -n "$pid" ] || { echo "none no TUI running (nothing to be stale)"; return 0; }
  src_epoch=$(tui_age_newest_mtime "$D/src/claude_swap") \
    || { echo "unknown cannot read any mtime under $D/src/claude_swap"; return 0; }
  started=$(ps -o lstart= -p "$pid" 2>/dev/null)
  pid_epoch=$(date -d "$started" +%s 2>/dev/null || date -j -f '%a %b %d %T %Y' "$started" +%s 2>/dev/null)
  [ -n "$pid_epoch" ] || { echo "unknown cannot parse the TUI start time ('$started')"; return 0; }
  # `-le`, not `-lt`: ps reports whole seconds, so a process that started in the
  # same second as the write cannot be shown to have imported the new bytes.
  # Ambiguous means stale — the cost is one extra restart, against a machine
  # silently serving replaced code.
  if [ "$pid_epoch" -le "$src_epoch" ]; then
    age=$(( (src_epoch - pid_epoch) / 60 ))
    echo "stale pid $pid started ${age}min before the newest src/ write"
  else
    echo "fresh pid $pid postdates the newest src/ write"
  fi
}
