#!/bin/bash
# Restart the cswap TUI on every machine, in its OWN cswap-tui window.
#
# The pull is the deploy for the FILES; a running TUI keeps the code it exec-ed.
# Measured: after the pin cutover all three TUIs served code from 16h before the
# commit and every check went green, because none of them looked at the process.
#
# The window is FOUND, never assumed — it is not `dotfiles` everywhere
# (via-personal-mac keeps its TUI in NeuRig). A machine whose window cannot be
# located is reported, not guessed at.
#
#   restart-tui.sh [--yes]
set -uo pipefail
R="${CSWAP_REPO:-$HOME/workspace/cswap/cswap_fork}"
. "$R/.claude/integrated.conf" 2>/dev/null || true
: "${DEPLOY_HOSTS:=lmd42-docker via-work-mac via-personal-mac}"
YES=0; for a in "$@"; do [ "$a" = --yes ] && YES=1; done

# Runs on the target host. Finds the pane by WINDOW NAME (cswap-tui), kills the
# process and re-runs it in that same pane — never creates a window, because a
# TUI in an unexpected window is worse than none: it drives account switches
# where nobody is looking.
remote='
  D="$HOME/workspace/cswap/cswap_fork"
  win=$(tmux list-panes -a -F "#{session_name}:#{window_index}.#{pane_index} #{window_name}" 2>/dev/null \
        | awk "/[[:space:]]cswap-tui\$/ {print \$1; exit}")
  if [ -z "$win" ]; then
    echo "$(hostname -s): NO cswap-tui WINDOW — cannot restart; locate it first"; exit 1
  fi
  # Resolve the TUI from the PANE, never from a global pgrep. Two reasons,
  # both measured on lambda-docker tonight:
  #
  # 1. SELF-MATCH. `pgrep -f "cswap (tui|auto)"` matches any process whose
  #    ARGV contains that string — including the shell running THIS script,
  #    because the script text below contains the literal `cswap tui --auto`.
  #    The local branch runs as `bash -c "$remote"`, so argv carries the whole
  #    body: the first match was the shell itself, `kill` killed it, and the
  #    run died with "Terminated" while the TUI kept serving pre-deploy code.
  #    Over ssh the same command survived, which is why this looked like a
  #    host-specific flake for two deploys.
  # 2. DUPLICATES. `| head -1` is a coin flip the moment a second TUI exists —
  #    and verify.sh has a tui-single check precisely because that happens.
  #
  # The pane IS the identity here: there is no listening socket to ask, and
  # the window is where the process was deliberately placed. Descend from the
  # pane pid so a shell wrapper cannot hide the real process.
  tui_in_pane() {
    ppid=$(tmux display -p -t "$1" "#{pane_pid}" 2>/dev/null) || return 1
    [ -n "$ppid" ] || return 1
    for p in $ppid $(pgrep -P "$ppid" 2>/dev/null); do
      case "$(ps -o args= -p "$p" 2>/dev/null)" in
        *cswap*tui*|*cswap*auto*) echo "$p"; return 0 ;;
      esac
    done
    return 1
  }
  old=$(tui_in_pane "$win")
  tmux send-keys -t "$win" C-c 2>/dev/null; sleep 1
  [ -n "$old" ] && kill "$old" 2>/dev/null; sleep 1
  tmux send-keys -t "$win" "cswap tui --auto" Enter
  sleep 6
  new=$(tui_in_pane "$win")
  commit=$(git -C "$D" log -1 --format=%ct HEAD)
  if [ -z "$new" ]; then
    echo "$(hostname -s): RESTART FAILED — no TUI running in $win"; exit 1
  fi
  started=$(ps -o lstart= -p "$new")
  pe=$(date -d "$started" +%s 2>/dev/null || date -j -f "%a %b %d %T %Y" "$started" +%s 2>/dev/null)
  if [ -n "$pe" ] && [ "$pe" -ge "$commit" ]; then
    echo "$(hostname -s): OK  $win  pid $old -> $new, now postdates the deployed commit"
  else
    echo "$(hostname -s): STILL STALE  $win  pid $new does not postdate the commit"; exit 1
  fi
'

if [ "$YES" = 0 ]; then
  printf 'restart the TUI on: %s  [y/N] ' "$DEPLOY_HOSTS"; read -r a
  case "$a" in y|Y) ;; *) echo aborted; exit 0;; esac
fi

me=$(hostname -s); rc=0
for h in $DEPLOY_HOSTS; do
  there=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'hostname -s' 2>/dev/null)
  if [ -z "$there" ]; then echo "$h: UNREACHABLE (no answer — re-probe before calling it down)"; rc=1; continue; fi
  if [ "$there" = "$me" ]; then bash -c "$remote" || rc=1
  else ssh -o BatchMode=yes -o ConnectTimeout=40 "$h" "$remote" || rc=1; fi
done
exit $rc
