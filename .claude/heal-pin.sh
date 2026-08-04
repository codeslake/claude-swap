#!/bin/bash
# Run `cswap pin --heal` on every machine, from a pane this script OWNS.
#
#   heal-pin.sh [--yes]
#
# WHY A SCRIPT AND NOT A COMMAND. `cswap pin --heal` cannot be run over plain
# ssh on macOS: an ssh-born process sits in a different audit session and
# cannot read the login keychain, so the daemon it starts is credential-blind
# and fails open. Measured on via-work-mac, all three paths, same host, minutes
# apart, as sha256[:12] of the credential (never the credential itself):
#
#     plain ssh                 e3b0c44298fc   <- sha256 of ""
#     launchctl asuser <uid>    e3b0c44298fc   <- also nothing
#     tmux window (GUI server)  16d4fa0941ce   <- the real credential
#
# The empty-string hash is the control that makes those distinguishable, and it
# is why the middle row is in this comment: `launchctl asuser` looks like the
# documented answer, FAILS here, and fails SILENTLY — piping into shasum masks
# its exit code, so without the control it reads as a success. Only the tmux
# server works, because that server was started from the GUI and its children
# inherit the audit session.
#
# NEVER SEND KEYS TO A PANE YOU DID NOT CREATE. This is the reason the script
# exists at all. `pane_current_command` reports `zsh` for an IDLE SHELL and for
# a RUNNING CLAUDE CODE SESSION alike — Claude Code runs under zsh, so the
# field cannot separate them, and neither can the window name or the pane
# index. Measured: `tmux send-keys ... Enter` aimed at what looked like the
# idle pane of a cswap-tui window submitted two shell lines as a PROMPT to a
# live Claude Code session, which spent 16 seconds executing them. Interrupting
# it cost that session whatever it had been doing, and nothing in the pane
# listing had said it was busy.
#
#     tmux new-window   -> the window is mine; nobody else is typing in it
#     tmux send-keys    -> a guess about who owns the cursor
#
# So: create a window, let it run the command directly, read the file it wrote,
# kill the window. Carrying the command as the window's argv rather than
# sending keystrokes means there is no prompt to submit to even in principle.
set -uo pipefail
R="${CSWAP_REPO:-$HOME/workspace/cswap/cswap_fork}"
. "$R/.claude/integrated.conf" 2>/dev/null || true
: "${DEPLOY_HOSTS:=lmd42-docker via-work-mac via-personal-mac}"
YES=0; for a in "$@"; do [ "$a" = --yes ] && YES=1; done

# Runs on the target. Prints, per host: the daemon before, what heal said, and
# the daemon after. The before/after pids are the evidence that a heal which
# claims to have restarted something actually did — "Restored the cloud pin"
# describes what the code attempted, not what the process table shows.
remote='
  hn=$(hostname -s)
  TD="$(NO_COLOR=1 uv tool dir 2>/dev/null)/claude-swap"
  PY="$TD/bin/python"; [ -x "$PY" ] || PY="$TD/Scripts/python.exe"
  [ -x "$PY" ] || { echo "$hn: no claude-swap tool env — skip"; exit 0; }

  # Ask the package for the fingerprint the recycle keys off, rather than
  # comparing version strings: a redeployed file with the same version is
  # exactly the stale case, and the version would call it current.
  fp=$("$PY" -c "import cswap_pin.proxy as P;print(P.daemon_fingerprint())" 2>/dev/null)
  before=$(pgrep -f "cswap_pin|pin_proxy" 2>/dev/null | head -1)
  echo "$hn: before pid=${before:-none} code-fp=${fp:-?}"

  case "$(uname -s)" in
    Darwin)
      command -v tmux >/dev/null 2>&1 || { echo "$hn: no tmux — cannot heal safely, SKIPPED"; exit 3; }
      sess=$(tmux list-sessions -F "#{session_name}" 2>/dev/null | head -1)
      [ -n "$sess" ] || { echo "$hn: no tmux session — cannot heal safely, SKIPPED"; exit 3; }
      out="$HOME/.cswap-heal.$$"
      rm -f "$out"
      tmux new-window -d -t "$sess" -n cswap-heal \
        "cswap pin --heal > $out 2>&1; echo rc=\$? >> $out; sleep 1"
      # Poll for the rc line rather than sleeping a fixed span: a heal that
      # respawns a daemon takes a variable time, and a fixed sleep either
      # wastes it or truncates the output into a false "no answer".
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
        grep -q "^rc=" "$out" 2>/dev/null && break
        sleep 2
      done
      tmux kill-window -t "$sess:cswap-heal" 2>/dev/null
      if [ -f "$out" ]; then
        sed "s/^/  $hn heal: /" "$out"
        rm -f "$out"
      else
        echo "  $hn heal: NO OUTPUT (window died before writing) — treat as FAILED"
      fi
      ;;
    *)
      # No audit-session split on Linux; run it directly.
      cswap pin --heal 2>&1 | sed "s/^/  $hn heal: /"
      echo "  $hn heal: rc=${PIPESTATUS[0]}"
      ;;
  esac

  after=$(pgrep -f "cswap_pin|pin_proxy" 2>/dev/null | head -1)
  echo "$hn: after  pid=${after:-none}"
  if [ -n "$after" ] && [ "$after" != "${before:-}" ]; then
    echo "$hn: RECYCLED ${before:-none} -> $after"
  elif [ -n "$after" ]; then
    # Not a failure: 0.1.5 and later self-recycle, so a daemon that is already
    # current has nothing to do. Distinguishable from a recycle, which is the
    # point — collapsing the two is how a no-op reads as a success.
    echo "$hn: unchanged pid $after (already current, or heal declined)"
  else
    echo "$hn: NO DAEMON after heal — the wiring should now be cleared; check \`cswap pin\`"
  fi
'

if [ "$YES" = 0 ]; then
  printf 'run `cswap pin --heal` on: %s  [y/N] ' "$DEPLOY_HOSTS"; read -r a
  case "$a" in y|Y) ;; *) echo aborted; exit 0;; esac
fi

# Ask each host its name rather than hardcoding which is local: the operator
# runs this from whichever machine they are at. And here it is not cosmetic —
# ssh-ing to your own mac is the exact failure this script exists to avoid.
me=$(hostname -s)
rc=0
for h in $DEPLOY_HOSTS; do
  there=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'hostname -s' 2>/dev/null)
  if [ -z "$there" ]; then
    echo "$h: no answer to \`hostname -s\` within 8s — SKIPPED"; rc=1; continue
  fi
  if [ "$there" = "$me" ]; then
    bash -c "$remote" || rc=1
  else
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$h" "$remote" || rc=1
  fi
done
exit $rc
