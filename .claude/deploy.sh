#!/bin/bash
# Put THIS machine on origin/integration. Run by /watch-pr-and-manage-integrated
# on each host in DEPLOY_HOSTS.
#
# This checkout IS the running cswap: the editable install's .pth points at its
# src/. So the update has two constraints that pull opposite ways —
#
#   the rebuild FORCE-PUSHES integration (it is rebuilt from upstream/main every
#   time), so the deployed HEAD is routinely not an ancestor and `merge --ff-only`
#   would refuse every rebuild that dropped a merged PR;
#
#   but `reset --hard` on a development tree silently discards whatever a session
#   left in it.
#
# Resolved by resetting, gated on the tree being CLEAN. A dirty live tree is a
# thing a human looks at; it is never something a deploy resolves by discarding.
#
# It also never checks the branch out. The main checkout stays on `integration`
# permanently; a checkout here has removed pin_proxy.py from all three hosts
# three times, while the running daemon kept serving the code it booted with,
# so every process check said healthy over a gutted build.
set -eu
# Overridable so the script can be exercised against a scratch tree. A deploy
# path testable only on the live install is one that gets tested in
# production, which is how it shipped an INCOMPLETE state as a success line.
D="${CSWAP_REPO:-$HOME/workspace/cswap/cswap_fork}"

branch=$(git -C "$D" rev-parse --abbrev-ref HEAD)
if [ "$branch" != integration ]; then
  echo "REFUSE: $(hostname -s):$D is on '$branch', not integration — a human decides this"
  exit 1
fi

# Only TRACKED modifications block. Measured: `reset --hard` destroys a
# modified tracked file and leaves an untracked one exactly where it is, so
# refusing over untracked files protects nothing the deploy could harm — it
# just means one stray file blocks every deploy until someone clears it, which
# is how a guard stops being read. They are reported below instead.
dirty=$(git -C "$D" status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
  echo "REFUSE: $(hostname -s):$D has uncommitted changes to TRACKED files — a human decides this, not a deploy:"
  printf '%s\n' "$dirty" | head -10 | sed 's/^/    /'
  exit 1
fi

git -C "$D" fetch origin integration --quiet
git -C "$D" reset --hard origin/integration >/dev/null

# The reset is not proof the FILES moved. `git update-ref` (which the rebuild
# uses, to avoid checking out the live install) advances the branch without
# touching the index or the worktree, so a tree can sit on the right HEAD with
# the wrong contents — and `git status` calls that "modified", which the dirty
# gate above then reports as somebody's uncommitted work rather than as drift.
#
# Measured: after a rebuild, lambda-docker had HEAD 0657d05 whose commit
# carries 11 cloud refs in tui/widgets.py while the file on disk had 0. The
# deploy reported success and the pin badge was gone.
drift=$(git -C "$D" diff --stat HEAD -- src/ tests/ | tail -1)
if [ -n "$drift" ]; then
  echo "REFUSE: $(hostname -s):$D is on $(git -C "$D" rev-parse --short HEAD) but its FILES do not match it:"
  echo "    $drift"
  echo "    the branch moved without the worktree — re-run after: git -C $D reset --hard HEAD"
  exit 1
fi
echo "$(hostname -s) @ $(git -C "$D" rev-parse --short HEAD)"

# The editable install serves this tree, so the FILES are current the moment the
# reset lands. The processes are not: a running `cswap tui --auto` keeps the
# modules it imported at startup. Say so; do not restart it from here —
# restarting a mac TUI over ssh strands it without keychain access (rc=36) until
# a GUI approval exists, and two live engines race each other's switches.
# Untracked files survive the reset, so they are stale litter rather than a
# hazard — but litter that reads as live code is what made the cutover
# confusing (a deleted pin_proxy.py sat here still importable).
stray=$(git -C "$D" ls-files --others --exclude-standard | head -5)
[ -n "$stray" ] && printf 'NOTE: %s has untracked files the deploy left alone:\n%s\n' \
  "$(hostname -s)" "$(printf '%s\n' "$stray" | sed 's/^/    /')"

# A running process keeps the code it exec-ed, so the pull is the deploy for the
# FILES and NOT for the TUI. Reported as INCOMPLETE with a non-zero exit, not as
# a note: I reported this deploy "LIVE on all three machines" while every TUI
# was serving code from 16 hours earlier, and the operator could see on screen a
# feature the deployed tree no longer contained. A note is walked past; a
# non-zero exit is not.
#
# The window is resolved, never assumed — it is NOT dotfiles:cswap-tui
# everywhere (via-personal-mac keeps its TUI in NeuRig:3.0). A machine whose
# TUI nobody can locate can never leave the stale state.
# Only a STALE TUI is incomplete. A pid that already carries the deployed code
# has it, so the deploy IS done — saying "restart required" there would train
# the operator to ignore the line that matters. The two answers must be
# distinguishable, or "required" and "confirmed" collapse.
#
# The verdict comes from tui-age.sh, the same file verify.sh reads. This was
# implemented separately in both, and they drifted: verify.sh was fixed to age
# the process against src/ mtime while this still compared against HEAD's
# commit date, so one deploy printed "INCOMPLETE, restart required" and
# "tui-fresh OK" about the same pid in the same second. An operator handed two
# answers takes the one that lets them stop.
. "$(dirname "$0")/tui-age.sh"
verdict=$(tui_age_verdict "$D"); detail="${verdict#* }"
case "${verdict%% *}" in
  fresh|none)
    echo "$(hostname -s): $detail — restart confirmed"
    ;;
  *)
    # stale, or a verdict this host could not reach. Both are INCOMPLETE: an
    # unanswerable question is not a pass, and the operator needs to look.
    tui_pid=$(pgrep -f 'cswap (tui|auto)' 2>/dev/null | head -1)
    win=$(tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{pane_pid}' 2>/dev/null \
          | awk -v p="$tui_pid" '$2==p || $2==p-1 {print $1; exit}')
    [ -n "$win" ] || win=$(tmux list-panes -a -F '#{session_name}:#{window_index}.#{pane_index} #{window_name}' 2>/dev/null \
          | awk '/cswap-tui/ {print $1; exit}')
    echo "INCOMPLETE: $(hostname -s) TUI — $detail"
    echo "    restart it in: ${win:-<no tmux window found — locate it before calling this deployed>}"
    echo "    the deploy is not done until that pid postdates the newest src/ write"
    exit 2   # distinct from the REFUSE codes: the files landed, the process did not
    ;;
esac
