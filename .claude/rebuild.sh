#!/bin/bash
# Rebuild the integration branch from its inputs, then deploy and verify every
# machine. The whole of the skill's Step 3/4/5, as code.
#
# WHY THIS IS A SCRIPT AND NOT A PROCEDURE IN SKILL.md: the procedure was in
# SKILL.md, and the session that wrote it still stopped mid-way to ask whether
# to rebuild. Prose does not bind. A refusal that is a `exit 1` binds; a
# refusal that is a paragraph is a suggestion.
#
#   rebuild.sh [--deploy] [--yes]
#     (default)   rebuild + suite, report, change nothing published
#     --deploy    also push and deploy every host in DEPLOY_HOSTS
#     --yes       skip the confirmation --deploy asks for
#
# It REFUSES rather than improvises. Every gate below is a state someone has
# actually shipped past.
set -uo pipefail

R="${CSWAP_REPO:-$HOME/workspace/cswap/cswap_fork}"
. "$R/.claude/integrated.conf" || { echo "no integrated.conf"; exit 2; }
: "${UPSTREAM_REMOTE:=upstream}"; : "${UPSTREAM_BRANCH:=main}"; : "${FORK_REMOTE:=origin}"
: "${PEER_OWNED_PRS:=}"; : "${PEER_OWNED_CONTACT:=a peer session}"
export GH_HOST=github.com
g() { git -C "$R" "$@"; }
die() { echo "REFUSE: $*"; exit 1; }

DEPLOY=0; YES=0; CHECK=0
for a in "$@"; do case "$a" in --deploy) DEPLOY=1;; --yes) YES=1;; --check-only) CHECK=1;; esac; done

# --check-only is what the skill's Stop hook runs: it says, every single time
# the turn ends, whether integration is behind its inputs. It never builds,
# pushes or deploys. It exists because the session that WROTE the rebuild
# procedure still ended a turn asking whether to run it — prose does not bind,
# and a line printed on every turn-end is harder to walk past than a paragraph.
if [ "$CHECK" = 1 ]; then
  hold=$(git -C "$R" show "$FORK_REMOTE/${INTEGRATED_BRANCH}-config:.claude/HOLD" 2>/dev/null)
  if [ -n "$hold" ]; then
    echo "[integration] HOLD: $hold"; exit 0
  fi
  git -C "$R" fetch "$FORK_REMOTE" --quiet 2>/dev/null
  git -C "$R" fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" --quiet 2>/dev/null
  behind=""
  up=$(git -C "$R" rev-list --count "$INTEGRATED_BRANCH..$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" 2>/dev/null || echo 0)
  [ "${up:-0}" -gt 0 ] && behind="upstream+$up"
  for n in $(gh pr list --repo "$UPSTREAM_REPO" --author "${PR_AUTHOR:-codeslake}" \
               --state open --json number --jq '.[].number' 2>/dev/null); do
    b=$(cat "$R/.claude/.watch-state/pr-$n.branch" 2>/dev/null) || continue
    [ -n "$b" ] || continue
    c=$(git -C "$R" rev-list --count "$INTEGRATED_BRANCH..$FORK_REMOTE/$b" 2>/dev/null || echo 0)
    [ "${c:-0}" -gt 0 ] && behind="$behind #$n+$c"
  done
  [ -n "$behind" ] && echo "[integration] BEHIND its inputs:$behind — run .claude/rebuild.sh --deploy"
  exit 0
fi

# The PRs to merge. Discovered, never hardcoded: a PR upstream merged must drop
# out of the rebuild or its change applies twice.
PRS=$(gh pr list --repo "$UPSTREAM_REPO" --author "${PR_AUTHOR:-codeslake}" \
        --state open --json number --jq '.[].number' | sort -n) \
  || die "cannot list PRs (gh unauthenticated?)"
[ -n "$PRS" ] || die "no open PRs — nothing to build"

g fetch "$UPSTREAM_REMOTE" "$UPSTREAM_BRANCH" --quiet || die "upstream fetch failed"
g fetch "$FORK_REMOTE" --quiet || die "fork fetch failed"

# GATE 0 — the fork's OWN main must mirror upstream. Nothing in this loop reads
# it, which is exactly why it rots unnoticed: we rebase PR branches and rebuild
# integration, and never touch <fork>/main.
#
# EXTERNAL tools do read it. Measured on CCF: their fork main sat 13 commits
# behind for two weeks, Codex's reviewer resolved its base with
# detectDefaultBranch() -> main, and 4 of 6 findings came back against files the
# PR never touched — they only looked new because they did not EXIST at that
# stale base. Same trap for `gh pr create`, `git diff main...`, IDE compare.
#
# Fast-forward only, gated on ahead==0, and REFUSE otherwise. A fork main
# carrying commits of its own is a repo doing something this script does not
# model, and force-pushing there destroys them. Rot is ahead==0; that is the
# case worth automating. Pushed as <remote-ref>:<branch> so the live tree is
# never checked out — same reason the config branch is built with plumbing.
fm_behind=$(g rev-list --count "$FORK_REMOTE/$UPSTREAM_BRANCH..$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" 2>/dev/null || echo 0)
fm_ahead=$(g rev-list --count "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH..$FORK_REMOTE/$UPSTREAM_BRANCH" 2>/dev/null || echo 0)
if [ "${fm_ahead:-0}" -gt 0 ]; then
  die "$FORK_REMOTE/$UPSTREAM_BRANCH is AHEAD of $UPSTREAM_REMOTE/$UPSTREAM_BRANCH by $fm_ahead — it has commits of its own; a human decides this, never a force-push"
fi
if [ "${fm_behind:-0}" -gt 0 ]; then
  echo "== fork $UPSTREAM_BRANCH is $fm_behind behind upstream — fast-forwarding it =="
  g push -q "$FORK_REMOTE" "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH:$UPSTREAM_BRANCH" \
    || die "could not fast-forward $FORK_REMOTE/$UPSTREAM_BRANCH"
fi

# GATE 1 — a HOLD outranks everything. A peer session owning a PR can stop the
# rebuild without being awake to argue about it; today one did, and the rebuild
# it stopped would have shipped a regression to three machines twice.
hold=$(g show "$FORK_REMOTE/${INTEGRATED_BRANCH}-config:.claude/HOLD" 2>/dev/null)
[ -z "$hold" ] || die "HOLD on the config branch: $hold"

# GATE 2 — every PR green. A red PR in the rebuild is a machine running code CI
# rejected.
echo "== inputs =="
for n in $PRS; do
  b=$(gh pr view "$n" --repo "$UPSTREAM_REPO" --json headRefName --jq .headRefName) \
    || die "#$n unreadable"
  ci=$(gh pr checks "$n" --repo "$UPSTREAM_REPO" --json bucket --jq \
        '[.[].bucket] | if any(.=="fail") then "fail" elif any(.=="pending") then "pending" else "pass" end' 2>/dev/null)
  case "$ci" in
    fail)    die "#$n CI is RED — fix it or drop it from the rebuild" ;;
    pending) die "#$n CI is still running — rerun when it settles" ;;
  esac
  case " $PEER_OWNED_PRS " in *" $n "*) own="peer($PEER_OWNED_CONTACT)";; *) own=ours;; esac
  printf '  #%-4s %-8s %-34s ci=%s %s\n' "$n" "$(g rev-parse --short "$FORK_REMOTE/$b")" "$b" "$ci" "$own"
  eval "BR_$n=\$b"
done

# Rebuild in a worktree: the main checkout IS the running editable install, and
# checking it out onto a build has gutted all three machines before.
W="$R/.claude/worktrees/rebuild-auto"; rm -rf "$W"
g worktree add -q --detach "$W" "$UPSTREAM_REMOTE/$UPSTREAM_BRANCH" || die "worktree failed"
cleanup() { git -C "$R" worktree remove --force "$W" >/dev/null 2>&1; }
trap cleanup EXIT

m() { git -C "$W" merge --no-ff --no-edit -q -m "$2" "$1"; }

# The config branch is the FIRST input: the rebuild starts from upstream/main,
# so a config committed on the integration branch would be erased every time.
m "${INTEGRATED_BRANCH}-config" "merge the fork-local config" \
  || die "config branch will not merge cleanly — resolve by hand"

for n in $PRS; do
  eval "b=\$BR_$n"
  # NO `#` before the number. GitHub reads `#N` in any PUSHED commit message as
  # an issue reference and posts a `referenced` event to that PR's timeline —
  # on every push, never deduplicated, and impossible to retract afterwards.
  # This branch is a deploy artifact rebuilt on every upstream move, every PR
  # push and every config re-cut, so each rebuild appended a line to six
  # maintainers' PRs. Measured on ours: 21 referenced events across #196-#210,
  # and on CCF the user read them as branch churn and asked why there were so
  # many force-pushes (there had been one). "PR 296" reads the same to a human
  # and links nothing; the branch name in the same line already says which PR.
  m "$FORK_REMOTE/$b" "merge $b (upstream PR $n)" && continue
  # A conflict is a human decision. Report which files, then stop — never
  # resolve one automatically.
  echo "CONFLICT merging #$n ($b):"
  git -C "$W" diff --name-only --diff-filter=U | sed 's/^/    /'
  git -C "$W" merge --abort 2>/dev/null
  die "#$n conflicts — resolve in a worktree, push, re-run"
done

# GATE 3 — the suite, on the merged result. Each PR passing alone is not the
# same claim.
echo "== suite =="
( cd "$W" && env -u FORCE_COLOR -u NO_COLOR ${TEST_CMD:-python3 -m pytest -q} ) 2>&1 | tail -3
[ "${PIPESTATUS[0]}" = 0 ] || die "suite failed on the merged tree${TEST_CAVEAT:+ (caveat: $TEST_CAVEAT)}"

new=$(git -C "$W" rev-parse HEAD)
echo "== built $(git -C "$W" rev-parse --short HEAD) =="
g diff --stat "$INTEGRATED_BRANCH" "$new" | tail -1

if [ "$DEPLOY" = 0 ]; then
  echo "(dry run — nothing pushed. --deploy to publish and roll out.)"
  exit 0
fi
if [ "$YES" = 0 ]; then
  printf 'deploy to: %s  [y/N] ' "$DEPLOY_HOSTS"; read -r ans
  case "$ans" in y|Y) ;; *) echo "aborted"; exit 0;; esac
fi

g branch -f "${INTEGRATED_BRANCH}-prev" "$INTEGRATED_BRANCH"   # escape hatch

# Move the branch. `update-ref` moves the REF and touches neither the index nor
# the worktree — deliberate, because this checkout IS the running editable
# install and checking it out onto a build has gutted all three machines
# before.
#
# But when the live tree is ON this branch, ref-only leaves it a commit behind
# holding the OLD files, and `git status` then calls every one of them
# "modified". The deploy's dirty gate reads that as somebody's uncommitted work
# and REFUSES — so the rebuild produced a state its own deploy could not
# repair, on every single run. Measured twice on lambda-docker: 11 tracked
# files "modified" with the worktree byte-identical to <branch>-prev.
#
# So: ref-only when the live tree is elsewhere, ref+index+worktree together
# when it is here. `reset --hard` destroys modified tracked files, so it is
# gated on the tree being clean — and refusing there is right, because real
# uncommitted work outranks a deploy. The build is already built either way;
# the operator resolves the tree and re-runs.
if [ "$(g rev-parse --abbrev-ref HEAD)" = "$INTEGRATED_BRANCH" ]; then
  live_dirty=$(g status --porcelain --untracked-files=no)
  [ -z "$live_dirty" ] || {
    echo "REFUSE: $R is on $INTEGRATED_BRANCH with uncommitted TRACKED changes:"
    printf '%s\n' "$live_dirty" | head -10 | sed 's/^/    /'
    die "a human decides these, not a rebuild — commit or discard, then re-run"
  }
  g reset --hard -q "$new"     # ref + index + worktree, in one step
else
  g update-ref "refs/heads/$INTEGRATED_BRANCH" "$new"
fi

g push --force-with-lease -q "$FORK_REMOTE" "$INTEGRATED_BRANCH" \
  || die "push rejected — another session moved the branch; re-run"
cleanup; trap - EXIT

# A running process keeps the code it booted with, so decide the restart from
# the DIFF, and decide it BEFORE deploying: both the deploy loop and the branch
# below need the answer.
#
# NOT `^src/claude_swap/tui/`, which is what this matched before. The TUI
# imports switcher, autoswitch, usage_store, oauth, pin, settings and models, so
# a rebuild that changed only switcher.py left the running TUI on pre-deploy
# code with nothing here to restart it. The narrow pattern named the directory
# the files sit in, not the code the process runs.
#
# Nearly every rebuild touches src/, so this now restarts nearly every time.
# That is the honest cost and it is the right one: the alternative is a TUI
# serving code the deploy already replaced, which is the exact failure that let
# three machines run 16-hour-old code with every check green.
needs_restart=0
g diff --name-only "$INTEGRATED_BRANCH@{1}" "$INTEGRATED_BRANCH" 2>/dev/null \
  | grep -q '^src/claude_swap/' && needs_restart=1

# Deploy. Each host runs the repo's OWN deploy.sh and verify.sh, so what runs
# here is what the repo says, not what this script guesses.
echo "== deploy =="
me=$(hostname -s); rc=0
for h in $DEPLOY_HOSTS; do
  there=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'hostname -s' 2>/dev/null)
  if [ -z "$there" ]; then
    echo "  $h UNREACHABLE — no answer; re-probe before calling it down"; rc=1; continue
  fi
  echo "-- $h --"
  # SEPARATE calls, never `deploy && verify`. deploy.sh exits 2 on the routine
  # INCOMPLETE (files landed, process still stale), and under `&&` that skipped
  # verify.sh outright — so the remote hosts printed one INCOMPLETE line and not
  # a single named check while the local host printed all six. "Report per
  # machine" cannot survive a gate that decides which machines get a report.
  if [ "$there" = "$me" ]; then
    bash "$R/.claude/deploy.sh"; drc=$?
    bash "$R/.claude/verify.sh"; vrc=$?
  else
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$h" \
      'bash $HOME/workspace/cswap/cswap_fork/.claude/deploy.sh'; drc=$?
    ssh -o BatchMode=yes -o ConnectTimeout=25 "$h" \
      'bash $HOME/workspace/cswap/cswap_fork/.claude/verify.sh'; vrc=$?
  fi
  # rc 2 from deploy.sh means "the files landed, the process did not" — the
  # restart below is precisely its repair, so it must not stick to rc. Same for
  # this pass's verify when a restart is coming: its tui-fresh FAILS by
  # construction. The authoritative pass is the one AFTER the action.
  #
  # Without this the script printed SOME MACHINE FAILED on every successful
  # deploy, under a line-by-line report showing all three machines green. An
  # alarm that fires when nothing is wrong teaches the operator to stop reading
  # it, which costs the one time it is right.
  [ "$drc" = 0 ] || [ "$drc" = 2 ] || rc=1
  [ "$vrc" = 0 ] || [ "$needs_restart" = 1 ] || rc=1
done

if [ "$needs_restart" = 1 ]; then
  echo "== product code changed — restarting the TUI, or the deploy is only half done =="
  bash "$R/.claude/restart-tui.sh" --yes || rc=1

  # RE-VERIFY. restart-tui.sh checks each pid as it restarts it, but a deploy
  # that reports "restart required" and then does not confirm it is the gap
  # that let three machines run 16-hour-old code while every check was green.
  # The pass that matters is the one AFTER the action, on every host — and when
  # a restart was needed, it is the ONLY pass that decides rc.
  echo "== confirming the restart reached every machine =="
  for h in $DEPLOY_HOSTS; do
    there=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" 'hostname -s' 2>/dev/null)
    [ -n "$there" ] || { echo "  $h UNREACHABLE — restart unconfirmed"; rc=1; continue; }
    if [ "$there" = "$me" ]; then out=$(bash "$R/.claude/verify.sh" 2>&1); vrc=$?
    else out=$(ssh -o BatchMode=yes -o ConnectTimeout=25 "$h" \
                 'bash $HOME/workspace/cswap/cswap_fork/.claude/verify.sh' 2>&1); vrc=$?; fi
    echo "$out" | grep -E 'tui-fresh|tui-pin' | sed "s/^/  $h  /"
    [ "$vrc" = 0 ] || rc=1
  done
fi

[ "$rc" = 0 ] && echo "== all machines deployed and verified ==" \
              || echo "== SOME MACHINE FAILED — see per-host lines above =="
exit $rc
