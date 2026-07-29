# Cloud pin — when does the pin NOT hold?

Every way the pin can be silently absent, defeated, or lost, with the
current behaviour and whether a session can recover without restarting.
Written from a code audit plus live measurement on pmac (2026-07-28).

The pin is **fail-open by design**: when anything below trips, the request
leaves with the session's own bearer instead of failing. That keeps a
proxy problem from taking the session down, and it is why every case here
needs to be *visible* — a silent fallback is what made two of tonight's
verification runs read wrong.

## A. The proxy is never wired in

| # | Case | Behaviour | Recover without restart? |
|---|---|---|---|
| A1 | `claude` launched outside cswap (no `pin-env` eval) | No proxy env → pin absent for that session | **No** — env is fixed at exec |
| A2 | Background tree whose daemon started before the pin | Daemon's children inherit the daemon's env | **No** for that tree |
| A3 | `CLAUDE_CODE_PROCESS_WRAPPER` users vs plain users | Same: the wrapper is not required. `cswap run`/`exec_default` wire the env directly (`session.py:_exec`), and `pin-env` covers hand-launched `claude`. A wrapper only changes *who* execs, not whether env is inherited | n/a |

`cswap tui` / `auto` / `watch` / `menubar` never exec `claude`, so they need
no wiring — they only change which account is active.

## B. The proxy is wired but the pin is not applied

| # | Case | Behaviour | Recover without restart? |
|---|---|---|---|
| B1 | Pinned account **is** the active account | Provider returns None deliberately (live credential is the client's own) | Yes — switch away and it applies again |
| B2 | Pinned account has no stored credential | Provider returns None → original bearer | Yes — `cswap add` for that slot |
| B3 | Pinned account's refresh lineage is dead (`invalid_grant`, or the ~4-week refresh-token expiry) | Refresh fails → provider returns None → original bearer, silently | Yes — re-login + `cswap add`; next request picks it up |
| B4 | Pin points at a removed/renamed account (**dangling pin**) | `ensure_proxy` resolves the account and returns None when it is gone, so no proxy starts at all | Yes — re-pin |

`remove_account` does **not** clear the pin, so B4 is reachable. It is
handled safely (no proxy rather than a wrong one), but the stale pin stays
in `settings.json` and reads as "pinned" in the UI. Clearing it on removal
is a small fix worth making.

## C. The proxy dies underneath a live session

| # | Case | Behaviour | Recover without restart? |
|---|---|---|---|
| C1 | Daemon killed / crashed, session keeps its `HTTPS_PROXY` | Requests to a dead port fail at the transport | Yes — `ensure_proxy` respawns on the **recorded port** (fix e98316b), so the live session reconnects to the same address |
| C2 | Daemon recycled onto a *different* port (pre-fix behaviour) | Session pointed at a dead port and its requests left **without the pin, silently** | This was the bug behind two bad verification calls; fixed by port reclamation |
| C3 | Idle teardown (last refcount holder closed) | Daemon exits by design; the next launch spawns a fresh one | Yes |

## D. Concurrency

| # | Case | Behaviour |
|---|---|---|
| D1 | Token expires while several pinned requests are in flight | **Was**: every connection thread refreshed the same one-time refresh token → one winner, the rest `invalid_grant`, last writer persisting a consumed grant (lineage death). Measured: 8 threads → 8 refreshes. **Now**: refresh is serialized and waiters re-read the store and reuse the winner's rotation |

## E. Not a pin problem, but adjacent (separate PR)

`cswap run <n>` copies the account token into a session profile, so the
same token then has two consumers — cswap's usage polling and the new
session's own `fetchUtilization`. The usage budget is per token, and that
doubling produced `http-429 per-token usage budget reached` twice on pmac
tonight (21:30, 22:30), 2 minutes after a `cswap run 1`. The pin path never
touches `/api/oauth/usage` (0 calls in 433 traced requests) — this is a
`cswap run` issue and belongs in its own PR.

## Visibility (the recurring theme)

Every case above is survivable; what makes them dangerous is that most are
*quiet*. The mitigations that matter:

1. Surface the pin in the UI — done (`○ cloud` marker, menu label).
2. Report daemon health where the pin is shown, so C1/C3 are visible
   rather than inferred.
3. Clear the pin when its account is removed (B4).
4. Keep fail-open, but log once per condition when the pin does not apply —
   a session that silently drops the pin is worse than one that says so.
