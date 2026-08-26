# Account-Pin Proxy — design

Status: IMPLEMENTED, in the `cswap-pin` package behind the `claude-swap[pin]`
extra. Kept as the RATIONALE -- why a proxy at all, and what the forensics
established -- never as a description of the call graph; where the two
disagree, the code wins. The draft's component breakdown and entry-point plan
were cut when they shipped, because a superseded plan reads as documentation.
Runtime failure modes live in `cloud-pin-failure-modes.md`, maintained against
the shipped build.

## Problem

cswap swaps the on-disk credential in place so *inference* follows whichever
account is active. But two other operations authenticate with that SAME
credential and therefore also follow the swap, which the user does NOT want:

- **Remote Control (RC)** — a REPL session with `remoteControlAtStartup` creates
  a claude.ai "code session" (`cse_*`). Ownership is fixed at creation by the
  bearer token, and the session re-reads the disk credential on every ~8h
  worker-JWT refresh. So a swap moves RC to the new account: the phone/web loses
  the session (it now lives under a different account) and stale "ghost"
  sessions pile up.
- **Artifacts** ("frames") — published via `POST /api/frame/deploy/init`,
  owned by the creating bearer. A swap makes republish fail (403/404) and the
  artifact "disappears" from the account you're logged into on the web.

Goal: **inference follows the swap; RC and artifacts stay pinned to one chosen
account** — within any session, without changing how the user runs cswap.

## Why a proxy is the only mechanism (established by forensics)

All three operations read a single global credential accessor (`ys()`/`_s()` in
the 2.1.217 binary). There is no per-operation token selector wired to anything,
no multi-account store, no live token override the client honors for RC (the
`CLAUDE_CODE_OAUTH_TOKEN` env is rejected by RC's full-scope gate), and no
RC/artifact lifecycle hook. Verified dead ends: org-uuid header (server derives
ownership from the bearer, not `x-organization-uuid` — live-tested: bearer #2 +
org-header #1 → session owned by #2), trusted-device-token, `CLAUDE_CONFIG_DIR`/
`CLAUDE_SECURESTORAGE_CONFIG_DIR` (isolates the whole session, not per-op), dead
bridge overrides, and hooks.

The ONLY way to split auth per-operation inside one session is to intercept the
HTTP requests and swap the bearer on the specific routes. That means a MITM
forward proxy. Live-tested: creating a code session / listing frames with the
PIN account's bearer (while disk = the other account) yields PIN ownership
(200 for PIN, 401/404 for the other). Inference (`/v1/messages`) is left
untouched, so it keeps following the disk swap.

## Architecture

```
claude session
  HTTPS_PROXY = cswap-proxy         ← cswap wires this (in session.run / TUI launch)
     │
     ▼
cswap-proxy  (NEW, this feature)
  - MITM api.anthropic.com
  - route match:
      /v1/code/sessions*  → replace Authorization: Bearer <PIN token>
      /api/frame/*        → replace Authorization: Bearer <PIN token>
      everything else (esp. /v1/messages) → pass through unchanged
  - chain onward to the PREVIOUS HTTPS_PROXY value (CCF 9901, corp proxy, or
    direct if none)
     │
     ▼
(previous proxy, if any: CCF 9901 → corp 8118) → api.anthropic.com
```

The proxy is generic: it does NOT know about CCF. It reads whatever HTTPS_PROXY
was set before cswap inserted itself and CONNECT-chains through it. Works for
users with CCF, without CCF, and behind a corp proxy.

### Coexistence with other proxies (the three user classes)

| user | prior HTTPS_PROXY | cswap-proxy chains to |
|---|---|---|
| plain (no CCF, no corp) | (unset) | direct `net.connect` to api.anthropic.com |
| a user behind a local caching proxy | `http://127.0.0.1:<its port>` | that proxy (which may itself chain onward) |
| corp-proxy user | `http://corp:8118` | corp proxy |

cswap-proxy captures the inbound HTTPS_PROXY at launch and uses it as its own
upstream. CCF is never modified.
