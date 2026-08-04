"""R11 CRITICAL: is the collector's own ACTIVE read a third collapse site?

`_build_accounts_info:3188` does `creds = active.value or ""` and records
`_active_read_degraded`, but the quarantine scan passes those bytes to
`_entry_token_dead` and `_static_usage_sentinel:3905` has only a
`keychain_unavailable` arm. A DEGRADED read returns BYTES, so every
"empty read" guard is bypassed.

The probe is INSTRUMENTED: it prints the strike state it actually built, so a
`sentinel=None` from never reaching the path cannot be mistaken for a fix.
"""
import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_swap import oauth
from claude_swap.credentials import ActiveCredentials
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.json_output import USAGE_RELOGIN_REQUIRED
from claude_swap.models import Platform
from claude_swap.usage_store import FetchRecord as FR, UsageEntry

OLD = json.dumps({"claudeAiOauth": {"accessToken": "sk-old",
                                    "refreshToken": "rt-old", "expiresAt": 1000}})
NEW = json.dumps({"claudeAiOauth": {"accessToken": "sk-new",
                                    "refreshToken": "rt-new", "expiresAt": 99999999999000}})


@pytest.mark.parametrize("degraded", [False, True], ids=["CONTROL-healthy", "PROBE-degraded"])
def test_degraded_active_read_must_not_condemn_a_healed_slot(
    degraded, temp_home: Path, mock_claude_config: Path,
    sample_sequence_data: dict, monkeypatch,
):
    sample_sequence_data["accounts"]["2"]["email"] = "b@example.com"
    s = ClaudeAccountSwitcher()
    s.platform = Platform.MACOS
    s._setup_directories()
    s._write_json(s.sequence_file, sample_sequence_data)
    idents = {"2": ("b@example.com", "")}
    s._usage_store.record(
        {"2": FR(error="invalid_grant", struck_fp=oauth.credential_fingerprint(OLD))},
        idents,
    )
    # HEALED: backup now holds the new generation; the struck fp matches nothing.
    s._write_account_credentials("2", "b@example.com", NEW)

    pre = s._usage_store.entries(idents, [])["2"]
    print(f"\n  [{'degraded' if degraded else 'healthy '}] PREMISE "
          f"strikes={pre.auth_dead_strikes} token_dead={pre.token_dead()}")

    # Patch the object _build_accounts_info ACTUALLY calls (self, not _store).
    # A degraded read serves the STALE generation — that is what "degraded"
    # means: the Keychain read failed and a plaintext fallback covered it,
    # so the bytes may be the superseded ones. Serving NEW here (my first
    # cut) can never condemn anything, which is why both rows passed.
    served = OLD if degraded else NEW
    monkeypatch.setattr(s, "_read_active_credentials",
                        lambda: ActiveCredentials(served, False, degraded))
    # `_build_accounts_info` derives active_num from _get_current_account()
    # (the live IDENTITY), NOT from current_account_number(). Patching the
    # latter left every row is_active=False, so the branch under test never
    # ran and both rows "passed" for the wrong reason.
    monkeypatch.setattr(s, "_get_current_account",
                        lambda: ("b@example.com", ""))
    with patch.object(s, "current_account_number", return_value="2"):
        info = s._build_accounts_info()
        print("  info rows:", [(r[0], r[4], (r[5] or '')[:12]) for r in info])
        entries = s._collect_usage_entries(info, fetch=set())

    e = entries["2"]
    print(f"  [{'degraded' if degraded else 'healthy '}] RESULT  "
          f"sentinel={e.sentinel!r}  _active_read_degraded="
          f"{getattr(s, '_active_read_degraded', '<absent>')}")
    assert e.sentinel != USAGE_RELOGIN_REQUIRED, (
        "an already-healed active slot was condemned on a degraded read"
    )


@pytest.mark.parametrize("degraded", [False, True],
                         ids=["CONTROL-healthy", "PROBE-degraded"])
def test_post_fetch_call_site_is_guarded_too(degraded):
    """`_collect_usage_entries` calls `_entry_token_dead` TWICE: the pre-fetch
    quarantine scan, and again after a fetch returns invalid_grant. The test
    above runs with ``fetch=set()`` and so only ever reaches the first one.

    Measured: dropping ``self._active_read_degraded`` from the SECOND call
    site leaves the whole suite green (447 passed), while the same inputs
    flip the verdict True<->False here. That mutant is not equivalent, it was
    merely untested — a guard no test kills is one the next refactor removes.

    Driven at the method, not through the collector: reaching the post-fetch
    branch needs a granted claim plus a fetch that re-strikes, and a fleet
    built by hand for it silently failed to claim the slot (strikes stayed 0),
    which reads as a pass while never entering the branch.

    NOTE, measured: this pins the GUARD, not the WIRING. Because it calls the
    method directly it does not pass through either call site, so dropping
    `self._active_read_degraded` from the post-fetch call still leaves it
    green. `test_post_fetch_call_site_passes_the_flag` below is what kills
    that mutant.
    """
    struck = oauth.credential_fingerprint(OLD)
    s = ClaudeAccountSwitcher.__new__(ClaudeAccountSwitcher)
    # HEALED: the struck generation matches nothing stored any more.
    s._read_account_credentials_ex = lambda num, email: (NEW, False)
    entry = UsageEntry(auth_dead_strikes=1, struck_fingerprint=struck)

    # OLD is what a DEGRADED read serves: the superseded generation, because
    # Claude Code rotates keychain-only and the plaintext fallback lags.
    verdict = ClaudeAccountSwitcher._entry_token_dead(
        s, entry, "2", "b@example.com", OLD, True, degraded,
    )
    if degraded:
        assert verdict is False, (
            "a degraded read serving the struck generation condemned an "
            "already-healed slot at the post-fetch call site"
        )
    else:
        assert verdict is True, (
            "CONTROL BROKEN: a healthy read of the struck generation must "
            "still confirm dead, or this test cannot detect the guard"
        )


@pytest.mark.parametrize("degraded", [False, True],
                         ids=["healthy", "degraded"])
def test_unstruck_row_is_unaffected_by_the_degraded_flag(degraded):
    """The guard must only narrow the STRUCK path. An unstruck row answers
    False either way — the same invariant the docstring commits to."""
    s = ClaudeAccountSwitcher.__new__(ClaudeAccountSwitcher)
    s._read_account_credentials_ex = lambda num, email: (NEW, False)
    entry = UsageEntry(auth_dead_strikes=0,
                       struck_fingerprint=oauth.credential_fingerprint(OLD))
    assert ClaudeAccountSwitcher._entry_token_dead(
        s, entry, "2", "b@example.com", OLD, True, degraded) is False


def test_post_fetch_call_site_passes_the_flag():
    """The post-fetch call site must actually HAND `_active_read_degraded` in.

    The guard test above pins the method's behaviour but calls it directly,
    so it cannot see a call site that forgot the argument — measured: with
    only that test present, dropping the argument here left 451 passing.
    A guard nothing kills is one the next refactor deletes.

    Read from the source rather than driven through a fleet: reaching the
    post-fetch branch for real needs a granted claim plus a re-striking
    fetch, and every hand-built fleet for it so far failed to claim the slot
    and reported a pass without ever entering the branch. What must hold is
    structural — the argument is present at BOTH call sites — so that is
    what is asserted, with the count as its own control.
    """
    import ast
    import inspect
    from claude_swap.switcher import ClaudeAccountSwitcher

    src = inspect.getsource(ClaudeAccountSwitcher._collect_usage_entries)
    tree = ast.parse(textwrap.dedent(src))
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "_entry_token_dead"
    ]
    assert len(calls) == 2, (
        f"_collect_usage_entries has {len(calls)} _entry_token_dead call "
        "sites, not 2 — this test's premise moved; re-derive it rather than "
        "loosening the count"
    )
    for i, call in enumerate(calls):
        passed = [
            a for a in call.args
            if isinstance(a, ast.Attribute) and a.attr == "_active_read_degraded"
        ] + [
            k for k in call.keywords if k.arg == "active_read_degraded"
        ]
        assert passed, (
            f"call site {i + 1} of _entry_token_dead does not pass "
            "_active_read_degraded: a degraded active read will confirm a "
            "dead verdict against possibly-stale bytes there"
        )
