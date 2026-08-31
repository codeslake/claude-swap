"""A foreign login parked while its own slot was still healthy.

``_adopt_into_dead_slot`` decides at STASH time, and a slot that is alive then
is refused — correctly, because a live slot's own refresh token must not be
overwritten. Nothing looked at the stash again, so when that slot's token died
hours later the login sitting in the stash was never reached and the slot asked
for a re-login it did not need.

Measured on a real machine: the login was stashed as ``foreign`` five minutes
after the slot's last good fetch, and the ``invalid_grant`` strike landed most
of a day later -- all of it time in which the remedy was already on disk.
"""

import json
import time

import pytest

from claude_swap import oauth
from claude_swap.locking import FileLock
from claude_swap.switcher import ClaudeAccountSwitcher
from claude_swap.json_output import USAGE_RELOGIN_REQUIRED
from claude_swap.usage_store import AUTH_DEAD_STRIKES, FetchRecord as FR


def _creds(refresh: str) -> str:
    return json.dumps({
        "claudeAiOauth": {
            "accessToken": "sk-" + refresh,
            "refreshToken": refresh,
            "expiresAt": 9999999999000,
        }
    })


def _dated(refresh: str, expires_at_ms: int) -> str:
    blob = json.loads(_creds(refresh))
    blob["claudeAiOauth"]["refreshTokenExpiresAt"] = expires_at_ms
    return json.dumps(blob)


# OFFSETS FROM NOW. Epoch constants (1970, the year 5138) sit either side of
# every plausible bug, so a guard comparing ms against seconds still passes.
_DAY_MS = 86_400_000
_NOW_MS = int(time.time() * 1000)

DEAD = _creds("rt-dead")
FRESH = _creds("rt-fresh-login")           # no refreshTokenExpiresAt at all
EXPIRED = _dated("rt-expired", _NOW_MS - _DAY_MS)
LIVE_DATED = _dated("rt-live-dated", _NOW_MS + 30 * _DAY_MS)
NEARLY_SPENT = _dated("rt-nearly-spent", _NOW_MS + 60_000)


def _strike(sw, creds=DEAD):
    """Quarantine slot 2 against ``creds``' generation."""
    path = sw._usage_store.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schemaVersion": 2,
        "accounts": {
            "2": {
                "email": "owner@example.com",
                "organizationUuid": "",
                "authDeadStrikes": AUTH_DEAD_STRIKES,
                "struckFingerprint": oauth.credential_fingerprint(creds),
                "consecutiveFailures": 2,
                "lastError": "invalid_grant",
                "lastGood": {"five_hour": {"pct": 10.0}},
            }
        },
    }))


class TestAdoptStashedLoginForSlot:
    """The stash gets a LATER adopter, at the moment the slot becomes dead."""

    @pytest.fixture
    def switcher(self, temp_home, mock_claude_config, sample_sequence_data):
        sw = ClaudeAccountSwitcher()
        sw._setup_directories()
        sample_sequence_data["accounts"]["2"]["email"] = "owner@example.com"
        sample_sequence_data["accounts"]["2"]["uuid"] = "uuid-owner"
        sw._write_json(sw.sequence_file, sample_sequence_data)
        sw._write_account_credentials("2", "owner@example.com", DEAD)
        return sw


    def _stash(self, sw, creds=FRESH, email="owner@example.com",
               uuid="uuid-owner"):
        return sw._store._write_unclaimed_credential(creds, {
            "reason": "foreign",
            "configSlot": "1",
            "fingerprint": oauth.credential_fingerprint(creds),
            "resolvedIdentity": {"uuid": uuid, "email": email,
                                 "organizationUuid": None},
        })

    def test_a_dead_slot_adopts_the_login_stashed_for_it(self, switcher):
        """The whole point: the slot holds the stashed login afterwards."""
        entry_id = self._stash(switcher)
        _strike(switcher)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is True

        stored, unreadable = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert not unreadable
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(FRESH)
        assert entry_id not in switcher._store._list_unclaimed_credentials()

    def test_the_adopt_lifts_the_quarantine_it_wrote_over(self, switcher):
        """A strike left standing keeps the healed slot out of every pass."""
        self._stash(switcher)
        _strike(switcher)

        switcher._adopt_stashed_login_for_slot("2", "owner@example.com")

        ident = {"2": ("owner@example.com", "")}
        entry = switcher._usage_store.entries(ident)["2"]
        assert entry.token_dead() is False

    def test_a_live_slot_keeps_its_own_credential(self, switcher):
        """No strike: the stored refresh token is the newer one to protect."""
        entry_id = self._stash(switcher)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False

        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(DEAD)
        assert entry_id in switcher._store._list_unclaimed_credentials()

    def test_the_struck_generation_is_not_adopted_back_in(self, switcher):
        """A stash of the very bytes the endpoint condemned heals nothing, and
        adopting it would clear the strike that describes it."""
        entry_id = self._stash(switcher, creds=DEAD)
        _strike(switcher)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert entry_id in switcher._store._list_unclaimed_credentials()

    def test_bytes_the_slot_already_holds_are_not_written_back(self, switcher):
        """`_adopt_login_into_slot` refuses this for its own reason — rewriting
        the identical credential only shifts `.prev`. Here it is worse: an
        UNBOUND strike (a row written before fingerprints were recorded) binds
        unconditionally, so the comparison against the struck generation
        cannot exclude these bytes, and adopting would clear a verdict that is
        accurate about what the slot still holds."""
        entry_id = self._stash(switcher, creds=DEAD)
        path = switcher._usage_store.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "schemaVersion": 2,
            "accounts": {
                "2": {
                    "email": "owner@example.com",
                    "organizationUuid": "",
                    "authDeadStrikes": AUTH_DEAD_STRIKES,   # no struckFingerprint
                    "consecutiveFailures": 2,
                    "lastError": "invalid_grant",
                    "lastGood": {"five_hour": {"pct": 10.0}},
                }
            },
        }))

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert entry_id in switcher._store._list_unclaimed_credentials(), (
            "a stash entry was spent on bytes the slot already held")
        ident = {"2": ("owner@example.com", "")}
        assert switcher._usage_store.entries(ident)["2"].token_dead() is True, (
            "an accurate strike was cleared by rewriting the same credential")

    def test_a_slot_whose_own_credential_cannot_be_READ_is_left_alone(
            self, switcher, monkeypatch):
        """An unreadable stored credential is not an empty slot: on a Mac a
        locked Keychain reads as a failure, and adopting on that answer would
        overwrite a credential nobody could see.

        `_slot_token_dead` is what refuses — it returns False on an unreadable
        read for both the idle and the active slot, so the adopt stops at its
        pre-check. This pins the BEHAVIOUR, not that check: a later refactor
        that moves the refusal must keep the answer."""
        entry_id = self._stash(switcher)
        _strike(switcher)
        monkeypatch.setattr(switcher, "_read_account_credentials_ex",
                            lambda num, email: ("", True))

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert entry_id in switcher._store._list_unclaimed_credentials()

    def test_a_write_that_landed_is_an_adopt_even_if_the_strike_cannot_lift(
            self, switcher, monkeypatch):
        """The write ADVANCES the slot. Reporting "could not adopt" after it
        lands makes the caller announce a re-login for a slot that now holds a
        working credential, and leaves a stash row pointing at bytes the slot
        already has — which the identical-bytes guard then refuses forever.
        The sibling adopt gets this right: it logs the unlifted quarantine and
        carries on."""
        entry_id = self._stash(switcher)
        _strike(switcher)

        def boom(*a, **k):
            raise OSError("usage store is unwritable")

        monkeypatch.setattr(switcher._usage_store, "clear_dead_token", boom)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is True
        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(FRESH), "the write did not land"
        assert entry_id not in switcher._store._list_unclaimed_credentials(), (
            "the adopted entry stayed in the stash")

    def test_a_write_that_FAILED_is_not_an_adopt(self, switcher, monkeypatch):
        """THE CONTROL. Without it the assertion above passes for a build that
        calls every failure an adopt."""
        entry_id = self._stash(switcher)
        _strike(switcher)

        def boom(*a, **k):
            raise OSError("backup is unwritable")

        monkeypatch.setattr(switcher, "_write_account_credentials", boom)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert entry_id in switcher._store._list_unclaimed_credentials()

    def test_an_EXPIRED_stash_is_not_adopted(self, switcher):
        """A refresh token already past its own expiry mints nothing, so
        adopting it writes a dead credential into a dead slot AND lifts the
        quarantine that was accurate about the slot."""
        entry_id = self._stash(switcher, creds=EXPIRED)
        _strike(switcher)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False

        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(DEAD), "a spent grant was written in"
        ident = {"2": ("owner@example.com", "")}
        assert switcher._usage_store.entries(ident)["2"].token_dead() is True, (
            "a credential that mints nothing lifted an accurate quarantine")
        assert entry_id in switcher._store._list_unclaimed_credentials()

    def test_an_UNREADABLE_stash_entry_is_not_adopted(
            self, switcher, monkeypatch):
        """Unreadable WITH BYTES. `("", True)` is caught by `not creds`, so
        the flag itself never runs and deleting it leaves such a test green;
        this returns a payload that would otherwise adopt, so only the flag
        can refuse it."""
        entry_id = self._stash(switcher, creds=FRESH)
        _strike(switcher)
        monkeypatch.setattr(switcher._store, "_read_unclaimed_credential",
                            lambda *a, **k: (FRESH, True))

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert entry_id in switcher._store._list_unclaimed_credentials()
        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(DEAD)

    def test_a_refresh_token_inside_the_expiry_buffer_is_refused(
            self, switcher):
        """`_sweep_unclaimed_stash` asks `oauth.is_oauth_token_expired`, which
        subtracts a 5-minute buffer, so it would DROP this row. Adopting it
        lifts the quarantine for bytes the sweep is about to delete and the
        slot re-strikes next pass. Both sides ask the same predicate."""
        entry_id = self._stash(switcher, creds=NEARLY_SPENT)
        _strike(switcher)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert entry_id in switcher._store._list_unclaimed_credentials()

    def test_CONTROL_an_unexpired_refreshTokenExpiresAt_is_adopted(
            self, switcher):
        """Without this the refusal above passes for a build that refuses
        every entry carrying the field. The no-field case is what `FRESH` is,
        so every other test in this class already covers it."""
        entry_id = self._stash(switcher, creds=LIVE_DATED)
        _strike(switcher)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is True
        assert entry_id not in switcher._store._list_unclaimed_credentials()

    def test_another_account_s_login_is_left_alone(self, switcher):
        """Identity is what authorizes the write; a dead slot is not a
        licence to absorb whatever is in the stash."""
        entry_id = self._stash(switcher, email="someone@else.example",
                               uuid="uuid-else")
        _strike(switcher)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False

        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(DEAD)
        assert entry_id in switcher._store._list_unclaimed_credentials()


class TestTheAdoptIsASlotMutation:
    """Every other path that writes a slot credential holds the slot lock, and
    `_adopt_login_into_slot` says why in its own words: identity, verdict and
    write must be one transaction, or a switch persisting a rotated refresh
    token in the gap is overwritten by a guard that had already passed.

    This adopt is reached from `_collect_usage_entries`, a READ path that holds
    no lock — so it has to take one itself.
    """

    @pytest.fixture
    def switcher(self, temp_home, mock_claude_config, sample_sequence_data):
        sw = ClaudeAccountSwitcher()
        sw._setup_directories()
        sample_sequence_data["accounts"]["2"]["email"] = "owner@example.com"
        sample_sequence_data["accounts"]["2"]["uuid"] = "uuid-owner"
        sw._write_json(sw.sequence_file, sample_sequence_data)
        sw._write_account_credentials("2", "owner@example.com", DEAD)
        sw._usage_store.record(
            {"2": FR(error="invalid_grant",
                     struck_fp=oauth.credential_fingerprint(DEAD))},
            {"2": ("owner@example.com", "")},
        )
        sw._store._write_unclaimed_credential(FRESH, {
            "reason": "foreign",
            "configSlot": "1",
            "fingerprint": oauth.credential_fingerprint(FRESH),
            "resolvedIdentity": {"uuid": "uuid-owner",
                                 "email": "owner@example.com",
                                 "organizationUuid": None},
        })
        return sw

    def test_it_does_not_write_while_another_holder_has_the_slot_lock(
            self, switcher):
        """A held lock means a slot mutation is in flight. The adopt must
        stand down for this pass, not write beside it, and not raise into a
        display refresh.

        AND IT MUST STAND DOWN QUICKLY. `_collect_usage_entries` runs on every
        list, status and TUI refresh and calls this once per dead slot, so the
        lock's default 10s wait would freeze the display for 10s PER SLOT
        against any concurrent switch. Measured on a held lock: the default
        acquire returns False after 10.01s."""
        held = FileLock(switcher.lock_file)
        assert held.acquire(timeout=5), "premise: the test could take the lock"
        try:
            start = time.monotonic()
            assert switcher._adopt_stashed_login_for_slot(
                "2", "owner@example.com") is False
            waited = time.monotonic() - start
            assert waited < 2.0, (
                "a display refresh waited %.1fs on a held lock" % waited)
            stored, _ = switcher._read_account_credentials_ex(
                "2", "owner@example.com")
            assert oauth.credential_fingerprint(stored) == \
                oauth.credential_fingerprint(DEAD), "wrote past a held lock"
        finally:
            held.release()

    def test_it_adopts_once_the_lock_is_free(self, switcher):
        """THE CONTROL. Without it the assertion above passes for a build whose
        adopt never writes at all."""
        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is True
        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(FRESH)


    def test_a_slot_that_heals_while_the_lock_is_waited_out_is_left_alone(
            self, switcher, monkeypatch):
        """The pre-check is lock-free, so its answer can be stale by the time
        the lock is granted. A slot that healed in the gap must not be written
        over: the credential it now holds is newer than anything in the stash.
        """
        calls = []

        def dead(num, email):
            calls.append(num)
            return len(calls) == 1        # true for the pre-check, false under the lock

        monkeypatch.setattr(switcher, "_slot_token_dead", dead)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert len(calls) == 2, (
            "the verdict was not re-derived under the lock", calls)
        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(DEAD)

    def test_a_slot_the_roster_moved_under_is_left_alone(
            self, switcher, monkeypatch):
        """`remove_account` holds no lock and the swap/move paths hold this
        one, so the roster can change while the lock is waited out. A stale
        (slot, address) pair would write a live credential into a slot that is
        now somebody else's."""
        real = switcher._get_sequence_data
        seen = {"n": 0}

        def moved():
            seen["n"] += 1
            data = real()
            if seen["n"] > 1:             # the read under the lock
                data["accounts"]["2"]["email"] = "someone@else.example"
            return data

        monkeypatch.setattr(switcher, "_get_sequence_data", moved)

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(DEAD)



class TestTheCollectPassReachesTheStash:
    """A method nobody calls heals nothing. The pass that decides to SAY
    "re-login needed" is the one that must look in the stash first."""

    def _switcher(self, sample_sequence_data):
        sw = ClaudeAccountSwitcher()
        sw._setup_directories()
        sample_sequence_data["accounts"]["2"]["email"] = "owner@example.com"
        sample_sequence_data["accounts"]["2"]["uuid"] = "uuid-owner"
        sw._write_json(sw.sequence_file, sample_sequence_data)
        sw._write_account_credentials("2", "owner@example.com", DEAD)
        sw._usage_store.record(
            {"2": FR(error="invalid_grant",
                     struck_fp=oauth.credential_fingerprint(DEAD))},
            {"2": ("owner@example.com", "")},
        )
        return sw

    def test_a_stashed_login_is_adopted_instead_of_asking_for_a_re_login(
        self, temp_home, mock_claude_config, sample_sequence_data
    ):
        sw = self._switcher(sample_sequence_data)
        sw._store._write_unclaimed_credential(FRESH, {
            "reason": "foreign",
            "configSlot": "1",
            "fingerprint": oauth.credential_fingerprint(FRESH),
            "resolvedIdentity": {"uuid": "uuid-owner",
                                 "email": "owner@example.com",
                                 "organizationUuid": None},
        })

        entries = sw._collect_usage_entries(sw._build_accounts_info(),
                                            fetch=set())

        assert entries["2"].sentinel != USAGE_RELOGIN_REQUIRED
        stored, _ = sw._read_account_credentials_ex("2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(FRESH)

    def test_with_nothing_stashed_the_slot_still_asks(
        self, temp_home, mock_claude_config, sample_sequence_data
    ):
        """THE CONTROL. Without it the assertion above passes for a pass that
        never reached the dead branch at all."""
        sw = self._switcher(sample_sequence_data)

        entries = sw._collect_usage_entries(sw._build_accounts_info(),
                                            fetch=set())

        assert entries["2"].sentinel == USAGE_RELOGIN_REQUIRED


class TestTheWriteSideTwinRefusesTheSameBytes:
    """`_adopt_into_dead_slot` gets the SAME string `_stash_live_credential`
    parked one line earlier, and decides alone: that stash sweeps its own
    expired row, so no reader ever reaches those bytes."""

    @pytest.fixture
    def switcher(self, temp_home, mock_claude_config, sample_sequence_data):
        sw = ClaudeAccountSwitcher()
        sw._setup_directories()
        sample_sequence_data["accounts"]["2"]["email"] = "owner@example.com"
        sw._write_json(sw.sequence_file, sample_sequence_data)
        sw._write_account_credentials("2", "owner@example.com", DEAD)
        _strike(sw)
        return sw

    def _dead(self, sw):
        ident = {"2": ("owner@example.com", "")}
        return sw._usage_store.entries(ident)["2"].token_dead()

    def test_an_EXPIRED_foreign_credential_does_not_heal_a_dead_slot(
            self, switcher):
        """Spent bytes written into the slot AND the accurate quarantine
        lifted — the exact pair the reader-side guard forbids."""
        assert switcher._adopt_into_dead_slot(
            "2", EXPIRED, switcher._get_sequence_data() or {}) is False

        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(DEAD), "a spent grant was written in"
        assert self._dead(switcher) is True, (
            "a credential that mints nothing lifted an accurate quarantine")

    def test_CONTROL_an_unexpired_foreign_credential_still_heals(
            self, switcher):
        """Without this the refusal passes for a build that heals nothing.
        This is issue #136's whole point: a dead slot has no freshness left to
        protect, so resolved bytes that can still mint are strictly better."""
        assert switcher._adopt_into_dead_slot(
            "2", LIVE_DATED, switcher._get_sequence_data() or {}) is True

        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(LIVE_DATED)
        assert self._dead(switcher) is False, "the quarantine was not lifted"


class TestTheResyncAdoptDeliberatelyHasNoSpentGuard:
    """The fifth writer into a dead slot, and the one that must NOT refuse a
    spent refresh token.

    THE DISTINGUISHER IS THE RECENCY GUARD, not the live access token. Both
    this path and `_adopt_into_dead_slot` reach their slot having had an
    Anthropic endpoint accept the access token, so "the server just accepted
    these bytes" argues equally for dropping the guard its sibling carries.
    What only this one has is the `_refresh_expiry` comparison that refuses a
    credential whose refresh lifetime ends earlier than the stored one: the
    spent guard is recency work, and here it is already done.

    On top of that its caller reaches it only past `outcome.usage is not
    None`, so refusing would forfeit a working access token's remaining life
    to avoid a quarantine that re-forms on its own. Two reviews split on this,
    so both halves are pinned rather than left to be "fixed" into a
    regression.
    """

    @pytest.fixture
    def switcher(self, temp_home, mock_claude_config, sample_sequence_data):
        sw = ClaudeAccountSwitcher()
        sw._setup_directories()
        sample_sequence_data["accounts"]["2"]["email"] = "owner@example.com"
        sample_sequence_data["accounts"]["2"]["uuid"] = "uuid-owner"
        sw._write_json(sw.sequence_file, sample_sequence_data)
        sw._write_account_credentials("2", "owner@example.com", DEAD)
        _strike(sw)
        return sw

    def test_a_spent_refresh_token_with_a_live_access_token_is_adopted(
            self, switcher):
        """`EXPIRED` is exactly that shape: `expiresAt` far future,
        `refreshTokenExpiresAt` a day past."""
        assert switcher._adopt_login_into_slot(
            "1", EXPIRED,
            {"uuid": "uuid-owner", "email": "owner@example.com",
             "organizationUuid": ""},
        ) is True

        stored, _ = switcher._read_account_credentials_ex(
            "2", "owner@example.com")
        assert oauth.credential_fingerprint(stored) == \
            oauth.credential_fingerprint(EXPIRED), (
                "the resync adopt grew a spent-token guard. Its own recency "
                "compare already does that work, and its caller has proven "
                "the access token still fetches — so this forfeits it")

    def _resync_reached(self, sw, monkeypatch, usage):
        """Drive `_fetch_active_usage`'s fast path; report whether the resync
        (and so the adopt below it) was reached."""
        reached = []
        monkeypatch.setattr(
            oauth, "try_fetch_usage_for_account",
            lambda *a, **k: oauth.UsageOutcome(usage=usage, error=None))
        monkeypatch.setattr(oauth, "build_usage_result", lambda *a, **k: None)
        monkeypatch.setattr(sw, "_resync_rotated_backup",
                            lambda *a, **k: reached.append(1))
        sw._fetch_active_usage("2", "owner@example.com", LIVE_DATED)
        return bool(reached)

    def test_the_server_accepted_these_bytes_is_what_gates_the_resync(
            self, switcher, monkeypatch):
        """THE PREMISE, PINNED SEPARATELY FROM THE CONCLUSION. Widen this gate
        and a credential the endpoint never accepted reaches the adopt, and
        the exception above quietly becomes the defect the other four paths
        were fixed for — while its own test goes on passing and defending it.
        Measured before this test existed: `if outcome.usage is not None` to
        `if True` left the whole suite green."""
        assert self._resync_reached(
            switcher, monkeypatch, usage=None) is False

    def test_CONTROL_a_usage_dict_does_reach_the_resync(
            self, switcher, monkeypatch):
        """Without this the assertion above passes for a build where the
        resync is unreachable altogether."""
        assert self._resync_reached(
            switcher, monkeypatch, usage={"five_hour": {"pct": 1.0}}) is True
