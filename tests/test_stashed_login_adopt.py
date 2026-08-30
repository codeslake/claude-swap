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


DEAD = _creds("rt-dead")
FRESH = _creds("rt-fresh-login")


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

    def _strike(self, sw, creds=DEAD):
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
        self._strike(switcher)

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
        self._strike(switcher)

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
        self._strike(switcher)

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
        self._strike(switcher)
        monkeypatch.setattr(switcher, "_read_account_credentials_ex",
                            lambda num, email: ("", True))

        assert switcher._adopt_stashed_login_for_slot(
            "2", "owner@example.com") is False
        assert entry_id in switcher._store._list_unclaimed_credentials()

    def test_another_account_s_login_is_left_alone(self, switcher):
        """Identity is what authorizes the write; a dead slot is not a
        licence to absorb whatever is in the stash."""
        entry_id = self._stash(switcher, email="someone@else.example",
                               uuid="uuid-else")
        self._strike(switcher)

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
        display refresh."""
        held = FileLock(switcher.lock_file)
        assert held.acquire(timeout=5), "premise: the test could take the lock"
        try:
            assert switcher._adopt_stashed_login_for_slot(
                "2", "owner@example.com") is False
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
