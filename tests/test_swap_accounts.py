"""Tests for `cswap swap` (ClaudeAccountSwitcher.swap_accounts)."""

import os
import sys
from pathlib import Path

import pytest

from claude_swap.exceptions import (
    AccountNotFoundError,
    ConfigError,
    CredentialError,
    ValidationError,
)
from claude_swap.switcher import ClaudeAccountSwitcher


def _refuse_write(self, num, email, creds):
    raise OSError("disk full (injected)")


class TestSwapAccounts:
    """Test ClaudeAccountSwitcher.swap_accounts()."""

    def _write(self, switcher, data):
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, data)

    def test_swap_by_number(self, temp_home: Path, sample_sequence_data: dict):
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)

        num_a, num_b = switcher.swap_accounts("1", "2")

        assert (num_a, num_b) == ("1", "2")
        data = switcher._get_sequence_data()
        assert data["accounts"]["1"]["email"] == "account2@example.com"
        assert data["accounts"]["2"]["email"] == "account1@example.com"

    def test_swap_moves_active_number_with_account(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        assert sample_sequence_data["activeAccountNumber"] == 1

        switcher.swap_accounts("1", "2")

        data = switcher._get_sequence_data()
        # account1 was active and now lives in slot 2
        assert data["activeAccountNumber"] == 2
        assert data["accounts"]["2"]["email"] == "account1@example.com"

    def test_swap_keeps_sequence_sorted(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """Sequence stays sorted, so rotation and list order follow the new
        numbers — the accounts genuinely trade places in `cswap list`."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)

        switcher.swap_accounts("1", "2")

        data = switcher._get_sequence_data()
        assert data["sequence"] == [1, 2]

    def test_swap_by_email_and_alias(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        sample_sequence_data["accounts"]["2"]["alias"] = "dev"
        self._write(switcher, sample_sequence_data)

        num_a, num_b = switcher.swap_accounts("account1@example.com", "dev")

        assert (num_a, num_b) == ("1", "2")
        data = switcher._get_sequence_data()
        # The alias travels with its account into the new slot.
        assert data["accounts"]["1"].get("alias") == "dev"
        assert data["accounts"]["2"].get("alias") is None

    def test_swap_moves_credential_and_config_backups(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        switcher._write_account_credentials("1", "account1@example.com", "creds-one")
        switcher._write_account_config("1", "account1@example.com", "config-one")
        switcher._write_account_credentials("2", "account2@example.com", "creds-two")
        switcher._write_account_config("2", "account2@example.com", "config-two")

        switcher.swap_accounts("1", "2")

        assert (
            switcher._read_account_credentials("2", "account1@example.com")
            == "creds-one"
        )
        assert (
            switcher._read_account_config("2", "account1@example.com") == "config-one"
        )
        assert (
            switcher._read_account_credentials("1", "account2@example.com")
            == "creds-two"
        )
        assert (
            switcher._read_account_config("1", "account2@example.com") == "config-two"
        )
        # Old keys are gone.
        assert switcher._read_account_credentials("1", "account1@example.com") == ""
        assert switcher._read_account_credentials("2", "account2@example.com") == ""

    def test_swap_with_one_slot_missing_backups(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """A never-backed-up slot swaps cleanly and stays credential-less."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        switcher._write_account_credentials("1", "account1@example.com", "creds-one")

        switcher.swap_accounts("1", "2")

        assert (
            switcher._read_account_credentials("2", "account1@example.com")
            == "creds-one"
        )
        assert switcher._read_account_credentials("1", "account2@example.com") == ""

    def test_swap_same_account_rejected(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)

        with pytest.raises(ValidationError):
            switcher.swap_accounts("1", "1")

    def test_swap_unknown_identifier_rejected(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)

        with pytest.raises(AccountNotFoundError):
            switcher.swap_accounts("1", "nosuch@example.com")

    def test_swap_same_email_accounts(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Same email, different orgs: the backup keys fully overlap."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")

        switcher.swap_accounts("1", "2")

        assert switcher._read_account_credentials("1", email) == "creds-personal"
        assert switcher._read_account_credentials("2", email) == "creds-org"
        data = switcher._get_sequence_data()
        assert data["accounts"]["2"]["organizationUuid"] == "org-uuid-5678"
        # The durable staging copies are cleaned up after the commit.
        assert not list(switcher.credentials_dir.glob(".swap-staging-*"))

    def test_swap_same_email_partial_failure_rolls_back(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """A write failure mid-swap must not destroy an overlapping backup.

        With a shared email the destination key IS the other account's key,
        so without a rollback the second account's credential would exist
        nowhere but in memory after the first write.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")

        real_write = ClaudeAccountSwitcher._write_account_credentials
        calls = {"n": 0}

        def failing_write(self, num, email, creds):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk full (injected)")
            return real_write(self, num, email, creds)

        # Scoped context, not the fixture's shared `monkeypatch`: that
        # instance also carries the autouse colour/keychain/home scrubs, and
        # `.undo()` on it would unwind those too (H-1) — restoring whatever
        # FORCE_COLOR/NO_COLOR the developer's shell actually has exported
        # for the rest of this test.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ClaudeAccountSwitcher, "_write_account_credentials", failing_write
            )
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")

        # Both originals are back under their pre-swap keys, and the account
        # table was never renumbered.
        assert switcher._read_account_credentials("1", email) == "creds-org"
        assert switcher._read_account_credentials("2", email) == "creds-personal"
        data = switcher._get_sequence_data()
        assert data["accounts"]["1"]["organizationUuid"] == "org-uuid-5678"
        assert data["activeAccountNumber"] == 1

    def _same_email_slots(self, switcher, data, profiles=("1", "2")) -> str:
        """Two slots sharing one email; each named slot gets a marked profile."""
        self._write(switcher, data)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")
        for num in profiles:
            profile = switcher._session_dir(num, email)
            profile.mkdir(parents=True, exist_ok=True)
            (profile / "marker").write_text(f"SLOT-{num}-HISTORY")
        return email

    def _marker(self, switcher, num: str, email: str) -> str:
        return (switcher._session_dir(num, email) / "marker").read_text()

    def test_swap_aborted_in_staging_touches_neither_slot(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """An abort before anything is mutated must leave both slots alone.

        Staging is the last step that can fail with nothing yet written, so it
        runs outside the rollback's reach: a reverse move would exchange two
        untouched profiles, and a credential restore would invalidate two live
        session profiles — both while the error says nothing was changed.
        """
        switcher = ClaudeAccountSwitcher()
        email = self._same_email_slots(switcher, sample_sequence_data_with_org)
        for num in ("1", "2"):
            (switcher._session_dir(num, email) / ".credentials.json").write_text("{}")

        # A leftover staging file is what makes staging refuse.
        (switcher.credentials_dir / ".swap-staging-creds-1.json").write_text("{}")

        with pytest.raises(ConfigError):
            switcher.swap_accounts("1", "2")

        assert self._marker(switcher, "1", email) == "SLOT-1-HISTORY"
        assert self._marker(switcher, "2", email) == "SLOT-2-HISTORY"
        for num in ("1", "2"):
            assert (
                switcher._session_dir(num, email) / ".credentials.json"
            ).exists(), f"slot {num}'s session credentials were invalidated"

    def test_swap_failing_after_the_move_still_restores_the_session_dirs(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """A failure after the forward move must still reverse it, or the gate
        on that reverse becomes a way to skip the rollback."""
        switcher = ClaudeAccountSwitcher()
        email = self._same_email_slots(switcher, sample_sequence_data_with_org)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ClaudeAccountSwitcher, "_write_account_credentials", _refuse_write
            )
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")

        assert self._marker(switcher, "1", email) == "SLOT-1-HISTORY"
        assert self._marker(switcher, "2", email) == "SLOT-2-HISTORY"

    def test_swap_does_not_reverse_a_forward_move_that_moved_nothing(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """``_swap_session_dirs`` swallows OSError, so reaching the end of it
        is not evidence that anything moved.

        A forward move that gave up leaves both profiles where they started,
        and reversing that exchanges two directories nobody touched.
        """
        switcher = ClaudeAccountSwitcher()
        email = self._same_email_slots(switcher, sample_sequence_data_with_org)

        real_replace = os.replace
        tripped: list[str] = []

        def fail_the_first_park(src, dst):
            # The forward move only; the rollback's reverse must run for real.
            if not tripped and str(dst).endswith(".swapping"):
                tripped.append(str(dst))
                raise OSError("cross-device link (injected)")
            return real_replace(src, dst)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", fail_the_first_park)
            mp.setattr(
                ClaudeAccountSwitcher, "_write_account_credentials", _refuse_write
            )
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")

        assert tripped, "the injected move failure never fired"
        assert self._marker(switcher, "1", email) == "SLOT-1-HISTORY"
        assert self._marker(switcher, "2", email) == "SLOT-2-HISTORY"

    def test_swap_reverses_a_forward_move_that_was_interrupted(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """A move aborted part-way through still has to be undone.

        ``swap_accounts`` catches BaseException, so a Ctrl-C between the two
        halves of the exchange reaches the rollback with one profile already
        parked under the other slot's key.
        """
        switcher = ClaudeAccountSwitcher()
        email = self._same_email_slots(switcher, sample_sequence_data_with_org)

        real_replace = os.replace

        def interrupt_the_last_park(src, dst):
            if str(src).endswith(".swapping"):
                raise KeyboardInterrupt
            return real_replace(src, dst)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", interrupt_the_last_park)
            with pytest.raises(KeyboardInterrupt):
                switcher.swap_accounts("1", "2")

        slot_2 = switcher._session_dir("2", email) / "marker"
        assert slot_2.exists(), "account 2's profile was left under slot 1's key"
        assert slot_2.read_text() == "SLOT-2-HISTORY"

    def test_swap_interrupted_just_past_a_move_still_reverses_it(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """An abort landing just PAST a rename must still undo that rename.

        The rename and the record of it are two statements, and a signal is
        delivered between them. Recording after the call returns therefore
        loses a move that is already on disk, and the slot then serves the
        other account's session history while the swap reports it aborted.
        """
        switcher = ClaudeAccountSwitcher()
        email = self._same_email_slots(switcher, sample_sequence_data_with_org)

        real_replace = os.replace
        landed: list[str] = []

        def interrupt_just_past_the_move(src, dst):
            # One shot: the rollback's own reverse must run for real.
            real_replace(src, dst)
            if not landed and not str(dst).endswith(".swapping"):
                landed.append(str(dst))
                raise KeyboardInterrupt

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", interrupt_just_past_the_move)
            with pytest.raises(KeyboardInterrupt):
                switcher.swap_accounts("1", "2")

        assert landed, "the injected interrupt never fired"
        slot_2 = switcher._session_dir("2", email) / "marker"
        assert slot_2.exists(), "account 2's profile was left under slot 1's key"
        assert slot_2.read_text() == "SLOT-2-HISTORY"

    def test_swap_interrupted_before_the_first_move_is_not_reversed(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Ctrl-C landing before the first rename must not reverse anything.

        Counterpart to the test above: the two together say the move has to
        report progress as it goes, since neither "assume it ran" nor "assume
        it did not" is right for an abort that can land on either side.
        """
        switcher = ClaudeAccountSwitcher()
        email = self._same_email_slots(switcher, sample_sequence_data_with_org)
        for num in ("1", "2"):
            (switcher._session_dir(num, email) / ".credentials.json").write_text("{}")

        real_replace = os.replace
        tripped: list[str] = []

        def interrupt_the_first_park(src, dst):
            if not tripped and str(dst).endswith(".swapping"):
                tripped.append(str(dst))
                raise KeyboardInterrupt
            return real_replace(src, dst)

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", interrupt_the_first_park)
            with pytest.raises(KeyboardInterrupt):
                switcher.swap_accounts("1", "2")

        assert tripped, "the injected interrupt never fired"
        assert self._marker(switcher, "1", email) == "SLOT-1-HISTORY"
        assert self._marker(switcher, "2", email) == "SLOT-2-HISTORY"
        alive = {
            num: (switcher._session_dir(num, email) / ".credentials.json").exists()
            for num in ("1", "2")
        }
        assert alive == {"1": True, "2": True}, f"session creds destroyed: {alive}"

    def test_swap_records_the_move_when_only_one_slot_has_a_profile(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """The last rename needs its own record, and only this shape shows it.

        With a profile in both slots the earlier rename already fills the sink,
        so losing the record of the last one is invisible. With one profile it
        is the only rename there is.
        """
        switcher = ClaudeAccountSwitcher()
        email = self._same_email_slots(
            switcher, sample_sequence_data_with_org, profiles=("1",)
        )

        real_replace = os.replace
        landed: list[str] = []

        def interrupt_just_past_the_last_park(src, dst):
            # One shot: the rollback's own reverse must run for real.
            real_replace(src, dst)
            if not landed and str(src).endswith(".swapping"):
                landed.append(str(dst))
                raise KeyboardInterrupt

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "replace", interrupt_just_past_the_last_park)
            with pytest.raises(KeyboardInterrupt):
                switcher.swap_accounts("1", "2")

        assert landed, "the injected interrupt never fired"
        slot_1 = switcher._session_dir("1", email) / "marker"
        assert slot_1.exists(), "account 1's profile was left under slot 2's key"
        assert slot_1.read_text() == "SLOT-1-HISTORY"
        assert not (switcher._session_dir("2", email) / "marker").exists()

    def test_swap_of_two_emails_still_reverses_the_move_on_failure(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """The gate exists for the same-email case, but it guards both.

        With two emails the four session-directory keys are distinct, so a
        reverse that never runs is silent: the profiles just stay under the
        keys the aborted swap handed them.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        emails = {
            num: sample_sequence_data["accounts"][num]["email"] for num in ("1", "2")
        }
        for num, email in emails.items():
            switcher._write_account_credentials(num, email, f"creds-{num}")
            profile = switcher._session_dir(num, email)
            profile.mkdir(parents=True, exist_ok=True)
            (profile / "marker").write_text(f"SLOT-{num}-HISTORY")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ClaudeAccountSwitcher, "_write_account_credentials", _refuse_write
            )
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")

        for num, email in emails.items():
            marker = switcher._session_dir(num, email) / "marker"
            assert marker.exists(), f"slot {num}'s profile was not moved back"
            assert marker.read_text() == f"SLOT-{num}-HISTORY"

    def test_swap_same_email_persistent_failure_keeps_staged_copy(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """When the restore writes fail too (persistent backend outage), the
        pre-swap material must survive on disk in the staged copies — not
        only in the dying process's memory."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")

        real_write = ClaudeAccountSwitcher._write_account_credentials
        calls = {"n": 0}

        def failing_write(self, num, email, creds):
            calls["n"] += 1
            if calls["n"] >= 2:
                raise OSError("disk full (injected, persistent)")
            return real_write(self, num, email, creds)

        # Scoped context: see H-1 comment above.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                ClaudeAccountSwitcher, "_write_account_credentials", failing_write
            )
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")

        # Slot 1's stored copy was never touched; slot 2's store now holds
        # the wrong material (restore failed), but the staged copy has it.
        assert switcher._read_account_credentials("1", email) == "creds-org"
        staged = switcher.credentials_dir / ".swap-staging-creds-2.json"
        assert staged.read_text(encoding="utf-8") == "creds-personal"
        if sys.platform != "win32":
            assert staged.stat().st_mode & 0o777 == 0o600

    def test_swap_same_email_rollback_restores_empty_slot(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Slot 2 was never backed up: after a failed swap, the shared key
        must read empty again — not keep account 1's credential under
        account 2's slot."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")

        def failing_write_json(self, path, data):
            raise OSError("disk full (injected)")

        # Scoped context: see H-1 comment above.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(ClaudeAccountSwitcher, "_write_json", failing_write_json)
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")

        assert switcher._read_account_credentials("1", email) == "creds-org"
        assert switcher._read_account_credentials("2", email) == ""
        data = switcher._get_sequence_data()
        assert data["accounts"]["1"]["organizationUuid"] == "org-uuid-5678"
        # Clean rollback: no staged copies left behind either.
        assert not list(switcher.credentials_dir.glob(".swap-staging-*"))

    def test_write_json_publishes_only_after_chmod(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """chmod runs on the temp file, making the rename the final commit —
        a chmod failure must abort *without* publishing, otherwise callers
        would roll files back around already-committed metadata."""
        if sys.platform == "win32":
            pytest.skip("_write_json skips chmod on Windows (no POSIX file modes)")
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        before = switcher.sequence_file.read_text(encoding="utf-8")

        def failing_chmod(path, mode):
            raise OSError("chmod denied (injected)")

        # Scoped context: see H-1 comment above.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("claude_swap.switcher.os.chmod", failing_chmod)
            with pytest.raises(OSError):
                switcher._write_json(switcher.sequence_file, {"x": 1})

        assert switcher.sequence_file.read_text(encoding="utf-8") == before

    def test_swap_same_email_one_sided_clears_destination(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Same email, only slot 1 backed up: after the swap, the unbacked
        account's new key must read empty — with fully overlapping keys the
        old key is never separately deleted, so it must be actively cleared,
        not skipped, or account 2 would serve account 1's credential."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")

        switcher.swap_accounts("1", "2")

        # Account 1 (backed) now lives in slot 2 with its credential;
        # account 2 (unbacked) now lives in slot 1 and must stay unbacked.
        assert switcher._read_account_credentials("2", email) == "creds-org"
        assert switcher._read_account_credentials("1", email) == ""
        data = switcher._get_sequence_data()
        assert data["accounts"]["2"]["organizationUuid"] == "org-uuid-5678"

    def test_swap_clears_stale_destination_key(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """Distinct emails, source unbacked: a stale file leaked under the
        destination key (e.g. by an earlier crash) must not be adopted."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        # Account 1 has no backup; plant a stale foreign file under the key
        # it will occupy after the swap: (slot 2, account1's email).
        switcher._write_account_credentials(
            "2", "account1@example.com", "stale-foreign"
        )

        switcher.swap_accounts("1", "2")

        assert switcher._read_account_credentials("2", "account1@example.com") == ""

    def test_swap_refuses_leftover_staging(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Leftover staging from an interrupted swap may be the only copy of
        a credential: a retry must refuse loudly, never overwrite it."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")
        leftover = switcher.credentials_dir / ".swap-staging-creds-1.json"
        leftover.write_text("only-surviving-copy", encoding="utf-8")

        with pytest.raises(ConfigError, match="interrupted swap"):
            switcher.swap_accounts("1", "2")

        # The leftover is untouched and nothing was swapped.
        assert leftover.read_text(encoding="utf-8") == "only-surviving-copy"
        assert switcher._read_account_credentials("1", email) == "creds-org"
        assert switcher._read_account_credentials("2", email) == "creds-personal"
        data = switcher._get_sequence_data()
        assert data["accounts"]["1"]["organizationUuid"] == "org-uuid-5678"

    def test_swap_failed_required_clear_aborts_commit(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Same-email one-sided swap where the required clear fails: the swap
        must abort pre-commit and roll back, instead of committing with
        account 1's credential still readable under the shared key."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")

        real_unlink = Path.unlink

        def failing_unlink(path, *args, **kwargs):
            if path.name.startswith(".creds-1-"):
                raise OSError("permission denied (injected)")
            return real_unlink(path, *args, **kwargs)

        # Scoped context: see H-1 comment above.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Path, "unlink", failing_unlink)
            with pytest.raises(CredentialError, match="aborting before commit"):
                switcher.swap_accounts("1", "2")

        # Table unrenumbered, slot 1's credential intact, and the rollback
        # reverted the half-written copy under the shared key.
        data = switcher._get_sequence_data()
        assert data["accounts"]["1"]["organizationUuid"] == "org-uuid-5678"
        assert switcher._read_account_credentials("1", email) == "creds-org"
        assert switcher._read_account_credentials("2", email) == ""

    def test_swap_same_email_clears_prev_generations(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Writing through the overlapping keys retains the displaced
        account's credential as each key's .prev generation; after the commit
        those must be gone — recovery must never resurrect another account's
        token onto a slot."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")

        switcher.swap_accounts("1", "2")

        assert not list(switcher.credentials_dir.glob("*.enc.prev"))

    def test_swap_holds_account_lock(
        self, temp_home: Path, sample_sequence_data: dict, monkeypatch
    ):
        """The whole mutation runs under the same lock switch/persist take."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        entered: list[object] = []

        class SpyLock:
            def __init__(self, path):
                self.path = path

            def __enter__(self):
                entered.append(self.path)
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr("claude_swap.switcher.FileLock", SpyLock)
        switcher.swap_accounts("1", "2")

        assert entered == [switcher.lock_file]

    def test_swap_moves_session_profiles(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        session_a = switcher._session_dir("1", "account1@example.com")
        session_a.mkdir(parents=True)
        (session_a / "marker.txt").write_text("history-of-account-one")

        switcher.swap_accounts("1", "2")

        moved = switcher._session_dir("2", "account1@example.com")
        assert (moved / "marker.txt").read_text() == "history-of-account-one"
        assert not session_a.exists()


class TestSwapUnreadableSourceIsNotAbsent:
    """Same defect family as C1/C2/move: the plain reader's ``""`` means both
    "no backup" and "the backup exists but could not be read right now".

    The pre-swap read (:1063-1064) used the plain reader — a permission
    glitch on either slot's ``.enc`` read as "no backup", and the swap
    committed BOTH destination keys from that snapshot: the unreadable
    slot's live refresh token would be silently dropped and replaced with
    an empty credential at its new number. Fixed with
    ``_read_account_credentials_ex``, aborting BEFORE anything moves.
    """

    def _write(self, switcher, data):
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, data)

    @pytest.mark.skipif(
        sys.platform == "win32" or os.geteuid() == 0,
        reason="needs POSIX permission semantics (non-root)",
    )
    def test_unreadable_enc_aborts_the_swap_before_anything_changes(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)
        switcher._write_account_credentials("1", "account1@example.com", "rt-1")
        switcher._write_account_credentials("2", "account2@example.com", "rt-2")

        # CONTROL: both readable, the swap lands cleanly (instrument says YES).
        switcher.swap_accounts("1", "2")
        assert (
            switcher._read_account_credentials("2", "account1@example.com")
            == "rt-1"
        )
        assert (
            switcher._read_account_credentials("1", "account2@example.com")
            == "rt-2"
        )
        # Swap back to the original layout for the probe below.
        switcher.swap_accounts("1", "2")

        enc = switcher._backup_enc_path("2", "account2@example.com")
        enc.chmod(0o000)
        try:
            with pytest.raises(ConfigError, match="could not be read"):
                switcher.swap_accounts("1", "2")
        finally:
            if enc.exists():
                enc.chmod(0o600)

        # Nothing committed: both accounts intact under their original
        # numbers, account 2 still holding its readable credential.
        data = switcher._get_sequence_data()
        assert data["accounts"]["1"]["email"] == "account1@example.com"
        assert data["accounts"]["2"]["email"] == "account2@example.com"
        assert (
            switcher._read_account_credentials("1", "account1@example.com")
            == "rt-1"
        )
        assert (
            switcher._read_account_credentials("2", "account2@example.com")
            == "rt-2"
        )

    def test_absent_source_still_swaps(
        self, temp_home: Path, sample_sequence_data: dict
    ):
        """Control in the other direction: genuinely unbacked slots (no
        .enc at all) are not mistaken for unreadable and still swap."""
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data)

        num_a, num_b = switcher.swap_accounts("1", "2")

        assert (num_a, num_b) == ("1", "2")
        data = switcher._get_sequence_data()
        assert data["accounts"]["1"]["email"] == "account2@example.com"
        assert data["accounts"]["2"]["email"] == "account1@example.com"
