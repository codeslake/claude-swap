"""Tests for `cswap swap` (ClaudeAccountSwitcher.swap_accounts)."""

import contextlib
import errno
import copy
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



@contextlib.contextmanager
def caplog_at_error():
    """Collect ERROR records from the switcher logger as plain strings."""
    import logging

    seen: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    logger = logging.getLogger("claude-swap")
    h = _Sink(level=logging.ERROR)
    logger.addHandler(h)
    try:
        yield seen
    finally:
        logger.removeHandler(h)


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

    def test_a_rollback_keeps_prev_generations_it_never_contaminated(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """The purge is only correct when the two slots share one email.

        The forward pass writes `(num_b, email_a)` and `(num_a, email_b)`; the
        purge deletes `.prev` for `(num_a, email_a)` and `(num_b, email_b)`.
        With two emails those key sets are DISJOINT, so the purge destroys a
        generation the swap never touched — and the stored credentials are
        still correct afterwards, so nothing signals the loss.
        """
        switcher = ClaudeAccountSwitcher()
        # TWO EMAILS, which is the whole case: the shared-email fixture is the
        # one shape where the purge's key set and the forward pass's coincide.
        data = copy.deepcopy(sample_sequence_data_with_org)
        emails = {"1": "account1@example.com", "2": "account2@example.com"}
        for num, email in emails.items():
            data["accounts"][num]["email"] = email
            data["accounts"][num]["uuid"] = f"uuid-{num}"
        self._write(switcher, data)
        for num, email in emails.items():
            switcher._write_account_credentials(num, email, f"gen1-{num}")
            switcher._write_account_credentials(num, email, f"gen2-{num}")

        prev = {
            num: switcher._store._prev_backup_path(num, email)
            for num, email in emails.items()
        }
        assert all(p.exists() for p in prev.values()), (
            "the fixture never produced a .prev, so this test would pass "
            "however the purge behaves"
        )

        # THE CONTROL. Without it a green result cannot separate "the purge
        # spared them" from "the rollback never reached the purge".
        purged: list[tuple[str, str]] = []
        real_purge = switcher._store.delete_previous_backup
        switcher._store.delete_previous_backup = (
            lambda n, e: (purged.append((n, e)), real_purge(n, e))[1])

        def failing_write(*_a, **_kw):
            raise ConfigError("commit failed")

        original = switcher._write_json
        switcher._write_json = failing_write
        try:
            with pytest.raises(ConfigError):
                switcher.swap_accounts("1", "2")
        finally:
            switcher._write_json = original
            switcher._store.delete_previous_backup = real_purge
        assert purged, (
            "the rollback never reached the purge, so this case measures "
            "nothing about it"
        )

        alive = {num: p.exists() for num, p in prev.items()}
        assert alive == {"1": True, "2": True}, (
            f"the rollback purged .prev for slots the swap never wrote "
            f"through: {alive}; purge calls were {purged}"
        )

    def test_a_staging_abort_never_enters_the_rollback(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """The hoist's own property, which its two observables do not pin.

        Staging sits outside the `try` so an abort there cannot reach the
        rollback at all. The existing case asserts untouched markers and live
        session credentials, but the `moved` and `wrote_backups` gates produce
        both of those independently — measured, moving the staging block back
        INSIDE the `try` leaves the whole suite green. What only the hoist
        gives is that the rollback is never entered, so that is what this
        asserts.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")
        # A leftover staging file is what makes staging refuse.
        (switcher.credentials_dir / ".swap-staging-creds-1.json").write_text("{}")

        entered: list[bool] = []
        real = switcher._rollback_swap
        switcher._rollback_swap = lambda *a, **kw: (
            entered.append(True), real(*a, **kw))[1]
        try:
            with pytest.raises(ConfigError):
                switcher.swap_accounts("1", "2")
        finally:
            switcher._rollback_swap = real

        assert entered == [], (
            "a staging abort reached the rollback — it reverses profiles and "
            "rewrites both slots' credentials for a failure that mutated "
            "nothing"
        )

    def test_a_rollback_that_restores_nothing_does_not_say_it_restored(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """The opening line is emitted before anything is decided.

        With `wrote_backups=False` and no moves, every step below it is
        skipped — the reverse, the restores, the cleanup and the purge — and
        the log still reads "restoring both slots". This PR opens by listing
        three pieces of text that told the user the opposite of what happened.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        switcher._write_account_credentials("1", email, "creds-org")
        switcher._write_account_credentials("2", email, "creds-personal")

        with caplog_at_error() as records:
            switcher._rollback_swap(
                "1", email, "creds-org", "{}",
                "2", email, "creds-personal", "{}",
                staging={}, moved=[], wrote_backups=False,
            )
        said = " ".join(records)
        assert "restoring both slots" not in said, (
            f"it announced a restore it then skipped entirely: {said!r}"
        )
        assert said, "nothing was logged at all — the rollback went silent"

    def test_a_rollback_keeps_prev_when_no_forward_write_landed(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """SAME email, failure on the FIRST forward write. The `overlap` gate
        does not reach this one.

        `wrote_backups` is armed one statement BEFORE the first write, and the
        comment prices that at "one needless restore". The restore is a no-op
        by construction — `_retain_previous_backup` short-circuits when the
        value it would displace equals the one going in — so no `.prev` is
        created, the purge's premise ("the restore writes pushed the
        half-written material into the retained generations") is false, and it
        deletes the generation that was there before the swap began.

        The condition that separates it from a legitimate purge is not the
        emails: it is whether a restore DISPLACED anything.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"gen1-{num}")
            switcher._write_account_credentials(num, email, f"gen2-{num}")

        prev = {n: switcher._store._prev_backup_path(n, email) for n in ("1", "2")}
        assert all(p.exists() for p in prev.values()), (
            "the fixture produced no .prev, so this would pass however the "
            "purge behaves"
        )

        calls = {"n": 0}
        real_write = switcher._write_account_credentials

        def fail_first(num, mail, creds):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(errno.EIO, "the first forward write failed")
            return real_write(num, mail, creds)

        switcher._write_account_credentials = fail_first
        try:
            # OSError, not ConfigError: the forward write is not wrapped, and
            # the rollback runs on the way out either way.
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")
        finally:
            del switcher._write_account_credentials

        alive = {n: p.exists() for n, p in prev.items()}
        assert alive == {"1": True, "2": True}, (
            f"the rollback purged .prev after a swap that wrote nothing: {alive}"
        )

    def test_a_profile_only_rollback_does_not_say_it_restored_the_slots(
        self, temp_home: Path, sample_sequence_data_with_org: dict, caplog
    ):
        """The announcement had three states and only said two.

        `bool(moved) or wrote_backups` reads as "restoring both slots" when a
        rename landed and no credential write ever ran -- the fourth instance
        of the class this change opens by naming. The existing case covers
        only the `(False, [])` corner, where the text is already right.
        """
        import logging

        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)

        def half_moved(a, ea, b, eb, moved):
            moved.append(temp_home / "one-profile-already-renamed")
            raise OSError(errno.EIO, "interrupted just past a rename")

        switcher._swap_session_dirs = half_moved
        try:
            with caplog.at_level(logging.ERROR, logger="claude-swap"):
                with pytest.raises(OSError):
                    switcher.swap_accounts("1", "2")
        finally:
            del switcher._swap_session_dirs

        # "failed; rolling back", not "failed mid-write": the `try` this unwinds
        # from BEGINS with the profile move, so a signal there wrote nothing
        # and "mid-write" was itself the class of wrong text this case guards.
        said = [r.message for r in caplog.records if "rolling back" in r.message]
        assert said, "premise: the rollback never announced anything"
        assert "restoring both slots" not in said[0], (
            f"no credential write ran, and the line says otherwise: {said[0]!r}"
        )

    def test_a_swap_that_stored_nothing_keeps_both_session_profiles(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """A restore that changes no value still invalidated a live profile.

        `wrote_backups` is armed one statement before the first write on
        purpose -- arming it after would let an abort skip a restore that was
        owed -- and the comment prices the gap at "one needless restore". It
        is not one and it is not free: all four restores run, each credential
        restore routes through `_post_backup_write`, and both slots lose their
        session credential material for a swap where zero writes landed. That
        is the same harm `..._touches_neither_slot` asserts against.

        The restore is a no-op by construction here: the key still holds
        exactly what the restore would write. A write that changes nothing
        should not cost a re-bootstrap.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"gen1-{num}")

        seeded = {}
        for num in ("1", "2"):
            d = switcher._session_dir(num, email)
            d.mkdir(parents=True, exist_ok=True)
            f = d / ".credentials.json"
            f.write_text('{"claudeAiOauth": {"accessToken": "session-tok"}}')
            seeded[num] = f
        assert all(f.exists() for f in seeded.values()), "premise: nothing seeded"

        calls = {"n": 0}
        real_write = switcher._write_account_credentials

        def fail_first(num, mail, creds):
            # ONLY THE FORWARD WRITE. Replacing the method outright would
            # intercept the four RESTORE writes too, which is the behaviour
            # under test.
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(errno.EIO, "the first forward write failed")
            return real_write(num, mail, creds)

        switcher._write_account_credentials = fail_first
        try:
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")
        finally:
            del switcher._write_account_credentials

        gone = [n for n, f in seeded.items() if not f.exists()]
        assert gone == [], (
            f"slots {gone} lost their session credentials to a swap that "
            "stored nothing"
        )

    def test_one_reader_decides_whether_a_restore_displaced_anything(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """Two reads of one value can disagree, and the disagreement strands.

        The purge used to probe the stored credential itself and then let
        `_retain_previous_backup` read it again a moment later. Deny only the
        probe -- an unreadable Keychain that clears between the two calls is
        the ordinary way -- and `displaced` stays empty while a `.prev` IS
        written. Nothing then purges a retained generation that holds the
        other slot's half-written material, which is the state
        `delete_previous_backup` exists to remove.

        One email, so the restore keys and the forward keys coincide and the
        retained generation really is contamination; failure at the COMMIT, so
        all four forward writes landed. The store's retention verdict is the
        only reader now, so there is no second answer to disagree with.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"gen1-{num}")
            switcher._write_account_credentials(num, email, f"gen2-{num}")
            switcher._store.delete_previous_backup(num, email)
        prev = {n: switcher._store._prev_backup_path(n, email) for n in ("1", "2")}
        assert not any(p.exists() for p in prev.values()), (
            "premise: a .prev already exists, so a survivor below proves nothing"
        )

        real_read = switcher._store._read_account_credentials_ex
        rolling_back = {"yet": False}
        seen: set[tuple[str, str]] = set()

        def first_rollback_read_is_unreadable(num, mail, *a, **kw):
            # ONLY DURING THE ROLLBACK. The forward pass reads the same keys to
            # find the material it is moving; denying those aborts the swap
            # before it reaches the state this case is about.
            if rolling_back["yet"] and (num, mail) not in seen:
                seen.add((num, mail))
                return "", True
            return real_read(num, mail, *a, **kw)

        def refuse_commit(path, data):
            rolling_back["yet"] = True
            raise OSError(errno.EIO, "the commit failed")

        switcher._store._read_account_credentials_ex = first_rollback_read_is_unreadable
        switcher._write_json = refuse_commit
        try:
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")
        finally:
            del switcher._write_json
            del switcher._store._read_account_credentials_ex
        assert seen, "premise: no read was denied, so nothing could disagree"

        survivors = {
            n: switcher._store._read_previous_backup(n, email)
            for n in ("1", "2") if prev[n].exists()
        }
        foreign = {n: v for n, v in survivors.items()
                   if v and not v.endswith(f"-{n}")}
        assert foreign == {}, (
            "a recovery generation was left holding the other slot's "
            f"half-written material: {sorted(foreign)}"
        )

    def test_the_purge_drops_only_the_keys_a_restore_displaced(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """`displaced` is built per key and then acted on for both.

        Line 1481 only truth-tests the set; the loop below it had no
        membership check, so any single displaced key purged every key. Same
        email, failure on the SECOND forward credential write: one key was
        written through and one was not. The written key's retained generation
        really is the half-written forward material and may go; the untouched
        key's is the user's genuine pre-swap copy and its restore, being a
        value-for-value no-op, creates nothing to replace it.

        The discriminator is which keys a FORWARD write reached, which the
        injection below records rather than assumes.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"gen1-{num}")
            switcher._write_account_credentials(num, email, f"gen2-{num}")

        prev = {n: switcher._store._prev_backup_path(n, email) for n in ("1", "2")}
        assert all(p.exists() for p in prev.values()), (
            "the fixture produced no .prev, so this would pass however the "
            "purge behaves"
        )

        calls = {"n": 0}
        forward: set[str] = set()
        real_write = switcher._write_account_credentials
        rolling_back = {"yet": False}

        def fail_second(num, mail, creds):
            calls["n"] += 1
            if calls["n"] == 2:
                rolling_back["yet"] = True
                raise OSError(errno.EIO, "the second forward write failed")
            if not rolling_back["yet"]:
                forward.add(num)
            return real_write(num, mail, creds)

        switcher._write_account_credentials = fail_second
        try:
            with pytest.raises(OSError):
                switcher.swap_accounts("1", "2")
        finally:
            del switcher._write_account_credentials

        untouched = sorted(set(prev) - forward)
        assert untouched, (
            "premise: every key took a forward write, so a blanket purge and "
            "a per-key one cannot be told apart here"
        )
        lost = [n for n in untouched if not prev[n].exists()]
        assert lost == [], (
            f"the purge dropped .prev for {lost}, which no forward write "
            "reached — that is the user's pre-swap generation, not "
            "contamination this swap created"
        )

    def test_a_rollback_drops_prev_it_really_did_contaminate(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """The other half, and without it the purge is untested.

        Measured: with the purge disabled outright the file stays green, so
        every case here pins only that it does NOT fire. Same email, failure
        at the COMMIT so all four forward writes landed — the restores then
        push that half-written material into each key's retained generation,
        and those really are contamination.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"gen1-{num}")
            switcher._write_account_credentials(num, email, f"gen2-{num}")

        prev = {n: switcher._store._prev_backup_path(n, email) for n in ("1", "2")}
        assert all(p.exists() for p in prev.values()), "no .prev to drop"

        def failing_commit(*_a, **_kw):
            raise ConfigError("commit failed")

        switcher._write_json = failing_commit
        try:
            with pytest.raises(ConfigError):
                switcher.swap_accounts("1", "2")
        finally:
            del switcher._write_json

        alive = {n: p.exists() for n, p in prev.items()}
        assert alive == {"1": False, "2": False}, (
            f"a generation the rollback itself contaminated was kept: {alive}"
        )
        assert switcher._read_account_credentials("1", email) == "gen2-1"
        assert switcher._read_account_credentials("2", email) == "gen2-2"


def _session_token_of(num: str) -> str:
    return '{"claudeAiOauth": {"accessToken": "SESSION-TOKEN-OF-SLOT-%s"}}' % num


class TestTheReverseCanFailAndTheSkipMustSeeIt:
    """`_swap_session_dirs` swallows `OSError` by design, so the rollback's
    reverse can put back FEWER profiles than the forward move took.

    The value-equal skip then finds both backups already holding their
    originals, writes nothing, and `_post_backup_write` never invalidates the
    profiles that are still crossed -- so both slots keep serving each other's
    session token, live, with no warning that says so. Base was clean here
    only because its unconditional restore write masked it; this branch
    removed the masking without replacing it.
    """

    def _seed(self, switcher, data):
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, data)

    def test_a_leftover_swapping_dir_leaves_both_profiles_crossed(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """NO INJECTED FAILURE. The only seeded state is a leftover
        `.swapping` directory -- which nothing in the codebase removes and an
        interrupt on any earlier swap leaves behind. It makes the park step
        raise a real ENOTEMPTY from a real `os.replace`.
        """
        switcher = ClaudeAccountSwitcher()
        self._seed(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"creds-{num}")
            d = switcher._session_dir(num, email)
            d.mkdir(parents=True, exist_ok=True)
            (d / "marker").write_text(f"SLOT-{num}-HISTORY")
            (d / ".credentials.json").write_text(_session_token_of(num))

        strand = switcher._session_dir("2", email)
        strand = strand.with_name(strand.name + ".swapping")
        strand.mkdir(parents=True)
        (strand / "leftover").write_text("from an earlier interrupt")

        calls = {"n": 0}
        real_write = switcher._write_account_credentials

        def fail_the_first(num, mail, creds):
            calls["n"] += 1
            if calls["n"] == 1:
                raise CredentialError("injected: the swap dies on its first write")
            return real_write(num, mail, creds)

        switcher._write_account_credentials = fail_the_first
        with pytest.raises((CredentialError, ConfigError)):
            switcher.swap_accounts("1", "2")
        switcher._write_account_credentials = real_write

        serving = {}
        for num in ("1", "2"):
            f = switcher._session_dir(num, email) / ".credentials.json"
            if f.exists():
                serving[num] = f.read_text()
        wrong = {
            num: text for num, text in serving.items()
            if text and f"SLOT-{num}" not in text
        }
        assert not wrong, (
            "a session profile was left serving another slot's token with no "
            f"invalidation: {wrong}"
        )


    def test_an_interrupt_before_the_first_write_still_uncrosses(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """`wrote_backups` alone cannot gate the repair.

        `moved` is appended from a `finally` precisely so a signal past a
        rename is recorded, and `KeyboardInterrupt` is not `OSError` -- so it
        leaves `_swap_session_dirs` with `moved` full while `wrote_backups` is
        still False. Gating `restores` on `wrote_backups` alone then skipped
        every restore, the repair never ran, and both profiles stayed crossed
        and live. Base was clean here because its restore was unconditional.
        """
        from claude_swap import switcher as switcher_mod

        switcher = ClaudeAccountSwitcher()
        self._seed(switcher, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"creds-{num}")
            d = switcher._session_dir(num, email)
            d.mkdir(parents=True, exist_ok=True)
            (d / ".credentials.json").write_text(_session_token_of(num))

        strand = switcher._session_dir("2", email)
        strand = strand.with_name(strand.name + ".swapping")
        strand.mkdir(parents=True)
        (strand / "leftover").write_text("from an earlier interrupt")

        real_replace = switcher_mod.os.replace
        calls = {"n": 0}

        def interrupt_after_the_last_forward_move(src, dst, *a, **kw):
            out = real_replace(src, dst, *a, **kw)
            calls["n"] += 1
            if calls["n"] == 3:      # both profiles now under each other's keys
                raise KeyboardInterrupt
            return out

        switcher_mod.os.replace = interrupt_after_the_last_forward_move
        try:
            with pytest.raises(KeyboardInterrupt):
                switcher.swap_accounts("1", "2")
        finally:
            switcher_mod.os.replace = real_replace

        serving = {}
        for num in ("1", "2"):
            f = switcher._session_dir(num, email) / ".credentials.json"
            if f.exists():
                serving[num] = f.read_text()
        wrong = {num: v for num, v in serving.items()
                 if v and f"SLOT-{num}" not in v}
        assert not wrong, (
            "an abort before the first credential write left a profile serving "
            f"another slot's token with no invalidation: {wrong}"
        )


class TestTheRetentionVerdictIsTheOneReader:
    """The purge that deletes the user's only recovery generation keys on this
    single boolean, and flipping its unchanged-value arm to `True` left the
    whole suite green -- the three guards above it (`wrote_backups`, the
    value-equal skip, `displaced`) are mutually redundant, so any one of them
    can be deleted invisibly.
    """

    def test_an_unchanged_write_displaces_nothing(self, temp_home: Path):
        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        email = "user@example.com"
        assert switcher._write_account_credentials("1", email, "gen-1") is False, (
            "premise: the FIRST write has nothing to displace"
        )
        assert switcher._write_account_credentials("1", email, "gen-2") is True, (
            "positive control: replacing a different value must retain a .prev"
        )
        assert switcher._write_account_credentials("1", email, "gen-2") is False, (
            "a write of the value already stored claimed it displaced "
            "something, so the rollback purge would delete a .prev this "
            "write never created -- the user's only recovery generation"
        )


class TestTheRollbackSummaryReportsWhatRan:
    """`wrote_backups` is armed one statement BEFORE the first write, so
    "armed" and "wrote something" are different claims.

    The old opening line made the second claim from the first fact: with both
    credential restores skipped it still said "restoring both slots", and the
    reversal line fired before a reverse that then moved nothing. Both are
    reachable, and the existing cases only assert a string is ABSENT -- which
    a rewording satisfies without making the text true.
    """

    def test_a_rollback_that_restored_nothing_says_so(
        self, temp_home: Path, sample_sequence_data_with_org: dict, caplog
    ):
        import logging

        switcher = ClaudeAccountSwitcher()
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, sample_sequence_data_with_org)
        email = "user@example.com"
        for num in ("1", "2"):
            switcher._write_account_credentials(num, email, f"creds-{num}")

        real_write = switcher._write_account_credentials

        def die_on_the_first(num, mail, creds):
            raise CredentialError("injected: the swap dies on its first write")

        switcher._write_account_credentials = die_on_the_first
        with caplog.at_level(logging.ERROR, logger="claude-swap"):
            with pytest.raises((CredentialError, ConfigError)):
                switcher.swap_accounts("1", "2")
        switcher._write_account_credentials = real_write

        said = "\n".join(r.getMessage() for r in caplog.records)
        assert "rollback:" in said, (
            "premise: no summary was emitted at all, so this proves nothing"
        )
        assert "credentials were restored" not in said, (
            "the rollback reported restoring credentials while every restore "
            f"was skipped or refused:\n{said}"
        )
        assert "Reversed the session-profile exchange" not in said, (
            "a reversal that moved nothing was announced as having happened:"
            f"\n{said}"
        )


class TestTheRollbackDecidesPerKey:
    """Which SLOT is still crossed, and which report the summary owes."""

    def _write(self, switcher, data):
        switcher._setup_directories()
        switcher._write_json(switcher.sequence_file, data)

    def _one_crossed_one_home(self, switcher, mail_a, mail_b):
        """Disk state where exactly ONE of the two profiles is still crossed.

        Slot B's profile sits in its crossed location and its home is
        occupied, so the reverse refuses to move it back. Slot A never had a
        profile, so nothing of A's ever left home.
        """
        crossed_b = switcher._session_dir("1", mail_b)
        crossed_b.mkdir(parents=True, exist_ok=True)
        switcher._session_dir("2", mail_b).mkdir(parents=True, exist_ok=True)
        return [crossed_b]

    def test_a_landed_staging_move_is_not_re_attempted(
        self, temp_home: Path, sample_sequence_data_with_org: dict, monkeypatch
    ):
        """`staging` names a path that stops existing once its move lands.

        Left set, the outer `finally`'s recovery `os.replace` runs on every
        SUCCESSFUL exchange and is carried by its own `except OSError` — so
        the happy path leans on an error handler rather than on there being
        no error, and a real ENOENT there is indistinguishable from that.
        """
        from claude_swap import switcher as switcher_mod

        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        mail_a, mail_b = "a@example.com", "b@example.com"
        switcher._session_dir("1", mail_a).mkdir(parents=True, exist_ok=True)
        switcher._session_dir("2", mail_b).mkdir(parents=True, exist_ok=True)

        real_replace = os.replace
        from_staging: list[str] = []

        def recording(src, dst, *a, **k):
            if str(src).endswith(".swapping"):
                from_staging.append(str(dst))
            return real_replace(src, dst, *a, **k)

        monkeypatch.setattr(switcher_mod.os, "replace", recording)
        moved: list = []
        switcher._swap_session_dirs("1", mail_a, "2", mail_b, moved)

        assert moved, "premise: the forward exchange moved nothing"
        assert len(from_staging) == 1, (
            "the staging name was replaced twice on a successful exchange — "
            "the strand recovery re-ran and only its `except OSError` hid "
            f"it: {from_staging}"
        )

    def test_only_the_slot_still_crossed_is_forced_through_a_write(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """`moved` is not a per-key fact, and neither is "some move happened".

        A reverse that put back one of two leaves the OTHER slot's profile
        crossed. Marking BOTH keys crossed forces a credential write on a
        slot whose profile never moved, and every credential write routes
        through `_post_backup_write` -- so a correctly-untouched slot loses
        its session credentials for a swap that did nothing to it.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        mail_a, mail_b = "a@example.com", "b@example.com"
        switcher._write_account_credentials("1", mail_a, "creds-a")
        switcher._write_account_credentials("2", mail_b, "creds-b")
        moved = self._one_crossed_one_home(switcher, mail_a, mail_b)

        wrote: list[str] = []
        real = switcher._write_account_credentials

        def record(num, mail, creds):
            wrote.append(num)
            return real(num, mail, creds)

        switcher._write_account_credentials = record
        try:
            switcher._rollback_swap(
                "1", mail_a, "creds-a", "{}",
                "2", mail_b, "creds-b", "{}",
                staging={}, moved=moved, wrote_backups=True,
            )
        finally:
            switcher._write_account_credentials = real

        assert "2" in wrote, (
            "premise: the slot that IS still crossed was not forced through a "
            f"write, so this case cannot see the other one either: {wrote}"
        )
        assert "1" not in wrote, (
            "a slot whose profile never left home was forced through a "
            "credential restore of the value already under it, which costs "
            f"it its session profile: writes were {wrote}"
        )

    def test_a_rollback_whose_every_restore_raised_does_not_say_nothing_ran(
        self, temp_home: Path, sample_sequence_data_with_org: dict
    ):
        """`wrote_any` is set AFTER the write, so a raising restore leaves it
        False -- and the summary then read that as "nothing was written".

        Those are opposite reports. Nothing-was-written needs no action; every
        restore failing is what keeps the staged copies on disk for manual
        recovery, and the line is the only place that says so.
        """
        switcher = ClaudeAccountSwitcher()
        self._write(switcher, sample_sequence_data_with_org)
        mail_a, mail_b = "a@example.com", "b@example.com"
        switcher._write_account_credentials("1", mail_a, "creds-a")
        switcher._write_account_credentials("2", mail_b, "creds-b")
        moved = self._one_crossed_one_home(switcher, mail_a, mail_b)

        attempted: list[str] = []

        def die(num, mail, creds):
            attempted.append(num)
            raise CredentialError("injected: the restore cannot land")

        real = switcher._write_account_credentials
        switcher._write_account_credentials = die
        try:
            with caplog_at_error() as records:
                switcher._rollback_swap(
                    "1", mail_a, "creds-a", "{}",
                    "2", mail_b, "creds-b", "{}",
                    staging={}, moved=moved, wrote_backups=True,
                )
        finally:
            switcher._write_account_credentials = real

        said = " ".join(records)
        assert attempted, (
            "premise: no restore was even attempted, so there was nothing "
            "that could have failed"
        )
        assert "rollback:" in said, "premise: no summary was emitted at all"
        summary = said.split("rollback:")[-1]
        assert "credentials were restored" not in summary, (
            f"a restore that raised was reported as a restore: {said!r}"
        )
        # THE ARM'S OWN CONTENT, not the absence of a word from a superseded
        # draft. Keying on "nothing" was satisfied by the arm AND by its
        # deletion, because the fallback wording carries no "nothing" either.
        assert "failed" in summary, (
            "every restore failed and the summary does not say so, so a "
            f"reader cannot tell it from a rollback with nothing to do: {said!r}"
        )
        assert "staged copies are kept" not in summary, (
            "these two slots have distinct emails, so nothing was staged — "
            f"the line names a recovery that does not exist: {said!r}"
        )
