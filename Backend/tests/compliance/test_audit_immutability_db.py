"""backend/tests/compliance/test_audit_immutability_db.py

Role: The two audit-immutability controls that exist only in PostgreSQL — the append-only
triggers and the revoked `UPDATE`/`DELETE`/`TRUNCATE` grants — plus the forged-row case that
proves hash verification localizes damage.

`audit-log-immutability.md` asks for three independent controls, and the other two are already
covered without a database: `AuditRepository` has no update or delete method
(`backend/tests/security/test_audit_logs.py` asserts the absence at source level), and
`AuditService` never offers one. Those tests pass against a database that has no triggers at
all. This module is the only place the database's own refusal is exercised, which is why it is
its own file with its own marker rather than three more assertions somewhere else.

Every test here skips when no PostgreSQL answers. **A skip is not a pass.** `db_available`
exists so the rest of the suite runs on a laptop, and `conftest.py`'s docstring explains the
engine disposal that keeps the second `requires_db` test in a run from skipping spuriously.
Whether these were executed for a given release belongs in `docs/RELEASE.md` — not inferred
from a green summary line.

Not executed on the machine this was written on
----------------------------------------------
No PostgreSQL was reachable here, so these assertions have never run. Two consequences shape
how they are written. Each refusal test asserts the *effect* — the row is unchanged when read
back in a new session — and treats the SQLSTATE as corroboration, because an untested exact
error code is a guess about which of two controls fired first, while an unchanged row is the
requirement itself. And `TRUNCATE` is never attempted: see below.

Why `TRUNCATE` is asserted from the catalog and never executed
-------------------------------------------------------------
`UPDATE` and `DELETE` are safe to attempt under any role — the row triggers fire for a
superuser too, so the statement is refused rather than merely unprivileged. `TRUNCATE` has no
trigger (a truncate guard has to be `FOR EACH STATEMENT`, and migration `0002_auth_audit`
installs only the two row triggers). Its single control is the revoked grant, which a superuser
bypasses. Attempting it while connected as `postgres` — the ordinary local setup — would empty
the audit table of the test database to prove a control that was not in force. So the grant is
read from `pg_class.relacl` instead, and the gap is named by
`test_truncate_has_no_trigger_behind_the_grant` rather than left for a reader to notice.

Closing that gap means a `BEFORE TRUNCATE ... FOR EACH STATEMENT` trigger in a new migration.
That is a schema change, so it is proposed in `docs/RELEASE.md` and not made here.

What these tests leave behind
-----------------------------
Rows, deliberately. Each test appends to `audit_log` under a scratch tenant and cannot clean up
afterwards, because cleanup is precisely what the controls under test prevent — and
`audit_log.tenant_id` is `ON DELETE RESTRICT`, so the scratch tenant cannot be removed either.
The tenants are named `audit-probe-*` so a reviewer looking at `docquery_test` can tell test
litter from data. This is an argument for pointing `DATABASE_URL` at a throwaway database when
running the suite, which `docs/RUNNING.md` already says.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, text

from backend.models.audit_log import (
    GENESIS_HASH,
    AuditAction,
    AuditLog,
    AuditObjectType,
    AuditOutcome,
)
from backend.repositories.audit_repository import AuditRepository
from backend.services.audit_service import AuditService, compute_integrity_hash

pytestmark = pytest.mark.requires_db

#: SQLSTATEs either control can raise. `42501` is the privilege check (the REVOKE) and `23001`
#: is `restrict_violation`, which migration `0002_auth_audit` sets on the trigger's exception.
#: Which one arrives depends on the role the test connected as; both satisfy the requirement.
REFUSAL_SQLSTATES = frozenset({"42501", "23001"})

#: Substrings that identify a refusal in the message, for the case where the driver does not
#: surface a SQLSTATE. `AUDIT_IMMUTABLE_MESSAGE` in the migration is the first one.
REFUSAL_PHRASES = ("append-only", "permission denied", "must be owner")

#: Fixed base timestamp. Entries get consecutive seconds from it so the chain has one defined
#: order — `iter_chain` walks `(occurred_at, log_id)`, and rows sharing a timestamp would make
#: the expected order depend on generated uuids.
BASE_TIME = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)


async def scratch_tenant(session: Any) -> uuid.UUID:
    """A tenant nobody else writes to, named so its rows are identifiable later."""
    from backend.models.tenant import Tenant

    tenant_id = uuid.uuid4()
    session.add(
        Tenant(
            id=tenant_id,
            name=f"audit-probe-{tenant_id.hex[:8]}",
            domain=f"audit-probe-{tenant_id.hex}.test",
        )
    )
    await session.flush()
    return tenant_id


async def append_chain(session: Any, tenant_id: uuid.UUID, count: int) -> list[AuditLog]:
    """`count` legitimate entries through the real service, in chain order."""
    service = AuditService(AuditRepository(session))
    entries = []
    for index in range(count):
        entries.append(
            await service.log(
                tenant_id=tenant_id,
                action=AuditAction.LOGIN_SUCCESS,
                object_type=AuditObjectType.SESSION,
                object_id=f"probe-{index}",
                outcome=AuditOutcome.SUCCESS,
                actor_email=f"probe-{index}@audit-probe.test",
                occurred_at=BASE_TIME + timedelta(seconds=index),
                detail={"probe": index},
            )
        )
    return entries


def unmatched_hash(marker: str) -> str:
    """64 hex characters that cannot be the hash of any row, unique to this run.

    `audit_log.integrity_hash` carries a UNIQUE constraint, and these tests deliberately leave
    their rows behind (see the module docstring). A constant like `"a" * 64` therefore collides
    with the forged row a *previous* run left in the table, and the test fails on the INSERT
    with a `UniqueViolationError` instead of on the verification it exists to exercise. The
    marker prefix survives so a row found later in the table is still recognisably this test's.
    """
    return (marker[0] * 8 + uuid.uuid4().hex + uuid.uuid4().hex)[:64]


def _one_char(value: Any) -> str:
    """Normalize a PostgreSQL `"char"` column to a one-character `str`.

    asyncpg hands `"char"` back as `bytes` (`b'O'`), psycopg as `str`. Comparing the raw value
    to `"O"` therefore fails on asyncpg even when the trigger is enabled — which is exactly how
    this file's first real run against PostgreSQL reported a correctly installed trigger as
    disabled.
    """
    return value.decode() if isinstance(value, bytes | bytearray) else str(value)


def refusal_reason(error: Exception) -> str:
    """The SQLSTATE or message fragment that shows this was a refusal, or `""`.

    Returns a *reason* rather than asserting, so a test can report which control fired instead
    of only that something failed. A refusal and an unrelated failure — a dropped connection,
    a syntax error introduced by an edit — both raise, and only one of them is the control
    working.
    """
    sqlstate = getattr(getattr(error, "orig", None), "sqlstate", None)
    if sqlstate in REFUSAL_SQLSTATES:
        return str(sqlstate)
    message = str(error).lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in message:
            return phrase
    return ""


async def row_snapshot(tenant_id: uuid.UUID, log_id: uuid.UUID) -> tuple[Any, ...] | None:
    """The fields a tamper would change, read in a session of its own.

    A fresh session matters: the identity map of the session that attempted the update would
    hand back the modified in-memory object and the assertion would pass while the row was
    gone.
    """
    from backend.core.database import session_scope

    async with session_scope() as session:
        entry = await AuditRepository(session).get(log_id, tenant_id)
        if entry is None:
            return None
        return (
            entry.outcome,
            entry.action,
            entry.actor_email,
            entry.object_id,
            entry.detail,
            entry.previous_hash,
            entry.integrity_hash,
        )


async def granted_privileges(session: Any) -> set[str]:
    """Privileges the connected role holds on `audit_log` according to the table's ACL.

    Read from `pg_class.relacl` rather than `has_table_privilege`, which answers True for every
    privilege when the caller is a superuser and would report the REVOKE as absent on the most
    common local setup. The ACL records what was granted, which is the control migration
    `0002_auth_audit` actually installs.
    """
    result = await session.execute(
        text(
            """
            SELECT acl.privilege_type
            FROM pg_class c
            CROSS JOIN aclexplode(c.relacl) AS acl
            JOIN pg_roles r ON r.oid = acl.grantee
            WHERE c.relname = 'audit_log' AND r.rolname = current_user
            """
        )
    )
    return {row[0] for row in result.all()}


async def audit_row_count(session: Any, tenant_id: uuid.UUID) -> int:
    """Rows for one tenant. Scoped, so another `requires_db` test appending its own audit rows
    in the same run cannot change the number this test is comparing."""
    stmt = select(func.count()).select_from(AuditLog).where(AuditLog.tenant_id == tenant_id)
    return int((await session.execute(stmt)).scalar_one())


class TestTheAppendPathWorks:
    """Asserted first, because "append-only" is two claims and the refusals are only half.

    A database that refused every write would pass every test below this class. These say the
    permitted operation is still permitted, so a failure elsewhere is attributable.
    """

    async def test_a_chain_of_appends_verifies_clean(self, db_available: bool) -> None:
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            entries = await append_chain(session, tenant_id, 4)
            log_ids = [entry.log_id for entry in entries]

        async with session_scope() as session:
            report = await AuditService(AuditRepository(session)).verify_chain(tenant_id)

        assert report.entries_checked == 4
        assert report.failures == ()
        assert report.is_valid
        assert len(set(log_ids)) == 4

    async def test_each_entry_links_to_the_one_before_it(self, db_available: bool) -> None:
        """The link is what makes a removed row detectable. Read back from PostgreSQL rather
        than asserted on the objects the service returned, because asyncpg round-trips the
        timestamp and the hash is computed over it."""
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            await append_chain(session, tenant_id, 3)

        async with session_scope() as session:
            stored = await AuditRepository(session).iter_chain(tenant_id)

        assert [entry.previous_hash for entry in stored] == [
            GENESIS_HASH,
            stored[0].integrity_hash,
            stored[1].integrity_hash,
        ]
        for entry in stored:
            assert compute_integrity_hash(entry, entry.previous_hash) == entry.integrity_hash


class TestTheDatabaseRefusesAMutation:
    """The control no application layer can provide: a refusal that holds for every caller."""

    async def test_an_update_is_refused_and_the_row_is_unchanged(
        self, db_available: bool
    ) -> None:
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            target = (await append_chain(session, tenant_id, 2))[0]
        before = await row_snapshot(tenant_id, target.log_id)
        assert before is not None

        with pytest.raises(Exception) as caught:  # noqa: B017 - the driver's type is not the point
            async with session_scope() as session:
                await session.execute(
                    text(
                        "UPDATE audit_log SET outcome = 'FAILURE' WHERE log_id = :log_id"
                    ),
                    {"log_id": str(target.log_id)},
                )

        assert refusal_reason(caught.value), f"not a refusal: {caught.value!r}"
        assert await row_snapshot(tenant_id, target.log_id) == before

    async def test_a_delete_is_refused_and_the_row_survives(
        self, db_available: bool
    ) -> None:
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            target = (await append_chain(session, tenant_id, 2))[0]
            total = await audit_row_count(session, tenant_id)

        with pytest.raises(Exception) as caught:  # noqa: B017
            async with session_scope() as session:
                await session.execute(
                    text("DELETE FROM audit_log WHERE log_id = :log_id"),
                    {"log_id": str(target.log_id)},
                )

        assert refusal_reason(caught.value), f"not a refusal: {caught.value!r}"
        assert await row_snapshot(tenant_id, target.log_id) is not None
        async with session_scope() as session:
            assert await audit_row_count(session, tenant_id) == total

    async def test_a_whole_table_update_is_refused_too(self, db_available: bool) -> None:
        """The row triggers are `FOR EACH ROW`, so a statement touching many rows has to be
        stopped on the first one. A guard that only fired for single-row statements would let
        the cheapest possible cover-up through."""
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            entries = await append_chain(session, tenant_id, 3)
        before = [await row_snapshot(tenant_id, entry.log_id) for entry in entries]

        with pytest.raises(Exception) as caught:  # noqa: B017
            async with session_scope() as session:
                await session.execute(
                    text(
                        "UPDATE audit_log SET detail = NULL WHERE tenant_id = :tenant_id"
                    ),
                    {"tenant_id": str(tenant_id)},
                )

        assert refusal_reason(caught.value), f"not a refusal: {caught.value!r}"
        assert [await row_snapshot(tenant_id, entry.log_id) for entry in entries] == before

    async def test_the_mutating_grants_are_revoked(self, db_available: bool) -> None:
        """The privilege half of the control, and the only assertion available for `TRUNCATE`.

        `INSERT` and `SELECT` are asserted present in the same breath: a REVOKE that took those
        with it would stop the application writing audit rows at all, and every other test in
        this module would then fail for a reason that has nothing to do with immutability.
        """
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            granted = await granted_privileges(session)

        assert granted, (
            "audit_log has no ACL entry for the connected role, so nothing was revoked from "
            "it — either the migration did not run or the app connects as a different role "
            "than Alembic did"
        )
        assert granted & {"UPDATE", "DELETE", "TRUNCATE"} == set()
        assert {"INSERT", "SELECT"} <= granted


class TestTheTriggersAreInstalled:
    """Read from the catalog, because the REVOKE alone protects only one role.

    A superuser, or any role that still holds `UPDATE`, gets past the privilege check. The
    triggers are what make the refusal a property of the table.
    """

    async def test_both_row_triggers_exist_and_are_enabled(
        self, db_available: bool
    ) -> None:
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            result = await session.execute(
                text(
                    """
                    SELECT t.tgname, p.proname, t.tgenabled,
                           pg_get_triggerdef(t.oid) AS definition
                    FROM pg_trigger t
                    JOIN pg_class c ON c.oid = t.tgrelid
                    JOIN pg_proc p ON p.oid = t.tgfoid
                    WHERE c.relname = 'audit_log' AND NOT t.tgisinternal
                    """
                )
            )
            triggers = {row[0]: row for row in result.all()}

        assert set(triggers) == {"audit_log_no_update", "audit_log_no_delete"}
        for name, row in triggers.items():
            assert row[1] == "audit_log_deny_mutation", name
            # 'O' is "enabled, origin and local"; 'D' would be a trigger left in place and
            # switched off, which is indistinguishable from absent at runtime. PostgreSQL's
            # `"char"` type arrives from asyncpg as a single byte, not a str — comparing it to
            # `"O"` fails on a correctly configured trigger, which is how this read the first
            # time it ever ran against a real database.
            assert _one_char(row[2]) == "O", f"{name} is disabled (tgenabled={row[2]!r})"
            # Asserted against the definition text rather than a `tgtype` bit mask. The bits
            # are easy to get wrong — an earlier draft of the TRUNCATE test below used 8
            # (DELETE) believing it was TRUNCATE (32), and so "found" a trigger that did not
            # exist. `pg_get_triggerdef` cannot be misread that way.
            definition = row[3]
            assert "FOR EACH ROW" in definition, f"{name} is not FOR EACH ROW: {definition}"
            assert definition.startswith(f"CREATE TRIGGER {name} BEFORE "), definition
            expected_event = "UPDATE" if name.endswith("update") else "DELETE"
            assert f"BEFORE {expected_event} ON public.audit_log" in definition, definition

    async def test_the_guard_function_raises_restrict_violation(
        self, db_available: bool
    ) -> None:
        """Asserted against the function's source, since the only way to observe the raise is
        to have a role the privilege check lets through — which the test may not have."""
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            body = (
                await session.execute(
                    text(
                        "SELECT prosrc FROM pg_proc WHERE proname = "
                        "'audit_log_deny_mutation'"
                    )
                )
            ).scalar_one()

        assert "append-only" in body
        assert "restrict_violation" in body
        assert "TG_OP" in body, "the exception does not say which operation was attempted"

    async def test_truncate_has_no_trigger_behind_the_grant(
        self, db_available: bool
    ) -> None:
        """A documented gap, asserted so it stays documented rather than assumed closed.

        `TRUNCATE` is revoked but not triggered, so a superuser can still empty the table. If
        this test ever fails because a `BEFORE TRUNCATE` trigger was added, the gap closed —
        delete the test and the waiver in `docs/RELEASE.md` together.

        Matched on the definition text. The `tgtype` bit for TRUNCATE is 32; 8 is DELETE, and
        an earlier draft of this test used 8 and therefore reported the `audit_log_no_delete`
        trigger as TRUNCATE coverage that does not exist. A test that claims a control is in
        force when it is not is worse than no test, so this one reads words instead of bits.
        """
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            definitions = (
                (
                    await session.execute(
                        text(
                            """
                            SELECT pg_get_triggerdef(t.oid)
                            FROM pg_trigger t
                            JOIN pg_class c ON c.oid = t.tgrelid
                            WHERE c.relname = 'audit_log' AND NOT t.tgisinternal
                            """
                        )
                    )
                )
                .scalars()
                .all()
            )

        covering = [text_ for text_ in definitions if "TRUNCATE" in text_]
        assert covering == [], (
            f"a TRUNCATE trigger now exists ({covering}) — remove this test and the "
            "corresponding waiver in docs/RELEASE.md"
        )


class TestAForgedRowIsFoundByVerification:
    """The attack the triggers do not stop.

    Append-only is not append-honest: `INSERT` has to remain permitted, so a caller with the
    application's own privileges can write a row whose hashes do not follow from the chain.
    Nothing in the database rejects it — the hash chain is what makes it visible, and
    `verify_chain` is the only control here. `test_audit_service.py` covers the same detection
    over fakes; this is the version where PostgreSQL stored and returned the row, timestamp
    round-trip and all.
    """

    @staticmethod
    def forged(
        tenant_id: uuid.UUID,
        *,
        previous_hash: str,
        integrity_hash: str,
        occurred_at: datetime,
    ) -> AuditLog:
        """A row assembled by hand — the service is bypassed, which is the point."""
        return AuditLog(
            log_id=uuid.uuid4(),
            tenant_id=tenant_id,
            user_id=None,
            actor_email="forger@audit-probe.test",
            action=AuditAction.LOGIN_SUCCESS,
            object_type=AuditObjectType.SESSION.value,
            object_id="forged",
            outcome=AuditOutcome.SUCCESS,
            occurred_at=occurred_at,
            ip_address="203.0.113.7",
            user_agent="forger",
            previous_hash=previous_hash,
            integrity_hash=integrity_hash,
            detail={"forged": True},
        )

    async def test_a_row_whose_hash_does_not_match_its_contents_is_named(
        self, db_available: bool
    ) -> None:
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            legitimate = await append_chain(session, tenant_id, 3)
            row = self.forged(
                tenant_id,
                # Links correctly to the head, so only the contents check can fail.
                previous_hash=legitimate[-1].integrity_hash,
                integrity_hash=unmatched_hash("a"),
                occurred_at=BASE_TIME + timedelta(seconds=10),
            )
            session.add(row)
            await session.flush()

        async with session_scope() as session:
            report = await AuditService(AuditRepository(session)).verify_chain(tenant_id)

        assert not report.is_valid
        assert report.entries_checked == 4
        assert [
            (log_id, reason) for log_id, reason in report.failures if log_id == row.log_id
        ] == [(row.log_id, "integrity_hash does not match contents")]

    async def test_the_legitimate_rows_are_not_named(self, db_available: bool) -> None:
        """Localization. An edit at row 2 of 100,000 must not produce 99,998 findings — a
        reviewer who has to read past thousands of consequential failures cannot see which row
        was actually touched."""
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            legitimate = await append_chain(session, tenant_id, 3)
            row = self.forged(
                tenant_id,
                previous_hash=legitimate[-1].integrity_hash,
                integrity_hash=unmatched_hash("b"),
                occurred_at=BASE_TIME + timedelta(seconds=10),
            )
            session.add(row)
            await session.flush()

        async with session_scope() as session:
            report = await AuditService(AuditRepository(session)).verify_chain(tenant_id)

        named = {log_id for log_id, _ in report.failures}
        assert named == {row.log_id}
        assert not named & {entry.log_id for entry in legitimate}

    async def test_a_row_spliced_into_the_chain_breaks_the_link(
        self, db_available: bool
    ) -> None:
        """The other half: a forged row spliced *between* two legitimate ones. A forger holding
        the source can compute its hash correctly, so the row itself passes both checks. What
        exposes the splice is the row that used to follow — its `previous_hash` names an entry
        that is no longer its predecessor. The finding lands one row later than the tampering,
        which is worth knowing before reading such a report."""
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        async with session_scope() as session:
            tenant_id = await scratch_tenant(session)
            legitimate = await append_chain(session, tenant_id, 3)
            row = self.forged(
                tenant_id,
                previous_hash=legitimate[0].integrity_hash,
                integrity_hash="c" * 64,
                # Between entry 0 (t+0) and entry 1 (t+1), so the walk visits it second.
                occurred_at=BASE_TIME + timedelta(milliseconds=500),
            )
            # Its own hash is self-consistent: a forger with the source can do this much.
            row.integrity_hash = compute_integrity_hash(row, row.previous_hash)
            session.add(row)
            await session.flush()

        async with session_scope() as session:
            report = await AuditService(AuditRepository(session)).verify_chain(tenant_id)

        assert not report.is_valid
        reasons = dict(report.failures)
        assert reasons.get(legitimate[1].log_id) == (
            "previous_hash does not match the preceding entry"
        )
        # The spliced row itself passes both checks — it links to a real head and its hash is
        # correct. What exposes it is the row that followed the row it displaced.
        assert row.log_id not in reasons
