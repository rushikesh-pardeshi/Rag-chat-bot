"""backend/tests/test_audit_service.py

Role: Unit tests for the append-only audit trail — canonicalization, hash chaining,
tamper detection, log-injection sanitizing and tenant separation.

The audit foundation is built in Phase 2 rather than Phase 3: the Phase 2 gate requires an
audit row for every auth decision, so the writer has to exist before the routes that call it.
The audit *read* route and its schemas remain Phase 3.

What is asserted here is the property a compliance reviewer relies on: the hash over a row
must depend on the row's contents and on every row before it, and on nothing else — not on
the reader's timezone and not on dict ordering.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.core.exceptions import ValidationError
from backend.models.audit_log import (
    GENESIS_HASH,
    AuditAction,
    AuditLog,
    AuditObjectType,
    AuditOutcome,
)
from backend.repositories.audit_repository import AuditRepository
from backend.services.audit_service import (
    UNKNOWN_IP,
    AuditService,
    canonical_form,
    compute_integrity_hash,
)
from backend.tests.fakes import FakeAuditRepository
from backend.utils.ids import utc_now


@pytest.fixture
def repo() -> FakeAuditRepository:
    return FakeAuditRepository()


@pytest.fixture
def service(repo: FakeAuditRepository) -> AuditService:
    return AuditService(repo)  # type: ignore[arg-type]


@pytest.fixture
def tenant_id() -> uuid.UUID:
    return uuid.uuid4()


def make_entry(**overrides: object) -> AuditLog:
    """A complete entry. Every canonical field is populated so nothing defaults silently."""
    fields: dict[str, object] = {
        "log_id": uuid.UUID("11111111-1111-1111-1111-111111111111"),
        "tenant_id": uuid.UUID("22222222-2222-2222-2222-222222222222"),
        "user_id": uuid.UUID("33333333-3333-3333-3333-333333333333"),
        "actor_email": "officer@examplebank.com",
        "action": AuditAction.QUERY_SUBMIT,
        "object_type": AuditObjectType.QUERY.value,
        "object_id": "q-1",
        "outcome": AuditOutcome.SUCCESS,
        "occurred_at": datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC),
        "ip_address": "203.0.113.10",
        "user_agent": "pytest",
        "detail": {"a": 1, "b": 2},
    }
    fields.update(overrides)
    return AuditLog(**fields)  # type: ignore[arg-type]


# -- The hash itself ---------------------------------------------------------
class TestIntegrityHash:
    def test_hash_is_sixty_four_hex_characters(self) -> None:
        digest = compute_integrity_hash(make_entry(), GENESIS_HASH)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_hash_is_deterministic(self) -> None:
        assert compute_integrity_hash(make_entry(), GENESIS_HASH) == (
            compute_integrity_hash(make_entry(), GENESIS_HASH)
        )

    def test_the_same_instant_in_a_different_offset_hashes_identically(self) -> None:
        """The digest must not depend on who is reading the row.

        asyncpg may hand back `+00:00` where the application wrote `+05:30`. If the
        canonical form kept the offset, verification would fail on a correct row.
        """
        utc = datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)
        india = utc.astimezone(timezone(timedelta(hours=5, minutes=30)))
        assert compute_integrity_hash(make_entry(occurred_at=utc), GENESIS_HASH) == (
            compute_integrity_hash(make_entry(occurred_at=india), GENESIS_HASH)
        )

    def test_detail_key_order_does_not_change_the_hash(self) -> None:
        """JSONB does not preserve insertion order, so the canonical form must sort keys."""
        assert compute_integrity_hash(
            make_entry(detail={"a": 1, "b": 2}), GENESIS_HASH
        ) == compute_integrity_hash(make_entry(detail={"b": 2, "a": 1}), GENESIS_HASH)

    def test_a_naive_timestamp_is_refused(self) -> None:
        """A naive datetime would hash as whatever the reader's clock means locally."""
        with pytest.raises(ValidationError):
            canonical_form(make_entry(occurred_at=datetime(2026, 8, 13, 12, 0, 0)))

    @pytest.mark.parametrize(
        "field,value",
        [
            ("actor_email", "someone.else@examplebank.com"),
            ("action", AuditAction.DOCUMENT_DELETE),
            ("outcome", AuditOutcome.DENIED),
            ("object_id", "q-2"),
            ("user_id", uuid.UUID("44444444-4444-4444-4444-444444444444")),
            ("detail", {"a": 1, "b": 3}),
        ],
    )
    def test_changing_any_audited_field_changes_the_hash(
        self, field: str, value: object
    ) -> None:
        baseline = compute_integrity_hash(make_entry(), GENESIS_HASH)
        assert compute_integrity_hash(make_entry(**{field: value}), GENESIS_HASH) != (
            baseline
        )

    def test_the_preceding_hash_is_part_of_the_digest(self) -> None:
        """This is what makes it a chain: editing row N invalidates every row after it."""
        assert compute_integrity_hash(make_entry(), GENESIS_HASH) != (
            compute_integrity_hash(make_entry(), "f" * 64)
        )

    def test_null_and_empty_string_are_distinguishable(self) -> None:
        """Otherwise `detail=None` and `detail={}` would collide with a missing field."""
        assert compute_integrity_hash(make_entry(detail=None), GENESIS_HASH) != (
            compute_integrity_hash(make_entry(detail={}), GENESIS_HASH)
        )


# -- Writing entries ---------------------------------------------------------
class TestLog:
    async def test_first_entry_for_a_tenant_chains_to_genesis(
        self, service: AuditService, tenant_id: uuid.UUID
    ) -> None:
        entry = await service.log(
            tenant_id=tenant_id,
            action=AuditAction.LOGIN_SUCCESS,
            object_type=AuditObjectType.SESSION,
            object_id="s-1",
            outcome=AuditOutcome.SUCCESS,
            actor_email="first@examplebank.com",
        )
        assert entry.previous_hash == GENESIS_HASH
        assert entry.integrity_hash == compute_integrity_hash(entry, GENESIS_HASH)

    async def test_each_entry_chains_to_the_previous_one(
        self, service: AuditService, repo: FakeAuditRepository, tenant_id: uuid.UUID
    ) -> None:
        for i in range(3):
            await service.log(
                tenant_id=tenant_id,
                action=AuditAction.QUERY_SUBMIT,
                object_type=AuditObjectType.QUERY,
                object_id=f"q-{i}",
                outcome=AuditOutcome.SUCCESS,
                actor_email="user@examplebank.com",
            )
        first, second, third = repo.entries
        assert second.previous_hash == first.integrity_hash
        assert third.previous_hash == second.integrity_hash

    async def test_two_tenants_keep_independent_chains(
        self, service: AuditService, repo: FakeAuditRepository
    ) -> None:
        """A tenant must not be able to infer another tenant's activity from a hash."""
        a, b = uuid.uuid4(), uuid.uuid4()
        for tenant in (a, b, a):
            await service.log(
                tenant_id=tenant,
                action=AuditAction.QUERY_SUBMIT,
                object_type=AuditObjectType.QUERY,
                object_id="q",
                outcome=AuditOutcome.SUCCESS,
                actor_email="user@examplebank.com",
            )
        a_first, b_first, a_second = repo.entries
        assert b_first.previous_hash == GENESIS_HASH
        assert a_second.previous_hash == a_first.integrity_hash

    async def test_missing_actor_or_object_is_refused(
        self, service: AuditService, tenant_id: uuid.UUID
    ) -> None:
        """An entry that cannot say who did what to which object is not an audit record."""
        for actor, object_id in (("", "o-1"), ("a@b.com", "")):
            with pytest.raises(ValidationError):
                await service.log(
                    tenant_id=tenant_id,
                    action=AuditAction.DOCUMENT_UPLOAD,
                    object_type=AuditObjectType.DOCUMENT,
                    object_id=object_id,
                    outcome=AuditOutcome.SUCCESS,
                    actor_email=actor,
                )

    async def test_crlf_in_user_input_cannot_forge_a_second_log_line(
        self, service: AuditService, tenant_id: uuid.UUID
    ) -> None:
        """Log injection: `.claude/rules/secure-coding-vapt.md` requires CR/LF stripped."""
        entry = await service.log(
            tenant_id=tenant_id,
            action=AuditAction.LOGIN_FAILURE,
            object_type=AuditObjectType.SESSION,
            object_id="s-1\r\nINJECTED",
            outcome=AuditOutcome.FAILURE,
            actor_email="evil@examplebank.com\r\nadmin@examplebank.com",
            user_agent="curl/8\nX-Forged: yes",
            detail={"note": "line one\r\nline two"},
        )
        for value in (entry.actor_email, entry.object_id, entry.user_agent):
            assert "\r" not in value
            assert "\n" not in value
        assert entry.detail is not None
        assert "\n" not in entry.detail["note"]

    async def test_a_missing_source_address_becomes_a_marker_not_null(
        self, service: AuditService, tenant_id: uuid.UUID
    ) -> None:
        """`ip_address` is NOT NULL; an ASGI call with no client still has to be recordable."""
        entry = await service.log(
            tenant_id=tenant_id,
            action=AuditAction.LOGIN_FAILURE,
            object_type=AuditObjectType.SESSION,
            object_id="s-1",
            outcome=AuditOutcome.FAILURE,
            actor_email="nobody@examplebank.com",
        )
        assert entry.ip_address == UNKNOWN_IP
        assert entry.user_agent == "unknown"

    async def test_occurred_at_is_timezone_aware(
        self, service: AuditService, tenant_id: uuid.UUID
    ) -> None:
        entry = await service.log(
            tenant_id=tenant_id,
            action=AuditAction.AUDIT_VIEW,
            object_type=AuditObjectType.AUDIT_EXPORT,
            object_id="all",
            outcome=AuditOutcome.SUCCESS,
            actor_email="admin@examplebank.com",
        )
        assert entry.occurred_at.tzinfo is not None
        assert entry.occurred_at <= utc_now()


# -- Verification ------------------------------------------------------------
class TestVerifyChain:
    async def _seed(
        self, service: AuditService, tenant_id: uuid.UUID, count: int = 4
    ) -> None:
        for i in range(count):
            await service.log(
                tenant_id=tenant_id,
                action=AuditAction.QUERY_SUBMIT,
                object_type=AuditObjectType.QUERY,
                object_id=f"q-{i}",
                outcome=AuditOutcome.SUCCESS,
                actor_email="user@examplebank.com",
            )

    async def test_an_untouched_chain_verifies(
        self, service: AuditService, tenant_id: uuid.UUID
    ) -> None:
        await self._seed(service, tenant_id)
        result = await service.verify_chain(tenant_id)
        assert result.is_valid
        assert result.entries_checked == 4
        assert result.failures == ()

    async def test_an_empty_chain_verifies(
        self, service: AuditService, tenant_id: uuid.UUID
    ) -> None:
        result = await service.verify_chain(tenant_id)
        assert result.is_valid
        assert result.entries_checked == 0

    async def test_editing_a_field_is_detected(
        self, service: AuditService, repo: FakeAuditRepository, tenant_id: uuid.UUID
    ) -> None:
        """The scenario the chain exists for: someone changes a stored row in place."""
        await self._seed(service, tenant_id)
        repo.entries[1].outcome = AuditOutcome.SUCCESS
        repo.entries[1].actor_email = "someone-else@examplebank.com"
        result = await service.verify_chain(tenant_id)
        assert not result.is_valid
        assert result.first_failure is not None
        assert result.first_failure[0] == repo.entries[1].log_id

    async def test_a_recomputed_hash_does_not_hide_the_edit(
        self, service: AuditService, repo: FakeAuditRepository, tenant_id: uuid.UUID
    ) -> None:
        """An attacker who fixes the edited row's own hash still breaks every later link."""
        await self._seed(service, tenant_id)
        target = repo.entries[1]
        target.actor_email = "someone-else@examplebank.com"
        target.integrity_hash = compute_integrity_hash(target, target.previous_hash)
        result = await service.verify_chain(tenant_id)
        assert not result.is_valid
        assert result.failures[0][0] == repo.entries[2].log_id

    async def test_deleting_an_entry_is_detected(
        self, service: AuditService, repo: FakeAuditRepository, tenant_id: uuid.UUID
    ) -> None:
        """Append-only means a gap is visible even though the remaining rows are intact."""
        await self._seed(service, tenant_id)
        del repo.entries[1]
        result = await service.verify_chain(tenant_id)
        assert not result.is_valid
        assert result.entries_checked == 3

    async def test_verification_does_not_stop_at_the_first_failure(
        self, service: AuditService, repo: FakeAuditRepository, tenant_id: uuid.UUID
    ) -> None:
        """A reviewer needs the extent of the damage, not only where it starts."""
        await self._seed(service, tenant_id)
        repo.entries[1].object_id = "tampered"  # contents no longer match the stored hash
        repo.entries[3].previous_hash = "0" * 64  # link to the preceding row broken
        result = await service.verify_chain(tenant_id)
        damaged = {log_id for log_id, _ in result.failures}
        assert damaged == {repo.entries[1].log_id, repo.entries[3].log_id}

    async def test_one_tampered_row_does_not_flood_the_report(
        self, service: AuditService, repo: FakeAuditRepository, tenant_id: uuid.UUID
    ) -> None:
        """Findings are localized: each row is judged against its own stored hashes, so an
        edit early in a long chain does not mark every later row as damaged too."""
        await self._seed(service, tenant_id, count=10)
        repo.entries[1].object_id = "tampered"
        result = await service.verify_chain(tenant_id)
        assert len(result.failures) == 1
        assert result.failures[0][0] == repo.entries[1].log_id

    async def test_verification_is_scoped_to_one_tenant(
        self, service: AuditService, repo: FakeAuditRepository, tenant_id: uuid.UUID
    ) -> None:
        other = uuid.uuid4()
        await self._seed(service, tenant_id, count=2)
        await self._seed(service, other, count=2)
        repo.entries[-1].object_id = "tampered"
        assert (await service.verify_chain(tenant_id)).is_valid
        assert not (await service.verify_chain(other)).is_valid


# -- Immutability at the repository boundary ---------------------------------
class TestAppendOnly:
    @pytest.mark.parametrize("forbidden", ["update", "delete", "remove", "purge", "edit"])
    def test_the_repository_exposes_no_mutation_method(self, forbidden: str) -> None:
        """First of three independent controls; the others are the DB trigger and REVOKE
        in `alembic/versions/20260813_0002_auth_and_audit.py`. A method added here would
        be the one an ORM caller could reach, so its absence is asserted."""
        assert not hasattr(AuditRepository, forbidden)

    def test_the_service_exposes_no_mutation_method(self) -> None:
        """The public surface is asserted exactly, not by forbidden-name pattern.

        An exact set is deliberately brittle: adding any method to `AuditService` fails this
        test and forces a reviewer to decide whether the new operation appends, reads, or
        mutates. A `not hasattr(..., "update")` check would pass for a mutation named
        anything else. Every name below is either the single append (`log`) or a read.
        """
        public = {name for name in vars(AuditService) if not name.startswith("_")}
        assert public == {"log", "list_entries", "verify_chain"}
