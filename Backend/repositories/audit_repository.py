"""backend/repositories/audit_repository.py

Role: Append-only audit writes and filtered reads. The only place `audit_log` is queried.

This class deliberately has **no** update or delete method, and none may be added. The
database revokes those privileges from the application role and a trigger raises on either,
so the absence here is the third of three independent controls rather than the only one.

`append` takes a row-level lock on the tenant's chain head. Two concurrent writes that both
read the same `previous_hash` would produce a fork, and a forked chain fails verification
for reasons that have nothing to do with tampering.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.audit_log import (
    GENESIS_HASH,
    AuditAction,
    AuditLog,
    AuditOutcome,
)


class AuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Write (the only mutation this table allows) -------------------------
    async def lock_chain(self, tenant_id: uuid.UUID) -> str:
        """Serialize appends for one tenant and return the current chain head hash.

        Uses a transaction-scoped advisory lock keyed on the tenant, not `SELECT … FOR
        UPDATE`: the first entry for a tenant has no row to lock, and a gap-locking
        predicate lock is not portable. The lock releases with the transaction.
        """
        # hashtext() gives a stable int4 key from the UUID; collisions between tenants
        # would only serialize two chains that could have run in parallel, never corrupt.
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
            {"key": f"audit_chain:{tenant_id}"},
        )
        return await self.last_hash(tenant_id)

    async def last_hash(self, tenant_id: uuid.UUID) -> str:
        """`integrity_hash` of the newest entry for a tenant, or the genesis value.

        Ordered by `(occurred_at, log_id)` so two entries sharing a timestamp still have
        one defined successor — verification walks the same order.
        """
        stmt = (
            select(AuditLog.integrity_hash)
            .where(AuditLog.tenant_id == tenant_id)
            .order_by(AuditLog.occurred_at.desc(), AuditLog.log_id.desc())
            .limit(1)
        )
        head = (await self._session.execute(stmt)).scalar_one_or_none()
        return head or GENESIS_HASH

    async def append(self, entry: AuditLog) -> AuditLog:
        """Insert one entry. The service has already computed its hashes."""
        self._session.add(entry)
        await self._session.flush()
        return entry

    # -- Read ----------------------------------------------------------------
    def _scoped(self, tenant_id: uuid.UUID) -> Select[tuple[AuditLog]]:
        return select(AuditLog).where(AuditLog.tenant_id == tenant_id)

    async def list_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        action: AuditAction | None = None,
        outcome: AuditOutcome | None = None,
        user_id: uuid.UUID | None = None,
        object_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Filtered compliance read. Every filter is a bound parameter, never a string."""
        stmt = self._scoped(tenant_id)
        stmt = self._apply_filters(
            stmt,
            action=action,
            outcome=outcome,
            user_id=user_id,
            object_type=object_type,
            occurred_from=occurred_from,
            occurred_to=occurred_to,
        )
        stmt = (
            stmt.order_by(AuditLog.occurred_at.desc(), AuditLog.log_id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self._session.execute(stmt)).scalars())

    async def count_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        action: AuditAction | None = None,
        outcome: AuditOutcome | None = None,
        user_id: uuid.UUID | None = None,
        object_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(AuditLog).where(
            AuditLog.tenant_id == tenant_id
        )
        stmt = self._apply_filters(
            stmt,
            action=action,
            outcome=outcome,
            user_id=user_id,
            object_type=object_type,
            occurred_from=occurred_from,
            occurred_to=occurred_to,
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def iter_chain(
        self, tenant_id: uuid.UUID, *, limit: int | None = None
    ) -> list[AuditLog]:
        """The tenant's entries in chain order, for verification.

        Ascending `(occurred_at, log_id)` — the exact inverse of `last_hash`, so the walk
        visits entries in the order they were linked.
        """
        stmt = self._scoped(tenant_id).order_by(
            AuditLog.occurred_at.asc(), AuditLog.log_id.asc()
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        return list((await self._session.execute(stmt)).scalars())

    async def get(self, log_id: uuid.UUID, tenant_id: uuid.UUID) -> AuditLog | None:
        """Tenant-scoped: a cross-tenant log id returns None, not another tenant's row."""
        stmt = self._scoped(tenant_id).where(AuditLog.log_id == log_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    @staticmethod
    def _apply_filters(
        stmt: Any,
        *,
        action: AuditAction | None,
        outcome: AuditOutcome | None,
        user_id: uuid.UUID | None,
        object_type: str | None,
        occurred_from: datetime | None,
        occurred_to: datetime | None,
    ) -> Any:
        if action is not None:
            stmt = stmt.where(AuditLog.action == action)
        if outcome is not None:
            stmt = stmt.where(AuditLog.outcome == outcome)
        if user_id is not None:
            stmt = stmt.where(AuditLog.user_id == user_id)
        if object_type is not None:
            stmt = stmt.where(AuditLog.object_type == object_type)
        if occurred_from is not None:
            stmt = stmt.where(AuditLog.occurred_at >= occurred_from)
        if occurred_to is not None:
            stmt = stmt.where(AuditLog.occurred_at <= occurred_to)
        return stmt
