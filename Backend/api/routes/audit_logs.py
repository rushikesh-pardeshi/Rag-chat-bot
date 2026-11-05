"""backend/api/routes/audit_logs.py

Role: Read-only HTTP access to the audit log, for Administrators (compliance officers hold
that role — there is no separate one).

**There is no write, edit or delete route in this file, and none may be added.** Entries are
created only as a side effect of the action they describe. That absence is one of four
independent controls on immutability: no route here, no mutating method on
`audit_repository`, `REVOKE UPDATE, DELETE` from the application role, and a trigger that
raises on either.

Both routes are `GET`. `/verify` recomputes the hash chain and reports every divergence with
the offending `log_id` — the answer to "has this log been altered?", which a boolean alone
does not give an auditor enough to act on.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query

from backend.api.dependencies.auth import AdminUser, AuditServiceDep, RequestCtx
from backend.models.audit_log import AuditAction, AuditOutcome
from backend.schemas.audit_schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    AuditLogFilter,
    AuditLogListResponse,
    AuditLogResponse,
    ChainVerificationResponse,
)
from backend.services.audit_service import AuditQuery

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get(
    "",
    response_model=AuditLogListResponse,
    summary="Read the audit log (Administrator only)",
)
async def list_audit_logs(
    actor: AdminUser,
    context: RequestCtx,
    audit: AuditServiceDep,
    action: AuditAction | None = Query(default=None),
    outcome: AuditOutcome | None = Query(default=None),
    user_id: str | None = Query(
        default=None, description="Filter by the user whose actions were logged"
    ),
    object_type: str | None = Query(default=None, max_length=32),
    occurred_from: datetime | None = Query(default=None),
    occurred_to: datetime | None = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> AuditLogListResponse:
    """A filtered page of the caller's own tenant's log, newest first.

    The tenant is not a parameter. It comes from the caller's row, so an Administrator in one
    organization cannot read another's log — role and tenant are independent checks and both
    apply (`.claude/rules/rbac-enforcement.md`).

    Reading the log writes an `AUDIT_VIEW` entry of its own. That is deliberate: a log that
    cannot say who read it is missing the access record a regulator asks for first.
    """
    # Parsed through the Pydantic filter rather than used directly, so the ordering rule on
    # the date range and the `user_id` UUID format are validated in one place.
    filters = AuditLogFilter(
        action=action,
        outcome=outcome,
        user_id=user_id,  # type: ignore[arg-type]  - coerced to UUID by the model
        object_type=object_type,
        occurred_from=occurred_from,
        occurred_to=occurred_to,
        limit=limit,
        offset=offset,
    )
    page = await audit.list_entries(
        tenant_id=actor.tenant_id,
        filters=AuditQuery(
            action=filters.action,
            outcome=filters.outcome,
            user_id=filters.user_id,
            object_type=filters.object_type,
            occurred_from=filters.occurred_from,
            occurred_to=filters.occurred_to,
            limit=filters.limit,
            offset=filters.offset,
        ),
        actor_email=actor.email,
        actor_user_id=actor.id,
        ip_address=context.ip_address,
        user_agent=context.user_agent,
    )
    return AuditLogListResponse(
        items=[AuditLogResponse.from_entity(entry) for entry in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/verify",
    response_model=ChainVerificationResponse,
    summary="Verify audit chain integrity (Administrator only)",
)
async def verify_audit_chain(
    actor: AdminUser,
    audit: AuditServiceDep,
    limit: int | None = Query(
        default=None,
        ge=1,
        description="Check only the oldest N entries. Omit to walk the whole chain.",
    ),
) -> ChainVerificationResponse:
    """Recompute every hash for the caller's tenant and report each divergence.

    Walks the whole chain rather than stopping at the first bad row, so a reviewer sees the
    full extent in one pass. This is a read: it recomputes hashes in memory and writes
    nothing, not even a correction.

    Unlike `GET /audit`, this route does not write an `AUDIT_VIEW` entry. It returns no entry
    contents — counts, a verdict, and the ids of rows that failed — and an integrity check
    that appends to the log it is checking makes the result harder to reason about, since the
    new row changes the chain head between two consecutive checks.
    """
    result = await audit.verify_chain(actor.tenant_id, limit=limit)
    return ChainVerificationResponse(
        tenant_id=result.tenant_id,
        entries_checked=result.entries_checked,
        is_valid=result.is_valid,
        failures=[
            {"log_id": str(log_id), "reason": reason}
            for log_id, reason in result.failures
        ],
    )
