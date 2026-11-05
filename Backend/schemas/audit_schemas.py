"""backend/schemas/audit_schemas.py

Role: Pydantic DTOs for the audit read endpoint. **Read-only by construction** — there is no
request model here that could create or amend an entry, because no such route exists.

`previous_hash` and `integrity_hash` are published. That is the point of a tamper-evident
log: an auditor who exports the entries must be able to recompute the chain outside this
system, and a hash they cannot see proves nothing to them.

`detail` is published as stored. `audit_service` already stripped control characters from it
and callers are forbidden from putting credentials, account numbers or document text in it —
so this schema does not filter it a second time, it just does not add a way to smuggle
anything in. Frontend rendering still escapes it (XSS is an output-encoding problem, solved
where the value is displayed).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.models.audit_log import AuditAction, AuditLog, AuditOutcome

_REQUEST_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)

MAX_PAGE_SIZE: Final = 200
DEFAULT_PAGE_SIZE: Final = 50


class AuditLogFilter(BaseModel):
    """Query-string filters for `GET /audit`.

    `tenant_id` is absent and would be rejected: the tenant comes from the caller's token.
    An Administrator who could name a tenant would be able to read every organization's log,
    which is the single worst read in the system.
    """

    model_config = _REQUEST_CONFIG

    action: AuditAction | None = None
    outcome: AuditOutcome | None = None
    #: Filter by the *subject* of the entries — which user acted, not who is asking.
    user_id: uuid.UUID | None = None
    object_type: str | None = Field(default=None, max_length=32)
    occurred_from: datetime | None = None
    occurred_to: datetime | None = None
    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def range_must_be_ordered(self) -> AuditLogFilter:
        if (
            self.occurred_from is not None
            and self.occurred_to is not None
            and self.occurred_from > self.occurred_to
        ):
            # A reversed range silently returns nothing, which reads as "no such activity"
            # in a compliance review. Better to say the filter is wrong.
            raise ValueError("occurred_from must not be later than occurred_to.")
        return self


class AuditLogResponse(BaseModel):
    """One audit entry, exactly as stored."""

    model_config = ConfigDict(extra="forbid")

    log_id: uuid.UUID
    #: Null only for a pre-authentication failure against an unknown account.
    user_id: uuid.UUID | None
    actor_email: str
    action: AuditAction
    object_type: str
    object_id: str
    outcome: AuditOutcome
    occurred_at: datetime
    ip_address: str
    user_agent: str
    previous_hash: str
    integrity_hash: str
    detail: dict[str, Any] | None = None

    @classmethod
    def from_entity(cls, entry: AuditLog) -> AuditLogResponse:
        return cls(
            log_id=entry.log_id,
            user_id=entry.user_id,
            actor_email=entry.actor_email,
            action=entry.action,
            object_type=entry.object_type,
            object_id=entry.object_id,
            outcome=entry.outcome,
            occurred_at=entry.occurred_at,
            ip_address=str(entry.ip_address),
            user_agent=entry.user_agent,
            previous_hash=entry.previous_hash,
            integrity_hash=entry.integrity_hash,
            detail=entry.detail,
        )


class AuditLogListResponse(BaseModel):
    """A page of audit entries plus the total, for the compliance table's pager."""

    model_config = ConfigDict(extra="forbid")

    items: list[AuditLogResponse]
    total: int
    limit: int
    offset: int


class ChainVerificationResponse(BaseModel):
    """Result of `verify_chain` — the answer to "has this log been altered?".

    `failures` names the offending `log_id` and what was wrong with it. A boolean alone would
    tell an auditor there is a problem without telling them where, which is the one thing they
    need in order to act.
    """

    model_config = ConfigDict(extra="forbid")

    tenant_id: uuid.UUID
    entries_checked: int
    is_valid: bool
    failures: list[dict[str, str]] = Field(default_factory=list)
