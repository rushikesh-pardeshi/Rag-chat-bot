"""backend/models/audit_log.py

Role: ORM entity for `audit_log` — append-only, hash-chained per tenant.

Mirrors `.claude/skills/audit-trail-generation/references/audit-schema.md` field for field.
Immutability is enforced in three places, because one is not enough:
  * this model exposes no mutation helper;
  * `audit_repository` has no update or delete method;
  * the migration revokes UPDATE and DELETE from the application role.

`actor_email` is denormalized on purpose: an audit entry has to stay readable after the
user row it refers to is gone, and a compliance review years later cannot depend on a join.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, ForeignKey, Index, String, Text, Uuid
from sqlalchemy.dialects.postgresql import INET, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.core.database import Base
from backend.utils.ids import UuidPk


class AuditAction(enum.StrEnum):
    """Every action that must leave a trace. Denials use the action that was attempted."""

    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGOUT = "LOGOUT"
    ACCESS_DENIED = "ACCESS_DENIED"
    DOCUMENT_UPLOAD = "DOCUMENT_UPLOAD"
    DOCUMENT_PREVIEW = "DOCUMENT_PREVIEW"
    DOCUMENT_DELETE = "DOCUMENT_DELETE"
    DOCUMENT_RESTORE = "DOCUMENT_RESTORE"
    #: Irreversible destruction: the row, its chunks, its vectors and every S3 version. Its
    #: own action rather than a `DOCUMENT_DELETE` with a flag, because the one question a
    #: compliance reviewer asks of this table is "was anything actually erased, and by
    #: whom" — and that must be answerable by filtering, not by reading detail JSON.
    DOCUMENT_PURGE = "DOCUMENT_PURGE"
    QUERY_SUBMIT = "QUERY_SUBMIT"
    HISTORY_VIEW = "HISTORY_VIEW"
    AUDIT_VIEW = "AUDIT_VIEW"
    ANALYTICS_VIEW = "ANALYTICS_VIEW"
    DATA_EXPORT = "DATA_EXPORT"
    ROLE_CHANGE = "ROLE_CHANGE"


class AuditOutcome(enum.StrEnum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED = "DENIED"


class AuditObjectType(enum.StrEnum):
    DOCUMENT = "document"
    QUERY = "query"
    USER = "user"
    AUDIT_EXPORT = "audit_export"
    SESSION = "session"
    ANALYTICS = "analytics"


#: First entry for a tenant chains to this fixed value rather than to nothing, so
#: verification has a defined starting point instead of a special case.
GENESIS_HASH = "0" * 64


class AuditLog(Base):
    __tablename__ = "audit_log"
    __table_args__ = (
        Index("ix_audit_tenant_occurred", "tenant_id", "occurred_at"),
        Index("ix_audit_tenant_action_occurred", "tenant_id", "action", "occurred_at"),
        # Chain verification walks per tenant in occurred_at order.
        Index("ix_audit_tenant_prev_hash", "tenant_id", "previous_hash"),
    )

    log_id: Mapped[UuidPk]
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenant.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    #: Null only for a pre-authentication failure — an unknown email has no user id.
    #: Deliberately *not* a foreign key: the entry has to outlive the user row, which is
    #: also why actor_email is stored alongside it.
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid(as_uuid=True), nullable=True)
    actor_email: Mapped[str] = mapped_column(String(320), nullable=False)
    action: Mapped[AuditAction] = mapped_column(
        Enum(AuditAction, name="audit_action", native_enum=True, validate_strings=True),
        nullable=False,
    )
    object_type: Mapped[str] = mapped_column(String(32), nullable=False)
    #: The id of the affected object, never its contents.
    object_id: Mapped[str] = mapped_column(String(128), nullable=False)
    outcome: Mapped[AuditOutcome] = mapped_column(
        Enum(AuditOutcome, name="audit_outcome", native_enum=True, validate_strings=True),
        nullable=False,
    )
    #: UTC, set by the service. A naive datetime would break both ordering across
    #: processes and the canonical form the integrity hash is computed over.
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    ip_address: Mapped[str] = mapped_column(INET, nullable=False)
    user_agent: Mapped[str] = mapped_column(Text, nullable=False)
    previous_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    integrity_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    #: Non-sensitive context only. Never credentials, account numbers, or document text.
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<AuditLog {self.action.value} outcome={self.outcome.value} "
            f"object={self.object_type}:{self.object_id}>"
        )
