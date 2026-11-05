"""backend/models/query.py

Role: ORM entity for `query` — one row per natural-language question asked.

**This table is also the search history.** `.claude/rules/architecture.md` is explicit that
`search_history` is a role-filtered projection over this table, not a second table: two
copies of an audited question-and-answer pair can drift, and a compliance review that finds
them disagreeing has found a defect it cannot resolve. Standard Users read their own rows,
Administrators read every row for the tenant — same table, different WHERE clause.

`source_references` is JSONB rather than a child table. The references are written once with
the answer and only ever read back whole, so a join buys nothing; and keeping them beside the
answer means the two cannot be separated by a partial write, which matters because BR-03 says
an answer without its sources must not exist.

Phase 3 only ever writes `PENDING` rows. Answer generation, and everything that fills
`answer_text`, `confidence_score` and `source_references`, arrives with the RAG pipeline in
Phase 4 — the columns exist now so the migration does not have to be revisited.
"""

from __future__ import annotations

import enum
import uuid  # runtime import: SQLAlchemy resolves Mapped[...] annotations at mapper setup
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base
from backend.utils.ids import CreatedAt, UuidPk

if TYPE_CHECKING:
    from backend.models.user import User


class QueryStatus(enum.StrEnum):
    """How a question ended.

    `NOT_FOUND` is a *success* of the grounding rule, not a failure: retrieval found nothing
    above the similarity threshold and the system correctly refused to answer from model
    knowledge. Keeping it distinct from `FAILED` is what makes "how often do we have to say
    we don't know" measurable — which is a coverage metric for the document repository, not
    an error rate.
    """

    PENDING = "PENDING"
    ANSWERED = "ANSWERED"
    NOT_FOUND = "NOT_FOUND"
    FAILED = "FAILED"


class Query(Base):
    __tablename__ = "query"
    __table_args__ = (
        # The search-history read: this tenant, this user, newest first. `created_at DESC`
        # is spelled out as an expression so page one is an index scan with no sort step —
        # a plain ascending index would be walked backwards, which costs the same here but
        # not once the trailing columns grow.
        Index(
            "ix_query_tenant_user_created",
            "tenant_id",
            "user_id",
            text("created_at DESC"),
        ),
        # The Administrator view and the analytics aggregations drop the user filter.
        Index("ix_query_tenant_created", "tenant_id", text("created_at DESC")),
        Index("ix_query_tenant_status", "tenant_id", "status"),
        CheckConstraint(
            "confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)",
            name="ck_query_confidence_range",
        ),
        CheckConstraint("length(trim(question_text)) > 0", name="ck_query_question_not_blank"),
    )

    id: Mapped[UuidPk]
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenant.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    #: RESTRICT: the question a user asked is an audited fact that must outlive nothing.
    #: Deleting a user requires dealing with their history deliberately.
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user_account.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: A list of `{source_document_name, section_or_page, document_id, chunk_id,
    #: confidence_level}`. Written whole with the answer; `[]` until then.
    source_references: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    confidence_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[QueryStatus] = mapped_column(
        Enum(QueryStatus, name="query_status", native_enum=True, validate_strings=True),
        nullable=False,
        default=QueryStatus.PENDING,
    )
    #: End-to-end wall time in milliseconds. Null while PENDING. This is the evidence for
    #: the <5s non-functional target, so it is recorded rather than inferred from logs.
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Set when the answer was produced, so the gap from `created_at` is visible even for a
    #: query answered by a background worker.
    answered_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[CreatedAt]

    user: Mapped[User] = relationship(lazy="raise")

    @property
    def has_sources(self) -> bool:
        return bool(self.source_references)

    def __repr__(self) -> str:
        # Never the question or answer text: both may quote customer financial data, and
        # __repr__ ends up in logs and tracebacks.
        return (
            f"<Query id={self.id} status={self.status.value} "
            f"sources={len(self.source_references)}>"
        )
