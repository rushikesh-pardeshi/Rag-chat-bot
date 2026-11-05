"""backend/models/document.py

Role: ORM entities for `document` and `document_chunk`.

Two decisions worth stating up front, because both are load-bearing elsewhere:

**Deletion is a status change.** `DELETED` hides a document from every listing and from
retrieval, but the row and the S3 object stay. A banking audit trail that says "this document
was consulted on 12 March" is worthless if the document itself can be erased, and BR-01 asks
for restore. Nothing in this codebase issues a DELETE against this table.

**Lineage is app-side, not bucket versioning.** Our IAM role cannot enable S3 versioning, so
every upload of a repeated name writes a new immutable key — `documents/{id}/v{n}/{name}` —
and gets its own `document` row with `version` incremented. Two rows, two keys, both readable.
The alternative (overwrite plus a version column) loses the earlier bytes, which is the one
thing lineage is supposed to preserve.

**Vectors live in Pinecone; everything else lives here.** `document_chunk` holds the text and
the citation (page or section) and *no* embedding column. The vector for a chunk is stored in
Pinecone under the id `str(chunk.id)`, in that tenant's namespace, and `embedded_at` records
that the upsert was acknowledged. PostgreSQL stays the authority on status and ownership, which
matters on soft-delete: a retrieval hit is re-checked against `document.status` here before it
can be cited, so a `DELETED` document cannot be quoted even in the window before its vectors
are removed from the index.

`DocumentChunk.page_number` and `section_label` are `NOT NULL` together with a CHECK that at
least one is meaningful: BR-03 requires every answer to cite a page or section, so a chunk
that cannot be cited must not be storable in the first place. Enforcing that at the column
level means a future ingestion path cannot quietly create uncitable text.
"""

from __future__ import annotations

import enum
import uuid  # runtime import: SQLAlchemy resolves Mapped[...] annotations at mapper setup
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.config import get_settings
from backend.core.database import Base
from backend.utils.ids import CreatedAt, UpdatedAt, UuidPk

if TYPE_CHECKING:
    from backend.models.user import User


class DocumentStatus(enum.StrEnum):
    """Lifecycle of a stored document.

    Only `ACTIVE` documents are retrievable. The rest each mean something specific:
      * `PROCESSING` — bytes are in S3, chunks are not ready yet (Phase 4 runs this async);
      * `NEEDS_OCR` — parsed successfully but yielded no text, so it is almost certainly a
        scan. Excluded from retrieval rather than silently returning nothing;
      * `FAILED` — parsing raised. The reason is audited, not stored on the row;
      * `DELETED` — soft-deleted, restorable, S3 object untouched.
    """

    PROCESSING = "PROCESSING"
    ACTIVE = "ACTIVE"
    NEEDS_OCR = "NEEDS_OCR"
    FAILED = "FAILED"
    DELETED = "DELETED"


#: Embedding width. Titan v2 emits 1024 floats and the Pinecone index is created with the
#: same number — a mismatch is rejected by Pinecone at upsert time, per vector, which is a
#: miserable way to find out. Kept here so ingestion can assert the width before sending.
#: The vectors themselves are **not** in this database; see `DocumentChunk.embedded_at`.
EMBEDDING_DIMENSIONS: int = get_settings().BEDROCK_EMBEDDING_DIMENSIONS

#: Statuses a search may retrieve from. A `NEEDS_OCR` document has no citable text, and
#: `PROCESSING` has no embeddings yet — including either would produce answers with no
#: sources, which `rag-answer-grounding.md` forbids.
RETRIEVABLE_STATUSES: frozenset[DocumentStatus] = frozenset({DocumentStatus.ACTIVE})


class Document(Base):
    __tablename__ = "document"
    __table_args__ = (
        # One row per (name, version) per tenant. Makes a double-submitted upload a
        # constraint violation instead of two rows claiming to be v2.
        UniqueConstraint("tenant_id", "name", "version", name="uq_document_name_version"),
        # The S3 key is derived from (document_id, version, filename) and must never repeat;
        # a duplicate would mean one upload overwrote another's bytes.
        UniqueConstraint("s3_key", name="uq_document_s3_key"),
        # Listing is always "this tenant, this status, newest first".
        Index("ix_document_tenant_status_uploaded", "tenant_id", "status", "upload_date"),
        Index("ix_document_tenant_name", "tenant_id", "name"),
        CheckConstraint("file_size > 0", name="ck_document_file_size_positive"),
        CheckConstraint("version >= 1", name="ck_document_version_positive"),
    )

    id: Mapped[UuidPk]
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenant.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    #: The original filename as sanitized on upload — never a path. See `storage_service`.
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Built by the application, never by a caller. Unique, and immutable once written.
    s3_key: Mapped[str] = mapped_column(String(1024), nullable=False)
    #: The type we *determined* from magic bytes, not the one the client claimed.
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    #: Pages for a PDF, sections for a DOCX. Null until parsing has run.
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Who uploaded it. RESTRICT, not CASCADE: removing a user must not remove the record
    #: of what they uploaded.
    uploaded_by: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user_account.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    upload_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            native_enum=True,
            validate_strings=True,
        ),
        nullable=False,
        default=DocumentStatus.PROCESSING,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    #: SHA-256 of the uploaded bytes. Lineage (`document-processing` skill step 5) and the
    #: cheapest possible answer to "is this the same file we stored?".
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Extracted metadata: title, author, producer, section titles. Non-sensitive only —
    #: never document body text, which belongs in `document_chunk`.
    #: Attribute name avoids `Document.metadata`, which is SQLAlchemy's own.
    doc_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]

    uploader: Mapped[User] = relationship(lazy="raise")
    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        # `raise` rather than `select`: a lazy load inside an async request raises
        # MissingGreenlet at runtime, and it is better to fail in a test than in production.
        # Callers that want chunks ask the repository for them.
        lazy="raise",
        cascade="all, delete-orphan",
    )

    @property
    def is_retrievable(self) -> bool:
        return self.status in RETRIEVABLE_STATUSES

    @property
    def extension(self) -> str:
        _, _, ext = self.name.rpartition(".")
        return ext.lower()

    def __repr__(self) -> str:
        # No name: a filename can carry a customer identifier, and __repr__ reaches logs.
        return f"<Document id={self.id} v{self.version} status={self.status.value}>"


class DocumentChunk(Base):
    """One citable passage of a document.

    The unit of retrieval *and* the unit of citation — those are deliberately the same
    thing. If retrieval returned something larger than a citation, a source reference would
    point at a region the answer only partly came from.
    """

    __tablename__ = "document_chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_document_index"),
        # Every retrieval filters tenant + document status; this covers the join side.
        Index("ix_chunk_tenant_document", "tenant_id", "document_id"),
        # BR-03: a chunk must be citable. Page number, section label, or the row is invalid.
        CheckConstraint(
            "page_number IS NOT NULL OR length(trim(section_label)) > 0",
            name="ck_chunk_has_citation",
        ),
        CheckConstraint("length(trim(text)) > 0", name="ck_chunk_text_not_blank"),
    )

    id: Mapped[UuidPk]
    #: Denormalized from `document` so retrieval can filter tenant without a join. The
    #: vector search runs on every query; a join per query to re-derive a column we already
    #: know is a cost with no benefit.
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenant.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("document.id", ondelete="CASCADE"), nullable=False, index=True
    )
    #: Position within the document, 0-based. Ordering for display and for overlap logic.
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    #: 1-based page for PDFs. Null for a format with no pages, in which case the CHECK
    #: above requires `section_label`.
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    #: Heading or section title. Empty string when unknown *and* a page number exists.
    section_label: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    #: When this chunk's vector was acknowledged by Pinecone. Null means "not embedded yet",
    #: which is what the ingestion task and any reconciliation job key off.
    #:
    #: **The vector is not stored here.** It lives in Pinecone under the id `str(chunk.id)`,
    #: in the namespace for `tenant_id`. This column is the local record that the upsert
    #: succeeded — without it, "which chunks still need embedding?" could only be answered by
    #: fetching every id from the index and diffing, which is a network round trip to learn
    #: something we already knew at write time.
    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Rough token count, used to keep prompt assembly inside the model's window without
    #: re-tokenizing at query time.
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[CreatedAt]

    document: Mapped[Document] = relationship(back_populates="chunks", lazy="raise")

    @property
    def citation(self) -> str:
        """Human-readable location, for `section_or_page` in a source reference.

        Both parts when both are known — "Section 4.2 (page 7)" tells a compliance reviewer
        more than either alone, and locating the passage is the whole point of the field.

        Must render **byte-identically** to `ChunkSpan.citation` (`utils/chunk_splitter.py`)
        and `RetrievedChunk.citation` (`repositories/query_repository.py`), the two other
        places the same location becomes a string. `source_tracer` deduplicates references on
        `(document_id, citation)`, so "Page 1" from here and "page 1" from there would be two
        distinct locations for one page — the same passage cited twice in one answer.
        """
        if self.section_label and self.page_number is not None:
            return f"{self.section_label} (page {self.page_number})"
        if self.section_label:
            return self.section_label
        if self.page_number is not None:
            return f"page {self.page_number}"
        return "unspecified section"

    def __repr__(self) -> str:
        # No text: chunk contents are the document body, which must not reach a log.
        return f"<DocumentChunk doc={self.document_id} idx={self.chunk_index}>"
