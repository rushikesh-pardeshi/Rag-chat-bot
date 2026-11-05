"""backend/schemas/document_schemas.py

Role: Pydantic DTOs for the document endpoints. No ORM entity ever leaves a route.

Every response is built by an explicit `from_entity` classmethod rather than by
`model_validate(orm_object)`. That is not ceremony: `from_attributes` would pick up whatever
the model happens to expose today, so adding a column to `Document` later would silently start
publishing it. Listing the fields means a new column reaches the API only when someone decides
it should.

`s3_key` is deliberately absent from every response. It is an internal storage locator; a
client that knows it learns the bucket layout and gains nothing it cannot get from the
presigned preview URL, which is scoped and expires.

`metadata` is filtered rather than passed through. It holds values extracted from an
attacker-supplied file — a PDF `/Title` is arbitrary text — so only known-safe keys are
published, already length-capped by the parser.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

from backend.models.document import Document, DocumentStatus

#: Shared by every request model here. `extra="forbid"` is the mass-assignment control:
#: a body naming `status`, `tenant_id`, `version` or `s3_key` is a 422, not a silent no-op.
_REQUEST_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)

#: Metadata keys allowed out to a client. An allowlist, because the values originate in an
#: uploaded file. Anything else the parser stored (`failed_pages`, producer strings) stays
#: server-side where it is diagnostic rather than published.
PUBLIC_METADATA_KEYS: Final[frozenset[str]] = frozenset(
    {
        "format",
        "title",
        "author",
        "subject",
        "page_count",
        "pages_extracted",
        "section_count",
        "sections",
        "chunk_count",
        "truncated",
        "duplicate_content",
        "chunk_limit_reached",
    }
)

MAX_PAGE_SIZE: Final = 100
DEFAULT_PAGE_SIZE: Final = 25


def _public_metadata(raw: dict[str, Any] | None) -> dict[str, Any]:
    return {key: value for key, value in (raw or {}).items() if key in PUBLIC_METADATA_KEYS}


class DocumentListParams(BaseModel):
    """Query-string filters for `GET /documents`.

    A model rather than loose `Query(...)` parameters so the bounds live in one place and
    `limit` cannot be raised past `MAX_PAGE_SIZE` by a caller asking for 1,000,000 rows.
    """

    model_config = _REQUEST_CONFIG

    status: DocumentStatus | None = None
    #: Substring match on the document name. Escaped and bound as a parameter by the
    #: repository — it is a filter, not a pattern language, and never a regex (ReDoS).
    search: str | None = Field(default=None, max_length=255)
    limit: int = Field(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE)
    offset: int = Field(default=0, ge=0)


class DocumentResponse(BaseModel):
    """One document as the API presents it."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    name: str
    content_type: str
    file_size: int
    page_count: int | None
    status: DocumentStatus
    version: int
    upload_date: datetime
    uploaded_by: uuid.UUID
    #: SHA-256 of the stored bytes. Published because it lets an Administrator verify that
    #: what they downloaded is what was ingested — the point of a lineage fingerprint.
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Convenience for the UI, computed server-side so the client cannot disagree with the
    #: retrieval rule about which statuses are searchable.
    is_retrievable: bool

    @classmethod
    def from_entity(cls, document: Document) -> DocumentResponse:
        return cls(
            id=document.id,
            name=document.name,
            content_type=document.content_type,
            file_size=document.file_size,
            page_count=document.page_count,
            status=document.status,
            version=document.version,
            upload_date=document.upload_date,
            uploaded_by=document.uploaded_by,
            content_hash=document.content_hash,
            metadata=_public_metadata(document.doc_metadata),
            is_retrievable=document.is_retrievable,
        )


class DocumentListResponse(BaseModel):
    """A page of documents plus the totals a pager needs."""

    model_config = ConfigDict(extra="forbid")

    items: list[DocumentResponse]
    total: int
    limit: int
    offset: int

    @property
    def has_more(self) -> bool:
        return self.offset + len(self.items) < self.total


class DocumentPreviewResponse(BaseModel):
    """A presigned download URL and its lifetime.

    `expires_in_seconds` is returned so the UI can re-request rather than hand the user a
    dead link — and so it is visible in a review that the TTL really is minutes.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    name: str
    content_type: str
    url: str
    expires_in_seconds: int


class DocumentPurgeResponse(BaseModel):
    """What a permanent deletion destroyed.

    Counts rather than an empty 204, because this is the one operation in the system with
    nothing to undo it: an Administrator should be able to see that the vectors and the
    stored file went too, not just the row. `vectors_deleted` is `-1` when no vector store is
    configured, which is different from `0` — nothing in the index versus no index to ask.

    `name` is echoed so the UI can name the document in its confirmation; it is not retained
    anywhere, since the row it came from no longer exists.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: uuid.UUID
    name: str
    chunks_deleted: int
    vectors_deleted: int
    s3_versions_deleted: int


class DocumentVersionsResponse(BaseModel):
    """Every version of one document name — the lineage view (BR-01)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    versions: list[DocumentResponse]
