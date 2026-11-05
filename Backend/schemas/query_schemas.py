"""backend/schemas/query_schemas.py

Role: Pydantic DTOs for the query endpoints, including the source-reference contract.

`SourceReference` is the schema-level enforcement of BR-03 and
`.claude/rules/document-source-traceability.md`: `source_document_name` and `section_or_page`
are **required**, with no default and `min_length=1`. An answer whose sources are unknown
cannot be serialized — constructing a `SourceReference` without them raises before a response
is built, so "we lost the citation" surfaces as a server error rather than as a plausible
answer a compliance reviewer cannot trace.

`confidence_level` is optional because it genuinely may be unavailable; the location of the
source never is.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import NoInspectionAvailable

from backend.models.query import Query, QueryStatus
from backend.services.query_service import MAX_QUESTION_LENGTH, MIN_QUESTION_LENGTH
from backend.services.source_tracer import NOT_FOUND_ANSWER

_REQUEST_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)

#: Longest document name or section label echoed back. Both originate in an uploaded file.
MAX_REFERENCE_FIELD: Final = 512

#: Re-exported so existing importers keep working. It is *defined* in
#: `backend/services/source_tracer.py`, next to the code that decides when an answer is
#: unsupported — a second definition here would be a second thing to keep in step with the
#: text `rag-answer-grounding.md` requires verbatim.
__all__ = [
    "MAX_REFERENCE_FIELD",
    "NOT_FOUND_ANSWER",
    "QueryCreate",
    "QueryResponse",
    "QuerySuggestionsResponse",
    "SearchHistoryPage",
    "SourceReference",
]


class QueryCreate(BaseModel):
    """A submitted question.

    `extra="forbid"` matters more here than elsewhere: a body carrying `user_id`, `tenant_id`,
    `status` or `answer_text` is rejected outright. Those are all server-assigned, and a
    caller who could set `answer_text` could write a fabricated answer into the audit trail.
    """

    model_config = _REQUEST_CONFIG

    question_text: str = Field(
        min_length=MIN_QUESTION_LENGTH,
        max_length=MAX_QUESTION_LENGTH,
        description="The natural-language question, answered only from uploaded documents.",
    )
    follow_up_to: uuid.UUID | None = Field(
        default=None,
        description=(
            "The id of the query this one follows up on. Supplying it lets a question like "
            "'more details about it' resolve what 'it' refers to, for retrieval and for the "
            "prompt. The earlier turn is context for reading the question only: the answer "
            "still has to come from retrieved passages and cite them, so an unsupported "
            "follow-up is still answered 'not found'. An id that does not exist, belongs to "
            "another tenant, is not the caller's own, or was never answered is ignored and "
            "the question is answered standalone."
        ),
    )


class QuerySuggestionsResponse(BaseModel):
    """Example questions the caller's corpus has been checked to be able to answer.

    `items` is empty rather than absent when there is nothing to suggest, and that is a normal
    response: an empty workspace, a corpus of unreadable scans, or an index that is temporarily
    unreachable all arrive here as `[]`.

    **`retrieval_available` is what separates the last of those from the first two**, and it is
    here because collapsing them misled a real user: with expired AWS credentials the panel
    reported that the documents' text could not be extracted, which was a confident statement
    about their files made from no evidence at all. `false` means the verifier never ran, so
    nothing is known about the corpus; `true` with an empty `items` is a finding about the
    documents themselves.

    Every string is a question assembled server-side from document text, so it is
    length-capped and free of markup by construction. It is still untrusted content — it
    derives from an uploaded file — and a client renders it as text, exactly as it does
    `question_text`.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[str] = Field(default_factory=list, max_length=10)
    total_available: int = Field(
        default=0,
        ge=0,
        description=(
            "Verified questions behind this window. Greater than len(items) means asking "
            "for the next offset returns different examples; equal means a refresh would "
            "show the same ones back, so the control can be hidden."
        ),
    )
    retrieval_available: bool = Field(
        default=True,
        description="False when embedding or the vector index could not be reached at all.",
    )


class SourceReference(BaseModel):
    """Where an answer came from. Both location fields are mandatory (BR-03).

    Built from a `DocumentChunk`: `section_or_page` is the chunk's own `citation`, which is
    why the chunk table has a CHECK constraint requiring a page number or a section label.
    The requirement is enforced three times over — database, service, schema — because a
    citation that goes missing at any one of those layers produces an untraceable answer.
    """

    model_config = ConfigDict(extra="forbid")

    source_document_name: str = Field(min_length=1, max_length=MAX_REFERENCE_FIELD)
    section_or_page: str = Field(min_length=1, max_length=MAX_REFERENCE_FIELD)
    #: The document this reference points at, so the UI can link straight to the preview.
    document_id: uuid.UUID | None = None
    #: 0–1 when the retriever provides one.
    confidence_level: float | None = Field(default=None, ge=0.0, le=1.0)
    #: The retrieved passage, for "show me where". Optional: it is document body text, and a
    #: caller who may see the answer may see the passage, but nothing requires it to be sent.
    excerpt: str | None = Field(default=None, max_length=2_000)
    #: The `[S#]` numbers in `answer_text` that point at this reference — `[S2]` and `[S5]`
    #: both land here when two passages from the same page were merged into one citation.
    #: Ascending.
    #:
    #: Optional with an empty default for one reason only: rows written before this field
    #: existed have no such key, and `extra="forbid"` plus the revalidation in `from_entity`
    #: would turn every one of them into a 500. A client that finds it empty is looking at a
    #: historical row and should fall back to list position; a fresh answer always carries it.
    markers: list[int] = Field(
        default_factory=list,
        description="[S#] marker numbers in the answer that refer to this source.",
    )


def _author_fields(query: Query) -> tuple[str | None, str | None]:
    """`(email, full_name)` of whoever asked, or `(None, None)` if the author is not loaded.

    `Query.user` is `lazy="raise"`, so this must never touch the attribute blind: reading it on
    a row fetched without `selectinload` would raise inside response serialization, turning a
    successful query into a 500. Only `query_repository._page` eager-loads it, which is exactly
    the history path — `POST /queries` and `GET /queries/{id}` legitimately arrive here with an
    unloaded author and get nulls, because the caller of those already knows who they are.
    """
    try:
        unloaded = sa_inspect(query).unloaded
    except NoInspectionAvailable:
        # Not an instrumented instance — a stub in a unit test. Nothing to report.
        return (None, None)
    if "user" in unloaded:
        return (None, None)
    author = query.user
    if author is None:
        return (None, None)
    return (author.email, author.full_name)


class QueryResponse(BaseModel):
    """A query record — question, answer if there is one, and its sources.

    The validator is the last line of the traceability rule: an `ANSWERED` response with an
    empty `source_references` list is rejected here rather than returned. That state should be
    unreachable (the repository writes answer and sources in one statement), and this is what
    makes it *stay* unreachable when Phase 4 adds the generation path.
    """

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    question_text: str
    answer_text: str | None
    source_references: list[SourceReference] = Field(default_factory=list)
    confidence_score: float | None = None
    status: QueryStatus
    latency_ms: int | None = None
    created_at: datetime
    answered_at: datetime | None = None
    #: Who asked. Populated on history rows, where an Administrator sees the whole tenant and
    #: needs the question attributed; null on `POST /queries` and `GET /queries/{id}`, whose
    #: caller is either the author or already holds the row.
    #:
    #: This is not a new disclosure. The scope of a history page is decided in
    #: `search_history_service` before any of it is serialized — a Standard User only ever
    #: receives their own rows, so the only reader who can see another person's address here is
    #: an Administrator, who can already list every user in the tenant.
    #:
    #: `email` is the identity in this system (`rbac-enforcement.md`); `full_name` is nullable
    #: on `user_account`, so a client displays it when set and falls back to the address.
    user_email: str | None = None
    user_full_name: str | None = None

    @model_validator(mode="after")
    def answered_must_cite(self) -> QueryResponse:
        if self.status is QueryStatus.ANSWERED and not self.source_references:
            raise ValueError(
                "An answered query must carry at least one source reference (BR-03)."
            )
        return self

    @classmethod
    def from_entity_list(cls, queries: Sequence[Query]) -> list[QueryResponse]:
        """Map a page of rows. Convenience only — `from_entity` holds the rules."""
        return [cls.from_entity(query) for query in queries]

    @classmethod
    def from_entity(cls, query: Query) -> QueryResponse:
        """Explicit field mapping, with the stored JSONB revalidated on the way out.

        `source_references` is JSONB written by the service, so it is trusted-ish — but it is
        re-parsed through `SourceReference` anyway. A row that somehow holds a reference with
        no `section_or_page` must fail loudly here, not be published as a citation with a
        missing location.
        """
        email, full_name = _author_fields(query)
        return cls(
            id=query.id,
            question_text=query.question_text,
            answer_text=query.answer_text,
            source_references=[
                SourceReference.model_validate(reference)
                for reference in (query.source_references or [])
            ],
            confidence_score=query.confidence_score,
            status=query.status,
            latency_ms=query.latency_ms,
            created_at=query.created_at,
            answered_at=query.answered_at,
            user_email=email,
            user_full_name=full_name,
        )


class SearchHistoryPage(BaseModel):
    """A page of search history plus the total, for the history table's pager.

    Reuses `QueryResponse` for the rows rather than defining a leaner history DTO. A separate
    shape would be a second place the source-reference contract has to be enforced, and
    `architecture.md` makes the same point about the history *table*: two representations of an
    audited fact are two things that can drift.

    `scope` is `"own"` or `"tenant"`, straight from the service. The UI needs it to label an
    Administrator's view accurately instead of implying the rows are the viewer's own activity.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[QueryResponse]
    total: int
    limit: int
    offset: int
    scope: str


class SearchHistoryStats(BaseModel):
    """Outcome counts for a history filter, with no rows paged.

    Exists so the history page's tiles cost one request and **one** `HISTORY_VIEW` audit row.
    They were previously assembled in the browser by calling the list endpoint once per
    outcome with `limit=1`, which wrote a row per call: four for one filter change, three of
    them counting reads indistinguishable in the trail from a person reading somebody's
    history. The audit log is the product here, so that had to stop.

    `by_status` is a mapping keyed by `QueryStatus` value, with every status present and zeros
    included — a tile row needs them all, and SQL omits a group with no rows.

    `scope` is `"own"` or `"tenant"`, as on `SearchHistoryPage`, and carries more weight here:
    a bare total with no rows beside it gives a reader nothing else to infer the scope from.
    """

    model_config = ConfigDict(extra="forbid")

    scope: str
    total: int
    by_status: dict[str, int]
