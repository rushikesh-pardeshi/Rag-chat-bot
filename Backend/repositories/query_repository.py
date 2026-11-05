"""backend/repositories/query_repository.py

Role: The only place `query` is queried. Serves both the query endpoints and search history.

`list_for_user` and `list_for_tenant` are two methods rather than one with an optional filter
because they are two different authorizations. A single `list(user_id=None)` would make the
Administrator case the *default* — forget the argument and you get everyone's history. Here,
reading another user's rows requires calling a differently-named method, which is visible in
a diff and in a review.

The role decision itself is not made here; `search_history_service` (Phase 4) picks the
method. This file only guarantees that whichever one is called is correctly scoped.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import Select, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.models.document import Document, DocumentChunk, DocumentStatus
from backend.models.query import Query, QueryStatus


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A passage that has passed the citability re-check, with everything a citation needs.

    A DTO rather than a `DocumentChunk` entity, for two reasons. The architecture rule says a
    repository returns domain objects or DTOs and never raw rows — but more concretely, the
    citation needs `document.name` and `document.upload_date`, and reaching them from an
    entity would mean a lazy load. `DocumentChunk.document` is `lazy="raise"`, so that would
    be an exception inside the query path rather than a silent N+1. Selecting the columns
    outright is the fix the model is asking for.

    Carries no `score`: relevance belongs to the vector store, and duplicating it into a
    database DTO invites two answers to the same question. `rag_service` holds the score
    alongside this, from the `VectorMatch` it came from.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    text: str
    page_number: int | None
    section_label: str
    chunk_index: int
    upload_date: datetime

    @property
    def citation(self) -> str:
        """Location string for `SourceReference.section_or_page`.

        Never empty — that field is `min_length=1`, and an empty value would fail schema
        validation after the model has already been paid for.
        """
        if self.section_label and self.page_number is not None:
            return f"{self.section_label} (page {self.page_number})"
        if self.section_label:
            return self.section_label
        if self.page_number is not None:
            return f"page {self.page_number}"
        return "unspecified section"


class QueryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def _scoped(self, tenant_id: uuid.UUID) -> Select[tuple[Query]]:
        return select(Query).where(Query.tenant_id == tenant_id)

    # -- Write --------------------------------------------------------------
    async def create(
        self,
        *,
        query_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        question_text: str,
        status: QueryStatus = QueryStatus.PENDING,
    ) -> Query:
        """Persist the question before answering it.

        Written first, deliberately: the audit trail must record that a question was asked
        even if answering then fails or times out. A row created only on success would hide
        exactly the queries a compliance review most wants to see.
        """
        query = Query(
            id=query_id,
            tenant_id=tenant_id,
            user_id=user_id,
            question_text=question_text,
            status=status,
            source_references=[],
        )
        self._session.add(query)
        await self._session.flush()
        return query

    async def attach_answer(
        self,
        query_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        answer_text: str | None,
        source_references: list[dict[str, Any]],
        confidence_score: float | None,
        status: QueryStatus,
        latency_ms: int,
        answered_at: datetime,
    ) -> bool:
        """Attach the outcome. Returns False if the id is not this tenant's.

        Answer, sources, confidence and status go in one UPDATE. BR-03 says an answer without
        traceable sources must not exist, so there must be no instant at which the answer is
        visible and its references are not.
        """
        result = await self._session.execute(
            update(Query)
            .where(Query.id == query_id, Query.tenant_id == tenant_id)
            .values(
                answer_text=answer_text,
                source_references=source_references,
                confidence_score=confidence_score,
                status=status,
                latency_ms=latency_ms,
                answered_at=answered_at,
            )
        )
        return bool(result.rowcount)

    # -- Read ---------------------------------------------------------------
    async def get(self, query_id: uuid.UUID, tenant_id: uuid.UUID) -> Query | None:
        """Tenant-scoped. Ownership is a separate check, made by the service."""
        stmt = self._scoped(tenant_id).where(Query.id == query_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def load_chunks(
        self, chunk_ids: Sequence[uuid.UUID], tenant_id: uuid.UUID
    ) -> list[RetrievedChunk]:
        """Load retrieval candidates that are still citable, in the order asked for.

        **This method is the enforcement point for the whole grounding rule.** Pinecone
        returned ids; this decides which of them may be quoted. Three conditions, all in the
        WHERE clause rather than in Python, so there is no code path that can skip one:

          * the chunk belongs to `tenant_id` — a namespace mistake in the index cannot leak
            another tenant's passage past here;
          * its document also belongs to `tenant_id` — checked on the join as well as the
            chunk, because `document_chunk.tenant_id` is denormalized and two copies of a
            fact can disagree; and
          * its document is `ACTIVE` — a soft-deleted document stops being quotable the
            instant its status changes, not whenever its vectors leave the index.

        The last one is why `PINECONE_OVERFETCH_FACTOR` exists: the caller asks the index for
        more candidates than it needs, because this filter discards some of them.

        Order is the caller's, not the database's. `chunk_ids` arrives in Pinecone score
        order, and that ordering is the relevance ranking — re-sorting by anything else here
        would silently rank the prompt by primary key. Postgres returns a set, so the order is
        restored in Python against the caller's list rather than trusted from the result.

        An empty `chunk_ids` short-circuits: `IN ()` is not valid SQL everywhere and an empty
        retrieval is a normal outcome, not an error.
        """
        if not chunk_ids:
            return []

        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.text,
                DocumentChunk.page_number,
                DocumentChunk.section_label,
                DocumentChunk.chunk_index,
                Document.name,
                Document.upload_date,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.id.in_(list(chunk_ids)),
                DocumentChunk.tenant_id == tenant_id,
                Document.tenant_id == tenant_id,
                Document.status == DocumentStatus.ACTIVE,
            )
        )
        rows = (await self._session.execute(stmt)).all()

        by_id = {
            row.id: RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                document_name=row.name,
                text=row.text,
                page_number=row.page_number,
                section_label=row.section_label,
                chunk_index=row.chunk_index,
                upload_date=row.upload_date,
            )
            for row in rows
        }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]

    async def sample_chunks(
        self, tenant_id: uuid.UUID, *, documents: int, per_document: int
    ) -> list[RetrievedChunk]:
        """A few citable chunks per recently-uploaded document, newest document first.

        Read by `rag_service.suggest_questions`, which needs the corpus's own wording to build
        an example question out of. It carries the same three conditions as `load_chunks` —
        tenant on the chunk, tenant on the document, document `ACTIVE` — because a suggestion
        built from a soft-deleted document would quote a withdrawn document in the UI, which
        is the failure `load_chunks` exists to prevent arriving by a second route.

        One condition is added: `embedded_at IS NOT NULL`. A chunk with no vector is not in
        Pinecone, so a question phrased from it could never retrieve it, and the suggestion
        would be exactly the unanswerable example this is meant to replace.

        `chunk_index < per_document` is what makes the sample per-document rather than
        whole-corpus: the index is 0-based and continuous within a document
        (`utils/chunk_splitter.py`), so the bound selects each document's opening chunks —
        where a document states its subject — without a window function. `limit` is then a
        backstop for the row count, not the grouping.
        """
        if documents <= 0 or per_document <= 0:
            return []

        stmt = (
            select(
                DocumentChunk.id,
                DocumentChunk.document_id,
                DocumentChunk.text,
                DocumentChunk.page_number,
                DocumentChunk.section_label,
                DocumentChunk.chunk_index,
                Document.name,
                Document.upload_date,
            )
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(
                DocumentChunk.tenant_id == tenant_id,
                Document.tenant_id == tenant_id,
                Document.status == DocumentStatus.ACTIVE,
                DocumentChunk.embedded_at.is_not(None),
                DocumentChunk.chunk_index < per_document,
            )
            .order_by(Document.upload_date.desc(), DocumentChunk.chunk_index)
            .limit(documents * per_document)
        )
        rows = (await self._session.execute(stmt)).all()

        return [
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                document_name=row.name,
                text=row.text,
                page_number=row.page_number,
                section_label=row.section_label,
                chunk_index=row.chunk_index,
                upload_date=row.upload_date,
            )
            for row in rows
        ]

    async def list_for_user(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        status: QueryStatus | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Query]:
        """One user's own history. Both `tenant_id` and `user_id` are required."""
        stmt = self._apply_filters(
            self._scoped(tenant_id).where(Query.user_id == user_id),
            status=status,
            keyword=keyword,
            created_from=created_from,
            created_to=created_to,
        )
        return await self._page(stmt, limit=limit, offset=offset)

    async def list_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None = None,
        status: QueryStatus | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Query]:
        """Every user's history for the tenant — Administrator only.

        `user_id` here is a *filter* an Administrator may apply, not an authorization check.
        The authorization happened before this method was chosen.
        """
        stmt = self._scoped(tenant_id)
        if user_id is not None:
            stmt = stmt.where(Query.user_id == user_id)
        stmt = self._apply_filters(
            stmt,
            status=status,
            keyword=keyword,
            created_from=created_from,
            created_to=created_to,
        )
        return await self._page(stmt, limit=limit, offset=offset)

    async def count(
        self,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None = None,
        status: QueryStatus | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> int:
        """Total matching rows, for pagination.

        Takes `user_id` as an ordinary filter because the caller has already decided which
        listing it is entitled to; passing None here after calling `list_for_user` would be
        a bug in the service, and the service is where that pairing is tested.
        """
        stmt = select(func.count()).select_from(Query).where(Query.tenant_id == tenant_id)
        if user_id is not None:
            stmt = stmt.where(Query.user_id == user_id)
        stmt = self._apply_filters(
            stmt,
            status=status,
            keyword=keyword,
            created_from=created_from,
            created_to=created_to,
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def count_by_status(
        self,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> dict[QueryStatus, int]:
        """How many matching rows sit in each status, in **one** statement.

        `GROUP BY status` rather than a `count()` per status, and that is the whole point of
        the method: the frontend used to get this breakdown by calling the history endpoint
        once per outcome with `limit=1` and reading the total, which meant four round trips
        *and* four `HISTORY_VIEW` audit rows for one filter change. Three of those rows were
        counting reads indistinguishable from a person reading somebody's history, which makes
        "how often was this history viewed" unanswerable from the trail
        (`.claude/rules/audit-log-immutability.md`).

        Deliberately takes no `status` filter: filtering by status *and* grouping by it returns
        one row, which is the degenerate case the caller already has `count()` for.

        Statuses with no rows are absent from the result, exactly as SQL returns it. The
        service fills the zeros, because that is a presentation decision and this layer does
        not make them.
        """
        stmt = (
            select(Query.status, func.count())
            .where(Query.tenant_id == tenant_id)
            .group_by(Query.status)
        )
        if user_id is not None:
            stmt = stmt.where(Query.user_id == user_id)
        stmt = self._apply_filters(
            stmt,
            status=None,
            keyword=keyword,
            created_from=created_from,
            created_to=created_to,
        )
        rows = (await self._session.execute(stmt)).all()
        return {row[0]: int(row[1]) for row in rows}

    async def _page(self, stmt: Any, *, limit: int, offset: int) -> list[Query]:
        """One ordered page, with each row's author loaded.

        `Query.user` is `lazy="raise"`, so the eager load is not an optimisation — it is the
        only way the author can be read at all. History rows carry it because an Administrator
        looking at the whole tenant needs to know who asked, and the alternative was making
        them copy a user id out of the audit log. One extra `SELECT ... IN (...)` per page.
        """
        stmt = (
            stmt.options(selectinload(Query.user))
            .order_by(Query.created_at.desc(), Query.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self._session.execute(stmt)).scalars())

    @staticmethod
    def _apply_filters(
        stmt: Any,
        *,
        status: QueryStatus | None,
        keyword: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
    ) -> Any:
        if status is not None:
            stmt = stmt.where(Query.status == status)
        if keyword:
            # ILIKE with escaped wildcards, never a user-supplied regex: PostgreSQL's `~`
            # operator would let a crafted pattern burn CPU on backtracking (ReDoS), and
            # `.claude/rules/secure-coding-vapt.md` forbids user regex outright.
            escaped = (
                keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            )
            stmt = stmt.where(Query.question_text.ilike(f"%{escaped}%", escape="\\"))
        if created_from is not None:
            stmt = stmt.where(Query.created_at >= created_from)
        if created_to is not None:
            stmt = stmt.where(Query.created_at <= created_to)
        return stmt
