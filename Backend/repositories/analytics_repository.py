"""backend/repositories/analytics_repository.py

Role: The only place the analytics aggregates are queried. Five reads over `query` — keyword
frequency, document citation counts, volume per period, per-user activity, and the outcome and
latency summary — each one a single grouped statement.

Aggregation happens in PostgreSQL, not in Python
------------------------------------------------
`WORKBREAKDOWN.md` requires it ("`GROUP BY` in SQL, not in Python") and the reason is size.
Counting keywords in Python means selecting every `question_text` for the window into the web
process; a tenant with a year of history would move tens of megabytes across the wire to
produce twenty rows. Every method here returns at most `limit` rows and PostgreSQL does the
grouping — which is also what makes the indexes on `(tenant_id, created_at DESC)` worth having.

Tokenization is a fixed pattern
-------------------------------
`KEYWORD_SPLIT_PATTERN` is a module constant and never comes from a caller.
`.claude/rules/secure-coding-vapt.md` forbids user-supplied regex outright (ReDoS), and
`regexp_split_to_table` running inside the database makes that worse rather than better: a
crafted pattern would burn *PostgreSQL's* CPU, where a statement timeout is the only thing that
would stop it. The stopword list is passed in as a bound parameter list, because *which* words
are uninteresting is a policy decision and policy belongs to `analytics_service`.

Every filter value — the window, the limits, the stopwords, the `date_trunc` granularity — is
bound, never interpolated. `granularity` additionally comes from an enum, so it cannot be an
arbitrary string even before binding.

Tenant scope, every statement
-----------------------------
`tenant_id` is a required positional argument on all five methods, and there is no unscoped
variant to reach for by mistake. Analytics is the read most likely to be written as "count
everything" — a single missing `WHERE` here would aggregate another bank's questions into this
one's dashboard, which is a disclosure even though no individual row is shown.
"""

from __future__ import annotations

import enum
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

from sqlalchemy import Select, distinct, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.query import Query, QueryStatus
from backend.models.user import User

#: How a question is split into candidate keywords. Fixed, tested, and never caller-supplied.
#: Splits on every run of characters that is not a letter, a digit or an internal apostrophe,
#: so "customer's" survives as one token while "account-number" becomes two.
KEYWORD_SPLIT_PATTERN: Final = r"[^a-z0-9']+"


class TrendGranularity(enum.StrEnum):
    """The `date_trunc` units the trend report allows.

    An enum rather than a string parameter. The value is bound, so a free string would not be
    an injection — but it would be a 500 from PostgreSQL on every typo, and the set of periods
    a dashboard can render is a closed set anyway.
    """

    DAY = "day"
    WEEK = "week"
    MONTH = "month"


@dataclass(frozen=True, slots=True)
class KeywordCount:
    """One extracted keyword and how often it was asked about.

    `occurrences` counts every appearance, `queries` counts distinct questions. They differ
    when one question repeats a word, and the pair is more honest than either alone: a single
    question about "wire wire wire" is not three people asking about wires.
    """

    keyword: str
    occurrences: int
    queries: int


@dataclass(frozen=True, slots=True)
class DocumentUsage:
    """How often a document was actually cited in an answer.

    Read out of `query.source_references`, not out of the `document` table, and that is the
    point: this measures what answers were *grounded in*, which is the number a compliance
    reviewer asks about. A document nobody's question ever touched does not appear here at all.

    `document_name` is the name recorded at answer time, so a document later renamed or deleted
    still reports under the name the answer cited — which is what makes the old answer's
    citation followable.
    """

    document_id: str | None
    document_name: str
    citations: int
    queries: int
    last_cited_at: datetime | None


@dataclass(frozen=True, slots=True)
class TrendPoint:
    """One bucket of the volume trend, split by outcome.

    `not_found` is reported beside `answered` rather than folded into a failure count. A
    not-found is the grounding rule working correctly; a rising not-found rate is a gap in the
    *document repository*, not an error rate, and the dashboard has to be able to say so.
    """

    period_start: datetime
    total: int
    answered: int
    not_found: int
    failed: int


@dataclass(frozen=True, slots=True)
class ActiveUser:
    """One user's activity within the window."""

    user_id: uuid.UUID
    email: str
    queries: int
    last_query_at: datetime | None


@dataclass(frozen=True, slots=True)
class QuerySummary:
    """Totals for the window, including the latency percentile.

    `p95_latency_ms` is computed by PostgreSQL (`percentile_cont`) over `query.latency_ms`,
    which `query_service` records around the whole pipeline. That makes this the production
    measurement for the 3-5 second target in the requirements — a figure from real traffic
    rather than from a fixture run, which is what the Phase 4 gate asks to be written down.
    """

    total: int
    answered: int
    not_found: int
    failed: int
    pending: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    avg_confidence: float | None


class AnalyticsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Keywords -----------------------------------------------------------
    async def keyword_counts(
        self,
        tenant_id: uuid.UUID,
        *,
        stopwords: Sequence[str],
        min_length: int,
        limit: int,
        ascending: bool = False,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[KeywordCount]:
        """Keyword frequencies for the window, most common first — or least, if `ascending`.

        Two directions from one statement because "most searched" and "least searched" are the
        same aggregate read in opposite order, and the dashboard shows both side by side
        (BR-06). Splitting them into two methods would duplicate the tokenization, and two
        copies of a tokenizer drift into disagreeing about what a word is.

        Ties break alphabetically. Without that the bottom of a long tail — where every count
        is 1 — would come back in whatever order the plan happened to produce, and the "least
        searched" panel would change on every refresh with no data having changed.
        """
        tokens = self._token_subquery(
            tenant_id, created_from=created_from, created_to=created_to
        )
        occurrences = func.count().label("occurrences")
        stmt = (
            select(
                tokens.c.word.label("keyword"),
                occurrences,
                func.count(distinct(tokens.c.query_id)).label("queries"),
            )
            .where(
                func.length(tokens.c.word) >= min_length,
                # A bound list, expanded by the driver. The stopword *policy* lives in
                # `analytics_service`; this is only where it is applied.
                tokens.c.word.not_in(list(stopwords)),
            )
            .group_by(tokens.c.word)
            .order_by(
                occurrences.asc() if ascending else occurrences.desc(),
                tokens.c.word.asc(),
            )
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            KeywordCount(keyword=row.keyword, occurrences=row.occurrences, queries=row.queries)
            for row in rows
        ]

    def _token_subquery(
        self,
        tenant_id: uuid.UUID,
        *,
        created_from: datetime | None,
        created_to: datetime | None,
    ) -> Any:
        """One row per (query, word), lower-cased, from the fixed split pattern.

        `regexp_split_to_table` sits in the SELECT list of a subquery rather than in a LATERAL
        join: PostgreSQL expands a set-returning function there row-by-row, and the subquery
        form keeps the outer statement a plain `GROUP BY` that reads like what it is.
        """
        stmt = select(
            Query.id.label("query_id"),
            func.regexp_split_to_table(
                func.lower(Query.question_text), KEYWORD_SPLIT_PATTERN
            ).label("word"),
        ).where(Query.tenant_id == tenant_id)
        stmt = self._window(stmt, created_from=created_from, created_to=created_to)
        return stmt.subquery("tokens")

    # -- Documents ----------------------------------------------------------
    async def document_usage(
        self,
        tenant_id: uuid.UUID,
        *,
        limit: int,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[DocumentUsage]:
        """Most-cited documents first, read out of the answers' own source references.

        `jsonb_array_elements` unnests `source_references`, so a query citing three documents
        contributes to three rows — which is the correct denominator for "how often was this
        document the basis of an answer".

        Grouped on the id *and* the name. Grouping on the id alone would drop the name from the
        result, and grouping on the name alone would merge two documents that share a name
        across versions — a re-uploaded fee schedule is a different document with the same
        name, and the citation named a specific one.
        """
        references = (
            select(
                Query.id.label("query_id"),
                Query.created_at.label("created_at"),
                # `type_=JSONB` is what makes `["key"].astext` below legal: without it the
                # function's return type is unknown and the index operator will not compile.
                func.jsonb_array_elements(Query.source_references, type_=JSONB).label(
                    "reference"
                ),
            )
            .where(Query.tenant_id == tenant_id)
            .where(Query.status == QueryStatus.ANSWERED)
            # `jsonb_array_elements` raises on a JSON scalar, and a raised error here is the
            # whole dashboard returning 500 rather than one row being wrong. The column is
            # `NOT NULL` with a `list` default so a scalar should be unreachable — but "should
            # be" is doing a lot of work for a statement that aggregates every row in the
            # tenant, and one hand-written row would take the page down.
            .where(func.jsonb_typeof(Query.source_references) == "array")
        )
        references = self._window(references, created_from=created_from, created_to=created_to)
        unnested = references.subquery("refs")

        document_id = unnested.c.reference["document_id"].astext.label("document_id")
        document_name = unnested.c.reference["source_document_name"].astext.label(
            "document_name"
        )
        citations = func.count().label("citations")
        stmt = (
            select(
                document_id,
                document_name,
                citations,
                func.count(distinct(unnested.c.query_id)).label("queries"),
                func.max(unnested.c.created_at).label("last_cited_at"),
            )
            .group_by(document_id, document_name)
            .order_by(citations.desc(), document_name.asc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            DocumentUsage(
                document_id=row.document_id,
                # A reference written before the name was recorded would group under an empty
                # string; the dashboard needs something to render either way.
                document_name=row.document_name or "unnamed document",
                citations=row.citations,
                queries=row.queries,
                last_cited_at=row.last_cited_at,
            )
            for row in rows
        ]

    # -- Trends -------------------------------------------------------------
    async def volume_trend(
        self,
        tenant_id: uuid.UUID,
        *,
        granularity: TrendGranularity,
        limit: int,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[TrendPoint]:
        """Query volume per period, oldest first, split by outcome.

        `FILTER (WHERE ...)` rather than `SUM(CASE ...)`: one pass, and the intent is on the
        line. Ordered ascending because a trend line is drawn left to right — but `limit` takes
        the *newest* periods, since a dashboard showing "the last 30 days" must not silently
        show the first 30 days of a two-year history.
        """
        period = func.date_trunc(granularity.value, Query.created_at).label("period_start")
        stmt = select(
            period,
            func.count().label("total"),
            func.count().filter(Query.status == QueryStatus.ANSWERED).label("answered"),
            func.count().filter(Query.status == QueryStatus.NOT_FOUND).label("not_found"),
            func.count().filter(Query.status == QueryStatus.FAILED).label("failed"),
        ).where(Query.tenant_id == tenant_id)
        stmt = self._window(stmt, created_from=created_from, created_to=created_to)
        newest_first = (
            stmt.group_by(period).order_by(period.desc()).limit(limit).subquery("trend")
        )
        ordered = select(newest_first).order_by(newest_first.c.period_start.asc())
        rows = (await self._session.execute(ordered)).all()
        return [
            TrendPoint(
                period_start=row.period_start,
                total=row.total,
                answered=row.answered,
                not_found=row.not_found,
                failed=row.failed,
            )
            for row in rows
        ]

    # -- Users --------------------------------------------------------------
    async def active_users(
        self,
        tenant_id: uuid.UUID,
        *,
        limit: int,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[ActiveUser]:
        """Who asked how much, busiest first.

        Both sides of the join are scoped to the tenant. `Query.user_id` is a foreign key into
        `user_account`, so the join alone would be safe — but "the join implies it" is exactly
        the reasoning that stops being true when someone adds an `OUTER` to it.
        """
        queries = func.count(Query.id).label("queries")
        stmt = (
            select(
                User.id.label("user_id"),
                User.email.label("email"),
                queries,
                func.max(Query.created_at).label("last_query_at"),
            )
            .join(Query, Query.user_id == User.id)
            .where(Query.tenant_id == tenant_id, User.tenant_id == tenant_id)
        )
        stmt = self._window(stmt, created_from=created_from, created_to=created_to)
        stmt = (
            stmt.group_by(User.id, User.email)
            .order_by(queries.desc(), User.email.asc())
            .limit(limit)
        )
        rows = (await self._session.execute(stmt)).all()
        return [
            ActiveUser(
                user_id=row.user_id,
                email=row.email,
                queries=row.queries,
                last_query_at=row.last_query_at,
            )
            for row in rows
        ]

    # -- Summary ------------------------------------------------------------
    async def summary(
        self,
        tenant_id: uuid.UUID,
        *,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> QuerySummary:
        """Outcome counts, average confidence, and the latency mean and p95.

        Always returns a row: the aggregates are over the whole window with no `GROUP BY`, so
        an empty window yields zeros and nulls rather than nothing. A dashboard that had to
        distinguish "no rows" from "zero queries" would show a blank panel on a tenant's first
        day, which reads as a broken page.
        """
        stmt = select(
            func.count().label("total"),
            func.count().filter(Query.status == QueryStatus.ANSWERED).label("answered"),
            func.count().filter(Query.status == QueryStatus.NOT_FOUND).label("not_found"),
            func.count().filter(Query.status == QueryStatus.FAILED).label("failed"),
            func.count().filter(Query.status == QueryStatus.PENDING).label("pending"),
            func.avg(Query.latency_ms).label("avg_latency_ms"),
            # Interpolated rather than discrete: with a handful of samples `percentile_disc`
            # would report an actual observation, which reads as more precise than it is.
            func.percentile_cont(0.95)
            .within_group(Query.latency_ms.asc())
            .label("p95_latency_ms"),
            func.avg(Query.confidence_score).label("avg_confidence"),
        ).where(Query.tenant_id == tenant_id)
        stmt = self._window(stmt, created_from=created_from, created_to=created_to)
        row = (await self._session.execute(stmt)).one()
        return QuerySummary(
            total=row.total,
            answered=row.answered,
            not_found=row.not_found,
            failed=row.failed,
            pending=row.pending,
            avg_latency_ms=_as_float(row.avg_latency_ms),
            p95_latency_ms=_as_float(row.p95_latency_ms),
            avg_confidence=_as_float(row.avg_confidence),
        )

    # -- Shared -------------------------------------------------------------
    @staticmethod
    def _window(
        stmt: Select[Any],
        *,
        created_from: datetime | None,
        created_to: datetime | None,
    ) -> Select[Any]:
        """The date window, applied identically by every method.

        Inclusive on both ends, matching `query_repository._apply_filters`. Two reads of the
        same window disagreeing by a boundary row is the kind of discrepancy that makes a
        reviewer distrust both numbers.
        """
        if created_from is not None:
            stmt = stmt.where(Query.created_at >= created_from)
        if created_to is not None:
            stmt = stmt.where(Query.created_at <= created_to)
        return stmt


def _as_float(value: Any) -> float | None:
    """PostgreSQL returns `Decimal` for `avg`; JSON and the charts want a float.

    Converted here rather than in the service, so the DTO's declared type is the truth. `None`
    survives as `None` — an average over no rows is not zero, and rendering it as zero would
    claim a latency of nothing at all.
    """
    if value is None:
        return None
    return float(value)
