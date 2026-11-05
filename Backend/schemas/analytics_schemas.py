"""backend/schemas/analytics_schemas.py

Role: Pydantic DTOs for the analytics endpoints — the request filters, the three report
responses, and the export request.

Every request model is `extra="forbid"`
---------------------------------------
`secure-coding-vapt.md` (Mass Assignment) requires the allowed fields to be named explicitly.
It matters more here than it looks, because the tempting extra fields are exactly the dangerous
ones: a caller who could send `tenant_id` would be asking for another bank's numbers, and one
who could send `user_id` on a keyword filter would be asking what a named colleague searches
for. Neither field exists on these models, and an unexpected key is a 422 rather than a
silently-ignored parameter.

Bounds live on the fields as well as in the service
---------------------------------------------------
`limit` is `ge=1, le=MAX_LIMIT` here and clamped again in `analytics_service`. The duplication
is deliberate: the schema turns an out-of-range request into a clear 422 at the edge, and the
service's clamp is what protects a caller that is not an HTTP request at all — a scheduled
export, a test, a future internal job.

Response text is user-derived
-----------------------------
`keyword` and `document_name` originate in typed questions and uploaded filenames. They are
returned as data, never as markup, and the UI must render them as text — React's default
escaping does that, and `SourceReference` in `query_schemas.py` carries the same warning.
Nothing here builds HTML.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.repositories.analytics_repository import (
    ActiveUser,
    DocumentUsage,
    KeywordCount,
    QuerySummary,
    TrendGranularity,
    TrendPoint,
)
from backend.services.analytics_service import (
    DEFAULT_LIMIT,
    DEFAULT_TREND_POINTS,
    MAX_LIMIT,
    MAX_TREND_POINTS,
    DocumentUsageReport,
    ExportFormat,
    ExportReport,
    KeywordTrends,
    ReportWindow,
    UsageTrendReport,
)

_REQUEST_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)
_RESPONSE_CONFIG = ConfigDict(extra="forbid")

#: Longest keyword or document name echoed back. Both are user-derived, and a response field
#: without a bound is a response size without a bound.
MAX_LABEL_LENGTH: Final = 512

__all__ = [
    "ActiveUserOut",
    "AnalyticsExportRequest",
    "AnalyticsFilter",
    "DocumentUsageOut",
    "DocumentUsageResponse",
    "KeywordCountOut",
    "KeywordTrendResponse",
    "QuerySummaryOut",
    "TrendFilter",
    "TrendPointOut",
    "UsageTrendResponse",
    "WindowOut",
]


# -- Requests ---------------------------------------------------------------
class AnalyticsFilter(BaseModel):
    """The window and row count shared by every analytics read.

    `extra="forbid"` here guards *programmatic* construction — a service or test that passes a
    field this model does not define fails immediately instead of silently dropping it.

    It does **not** reject an unknown query string. The routes in `api/routes/analytics.py`
    declare `created_from`, `created_to` and `limit` as individual query parameters and build
    this model inside the handler, so an undeclared parameter never reaches Pydantic: FastAPI
    ignores it and the report is returned unfiltered. A typo'd `?form=2026-01-01` therefore
    succeeds and answers a wider question than the caller asked.

    That is a known gap, not an authorization one — the tenant and role filters are derived
    from the token server-side and are never caller-supplied. It is recorded in
    `backend/tests/security/test_mass_assignment.py`, which asserts the behaviour that exists
    rather than the behaviour this docstring used to claim. Closing it means changing the query
    contract on every read route, which needs a decision rather than a quiet edit.
    """

    model_config = _REQUEST_CONFIG

    created_from: datetime | None = Field(
        default=None, description="Inclusive start of the window. Omit for all history."
    )
    created_to: datetime | None = Field(
        default=None, description="Inclusive end of the window. Omit for up to now."
    )
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)

    @model_validator(mode="after")
    def window_must_be_ordered(self) -> AnalyticsFilter:
        """Reject an inverted window instead of swapping it.

        Silently swapping would answer a question the caller did not ask, and a report that
        reinterprets its own filter is a figure nobody can reproduce.
        """
        if self.created_from and self.created_to and self.created_from > self.created_to:
            raise ValueError("created_from must not be after created_to.")
        return self


class TrendFilter(AnalyticsFilter):
    """`AnalyticsFilter` plus the bucket size and how many buckets.

    `granularity` is the enum, so `?granularity=drop table` is a 422 at the edge and never
    reaches `date_trunc`. The value is bound as a parameter in the repository too — this is the
    outer of the two controls, not the only one.
    """

    model_config = _REQUEST_CONFIG

    granularity: TrendGranularity = TrendGranularity.DAY
    points: int = Field(default=DEFAULT_TREND_POINTS, ge=1, le=MAX_TREND_POINTS)


class AnalyticsExportRequest(BaseModel):
    """Which report to render, and in which format.

    A POST body rather than a GET query string, for two reasons that both matter here. It is a
    state-*recording* operation — every export writes a `DATA_EXPORT` audit row — and it must
    therefore carry the anti-CSRF token, which the middleware requires on unsafe methods only
    (`secure-coding-vapt.md`). And a filename is never accepted: the server composes it,
    because a caller-supplied name reaches a `Content-Disposition` header.
    """

    model_config = _REQUEST_CONFIG

    report: ExportReport
    format: ExportFormat
    granularity: TrendGranularity = TrendGranularity.DAY
    limit: int = Field(default=MAX_LIMIT, ge=1, le=MAX_LIMIT)
    created_from: datetime | None = None
    created_to: datetime | None = None

    @model_validator(mode="after")
    def window_must_be_ordered(self) -> AnalyticsExportRequest:
        if self.created_from and self.created_to and self.created_from > self.created_to:
            raise ValueError("created_from must not be after created_to.")
        return self


# -- Response parts ---------------------------------------------------------
class WindowOut(BaseModel):
    """The window the figures cover, echoed back with the label used on exports.

    Returned rather than assumed: a caller that sent no window needs to know what it got, and a
    dashboard tile reading "1,204 queries" means nothing without the period beside it.
    """

    model_config = _RESPONSE_CONFIG

    created_from: datetime | None
    created_to: datetime | None
    label: str

    @classmethod
    def from_dto(cls, window: ReportWindow) -> WindowOut:
        return cls(
            created_from=window.created_from,
            created_to=window.created_to,
            label=window.label,
        )


class QuerySummaryOut(BaseModel):
    """Outcome counts and latency for the window.

    `p95_latency_ms` is measured by PostgreSQL over `query.latency_ms`. It is the production
    figure for the 3–5 second target in the requirements — worth knowing, because a p95 from a
    fixture run measures the test harness rather than the system.
    """

    model_config = _RESPONSE_CONFIG

    total: int
    answered: int
    not_found: int
    failed: int
    pending: int
    avg_latency_ms: float | None
    p95_latency_ms: float | None
    avg_confidence: float | None

    @classmethod
    def from_dto(cls, summary: QuerySummary) -> QuerySummaryOut:
        return cls(
            total=summary.total,
            answered=summary.answered,
            not_found=summary.not_found,
            failed=summary.failed,
            pending=summary.pending,
            avg_latency_ms=summary.avg_latency_ms,
            p95_latency_ms=summary.p95_latency_ms,
            avg_confidence=summary.avg_confidence,
        )


class KeywordCountOut(BaseModel):
    """One keyword and its two counts.

    `queries` is the number of distinct questions containing the word, `occurrences` every
    appearance. Both are returned because one question repeating a word is not two people
    asking about it, and a ranking that cannot distinguish those invites a wrong conclusion.
    """

    model_config = _RESPONSE_CONFIG

    keyword: str = Field(max_length=MAX_LABEL_LENGTH)
    occurrences: int
    queries: int

    @classmethod
    def from_dto(cls, item: KeywordCount) -> KeywordCountOut:
        return cls(keyword=item.keyword, occurrences=item.occurrences, queries=item.queries)


class DocumentUsageOut(BaseModel):
    """One document and how often answers cited it.

    `document_id` may be null: the citation is read from the answer's stored source references,
    and a reference written before the id was recorded still counts as usage. The name is the
    one the answer cited, which is what makes an old citation followable after a rename.
    """

    model_config = _RESPONSE_CONFIG

    document_id: uuid.UUID | None
    document_name: str = Field(max_length=MAX_LABEL_LENGTH)
    citations: int
    queries: int
    last_cited_at: datetime | None

    @classmethod
    def from_dto(cls, item: DocumentUsage) -> DocumentUsageOut:
        return cls(
            # The repository reads this out of JSONB as text, so it is parsed here rather than
            # trusted. A malformed id becomes null instead of failing the whole report: the
            # usage count is still true, and the UI simply cannot deep-link that row.
            document_id=_as_uuid(item.document_id),
            document_name=item.document_name,
            citations=item.citations,
            queries=item.queries,
            last_cited_at=item.last_cited_at,
        )


class TrendPointOut(BaseModel):
    """One bucket of the volume trend.

    `not_found` is its own field, not folded into `failed`. A not-found is the grounding rule
    working as specified (`rag-answer-grounding.md`); a rising not-found rate is a gap in the
    document repository, and a chart that called it an error would send someone to fix the
    wrong thing.
    """

    model_config = _RESPONSE_CONFIG

    period_start: datetime
    total: int
    answered: int
    not_found: int
    failed: int

    @classmethod
    def from_dto(cls, point: TrendPoint) -> TrendPointOut:
        return cls(
            period_start=point.period_start,
            total=point.total,
            answered=point.answered,
            not_found=point.not_found,
            failed=point.failed,
        )


class ActiveUserOut(BaseModel):
    """One user's query count. Administrator-only, like everything in this module."""

    model_config = _RESPONSE_CONFIG

    user_id: uuid.UUID
    email: str = Field(max_length=MAX_LABEL_LENGTH)
    queries: int
    last_query_at: datetime | None

    @classmethod
    def from_dto(cls, user: ActiveUser) -> ActiveUserOut:
        return cls(
            user_id=user.user_id,
            email=user.email,
            queries=user.queries,
            last_query_at=user.last_query_at,
        )


# -- Responses --------------------------------------------------------------
class KeywordTrendResponse(BaseModel):
    """Most- and least-searched keywords for the window, with its totals."""

    model_config = _RESPONSE_CONFIG

    window: WindowOut
    most_searched: list[KeywordCountOut]
    least_searched: list[KeywordCountOut]
    summary: QuerySummaryOut

    @classmethod
    def from_dto(cls, trends: KeywordTrends) -> KeywordTrendResponse:
        return cls(
            window=WindowOut.from_dto(trends.window),
            most_searched=[KeywordCountOut.from_dto(item) for item in trends.most_searched],
            least_searched=[KeywordCountOut.from_dto(item) for item in trends.least_searched],
            summary=QuerySummaryOut.from_dto(trends.summary),
        )


class DocumentUsageResponse(BaseModel):
    """Which documents answers were grounded in."""

    model_config = _RESPONSE_CONFIG

    window: WindowOut
    documents: list[DocumentUsageOut]

    @classmethod
    def from_dto(cls, report: DocumentUsageReport) -> DocumentUsageResponse:
        return cls(
            window=WindowOut.from_dto(report.window),
            documents=[DocumentUsageOut.from_dto(item) for item in report.documents],
        )


class UsageTrendResponse(BaseModel):
    """Volume over time, the window's totals, and the busiest users."""

    model_config = _RESPONSE_CONFIG

    window: WindowOut
    granularity: TrendGranularity
    points: list[TrendPointOut]
    summary: QuerySummaryOut
    active_users: list[ActiveUserOut]

    @classmethod
    def from_dto(cls, report: UsageTrendReport) -> UsageTrendResponse:
        return cls(
            window=WindowOut.from_dto(report.window),
            granularity=report.granularity,
            points=[TrendPointOut.from_dto(point) for point in report.points],
            summary=QuerySummaryOut.from_dto(report.summary),
            active_users=[ActiveUserOut.from_dto(user) for user in report.active_users],
        )


def _as_uuid(value: str | None) -> uuid.UUID | None:
    """Parse a JSONB-sourced id, or None. Never raises — see `DocumentUsageOut.from_dto`."""
    if not value:
        return None
    try:
        return uuid.UUID(value)
    except ValueError:
        return None
