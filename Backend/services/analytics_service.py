"""backend/services/analytics_service.py

Role: The analytics business rules — who may read a report, what window it covers, which words
count as keywords, and how a report becomes a downloadable file. Owns no SQL and no rendering:
`analytics_repository` runs the aggregates, `backend/utils/export_utils.py` writes the bytes.

Administrator-only, and checked here as well as in the route
------------------------------------------------------------
`rbac-enforcement.md` puts analytics on the Administrator list, and `require_admin` on the
router already enforces it. This service checks again anyway — not from distrust of the
dependency, but because the check and the *audit row* belong together. A denial recorded only
by the route dependency is a denial that disappears the moment analytics is called from
anywhere else: a scheduled report, an internal tool, a future export job. Two cheap checks, one
of which survives refactoring.

Tenant comes from the actor, never from a parameter — same reasoning as
`search_history_service`. An Administrator is an administrator *of one tenant*.

Keywords: a fixed tokenizer and a stopword list
-----------------------------------------------
`STOPWORDS` and `MIN_KEYWORD_LENGTH` live here because they are policy, and the tokenizer lives
in the repository because it is SQL. Neither is caller-supplied: `secure-coding-vapt.md`
forbids a user-supplied regex outright, and there is no parameter on any method here that could
become one. The list is deliberately function words only — "account", "wire" and "overdraft"
are exactly what a search-trends report exists to surface, so nothing domain-specific is
filtered.

What never reaches the audit trail
----------------------------------
Every read writes an `ANALYTICS_VIEW` row and every export a `DATA_EXPORT` row (BR-05,
`audit-log-immutability.md`). The detail records *shapes* — row counts, the window, the
granularity, the format — and never a keyword, a question or a document name. Extracted
keywords are user text: a question mentioning an account number produces a keyword containing
it, and `secure-coding-vapt.md` forbids that in a log. The same reason `search_history_service`
records `keyword_used` as a boolean.

Exports are assembled here, not in the route
--------------------------------------------
The history report needs `search_history_service`, and `architecture.md` allows a service to
call a service while forbidding a route to orchestrate two. It also means the history rows in
an export come back through exactly the same authorized, audited read the history *page* uses —
there is no second query with its own idea of who may see what.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from backend.core.exceptions import UnauthorizedAccess, ValidationError
from backend.core.logging import get_logger
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.query import Query
from backend.models.user import User, UserRole
from backend.repositories.analytics_repository import (
    ActiveUser,
    AnalyticsRepository,
    DocumentUsage,
    KeywordCount,
    QuerySummary,
    TrendGranularity,
    TrendPoint,
)
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.services.search_history_service import MAX_PAGE_SIZE, SearchHistoryService
from backend.utils.export_utils import (
    ReportDocument,
    ReportTable,
    render_excel,
    render_pdf,
)

logger = get_logger(__name__)

#: Shortest token counted as a keyword. Three drops "is", "of", "to" and every other
#: two-letter function word without needing them in the list below.
MIN_KEYWORD_LENGTH: Final = 3

#: Rows per panel. The maximum is a response-size bound as much as a UX one, for the same
#: reason `search_history_service` caps its page: an unbounded `limit` on a grouped query over
#: a year of history is a cheap way for an authenticated Administrator to make the server work
#: hard.
DEFAULT_LIMIT: Final = 10
MAX_LIMIT: Final = 50

#: Trend buckets returned. 366 covers a year of daily points — beyond that a chart is
#: unreadable and the caller wants a coarser `granularity`, not more points.
DEFAULT_TREND_POINTS: Final = 30
MAX_TREND_POINTS: Final = 366

#: Rows an export may contain. Reached by walking the ordinary history page, `MAX_PAGE_SIZE` at
#: a time — deliberately not by adding an unbounded query, which would be a second read path
#: with its own idea of authorization. When the cap truncates an export it is recorded in the
#: audit detail and printed on the report, because a truncated export that looks complete is
#: worse evidence than no export.
MAX_EXPORT_ROWS: Final = 1_000

#: Function words only. A keyword report is meant to surface what customers ask *about*, so no
#: domain term is filtered — "overdraft" being the top keyword is the finding, not noise.
#: The repository lower-cases before splitting, so every entry here is lower-case.
STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "about", "after", "again", "all", "also", "and", "any", "are", "aren't", "back",
        "because", "been", "before", "being", "both", "but", "can", "cannot", "can't",
        "come", "could", "couldn't", "did", "didn't", "does", "doesn't", "doing", "don't",
        "down", "each", "even", "ever", "every", "for", "from", "get", "gets", "give",
        "goes", "going", "gone", "got", "had", "hadn't", "has", "hasn't", "have",
        "haven't", "having", "her", "here", "hers", "him", "his", "how", "i'd", "i'll",
        "i'm", "into", "isn't", "it's", "its", "just", "know", "less", "let", "like",
        "make", "many", "may", "me", "mean", "means", "might", "mine", "more", "most",
        "much", "must", "need", "needed", "needs", "not", "now", "off", "one", "only",
        "onto", "other", "our", "ours", "out", "over", "own", "per", "please", "put",
        "same", "say", "says", "see", "shall", "she", "should", "shouldn't", "since",
        "some", "such", "take", "tell", "than", "that", "that's", "the", "their",
        "theirs", "them", "then", "there", "there's", "these", "they", "this", "those",
        "through", "too", "under", "until", "upon", "use", "used", "uses", "very", "want",
        "was", "wasn't", "way", "we", "well", "were", "weren't", "what", "what's", "when",
        "where", "whether", "which", "while", "who", "whom", "why", "will", "with",
        "within", "without", "won't", "would", "wouldn't", "you", "your", "yours",
    }
)  # fmt: skip


class ExportReport(StrEnum):
    """Which report an export renders. A closed set, not a table or column name.

    The value never reaches SQL — each member selects a Python branch in `_build_report`. That
    is the point: a "which report" parameter that became part of a query would be the classic
    injection surface, and there is nothing here for a caller to widen.
    """

    KEYWORDS = "keywords"
    DOCUMENTS = "documents"
    TRENDS = "trends"
    HISTORY = "history"


class ExportFormat(StrEnum):
    PDF = "pdf"
    EXCEL = "excel"


_MEDIA_TYPES: Final[dict[ExportFormat, str]] = {
    ExportFormat.PDF: "application/pdf",
    ExportFormat.EXCEL: (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ),
}
_EXTENSIONS: Final[dict[ExportFormat, str]] = {
    ExportFormat.PDF: "pdf",
    ExportFormat.EXCEL: "xlsx",
}


@dataclass(frozen=True, slots=True)
class ReportWindow:
    """The date range a report covers, and how it is described on the page.

    Both ends optional: `None` means unbounded, which is the honest rendering of "everything so
    far" on a tenant's first month. `label` is what goes in the export subtitle, so an exported
    file can still be interpreted a year later.
    """

    created_from: datetime | None
    created_to: datetime | None

    @property
    def label(self) -> str:
        start = self.created_from.date().isoformat() if self.created_from else "inception"
        end = self.created_to.date().isoformat() if self.created_to else "now"
        return f"{start} to {end}"


@dataclass(frozen=True, slots=True)
class KeywordTrends:
    """What was searched most, and least, in the window.

    Both directions in one DTO because the dashboard shows them together and they come from one
    aggregate read in two orders (BR-06). `least` answers a different question from `most`: it
    is where a document nobody knew to ask about shows up.
    """

    window: ReportWindow
    most_searched: tuple[KeywordCount, ...]
    least_searched: tuple[KeywordCount, ...]
    summary: QuerySummary


@dataclass(frozen=True, slots=True)
class DocumentUsageReport:
    """Which documents answers actually came from."""

    window: ReportWindow
    documents: tuple[DocumentUsage, ...]


@dataclass(frozen=True, slots=True)
class UsageTrendReport:
    """Volume over time, plus the totals and the busiest users."""

    window: ReportWindow
    granularity: TrendGranularity
    points: tuple[TrendPoint, ...]
    summary: QuerySummary
    active_users: tuple[ActiveUser, ...]


@dataclass(frozen=True, slots=True)
class ExportedFile:
    """A rendered report: the bytes, a filename, and the media type to serve it as.

    The filename is composed here from a fixed prefix, the enum value and a UTC timestamp —
    never from user input. It ends up in a `Content-Disposition` header, where a
    caller-supplied name would be both a header-injection vector and a path-traversal one.
    """

    filename: str
    media_type: str
    content: bytes
    rows: int
    truncated: bool


class AnalyticsService:
    def __init__(
        self,
        analytics: AnalyticsRepository,
        audit: AuditService,
        history: SearchHistoryService,
    ) -> None:
        self._analytics = analytics
        self._audit = audit
        self._history = history

    # -- Reads --------------------------------------------------------------
    async def keyword_trends(
        self,
        *,
        actor: User,
        context: RequestContext,
        limit: int = DEFAULT_LIMIT,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> KeywordTrends:
        """Most- and least-searched keywords, with the window's totals for context.

        The totals travel with the keywords deliberately: "wire" appearing 40 times means one
        thing over 60 queries and another over 6,000, and a panel showing only the ranking
        invites the wrong reading.
        """
        await self._authorize(actor, context, "keyword_trends")
        window = self._window(created_from, created_to)
        bounded = self._bounded(limit, MAX_LIMIT)

        most = await self._analytics.keyword_counts(
            actor.tenant_id,
            stopwords=sorted(STOPWORDS),
            min_length=MIN_KEYWORD_LENGTH,
            limit=bounded,
            created_from=window.created_from,
            created_to=window.created_to,
        )
        least = await self._analytics.keyword_counts(
            actor.tenant_id,
            stopwords=sorted(STOPWORDS),
            min_length=MIN_KEYWORD_LENGTH,
            limit=bounded,
            ascending=True,
            created_from=window.created_from,
            created_to=window.created_to,
        )
        summary = await self._analytics.summary(
            actor.tenant_id,
            created_from=window.created_from,
            created_to=window.created_to,
        )

        await self._record_view(
            actor,
            context,
            report="keywords",
            window=window,
            # Counts, never the keywords themselves — they are user text.
            detail={"returned": len(most), "queries_in_window": summary.total},
        )
        return KeywordTrends(
            window=window,
            most_searched=tuple(most),
            least_searched=tuple(least),
            summary=summary,
        )

    async def document_usage(
        self,
        *,
        actor: User,
        context: RequestContext,
        limit: int = DEFAULT_LIMIT,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> DocumentUsageReport:
        """Which documents were cited, most first."""
        await self._authorize(actor, context, "document_usage")
        window = self._window(created_from, created_to)
        documents = await self._analytics.document_usage(
            actor.tenant_id,
            limit=self._bounded(limit, MAX_LIMIT),
            created_from=window.created_from,
            created_to=window.created_to,
        )
        await self._record_view(
            actor,
            context,
            report="documents",
            window=window,
            # Document names are uploaded-file text; the count is enough for the trail.
            detail={"returned": len(documents)},
        )
        return DocumentUsageReport(window=window, documents=tuple(documents))

    async def usage_trends(
        self,
        *,
        actor: User,
        context: RequestContext,
        granularity: TrendGranularity = TrendGranularity.DAY,
        points: int = DEFAULT_TREND_POINTS,
        user_limit: int = DEFAULT_LIMIT,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> UsageTrendReport:
        """Query volume per period, the window's totals, and the most active users."""
        await self._authorize(actor, context, "usage_trends")
        window = self._window(created_from, created_to)
        trend = await self._analytics.volume_trend(
            actor.tenant_id,
            granularity=granularity,
            limit=self._bounded(points, MAX_TREND_POINTS),
            created_from=window.created_from,
            created_to=window.created_to,
        )
        summary = await self._analytics.summary(
            actor.tenant_id,
            created_from=window.created_from,
            created_to=window.created_to,
        )
        users = await self._analytics.active_users(
            actor.tenant_id,
            limit=self._bounded(user_limit, MAX_LIMIT),
            created_from=window.created_from,
            created_to=window.created_to,
        )
        await self._record_view(
            actor,
            context,
            report="trends",
            window=window,
            detail={
                "granularity": granularity.value,
                "points": len(trend),
                "queries_in_window": summary.total,
                # Emails are PII and the *count* is what the trail needs.
                "users_returned": len(users),
            },
        )
        return UsageTrendReport(
            window=window,
            granularity=granularity,
            points=tuple(trend),
            summary=summary,
            active_users=tuple(users),
        )

    # -- Export -------------------------------------------------------------
    async def export(
        self,
        *,
        actor: User,
        context: RequestContext,
        report: ExportReport,
        export_format: ExportFormat,
        granularity: TrendGranularity = TrendGranularity.DAY,
        limit: int = MAX_LIMIT,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> ExportedFile:
        """Render one report as a PDF or Excel file.

        Audited as `DATA_EXPORT` rather than `ANALYTICS_VIEW`: a file leaves the system and can
        be forwarded, and `audit-log-immutability.md` names data export as a critical activity
        in its own right. The row records the format, the report, the window and the row count
        — the facts a reviewer needs to match a file someone produces against a request that
        was made.
        """
        await self._authorize(actor, context, f"export:{report.value}")
        window = self._window(created_from, created_to)
        bounded = self._bounded(limit, MAX_LIMIT)

        document, rows, truncated = await self._build_report(
            actor=actor,
            context=context,
            report=report,
            window=window,
            granularity=granularity,
            limit=bounded,
        )
        content = (
            render_pdf(document)
            if export_format is ExportFormat.PDF
            else render_excel(document)
        )
        # A fixed prefix, the enum value, and a timestamp. No user input reaches this string.
        stamp = _now().strftime("%Y%m%d-%H%M%S")
        filename = f"docquery-{report.value}-{stamp}.{_EXTENSIONS[export_format]}"

        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=AuditAction.DATA_EXPORT,
            object_type=AuditObjectType.ANALYTICS,
            object_id=str(actor.tenant_id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={
                "report": report.value,
                "format": export_format.value,
                "window": window.label,
                "rows": rows,
                # Explicit, because "1,000 rows" and "the first 1,000 of 40,000 rows" are
                # different pieces of evidence.
                "truncated": truncated,
                "bytes": len(content),
                "filename": filename,
            },
        )
        logger.info(
            "analytics.exported",
            report=report.value,
            export_format=export_format.value,
            rows=rows,
            truncated=truncated,
        )
        return ExportedFile(
            filename=filename,
            media_type=_MEDIA_TYPES[export_format],
            content=content,
            rows=rows,
            truncated=truncated,
        )

    async def _build_report(
        self,
        *,
        actor: User,
        context: RequestContext,
        report: ExportReport,
        window: ReportWindow,
        granularity: TrendGranularity,
        limit: int,
    ) -> tuple[ReportDocument, int, bool]:
        """Assemble the tables for one report. Returns the document, its row count, and whether
        the row cap truncated it.

        Each branch reuses the read methods above rather than calling the repository directly,
        so an exported figure and the same figure on screen cannot disagree — and so the export
        inherits their `ANALYTICS_VIEW` audit row as well as its own `DATA_EXPORT` one. Two
        rows for one export is correct: a report was read, and then a file was produced.
        """
        subtitle = (
            f"Tenant {actor.tenant_id} | {window.label} | requested by {actor.email}"
        )
        truncated = False

        if report is ExportReport.KEYWORDS:
            trends = await self.keyword_trends(
                actor=actor,
                context=context,
                limit=limit,
                created_from=window.created_from,
                created_to=window.created_to,
            )
            tables = (
                _keyword_table("Most searched keywords", trends.most_searched),
                _keyword_table("Least searched keywords", trends.least_searched),
                _summary_table(trends.summary),
            )
            rows = len(trends.most_searched) + len(trends.least_searched)
            title = "DocQuery search keyword report"

        elif report is ExportReport.DOCUMENTS:
            usage = await self.document_usage(
                actor=actor,
                context=context,
                limit=limit,
                created_from=window.created_from,
                created_to=window.created_to,
            )
            tables = (_document_table(usage.documents),)
            rows = len(usage.documents)
            title = "DocQuery document usage report"

        elif report is ExportReport.TRENDS:
            usage_trend = await self.usage_trends(
                actor=actor,
                context=context,
                granularity=granularity,
                points=MAX_TREND_POINTS,
                user_limit=limit,
                created_from=window.created_from,
                created_to=window.created_to,
            )
            tables = (
                _trend_table(usage_trend.points, usage_trend.granularity),
                _summary_table(usage_trend.summary),
                _active_user_table(usage_trend.active_users),
            )
            rows = len(usage_trend.points)
            title = "DocQuery usage trend report"

        else:
            history_rows, truncated = await self._collect_history(
                actor=actor,
                context=context,
                created_from=window.created_from,
                created_to=window.created_to,
            )
            tables = (_history_table(history_rows, truncated),)
            rows = len(history_rows)
            title = "DocQuery search history report"

        return (
            ReportDocument(
                title=title,
                generated_at=_now(),
                subtitle=subtitle,
                tables=tables,
            ),
            rows,
            truncated,
        )

    async def _collect_history(
        self,
        *,
        actor: User,
        context: RequestContext,
        created_from: datetime | None,
        created_to: datetime | None,
    ) -> tuple[list[Query], bool]:
        """Walk history pages up to `MAX_EXPORT_ROWS`, through the authorized read.

        Paging rather than one large query, because `search_history_service` is where the
        "whose history may I see" decision lives and `MAX_PAGE_SIZE` is its bound. Adding an
        unbounded export query would duplicate that decision in a second place — and a
        divergence between the two would be a Standard User's rows appearing in an export.

        The trade-off is one round trip per hundred rows and one `HISTORY_VIEW` audit row per
        page. Both are acceptable: an export is a rare, deliberate act, and an audit trail
        showing exactly how much history was read is the point rather than a cost.
        """
        collected: list[Query] = []
        offset = 0
        while len(collected) < MAX_EXPORT_ROWS:
            page = await self._history.list_history(
                actor=actor,
                context=context,
                created_from=created_from,
                created_to=created_to,
                limit=MAX_PAGE_SIZE,
                offset=offset,
            )
            if not page.items:
                return collected, False
            collected.extend(page.items)
            offset += len(page.items)
            if offset >= page.total:
                return collected[:MAX_EXPORT_ROWS], False
        # The cap stopped the walk with rows still unread.
        logger.warning("analytics.export_truncated", cap=MAX_EXPORT_ROWS)
        return collected[:MAX_EXPORT_ROWS], True

    # -- Shared -------------------------------------------------------------
    async def _authorize(
        self, actor: User, context: RequestContext, operation: str
    ) -> None:
        """Administrator or nothing, and the denial is recorded.

        Logged as `ACCESS_DENIED` with the operation name — a fixed string from this module,
        never request text. `rbac-enforcement.md` requires denied attempts in the trail, and a
        denial with no record is indistinguishable from an attempt that never happened.
        """
        if actor.role is UserRole.ADMINISTRATOR:
            return
        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=AuditAction.ACCESS_DENIED,
            object_type=AuditObjectType.ANALYTICS,
            object_id=str(actor.tenant_id),
            outcome=AuditOutcome.DENIED,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={
                "reason": "administrator_required",
                "operation": operation,
                "role": actor.role.value,
            },
        )
        logger.warning("analytics.denied", operation=operation, role=actor.role.value)
        raise UnauthorizedAccess()

    async def _record_view(
        self,
        actor: User,
        context: RequestContext,
        *,
        report: str,
        window: ReportWindow,
        detail: dict[str, object],
    ) -> None:
        """One `ANALYTICS_VIEW` row per read (BR-05).

        `object_id` is the tenant: a report is an aggregate with no single subject row, and
        `search_history_service` records its collection reads the same way.
        """
        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=AuditAction.ANALYTICS_VIEW,
            object_type=AuditObjectType.ANALYTICS,
            object_id=str(actor.tenant_id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={"report": report, "window": window.label, **detail},
        )

    @staticmethod
    def _window(
        created_from: datetime | None, created_to: datetime | None
    ) -> ReportWindow:
        """Validate the range. An inverted window is rejected, not silently swapped.

        Swapping the ends would answer a different question from the one asked, and a report
        that quietly reinterprets its own filter is a number nobody can check.
        """
        if created_from and created_to and created_from > created_to:
            raise ValidationError("The start of the window must not be after its end.")
        return ReportWindow(created_from=created_from, created_to=created_to)

    @staticmethod
    def _bounded(value: int, maximum: int) -> int:
        """Clamp a limit rather than rejecting it — same rule as history pagination."""
        return min(max(1, int(value)), maximum)


# -- Table builders ---------------------------------------------------------
# Plain functions, not methods: each one is a pure mapping from a DTO tuple to a `ReportTable`,
# and none of them needs the repository, the audit trail or the actor. Keeping them out of the
# class is what makes them readable next to each other — the whole export contract is here.
def _keyword_table(title: str, keywords: tuple[KeywordCount, ...]) -> ReportTable:
    return ReportTable(
        title=title,
        columns=("Keyword", "Occurrences", "Distinct queries"),
        rows=tuple(
            (item.keyword, item.occurrences, item.queries) for item in keywords
        ),
    )


def _document_table(documents: tuple[DocumentUsage, ...]) -> ReportTable:
    return ReportTable(
        title="Most referenced documents",
        columns=("Document", "Citations", "Distinct queries", "Last cited"),
        rows=tuple(
            (item.document_name, item.citations, item.queries, item.last_cited_at)
            for item in documents
        ),
    )


def _trend_table(
    points: tuple[TrendPoint, ...], granularity: TrendGranularity
) -> ReportTable:
    return ReportTable(
        title=f"Query volume per {granularity.value}",
        columns=("Period start", "Total", "Answered", "Not found", "Failed"),
        rows=tuple(
            (point.period_start, point.total, point.answered, point.not_found, point.failed)
            for point in points
        ),
    )


def _active_user_table(users: tuple[ActiveUser, ...]) -> ReportTable:
    return ReportTable(
        title="Most active users",
        columns=("User", "Queries", "Last query"),
        rows=tuple((user.email, user.queries, user.last_query_at) for user in users),
    )


def _summary_table(summary: QuerySummary) -> ReportTable:
    """The window's totals as label/value pairs rather than one wide row.

    Two columns survive a narrow page and a printout; a nine-column single-row table does not.
    """
    return ReportTable(
        title="Summary for the period",
        columns=("Measure", "Value"),
        rows=(
            ("Queries submitted", summary.total),
            ("Answered from documents", summary.answered),
            ("No supporting document found", summary.not_found),
            ("Failed", summary.failed),
            ("Awaiting an answer", summary.pending),
            ("Average latency (ms)", summary.avg_latency_ms),
            ("95th percentile latency (ms)", summary.p95_latency_ms),
            ("Average confidence", summary.avg_confidence),
        ),
    )


def _history_table(rows: list[Query], truncated: bool) -> ReportTable:
    """Search history, one row per query, with its citations flattened into a cell.

    Takes the ORM rows the history service returns and maps each to plain values here. Not
    `QueryResponse`: that is an API DTO, and a service reaching into `backend/schemas/` to
    assemble a file would invert the layer order `architecture.md` sets out. The mapping is
    explicit either way, which is what keeps a column from appearing in an export by accident.

    The citation column is the reason a history export is worth producing:
    `rag-answer-grounding` and BR-03 make the source part of the answer, so an export of
    answers without their sources would be an export that loses the property being audited.

    Answer text is included in full. It is customer-visible content the Administrator asking
    for the export can already read on screen, the read is audited, and an export that silently
    dropped the answers would not be a history report. What is *not* here is anything the UI
    does not show: no internal ids beyond the query's own, no chunk text, no user id.
    """
    title = "Search history"
    if truncated:
        title = f"Search history (first {MAX_EXPORT_ROWS} rows — the period holds more)"
    return ReportTable(
        title=title,
        columns=(
            "Asked at",
            "Question",
            "Status",
            "Answer",
            "Sources",
            "Confidence",
            "Latency (ms)",
        ),
        rows=tuple(
            (
                row.created_at,
                row.question_text,
                row.status.value,
                row.answer_text,
                _citations(row),
                row.confidence_score,
                row.latency_ms,
            )
            for row in rows
        ),
    )


def _citations(row: Query) -> str:
    """`name (section)` per source, newline-separated. Empty for a not-found.

    Reads the stored JSONB defensively. A reference missing its location would have failed the
    schema on the way out of the query endpoint; here a placeholder is preferable to a raised
    exception, because one malformed row from an early migration must not make the whole export
    fail — and `unspecified` in a cell is visible, which is the point.
    """
    references: list[dict[str, Any]] = row.source_references or []
    return "\n".join(
        f"{reference.get('source_document_name', 'unknown document')} "
        f"({reference.get('section_or_page', 'unspecified')})"
        for reference in references
    )


def _now() -> datetime:
    """UTC, and the only clock this module reads."""
    return datetime.now(UTC)
