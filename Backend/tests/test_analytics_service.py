"""backend/tests/test_analytics_service.py

Role: Tests for `backend/services/analytics_service.py` — who may read a report, what the
service passes down to the aggregates, what reaches the audit trail, and what an export is.

The gate this file closes, from `WORKBREAKDOWN.md`: analytics is Administrator-only, keyword
trends cover both ends of the distribution, document usage is counted from citations, the
volume trend is bucketed, and a report can be exported as PDF or Excel with an audit row.

Why the repository is faked and the audit service is real
--------------------------------------------------------
`FakeAnalyticsRepository` records calls and returns queued rows; it is deliberately not a
Python reimplementation of the SQL (`test_analytics_repository.py` compiles the statements
instead). The claims here are all *policy*: the role check, the fixed stopword list, the
clamped limits, the validated window, the row cap, and what the audit detail may contain. None
of those are visible in the numbers an aggregate returns, and all of them are decided before
the repository is reached.

`AuditService` is the real one over `FakeAuditRepository`, because half of what is asserted
here is the shape of an audit row — the action, the outcome, and above all the *absence* of
user text in the detail. A faked audit service would let a row that violates
`secure-coding-vapt.md` pass unnoticed.

The export tests render real bytes
----------------------------------
`render_pdf` and `render_excel` are not stubbed. An export that produced an unopenable file
would satisfy every mock-based assertion, and the magic-byte checks here are what catch it.
`test_export_utils.py` covers the escaping and truncation inside those renderers; this file
covers the service's part — which report, which rows, the filename, and the audit row.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from backend.core.exceptions import UnauthorizedAccess, ValidationError
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.query import Query, QueryStatus
from backend.models.user import User, UserRole
from backend.repositories.analytics_repository import (
    ActiveUser,
    DocumentUsage,
    KeywordCount,
    QuerySummary,
    TrendGranularity,
    TrendPoint,
)
from backend.services.analytics_service import (
    MAX_EXPORT_ROWS,
    MAX_LIMIT,
    MAX_TREND_POINTS,
    MIN_KEYWORD_LENGTH,
    STOPWORDS,
    AnalyticsService,
    ExportFormat,
    ExportReport,
)
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.services.search_history_service import SearchHistoryService
from backend.tests.fakes import (
    FakeAnalyticsRepository,
    FakeAuditRepository,
    FakeQueryRepository,
)
from backend.utils.ids import new_id, utc_now

CONTEXT = RequestContext(ip_address="203.0.113.10", user_agent="pytest")

FROM = datetime(2026, 1, 1, tzinfo=UTC)
TO = datetime(2026, 6, 30, tzinfo=UTC)

#: An account number inside user-supplied text. Used to prove it never reaches a log: a real
#: question mentioning an account produces a keyword containing it, and
#: `.claude/rules/secure-coding-vapt.md` forbids that in the audit trail.
SENSITIVE = "4111111111111111"


def _user(tenant_id: uuid.UUID, role: UserRole, email: str) -> User:
    return User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email=email,
        full_name=None,
        hashed_password="unused-in-these-tests",
        role=role,
        is_active=True,
        failed_login_attempts=0,
        locked_until=None,
    )


@pytest.fixture
def tenant_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def admin(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.ADMINISTRATOR, "admin@bank.example")


@pytest.fixture
def analyst(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.STANDARD_USER, "analyst@bank.example")


@pytest.fixture
def analytics_repo() -> FakeAnalyticsRepository:
    return FakeAnalyticsRepository()


@pytest.fixture
def audit_repo() -> FakeAuditRepository:
    return FakeAuditRepository()


@pytest.fixture
def queries() -> FakeQueryRepository:
    return FakeQueryRepository()


@pytest.fixture
def service(
    analytics_repo: FakeAnalyticsRepository,
    audit_repo: FakeAuditRepository,
    queries: FakeQueryRepository,
) -> AnalyticsService:
    """The real service over a recording repository, a real audit service, and the real
    history service.

    `SearchHistoryService` is real rather than faked because the history export's whole point
    is that it returns exactly the rows `GET /history` would — a faked history service would
    make that property untestable here, which is the one place it can be checked.
    """
    audit = AuditService(audit_repo)
    return AnalyticsService(analytics_repo, audit, SearchHistoryService(queries, audit))


async def _ask(
    queries: FakeQueryRepository,
    actor: User,
    question: str,
    *,
    minutes_ago: int = 0,
    answer: str | None = "Wire transfers cost 25 USD.",
) -> Query:
    """One answered history row with a citation, for the history export."""
    row = await queries.create(
        query_id=new_id(),
        tenant_id=actor.tenant_id,
        user_id=actor.id,
        question_text=question,
    )
    row.status = QueryStatus.ANSWERED
    row.answer_text = answer
    row.confidence_score = 0.87
    row.latency_ms = 1_400
    row.source_references = [
        {"source_document_name": "Fee schedule 2026.pdf", "section_or_page": "page 4"}
    ]
    row.created_at = utc_now() - timedelta(minutes=minutes_ago)
    return row


def _seed(repo: FakeAnalyticsRepository) -> None:
    """Rows for every panel, including one containing an account number.

    The sensitive value goes in the *keyword* and the *document name* deliberately — those are
    the two places user- and file-supplied text enters an analytics report, so a leak into a
    log would show up in `TestTheAuditTrailCarriesNoUserText`.
    """
    repo.keywords[False] = [
        KeywordCount(keyword="wire", occurrences=40, queries=31),
        KeywordCount(keyword=SENSITIVE, occurrences=12, queries=12),
    ]
    repo.keywords[True] = [KeywordCount(keyword="escheatment", occurrences=1, queries=1)]
    repo.documents = [
        DocumentUsage(
            document_id=str(uuid.uuid4()),
            document_name=f"Statement {SENSITIVE}.pdf",
            citations=9,
            queries=7,
            last_cited_at=TO,
        )
    ]
    repo.trend = [
        TrendPoint(period_start=FROM, total=10, answered=8, not_found=1, failed=1),
        TrendPoint(period_start=TO, total=4, answered=4, not_found=0, failed=0),
    ]
    repo.users = [
        ActiveUser(
            user_id=uuid.uuid4(),
            email="analyst@bank.example",
            queries=22,
            last_query_at=TO,
        )
    ]
    repo.query_summary = QuerySummary(
        total=14,
        answered=12,
        not_found=1,
        failed=1,
        pending=0,
        avg_latency_ms=1_320.5,
        p95_latency_ms=2_800.0,
        avg_confidence=0.82,
    )


class TestOnlyAdministratorsReachAnalytics:
    """`rbac-enforcement.md` puts analytics on the Administrator list.

    Checked here as well as in the route dependency, and this is the test that would fail if
    the service's own check were removed as redundant. It is not redundant: analytics called
    from a scheduled job or a CLI never passes through `require_admin`.
    """

    @pytest.mark.parametrize(
        "method", ["keyword_trends", "document_usage", "usage_trends"]
    )
    async def test_a_standard_user_is_refused(
        self, service: AnalyticsService, analyst: User, method: str
    ) -> None:
        with pytest.raises(UnauthorizedAccess):
            await getattr(service, method)(actor=analyst, context=CONTEXT)

    async def test_a_standard_user_cannot_export(
        self, service: AnalyticsService, analyst: User
    ) -> None:
        with pytest.raises(UnauthorizedAccess):
            await service.export(
                actor=analyst,
                context=CONTEXT,
                report=ExportReport.HISTORY,
                export_format=ExportFormat.EXCEL,
            )

    async def test_a_refusal_reads_nothing(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        analyst: User,
    ) -> None:
        """The denial comes before the aggregate, so no tenant data is read at all.

        Refusing after the query would leave the numbers in the process's memory and the load
        on the database — a check that only suppresses the response is not an access control.
        """
        with pytest.raises(UnauthorizedAccess):
            await service.keyword_trends(actor=analyst, context=CONTEXT)
        assert analytics_repo.calls == []

    async def test_the_denial_is_audited_with_the_operation(
        self,
        service: AnalyticsService,
        audit_repo: FakeAuditRepository,
        analyst: User,
    ) -> None:
        """`ACCESS_DENIED`, `DENIED`, and a fixed operation name — never request text."""
        with pytest.raises(UnauthorizedAccess):
            await service.usage_trends(actor=analyst, context=CONTEXT)

        entry = audit_repo.entries[-1]
        assert entry.action is AuditAction.ACCESS_DENIED
        assert entry.outcome is AuditOutcome.DENIED
        assert entry.object_type == AuditObjectType.ANALYTICS
        assert entry.detail["reason"] == "administrator_required"
        assert entry.detail["operation"] == "usage_trends"
        assert entry.detail["role"] == UserRole.STANDARD_USER.value

    async def test_an_export_denial_names_the_report(
        self,
        service: AnalyticsService,
        audit_repo: FakeAuditRepository,
        analyst: User,
    ) -> None:
        """The operation string distinguishes a refused read from a refused download.

        A reviewer asking "did anyone try to take data out of the system" needs those apart:
        an attempted export is a different event from an attempted glance at a dashboard.
        """
        with pytest.raises(UnauthorizedAccess):
            await service.export(
                actor=analyst,
                context=CONTEXT,
                report=ExportReport.KEYWORDS,
                export_format=ExportFormat.PDF,
            )
        assert audit_repo.entries[-1].detail["operation"] == "export:keywords"


class TestTheTenantComesFromTheActor:
    """Role and tenant are independent checks, and both apply on every read."""

    async def test_every_read_uses_the_actors_tenant(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        _seed(analytics_repo)
        await service.keyword_trends(actor=admin, context=CONTEXT)
        await service.document_usage(actor=admin, context=CONTEXT)
        await service.usage_trends(actor=admin, context=CONTEXT)
        assert analytics_repo.tenants_seen() == {admin.tenant_id}

    async def test_two_administrators_of_different_tenants_do_not_mix(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """There is no parameter that could name another organization.

        Asserted by driving the same service with a second Administrator: the only thing that
        changes the scope is who the actor is.
        """
        other = _user(uuid.uuid4(), UserRole.ADMINISTRATOR, "admin@other.example")
        await service.document_usage(actor=admin, context=CONTEXT)
        await service.document_usage(actor=other, context=CONTEXT)
        assert analytics_repo.tenants_seen() == {admin.tenant_id, other.tenant_id}
        assert admin.tenant_id != other.tenant_id


class TestKeywordPolicyIsFixed:
    async def test_the_stopword_list_and_min_length_are_the_modules(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """The policy travels down; no caller can supply either.

        `sorted(STOPWORDS)` rather than the frozenset, because the value becomes a bound
        parameter list and a set's iteration order would make the statement's parameters differ
        between processes for no reason.
        """
        await service.keyword_trends(actor=admin, context=CONTEXT)
        for kwargs in analytics_repo.kwargs_for("keyword_counts"):
            assert kwargs["stopwords"] == sorted(STOPWORDS)
            assert kwargs["min_length"] == MIN_KEYWORD_LENGTH

    async def test_no_domain_term_is_filtered(self) -> None:
        """The list is function words only.

        A keyword report exists to surface what customers ask about, so filtering "overdraft"
        would remove the finding rather than the noise. This asserts the intent of the list,
        which is the kind of thing that erodes one well-meant addition at a time.
        """
        for term in ("account", "wire", "overdraft", "fee", "loan", "kyc", "escheatment"):
            assert term not in STOPWORDS

    async def test_both_directions_are_read_from_one_aggregate(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """Most- and least-searched are the same read in two orders (BR-06)."""
        _seed(analytics_repo)
        report = await service.keyword_trends(actor=admin, context=CONTEXT)

        directions = [k["ascending"] for k in analytics_repo.kwargs_for("keyword_counts")]
        assert directions == [False, True]
        assert [k.keyword for k in report.most_searched] == ["wire", SENSITIVE]
        assert [k.keyword for k in report.least_searched] == ["escheatment"]

    async def test_the_totals_travel_with_the_keywords(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """"wire, 40 times" means one thing over 14 queries and another over 14,000."""
        _seed(analytics_repo)
        report = await service.keyword_trends(actor=admin, context=CONTEXT)
        assert report.summary.total == 14
        assert report.summary.p95_latency_ms == 2_800.0


class TestLimitsAreClamped:
    """Clamped rather than rejected — the same rule as history pagination.

    The bound is a response-size control as much as a UX one: an unbounded `limit` on a grouped
    query over a year of history is a cheap way for an authenticated Administrator to make the
    server work hard. The schema also bounds it, which turns an out-of-range request into a 422
    at the edge; this clamp is what protects the non-HTTP callers.
    """

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [(1, 1), (10, 10), (MAX_LIMIT, MAX_LIMIT), (MAX_LIMIT + 500, MAX_LIMIT)],
    )
    async def test_the_row_limit_is_bounded(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
        requested: int,
        expected: int,
    ) -> None:
        await service.document_usage(actor=admin, context=CONTEXT, limit=requested)
        assert analytics_repo.kwargs_for("document_usage")[0]["limit"] == expected

    @pytest.mark.parametrize("requested", [0, -1, -10_000])
    async def test_a_non_positive_limit_becomes_one(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
        requested: int,
    ) -> None:
        """`LIMIT 0` would render an empty panel and `LIMIT -1` is a PostgreSQL error.

        Neither is what a caller asking for "no rows, please" meant, so the floor is one row.
        """
        await service.document_usage(actor=admin, context=CONTEXT, limit=requested)
        assert analytics_repo.kwargs_for("document_usage")[0]["limit"] == 1

    async def test_trend_points_have_their_own_ceiling(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """366 points, because a year of daily buckets is the widest readable chart."""
        await service.usage_trends(
            actor=admin, context=CONTEXT, points=10_000, user_limit=10_000
        )
        assert analytics_repo.kwargs_for("volume_trend")[0]["limit"] == MAX_TREND_POINTS
        # The user panel keeps the row limit, not the trend limit.
        assert analytics_repo.kwargs_for("active_users")[0]["limit"] == MAX_LIMIT


class TestTheWindow:
    async def test_both_bounds_reach_every_aggregate(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        await service.usage_trends(
            actor=admin, context=CONTEXT, created_from=FROM, created_to=TO
        )
        assert len(analytics_repo.calls) == 3
        for _, _, kwargs in analytics_repo.calls:
            assert kwargs["created_from"] == FROM
            assert kwargs["created_to"] == TO

    async def test_an_inverted_window_is_rejected_not_swapped(
        self, service: AnalyticsService, admin: User
    ) -> None:
        """Swapping the ends would answer a different question from the one asked.

        A report that quietly reinterprets its own filter produces a number nobody can check
        against the request that produced it, which is the opposite of what an audited report
        is for.
        """
        with pytest.raises(ValidationError):
            await service.keyword_trends(
                actor=admin, context=CONTEXT, created_from=TO, created_to=FROM
            )

    async def test_an_inverted_window_is_refused_before_the_read(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        with pytest.raises(ValidationError):
            await service.document_usage(
                actor=admin, context=CONTEXT, created_from=TO, created_to=FROM
            )
        assert analytics_repo.calls == []

    async def test_an_unbounded_window_is_labelled_honestly(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """"inception to now" — the truthful description of a report with no filter.

        The label lands in the export subtitle and the audit detail, so it has to be readable a
        year later. Printing today's date for an unbounded window would misdate the evidence.
        """
        report = await service.document_usage(actor=admin, context=CONTEXT)
        assert report.window.label == "inception to now"

    async def test_a_bounded_window_labels_both_dates(
        self, service: AnalyticsService, admin: User
    ) -> None:
        report = await service.document_usage(
            actor=admin, context=CONTEXT, created_from=FROM, created_to=TO
        )
        assert report.window.label == "2026-01-01 to 2026-06-30"


class TestEveryReadIsAudited:
    """One `ANALYTICS_VIEW` row per read (BR-05, `audit-log-immutability.md`)."""

    @pytest.mark.parametrize(
        ("method", "report"),
        [
            ("keyword_trends", "keywords"),
            ("document_usage", "documents"),
            ("usage_trends", "trends"),
        ],
    )
    async def test_the_row_names_the_report_and_the_window(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
        method: str,
        report: str,
    ) -> None:
        _seed(analytics_repo)
        await getattr(service, method)(
            actor=admin, context=CONTEXT, created_from=FROM, created_to=TO
        )

        entry = audit_repo.entries[-1]
        assert entry.action is AuditAction.ANALYTICS_VIEW
        assert entry.outcome is AuditOutcome.SUCCESS
        assert entry.object_type == AuditObjectType.ANALYTICS
        # A report is an aggregate with no single subject row, so the tenant is the object.
        assert entry.object_id == str(admin.tenant_id)
        assert entry.detail["report"] == report
        assert entry.detail["window"] == "2026-01-01 to 2026-06-30"

    async def test_the_row_records_who_asked(
        self,
        service: AnalyticsService,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        await service.document_usage(actor=admin, context=CONTEXT)
        entry = audit_repo.entries[-1]
        assert entry.user_id == admin.id
        assert entry.actor_email == admin.email
        assert entry.ip_address == CONTEXT.ip_address
        assert entry.tenant_id == admin.tenant_id

    async def test_the_trend_row_records_the_granularity(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        _seed(analytics_repo)
        await service.usage_trends(
            actor=admin, context=CONTEXT, granularity=TrendGranularity.WEEK
        )
        detail = audit_repo.entries[-1].detail
        assert detail["granularity"] == "week"
        assert detail["points"] == 2
        assert detail["users_returned"] == 1


class TestTheAuditTrailCarriesNoUserText:
    """The detail records shapes — counts, the window, the format — and never content.

    `secure-coding-vapt.md` forbids customer account numbers and PII in a log, and an extracted
    keyword is user text: a question mentioning an account produces a keyword containing it.
    Document names are file-supplied text with the same problem, and an active-user row is an
    email address. Each seeded row here carries a value that must not appear.
    """

    @staticmethod
    def _details(audit_repo: FakeAuditRepository) -> str:
        return " ".join(repr(entry.detail) for entry in audit_repo.entries)

    async def test_keywords_are_counted_not_logged(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        _seed(analytics_repo)
        await service.keyword_trends(actor=admin, context=CONTEXT)

        detail = audit_repo.entries[-1].detail
        assert detail["returned"] == 2
        assert detail["queries_in_window"] == 14
        assert SENSITIVE not in self._details(audit_repo)
        assert "wire" not in self._details(audit_repo)

    async def test_document_names_are_counted_not_logged(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        _seed(analytics_repo)
        await service.document_usage(actor=admin, context=CONTEXT)
        assert audit_repo.entries[-1].detail["returned"] == 1
        assert SENSITIVE not in self._details(audit_repo)
        assert "Statement" not in self._details(audit_repo)

    async def test_active_user_emails_are_counted_not_logged(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        """The *count* of users is the fact the trail needs; the addresses are PII.

        `actor_email` is a separate field and is recorded — the trail must say who ran the
        report. That is the subject of the action, not a list of colleagues it disclosed.
        """
        _seed(analytics_repo)
        await service.usage_trends(actor=admin, context=CONTEXT)
        assert audit_repo.entries[-1].detail["users_returned"] == 1
        assert "analyst@bank.example" not in self._details(audit_repo)


class TestExport:
    async def test_a_pdf_export_returns_a_real_pdf(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """Rendered, not mocked: the magic bytes are what catch an unopenable file."""
        _seed(analytics_repo)
        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.KEYWORDS,
            export_format=ExportFormat.PDF,
        )
        assert exported.content.startswith(b"%PDF-")
        assert exported.media_type == "application/pdf"
        assert exported.filename.endswith(".pdf")

    async def test_an_excel_export_returns_a_real_workbook(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        _seed(analytics_repo)
        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.TRENDS,
            export_format=ExportFormat.EXCEL,
        )
        # A .xlsx is a zip archive; `PK` is its local file header.
        assert exported.content.startswith(b"PK")
        assert exported.filename.endswith(".xlsx")
        assert "spreadsheetml" in exported.media_type

    @pytest.mark.parametrize("report", list(ExportReport))
    async def test_every_report_renders(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        queries: FakeQueryRepository,
        admin: User,
        report: ExportReport,
    ) -> None:
        """All four branches of `_build_report`, including the history one.

        Parametrized over the enum rather than listing the members, so adding a report without
        a rendering branch fails here instead of at a user's download.
        """
        _seed(analytics_repo)
        await _ask(queries, admin, "What is the wire fee?")
        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=report,
            export_format=ExportFormat.EXCEL,
        )
        assert exported.content.startswith(b"PK")
        assert exported.rows >= 1

    async def test_the_filename_contains_no_user_input(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        admin: User,
    ) -> None:
        """A fixed prefix, the enum value, and a UTC stamp — nothing else.

        It lands in `Content-Disposition`, where a caller-supplied name would be both header
        injection and path traversal. Asserted as a shape because that is the property: no
        component of this string comes from a request.
        """
        _seed(analytics_repo)
        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.DOCUMENTS,
            export_format=ExportFormat.PDF,
        )
        stem, _, extension = exported.filename.rpartition(".")
        assert extension == "pdf"
        prefix = "docquery-documents-"
        assert stem.startswith(prefix)
        # The remainder is `%Y%m%d-%H%M%S` and nothing else, parsed rather than pattern-matched
        # so a stamp that is the right shape but not a real timestamp still fails.
        datetime.strptime(stem.removeprefix(prefix), "%Y%m%d-%H%M%S")  # noqa: DTZ007
        for character in ('"', "\\", "/", "\r", "\n", ".."):
            assert character not in exported.filename

    async def test_an_export_writes_a_data_export_row(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        """`DATA_EXPORT`, not `ANALYTICS_VIEW`: a file left the system and can be forwarded."""
        _seed(analytics_repo)
        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.KEYWORDS,
            export_format=ExportFormat.PDF,
            created_from=FROM,
            created_to=TO,
        )

        entry = audit_repo.entries[-1]
        assert entry.action is AuditAction.DATA_EXPORT
        assert entry.outcome is AuditOutcome.SUCCESS
        assert entry.detail["report"] == "keywords"
        assert entry.detail["format"] == "pdf"
        assert entry.detail["window"] == "2026-01-01 to 2026-06-30"
        assert entry.detail["rows"] == exported.rows
        assert entry.detail["bytes"] == len(exported.content)
        assert entry.detail["filename"] == exported.filename

    async def test_an_export_is_two_audit_rows(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        """A read happened and then a file was produced — both are facts worth recording.

        It follows from the export reusing the audited read methods rather than querying the
        repository again, which is also what keeps an exported figure and the same figure on
        screen from disagreeing.
        """
        _seed(analytics_repo)
        await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.DOCUMENTS,
            export_format=ExportFormat.EXCEL,
        )
        actions = [entry.action for entry in audit_repo.entries]
        assert actions == [AuditAction.ANALYTICS_VIEW, AuditAction.DATA_EXPORT]

    async def test_the_export_detail_carries_no_report_content(
        self,
        service: AnalyticsService,
        analytics_repo: FakeAnalyticsRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        _seed(analytics_repo)
        await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.DOCUMENTS,
            export_format=ExportFormat.EXCEL,
        )
        details = " ".join(repr(entry.detail) for entry in audit_repo.entries)
        assert SENSITIVE not in details
        assert "Statement" not in details


class TestTheHistoryExport:
    async def test_it_reads_through_the_authorized_history_service(
        self,
        service: AnalyticsService,
        queries: FakeQueryRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
        analyst: User,
    ) -> None:
        """The rows are exactly what `GET /history` would return for this caller.

        An Administrator's scope is the tenant, so a colleague's question belongs in the file —
        and the `HISTORY_VIEW` row proves the export went through the read that decides that,
        rather than a second query with its own idea of who may see what.
        """
        await _ask(queries, analyst, "What is the overdraft rate?")
        await _ask(queries, admin, "What is the wire fee?")

        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.HISTORY,
            export_format=ExportFormat.EXCEL,
        )
        assert exported.rows == 2
        actions = [entry.action for entry in audit_repo.entries]
        assert AuditAction.HISTORY_VIEW in actions
        assert actions[-1] is AuditAction.DATA_EXPORT

    async def test_an_empty_history_still_produces_a_file(
        self, service: AnalyticsService, admin: User
    ) -> None:
        """Zero rows is an answer, and a reviewer asked for it in writing.

        Returning an error for an empty period would leave "nobody searched in March"
        undocumented, which is exactly the sort of negative finding an audit asks to see.
        """
        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.HISTORY,
            export_format=ExportFormat.EXCEL,
        )
        assert exported.rows == 0
        assert exported.truncated is False
        assert exported.content.startswith(b"PK")

    async def test_the_row_cap_truncates_and_says_so(
        self,
        service: AnalyticsService,
        queries: FakeQueryRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        """`MAX_EXPORT_ROWS` stops the walk, and the fact is recorded in three places.

        A truncated export that looks complete is worse evidence than no export: someone would
        cite it as the whole period. The flag on the DTO becomes a response header, the audit
        detail records it, and the report's own table title says it on the page.
        """
        for index in range(MAX_EXPORT_ROWS + 5):
            await _ask(queries, admin, f"Question {index}", minutes_ago=index)

        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.HISTORY,
            export_format=ExportFormat.EXCEL,
        )
        assert exported.truncated is True
        assert exported.rows == MAX_EXPORT_ROWS
        assert audit_repo.entries[-1].detail["truncated"] is True

    async def test_a_full_but_uncapped_history_is_not_flagged(
        self,
        service: AnalyticsService,
        queries: FakeQueryRepository,
        admin: User,
    ) -> None:
        """Exactly `MAX_EXPORT_ROWS` rows is complete, not truncated.

        The off-by-one that matters: flagging a full-but-complete export would make the warning
        meaningless, and the walk stops on `offset >= total` before the cap is reached.
        """
        for index in range(MAX_EXPORT_ROWS):
            await _ask(queries, admin, f"Question {index}", minutes_ago=index)

        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.HISTORY,
            export_format=ExportFormat.EXCEL,
        )
        assert exported.rows == MAX_EXPORT_ROWS
        assert exported.truncated is False

    async def test_the_window_reaches_the_history_read(
        self,
        service: AnalyticsService,
        queries: FakeQueryRepository,
        admin: User,
    ) -> None:
        """An export of "March" must not quietly contain April.

        The rows are filtered by the history service, so this asserts on the count: the older
        row falls outside the window and the recent one does not.
        """
        await _ask(queries, admin, "Recent question", minutes_ago=1)
        await _ask(queries, admin, "Ancient question", minutes_ago=60 * 24 * 400)

        exported = await service.export(
            actor=admin,
            context=CONTEXT,
            report=ExportReport.HISTORY,
            export_format=ExportFormat.EXCEL,
            created_from=utc_now() - timedelta(days=7),
        )
        assert exported.rows == 1
