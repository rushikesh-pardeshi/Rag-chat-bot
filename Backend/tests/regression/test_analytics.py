"""backend/tests/regression/test_analytics.py

Role: BR-AN (Analytics & Reporting, AC-11.1) asserted over HTTP.

The requirement, quoted from `WORKBREAKDOWN.md`:

    Analytics & Reporting: Most- and least-searched keywords, document usage, query volume
    over time, most active users; exportable as PDF or Excel; Administrator only (BR-AN,
    AC-11.1)

What is asserted here, and what deliberately is not
---------------------------------------------------
`FakeAnalyticsRepository` **does not reimplement the aggregation SQL** — it records the
arguments each aggregate was called with and returns what the test queued. That is the right
fake for this layer (its own docstring explains why: a hand-written `regexp_split_to_table`
that tokenized slightly differently from PostgreSQL would hide a real disagreement), but it
means two different claims must not be confused:

  * **Asserted here.** The policy around the aggregate — that the tenant passed down is the
    caller's and cannot be named by the caller, that the stopword list and minimum keyword
    length are applied, that `limit` is bounded, that both ends of the keyword distribution are
    returned, that the window is echoed back, that every read is audited and every export is
    audited separately, and that the exported bytes really are a PDF or a real spreadsheet.
  * **Not asserted here.** That "wire" ranks above "fee" for a given corpus. The counting is
    `AnalyticsRepository`'s, and `backend/tests/test_analytics_repository.py` compiles those
    statements against the PostgreSQL dialect. A test in this module that queued
    `[KeywordCount("wire", 9, 7)]` and then asserted "wire is the top keyword" would be
    asserting its own fixture.

The p95 in this report is the production latency figure
------------------------------------------------------
`summary.p95_latency_ms` is `percentile_cont` over `query.latency_ms`, measured by PostgreSQL
over real requests. It — not any number produced by a fixture run — is the source for the 3–5
second target in the requirements. `docs/RELEASE.md` records that distinction; a wall-clock
figure from this suite measures the harness.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from backend.repositories.analytics_repository import (
    ActiveUser,
    DocumentUsage,
    KeywordCount,
    QuerySummary,
    TrendPoint,
)
from backend.services.analytics_service import (
    DEFAULT_LIMIT,
    DEFAULT_TREND_POINTS,
    MAX_LIMIT,
    MAX_TREND_POINTS,
)
from backend.tests.regression.conftest import (
    API,
    TAX_QUESTION,
    Api,
    Stack,
    audit_rows,
    csrf_headers,
    switch_to,
    upload,
)

ANALYTICS = f"{API}/analytics"
KEYWORDS = f"{ANALYTICS}/keywords"
DOCUMENTS = f"{ANALYTICS}/documents"
TRENDS = f"{ANALYTICS}/trends"
EXPORT = f"{ANALYTICS}/export"

#: A fixed instant, so nothing in this module depends on the clock.
WHEN = datetime(2026, 3, 12, 9, 30, tzinfo=UTC)

#: The magic bytes of each export format. Asserting the media *type* alone would pass on an
#: empty body or an HTML error page served with the wrong header.
PDF_MAGIC = b"%PDF-"
XLSX_MAGIC = b"PK\x03\x04"  # xlsx is a zip archive


def seed_reports(stack: Stack) -> None:
    """Queue one plausible row per aggregate.

    The values are arbitrary and no test asserts a ranking derived from them — see the module
    docstring. They exist so the response shape is exercised with something other than empty
    lists, which is where a missing field mapping hides.
    """
    stack.analytics_repo.keywords[False] = [
        KeywordCount(keyword="withholding", occurrences=9, queries=7),
        KeywordCount(keyword="maintenance", occurrences=4, queries=4),
    ]
    stack.analytics_repo.keywords[True] = [
        KeywordCount(keyword="surcharge", occurrences=1, queries=1),
    ]
    stack.analytics_repo.documents = [
        DocumentUsage(
            document_id=str(uuid.uuid4()),
            document_name="retail-account-fees.pdf",
            citations=12,
            queries=9,
            last_cited_at=WHEN,
        ),
        # A citation written before the id was recorded. Counted as usage, not deep-linkable.
        DocumentUsage(
            document_id=None,
            document_name="legacy-tariff.pdf",
            citations=2,
            queries=2,
            last_cited_at=None,
        ),
    ]
    # `not_found` and `failed` differ, so a test asserting they are charted apart cannot pass
    # by accident on two equal numbers.
    stack.analytics_repo.trend = [
        TrendPoint(period_start=WHEN, total=6, answered=3, not_found=2, failed=1),
    ]
    stack.analytics_repo.users = [
        ActiveUser(
            user_id=uuid.uuid4(),
            email="standard@tenant-a.example.com",
            queries=6,
            last_query_at=WHEN,
        ),
    ]
    stack.analytics_repo.query_summary = QuerySummary(
        total=6,
        answered=3,
        not_found=2,
        failed=1,
        pending=0,
        avg_latency_ms=812.5,
        p95_latency_ms=1980.0,
        avg_confidence=0.71,
    )


def call_args(stack: Stack, method: str) -> dict[str, Any]:
    """The kwargs the service passed to one aggregate, most recent call."""
    matching = [call for call in stack.analytics_repo.calls if call[0] == method]
    assert matching, f"{method} was never called"
    return matching[-1][2]


def call_tenants(stack: Stack, method: str) -> set[uuid.UUID]:
    return {call[1] for call in stack.analytics_repo.calls if call[0] == method}


class TestKeywordTrends:
    """"Most- and least-searched keywords" — both ends, in one response."""

    async def test_both_ends_of_the_distribution_are_returned(
        self, api: Api, stack: Stack
    ) -> None:
        """The quiet end is the more actionable one: a term nobody searches may be a document
        category users do not know exists. A report with only `most_searched` answers half the
        requirement."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(KEYWORDS)).json()

        assert body["most_searched"]
        assert body["least_searched"]

    async def test_the_aggregate_is_asked_for_both_directions(
        self, api: Api, stack: Stack
    ) -> None:
        """Not two hard-coded lists: the service issues the same aggregate twice, ascending and
        descending, so "least searched" is really the other end of the same counting."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        await client.get(KEYWORDS)

        directions = {
            call[2]["ascending"]
            for call in stack.analytics_repo.calls
            if call[0] == "keyword_counts"
        }
        assert directions == {True, False}

    async def test_each_keyword_carries_both_counts(
        self, api: Api, stack: Stack
    ) -> None:
        """One question repeating a word is not two people asking about it. A ranking that
        cannot distinguish those invites a wrong conclusion, so both numbers are published."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(KEYWORDS)).json()

        for item in body["most_searched"]:
            assert item["keyword"]
            assert item["occurrences"] >= item["queries"] >= 1

    async def test_stopwords_and_a_minimum_length_are_applied(
        self, api: Api, stack: Stack
    ) -> None:
        """Otherwise the top of the ranking is "the", "and", "is" on every corpus. The policy
        belongs to the service; this asserts it reaches the aggregate."""
        client = switch_to(api, stack.a.admin)

        await client.get(KEYWORDS)

        args = call_args(stack, "keyword_counts")
        assert args["min_length"] >= 3
        assert "the" in args["stopwords"]

    async def test_the_window_is_echoed_back(self, api: Api, stack: Stack) -> None:
        """A dashboard tile reading "1,204 queries" means nothing without the period beside
        it, and a caller that sent no window still needs to know what it got."""
        client = switch_to(api, stack.a.admin)

        body = (await client.get(KEYWORDS)).json()

        assert body["window"]["label"]

    async def test_a_requested_window_comes_back_unchanged(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        body = (
            await client.get(
                f"{KEYWORDS}?created_from=2026-01-01T00:00:00Z"
                f"&created_to=2026-02-01T00:00:00Z"
            )
        ).json()

        assert body["window"]["created_from"].startswith("2026-01-01")
        assert body["window"]["created_to"].startswith("2026-02-01")
        assert call_args(stack, "keyword_counts")["created_from"] is not None


class TestDocumentUsage:
    """"Document usage" — which documents answers were actually grounded in."""

    async def test_usage_is_reported_per_document(
        self, api: Api, stack: Stack
    ) -> None:
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(DOCUMENTS)).json()

        assert body["documents"]
        for item in body["documents"]:
            assert item["document_name"]
            assert item["citations"] >= 1

    async def test_a_citation_without_an_id_is_still_counted(
        self, api: Api, stack: Stack
    ) -> None:
        """The id is read out of JSONB written at answer time, so an older reference may not
        carry one. The usage count is still true; the row simply cannot be deep-linked, and
        dropping it would understate how often the document was consulted."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(DOCUMENTS)).json()

        idless = [item for item in body["documents"] if item["document_id"] is None]
        assert idless
        assert idless[0]["citations"] >= 1

    async def test_a_document_nobody_cited_does_not_appear(
        self, api: Api, stack: Stack
    ) -> None:
        """The absence is the finding. This report measures what the system *used*, not what
        was uploaded, so an unread document showing a zero would be a different report."""
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        body = (await client.get(DOCUMENTS)).json()

        assert uploaded["id"] not in [
            item["document_id"] for item in body["documents"]
        ]


class TestQueryVolumeAndActiveUsers:
    """"Query volume over time, most active users"."""

    async def test_volume_is_bucketed_and_split_by_outcome(
        self, api: Api, stack: Stack
    ) -> None:
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(TRENDS)).json()

        assert body["points"]
        point = body["points"][0]
        assert point["total"] == point["answered"] + point["not_found"] + point["failed"]

    async def test_not_found_is_charted_apart_from_failed(
        self, api: Api, stack: Stack
    ) -> None:
        """A not-found is the grounding rule refusing to answer without a source, which is
        correct behaviour. A rising rate means the document repository has a gap — folding it
        into `failed` would send someone to fix the wrong thing."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        point = (await client.get(TRENDS)).json()["points"][0]

        assert "not_found" in point
        assert "failed" in point
        assert point["not_found"] != point["failed"]

    async def test_the_granularity_is_a_closed_set(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(f"{TRENDS}?granularity=week")).status_code == 200
        assert (await client.get(f"{TRENDS}?granularity=fortnight")).status_code == 422

    async def test_the_most_recent_buckets_are_the_ones_returned(
        self, api: Api, stack: Stack
    ) -> None:
        """A dashboard asking for thirty days must not be shown the first thirty days of a
        two-year history."""
        client = switch_to(api, stack.a.admin)

        await client.get(f"{TRENDS}?points=7")

        assert call_args(stack, "volume_trend")["limit"] == 7

    async def test_the_bucket_count_is_bounded(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{TRENDS}?points={MAX_TREND_POINTS + 1}")

        assert response.status_code == 422

    async def test_the_default_bucket_count_is_the_documented_one(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        await client.get(TRENDS)

        assert call_args(stack, "volume_trend")["limit"] == DEFAULT_TREND_POINTS

    async def test_the_busiest_users_are_named(self, api: Api, stack: Stack) -> None:
        """Administrator-only information, and the reason this whole module is admin-only:
        "who searched the most" is a statement about a colleague's behaviour."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(TRENDS)).json()

        assert body["active_users"]
        assert body["active_users"][0]["email"]
        assert body["active_users"][0]["queries"] >= 1

    async def test_the_windows_totals_include_the_production_latency_figure(
        self, api: Api, stack: Stack
    ) -> None:
        """`p95_latency_ms` is `percentile_cont` over `query.latency_ms`. It is the number the
        3–5 second requirement is measured against — see the module docstring on why no
        assertion here compares it to a threshold."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        summary = (await client.get(TRENDS)).json()["summary"]

        assert summary["p95_latency_ms"] is not None
        assert summary["avg_latency_ms"] is not None
        assert summary["total"] == (
            summary["answered"]
            + summary["not_found"]
            + summary["failed"]
            + summary["pending"]
        )


class TestLimitsAreBounded:
    """A report is a table, and an unbounded `limit` on an aggregate over every query in a
    tenant is a denial of service with a legitimate-looking URL."""

    async def test_the_default_limit_is_the_documented_one(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        await client.get(DOCUMENTS)

        assert call_args(stack, "document_usage")["limit"] == DEFAULT_LIMIT

    async def test_a_requested_limit_within_the_cap_is_honoured(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        await client.get(f"{DOCUMENTS}?limit=25")

        assert call_args(stack, "document_usage")["limit"] == 25

    async def test_a_limit_over_the_cap_is_refused_at_the_edge(
        self, api: Api, stack: Stack
    ) -> None:
        """422 rather than a silent clamp, because a caller that asked for 5,000 rows and got
        50 would page through the same 50 forever."""
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{DOCUMENTS}?limit={MAX_LIMIT + 1}")

        assert response.status_code == 422

    async def test_a_reversed_window_is_refused(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        response = await client.get(
            f"{KEYWORDS}?created_from=2026-06-01T00:00:00Z"
            f"&created_to=2026-01-01T00:00:00Z"
        )

        assert response.status_code == 422


class TestTheReportIsScopedToOneTenant:
    """`rbac-enforcement.md`: role and tenant are independent checks. An aggregate is the whole
    tenant's search behaviour, so a leak here is not one row — it is the shape of another
    organization's business."""

    async def test_the_tenant_comes_from_the_caller(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        await client.get(KEYWORDS)
        await client.get(DOCUMENTS)
        await client.get(TRENDS)

        for method in ("keyword_counts", "document_usage", "volume_trend", "active_users"):
            assert call_tenants(stack, method) == {stack.a.id}, method

    async def test_naming_another_tenant_changes_nothing(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{KEYWORDS}?tenant_id={stack.b.id}")

        assert response.status_code == 200
        assert call_tenants(stack, "keyword_counts") == {stack.a.id}

    async def test_two_administrators_get_their_own_tenants(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        await client.get(KEYWORDS)
        client = switch_to(api, stack.b.admin)
        await client.get(KEYWORDS)

        assert call_tenants(stack, "keyword_counts") == {stack.a.id, stack.b.id}

    async def test_a_standard_user_is_refused_every_report(
        self, api: Api, stack: Stack
    ) -> None:
        """Also covered route-by-route in `test_rbac.py`; asserted here because "Administrator
        only" is part of AC-11.1 itself, not only of BR-06."""
        client = switch_to(api, stack.a.user)

        for path in (KEYWORDS, DOCUMENTS, TRENDS):
            assert (await client.get(path)).status_code == 403, path

    async def test_a_refused_report_never_reaches_the_aggregate(
        self, api: Api, stack: Stack
    ) -> None:
        """The check runs before the query, so a denial costs nothing and leaks nothing — not
        even a row count through a timing difference."""
        client = switch_to(api, stack.a.user)

        await client.get(KEYWORDS)

        assert stack.analytics_repo.calls == []


class TestExport:
    """"Exportable as PDF or Excel". A file that leaves the system is the one artifact nobody
    can revoke afterwards, which is why it is a POST, why it is audited separately, and why the
    filename is composed by the server."""

    async def test_a_pdf_export_is_really_a_pdf(self, api: Api, stack: Stack) -> None:
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            EXPORT, json={"report": "keywords", "format": "pdf"}, headers=csrf_headers()
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/pdf")
        assert response.content.startswith(PDF_MAGIC)

    async def test_an_excel_export_is_really_a_spreadsheet(
        self, api: Api, stack: Stack
    ) -> None:
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            EXPORT,
            json={"report": "documents", "format": "excel"},
            headers=csrf_headers(),
        )

        assert response.status_code == 200
        assert "spreadsheetml" in response.headers["content-type"]
        assert response.content.startswith(XLSX_MAGIC)

    async def test_every_report_can_be_exported_in_both_formats(
        self, api: Api, stack: Stack
    ) -> None:
        """Four reports times two formats. A format that only works for one report is the kind
        of gap a dashboard finds in production."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        for report in ("keywords", "documents", "trends", "history"):
            for export_format in ("pdf", "excel"):
                response = await client.post(
                    EXPORT,
                    json={"report": report, "format": export_format},
                    headers=csrf_headers(),
                )
                assert response.status_code == 200, f"{report}/{export_format}"
                assert response.content, f"{report}/{export_format}"

    async def test_the_filename_is_composed_by_the_server(
        self, api: Api, stack: Stack
    ) -> None:
        """It lands in `Content-Disposition`, where a newline is header injection and a `../`
        is a path the client will happily write to. The request has no field for it."""
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            EXPORT, json={"report": "keywords", "format": "pdf"}, headers=csrf_headers()
        )

        disposition = response.headers["content-disposition"]
        assert disposition.startswith("attachment; ")
        assert "docquery-keywords-" in disposition
        assert ".pdf" in disposition

    async def test_a_caller_supplied_filename_is_rejected(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            EXPORT,
            json={
                "report": "keywords",
                "format": "pdf",
                "filename": "../../etc/passwd",
            },
            headers=csrf_headers(),
        )

        assert response.status_code == 422

    async def test_the_row_count_is_reported_alongside_the_file(
        self, api: Api, stack: Stack
    ) -> None:
        """A client cannot read a row count out of a PDF, and an export that quietly stopped at
        the cap and looked complete is a compliance defect rather than a display bug."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            EXPORT, json={"report": "keywords", "format": "pdf"}, headers=csrf_headers()
        )

        assert int(response.headers["X-Export-Rows"]) > 0
        assert response.headers["X-Export-Truncated"] == "false"

    async def test_an_export_is_never_cached(self, api: Api, stack: Stack) -> None:
        """The next request may come from a different Administrator, and the audit row says who
        received these bytes."""
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            EXPORT, json={"report": "keywords", "format": "pdf"}, headers=csrf_headers()
        )

        assert response.headers["cache-control"] == "no-store"

    async def test_an_export_without_a_csrf_token_is_refused(
        self, api: Api, stack: Stack
    ) -> None:
        """The JWT lives in an httpOnly cookie, so a cross-site page could otherwise make an
        Administrator's browser export the tenant's search behaviour to an attacker."""
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            EXPORT, json={"report": "keywords", "format": "pdf"}
        )

        assert response.status_code == 403

    async def test_a_standard_user_cannot_export(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        response = await client.post(
            EXPORT, json={"report": "keywords", "format": "pdf"}, headers=csrf_headers()
        )

        assert response.status_code == 403

    async def test_an_unknown_report_is_refused(self, api: Api, stack: Stack) -> None:
        """`ExportReport` is a closed enum whose value selects a Python branch and never
        reaches SQL, so there is nothing here for a caller to widen — including to the audit
        log, which has no export at all."""
        client = switch_to(api, stack.a.admin)

        for report in ("audit", "users", "documents; drop table document"):
            response = await client.post(
                EXPORT,
                json={"report": report, "format": "pdf"},
                headers=csrf_headers(),
            )
            assert response.status_code == 422, report


class TestEveryReportAndExportIsAudited:
    """BR-05 applies to analytics too: a report is a read of the whole tenant's behaviour, and
    an export takes it out of the system."""

    async def test_a_report_writes_an_analytics_view_row(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        await client.get(KEYWORDS)

        rows = audit_rows(stack, "ANALYTICS_VIEW", actor_email=stack.a.admin.email)
        assert rows

    async def test_an_export_writes_a_data_export_row(
        self, api: Api, stack: Stack
    ) -> None:
        """`DATA_EXPORT`, not `ANALYTICS_VIEW`: a file left the system and can be forwarded
        anywhere, so it is a different event from looking at a chart."""
        client = switch_to(api, stack.a.admin)

        await client.post(
            EXPORT, json={"report": "keywords", "format": "pdf"}, headers=csrf_headers()
        )

        rows = audit_rows(stack, "DATA_EXPORT", actor_email=stack.a.admin.email)
        assert rows
        assert rows[-1].detail is not None
        assert rows[-1].detail["truncated"] is False

    async def test_the_export_row_records_the_filename_and_row_count(
        self, api: Api, stack: Stack
    ) -> None:
        """What was taken, how much of it, and as what — the three questions asked after a
        leak. The server composed the filename, so recording it introduces no user input."""
        seed_reports(stack)
        client = switch_to(api, stack.a.admin)

        await client.post(
            EXPORT,
            json={"report": "documents", "format": "excel"},
            headers=csrf_headers(),
        )

        detail = audit_rows(stack, "DATA_EXPORT")[-1].detail
        assert detail is not None
        assert detail["filename"].endswith(".xlsx")
        assert detail["rows"] >= 1

    async def test_a_refused_report_is_audited_as_denied(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        await client.get(TRENDS)

        assert stack.denials()

    async def test_no_audit_detail_carries_a_question_or_a_keyword(
        self, api: Api, stack: Stack
    ) -> None:
        """A history export contains question text by design — that is what was asked for. The
        *audit row* about it must not, or the append-only log becomes a second copy of every
        question, in the one place nothing can be removed from."""
        stack.bedrock.queue_answer("Withholding tax is 10 percent [S1].")
        user = switch_to(api, stack.a.user)
        await user.post(
            f"{API}/queries",
            json={"question_text": TAX_QUESTION},
            headers=csrf_headers(),
        )
        client = switch_to(api, stack.a.admin)

        await client.post(
            EXPORT, json={"report": "history", "format": "pdf"}, headers=csrf_headers()
        )

        written = stack.audit_details()
        assert TAX_QUESTION not in written
        assert "withholding tax" not in written.lower()
