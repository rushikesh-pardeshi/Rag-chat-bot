"""backend/api/routes/analytics.py

Role: Administrator-only endpoints — keyword trends, document usage, query volume, and
PDF/Excel export.

Every route takes `AdminUser`, and the service checks the role again
------------------------------------------------------------------
The duplication is intentional and `analytics_service._authorize` explains the other half: the
dependency's denial covers HTTP callers, and the service's covers every other caller — a
scheduled report, a future CLI, a test. Analytics is the whole tenant's search behaviour
aggregated, which is the one thing a Standard User must never see even in summary
(`.claude/rules/rbac-enforcement.md`).

The tenant is never a parameter
-------------------------------
It comes off `actor.tenant_id`. There is no route argument that could name another
organization, which is the same reasoning `audit_logs.py` records: role and tenant are
independent checks and both apply on every request.

The export is a POST
--------------------
Not because it changes domain state, but because it *records* state — every export writes a
`DATA_EXPORT` audit row — and because an unsafe method is what carries the double-submit CSRF
token (`.claude/rules/secure-coding-vapt.md`). A `GET` that appends to the audit log would also
be cacheable and pre-fetchable, and a browser prefetching a compliance export is a false entry
in the record of who took data out of the system.

No route here builds a report
-----------------------------
Each function validates, calls one service method, and maps the result to a DTO. The export
route wraps bytes the service produced in a `Response`; it does not choose the columns, the row
cap, or the filename — `architecture.md` puts that in the service, where the RBAC check is.
"""

from __future__ import annotations

from datetime import datetime
from urllib.parse import quote

from fastapi import APIRouter, Query, Response, status

from backend.api.dependencies.auth import AdminUser, RequestCtx
from backend.api.dependencies.services import AnalyticsServiceDep
from backend.repositories.analytics_repository import TrendGranularity
from backend.schemas.analytics_schemas import (
    AnalyticsExportRequest,
    AnalyticsFilter,
    DocumentUsageResponse,
    KeywordTrendResponse,
    TrendFilter,
    UsageTrendResponse,
)
from backend.services.analytics_service import (
    DEFAULT_LIMIT,
    DEFAULT_TREND_POINTS,
    MAX_LIMIT,
    MAX_TREND_POINTS,
)

router = APIRouter(prefix="/analytics", tags=["analytics"])

#: Row count and truncation flag, surfaced as headers because a client cannot read them out of
#: a PDF. `MAX_EXPORT_ROWS` truncation is also written into the report title and the audit
#: row — three places, because an export that quietly stopped at a thousand rows and looked
#: complete is a compliance defect, not a display bug.
ROWS_HEADER = "X-Export-Rows"
TRUNCATED_HEADER = "X-Export-Truncated"


@router.get(
    "/keywords",
    response_model=KeywordTrendResponse,
    summary="Most- and least-searched keywords (Administrator only)",
)
async def keyword_trends(
    actor: AdminUser,
    context: RequestCtx,
    service: AnalyticsServiceDep,
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> KeywordTrendResponse:
    """Both ends of the keyword distribution for the window, plus the window's totals.

    Least-searched is returned alongside most-searched because BR-06 asks for both, and the
    quiet end is the more actionable one: a term nobody searches for may be a document category
    users do not know exists.

    The parameters are re-validated through `AnalyticsFilter` rather than passed straight on,
    so the "from must not be after to" rule lives in one place and reads the same on all three
    of these endpoints.
    """
    filters = AnalyticsFilter(created_from=created_from, created_to=created_to, limit=limit)
    report = await service.keyword_trends(
        actor=actor,
        context=context,
        limit=filters.limit,
        created_from=filters.created_from,
        created_to=filters.created_to,
    )
    return KeywordTrendResponse.from_dto(report)


@router.get(
    "/documents",
    response_model=DocumentUsageResponse,
    summary="Most-cited documents (Administrator only)",
)
async def document_usage(
    actor: AdminUser,
    context: RequestCtx,
    service: AnalyticsServiceDep,
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
) -> DocumentUsageResponse:
    """Which documents answers were actually grounded in, most-cited first.

    Counted from the citations stored on each answer, not from the `document` table — so this
    measures what the system *used*, which is the figure a compliance reviewer asks for. A
    document nobody's question ever reached does not appear at all, and that absence is the
    finding.
    """
    filters = AnalyticsFilter(created_from=created_from, created_to=created_to, limit=limit)
    report = await service.document_usage(
        actor=actor,
        context=context,
        limit=filters.limit,
        created_from=filters.created_from,
        created_to=filters.created_to,
    )
    return DocumentUsageResponse.from_dto(report)


@router.get(
    "/trends",
    response_model=UsageTrendResponse,
    summary="Query volume over time and the busiest users (Administrator only)",
)
async def usage_trends(
    actor: AdminUser,
    context: RequestCtx,
    service: AnalyticsServiceDep,
    granularity: TrendGranularity = Query(
        default=TrendGranularity.DAY,
        description="Bucket size: day, week or month.",
    ),
    points: int = Query(
        default=DEFAULT_TREND_POINTS,
        ge=1,
        le=MAX_TREND_POINTS,
        description="How many of the most recent buckets to return.",
    ),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    limit: int = Query(
        default=DEFAULT_LIMIT,
        ge=1,
        le=MAX_LIMIT,
        description="How many of the most active users to include.",
    ),
) -> UsageTrendResponse:
    """Volume per period split by outcome, the window's totals, and the most active users.

    `granularity` is typed as the enum, so an unknown unit is a 422 here and never reaches
    `date_trunc`. `points` takes the *most recent* buckets: a dashboard asking for thirty days
    must not be shown the first thirty days of a two-year history.

    `not_found` is charted separately from `failed`. A not-found is the grounding rule refusing
    to answer without a source, which is correct behaviour — a rising rate means the document
    repository has a gap, not that the system is erring.
    """
    filters = TrendFilter(
        created_from=created_from,
        created_to=created_to,
        limit=limit,
        granularity=granularity,
        points=points,
    )
    report = await service.usage_trends(
        actor=actor,
        context=context,
        granularity=filters.granularity,
        points=filters.points,
        user_limit=filters.limit,
        created_from=filters.created_from,
        created_to=filters.created_to,
    )
    return UsageTrendResponse.from_dto(report)


@router.post(
    "/export",
    status_code=status.HTTP_200_OK,
    summary="Export a report as PDF or Excel (Administrator only)",
    responses={
        200: {
            "content": {
                "application/pdf": {},
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": {},
            },
            "description": "The rendered report.",
        }
    },
)
async def export_report(
    payload: AnalyticsExportRequest,
    actor: AdminUser,
    context: RequestCtx,
    service: AnalyticsServiceDep,
) -> Response:
    """Render one report to a file and return the bytes.

    There is no `response_model`: the body is a binary document, so the route returns a
    `Response` directly. That is the route's job — choosing a status code and content type —
    and not the service's, which hands back `ExportedFile` and knows nothing about HTTP
    (`architecture.md`).

    The filename comes from the service and contains no user input, because it lands in
    `Content-Disposition`, where a newline is header injection and a `../` is a path the client
    will happily write to. It is still `quote`d here: two independent controls on the same
    header cost nothing, and this one holds even if the naming rule upstream is later relaxed.
    """
    exported = await service.export(
        actor=actor,
        context=context,
        report=payload.report,
        export_format=payload.format,
        granularity=payload.granularity,
        limit=payload.limit,
        created_from=payload.created_from,
        created_to=payload.created_to,
    )
    return Response(
        content=exported.content,
        media_type=exported.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{quote(exported.filename)}"',
            ROWS_HEADER: str(exported.rows),
            TRUNCATED_HEADER: "true" if exported.truncated else "false",
            # A compliance export must not be served from a cache: the next request may come
            # from a different Administrator, and the audit row says who received these bytes.
            "Cache-Control": "no-store",
        },
    )
