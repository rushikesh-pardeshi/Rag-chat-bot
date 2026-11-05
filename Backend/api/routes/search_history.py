"""backend/api/routes/search_history.py

Role: `GET /api/v1/history` — one endpoint that serves both roles, scoped by the caller's role.

One endpoint, not two
---------------------
There is no `/history/mine` and no `/history/all`. A Standard User calling this gets their own
rows; an Administrator gets the tenant's. Two endpoints would mean the *route* decides the
scope, and then `require_admin` on one of them would be the only thing standing between a
Standard User and everyone's questions — a single missing decorator away from a breach. With
one endpoint the scope is derived from the authenticated principal inside
`search_history_service`, where it is unit-tested against both roles.

That is also why there is no `require_admin` here and its absence is not an oversight. The
route is open to any authenticated user *because the service narrows what they see*.

`user_id` is a filter, not a selector
-------------------------------------
Accepted for both roles and interpreted differently by each: an Administrator's filter, and
ignored for a Standard User (whose scope is already their own id). It is declared here rather
than rejected for non-admins so a Standard User probing it gets their own history and an audit
entry — not a 403 that confirms the parameter does something.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Query

from backend.api.dependencies.auth import CurrentUser, RequestCtx
from backend.api.dependencies.services import SearchHistoryServiceDep
from backend.models.query import QueryStatus
from backend.schemas.query_schemas import (
    QueryResponse,
    SearchHistoryPage,
    SearchHistoryStats,
)
from backend.services.search_history_service import (
    DEFAULT_PAGE_SIZE,
    MAX_KEYWORD_LENGTH,
    MAX_PAGE_SIZE,
)

router = APIRouter(prefix="/history", tags=["history"])


@router.get(
    "",
    response_model=SearchHistoryPage,
    summary="Search history — own for Standard Users, tenant-wide for Administrators",
)
async def list_history(
    actor: CurrentUser,
    context: RequestCtx,
    service: SearchHistoryServiceDep,
    user_id: uuid.UUID | None = Query(
        default=None,
        description=(
            "Administrator filter: show one user's history. Ignored for Standard Users, "
            "whose scope is always their own."
        ),
    ),
    query_status: QueryStatus | None = Query(
        default=None,
        alias="status",
        description="Filter by outcome: PENDING, ANSWERED, NOT_FOUND or FAILED.",
    ),
    keyword: str | None = Query(default=None, max_length=MAX_KEYWORD_LENGTH),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> SearchHistoryPage:
    """One page of history, with `scope` saying which authorization path produced it.

    `query_status` is aliased to `status` on the wire: the query parameter reads naturally as
    `?status=ANSWERED`, while the Python name avoids shadowing `fastapi.status`, which this
    module's siblings import. Typing it as `QueryStatus` means an unknown value is a 422 from
    FastAPI rather than a string reaching the repository.

    `keyword` is bounded here and matched with an escaped `ILIKE` in the repository — never a
    user-supplied regex (`secure-coding-vapt.md`, ReDoS).
    """
    page = await service.list_history(
        actor=actor,
        context=context,
        user_id=user_id,
        status=query_status,
        keyword=keyword,
        created_from=created_from,
        created_to=created_to,
        limit=limit,
        offset=offset,
    )
    return SearchHistoryPage(
        items=QueryResponse.from_entity_list(page.items),
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        scope=page.scope,
    )


@router.get(
    "/stats",
    response_model=SearchHistoryStats,
    summary="Question counts by outcome, under the caller's own scope",
)
async def history_stats(
    actor: CurrentUser,
    context: RequestCtx,
    service: SearchHistoryServiceDep,
    user_id: uuid.UUID | None = Query(
        default=None,
        description=(
            "Administrator filter: count one user's questions. Ignored for Standard Users, "
            "whose scope is always their own."
        ),
    ),
    keyword: str | None = Query(default=None, max_length=MAX_KEYWORD_LENGTH),
    created_from: datetime | None = Query(default=None),
    created_to: datetime | None = Query(default=None),
) -> SearchHistoryStats:
    """The outcome breakdown for a filter, in one request and one audit row.

    Declared **before** nothing and after the list route, but note it is a literal path under
    the same prefix as no parameterised sibling — `/history` has no `/{id}` route, so there is
    no ordering hazard of the kind `queries.py` documents for `/queries/suggestions`.

    Takes no `status` parameter on purpose: the response *is* the breakdown by status, and
    narrowing to one outcome first would return that single number, which the list endpoint's
    `total` already gives.

    `CurrentUser`, not `AdminUser`: a Standard User may see the shape of **their own** history.
    The service narrows the scope and discards a foreign `user_id` rather than refusing it, so
    the parameter is not an authorization input (`rbac-enforcement.md`).
    """
    stats = await service.history_stats(
        actor=actor,
        context=context,
        user_id=user_id,
        keyword=keyword,
        created_from=created_from,
        created_to=created_to,
    )
    return SearchHistoryStats(
        scope=stats.scope,
        total=stats.total,
        # Keyed by the enum's *value*, so the JSON reads `{"ANSWERED": 102}` rather than
        # carrying Python enum reprs.
        by_status={status.value: count for status, count in stats.by_status.items()},
    )
