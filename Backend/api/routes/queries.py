"""backend/api/routes/queries.py

Role: HTTP routes for submitting a question and reading its record.

**Submission answers the question in the same request.** `query_service.submit` records the
row, then runs retrieval and generation and fills that row in place, so a caller normally
receives `ANSWERED` or `NOT_FOUND` straight away. The response is nonetheless **202, not 201**:
the row is created `PENDING`, and it stays `PENDING` when no RAG pipeline is configured. The
status code says "accepted", which is true in both cases, and `GET /queries/{id}` is the poll
endpoint for the second one — so the contract did not have to change when answers arrived.

All three routes are open to any authenticated user. The ownership rule — a Standard User reads
their own queries, an Administrator reads any in their tenant — is enforced in
`query_service.get_query`, not here: it depends on the row, and a route that fetched the row
to check it would be doing the service's job.

`GET /queries/suggestions` is a *literal* path sharing a prefix with `GET /queries/{query_id}`,
so it is declared first. FastAPI matches in declaration order and would otherwise try
`suggestions` as a UUID and answer 422. Any further literal sub-path added here goes above the
parameterised route too.
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, Response, status

from backend.api.dependencies.auth import CurrentUser, RequestCtx
from backend.api.dependencies.services import QueryServiceDep, RagServiceDep
from backend.schemas.query_schemas import (
    QueryCreate,
    QueryResponse,
    QuerySuggestionsResponse,
)

router = APIRouter(prefix="/queries", tags=["queries"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a natural-language question",
)
async def submit_query(
    payload: QueryCreate,
    actor: CurrentUser,
    context: RequestCtx,
    service: QueryServiceDep,
) -> QueryResponse:
    """Record the question as `PENDING` and audit `QUERY_SUBMIT`.

    `user_id` and `tenant_id` are taken from the session, never from the body — `QueryCreate`
    forbids extra fields, so a request that tries to supply either is a 422.
    """
    query = await service.submit(
        question=payload.question_text,
        follow_up_to=payload.follow_up_to, actor=actor, context=context
    )
    return QueryResponse.from_entity(query)


@router.get(
    "/suggestions",
    response_model=QuerySuggestionsResponse,
    summary="Example questions this workspace can answer",
)
async def suggest_questions(
    actor: CurrentUser,
    service: RagServiceDep,
    response: Response,
    offset: Annotated[
        int,
        Query(
            ge=0,
            le=10_000,
            description=(
                "Which window of the verified pool to return. The panel's refresh passes "
                "the next multiple of the window size; the server wraps around, so an "
                "offset past the end is never an error and never an empty panel."
            ),
        ),
    ] = 0,
) -> QuerySuggestionsResponse:
    """Questions built from the tenant's own documents and verified to retrieve.

    **Declared before `GET /{query_id}` on purpose.** FastAPI matches in declaration order, so
    with the parameterised route first this path would be parsed as a query id and answered
    with a 422 about an invalid UUID.

    Open to both roles, and tenant-scoped by `actor.tenant_id` alone — there is no parameter
    that could widen it. Nothing here is admin-only: a Standard User may already submit any
    question and receive whole passages as citations, so a phrase drawn from a document they
    can already search exposes no new class of data.

    Not audited, and that is a deliberate omission rather than an oversight. The audit action
    enum has no member for reading a hint, and adding one is a database migration (the F-13
    pattern in `docs/RELEASE.md`); the questions a user actually asks are audited when they are
    asked, which is the event a compliance review is looking for.

    **`no-store`, deliberately, and this is the one thing not to "optimise" here.** Serving
    these fast is a real requirement, and a `private, max-age=…` would make a repeat visit
    instant — but the browser's HTTP cache is keyed by URL, not by session, so a second person
    signing in on the same machine inside the window would be handed the first person's
    tenant's document names. Speed belongs on the server side instead, where the cache is keyed
    by `tenant_id`: see `SUGGESTION_CACHE_TTL_SECONDS` in `rag_service.py`.
    """
    response.headers["Cache-Control"] = "no-store"
    suggested = await service.suggest_questions(
        tenant_id=actor.tenant_id, offset=offset
    )
    return QuerySuggestionsResponse(
        items=list(suggested.questions),
        retrieval_available=suggested.available,
        total_available=suggested.total,
    )


@router.get("/{query_id}", response_model=QueryResponse, summary="Get one query record")
async def get_query(
    query_id: uuid.UUID,
    actor: CurrentUser,
    context: RequestCtx,
    service: QueryServiceDep,
) -> QueryResponse:
    """One query with its answer and sources, if it has been answered.

    Another user's query returns the same 404 as a nonexistent id, and audits the denial.
    This is also the endpoint a client polls while a query is `PENDING`.
    """
    query = await service.get_query(query_id, actor=actor, context=context)
    return QueryResponse.from_entity(query)
