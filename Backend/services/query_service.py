"""backend/services/query_service.py

Role: Query submission, answering, and retrieval of a query's own record. Owns the rules about
what a question may be, who may read the answer, and what is written to the audit trail.

**The row is written before the answer is attempted, always.** `submit` validates the
question, writes a `PENDING` row, audits `QUERY_SUBMIT`, and only then runs the RAG pipeline.
The split is deliberate rather than incidental: a question that later fails, times out or is
refused for lack of grounding is still in the audit trail. A query recorded only on success
would hide exactly the queries a compliance review most wants to see (BR-04, BR-05).

**A failure downgrades the answer; it never loses the question.** If `rag_service` raises,
this service records `FAILED` on the existing row, audits the outcome, and re-raises so the
caller gets a 503. There is no path where an exception leaves a `PENDING` row behind with
nothing to explain it.

`rag_service` is optional on the constructor. With no pipeline injected, `submit` records the
question and stops — which is what the Phase 3 tests assert, and what a deployment with no
Bedrock access degrades to rather than erroring on every question.

**Answers will carry sources or they will not exist.** Nothing in this file can attach an
answer without source references — `attach_answer` in the repository writes answer, sources
and status in a single statement, so there is no reachable state where an answer is visible
and its citations are not (BR-03, `.claude/rules/rag-answer-grounding.md`).

Ownership: a Standard User may read their own queries only; an Administrator may read any
query in their tenant. Both dimensions are checked — role *and* tenant — because an
Administrator in tenant A has no claim on tenant B's questions.

Listing history is *not* here. `search_history_service` (Phase 4) owns that read, and it picks
between `list_for_user` and `list_for_tenant` by role; putting a second listing here would give
the same authorization decision two implementations free to diverge.
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any, Final

from backend.core.exceptions import QueryNotFound, ValidationError
from backend.core.logging import get_logger
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.query import Query, QueryStatus
from backend.models.user import User, UserRole
from backend.repositories.query_repository import QueryRepository
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.utils.ids import new_id, utc_now

if TYPE_CHECKING:  # pragma: no cover - import cycle avoidance only
    # `rag_service` imports nothing from here, so this is not a true cycle — but the
    # annotation-only import keeps `query_service` importable without Bedrock or Pinecone
    # present, which is what lets `query_schemas` import its length constants from here.
    from backend.services.rag_service import PriorTurn, RagService

logger = get_logger(__name__)

#: Shortest question accepted. Below this there is nothing to retrieve against — "tax" is a
#: keyword, not a question, and embedding it produces a vector that matches everything.
MIN_QUESTION_LENGTH: Final = 3
#: Longest question accepted. Also enforced by the schema; repeated here because a service
#: invariant that only exists in a DTO stops existing the moment another caller appears.
MAX_QUESTION_LENGTH: Final = 2_000

#: Discriminates the two audit entries a single query produces, both written under
#: `AuditAction.QUERY_SUBMIT`.
#:
#: There is no `QUERY_ANSWER` action, and that is a deliberate omission rather than an
#: oversight. `audit_log.action` is a native PostgreSQL enum (`audit_action`), so a new member
#: needs `ALTER TYPE ... ADD VALUE` in a migration — a schema change, which `workflow.md`
#: requires asking about before making. Until that decision is taken, the phase lives in
#: `detail`, which is JSONB and needs no migration.
#:
#: Both rules are still satisfied: `audit-log-immutability.md` gets its who/what/when/object on
#: each entry, and `document-source-traceability.md` gets its source references logged. The
#: cost is that a reviewer filters on `detail->>'phase'` instead of on `action`, which is
#: indexed. Worth revisiting if audit queries get slow.
AUDIT_PHASE_SUBMIT: Final = "submit"
AUDIT_PHASE_ANSWER: Final = "answer"


class QueryService:
    def __init__(
        self,
        queries: QueryRepository,
        audit: AuditService,
        rag: RagService | None = None,
    ) -> None:
        self._queries = queries
        self._audit = audit
        self._rag = rag

    # -- Submit -------------------------------------------------------------
    async def submit(
        self,
        *,
        question: str,
        actor: User,
        context: RequestContext,
        follow_up_to: uuid.UUID | None = None,
    ) -> Query:
        """Record a question, then answer it from the tenant's own documents.

        `tenant_id` and `user_id` come from the authenticated user, never from the request
        body — a caller who could name their own tenant would be able to file questions
        against, and later read history from, another organization. The same `tenant_id` is
        what scopes retrieval, so tenant isolation in the answer is the same fact as tenant
        isolation in the record.
        """
        cleaned = _normalize_question(question)

        query = await self._queries.create(
            query_id=new_id(),
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            question_text=cleaned,
            status=QueryStatus.PENDING,
        )

        await self._audit_query(
            actor,
            context,
            action=AuditAction.QUERY_SUBMIT,
            query_id=query.id,
            outcome=AuditOutcome.SUCCESS,
            # Length, not text. The question itself can name a customer or an account —
            # `secure-coding-vapt.md` forbids that in a log, and the question is already
            # stored once, in the `query` row a reviewer reads through the audit entry's id.
            detail={
                "phase": AUDIT_PHASE_SUBMIT,
                "question_length": len(cleaned),
                "status": query.status.value,
                # Which turn this one followed, when it followed one. An id, not text, and it
                # is the only record of the thread: `query` has no parent column, so adding
                # one would be a migration. JSONB `detail` carries it without changing the
                # schema, and a reviewer reconstructing a conversation can follow it.
                **({"follow_up_to": str(follow_up_to)} if follow_up_to else {}),
            },
        )
        logger.info("query.submitted", query_id=str(query.id))

        if self._rag is None:
            # No pipeline configured. The question is recorded and stays `PENDING`; nothing
            # fabricates an answer to fill the gap.
            return query

        prior = await self._prior_turn(follow_up_to, actor=actor)
        return await self._answer(
            query, cleaned, actor=actor, context=context, prior=prior
        )

    async def _prior_turn(
        self, follow_up_to: uuid.UUID | None, *, actor: User
    ) -> PriorTurn | None:
        """The turn a follow-up refers to, or `None`.

        **Read through the same tenant and ownership rules as `get_query`**, and that is the
        point of doing it here rather than trusting the client: `follow_up_to` is a query id
        from a request body, so without the ownership check a Standard User could name another
        person's query id and have its question and answer summarised into their own answer's
        context. Reading someone else's history is exactly what `rbac-enforcement.md` reserves
        to an Administrator, and a conversational feature is a poor reason to open it.

        A missing, unanswered or cross-tenant id degrades to `None` rather than raising: the
        follow-up is then answered as a standalone question, which is what it would have been
        before this existed. A 404 would turn a stale browser tab into a failed question.
        """
        if follow_up_to is None:
            return None
        # Imported here rather than at module scope for the reason given at the top of this
        # file: the annotation-only import is what keeps `query_service` loadable without
        # Bedrock or Pinecone, and `query_schemas` depends on that to read its length
        # constants from here. A function-level import is the cost of that property.
        from backend.services.rag_service import PriorTurn

        previous = await self._queries.get(follow_up_to, actor.tenant_id)
        if previous is None:
            return None
        if actor.role is not UserRole.ADMINISTRATOR and previous.user_id != actor.id:
            logger.info("query.follow_up_not_owner", query_id=str(follow_up_to))
            return None
        if not previous.answer_text or previous.status is not QueryStatus.ANSWERED:
            # A not-found or failed turn carries no resolvable referent — "more details about
            # it" after "Information not found in uploaded documents" refers to nothing.
            return None
        return PriorTurn(question=previous.question_text, answer=previous.answer_text)

    async def _answer(
        self,
        query: Query,
        question: str,
        *,
        actor: User,
        context: RequestContext,
        prior: PriorTurn | None = None,
    ) -> Query:
        """Run the pipeline and attach the outcome to the existing row.

        Latency is measured around the whole pipeline — embedding, retrieval, the PostgreSQL
        re-check and generation — because that is what the user waited for and what the 3–5
        second target in the requirements refers to. Timing only the model call would report a
        number no user experiences.
        """
        started = time.monotonic()
        try:
            result = await self._rag.answer(  # type: ignore[union-attr]
                question=question, tenant_id=actor.tenant_id, prior=prior
            )
        except Exception as exc:
            latency_ms = int((time.monotonic() - started) * 1000)
            await self._queries.attach_answer(
                query.id,
                actor.tenant_id,
                # No answer text on a failure. A message like "Bedrock unavailable" stored
                # here would later be read back as the answer the system gave, and a
                # compliance reviewer would find an error string in an answer field.
                answer_text=None,
                source_references=[],
                confidence_score=None,
                status=QueryStatus.FAILED,
                latency_ms=latency_ms,
                answered_at=utc_now(),
            )
            await self._audit_query(
                actor,
                context,
                action=AuditAction.QUERY_SUBMIT,
                query_id=query.id,
                outcome=AuditOutcome.FAILURE,
                # The exception *type*, never its message: a botocore error carries the AWS
                # account id and the model arn, and a database error can echo row values.
                detail={
                    "phase": AUDIT_PHASE_ANSWER,
                    "status": QueryStatus.FAILED.value,
                    "error_type": type(exc).__name__,
                    "latency_ms": latency_ms,
                },
            )
            logger.warning(
                "query.failed", query_id=str(query.id), error_type=type(exc).__name__
            )
            raise

        latency_ms = int((time.monotonic() - started) * 1000)
        status = QueryStatus.ANSWERED if result.answered else QueryStatus.NOT_FOUND
        sources = [dict(reference) for reference in result.sources]

        await self._queries.attach_answer(
            query.id,
            actor.tenant_id,
            answer_text=result.answer_text,
            source_references=sources,
            confidence_score=result.confidence,
            status=status,
            latency_ms=latency_ms,
            answered_at=utc_now(),
        )

        await self._audit_query(
            actor,
            context,
            action=AuditAction.QUERY_SUBMIT,
            query_id=query.id,
            outcome=AuditOutcome.SUCCESS,
            # The source references are recorded in the audit trail as document id plus
            # location — that is the traceability requirement in
            # `document-source-traceability.md` ("Log all source references in audit trail").
            # The passage text is not: it is customer document content, it is already stored
            # in `document_chunk`, and the audit log is append-only and never redactable.
            detail={
                "phase": AUDIT_PHASE_ANSWER,
                "status": status.value,
                "latency_ms": latency_ms,
                "confidence": result.confidence,
                "candidates": result.candidates,
                "sources": [
                    {
                        "document_id": reference.get("document_id"),
                        "section_or_page": reference.get("section_or_page"),
                    }
                    for reference in sources
                ],
            },
        )
        logger.info(
            "query.answered",
            query_id=str(query.id),
            status=status.value,
            latency_ms=latency_ms,
            sources=len(sources),
        )

        # Re-read so the caller sees the attached answer. `attach_answer` issues an UPDATE
        # rather than mutating the instance, so the in-memory object is stale otherwise — and
        # returning it would send the user a `PENDING` row with no answer.
        refreshed = await self._queries.get(query.id, actor.tenant_id)
        return refreshed or query

    # -- Read ---------------------------------------------------------------
    async def get_query(
        self, query_id: uuid.UUID, *, actor: User, context: RequestContext
    ) -> Query:
        """One query record, subject to tenant *and* ownership.

        A Standard User asking for another user's query gets the same `QueryNotFound` as a
        nonexistent id, plus an audit row with `outcome='DENIED'`. Distinguishing the two
        would confirm that a given query id exists, which is enough to enumerate activity.
        """
        query = await self._queries.get(query_id, actor.tenant_id)
        if query is None:
            await self._audit_query(
                actor,
                context,
                action=AuditAction.HISTORY_VIEW,
                query_id=query_id,
                outcome=AuditOutcome.DENIED,
                detail={"reason": "not_found_or_other_tenant"},
            )
            raise QueryNotFound()

        if actor.role is not UserRole.ADMINISTRATOR and query.user_id != actor.id:
            await self._audit_query(
                actor,
                context,
                action=AuditAction.HISTORY_VIEW,
                query_id=query_id,
                outcome=AuditOutcome.DENIED,
                detail={"reason": "not_owner"},
            )
            raise QueryNotFound()
        return query

    # -- Audit helper -------------------------------------------------------
    async def _audit_query(
        self,
        actor: User,
        context: RequestContext,
        *,
        action: AuditAction,
        query_id: uuid.UUID,
        outcome: AuditOutcome,
        detail: dict[str, Any] | None = None,
    ) -> None:
        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=action,
            object_type=AuditObjectType.QUERY,
            object_id=str(query_id),
            outcome=outcome,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail=detail,
        )


def _normalize_question(question: str) -> str:
    """Trim, collapse the obvious whitespace, and bound the length.

    Control characters are stripped rather than rejected: a question pasted from a PDF picks
    up form feeds and NULs, and NUL in particular aborts the INSERT. The remaining text is
    stored verbatim — it is the user's question and a reviewer must see what was asked.
    """
    if question is None:
        raise ValidationError("A question is required.")
    cleaned = "".join(
        character
        for character in question
        if character in "\n\t" or character.isprintable()
    ).strip()
    if len(cleaned) < MIN_QUESTION_LENGTH:
        raise ValidationError(
            f"A question must be at least {MIN_QUESTION_LENGTH} characters."
        )
    if len(cleaned) > MAX_QUESTION_LENGTH:
        raise ValidationError(
            f"A question may not exceed {MAX_QUESTION_LENGTH} characters."
        )
    return cleaned
