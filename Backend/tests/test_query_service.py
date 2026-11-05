"""backend/tests/test_query_service.py

Role: Unit tests for query submission, answering, and query reads — question validation, the
`PENDING` row written before any answering, what the answer path writes to the row and the
audit trail, ownership and tenant checks on the read, and the schema guarantees that stop an
answer existing without its sources.

`TestSubmit` runs with no pipeline injected, which is a real deployment shape and not just a
convenience: with no Bedrock access the service records the question and stops.
`TestAnswerPath` injects the *real* `RagService` over the Bedrock and Pinecone fakes, so the
five outcomes a query can have are exercised through the same code production runs — answered,
nothing retrieved, everything below the threshold, a citation that resolves to nothing, and a
match pointing at a document that was soft-deleted after it was indexed.

Overlap with `test_rag_service.py` is deliberate and narrow. That file asks what the pipeline
decides; this one asks what is *recorded* about the decision — `status`, `latency_ms`,
`source_references`, and an audit entry with the citations but none of the passage text. A
pipeline that answered correctly and wrote `PENDING` to the row would pass every test there.

The schema tests at the bottom belong in this file rather than in a schema-only module because
`QueryResponse.answered_must_cite` is BR-03 expressed as a validator — it is a rule about the
query domain, not about serialization.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.core.config import Settings, get_settings
from backend.core.exceptions import QueryNotFound, RetrievalUnavailable, ValidationError
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.document import DocumentStatus
from backend.models.query import Query, QueryStatus
from backend.models.user import User, UserRole
from backend.schemas.query_schemas import (
    NOT_FOUND_ANSWER,
    QueryCreate,
    QueryResponse,
    SourceReference,
)
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.services.bedrock_client import BedrockUnavailable
from backend.services.query_service import (
    AUDIT_PHASE_ANSWER,
    AUDIT_PHASE_SUBMIT,
    MAX_QUESTION_LENGTH,
    MIN_QUESTION_LENGTH,
    QueryService,
)
from backend.services.rag_service import RagService
from backend.tests.fakes import (
    Corpus,
    FakeAuditRepository,
    FakeBedrockClient,
    FakeDocumentRepository,
    FakePineconeIndex,
    FakeQueryRepository,
)
from backend.utils.ids import new_id, utc_now
from backend.utils.vector_store import VectorStore

CONTEXT = RequestContext(ip_address="203.0.113.10", user_agent="pytest")

QUESTION = "What is the wire transfer fee for outbound payments?"

WIRE_FEE = "The outbound wire transfer fee is 25 USD for every retail payment."
DORMANCY = "An account is classified dormant after 24 months without customer activity."


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
def analyst(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.STANDARD_USER, "analyst@bank.example")


@pytest.fixture
def colleague(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.STANDARD_USER, "colleague@bank.example")


@pytest.fixture
def admin(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.ADMINISTRATOR, "admin@bank.example")


@pytest.fixture
def documents() -> FakeDocumentRepository:
    return FakeDocumentRepository()


@pytest.fixture
def queries(documents: FakeDocumentRepository) -> FakeQueryRepository:
    """Wired to the document fake so `load_chunks` can perform the citability re-check.

    Passing `documents` even to the tests that never retrieve anything keeps one wiring in the
    file. An unwired `FakeQueryRepository` returns no chunks at all, which would make a
    grounding test pass for the wrong reason.
    """
    return FakeQueryRepository(documents)


@pytest.fixture
def audit_repo() -> FakeAuditRepository:
    return FakeAuditRepository()


@pytest.fixture
def service(
    queries: FakeQueryRepository, audit_repo: FakeAuditRepository
) -> QueryService:
    """No pipeline injected — `submit` records the question and stops.

    That is a real deployment shape, not a test shortcut: without Bedrock access the service
    degrades to recording questions rather than erroring on every one of them.
    """
    return QueryService(queries, AuditService(audit_repo))


# -- The answering wiring ----------------------------------------------------
@pytest.fixture
def index() -> FakePineconeIndex:
    return FakePineconeIndex()


@pytest.fixture
def bedrock() -> FakeBedrockClient:
    return FakeBedrockClient()


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def vectors(index: FakePineconeIndex, settings: Settings) -> VectorStore:
    return VectorStore(index=index, settings=settings)


@pytest.fixture
def corpus(
    documents: FakeDocumentRepository, vectors: VectorStore, bedrock: FakeBedrockClient
) -> Corpus:
    return Corpus(documents, vectors, bedrock)


@pytest.fixture
def answering(
    queries: FakeQueryRepository,
    audit_repo: FakeAuditRepository,
    bedrock: FakeBedrockClient,
    vectors: VectorStore,
    settings: Settings,
) -> QueryService:
    """The same service with the *real* `RagService` behind it.

    Only Bedrock and Pinecone are faked. `RagService`, `VectorStore` and `source_tracer` are
    the real classes, because the properties under test here — a soft-deleted document dropping
    out of retrieval, an unresolvable citation downgrading the answer — live in that code and
    not in any stub of it.
    """
    return QueryService(
        queries, AuditService(audit_repo), RagService(queries, bedrock, vectors, settings)
    )


# -- Submission --------------------------------------------------------------
class TestSubmit:
    async def test_a_question_is_recorded_as_pending_before_it_is_answered(
        self, service: QueryService, queries: FakeQueryRepository, analyst: User
    ) -> None:
        """The row exists before any answering happens.

        A query written only on success would hide the timeouts, failures and
        refused-for-lack-of-grounding cases — exactly the ones a compliance review wants
        (BR-04, BR-05).
        """
        query = await service.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        assert query.status is QueryStatus.PENDING
        assert query.question_text == QUESTION
        assert query.answer_text is None
        assert query.source_references == []
        assert queries.queries[query.id] is query

    async def test_tenant_and_user_come_from_the_actor_not_the_request(
        self, service: QueryService, analyst: User
    ) -> None:
        """A caller who could name their own tenant could file — and later read — elsewhere."""
        query = await service.submit(question=QUESTION, actor=analyst, context=CONTEXT)
        assert query.tenant_id == analyst.tenant_id
        assert query.user_id == analyst.id

    async def test_the_audit_row_records_the_length_and_never_the_question(
        self, service: QueryService, audit_repo: FakeAuditRepository, analyst: User
    ) -> None:
        """A question can name a customer or an account number.

        It is stored once, in the `query` row the audit entry's `object_id` points at. Copying
        it into the log would put PII somewhere with different retention and wider read access.
        """
        sensitive = "What is the balance of account 4417829930 held by J. Okafor?"
        query = await service.submit(question=sensitive, actor=analyst, context=CONTEXT)

        entry = audit_repo.entries[-1]
        assert entry.action is AuditAction.QUERY_SUBMIT
        assert entry.outcome is AuditOutcome.SUCCESS
        assert entry.object_type == AuditObjectType.QUERY.value
        assert entry.object_id == str(query.id)
        assert entry.detail == {
            # `phase` is how the two entries a single query writes are told apart, both being
            # `QUERY_SUBMIT` — see `AUDIT_PHASE_SUBMIT` for why there is no `QUERY_ANSWER`.
            "phase": AUDIT_PHASE_SUBMIT,
            "question_length": len(sensitive),
            "status": "PENDING",
        }
        assert "4417829930" not in repr(entry.detail)
        assert "Okafor" not in repr(entry.detail)

    @pytest.mark.parametrize("question", ["", "  ", "\n\t", "a", "ab", "  x  "])
    async def test_a_question_with_nothing_to_retrieve_against_is_refused(
        self, service: QueryService, analyst: User, question: str
    ) -> None:
        with pytest.raises(ValidationError):
            await service.submit(question=question, actor=analyst, context=CONTEXT)

    async def test_a_question_at_the_minimum_length_is_accepted(
        self, service: QueryService, analyst: User
    ) -> None:
        query = await service.submit(
            question="x" * MIN_QUESTION_LENGTH, actor=analyst, context=CONTEXT
        )
        assert query.status is QueryStatus.PENDING

    async def test_an_over_long_question_is_refused_by_the_service_not_only_the_schema(
        self, service: QueryService, analyst: User
    ) -> None:
        """A bound that exists only in a DTO stops existing the moment another caller appears.

        Phase 4's Celery task will call `submit` without passing through the route, so the
        length check has to be a service invariant too.
        """
        with pytest.raises(ValidationError):
            await service.submit(
                question="x" * (MAX_QUESTION_LENGTH + 1), actor=analyst, context=CONTEXT
            )

    async def test_control_characters_are_stripped_rather_than_rejected(
        self, service: QueryService, analyst: User
    ) -> None:
        """A question pasted from a PDF carries NULs and form feeds.

        NUL aborts the INSERT outright, so it must go; rejecting the whole question would
        punish the user for their clipboard. Newlines and tabs survive — they are structure.
        """
        query = await service.submit(
            question="What is the\x00 fee\x0c for\twires?\nSecond line.",
            actor=analyst,
            context=CONTEXT,
        )
        assert "\x00" not in query.question_text
        assert "\x0c" not in query.question_text
        assert "\t" in query.question_text
        assert "\n" in query.question_text

    async def test_a_log_injection_attempt_in_a_question_cannot_forge_an_audit_row(
        self, service: QueryService, audit_repo: FakeAuditRepository, analyst: User
    ) -> None:
        """CR/LF sanitization, checked end to end.

        The question text does not reach the audit row at all here, but the actor's email does,
        and `sanitize_log_value` is what stops either from opening a second log line.
        """
        crafted = _user(
            analyst.tenant_id,
            UserRole.STANDARD_USER,
            "attacker@bank.example\nQUERY_SUBMIT SUCCESS forged",
        )
        await service.submit(question=QUESTION, actor=crafted, context=CONTEXT)
        entry = audit_repo.entries[-1]
        assert "\n" not in entry.actor_email
        assert "\r" not in entry.actor_email


# -- Answering ---------------------------------------------------------------
class TestAnswerPath:
    """The five outcomes a query can have, each checked on what was *recorded*."""

    async def test_a_grounded_answer_is_attached_to_the_row_with_its_sources(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """Answer, citations, confidence and latency all land on the one row.

        `attach_answer` writes them in a single statement, so this also asserts the invariant
        that matters for BR-03: there is no reachable state where the answer is visible and its
        citations are not.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("The outbound wire transfer fee is 25 USD [S1].")

        query = await answering.submit(
            question="What is the outbound wire transfer fee?",
            actor=analyst,
            context=CONTEXT,
        )

        assert query.status is QueryStatus.ANSWERED
        assert "25 USD" in (query.answer_text or "")
        assert query.source_references[0]["source_document_name"] == "retail-fees.pdf"
        assert query.source_references[0]["section_or_page"] == "page 1"
        assert query.confidence_score is not None
        assert query.latency_ms is not None
        assert query.answered_at is not None

    async def test_the_caller_receives_the_answered_row_not_the_pending_one(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """`attach_answer` issues an UPDATE, so the object `create` returned is stale.

        Returning that object would send the user a `PENDING` row with no answer for a query
        that had in fact been answered — and the UI would show a spinner that never resolves.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("Fee is 25 USD [S1].")

        query = await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)
        assert query.answer_text is not None

    async def test_the_answer_audit_row_logs_the_citations_and_not_the_passage(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        audit_repo: FakeAuditRepository,
        analyst: User,
    ) -> None:
        """Both halves of the compliance requirement, and the line between them.

        `document-source-traceability.md` requires the source references in the audit trail, so
        the document id and the location are there. The passage text is not: it is customer
        content, it already lives in `document_chunk`, and the audit log is append-only — a
        passage written here can never be redacted.
        """
        document = await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("Fee is 25 USD [S1].")

        query = await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        entry = audit_repo.entries[-1]
        assert entry.action is AuditAction.QUERY_SUBMIT
        assert entry.outcome is AuditOutcome.SUCCESS
        assert entry.object_id == str(query.id)
        assert entry.detail is not None
        assert entry.detail["phase"] == AUDIT_PHASE_ANSWER
        assert entry.detail["status"] == QueryStatus.ANSWERED.value
        assert entry.detail["sources"] == [
            {"document_id": str(document.id), "section_or_page": "page 1"}
        ]
        assert WIRE_FEE not in repr(entry.detail)
        assert "excerpt" not in repr(entry.detail)

    async def test_one_query_writes_two_audit_rows_told_apart_by_phase(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        audit_repo: FakeAuditRepository,
        analyst: User,
    ) -> None:
        """Submission and answering are separate facts with separate timestamps.

        Both are `QUERY_SUBMIT` because `audit_log.action` is a native PostgreSQL enum and a
        `QUERY_ANSWER` member needs `ALTER TYPE` in a migration — see `AUDIT_PHASE_SUBMIT`. The
        phase in `detail` is what a reviewer filters on until that decision is taken.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("Fee is 25 USD [S1].")

        await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        phases = [(entry.detail or {}).get("phase") for entry in audit_repo.entries]
        assert phases == [AUDIT_PHASE_SUBMIT, AUDIT_PHASE_ANSWER]
        assert {entry.action for entry in audit_repo.entries} == {
            AuditAction.QUERY_SUBMIT
        }

    async def test_an_empty_corpus_is_recorded_as_not_found_and_never_reaches_the_model(
        self,
        answering: QueryService,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """The empty-retrieval path. The assertion is the *absence* of the generate call.

        A model given no passages answers from its training data, which
        `rag-answer-grounding.md` prohibits outright. Asserting only on the returned text would
        pass even if we had called Bedrock and then thrown the answer away — which would still
        have sent the question out and produced content we must never use.
        """
        query = await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        assert query.status is QueryStatus.NOT_FOUND
        assert query.answer_text == NOT_FOUND_ANSWER
        assert query.source_references == []
        assert query.confidence_score is None
        assert bedrock.generate_calls == []

    async def test_a_below_threshold_match_is_not_found_with_retrieval_recorded(
        self,
        queries: FakeQueryRepository,
        audit_repo: FakeAuditRepository,
        bedrock: FakeBedrockClient,
        vectors: VectorStore,
        corpus: Corpus,
        settings: Settings,
        analyst: User,
    ) -> None:
        """The below-threshold path, driven by the threshold rather than by unrelated text.

        `candidates` in the audit detail is how an operator later distinguishes "the corpus has
        nothing on this" from "retrieval found passages and none were close enough". The second
        is a tuning problem and the first is a content gap, and they have different fixes.
        """
        strict = settings.model_copy(update={"RAG_MIN_SIMILARITY": 1.0})
        service = QueryService(
            queries, AuditService(audit_repo), RagService(queries, bedrock, vectors, strict)
        )
        await corpus.add(analyst.tenant_id, "policies.pdf", [DORMANCY])

        query = await service.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        assert query.status is QueryStatus.NOT_FOUND
        assert bedrock.generate_calls == []
        assert (audit_repo.entries[-1].detail or {})["candidates"] == 1

    async def test_a_citation_that_resolves_to_nothing_downgrades_the_answer(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """The unresolvable-citation path: `[S9]` when one passage was retrieved.

        The marker is a manufactured citation, so it is dropped — and because it was the only
        one, the answer is left uncited and must be replaced. Recording the prose with an empty
        reference list would publish an unattributable claim under the bank's name; recording
        it with the retrieved passage attached would be worse, because the citation would be a
        guess presented as provenance.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("The outbound wire transfer fee is 25 USD [S9].")

        query = await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        assert query.status is QueryStatus.NOT_FOUND
        assert query.answer_text == NOT_FOUND_ANSWER
        assert query.source_references == []

    async def test_a_soft_deleted_document_cannot_be_cited_while_still_indexed(
        self,
        answering: QueryService,
        corpus: Corpus,
        documents: FakeDocumentRepository,
        index: FakePineconeIndex,
        bedrock: FakeBedrockClient,
        settings: Settings,
        analyst: User,
    ) -> None:
        """The match-points-at-a-soft-deleted-document path, end to end through the service.

        The vectors are deliberately left in the index: that models the seconds between the
        status change landing in PostgreSQL and the Pinecone delete completing. PostgreSQL is
        the authority, so the document stops being quotable at the status change — not when the
        index catches up. This is what `PINECONE_OVERFETCH_FACTOR` pays for.
        """
        document = await corpus.add(analyst.tenant_id, "withdrawn.pdf", [WIRE_FEE])
        namespace = settings.pinecone_namespace(analyst.tenant_id)
        await documents.set_status(document.id, analyst.tenant_id, DocumentStatus.DELETED)

        query = await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        assert index.count(namespace) == 1  # still indexed...
        assert query.status is QueryStatus.NOT_FOUND  # ...and still not quotable
        assert bedrock.generate_calls == []

    async def test_an_outage_records_failed_and_re_raises_rather_than_answering(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        audit_repo: FakeAuditRepository,
        queries: FakeQueryRepository,
        analyst: User,
    ) -> None:
        """A failure downgrades the answer; it never loses the question.

        `FAILED` and `NOT_FOUND` are different statements. "Not in the uploaded documents" said
        of a question the documents *do* answer is a false claim about the corpus, so the
        exception propagates to a 503 the user can retry — question still on the record.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.fail_with = BedrockUnavailable()

        with pytest.raises(RetrievalUnavailable):
            await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        stored = next(iter(queries.queries.values()))
        assert stored.status is QueryStatus.FAILED
        assert stored.question_text == QUESTION
        # No answer text on a failure. "Bedrock unavailable" stored here would later be read
        # back as the answer the system gave, and a reviewer would find an error string in an
        # answer field.
        assert stored.answer_text is None
        assert audit_repo.entries[-1].outcome is AuditOutcome.FAILURE

    async def test_a_failure_audits_the_exception_type_and_never_its_message(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        audit_repo: FakeAuditRepository,
        analyst: User,
    ) -> None:
        """A botocore message carries the AWS account id and the model ARN.

        Both are infrastructure detail that `secure-coding-vapt.md` keeps out of logs, and the
        audit log is append-only — anything written here is permanent. The type is enough to
        triage; the message is looked up in the application log, which does get redacted.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.fail_with = BedrockUnavailable(
            "account 286134054656 model arn:aws:bedrock:us-east-1::foundation-model/nova"
        )

        with pytest.raises(RetrievalUnavailable):
            await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        detail = audit_repo.entries[-1].detail or {}
        assert detail["error_type"] == "BedrockUnavailable"
        assert detail["phase"] == AUDIT_PHASE_ANSWER
        assert "286134054656" not in repr(detail)
        assert "arn:aws" not in repr(detail)

    async def test_retrieval_is_scoped_to_the_actors_tenant_not_a_request_field(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """Tenant isolation in the answer is the same fact as tenant isolation in the record.

        Another tenant's passage is a *better* match for the question than anything this tenant
        holds, so a leak would surface as a confident, correct-looking answer rather than as a
        marginally worse one.
        """
        await corpus.add(analyst.tenant_id, "our-policies.pdf", [DORMANCY])
        await corpus.add(uuid.uuid4(), "their-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("Fee is 25 USD [S1].")

        query = await answering.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        assert query.status is QueryStatus.NOT_FOUND
        assert bedrock.generate_calls == []


# -- Reading a query ---------------------------------------------------------
class TestGetQuery:
    async def test_a_user_reads_their_own_query(
        self, service: QueryService, analyst: User
    ) -> None:
        query = await service.submit(question=QUESTION, actor=analyst, context=CONTEXT)
        assert (
            await service.get_query(query.id, actor=analyst, context=CONTEXT)
        ).id == query.id

    async def test_another_users_query_is_a_404_and_a_denied_audit_row(
        self,
        service: QueryService,
        audit_repo: FakeAuditRepository,
        analyst: User,
        colleague: User,
    ) -> None:
        """Distinguishing "not yours" from "does not exist" would confirm the id exists.

        That is enough to enumerate a colleague's activity, so both answers are the same 404 —
        and the attempt is recorded (`.claude/rules/rbac-enforcement.md`).
        """
        query = await service.submit(question=QUESTION, actor=analyst, context=CONTEXT)

        with pytest.raises(QueryNotFound):
            await service.get_query(query.id, actor=colleague, context=CONTEXT)

        denied = audit_repo.entries[-1]
        assert denied.action is AuditAction.HISTORY_VIEW
        assert denied.outcome is AuditOutcome.DENIED
        assert denied.detail == {"reason": "not_owner"}
        assert denied.actor_email == colleague.email

    async def test_an_administrator_reads_any_query_in_their_own_tenant(
        self, service: QueryService, analyst: User, admin: User
    ) -> None:
        query = await service.submit(question=QUESTION, actor=analyst, context=CONTEXT)
        found = await service.get_query(query.id, actor=admin, context=CONTEXT)
        assert found.id == query.id

    async def test_an_administrator_of_another_tenant_gets_nothing(
        self, service: QueryService, audit_repo: FakeAuditRepository, analyst: User
    ) -> None:
        """Role and tenant are independent checks and both apply on every request."""
        query = await service.submit(question=QUESTION, actor=analyst, context=CONTEXT)
        outsider = _user(uuid.uuid4(), UserRole.ADMINISTRATOR, "admin@other.example")

        with pytest.raises(QueryNotFound):
            await service.get_query(query.id, actor=outsider, context=CONTEXT)
        assert audit_repo.entries[-1].detail == {"reason": "not_found_or_other_tenant"}

    async def test_an_unknown_id_is_audited_too(
        self, service: QueryService, audit_repo: FakeAuditRepository, analyst: User
    ) -> None:
        with pytest.raises(QueryNotFound):
            await service.get_query(new_id(), actor=analyst, context=CONTEXT)
        assert audit_repo.entries[-1].outcome is AuditOutcome.DENIED

    async def test_the_service_offers_no_way_to_list_another_users_history(self) -> None:
        """History listing deliberately lives in `search_history_service` (Phase 4).

        Two implementations of one authorization decision are free to diverge, and the one
        that diverges is the one nobody reviewed. Asserted so a "convenience" listing added
        here fails a test rather than quietly duplicating the rule.
        """
        public = {name for name in vars(QueryService) if not name.startswith("_")}
        assert public == {"submit", "get_query"}


# -- BR-03 in the response DTO ----------------------------------------------
class TestAnswerGroundingSchema:
    def _query(self, **overrides: object) -> Query:
        base: dict[str, object] = {
            "id": new_id(),
            "tenant_id": uuid.uuid4(),
            "user_id": uuid.uuid4(),
            "question_text": QUESTION,
            "answer_text": None,
            "source_references": [],
            "confidence_score": None,
            "status": QueryStatus.PENDING,
            "latency_ms": None,
            "answered_at": None,
            "created_at": utc_now(),
        }
        base.update(overrides)
        return Query(**base)  # type: ignore[arg-type]

    def test_an_answered_query_without_sources_cannot_be_serialized(self) -> None:
        """BR-03 as a validator, not as a convention.

        An answer with no citations must not be representable — if the DTO could express it,
        some future code path would produce it and the API would return it.
        """
        with pytest.raises(PydanticValidationError) as raised:
            QueryResponse.from_entity(
                self._query(
                    status=QueryStatus.ANSWERED,
                    answer_text="The wire fee is 25 USD.",
                    source_references=[],
                )
            )
        assert "source reference" in str(raised.value)

    def test_an_answered_query_with_a_complete_reference_serializes(self) -> None:
        response = QueryResponse.from_entity(
            self._query(
                status=QueryStatus.ANSWERED,
                answer_text="The wire fee is 25 USD.",
                confidence_score=0.82,
                source_references=[
                    {
                        "source_document_name": "retail-fees.docx",
                        "section_or_page": "Fee Schedule",
                        "confidence_level": 0.82,
                    }
                ],
            )
        )
        assert response.source_references[0].source_document_name == "retail-fees.docx"
        assert response.source_references[0].section_or_page == "Fee Schedule"

    @pytest.mark.parametrize(
        "reference",
        [
            {"section_or_page": "Fee Schedule"},
            {"source_document_name": "retail-fees.docx"},
            {"source_document_name": "", "section_or_page": "Fee Schedule"},
            {"source_document_name": "retail-fees.docx", "section_or_page": ""},
        ],
    )
    def test_a_reference_missing_either_half_is_rejected(
        self, reference: dict[str, str]
    ) -> None:
        """Both fields are mandatory and neither may be blank.

        "Which document" without "where in it" is not a citation a compliance reviewer can
        act on, and the reverse is worse.
        """
        with pytest.raises(PydanticValidationError):
            SourceReference.model_validate(reference)

    def test_a_confidence_level_outside_zero_to_one_is_rejected(self) -> None:
        with pytest.raises(PydanticValidationError):
            SourceReference.model_validate(
                {
                    "source_document_name": "a.pdf",
                    "section_or_page": "Page 1",
                    "confidence_level": 1.4,
                }
            )

    def test_a_not_found_query_may_have_no_sources(self) -> None:
        """`NOT_FOUND` is the grounding rule working, not a failure.

        Retrieval found nothing above the threshold and the system refused to answer from
        model knowledge — so there is nothing to cite, and requiring a citation here would
        force the very fabrication the rule exists to prevent.
        """
        response = QueryResponse.from_entity(
            self._query(status=QueryStatus.NOT_FOUND, answer_text=NOT_FOUND_ANSWER)
        )
        assert response.source_references == []
        assert response.answer_text == NOT_FOUND_ANSWER

    def test_the_request_body_rejects_unexpected_fields(self) -> None:
        """Mass assignment: naming `user_id` or `tenant_id` in the body must not be accepted.

        `extra="forbid"` means an unknown field is a 422 rather than a silently ignored
        attempt (`.claude/rules/secure-coding-vapt.md`).
        """
        assert QueryCreate(question_text=QUESTION).question_text == QUESTION
        for body in [
            {"question_text": QUESTION, "tenant_id": str(uuid.uuid4())},
            {"question_text": QUESTION, "status": "ANSWERED"},
            # The one that matters most: a caller who could set `answer_text` could write a
            # fabricated answer into the record a compliance review reads.
            {"question_text": QUESTION, "answer_text": "The fee is zero."},
            {"question_text": "x"},
        ]:
            with pytest.raises(PydanticValidationError):
                QueryCreate.model_validate(body)


# -- Follow-up questions -----------------------------------------------------
class TestFollowUp:
    """A second question that only makes sense after the first.

    "more details about it" has no referent: on its own it embeds to nothing and retrieves
    nothing, so an answered question followed by a follow-up produced a not-found. These tests
    pin the fix *and* the limit on it — conversational memory must not become a second source
    of truth beside the documents.
    """

    async def test_a_follow_up_retrieves_using_the_question_it_follows(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """The referent comes from the previous question, not from the bare pronoun."""
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("The outbound wire transfer fee is 25 USD [S1].")
        first = await answering.submit(
            question="What is the outbound wire transfer fee?",
            actor=analyst,
            context=CONTEXT,
        )
        assert first.status is QueryStatus.ANSWERED

        bedrock.queue_answer("It is charged on every retail payment [S1].")
        second = await answering.submit(
            question="more details about it",
            actor=analyst,
            context=CONTEXT,
            follow_up_to=first.id,
        )

        assert second.status is QueryStatus.ANSWERED, (
            "a follow-up resolving to a covered topic should not be a not-found"
        )
        assert second.source_references
        # The text embedded for retrieval carried the earlier question, so the vector landed
        # near the same passages rather than near a two-word pronoun.
        embedded = bedrock.embed_calls[-1][0]
        assert "outbound wire transfer fee" in embedded
        assert "more details about it" in embedded

    async def test_the_previous_turn_reaches_the_model_as_a_conversation(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """Prior question and prior answer, each in its own turn.

        Not pasted into the new question: a transcript inside the user turn is text the
        grounding rules then have to tell apart from the question itself, which is the
        confusion rule 6 exists to prevent for passages.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("The fee is 25 USD [S1].")
        first = await answering.submit(
            question="What is the outbound wire transfer fee?",
            actor=analyst,
            context=CONTEXT,
        )

        bedrock.queue_answer("It applies to retail payments [S1].")
        await answering.submit(
            question="more details about it",
            actor=analyst,
            context=CONTEXT,
            follow_up_to=first.id,
        )

        messages = bedrock.generate_calls[-1]["messages"]
        assert [message["role"] for message in messages] == [
            "user",
            "assistant",
            "user",
        ]
        assert "outbound wire transfer fee" in messages[0]["content"][0]["text"]
        assert "25 USD" in messages[1]["content"][0]["text"]
        assert "more details about it" in messages[2]["content"][0]["text"]

    async def test_a_follow_up_the_documents_do_not_support_is_still_not_found(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """The property a conversational memory most easily breaks.

        With an earlier turn in the window a model can answer from its own previous reply and
        cite nothing. Grounding does not move for a follow-up: no marker into a retrieved
        passage means no answer, exactly as for a standalone question.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("The fee is 25 USD [S1].")
        first = await answering.submit(
            question="What is the outbound wire transfer fee?",
            actor=analyst,
            context=CONTEXT,
        )

        # The model answers confidently from the conversation, citing nothing.
        bedrock.queue_answer("It was introduced in 2019 and reviewed every March.")
        second = await answering.submit(
            question="when was it introduced",
            actor=analyst,
            context=CONTEXT,
            follow_up_to=first.id,
        )

        assert second.status is QueryStatus.NOT_FOUND
        assert second.source_references == []

    async def test_a_follow_up_to_another_users_query_is_ignored(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
        tenant_id: uuid.UUID,
    ) -> None:
        """`follow_up_to` arrives in a request body, so it is an IDOR surface.

        Without the ownership check a Standard User could name a colleague's query id and have
        its question and answer carried into the context of their own — reading another
        person's history, which `rbac-enforcement.md` reserves to an Administrator. The id is
        ignored rather than refused, so the question is simply answered standalone.
        """
        colleague = _user(tenant_id, UserRole.STANDARD_USER, "someone-else@bank.example")
        await corpus.add(tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("The fee is 25 USD [S1].")
        theirs = await answering.submit(
            question="What is the outbound wire transfer fee?",
            actor=colleague,
            context=CONTEXT,
        )

        bedrock.queue_answer("The fee is 25 USD [S1].")
        await answering.submit(
            question="more details about it",
            actor=analyst,
            context=CONTEXT,
            follow_up_to=theirs.id,
        )

        # No prior turn was used: the model saw one user turn and nothing else.
        assert [m["role"] for m in bedrock.generate_calls[-1]["messages"]] == ["user"]

    async def test_a_follow_up_to_a_not_found_turn_is_answered_standalone(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        analyst: User,
    ) -> None:
        """A pronoun after a refusal refers to nothing, so there is nothing to carry."""
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        # No answer is queued for this one on purpose: an off-topic question is refused at the
        # similarity threshold and never reaches the model, so a queued reply would sit in the
        # fake's queue and be served to the *next* call — which is what made this test fail
        # the first time, with the follow-up reading as "model_declined".
        refused = await answering.submit(
            question="what is the dividend policy",
            actor=analyst,
            context=CONTEXT,
        )
        assert refused.status is QueryStatus.NOT_FOUND
        assert bedrock.generate_calls == []

        bedrock.queue_answer("The outbound wire transfer fee is 25 USD [S1].")
        second = await answering.submit(
            question="What is the outbound wire transfer fee?",
            actor=analyst,
            context=CONTEXT,
            follow_up_to=refused.id,
        )

        assert second.status is QueryStatus.ANSWERED
        assert [m["role"] for m in bedrock.generate_calls[-1]["messages"]] == ["user"]

    async def test_the_audit_row_records_which_turn_was_followed(
        self,
        answering: QueryService,
        corpus: Corpus,
        bedrock: FakeBedrockClient,
        audit_repo: FakeAuditRepository,
        analyst: User,
    ) -> None:
        """An id, not text — and the only record of the thread.

        `query` has no parent column, so a thread would be a migration; JSONB `detail` carries
        the link without one, and a reviewer reconstructing a conversation can follow it.
        """
        await corpus.add(analyst.tenant_id, "retail-fees.pdf", [WIRE_FEE])
        bedrock.queue_answer("The fee is 25 USD [S1].")
        first = await answering.submit(
            question="What is the outbound wire transfer fee?",
            actor=analyst,
            context=CONTEXT,
        )

        bedrock.queue_answer("It applies to retail payments [S1].")
        await answering.submit(
            question="more details about it",
            actor=analyst,
            context=CONTEXT,
            follow_up_to=first.id,
        )

        submits = [
            e for e in audit_repo.entries if (e.detail or {}).get("phase") == "submit"
        ]
        assert submits[-1].detail is not None
        assert submits[-1].detail["follow_up_to"] == str(first.id)
        # The first question was not a follow-up, so it carries no link at all.
        assert "follow_up_to" not in (submits[0].detail or {})
