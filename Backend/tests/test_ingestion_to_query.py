"""backend/tests/test_ingestion_to_query.py

Role: The Phase 4 integration test — one PDF travels from an administrator's upload through
Celery-eager ingestion into a Standard User's answer, and the citation it comes back with is
checked against the chunk rows that were actually stored.

The gate item, verbatim from `WORKBREAKDOWN.md`: *"Integration test upload -> processing ->
query end to end with Celery eager, asserting the answer carries a resolvable source."*

What makes this different from the unit tests
---------------------------------------------
`test_document_service.py` ends where the document becomes `ACTIVE`; `test_rag_service.py` and
`test_query_service.py` begin with a corpus already staged by `Corpus`. Between them sits the
seam this file covers: the chunk rows, page numbers and vector ids that ingestion *writes* are
the ones retrieval *reads*. A page-numbering change in the splitter, a vector id that stopped
being `str(chunk.id)`, or metadata that stopped carrying `page` would leave both neighbouring
suites green and break every citation in production.

So nothing here is staged. The document is a real PDF built by `document_fixtures`, parsed by
the real parser, split by the real splitter, embedded, upserted through the real `VectorStore`,
and retrieved by the real `RagService`. Only the three boundaries the project does not own are
faked: S3, Bedrock and the Pinecone index.

One shared `FakeDocumentRepository`
-----------------------------------
`DocumentService` writes to it and `FakeQueryRepository` reads chunks through it, which is what
makes this an integration test rather than two unit tests in one file. If ingestion and
retrieval disagreed about what a chunk row looks like, that disagreement would surface here as
a missing or misattributed citation.

On the questions
----------------
They reuse the fixture pages' own word forms, for the reason `backend/tests/eval/` states at
length: `FakeBedrockClient.embed` is a hashed bag of words, so cosine here approximates word
overlap and has no morphology. That is a limitation of the *relevance* signal, not of the
plumbing under test — whether a retrieved chunk resolves to the right document and page is
independent of how it was scored.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from backend.core.config import get_settings
from backend.models.audit_log import AuditAction, AuditOutcome
from backend.models.document import DocumentStatus
from backend.models.query import QueryStatus
from backend.models.user import User, UserRole
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.services.document_service import DocumentService
from backend.services.query_service import AUDIT_PHASE_ANSWER, QueryService
from backend.services.rag_service import RagService
from backend.services.source_tracer import NOT_FOUND_ANSWER
from backend.services.storage_service import S3StorageService
from backend.tests.document_fixtures import make_pdf
from backend.tests.fakes import (
    FakeAuditRepository,
    FakeBedrockClient,
    FakeDocumentRepository,
    FakeIngestionQueue,
    FakePineconeIndex,
    FakeQueryRepository,
    FakeS3Client,
)
from backend.utils.document_parser import CONTENT_TYPE_PDF
from backend.utils.vector_store import VectorStore
from backend.workers.document_processor import AUDIT_PHASE_INGEST

CONTEXT = RequestContext(ip_address="203.0.113.10", user_agent="pytest")

#: Three pages that price three different events. The pages are deliberately about neighbouring
#: subjects — all three are fees — so a citation naming the right *document* and the wrong page
#: fails here. Citing a document is the weak claim; BR-03 asks for the location.
FEE_PAGES = [
    "The outbound domestic wire transfer fee is 25 USD for every retail payment "
    "instruction, debited from the originating account on the day of submission.",
    "The monthly account maintenance fee is 12 USD and is waived for any month in which "
    "the average daily balance stays above 5,000 USD.",
    "Cash withdrawals at an out-of-network ATM carry a 3 USD surcharge in addition to any "
    "amount the operator of that machine chooses to levy.",
]
FEE_PDF = make_pdf(FEE_PAGES)

WIRE_QUESTION = "What is the outbound domestic wire transfer fee for a payment instruction?"
#: Page 1 of the fixture. `make_pdf` numbers from 1 in list order, and the splitter propagates
#: that page onto every chunk it produces from it.
WIRE_PAGE = "page 1"


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
    """The Standard User who asks the question. Uploading is Administrator-only."""
    return _user(tenant_id, UserRole.STANDARD_USER, "analyst@bank.example")


@pytest.fixture
def documents() -> FakeDocumentRepository:
    """Written by ingestion, read by retrieval. One instance, deliberately — see the header."""
    return FakeDocumentRepository()


@pytest.fixture
def s3() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture
def storage(s3: FakeS3Client) -> S3StorageService:
    return S3StorageService(client=s3)


@pytest.fixture
def bedrock() -> FakeBedrockClient:
    return FakeBedrockClient(dimensions=get_settings().BEDROCK_EMBEDDING_DIMENSIONS)


@pytest.fixture
def index() -> FakePineconeIndex:
    return FakePineconeIndex()


@pytest.fixture
def vectors(index: FakePineconeIndex) -> VectorStore:
    return VectorStore(index=index, settings=get_settings())


@pytest.fixture
def audit_repo() -> FakeAuditRepository:
    """One trail for both halves, so the upload and the answer can be read as one story."""
    return FakeAuditRepository()


@pytest.fixture
def audit(audit_repo: FakeAuditRepository) -> AuditService:
    return AuditService(audit_repo)


@pytest.fixture
def queue(
    storage: S3StorageService, bedrock: FakeBedrockClient, vectors: VectorStore
) -> FakeIngestionQueue:
    """`CELERY_TASK_ALWAYS_EAGER` in one object: the pipeline runs in the calling coroutine.

    Substituting the queue rather than the Celery app is what keeps this test free of a broker.
    The pipeline it runs is `ingest_document_in_session` — the exact function the real eager
    branch calls, on the same repository the request is using.
    """
    return FakeIngestionQueue(
        storage=storage, bedrock=bedrock, vectors=vectors, settings=get_settings()
    )


@pytest.fixture
def uploads(
    documents: FakeDocumentRepository,
    storage: S3StorageService,
    audit: AuditService,
    queue: FakeIngestionQueue,
    vectors: VectorStore,
) -> DocumentService:
    return DocumentService(documents, storage, audit, queue, vectors)


@pytest.fixture
def queries(documents: FakeDocumentRepository) -> FakeQueryRepository:
    return FakeQueryRepository(documents)


@pytest.fixture
def asking(
    queries: FakeQueryRepository,
    audit: AuditService,
    bedrock: FakeBedrockClient,
    vectors: VectorStore,
) -> QueryService:
    """The full query path, with the real `RagService` at production settings."""
    return QueryService(
        queries, audit, RagService(queries, bedrock, vectors, get_settings())
    )


def _answer_details(audit_repo: FakeAuditRepository) -> list[dict[str, Any]]:
    return [
        entry.detail or {}
        for entry in audit_repo.entries
        if (entry.detail or {}).get("phase") == AUDIT_PHASE_ANSWER
    ]


class TestUploadThenQuery:
    """The whole path, in the order a user experiences it."""

    async def test_ingestion_publishes_the_document_with_embedded_chunks(
        self,
        uploads: DocumentService,
        documents: FakeDocumentRepository,
        index: FakePineconeIndex,
        admin: User,
    ) -> None:
        """The three facts that have to be simultaneously true before anything is answerable.

        `ACTIVE`, one vector per chunk row, and `embedded_at` set on every one. Any two without
        the third is a state `document_processor` is written to never leave behind: `ACTIVE`
        with no vectors answers everything with not-found, and vectors with no `ACTIVE` is
        invisible.
        """
        document = await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )

        assert document.status is DocumentStatus.ACTIVE
        chunks = await documents.get_chunks(document.id, admin.tenant_id)
        assert len(chunks) == len(FEE_PAGES)
        assert all(chunk.embedded_at is not None for chunk in chunks)

        namespace = get_settings().pinecone_namespace(admin.tenant_id)
        assert index.ids_in(namespace) == {str(chunk.id) for chunk in chunks}

    async def test_a_standard_user_gets_an_answer_citing_the_right_page(
        self,
        uploads: DocumentService,
        asking: QueryService,
        bedrock: FakeBedrockClient,
        admin: User,
        analyst: User,
    ) -> None:
        """The gate item itself: the answer carries a source, and the source resolves.

        "Resolvable" is asserted three ways — the document name is the uploaded one, the page
        is the page that contains the sentence, and the `document_id` is the row's own id. A
        citation missing any of those cannot be followed by a compliance reviewer, which is the
        only thing `section_or_page` is for.
        """
        document = await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )
        bedrock.queue_answer("The outbound domestic wire transfer fee is 25 USD [S1].")

        query = await asking.submit(
            question=WIRE_QUESTION, actor=analyst, context=CONTEXT
        )

        assert query.status is QueryStatus.ANSWERED
        assert query.source_references, "An answer with no source violates BR-03."
        first = query.source_references[0]
        assert first["source_document_name"] == document.name
        assert first["section_or_page"] == WIRE_PAGE
        assert first["document_id"] == str(document.id)

    async def test_the_cited_page_exists_in_the_stored_chunks(
        self,
        uploads: DocumentService,
        asking: QueryService,
        documents: FakeDocumentRepository,
        bedrock: FakeBedrockClient,
        admin: User,
        analyst: User,
    ) -> None:
        """Every citation is checked against the rows ingestion wrote, not against a constant.

        This is the assertion that catches the seam: a splitter that renumbered pages, or a
        tracer that formatted the location differently from the way the chunk records it, still
        produces a plausible-looking citation. Comparing it to the actual `page_number` on the
        actual chunk row is what makes "resolvable" mean something.
        """
        document = await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )
        bedrock.queue_answer("Fees are set out in the schedule [S1][S2].")

        query = await asking.submit(
            question=WIRE_QUESTION, actor=analyst, context=CONTEXT
        )

        chunks = await documents.get_chunks(document.id, admin.tenant_id)
        real_pages = {f"page {chunk.page_number}" for chunk in chunks}
        assert query.source_references
        for reference in query.source_references:
            assert reference["section_or_page"] in real_pages, reference

    async def test_the_query_row_records_the_latency_and_the_sources_together(
        self,
        uploads: DocumentService,
        asking: QueryService,
        bedrock: FakeBedrockClient,
        admin: User,
        analyst: User,
    ) -> None:
        """`attach_answer` writes answer, sources, confidence and status in one statement.

        Asserted end to end because the invariant is about a *reachable state*: there must be
        no moment where a user can read an answer whose citations have not been stored with it.
        """
        await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )
        bedrock.queue_answer("The fee is 25 USD [S1].")

        query = await asking.submit(
            question=WIRE_QUESTION, actor=analyst, context=CONTEXT
        )

        assert query.answer_text
        assert query.answered_at is not None
        assert query.latency_ms is not None and query.latency_ms >= 0
        assert query.confidence_score is not None

    async def test_the_audit_trail_holds_the_ingestion_and_the_answer(
        self,
        uploads: DocumentService,
        asking: QueryService,
        audit_repo: FakeAuditRepository,
        bedrock: FakeBedrockClient,
        admin: User,
        analyst: User,
    ) -> None:
        """One trail, both halves, and the answer's row carries its source references.

        `audit-log-immutability.md` wants every critical activity recorded, and
        `document-source-traceability.md` wants the source references among them. Checked here
        rather than only in the unit tests because a reviewer reconstructing "where did this
        answer come from" reads these rows in sequence, and the sequence only exists once both
        services share a trail.
        """
        await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )
        bedrock.queue_answer("The fee is 25 USD [S1].")
        await asking.submit(question=WIRE_QUESTION, actor=analyst, context=CONTEXT)

        phases = [(entry.detail or {}).get("phase") for entry in audit_repo.entries]
        assert AUDIT_PHASE_INGEST in phases
        assert AUDIT_PHASE_ANSWER in phases
        assert AuditAction.QUERY_SUBMIT in [entry.action for entry in audit_repo.entries]

        answered = _answer_details(audit_repo)[0]
        assert answered["status"] == QueryStatus.ANSWERED.value
        assert answered["sources"], "The answer's audit row must carry its source references."
        assert answered["sources"][0]["section_or_page"] == WIRE_PAGE

    async def test_no_audit_detail_repeats_the_document_text(
        self,
        uploads: DocumentService,
        asking: QueryService,
        audit_repo: FakeAuditRepository,
        bedrock: FakeBedrockClient,
        admin: User,
        analyst: User,
    ) -> None:
        """The passage travelled through parse, chunk, embed, retrieve and generate.

        Five places it could have been copied into a log line. `secure-coding-vapt.md` keeps
        customer document content out of the audit trail, which is append-only and therefore
        not redactable — so this is checked once, at the end, over everything that was written.
        """
        await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )
        bedrock.queue_answer("The fee is 25 USD [S1].")
        await asking.submit(question=WIRE_QUESTION, actor=analyst, context=CONTEXT)

        written = repr([entry.detail for entry in audit_repo.entries])
        for page in FEE_PAGES:
            assert page[:60] not in written
        # Nor the question, nor the filename: both are user-supplied text that can name a
        # customer. The filename is recorded on the `document` row, the question on `query`.
        assert WIRE_QUESTION not in written
        assert "retail-account-fees" not in written


class TestGroundingSurvivesTheRoundTrip:
    """The refusals, asserted against a document that genuinely exists."""

    async def test_a_question_the_document_does_not_answer_is_declined(
        self,
        uploads: DocumentService,
        asking: QueryService,
        admin: User,
        analyst: User,
    ) -> None:
        """An `ACTIVE`, embedded, retrievable document — and still no answer.

        The strongest form of the not-found requirement: retrieval had something to return and
        the pipeline declined anyway, rather than the trivial case of an empty index. No answer
        queued, so a model that was called at all would raise instead of inventing.
        """
        await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )

        query = await asking.submit(
            question="photosynthesis chloroplast thylakoid membrane grana stroma",
            actor=analyst,
            context=CONTEXT,
        )

        assert query.status is QueryStatus.NOT_FOUND
        assert query.answer_text == NOT_FOUND_ANSWER
        assert query.source_references == []

    async def test_a_deleted_document_stops_being_quotable_at_once(
        self,
        uploads: DocumentService,
        asking: QueryService,
        bedrock: FakeBedrockClient,
        admin: User,
        analyst: User,
    ) -> None:
        """Answerable, then deleted, then not answerable — the same question both times.

        PostgreSQL is the authority on `status`, so the second question must be declined even
        though this exercise cannot prove the vectors were gone: `soft_delete` removes them, so
        what is really asserted is that deletion closes the window. The unit tests in
        `test_rag_service.py` cover the narrower case where the vector is deliberately left
        behind; this one covers the operation an administrator actually performs.
        """
        document = await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )
        bedrock.queue_answer("The fee is 25 USD [S1].")
        answered = await asking.submit(
            question=WIRE_QUESTION, actor=analyst, context=CONTEXT
        )
        assert answered.status is QueryStatus.ANSWERED

        await uploads.soft_delete(document.id, actor=admin, context=CONTEXT)

        after = await asking.submit(
            question=WIRE_QUESTION, actor=analyst, context=CONTEXT
        )
        assert after.status is QueryStatus.NOT_FOUND
        assert after.source_references == []

    async def test_another_tenant_cannot_reach_the_uploaded_document(
        self,
        uploads: DocumentService,
        asking: QueryService,
        admin: User,
    ) -> None:
        """The same question, from a user whose tenant uploaded nothing.

        Isolation is the Pinecone namespace and the PostgreSQL re-check, both keyed on the
        authenticated user's `tenant_id` — never on anything in the request. An outsider gets a
        not-found, not a partial answer.
        """
        await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )
        outsider = _user(uuid.uuid4(), UserRole.STANDARD_USER, "user@other-bank.example")

        query = await asking.submit(
            question=WIRE_QUESTION, actor=outsider, context=CONTEXT
        )

        assert query.status is QueryStatus.NOT_FOUND
        assert query.source_references == []


class TestTheUploadRequestDoesNotWait:
    async def test_upload_returns_processing_when_the_task_is_dispatched(
        self,
        documents: FakeDocumentRepository,
        storage: S3StorageService,
        audit: AuditService,
        vectors: VectorStore,
        admin: User,
    ) -> None:
        """With a queue that dispatches instead of running, the response is `PROCESSING`.

        The gate wants the upload request to return without waiting for embedding. Every other
        test in this file uses the eager queue, which finishes the work inside the call — so
        without this one, "does not block" would be untested and the `PROCESSING` status would
        appear to be dead code.
        """
        queue = FakeIngestionQueue(autorun=False)
        service = DocumentService(documents, storage, audit, queue, vectors)

        document = await service.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )

        assert document.status is DocumentStatus.PROCESSING
        assert queue.calls == [(document.id, admin.tenant_id)]
        # Nothing is citable yet, and the chunk rows do not exist either.
        assert await documents.get_chunks(document.id, admin.tenant_id) == []

    async def test_the_document_only_becomes_active_once_the_pipeline_runs(
        self,
        documents: FakeDocumentRepository,
        storage: S3StorageService,
        audit: AuditService,
        bedrock: FakeBedrockClient,
        vectors: VectorStore,
        admin: User,
    ) -> None:
        """The deferred half of the previous test: dispatch, then run, then `ACTIVE`.

        Runs the pipeline through the same `enqueue` entry point rather than calling
        `ingest_document_in_session` directly, so what is exercised is the transition an
        administrator watching the list actually sees.
        """
        queue = FakeIngestionQueue(autorun=False)
        service = DocumentService(documents, storage, audit, queue, vectors)
        document = await service.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )

        queue.autorun = True
        queue.storage, queue.bedrock, queue.vectors = storage, bedrock, vectors
        queue.settings = get_settings()
        await queue.enqueue(
            document.id, admin.tenant_id, documents=documents, audit=audit
        )

        refreshed = await documents.get(document.id, admin.tenant_id)
        assert refreshed is not None
        assert refreshed.status is DocumentStatus.ACTIVE
        assert len(await documents.get_chunks(document.id, admin.tenant_id)) == len(
            FEE_PAGES
        )


class TestIngestionFailureIsVisible:
    async def test_a_bedrock_outage_leaves_a_failed_document_and_no_vectors(
        self,
        uploads: DocumentService,
        documents: FakeDocumentRepository,
        bedrock: FakeBedrockClient,
        index: FakePineconeIndex,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        """Embedding fails mid-pipeline. The upload still succeeds; the document does not.

        Two properties, and the second is the compliance one. The request must not fail — the
        bytes are already in S3, and failing it makes the administrator re-upload and create a
        v2 they did not want. And the outcome must be *visible*: `FAILED` in the list, with an
        audit row saying which step failed, rather than a document stuck at `PROCESSING` with
        nothing explaining why it never became searchable.
        """

        async def _unavailable(_texts: list[str]) -> list[list[float]]:
            from backend.services.bedrock_client import BedrockUnavailable

            raise BedrockUnavailable("Simulated throttling.")

        bedrock.embed = _unavailable  # type: ignore[method-assign]

        document = await uploads.upload(
            actor=admin,
            context=CONTEXT,
            filename="retail-account-fees.pdf",
            data=FEE_PDF,
            declared_content_type=CONTENT_TYPE_PDF,
        )

        assert document.status is DocumentStatus.FAILED
        assert index.total() == 0
        failures = [
            entry.detail or {}
            for entry in audit_repo.entries
            if entry.outcome == AuditOutcome.FAILURE
        ]
        assert failures and failures[0]["reason"] == "embedding_failed"
        # The exception message never reaches the trail — see `_failure_detail`.
        assert "Simulated throttling" not in repr(failures)
