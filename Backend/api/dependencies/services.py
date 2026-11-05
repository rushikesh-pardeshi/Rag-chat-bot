"""backend/api/dependencies/services.py

Role: Construct the document, query and analytics services for a request, and nothing else.

Separate from `dependencies/auth.py` because that file's job is identity and role — it
resolves the caller and refuses them. Domain-service wiring alongside it would mean every
route that needs a document also imports the authorization boundary's module, and the two
would slowly merge.

`get_storage_service` returns the real S3 client. Tests override this provider with a fake;
no test in this project reaches AWS (`app.dependency_overrides[get_storage]`).

The same pattern extends to the RAG collaborators. `get_bedrock` and `get_vectors` are thin
providers over the process-wide singletons, and they exist to be overridden — a test replaces
them with `FakeBedrockClient` and a `VectorStore` holding a `FakePineconeIndex`. That is what
makes `pytest -q` pass offline with `PINECONE_API_KEY` blank and no AWS credentials.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from backend.api.dependencies.auth import get_audit_repository, get_audit_service
from backend.api.dependencies.database import DbSession
from backend.repositories.analytics_repository import AnalyticsRepository
from backend.repositories.audit_repository import AuditRepository
from backend.repositories.document_repository import DocumentRepository
from backend.repositories.query_repository import QueryRepository
from backend.services.analytics_service import AnalyticsService
from backend.services.audit_service import AuditService
from backend.services.bedrock_client import BedrockClient, get_bedrock_client
from backend.services.document_service import DocumentService
from backend.services.query_service import QueryService
from backend.services.rag_service import RagService
from backend.services.search_history_service import SearchHistoryService
from backend.services.storage_service import (
    S3StorageService,
    StorageBackend,
    get_storage_service,
)
from backend.utils.vector_store import VectorStore, get_vector_store
from backend.workers.document_processor import IngestionQueue, get_ingestion_queue


def get_document_repository(session: DbSession) -> DocumentRepository:
    return DocumentRepository(session)


def get_query_repository(session: DbSession) -> QueryRepository:
    return QueryRepository(session)


def get_analytics_repository(session: DbSession) -> AnalyticsRepository:
    return AnalyticsRepository(session)


def get_storage() -> S3StorageService:
    """Indirection that exists to be overridden.

    The route depends on this function, not on `S3StorageService`, which is what lets a test
    substitute a fake without patching a module global.
    """
    return get_storage_service()


def get_queue() -> IngestionQueue:
    """The ingestion queue. Overridden in tests with one that records calls.

    This is the only place the API layer touches `backend/workers/`. `DocumentService` takes
    the queue as a constructor argument and never imports it, so importing the service does not
    pull in a Celery application or a broker URL.
    """
    return get_ingestion_queue()


def get_bedrock() -> BedrockClient:
    """The process-wide Bedrock client. Overridden in tests.

    Returns the shared instance rather than constructing one: a boto3 client costs a
    credential resolution and a session build, and paying that per request is measurable
    against the 3–5 second query target.
    """
    return get_bedrock_client()


def get_vectors() -> VectorStore:
    """The process-wide Pinecone store. Overridden in tests.

    Shared because it owns a connection pool that the app's lifespan closes on shutdown
    (`dispose_vector_store`). A per-request store would leak one pool per request.
    """
    return get_vector_store()


def get_document_service(
    documents: Annotated[DocumentRepository, Depends(get_document_repository)],
    storage: Annotated[StorageBackend, Depends(get_storage)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
    queue: Annotated[IngestionQueue, Depends(get_queue)],
    vectors: Annotated[VectorStore, Depends(get_vectors)],
) -> DocumentService:
    """The document service with its ingestion and vector collaborators.

    `vectors` is here for `soft_delete`, which clears a document's vectors from the index after
    the status change. `queue` is here for `upload` and `restore`.
    """
    return DocumentService(documents, storage, audit, queue, vectors)


def get_rag_service(
    queries: Annotated[QueryRepository, Depends(get_query_repository)],
    bedrock: Annotated[BedrockClient, Depends(get_bedrock)],
    vectors: Annotated[VectorStore, Depends(get_vectors)],
) -> RagService:
    return RagService(queries, bedrock, vectors)


def get_query_service(
    queries: Annotated[QueryRepository, Depends(get_query_repository)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
    rag: Annotated[RagService, Depends(get_rag_service)],
) -> QueryService:
    """The query service with its pipeline attached.

    Note that `queries` is resolved once and shared with `RagService` — FastAPI caches a
    dependency per request, so both hold the same repository and therefore the same session.
    That matters: `load_chunks` and `attach_answer` must see the same transaction.
    """
    return QueryService(queries, audit, rag)


def get_search_history_service(
    queries: Annotated[QueryRepository, Depends(get_query_repository)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
) -> SearchHistoryService:
    return SearchHistoryService(queries, audit)


def get_analytics_service(
    analytics: Annotated[AnalyticsRepository, Depends(get_analytics_repository)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
    history: Annotated[SearchHistoryService, Depends(get_search_history_service)],
) -> AnalyticsService:
    """The analytics service, including the history service its export path reuses.

    `history` is a dependency rather than a second query path because the history export must
    return exactly the rows `GET /history` would. Two implementations of "which rows may this
    caller see" is how a Standard User's questions end up in a file they were never shown.
    """
    return AnalyticsService(analytics, audit, history)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
QueryServiceDep = Annotated[QueryService, Depends(get_query_service)]
RagServiceDep = Annotated[RagService, Depends(get_rag_service)]
SearchHistoryServiceDep = Annotated[SearchHistoryService, Depends(get_search_history_service)]
AnalyticsServiceDep = Annotated[AnalyticsService, Depends(get_analytics_service)]
AuditRepositoryDep = Annotated[AuditRepository, Depends(get_audit_repository)]
