"""backend/tests/harness.py

Role: The shared two-tenant, four-user application harness. Builds the whole service graph
over the fakes in `backend/tests/fakes.py` and mounts the *real* `create_app()` on top, so a
test can make HTTP requests through the production routers, dependencies and middleware
without a database, without AWS and without Pinecone.

Why this is a module rather than a `conftest.py`
------------------------------------------------
Two packages need the same stack for different reasons. `backend/tests/security/` needs it to
prove a rule cannot be bypassed; `backend/tests/regression/` needs it to prove a business
requirement still works. Neither should import the other's `conftest.py` — a regression suite
that depends on the security package's fixtures inverts the dependency (the security suite
would then be load-bearing for BR coverage, and deleting a security test could break a BR
test). So the harness lives here, in neither package, and each `conftest.py` is a thin set of
fixtures over it.

Why two tenants and four users
------------------------------
An authorization test needs someone to *be* the victim. One tenant with one user can only
prove that a made-up UUID 404s, which is the easy half — the interesting failure is a real id
belonging to a real user in a real other tenant, because that is the id an attacker actually
obtains (from a citation, an export, a shared screenshot). So `build_stack()` builds:

    tenant A: admin_a (ADMINISTRATOR), user_a (STANDARD_USER)
    tenant B: admin_b (ADMINISTRATOR), user_b (STANDARD_USER)

Both tenants get a document and an answered query, and the ids of both are available to every
test.

Why the app is overridden rather than mocked
--------------------------------------------
Much of what these suites are about lives in the layers *between* the socket and the service:
the CSRF middleware, `get_current_user`, `require_admin`, Pydantic's `extra='forbid'`, the
exception handlers that decide what a client is told. Calling services directly would skip all
of it. So the real `create_app()` is used and only the leaves are replaced — the four
repository providers, the storage client, the queue, Bedrock and the vector index. Nothing that
enforces a rule is faked.

`as_user` is the switch that makes a request "come from" someone. It overrides
`get_current_user` only; `require_admin` is left real, so a 403 in these tests is produced by
the same code that produces it in production.
"""

from __future__ import annotations

import pathlib
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

import httpx

from backend.api.app import create_app
from backend.api.dependencies import auth as auth_deps
from backend.api.dependencies import services as service_deps
from backend.core.config import get_settings
from backend.core.security import generate_csrf_token
from backend.models.document import Document
from backend.models.query import Query
from backend.models.user import User, UserRole
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.services.document_service import DocumentService
from backend.services.query_service import QueryService
from backend.services.rag_service import RagService
from backend.services.search_history_service import SearchHistoryService
from backend.services.storage_service import S3StorageService
from backend.tests.fakes import (
    Corpus,
    FakeAnalyticsRepository,
    FakeAuditRepository,
    FakeBedrockClient,
    FakeDocumentRepository,
    FakeIngestionQueue,
    FakePineconeIndex,
    FakeQueryRepository,
    FakeS3Client,
    FakeUserRepository,
)
from backend.utils.vector_store import VectorStore

API = "/api/v1"

#: The repository root, so `ROOT / "backend"` is the package the static scans walk. Taken from
#: this file's own location rather than the working directory, because `pytest` can be invoked
#: from anywhere and a scan that silently walked an empty tree would pass every assertion made
#: over it — which is exactly what happened on the first run of `test_vector_isolation.py`, and
#: why every caller of `backend_modules()` also asserts the set is non-empty.
ROOT = pathlib.Path(__file__).resolve().parents[2]


def backend_modules() -> list[pathlib.Path]:
    """Every shipped module. Tests are excluded; scripts are not shipped.

    Shared by the modules that make *structural* claims — "no module builds a namespace
    string of its own", "nothing in the backend opens a file" — because a structural claim is
    only as good as the tree it walked, and one copy of this function is one place for that
    trap to be sprung.
    """
    return [
        path
        for path in ROOT.joinpath("backend").rglob("*.py")
        if "tests" not in path.parts and "__pycache__" not in path.parts
    ]


#: Passed to services called directly (not over HTTP). A documentation-range address, so a
#: grep for it in a log or an audit row is unambiguous.
CONTEXT = RequestContext(ip_address="203.0.113.42", user_agent="pytest-harness")

#: Two passages per tenant, worded almost identically on purpose. Near-identical text is what
#: makes `test_vector_isolation.py` meaningful: if isolation were by metadata filter rather
#: than by namespace, tenant A's question would rank tenant B's passage highly, and only the
#: filter would be standing between them. With one namespace per tenant the match cannot happen
#: at all.
TENANT_A_PASSAGES = [
    "The outbound domestic wire transfer fee is 25 USD per payment instruction at "
    "Tenant A Bank.",
    "The monthly account maintenance fee is 12 USD and is waived above a 5,000 USD "
    "average daily balance.",
]
TENANT_B_PASSAGES = [
    "The outbound domestic wire transfer fee is 40 USD per payment instruction at "
    "Tenant B Trust.",
    "The monthly account maintenance fee is 18 USD and is waived above a 9,000 USD "
    "average daily balance.",
]

#: A question both corpora would answer, which is the point — the answer must depend on who
#: asks, and on nothing else.
SHARED_QUESTION = "What is the outbound domestic wire transfer fee per payment instruction?"


@dataclass
class Tenant:
    """One tenant's ids, so a test can name a victim without reaching into the fakes."""

    id: uuid.UUID
    admin: User
    user: User
    document: Document | None = None
    query: Query | None = None
    passages: list[str] = field(default_factory=list)


@dataclass
class Stack:
    """Every fake and every service, plus the two tenants.

    Held as one object rather than a dozen fixtures because these tests are mostly
    *relational* — "this user, that tenant's id, and what the audit trail says afterwards" —
    and threading nine fixtures through each of those signatures obscures the assertion.
    """

    documents: FakeDocumentRepository
    queries: FakeQueryRepository
    users: FakeUserRepository
    audit_repo: FakeAuditRepository
    analytics_repo: FakeAnalyticsRepository
    s3: FakeS3Client
    index: FakePineconeIndex
    bedrock: FakeBedrockClient
    vectors: VectorStore
    storage: S3StorageService
    audit: AuditService
    queue: FakeIngestionQueue
    document_service: DocumentService
    query_service: QueryService
    history_service: SearchHistoryService
    a: Tenant
    b: Tenant

    # -- Assertion helpers ---------------------------------------------------
    def audit_details(self) -> str:
        """Every audit `detail` as one string, for "does X appear anywhere" assertions."""
        return repr([entry.detail for entry in self.audit_repo.entries])

    def denials(self) -> list[Any]:
        from backend.models.audit_log import AuditOutcome

        return [e for e in self.audit_repo.entries if e.outcome is AuditOutcome.DENIED]

    def namespace(self, tenant_id: uuid.UUID) -> str:
        return get_settings().pinecone_namespace(tenant_id)


def make_user(tenant_id: uuid.UUID, role: UserRole, email: str) -> User:
    """A `User` ORM entity that is never persisted and never authenticates.

    `hashed_password` is deliberately not a real hash: no test built on this harness signs in
    over `POST /auth/login`, and a bcrypt call per user would cost more than the rest of the
    suite. The tests that *do* exercise password verification build their own user.
    """
    return User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email=email,
        full_name=None,
        hashed_password="not-a-real-hash",
        role=role,
        is_active=True,
        failed_login_attempts=0,
        locked_until=None,
    )


async def build_stack() -> Stack:
    """Two populated tenants over the fakes. The starting point for every test.

    Ordering matters here and is not incidental:

      * the tenants are registered in `FakeUserRepository` *before* any document exists, so a
        tenant-scoped `get_by_id` can succeed — the IDOR tests read a victim's id back through
        that lookup;
      * `Corpus.add` writes chunk rows, vectors and an `ACTIVE` status but no S3 bytes, so a
        test that needs a *retrievable* document uploads one itself rather than assuming;
      * one answered query per tenant is submitted last, through the real `QueryService`, so
        history, analytics, traceability and the audit chain all have a genuine row to read
        rather than a hand-built one.
    """
    settings = get_settings()
    documents = FakeDocumentRepository()
    queries = FakeQueryRepository(documents)
    users = FakeUserRepository()
    audit_repo = FakeAuditRepository()
    analytics_repo = FakeAnalyticsRepository()
    s3 = FakeS3Client()
    index = FakePineconeIndex()
    bedrock = FakeBedrockClient(dimensions=settings.BEDROCK_EMBEDDING_DIMENSIONS)
    vectors = VectorStore(index=index, settings=settings)
    storage = S3StorageService(client=s3)
    audit = AuditService(audit_repo)
    queue = FakeIngestionQueue(
        storage=storage, bedrock=bedrock, vectors=vectors, settings=settings
    )

    tenants: list[Tenant] = []
    for domain, passages in (
        ("tenant-a.example.com", TENANT_A_PASSAGES),
        ("tenant-b.example.com", TENANT_B_PASSAGES),
    ):
        record = users.add_tenant(domain=domain, name=domain)
        tenants.append(
            Tenant(
                id=record.id,
                admin=make_user(record.id, UserRole.ADMINISTRATOR, f"admin@{domain}"),
                user=make_user(record.id, UserRole.STANDARD_USER, f"standard@{domain}"),
                passages=list(passages),
            )
        )
    a, b = tenants

    for tenant in tenants:
        for person in (tenant.admin, tenant.user):
            users.users[person.id] = person

    corpus = Corpus(documents, vectors, bedrock)
    for tenant in tenants:
        tenant.document = await corpus.add(
            tenant.id, f"fee-schedule-{tenant.id.hex[:6]}.pdf", tenant.passages
        )

    document_service = DocumentService(documents, storage, audit, queue, vectors)
    query_service = QueryService(
        queries, audit, RagService(queries, bedrock, vectors, settings)
    )
    history_service = SearchHistoryService(queries, audit)

    for tenant in tenants:
        bedrock.queue_answer("The fee is stated in the schedule [S1].")
        tenant.query = await query_service.submit(
            question=SHARED_QUESTION, actor=tenant.user, context=CONTEXT
        )

    return Stack(
        documents=documents,
        queries=queries,
        users=users,
        audit_repo=audit_repo,
        analytics_repo=analytics_repo,
        s3=s3,
        index=index,
        bedrock=bedrock,
        vectors=vectors,
        storage=storage,
        audit=audit,
        queue=queue,
        document_service=document_service,
        query_service=query_service,
        history_service=history_service,
        a=a,
        b=b,
    )


#: The provider → fake map. Named so a test can assert *which* providers are replaced, and so
#: the list stays reviewable: anything that enforces a rule must be absent from it.
def _overrides(stack: Stack, current: dict[str, User | None]) -> dict[Any, Any]:
    async def _current_user() -> User:
        user = current["user"]
        if user is None:
            from backend.core.exceptions import AuthenticationError

            raise AuthenticationError()
        return user

    return {
        auth_deps.get_current_user: _current_user,
        auth_deps.get_optional_user: lambda: current["user"],
        auth_deps.get_audit_service: lambda: stack.audit,
        auth_deps.get_audit_repository: lambda: stack.audit_repo,
        auth_deps.get_user_repository: lambda: stack.users,
        service_deps.get_document_repository: lambda: stack.documents,
        service_deps.get_query_repository: lambda: stack.queries,
        service_deps.get_analytics_repository: lambda: stack.analytics_repo,
        service_deps.get_storage: lambda: stack.storage,
        service_deps.get_queue: lambda: stack.queue,
        service_deps.get_bedrock: lambda: stack.bedrock,
        service_deps.get_vectors: lambda: stack.vectors,
    }


Api = tuple[httpx.AsyncClient, Callable[[User | None], None]]


@asynccontextmanager
async def build_api(stack: Stack) -> AsyncIterator[Api]:
    """The real app over `stack`, plus an `as_user` switch.

    Yields `(client, as_user)`. Call `as_user(stack.a.user)` and every subsequent request is
    authenticated as that person. Nothing else about authorization is replaced:
    `require_admin`, the CSRF middleware and the exception handlers are the production ones, so
    a 403 here is the same 403 a browser would get.

    Note that the analytics *service* is not on `Stack`: only
    `get_analytics_repository` is overridden, so the real `AnalyticsService` is constructed by
    dependency injection over the fake repository. That is deliberate — the analytics
    authorization re-check runs for real.
    """
    app = create_app()
    current: dict[str, User | None] = {"user": None}
    app.dependency_overrides.update(_overrides(stack, current))

    def as_user(user: User | None) -> None:
        current["user"] = user

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as client:
        yield client, as_user


def csrf_headers() -> dict[str, str]:
    """A matching cookie/header pair, so a test about something *else* is not blocked by CSRF.

    Every unsafe request carries these unless the test is specifically about CSRF — see
    `backend/tests/security/test_csrf.py`, which asserts what happens without them.
    """
    token = generate_csrf_token()
    settings = get_settings()
    return {
        "Cookie": f"{settings.CSRF_COOKIE_NAME}={token}",
        settings.CSRF_HEADER_NAME: token,
    }
