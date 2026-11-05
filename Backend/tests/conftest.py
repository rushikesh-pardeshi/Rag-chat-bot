"""backend/tests/conftest.py

Role: Shared pytest fixtures.

Two principles hold across the whole suite:
  * no test contacts a network service — Bedrock, S3 and Pinecone are always faked;
  * tests that need a database are marked `requires_db` and skip cleanly when none is
    reachable, so the rest of the suite still runs on a laptop without Postgres.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest

# The test environment (DATABASE_URL, SECRET_KEY, blanked AWS credentials) is set in the
# repository-root `conftest.py`, which pytest imports before this one. It lives there and
# not here so that nothing under `backend/` reads the environment directly.


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "requires_db: test needs a reachable PostgreSQL 17"
    )


@pytest.fixture(scope="session")
def event_loop() -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings():
    from backend.core.config import get_settings

    return get_settings()


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    """ASGI client that exercises the real app without binding a socket."""
    from backend.api.app import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as ac:
        yield ac


@pytest.fixture
async def db_available() -> AsyncIterator[bool]:
    """True when a database answers. Used by `requires_db` tests to skip, not fail.

    The engine is disposed on both sides of the test, and that is not tidiness — it is what
    makes the second `requires_db` test in a run behave like the first.

    `backend/core/database.py` caches one engine per process, and its connection pool holds
    asyncpg connections bound to the event loop that opened them. Because
    `asyncio_default_fixture_loop_scope` is `function`, every test gets a fresh loop; a pooled
    connection from the previous test's loop then fails on checkout. The failure surfaced here
    as `db_available` returning False — so the test *skipped*, reporting "no PostgreSQL
    reachable" on a machine where PostgreSQL was running and the previous test had just
    used it.

    A silent skip is the worst possible outcome for a compliance suite: the run is green, the
    count of passing tests goes down, and nothing says the tamper-evidence proof never
    executed. Disposing per test costs one connection handshake and removes the failure mode.
    """
    from sqlalchemy import text

    from backend.core.database import dispose_engine, session_scope

    await dispose_engine()
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
        reachable = True
    except Exception:
        reachable = False

    try:
        yield reachable
    finally:
        # Leaves no connection bound to a loop that is about to close, which is also what
        # silences `RuntimeWarning: coroutine 'Connection._cancel' was never awaited`.
        await dispose_engine()


#: Re-exported so a fixture and a direct import name the *same* class. This file used to
#: define a second, weaker `FakeS3Client` that accepted any `put_object` — including one that
#: overwrote an existing key. Keeping both would mean the no-overwrite guard in
#: `storage_service.put` was exercised by some tests and silently ignored by others, which is
#: the failure mode a fake is supposed to prevent.
#:
#: `FakeBedrockClient` moved to `fakes.py` for the same reason. The copy that lived here hashed
#: the whole string, which made every pair of texts roughly equidistant — reproducible, but
#: useless for ranking, and ranking is what the RAG tests are about.
from backend.tests.fakes import (  # noqa: E402
    FakeBedrockClient,
    FakePineconeIndex,
    FakeS3Client,
)


@pytest.fixture
def fake_s3() -> FakeS3Client:
    return FakeS3Client()


@pytest.fixture
def fake_bedrock() -> FakeBedrockClient:
    return FakeBedrockClient()


@pytest.fixture
def fake_index() -> FakePineconeIndex:
    return FakePineconeIndex()


@pytest.fixture
def fake_vectors(fake_index: FakePineconeIndex, settings):
    """A real `VectorStore` over a fake index.

    The store itself is *not* faked: namespace construction, upsert batching, the metadata
    allowlist and match normalization are the parts that keep one tenant's passages out of
    another's answers, so they stay under test. Only the network is replaced.
    """
    from backend.utils.vector_store import VectorStore

    return VectorStore(index=fake_index, settings=settings)
