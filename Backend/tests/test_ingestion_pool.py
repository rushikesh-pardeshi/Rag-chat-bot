"""backend/tests/test_ingestion_pool.py

Role: Tests for `LocalIngestionPool` and the dispatch mode `IngestionQueue` picks.

This is the machinery that makes a multi-file upload genuinely concurrent on a host with no
broker. Three properties are worth defending, and each one is a bug that actually happens:

  * **It is concurrent.** The whole reason the mode exists. A pool that ran the pipeline one
    document at a time would be eager mode with extra steps, and the test that proves
    otherwise has to observe overlap, not just a finishing time.
  * **It is bounded.** A drag-and-drop of forty files must not become forty simultaneous
    database connections and forty concurrent Bedrock conversations. `INGEST_LOCAL_CONCURRENCY`
    is a ceiling, and a semaphore built per request would not be one.
  * **It dispatches after the commit, not before.** The pool's task opens its *own* session,
    so it can only see the `document` row once the upload request's transaction has committed.
    Submitting at the point of call is a race the loser of which is silent: the task finds no
    document, logs a warning, and the upload looks fine while the document is never indexed.

`ingest_document` is substituted throughout. What the pipeline does with a document is tested
against the fakes in `test_ingestion_to_query.py`; what is under test here is purely the
scheduling around it, and a real pipeline would make the concurrency assertions depend on
Bedrock timing.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from backend.core.config import get_settings
from backend.tests.fakes import FakeDocumentRepository
from backend.workers import document_processor
from backend.workers.document_processor import IngestionQueue, LocalIngestionPool


@pytest.fixture
def ids() -> tuple[uuid.UUID, uuid.UUID]:
    """One tenant, used for every document in a test."""
    return uuid.uuid4(), uuid.uuid4()


class ConcurrencyProbe:
    """Stands in for `ingest_document` and records how much overlap actually happened.

    `peak` is the assertion that matters. Counting completions would pass for a serial pool,
    and timing would make the test flaky on a loaded machine; the number of tasks inside the
    body at once is the direct observation.
    """

    def __init__(self, hold: asyncio.Event | None = None) -> None:
        #: Set externally to pin tasks inside the body, so overlap is observable rather than
        #: a matter of scheduling luck.
        self.hold = hold
        self.active = 0
        self.peak = 0
        self.started: list[uuid.UUID] = []
        self.finished: list[uuid.UUID] = []
        self.failures: set[uuid.UUID] = set()

    async def __call__(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.append(document_id)
        try:
            if self.hold is not None:
                await self.hold.wait()
            else:
                # One loop turn, so tasks interleave rather than each running to completion
                # inside its own `create_task` step.
                await asyncio.sleep(0)
            if document_id in self.failures:
                raise RuntimeError("ingestion blew up")
        finally:
            self.active -= 1
            self.finished.append(document_id)


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> ConcurrencyProbe:
    probe = ConcurrencyProbe()
    monkeypatch.setattr(document_processor, "ingest_document", probe)
    return probe


class TestItIsActuallyConcurrent:
    async def test_several_documents_are_ingested_at_the_same_time(
        self, probe: ConcurrencyProbe
    ) -> None:
        """The property the whole mode exists for."""
        hold = asyncio.Event()
        probe.hold = hold
        pool = LocalIngestionPool(concurrency=4)

        for _ in range(4):
            pool.submit(uuid.uuid4(), uuid.uuid4())
        # Let every task reach the body. Nothing has finished yet, so `active` is the number
        # genuinely in flight together.
        await asyncio.sleep(0)

        assert probe.active == 4
        hold.set()
        assert await pool.drain(grace_seconds=5) == 4
        assert probe.peak == 4

    async def test_concurrency_is_capped_at_the_configured_limit(
        self, probe: ConcurrencyProbe
    ) -> None:
        """Eight documents, two slots. The ninth connection is the one that must not exist."""
        hold = asyncio.Event()
        probe.hold = hold
        pool = LocalIngestionPool(concurrency=2)

        for _ in range(8):
            pool.submit(uuid.uuid4(), uuid.uuid4())
        await asyncio.sleep(0)

        assert probe.active == 2
        assert len(probe.started) == 2
        hold.set()
        await pool.drain(grace_seconds=5)

        # All eight ran, but never more than two together.
        assert len(probe.finished) == 8
        assert probe.peak == 2

    async def test_a_batch_of_25_all_run_four_at_a_time(
        self, probe: ConcurrencyProbe
    ) -> None:
        """An administrator dropping 25 files at once, from the pool's side.

        The claim being pinned is two-sided and both halves matter: **all 25 are ingested**
        (none parked and forgotten behind the semaphore), and **never more than 4 at once**
        (the ceiling holds at a batch size well past it, so `INGEST_LOCAL_CONCURRENCY` is a
        real limit on database connections and concurrent Bedrock conversations rather than a
        number that only applies to small batches).

        `test_document_service.py::TestABulkUpload` is the other half of the same claim: that
        25 documents come out `ACTIVE` with every chunk in the vector store.
        """
        pool = LocalIngestionPool(concurrency=4)
        document_ids = [uuid.uuid4() for _ in range(25)]
        tenant_id = uuid.uuid4()

        for document_id in document_ids:
            pool.submit(document_id, tenant_id)

        assert await pool.drain(grace_seconds=10) == 25
        assert sorted(map(str, probe.finished)) == sorted(map(str, document_ids))
        assert probe.peak == 4
        # Nothing left behind: the pool is empty and holds no strong references to tasks.
        assert pool.in_flight == 0

    async def test_a_zero_or_negative_limit_still_runs_one_at_a_time(
        self, probe: ConcurrencyProbe
    ) -> None:
        """`LocalIngestionPool` is never constructed below one slot.

        The settings field refuses zero for this path — `INGEST_LOCAL_CONCURRENCY = 0` means
        "inline mode" and never builds a pool — but a pool that silently accepted zero would
        deadlock every ingestion rather than fail, which is the worst way to be wrong.
        """
        pool = LocalIngestionPool(concurrency=0)
        document_id = uuid.uuid4()
        pool.submit(document_id, uuid.uuid4())
        await pool.drain(grace_seconds=5)
        assert probe.finished == [document_id]


class TestFailureIsContained:
    async def test_one_document_failing_does_not_stop_the_others(
        self, probe: ConcurrencyProbe
    ) -> None:
        """`ingest_document` has already recorded `FAILED` by the time it raises.

        The pool has no caller to report to, so the alternative to swallowing is an
        "exception was never retrieved" warning at garbage-collection time — and a sibling
        document left unprocessed.
        """
        doomed = uuid.uuid4()
        healthy = [uuid.uuid4() for _ in range(3)]
        probe.failures = {doomed}
        pool = LocalIngestionPool(concurrency=4)

        for document_id in [doomed, *healthy]:
            pool.submit(document_id, uuid.uuid4())
        await pool.drain(grace_seconds=5)

        assert set(probe.finished) == {doomed, *healthy}

    async def test_drain_cancels_what_it_cannot_wait_for(
        self, probe: ConcurrencyProbe
    ) -> None:
        """A wedged Textract poll must not hold the process open through shutdown."""
        probe.hold = asyncio.Event()  # never set
        pool = LocalIngestionPool(concurrency=2)
        pool.submit(uuid.uuid4(), uuid.uuid4())
        await asyncio.sleep(0)

        assert await pool.drain(grace_seconds=0.05) == 1
        assert pool.in_flight == 0

    async def test_draining_an_idle_pool_is_a_no_op(self) -> None:
        assert await LocalIngestionPool(concurrency=2).drain(grace_seconds=5) == 0


class TestWhichModeTheQueuePicks:
    """`parallel` is the branch selector, so it is worth asserting directly."""

    @pytest.mark.parametrize(
        ("eager", "concurrency", "expected"),
        [
            # The deployment default here: no broker, concurrent ingestion.
            (True, 4, True),
            # Explicitly asked for the original inline behaviour.
            (True, 0, False),
            # A real broker and a real worker. Nothing in this module runs.
            (False, 4, False),
            (False, 0, False),
        ],
    )
    def test_parallel_requires_eager_mode_and_a_positive_limit(
        self, eager: bool, concurrency: int, expected: bool
    ) -> None:
        settings = get_settings().model_copy(
            update={
                "CELERY_TASK_ALWAYS_EAGER": eager,
                "INGEST_LOCAL_CONCURRENCY": concurrency,
            }
        )
        assert IngestionQueue(settings).parallel is expected

    async def test_parallel_mode_defers_dispatch_to_the_commit(
        self, probe: ConcurrencyProbe, ids: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        """The race this avoids is silent, which is why it is pinned.

        `FakeDocumentRepository.after_commit` fires the callback immediately — it has no
        transaction to wait for — so what this asserts is that `enqueue` went through
        `after_commit` *at all* rather than submitting directly. The real repository defers to
        the session, and `TestAfterCommitAgainstPostgres` covers that end.
        """
        document_id, tenant_id = ids
        documents = FakeDocumentRepository()
        queue = IngestionQueue(
            get_settings().model_copy(
                update={
                    "CELERY_TASK_ALWAYS_EAGER": True,
                    "INGEST_LOCAL_CONCURRENCY": 4,
                }
            ),
            pool=LocalIngestionPool(concurrency=4),
        )

        await queue.enqueue(document_id, tenant_id, documents=documents)

        assert len(documents.after_commit_callbacks) == 1
        # `call_soon_threadsafe` schedules onto the loop, so the task does not exist until the
        # loop next runs. That one turn of latency is the mechanism, not an accident.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert probe.started == [document_id]

    async def test_parallel_mode_does_not_run_the_pipeline_inline(
        self, probe: ConcurrencyProbe, ids: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        """`enqueue` must return before ingestion starts, or the upload blocks on it again."""
        document_id, tenant_id = ids
        queue = IngestionQueue(
            get_settings().model_copy(
                update={
                    "CELERY_TASK_ALWAYS_EAGER": True,
                    "INGEST_LOCAL_CONCURRENCY": 4,
                }
            ),
            pool=LocalIngestionPool(concurrency=4),
        )

        await queue.enqueue(document_id, tenant_id, documents=FakeDocumentRepository())

        # Nothing has run yet: the work is queued on the loop, and the upload request is free
        # to commit and respond.
        assert probe.started == []

    async def test_inline_mode_is_unchanged(
        self, monkeypatch: pytest.MonkeyPatch, ids: tuple[uuid.UUID, uuid.UUID]
    ) -> None:
        """`INGEST_LOCAL_CONCURRENCY = 0` still awaits the pipeline on the caller's session."""
        document_id, tenant_id = ids
        calls: list[tuple[uuid.UUID, object]] = []

        async def fake_in_session(
            document_id: uuid.UUID, tenant_id: uuid.UUID, **kwargs: object
        ) -> None:
            calls.append((document_id, kwargs.get("documents")))

        monkeypatch.setattr(
            document_processor, "ingest_document_in_session", fake_in_session
        )
        documents = FakeDocumentRepository()
        queue = IngestionQueue(
            get_settings().model_copy(
                update={
                    "CELERY_TASK_ALWAYS_EAGER": True,
                    "INGEST_LOCAL_CONCURRENCY": 0,
                }
            )
        )

        await queue.enqueue(document_id, tenant_id, documents=documents)

        # Already done by the time `enqueue` returned, and on the caller's repository.
        assert calls == [(document_id, documents)]
        assert documents.after_commit_callbacks == []


@pytest.mark.requires_db
class TestAfterCommitAgainstPostgres:
    """The half the fake cannot prove: the real hook fires on commit, and only on commit.

    Marked as well as fixture-guarded: `db_available` makes these skip on a machine with no
    PostgreSQL, and the marker is what gets them selected by `scripts/run_db_tests.py test`.

    `FakeDocumentRepository.after_commit` calls back immediately, which is the right stand-in
    for a test with no transaction but says nothing about the SQLAlchemy event. Skipped rather
    than failed without a database, like the rest of the `db_available` suite.
    """

    async def test_the_callback_fires_when_the_session_commits(
        self, db_available: bool
    ) -> None:
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope
        from backend.repositories.document_repository import DocumentRepository

        fired: list[str] = []
        async with session_scope() as session:
            DocumentRepository(session).after_commit(lambda: fired.append("committed"))
            # Nothing yet — the transaction is still open, and this is exactly the window in
            # which a pool task would find no document row.
            assert fired == []

        assert fired == ["committed"]

    async def test_the_callback_does_not_fire_on_a_rollback(
        self, db_available: bool
    ) -> None:
        """No document was stored, so there is nothing to ingest."""
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope
        from backend.repositories.document_repository import DocumentRepository

        fired: list[str] = []
        with pytest.raises(RuntimeError):
            async with session_scope() as session:
                DocumentRepository(session).after_commit(lambda: fired.append("committed"))
                raise RuntimeError("the upload failed")

        assert fired == []

    async def test_lock_name_excludes_a_concurrent_transaction_on_the_same_name(
        self, db_available: bool
    ) -> None:
        """The deadlock guard, proved where it has to hold: against a second transaction.

        Version supersession writes the rows of *other* versions of a name while the pipeline
        holds the tenant's audit-chain lock, and another pipeline can hold one of those rows
        while waiting for that same chain lock. That cycle is a PostgreSQL deadlock, and the
        aborted side rolls back a document that had already embedded and upserted — its chunk
        rows vanish, its vectors do not, and nothing can enumerate them again.

        `lock_name` removes the cycle by making ingestion of one `(tenant, name)` mutually
        exclusive, so the two transactions never hold each other's rows. That claim is about
        two transactions, which the fakes cannot express; this is the test that can.
        """
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope
        from backend.repositories.document_repository import DocumentRepository

        tenant_id = uuid.uuid4()
        release_first = asyncio.Event()
        second_acquired = asyncio.Event()

        async def hold() -> None:
            async with session_scope() as session:
                await DocumentRepository(session).lock_name("policy.pdf", tenant_id)
                await release_first.wait()

        async def contend() -> None:
            async with session_scope() as session:
                await DocumentRepository(session).lock_name("policy.pdf", tenant_id)
                second_acquired.set()

        first = asyncio.create_task(hold())
        # Long enough for the first transaction to have taken the lock.
        await asyncio.sleep(0.25)
        second = asyncio.create_task(contend())

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(asyncio.shield(second_acquired.wait()), timeout=0.75)

        release_first.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=10)
        # Granted the moment the first transaction committed, not before it.
        assert second_acquired.is_set()

    async def test_lock_name_does_not_serialize_two_different_names(
        self, db_available: bool
    ) -> None:
        """Otherwise the fix would cost the parallelism the pool exists to provide.

        Two documents with different names share no rows that supersession could contend on,
        so there is nothing to exclude — and a lock keyed on the tenant instead of the name
        would have quietly made a four-slot pool serial.
        """
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope
        from backend.repositories.document_repository import DocumentRepository

        tenant_id = uuid.uuid4()
        release = asyncio.Event()
        other_acquired = asyncio.Event()

        async def hold(name: str) -> None:
            async with session_scope() as session:
                await DocumentRepository(session).lock_name(name, tenant_id)
                await release.wait()

        async def take(name: str) -> None:
            async with session_scope() as session:
                await DocumentRepository(session).lock_name(name, tenant_id)
                other_acquired.set()

        first = asyncio.create_task(hold("policy.pdf"))
        await asyncio.sleep(0.25)
        second = asyncio.create_task(take("fees.pdf"))

        # Granted immediately, while the first lock is still held.
        await asyncio.wait_for(other_acquired.wait(), timeout=5)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, second), timeout=10)

    async def test_the_callback_fires_once_per_registration(
        self, db_available: bool
    ) -> None:
        """`once=True`. Two uploads in one request must not ingest the first one twice."""
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from sqlalchemy import text

        from backend.core.database import session_scope
        from backend.repositories.document_repository import DocumentRepository

        fired: list[str] = []
        async with session_scope() as session:
            documents = DocumentRepository(session)
            documents.after_commit(lambda: fired.append("first"))
            documents.after_commit(lambda: fired.append("second"))
            # A commit mid-session, so the listeners have a chance to fire more than once.
            await session.execute(text("SELECT 1"))
            await session.commit()
            assert fired == ["first", "second"]

        assert fired == ["first", "second"]
