"""backend/tests/test_vector_store.py

Role: Unit tests for `backend/utils/vector_store.py` — the only module in the codebase allowed
to import `pinecone`, and therefore the only place tenant isolation in the index is decided.

Run against `FakePineconeIndex`, which satisfies the `AsyncIndex` protocol with real cosine
ranking and separate dicts per namespace. The store under test is the real class: namespace
construction, batching, the metadata allowlist and match normalization are precisely the code
that keeps one bank tenant's passages out of another's answers.

The three standing invariants from `.claude/rules/architecture.md`
------------------------------------------------------------------
1. **One namespace per tenant, built solely by `Settings.pinecone_namespace`.** Isolation is
   never a metadata filter — a wrong namespace returns nothing, a forgotten filter returns
   another tenant's chunks, and only the first failure mode is safe.
2. **The vector id is `str(chunk.id)`.** That is what makes re-ingestion overwrite its own
   vectors instead of doubling them, so a restore does not leave the index with two copies of
   every passage at two different scores.
3. **Metadata carries citations, never document text.** A third copy of customer content in a
   system with its own access model is what `secure-coding-vapt.md` forbids.

The last test in the file is the import-boundary gate itself, read off the source tree.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from backend.core.config import Settings, get_settings
from backend.tests.fakes import FakePineconeIndex
from backend.utils import vector_store
from backend.utils.vector_store import (
    ALLOWED_METADATA_KEYS,
    FORBIDDEN_METADATA_KEYS,
    ChunkVector,
    ForbiddenVectorMetadata,
    VectorMatch,
    VectorStore,
    VectorStoreUnavailable,
    _build_metadata,
    _read_matches,
)


@pytest.fixture
def settings() -> Settings:
    return get_settings()


@pytest.fixture
def index() -> FakePineconeIndex:
    return FakePineconeIndex()


@pytest.fixture
def store(index: FakePineconeIndex, settings: Settings) -> VectorStore:
    return VectorStore(index=index, settings=settings)


@pytest.fixture
def tenant_id() -> uuid.UUID:
    return uuid.uuid4()


def _vector(seed: int, dimensions: int = 8) -> list[float]:
    """A unit vector pointing along one axis, so cosine between distinct seeds is 0."""
    values = [0.0] * dimensions
    values[seed % dimensions] = 1.0
    return values


def _chunk_vector(
    seed: int = 0,
    *,
    chunk_id: uuid.UUID | None = None,
    document_id: uuid.UUID | None = None,
    name: str = "retail-fees.pdf",
    page: int | None = 3,
    label: str = "",
) -> ChunkVector:
    return ChunkVector(
        chunk_id=chunk_id or uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        document_name=name,
        vector=_vector(seed),
        page_number=page,
        section_label=label,
        chunk_index=seed,
    )


# -- Namespaces --------------------------------------------------------------
class TestNamespaces:
    async def test_an_upsert_lands_in_the_tenants_namespace_and_nowhere_else(
        self,
        store: VectorStore,
        index: FakePineconeIndex,
        settings: Settings,
        tenant_id: uuid.UUID,
    ) -> None:
        await store.upsert_embeddings(tenant_id, [_chunk_vector()])

        assert list(index.namespaces) == [settings.pinecone_namespace(tenant_id)]

    async def test_the_namespace_comes_only_from_settings(
        self, store: VectorStore, index: FakePineconeIndex, settings: Settings
    ) -> None:
        """One construction site for the namespace string.

        A second place that formats `f"tenant-{id}"` is a second place that can drift, and a
        drifted namespace does not fail loudly — it quietly returns no matches, which reads
        exactly like "this tenant has no documents".
        """
        tenant_id = uuid.uuid4()
        await store.upsert_embeddings(tenant_id, [_chunk_vector()])
        await store.search(tenant_id, _vector(0), 5)

        expected = settings.pinecone_namespace(tenant_id)
        assert expected.startswith(settings.PINECONE_NAMESPACE_PREFIX)
        assert str(tenant_id) in expected
        assert {namespace for namespace, _ in index.query_calls} == {expected}

    async def test_a_search_in_one_tenant_cannot_see_anothers_vectors(
        self, store: VectorStore, index: FakePineconeIndex
    ) -> None:
        """The isolation test that matters, with the same vector in both namespaces.

        Tenant B's vector is a *perfect* match for the query. A leak would therefore surface as
        a score of 1.0 rather than as a marginally different ranking.
        """
        tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
        await store.upsert_embeddings(tenant_b, [_chunk_vector(seed=0)])

        assert await store.search(tenant_a, _vector(0), 5) == []
        assert len(await store.search(tenant_b, _vector(0), 5)) == 1

    async def test_search_takes_no_filter_argument(self) -> None:
        """Asserted on the signature, because the absence of an option is the safety property.

        With a filter parameter available, a caller could believe a filter was protecting them
        when it was optional. There is exactly one way to scope a search here.
        """
        import inspect

        parameters = set(inspect.signature(VectorStore.search).parameters)
        assert parameters == {"self", "tenant_id", "vector", "top_k"}


# -- Vector ids and idempotency ----------------------------------------------
class TestVectorIds:
    async def test_the_vector_id_is_the_chunk_id(
        self,
        store: VectorStore,
        index: FakePineconeIndex,
        settings: Settings,
        tenant_id: uuid.UUID,
    ) -> None:
        chunk_id = uuid.uuid4()
        await store.upsert_embeddings(tenant_id, [_chunk_vector(chunk_id=chunk_id)])

        namespace = settings.pinecone_namespace(tenant_id)
        assert index.ids_in(namespace) == {str(chunk_id)}

    async def test_re_upserting_the_same_chunk_overwrites_rather_than_duplicates(
        self,
        store: VectorStore,
        index: FakePineconeIndex,
        settings: Settings,
        tenant_id: uuid.UUID,
    ) -> None:
        """Idempotency per `(document_id, version)`, and the reason it comes for free.

        A restore re-runs the whole pipeline. With a generated vector id, the index would end
        up holding two copies of every passage — the same text retrieved twice at two scores,
        which a reviewer reads as two independent sources agreeing.
        """
        chunk_id = uuid.uuid4()
        await store.upsert_embeddings(tenant_id, [_chunk_vector(0, chunk_id=chunk_id)])
        await store.upsert_embeddings(tenant_id, [_chunk_vector(1, chunk_id=chunk_id)])

        namespace = settings.pinecone_namespace(tenant_id)
        assert index.count(namespace) == 1
        # The second write won: the stored values are the new direction.
        values, _metadata = index.namespaces[namespace][str(chunk_id)]
        assert values == _vector(1)

    async def test_a_search_returns_chunk_ids_as_uuids_not_strings(
        self, store: VectorStore, tenant_id: uuid.UUID
    ) -> None:
        """`load_chunks` binds these into a SQL `IN` clause; a string would not match."""
        chunk_id = uuid.uuid4()
        await store.upsert_embeddings(tenant_id, [_chunk_vector(chunk_id=chunk_id)])

        matches = await store.search(tenant_id, _vector(0), 5)
        assert matches[0].chunk_id == chunk_id
        assert isinstance(matches[0].chunk_id, uuid.UUID)


# -- Batching ----------------------------------------------------------------
class TestBatching:
    async def test_upserts_are_split_at_the_configured_batch_size(
        self, index: FakePineconeIndex, settings: Settings, tenant_id: uuid.UUID
    ) -> None:
        """Pinecone caps a single upsert by payload size.

        A 200-chunk document at 1024 float32 dimensions is well past it, so an unbatched upsert
        fails on exactly the large documents a bank cares most about indexing.
        """
        tuned = settings.model_copy(update={"PINECONE_UPSERT_BATCH_SIZE": 10})
        store = VectorStore(index=index, settings=tuned)

        sent = await store.upsert_embeddings(
            tenant_id, [_chunk_vector(seed) for seed in range(25)]
        )

        assert sent == 25
        assert [size for _namespace, size in index.upsert_calls] == [10, 10, 5]
        assert index.total() == 25

    async def test_an_empty_upsert_never_touches_the_index(
        self, store: VectorStore, index: FakePineconeIndex, tenant_id: uuid.UUID
    ) -> None:
        assert await store.upsert_embeddings(tenant_id, []) == 0
        assert index.upsert_calls == []

    async def test_deletes_are_batched_the_same_way(
        self, index: FakePineconeIndex, settings: Settings, tenant_id: uuid.UUID
    ) -> None:
        tuned = settings.model_copy(update={"PINECONE_UPSERT_BATCH_SIZE": 10})
        store = VectorStore(index=index, settings=tuned)
        chunk_ids = [uuid.uuid4() for _ in range(22)]
        await store.upsert_embeddings(
            tenant_id,
            [_chunk_vector(i, chunk_id=chunk_id) for i, chunk_id in enumerate(chunk_ids)],
        )

        deleted = await store.delete_document_vectors(tenant_id, uuid.uuid4(), chunk_ids)

        assert deleted == 22
        assert [size for _namespace, size, _all in index.delete_calls] == [10, 10, 2]
        assert index.total() == 0


# -- Metadata ----------------------------------------------------------------
class TestMetadata:
    async def test_metadata_holds_the_citation_fields(
        self,
        store: VectorStore,
        index: FakePineconeIndex,
        settings: Settings,
        tenant_id: uuid.UUID,
    ) -> None:
        await store.upsert_embeddings(
            tenant_id,
            [_chunk_vector(name="retail-fees.pdf", page=4, label="4.2 Accrual")],
        )
        namespace = settings.pinecone_namespace(tenant_id)
        _values, metadata = next(iter(index.namespaces[namespace].values()))

        assert metadata["document_name"] == "retail-fees.pdf"
        assert metadata["page_number"] == 4
        assert metadata["section_label"] == "4.2 Accrual"
        assert "document_id" in metadata
        assert "chunk_index" in metadata

    async def test_no_passage_text_reaches_the_index(
        self,
        store: VectorStore,
        index: FakePineconeIndex,
        settings: Settings,
        tenant_id: uuid.UUID,
    ) -> None:
        """The gate item "Pinecone metadata contains no chunk text", asserted on the index.

        Checked on what actually landed rather than on the builder, because the property that
        matters is about the far end: whatever route the metadata took, the passage must not be
        there.
        """
        passage = "The outbound wire transfer fee is 25 USD."
        await store.upsert_embeddings(tenant_id, [_chunk_vector(name=passage[:0] or "a.pdf")])

        namespace = settings.pinecone_namespace(tenant_id)
        _values, metadata = next(iter(index.namespaces[namespace].values()))
        assert passage not in repr(metadata)

    def test_the_allowlist_and_the_forbidden_set_cannot_both_admit_a_key(self) -> None:
        """`_build_metadata` enforces both, so an overlap would make one of them a no-op.

        `ChunkVector` is frozen with slots and `_build_metadata` writes every key itself, so
        the realistic regression is not a mutated instance — it is someone adding `"text"` to
        the allowlist along with a new `ChunkVector.text` field. This is the assertion that
        catches that, and it is cheap enough to be worth having.
        """
        assert not (ALLOWED_METADATA_KEYS & FORBIDDEN_METADATA_KEYS)
        assert ALLOWED_METADATA_KEYS.isdisjoint({"text", "excerpt", "chunk_text"})

    def test_a_key_outside_the_allowlist_raises_instead_of_being_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard's live behaviour, reached the only honest way available.

        `_build_metadata` writes every key itself, so with today's `ChunkVector` no caller can
        smuggle a key in — the guard is unreachable *by construction*, which is why it looks
        redundant. What can change is the allowlist, and the function reads it as a module
        global on every call. Narrowing it here is therefore the same code path a new
        `ChunkVector.text` field would take: the key is present, the allowlist does not admit
        it, and the result is a raise rather than a silent drop.

        Silent dropping is the outcome worth ruling out. It would leave the calling code
        believing the passage had been stored somewhere, which is how "why is retrieval empty"
        turns into a change that stores it for real.
        """
        monkeypatch.setattr(vector_store, "ALLOWED_METADATA_KEYS", frozenset({"document_id"}))

        with pytest.raises(ForbiddenVectorMetadata, match="not on the allowlist"):
            _build_metadata(_chunk_vector())

    async def test_the_fake_index_is_the_second_independent_check(
        self, index: FakePineconeIndex, settings: Settings, tenant_id: uuid.UUID
    ) -> None:
        """A caller that bypassed `_build_metadata` still cannot get text past the fake.

        Two checks rather than one because `_build_metadata` is a function a future writer can
        route around, whereas every test in the suite goes through the fake. A third copy of
        customer content in a system with its own access model is what `secure-coding-vapt.md`
        forbids, so it is worth enforcing twice.
        """
        with pytest.raises(AssertionError, match="never carry document text"):
            await index.upsert(
                vectors=[
                    {
                        "id": str(uuid.uuid4()),
                        "values": _vector(0),
                        "metadata": {"chunk_text": "The wire fee is 25 USD."},
                    }
                ],
                namespace=settings.pinecone_namespace(tenant_id),
            )

    async def test_search_does_not_ask_for_metadata_at_all(
        self, store: VectorStore, index: FakePineconeIndex, tenant_id: uuid.UUID
    ) -> None:
        """The citation is read from PostgreSQL, which is being consulted anyway.

        Trusting index metadata would let a stale document name be cited after a rename — the
        reference would name a document that no longer exists under that name.
        """
        await store.upsert_embeddings(tenant_id, [_chunk_vector()])
        matches = await store.search(tenant_id, _vector(0), 5)

        assert isinstance(matches[0], VectorMatch)
        assert not hasattr(matches[0], "metadata")


# -- Ordering and normalization ----------------------------------------------
class TestOrdering:
    async def test_matches_come_back_best_first(
        self, store: VectorStore, tenant_id: uuid.UUID
    ) -> None:
        """Position is what `[S1]`…`[S5]` mean — order is part of the citation contract."""
        await store.upsert_embeddings(
            tenant_id, [_chunk_vector(seed) for seed in range(4)]
        )
        query = [0.9, 0.1, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0]

        matches = await store.search(tenant_id, query, 4)
        scores = [match.score for match in matches]
        assert scores == sorted(scores, reverse=True)

    async def test_top_k_is_honoured(
        self, store: VectorStore, tenant_id: uuid.UUID
    ) -> None:
        await store.upsert_embeddings(
            tenant_id, [_chunk_vector(seed) for seed in range(8)]
        )
        assert len(await store.search(tenant_id, _vector(0), 3)) == 3

    @pytest.mark.parametrize(("vector", "top_k"), [([], 5), ([1.0], 0), ([], 0)])
    async def test_a_degenerate_search_returns_nothing_without_a_round_trip(
        self,
        store: VectorStore,
        index: FakePineconeIndex,
        tenant_id: uuid.UUID,
        vector: list[float],
        top_k: int,
    ) -> None:
        assert await store.search(tenant_id, vector, top_k) == []
        assert index.query_calls == []

    def test_a_foreign_vector_id_is_skipped_not_fatal(self) -> None:
        """Something else wrote into our namespace. An operational problem, not a 500.

        Failing the whole query would take DocQuery down for every user because one stray
        vector exists; skipping it degrades that one candidate and answers the question.
        """
        good = uuid.uuid4()
        matches = _read_matches(
            {
                "matches": [
                    {"id": "not-a-uuid", "score": 0.99},
                    {"id": str(good), "score": 0.5},
                ]
            }
        )
        assert [match.chunk_id for match in matches] == [good]

    def test_a_match_missing_its_score_is_skipped(self) -> None:
        assert _read_matches({"matches": [{"id": str(uuid.uuid4())}]}) == []

    def test_an_attribute_style_response_is_read_too(self) -> None:
        """The SDK's response shape varies by client version, so both are handled."""

        class Match:
            def __init__(self, identifier: str, score: float) -> None:
                self.id = identifier
                self.score = score

        class Response:
            def __init__(self, matches: list[Match]) -> None:
                self.matches = matches

        chunk_id = uuid.uuid4()
        matches = _read_matches(Response([Match(str(chunk_id), 0.7)]))
        assert matches == [VectorMatch(chunk_id=chunk_id, score=0.7)]


# -- Failure behaviour -------------------------------------------------------
class TestFailures:
    async def test_an_upsert_failure_raises_so_the_document_never_goes_active(
        self, store: VectorStore, index: FakePineconeIndex, tenant_id: uuid.UUID
    ) -> None:
        """`status='ACTIVE'` is written after this returns, which is what makes raising safe.

        A swallowed failure here would mark a document searchable with no vectors — it would
        answer nothing, and look like a retrieval bug rather than an ingestion one.
        """
        index.fail_with = RuntimeError("503 from the control plane")
        with pytest.raises(VectorStoreUnavailable):
            await store.upsert_embeddings(tenant_id, [_chunk_vector()])

    async def test_a_search_failure_raises_rather_than_returning_no_matches(
        self, store: VectorStore, index: FakePineconeIndex, tenant_id: uuid.UUID
    ) -> None:
        """An empty list would be read as "the documents do not cover this".

        That is a false statement about the corpus. A 503 the user can retry is the honest
        outcome of an index outage.
        """
        index.fail_with = RuntimeError("connection reset by peer")
        with pytest.raises(VectorStoreUnavailable):
            await store.search(tenant_id, _vector(0), 5)

    async def test_a_delete_failure_is_swallowed_because_postgresql_already_decided(
        self, store: VectorStore, index: FakePineconeIndex, tenant_id: uuid.UUID
    ) -> None:
        """The asymmetry with upsert, and the reason for it.

        By the time this runs, `document.status` is already `DELETED` and `load_chunks` will
        refuse to cite the document whether or not its vectors are gone. Raising would roll
        back the status change and leave the document *citable* — strictly worse.
        """
        index.fail_with = RuntimeError("timeout")
        assert await store.delete_document_vectors(
            tenant_id, uuid.uuid4(), [uuid.uuid4()]
        ) == 0

    async def test_deleting_no_chunks_is_a_no_op(
        self, store: VectorStore, index: FakePineconeIndex, tenant_id: uuid.UUID
    ) -> None:
        assert await store.delete_document_vectors(tenant_id, uuid.uuid4(), []) == 0
        assert index.delete_calls == []

    async def test_a_namespace_drop_removes_everything_for_that_tenant_only(
        self, store: VectorStore, index: FakePineconeIndex, settings: Settings
    ) -> None:
        tenant_a, tenant_b = uuid.uuid4(), uuid.uuid4()
        await store.upsert_embeddings(tenant_a, [_chunk_vector(seed) for seed in range(3)])
        await store.upsert_embeddings(tenant_b, [_chunk_vector(9)])

        assert await store.delete_namespace(tenant_a) is True
        assert index.count(settings.pinecone_namespace(tenant_a)) == 0
        assert index.count(settings.pinecone_namespace(tenant_b)) == 1

    async def test_no_api_key_and_no_injected_index_is_unavailable_not_a_crash(
        self, settings: Settings, tenant_id: uuid.UUID
    ) -> None:
        """The test harness blanks `PINECONE_API_KEY`, so this is the harness's own state.

        Reaching the real client from a test would be a network call; it raises instead, which
        is also the correct production behaviour for a misconfigured deployment.
        """
        store = VectorStore(index=None, settings=settings)
        with pytest.raises(VectorStoreUnavailable):
            await store.search(tenant_id, _vector(0), 5)


# -- The import-boundary gate ------------------------------------------------
class TestImportBoundary:
    def test_only_vector_store_imports_pinecone(self) -> None:
        """Phase 4's `grep -rn "import pinecone" backend/` gate, as a test.

        A gate that only exists as a command in a checklist is a gate nobody runs. The rule is
        that the SDK has exactly one point of contact with this codebase — services call the
        store, never the client — so a second importer is an architecture violation and should
        fail the suite, not a review.

        Matched on the start of a stripped line rather than anywhere in the file, which the
        naive `grep` the checklist names would do. A substring search reports this very file,
        because the pattern it searches for is written in it — and a gate whose first result is
        itself gets muted rather than fixed.
        """
        root = Path(__file__).resolve().parents[2]
        offenders = []
        for path in (root / "backend").rglob("*.py"):
            lines = path.read_text(encoding="utf-8").splitlines()
            if any(
                line.lstrip().startswith(("import pinecone", "from pinecone"))
                for line in lines
            ):
                offenders.append(path.relative_to(root).as_posix())

        assert offenders == ["backend/utils/vector_store.py"], (
            "pinecone must be imported only by backend/utils/vector_store.py; "
            f"found {offenders}"
        )
