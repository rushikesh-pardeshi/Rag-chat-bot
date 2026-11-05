"""backend/utils/vector_store.py

Role: The only module that talks to Pinecone. Every embedding in DocQuery is written and
read through here; services call these methods and never the SDK
(`.claude/rules/architecture.md`).

Four decisions, already made
----------------------------
1. **One namespace per tenant**, built solely by `Settings.pinecone_namespace(tenant_id)`.
   Isolation is the namespace, never a metadata filter. The difference is the failure mode:
   a wrong namespace returns nothing, while a forgotten filter returns another tenant's
   chunks. One is a visible bug, the other is a data breach that looks like a working
   feature.
2. **The vector id is `str(chunk.id)`.** Re-processing a document therefore overwrites its
   own vectors instead of accumulating duplicates, which is what makes the Celery task safe
   to retry.
3. **PostgreSQL stays the authority on `document.status` and ownership.** A match from here
   is a *candidate*. `query_repository.load_chunks` re-checks tenant and `ACTIVE` status
   before anything can be cited, so a soft-deleted document is unquotable immediately rather
   than whenever its vectors finish leaving the index. `PINECONE_OVERFETCH_FACTOR` pays for
   the candidates that re-check discards.
4. **Metadata carries citations, never document text.** The passage lives in
   `document_chunk.text`. A third copy of customer content in a system with its own access
   model is what `secure-coding-vapt.md` forbids, and `_build_metadata` raises rather than
   trusting callers to remember.

A class, not four bare functions
--------------------------------
`WORKBREAKDOWN.md` describes four functions and this file provides them as four methods on
one object. The reason is the test suite: `PINECONE_API_KEY` is blanked in the harness and
no test may reach the network, so the index has to be *injectable*. Module-level functions
holding a private global would have to be monkeypatched, and a monkeypatched global is
exactly the shape that lets one test leak state into the next.

Corporate TLS interception
--------------------------
On a network that terminates TLS, `api.pinecone.io` fails to verify because Python trusts
only certifi's roots while the interception root lives in the OS store. The fix is process
wide, not per-client: run `python scripts/export_ca_bundle.py` and point `SSL_CERT_FILE` at
the result. `pinecone==9.1.0` accepts an `ssl_ca_certs` argument but does not apply it —
verified against a bundle `ssl.create_default_context(cafile=...)` accepts and the Pinecone
client still rejects — so passing it here would look like a fix and be none.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any, Final, Protocol

from backend.core.config import Settings, get_settings
from backend.core.exceptions import RetrievalUnavailable
from backend.core.logging import get_logger

logger = get_logger(__name__)

#: Metadata keys allowed on a vector. Anything else is dropped, and `text` specifically
#: raises — see decision 4. This is an allowlist rather than a denylist because a denylist
#: has to anticipate every name someone might give a passage field ("body", "content",
#: "passage", "excerpt"), and it only has to miss once.
ALLOWED_METADATA_KEYS: Final = frozenset(
    {"document_id", "document_name", "page_number", "section_label", "chunk_index"}
)

#: Metadata keys that indicate a caller is about to write document body text. Rejected
#: loudly, because the alternative is a silent compliance breach.
FORBIDDEN_METADATA_KEYS: Final = frozenset(
    {"text", "content", "body", "passage", "excerpt", "chunk_text"}
)


class VectorStoreUnavailable(RetrievalUnavailable):
    """Pinecone could not be reached. 503, and the SDK's message never reaches the client."""

    code = "vector_store_unavailable"


class ForbiddenVectorMetadata(ValueError):
    """Raised when a caller tries to put document text into vector metadata.

    A programming error, not a request error, so it is a plain `ValueError` and not a
    `DocQueryError`: no HTTP status can make this the caller's problem. It should fail the
    test suite, which is the point.
    """


@dataclass(frozen=True, slots=True)
class ChunkVector:
    """One chunk paired with its embedding, ready to upsert.

    Holds the citation fields explicitly rather than a `DocumentChunk` so this module never
    touches the ORM — and so a reader can see, at the call site, exactly which fields cross
    the boundary into Pinecone.
    """

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    document_name: str
    vector: list[float]
    page_number: int | None = None
    section_label: str = ""
    chunk_index: int = 0


@dataclass(frozen=True, slots=True)
class VectorMatch:
    """A retrieval candidate: which chunk, and how similar.

    Deliberately not the chunk itself. This module cannot know whether the chunk is still
    citable — that is a PostgreSQL question — so it returns an id and a score and stops.
    """

    chunk_id: uuid.UUID
    score: float


class AsyncIndex(Protocol):
    """The slice of Pinecone's async index API this module uses.

    Declared as a Protocol so `FakePineconeIndex` in `backend/tests/fakes.py` is checked
    against the same shape the real client has to satisfy.
    """

    async def upsert(self, *, vectors: list[dict[str, Any]], namespace: str) -> Any: ...

    async def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str,
        include_metadata: bool = False,
        include_values: bool = False,
    ) -> Any: ...

    async def delete(
        self,
        *,
        ids: list[str] | None = None,
        namespace: str = "",
        delete_all: bool = False,
    ) -> Any: ...


class VectorStore:
    """Pinecone access, scoped by tenant on every call.

    `index` may be injected; when it is not, the real async client is built on first use so
    that importing this module never needs an API key.
    """

    def __init__(
        self, *, index: AsyncIndex | None = None, settings: Settings | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._index = index
        self._client: Any = None
        # Guards lazy construction: two concurrent queries on a cold process would
        # otherwise each build a client, and one of them would be leaked.
        self._lock = asyncio.Lock()

    # -- Connection ---------------------------------------------------------
    async def _get_index(self) -> AsyncIndex:
        if self._index is not None:
            return self._index

        async with self._lock:
            if self._index is not None:  # settled while we waited for the lock
                return self._index
            if not self._settings.PINECONE_API_KEY:
                # Reached only if a service tries to retrieve with no key configured —
                # in the test harness the key is blank and a fake index is injected.
                logger.error("vector_store.no_api_key")
                raise VectorStoreUnavailable()
            try:
                from pinecone import PineconeAsyncio

                self._client = PineconeAsyncio(
                    api_key=self._settings.PINECONE_API_KEY.get_secret_value(),
                )
                # The host is per-index and stable; resolving it once avoids a control-plane
                # round trip on every data-plane call.
                description = await self._client.describe_index(
                    self._settings.PINECONE_INDEX_NAME
                )
                self._index = self._client.IndexAsyncio(host=description.host)
            except Exception as exc:  # noqa: BLE001 - SDK raises many unrelated types
                logger.error(
                    "vector_store.connect_failed", error_type=type(exc).__name__
                )
                raise VectorStoreUnavailable() from exc
        return self._index

    async def close(self) -> None:
        """Release the SDK's connection pool. Called from the app's lifespan shutdown."""
        for candidate in (self._index, self._client):
            closer = getattr(candidate, "close", None)
            if closer is None:
                continue
            try:
                result = closer()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.warning("vector_store.close_failed")
        self._index = None
        self._client = None

    # -- Write --------------------------------------------------------------
    async def upsert_embeddings(
        self, tenant_id: uuid.UUID, chunk_vectors: list[ChunkVector]
    ) -> int:
        """Write vectors into the tenant's namespace. Returns how many were sent.

        Batched at `PINECONE_UPSERT_BATCH_SIZE` because Pinecone caps a single upsert
        request by payload size, and a 200-chunk document at 1024 float32 dimensions is well
        past it.

        Not transactional: a failure partway leaves earlier batches written. That is
        deliberate and safe, because `document.status` only becomes `ACTIVE` after this
        returns — chunks whose vectors landed belong to a document that is not yet
        retrievable, and a retry overwrites them by id rather than duplicating them.
        """
        if not chunk_vectors:
            return 0

        namespace = self._settings.pinecone_namespace(tenant_id)
        index = await self._get_index()
        batch_size = max(1, self._settings.PINECONE_UPSERT_BATCH_SIZE)
        sent = 0

        for start in range(0, len(chunk_vectors), batch_size):
            batch = chunk_vectors[start : start + batch_size]
            payload = [
                {
                    "id": str(item.chunk_id),
                    "values": item.vector,
                    "metadata": _build_metadata(item),
                }
                for item in batch
            ]
            try:
                await index.upsert(vectors=payload, namespace=namespace)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "vector_store.upsert_failed",
                    error_type=type(exc).__name__,
                    namespace=namespace,
                    batch_size=len(batch),
                )
                raise VectorStoreUnavailable() from exc
            sent += len(batch)

        logger.info("vector_store.upserted", namespace=namespace, vectors=sent)
        return sent

    # -- Read ---------------------------------------------------------------
    async def search(
        self, tenant_id: uuid.UUID, vector: list[float], top_k: int
    ) -> list[VectorMatch]:
        """Nearest neighbours in the tenant's namespace, best score first.

        No filter argument, by design. Adding one would create a second isolation mechanism
        alongside the namespace, and a caller could then believe a filter was protecting
        them when it was optional. There is exactly one way to scope a search here.
        """
        if not vector or top_k <= 0:
            return []

        namespace = self._settings.pinecone_namespace(tenant_id)
        index = await self._get_index()
        try:
            response = await index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                # Metadata is not requested: the citation is read from PostgreSQL, which is
                # authoritative and is being consulted anyway to re-check status. Trusting
                # index metadata would let a stale document name be cited after a rename.
                include_metadata=False,
                include_values=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "vector_store.query_failed",
                error_type=type(exc).__name__,
                namespace=namespace,
            )
            raise VectorStoreUnavailable() from exc

        matches = _read_matches(response)
        logger.info(
            "vector_store.searched",
            namespace=namespace,
            requested=top_k,
            returned=len(matches),
        )
        return matches

    # -- Delete -------------------------------------------------------------
    async def delete_document_vectors(
        self,
        tenant_id: uuid.UUID,
        document_id: uuid.UUID,
        chunk_ids: list[uuid.UUID],
    ) -> int:
        """Remove one document's vectors from the tenant's namespace.

        `chunk_ids` is a parameter, not something this method derives, because **Pinecone
        serverless cannot delete by metadata filter** — that is a pod-based feature. The ids
        come from `document_repository.get_chunks`, which is also the only place that knows
        which chunks a document has. `document_id` is still taken, purely so the log line
        says which document was cleared.

        A failure here is logged and swallowed rather than raised. By the time this is
        called, `document.status` is already `DELETED`, and PostgreSQL is the authority:
        `load_chunks` will refuse to cite the document whether or not its vectors are gone.
        Raising would roll back the status change and leave the document *citable*, which is
        the strictly worse outcome.
        """
        if not chunk_ids:
            return 0

        namespace = self._settings.pinecone_namespace(tenant_id)
        batch_size = max(1, self._settings.PINECONE_UPSERT_BATCH_SIZE)
        deleted = 0
        try:
            index = await self._get_index()
            for start in range(0, len(chunk_ids), batch_size):
                batch = chunk_ids[start : start + batch_size]
                await index.delete(
                    ids=[str(chunk_id) for chunk_id in batch], namespace=namespace
                )
                deleted += len(batch)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vector_store.delete_failed",
                error_type=type(exc).__name__,
                document_id=str(document_id),
                namespace=namespace,
                deleted_before_failure=deleted,
            )
            return deleted

        logger.info(
            "vector_store.deleted",
            document_id=str(document_id),
            namespace=namespace,
            vectors=deleted,
        )
        return deleted

    async def delete_namespace(self, tenant_id: uuid.UUID) -> bool:
        """Drop a tenant's entire namespace — offboarding and test teardown.

        Not reachable from any route. Deleting every embedding a tenant owns is an
        operational action, and putting it behind an HTTP endpoint would make it an IDOR
        target with no legitimate caller.
        """
        namespace = self._settings.pinecone_namespace(tenant_id)
        try:
            index = await self._get_index()
            await index.delete(delete_all=True, namespace=namespace)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vector_store.delete_namespace_failed",
                error_type=type(exc).__name__,
                namespace=namespace,
            )
            return False
        logger.info("vector_store.namespace_deleted", namespace=namespace)
        return True


def _build_metadata(item: ChunkVector) -> dict[str, Any]:
    """Citation fields only. Raises if a forbidden key is somehow present.

    The check looks redundant given that this function builds the dict itself — and it is,
    today. It is here because the *next* change to `ChunkVector` is the one that adds a
    `text` field for convenience, and this raises in the test suite the moment that happens
    rather than shipping customer document text to a third-party index.
    """
    metadata: dict[str, Any] = {
        "document_id": str(item.document_id),
        "document_name": item.document_name,
        "chunk_index": item.chunk_index,
    }
    if item.page_number is not None:
        metadata["page_number"] = item.page_number
    if item.section_label:
        metadata["section_label"] = item.section_label

    forbidden = FORBIDDEN_METADATA_KEYS & metadata.keys()
    if forbidden:
        raise ForbiddenVectorMetadata(
            f"Vector metadata may not contain document text: {sorted(forbidden)}"
        )
    unexpected = metadata.keys() - ALLOWED_METADATA_KEYS
    if unexpected:
        raise ForbiddenVectorMetadata(
            f"Vector metadata keys not on the allowlist: {sorted(unexpected)}"
        )
    return metadata


def _read_matches(response: Any) -> list[VectorMatch]:
    """Normalize a Pinecone query response into `VectorMatch` objects.

    The SDK returns an object that also behaves like a mapping, and which shape you get
    depends on the client version. Both are handled, and a match whose id is not a UUID is
    skipped rather than raising: a foreign vector in the namespace is an operational problem,
    not a reason to fail a user's query.
    """
    raw = response.get("matches") if hasattr(response, "get") else None
    if raw is None:
        raw = getattr(response, "matches", None) or []

    matches: list[VectorMatch] = []
    for entry in raw:
        identifier = entry.get("id") if hasattr(entry, "get") else getattr(entry, "id", None)
        score = entry.get("score") if hasattr(entry, "get") else getattr(entry, "score", None)
        if identifier is None or score is None:
            continue
        try:
            chunk_id = uuid.UUID(str(identifier))
        except ValueError:
            logger.warning("vector_store.non_uuid_vector_id")
            continue
        matches.append(VectorMatch(chunk_id=chunk_id, score=float(score)))

    # Pinecone already returns best-first; sorting makes the contract independent of that.
    matches.sort(key=lambda match: match.score, reverse=True)
    return matches


_vector_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    """The process-wide store. Overridden in tests via a FastAPI dependency.

    A module global rather than `functools.lru_cache` because `close()` has to be able to
    reset it during shutdown, and a cached function's state is not reachable for that.
    """
    global _vector_store  # noqa: PLW0603 - single process-wide connection pool
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


async def dispose_vector_store() -> None:
    """Close the shared store. Called from the app lifespan alongside `dispose_engine`."""
    global _vector_store  # noqa: PLW0603
    if _vector_store is not None:
        await _vector_store.close()
        _vector_store = None
