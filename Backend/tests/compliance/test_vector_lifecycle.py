"""backend/tests/compliance/test_vector_lifecycle.py

Role: PostgreSQL and Pinecone have to agree about what is citable — at every point in a
document's life, including the window where they genuinely disagree.

The two stores hold different halves of one fact
-----------------------------------------------
`document_chunk` holds the passage and PostgreSQL holds the status; Pinecone holds the
embedding. Neither is complete on its own, and `architecture.md` settles the tie in advance:
**PostgreSQL is the authority on `document.status` and ownership**, so a Pinecone match is a
candidate and never a citation. This module asserts that arrangement holds in both directions:

  * *nothing missing* — every chunk of an `ACTIVE` document has a vector in its tenant's
    namespace, and (for documents that went through the real pipeline) `embedded_at` records
    the acknowledgement;
  * *nothing extra* — every vector in a namespace belongs to a chunk row of that tenant;
  * *the disagreement window is safe* — a soft-deleted document is uncitable from the instant
    its status changes, **even when its vectors are still in the index**;
  * *a purge is complete* — `delete_namespace` leaves nothing behind, and leaves the other
    tenant untouched.

The window is the interesting one, and it is tested by breaking the index
------------------------------------------------------------------------
`soft_delete` writes the status first and removes vectors afterwards, best-effort — the
reasoning is in its docstring, and it is deliberately the opposite of the intuitive order. That
choice is only defensible if the status change alone is sufficient, so
`TestTheDisagreementWindowIsSafe` makes the vector deletion *fail* (`FakePineconeIndex.
fail_with`), confirms the vectors are still in the namespace, and then asks a question. The
document must not be cited. If it were, the ordering in `soft_delete` would be a compliance
defect rather than a considered trade-off.

A fixture limit worth knowing before reading a failure
------------------------------------------------------
`Corpus.add` in `backend/tests/fakes.py` stages a document the way ingestion would — chunk
rows, then vectors, then `ACTIVE` — but it does **not** call `mark_chunks_embedded`, so
`build_stack()`'s two seeded documents have `embedded_at = None`. That is a property of the
harness, not of the product. So the vector-presence check runs over every `ACTIVE` document
(the seeded ones do have vectors), while the `embedded_at` check runs only over documents this
module uploaded through the real pipeline. Widening the second check would report a fixture gap
as a product defect, which is the failure mode `docs/RELEASE.md` asks these suites to avoid.

Scope: this is the *agreement* check. Namespace isolation between tenants — that a wrong
namespace returns nothing rather than someone else's passages — is
`backend/tests/security/test_vector_isolation.py`, and it is not re-argued here.
"""

from __future__ import annotations

import uuid

from backend.models.audit_log import AuditAction
from backend.models.document import RETRIEVABLE_STATUSES, DocumentStatus
from backend.models.query import QueryStatus
from backend.tests.compliance.conftest import (
    API,
    SURCHARGE_QUESTION,
    TAX_QUESTION,
    Api,
    Stack,
    ask,
    csrf_headers,
    switch_to,
    upload,
)
from backend.tests.harness import ROOT
from backend.utils.vector_store import VectorStore

#: Raised by the fake index when a test needs a Pinecone outage. A plain exception type, since
#: `delete_document_vectors` catches broadly on purpose — the SDK raises many unrelated types.
INDEX_DOWN = RuntimeError("pinecone is unreachable")


def namespace_ids(stack: Stack, tenant_id: uuid.UUID) -> set[str]:
    """Every vector id in one tenant's namespace, as the index holds them."""
    return stack.index.ids_in(stack.namespace(tenant_id))


def chunk_ids_of(stack: Stack, document_id: uuid.UUID) -> set[str]:
    """The chunk ids stored for one document, as vector ids would be written."""
    return {
        str(chunk.id) for chunk in stack.documents.chunks if chunk.document_id == document_id
    }


def presence_problems(stack: Stack, tenant_id: uuid.UUID) -> list[str]:
    """Every `ACTIVE` chunk in this tenant must have a vector. Empty list means it agrees.

    A missing vector is the state ingestion is ordered to never leave behind: a document that
    is listed, looks searchable, and matches nothing — answering every question about itself
    with "not found" while appearing healthy.
    """
    present = namespace_ids(stack, tenant_id)
    problems: list[str] = []
    for document in stack.documents.documents.values():
        if document.tenant_id != tenant_id or document.status not in RETRIEVABLE_STATUSES:
            continue
        missing = chunk_ids_of(stack, document.id) - present
        if missing:
            problems.append(
                f"{document.name} is {document.status.value} with {len(missing)} "
                "chunk(s) that have no vector"
            )
    return problems


def orphan_problems(stack: Stack, tenant_id: uuid.UUID) -> list[str]:
    """Every vector in this tenant's namespace must belong to a chunk row of this tenant.

    The other direction, and the one that catches a delete that only half happened. An orphan
    vector cannot be cited — `load_chunks` drops a match with no row — but it is still a
    paid-for embedding of customer content in a third-party index with no record behind it.
    """
    rows = {str(chunk.id) for chunk in stack.documents.chunks if chunk.tenant_id == tenant_id}
    orphans = namespace_ids(stack, tenant_id) - rows
    if not orphans:
        return []
    return [f"{len(orphans)} vector(s) with no chunk row: {sorted(orphans)[:3]}"]


def acknowledged_problems(stack: Stack, document_id: uuid.UUID) -> list[str]:
    """`embedded_at` must be set on every chunk of a pipeline-ingested document.

    Restricted to one document id by design — see the module docstring on `Corpus.add`.
    """
    return [
        f"chunk {chunk.chunk_index} has no embedded_at"
        for chunk in stack.documents.chunks
        if chunk.document_id == document_id and chunk.embedded_at is None
    ]


def delete_rows(stack: Stack, document_id: str) -> list[dict]:
    """The `DOCUMENT_DELETE` audit details for one document, oldest first."""
    return [
        entry.detail or {}
        for entry in stack.audit_repo.entries
        if entry.action is AuditAction.DOCUMENT_DELETE
        and str(entry.object_id) == str(document_id)
    ]


class TestTheTwoStoresAgree:
    """Nothing missing, nothing extra — in both tenants, before and after activity."""

    async def test_the_seeded_stack_agrees(self, stack: Stack) -> None:
        """Asserted first, on an untouched stack, so a later failure is attributable."""
        for tenant_id in (stack.a.id, stack.b.id):
            assert presence_problems(stack, tenant_id) == []
            assert orphan_problems(stack, tenant_id) == []

    async def test_an_uploaded_document_has_a_vector_for_every_chunk(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)

        document_id = uuid.UUID(document["id"])
        chunks = chunk_ids_of(stack, document_id)
        assert chunks, "ingestion produced no chunk rows"
        assert chunks <= namespace_ids(stack, stack.a.id)
        assert presence_problems(stack, stack.a.id) == []
        assert orphan_problems(stack, stack.a.id) == []

    async def test_an_active_document_has_its_embedding_acknowledged(
        self, api: Api, stack: Stack
    ) -> None:
        """`embedded_at` is written *after* the upsert returns and *before* the status becomes
        `ACTIVE`. It is the row-level record that the vector exists, and the only evidence
        available to a reviewer who cannot query Pinecone."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)

        assert document["status"] == DocumentStatus.ACTIVE.value
        assert acknowledged_problems(stack, uuid.UUID(document["id"])) == []

    async def test_the_vectors_land_in_the_uploader_s_own_namespace(
        self, api: Api, stack: Stack
    ) -> None:
        """The namespace is the isolation boundary, so the upsert has to name the right one.
        Tenant B's namespace must be untouched by tenant A's upload."""
        before = stack.index.count(stack.namespace(stack.b.id))

        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)

        staged = chunk_ids_of(stack, uuid.UUID(document["id"]))
        assert staged <= namespace_ids(stack, stack.a.id)
        assert stack.index.count(stack.namespace(stack.b.id)) == before
        assert {call[0] for call in stack.index.upsert_calls} == {
            stack.namespace(stack.a.id),
            stack.namespace(stack.b.id),
        }

    async def test_both_stores_still_agree_after_a_delete_and_a_restore(
        self, api: Api, stack: Stack
    ) -> None:
        """A restore is a re-ingest, so it writes *new* chunk rows with new ids. The count has
        to come back to where it started: an accumulating namespace would mean the old vectors
        were never removed, and every restore would leave another copy behind."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        settled = stack.index.count(stack.namespace(stack.a.id))

        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
        restored = await admin.post(
            f"{API}/documents/{document['id']}/restore", headers=csrf_headers()
        )
        assert restored.status_code == 200, restored.text

        assert restored.json()["status"] == DocumentStatus.ACTIVE.value
        assert stack.index.count(stack.namespace(stack.a.id)) == settled
        assert presence_problems(stack, stack.a.id) == []
        assert orphan_problems(stack, stack.a.id) == []
        assert acknowledged_problems(stack, uuid.UUID(document["id"])) == []

    async def test_a_restored_document_is_answerable_again(
        self, api: Api, stack: Stack
    ) -> None:
        """The point of re-ingesting rather than flipping the status. A restored document that
        matched nothing would be listed, look healthy, and answer nothing."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
        await admin.post(
            f"{API}/documents/{document['id']}/restore", headers=csrf_headers()
        )

        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        assert body["status"] == QueryStatus.ANSWERED.value
        assert {entry["document_id"] for entry in body["source_references"]} == {
            document["id"]
        }


class TestASoftDeleteRemovesTheVectors:
    """The normal path: status first, then the index is cleared."""

    async def test_the_documents_vectors_leave_the_namespace(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        document_id = uuid.UUID(document["id"])
        staged = chunk_ids_of(stack, document_id)
        assert staged <= namespace_ids(stack, stack.a.id)

        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        assert not staged & namespace_ids(stack, stack.a.id)

    async def test_the_chunk_rows_are_kept(self, api: Api, stack: Stack) -> None:
        """BR-01: a soft delete erases nothing. The rows are what make a restore a re-embed
        rather than a re-parse, and what a reviewer reads to see what the document said."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        document_id = uuid.UUID(document["id"])
        staged = chunk_ids_of(stack, document_id)

        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        assert chunk_ids_of(stack, document_id) == staged

    async def test_the_audit_row_records_how_many_vectors_went(
        self, api: Api, stack: Stack
    ) -> None:
        """"Delete" in a compliance review must not be read as "erased", so the row says
        `soft_delete: true` *and* how much of the index was cleared."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        expected = len(chunk_ids_of(stack, uuid.UUID(document["id"])))

        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        rows = delete_rows(stack, document["id"])
        assert len(rows) == 1
        assert rows[0]["soft_delete"] is True
        assert rows[0]["vectors_removed"] == expected

    async def test_the_other_tenants_namespace_is_untouched(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        before = namespace_ids(stack, stack.b.id)

        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        assert namespace_ids(stack, stack.b.id) == before
        assert {call[0] for call in stack.index.delete_calls} == {
            stack.namespace(stack.a.id)
        }

    async def test_a_second_delete_clears_nothing_twice(
        self, api: Api, stack: Stack
    ) -> None:
        """`soft_delete` is idempotent. A second call must not write a second audit row
        claiming a second deletion, nor issue a second index call."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
        calls = len(stack.index.delete_calls)

        again = await admin.delete(
            f"{API}/documents/{document['id']}", headers=csrf_headers()
        )

        assert again.status_code == 200
        assert len(delete_rows(stack, document["id"])) == 1
        assert len(stack.index.delete_calls) == calls


class TestTheDisagreementWindowIsSafe:
    """The claim `soft_delete`'s ordering rests on: PostgreSQL alone is sufficient.

    Each test here leaves the vectors *in place* — by breaking the index — and then asserts the
    document is already uncitable. This is the window a real deployment lives in for the
    seconds between the status write and Pinecone's eventual consistency, and the reason
    `PINECONE_OVERFETCH_FACTOR` exists.
    """

    async def test_a_deleted_document_is_uncitable_while_its_vectors_remain(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        staged = chunk_ids_of(stack, uuid.UUID(document["id"]))

        stack.index.fail_with = INDEX_DOWN
        deleted = await admin.delete(
            f"{API}/documents/{document['id']}", headers=csrf_headers()
        )
        assert deleted.status_code == 200, deleted.text
        stack.index.fail_with = None

        # The premise: the vectors really are still there. Without this the test would pass
        # for the ordinary reason and prove nothing about the window.
        assert staged <= namespace_ids(stack, stack.a.id)
        assert delete_rows(stack, document["id"])[0]["vectors_removed"] == 0

        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        assert document["id"] not in {
            entry["document_id"] for entry in body["source_references"]
        }
        # Nothing else in this tenant covers withholding tax, so a refusal is the only honest
        # answer left.
        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert staged <= namespace_ids(stack, stack.a.id), "the vectors were the reason"

    async def test_the_index_outage_does_not_block_the_delete(
        self, api: Api, stack: Stack
    ) -> None:
        """A raise from the vector deletion would roll back the status change and leave the
        document *citable* — strictly the worse outcome, which is why it is swallowed."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)

        stack.index.fail_with = INDEX_DOWN
        response = await admin.delete(
            f"{API}/documents/{document['id']}", headers=csrf_headers()
        )
        stack.index.fail_with = None

        assert response.status_code == 200
        assert response.json()["status"] == DocumentStatus.DELETED.value
        assert (
            stack.documents.documents[uuid.UUID(document["id"])].status
            is DocumentStatus.DELETED
        )

    async def test_a_leftover_vector_cannot_answer_for_another_document(
        self, api: Api, stack: Stack
    ) -> None:
        """The generalisation: with the tax document deleted but its vectors present, the
        questions it used to answer must not be answered from another document either. A
        retrieval layer that fell back to the next-best match would produce a confident
        citation of an unrelated page."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)

        stack.index.fail_with = INDEX_DOWN
        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
        stack.index.fail_with = None

        user = switch_to(api, stack.a.user)
        for question in (TAX_QUESTION, SURCHARGE_QUESTION):
            body = await ask(user, stack, question, "Stated in the schedule [S1].")
            assert body["status"] == QueryStatus.NOT_FOUND.value, question
            assert body["source_references"] == [], question


class TestAPurgeLeavesNothingBehind:
    """`delete_namespace` — offboarding. The one operation that is supposed to erase."""

    async def test_the_namespace_is_empty_afterwards(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        assert namespace_ids(stack, stack.a.id)

        assert await stack.vectors.delete_namespace(stack.a.id) is True

        assert namespace_ids(stack, stack.a.id) == set()
        assert stack.index.count(stack.namespace(stack.a.id)) == 0

    async def test_the_other_tenant_survives_the_purge(
        self, api: Api, stack: Stack
    ) -> None:
        """An offboarding that took a second tenant's embeddings with it would be unrecoverable
        — the vectors are derived, but re-deriving them means re-embedding every document."""
        before = namespace_ids(stack, stack.b.id)
        assert before

        await stack.vectors.delete_namespace(stack.a.id)

        assert namespace_ids(stack, stack.b.id) == before
        assert stack.index.total() == len(before)

    async def test_nothing_is_answerable_after_a_purge(
        self, api: Api, stack: Stack
    ) -> None:
        """The purge is complete when the tenant's questions stop being answerable. Their
        documents and chunk rows are still in PostgreSQL — erasing those is a retention
        decision (C-05), not this method's job."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        await stack.vectors.delete_namespace(stack.a.id)

        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert stack.documents.documents[uuid.UUID(document["id"])].status is (
            DocumentStatus.ACTIVE
        )
        assert chunk_ids_of(stack, uuid.UUID(document["id"]))

    def test_no_route_can_purge_a_namespace(self) -> None:
        """"Not reachable from any route" is a claim in `vector_store.delete_namespace`'s
        docstring, and this is what keeps it true. Deleting every embedding a tenant owns
        behind an HTTP endpoint would be an IDOR target with no legitimate caller."""
        routes = sorted(ROOT.joinpath("backend", "api").rglob("*.py"))
        assert routes, "no API modules found — the scan would pass over an empty tree"
        for path in routes:
            assert "delete_namespace" not in path.read_text(encoding="utf-8"), path

    def test_the_purge_is_still_available_to_an_operator(self) -> None:
        """The other half of the test above. "No route calls it" is also satisfiable by
        deleting the capability, which would leave offboarding with no implementation."""
        assert callable(getattr(VectorStore, "delete_namespace", None))
