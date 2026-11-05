"""backend/tests/security/test_vector_isolation.py

Role: prove that one tenant's embeddings are unreachable from another tenant's question, that
the namespace is the only thing standing between them, and that PostgreSQL — not the index —
decides what may be quoted.

Why this module exists separately from `test_idor.py`
----------------------------------------------------
Every other authorization boundary in this system is a `WHERE` clause a reviewer can read.
This one is not. Retrieval leaves PostgreSQL entirely, asks a third-party index for nearest
neighbours, and comes back with ids. If the scoping on that round trip is wrong, no route is
missing a check and no repository is missing a filter — the answer is simply built from
someone else's documents, and it looks like a working feature. `architecture.md` is explicit
about the shape of the defence:

    *"**One namespace per tenant**, built solely by `Settings.pinecone_namespace(tenant_id)`.
    Isolation is by namespace, never by metadata filter — a wrong namespace returns nothing, a
    forgotten filter returns another tenant's chunks."*

The two failure modes are not symmetric, and that asymmetry is the whole design. A wrong
namespace is a visible bug: the user gets "Information not found in uploaded documents" and
files a ticket. A forgotten filter is a breach that reports success. So the tests below are
written to fail loudly in the first direction and are structured to be *capable* of catching
the second.

Why the fixture's two corpora are worded almost identically
-----------------------------------------------------------
`conftest.py` gives tenant A "the outbound domestic wire transfer fee is 25 USD … at Tenant A
Bank" and tenant B the same sentence with 40 USD and Tenant B. That is deliberate. If
isolation were by metadata filter over one shared namespace, A's question would rank B's
passage at nearly the same score as its own, and the filter would be the only thing between
them — so a dropped filter would surface immediately as a cross-tenant citation. With separate
namespaces the match cannot happen at all, and `SHARED_QUESTION` is the question that tells the
two designs apart. A test corpus about unrelated topics would pass under either.

Three independent claims, three sections
----------------------------------------
1. **The namespace is the only isolation mechanism** — asserted statically. `search()` takes no
   filter argument, nothing outside `config.py` builds a namespace string, and every index call
   made while serving a request carried one.
2. **A tenant retrieves only its own vectors** — asserted behaviourally, over the real
   `VectorStore` and the real `RagService`, with the second tenant's near-identical passage
   sitting there to be wrongly cited.
3. **PostgreSQL is the authority on citability** — a vector that is still in the index is not a
   citation. Soft-deleted document, foreign chunk row, below-threshold score: each is dropped
   after the index has already returned it.

What the fake can and cannot prove
----------------------------------
`FakePineconeIndex` keeps one dict per namespace and ranks by real cosine similarity. It can
therefore prove that our code asks the right namespace and uses what comes back correctly, and
it can prove metadata never carries document text. It cannot prove Pinecone honours namespace
isolation on its side — that is the vendor's guarantee, and the only thing that would test it
is a live index with two namespaces, which needs the credentials recorded as blocked in
`docs/RELEASE.md`. What is under test here is every line of the boundary this project owns.

Positive controls
-----------------
"Tenant A cannot see tenant B's passage" also passes when retrieval is broken for everyone, so
`TestBothTenantsCanActuallyRetrieve` runs first: each tenant must get a grounded answer citing
its *own* document, with its own number in it. Every negative assertion below rests on it.
"""

from __future__ import annotations

import ast
import inspect
import re
import uuid
from collections.abc import Callable

import httpx
import pytest

from backend.core.config import get_settings
from backend.models.document import DocumentStatus
from backend.models.query import QueryStatus
from backend.models.user import User, UserRole
from backend.services.source_tracer import NOT_FOUND_ANSWER
from backend.tests.document_fixtures import make_pdf
from backend.tests.fakes import Corpus
from backend.tests.security.conftest import (
    API,
    CONTEXT,
    ROOT,
    SHARED_QUESTION,
    Stack,
    backend_modules,
    csrf_headers,
)
from backend.utils.vector_store import (
    ALLOWED_METADATA_KEYS,
    FORBIDDEN_METADATA_KEYS,
    VectorStore,
)

Api = tuple[httpx.AsyncClient, Callable[[User | None], None]]

#: `ROOT` and `backend_modules()` come from `conftest`, where `test_path_traversal.py` also
#: reads them. They started here, and moved once a second module needed the same tree: a
#: structural claim is only as good as the tree it walked, and two copies of the walk is two
#: places for it to silently become empty. `test_the_static_scans_are_reaching_the_source`
#: below is the trap, and it now guards both modules.
#:
#: A fact only tenant B's corpus can support, so a not-found from tenant A is unambiguous.
#: Phrased as a question about a named counterparty rather than about a fee, because a
#: fee question would legitimately match A's own fee passages and prove nothing.
B_ONLY_QUESTION = "What is the settlement cut-off time for the Tenant B Trust escrow ledger?"
B_ONLY_PASSAGE = (
    "The settlement cut-off time for the Tenant B Trust escrow ledger is 16:30 GMT, after "
    "which instructions are carried to the next business day."
)


class TestBothTenantsCanActuallyRetrieve:
    """The positive control for the whole module.

    Every isolation assertion here is of the form "X does not appear". That is also true of a
    pipeline that retrieves nothing, embeds to zeros, or raises and is swallowed. So before any
    of it: both tenants ask the same question, and both must get a grounded answer citing their
    own document — with their own figure in it, which is the part a shared namespace would get
    wrong.
    """

    async def test_each_tenant_gets_its_own_figure(self, stack: Stack) -> None:
        for tenant, own_fee in ((stack.a, "25"), (stack.b, "40")):
            passage = next(p for p in tenant.passages if own_fee in p)
            stack.bedrock.queue_answer(f"{passage} [S1]")
            query = await stack.query_service.submit(
                question=SHARED_QUESTION, actor=tenant.user, context=CONTEXT
            )
            assert query.status is QueryStatus.ANSWERED, query.answer_text
            assert query.source_references, "a grounded answer must carry a citation"
            assert own_fee in (query.answer_text or "")

    async def test_the_index_holds_both_corpora_in_separate_namespaces(
        self, stack: Stack
    ) -> None:
        """Both namespaces are populated, and neither is a subset of the other.

        If the fixture had written both tenants into one namespace, the isolation tests below
        would be asserting against a corpus that never existed.
        """
        namespace_a = stack.namespace(stack.a.id)
        namespace_b = stack.namespace(stack.b.id)
        assert namespace_a != namespace_b
        assert stack.index.count(namespace_a) == len(stack.a.passages)
        assert stack.index.count(namespace_b) == len(stack.b.passages)
        assert not stack.index.ids_in(namespace_a) & stack.index.ids_in(namespace_b)
        assert stack.index.total() == len(stack.a.passages) + len(stack.b.passages)


class TestTheNamespaceIsTheOnlyIsolation:
    """Static assertions about the shape of the boundary, not about one request's behaviour.

    A behavioural test proves the namespace was right on the paths it exercised. These prove
    there is no second mechanism to get wrong and no other way to build the string — which is
    what makes the behavioural result generalize to the paths no test ran.
    """

    def test_the_static_scans_are_reaching_the_source(self) -> None:
        """The non-vacuity control for every scan in this class.

        `backend_modules()` walking an empty tree makes "no module builds its own namespace"
        and "nothing references `delete_namespace`" both pass by finding nothing at all. That
        is not hypothetical — the first version of this module rooted the walk one directory
        too deep and three tests went green against zero files. Only the one with an explicit
        floor on what it found reported it.
        """
        modules = backend_modules()
        names = {path.name for path in modules}
        assert len(modules) > 40, f"only {len(modules)} modules found under {ROOT / 'backend'}"
        for required in ("vector_store.py", "config.py", "rag_service.py"):
            assert required in names, f"{required} is not in the scanned set"
        assert not any("tests" in path.parts for path in modules)

    def test_search_accepts_no_filter_argument(self) -> None:
        """A filter parameter is the defect, not its misuse.

        `VectorStore.search` deliberately has no `filter`: offering one would create a second
        isolation mechanism beside the namespace, and a caller could then believe a filter was
        protecting them when passing it was optional. The `AsyncIndex` protocol is checked too,
        so the capability is not merely unused but unavailable through this module's own
        interface.
        """
        parameters = set(inspect.signature(VectorStore.search).parameters)
        assert parameters == {"self", "tenant_id", "vector", "top_k"}, parameters

        source = inspect.getsource(VectorStore)
        for forbidden in ("filter=", "metadata_filter", '"filter"', "'filter'"):
            assert forbidden not in source, (
                f"{forbidden!r} appears in VectorStore — isolation must be the namespace "
                "alone; see .claude/rules/architecture.md"
            )

    def test_only_config_builds_a_namespace_string(self) -> None:
        """`pinecone_namespace()` is the single producer, so there is one place to audit.

        The prefix is what a hand-rolled namespace would have to reference. Any module other
        than `config.py` naming `PINECONE_NAMESPACE_PREFIX` is building the string itself, and
        a second construction site is a second chance to interpolate the wrong id.
        """
        offenders = [
            path.relative_to(ROOT).as_posix()
            for path in backend_modules()
            if "PINECONE_NAMESPACE_PREFIX" in path.read_text(encoding="utf-8")
            and path.name != "config.py"
        ]
        assert offenders == [], offenders

    def test_every_namespace_argument_comes_from_that_one_function(self) -> None:
        """The `namespace=` keyword is only ever passed a `pinecone_namespace(...)` result.

        Read out of the AST rather than trusted, because the failure this guards against is a
        convenience f-string added later — `namespace=f"tenant-{user.tenant_id}"` — which is
        correct on the day it is written and silently wrong the day the prefix changes.
        """
        found = 0
        for path in backend_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                for keyword in node.keywords:
                    if keyword.arg != "namespace":
                        continue
                    found += 1
                    value = keyword.value
                    # The only accepted shapes: a local named `namespace` (assigned from the
                    # settings call a few lines above) or the call itself.
                    if isinstance(value, ast.Name) and value.id == "namespace":
                        continue
                    if (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Attribute)
                        and value.func.attr == "pinecone_namespace"
                    ):
                        continue
                    raise AssertionError(
                        f"{path.relative_to(ROOT).as_posix()}:{node.lineno} passes a "
                        "hand-built namespace; it must come from "
                        "Settings.pinecone_namespace()"
                    )
        assert found >= 4, (
            f"only {found} namespace= arguments found in backend/ — the scan is not reaching "
            "vector_store.py and would pass vacuously"
        )

    def test_the_namespace_local_is_always_assigned_from_settings(self) -> None:
        """Closes the loophole the test above leaves open.

        Accepting a local named `namespace` is only safe if every assignment to that name is a
        `pinecone_namespace()` call. Otherwise `namespace = f"tenant-{x}"` on the line before
        satisfies the previous test exactly.
        """
        for path in backend_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                targets = [
                    t.id for t in node.targets if isinstance(t, ast.Name)
                ]
                if "namespace" not in targets:
                    continue
                call = node.value
                assert (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "pinecone_namespace"
                ), (
                    f"{path.relative_to(ROOT).as_posix()}:{node.lineno} assigns a namespace "
                    "from something other than Settings.pinecone_namespace()"
                )

    async def test_every_index_call_a_request_makes_is_namespaced(
        self, api: Api, stack: Stack
    ) -> None:
        """The dynamic counterpart: no call reached the index with a blank namespace.

        A blank namespace is Pinecone's default namespace, not an error — an upsert with one
        writes vectors into a space every tenant's search would also have to be told to avoid.
        """
        client, as_user = api
        as_user(stack.a.user)
        response = await client.post(
            f"{API}/queries",
            json={"question_text": SHARED_QUESTION},
            headers=csrf_headers(),
        )
        assert response.status_code == 202, response.text

        recorded = (
            [namespace for namespace, _ in stack.index.query_calls]
            + [namespace for namespace, _ in stack.index.upsert_calls]
            + [namespace for namespace, _, _ in stack.index.delete_calls]
        )
        assert recorded, "no index call was recorded; the request did not reach retrieval"
        prefix = get_settings().PINECONE_NAMESPACE_PREFIX
        for namespace in recorded:
            assert namespace, "an index call carried the default (blank) namespace"
            assert namespace.startswith(prefix), namespace
            # The suffix must parse as a UUID: `tenant-` plus anything else means some other
            # identifier — an email, a slug, a document id — is scoping the vectors.
            uuid.UUID(namespace.removeprefix(prefix))


class TestATenantRetrievesOnlyItsOwnVectors:
    """The behavioural half, with the other tenant's near-identical passage there to leak."""

    async def test_the_shared_question_never_cites_the_other_tenant(
        self, stack: Stack
    ) -> None:
        """The central assertion of the module.

        Both corpora answer this question. Tenant A must cite only `fee-schedule-<a>.pdf`, and
        neither B's document name nor B's figure may appear anywhere in the answer, the
        references, or the audit detail.
        """
        stack.bedrock.queue_answer("The outbound domestic wire fee is stated as 25 USD [S1].")
        query = await stack.query_service.submit(
            question=SHARED_QUESTION, actor=stack.a.user, context=CONTEXT
        )
        assert query.status is QueryStatus.ANSWERED, query.answer_text

        cited = {reference["source_document_name"] for reference in query.source_references}
        assert cited == {stack.a.document.name}, cited
        assert stack.b.document.name not in repr(query.source_references)
        assert "Tenant B" not in (query.answer_text or "")
        assert "Tenant B" not in stack.audit_details()

    async def test_the_query_used_the_askers_own_namespace(self, stack: Stack) -> None:
        """Not "a namespace" — the asker's.

        The fixture answers one query per tenant during setup, so the assertion is on the call
        this test provoked rather than on the whole list.
        """
        before = len(stack.index.query_calls)
        stack.bedrock.queue_answer("Stated in the schedule [S1].")
        await stack.query_service.submit(
            question=SHARED_QUESTION, actor=stack.b.user, context=CONTEXT
        )
        new_calls = stack.index.query_calls[before:]
        assert new_calls, "the question did not reach the index"
        assert {namespace for namespace, _ in new_calls} == {stack.namespace(stack.b.id)}

    async def test_a_fact_only_the_other_tenant_holds_is_not_found(
        self, stack: Stack
    ) -> None:
        """The clean cross-tenant probe: the answer exists, in the wrong namespace.

        The passage is added to tenant B through the same `Corpus` helper ingestion uses, so it
        is genuinely retrievable — asked as B, this returns an answer (asserted in the next
        test). Asked as A, the pipeline must not even call the model: with nothing citable
        there is no prompt that reliably stops a model answering from training data, which is
        why `rag_service` returns before generating.
        """
        corpus = Corpus(stack.documents, stack.vectors, stack.bedrock)
        await corpus.add(stack.b.id, "escrow-cutoffs.pdf", [B_ONLY_PASSAGE])

        # Queued but must go unused: `bedrock.generate` is never called on a not-found, so a
        # leaked answer would show up as this string in the response.
        stack.bedrock.queue_answer("The cut-off is 16:30 GMT [S1].")
        query = await stack.query_service.submit(
            question=B_ONLY_QUESTION, actor=stack.a.user, context=CONTEXT
        )
        assert query.status is QueryStatus.NOT_FOUND, query.answer_text
        assert query.answer_text == NOT_FOUND_ANSWER
        assert query.source_references == []
        assert "16:30" not in (query.answer_text or "")

    async def test_the_same_fact_is_found_by_the_tenant_that_holds_it(
        self, stack: Stack
    ) -> None:
        """The paired control. Without it, the test above passes on a typo in the question."""
        corpus = Corpus(stack.documents, stack.vectors, stack.bedrock)
        await corpus.add(stack.b.id, "escrow-cutoffs.pdf", [B_ONLY_PASSAGE])

        stack.bedrock.queue_answer("The cut-off is 16:30 GMT [S1].")
        query = await stack.query_service.submit(
            question=B_ONLY_QUESTION, actor=stack.b.user, context=CONTEXT
        )
        assert query.status is QueryStatus.ANSWERED, query.answer_text
        assert "escrow-cutoffs.pdf" in repr(query.source_references)

    async def test_a_tenant_with_no_vectors_gets_not_found_not_a_neighbour(
        self, stack: Stack
    ) -> None:
        """An empty namespace must return nothing, never fall back to a populated one.

        A third tenant is created with no documents at all. This is the case a
        `.get(namespace, first_available)` bug or a blank-namespace default would fail: with
        two well-stocked corpora next door, "no matches" has to mean no matches.
        """
        record = stack.users.add_tenant(domain="tenant-c.example.com", name="Tenant C")
        newcomer = User(
            id=uuid.uuid4(),
            tenant_id=record.id,
            email="standard@tenant-c.example.com",
            full_name=None,
            hashed_password="not-a-real-hash",
            role=UserRole.STANDARD_USER,
            is_active=True,
            failed_login_attempts=0,
            locked_until=None,
        )
        stack.users.users[newcomer.id] = newcomer

        before = len(stack.index.query_calls)
        stack.bedrock.queue_answer("The fee is 25 USD [S1].")
        query = await stack.query_service.submit(
            question=SHARED_QUESTION, actor=newcomer, context=CONTEXT
        )
        assert query.status is QueryStatus.NOT_FOUND, query.answer_text
        assert query.answer_text == NOT_FOUND_ANSWER
        assert "25" not in (query.answer_text or "")
        assert {namespace for namespace, _ in stack.index.query_calls[before:]} == {
            stack.namespace(record.id)
        }
        assert stack.index.count(stack.namespace(record.id)) == 0

    async def test_history_shows_each_user_only_their_own_grounded_answer(
        self, api: Api, stack: Stack
    ) -> None:
        """Isolation has to hold on the way back out, too.

        The citations are stored on the query row and read back through `/history`. A leak that
        was avoided at retrieval time and then re-introduced by an unscoped history read is the
        same disclosure with a different line number.
        """
        client, as_user = api
        as_user(stack.a.user)
        response = await client.get(f"{API}/history")
        assert response.status_code == 200, response.text
        body = response.text
        assert stack.b.document.name not in body
        assert "Tenant B" not in body


class TestPostgresIsTheAuthorityOnCitability:
    """A vector in the index is a candidate. Only PostgreSQL turns one into a citation."""

    async def test_a_soft_deleted_document_is_not_cited_while_its_vectors_remain(
        self, stack: Stack
    ) -> None:
        """The window `PINECONE_OVERFETCH_FACTOR` exists to cover.

        The status is changed directly in the repository and the vectors are deliberately left
        in place — that is the real state between an administrator's delete and the index
        catching up, and `load_chunks` is what has to close it. Asserting the vectors are still
        present is what makes this a test of the re-check rather than of the cleanup.
        """
        await stack.documents.set_status(
            stack.a.document.id, stack.a.id, DocumentStatus.DELETED
        )
        assert stack.index.count(stack.namespace(stack.a.id)) == len(stack.a.passages)

        stack.bedrock.queue_answer("The fee is 25 USD [S1].")
        query = await stack.query_service.submit(
            question=SHARED_QUESTION, actor=stack.a.user, context=CONTEXT
        )
        assert query.status is QueryStatus.NOT_FOUND, query.answer_text
        assert query.source_references == []
        assert "25" not in (query.answer_text or "")

    async def test_a_foreign_chunk_id_planted_in_the_namespace_is_dropped(
        self, stack: Stack
    ) -> None:
        """Defence in depth against the index itself being wrong.

        A vector whose id is one of tenant B's chunks is written into tenant A's namespace —
        which models an index-side isolation failure, a botched migration, or a restored
        snapshot. The search returns it, so this is not a test of retrieval; it is a test that
        `load_chunks` refuses an id it cannot tie to a document in the caller's tenant.
        """
        namespace_a = stack.namespace(stack.a.id)
        namespace_b = stack.namespace(stack.b.id)
        foreign_id, (values, metadata) = next(
            iter(stack.index.namespaces[namespace_b].items())
        )
        stack.index.namespaces[namespace_a][foreign_id] = (values, metadata)

        stack.bedrock.queue_answer("The fee is stated in the schedule [S1].")
        query = await stack.query_service.submit(
            question=SHARED_QUESTION, actor=stack.a.user, context=CONTEXT
        )
        cited = {reference["source_document_name"] for reference in query.source_references}
        assert stack.b.document.name not in cited, cited
        assert cited <= {stack.a.document.name}, cited

    async def test_the_search_overfetches_so_the_recheck_can_discard(
        self, stack: Stack
    ) -> None:
        """`top_k * PINECONE_OVERFETCH_FACTOR`, asserted as the arithmetic it is.

        If retrieval asked for exactly `RAG_TOP_K`, then every candidate the re-check discarded
        would shrink the answer's evidence instead of being replaced. Overfetching is what
        makes "a deleted document is silently skipped" behave like a deleted document rather
        than like a degraded search.
        """
        settings = get_settings()
        before = len(stack.index.query_calls)
        stack.bedrock.queue_answer("Stated in the schedule [S1].")
        await stack.query_service.submit(
            question=SHARED_QUESTION, actor=stack.a.user, context=CONTEXT
        )
        requested = [top_k for _, top_k in stack.index.query_calls[before:]]
        assert requested == [settings.RAG_TOP_K * settings.PINECONE_OVERFETCH_FACTOR]
        assert settings.PINECONE_OVERFETCH_FACTOR > 1, (
            "an overfetch factor of 1 removes the margin the citability re-check spends"
        )

    async def test_the_search_does_not_ask_for_metadata(self, stack: Stack) -> None:
        """Citations are read from PostgreSQL, so index metadata is never requested.

        Not a performance choice. Index metadata is written once at ingestion and never
        updated; citing it would quote a document's name as it was at upload time, and a
        renamed document would be cited under a name no longer in the system.
        """
        source = inspect.getsource(VectorStore.search)
        assert "include_metadata=False" in source
        assert "include_metadata=True" not in source


class TestVectorMetadataCarriesNoDocumentText:
    """`architecture.md`: *"Pinecone metadata carries citations, never document text."*

    The rule is a data-residency one, not an efficiency one: a third copy of customer document
    text in a system with its own access model and its own retention is what
    `secure-coding-vapt.md` forbids. Both guards are exercised — the builder's allowlist and
    the fake index's independent check — because the second is what still catches a caller who
    bypasses the first.
    """

    async def test_no_stored_metadata_contains_a_passage(self, stack: Stack) -> None:
        stored = repr(stack.index.namespaces)
        for tenant in (stack.a, stack.b):
            for passage in tenant.passages:
                # A distinctive fragment rather than the whole passage: a truncated copy is
                # still a copy, and the whole-string check would miss it.
                for fragment in (passage[:40], passage[-40:]):
                    assert fragment not in stored, (
                        "document text reached vector metadata; see "
                        ".claude/rules/architecture.md"
                    )

    async def test_stored_metadata_keys_are_on_the_allowlist(self, stack: Stack) -> None:
        for space in stack.index.namespaces.values():
            for _values, metadata in space.values():
                unexpected = metadata.keys() - ALLOWED_METADATA_KEYS
                assert unexpected == set(), unexpected

    async def test_the_index_rejects_a_text_metadata_key(self, stack: Stack) -> None:
        """The second guard, proven live rather than assumed.

        `_build_metadata` raises on the way in, so this writes past it — straight to the fake
        index — to show the backstop is real. A test that only exercised the builder would go
        green on the day a caller stops using it.
        """
        with pytest.raises(AssertionError, match="never carry document text"):
            await stack.index.upsert(
                vectors=[
                    {
                        "id": str(uuid.uuid4()),
                        "values": [0.0, 1.0],
                        "metadata": {"text": stack.a.passages[0]},
                    }
                ],
                namespace=stack.namespace(stack.a.id),
            )

    def test_the_forbidden_list_covers_the_obvious_synonyms(self) -> None:
        """A denylist that only knows the word `text` is one rename from useless."""
        for name in ("text", "content", "body", "passage", "excerpt", "chunk_text"):
            assert name in FORBIDDEN_METADATA_KEYS
        assert not FORBIDDEN_METADATA_KEYS & ALLOWED_METADATA_KEYS


class TestDeletionStaysInsideOneNamespace:
    """A delete is a write, and one to the wrong namespace erases another tenant's index."""

    async def test_deleting_a_document_leaves_the_other_tenant_untouched(
        self, api: Api, stack: Stack
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        namespace_b = stack.namespace(stack.b.id)
        before_b = stack.index.ids_in(namespace_b)

        response = await client.delete(
            f"{API}/documents/{stack.a.document.id}", headers=csrf_headers()
        )
        assert response.status_code in (200, 204), response.text

        assert stack.index.count(stack.namespace(stack.a.id)) == 0
        assert stack.index.ids_in(namespace_b) == before_b
        assert {namespace for namespace, _, _ in stack.index.delete_calls} == {
            stack.namespace(stack.a.id)
        }

    async def test_no_delete_call_ever_asks_for_delete_all(
        self, api: Api, stack: Stack
    ) -> None:
        """`delete_all=True` empties a whole tenant's namespace.

        It exists for offboarding and test teardown and is deliberately unroutable. A request
        path that reached it would turn one document deletion into the loss of every embedding
        the tenant owns — recoverable only by re-ingesting every document.
        """
        client, as_user = api
        as_user(stack.a.admin)
        await client.delete(
            f"{API}/documents/{stack.a.document.id}", headers=csrf_headers()
        )
        assert [flag for _, _, flag in stack.index.delete_calls] == [False]

    def test_delete_namespace_is_not_reachable_from_any_route(self) -> None:
        """Asserted by grep, because the safe number of call sites is zero.

        `delete_namespace` is the one method here with no legitimate HTTP caller. Behind an
        endpoint it would be an IDOR target whose successful exploitation is indistinguishable
        from a tenant with no documents.
        """
        pattern = re.compile(r"\bdelete_namespace\b")
        callers = [
            path.relative_to(ROOT).as_posix()
            for path in backend_modules()
            if pattern.search(path.read_text(encoding="utf-8"))
            and path.name != "vector_store.py"
        ]
        assert callers == [], (
            f"delete_namespace is referenced outside vector_store.py: {callers}"
        )

    async def test_a_restore_re_embeds_into_the_owners_namespace(
        self, api: Api, stack: Stack
    ) -> None:
        """Upload, delete, restore — all over HTTP as tenant A's administrator.

        A restore is a re-ingest, so it is the one request that *writes* vectors outside an
        upload, and the namespace it writes to is derived a second time. A restore that
        re-embedded under a blank namespace, or under the id of whichever tenant the admin
        looked at last, would return 200 and leave the document listed and unsearchable.

        The upload is not incidental. `conftest`'s fixture documents are staged through
        `Corpus`, which writes chunk rows and vectors directly and never puts bytes in S3 — so
        restoring one re-ingests from an object that does not exist, and the document lands
        `FAILED` with no vectors. Uploading a real PDF first gives this test the state a
        restore actually operates on. (That the route still answers 200 for the failed-reingest
        case, with the truth only in the audit row's `restored_to`, is recorded as a Low
        finding in `docs/RELEASE.md` — it is an audit-fidelity question, not an isolation one.)
        """
        client, as_user = api
        as_user(stack.a.admin)
        namespace_a = stack.namespace(stack.a.id)
        namespace_b = stack.namespace(stack.b.id)
        before_b = stack.index.ids_in(namespace_b)
        before_a = stack.index.count(namespace_a)

        uploaded = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "wire-fees.pdf",
                    make_pdf(["The outbound domestic wire transfer fee is 25 USD."]),
                    "application/pdf",
                )
            },
            headers=csrf_headers(),
        )
        assert uploaded.status_code == 201, uploaded.text
        document_id = uploaded.json()["id"]
        embedded = stack.index.count(namespace_a) - before_a
        assert embedded > 0, "the upload produced no vectors; the rest of this test is vacuous"

        deleted = await client.delete(
            f"{API}/documents/{document_id}", headers=csrf_headers()
        )
        assert deleted.status_code in (200, 204), deleted.text
        assert stack.index.count(namespace_a) == before_a

        restored = await client.post(
            f"{API}/documents/{document_id}/restore", headers=csrf_headers()
        )
        assert restored.status_code == 200, restored.text
        assert stack.index.count(namespace_a) == before_a + embedded
        assert stack.index.ids_in(namespace_b) == before_b
