"""backend/tests/compliance/test_traceability.py

Role: The BR-03 compliance check — every stored answer resolves to a live document and a real
section or page. An orphaned reference fails, and this module makes one to prove it.

What "resolves" means here
--------------------------
`resolution_problems()` below is the whole check, and it is deliberately stricter than the
schema. `SourceReference` guarantees the two location fields are *present*; that is a shape
check and it passes on `{"source_document_name": "x", "section_or_page": "page 400"}`. A
compliance reviewer is asking something else: does this citation lead anywhere? So each
reference is followed back through PostgreSQL:

  1. `document_id` parses as a UUID;
  2. a document with that id exists **in the answer's own tenant**;
  3. that document is `ACTIVE` — in `RETRIEVABLE_STATUSES`, the same set retrieval uses;
  4. the stored `source_document_name` is that document's name;
  5. the stored `section_or_page` is the `citation` of a chunk that document really has.

Step 5 is the one worth arguing for. Naming the right document is the weak claim — a four-page
fee schedule has one name and four locations, and an answer citing page 1 for a fact on page 3
is exactly the failure BR-03 exists to catch. `models/document.py`'s `DocumentChunk.citation`
docstring records that the same string is built in three places and must render byte-
identically; this module is where a divergence between them surfaces as a citation that leads
nowhere.

The requirement is scoped to live documents, and that is a real limit
--------------------------------------------------------------------
"Every stored answer resolves to a live document" cannot hold unconditionally: an Administrator
may legitimately delete a document tomorrow, and yesterday's answers keep the references they
were given. Rewriting or deleting those answers to keep this check green would be worse than
the finding — search history is audited content, and `architecture.md` treats a second,
divergent copy of an audited fact as a compliance defect in itself.

So the honest statement of the invariant, and the one asserted below, is two-part: an answer's
references resolve for as long as their documents are live, and the moment a document stops
being live the reference stops resolving *and says why*. What a reviewer then sees is a history
row whose citation can no longer be opened, next to a `DOCUMENT_DELETE` row that explains it.
`docs/RELEASE.md` records this as documented behaviour rather than an open defect.

Why the questions here are mostly about withholding tax
-------------------------------------------------------
`build_stack()` gives tenant A a seeded `fee-schedule-*.pdf` whose two passages are worded
almost identically to pages 1 and 2 of `FEE_PDF` — the security suite needs near-duplicate text
across tenants. So a wire-fee question in tenant A has two plausible sources, and an assertion
naming the uploaded document could fail for a ranking reason that has nothing to do with
traceability. Pages 3 and 4 (withholding tax, out-of-network ATM) have no seeded counterpart,
which makes the uploaded document the only possible source and keeps each assertion about the
thing it claims to be about. `backend/tests/regression/test_ai_query.py` records the same
lesson, having paid for it first.

What this module cannot tell you
--------------------------------
That the answer is *correct*. `confidence_level` is retrieval similarity — `source_tracer` says
so at length — and nothing here measures whether the model reasoned correctly over a passage it
genuinely retrieved. Every assertion in this file is about provenance, which is what BR-03 and
`document-source-traceability.md` actually require.
"""

from __future__ import annotations

import uuid
from typing import Any

from backend.models.audit_log import AuditAction
from backend.models.document import RETRIEVABLE_STATUSES, DocumentStatus
from backend.models.query import QueryStatus
from backend.services.source_tracer import NOT_FOUND_ANSWER
from backend.tests.compliance.conftest import (
    API,
    SURCHARGE_QUESTION,
    TAX_QUESTION,
    UNANSWERABLE_QUESTION,
    WIRE_QUESTION,
    Api,
    Stack,
    ask,
    csrf_headers,
    new_uuid,
    switch_to,
    upload,
)

#: Where each answerable fact lives in `PAGES`, as the citation renders it. Only the last two
#: are unique to the uploaded document — see the module docstring.
WIRE_PAGE = "page 1"
TAX_PAGE = "page 3"
SURCHARGE_PAGE = "page 4"

#: A phrase from each of those two pages, so a test can assert the citation resolves to the
#: passage carrying the figure rather than merely to some page of the right document.
TAX_PHRASE = "10 percent"
SURCHARGE_PHRASE = "out-of-network"


async def resolution_problems(
    stack: Stack, tenant_id: uuid.UUID, reference: dict[str, Any]
) -> list[str]:
    """Follow one stored reference back to PostgreSQL. An empty list means it resolves.

    Returns *reasons* rather than raising, for two reasons. One answer can carry several
    references and a reviewer wants every problem, not the first. And the orphan tests below
    assert on which check caught the orphan — a bare `pytest.raises` passes when a reference
    fails for an unrelated reason, which is how a compliance test quietly stops testing the
    thing it is named after.
    """
    problems: list[str] = []

    try:
        document_id = uuid.UUID(str(reference.get("document_id")))
    except (TypeError, ValueError):
        return [f"document_id is not a uuid: {reference.get('document_id')!r}"]

    document = await stack.documents.get(document_id, tenant_id)
    if document is None:
        return [f"no document {document_id} in tenant {tenant_id}"]

    if document.status not in RETRIEVABLE_STATUSES:
        problems.append(f"document is {document.status.value}, not retrievable")
    if document.name != reference.get("source_document_name"):
        problems.append(
            f"cited name {reference.get('source_document_name')!r} is not {document.name!r}"
        )

    chunks = await stack.documents.get_chunks(document.id, tenant_id)
    if not chunks:
        problems.append("document has no chunk rows, so no location can be verified")
    elif reference.get("section_or_page") not in {chunk.citation for chunk in chunks}:
        problems.append(
            f"location {reference.get('section_or_page')!r} is not a section or page "
            "this document has"
        )
    return problems


async def problems_for_every_answer(stack: Stack) -> dict[uuid.UUID, list[str]]:
    """Every answered query in the store, checked. Keyed by query id so a failure names it."""
    found: dict[uuid.UUID, list[str]] = {}
    for query in stack.queries.queries.values():
        if query.status is not QueryStatus.ANSWERED:
            continue
        problems: list[str] = []
        if not query.source_references:
            problems.append("answered with no source reference at all")
        for reference in query.source_references or []:
            problems.extend(await resolution_problems(stack, query.tenant_id, reference))
        if problems:
            found[query.id] = problems
    return found


async def resolved_passages(
    stack: Stack, tenant_id: uuid.UUID, body: dict[str, Any]
) -> list[str]:
    """The stored chunk text each citation resolves to.

    Goes through the published `document_id` and `section_or_page` rather than reading the
    `excerpt` in the same payload: the claim is that the *citation* leads a reviewer to the
    passage, and an echoed excerpt only proves the answer quoted something.
    """
    passages: list[str] = []
    for reference in body["source_references"]:
        assert await resolution_problems(stack, tenant_id, reference) == []
        chunks = await stack.documents.get_chunks(
            uuid.UUID(reference["document_id"]), tenant_id
        )
        passages.extend(
            chunk.text for chunk in chunks if chunk.citation == reference["section_or_page"]
        )
    return passages


def only_reference(body: dict[str, Any]) -> dict[str, Any]:
    """The single reference on an answer that cited exactly one passage."""
    references = body["source_references"]
    assert len(references) == 1, references
    return references[0]


def answer_rows(stack: Stack, query_id: str) -> list[Any]:
    """The audit rows for one query that carry a `sources` list.

    Filtered by `object_id`, not by action alone: `build_stack()` already answered one question
    per tenant, so an unfiltered scan would mix another query's citations into the assertion.
    """
    return [
        entry
        for entry in stack.audit_repo.entries
        if entry.action is AuditAction.QUERY_SUBMIT
        and str(entry.object_id) == str(query_id)
        and isinstance((entry.detail or {}).get("sources"), list)
    ]


class TestEveryStoredAnswerResolves:
    """The check itself, over answers produced by the real pipeline."""

    async def test_the_seeded_answers_resolve(self, stack: Stack) -> None:
        """`build_stack()` answers one question per tenant through the real `QueryService`.

        Run before anything in this module touches the stack, so a failure here is about
        ingestion and retrieval rather than about a test's own setup.
        """
        assert await problems_for_every_answer(stack) == {}

    async def test_an_answer_from_an_uploaded_document_resolves(
        self, api: Api, stack: Stack
    ) -> None:
        """The full path: an Administrator uploads, a Standard User asks, and the citation
        leads back to the page the fact is on."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)

        user = switch_to(api, stack.a.user)
        body = await ask(
            user, stack, TAX_QUESTION, "Withholding tax is deducted at 10 percent [S1]."
        )

        assert body["status"] == QueryStatus.ANSWERED.value
        reference = only_reference(body)
        assert reference["source_document_name"] == document["name"]
        assert reference["section_or_page"] == TAX_PAGE
        assert await resolution_problems(stack, stack.a.id, reference) == []

    async def test_the_cited_page_is_the_page_the_fact_is_on(
        self, api: Api, stack: Stack
    ) -> None:
        """Naming the document is the weak claim. Two questions whose answers live on two
        different pages must produce two different locations, each resolving to the passage
        that carries the figure."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)

        cited: list[str] = []
        for question, page, phrase in (
            (TAX_QUESTION, TAX_PAGE, TAX_PHRASE),
            (SURCHARGE_QUESTION, SURCHARGE_PAGE, SURCHARGE_PHRASE),
        ):
            body = await ask(user, stack, question, "Stated in the schedule [S1].")
            assert only_reference(body)["section_or_page"] == page, question
            passages = await resolved_passages(stack, stack.a.id, body)
            assert any(phrase in passage for passage in passages), question
            cited.append(page)

        assert len(set(cited)) == len(cited)

    async def test_every_answer_still_resolves_after_a_page_of_activity(
        self, api: Api, stack: Stack
    ) -> None:
        """The suite-level assertion again, after uploads and several questions rather than
        on a freshly built stack."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        for question in (WIRE_QUESTION, TAX_QUESTION, SURCHARGE_QUESTION):
            await ask(user, stack, question, "Stated in the fee schedule [S1].")

        assert await problems_for_every_answer(stack) == {}

    async def test_the_published_dto_carries_what_the_store_holds(
        self, api: Api, stack: Stack
    ) -> None:
        """A citation that is right in PostgreSQL and lost in the DTO is still an untraceable
        answer to the person reading it, which is why this suite goes over HTTP."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        fetched = (await user.get(f"{API}/queries/{body['id']}")).json()

        stored = stack.queries.queries[uuid.UUID(body["id"])].source_references
        assert [entry["section_or_page"] for entry in fetched["source_references"]] == [
            entry["section_or_page"] for entry in stored
        ]
        assert [entry["document_id"] for entry in fetched["source_references"]] == [
            entry["document_id"] for entry in stored
        ]

    async def test_one_passage_cited_twice_is_one_reference(
        self, api: Api, stack: Stack
    ) -> None:
        """`source_tracer` deduplicates on `(document_id, citation)`. A list repeating "page 3"
        with two scores would read as an unreliable citation to a reviewer."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)

        body = await ask(
            user, stack, TAX_QUESTION, "Ten percent [S1], on credited interest [S1]."
        )

        pairs = [
            (entry["source_document_name"], entry["section_or_page"])
            for entry in body["source_references"]
        ]
        assert len(pairs) == len(set(pairs))

    async def test_a_reference_carries_the_id_the_ui_needs_to_open_it(
        self, api: Api, stack: Stack
    ) -> None:
        """A citation a Standard User cannot open is a citation they cannot check, which is why
        `preview` is not Administrator-only. The id is what makes the link possible."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        reference = only_reference(body)
        assert reference["document_id"] == document["id"]
        opened = await user.get(f"{API}/documents/{reference['document_id']}/preview")
        assert opened.status_code == 200
        assert opened.json()["name"] == reference["source_document_name"]


class TestAnOrphanedReferenceFails:
    """The proof that the check above can fail.

    Every test here breaks one link in the chain of a real stored answer and asserts
    `resolution_problems` names it. A compliance check nobody has watched fail is a claim about
    its own fixtures.
    """

    async def test_a_reference_to_no_document_at_all_fails(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        orphan = dict(only_reference(body), document_id=str(new_uuid()))

        problems = await resolution_problems(stack, stack.a.id, orphan)
        assert problems and "no document" in problems[0]

    async def test_a_reference_to_another_tenants_document_fails(
        self, api: Api, stack: Stack
    ) -> None:
        """The id an attacker actually has: a real document id, from a real citation, in the
        wrong tenant. `resolution_problems` scopes the lookup by tenant for the same reason
        every repository method does."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        assert stack.b.document is not None
        orphan = dict(only_reference(body), document_id=str(stack.b.document.id))

        problems = await resolution_problems(stack, stack.a.id, orphan)
        assert problems and "no document" in problems[0]

    async def test_a_reference_to_a_page_that_does_not_exist_fails(
        self, api: Api, stack: Stack
    ) -> None:
        """The document is right and the location is invented. This is the failure the schema
        cannot see, because "page 400" is a perfectly valid string."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        orphan = dict(only_reference(body), section_or_page="page 400")

        problems = await resolution_problems(stack, stack.a.id, orphan)
        assert any("is not a section or page" in problem for problem in problems)

    async def test_a_reference_naming_the_wrong_document_fails(
        self, api: Api, stack: Stack
    ) -> None:
        """The id and the name have to agree. A stored name that drifted from the document it
        points at would send a reviewer looking for a file that does not hold the passage."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        orphan = dict(only_reference(body), source_document_name="something-else.pdf")

        problems = await resolution_problems(stack, stack.a.id, orphan)
        assert any("is not" in problem for problem in problems)

    async def test_a_malformed_document_id_fails(self, api: Api, stack: Stack) -> None:
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        for bad in ("", "not-a-uuid", None, "../../etc/passwd"):
            orphan = dict(only_reference(body), document_id=bad)
            problems = await resolution_problems(stack, stack.a.id, orphan)
            assert problems and "not a uuid" in problems[0], bad

    async def test_a_reference_whose_chunks_are_gone_fails(
        self, api: Api, stack: Stack
    ) -> None:
        """A purge removes the chunk rows. The document may still exist, but nothing is left to
        verify the location against — and "cannot be verified" must not read as "verified"."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        await stack.documents.delete_chunks(uuid.UUID(document["id"]), stack.a.id)

        problems = await resolution_problems(stack, stack.a.id, only_reference(body))
        assert any("no chunk rows" in problem for problem in problems)

    async def test_a_deleted_document_makes_its_historical_citations_unresolvable(
        self, api: Api, stack: Stack
    ) -> None:
        """The legitimate case, asserted as behaviour rather than as a defect.

        An Administrator deletes a document; yesterday's answer keeps the reference it was
        given. The reference stops resolving, and the reason names the status — which is what
        lets a reviewer tell "this citation was never real" apart from "this document was
        withdrawn, see the `DOCUMENT_DELETE` row". The stored answer is not rewritten; see the
        module docstring.
        """
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        admin = switch_to(api, stack.a.admin)
        deleted = await admin.delete(
            f"{API}/documents/{document['id']}", headers=csrf_headers()
        )
        assert deleted.status_code in (200, 204), deleted.text

        problems = await resolution_problems(stack, stack.a.id, only_reference(body))
        assert any("not retrievable" in problem for problem in problems)
        stored = stack.documents.documents[uuid.UUID(document["id"])]
        assert stored.status is DocumentStatus.DELETED
        assert stack.queries.queries[uuid.UUID(body["id"])].source_references

    async def test_the_deleted_document_cannot_be_cited_by_a_new_answer(
        self, api: Api, stack: Stack
    ) -> None:
        """The other half of the same delete: history keeps its citation, and no *new* answer
        may acquire one. This is `rag-answer-grounding.md`'s PostgreSQL re-check, seen from the
        compliance side rather than the retrieval side."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        cited = {entry["document_id"] for entry in body["source_references"]}
        assert document["id"] not in cited


class TestAnUngroundedAnswerIsNeverStoredAsAnAnswer:
    """`rag-answer-grounding.md`: no answer without a source, and the refusal is fixed text."""

    async def test_a_question_the_corpus_cannot_answer_is_refused(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)

        body = await ask(user, stack, UNANSWERABLE_QUESTION, "Brazil won the 1994 final.")

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert body["answer_text"] == NOT_FOUND_ANSWER
        assert body["source_references"] == []

    async def test_an_answer_with_no_citation_marker_is_discarded(
        self, api: Api, stack: Stack
    ) -> None:
        """The model answered from something other than the passages it was given. In a banking
        context that is a fabrication with the bank's name on it, so the text is replaced
        rather than published with a caveat."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)

        body = await ask(user, stack, TAX_QUESTION, "Withholding tax is 10 percent.")

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert body["answer_text"] == NOT_FOUND_ANSWER
        assert body["source_references"] == []

    async def test_an_invented_marker_produces_no_reference(
        self, api: Api, stack: Stack
    ) -> None:
        """`[S99]` when ninety-nine passages were never supplied is the model inventing a
        citation — precisely what BR-03 exists to catch. The marker is dropped, and with
        nothing valid left the answer becomes a not-found rather than an answer with one fewer
        source."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)

        body = await ask(user, stack, TAX_QUESTION, "Withholding tax is 10 percent [S99].")

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert body["source_references"] == []

    async def test_no_refusal_is_stored_as_answered(self, stack: Stack) -> None:
        """Belt and braces on the store rather than the response: no row anywhere in the stack
        is `ANSWERED` while holding the refusal text."""
        for query in stack.queries.queries.values():
            if query.answer_text == NOT_FOUND_ANSWER:
                assert query.status is not QueryStatus.ANSWERED


class TestTheAuditTrailRecordsTheSources:
    """`document-source-traceability.md`: "Log all source references in audit trail for
    compliance reviews." The trail must hold enough to reconstruct which document answered a
    question, and no more than that."""

    async def test_the_answer_row_records_every_citation(
        self, api: Api, stack: Stack
    ) -> None:
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        rows = answer_rows(stack, body["id"])
        assert rows
        logged = rows[-1].detail["sources"]
        assert [entry["section_or_page"] for entry in logged] == [
            entry["section_or_page"] for entry in body["source_references"]
        ]

    async def test_the_logged_citation_resolves_the_same_way(
        self, api: Api, stack: Stack
    ) -> None:
        """The point of logging the reference is that a reviewer can follow it years later, so
        the logged pair has to satisfy the same check the published one does."""
        admin = switch_to(api, stack.a.admin)
        document = await upload(admin)
        user = switch_to(api, stack.a.user)
        body = await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        logged = [
            item for row in answer_rows(stack, body["id"]) for item in row.detail["sources"]
        ]
        assert logged
        for item in logged:
            assert item["document_id"] == document["id"]
            reference = dict(item, source_document_name=document["name"])
            assert await resolution_problems(stack, stack.a.id, reference) == []

    async def test_the_trail_records_the_location_and_not_the_passage(
        self, api: Api, stack: Stack
    ) -> None:
        """A reference also carries an `excerpt` — document body text. `secure-coding-vapt.md`
        keeps that out of the log, so the audit row holds the id and the location only: enough
        to find the passage, not a second copy of it in the one store nothing can be removed
        from."""
        admin = switch_to(api, stack.a.admin)
        await upload(admin)
        user = switch_to(api, stack.a.user)
        await ask(user, stack, TAX_QUESTION, "Ten percent [S1].")

        for entry in stack.audit_repo.entries:
            for item in (entry.detail or {}).get("sources") or []:
                assert set(item) == {"document_id", "section_or_page"}
        written = stack.audit_details()
        assert "valid exemption declaration" not in written
        assert TAX_QUESTION not in written
