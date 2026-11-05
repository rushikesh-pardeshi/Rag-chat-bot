"""backend/tests/regression/test_ai_query.py

Role: BR-02 (AI Query Engine), BR-03 (Source Traceability) and the Query Accuracy family
(R-04) asserted over HTTP.

The requirements, quoted from `WORKBREAKDOWN.md`:

    AI Query Engine: Natural language queries processed in 3-5 seconds, answers generated
    strictly from uploaded documents, domain-specific terminology (tax, customer accounts,
    compliance) interpreted accurately (BR-AIQ, BR-02, NFR-9.1)

    Source Traceability: Every AI answer displays source document name, relevant section/page,
    and optional confidence level; prevents unsupported answer generation (BR-03, BR-AIQ)

What these tests can and cannot prove
-------------------------------------
`FakeBedrockClient` stands in for Titan embeddings and Nova generation. Its `embed` is a
hashed bag-of-words projection, so retrieval *ranking* is real in the sense that word overlap
decides it — a passage about maintenance fees does not outrank one about wire fees for a
question about wire fees. But it is **not** Titan, and no assertion here is evidence about
Titan's ranking quality. The accuracy figure that goes in `docs/RELEASE.md` comes from
`backend/tests/eval/`, and it carries the same caveat.

Likewise, `latency_ms` is asserted to be *recorded*, not to be under three seconds. A fake
embedder answers in single-digit milliseconds; the 3–5 second target in NFR-9.1 is measured by
`AnalyticsRepository.summary()`'s `percentile_cont` over `query.latency_ms` against real
traffic. A test that asserted `latency_ms < 5000` here would be a green gate over an
unmeasured requirement, which `.claude/rules/workflow.md` forbids.

Why the answer is queued per test
---------------------------------
`stack.bedrock.queue_answer(...)` puts one `generate` response in line. With nothing queued the
fake returns the refusal string, deliberately — a fake that invented prose would make a
grounding test pass by accident. So a test that wants an *answered* query queues one, and a
test about the not-found path queues nothing.
"""

from __future__ import annotations

import uuid
from typing import Any

from backend.models.document import DocumentStatus
from backend.models.query import QueryStatus
from backend.services.source_tracer import NOT_FOUND_ANSWER
from backend.tests.regression.conftest import (
    API,
    FEE_PAGES,
    MAINTENANCE_PAGE,
    OFF_CORPUS_QUESTION,
    TAX_QUESTION,
    WIRE_PAGE,
    WIRE_QUESTION,
    Api,
    Stack,
    audit_rows,
    citations,
    csrf_headers,
    new_uuid,
    source_names,
    switch_to,
    upload,
)

MAINTENANCE_QUESTION = (
    "What is the monthly account maintenance fee and when is it waived?"
)

#: `build_stack()` already gives tenant A a `fee-schedule-*.pdf` whose two passages are worded
#: almost identically to pages 1 and 2 of `FEE_PDF`, because the security suite needs
#: near-duplicate text across tenants. Two consequences for this module, both of which cost a
#: test here before they were noticed:
#:
#:   * a test that only asserts "page 1 was cited" can be satisfied by the *seeded* document,
#:     whose passages also live on pages 1 and 2 — so the assertions below check the passage
#:     the citation resolves to, not just its page label;
#:   * a test that needs the uploaded document to be the *only* possible source must ask about
#:     pages 3 or 4 (withholding tax, out-of-network ATM), which the seeded corpus has no
#:     counterpart for. `TAX_QUESTION` is that question.


async def ask(client: Any, question: str) -> Any:
    """`POST /api/v1/queries`. 202, because the work is accepted rather than complete."""
    return await client.post(
        f"{API}/queries", json={"question_text": question}, headers=csrf_headers()
    )


def resolved_passages(stack: Stack, body: dict[str, Any]) -> list[str]:
    """The stored chunk text each citation resolves to, in citation order.

    Goes through the `document_id` and `section_or_page` the response published, rather than
    reading the `excerpt` the response also carries. The claim being tested is that the
    *citation* leads a reviewer to the passage — an excerpt echoed back in the same payload
    proves only that the answer quoted something.
    """
    chunks = stack.queries.chunks_for(stack.a.id)
    passages: list[str] = []
    for reference in body["source_references"]:
        matches = [
            chunk.text
            for chunk in chunks
            if str(chunk.document_id) == str(reference["document_id"])
            and chunk.citation == reference["section_or_page"]
        ]
        assert matches, (
            f"citation {reference['section_or_page']!r} in "
            f"{reference['source_document_name']!r} resolves to no stored chunk"
        )
        passages.extend(matches)
    return passages


async def seed_and_ask(api: Api, stack: Stack, question: str, answer: str | None) -> Any:
    """Upload the fee schedule as tenant A's Administrator, then ask as its Standard User.

    Both halves matter to BR-02: uploading is Administrator-only, asking is what a Standard
    User does, and the answer has to cross that boundary correctly.
    """
    client = switch_to(api, stack.a.admin)
    await upload(client)
    client = switch_to(api, stack.a.user)
    if answer is not None:
        stack.bedrock.queue_answer(answer)
    return await ask(client, question)


class TestANaturalLanguageQuestionIsAnswered:
    """"Natural language queries ... answers generated strictly from uploaded documents"."""

    async def test_a_question_about_the_corpus_is_answered(
        self, api: Api, stack: Stack
    ) -> None:
        response = await seed_and_ask(
            api,
            stack,
            WIRE_QUESTION,
            "The outbound domestic wire transfer fee is 25 USD [S1].",
        )

        assert response.status_code == 202
        body = response.json()
        assert body["status"] == QueryStatus.ANSWERED.value
        assert "25 USD" in body["answer_text"]

    async def test_a_standard_user_may_ask_without_being_an_administrator(
        self, api: Api, stack: Stack
    ) -> None:
        """Querying is the Standard User's core operation. If this ever needed an admin role,
        the product would have no users."""
        response = await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        assert response.status_code == 202

    async def test_an_administrator_may_ask_too(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)
        await upload(client)
        stack.bedrock.queue_answer("25 USD [S1].")

        response = await ask(client, WIRE_QUESTION)

        assert response.status_code == 202
        assert response.json()["status"] == QueryStatus.ANSWERED.value

    async def test_an_unauthenticated_question_is_401(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, None)

        assert (await ask(client, WIRE_QUESTION)).status_code == 401

    async def test_the_question_is_echoed_back_unchanged(
        self, api: Api, stack: Stack
    ) -> None:
        """The row records what was asked, verbatim. A history entry whose question was
        rewritten is not the question that was answered."""
        response = await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        assert response.json()["question_text"] == WIRE_QUESTION


class TestDomainTerminologyIsInterpreted:
    """"domain-specific terminology (tax, customer accounts, compliance) interpreted
    accurately".

    Each question below is answerable from exactly one page of a four-page document whose other
    pages are all about neighbouring subjects — fees, tax, ATM surcharges. Retrieving the
    *document* is the weak claim, so each test resolves the citation back to the stored passage
    and asserts the passage actually carries the figure the question asked for.

    Note the ranking these tests rely on is `FakeBedrockClient`'s hashed bag-of-words
    projection, not Titan. They assert that the pipeline cites the passage it retrieved; they
    are not evidence about Titan's ranking quality.
    """

    async def test_a_tax_question_retrieves_the_tax_passage(
        self, api: Api, stack: Stack
    ) -> None:
        response = await seed_and_ask(
            api, stack, TAX_QUESTION, "Withholding tax is deducted at 10 percent [S1]."
        )

        body = response.json()
        assert body["status"] == QueryStatus.ANSWERED.value
        assert "page 3" in citations(body)
        assert any("10 percent" in passage for passage in resolved_passages(stack, body))

    async def test_a_maintenance_fee_question_retrieves_the_maintenance_passage(
        self, api: Api, stack: Stack
    ) -> None:
        response = await seed_and_ask(
            api,
            stack,
            MAINTENANCE_QUESTION,
            "The monthly maintenance fee is 12 USD and is waived above 5,000 USD [S1].",
        )

        body = response.json()
        assert MAINTENANCE_PAGE in citations(body)
        passages = resolved_passages(stack, body)
        assert any("maintenance fee is 12 USD" in passage for passage in passages)
        # The neighbouring page is about wire fees, and a citation that landed there would be
        # the failure this test exists for.
        assert not any("wire transfer" in passage for passage in passages)

    async def test_a_wire_fee_question_retrieves_the_wire_passage(
        self, api: Api, stack: Stack
    ) -> None:
        response = await seed_and_ask(
            api, stack, WIRE_QUESTION, "The wire transfer fee is 25 USD [S1]."
        )

        body = response.json()
        assert WIRE_PAGE in citations(body)
        passages = resolved_passages(stack, body)
        assert any("wire transfer fee is 25 USD" in passage for passage in passages)
        assert not any("maintenance fee" in passage for passage in passages)


class TestEveryAnswerCarriesItsSources:
    """BR-03: "Every AI answer displays source document name, relevant section/page, and
    optional confidence level"."""

    async def test_the_answer_names_the_source_document(
        self, api: Api, stack: Stack
    ) -> None:
        """`TAX_QUESTION`, not the wire question: only the uploaded document can answer it, so
        the document name asserted here is the name of a file this test actually uploaded."""
        response = await seed_and_ask(
            api, stack, TAX_QUESTION, "Withholding tax is 10 percent [S1]."
        )

        body = response.json()
        assert body["source_references"]
        assert source_names(body) == ["retail-account-fees.pdf"]

    async def test_every_reference_carries_a_section_or_page(
        self, api: Api, stack: Stack
    ) -> None:
        """Both fields are mandatory on `SourceReference`. A citation with a document name and
        no location cannot be followed by a compliance reviewer, which is the only thing
        `section_or_page` is for."""
        response = await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        for reference in response.json()["source_references"]:
            assert reference["source_document_name"].strip()
            assert reference["section_or_page"].strip()

    async def test_the_confidence_level_is_within_range_when_present(
        self, api: Api, stack: Stack
    ) -> None:
        """Optional by BR-03, and a *retrieval* measure rather than a claim about correctness —
        it is the mean similarity of the cited passages."""
        response = await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        body = response.json()
        for reference in body["source_references"]:
            level = reference["confidence_level"]
            if level is not None:
                assert 0.0 <= level <= 1.0
        if body["confidence_score"] is not None:
            assert 0.0 <= body["confidence_score"] <= 1.0

    async def test_the_cited_page_exists_in_the_stored_chunks(
        self, api: Api, stack: Stack
    ) -> None:
        """"Resolvable" is the whole claim. A citation is only traceable if the page it names
        is a page that actually exists on a stored chunk row."""
        response = await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        stored_pages = {
            f"page {chunk.page_number}"
            for chunk in stack.queries.chunks_for(stack.a.id)
            if chunk.page_number is not None
        }
        for citation in citations(response.json()):
            assert citation in stored_pages, f"{citation} is not a stored page"

    async def test_the_cited_document_id_resolves_to_a_readable_document(
        self, api: Api, stack: Stack
    ) -> None:
        """The end-to-end traceability claim: a reference the asker can actually open."""
        client = switch_to(api, stack.a.admin)
        await upload(client)
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("25 USD [S1].")

        body = (await ask(client, WIRE_QUESTION)).json()

        for reference in body["source_references"]:
            document_id = reference["document_id"]
            assert document_id, "a citation without a document id cannot be opened"
            fetched = await client.get(f"{API}/documents/{document_id}")
            assert fetched.status_code == 200
            assert fetched.json()["name"] == reference["source_document_name"]

    async def test_several_citations_each_appear_once(
        self, api: Api, stack: Stack
    ) -> None:
        """An answer citing three passages gets three distinct references, deduplicated on
        `(document_id, section_or_page)`. A reviewer reading "page 1, page 1, page 1" learns
        nothing the first entry did not already say, and two chunks from one page are one
        location."""
        response = await seed_and_ask(
            api,
            stack,
            WIRE_QUESTION,
            "Wire transfers cost 25 USD [S1], and other fees apply [S2] [S3] [S1].",
        )

        references = response.json()["source_references"]
        assert len(references) > 1, "a multi-marker answer should carry several references"
        pairs = [(ref["document_id"], ref["section_or_page"]) for ref in references]
        assert len(pairs) == len(set(pairs))


class TestUnsupportedAnswersArePrevented:
    """BR-03: "prevents unsupported answer generation", and the Data Integrity requirement
    "Answers derived only from uploaded documents, no external content" (C-04, R-01)."""

    async def test_an_off_corpus_question_is_declined(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        await upload(client)
        client = switch_to(api, stack.a.user)

        body = (await ask(client, OFF_CORPUS_QUESTION)).json()

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert body["answer_text"] == NOT_FOUND_ANSWER

    async def test_a_declined_answer_carries_no_citations(
        self, api: Api, stack: Stack
    ) -> None:
        """The inverse of BR-03's mandate. Citing a source for "I could not find this" would be
        worse than citing nothing."""
        client = switch_to(api, stack.a.admin)
        await upload(client)
        client = switch_to(api, stack.a.user)

        body = (await ask(client, OFF_CORPUS_QUESTION)).json()

        assert body["source_references"] == []

    async def test_a_question_the_corpus_does_not_cover_is_declined(
        self, api: Api, stack: Stack
    ) -> None:
        """The tax question asked *before* the tax document is uploaded.

        Tenant A is not empty — the harness seeded a fee schedule — so this is the interesting
        version of the not-found path: a corpus that is topically adjacent but does not contain
        the answer. An answer is queued deliberately; the pipeline must not reach it, because
        nothing retrieved clears the similarity floor.
        """
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("Withholding tax is 10 percent [S1].")

        body = (await ask(client, TAX_QUESTION)).json()

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert body["answer_text"] == NOT_FOUND_ANSWER

    async def test_a_model_that_declines_is_taken_at_its_word(
        self, api: Api, stack: Stack
    ) -> None:
        """Passages retrieved, and the model still says it cannot answer from them. The verdict
        is not-found, not an answered row with a citation nobody supported."""
        client = switch_to(api, stack.a.admin)
        await upload(client)
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("I cannot answer this from the provided documents.")

        body = (await ask(client, WIRE_QUESTION)).json()

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert body["source_references"] == []

    async def test_an_uncited_answer_is_not_published_as_answered(
        self, api: Api, stack: Stack
    ) -> None:
        """Prose with no `[Sn]` marker cannot be traced to a passage, so it cannot be an
        `ANSWERED` row — `QueryResponse` refuses that combination outright."""
        client = switch_to(api, stack.a.admin)
        await upload(client)
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("The fee is 25 USD.")

        body = (await ask(client, WIRE_QUESTION)).json()

        assert body["status"] != QueryStatus.ANSWERED.value
        assert body["source_references"] == []

    async def test_a_soft_deleted_document_stops_being_quotable_at_once(
        self, api: Api, stack: Stack
    ) -> None:
        """A vector can outlive a soft-delete by seconds; an answer must not
        (`rag-answer-grounding.md`). PostgreSQL stays the authority on status.

        Asked with `TAX_QUESTION`, because the seeded fee schedule stays ACTIVE and would
        answer a fee question on its own — which would make this pass whether the re-check
        worked or not. Only the uploaded document covers withholding tax.
        """
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        vectors_before = stack.index.count(stack.namespace(stack.a.id))
        # Change the status in the database only, leaving the index untouched — exactly the
        # window the re-check exists to close.
        await stack.documents.set_status(
            uuid.UUID(uploaded["id"]), stack.a.id, DocumentStatus.DELETED
        )
        assert stack.index.count(stack.namespace(stack.a.id)) == vectors_before, (
            "the vectors must still be in the index, or this test proves nothing"
        )

        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("Withholding tax is 10 percent [S1].")
        body = (await ask(client, TAX_QUESTION)).json()

        assert body["status"] == QueryStatus.NOT_FOUND.value
        assert body["source_references"] == []


class TestTheRequestBodyIsNarrow:
    """Mass assignment: `QueryCreate` forbids extra fields, so nothing server-assigned can be
    supplied by a caller."""

    async def test_a_body_naming_a_server_assigned_field_is_422(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        for field, value in (
            ("user_id", str(stack.b.user.id)),
            ("tenant_id", str(stack.b.id)),
            ("status", "ANSWERED"),
            ("answer_text", "A fabricated answer."),
            ("source_references", []),
        ):
            response = await client.post(
                f"{API}/queries",
                json={"question_text": WIRE_QUESTION, field: value},
                headers=csrf_headers(),
            )
            assert response.status_code == 422, field

    async def test_a_question_that_is_too_short_is_422(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        assert (await ask(client, "a")).status_code == 422

    async def test_a_question_that_is_too_long_is_422(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        assert (await ask(client, "fee " * 1000)).status_code == 422

    async def test_a_missing_question_is_422(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        response = await client.post(f"{API}/queries", json={}, headers=csrf_headers())

        assert response.status_code == 422

    async def test_submitting_without_a_csrf_token_is_refused(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        response = await client.post(
            f"{API}/queries", json={"question_text": WIRE_QUESTION}
        )

        assert response.status_code == 403


class TestReadingAQueryBack:
    """`GET /queries/{id}` — the endpoint a client polls, and the one an IDOR attempt hits."""

    async def test_the_asker_can_read_their_own_query(
        self, api: Api, stack: Stack
    ) -> None:
        submitted = (await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")).json()
        client, _ = api

        response = await client.get(f"{API}/queries/{submitted['id']}")

        assert response.status_code == 200
        assert response.json()["answer_text"] == submitted["answer_text"]

    async def test_another_users_query_is_404_for_a_standard_user(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.a.query is not None
        client = switch_to(api, stack.a.user)
        mine = (await ask(client, WIRE_QUESTION)).json()

        # `stack.a.query` was submitted by the same Standard User, so use tenant B's instead —
        # a real id belonging to a real person in another tenant.
        assert stack.b.query is not None
        response = await client.get(f"{API}/queries/{stack.b.query.id}")

        assert response.status_code == 404
        assert mine["id"] != str(stack.b.query.id)

    async def test_an_administrator_reads_any_query_in_their_own_tenant(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.a.query is not None
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{API}/queries/{stack.a.query.id}")

        assert response.status_code == 200

    async def test_an_administrator_cannot_read_another_tenants_query(
        self, api: Api, stack: Stack
    ) -> None:
        """Role and tenant are independent dimensions and both apply on every request."""
        assert stack.b.query is not None
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{API}/queries/{stack.b.query.id}")

        assert response.status_code == 404

    async def test_an_unknown_query_id_is_404(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        assert (await client.get(f"{API}/queries/{new_uuid()}")).status_code == 404

    async def test_a_denied_read_is_audited(self, api: Api, stack: Stack) -> None:
        assert stack.b.query is not None
        client = switch_to(api, stack.a.user)

        await client.get(f"{API}/queries/{stack.b.query.id}")

        assert stack.denials(), "a cross-tenant 404 is an authorization failure wearing a 404"


class TestLatencyAndAuditAreRecordedTogether:
    """NFR-9.1's measurement point, and BR-05's "query submissions" clause."""

    async def test_the_row_records_a_latency(self, api: Api, stack: Stack) -> None:
        """Recorded, not asserted to be fast. See the module docstring: the 3–5 second target
        is measured by `AnalyticsRepository.summary()` against production traffic, and a
        threshold asserted against a fake embedder would be a green gate over nothing."""
        body = (await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")).json()

        assert body["latency_ms"] is not None
        assert body["latency_ms"] >= 0
        assert body["answered_at"]

    async def test_submission_and_answer_are_both_audited(
        self, api: Api, stack: Stack
    ) -> None:
        """Two rows, one action. The phase lives in `detail` because `AuditAction` is a native
        PostgreSQL enum and adding a member needs an `ALTER TYPE` migration."""
        await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        phases = [
            row.detail.get("phase")
            for row in audit_rows(stack, "QUERY_SUBMIT")
            if row.detail
        ]
        assert "submit" in phases
        assert "answer" in phases

    async def test_the_answer_row_records_the_sources_it_cited(
        self, api: Api, stack: Stack
    ) -> None:
        """"Log all source references in audit trail for compliance reviews"
        (`document-source-traceability.md`)."""
        body = (await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")).json()

        answer_rows = [
            row
            for row in audit_rows(stack, "QUERY_SUBMIT")
            if row.detail and row.detail.get("phase") == "answer"
        ]
        assert answer_rows
        logged = answer_rows[-1].detail["sources"]
        assert [entry["section_or_page"] for entry in logged] == citations(body)

    async def test_no_audit_detail_quotes_the_question_or_the_document(
        self, api: Api, stack: Stack
    ) -> None:
        """A log that quotes a customer's question or a banking document is a second, less
        protected copy of it (`secure-coding-vapt.md`, Secrets/PII)."""
        await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        written = stack.audit_details()
        assert WIRE_QUESTION not in written
        for page in FEE_PAGES:
            assert page[:60] not in written

    async def test_the_submit_row_records_only_the_question_length(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_and_ask(api, stack, WIRE_QUESTION, "25 USD [S1].")

        submit_rows = [
            row
            for row in audit_rows(stack, "QUERY_SUBMIT")
            if row.detail and row.detail.get("phase") == "submit"
        ]
        assert submit_rows[-1].detail["question_length"] == len(WIRE_QUESTION)
