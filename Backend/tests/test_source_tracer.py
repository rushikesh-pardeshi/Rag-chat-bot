"""backend/tests/test_source_tracer.py

Role: Unit tests for `backend/services/source_tracer.py` — the grounding verdict and the
source references built from it.

This is the file that defends BR-03 and `.claude/rules/rag-answer-grounding.md` at the point
where they are still enforceable. Everything here runs on hand-written answer strings and
in-memory chunks: no AWS, no Pinecone, no database. That is the payoff of `rag_service` and
`source_tracer` being separate modules — the verdict is the part most likely to be wrong and
the part cheapest to test exhaustively.

The cases that matter most, and why
-----------------------------------
  * **An answer with no marker is discarded.** In a bank, an answer that cannot be attributed
    to a retrieved passage is not a lower-quality answer — it is a fabrication published under
    the bank's name. It must come back as `NOT_FOUND`, with the answer text *replaced*.
  * **An invented marker is dropped, not honoured.** `[S9]` when six passages were supplied is
    the model manufacturing a citation, which is the specific failure BR-03 exists to catch.
  * **References are built from the chunk, never from the model's prose.** A model that
    misnames a document in its sentences cannot make that name appear in the reference list.
  * **A reference always has both halves.** "Which document" with no "where in it" is not
    something a compliance reviewer can act on, and `SourceReference` rejects it downstream —
    so nothing here may construct one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from backend.core.config import Settings, get_settings
from backend.repositories.query_repository import RetrievedChunk
from backend.schemas.query_schemas import SourceReference
from backend.services.source_tracer import (
    MAX_EXCERPT_CHARS,
    NOT_FOUND_ANSWER,
    ScoredChunk,
    build_prompt_context,
    trace,
)

UPLOADED = datetime(2026, 3, 1, 9, 30, tzinfo=UTC)


@pytest.fixture
def settings() -> Settings:
    return get_settings()


def _chunk(
    *,
    name: str = "retail-fees.pdf",
    text: str = "The outbound wire transfer fee is 25 USD per payment.",
    page: int | None = 4,
    label: str = "",
    document_id: uuid.UUID | None = None,
    index: int = 0,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=document_id or uuid.uuid4(),
        document_name=name,
        text=text,
        page_number=page,
        section_label=label,
        chunk_index=index,
        upload_date=UPLOADED,
    )


def _passages(*scored: tuple[RetrievedChunk, float]) -> list[ScoredChunk]:
    return [ScoredChunk(chunk=chunk, score=score) for chunk, score in scored]


def _one(score: float = 0.9, **kwargs: object) -> list[ScoredChunk]:
    return _passages((_chunk(**kwargs), score))  # type: ignore[arg-type]


# -- The ungrounded answer ---------------------------------------------------
class TestGrounding:
    def test_an_answer_with_no_marker_is_replaced_not_returned(
        self, settings: Settings
    ) -> None:
        """The single most important assertion in this file.

        The model produced a fluent, plausible, entirely uncited sentence. Returning it with an
        empty reference list would publish an unattributable claim; returning it *with* the
        retrieved passages attached would be worse, because the citation would be a guess
        presented as provenance.
        """
        result = trace("The outbound wire fee is 25 USD.", _one(), settings=settings)

        assert result.grounded is False
        assert result.is_not_found is True
        assert result.answer_text == NOT_FOUND_ANSWER
        assert result.sources == ()
        assert result.confidence is None

    def test_a_cited_answer_is_kept_verbatim_with_its_markers(
        self, settings: Settings
    ) -> None:
        """Markers stay in the text on purpose.

        `[S1]` beside a sentence is how a reviewer connects that specific claim to that
        specific passage. Stripping them would leave a paragraph and a bibliography with no
        way to join them.
        """
        answer = "The outbound wire fee is 25 USD [S1]."
        result = trace(answer, _one(), settings=settings)

        assert result.grounded is True
        assert result.answer_text == answer
        assert "[S1]" in result.answer_text

    def test_an_invented_marker_cites_nothing(self, settings: Settings) -> None:
        """`[S9]` of one passage is a manufactured citation.

        The marker is dropped rather than mapped onto some passage, and because it was the
        answer's only marker the answer itself is then uncited — so the verdict is NOT_FOUND.
        """
        result = trace("The fee is 25 USD [S9].", _one(), settings=settings)
        assert result.grounded is False
        assert result.sources == ()

    def test_a_valid_marker_survives_alongside_an_invented_one(
        self, settings: Settings
    ) -> None:
        """The good citation is not thrown away with the bad one.

        Discarding a whole answer because the model added one spurious marker would refuse
        answers that are in fact supported. Only the invented marker is dropped.
        """
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9), (_chunk(name="b.pdf", page=2), 0.8)
        )
        result = trace(
            "Fee is 25 USD [S1], reviewed annually [S7].", passages, settings=settings
        )

        assert result.grounded is True
        assert [s["source_document_name"] for s in result.sources] == ["a.pdf"]

    @pytest.mark.parametrize(
        "answer",
        [
            "Information not found in uploaded documents.",
            "The documents do not contain the wire transfer fee [S1].",
            "The provided documents do not address overdraft limits.",
            "I cannot answer this from the provided passages [S1].",
        ],
    )
    def test_a_model_that_declines_is_recorded_as_not_found_even_if_it_cited(
        self, answer: str, settings: Settings
    ) -> None:
        """A refusal with an incidental marker is still a refusal.

        Two of these cite `[S1]` while declining. Treating them as answered would put a
        citation on a sentence that makes no claim — a reference trail leading to a passage
        that supports nothing.
        """
        result = trace(answer, _one(), settings=settings)
        assert result.grounded is False
        assert result.answer_text == NOT_FOUND_ANSWER
        assert result.sources == ()

    @pytest.mark.parametrize("answer", ["", "   ", "\n\t "])
    def test_an_empty_generation_is_not_found_rather_than_an_error(
        self, answer: str, settings: Settings
    ) -> None:
        """A model returning nothing is a bad answer, not a broken system.

        Raising here would surface as a 500 and tell the user DocQuery failed, when what
        actually happened is that no answer was produced — which is a NOT_FOUND.
        """
        assert trace(answer, _one(), settings=settings).grounded is False

    def test_no_passages_means_no_answer_regardless_of_what_was_generated(
        self, settings: Settings
    ) -> None:
        """Defensive: `rag_service` returns before generating when nothing survives retrieval.

        Asserted anyway, because this function is the last gate. If a future caller reaches
        generation with an empty passage list, the answer it produced came from model memory.
        """
        result = trace("The fee is 25 USD [S1].", [], settings=settings)
        assert result.grounded is False
        assert result.answer_text == NOT_FOUND_ANSWER


# -- The reference list ------------------------------------------------------
class TestReferences:
    def test_a_reference_carries_both_mandatory_halves(self, settings: Settings) -> None:
        """BR-03's two required fields, and the excerpt a reviewer reads without leaving."""
        result = trace("Fee is 25 USD [S1].", _one(page=4), settings=settings)
        reference = result.sources[0]

        assert reference["source_document_name"] == "retail-fees.pdf"
        assert reference["section_or_page"] == "page 4"
        assert reference["confidence_level"] == 0.9
        assert "25 USD" in reference["excerpt"]

    def test_every_reference_validates_as_a_source_reference(
        self, settings: Settings
    ) -> None:
        """These dicts are stored in `query.source_references` and revalidated on the way out.

        A payload that fails `SourceReference` would fail *after* retrieval and generation had
        been paid for, as a 500 on a query that actually worked. Checked at the source instead.
        """
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.91),
            (_chunk(name="b.pdf", page=None, label="4.2 Accrual"), 0.72),
        )
        result = trace("Both [S1] and [S2] apply.", passages, settings=settings)

        assert len(result.sources) == 2
        for reference in result.sources:
            model = SourceReference.model_validate(reference)
            assert model.source_document_name
            assert model.section_or_page

    def test_only_the_cited_passages_are_listed(self, settings: Settings) -> None:
        """A reference the answer did not rely on is a false trail for whoever audits it."""
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9),
            (_chunk(name="b.pdf", page=2), 0.8),
            (_chunk(name="c.pdf", page=3), 0.7),
        )
        result = trace("Only the second matters [S2].", passages, settings=settings)

        assert [s["source_document_name"] for s in result.sources] == ["b.pdf"]

    def test_references_are_ordered_as_the_answer_cited_them(
        self, settings: Settings
    ) -> None:
        """Citation order, not retrieval order, so the list reads with the prose."""
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9),
            (_chunk(name="b.pdf", page=2), 0.8),
            (_chunk(name="c.pdf", page=3), 0.7),
        )
        result = trace("Third [S3], then first [S1].", passages, settings=settings)
        assert [s["source_document_name"] for s in result.sources] == ["c.pdf", "a.pdf"]

    def test_a_passage_cited_three_times_is_one_source(self, settings: Settings) -> None:
        result = trace("A [S1]. B [S1]. C [S1].", _one(), settings=settings)
        assert len(result.sources) == 1

    def test_two_chunks_on_the_same_page_collapse_to_one_citation(
        self, settings: Settings
    ) -> None:
        """Adjacent chunks are one location to a reviewer.

        Listing "page 4" twice with two different confidence figures makes the reference list
        look unreliable, and invites the reader to wonder which of the two is the real one.
        """
        document_id = uuid.uuid4()
        passages = _passages(
            (_chunk(document_id=document_id, page=4, index=0), 0.9),
            (_chunk(document_id=document_id, page=4, index=1), 0.85),
        )
        result = trace("Fee is 25 USD [S1][S2].", passages, settings=settings)
        assert len(result.sources) == 1

    def test_the_same_page_of_two_different_documents_stays_two_citations(
        self, settings: Settings
    ) -> None:
        """The dedup key is (document, location) — page 4 of two policies is two facts."""
        passages = _passages(
            (_chunk(name="a.pdf", page=4), 0.9), (_chunk(name="b.pdf", page=4), 0.8)
        )
        result = trace("Both say so [S1][S2].", passages, settings=settings)
        assert len(result.sources) == 2

    def test_the_model_cannot_name_a_document_in_the_reference_list(
        self, settings: Settings
    ) -> None:
        """The separation of prose from provenance, asserted.

        The answer misnames the document and invents a page. Neither string reaches the
        reference, because the reference is built from the chunk record in PostgreSQL — the
        model only ever *selects* a passage, it never describes one.
        """
        answer = "According to fabricated-policy.pdf page 99, the fee is 25 USD [S1]."
        result = trace(answer, _one(name="retail-fees.pdf", page=4), settings=settings)

        reference = result.sources[0]
        assert reference["source_document_name"] == "retail-fees.pdf"
        assert reference["section_or_page"] == "page 4"

    def test_a_long_passage_is_truncated_rather_than_failing_validation(
        self, settings: Settings
    ) -> None:
        """`SourceReference.excerpt` caps at 2,000 characters and a chunk can be longer."""
        result = trace(
            "Fee is 25 USD [S1].", _one(text="x" * 5_000), settings=settings
        )
        assert len(result.sources[0]["excerpt"]) == MAX_EXCERPT_CHARS

    def test_an_uncitable_chunk_is_skipped_and_takes_the_answer_with_it(
        self, settings: Settings
    ) -> None:
        """Unreachable through ingestion — `ck_chunk_has_citation` forbids it — handled anyway.

        If the only cited passage cannot produce a reference, the result is NOT_FOUND rather
        than an answered query with an empty reference list, which `QueryResponse` would
        reject at the boundary.
        """
        result = trace(
            "Fee is 25 USD [S1].", _one(name="   ", page=None, label=""), settings=settings
        )
        assert result.grounded is False
        assert result.sources == ()


# -- Confidence --------------------------------------------------------------
class TestConfidence:
    def test_confidence_is_the_mean_of_the_cited_scores(self, settings: Settings) -> None:
        """The mean, not the maximum.

        A maximum would rate an answer leaning on one strong passage and three weak ones as
        highly as one supported throughout, and the second is the more trustworthy answer.
        """
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9), (_chunk(name="b.pdf", page=2), 0.5)
        )
        result = trace("Both [S1][S2].", passages, settings=settings)
        assert result.confidence == pytest.approx(0.7)

    def test_an_uncited_passages_score_does_not_move_the_confidence(
        self, settings: Settings
    ) -> None:
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9), (_chunk(name="b.pdf", page=2), 0.1)
        )
        result = trace("Only the first [S1].", passages, settings=settings)
        assert result.confidence == pytest.approx(0.9)

    @pytest.mark.parametrize(("score", "expected"), [(-0.4, 0.0), (1.8, 1.0)])
    def test_a_score_outside_zero_to_one_is_clamped(
        self, score: float, expected: float, settings: Settings
    ) -> None:
        """Cosine similarity is nominally -1..1; `confidence_level` is `ge=0.0, le=1.0`.

        An unclamped negative would fail validation and turn a working answer into a 500.
        """
        result = trace("Fee is 25 USD [S1].", _one(score=score), settings=settings)
        assert result.sources[0]["confidence_level"] == expected
        assert result.confidence == expected


# -- The prompt context ------------------------------------------------------
class TestPromptContext:
    def test_each_passage_is_numbered_from_one_with_its_citation(self) -> None:
        """The numbering is the contract the marker check reads back.

        An off-by-one here would make every marker in every answer point at the wrong passage
        — the citations would look valid and be systematically wrong.
        """
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9),
            (_chunk(name="b.pdf", page=None, label="4.2 Accrual"), 0.8),
        )
        context = build_prompt_context(passages)

        assert "[S1] Document: a.pdf" in context
        assert "Location: page 1" in context
        assert "[S2] Document: b.pdf" in context
        assert "Location: 4.2 Accrual" in context
        assert "[S0]" not in context

    def test_the_passage_text_is_present_because_that_is_the_whole_point(self) -> None:
        context = build_prompt_context(_one(text="The fee is 25 USD."))
        assert "The fee is 25 USD." in context

    def test_no_passages_renders_empty_rather_than_raising(self) -> None:
        assert build_prompt_context([]) == ""


class TestNumbering:
    """The cards are numbered 1..n, and the prose is rewritten to agree.

    The model's own markers are *retrieval ranks*: handed five passages it cites the first and
    the third, so the cards came out "1" and "3" under a heading reading "SOURCES (2)". Each
    test here pins one half of the fix — contiguous numbers, and prose that still points at
    the card it meant.
    """

    def test_cards_are_numbered_from_one_when_the_model_skips_a_rank(
        self, settings: Settings
    ) -> None:
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9),
            (_chunk(name="b.pdf", page=2), 0.8),
            (_chunk(name="c.pdf", page=3), 0.7),
        )
        result = trace(
            "Fees are listed [S1] and reviewed annually [S3].", passages, settings=settings
        )

        assert result.grounded is True
        assert [s["markers"] for s in result.sources] == [[1], [2]]
        # The prose follows the cards: no reader should see a [S3] above a list of two.
        assert "[S1]" in result.answer_text
        assert "[S2]" in result.answer_text
        assert "[S3]" not in result.answer_text
        # And each number still points at the passage the model actually used.
        assert [s["source_document_name"] for s in result.sources] == ["a.pdf", "c.pdf"]

    def test_citation_order_decides_the_numbers_not_retrieval_rank(
        self, settings: Settings
    ) -> None:
        """`[1]` is the first citation the reader meets, not Pinecone's best match.

        The answer cites the third passage first, so that passage becomes card 1 — otherwise
        the list reads in an order the prose does not.
        """
        passages = _passages(
            (_chunk(name="first.pdf", page=1), 0.9),
            (_chunk(name="second.pdf", page=2), 0.8),
            (_chunk(name="third.pdf", page=3), 0.7),
        )
        result = trace(
            "Start with the annex [S3], then the schedule [S1].", passages, settings=settings
        )

        assert [s["source_document_name"] for s in result.sources] == [
            "third.pdf",
            "first.pdf",
        ]
        assert result.answer_text.index("[S1]") < result.answer_text.index("[S2]")

    def test_two_passages_from_one_page_become_one_card_with_one_number(
        self, settings: Settings
    ) -> None:
        """A merged card used to answer to two markers; now it answers to one.

        Merging exists so a reviewer reads "page 4" once rather than twice with different
        scores. Keeping both of the model's numbers on it meant the prose said `[S1]` in one
        sentence and `[S3]` in another for the *same* card — which reads as two sources.
        """
        shared = uuid.uuid4()
        passages = _passages(
            (_chunk(name="lease.pdf", page=4, document_id=shared, index=0), 0.9),
            (_chunk(name="other.pdf", page=1), 0.5),
            (_chunk(name="lease.pdf", page=4, document_id=shared, index=1), 0.8),
        )
        result = trace(
            "The term is 36 months [S1] and rent is monthly [S3].",
            passages,
            settings=settings,
        )

        assert len(result.sources) == 1
        assert result.sources[0]["markers"] == [1]
        # Both sentences now cite the one card, and nothing points past it.
        assert result.answer_text.count("[S1]") == 2
        assert "[S3]" not in result.answer_text
        # The card keeps the stronger of the two merged scores.
        assert result.sources[0]["confidence_level"] == pytest.approx(0.9)

    def test_an_invented_marker_is_removed_from_the_prose(
        self, settings: Settings
    ) -> None:
        """It pointed at nothing before and rendered as inert text.

        Leaving it after renumbering would be worse than untidy: an invented `[S2]` sitting
        beside a renumbered `[S2]` reads as a citation of a card it has no relation to.
        """
        passages = _passages(
            (_chunk(name="a.pdf", page=1), 0.9), (_chunk(name="b.pdf", page=2), 0.8)
        )
        result = trace(
            "Fee is 25 USD [S1], reviewed annually [S7].", passages, settings=settings
        )

        assert result.grounded is True
        assert "[S7]" not in result.answer_text
        assert "[S1]" in result.answer_text
        # No double space or orphaned space before the full stop where it was removed.
        assert "  " not in result.answer_text
        assert " ." not in result.answer_text
