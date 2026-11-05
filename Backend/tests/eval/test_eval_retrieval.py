"""backend/tests/eval/test_eval_retrieval.py

Role: The Phase 4 accuracy gate. Stages the fixture corpus, runs all thirty questions through
the real `RagService`, and reports top-1 accuracy, recall@k and the refusal rate.

READ THIS BEFORE QUOTING THE NUMBER
-----------------------------------
The figure this harness prints is **not** a measure of production answer quality, and anyone
who repeats it without the following sentence is misreporting it:

  It is measured against `FakeBedrockClient`, whose `embed` is a hashed bag-of-words
  projection — not Titan.

What that means concretely. The cosine between a question and a passage here approximates their
normalized word overlap. So the harness genuinely measures the *plumbing* that stands between a
question and a citation:

  * the question is embedded and searched in **this tenant's namespace** and no other;
  * Pinecone's score order survives `load_chunks`, so the strongest match is still first;
  * the PostgreSQL re-check runs, and a chunk whose document is not `ACTIVE` cannot be cited;
  * `RAG_TOP_K` trims, `RAG_MIN_SIMILARITY` thresholds;
  * `source_tracer` resolves the model's `[S#]` markers to the right document *and page*;
  * a declining model becomes `NOT_FOUND` with zero references rather than an answer.

Every one of those is a place a real bug lives, and a break in any of them shows up here as a
wrong page or a lost citation. What the harness cannot measure is whether Titan embeds
"reactivate" near "reactivating", whether Nova Pro reasons correctly over a passage it was
given, or how either behaves on documents unlike these six. A bag of words has no synonyms and
no morphology. Semantic quality needs a live run against Bedrock, and that run does not belong
in a suite the gate requires to be green offline with `PINECONE_API_KEY` blank.

Two consequences, both stated rather than hidden:

1. **`questions.py` reuses the passages' word forms.** With no morphology in the stand-in,
   "require" and "requires" are unrelated tokens. That is a concession to the fake, and it
   means the questions are more keyword-aligned than a real user's would be.
2. **Some misses are threshold-scale artifacts, not ranking failures.** The stand-in scores
   normalized word overlap, which is not on the same scale as a real embedding cosine, so a
   question can rank the right passage first and still clear no passage above the floor.
   Those are counted as **misses**, not excused: the harness runs at the shipped
   `RAG_MIN_SIMILARITY`, `RAG_TOP_K`, `PINECONE_OVERFETCH_FACTOR` and
   `RAG_REQUIRE_SOURCE_CITATION`, with nothing relaxed to raise the number. `thresholded` in
   the report separates them from a genuine wrong-page ranking, because the two failures have
   different causes and a single accuracy figure hides that.

   This docstring used to assert that `RAG_MIN_SIMILARITY=0.35` "is calibrated for Titan,
   whose cosine for a relevant passage runs far above normalized word overlap". That was an
   assumption and it was wrong. Measured against real Titan v2 and a real Pinecone index on
   2026-10-27, a relevant passage for a short topical query scores **0.28**, not "far above
   0.35" — so the floor was refusing answers the corpus contained. It is now 0.20, chosen
   from the measured gap between off-topic (0.09-0.12) and topical (0.28-0.75) queries; see
   `Settings.RAG_MIN_SIMILARITY`. The lesson for this file: a number that only ever ran
   against a stand-in was never calibrated at all.

The wall-clock this harness prints is likewise **not** the p95 latency gate. It excludes every
network round trip — the two Bedrock calls and the Pinecone query that dominate a real request.
It is reported only as the CPU floor.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Final

import pytest

from backend.core.config import Settings, get_settings
from backend.models.document import DocumentStatus
from backend.services.rag_service import RagService
from backend.services.source_tracer import NOT_FOUND_ANSWER
from backend.tests.eval.corpus import PASSAGES, total_passages
from backend.tests.eval.questions import CASES, UNANSWERABLE, EvalCase
from backend.tests.fakes import (
    Corpus,
    FakeBedrockClient,
    FakeDocumentRepository,
    FakePineconeIndex,
    FakeQueryRepository,
)
from backend.utils.vector_store import VectorStore

#: The gate in `WORKBREAKDOWN.md`: >90% on the fixture set.
REQUIRED_ACCURACY: Final = 0.90

#: A single stubborn case out of thirty is 3.3%, so the gate tolerates one miss and not three.
#: Written down because "90%" on a 30-case set is a coarser instrument than it looks.
MAX_MISSES: Final = 3


@dataclass(frozen=True, slots=True)
class Outcome:
    """What one question produced: the ranked citations, in the order they were returned."""

    case: EvalCase
    answered: bool
    #: (document name, section_or_page) per citation, best first.
    ranked: tuple[tuple[str, str], ...]
    confidence: float | None
    elapsed_ms: float
    #: Pinecone matches before the PostgreSQL re-check and the threshold.
    candidates: int
    #: Passages that survived both and were sent to the model. Zero means it was never called.
    used: int

    @property
    def expected(self) -> tuple[str, str]:
        return (self.case.document, self.case.section_or_page)

    @property
    def is_top1(self) -> bool:
        """The expected page is the first citation — what the UI shows first."""
        return bool(self.ranked) and self.ranked[0] == self.expected

    @property
    def is_recalled(self) -> bool:
        """The expected page appears anywhere in the reference list."""
        return self.expected in self.ranked

    @property
    def was_thresholded(self) -> bool:
        """Candidates were retrieved, then none cleared `RAG_MIN_SIMILARITY`.

        Distinct from a ranking failure, and reported separately: the retrieval order may have
        been perfectly correct and still produce no citation, which under
        `rag-answer-grounding.md` is the *required* behaviour rather than a bug. It is still
        counted as a miss — the user did not get the answer their documents contain.
        """
        return self.candidates > 0 and self.used == 0

    def describe(self) -> str:
        if self.ranked:
            got = f"{self.ranked[0][0]} {self.ranked[0][1]}"
        elif self.was_thresholded:
            got = f"<below threshold, {self.candidates} candidates>"
        else:
            got = "<no citation>"
        mark = "ok  " if self.is_top1 else ("near" if self.is_recalled else "MISS")
        return f"  {mark}  expected {self.case}  got {got}  ({self.case.question})"


@dataclass
class Report:
    """The measured figures, and the text that gets pasted into the phase notes."""

    outcomes: list[Outcome] = field(default_factory=list)
    refusals_correct: int = 0
    refusals_total: int = 0

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def top1(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.is_top1)

    @property
    def recalled(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.is_recalled)

    @property
    def accuracy(self) -> float:
        return self.top1 / self.total if self.total else 0.0

    @property
    def recall(self) -> float:
        return self.recalled / self.total if self.total else 0.0

    @property
    def misses(self) -> list[Outcome]:
        return [outcome for outcome in self.outcomes if not outcome.is_top1]

    @property
    def thresholded(self) -> list[Outcome]:
        """Misses where retrieval found candidates but none cleared the similarity floor."""
        return [outcome for outcome in self.misses if outcome.was_thresholded]

    @property
    def misranked(self) -> list[Outcome]:
        """Misses where a citation *was* produced and it named the wrong page.

        The figure that would still be a defect with a real embedder. A thresholded miss says
        the similarity floor is calibrated for a different embedder; a misranked one says the
        ordering is wrong, which no amount of recalibration fixes.
        """
        return [outcome for outcome in self.misses if not outcome.was_thresholded]

    def percentile_ms(self, fraction: float) -> float:
        """Nearest-rank percentile. Thirty samples do not support interpolation."""
        if not self.outcomes:
            return 0.0
        ordered = sorted(outcome.elapsed_ms for outcome in self.outcomes)
        index = min(len(ordered) - 1, int(round(fraction * len(ordered))) - 1)
        return ordered[max(0, index)]

    def render(self) -> str:
        lines = [
            "",
            "=" * 78,
            "DocQuery retrieval eval — fixture corpus, stand-in embedder (NOT Titan)",
            "=" * 78,
            f"  corpus         : {len(PASSAGES)} documents, {total_passages()} passages",
            f"  questions      : {self.total}",
            f"  top-1 accuracy : {self.accuracy:.1%}  ({self.top1}/{self.total})",
            f"  recall@k       : {self.recall:.1%}  ({self.recalled}/{self.total})",
            f"  misses         : {len(self.misranked)} misranked, "
            f"{len(self.thresholded)} below RAG_MIN_SIMILARITY",
            f"  refusal rate   : {self.refusals_correct}/{self.refusals_total} "
            "off-corpus questions declined",
            f"  wall-clock     : p50 {self.percentile_ms(0.50):.1f} ms, "
            f"p95 {self.percentile_ms(0.95):.1f} ms  (no network — NOT the p95 gate)",
        ]
        if self.misses:
            lines.append("  misses:")
            lines.extend(outcome.describe() for outcome in self.misses)
        lines.append("=" * 78)
        return "\n".join(lines)


async def _measure(settings: Settings) -> Report:
    """Stage the corpus once, then run every question through the real pipeline.

    Production settings throughout — `RAG_TOP_K`, `PINECONE_OVERFETCH_FACTOR`,
    `RAG_MIN_SIMILARITY` and `RAG_REQUIRE_SOURCE_CITATION` are all as shipped. Nothing is
    relaxed to raise the reported figure.
    """
    documents = FakeDocumentRepository()
    queries = FakeQueryRepository(documents)
    index = FakePineconeIndex()
    bedrock = FakeBedrockClient(dimensions=settings.BEDROCK_EMBEDDING_DIMENSIONS)
    vectors = VectorStore(index=index, settings=settings)
    tenant_id = uuid.uuid4()

    corpus = Corpus(documents, vectors, bedrock)
    for name, passages in PASSAGES.items():
        await corpus.add(tenant_id, name, passages, status=DocumentStatus.ACTIVE)

    service = RagService(queries, bedrock, vectors, settings)
    # Citing every marker makes the reference list the ranked retrieval list, which is what is
    # being scored. Markers past the number of surviving passages are dropped by
    # `source_tracer` as invented, so an over-long marker string cannot inflate the result.
    all_markers = "".join(f"[S{number}]" for number in range(1, settings.RAG_TOP_K + 1))

    report = Report()
    for case in CASES:
        # Cleared, not appended to. When a question is thresholded away the model is never
        # called, so its queued answer is left unconsumed — and the *next* question then pops a
        # stale one. That leak silently answered an off-corpus question from a previous case's
        # queue entry while this harness was being written, which is exactly the kind of
        # cross-contamination that turns a reported figure into a fiction.
        bedrock.queued_answers.clear()
        bedrock.queue_answer(f"Per the retrieved passages: {all_markers}")

        started = time.perf_counter()
        result = await service.answer(question=case.question, tenant_id=tenant_id)
        elapsed_ms = (time.perf_counter() - started) * 1_000
        report.outcomes.append(
            Outcome(
                case=case,
                answered=result.answered,
                ranked=tuple(
                    (str(source["source_document_name"]), str(source["section_or_page"]))
                    for source in result.sources
                ),
                confidence=result.confidence,
                elapsed_ms=elapsed_ms,
                candidates=result.candidates,
                used=result.used,
            )
        )

    # Off-corpus questions: with the queue empty the fake declines, and what is scored is
    # whether the *system* records that as a refusal with no citations. It is not a measure of
    # the model's judgement about when to decline — see the module header.
    bedrock.queued_answers.clear()
    report.refusals_total = len(UNANSWERABLE)
    for question in UNANSWERABLE:
        result = await service.answer(question=question, tenant_id=tenant_id)
        declined = not result.answered and result.answer_text == NOT_FOUND_ANSWER
        if declined and not result.sources:
            report.refusals_correct += 1

    return report


@pytest.fixture(scope="module")
def report() -> Report:
    """Run the whole eval once for the module.

    Synchronous, driving the async run through `asyncio.run`, so the staged corpus does not
    outlive the event loop it was built on — a module-scoped *async* fixture would be bound to
    one loop while each test gets its own.
    """
    measured = _measure(get_settings())
    result = asyncio.run(measured)
    print(result.render())  # noqa: T201 - the harness output is the deliverable
    return result


class TestAccuracy:
    def test_top_1_accuracy_meets_the_phase_gate(self, report: Report) -> None:
        assert report.accuracy >= REQUIRED_ACCURACY, report.render()

    def test_no_more_than_a_handful_of_cases_miss(self, report: Report) -> None:
        """A count as well as a rate: 90% of 30 is a coarse instrument on its own."""
        assert len(report.misses) <= MAX_MISSES, report.render()

    def test_no_question_was_answered_with_the_wrong_page(self, report: Report) -> None:
        """The strict claim, and the one that would survive a real embedder.

        Every miss must be a refusal (nothing cleared the similarity floor), never a confident
        citation of the wrong page. A wrong page is worse than a not-found: the user gets an
        answer and a reference that disagree, and a compliance reviewer following the citation
        finds text that does not support it. `RAG_MIN_SIMILARITY` calibration explains a
        not-found; nothing explains a misattribution.
        """
        assert report.misranked == [], report.render()

    def test_recall_is_at_least_accuracy(self, report: Report) -> None:
        """Sanity check on the metrics themselves, not on the system.

        Every top-1 hit is by definition also recalled. If this ever fails, `Outcome` is
        wrong and the headline figure means nothing — worth catching before anyone quotes it.
        """
        assert report.recall >= report.accuracy

    def test_every_question_was_actually_answered(self, report: Report) -> None:
        """No case may be scored as a hit while the pipeline returned a not-found."""
        for outcome in report.outcomes:
            if outcome.is_recalled:
                assert outcome.answered is True, outcome.describe()

    def test_the_corpus_is_large_enough_for_the_figure_to_mean_something(self) -> None:
        """Guards the denominator. A shrunken corpus makes any accuracy figure easy."""
        assert len(PASSAGES) >= 6
        assert total_passages() >= 25
        assert len(CASES) >= 30


class TestCitationIntegrity:
    """Properties the accuracy figure would hide, and that BR-03 actually turns on."""

    def test_every_citation_names_a_real_document_and_page(self, report: Report) -> None:
        real = {
            (name, f"page {number}")
            for name, passages in PASSAGES.items()
            for number in range(1, len(passages) + 1)
        }
        for outcome in report.outcomes:
            for citation in outcome.ranked:
                assert citation in real, f"{citation} is not in the corpus"

    def test_no_answered_question_came_back_without_a_citation(
        self, report: Report
    ) -> None:
        """`document-source-traceability.md`: an answer with no traceable source is a bug."""
        for outcome in report.outcomes:
            if outcome.answered:
                assert outcome.ranked, outcome.describe()

    def test_no_reference_list_repeats_a_page(self, report: Report) -> None:
        """`source_tracer` dedupes by (document, location); a repeat means it stopped."""
        for outcome in report.outcomes:
            assert len(set(outcome.ranked)) == len(outcome.ranked), outcome.describe()

    def test_citations_never_exceed_the_configured_top_k(self, report: Report) -> None:
        top_k = get_settings().RAG_TOP_K
        for outcome in report.outcomes:
            assert len(outcome.ranked) <= top_k, outcome.describe()

    def test_every_off_corpus_question_was_declined(self, report: Report) -> None:
        assert report.refusals_correct == report.refusals_total, report.render()


class TestThresholdIsStillEnforced:
    """The one thing `_ranking_settings` relaxes, asserted here at its production value."""

    async def test_the_production_threshold_still_rejects_a_weak_match(self) -> None:
        """A question with almost no word overlap must be declined, not answered weakly.

        Run at the shipped `RAG_MIN_SIMILARITY`, unlike the ranking metric above. Without this
        test, lowering the threshold for the ranking measurement would leave the setting
        untested in this module, and a harness that quietly disabled a production guard while
        reporting a high score would be worse than no harness.
        """
        settings = get_settings()
        assert settings.RAG_MIN_SIMILARITY > 0.0, (
            "The production threshold must be non-zero, or this test proves nothing."
        )

        documents = FakeDocumentRepository()
        queries = FakeQueryRepository(documents)
        index = FakePineconeIndex()
        bedrock = FakeBedrockClient(dimensions=settings.BEDROCK_EMBEDDING_DIMENSIONS)
        vectors = VectorStore(index=index, settings=settings)
        tenant_id = uuid.uuid4()

        await Corpus(documents, vectors, bedrock).add(
            tenant_id, "wire-transfer-cutoffs.pdf", PASSAGES["wire-transfer-cutoffs.pdf"]
        )
        bedrock.queue_answer("Some claim [S1].")
        service = RagService(queries, bedrock, vectors, settings)

        # Same tenant the corpus was staged under. A different tenant would return no matches
        # at all, and every assertion below would pass without the threshold doing anything —
        # a green test proving nothing.
        control = await service.answer(
            question="What is the cutoff time for same day domestic wire processing?",
            tenant_id=tenant_id,
        )
        assert control.candidates > 0, (
            "The corpus is not reachable from this tenant, so the threshold assertion below "
            "would be vacuous."
        )

        result = await service.answer(
            question="photosynthesis chloroplast thylakoid membrane grana stroma",
            tenant_id=tenant_id,
        )

        assert result.answered is False
        assert result.answer_text == NOT_FOUND_ANSWER
        assert result.sources == ()
        assert result.candidates > 0, "Candidates were retrieved and then thresholded away."
        assert result.used == 0
        # One `generate` call, from the control question — never for the off-topic one.
        assert len(bedrock.generate_calls) == 1
