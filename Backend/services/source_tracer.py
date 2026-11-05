"""backend/services/source_tracer.py

Role: Decide whether a generated answer is grounded, and turn the passages it used into
source references. This module owns BR-03 and `.claude/rules/rag-answer-grounding.md` at the
point where they can still be enforced.

`rag_service` runs the pipeline; this module makes the verdict. They are separate because the
verdict is the part that must be testable on strings alone — every interesting case here
("the model answered without citing anything", "the model cited [S7] when six passages were
supplied") is reachable with a hand-written answer string and needs no AWS, no Pinecone and no
database.

How grounding is actually checked
---------------------------------
The prompt numbers each retrieved passage `[S1]`, `[S2]`, … and instructs the model to mark
every claim with the marker it came from. This module then reads the markers back:

  * **No marker at all** → the answer is not attributable to any retrieved passage, so it is
    discarded and replaced with `NOT_FOUND_ANSWER`. This is the case that matters. An
    ungrounded answer is not a lower-quality answer; in a banking compliance context it is a
    fabrication with a bank's name on it.
  * **A marker outside the supplied range** (`[S9]` of six passages) → that one marker is
    dropped. It is the model inventing a citation, and an invented citation is exactly what
    BR-03 exists to catch.
  * **Only some passages cited** → normal and expected. References list what was used, not
    everything retrieved, because a reference the answer did not rely on is a false trail for
    whoever audits it.

This is verification, not trust: the check is on what the model *pointed at*, and the
reference text is then built from the chunk in PostgreSQL, never from anything the model
wrote. The model cannot name a document, a page, or a section in the reference list — it can
only select from passages it was given. That is the difference between a citation and a
plausible-looking string.

What this module does not claim
-------------------------------
`confidence_level` is the **retrieval** similarity for the cited passage, not a probability
that the answer is correct, and `_answer_confidence` is the same figure aggregated. Nothing
here measures whether the model reasoned correctly over a passage it genuinely retrieved. The
docstrings say so rather than letting a number imply a guarantee it cannot make — a
compliance reviewer reading "confidence 0.87" is owed an accurate account of what was
measured.

Markers stay in the answer text, deliberately. `[S2]` next to a sentence is how a reviewer
sees which passage that specific sentence rests on; stripping them would leave a paragraph
with a bibliography and no way to connect the two.

Which is why each reference carries the marker numbers that point at it (`markers`). The list
is neither in retrieval order nor one-entry-per-marker — it is in citation order with
same-page passages merged — so a client that numbered the cards by position would label the
wrong card and hide any marker whose card had been merged away. `_build_references` explains
the two mechanisms; the field is the contract that lets a reader follow `[S4]` to the passage
it names.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Any, Final

from backend.core.config import Settings, get_settings
from backend.core.logging import get_logger
from backend.repositories.query_repository import RetrievedChunk

logger = get_logger(__name__)

#: Returned instead of an answer whenever retrieval or grounding fails. Fixed text, required
#: verbatim by `rag-answer-grounding.md`: a model-authored apology is still model-authored
#: content and could carry an unsupported claim inside the refusal.
#:
#: Defined here rather than in `backend/schemas/query_schemas.py` (which re-exports it) because
#: this module decides *when* it is used, and the constant belongs with the decision.
NOT_FOUND_ANSWER: Final = "Information not found in uploaded documents."

#: How a passage is labelled in the prompt and cited in the answer. `[S12]` as well as `[S1]`.
#:
#: A fixed pattern with a bounded quantifier — `{1,3}` caps it at 999 passages, which is three
#: orders of magnitude past `RAG_TOP_K`. `secure-coding-vapt.md` forbids a user-supplied regex;
#: this one is a literal, and the text it scans is model output rather than user input, but the
#: bound is there so a pathological answer full of `[S` cannot make matching expensive.
_MARKER: Final = re.compile(r"\[S(\d{1,3})\]")

#: Longest excerpt attached to a reference. The full passage can be several thousand
#: characters; `SourceReference.excerpt` caps at 2,000 and truncating here keeps the response
#: from failing validation on a long chunk.
MAX_EXCERPT_CHARS: Final = 1_200

#: Phrases a model uses when it is declining. Matched case-insensitively against the whole
#: answer, and only used as a *hint* — the marker check below is the real gate. This exists so
#: a model that correctly says "the documents do not cover this" is recorded as `NOT_FOUND`
#: rather than as an answered query with one incidental citation.
_REFUSAL_HINTS: Final = (
    "information not found in uploaded documents",
    "not found in the uploaded documents",
    "the provided documents do not",
    "the documents do not contain",
    "i cannot answer this from the provided",
)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A citable passage together with the similarity that retrieved it.

    Pairs a `RetrievedChunk` (PostgreSQL, authoritative on content and citation) with a score
    from a `VectorMatch` (Pinecone, authoritative on relevance). They are joined here rather
    than in either source because neither system should be asked to hold the other's fact.
    """

    chunk: RetrievedChunk
    score: float


@dataclass(frozen=True, slots=True)
class TraceResult:
    """The verdict: the answer to publish, its references, and its confidence.

    `grounded` is False when the answer was rejected. In that case `answer_text` has already
    been replaced with `NOT_FOUND_ANSWER` and `sources` is empty — the caller does not have to
    remember to do either, because "forgot to overwrite the ungrounded answer" is the one
    mistake in this file that ships a fabrication.
    """

    answer_text: str
    sources: tuple[dict[str, Any], ...]
    confidence: float | None
    grounded: bool

    @property
    def is_not_found(self) -> bool:
        return not self.grounded


def build_prompt_context(passages: list[ScoredChunk]) -> str:
    """Render retrieved passages as the numbered block the model is asked to cite.

    The marker, the document name and the location are all included. The name is in the prompt
    so the model can say "the 2024 audit report states…" naturally — but note that the
    reference list is built from the chunk record regardless of what the model writes, so a
    misquoted name in the prose cannot become a citation.
    """
    blocks = []
    for index, passage in enumerate(passages, start=1):
        chunk = passage.chunk
        blocks.append(
            f"[S{index}] Document: {chunk.document_name}\n"
            f"Location: {chunk.citation}\n"
            f"Passage:\n{chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def trace(
    answer: str, passages: list[ScoredChunk], *, settings: Settings | None = None
) -> TraceResult:
    """Verify an answer against the passages it was given, and build its references.

    Returns a `TraceResult` in every case; nothing here raises for an ungrounded answer,
    because an answer the model could not support is a normal outcome of asking a question the
    documents do not cover. It is a `NOT_FOUND` result, not an error — a 500 here would tell
    the user the system broke when in fact it worked exactly as required.
    """
    config = settings or get_settings()
    cleaned = (answer or "").strip()

    if not cleaned or not passages:
        return _not_found("empty_answer" if passages else "no_passages")

    if _looks_like_refusal(cleaned):
        # The model declined. Recorded as NOT_FOUND with no references even if it happened to
        # emit a marker while declining, since there is no claim for that marker to support.
        return _not_found("model_declined")

    cited, invented = _cited_indexes(cleaned, len(passages))

    if invented:
        # Logged as a count, not as the answer text: the answer is derived from customer
        # documents and `secure-coding-vapt.md` keeps document content out of logs.
        logger.warning(
            "source_tracer.invented_markers",
            invented=len(invented),
            supplied=len(passages),
        )

    if not cited:
        if config.RAG_REQUIRE_SOURCE_CITATION:
            return _not_found("no_citation")
        # Only reachable with the setting off, which is not a supported production
        # configuration. Every passage is then attributed, which is honest about the fact
        # that we do not know which one was used.
        cited = list(range(1, len(passages) + 1))

    references = _build_references(cited, passages)
    if not references:
        # Defensive: `cited` was non-empty but nothing survived reference construction. An
        # answered query with no references is unrepresentable downstream (QueryResponse
        # rejects it), so it is turned into NOT_FOUND here rather than raising there.
        logger.error("source_tracer.references_empty", cited=len(cited))
        return _not_found("references_empty")

    confidence = _answer_confidence([passages[index - 1].score for index in cited])
    renumbered = _renumber(cleaned, references)
    logger.info(
        "source_tracer.grounded",
        passages=len(passages),
        cited=len(references),
        confidence=confidence,
    )
    return TraceResult(
        answer_text=renumbered,
        sources=tuple(references),
        confidence=confidence,
        grounded=True,
    )


def _renumber(answer: str, references: list[dict[str, Any]]) -> str:
    """Rewrite the answer's markers so the cards read 1, 2, 3.

    Mutates `references` to match, so prose and cards cannot disagree.

    The numbers the model emits are *retrieval ranks*: it is handed five passages and cites,
    say, the first and the third, so the cards came out numbered "1" and "3" under a heading
    saying "SOURCES (2)". Two passages from the same page merge into one card as well, which
    could make a single card answer to both `[S2]` and `[S5]`. Both are faithful to what
    happened and neither is readable — a reader counting cards cannot find "3" when there are
    two of them.

    So the prose and the cards are renumbered together, in one pass, after the merge:

      * each reference takes its position in the list, which is already *citation order*, so
        `[1]` is the first citation the reader meets rather than the best Pinecone match;
      * every old marker that resolved to that reference is rewritten to its new number, so a
        merged card's two markers collapse into one — which is what a merged card means;
      * **invented markers are removed from the prose.** They point at nothing, so they
        rendered as inert text before; leaving them now would be worse than untidy, because
        an invented `[S2]` sitting next to a renumbered `[S2]` would read as a citation of a
        card it has no relation to.

    Renumbering here rather than in the UI is deliberate: `answer_text` and `source_references`
    are stored on the `query` row and re-read by the history page, the export and any
    compliance review. If the UI renumbered for display, the stored answer and the stored
    references would disagree with what the reviewer was shown, which is the traceability gap
    `markers` was added to close in the first place.
    """
    mapping: dict[int, int] = {}
    for position, reference in enumerate(references, start=1):
        for marker in reference["markers"]:
            mapping[marker] = position
        # The card now answers to exactly one number: its own.
        reference["markers"] = [position]

    def substitute(match: re.Match[str]) -> str:
        new = mapping.get(int(match.group(1)))
        # No mapping means the model invented it — drop it, and the stray space with it.
        return f"[S{new}]" if new is not None else ""

    rewritten = _MARKER.sub(substitute, answer)
    # Dropping a marker can leave a double space or a space before punctuation.
    return re.sub(r" {2,}", " ", rewritten).replace(" .", ".").replace(" ,", ",").strip()


def _not_found(reason: str) -> TraceResult:
    """The single construction site for a not-found verdict."""
    logger.info("source_tracer.not_found", reason=reason)
    return TraceResult(
        answer_text=NOT_FOUND_ANSWER, sources=(), confidence=None, grounded=False
    )


def _looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(hint in lowered for hint in _REFUSAL_HINTS)


def _cited_indexes(answer: str, supplied: int) -> tuple[list[int], list[int]]:
    """Markers the answer used, split into valid and invented, in first-appearance order.

    Order is the order the model cited them, not retrieval order, so the reference list reads
    in the same sequence as the answer. Duplicates collapse — a passage cited in three
    sentences is one source.
    """
    seen: dict[int, None] = {}
    invented: list[int] = []
    for match in _MARKER.finditer(answer):
        index = int(match.group(1))
        if 1 <= index <= supplied:
            seen.setdefault(index, None)
        elif index not in invented:
            invented.append(index)
    return list(seen), invented


def _build_references(cited: list[int], passages: list[ScoredChunk]) -> list[dict[str, Any]]:
    """Build the JSONB reference payloads for the cited passages.

    Keys match `SourceReference` field names exactly, because these dicts are stored in
    `query.source_references` and revalidated through that model on the way out. Plain dicts
    rather than schema instances keeps `services/` from importing `schemas/`.

    Every reference carries `markers`: the `[S#]` numbers in the answer that resolve to it.
    That field is what makes the list checkable. Two facts about this function make position
    the wrong way to match a marker to a card, and both are wanted:

      * `cited` is in the order the model cited them, not retrieval order, so `[S4]` can be
        the first entry in the list;
      * two adjacent chunks on the same page are merged into one citation — a reviewer reads
        "page 4" once, not twice with different scores — so one card can answer to `[S2]`
        *and* `[S5]`.

    Before `markers` existed, a UI numbering the cards 1..n showed "S1, S2" under an answer
    citing `[S4]`, and the reader had no way to reach the passage the sentence rested on. That
    is a BR-03 traceability failure, not a display quirk: an unreachable citation is not a
    citation. The merged card now lists both markers, and no cited marker is ever absent.
    """
    references: list[dict[str, Any]] = []
    by_key: dict[tuple[uuid.UUID, str], dict[str, Any]] = {}

    for index in cited:
        passage = passages[index - 1]
        chunk = passage.chunk
        key = (chunk.document_id, chunk.citation)
        merged = by_key.get(key)
        if merged is not None:
            # Same page, cited again under another marker. One card, both markers — dropping
            # the marker here is what hid S4.
            markers = merged["markers"]
            if index not in markers:
                markers.append(index)
            # The stronger of the two scores: the card stands for both passages, and the
            # weaker one would understate how well the page matched.
            merged["confidence_level"] = max(merged["confidence_level"], _clamp(passage.score))
            continue

        name = chunk.document_name.strip()
        location = chunk.citation.strip()
        if not name or not location:
            # Unreachable through normal ingestion: `document.name` is non-null and the
            # `ck_chunk_has_citation` constraint guarantees a location. Skipped rather than
            # allowed to fail `min_length=1` validation at the response boundary.
            logger.error("source_tracer.uncitable_chunk", chunk_id=str(chunk.chunk_id))
            continue

        reference: dict[str, Any] = {
            "source_document_name": name[:512],
            "section_or_page": location[:512],
            "document_id": str(chunk.document_id),
            "confidence_level": _clamp(passage.score),
            "excerpt": chunk.text[:MAX_EXCERPT_CHARS],
            "markers": [index],
        }
        by_key[key] = reference
        references.append(reference)

    # The list keeps the order the answer cited them — insertion order here *is* citation
    # order, because `cited` is. Deliberately not re-sorted by marker number: marker number is
    # retrieval rank, so sorting on it would put the list back in the order Pinecone returned
    # rather than the order the prose reads in, which is the distinction
    # `test_references_are_ordered_as_the_answer_cited_them` exists to hold.
    #
    # Within one card the markers *are* ascending, so a merged citation renders "S2 S5" rather
    # than "S5 S2" when a later sentence reached the page first.
    for reference in references:
        reference["markers"].sort()
    return references


def _answer_confidence(scores: list[float]) -> float | None:
    """Aggregate retrieval similarity for the cited passages.

    The mean, not the maximum. A maximum reports the single best passage and would rate an
    answer that leaned on one strong match and three weak ones as highly as one supported
    throughout — and the second is the more trustworthy answer.

    Still a retrieval measure. It says the cited passages resembled the question, not that the
    answer drawn from them is correct.
    """
    usable = [_clamp(score) for score in scores]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 4)


def _clamp(score: float) -> float:
    """Force a score into 0–1.

    Cosine similarity from Pinecone is nominally -1..1 and in practice 0..1 for Titan
    embeddings, but `SourceReference.confidence_level` is `ge=0.0, le=1.0`. A negative score
    arriving unclamped would fail validation and turn a working answer into a 500.
    """
    return round(min(1.0, max(0.0, float(score))), 4)
