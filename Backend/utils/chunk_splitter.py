"""backend/utils/chunk_splitter.py

Role: Turn parsed sections into retrievable, citable chunks. Token-aware, with overlap.

This replaces the character-based chunking that Phase 3 kept inline in
`document_service._build_chunks`. Two things changed: sizes are expressed in tokens
(`CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`) rather than raw characters, and consecutive
chunks overlap so a sentence split across a boundary is still retrievable from both sides.

Why there is no tokenizer dependency
------------------------------------
Tokens are estimated at `CHARS_PER_TOKEN` characters each, not counted. Adding `tiktoken`
would buy precision we cannot use: it implements OpenAI's BPE, and this project embeds with
Titan and generates with Nova, whose tokenizers differ from it and from each other. An exact
count against the wrong vocabulary is false precision — a number that looks authoritative and
is not. The estimate is used for two things only, and both are tolerant of being ~15% off:

  * choosing a split point, where the true constraint is Titan's ~8k-token input and a chunk
    is targeted at 512; and
  * `token_count` on the row, which `rag_service` uses to budget how many chunks fit in a
    prompt — where it is bounded by `RAG_TOP_K` anyway.

If exact accounting is ever needed, Bedrock returns real token usage per call, and that is a
measurement rather than a guess.

Citations survive the split
---------------------------
Every chunk inherits its section's `page_number` and `section_label`. This is BR-03 in code
form: a chunk that cannot say where it came from cannot be cited, so it must not exist.
`document_chunk`'s CHECK constraint is the same requirement expressed in the database.

Overlap never crosses a section boundary. Two pages joined by an overlap window would produce
a chunk whose text spans page 4 and page 5 while its citation claims one of them — a wrong
citation, which is worse than a missed retrieval. Overlap is taken only from earlier text
within the same section.

Pure and ORM-free
-----------------
Nothing here imports a model or a session. It takes `ParsedSection` values and returns
`TextChunk` values; the caller (`document_service`, and the Celery task in
`backend/workers/document_processor.py`) maps those onto `DocumentChunk` rows. That is what
lets the splitter be tested on strings alone, which is the only way the boundary cases below
are cheap enough to cover.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final

from backend.core.config import Settings, get_settings
from backend.utils.document_parser import ParsedSection

#: Characters per token for English prose. Titan and Nova both land near this for the
#: business documents this system ingests; see the module docstring for why it is an estimate.
CHARS_PER_TOKEN: Final = 4

#: A fragment shorter than this is merged into its predecessor instead of standing alone. A
#: chunk of two words retrieves noisily — it can match a question on one incidental term — and
#: cites nothing a reviewer can act on.
MIN_CHUNK_CHARS: Final = 80

#: Hard ceiling on rows written for one document, independent of chunk size. A 2,000-page PDF
#: at four chunks a page is 8,000 rows; past this the document is stored and searchable up to
#: the cap and `truncated` says so, rather than the shortfall being silent.
MAX_CHUNKS_PER_DOCUMENT: Final = 10_000

#: Paragraph, then sentence, then whitespace. Tried in order, so a split lands at the most
#: semantically complete boundary available before falling back to a blunt character cut.
#:
#: All three are fixed literals with no user input and no nested quantifier — the ReDoS rule in
#: `secure-coding-vapt.md` forbids a caller-supplied pattern, and there is no path to one here.
_PARAGRAPH_BREAK: Final = re.compile(r"\n\s*\n")
_SENTENCE_END: Final = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One citable passage, ready to be written as a `document_chunk` row.

    `chunk_index` is global across the document rather than per section, because it is the
    order a reviewer reads the document in — and the order `rag_service` presents adjacent
    chunks to the model.
    """

    text: str
    chunk_index: int
    page_number: int | None
    section_label: str
    token_count: int

    @property
    def citation(self) -> str:
        """How this chunk names its own location, for a `section_or_page` reference.

        Prefers the section label when the document has real headings, because "Section 4.2
        Interest Accrual" is checkable by a reader in a way "page 17" is not. Falls back to
        the page, and never returns an empty string: `SourceReference.section_or_page` is
        `min_length=1`, so an empty value would fail validation at the response boundary
        after the expensive work was already done.
        """
        if self.section_label and self.page_number is not None:
            return f"{self.section_label} (page {self.page_number})"
        if self.section_label:
            return self.section_label
        if self.page_number is not None:
            return f"page {self.page_number}"
        return "unspecified section"


@dataclass(frozen=True, slots=True)
class ChunkSet:
    """The result of splitting a document: the chunks, and whether the cap was hit."""

    chunks: tuple[TextChunk, ...]
    truncated: bool

    def __len__(self) -> int:
        return len(self.chunks)

    def __iter__(self) -> Iterable[TextChunk]:  # type: ignore[override]
        return iter(self.chunks)


def estimate_tokens(text: str) -> int:
    """Approximate token count. At least 1 for any non-empty text.

    Never returns 0 for real content: `document_chunk.token_count` is used as a divisor when
    budgeting a prompt, and a zero there is a crash in a code path that runs on every query.
    """
    stripped = text.strip()
    if not stripped:
        return 0
    return max(1, len(stripped) // CHARS_PER_TOKEN)


def split_sections(
    sections: Sequence[ParsedSection], *, settings: Settings | None = None
) -> ChunkSet:
    """Split parsed sections into overlapping, citable chunks.

    Sections are processed in order and `chunk_index` runs continuously across them, so the
    index doubles as document reading order. A section that is empty after cleaning
    contributes nothing — a blank page in a scan is not a chunk.
    """
    config = settings or get_settings()
    target_chars = max(1, config.CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN)
    overlap_chars = max(0, config.CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN)
    # Overlap must stay strictly below the target, or each chunk would begin where the last
    # one began and the loop would not advance.
    overlap_chars = min(overlap_chars, target_chars // 2)

    chunks: list[TextChunk] = []
    truncated = False

    for section in sections:
        for piece in _split_text(section.text, target_chars, overlap_chars):
            if len(chunks) >= MAX_CHUNKS_PER_DOCUMENT:
                truncated = True
                return ChunkSet(chunks=tuple(chunks), truncated=truncated)
            chunks.append(
                TextChunk(
                    text=piece,
                    chunk_index=len(chunks),
                    page_number=section.page_number,
                    section_label=section.section_label,
                    token_count=estimate_tokens(piece),
                )
            )

    return ChunkSet(chunks=tuple(chunks), truncated=truncated)


def _split_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split one section's text into pieces of at most `target_chars`, with overlap.

    The strategy is greedy from the front: take as much as fits, back up to the nearest clean
    boundary, emit, then restart `overlap_chars` earlier than where the last piece ended.

    Greedy rather than balanced on purpose. A balanced split produces pieces of even size but
    moves every boundary when the document changes by one paragraph; greedy keeps earlier
    boundaries stable, which means re-processing an amended document leaves most chunks
    byte-identical and their embeddings unchanged.
    """
    stripped = text.strip()
    if not stripped:
        return []
    if len(stripped) <= target_chars:
        return [stripped]

    pieces: list[str] = []
    start = 0
    length = len(stripped)

    while start < length:
        end = start + target_chars
        if end >= length:
            pieces.append(stripped[start:].strip())
            break

        cut = _find_boundary(stripped, start, end)
        piece = stripped[start:cut].strip()
        if piece:
            pieces.append(piece)

        next_start = cut - overlap_chars if overlap_chars else cut
        # Guard against a boundary search that found nothing useful and left us where we
        # started. Without this a pathological section is an infinite loop, and an infinite
        # loop inside a Celery task holds a worker slot forever.
        start = max(next_start, start + 1) if cut > start else start + target_chars

    return _merge_slivers(pieces)


def _find_boundary(text: str, start: int, end: int) -> int:
    """Best split point at or before `end`, searching back no further than halfway.

    Halfway is the floor because a boundary found in the first half of the window would
    produce a chunk under half the target size, and many such chunks retrieve worse than
    fewer well-filled ones. Below that floor a blunt character cut is the better trade.
    """
    floor = start + (end - start) // 2
    window = text[start:end]

    for pattern in (_PARAGRAPH_BREAK, _SENTENCE_END):
        best = None
        for match in pattern.finditer(window):
            absolute = start + match.end()
            if absolute > floor:
                best = absolute
        if best is not None:
            return best

    space = text.rfind(" ", floor, end)
    if space > floor:
        return space
    # No boundary at all — a URL, a table row, or OCR output with no spaces. Cut mid-token
    # rather than emitting an oversized chunk that Titan would truncate silently.
    return end


def _merge_slivers(pieces: list[str]) -> list[str]:
    """Fold any piece shorter than `MIN_CHUNK_CHARS` into its neighbour.

    Applied after splitting rather than during it, because whether a piece is a sliver depends
    on where the *next* boundary landed. A trailing "Page 4 of 12" footer is the common case.
    """
    result: list[str] = []
    for piece in pieces:
        cleaned = piece.strip()
        if not cleaned:
            continue
        if len(cleaned) < MIN_CHUNK_CHARS and result:
            result[-1] = f"{result[-1]}\n\n{cleaned}"
        else:
            result.append(cleaned)
    return result
