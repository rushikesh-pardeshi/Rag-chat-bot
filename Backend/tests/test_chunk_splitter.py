"""backend/tests/test_chunk_splitter.py

Role: Unit tests for `backend/utils/chunk_splitter.py` — sizing, boundaries, overlap, and the
citation fields every chunk has to carry.

These tests were part of `test_document_service.py` in Phase 3, when chunking was inline in
`document_service._split_section`. Phase 4 moved chunking into its own module and made it
token-aware and overlapping, so the tests moved with it and were rewritten against
`split_sections` — the function the ingestion pipeline actually calls.

Nothing here builds a document, touches a repository, or needs a settings override beyond
`CHUNK_SIZE_TOKENS`. That is the payoff of keeping the splitter pure: the boundary cases below
are the ones most likely to break, and they are cheap enough to cover exhaustively.

Two properties are defended harder than the rest, because getting either wrong is a compliance
defect rather than a quality one:

  * **Every chunk is citable.** A chunk with neither a page number nor a section label cannot
    satisfy BR-03, so `split_sections` must never produce one — inheritance from the section is
    what guarantees it.
  * **Overlap never crosses a section boundary.** A chunk whose text spans page 4 and page 5
    while its citation names one of them is a *wrong* citation, which is worse than a missed
    retrieval: a reviewer following the reference finds text that is not there.
"""

from __future__ import annotations

import pytest

from backend.core.config import Settings, get_settings
from backend.utils.chunk_splitter import (
    CHARS_PER_TOKEN,
    MAX_CHUNKS_PER_DOCUMENT,
    MIN_CHUNK_CHARS,
    ChunkSet,
    _find_boundary,
    _merge_slivers,
    _split_text,
    estimate_tokens,
    split_sections,
)
from backend.utils.document_parser import ParsedSection

#: Small enough that a test fixture is readable, large enough that `MIN_CHUNK_CHARS` (80) is
#: still well below it — otherwise the sliver-merging tests would be testing nothing.
TARGET_TOKENS = 100
TARGET_CHARS = TARGET_TOKENS * CHARS_PER_TOKEN  # 400
OVERLAP_TOKENS = 10
OVERLAP_CHARS = OVERLAP_TOKENS * CHARS_PER_TOKEN  # 40


@pytest.fixture
def settings() -> Settings:
    """Production settings with the two chunk sizes overridden.

    `model_copy` rather than a fresh `Settings()`: everything else — the database URL, the
    secret key — should stay exactly as the harness configured it, and re-reading the
    environment here would make this fixture the second place that does so.
    """
    return get_settings().model_copy(
        update={
            "CHUNK_SIZE_TOKENS": TARGET_TOKENS,
            "CHUNK_OVERLAP_TOKENS": OVERLAP_TOKENS,
        }
    )


def _page(text: str, page: int = 1) -> ParsedSection:
    return ParsedSection(text=text, page_number=page)


def _heading(text: str, label: str) -> ParsedSection:
    return ParsedSection(text=text, page_number=None, section_label=label)


def _sentences(count: int, marker: str = "word") -> str:
    """`count` short sentences, each ending in a period so a sentence boundary exists."""
    return " ".join(f"The {marker} number {i} appears here." for i in range(count))


# -- Sizing ------------------------------------------------------------------
class TestSizing:
    def test_text_under_the_target_is_one_chunk_unchanged(
        self, settings: Settings
    ) -> None:
        result = split_sections([_page("The base rate is 6.5 percent.")], settings=settings)
        assert len(result.chunks) == 1
        assert result.chunks[0].text == "The base rate is 6.5 percent."
        assert result.truncated is False

    def test_an_over_long_paragraph_is_hard_split_rather_than_dropped(self) -> None:
        """A wall of text with no boundary at all is real — OCR output, or a table row.

        It must not become one oversized chunk that Titan truncates silently, and it must not
        be discarded. `_split_text` is called directly because the point is the character
        arithmetic, not the section plumbing.
        """
        pieces = _split_text("x" * (TARGET_CHARS * 2 + 200), TARGET_CHARS, 0)
        assert len(pieces) == 3
        assert all(len(piece) <= TARGET_CHARS for piece in pieces)
        # Nothing lost: with no overlap the pieces reassemble to the original.
        assert sum(len(piece) for piece in pieces) == TARGET_CHARS * 2 + 200

    def test_no_chunk_exceeds_the_target_even_with_overlap(
        self, settings: Settings
    ) -> None:
        """Overlap adds text to the *front* of a chunk, so it is the constraint most easily
        broken by an off-by-one — and an oversized chunk is truncated by the embedding model
        without an error, producing a vector for text the citation does not cover."""
        result = split_sections([_page(_sentences(200))], settings=settings)
        assert len(result.chunks) > 1
        assert all(len(chunk.text) <= TARGET_CHARS for chunk in result.chunks)

    def test_the_document_wide_chunk_cap_is_reported_not_silent(
        self, settings: Settings
    ) -> None:
        """`truncated` exists so a partial corpus is visible rather than assumed complete.

        Driven with many small sections rather than one enormous one: the cap is enforced in
        `split_sections`, and a 2,000-page scan is a long list of pages, not a single blob.
        """
        page = "The quarterly fee schedule applies to every retail account. " * 2
        sections = [
            _page(page, number) for number in range(MAX_CHUNKS_PER_DOCUMENT + 50)
        ]
        result = split_sections(sections, settings=settings)
        assert result.truncated is True
        assert len(result.chunks) == MAX_CHUNKS_PER_DOCUMENT


# -- Boundaries --------------------------------------------------------------
class TestBoundaries:
    def test_a_paragraph_break_is_preferred_over_a_sentence_break(self) -> None:
        """Both are present in the window; the split has to land on the paragraph.

        A chunk that ends mid-topic retrieves for the wrong question. The paragraph break is
        the strongest signal the text itself offers about where a topic ends.
        """
        # Sized so the paragraph break lands in the *second* half of the window — see
        # `test_a_boundary_search_never_backs_up_past_halfway` for why the floor exists.
        first = _sentences(8, "alpha")
        second = _sentences(8, "beta")
        text = f"{first}\n\n{second}"
        assert TARGET_CHARS // 2 < len(first) < TARGET_CHARS  # the fixture's premise

        cut = _find_boundary(text, 0, TARGET_CHARS)
        # Exactly the paragraph break, not a sentence end inside `second`.
        assert text[:cut].rstrip() == first

    def test_a_sentence_break_is_used_when_there_is_no_paragraph(self) -> None:
        text = _sentences(40)
        cut = _find_boundary(text, 0, TARGET_CHARS)
        assert text[:cut].rstrip().endswith(".")

    def test_a_boundary_search_never_backs_up_past_halfway(self) -> None:
        """Below halfway a clean boundary costs more than it buys.

        A chunk under half the target size retrieves worse than a slightly ragged full one, so
        an early paragraph break must be ignored rather than honoured.
        """
        text = "Short intro.\n\n" + ("y" * (TARGET_CHARS * 2))
        cut = _find_boundary(text, 0, TARGET_CHARS)
        assert cut > TARGET_CHARS // 2

    def test_text_with_no_boundary_at_all_is_cut_at_the_target(self) -> None:
        """A URL or a spaceless OCR line. A blunt cut beats an oversized chunk."""
        assert _find_boundary("z" * (TARGET_CHARS * 2), 0, TARGET_CHARS) == TARGET_CHARS

    def test_a_pathological_section_terminates(self) -> None:
        """The `start + 1` guard in `_split_text`, asserted rather than trusted.

        An infinite loop here would not raise — it would hold a Celery worker slot forever,
        which is the failure mode that looks like a hung queue rather than an error.
        """
        pieces = _split_text("​" * (TARGET_CHARS * 4), TARGET_CHARS, OVERLAP_CHARS)
        assert pieces  # terminated at all, which is the assertion


# -- Overlap -----------------------------------------------------------------
class TestOverlap:
    def test_consecutive_chunks_share_text_so_a_split_sentence_is_retrievable(
        self, settings: Settings
    ) -> None:
        result = split_sections([_page(_sentences(120))], settings=settings)
        assert len(result.chunks) >= 2
        first, second = result.chunks[0].text, result.chunks[1].text
        # The tail of the first chunk reappears at the head of the second. Compared on a short
        # slice rather than the full overlap window, because the boundary search moves the
        # exact split point: the guarantee is "they share text", not "they share N characters".
        tail = first[-20:]
        assert tail in second

    def test_overlap_never_crosses_a_section_boundary(self, settings: Settings) -> None:
        """The wrong-citation case, and the reason overlap is per-section.

        Page 1's closing words must not appear in a chunk that cites page 2. A reviewer who
        follows that citation reads a page that does not contain the quoted text.
        """
        page_one = _sentences(60, "alpha")
        page_two = _sentences(60, "beta")
        result = split_sections(
            [_page(page_one, 1), _page(page_two, 2)], settings=settings
        )
        for chunk in result.chunks:
            if chunk.page_number == 2:
                assert "alpha" not in chunk.text
            else:
                assert "beta" not in chunk.text

    def test_an_overlap_at_or_above_the_target_is_clamped(self) -> None:
        """Otherwise each chunk begins where the last one began and the loop never advances."""
        absurd = get_settings().model_copy(
            update={
                "CHUNK_SIZE_TOKENS": TARGET_TOKENS,
                # Larger than the chunk itself — a plausible misconfiguration.
                "CHUNK_OVERLAP_TOKENS": TARGET_TOKENS * 2,
            }
        )
        result = split_sections([_page(_sentences(120))], settings=absurd)
        assert len(result.chunks) > 1
        # Distinct chunks, not the same window repeated.
        assert len({chunk.text for chunk in result.chunks}) == len(result.chunks)


# -- Slivers -----------------------------------------------------------------
class TestSlivers:
    def test_a_trailing_sliver_is_merged_rather_than_stored_alone(self) -> None:
        """A "Page 4 of 12" footer is the common case. Two words cite nothing useful."""
        body = "z" * TARGET_CHARS
        merged = _merge_slivers([body, "short tail"])
        assert len(merged) == 1
        assert merged[0].endswith("short tail")

    def test_a_piece_at_the_minimum_length_stands_on_its_own(self) -> None:
        body = "z" * TARGET_CHARS
        keeper = "k" * MIN_CHUNK_CHARS
        assert _merge_slivers([body, keeper]) == [body, keeper]

    def test_a_leading_sliver_survives_because_it_has_no_predecessor(self) -> None:
        """Dropping it would lose text. A short first paragraph is still content."""
        assert _merge_slivers(["Preface.", "z" * TARGET_CHARS])[0] == "Preface."

    def test_blank_pieces_are_dropped(self) -> None:
        assert _merge_slivers(["  ", "\n\t", ""]) == []


# -- Citability (BR-03) ------------------------------------------------------
class TestCitability:
    def test_a_blank_section_produces_no_chunks(self, settings: Settings) -> None:
        assert split_sections([_page("   \n\n  \t ")], settings=settings).chunks == ()

    def test_no_sections_at_all_produces_an_empty_set_not_an_error(
        self, settings: Settings
    ) -> None:
        result = split_sections([], settings=settings)
        assert isinstance(result, ChunkSet)
        assert result.chunks == ()
        assert result.truncated is False

    def test_every_chunk_inherits_its_sections_page_and_label(
        self, settings: Settings
    ) -> None:
        result = split_sections(
            [
                _page(_sentences(120), 7),
                _heading(_sentences(120), "4.2 Interest Accrual"),
            ],
            settings=settings,
        )
        assert len(result.chunks) > 2
        for chunk in result.chunks:
            assert chunk.page_number == 7 or chunk.section_label == "4.2 Interest Accrual"

    def test_no_chunk_is_ever_blank_or_uncitable(self, settings: Settings) -> None:
        """The two CHECK constraints on `document_chunk`, asserted at the source.

        The database would reject a violation, but no database runs here — and a chunk that
        cannot be cited should be impossible to *construct*, not merely impossible to insert.
        """
        result = split_sections(
            [_page(_sentences(80), 3), _heading(_sentences(80), "Fees")],
            settings=settings,
        )
        for chunk in result.chunks:
            assert chunk.text.strip()
            assert chunk.page_number is not None or chunk.section_label.strip()
            assert chunk.token_count >= 1
            assert chunk.citation.strip()

    def test_chunk_index_runs_continuously_across_sections(
        self, settings: Settings
    ) -> None:
        """The index is document reading order, which is what makes adjacent chunks adjacent.

        Per-section numbering would collide on `uq_chunk_document_index` and would make
        "the passage after this one" unanswerable.
        """
        result = split_sections(
            [_page(_sentences(120), 1), _page(_sentences(120), 2)], settings=settings
        )
        assert [c.chunk_index for c in result.chunks] == list(range(len(result.chunks)))

    @pytest.mark.parametrize(
        ("page", "label", "expected"),
        [
            (17, "", "page 17"),
            (None, "Fee Schedule", "Fee Schedule"),
            (17, "4.2 Accrual", "4.2 Accrual (page 17)"),
            (None, "", "unspecified section"),
        ],
    )
    def test_the_citation_string_is_never_empty(
        self, page: int | None, label: str, expected: str
    ) -> None:
        """`SourceReference.section_or_page` is `min_length=1`.

        An empty citation would fail validation at the response boundary — after retrieval and
        generation had already been paid for.
        """
        from backend.utils.chunk_splitter import TextChunk

        chunk = TextChunk(
            text="body", chunk_index=0, page_number=page, section_label=label, token_count=1
        )
        assert chunk.citation == expected


# -- Token estimation --------------------------------------------------------
class TestTokenEstimation:
    def test_empty_text_is_zero_tokens(self) -> None:
        assert estimate_tokens("   \n ") == 0

    def test_any_real_content_is_at_least_one_token(self) -> None:
        """`token_count` is used as a divisor when budgeting a prompt.

        A zero there is a crash on a code path that runs on every single query.
        """
        assert estimate_tokens("a") == 1

    def test_the_estimate_scales_with_length(self) -> None:
        assert estimate_tokens("x" * 400) == 100
