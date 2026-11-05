"""backend/tests/test_document_parser.py

Role: Unit tests for `backend/utils/document_parser.py` — type detection from magic bytes, text
normalization, and the citable-section shape that BR-03 depends on.

Roughly half of this file is hostile input. A document upload hands an attacker-supplied file
to a parsing library, and the four defences in that module (magic-byte sniffing, XXE off,
zip-bomb limits, page and text caps) are each asserted here against a fixture that would defeat
their absence: a renamed executable, a Word document declaring an external entity, an archive
whose declared expansion is a thousand-fold. A defence with no test is a comment.

The other half is about citations rather than safety. A page number that shifts when a blank
page is skipped, or a heading that fails to open a new section, produces an answer whose source
reference points at the wrong place — which is worse than no answer at all in a compliance
review, because it looks correct.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from backend.core.exceptions import UnsupportedDocumentType, ValidationError
from backend.tests.document_fixtures import (
    PLAIN_TEXT,
    WINDOWS_EXECUTABLE,
    make_docx,
    make_docx_with_external_entity,
    make_pdf,
    make_zip_bomb,
    make_zip_that_is_not_docx,
)
from backend.utils.document_parser import (
    CONTENT_TYPE_DOCX,
    CONTENT_TYPE_PDF,
    ParsedSection,
    clean_text,
    detect_content_type,
    parse_document,
    parse_docx,
    parse_pdf,
)


# -- Type detection ----------------------------------------------------------
class TestDetectContentType:
    def test_a_pdf_is_detected_from_its_leading_bytes(self) -> None:
        assert detect_content_type(make_pdf(["Statement of fees."])) == CONTENT_TYPE_PDF

    def test_a_docx_is_detected_as_a_zip_container(self) -> None:
        assert detect_content_type(make_docx([("Fees", "25 USD")])) == CONTENT_TYPE_DOCX

    @pytest.mark.parametrize(
        ("label", "data"),
        [
            ("windows executable", WINDOWS_EXECUTABLE),
            ("plain text", PLAIN_TEXT),
            ("empty file", b""),
            ("html", b"<!DOCTYPE html><html><body>hi</body></html>"),
            ("rtf", b"{\\rtf1\\ansi fee schedule}"),
        ],
    )
    def test_anything_else_is_refused(self, label: str, data: bytes) -> None:
        """The extension and the client's `Content-Type` are never consulted.

        Both are caller-controlled. `label` exists only so a failure names the payload.
        """
        with pytest.raises(UnsupportedDocumentType):
            detect_content_type(data)

    def test_the_magic_prefix_must_start_the_file(self) -> None:
        """A PDF signature further in does not make the file a PDF.

        Polyglot files — an executable with `%PDF-` a few bytes in — are the reason this is a
        `startswith` rather than a search.
        """
        with pytest.raises(UnsupportedDocumentType):
            detect_content_type(b"GIF89a" + make_pdf(["hidden"]))

    def test_parse_document_re_derives_the_type_when_none_is_given(self) -> None:
        parsed = parse_document(make_pdf(["Wire transfer fee: 25 USD."]))
        assert parsed.metadata["format"] == "pdf"
        assert parsed.has_text


# -- Text normalization ------------------------------------------------------
class TestCleanText:
    def test_a_nul_byte_is_removed(self) -> None:
        """NUL aborts a PostgreSQL INSERT, and a PDF text layer really does contain them."""
        assert "\x00" not in clean_text("Fee\x00 schedule")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Fee    schedule", "Fee schedule"),
            ("Fee\t\tschedule", "Fee schedule"),
            ("  Fee schedule  ", "Fee schedule"),
            ("Fee\x0bschedule", "Feeschedule"),
            ("Line one\n\n\n\n\nLine two", "Line one\n\nLine two"),
            ("", ""),
            ("\x00\x00", ""),
        ],
    )
    def test_normalization_table(self, raw: str, expected: str) -> None:
        assert clean_text(raw) == expected

    def test_newlines_survive_because_they_carry_structure(self) -> None:
        """A reviewer reading a citation needs the paragraph breaks the author wrote."""
        assert clean_text("Clause 1.\nClause 2.") == "Clause 1.\nClause 2."


# -- Citations ---------------------------------------------------------------
class TestCitation:
    @pytest.mark.parametrize(
        ("section", "expected"),
        [
            (ParsedSection(text="t", page_number=4), "Page 4"),
            (
                ParsedSection(text="t", page_number=None, section_label="Fee Schedule"),
                "Fee Schedule",
            ),
            (
                ParsedSection(text="t", page_number=4, section_label="Fee Schedule"),
                "Fee Schedule (page 4)",
            ),
        ],
    )
    def test_every_section_can_state_where_it_came_from(
        self, section: ParsedSection, expected: str
    ) -> None:
        """`section_or_page` in the API response is this string.

        There is no fourth case: a section with neither a page nor a label cannot be
        constructed by either parser, because a chunk with no citation fails BR-03.
        """
        assert section.citation == expected


# -- PDF ---------------------------------------------------------------------
class TestParsePdf:
    def test_pages_are_numbered_from_one(self) -> None:
        parsed = parse_pdf(make_pdf(["First page.", "Second page.", "Third page."]))
        assert [section.page_number for section in parsed.sections] == [1, 2, 3]
        assert parsed.page_count == 3
        assert parsed.truncated is False

    def test_skipping_a_blank_page_does_not_renumber_the_rest(self) -> None:
        """The citation must name the page the reader will open.

        Page 2 has no text layer, so it produces no section — but page 3 is still page 3. A
        parser that numbered sections by position would send a reviewer to the wrong page and
        look right while doing it.
        """
        parsed = parse_pdf(make_pdf(["First page.", "", "Third page."]))
        assert [section.page_number for section in parsed.sections] == [1, 3]
        assert parsed.page_count == 3
        assert parsed.metadata["pages_extracted"] == 2

    def test_a_document_with_no_text_layer_is_not_an_error(self) -> None:
        """This is a scan. The caller marks it `NEEDS_OCR` rather than failing the upload."""
        parsed = parse_pdf(make_pdf(["", "", ""]))
        assert parsed.sections == ()
        assert parsed.has_text is False
        assert parsed.page_count == 3

    def test_multi_line_page_text_is_kept_together(self) -> None:
        parsed = parse_pdf(make_pdf(["Clause 1.\nClause 2.\nClause 3."]))
        assert "Clause 1." in parsed.sections[0].text
        assert "Clause 3." in parsed.sections[0].text
        assert parsed.total_characters == len(parsed.sections[0].text)

    @pytest.mark.parametrize(
        ("label", "data"),
        [
            ("header only", b"%PDF-1.4\n"),
            ("truncated body", make_pdf(["Fee schedule."])[:120]),
            ("garbage after header", b"%PDF-1.4\n" + b"\xff" * 400),
        ],
    )
    def test_a_corrupt_pdf_raises_a_generic_validation_error(
        self, label: str, data: bytes
    ) -> None:
        """The message must not quote pypdf.

        pypdf's errors describe file structure and object offsets; that text would reach an API
        client and tell an attacker how far their malformed file got.
        """
        with pytest.raises(ValidationError) as raised:
            parse_pdf(data)
        assert "pypdf" not in str(raised.value).lower()

    def test_metadata_carries_no_document_body(self) -> None:
        """`metadata` lands in JSONB and then on a dashboard.

        Body text may hold account numbers, so only descriptive fields belong here
        (`.claude/rules/secure-coding-vapt.md`).
        """
        parsed = parse_pdf(make_pdf(["Account 4417829930 balance is 12,004.55."]))
        assert "4417829930" not in repr(parsed.metadata)
        assert set(parsed.metadata) <= {
            "format",
            "title",
            "author",
            "subject",
            "producer",
            "creator",
            "page_count",
            "pages_extracted",
            "failed_pages",
        }


# -- DOCX --------------------------------------------------------------------
class TestParseDocx:
    def test_each_heading_opens_a_new_citable_section(self) -> None:
        parsed = parse_docx(
            make_docx(
                [
                    ("Fee Schedule", "Outbound wires cost 25 USD."),
                    ("Eligibility", "Accounts open at least 30 days."),
                ]
            )
        )
        assert [section.section_label for section in parsed.sections] == [
            "Fee Schedule",
            "Eligibility",
        ]
        assert [section.page_number for section in parsed.sections] == [None, None]
        assert parsed.sections[0].citation == "Fee Schedule"

    def test_text_before_any_heading_is_cited_as_the_introduction(self) -> None:
        """`""` is not a citation, and a chunk that cannot be cited cannot be returned."""
        parsed = parse_docx(make_docx([("", "This policy applies to all retail accounts.")]))
        assert parsed.sections[0].section_label == "Introduction"

    def test_a_docx_has_no_page_numbers_at_all(self) -> None:
        """Pagination is a rendering decision Word makes, not something stored in the file.

        `page_count` is therefore the section count — the thing a reviewer can navigate by.
        """
        parsed = parse_docx(make_docx([("A", "one"), ("B", "two")]))
        assert all(section.page_number is None for section in parsed.sections)
        assert parsed.page_count == len(parsed.sections)
        assert parsed.metadata["section_count"] == len(parsed.sections)

    def test_the_heading_outline_is_kept_as_navigational_metadata(self) -> None:
        parsed = parse_docx(make_docx([("Fee Schedule", "x"), ("Eligibility", "y")]))
        assert parsed.metadata["sections"] == ["Fee Schedule", "Eligibility"]
        assert parsed.metadata["format"] == "docx"

    def test_core_properties_become_metadata(self) -> None:
        parsed = parse_docx(
            make_docx(
                [("Fee Schedule", "x")],
                title="Retail Fee Schedule 2026",
                author="Treasury Operations",
            )
        )
        assert parsed.metadata["title"] == "Retail Fee Schedule 2026"
        assert parsed.metadata["author"] == "Treasury Operations"

    def test_table_content_is_extracted_rather_than_dropped(self) -> None:
        """A rate card is often *only* a table.

        Dropping tables would make those documents unanswerable while the upload still looked
        like a success — the worst failure mode available, because nothing reports it.
        """
        from docx import Document as DocxDocument

        document = DocxDocument()
        document.add_heading("Fee Schedule", level=1)
        table = document.add_table(rows=2, cols=2)
        table.cell(0, 0).text = "Outbound wire"
        table.cell(0, 1).text = "25 USD"
        table.cell(1, 0).text = "Inbound wire"
        table.cell(1, 1).text = "10 USD"
        buffer = io.BytesIO()
        document.save(buffer)

        parsed = parse_docx(buffer.getvalue())
        tables = [s for s in parsed.sections if s.section_label == "Tables"]
        assert len(tables) == 1
        assert "Outbound wire | 25 USD" in tables[0].text
        assert "Inbound wire | 10 USD" in tables[0].text

    def test_a_plain_zip_is_refused_even_though_it_sniffs_as_a_docx(self) -> None:
        """Magic bytes get as far as `PK\\x03\\x04` and no further.

        `word/document.xml` is the only thing that distinguishes a Word document from any
        other archive, which is why detection is not the last line of defence.
        """
        with pytest.raises(UnsupportedDocumentType):
            parse_docx(make_zip_that_is_not_docx())

    def test_a_truncated_archive_is_a_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            parse_docx(make_docx([("A", "one")])[:200])


# -- Hostile archives and XML ------------------------------------------------
class TestArchiveDefences:
    def test_an_implausible_compression_ratio_is_refused_before_decompression(self) -> None:
        """Checked against the *declared* sizes in the central directory.

        That is the only way to know the cost before paying it: by the time a bomb has been
        handed to `python-docx`, the worker has already inflated it.
        """
        with pytest.raises(ValidationError):
            parse_docx(make_zip_bomb())

    def test_an_external_entity_is_not_resolved(self, tmp_path: Path) -> None:
        """XXE, end to end, against a real file on disk.

        The document is a valid Word package — only its XML is hostile — so nothing before the
        XML parser can reject it. `_harden_xml` sets `resolve_entities=False` process-wide at
        import; without it, the secret below would be extracted into a section, stored in
        PostgreSQL, and returned as a cited answer.

        A raised error is an equally acceptable outcome. What must never happen is the leak.
        """
        secret = tmp_path / "secret.txt"
        secret.write_text("ROOT_PASSWORD=hunter2", encoding="utf-8")
        uri = secret.as_uri()
        hostile = make_docx_with_external_entity(uri)

        # The package itself is intact — the archive checks cannot be what saves us here.
        with zipfile.ZipFile(io.BytesIO(hostile)) as archive:
            assert "word/document.xml" in archive.namelist()

        try:
            parsed = parse_docx(hostile)
        except ValidationError:
            return
        assert "hunter2" not in repr(parsed.sections)
        assert "ROOT_PASSWORD" not in repr(parsed.sections)
        assert "hunter2" not in repr(parsed.metadata)

    def test_a_remote_entity_is_not_fetched(self) -> None:
        """`no_network=True` — an entity naming an internal URL is SSRF with XML for a client.

        A DOCX uploaded by any authenticated user could otherwise make the server issue a
        request to `169.254.169.254` and store the credentials it returned.
        """
        hostile = make_docx_with_external_entity(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
        )
        try:
            parsed = parse_docx(hostile)
        except ValidationError:
            return
        assert "AccessKeyId" not in repr(parsed.sections)
        assert "169.254.169.254" not in repr(parsed.sections)
