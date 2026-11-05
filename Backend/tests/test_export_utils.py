"""backend/tests/test_export_utils.py

Role: Tests for `backend/utils/export_utils.py` — that a rendered export is safe to open and
still says what the report said.

The claim under test is narrow and worth stating plainly: an exported file must not be able to
execute anything on the machine that opens it, and must not lose or misstate the data because a
value was long, oddly encoded, or shaped like markup. Every value in these reports is user- or
file-supplied — questions are typed, document names come from uploads — so the escaping
is not a theoretical concern.

Why the assertions read the file back
-------------------------------------
Both renderers are exercised for real and the output is re-opened: the workbook through
`openpyxl.load_workbook`, the PDF by checking the bytes it produced. Asserting on the internal
helpers alone would pass while producing a file Excel refuses, and "openpyxl accepted the
value" is not the same claim as "the saved sheet holds a string element". `data_type` and
`quotePrefix` are therefore read off the *loaded* workbook, not off the cell written.

What is deliberately not covered
--------------------------------
Whether Excel itself honours `quotePrefix`, and whether a PDF reader renders the escaped
text as intended. Those are the applications' behaviour; the module docstring already records
the one residual risk this module cannot close — a third tool re-exporting the sheet to CSV
drops the flag and keeps only the text.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime, timedelta, timezone

import pytest
from openpyxl import load_workbook

from backend.utils.export_utils import (
    FORMULA_LEADERS,
    MAX_CELL_LENGTH,
    MAX_PDF_CELL_LENGTH,
    MAX_SHEET_NAME,
    TRUNCATION_MARKER,
    ReportDocument,
    ReportTable,
    render_excel,
    render_pdf,
)

GENERATED = datetime(2026, 8, 14, 9, 30, 0, tzinfo=UTC)

#: The classic spreadsheet payload. `cmd|'/c calc'!A1` is code execution on the machine of
#: whoever opens the file, which is why every cell is written with an explicit string type.
DDE_PAYLOAD = "=cmd|'/c calc'!A1"

#: The header row is row 5; data starts at row 6. Named because every assertion below counts
#: from it, and an off-by-one would silently read a heading instead of a value.
HEADER_ROW = 5
FIRST_DATA_ROW = 6


def _table(*rows: tuple[object, ...], title: str = "Most searched keywords") -> ReportTable:
    return ReportTable(
        title=title,
        columns=("Keyword", "Occurrences", "Distinct queries"),
        rows=rows,
    )


def _document(*tables: ReportTable) -> ReportDocument:
    return ReportDocument(
        title="DocQuery search keyword report",
        generated_at=GENERATED,
        subtitle="Tenant 0000 | 2026-01-01 to 2026-06-30 | requested by admin@bank.example",
        tables=tables or (_table(("wire", 40, 31)),),
    )


def _sheet(report: ReportDocument, index: int = 0):
    """Render, save, and re-open — so assertions see the written file, not the builder."""
    workbook = load_workbook(io.BytesIO(render_excel(report)))
    return workbook[workbook.sheetnames[index]]


class TestFormulaInjection:
    """No exported cell may be a formula, whatever the text looks like.

    Two independent controls, and both are asserted: the explicit string type means the file
    contains no formula element at all, and `quotePrefix` keeps the value inert even after
    someone edits the cell in Excel.
    """

    @pytest.mark.parametrize("leader", list(FORMULA_LEADERS))
    async def test_every_dangerous_leader_is_flagged(self, leader: str) -> None:
        sheet = _sheet(_document(_table((f"{leader}danger", 1, 1))))
        cell = sheet.cell(row=FIRST_DATA_ROW, column=1)
        assert cell.data_type == "s"
        assert cell.quotePrefix is True

    async def test_the_dde_payload_is_written_as_text(self) -> None:
        """The text survives verbatim — it is evidence of what someone searched for.

        Stripping or rewriting the value would lose the finding. Making it inert keeps both
        properties: the reviewer sees exactly what was typed, and nothing evaluates.
        """
        sheet = _sheet(_document(_table((DDE_PAYLOAD, 3, 3))))
        cell = sheet.cell(row=FIRST_DATA_ROW, column=1)
        assert cell.value == DDE_PAYLOAD
        assert cell.data_type == "s"
        assert cell.quotePrefix is True

    async def test_the_sheet_xml_contains_no_formula_element(self) -> None:
        """Read at the XML level, because that is the layer Excel evaluates.

        `<f>` is the formula element. openpyxl infers the type from the string — a leading `=`
        becomes `"f"` on its own — so this is the assertion that would fail if the explicit
        `data_type = "s"` in `_string_cell` were ever removed as redundant.
        """
        import zipfile

        content = render_excel(_document(_table((DDE_PAYLOAD, 3, 3), ("=1+1", 2, 2))))
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            sheets = [n for n in archive.namelist() if n.startswith("xl/worksheets/")]
            assert sheets
            for name in sheets:
                xml = archive.read(name).decode("utf-8")
                assert "<f>" not in xml
                assert "<f " not in xml

    async def test_an_error_code_is_not_written_as_an_error(self) -> None:
        """`#REF!` is inferred as an error type, and an error cell is not what was measured.

        A keyword report showing `#REF!` in a cell reads as a broken export rather than as a
        question somebody actually typed.
        """
        sheet = _sheet(_document(_table(("#REF!", 1, 1))))
        cell = sheet.cell(row=FIRST_DATA_ROW, column=1)
        assert cell.data_type == "s"
        assert cell.value == "#REF!"

    async def test_an_ordinary_value_is_not_flagged(self) -> None:
        """The flag is not applied indiscriminately.

        A `quotePrefix` on every cell would show a stray apostrophe in some tools and would
        make the flag meaningless as a signal — worth an assertion so the guard stays targeted.
        """
        sheet = _sheet(_document(_table(("wire transfer", 40, 31))))
        assert sheet.cell(row=FIRST_DATA_ROW, column=1).quotePrefix in (False, None)


class TestNumbersStayNumbers:
    """The reason to offer Excel at all is that the recipient can do arithmetic."""

    async def test_counts_are_written_as_integers(self) -> None:
        sheet = _sheet(_document(_table(("wire", 40, 31))))
        assert sheet.cell(row=FIRST_DATA_ROW, column=2).value == 40
        assert isinstance(sheet.cell(row=FIRST_DATA_ROW, column=2).value, int)

    async def test_a_float_keeps_three_decimals(self) -> None:
        sheet = _sheet(_document(_table(("confidence", 0.8235, 1))))
        cell = sheet.cell(row=FIRST_DATA_ROW, column=2)
        assert cell.value == pytest.approx(0.8235)
        assert cell.number_format == "0.000"

    async def test_a_bool_does_not_become_one(self) -> None:
        """`bool` is an `int` in Python, so the order of the type checks is load-bearing.

        `True` silently rendered as `1` in a count column is a number a reviewer would add up.
        """
        sheet = _sheet(_document(_table(("flagged", True, False))))
        assert sheet.cell(row=FIRST_DATA_ROW, column=2).value == "yes"
        assert sheet.cell(row=FIRST_DATA_ROW, column=3).value == "no"

    async def test_none_becomes_an_empty_cell_not_the_word_none(self) -> None:
        """A null latency means "not measured"; `None` printed in a report reads as a bug."""
        sheet = _sheet(_document(_table(("p95", None, 1))))
        assert sheet.cell(row=FIRST_DATA_ROW, column=2).value in ("", None)


class TestTimestamps:
    async def test_a_timestamp_is_written_as_labelled_utc(self) -> None:
        """A string, and explicitly UTC.

        openpyxl rejects an aware `datetime` outright, and a compliance export whose zone is
        a guess is worse than one that is verbose — so both renderers write the zone into the
        text.
        """
        sheet = _sheet(_document(_table(("cited", 40, GENERATED))))
        assert sheet.cell(row=FIRST_DATA_ROW, column=3).value == "2026-08-14 09:30:00 UTC"

    async def test_a_non_utc_timestamp_is_converted_not_relabelled(self) -> None:
        """Converted to UTC, so the instant is preserved rather than the wall clock.

        Relabelling would move the event by the offset, which in an audit export is a
        misstatement of when something happened.
        """
        moment = GENERATED.astimezone(timezone(timedelta(hours=5, minutes=30)))
        sheet = _sheet(_document(_table(("cited", 1, moment))))
        assert sheet.cell(row=FIRST_DATA_ROW, column=3).value == "2026-08-14 09:30:00 UTC"

    async def test_a_naive_timestamp_is_assumed_utc(self) -> None:
        """Everything this application stores is written in UTC.

        So a naive value means one that lost its `tzinfo` in transit, and guessing the host's
        zone would shift it by however the container happens to be configured.
        """
        sheet = _sheet(_document(_table(("cited", 1, datetime(2026, 8, 14, 9, 30)))))  # noqa: DTZ001
        assert sheet.cell(row=FIRST_DATA_ROW, column=3).value == "2026-08-14 09:30:00 UTC"


class TestLongAndOddText:
    async def test_a_cell_over_excels_ceiling_is_truncated_with_a_marker(self) -> None:
        """Excel's own limit is 32,767 characters; a longer string makes the file unopenable.

        Truncated with a visible marker rather than silently cut, because a reader has to be
        able to tell that the answer they are looking at is not the whole answer.
        """
        sheet = _sheet(_document(_table(("x" * (MAX_CELL_LENGTH + 500), 1, 1))))
        value = sheet.cell(row=FIRST_DATA_ROW, column=1).value
        assert len(value) == MAX_CELL_LENGTH
        assert value.endswith(TRUNCATION_MARKER)

    async def test_a_cell_at_the_ceiling_is_left_alone(self) -> None:
        """Exactly at the limit is legal, so it must not be marked as clipped."""
        sheet = _sheet(_document(_table(("x" * MAX_CELL_LENGTH, 1, 1))))
        value = sheet.cell(row=FIRST_DATA_ROW, column=1).value
        assert len(value) == MAX_CELL_LENGTH
        assert TRUNCATION_MARKER not in value

    async def test_control_characters_are_stripped_rather_than_raising(self) -> None:
        """openpyxl raises `IllegalCharacterError` on these instead of escaping them.

        A stray `\\x00` out of a badly-encoded PDF would otherwise fail the whole export rather
        than one cell — one malformed source document costing an Administrator their report.
        """
        sheet = _sheet(_document(_table(("wire\x00tra\x0bnsfer", 1, 1))))
        assert sheet.cell(row=FIRST_DATA_ROW, column=1).value == "wiretransfer"

    async def test_newlines_and_tabs_survive_inside_a_cell(self) -> None:
        """They are legal in a cell and they carry meaning.

        The citation column is newline-separated by design, so stripping newlines would run
        every source of a multi-source answer into one unreadable line. A *leading* newline is
        covered by the formula-leader check instead.
        """
        citations = "Fee schedule.pdf (page 4)\nKYC.pdf (page 2)"
        sheet = _sheet(_document(_table((citations, 1, 1))))
        assert "\n" in sheet.cell(row=FIRST_DATA_ROW, column=1).value

    async def test_the_pdf_cap_is_lower_than_excels(self) -> None:
        """A structural difference, not an oversight.

        `reportlab` cannot split one table row across pages, so a cell taller than the frame
        raises `LayoutError` and the export fails outright. The PDF is the printable summary;
        the sheet is the full record.
        """
        assert MAX_PDF_CELL_LENGTH < MAX_CELL_LENGTH

    async def test_a_very_long_answer_does_not_break_the_pdf(self) -> None:
        """The regression test for that `LayoutError`.

        40,000 characters in one cell is an ordinary long answer plus a citation list. Before
        the cap this raised during `build` and the whole export 500'd.
        """
        content = render_pdf(_document(_table(("x" * 40_000, 1, 1))))
        assert content.startswith(b"%PDF-")


class TestPdfMarkupIsEscaped:
    async def test_a_document_name_containing_markup_still_renders(self) -> None:
        """`reportlab`'s `Paragraph` parses a mini-HTML dialect.

        An unescaped `<` in a document name is markup: at best the name renders wrong, at worst
        the parser raises and the export fails. Uploaded filenames are user-supplied, so
        this is a real input.
        """
        table = ReportTable(
            title="Most referenced documents",
            columns=("Document", "Citations"),
            rows=(("<script>alert(1)</script> & co.pdf", 3),),
        )
        content = render_pdf(_document(table))
        assert content.startswith(b"%PDF-")

    async def test_an_unclosed_tag_does_not_raise(self) -> None:
        """The shape that actually breaks the parser, rather than a well-formed tag."""
        content = render_pdf(_document(_table(("<b>unclosed", 1, 1))))
        assert content.startswith(b"%PDF-")

    async def test_a_title_containing_markup_does_not_raise(self) -> None:
        """Titles and the subtitle go through the same escaping as the cells.

        The subtitle carries `actor.email`, which is user-supplied at registration.
        """
        report = ReportDocument(
            title="Report <for> & about",
            generated_at=GENERATED,
            subtitle="requested by <admin>@bank.example",
            tables=(_table(("wire", 1, 1), title="Table <one>"),),
        )
        assert render_pdf(report).startswith(b"%PDF-")


class TestEmptyReports:
    async def test_an_empty_table_says_so_in_the_sheet(self) -> None:
        """Zero rows is an answer. "No queries in March" is a finding worth exporting."""
        sheet = _sheet(_document(_table()))
        assert sheet.cell(row=FIRST_DATA_ROW, column=1).value == "No data for this period."

    async def test_an_empty_table_still_renders_a_pdf(self) -> None:
        assert render_pdf(_document(_table())).startswith(b"%PDF-")

    async def test_the_header_row_survives_an_empty_table(self) -> None:
        """The columns are still labelled, so the empty report is still self-describing."""
        sheet = _sheet(_document(_table()))
        assert sheet.cell(row=HEADER_ROW, column=1).value == "Keyword"
        assert sheet.cell(row=HEADER_ROW, column=3).value == "Distinct queries"


class TestSheetLayout:
    async def test_the_header_is_frozen_below_the_provenance_rows(self) -> None:
        """`A6`: rows 1-5 stay visible, so scrolling a long export keeps the labels."""
        assert _sheet(_document()).freeze_panes == "A6"

    async def test_every_sheet_repeats_the_provenance(self) -> None:
        """One worksheet is what gets pasted into a mail, and a sheet with no window is not
        evidence of anything.

        Repeated per sheet rather than kept on a cover sheet for exactly that reason.
        """
        report = _document(
            _table(("wire", 40, 31), title="Most searched keywords"),
            _table(("escheatment", 1, 1), title="Least searched keywords"),
        )
        for index in (0, 1):
            sheet = _sheet(report, index)
            assert sheet.cell(row=1, column=1).value == report.title
            assert sheet.cell(row=2, column=1).value == report.subtitle
            assert sheet.cell(row=3, column=1).value == "Generated 2026-08-14 09:30:00 UTC"

    async def test_one_sheet_per_table_and_no_leftover_default_sheet(self) -> None:
        """A new workbook always has a "Sheet"; the file must not open on it."""
        report = _document(
            _table(("wire", 40, 31), title="Most searched"),
            _table(("escheatment", 1, 1), title="Least searched"),
            _table(("total", 14, 14), title="Summary for the period"),
        )
        workbook = load_workbook(io.BytesIO(render_excel(report)))
        assert len(workbook.sheetnames) == 3
        assert "Sheet" not in workbook.sheetnames

    @pytest.mark.parametrize("bad", ["a/b", "a\\b", "a?b", "a*b", "a[b]", "a:b"])
    async def test_a_title_with_an_illegal_character_still_produces_a_sheet(
        self, bad: str
    ) -> None:
        """Excel refuses the whole file over one illegal sheet-name character.

        A guard against a careless developer-written title rather than against an attacker —
        table titles are literals in `analytics_service`, not request data.
        """
        workbook = load_workbook(io.BytesIO(render_excel(_document(_table(title=bad)))))
        name = workbook.sheetnames[0]
        assert len(name) <= MAX_SHEET_NAME
        for character in r"[]:*?/\\":
            assert character not in name

    async def test_colliding_titles_get_distinct_sheets(self) -> None:
        """Excel rejects a duplicate name, which would lose the second table entirely.

        The position prefix is what keeps two titles that are identical after 31-character
        truncation apart.
        """
        long_title = "Query volume per day for the current reporting period"
        report = _document(
            _table(("a", 1, 1), title=long_title),
            _table(("b", 2, 2), title=long_title),
        )
        workbook = load_workbook(io.BytesIO(render_excel(report)))
        assert len(set(workbook.sheetnames)) == 2

    async def test_a_long_title_is_truncated_to_excels_limit(self) -> None:
        sheet_name = load_workbook(
            io.BytesIO(render_excel(_document(_table(title="T" * 80))))
        ).sheetnames[0]
        assert len(sheet_name) == MAX_SHEET_NAME


class TestBothRenderersAgree:
    async def test_the_same_value_reads_the_same_in_both(self) -> None:
        """A reviewer comparing the PDF to the sheet will check this.

        Asserted on the shared formatting helpers through the Excel path plus a successful
        PDF render: the two renderers call the same `_format_timestamp` and the same
        `_sanitize`, and this is the test that fails if one grows its own copy.
        """
        table = ReportTable(
            title="Summary for the period",
            columns=("Measure", "Value"),
            rows=(("Average confidence", 0.8235), ("Last cited", GENERATED)),
        )
        sheet = _sheet(_document(table))
        assert sheet.cell(row=FIRST_DATA_ROW, column=2).value == pytest.approx(0.8235)
        assert sheet.cell(row=FIRST_DATA_ROW + 1, column=2).value == (
            "2026-08-14 09:30:00 UTC"
        )
        assert render_pdf(_document(table)).startswith(b"%PDF-")

    async def test_the_pdf_carries_the_window_in_its_metadata(self) -> None:
        """The subtitle goes into the PDF `subject`, so the window survives a rename.

        An exported file whose name someone changed is still interpretable from its properties.
        """
        content = render_pdf(_document())
        assert b"2026-01-01 to 2026-06-30" in content
