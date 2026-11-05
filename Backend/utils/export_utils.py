"""backend/utils/export_utils.py

Role: Renders a `ReportDocument` to PDF or Excel bytes. Server-side, synchronous, and pure —
no database, no ORM entity, no request object. The caller decides *what* a report contains;
this module only decides how it is written down safely.

Why the rendering is server-side
--------------------------------
`WORKBREAKDOWN.md` puts the export here rather than in the browser, and the reason is
authorization. A client-side exporter can only write down what was already sent to the client,
so any field the UI masks is either absent from the export or the masking was never real. Doing
it here means the *service* decides which columns exist, under the same RBAC check as the read.

Formula injection
-----------------
A cell beginning `=`, `+`, `-`, `@`, a tab or a carriage return is a formula to Excel, and
`=cmd|'/c calc'!A1` in a spreadsheet a compliance officer opens is code execution on their
machine. Every one of those values reaches this module from user input — questions are typed by
users, document names come from uploaded files. Two independent controls, in `_string_cell`:

  * the value is written with an **explicit string type**, so the sheet never contains a
    formula element at all; and
  * `quotePrefix` is set on the dangerous shapes, which is exactly what Excel records when a
    user types `'=x` — the text displays unchanged and stays inert if the cell is edited.

The residual risk is a third tool re-exporting the sheet to CSV, which drops the flag and
carries only the text. That is out of this process's reach; it is written down rather than
papered over. What is *in* reach — the file DocQuery hands out — cannot evaluate.

PDF markup injection
--------------------
`reportlab`'s `Paragraph` parses a mini-HTML dialect, so an unescaped `<` in a document name is
markup: at best the name renders wrong, at worst the parser raises and the whole export 500s.
`_pdf_text` escapes `&`, `<` and `>` before any text becomes a `Paragraph`.

Masking
-------
This module never unmasks anything and never reaches for a value it was not handed. If a report
must exclude account numbers, the columns are simply not in the `ReportTable` — there is no
"full detail" path here to switch on.
"""

from __future__ import annotations

import html
import io
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from openpyxl import Workbook
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE, Cell
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

#: Excel's own per-cell ceiling. A longer string makes the file unopenable, so it is truncated
#: with a marker rather than silently cut — a reader must be able to tell the text was clipped.
MAX_CELL_LENGTH: Final = 32_767
TRUNCATION_MARKER: Final = " […truncated]"

#: The PDF cap, and it is far lower than Excel's for a structural reason: `reportlab` cannot
#: split one table row across pages, so a cell taller than the frame raises `LayoutError` and
#: the whole export fails. An answer or a question is easily long enough to do that. So the PDF
#: is the printable summary and the Excel sheet is the full record — a difference worth knowing
#: about, which is why the clipped cells carry a visible marker rather than ending mid-word.
MAX_PDF_CELL_LENGTH: Final = 700

#: Leading characters a spreadsheet treats as the start of a formula. Tab and CR are here
#: because some importers strip leading whitespace before deciding, so `\t=cmd` becomes `=cmd`.
FORMULA_LEADERS: Final = ("=", "+", "-", "@", "\t", "\r", "\n")

#: How a timestamp is written in both formats. Excel cells carry no timezone — openpyxl rejects
#: an aware `datetime` outright — and a compliance export whose zone is a guess is worse than
#: one that is verbose. So timestamps become explicit UTC strings in both renderers, and both
#: renderers therefore agree, which a reviewer comparing the PDF to the sheet will check.
TIMESTAMP_FORMAT: Final = "%Y-%m-%d %H:%M:%S UTC"

#: Sheet names are limited to 31 characters and may not contain these. Excel refuses the file
#: otherwise, and a table title is written by a developer, not a user — this is a guard against
#: a careless title, not against an attacker.
_INVALID_SHEET_CHARS: Final = str.maketrans(dict.fromkeys(r"[]:*?/\\", "-"))
MAX_SHEET_NAME: Final = 31

_HEADER_FILL: Final = PatternFill("solid", fgColor="1F3864")
_HEADER_FONT: Final = Font(bold=True, color="FFFFFF")


@dataclass(frozen=True, slots=True)
class ReportTable:
    """One table: a title, its column labels, and its rows.

    `rows` holds plain Python values — `str`, `int`, `float`, `datetime`, `None`. Deliberately
    not ORM entities: a renderer that could reach `.user` off a row could pull a column the
    service chose not to disclose, and the whole point of assembling the report in the service
    is that the disclosure decision happens once, where the RBAC check is.
    """

    title: str
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class ReportDocument:
    """A whole export: heading, provenance line, and one or more tables."""

    title: str
    generated_at: datetime
    #: Who asked and over what window. Rendered under the title, and it is what makes an
    #: exported file self-describing months later — a table of counts with no window is
    #: unusable evidence.
    subtitle: str
    tables: tuple[ReportTable, ...]


# -- Excel ------------------------------------------------------------------
def render_excel(report: ReportDocument) -> bytes:
    """The report as a `.xlsx` byte string, one worksheet per table.

    Returned as bytes rather than written to a path: `architecture.md` forbids writing
    documents to local disk, and the same reasoning covers a generated export — a file on the
    API host is a copy of customer data outside S3's access model, living as long as the
    container does.
    """
    workbook = Workbook()
    # A new Workbook always has one sheet; the first table takes it over so the file never
    # opens on an empty "Sheet".
    workbook.remove(workbook.active)

    for index, table in enumerate(report.tables, start=1):
        sheet = workbook.create_sheet(_sheet_name(table.title, index))
        _write_excel_sheet(sheet, report, table)

    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _write_excel_sheet(sheet: Worksheet, report: ReportDocument, table: ReportTable) -> None:
    """Heading rows, then the header, then the data — with the header frozen.

    The provenance lines are repeated on every sheet rather than kept on a cover sheet. One
    worksheet is what usually gets pasted into a mail or a review pack, and a sheet of counts
    that has lost its date window is not evidence of anything.
    """
    _string_cell(sheet.cell(row=1, column=1), report.title).font = Font(bold=True, size=14)
    _string_cell(sheet.cell(row=2, column=1), report.subtitle)
    _string_cell(
        sheet.cell(row=3, column=1),
        f"Generated {_format_timestamp(report.generated_at)}",
    )
    _string_cell(sheet.cell(row=4, column=1), table.title).font = Font(bold=True)

    header_row = 5
    for column, label in enumerate(table.columns, start=1):
        cell = _string_cell(sheet.cell(row=header_row, column=column), label)
        cell.font = _HEADER_FONT
        cell.fill = _HEADER_FILL
        cell.alignment = Alignment(horizontal="center")

    for offset, row in enumerate(table.rows, start=1):
        for column, value in enumerate(row, start=1):
            _write_value(sheet.cell(row=header_row + offset, column=column), value)

    if not table.rows:
        _string_cell(sheet.cell(row=header_row + 1, column=1), "No data for this period.")

    # Below the header, so scrolling a long report keeps the column labels visible.
    sheet.freeze_panes = sheet.cell(row=header_row + 1, column=1)
    _fit_columns(sheet, table)


def _write_value(cell: Cell, value: object) -> Cell:
    """Write one value with a type Excel can rely on.

    Numbers stay numbers — a count written as text cannot be summed or charted, and the point
    of the Excel format over the PDF is that the recipient can do arithmetic on it. `bool` is
    checked before `int` because `bool` *is* an `int` in Python and `True` in a count column
    would otherwise silently become `1`.
    """
    if value is None:
        return _string_cell(cell, "")
    if isinstance(value, datetime):
        return _string_cell(cell, _format_timestamp(value))
    if isinstance(value, bool):
        return _string_cell(cell, "yes" if value else "no")
    if isinstance(value, int):
        cell.value = value
        return cell
    if isinstance(value, float):
        cell.value = value
        cell.number_format = "0.000"
        return cell
    return _string_cell(cell, str(value))


def _string_cell(cell: Cell, value: str) -> Cell:
    """Write text as text — never as a formula, never with a character Excel rejects.

    Forcing `data_type` after the assignment is the load-bearing part. openpyxl *infers* the
    type from the string: a leading `=` becomes `"f"`, a formula, and `#REF!` becomes `"e"`, an
    error code. Overwriting it with `"s"` means the written sheet holds a string element in
    every case, so there is no formula for Excel to evaluate whatever the text was.
    """
    text = _sanitize(value)
    cell.value = text
    cell.data_type = "s"
    if text.startswith(FORMULA_LEADERS):
        # What Excel itself records for a user-typed `'=x`: displayed verbatim, never
        # evaluated, and still inert after someone edits the cell.
        cell.quotePrefix = True
    return cell


def _sanitize(value: str, *, limit: int = MAX_CELL_LENGTH) -> str:
    """Strip the control characters openpyxl refuses, and bound the length.

    openpyxl raises `IllegalCharacterError` on these rather than escaping them, so an answer
    containing a stray `\\x00` — perfectly possible out of a badly-encoded PDF — would fail
    the whole export instead of one cell. Newlines and tabs survive: they are legal in a cell,
    they carry meaning in an answer, and the formula-leader check above already covers the
    case where one leads.

    `limit` differs by format — see `MAX_PDF_CELL_LENGTH`. Both paths truncate through here so
    the marker and the control-character filter cannot diverge between the two renderers.
    """
    cleaned = ILLEGAL_CHARACTERS_RE.sub("", value)
    if len(cleaned) > limit:
        keep = max(0, limit - len(TRUNCATION_MARKER))
        cleaned = cleaned[:keep] + TRUNCATION_MARKER
    return cleaned


def _fit_columns(sheet: Worksheet, table: ReportTable) -> None:
    """Approximate autofit from the header and the first rows.

    Sampled rather than exhaustive — width is cosmetic, and measuring every cell of a
    thousand-row export to choose a column width is work nobody asked for.
    """
    for index, label in enumerate(table.columns, start=1):
        widest = len(label)
        for row in table.rows[:200]:
            if index <= len(row):
                widest = max(widest, len(str(row[index - 1] or "")))
        sheet.column_dimensions[get_column_letter(index)].width = min(60, widest + 4)


def _sheet_name(title: str, index: int) -> str:
    """A legal, unique worksheet name.

    Prefixed with its position so two tables whose titles collide after truncation still get
    distinct sheets — Excel rejects a duplicate name and would lose the second table entirely.
    """
    prefix = f"{index}. "
    cleaned = title.translate(_INVALID_SHEET_CHARS).strip() or "Report"
    return (prefix + cleaned)[:MAX_SHEET_NAME]


# -- PDF --------------------------------------------------------------------
def render_pdf(report: ReportDocument) -> bytes:
    """The report as PDF bytes: heading, provenance, then each table on its own page.

    Landscape A4 because these tables are wide — a portrait page puts a document name and four
    counts into a column two centimetres across.
    """
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        # PDF metadata is part of the file a reviewer receives; `subject` carries the window.
        title=_pdf_text(report.title),
        author="DocQuery",
        subject=_pdf_text(report.subtitle),
    )

    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle(
        "cell", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=0
    )
    header_style = ParagraphStyle(
        "cellHeader", parent=cell_style, textColor=colors.white, fontName="Helvetica-Bold"
    )

    story: list[object] = [
        Paragraph(_pdf_text(report.title), styles["Title"]),
        Paragraph(_pdf_text(report.subtitle), styles["Normal"]),
        Paragraph(
            f"Generated {_pdf_text(_format_timestamp(report.generated_at))}",
            styles["Normal"],
        ),
        Spacer(1, 8 * mm),
    ]
    for position, table in enumerate(report.tables):
        if position:
            story.append(PageBreak())
        story.append(Paragraph(_pdf_text(table.title), styles["Heading2"]))
        story.append(Spacer(1, 3 * mm))
        story.append(_pdf_table(table, cell_style, header_style))

    document.build(story)
    return buffer.getvalue()


def _pdf_table(
    table: ReportTable, cell_style: ParagraphStyle, header_style: ParagraphStyle
) -> Table:
    """One flowable table, every cell a `Paragraph` so long text wraps instead of overflowing.

    `repeatRows=1` reprints the header on each page of a table that spans several — without it,
    page three of a keyword report is a wall of unlabelled numbers.
    """
    header = [Paragraph(_pdf_text(label), header_style) for label in table.columns]
    body = [
        [Paragraph(_pdf_text(_format_cell(value)), cell_style) for value in row]
        for row in table.rows
    ]
    if not body:
        body = [
            [Paragraph("No data for this period.", cell_style)]
            + [Paragraph("", cell_style)] * (len(table.columns) - 1)
        ]

    available = landscape(A4)[0] - 30 * mm
    width = available / max(1, len(table.columns))
    flowable = Table([header, *body], colWidths=[width] * len(table.columns), repeatRows=1)
    flowable.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F3864")),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9AA5B1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F2F4F7")],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return flowable


def _pdf_text(value: str) -> str:
    """Escape the three characters `reportlab`'s paragraph parser treats as markup.

    `quote=False`: `"` and `'` are legal in the mini-HTML dialect and appear constantly in
    document names and questions. Escaping them would put `&quot;` in the rendered PDF, which
    is a display bug rather than a safety gain.

    Truncation happens *before* escaping, so an entity can never be cut in half — `&am` in the
    output would be a literal, and the length cap is about page geometry, not bytes.
    """
    return html.escape(_sanitize(value, limit=MAX_PDF_CELL_LENGTH), quote=False)


def _format_cell(value: object) -> str:
    """Everything becomes text in a PDF — but the *same* text the Excel sheet shows."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return _format_timestamp(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _format_timestamp(value: datetime) -> str:
    """Render in UTC, labelled.

    A naive value is *assumed* UTC rather than localized to the server: everything this
    application stores is `timezone=True` and written in UTC, so a naive datetime here means a
    value that lost its tzinfo in transit, and guessing the host's zone would silently shift a
    timestamp in an audit export.
    """
    moment = value if value.tzinfo else value.replace(tzinfo=UTC)
    return moment.astimezone(UTC).strftime(TIMESTAMP_FORMAT)
