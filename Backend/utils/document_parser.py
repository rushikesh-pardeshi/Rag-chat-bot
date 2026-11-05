"""backend/utils/document_parser.py

Role: Extract citable text and metadata from PDF and DOCX bytes. Pure functions — no
database, no S3, no logging of content.

**Every parser here treats its input as hostile.** A document upload is an attacker-supplied
file handed to a parsing library, which is one of the classic paths to code execution and
resource exhaustion. The defences, and what each one is actually for:

  * **Magic-byte sniffing** (`detect_content_type`). The extension and the `Content-Type`
    header both come from the client. An `.exe` renamed `.pdf` must be refused before any
    parser sees it, so type detection reads the first bytes of the file and nothing else.
  * **XXE off.** A DOCX is a ZIP of XML. `python-docx` parses that XML with lxml, so an
    external-entity reference could read a local file or reach an internal URL. `_harden_xml`
    disables entity resolution and network access globally, at import.
  * **Zip-bomb limits.** A DOCX declaring a 4 GB uncompressed part fits in a few kilobytes on
    the wire. `_assert_safe_archive` checks the declared sizes and the compression ratio
    before anything is decompressed.
  * **Page and text caps.** A PDF with 200,000 pages is a valid PDF. Extraction stops at
    `MAX_PAGES` and `MAX_TEXT_CHARS` so one upload cannot exhaust the worker.

Nothing in this module logs extracted text. Document bodies may contain customer account
numbers and financial data, which `.claude/rules/secure-coding-vapt.md` forbids logging.

A document that parses cleanly but yields no text is **not** an error: it is almost certainly
a scan, and the caller marks it `NEEDS_OCR`. Adding an OCR dependency needs approval
(`.claude/rules/workflow.md`), so this module reports the condition rather than handling it.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any, Final

from backend.core.exceptions import UnsupportedDocumentType, ValidationError

# -- Limits ------------------------------------------------------------------
#: Pages read from one PDF. Beyond this the document is truncated, and the caller is told.
MAX_PAGES: Final = 2_000
#: Total characters kept across all pages. ~5 MB of text, far past any real business document.
MAX_TEXT_CHARS: Final = 5_000_000
#: Largest uncompressed size any single DOCX part may declare.
MAX_ARCHIVE_MEMBER_BYTES: Final = 100 * 1024 * 1024
#: Largest total uncompressed size for the whole archive.
MAX_ARCHIVE_TOTAL_BYTES: Final = 250 * 1024 * 1024
#: Compression ratio above which an archive is treated as a bomb rather than a document.
MAX_COMPRESSION_RATIO: Final = 200
#: Characters of a heading kept as a section label; the column is 512.
MAX_SECTION_LABEL = 512

#: Magic-byte prefixes. A DOCX is a ZIP, so `PK\x03\x04` is as specific as the first bytes
#: get — `parse_docx` confirms the ZIP actually contains `word/document.xml`.
PDF_MAGIC: Final = b"%PDF-"
ZIP_MAGIC: Final = b"PK\x03\x04"

CONTENT_TYPE_PDF: Final = "application/pdf"
CONTENT_TYPE_DOCX: Final = (
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
)

#: Collapses runs of whitespace. A fixed pattern with no nested quantifier — a user-supplied
#: or backtracking-prone regex is forbidden by `secure-coding-vapt.md`.
_WHITESPACE_RUN = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
#: Characters PostgreSQL will not accept in a text column, plus other C0 controls. NUL in
#: particular aborts the insert, and a PDF's text layer can legitimately contain one.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _harden_xml() -> None:
    """Disable XML external entities and network access for lxml, process-wide.

    Called at import so no code path can reach `python-docx` before it has run. This is the
    XXE control from `.claude/rules/secure-coding-vapt.md`: a DOCX is attacker-supplied XML,
    and `resolve_entities=False` is what stops `<!ENTITY x SYSTEM "file:///etc/passwd">`
    from being expanded during parsing.

    `python-docx` builds its own parser, so the defence is applied to lxml's default
    `etree.XMLParser` — which is the object `python-docx` derives from.
    """
    try:
        from lxml import etree
    except ImportError:  # pragma: no cover - lxml ships with python-docx
        return

    etree.set_default_parser(
        etree.XMLParser(
            resolve_entities=False,  # the XXE control
            no_network=True,  # no SSRF via a DTD or schema location
            huge_tree=False,  # bounded depth and node count
            recover=False,  # malformed XML is a rejection, not a repair job
        )
    )


_harden_xml()


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """One citable unit of text, before chunking.

    Carries its own location because that is the only moment it is knowable — page numbers
    come from the PDF reader's iteration order, and heading levels from the DOCX style names.
    Recovering either later would mean re-parsing.
    """

    text: str
    #: 1-based. None for a format with no pages, where `section_label` carries the citation.
    page_number: int | None
    #: Heading or section title. Empty when the format offers none.
    section_label: str = ""

    @property
    def citation(self) -> str:
        if self.section_label and self.page_number is not None:
            return f"{self.section_label} (page {self.page_number})"
        if self.section_label:
            return self.section_label
        return f"Page {self.page_number}"


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Everything extraction produced.

    `has_text` is what the caller keys the `NEEDS_OCR` decision on. It is a property rather
    than a stored flag so it cannot disagree with `sections`.
    """

    sections: tuple[ParsedSection, ...]
    #: Pages for a PDF, top-level sections for a DOCX.
    page_count: int
    #: Non-sensitive descriptive fields only: title, author, producer, section titles.
    #: Never document body text — that is what `sections` is for.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: True when a limit stopped extraction early. Surfaced so a truncated document is not
    #: silently treated as complete.
    truncated: bool = False

    @property
    def has_text(self) -> bool:
        return any(section.text.strip() for section in self.sections)

    @property
    def total_characters(self) -> int:
        return sum(len(section.text) for section in self.sections)


def detect_content_type(data: bytes) -> str:
    """The document's real type, from its leading bytes.

    Neither the filename nor the client's `Content-Type` is consulted. Both are
    caller-controlled, and "the extension said PDF" is precisely the assumption that lets a
    payload reach a parser that was never meant to see it.

    Raises `UnsupportedDocumentType` for anything that is not a PDF or a ZIP container.
    """
    if data.startswith(PDF_MAGIC):
        return CONTENT_TYPE_PDF
    if data.startswith(ZIP_MAGIC):
        # Provisional: every DOCX is a ZIP, but not every ZIP is a DOCX. `parse_docx`
        # confirms by looking for `word/document.xml`.
        return CONTENT_TYPE_DOCX
    raise UnsupportedDocumentType()


def clean_text(value: str) -> str:
    """Normalize extracted text for storage.

    Strips characters PostgreSQL rejects (NUL above all — a PDF text layer can contain one,
    and it aborts the INSERT), collapses whitespace runs, and trims. Newlines survive because
    they carry paragraph structure that a reviewer reading a citation needs.
    """
    without_controls = _CONTROL_CHARS.sub("", value)
    collapsed = _WHITESPACE_RUN.sub(" ", without_controls)
    return _BLANK_LINES.sub("\n\n", collapsed).strip()


def parse_document(data: bytes, *, content_type: str | None = None) -> ParsedDocument:
    """Parse by detected type. The entry point for the ingestion path.

    `content_type` may be passed when the caller has already sniffed it, to avoid inspecting
    the bytes twice. It is *re-derived* rather than trusted if absent — and a caller passing
    a type that contradicts the bytes gets the parser for the type they named, which is why
    only `document_service` (which sniffs first) supplies it.
    """
    detected = content_type or detect_content_type(data)
    if detected == CONTENT_TYPE_PDF:
        return parse_pdf(data)
    if detected == CONTENT_TYPE_DOCX:
        return parse_docx(data)
    raise UnsupportedDocumentType()


def parse_pdf(data: bytes) -> ParsedDocument:
    """Per-page text plus document metadata.

    One `ParsedSection` per page, keeping the 1-based page number BR-03 requires for
    citations. Pages that yield nothing are skipped rather than stored empty — a chunk with
    no text is uncitable, and the CHECK constraint on `document_chunk` would reject it.

    A page whose extraction raises is skipped too, and recorded in `metadata['failed_pages']`.
    One corrupt page in a 300-page annual report should not cost the other 299.
    """
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
    except (PdfReadError, ValueError, OSError) as exc:
        # Message deliberately generic: pypdf's text can quote file structure, and this
        # string reaches an API client.
        raise ValidationError("The PDF could not be read. It may be corrupt.") from exc

    if reader.is_encrypted:
        # Try the empty password, which is what an "owner password only" PDF uses. Anything
        # else needs a credential we do not have and must not guess at.
        try:
            if reader.decrypt("") == 0:
                raise ValidationError(
                    "The PDF is password-protected and cannot be indexed."
                )
        except (NotImplementedError, PdfReadError) as exc:
            raise ValidationError(
                "The PDF uses an unsupported encryption scheme."
            ) from exc

    total_pages = len(reader.pages)
    page_limit = min(total_pages, MAX_PAGES)
    sections: list[ParsedSection] = []
    failed_pages: list[int] = []
    characters = 0
    truncated = total_pages > page_limit

    for index in range(page_limit):
        try:
            raw = reader.pages[index].extract_text() or ""
        except Exception:  # noqa: BLE001 - one bad page must not fail the document
            failed_pages.append(index + 1)
            continue

        text = clean_text(raw)
        if not text:
            continue
        if characters + len(text) > MAX_TEXT_CHARS:
            # Keep the part that fits, then stop. A hard stop mid-document is still more
            # useful than refusing the whole upload.
            text = text[: MAX_TEXT_CHARS - characters]
            truncated = True
        characters += len(text)
        sections.append(ParsedSection(text=text, page_number=index + 1))
        if characters >= MAX_TEXT_CHARS:
            truncated = True
            break

    metadata = _pdf_metadata(reader)
    metadata["page_count"] = total_pages
    metadata["pages_extracted"] = len(sections)
    if failed_pages:
        metadata["failed_pages"] = failed_pages[:50]

    return ParsedDocument(
        sections=tuple(sections),
        page_count=total_pages,
        metadata=metadata,
        truncated=truncated,
    )


def _pdf_metadata(reader: Any) -> dict[str, Any]:
    """Descriptive fields from the PDF's info dictionary.

    Every value is length-capped and control-stripped: these strings are attacker-controlled
    and end up in JSONB and then on a dashboard.
    """
    metadata: dict[str, Any] = {"format": "pdf"}
    try:
        info = reader.metadata or {}
    except Exception:  # noqa: BLE001 - a malformed info dict is not a parse failure
        return metadata

    for key, target in (
        ("/Title", "title"),
        ("/Author", "author"),
        ("/Subject", "subject"),
        ("/Producer", "producer"),
        ("/Creator", "creator"),
    ):
        value = info.get(key)
        if value:
            metadata[target] = clean_text(str(value))[:255]
    return metadata


def parse_docx(data: bytes) -> ParsedDocument:
    """Section-aware text from a Word document.

    DOCX has no pages — pagination is a rendering decision made by Word, not something stored
    in the file. So the citation is the *heading path*: paragraphs accumulate under the most
    recent heading, and that heading becomes `section_label`. This is why
    `DocumentChunk`'s CHECK accepts either a page number or a section label; for this format
    only the latter exists.

    Text before any heading goes to "Introduction" rather than to an empty label, because a
    chunk must be citable and "" is not a citation.
    """
    _assert_safe_archive(data)

    from docx import Document as DocxDocument
    from docx.opc.exceptions import PackageNotFoundError

    try:
        document = DocxDocument(io.BytesIO(data))
    except (PackageNotFoundError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise ValidationError(
            "The Word document could not be read. It may be corrupt."
        ) from exc

    sections: list[ParsedSection] = []
    current_label = "Introduction"
    buffer: list[str] = []
    headings: list[str] = []
    characters = 0
    truncated = False

    def flush() -> None:
        nonlocal buffer, characters, truncated
        if not buffer:
            return
        text = clean_text("\n".join(buffer))
        buffer = []
        if not text:
            return
        if characters + len(text) > MAX_TEXT_CHARS:
            text = text[: MAX_TEXT_CHARS - characters]
            truncated = True
        if text:
            characters += len(text)
            sections.append(
                ParsedSection(text=text, page_number=None, section_label=current_label)
            )

    for paragraph in document.paragraphs:
        if characters >= MAX_TEXT_CHARS:
            truncated = True
            break
        raw = paragraph.text
        if not raw or not raw.strip():
            continue
        if _is_heading(paragraph):
            # A heading closes the previous section and opens the next one.
            flush()
            current_label = clean_text(raw)[:MAX_SECTION_LABEL] or current_label
            headings.append(current_label)
            continue
        buffer.append(raw)

    flush()

    # Tables carry the numeric content of a financial document — a rate card or a fee
    # schedule is often *only* a table. Dropping them would make those documents
    # unanswerable while looking like a successful ingest.
    table_text = _extract_tables(document, remaining=MAX_TEXT_CHARS - characters)
    if table_text:
        sections.append(
            ParsedSection(text=table_text, page_number=None, section_label="Tables")
        )

    metadata: dict[str, Any] = {"format": "docx", "section_count": len(sections)}
    if headings:
        # Capped: the outline is navigational context, not a copy of the document.
        metadata["sections"] = headings[:100]
    metadata.update(_docx_core_properties(document))

    return ParsedDocument(
        sections=tuple(sections),
        # No real page count for this format. The number of sections is what a reviewer can
        # actually navigate by, and it is what `section_or_page` cites.
        page_count=len(sections),
        metadata=metadata,
        truncated=truncated,
    )


def _is_heading(paragraph: Any) -> bool:
    """Whether a paragraph is a heading, by style name.

    Style names are localized in non-English Word installations, so this is a heuristic and
    is treated as one: a missed heading costs a less precise `section_label`, never a lost or
    uncitable chunk, because the previous label still applies.
    """
    try:
        style_name = (paragraph.style.name or "").lower()
    except AttributeError:
        return False
    return style_name.startswith("heading") or style_name in {"title", "subtitle"}


def _extract_tables(document: Any, *, remaining: int) -> str:
    """Flatten tables to pipe-separated rows, within the remaining character budget."""
    if remaining <= 0:
        return ""
    rows: list[str] = []
    used = 0
    for table in document.tables:
        for row in table.rows:
            cells = [clean_text(cell.text) for cell in row.cells]
            line = " | ".join(cell for cell in cells if cell)
            if not line:
                continue
            if used + len(line) > remaining:
                return "\n".join(rows)
            rows.append(line)
            used += len(line)
    return "\n".join(rows)


def _docx_core_properties(document: Any) -> dict[str, Any]:
    """Title, author and revision from the OPC core properties part."""
    metadata: dict[str, Any] = {}
    try:
        core = document.core_properties
    except Exception:  # noqa: BLE001 - the part is optional
        return metadata

    for attribute, target in (
        ("title", "title"),
        ("author", "author"),
        ("subject", "subject"),
        ("last_modified_by", "last_modified_by"),
    ):
        value = getattr(core, attribute, None)
        if value:
            metadata[target] = clean_text(str(value))[:255]
    revision = getattr(core, "revision", None)
    if revision:
        metadata["revision"] = int(revision)
    return metadata


def _assert_safe_archive(data: bytes) -> None:
    """Reject zip bombs before any part is decompressed.

    Checks the *declared* uncompressed sizes in the central directory, which is the only way
    to know the cost before paying it. A `python-docx` call on a bomb would inflate hundreds
    of megabytes inside a worker before anything could intervene.

    Also confirms the archive is actually a Word document: a plain ZIP that reached
    `detect_content_type` looks like a DOCX from its first four bytes.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            if "word/document.xml" not in names:
                raise UnsupportedDocumentType()

            total = 0
            for info in archive.infolist():
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise ValidationError(
                        "The document contains an implausibly large internal part."
                    )
                total += info.file_size
                if total > MAX_ARCHIVE_TOTAL_BYTES:
                    raise ValidationError("The document expands to an unsafe size.")
                if (
                    info.compress_size > 0
                    and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO
                ):
                    raise ValidationError(
                        "The document's compression ratio is implausible."
                    )
    except zipfile.BadZipFile as exc:
        raise ValidationError("The document is not a readable Word file.") from exc
