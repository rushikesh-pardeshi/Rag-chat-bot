"""backend/tests/document_fixtures.py

Role: Builders for real PDF, DOCX and hostile-file bytes used by the ingestion tests.

**Generated, not committed as binaries.** A checked-in `sample.pdf` is a blob nobody can
review: a reader cannot tell what it is supposed to contain, and a reviewer cannot tell whether
it is a fixture or an attack payload someone pasted in. These builders make the content of
every fixture readable in the test that uses it, and they let a test ask for exactly the shape
it needs — three pages, no text at all, one 12,000-character paragraph.

`make_pdf` writes the PDF structure by hand rather than through a rendering library, because
the project has no PDF *writer* dependency and adding one to run tests would need approval
(`.claude/rules/workflow.md`). The output is a genuine PDF with a correct cross-reference
table, so `pypdf` reads it through the same path a real upload takes — not a stub that only
looks like one.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Sequence

#: Where the first text line sits on the page, in PDF user space (origin bottom-left).
_FIRST_LINE_Y = 720
_LINE_HEIGHT = 16


def _escape_pdf_text(value: str) -> str:
    """Escape the three characters that are syntax inside a PDF string literal."""
    return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: Sequence[str]) -> bytes:
    """A page's drawing instructions: one `Tj` per line, top to bottom."""
    parts = []
    for offset, line in enumerate(lines):
        y = _FIRST_LINE_Y - offset * _LINE_HEIGHT
        parts.append(
            f"BT /F1 12 Tf 72 {y} Td ({_escape_pdf_text(line)}) Tj ET\n"
        )
    return "".join(parts).encode("latin-1", errors="replace")


def make_pdf(pages: Sequence[str]) -> bytes:
    """A valid single-font PDF, one page per entry in `pages`.

    Pass an empty string for a page with no text layer — that is what a scanned page looks
    like to an extractor, and it is how the `NEEDS_OCR` path is exercised. Newlines within a
    page become separate text lines.
    """
    # Object numbering is fixed so the references below can be written literally:
    # 1 catalog, 2 page tree, 3 font, then (content, page) pairs from 4 upward.
    page_object_numbers = [5 + 2 * index for index in range(len(pages))]
    kids = " ".join(f"{number} 0 R" for number in page_object_numbers)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for index, page_text in enumerate(pages):
        lines = [line for line in page_text.split("\n") if line]
        stream = _content_stream(lines)
        objects.append(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                "/Resources << /Font << /F1 3 0 R >> >> "
                f"/Contents {4 + 2 * index} 0 R >>"
            ).encode()
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    # The cross-reference table has to record the byte offset of every object, which is why
    # it is built after the body rather than assembled inline.
    xref_offset = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    out += f"startxref\n{xref_offset}\n%%EOF\n".encode()
    return bytes(out)


def make_docx(
    sections: Sequence[tuple[str, str]],
    *,
    title: str | None = None,
    author: str | None = None,
) -> bytes:
    """A real DOCX built with `python-docx`, from `(heading, body)` pairs.

    Uses the library the parser itself uses, so the styles the heading heuristic reads
    (`Heading 1`) are the ones Word would actually write. Hand-rolling the XML would test our
    own XML, not `parse_docx`.
    """
    from docx import Document as DocxDocument

    document = DocxDocument()
    if title is not None:
        document.core_properties.title = title
    if author is not None:
        document.core_properties.author = author
    for heading, body in sections:
        if heading:
            document.add_heading(heading, level=1)
        for paragraph in body.split("\n\n"):
            if paragraph.strip():
                document.add_paragraph(paragraph)

    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def make_zip_that_is_not_docx() -> bytes:
    """A ZIP with no `word/document.xml`.

    Its magic bytes are indistinguishable from a DOCX — `PK\\x03\\x04` is as far as sniffing
    can get — so this is the fixture that proves detection is not the last line of defence.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "not a word document")
    return buffer.getvalue()


def make_docx_with_external_entity(entity_uri: str) -> bytes:
    """A DOCX whose `word/document.xml` declares an external entity and references it.

    Built by repacking a real `python-docx` archive so every *other* part stays valid — the
    package opens, and the only hostile thing in it is the XML. That is the shape of a real
    XXE attempt: a document Word itself would accept.

    `entity_uri` is a `file://` or `http://` URI. If entity resolution were on, its content
    would appear in the extracted text; `_harden_xml` is what stops that.
    """
    base = make_docx([("Cover", "placeholder")])
    hostile = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<!DOCTYPE w:document [ <!ENTITY xxe SYSTEM "{entity_uri}"> ]>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/'
        '2006/main"><w:body><w:p><w:r><w:t>Leaked: &xxe;</w:t></w:r></w:p>'
        "</w:body></w:document>"
    ).encode()

    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(base)) as source,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for item in source.infolist():
            payload = (
                hostile
                if item.filename == "word/document.xml"
                else source.read(item.filename)
            )
            target.writestr(item.filename, payload)
    return out.getvalue()


def make_docx_with_traversal_member(member_name: str) -> bytes:
    """A valid DOCX carrying one extra part whose *name* is a path escaping the archive.

    The Zip Slip fixture. A DOCX is a ZIP, and any code that unpacks one by writing each
    member to `destination / member.filename` will happily write `../../../../etc/passwd` —
    the archive entry name is attacker-controlled and `zipfile` does not sanitize it.

    Every other part is the one `python-docx` wrote, so the package opens and parses normally.
    The hostile member is inert *unless something extracts it*, which is exactly the property
    `test_path_traversal.py` asserts: DocQuery reads `word/document.xml` by name, in memory,
    and never extracts anything to a path.
    """
    base = make_docx([("Cover", "placeholder")])
    out = io.BytesIO()
    with (
        zipfile.ZipFile(io.BytesIO(base)) as source,
        zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as target,
    ):
        for item in source.infolist():
            target.writestr(item.filename, source.read(item.filename))
        target.writestr(member_name, b"pwned by zip slip\n")
    return out.getvalue()


def make_zip_bomb() -> bytes:
    """A DOCX-shaped archive whose declared expansion is wildly out of proportion.

    2 MiB of a single repeated byte deflates to a couple of kilobytes — a compression ratio
    near 1000 against a limit of 200. Kept small on purpose: the point is to trip the *declared
    size* check in `_assert_safe_archive`, which reads the central directory and never
    decompresses, so a fixture that actually allocated hundreds of megabytes would only make
    the test slow.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # Present, so the "is this really a Word document?" check passes and the size checks
        # are the ones under test.
        archive.writestr("word/document.xml", b"0" * (2 * 1024 * 1024))
    return buffer.getvalue()


#: A Windows PE header. The canonical "`.exe` renamed to `.pdf`" upload.
WINDOWS_EXECUTABLE = (
    b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff\x00\x00" + b"\x00" * 64
)

#: Plain text. Neither magic prefix matches, so `detect_content_type` refuses it outright.
PLAIN_TEXT = b"Quarterly revenue was 4.2 million.\n"
