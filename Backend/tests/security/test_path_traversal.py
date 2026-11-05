"""backend/tests/security/test_path_traversal.py

Role: Prove that no client-supplied string ever becomes a path — not a filesystem path, not an
S3 key, and not a `Content-Disposition` filename. Closes the **Path Traversal** requirement in
`.claude/rules/secure-coding-vapt.md`: *"Validate and sanitize all file paths for document
uploads/downloads. Use secure file storage APIs with access controls."*

Where a path could come from, in this application
-------------------------------------------------
There are exactly four places a caller's string could be read as a path, and there is one
section below for each:

1. **The upload filename.** `multipart/form-data` carries a filename the client chooses, and
   browsers on Windows really do send `C:\\Users\\x\\report.pdf`. It reaches
   `sanitize_filename`, then `build_key`, then the `document.name` column.
2. **A member name inside an uploaded DOCX.** A DOCX is a ZIP, and a ZIP entry name is
   attacker-controlled — `../../../../etc/passwd` is a legal entry name that `zipfile` will not
   sanitize. Anything that unpacks by writing `destination / entry.filename` is Zip Slip.
3. **A path parameter on a read route.** `GET /documents/{document_id}` would be a file-read
   primitive if `document_id` were a string that reached storage.
4. **The export filename.** It lands in a response header the browser writes to disk under.

What this module asserts, and what it deliberately does not
-----------------------------------------------------------
The strongest claim available here is not "traversal is filtered" but **"there is nothing to
traverse"**: `TestNothingInTheBackendTouchesTheFilesystem` walks all 68 shipped modules and
asserts that not one of them imports `os`, `pathlib`, `tempfile`, `shutil` or `tarfile`, calls
`open`, or extracts an archive member. Document bytes go to S3 and text is parsed in memory, so
a traversal payload has no filesystem to escape into. That is a structural property, and a
structural property is worth a test precisely because the *next* commit is the one that adds a
`tempfile` for "just a quick OCR spike".

Two things are consciously out of scope. Windows reserved device names (`CON`, `PRN`, `NUL`,
`COM1`) survive `sanitize_filename` — they are single path components and therefore not
traversal, and they are inert here because nothing opens a file by name; a client that saves a
download under such a name is a browser-side concern, handled by `safeFilename()` in
`frontend/lib/api-client.ts`. And S3 key *permissions* are not asserted: the bucket policy is
infrastructure, recorded in `docs/RELEASE.md`, not something a pytest can reach.

Positive controls
-----------------
Every assertion in sections 1 and 2 is of the form "the dangerous part is gone", which is also
true of a function that returns a constant. So `TestALegitimateNameSurvivesIntact` runs first:
ordinary filenames must come through unchanged, extension and all, or the sanitizer is simply
destroying input and the rest of this module proves nothing.
"""

from __future__ import annotations

import ast
import pathlib
import re
import uuid
from collections.abc import Callable

import httpx
import pytest

from backend.core.config import get_settings
from backend.core.exceptions import ValidationError
from backend.models.audit_log import AuditAction
from backend.services.storage_service import (
    FALLBACK_FILENAME,
    MAX_FILENAME_LENGTH,
    S3StorageService,
    file_extension,
    sanitize_filename,
)
from backend.tests.document_fixtures import (
    make_docx_with_traversal_member,
    make_pdf,
)
from backend.tests.security.conftest import (
    API,
    ROOT,
    Stack,
    backend_modules,
    csrf_headers,
)

Api = tuple[httpx.AsyncClient, Callable[["object | None"], None]]

#: Traversal payloads. Local to this module rather than in `conftest`, unlike the SQL/XSS/log
#: sets: a filename is the only field in this API that could be read as a path, so there is no
#: second module for these to be reused by.
#:
#: The list is grouped by the *mechanism* each payload attacks, because a sanitizer usually
#: stops one group and not the next:
TRAVERSAL_PAYLOADS: tuple[str, ...] = (
    # Plain relative traversal, both separators.
    "../../../../etc/passwd",
    "..\\..\\..\\..\\windows\\system32\\config\\sam",
    # Doubled dots, which survive a naive `replace("../", "")`.
    "....//....//etc/passwd",
    "..../\\..../\\boot.ini",
    # Percent-encoded — this string arrives *already decoded* by Starlette's multipart parser,
    # so a payload that is still encoded here is what a double-decoding bug looks like.
    "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
    "..%252f..%252fetc%252fpasswd",
    # Absolute paths and UNC, which need no dots at all.
    "/etc/shadow",
    "C:\\Users\\treasury\\Desktop\\statement.pdf",
    "\\\\attacker.invalid\\share\\payload.pdf",
    "file:///etc/passwd",
    # Unicode separators. U+FF0F FULLWIDTH SOLIDUS becomes a real `/` under NFKD, which is why
    # normalization happens *before* the component split; U+2215 DIVISION SLASH does not, and
    # is caught by the allowlist instead. Two different controls, one payload each.
    "..\uff0f..\uff0fetc\uff0fpasswd",
    "..\u2215..\u2215etc\u2215passwd",
    # A NUL truncation attempt: some C-backed APIs stop reading at \x00, so `x.pdf\x00.exe`
    # passes an extension check as `.exe` and is written as `x.pdf`.
    "report.pdf\x00.exe",
    # A newline, which would also forge a log line — see `test_log_injection.py`.
    "report\n../../evil.pdf",
    # Degenerate components: each of these would collapse a key to a directory prefix.
    "..",
    ".",
    "...",
    "",
    "   ",
    "./",
    # Names that are single components but would be hidden files or look like CLI flags.
    ".hidden.pdf",
    "-rf.pdf",
    "--output=/etc/cron.d/x",
    # Over-long, to prove the length cap keeps the extension the allowlist reads.
    "a" * 400 + ".pdf",
)

#: Payloads whose separators really are separators, paired with the *only* component that may
#: contribute to the key. An exact expectation rather than "no `/` survives", because the
#: interesting failure is not a surviving slash — it is a leaf assembled from the wrong part of
#: the path, e.g. `..` collapsing to `.._.._etc_passwd` and taking the attacker's directory
#: names into an object name a compliance reviewer then reads.
#:
#: Payloads *not* in this list are the ones whose separators were never separators: a
#: percent-encoded `%2f` (nothing decodes it — arriving still-encoded is what a double-decode
#: bug looks like) and U+2215 DIVISION SLASH (which NFKD leaves alone). Those legitimately
#: collapse to a single inert filename, and `etc` surviving as a *substring* of a filename is
#: not traversal — see `test_a_payload_whose_separators_are_not_separators_is_inert`.
SEPARATED_PAYLOADS: tuple[tuple[str, str], ...] = (
    ("../../../../etc/passwd", "passwd"),
    ("..\\..\\..\\..\\windows\\system32\\config\\sam", "sam"),
    ("....//....//etc/passwd", "passwd"),
    ("..../\\..../\\boot.ini", "boot.ini"),
    ("/etc/shadow", "shadow"),
    ("C:\\Users\\treasury\\Desktop\\statement.pdf", "statement.pdf"),
    ("\\\\attacker.invalid\\share\\payload.pdf", "payload.pdf"),
    ("file:///etc/passwd", "passwd"),
    ("..\uff0f..\uff0fetc\uff0fpasswd", "passwd"),
    ("report\n../../evil.pdf", "evil.pdf"),
)

#: Ordinary filenames. If any of these comes back altered, the sanitizer is destroying real
#: input and every "the payload is gone" assertion below is met by a function that returns
#: nothing useful.
LEGITIMATE_NAMES: tuple[str, ...] = (
    "fee-schedule.pdf",
    "Q3-2026_treasury_report.docx",
    "AML.Policy.v4.pdf",
    "statement_2026-08-14.pdf",
    "a.pdf",
)

#: Field names that would mean a client is choosing a path. `file` is deliberately absent: it
#: is the multipart part name on the upload route, holds an `UploadFile`, and is not a path.
PATHISH_NAMES: frozenset[str] = frozenset(
    {
        "path",
        "filepath",
        "file_path",
        "filename",
        "file_name",
        "key",
        "s3_key",
        "object_key",
        "dir",
        "directory",
        "folder",
        "template",
        "template_path",
    }
)

#: Modules a shipped backend module has no business importing. Each one is a way to reach the
#: filesystem, and the point of the list is that document bytes live in S3 and text is parsed
#: in memory — see the module docstring.
FILESYSTEM_MODULES: frozenset[str] = frozenset(
    {"os", "pathlib", "tempfile", "shutil", "tarfile", "glob", "fileinput"}
)

#: Calls that write an archive member to a path. `zipfile.ZipFile(...).read(name)` is fine and
#: is what `document_parser` uses; `extract`/`extractall` are the Zip Slip primitives.
EXTRACTION_CALLS: frozenset[str] = frozenset({"extract", "extractall", "unpack_archive"})

#: Written into the hostile DOCX in section 2. Distinctive, so a filesystem check for it cannot
#: match something a different test left behind.
ZIP_SLIP_MARKER = "docquery-zip-slip-marker.txt"


def assert_single_component(name: str) -> None:
    """`name` must be one path component, on POSIX and on Windows, with nothing hidden in it.

    Both `PurePosixPath` and `PureWindowsPath` are asked, because they disagree: `a\\b` is a
    single component to POSIX and two to Windows, and a name that only satisfies one of them is
    a name that is safe on the developer's machine and not on the server's.
    """
    assert name, "sanitize_filename returned an empty component"
    assert "/" not in name, f"{name!r} still contains a POSIX separator"
    assert "\\" not in name, f"{name!r} still contains a Windows separator"
    assert name not in {".", ".."}, f"{name!r} is a directory reference"
    assert not name.startswith("."), f"{name!r} would be a hidden file"
    assert not name.startswith("-"), f"{name!r} would be read as a command-line flag"
    assert pathlib.PurePosixPath(name).name == name, f"{name!r} is not one POSIX component"
    assert pathlib.PureWindowsPath(name).name == name, f"{name!r} is not one Windows component"
    assert len(name) <= MAX_FILENAME_LENGTH, f"{name!r} exceeds the length cap"
    control = [char for char in name if ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F]
    assert not control, f"{name!r} carries control characters {control!r}"


def assert_nothing_was_written(working_directory: pathlib.Path) -> None:
    """No file appeared in the working directory, and none escaped above it.

    Both halves are needed: a Zip Slip member with no `..` lands *inside* the directory, and
    one with `../../` lands outside it, so checking one place misses half the attack. Six
    levels is more than the deepest payload here reaches.

    Sync on purpose. Called from `async def` tests, where ruff's ASYNC240 correctly objects to
    blocking `pathlib` calls — the fix is to keep the blocking work in a plain function rather
    than to silence the rule at the call site.
    """
    left_behind = sorted(path.name for path in working_directory.rglob("*"))
    assert not left_behind, f"files appeared under {working_directory}: {left_behind}"
    escaped = [
        candidate
        for candidate in (
            working_directory.joinpath(*[".."] * depth, ZIP_SLIP_MARKER) for depth in range(6)
        )
        if candidate.exists()
    ]
    assert not escaped, f"a zip member escaped to {escaped}"


def assert_key_inside_prefix(key: str, *, prefix: str = "documents") -> None:
    """The key must be exactly `prefix/{uuid}/v{n}/{name}` with no escape and no empty part.

    An empty part matters as much as a `..` one: `documents//v1/x.pdf` is a different object in
    S3 than the audit row names, and `documents/{id}/v1/` is a prefix rather than an object.
    """
    assert not key.startswith("/"), f"{key!r} is absolute"
    assert "\\" not in key, f"{key!r} contains a Windows separator"
    parts = key.split("/")
    assert parts[0] == prefix, f"{key!r} does not start with the configured prefix"
    assert len(parts) == 4, f"{key!r} has {len(parts)} components, expected 4"
    assert all(parts), f"{key!r} contains an empty component"
    assert ".." not in parts, f"{key!r} contains a parent reference"
    assert "." not in parts, f"{key!r} contains a self reference"
    uuid.UUID(parts[1])  # raises if the id component is not a UUID we generated
    assert re.fullmatch(r"v[1-9]\d*", parts[2]), f"{key!r} has a malformed version component"


class TestALegitimateNameSurvivesIntact:
    """The positive control. Run first, because everything after it is a negative assertion."""

    @pytest.mark.parametrize("name", LEGITIMATE_NAMES)
    def test_an_ordinary_filename_is_unchanged(self, name: str) -> None:
        assert sanitize_filename(name) == name

    def test_a_trailing_space_does_not_corrupt_the_extension(self) -> None:
        """`"report.pdf "` must sanitize to `report.pdf`, not to `report.pdf_`.

        This is the reason `sanitize_filename` strips before substituting. A trailing space is
        invisible in a file picker, and turning it into `_` produces extension `pdf_`, which
        the upload allowlist then refuses — a real document rejected for a character the
        uploader cannot see. A security control that rejects legitimate work gets switched off.
        """
        assert sanitize_filename("report.pdf ") == "report.pdf"
        assert sanitize_filename("  fee-schedule.pdf  ") == "fee-schedule.pdf"

    def test_the_extension_the_allowlist_reads_is_the_real_one(self) -> None:
        assert file_extension("statement.PDF") == "pdf"
        assert file_extension("../../etc/report.docx") == "docx"
        # The traversal payload's real extension, not the one before the NUL.
        assert file_extension("report.pdf\x00.exe") == "exe"
        assert file_extension("noextension") == ""


class TestSanitizeFilenameReducesAnythingToOneComponent:
    """Section 1a — the filename, at the unit level."""

    @pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
    def test_a_traversal_payload_becomes_one_safe_component(self, payload: str) -> None:
        assert_single_component(sanitize_filename(payload))

    @pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
    def test_sanitizing_twice_changes_nothing(self, payload: str) -> None:
        """Idempotence, which is what makes the function safe to apply at more than one layer.

        `document_service.upload` sanitizes and then hands the result to `build_key`, which
        sanitizes again. If the function were not idempotent that second pass would rename the
        object relative to the `document.name` column, and the row would point at a key that
        does not exist.
        """
        once = sanitize_filename(payload)
        assert sanitize_filename(once) == once

    def test_nothing_usable_becomes_the_fallback_not_the_empty_string(self) -> None:
        """An empty component would collapse a key to a directory prefix, so there is a name.

        `documents/{id}/v1/` is a prefix, not an object: it stores nothing, and the audit row
        that names it points at nothing.
        """
        for payload in ("", "   ", "..", ".", "...", "///", "\\\\", "._-"):
            assert sanitize_filename(payload) == FALLBACK_FILENAME

    def test_the_length_cap_preserves_the_extension(self) -> None:
        result = sanitize_filename("a" * 400 + ".pdf")
        assert len(result) <= MAX_FILENAME_LENGTH
        assert result.endswith(".pdf"), "the cap truncated the extension the allowlist reads"

    def test_a_unicode_separator_does_not_survive_normalization(self) -> None:
        """U+FF0F normalizes to `/` under NFKD, so the split must happen after normalizing.

        Asserted on its own because the ordering is invisible in the result: a sanitizer that
        split first and normalized second returns `..／..／etc／passwd` → `.._.._etc_passwd`,
        which passes `assert_single_component` while proving the steps are in the wrong order.
        """
        assert sanitize_filename("..\uff0f..\uff0fetc\uff0fpasswd") == "passwd"


class TestTheStorageKeyCannotLeaveItsPrefix:
    """Section 1b — the key, which is where a filename becomes a location."""

    @pytest.fixture
    def storage(self, stack: Stack) -> S3StorageService:
        return stack.storage

    @pytest.mark.parametrize("payload", TRAVERSAL_PAYLOADS)
    def test_a_traversal_filename_still_produces_a_key_inside_the_prefix(
        self, storage: S3StorageService, payload: str
    ) -> None:
        key = storage.build_key(document_id=uuid.uuid4(), version=1, filename=payload)
        assert_key_inside_prefix(key)

    @pytest.mark.parametrize(("payload", "expected_leaf"), SEPARATED_PAYLOADS)
    def test_only_the_last_component_of_a_path_reaches_the_key(
        self, storage: S3StorageService, payload: str, expected_leaf: str
    ) -> None:
        """Not just "the key is well-formed" — only the *final* component may contribute.

        A key can be structurally valid and still carry the attack: a leaf of
        `.._.._etc_passwd` is inside the prefix but has taken its name from directories the
        client chose, and that string is then what a compliance reviewer reads in a document
        list. The separators are dropped along with everything before the last one.
        """
        key = storage.build_key(document_id=uuid.uuid4(), version=1, filename=payload)
        assert_key_inside_prefix(key)
        assert key.rsplit("/", 1)[-1] == expected_leaf

    @pytest.mark.parametrize(
        "payload",
        [
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "..%252f..%252fetc%252fpasswd",
            "..∕..∕etc∕passwd",
        ],
    )
    def test_a_payload_whose_separators_are_not_separators_is_inert(
        self, storage: S3StorageService, payload: str
    ) -> None:
        """A still-encoded `%2f`, or a look-alike NFKD leaves alone, becomes one filename.

        The result contains `etc` and `passwd` — as a *substring of a filename*, which is not
        traversal and must not be treated as a finding. The claim worth asserting is that the
        string never became a path: one component, one key, four segments. What would make
        these dangerous is a second decode pass downstream, and the reason there isn't one is
        that nothing downstream re-reads the name — the key is built once, here.
        """
        safe = sanitize_filename(payload)
        assert_single_component(safe)
        key = storage.build_key(document_id=uuid.uuid4(), version=1, filename=payload)
        assert_key_inside_prefix(key)
        assert key.rsplit("/", 1)[-1] == safe

    def test_the_version_component_cannot_be_zero_or_negative(
        self, storage: S3StorageService
    ) -> None:
        """`v0` and `v-1` are refused, so the version segment is always one non-empty token.

        `version` is server-computed and no request can set it, so this is defence in depth —
        the kind that costs one `if` and catches an off-by-one in `next_version` before it
        writes an object at a key no audit row can name.
        """
        for version in (0, -1, -999):
            with pytest.raises(ValidationError):
                storage.build_key(
                    document_id=uuid.uuid4(), version=version, filename="report.pdf"
                )

    def test_two_versions_of_the_same_name_are_different_objects(
        self, storage: S3StorageService
    ) -> None:
        """Lineage is app-side, so no version may ever share a key with another."""
        document_id = uuid.uuid4()
        keys = {
            storage.build_key(document_id=document_id, version=n, filename="../../fee.pdf")
            for n in (1, 2, 3)
        }
        assert len(keys) == 3
        for key in keys:
            assert_key_inside_prefix(key)


class TestAnUploadedTraversalNameLandsInsideThePrefix:
    """Section 1c — the same claim, over HTTP, through the real route and service."""

    async def test_the_object_is_written_under_the_documents_prefix(
        self, api: Api, stack: Stack
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        # `Corpus` stages chunk rows and vectors directly and never puts bytes in S3, so the
        # bucket starts empty and "exactly one object" below is a statement about this upload.
        assert not stack.s3.objects, "the fixture staged an S3 object"

        response = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "../../../../etc/passwd.pdf",
                    make_pdf(["The outbound domestic wire transfer fee is 25 USD."]),
                    "application/pdf",
                )
            },
            headers=csrf_headers(),
        )

        assert response.status_code == 201, response.text
        assert len(stack.s3.objects) == 1, "expected exactly one object written"
        key = next(iter(stack.s3.objects))
        assert_key_inside_prefix(key)
        assert key.endswith("/passwd.pdf")

    async def test_the_persisted_name_is_the_sanitized_one(
        self, api: Api, stack: Stack
    ) -> None:
        """The stored `document.name` is what the UI renders and what `next_version` counts."""
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "..\\..\\..\\treasury\\statement.pdf",
                    make_pdf(["The monthly account maintenance fee is 12 USD."]),
                    "application/pdf",
                )
            },
            headers=csrf_headers(),
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["name"] == "statement.pdf"
        assert_single_component(body["name"])

        listed = await client.get(f"{API}/documents")
        assert listed.status_code == 200, listed.text
        names = [item["name"] for item in listed.json()["items"]]
        assert "statement.pdf" in names
        for name in names:
            assert_single_component(name)

    async def test_no_traversal_fragment_reaches_the_audit_trail(
        self, api: Api, stack: Stack
    ) -> None:
        """An audit row is read by a human during a compliance review, and it is also a log.

        The positive control is the *row*, not the name: `DOCUMENT_UPLOAD` records the document
        id in `object_id` and deliberately keeps the filename out of `detail`, because the
        document row is the authority on what a document is called and a second copy in an
        append-only log can only drift from it. So this asserts the row exists, references the
        document that was created, and carries no fragment of the path the uploader tried —
        a reviewer reading `../../etc/passwd.pdf` in an audit detail cannot tell whether it was
        blocked.
        """
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "../../../../etc/passwd.pdf",
                    make_pdf(["Fee schedule."]),
                    "application/pdf",
                )
            },
            headers=csrf_headers(),
        )
        assert response.status_code == 201, response.text
        document_id = response.json()["id"]

        uploads = [
            entry
            for entry in stack.audit_repo.entries
            if entry.action is AuditAction.DOCUMENT_UPLOAD
        ]
        assert uploads, "the upload was not audited at all"
        assert any(entry.object_id == document_id for entry in uploads), (
            "no upload row references the document that was created"
        )

        trail = stack.audit_details() + repr(
            [(entry.object_type, entry.object_id) for entry in stack.audit_repo.entries]
        )
        assert "../" not in trail
        assert "..\\" not in trail
        assert "etc/passwd" not in trail

    async def test_the_presigned_url_is_minted_for_the_server_built_key(
        self, api: Api, stack: Stack
    ) -> None:
        """The download path takes no filename from the caller — only the document's own id."""
        client, as_user = api
        as_user(stack.a.admin)
        uploaded = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "/etc/shadow.pdf",
                    make_pdf(["Fee schedule."]),
                    "application/pdf",
                )
            },
            headers=csrf_headers(),
        )
        assert uploaded.status_code == 201, uploaded.text
        document_id = uploaded.json()["id"]
        written_key = next(iter(stack.s3.objects))

        preview = await client.get(f"{API}/documents/{document_id}/preview")
        assert preview.status_code == 200, preview.text

        assert stack.s3.presigned, "no presigned URL was requested"
        presigned_key, ttl = stack.s3.presigned[-1]
        assert presigned_key == written_key
        assert_key_inside_prefix(presigned_key)
        assert 0 < ttl <= get_settings().S3_PRESIGNED_URL_EXPIRY_SECONDS


class TestNothingInTheBackendTouchesTheFilesystem:
    """Section 3a — the structural claim, and the strongest one available.

    A traversal payload needs a filesystem to escape into. There is none: document bytes go to
    S3 and are parsed from memory. These scans are what keeps that true.
    """

    def test_the_scan_is_reaching_the_source(self) -> None:
        """Non-vacuity. Asserted here as well as in `test_vector_isolation.py`.

        `backend_modules()` returning `[]` would make every assertion in this class pass. That
        has already happened once in this suite (a wrong `parents[]` index), which is why the
        floor is explicit rather than implied.
        """
        modules = backend_modules()
        names = {path.name for path in modules}
        assert len(modules) > 40, f"only {len(modules)} modules found under {ROOT / 'backend'}"
        for required in ("storage_service.py", "document_parser.py", "documents.py"):
            assert required in names, f"{required} is not in the scanned set"

    def test_no_shipped_module_imports_a_filesystem_library(self) -> None:
        offenders: list[str] = []
        for path in backend_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                for name in names:
                    if name.split(".")[0] in FILESYSTEM_MODULES:
                        location = path.relative_to(ROOT).as_posix()
                        offenders.append(f"{location}:{node.lineno} imports {name}")
        assert not offenders, (
            "the backend stores documents in S3 and parses them in memory; a filesystem "
            f"import is how that stops being true: {offenders}"
        )

    def test_no_shipped_module_opens_a_file(self) -> None:
        offenders: list[str] = []
        for path in backend_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name) and func.id == "open":
                    location = path.relative_to(ROOT).as_posix()
                    offenders.append(f"{location}:{node.lineno} open()")
        assert not offenders, f"a shipped module opens a file by path: {offenders}"

    def test_no_archive_member_is_ever_extracted_to_a_path(self) -> None:
        """Zip Slip, statically — no member name is ever turned into a write location.

        `extract` and `extractall` are the two calls that do that, and
        `shutil.unpack_archive` is the same thing one import away. `ZipFile.read(name)`
        returns bytes and is not on the list: it is what `document_parser` uses, and it
        writes nothing.
        """
        offenders: list[str] = []
        for path in backend_modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in EXTRACTION_CALLS
                ):
                    offenders.append(
                        f"{path.relative_to(ROOT).as_posix()}:{node.lineno} "
                        f"calls .{node.func.attr}()"
                    )
        assert not offenders, f"an archive member is extracted to a path: {offenders}"


class TestNoRequestFieldIsAPath:
    """Section 3b — nothing a client can set is named, or shaped, like a location."""

    def test_no_route_parameter_is_pathish(self) -> None:
        """Route signatures are the API's whole input surface for paths.

        Scanned by name rather than by type, because the type that would make this dangerous
        (`str`) is the same type that makes `search` harmless. A parameter called `path` on a
        route is the review signal; there is no legitimate one in this API.
        """
        offenders: list[str] = []
        routes = ROOT / "backend" / "api" / "routes"
        for path in sorted(routes.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not node.decorator_list:  # helpers, not routes
                    continue
                arguments = [*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs]
                for argument in arguments:
                    if argument.arg in PATHISH_NAMES:
                        offenders.append(
                            f"{path.relative_to(ROOT).as_posix()}:{node.lineno} "
                            f"{node.name}({argument.arg})"
                        )
        assert not offenders, f"a route accepts a path-like parameter: {offenders}"

    def test_no_request_schema_declares_a_pathish_field(self) -> None:
        """The other half: a field on a POST body reaches a service just as directly.

        Covers the whole of `backend/schemas/`, request and response models alike. A response
        model carrying a `key` would be an information leak rather than traversal, and neither
        belongs.
        """
        offenders: list[str] = []
        for path in sorted((ROOT / "backend" / "schemas").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                for statement in node.body:
                    if not isinstance(statement, ast.AnnAssign):
                        continue
                    target = statement.target
                    if isinstance(target, ast.Name) and target.id in PATHISH_NAMES:
                        offenders.append(
                            f"{path.relative_to(ROOT).as_posix()}:{statement.lineno} "
                            f"{node.name}.{target.id}"
                        )
        assert not offenders, f"a schema field is path-like: {offenders}"

    @pytest.mark.parametrize(
        "suffix",
        [
            "..%2F..%2Fetc%2Fpasswd",
            "..%252f..%252fetc%252fpasswd",
            "%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//etc/passwd",
            "..\\..\\windows\\win.ini",
            "file:///etc/passwd",
        ],
    )
    async def test_a_traversal_document_id_is_refused(
        self, api: Api, stack: Stack, suffix: str
    ) -> None:
        """`{document_id}` is typed `uuid.UUID`, so a path is a validation failure, not a read.

        The status is asserted as a set because it depends on where the request dies —
        Starlette's router (404), the path converter (422), or method matching (405) — and all
        three are correct. What matters is that it is never 200 and never carries file content.
        """
        client, as_user = api
        as_user(stack.a.user)
        response = await client.get(f"{API}/documents/{suffix}")

        assert response.status_code in (404, 405, 422), (
            f"{suffix!r} produced {response.status_code}: {response.text[:400]}"
        )
        body = response.text
        assert "root:" not in body, "a passwd file reached the response"
        assert "Traceback" not in body
        assert "documents/" not in body, "an S3 key was echoed back"


class TestAHostileArchiveIsNeverUnpacked:
    """Section 2 — Zip Slip, behaviourally.

    The static scan above proves no code calls `extractall`. This proves the whole ingestion
    path — magic-byte detection, `_assert_safe_archive`, `python-docx`, chunking, embedding —
    leaves nothing on disk when handed an archive whose member name is a traversal path.
    """

    async def test_a_docx_with_a_traversal_member_writes_no_file(
        self,
        api: Api,
        stack: Stack,
        tmp_path: pathlib.Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        # The working directory is where a relative `../../x` resolves from, so it is moved
        # somewhere disposable before the upload rather than after.
        monkeypatch.chdir(tmp_path)

        response = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "quarterly.docx",
                    make_docx_with_traversal_member(f"../../../../{ZIP_SLIP_MARKER}"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            headers=csrf_headers(),
        )
        # Accepted or rejected is not the claim; either is a defensible product decision. The
        # claim is that nothing was written to a path.
        assert response.status_code in (201, 400, 415, 422), response.text
        assert_nothing_was_written(tmp_path)

    async def test_the_traversal_member_is_not_in_the_extracted_text(
        self, api: Api, stack: Stack
    ) -> None:
        """A weaker but independent claim: the hostile part contributed no content.

        If its bytes appeared in a chunk they would also appear in an answer and a citation,
        which is a grounding defect (`rag-answer-grounding.md`) even though nothing was written
        to disk.
        """
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "quarterly.docx",
                    make_docx_with_traversal_member(f"../../{ZIP_SLIP_MARKER}"),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
            headers=csrf_headers(),
        )
        assert response.status_code in (201, 400, 415, 422), response.text

        stored = repr([chunk.text for chunk in stack.documents.chunks])
        assert "pwned by zip slip" not in stored
        assert ZIP_SLIP_MARKER not in stored


class TestTheExportFilenameIsServerComposed:
    """Section 4 — `Content-Disposition`, the one header that names a file on the client."""

    async def test_the_filename_is_a_fixed_pattern_with_no_caller_input(
        self, api: Api, stack: Stack
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.post(
            f"{API}/analytics/export",
            json={"report": "documents", "format": "pdf"},
            headers=csrf_headers(),
        )
        assert response.status_code == 200, response.text

        disposition = response.headers["Content-Disposition"]
        assert re.fullmatch(
            r'attachment; filename="docquery-(keywords|documents|trends|history)'
            r'-\d{8}-\d{6}\.(pdf|xlsx)"',
            disposition,
        ), disposition
        assert "/" not in disposition.split("filename=", 1)[1]
        assert "\\" not in disposition
        assert "\r" not in disposition and "\n" not in disposition

    async def test_an_unknown_field_cannot_smuggle_a_filename_in(
        self, api: Api, stack: Stack
    ) -> None:
        """`extra='forbid'` on the request schema, from the traversal side rather than the
        mass-assignment side.

        The interesting outcome is not the 422 itself — `test_mass_assignment.py` owns that —
        but that a request naming a file is refused outright rather than ignored, so there is
        no version of this call where a caller's filename is silently in play.
        """
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.post(
            f"{API}/analytics/export",
            json={
                "report": "documents",
                "format": "pdf",
                "filename": "../../../../etc/cron.d/docquery",
            },
            headers=csrf_headers(),
        )
        assert response.status_code == 422, response.text
        assert "cron.d" not in response.text.replace("filename", ""), response.text
