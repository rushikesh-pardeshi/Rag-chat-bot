"""backend/tests/security/test_log_injection.py

Role: Prove that no user-supplied string can forge a log record, and that no user-supplied
string is reflected into a response header.

`.claude/rules/secure-coding-vapt.md`: *"**Log Injection**: Sanitize CR/LF characters from all
user inputs before logging. Never log raw untrusted data (queries, document metadata,
usernames) without sanitization."*

Why a forged log line matters here specifically
----------------------------------------------
`audit-log-immutability.md` makes the audit trail the evidence of record for a compliance
review. The structlog stream is not that trail — it lives in PostgreSQL with an integrity
hash — but the stream is what an operator reads during an incident and what a SIEM alerts on.
A question containing `\\r\\nWARNING login.success email=someone@bank.example` puts a second,
fabricated record into it. Nothing is tampered with; the *interpretation* is, which is harder
to detect and just as effective.

Two layers, and why the assertions are on the rendered line
-----------------------------------------------------------
`sanitize_log_value` replaces C0/C1 control characters with a space, and `JSONRenderer` escapes
whatever is left. `TestTheRenderedLine` asserts the rendered output rather than the sanitizer's
return value, because the two layers do not cover the same characters. U+2028 LINE SEPARATOR
passes the sanitizer untouched — it is in neither the C0 nor the C1 range — and is neutralized
only by `json.dumps`'s `ensure_ascii`. A test that stopped at `sanitize_log_value` would report
that payload as handled while proving nothing about the line an operator sees.

Which layer is load-bearing, measured rather than assumed
---------------------------------------------------------
I checked by breaking each layer and re-running. Replacing `_CONTROL_CHARS.sub(" ", value)`
with `value` failed 6 tests, all of them in `TestSanitizeLogValue` — and **not one** in
`TestTheRenderedLine`. That is the honest picture: while `configure_logging` installs
`JSONRenderer`, a CR/LF inside a string value is escaped to `\\r\\n` and cannot open a record,
so the renderer is what actually prevents forgery today. `sanitize_log_value` is the second
layer, and it earns its place two ways: it is what keeps the *field* readable rather than
escape-littered, and it is the only control that survives someone swapping in
`structlog.dev.ConsoleRenderer` — a one-line change that is routinely made for local
readability and that would make CR/LF injection live again.

Both layers are asserted separately for exactly that reason. A single combined test would go
green on either one alone, and the day the renderer changed nobody would learn which had been
carrying the guarantee.

The response header
-------------------
`TestTheRequestIdHeader` covers a defect this module found rather than one it was written to
confirm. `X-Request-ID` was read from the request and echoed into the response verbatim, and
uvicorn's httptools implementation — the one `uvicorn[standard]` selects — serializes response
headers by byte concatenation with no validation, so `abc\\r\\nSet-Cookie: …` was a genuine
response-splitting primitive on our own origin. `_safe_request_id` now validates it. The full
finding, including why the impact varied with a deployment flag, is in that class's docstring.
"""

from __future__ import annotations

import io
import json
import logging
import re
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx
import pytest
import structlog

from backend.api.app import REQUEST_ID_HEADER, _safe_request_id
from backend.core.logging import (
    _SHARED_PROCESSORS,
    MAX_VALUE_LENGTH,
    REDACTED,
    _redact_and_sanitize,
    configure_logging,
    sanitize_log_value,
)
from backend.models.user import User
from backend.tests.security.conftest import (
    API,
    LOG_INJECTION_PAYLOADS,
    Stack,
    assert_no_forged_line,
    csrf_headers,
)

Api = tuple[httpx.AsyncClient, Callable[[User | None], None]]

#: U+2028 LINE SEPARATOR and U+2029 PARAGRAPH SEPARATOR. Line breaks to a JSON parser and to
#: most log viewers, but not control characters as far as `sanitize_log_value` is concerned.
#: Written as escapes deliberately: as literal characters they are invisible in a diff, and an
#: editor that normalized them to ordinary spaces would leave tests that assert nothing.
UNICODE_LINE_BREAKS: tuple[str, ...] = (" ", " ")


def render(**fields: object) -> str:
    """One log line, through the real processor and the real renderer.

    Deliberately not `structlog.testing.capture_logs`, which captures the event dict *before*
    the processor chain runs — the exact step under test.
    """
    processed = _redact_and_sanitize(None, "info", dict(fields))
    rendered = structlog.processors.JSONRenderer()(None, "info", processed)
    return str(rendered)


@contextmanager
def captured_log_stream() -> Iterator[io.StringIO]:
    """Every record the root logger emits, rendered exactly as production renders it.

    Not `capsys`. `configure_logging` does `logging.StreamHandler(sys.stdout)`, which binds the
    stream object *at configure time* — so whether pytest's replacement stdout is the one the
    handler holds depends on whether the app fixture happened to be built inside the capture.
    A test whose subject is "what lands in the log" must not rest on that ordering.

    The formatter is built from `_SHARED_PROCESSORS` plus `JSONRenderer` — the same pair
    `configure_logging` installs — so `_redact_and_sanitize` runs here because production runs
    it, not because this helper asked for it.

    `configure_logging()` is called first, and that call is load-bearing rather than tidy.
    structlog only routes into the stdlib root logger once `ProcessorFormatter` is installed;
    unconfigured, `structlog.get_logger()` writes straight to stdout through `PrintLogger` and
    nothing attached here would see a single record. Every test that reached this helper
    through the `api` fixture was configured as a side effect of `create_app()` — so a security
    assertion had anything to assert on depended on fixture ordering. The root handler list is
    snapshotted and restored, so a caller that was not expecting its logging to be reconfigured
    is left as it was found.
    """
    root = logging.getLogger()
    # Snapshot before `configure_logging`, which clears the handler list — taken afterwards it
    # would restore that call's state rather than the caller's.
    previous_level = root.level
    previous_handlers = list(root.handlers)

    configure_logging()
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=_SHARED_PROCESSORS,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )
    root.addHandler(handler)
    # An INFO floor, read from the level `configure_logging` just applied: `LOG_LEVEL` in the
    # test environment is not this module's business, and a WARNING root would make every
    # assertion below vacuously true.
    if root.level > logging.INFO:
        root.setLevel(logging.INFO)
    try:
        yield stream
    finally:
        root.handlers[:] = previous_handlers
        root.setLevel(previous_level)


def json_lines(stream: io.StringIO) -> list[dict[str, object]]:
    """Parse the captured stream, failing loudly on any line that is not one JSON object.

    An unparseable line *is* the finding this module looks for, so it has to be a failure
    carrying the offending text — not a silently skipped entry.
    """
    parsed: list[dict[str, object]] = []
    for line in stream.getvalue().splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - the failure path
            pytest.fail(
                f"a log line is not one JSON object, so a record was forged: {line!r} ({exc})"
            )
    return parsed


class TestSanitizeLogValue:
    """The first layer, tested alone so a failure says which layer broke."""

    @pytest.mark.parametrize("payload", LOG_INJECTION_PAYLOADS)
    def test_a_payload_loses_its_control_characters(self, payload: str) -> None:
        assert_no_forged_line(str(sanitize_log_value(payload)))

    def test_nested_values_are_walked(self) -> None:
        """Audit `detail` is a dict and `source_references` is a list of dicts.

        A sanitizer that only handled top-level strings would leave both raw — and those are
        exactly where user text lives in this application's log calls.
        """
        dirty = {
            "detail": {"question": "a\r\nFORGED", "nested": ["b\nFORGED", {"c": "d\rFORGED"}]},
            "count": 3,
        }

        cleaned = sanitize_log_value(dirty)

        assert_no_forged_line(json.dumps(cleaned))
        assert cleaned["count"] == 3, "a non-string scalar was altered"

    def test_a_long_value_is_truncated_with_a_marker(self) -> None:
        """Truncation is flood control, and the marker is what keeps it honest.

        A silently clipped value looks like the whole value. The cap matters because a document
        name is user-supplied and a question can be 2000 characters.
        """
        cleaned = sanitize_log_value("a" * (MAX_VALUE_LENGTH + 500))

        assert cleaned.endswith("…[truncated]")
        assert len(cleaned) < MAX_VALUE_LENGTH + 50

    def test_ordinary_text_and_non_latin_scripts_survive(self) -> None:
        """The sanitizer is not a general-purpose stripper.

        A question in a non-Latin script has to stay readable in the log, or an operator
        investigating an incident cannot tell which query they are looking at.
        """
        assert sanitize_log_value("wire transfer fee") == "wire transfer fee"
        assert sanitize_log_value("शुल्क क्या है") == "शुल्क क्या है"

    @pytest.mark.parametrize("separator", UNICODE_LINE_BREAKS)
    def test_a_unicode_line_break_is_not_stripped_here(self, separator: str) -> None:
        """Documenting the boundary, not asserting an ideal.

        `_CONTROL_CHARS` is `[\\x00-\\x1f\\x7f-\\x9f]`, and U+2028 is in neither range, so it
        reaches the renderer intact. That is safe *because* `JSONRenderer` escapes it — see
        `TestTheRenderedLine::test_a_unicode_line_break_is_escaped_by_the_renderer`, which is
        where the guarantee actually lives. If this test ever fails because someone widened the
        pattern, that is an improvement: delete this test and keep the other one.
        """
        assert separator in str(sanitize_log_value(f"before{separator}after"))


class TestTheRenderedLine:
    """What an operator and a SIEM actually receive. The assertions that count."""

    @pytest.mark.parametrize("payload", LOG_INJECTION_PAYLOADS)
    def test_a_payload_cannot_forge_a_second_record(self, payload: str) -> None:
        line = render(event="query.submitted", question=payload)

        assert line.count("\n") == 0, f"the rendered line spans records: {line!r}"
        assert json.loads(line)["event"] == "query.submitted", (
            "the payload changed which event this line reports"
        )

    @pytest.mark.parametrize("payload", LOG_INJECTION_PAYLOADS)
    def test_a_payload_cannot_forge_a_field(self, payload: str) -> None:
        """The JSON-shaped payload is `benign","level":"critical","event":"fabricated`.

        A renderer that concatenated strings would produce a line whose `level` is `critical`
        and whose `event` is `fabricated`, and a SIEM rule keyed on either would fire on an
        event that never happened. `json.dumps` escaping the quotes is what stops it, so the
        assertions are on the parsed object rather than on the text.
        """
        parsed = json.loads(render(event="query.submitted", level="info", question=payload))

        assert parsed["level"] == "info", f"the payload rewrote the level: {parsed}"
        assert parsed["event"] == "query.submitted", f"the payload rewrote the event: {parsed}"
        assert parsed["question"] == sanitize_log_value(payload), (
            "the question field is neither the sanitized payload nor anything explicable: "
            f"{parsed['question']!r}"
        )
        assert len(parsed) == 3, f"the payload added a field: {sorted(parsed)}"

    @pytest.mark.parametrize("separator", UNICODE_LINE_BREAKS)
    def test_a_unicode_line_break_is_escaped_by_the_renderer(self, separator: str) -> None:
        """The layer that catches what the sanitizer does not.

        `json.dumps` defaults to `ensure_ascii=True`, so U+2028 leaves as the six characters
        `\\u2028` and a viewer splitting on it sees one line. If someone later sets
        `ensure_ascii=False` for readability this test fails, and the fix is then to widen
        `_CONTROL_CHARS` — not to relax the assertion.
        """
        line = render(event="query.submitted", question=f"before{separator}after")

        assert separator not in line, "the raw separator reached the rendered line"
        assert "\\u202" in line, f"the separator was neither escaped nor stripped: {line!r}"
        assert json.loads(line)["question"] == f"before{separator}after"

    def test_a_sensitive_key_is_redacted_rather_than_sanitized(self) -> None:
        """Ordering matters: redaction replaces the value instead of cleaning it.

        A sanitized password is still a password. `_redact_and_sanitize` branches on the key
        before it ever looks at the value, and this asserts the value appears nowhere.
        """
        line = render(event="auth.login", password="DocQuery!Local2026", email="a@b.example")

        assert REDACTED in line
        assert "DocQuery!Local2026" not in line


class TestUserInputThroughTheRealApplication:
    """End to end, because a sanitizer only helps on the code paths that call it."""

    @pytest.mark.parametrize("payload", LOG_INJECTION_PAYLOADS)
    async def test_a_question_carrying_a_payload_forges_no_record(
        self, stack: Stack, api: Api, payload: str
    ) -> None:
        """Every line the request emits must parse as exactly one JSON object.

        This is the assertion that survives a refactor: it names no field and no logger, so a
        log call added anywhere in the query path is covered the day it ships. `json_lines`
        fails on the first line that is not one object, which is what a forged record looks
        like from the outside.
        """
        client, as_user = api
        as_user(stack.a.user)

        with captured_log_stream() as stream:
            response = await client.post(
                f"{API}/queries",
                json={"question_text": f"What is the fee for {payload}"},
                headers=csrf_headers(),
            )

        assert response.status_code == 202, response.text
        lines = json_lines(stream)
        assert lines, "the request emitted no log lines at all, so this proves nothing"
        for line in lines:
            assert_no_forged_line(json.dumps(line))

    @pytest.mark.parametrize("payload", LOG_INJECTION_PAYLOADS)
    async def test_the_http_line_reports_the_real_outcome(
        self, stack: Stack, api: Api, payload: str
    ) -> None:
        """The specific lie a CRLF payload is for: a fabricated status or path.

        `http.request` carries `status_code` and `path`. If a payload could open a new record,
        the cheapest useful forgery is a second `http.request` line reporting 200 on a path
        that was never called — which is how an attacker hides the 403s from a probing run.
        """
        client, as_user = api
        as_user(stack.a.user)

        with captured_log_stream() as stream:
            await client.post(
                f"{API}/queries",
                json={"question_text": f"Fee for {payload}"},
                headers=csrf_headers(),
            )

        http_lines = [
            line for line in json_lines(stream) if line.get("event") == "http.request"
        ]
        assert len(http_lines) == 1, f"expected one http.request line, got {http_lines}"
        assert http_lines[0]["status_code"] == 202
        assert http_lines[0]["path"] == f"{API}/queries"

    async def test_a_hostile_document_name_does_not_reach_the_log_raw(
        self, stack: Stack, api: Api
    ) -> None:
        """Filenames take a different path: `sanitize_filename` first, then the log.

        Two independent controls, and the upload audit entry deliberately carries no filename
        at all (`document_service.upload`: a document name can carry a customer identifier).
        """
        client, as_user = api
        as_user(stack.a.admin)

        with captured_log_stream() as stream:
            response = await client.post(
                f"{API}/documents",
                files={
                    "file": (
                        "statement\r\nERROR fabricated.event.pdf",
                        b"%PDF-1.4\n%%EOF\n",
                        "application/pdf",
                    )
                },
                headers=csrf_headers(),
            )

        assert response.status_code == 201, response.text
        text = stream.getvalue()
        assert "fabricated.event" not in text, (
            "the hostile filename reached the log stream; the upload audit entry is supposed "
            f"to carry ids only: {text[:600]}"
        )
        for line in json_lines(stream):
            assert_no_forged_line(json.dumps(line))

    async def test_a_hostile_request_id_forges_no_record(self, stack: Stack, api: Api) -> None:
        """The header is bound into *every* line of the request via `bind_request_context`.

        That makes it the highest-leverage injection point in the module: one hostile header
        contaminates every record the request produces, not just one.
        """
        client, as_user = api
        as_user(stack.a.user)

        with captured_log_stream() as stream:
            await client.get(
                f"{API}/auth/me",
                headers={REQUEST_ID_HEADER: "abc\r\n{\"event\":\"audit.tampered\"}"},
            )

        lines = json_lines(stream)
        assert lines, "no log lines captured"
        for line in lines:
            assert line.get("event") != "audit.tampered", f"the header forged a record: {line}"

        # Only lines emitted *inside* the request carry a `request_id`; httpx's own client
        # line fires after `clear_request_context`, and demanding an id there would be
        # asserting that the context leaks. Every line that has one must have the substitute.
        bound = [line for line in lines if line.get("request_id") is not None]
        assert bound, f"no line carried a request_id, so the binding is untested: {lines}"
        for line in bound:
            identifier = str(line["request_id"])
            assert "audit.tampered" not in identifier, (
                f"the hostile header value was bound into the log context: {identifier!r}"
            )
            uuid.UUID(identifier)  # raises unless the middleware substituted a fresh UUID


class TestTheRequestIdHeader:
    """A defect this module found. The class exists to keep it fixed.

    What was wrong: the `request_context` middleware read `X-Request-ID` from the request and
    wrote it back on the response unchanged. Both halves are user-controlled output — one into
    the log stream, one into the HTTP response — and only the log half was sanitized.

    Why it was more than untidy: `requirements.txt` pins `uvicorn[standard]`, which installs
    `httptools`, and `httptools_impl.py` builds the response head with
    `content.extend([name, b": ", value, b"\\r\\n"])` — no validation of `value`. A CR/LF in
    the echoed id is therefore an injected response header, or a second response body, served
    from our own origin: a `Set-Cookie`, or HTML the browser trusts because the origin is ours.
    Under the pure-Python `h11` implementation the same value raises `LocalProtocolError` and
    fails the request instead. A guarantee that changes with a server flag is not a guarantee,
    which is why the validation now lives in the application.

    Reachability, stated plainly rather than at worst case: a browser sends this header only if
    script sets it, and CORS lists `X-Request-ID` in `allow_headers` against an explicit origin
    allowlist, so an arbitrary attacker page cannot preflight it. The realistic injectors are
    an allowed-origin script and any intermediary that forwards a client-supplied value.
    Bounded — but not theoretical, and the fix costs one regex.
    """

    @pytest.mark.parametrize(
        ("name", "hostile"),
        [
            ("header injection", "abc\r\nSet-Cookie: docquery_access=stolen"),
            ("response splitting", "abc\r\n\r\n<html>attacker</html>"),
            ("bare LF", "abc\ndef"),
            ("NUL", "abc\x00def"),
            ("ANSI escape", "abc\x1b[31mred"),
            ("oversized", "a" * 5000),
            ("spaces", "id with spaces"),
            ("empty", ""),
        ],
    )
    async def test_a_hostile_request_id_is_not_echoed(
        self, stack: Stack, api: Api, name: str, hostile: str
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)

        response = await client.get(f"{API}/auth/me", headers={REQUEST_ID_HEADER: hostile})

        echoed = response.headers[REQUEST_ID_HEADER]
        assert echoed != hostile, f"{name}: the header was echoed verbatim: {echoed!r}"
        # A UUID, because that is what replaces a rejected value.
        uuid.UUID(echoed)

    async def test_a_legitimate_request_id_is_preserved(self, stack: Stack, api: Api) -> None:
        """The positive control. A correlation id nobody can rely on is not one.

        Tracing across services depends on the caller's id surviving. A validator that replaced
        every value would be perfectly "safe", useless, and would pass every test above.
        """
        client, as_user = api
        as_user(stack.a.user)
        given = "req-01JB4KQ7Z8.trace_9:0"

        response = await client.get(f"{API}/auth/me", headers={REQUEST_ID_HEADER: given})

        assert response.headers[REQUEST_ID_HEADER] == given

    @pytest.mark.parametrize(
        "value",
        [
            "0f9c2d3e-4b5a-4c6d-8e7f-a1b2c3d4e5f6",
            # A W3C `traceparent`, which is what a service mesh forwards.
            "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "req-abc123",
            "a",
            "A" * 128,
        ],
    )
    def test_the_accepted_shapes_cover_real_correlation_ids(self, value: str) -> None:
        assert _safe_request_id(value) == value

    @pytest.mark.parametrize(
        "value",
        [None, "", "A" * 129, "abc\r\nx", "abc def", "abc<script>", "id;drop", "über-id"],
    )
    def test_the_rejected_shapes_get_a_fresh_uuid(self, value: str | None) -> None:
        replacement = _safe_request_id(value)

        assert replacement != value
        uuid.UUID(replacement)

    def test_the_match_is_anchored(self) -> None:
        """`fullmatch`, not `search` — the distinction is the entire control.

        `search` would accept `abc\\r\\nSet-Cookie: x`, because the safe pattern matches the
        leading `abc`. Asserted against the source so a later "simplification" fails here
        rather than in production, where the symptom would be a stolen session.
        """
        import inspect

        from backend.api import app as app_module

        source = inspect.getsource(app_module._safe_request_id)
        assert ".fullmatch(" in source, "the request-id check is no longer anchored"
        assert re.search(r"_SAFE_REQUEST_ID\.(search|match)\(", source) is None, (
            "`search`/`match` accept a safe prefix followed by anything, which is the bug"
        )

    async def test_every_response_still_carries_an_id(self, stack: Stack, api: Api) -> None:
        """Traceability does not become optional because a header was refused.

        `audit-log-immutability.md` wants 100% traceability; a request whose response carried
        no correlation id would be one an operator cannot tie back to its log lines.
        """
        client, as_user = api
        as_user(stack.a.user)

        for headers in ({}, {REQUEST_ID_HEADER: "abc\r\nx"}, {REQUEST_ID_HEADER: "req-ok"}):
            response = await client.get(f"{API}/auth/me", headers=headers)
            assert response.headers.get(REQUEST_ID_HEADER), (
                f"no request id on the response for headers={headers}"
            )
