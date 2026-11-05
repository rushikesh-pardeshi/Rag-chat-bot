"""backend/tests/security/test_secrets_in_logs.py

Role: prove that nothing this process holds as a credential — and no password a user types —
reaches the log stream, an audit row, or an HTTP response.

The first rule of this module is about its own failure messages
--------------------------------------------------------------
A test that proved the log was clean by printing the leaked key into a pytest failure message
would have moved the leak from the application log to the CI log, which is at least as widely
readable and usually kept longer. So every assertion here names the *setting* and never the
value. When one fails, the next step is to grep the log for the field name it reports.

What counts as a secret here
---------------------------
Five fields are `SecretStr` and mask themselves: `SECRET_KEY`, the three `AWS_*` credentials
and `PINECONE_API_KEY`. Four are plain `str` and do not: `DATABASE_URL`, `REDIS_URL`,
`CELERY_BROKER_URL` and `RATE_LIMIT_STORAGE_URI` are connection URLs, and a deployed one reads
`postgresql+asyncpg://user:PASSWORD@host/db` — a password sitting inside an ordinary string
whose field *name* matches nothing in `_SENSITIVE_KEY_PATTERN`. That asymmetry is what
`TestSettingsDoNotLeakThroughRepr` is about, and the fix it asserts — `repr=False` on those
fields and on the `sync_database_url` computed field — came out of writing it.

The secret list is derived from `Settings`, not typed out
--------------------------------------------------------
`_secret_values()` reads `Settings.model_fields`: anything annotated `SecretStr` and anything
carrying `repr=False`. A credential-bearing field added next quarter is therefore covered
without anyone remembering this file exists — the only way a list like this stays true.

Redaction by key name cannot be the only control
------------------------------------------------
`_redact_and_sanitize` matches *keys*. botocore at DEBUG puts an entire signed request — the
`Authorization` header, an STS session token — into `event`, a key that matches nothing, so
redaction is structurally unable to help there. The control is the INFO floor on
`_WIRE_LEVEL_LOGGERS`, and `TestWireLevelLoggersAreFloored` asserts it on each library's own
logger rather than through inheritance: an inherited level follows the root down the moment
someone sets `LOG_LEVEL=DEBUG`, which is exactly when a developer is chasing a wire-level bug.

Why the password test builds `AuthService` by hand
-------------------------------------------------
`conftest.py`'s `api` fixture overrides the repository providers but not `get_auth_service`, so
`/auth/login` cannot be driven over HTTP in this suite. A password only ever exists inside
`AuthService`, so the test constructs one over the same fakes and calls it directly. Nothing is
lost — no middleware touches a password.

Every "the secret is absent" assertion needs a positive control
--------------------------------------------------------------
"The sentinel is not in the captured stream" also passes when the stream is empty, when the
capture is attached to the wrong logger, and when the request failed before reaching the code
under test. `test_the_capture_would_notice_a_leak` logs a sentinel under a harmless key and
asserts it *does* appear, so the machinery is proven live in the same run.
"""

from __future__ import annotations

import ast
import logging
import pathlib
import re
import uuid
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest

from backend.core.config import Settings, get_settings
from backend.core.logging import (
    _SENSITIVE_KEY_PATTERN,
    _WIRE_LEVEL_LOGGERS,
    REDACTED,
    configure_logging,
    get_logger,
    sanitize_log_value,
)
from backend.models.user import User
from backend.services.auth_service import AuthService
from backend.tests.fakes import FakeAuthRepository
from backend.tests.security.conftest import API, CONTEXT, SHARED_QUESTION, Stack, csrf_headers
from backend.tests.security.test_log_injection import (
    Api,
    captured_log_stream,
    json_lines,
    render,
)

#: A password no other fixture uses, long enough to satisfy `MIN_PASSWORD_LENGTH` (12) and
#: distinctive enough that a substring hit anywhere is a real finding, not a coincidence.
SENTINEL_PASSWORD = "Sentinel-Wire-Fee-2026-nolog"

#: Goes into a dependency's exception message. Nothing legitimate produces this string.
SENTINEL_IN_EXCEPTION = "SENTINEL-BEDROCK-CREDENTIAL-9f3a2c"


def _public_configuration() -> str:
    """Every setting value that is *not* a credential, as one string.

    Used to decide whether a candidate secret is distinctive enough to search for. See
    `_secret_values`.
    """
    settings = get_settings()
    values = []
    for name, field in Settings.model_fields.items():
        if "SecretStr" in str(field.annotation) or not field.repr:
            continue
        values.append(f"{name}={getattr(settings, name)!r}")
    return " ".join(values)


def _secret_values() -> tuple[list[tuple[str, str]], list[str]]:
    """Every credential this process holds, as `(name, value)` pairs, plus what was skipped.

    `SecretStr` fields are unwrapped. Fields carrying `repr=False` are the connection URLs, and
    each contributes two candidates: the whole URL, and its password component on its own —
    because a DSN reassembled by a driver will not match the URL we configured, while the
    password inside it is still the password.

    A candidate is skipped when it also appears in the non-credential configuration, and that
    filter is not a convenience. This repository's test DSN is
    `postgresql+asyncpg://docquery:docquery@localhost/docquery_test`: the password is the word
    `docquery`, which is also in `APP_NAME`, the S3 bucket and the Pinecone index name.
    Searching the log for it reports a leak on every line that names the application. A local
    password equal to the project name is simply not testable by substring search — the
    whole-URL candidate still covers it, and a deployed password is random. Skips are returned
    rather than dropped so `test_the_scan_covers_what_it_claims_to` can assert what was lost.
    """
    settings = get_settings()
    public = _public_configuration()
    found: list[tuple[str, str]] = []
    skipped: list[str] = []

    def offer(name: str, value: str | None) -> None:
        if not value:
            return
        if value in public:
            skipped.append(name)
            return
        found.append((name, value))

    for name, field in Settings.model_fields.items():
        if "SecretStr" in str(field.annotation):
            holder = getattr(settings, name)
            offer(name, None if holder is None else holder.get_secret_value())
            continue
        if field.repr:
            continue
        raw = getattr(settings, name)
        if not isinstance(raw, str):
            continue
        offer(name, raw)
        offer(f"{name}:password", urlsplit(raw).password)
    return found, skipped


def _credential_field_names() -> frozenset[str]:
    """Setting names a log call must never read. Derived the same way as `_secret_values`."""
    names = {
        name
        for name, field in Settings.model_fields.items()
        if "SecretStr" in str(field.annotation) or not field.repr
    }
    # Not a model field, but it is the DSN with the driver renamed.
    names.add("sync_database_url")
    return frozenset(names)


async def _exercise_the_api(
    client: httpx.AsyncClient, as_user: Callable[[User | None], None], stack: Stack
) -> list[tuple[str, int]]:
    """Drive every read and write route both roles can reach, and report what each returned.

    Breadth is the point. A credential does not leak from the endpoint anyone would guess; it
    leaks from the one that happens to log a client construction, a presigned URL or a retry.
    """
    as_user(stack.a.admin)
    calls = [
        ("GET", f"{API}/auth/me", {}),
        ("GET", f"{API}/documents", {}),
        ("GET", f"{API}/documents/{stack.a.document.id}", {}),
        ("GET", f"{API}/documents/{stack.a.document.id}/preview", {}),
        ("GET", f"{API}/documents/{stack.a.document.id}/versions", {}),
        ("GET", f"{API}/analytics/keywords", {}),
        ("GET", f"{API}/analytics/documents", {}),
        ("GET", f"{API}/analytics/trends", {}),
        ("GET", f"{API}/audit", {}),
        ("GET", f"{API}/audit/verify", {}),
        ("GET", f"{API}/history", {}),
    ]
    outcomes: list[tuple[str, int]] = []
    for _method, path, _kwargs in calls:
        response = await client.get(path)
        outcomes.append((path, response.status_code))

    upload = await client.post(
        f"{API}/documents",
        files={"file": ("fee-schedule.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")},
        headers=csrf_headers(),
    )
    outcomes.append((f"{API}/documents", upload.status_code))

    export = await client.post(
        f"{API}/analytics/export",
        json={"report": "keywords", "format": "pdf"},
        headers=csrf_headers(),
    )
    outcomes.append((f"{API}/analytics/export", export.status_code))

    as_user(stack.a.user)
    stack.bedrock.queue_answer("The outbound domestic wire transfer fee is 25 USD [S1].")
    submitted = await client.post(
        f"{API}/queries", json={"question_text": SHARED_QUESTION}, headers=csrf_headers()
    )
    outcomes.append((f"{API}/queries", submitted.status_code))
    history = await client.get(f"{API}/history")
    outcomes.append((f"{API}/history", history.status_code))
    return outcomes


class TestTheRedactionGuard:
    """`_SENSITIVE_KEY_PATTERN` against the names this codebase and its libraries actually use.

    The should-match list is not aspirational: every entry is either a field on `Settings`, a
    key a service passes to `logger`, or a header name boto3 and httpx put in an event dict.
    Three defects were found by writing it out — `aws_access_key_id` matched nothing,
    `pan`/`card` matched `company` and `span_count`, and a sensitive key one level deep was
    sanitized rather than redacted.
    """

    SHOULD_REDACT = (
        "password",
        "passwd",
        "user_password",
        "new_password",
        "hashed_password",
        "secret",
        "secret_key",
        "SECRET_KEY",
        "client_secret",
        "token",
        "access_token",
        "refresh_token",
        "csrf_token",
        "authorization",
        "Authorization",
        "api_key",
        "apikey",
        "api-key",
        "pinecone_api_key",
        "aws_access_key_id",
        "AWS_SECRET_ACCESS_KEY",
        "aws_session_token",
        "private_key",
        "credential",
        "credentials",
        "bearer",
        "bearer_token",
        "jwt",
        "jwt_claims",
        "cookie",
        "set_cookie",
        "session",
        "session_id",
        "account_number",
        "accountNumber",
        "ssn",
        "aadhaar",
        "pan",
        "pan_number",
        "card",
        "card_number",
        "customer.card",
    )

    #: Operational field names that must survive. Over-redaction is not a breach, but a masked
    #: counter is indistinguishable from a missing one, and an operator who cannot trust the
    #: fields stops reading the log — which costs more than the redaction saved.
    SHOULD_NOT_REDACT = (
        "company",
        "company_name",
        "span_count",
        "panel",
        "panel_id",
        "expanded_pages",
        "email",
        "actor_email",
        "document_name",
        "document_id",
        "question",
        "tenant_id",
        "user_id",
        "request_id",
        "page_count",
        "chunk_count",
        "candidate_count",
        "status_code",
        "latency_ms",
        "integrity_hash",
        "error_type",
        "outcome",
        "scanned",
    )

    @pytest.mark.parametrize("key", SHOULD_REDACT)
    def test_a_sensitive_key_is_redacted(self, key: str) -> None:
        line = render(**{key: "the-value-under-test"})
        assert REDACTED in line, f"{key} was not redacted"
        assert "the-value-under-test" not in line, f"{key} kept its value"

    @pytest.mark.parametrize("key", SHOULD_NOT_REDACT)
    def test_an_operational_key_survives(self, key: str) -> None:
        line = render(**{key: "the-value-under-test"})
        assert "the-value-under-test" in line, f"{key} was redacted but carries no secret"

    def test_a_secret_is_redacted_at_any_depth(self) -> None:
        """The nesting level that matters, because it is where this application's data is.

        The deepest user- and system-influenced structure in the codebase is the audit `detail`
        dict. `_redact_and_sanitize` inspects top-level keys only, so before
        `sanitize_log_value` learned to redact, `detail={"authorization": token}` sanitized the
        token and logged it in full — the guard sat one level above the data.
        """
        line = render(
            detail={
                "phase": "ingest",
                "authorization": "Bearer eyJleUnMistakable",
                "inner": [{"api_key": "pcsk_UnMistakable"}],
            }
        )
        assert "eyJleUnMistakable" not in line
        assert "pcsk_UnMistakable" not in line
        assert line.count(REDACTED) == 2, line
        assert "ingest" in line, "the non-sensitive sibling should still be readable"

    def test_a_non_string_secret_is_still_redacted(self) -> None:
        """An int or a list under a sensitive key is replaced, not walked.

        `logger.info("x", token=1234)` is a mistake somebody makes with a numeric OTP or a
        list of session ids, and a guard that only handled `str` would let both through.
        """
        assert REDACTED in render(token=1234567890)
        assert "1234567890" not in render(token=1234567890)
        assert "aaa" not in render(credentials=["aaa", "bbb"])

    def test_redaction_replaces_rather_than_truncates(self) -> None:
        """The whole value goes. A truncated key is still a key with its entropy reduced."""
        secret = "k" * 64
        line = render(secret_key=secret)
        assert "kkkk" not in line
        assert "…[truncated]" not in line

    def test_email_is_deliberately_not_redacted(self) -> None:
        """A recorded trade-off, not a gap — assert it so the decision has to be re-made.

        `rbac-enforcement.md` requires denied access attempts to be logged, and
        `auth_service._audit_login_failure` logs the attempted address for an unknown account
        because `audit_log.tenant_id` is NOT NULL and there is no tenant to chain the row to.
        An attempt with no subject is not an audit record. Adding `email` to the pattern would
        silently empty that control, so if a future PII review decides to, it must also decide
        where the denied-attempt evidence goes.
        """
        assert _SENSITIVE_KEY_PATTERN.search("email") is None
        assert "attacker@evil.invalid" in render(
            event="auth.login_failed_unknown_account", email="attacker@evil.invalid"
        )

    def test_a_single_word_cardholder_is_not_matched(self) -> None:
        """A known limit of anchoring `pan` and `card` to token boundaries.

        `card_number` and `customer.card` match; `cardholder` does not, because unanchored
        alternatives matched `company`, `panel` and `expanded_pages` and the trade favoured the
        operational fields. Snake_case is this codebase's convention, so the shape that misses
        is one nothing here produces.

        If this test ever fails because someone widened the pattern *and* kept the false
        positives out, that is an improvement: delete this test and keep the ones above.
        """
        assert _SENSITIVE_KEY_PATTERN.search("cardholder") is None

    def test_the_capture_would_notice_a_leak(self) -> None:
        """The positive control for every absence assertion in this module."""
        sentinel = "SENTINEL-POSITIVE-CONTROL"
        with captured_log_stream() as stream:
            get_logger(__name__).info("test.positive_control", note=sentinel)
        assert sentinel in stream.getvalue(), (
            "the capture mechanism is not seeing log records, so every 'absent' "
            "assertion in this module would pass vacuously"
        )


class TestSettingsDoNotLeakThroughRepr:
    """`repr(settings)` is one `logger.debug` away from every log line, so it must be clean.

    `SecretStr` handles five of the nine credential-bearing fields by itself; `test_health.py`
    already covers `SECRET_KEY` there. The four connection URLs are plain `str` and were not
    handled at all until `repr=False` was added to them, which is what these tests pin.
    """

    def test_a_secretstr_masks_itself(self) -> None:
        settings = get_settings()
        assert str(settings.SECRET_KEY) == "**********"
        assert settings.SECRET_KEY.get_secret_value() not in repr(settings)

    def test_no_credential_field_appears_in_the_repr(self) -> None:
        rendered = repr(get_settings())
        scanned, _ = _secret_values()
        for name, value in scanned:
            assert value not in rendered, f"{name} is readable in repr(settings)"

    def test_the_scan_covers_what_it_claims_to(self) -> None:
        """The credential list must not be quietly empty, nor may a whole field be skipped.

        `_secret_values` drops candidates that also appear in ordinary configuration, which is
        right for a weak local password but would be a hole if it ever dropped an entire field.
        A skipped *field* means the value is un-searchable and this module is not covering it —
        that has to fail, not pass with a shorter list.
        """
        scanned, skipped = _secret_values()
        names = {name for name, _ in scanned}
        assert "SECRET_KEY" in names, "the signing key is not being scanned for"
        assert "DATABASE_URL" in names, "the database DSN is not being scanned for"
        whole_fields = [name for name in skipped if not name.endswith(":password")]
        assert not whole_fields, (
            f"these credentials are indistinguishable from ordinary configuration and are "
            f"therefore not covered by any assertion here: {whole_fields}"
        )

    def test_every_connection_url_carries_repr_false(self) -> None:
        """A URL field added later without `repr=False` fails here rather than in an audit.

        The test is on the *shape* of the field rather than on a hard-coded list, because the
        next one will be a message queue or a metrics sink and nobody will think of this file.
        """
        offenders = [
            name
            for name, field in Settings.model_fields.items()
            if field.repr
            and str(field.annotation).startswith("<class 'str'>")
            and (name.endswith(("_URL", "_URI", "_DSN")))
        ]
        assert not offenders, (
            f"these connection settings are rendered in repr(settings) and can carry a "
            f"password: {offenders}. Add Field(repr=False)."
        )

    def test_the_derived_database_url_is_hidden_too(self) -> None:
        """`sync_database_url` is the DSN with the driver renamed — the password is unchanged.

        A computed field is rendered in `repr` and included in `model_dump()` independently of
        the field it derives from, so hiding `DATABASE_URL` alone left the same password one
        attribute to the left.
        """
        settings = get_settings()
        assert settings.sync_database_url not in repr(settings)
        assert "sync_database_url" not in repr(settings)

    def test_model_dump_still_carries_the_values(self) -> None:
        """`repr=False` must not have hidden the values from the code that needs them.

        `model_dump()` is how the Celery config and the Alembic environment read settings; a
        fix that quietly dropped fields from it would swap a logging defect for an outage.
        """
        dumped = get_settings().model_dump()
        for name in ("DATABASE_URL", "REDIS_URL", "CELERY_BROKER_URL"):
            assert dumped.get(name), f"{name} disappeared from model_dump()"
        assert dumped["sync_database_url"].startswith("postgresql://")


class TestNoLogCallCanReachACredential:
    """A static check on every `logger.*` call in `backend/`, parsed rather than grepped.

    Dynamic tests can only cover the paths they exercise, and the log call that leaks a
    credential is by definition on a path nobody exercised. This one reads the AST of every
    module and rejects four argument shapes:

      * a whole settings object (`settings`, `self._settings`, `get_settings()`) — structlog
        pretty-prints it, so it prints every field
      * `.model_dump()` — the same content under a different name, and `repr=False` does not
        affect it
      * `.get_secret_value()` — the only way a `SecretStr` can be unwrapped
      * `os.environ` / `os.getenv` — the environment holds everything `.env` does

    Naming a specific safe field is allowed: `logger.info("startup", environment=...)` is how
    the lifespan hook works and there is nothing wrong with it. Naming a *credential* field is
    not, and the list of those is derived from `Settings` so it cannot go stale.
    """

    LOG_METHODS = ("debug", "info", "warning", "error", "exception", "critical")

    #: `logger.info`, `log.warning`, `self._logger.error`, `_log.debug` — matched on the
    #: unparsed callee rather than by assuming one attribute name.
    CALLEE = re.compile(r"(^|\.)_?log(ger)?\.(" + "|".join(LOG_METHODS) + r")$")

    #: The whole object, under any of the names this codebase binds it to.
    WHOLE_SETTINGS = re.compile(r"^(self\.)?_?settings$|^get_settings\(\)$")

    FORBIDDEN_CALLS = ("get_secret_value", "model_dump", "os.environ", "os.getenv")

    def _source_files(self) -> list[pathlib.Path]:
        root = pathlib.Path(__file__).resolve().parents[2]
        return [p for p in sorted(root.rglob("*.py")) if "tests" not in p.parts]

    def test_the_scan_actually_finds_log_calls(self) -> None:
        """Guard against the scan silently matching nothing — the usual way this test rots."""
        total = sum(len(self._log_call_arguments(path)) for path in self._source_files())
        assert total > 40, f"only {total} log arguments found; the AST scan is not working"

    def _log_call_arguments(self, path: pathlib.Path) -> list[tuple[int, str]]:
        """Every argument expression passed to a logging call in one file, with its line."""
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        found: list[tuple[int, str]] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not self.CALLEE.search(ast.unparse(node.func)):
                continue
            expressions = [*node.args, *(kw.value for kw in node.keywords)]
            found.extend((node.lineno, ast.unparse(expr)) for expr in expressions)
        return found

    def test_no_log_call_passes_a_whole_settings_object(self) -> None:
        offenders = [
            f"{path.name}:{line} -> {expr}"
            for path in self._source_files()
            for line, expr in self._log_call_arguments(path)
            if self.WHOLE_SETTINGS.match(expr)
        ]
        assert not offenders, f"a log call renders every setting: {offenders}"

    def test_no_log_call_unwraps_or_dumps_configuration(self) -> None:
        offenders = [
            f"{path.name}:{line} -> {expr}"
            for path in self._source_files()
            for line, expr in self._log_call_arguments(path)
            if any(forbidden in expr for forbidden in self.FORBIDDEN_CALLS)
        ]
        assert not offenders, f"a log call reads a secret or the environment: {offenders}"

    def test_no_log_call_names_a_credential_setting(self) -> None:
        credentials = _credential_field_names()
        offenders = [
            f"{path.name}:{line} -> {expr}"
            for path in self._source_files()
            for line, expr in self._log_call_arguments(path)
            if any(re.search(rf"\.{name}\b", expr) for name in credentials)
        ]
        assert not offenders, f"a log call reads a credential setting: {offenders}"


class TestNoSettingValueReachesTheLogStream:
    """Every route both roles can reach, driven with the real middleware, then the stream read.

    This is the dynamic half. It cannot prove a negative about code it did not run, which is
    why the static scan above exists; what it does prove is that the paths a request actually
    takes — client construction, presigned URLs, Pinecone queries, the export writer — do not
    print what they were configured with.
    """

    async def test_no_credential_appears_anywhere_in_the_stream(
        self, api: Api, stack: Stack
    ) -> None:
        client, as_user = api
        with captured_log_stream() as stream:
            outcomes = await _exercise_the_api(client, as_user, stack)
        text = stream.getvalue()

        served = [path for path, status in outcomes if status < 500]
        assert len(served) >= 12, f"too little was exercised to conclude anything: {outcomes}"
        assert text.strip(), "nothing was logged, so this test proves nothing"

        scanned, _ = _secret_values()
        for name, value in scanned:
            assert value not in text, f"{name} reached the log stream"

    async def test_the_stream_stayed_machine_readable(
        self, api: Api, stack: Stack
    ) -> None:
        """A secret hidden in an unparseable line is a secret nobody would find by grepping.

        Kept here rather than left to `test_log_injection.py` because this run exercises far
        more routes, and an export writer or a presigned-URL path is exactly the sort of code
        that logs something structlog was not given a chance to render.
        """
        client, as_user = api
        with captured_log_stream() as stream:
            await _exercise_the_api(client, as_user, stack)
        lines = json_lines(stream)
        assert lines
        for line in lines:
            assert "event" in line, f"a record with no event field: {line}"

    async def test_no_credential_appears_in_any_response_body(
        self, api: Api, stack: Stack
    ) -> None:
        """The other direction. A settings value on the wire is worse than one in the log."""
        client, as_user = api
        as_user(stack.a.admin)
        bodies = []
        for path in (
            f"{API}/auth/me",
            f"{API}/documents",
            f"{API}/documents/{stack.a.document.id}/preview",
            f"{API}/analytics/keywords",
            f"{API}/audit",
            f"{API}/history",
        ):
            response = await client.get(path)
            bodies.append(response.text)
        combined = "\n".join(bodies)
        scanned, _ = _secret_values()
        for name, value in scanned:
            assert value not in combined, f"{name} was returned to the client"


class TestPasswordsNeverReachTheLogOrTheAuditTrail:
    """Register, sign in, get it wrong — with a sentinel password, watching everything.

    `AuthService` is built directly over the same fakes because the `api` fixture does not
    override `get_auth_service`. The four calls below cover every branch that has the plaintext
    in scope: registration hashes it, a success verifies it, a wrong password increments a
    counter and audits a failure, and an unknown email takes the timing-equalized path.
    """

    def _service(self, stack: Stack) -> AuthService:
        return AuthService(stack.users, FakeAuthRepository(), stack.audit)

    async def test_a_password_reaches_neither_the_log_nor_an_audit_row(
        self, stack: Stack
    ) -> None:
        service = self._service(stack)
        email = f"newcomer-{uuid.uuid4().hex[:8]}@tenant-a.example.com"

        with captured_log_stream() as stream:
            user = await service.register(
                email=email,
                password=SENTINEL_PASSWORD,
                full_name="Newcomer",
                context=CONTEXT,
            )
            await service.authenticate(
                email=email, password=SENTINEL_PASSWORD, context=CONTEXT
            )
            with pytest.raises(Exception):  # noqa: B017 - InvalidCredentials, by contract
                await service.authenticate(
                    email=email, password="Wrong-Sentinel-2026-x", context=CONTEXT
                )
            with pytest.raises(Exception):  # noqa: B017 - InvalidCredentials, by contract
                await service.authenticate(
                    email="nobody@tenant-a.example.com",
                    password=SENTINEL_PASSWORD,
                    context=CONTEXT,
                )
        text = stream.getvalue()

        assert text.strip(), "nothing was logged, so this test proves nothing"
        assert SENTINEL_PASSWORD not in text, "the plaintext password reached the log stream"
        assert "Wrong-Sentinel-2026-x" not in text, "a rejected password reached the log"
        assert SENTINEL_PASSWORD not in stack.audit_details(), (
            "the plaintext password reached an audit row, which is append-only and retained"
        )
        assert user.hashed_password not in text, (
            "the bcrypt hash reached the log stream; it is offline-crackable and identifies "
            "the account across log entries"
        )
        assert user.hashed_password not in stack.audit_details()

    async def test_the_unknown_account_line_is_still_written(self, stack: Stack) -> None:
        """The positive control, and a compliance requirement in its own right.

        If this fails, the test above is passing because nothing was logged at all — and
        `rbac-enforcement.md`'s "log all access attempts (authorized and denied)" is broken.
        """
        service = self._service(stack)
        with (
            captured_log_stream() as stream,
            pytest.raises(Exception),  # noqa: B017 - InvalidCredentials, by contract
        ):
            await service.authenticate(
                email="ghost@tenant-a.example.com",
                password=SENTINEL_PASSWORD,
                context=CONTEXT,
            )
        events = [line.get("event") for line in json_lines(stream)]
        assert "auth.login_failed_unknown_account" in events, events

    async def test_an_issued_token_is_not_logged(self, stack: Stack) -> None:
        """The access and refresh tokens are bearer credentials for the length of their TTL."""
        service = self._service(stack)
        email = f"tokens-{uuid.uuid4().hex[:8]}@tenant-a.example.com"
        await service.register(
            email=email, password=SENTINEL_PASSWORD, full_name=None, context=CONTEXT
        )
        with captured_log_stream() as stream:
            session = await service.authenticate(
                email=email, password=SENTINEL_PASSWORD, context=CONTEXT
            )
        text = stream.getvalue()
        for label, token in (
            ("access_token", session.access_token),
            ("refresh_token", session.refresh_token),
            ("csrf_token", session.csrf_token),
        ):
            assert token not in text, f"the issued {label} reached the log stream"

    async def test_a_password_field_name_is_redacted_even_when_the_caller_is_careless(
        self,
    ) -> None:
        """The backstop for the log call nobody has written yet.

        Everything above tests today's code. This asserts the guard that covers tomorrow's:
        a future `logger.info("auth.attempt", password=...)` is redacted at the processor, not
        left to review.
        """
        line = render(event="auth.attempt", password=SENTINEL_PASSWORD)
        assert SENTINEL_PASSWORD not in line
        assert REDACTED in line


class TestWireLevelLoggersAreFloored:
    """The control that does not depend on key names, because botocore does not use them.

    At DEBUG, botocore renders a whole signed request — `Authorization`, the STS session token,
    every header — into the message body. `_redact_and_sanitize` matches keys, and prose in
    `event` has no key to match, so redaction cannot reach it. The only reliable control is not
    asking these libraries for wire-level output.
    """

    def test_every_wire_level_logger_is_pinned_at_its_own_level(self) -> None:
        """`.level`, not `getEffectiveLevel()`, and that distinction is the whole test.

        An inherited level follows the root logger down to DEBUG the moment someone sets
        `LOG_LEVEL=DEBUG` — which is precisely when a developer is chasing an S3 or Pinecone
        problem and would get credentials in their terminal. An explicit `setLevel` on the
        library's own logger wins over the root, so it holds in that exact case.
        """
        configure_logging()
        for name in _WIRE_LEVEL_LOGGERS:
            level = logging.getLogger(name).level
            assert level >= logging.INFO, (
                f"{name} is not pinned above DEBUG (level={level}); with LOG_LEVEL=DEBUG it "
                f"would print signed requests"
            )

    def test_the_floor_covers_every_library_that_signs_a_request(self) -> None:
        """The AWS chain, the HTTP clients underneath it, and Pinecone's SDK.

        Named individually rather than checked as a set so that a library added to the tuple
        without thought does not quietly satisfy this test, and one removed fails it.
        """
        for name in ("boto3", "botocore", "s3transfer", "urllib3", "httpx", "pinecone"):
            assert name in _WIRE_LEVEL_LOGGERS, f"{name} can log a signed request"

    def test_uvicorn_access_stays_disabled(self) -> None:
        """It carries no request_id, and the middleware already logs the same request.

        Relevant here because an access line logs the full URL: any credential a client puts in
        a query string would be recorded by it, in a line the request-id binding never touches.
        """
        configure_logging()
        assert logging.getLogger("uvicorn.access").disabled is True


class TestAFailingDependencyDoesNotEchoItsMessage:
    """What happens when Bedrock fails with a credential in its exception text.

    The service layer's rule is `error_type=type(exc).__name__`, never `str(exc)`, and the
    reason is written at `query_service._answer`: a botocore error carries the AWS account id
    and the model ARN, and a database error can echo row values. These tests hold it to that.

    One thing here is recorded rather than fixed. An exception the handlers do not recognize
    reaches `handle_unexpected`, which calls `logger.exception` — and `format_exc_info` renders
    a traceback that *does* include the message. That stays: a traceback is how an operator
    diagnoses an outage, and removing it would trade a real capability for a hypothetical leak.
    What follows from it is a rule on the other side — an exception message must never be
    *constructed* from a credential — and `test_the_traceback_is_the_documented_exception` pins
    the behaviour so the trade-off is visible rather than discovered during an incident.
    """

    async def test_the_service_log_line_names_the_type_not_the_message(
        self, stack: Stack
    ) -> None:
        """Driven through the service rather than the route, because the route re-raises.

        `httpx.ASGITransport` re-raises an unhandled application exception by default, which is
        the right setting for the rest of the suite — a 500 should fail a test loudly rather
        than be asserted about. The subject here is the log line the service writes on its way
        out, and that is written before anything re-raises.
        """
        with captured_log_stream() as stream:
            stack.bedrock.fail_with = RuntimeError(f"boom {SENTINEL_IN_EXCEPTION}")
            with pytest.raises(RuntimeError):
                await stack.query_service.submit(
                    question=SHARED_QUESTION, actor=stack.a.user, context=CONTEXT
                )
        lines = json_lines(stream)

        failed = [line for line in lines if line.get("event") == "query.failed"]
        assert failed, f"the failure was not logged at all: {[x.get('event') for x in lines]}"
        assert failed[0]["error_type"] == "RuntimeError"
        assert SENTINEL_IN_EXCEPTION not in str(failed[0]), (
            "the dependency's exception message reached the service's own log line"
        )

    async def test_the_audit_row_names_the_type_not_the_message(self, stack: Stack) -> None:
        """A `FAILED` query is audited; what it records is the exception class only.

        The audit trail is append-only and retained per policy, so anything written into a
        `detail` dict is written for years. A message is the one field of an exception that can
        contain an account id, a row value or a URL with a token in it.
        """
        stack.bedrock.fail_with = RuntimeError(f"boom {SENTINEL_IN_EXCEPTION}")
        with pytest.raises(RuntimeError):
            await stack.query_service.submit(
                question=SHARED_QUESTION, actor=stack.a.user, context=CONTEXT
            )

        details = stack.audit_details()
        assert SENTINEL_IN_EXCEPTION not in details
        assert "RuntimeError" in details, "the failure was not audited with its type"

    async def test_the_client_is_told_nothing_about_the_failure(
        self, api: Api, stack: Stack
    ) -> None:
        """The 500 envelope, fetched over a transport that returns it instead of re-raising.

        This is the only test in the suite needing `raise_app_exceptions=False`, so it builds
        its own client over the app the fixture already wired rather than changing the fixture
        for everyone. What it asserts is the part of an outage the *caller* sees: a fixed code
        and message, no exception class, no traceback, nothing to fingerprint the stack with.
        """
        client, as_user = api
        as_user(stack.a.user)
        stack.bedrock.fail_with = RuntimeError(f"boom {SENTINEL_IN_EXCEPTION}")

        transport = httpx.ASGITransport(
            app=client._transport.app,  # type: ignore[union-attr] # noqa: SLF001
            raise_app_exceptions=False,
        )
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as tolerant:
            response = await tolerant.post(
                f"{API}/queries",
                json={"question_text": SHARED_QUESTION},
                headers=csrf_headers(),
            )

        assert response.status_code >= 500, response.text
        assert SENTINEL_IN_EXCEPTION not in response.text
        assert "RuntimeError" not in response.text
        assert "Traceback" not in response.text
        assert response.json()["error"]["code"] == "internal_error"

    async def test_the_traceback_is_the_documented_exception(self, stack: Stack) -> None:
        """The operator log does keep the message, via `logger.exception`'s rendered traceback.

        Asserted rather than left implicit. If someone later removes `format_exc_info` from the
        processor chain this test fails and they will find out that tracebacks were deliberate;
        if someone adds a redaction pass over rendered tracebacks, it also fails and they will
        find out that this was the accepted boundary. Either way the decision gets re-made by a
        person instead of drifting.
        """
        from backend.core.logging import get_logger as get_bound_logger

        with captured_log_stream() as stream:
            try:
                raise RuntimeError(f"boom {SENTINEL_IN_EXCEPTION}")
            except RuntimeError:
                get_bound_logger(__name__).exception("unhandled.error", path="/api/v1/queries")
        rendered = stream.getvalue()

        assert "unhandled.error" in rendered
        assert SENTINEL_IN_EXCEPTION in rendered, (
            "a traceback no longer carries the exception message — if that was deliberate, "
            "delete this test and record the change in docs/RELEASE.md"
        )


class TestResponsesCarryNoInternalSecret:
    """Two fields that are not credentials but must not travel: the hash and the S3 key."""

    async def test_the_profile_carries_no_password_hash(
        self, api: Api, stack: Stack
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)
        response = await client.get(f"{API}/auth/me")
        assert response.status_code == 200, response.text
        body = response.json()
        assert "hashed_password" not in body
        assert stack.a.user.hashed_password not in response.text
        # `extra="forbid"` on the response model is what keeps it that way; assert the shape
        # rather than only the one field, so a field added by accident is caught too.
        assert set(body) == {
            "id",
            "email",
            "full_name",
            "role",
            "tenant_id",
            "is_active",
            "last_login_at",
            "created_at",
        }, sorted(body)

    async def test_a_document_response_carries_no_storage_key(
        self, api: Api, stack: Stack
    ) -> None:
        """`s3_key` is an internal locator, and it names the tenant's prefix.

        Leaking it does not grant access — the bucket is private and reads go through a
        presigned URL — but it hands an attacker the layout to aim a misconfigured-bucket or
        path-traversal attempt at, and it is not a field any client needs.
        """
        client, as_user = api
        as_user(stack.a.admin)
        listing = await client.get(f"{API}/documents")
        assert listing.status_code == 200, listing.text
        single = await client.get(f"{API}/documents/{stack.a.document.id}")
        assert single.status_code == 200, single.text

        for response in (listing, single):
            assert "s3_key" not in response.text
            assert stack.a.document.s3_key not in response.text

    async def test_the_sanitizer_leaves_a_non_secret_value_usable(self) -> None:
        """A closing sanity check on the layer everything above depends on.

        `sanitize_log_value` is applied to every field that is not redacted. If it mangled
        ordinary values, the log would be useless and the redaction tests would be measuring
        a stream nobody reads.
        """
        assert sanitize_log_value("fee-schedule-2026.pdf") == "fee-schedule-2026.pdf"
        assert sanitize_log_value(42) == 42
        assert sanitize_log_value(None) is None
        nested: dict[str, Any] = {"phase": "ingest", "chunks": [1, 2, 3]}
        assert sanitize_log_value(nested) == nested
