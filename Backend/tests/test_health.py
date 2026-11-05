"""backend/tests/test_health.py

Role: Smoke tests for the app factory, probes, CORS and the logging sanitizers.
"""

from __future__ import annotations

import json

import pytest

# asyncio_mode = "auto" (pyproject.toml) collects async tests; no module-level mark, or
# the sync tests in this file get one too.


async def test_health_returns_expected_schema(client) -> None:
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert set(body) == {"status", "version", "timestamp"}
    assert body["version"]


async def test_health_echoes_request_id(client) -> None:
    response = await client.get("/health", headers={"X-Request-ID": "abc-123"})

    assert response.headers["X-Request-ID"] == "abc-123"


async def test_health_generates_request_id_when_absent(client) -> None:
    response = await client.get("/health")

    assert response.headers.get("X-Request-ID")


async def test_readiness_reports_database_state(client) -> None:
    """Without a database this must be 503 — never a cheerful 200."""
    response = await client.get("/readiness")

    assert response.status_code in (200, 503)
    body = response.json()
    if response.status_code == 200:
        assert body["status"] == "ready"
        assert body["checks"]["database"] == "ok"
    else:
        assert body["status"] == "not_ready"
        assert body["checks"]["database"] == "unavailable"


async def test_cors_allows_configured_origin(client) -> None:
    response = await client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    # The session is a cookie, so credentialed CORS is mandatory.
    assert response.headers["access-control-allow-credentials"] == "true"


async def test_cors_rejects_unknown_origin(client) -> None:
    response = await client.options(
        "/health",
        headers={
            "Origin": "https://evil.example.com",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers


async def test_unhandled_error_does_not_leak_internals() -> None:
    """A crash returns a generic 500 body — no exception text, no connection string."""
    import httpx
    from fastapi import APIRouter

    from backend.api.app import create_app

    app = create_app()
    router = APIRouter()

    @router.get("/boom")  # vapt-ok: test-only route, never registered on the real app
    async def boom() -> None:
        raise RuntimeError("connection string postgres://user:pw@host/db")

    app.include_router(router)

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        response = await ac.get("/boom")

    assert response.status_code == 500
    body = response.json()
    assert body == {
        "error": {"code": "internal_error", "message": "An unexpected error occurred."}
    }
    assert "postgres://" not in response.text
    assert "RuntimeError" not in response.text


class TestSessionDependency:
    """The gate asks for 50 concurrent requests with no errors and no leaked connections."""

    async def test_fifty_concurrent_requests_share_the_pool_cleanly(self) -> None:
        import asyncio

        import httpx

        from backend.api.app import create_app
        from backend.core.database import dispose_engine, get_engine

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as ac:
            responses = await asyncio.gather(
                *(ac.get("/readiness") for _ in range(50))
            )

        codes = {r.status_code for r in responses}
        # Every request must agree: all ready, or all not-ready. A mix means the pool
        # starved some requests, which is exactly the leak this check exists to catch.
        assert codes in ({200}, {503}), f"inconsistent readiness under load: {codes}"

        # Whatever the database answered, every session must have been returned.
        pool = get_engine().pool
        assert pool.checkedout() == 0, "session dependency leaked a connection"
        await dispose_engine()


class TestStructuredLogging:
    def test_log_line_is_one_json_object_with_required_fields(self, capsys) -> None:
        """CR/LF in a value must not become a second log line."""
        import structlog

        from backend.core.logging import bind_request_context, configure_logging

        configure_logging()
        bind_request_context(request_id="req-1", user_id="user-1", action="QUERY_SUBMIT")
        structlog.get_logger("test").info(
            "query_submitted", question="line one\nWARNING fake entry\r\nlevel=critical"
        )

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(lines) == 1, f"log injection produced {len(lines)} lines: {lines}"

        record = json.loads(lines[0])
        for field in ("request_id", "timestamp", "level", "event"):
            assert field in record, f"missing {field} in {record}"
        assert record["request_id"] == "req-1"
        assert "\n" not in record["question"]

    def test_third_party_stdlib_logs_are_json_too(self, capsys) -> None:
        """uvicorn and friends log through stdlib; a plain-text line breaks log shipping."""
        import logging

        from backend.core.logging import configure_logging

        configure_logging()
        logging.getLogger("uvicorn.error").warning("Application startup complete.")

        captured = capsys.readouterr()
        lines = [line for line in captured.out.splitlines() if line.strip()]
        assert len(lines) == 1

        record = json.loads(lines[0])  # would raise if uvicorn's own formatter won
        assert record["event"] == "Application startup complete."
        assert record["level"] == "warning"

    def test_uvicorn_access_log_is_disabled(self) -> None:
        """The middleware logs requests with a request_id; uvicorn's duplicate has none."""
        import logging

        from backend.core.logging import configure_logging

        configure_logging()

        assert logging.getLogger("uvicorn.access").disabled is True

    def test_configure_logging_is_idempotent(self, capsys) -> None:
        """A second call must not double every line — a real risk with stdlib handlers."""
        import structlog

        from backend.core.logging import configure_logging

        configure_logging()
        configure_logging()
        structlog.get_logger("test").info("once")

        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 1, f"logging configured twice emitted {len(lines)} lines"


class TestLogSanitizers:
    """The log-injection and secret-redaction controls, tested rather than assumed."""

    def test_crlf_is_stripped_from_values(self) -> None:
        from backend.core.logging import sanitize_log_value

        forged = "normal value\nWARNING forged log line"
        cleaned = sanitize_log_value(forged)

        assert "\n" not in cleaned
        assert "\r" not in cleaned
        assert "forged log line" in cleaned  # kept, but on the same line

    def test_control_characters_are_stripped(self) -> None:
        from backend.core.logging import sanitize_log_value

        assert "\x00" not in sanitize_log_value("null\x00byte")
        assert "\x1b" not in sanitize_log_value("esc\x1b[31mred")

    def test_long_values_are_truncated(self) -> None:
        from backend.core.logging import sanitize_log_value

        cleaned = sanitize_log_value("x" * 10_000)

        assert len(cleaned) < 10_000
        assert cleaned.endswith("[truncated]")

    def test_nested_structures_are_sanitized(self) -> None:
        from backend.core.logging import sanitize_log_value

        cleaned = sanitize_log_value({"doc": ["a\nb", {"name": "c\rd"}]})

        assert "\n" not in json.dumps(cleaned)
        assert "\r" not in json.dumps(cleaned)

    @pytest.mark.parametrize(
        "key",
        [
            "password",
            "Password",
            "secret_key",
            "access_token",
            "authorization",
            "api_key",
            "account_number",
            "customer_card",
        ],
    )
    def test_sensitive_keys_are_redacted(self, key: str) -> None:
        from backend.core.logging import REDACTED, _redact_and_sanitize

        result = _redact_and_sanitize(None, "info", {key: "hunter2", "safe": "kept"})

        assert result[key] == REDACTED
        assert result["safe"] == "kept"


class TestSettings:
    def test_rejects_sync_database_url(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from backend.core.config import Settings

        with pytest.raises(PydanticValidationError, match="asyncpg"):
            Settings(  # type: ignore[call-arg]
                DATABASE_URL="postgresql://u:p@localhost/db",
                SECRET_KEY="x" * 40,
                S3_BUCKET_NAME="b",
            )

    def test_rejects_short_secret_key(self) -> None:
        from pydantic import ValidationError as PydanticValidationError

        from backend.core.config import Settings

        with pytest.raises(PydanticValidationError, match="at least 32"):
            Settings(  # type: ignore[call-arg]
                DATABASE_URL="postgresql+asyncpg://u:p@localhost/db",
                SECRET_KEY="short",
                S3_BUCKET_NAME="b",
            )

    @pytest.mark.parametrize("missing", ["DATABASE_URL", "SECRET_KEY"])
    def test_missing_required_key_names_itself(self, monkeypatch, missing: str) -> None:
        """The gate: dropping a required key stops startup with a message naming it."""
        from pydantic import ValidationError as PydanticValidationError

        from backend.core.config import Settings

        supplied = {
            "DATABASE_URL": "postgresql+asyncpg://u:p@localhost/db",
            "SECRET_KEY": "x" * 40,
            "S3_BUCKET_NAME": "b",
        }
        del supplied[missing]
        # Both the process environment and .env would otherwise satisfy the key.
        monkeypatch.delenv(missing, raising=False)

        with pytest.raises(PydanticValidationError, match=missing):
            Settings(_env_file=None, **supplied)  # type: ignore[call-arg]

    def test_secret_key_is_not_printed_in_repr(self, settings) -> None:
        """A stray log of the settings object must not spill the signing key."""
        assert settings.SECRET_KEY.get_secret_value() not in repr(settings)

    def test_extension_allowlist_is_normalized(self, settings) -> None:
        assert settings.allowed_extensions == frozenset({"pdf", "docx"})

    def test_sync_url_derived_for_alembic(self, settings) -> None:
        assert settings.sync_database_url.startswith("postgresql://")
        assert "asyncpg" not in settings.sync_database_url
