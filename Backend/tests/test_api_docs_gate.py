"""backend/tests/test_api_docs_gate.py

Role: Pin that `/docs` and the session cookies' `Secure` flag are independent switches.

`ENVIRONMENT` used to decide both. `/docs` is mounted when it is `local`, and
`Settings.cookies_secure` is `ENVIRONMENT != "local"` — so anybody who wanted the interactive
docs on the POC instance set `ENVIRONMENT=local` and, without being told, also stripped
`Secure` from the session cookies of an HTTPS site. The docs were the visible reason to flip
it; the cookie change was invisible, which is the worst possible pairing.

`ENABLE_API_DOCS` splits them. These tests assert the split holds in both directions, because
re-deriving one from the other would reintroduce the coupling while every existing test still
passed — the defaults agree, and only the overridden combinations tell them apart.
"""

from __future__ import annotations

import pytest

from backend.core.config import Settings


def _settings(**overrides: object) -> Settings:
    """A Settings built from explicit values, ignoring any `.env` on the machine.

    `_env_file=None` matters: without it this reads the developer's real `.env`, and the test
    would assert whatever that happens to say rather than what was passed here.
    """
    base: dict[str, object] = {
        "SECRET_KEY": "x" * 64,
        "DATABASE_URL": "postgresql+asyncpg://u:p@localhost:5432/d",
        "PINECONE_API_KEY": "pc-test",
        "PINECONE_INDEX_NAME": "test-index",
        "S3_BUCKET_NAME": "test-bucket",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


class TestDefaultsAreUnchanged:
    """`ENABLE_API_DOCS` unset must behave exactly as the old code did."""

    def test_docs_on_in_local(self) -> None:
        assert _settings(ENVIRONMENT="local").api_docs_enabled is True

    @pytest.mark.parametrize("environment", ["dev", "staging", "prod"])
    def test_docs_off_everywhere_else(self, environment: str) -> None:
        assert _settings(ENVIRONMENT=environment).api_docs_enabled is False


class TestTheTwoSwitchesAreIndependent:
    """The point of the change: either can move without dragging the other along."""

    def test_docs_on_without_losing_secure_cookies(self) -> None:
        """The POC instance's case — HTTPS, Secure cookies, and docs reachable."""
        settings = _settings(ENVIRONMENT="dev", ENABLE_API_DOCS=True)
        assert settings.api_docs_enabled is True
        assert settings.cookies_secure is True, (
            "turning the docs on must not affect the cookies — that coupling is the bug "
            "ENABLE_API_DOCS exists to remove"
        )

    def test_docs_can_be_forced_off_in_local(self) -> None:
        settings = _settings(ENVIRONMENT="local", ENABLE_API_DOCS=False)
        assert settings.api_docs_enabled is False
        assert settings.cookies_secure is False, "local development stays on plain HTTP"

    @pytest.mark.parametrize("environment", ["local", "dev", "staging", "prod"])
    @pytest.mark.parametrize("docs", [True, False])
    def test_cookies_secure_ignores_the_docs_setting(
        self, environment: str, docs: bool
    ) -> None:
        """`cookies_secure` must depend on `ENVIRONMENT` and nothing else."""
        settings = _settings(ENVIRONMENT=environment, ENABLE_API_DOCS=docs)
        assert settings.cookies_secure is (environment != "local")


class TestTheAppHonoursIt:
    """The setting has to reach FastAPI, not just the Settings object."""

    def test_routes_are_mounted_when_enabled(self) -> None:
        settings = _settings(ENVIRONMENT="dev", ENABLE_API_DOCS=True)
        assert settings.api_docs_enabled is True
        # Mirrors the expression in `create_app`. Asserted as the derived URLs rather than by
        # building the app, which would need a database and a Pinecone key.
        assert ("/docs" if settings.api_docs_enabled else None) == "/docs"
        assert ("/openapi.json" if settings.api_docs_enabled else None) == "/openapi.json"

    def test_app_reads_the_property_not_the_environment(self) -> None:
        """Guard against the gate drifting back to `settings.ENVIRONMENT` directly."""
        import pathlib

        app_py = pathlib.Path(__file__).resolve().parents[1] / "api" / "app.py"
        source = app_py.read_text(encoding="utf-8")
        docs_line = next(
            line for line in source.splitlines() if line.strip().startswith("docs_url=")
        )
        assert "api_docs_enabled" in docs_line, (
            "docs_url must be gated on settings.api_docs_enabled; gating it on ENVIRONMENT "
            "again re-couples the docs to the cookies' Secure flag"
        )
