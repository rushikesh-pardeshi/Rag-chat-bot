"""backend/api/app.py

Role: FastAPI application factory. Registers middleware, exception handlers, and route
modules. All wiring lives here — `main.py` only boots what this builds.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.api.routes import analytics as analytics_routes
from backend.api.routes import audit_logs as audit_routes
from backend.api.routes import auth as auth_routes
from backend.api.routes import documents as document_routes
from backend.api.routes import health
from backend.api.routes import queries as query_routes
from backend.api.routes import search_history as history_routes
from backend.api.routes import users as user_routes
from backend.core.config import Settings, get_settings
from backend.core.database import dispose_engine
from backend.core.exceptions import CSRFValidationError, DocQueryError
from backend.core.limiter import limiter
from backend.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)
from backend.core.security import verify_csrf_token
from backend.utils.vector_store import dispose_vector_store
from backend.workers.document_processor import drain_ingestion_pool

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

#: What a caller-supplied `X-Request-ID` may contain. Anything else is replaced with a fresh
#: UUID rather than echoed.
#:
#: The value is reflected into the *response* header of the same name, and uvicorn's httptools
#: implementation — the one `uvicorn[standard]` selects — serializes response headers by
#: concatenation with no validation (`httptools_impl.py`: `content.extend([name, b": ", value,
#: b"\r\n"])`). A CR/LF in this value therefore becomes a real injected header, or a second
#: response entirely: HTTP response splitting, on our own origin. Under the pure-Python `h11`
#: implementation the same value raises `LocalProtocolError` and fails the request instead, so
#: the observable damage depends on a deployment flag — which is exactly the kind of guarantee
#: that should not be left to one.
#:
#: The character set is what a correlation id legitimately needs: a UUID, a W3C trace id, or a
#: gateway's `req-…`. The length cap keeps a 5 KB header out of the log stream and off the
#: wire.
_SAFE_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")

#: Methods that change state and therefore require a CSRF token.
UNSAFE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Paths exempt from the CSRF check, and why each one is safe to exempt.
#:   * `/login` and `/register` — there is no session cookie yet, so there is nothing for a
#:     cross-site request to ride on. Requiring a token here would mean minting one for
#:     anonymous callers, which is a rate-limit target and buys nothing.
#:   * `/refresh` — protected instead by `SameSite=Strict` plus the refresh cookie's path
#:     scope, and by rotation: a replayed refresh revokes the whole family.
#: Everything else, including `/logout` and `/password`, is checked.
CSRF_EXEMPT_SUFFIXES = ("/auth/login", "/auth/register", "/auth/refresh")

#: Response headers set on every response. Held here rather than in a reverse proxy so the
#: guarantee travels with the application.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info(
        "application.startup",
        environment=settings.ENVIRONMENT,
        version=settings.APP_VERSION,
    )
    yield
    # Ingestion first, and before the engine goes: when `INGEST_LOCAL_CONCURRENCY` is set, the
    # documents being parsed and embedded right now are running in *this* process, and they
    # each hold a session. Disposing the engine under them would turn a clean stop into a row
    # stuck at `PROCESSING` plus a stack trace. `drain` waits a bounded time and then cancels,
    # so a wedged Textract poll cannot hold the shutdown open.
    drained = await drain_ingestion_pool()
    if drained:
        logger.info("application.ingestion_drained", documents=drained)
    # Both pools, in the order they were opened. `dispose_vector_store` closes the
    # Pinecone connection pool; skipping it leaks sockets on every reload, which under
    # `--reload` means a developer's process accumulates them all afternoon.
    await dispose_vector_store()
    await dispose_engine()
    logger.info("application.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging()

    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "AI-driven document search with full audit trails and source traceability."
        ),
        lifespan=lifespan,
        # Interactive docs are a development convenience, not a production surface. Gated on
        # `api_docs_enabled` rather than `ENVIRONMENT` directly: the latter also derives
        # `cookies_secure`, so turning the docs on used to mean quietly turning the session
        # cookies' `Secure` flag off. See `ENABLE_API_DOCS` in `backend/core/config.py`.
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
    )

    # SlowAPIMiddleware and the RateLimitExceeded handler both read the limiter off
    # app.state; without this every declared limit is silently a no-op.
    app.state.limiter = limiter

    _register_middleware(app, settings)
    _register_exception_handlers(app)
    _register_routes(app, settings)
    return app


def _register_middleware(app: FastAPI, settings: Settings) -> None:
    # -- Unhandled exceptions, turned into a response at the bottom of the stack ---------
    #
    # **Registered first on purpose, which makes it innermost.** `add_middleware` prepends, so
    # the last one registered is the outermost and the resulting stack is
    # `CORS -> SlowAPI -> csrf_protection -> request_context -> this -> router`.
    #
    # Starlette installs the `@app.exception_handler(Exception)` handler on
    # `ServerErrorMiddleware`, which wraps the *whole* application — outside every middleware
    # registered here. A 500 produced there has therefore been through none of them, and the
    # consequences compound:
    #
    #   * **No `Access-Control-Allow-Origin`.** Measured before this existed: a 401 carried
    #     `http://localhost:3000`, a 500 carried nothing. A browser cannot read a cross-origin
    #     response without that header, so it discards the 500 and reports a **CORS policy
    #     error** instead. The real failure — a database the API could not reach, a foreign key
    #     it violated — becomes invisible in the one place a frontend developer is looking, and
    #     the symptom accuses the one subsystem that is working. That is precisely how
    #     `/auth/refresh` returning 500 came to be investigated as a missing preflight.
    #   * **No `X-Request-Id`**, so the user reporting the 500 has no correlation id to quote
    #     and the traceback cannot be found in the log.
    #   * **No security headers**, and no `http.request` log line recording the outcome.
    #
    # Converting the exception to a response *here* fixes all four at once, because every
    # middleware above then treats it as an ordinary response and decorates it as usual. The
    # alternative — catching just below CORS — restores the CORS header alone and leaves the
    # 500 the only response in the app with no request id, which `test_error_cors.py` caught.
    #
    # `handle_unexpected` stays registered as the backstop for anything raised *above* this
    # point: the three middlewares around it, and CORS itself.
    @app.middleware("http")
    async def surface_unhandled_errors(
        request: Request, call_next: Callable[[Request], Awaitable[object]]
    ):
        try:
            return await call_next(request)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - re-raising puts it back outside the stack
            # Same shape and same log event as `handle_unexpected`, so a reader does not have
            # to know which of the two caught it. `logger.exception` keeps the traceback on
            # the server; the body stays generic because an exception message can name a
            # bucket, a model arn or the contents of a row.
            logger.exception("unhandled.error", path=request.url.path)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "An unexpected error occurred.",
                    }
                },
            )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[object]]
    ):
        """Give every request an id, log its outcome, and never leak the context."""
        request_id = _safe_request_id(request.headers.get(REQUEST_ID_HEADER))
        request.state.request_id = request_id
        bind_request_context(request_id=request_id)
        try:
            response = await call_next(request)  # type: ignore[misc]
            logger.info(
                "http.request",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,  # type: ignore[attr-defined]
            )
            response.headers[REQUEST_ID_HEADER] = request_id  # type: ignore[attr-defined]
            for header, value in SECURITY_HEADERS.items():
                response.headers.setdefault(header, value)  # type: ignore[attr-defined]
            if settings.cookies_secure:
                # Only over TLS. Sending HSTS from a plain-HTTP dev server would pin
                # localhost to https:// in the developer's browser for a year.
                response.headers.setdefault(  # type: ignore[attr-defined]
                    "Strict-Transport-Security",
                    "max-age=31536000; includeSubDomains",
                )
            return response
        finally:
            clear_request_context()

    @app.middleware("http")
    async def csrf_protection(
        request: Request, call_next: Callable[[Request], Awaitable[object]]
    ):
        """Double-submit CSRF check on every unsafe method.

        This is required *because* the JWT lives in an httpOnly cookie: the browser
        attaches it to cross-site requests automatically, so `SameSite=Strict` alone is the
        only thing standing between a malicious page and an authenticated POST — and
        SameSite has gaps (older browsers, some redirect flows). The token proves the
        request came from a page that could read our own cookie.

        If the token transport ever moves to an `Authorization` header this check becomes
        unnecessary. The two decisions are coupled; change them together
        (`.claude/rules/secure-coding-vapt.md`).
        """
        if request.method in UNSAFE_METHODS and not _is_csrf_exempt(request.url.path):
            cookie_token = request.cookies.get(settings.CSRF_COOKIE_NAME)
            header_token = request.headers.get(settings.CSRF_HEADER_NAME)
            if not verify_csrf_token(cookie_token, header_token):
                logger.warning(
                    "csrf.rejected",
                    method=request.method,
                    path=request.url.path,
                    has_cookie=bool(cookie_token),
                    has_header=bool(header_token),
                )
                exc = CSRFValidationError()
                return JSONResponse(
                    status_code=exc.status_code,
                    content={"error": {"code": exc.code, "message": exc.message}},
                )
        return await call_next(request)  # type: ignore[return-value]

    # Runs the limits declared by `@limiter.limit(...)` on individual routes and puts
    # `view_rate_limit` on request.state, which the 429 handler needs for `Retry-After`.
    # Without this middleware every decorator is a silent no-op.
    app.add_middleware(SlowAPIMiddleware)

    # CORS must allow credentials: the session travels as an httpOnly cookie, so a
    # wildcard origin is both invalid here and unsafe.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", settings.CSRF_HEADER_NAME, REQUEST_ID_HEADER],
        # `Content-Disposition` and the two export headers must be readable by the browser, or
        # the analytics page cannot name the downloaded file or warn that it was truncated —
        # a cross-origin response exposes no headers beyond the CORS-safelisted set by default.
        expose_headers=[
            REQUEST_ID_HEADER,
            "Content-Disposition",
            analytics_routes.ROWS_HEADER,
            analytics_routes.TRUNCATED_HEADER,
        ],
        max_age=600,
    )


def _safe_request_id(incoming: str | None) -> str:
    """The caller's correlation id if it is one, otherwise a fresh UUID.

    `fullmatch`, not `search`: a partial match would accept `abc\\r\\nSet-Cookie: x` because
    the safe pattern matches the `abc` at the front. That distinction is the whole control.

    A rejected value is replaced silently rather than 400'd. The header is a debugging
    convenience, and failing a legitimate request because a gateway prefixed something odd
    would trade a real outage for a cosmetic gain — while the replacement costs the caller
    only the ability to correlate on an id they chose.
    """
    if incoming and _SAFE_REQUEST_ID.fullmatch(incoming):
        return incoming
    return str(uuid.uuid4())


def _is_csrf_exempt(path: str) -> bool:
    """Suffix match, so the check is independent of the API version prefix."""
    return any(path.endswith(suffix) for suffix in CSRF_EXEMPT_SUFFIXES)


def _field_errors(
    errors: Sequence[Any], *, drop_first: bool
) -> list[dict[str, str]]:
    """Which fields failed and why — never the values that were submitted.

    `drop_first` strips FastAPI's source segment ("query", "body") from `loc`. A model
    validated directly has no such segment, and dropping one there would leave the field
    unnamed, which is the one piece of information the caller actually needs.
    """
    return [
        {
            "field": ".".join(
                str(part) for part in (err["loc"][1:] if drop_first else err["loc"])
            ),
            "reason": err["msg"],
        }
        for err in errors
    ]


def _validation_response(fields: list[dict[str, str]]) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "The request was not valid.",
                "fields": fields,
            }
        },
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DocQueryError)
    async def handle_domain_error(request: Request, exc: DocQueryError) -> JSONResponse:
        # Context goes to the log; the client sees only code and message.
        logger.warning(
            "domain.error",
            code=exc.code,
            path=request.url.path,
            **{k: str(v) for k, v in exc.context.items()},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": exc.code, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # FastAPI prefixes `loc` with the source ("query", "body"), which is noise to a
        # client that knows what it sent.
        return _validation_response(_field_errors(exc.errors(), drop_first=True))

    @app.exception_handler(PydanticValidationError)
    async def handle_model_validation_error(
        request: Request, exc: PydanticValidationError
    ) -> JSONResponse:
        """A filter model a route built by hand, rejected — still a 422, not a 500.

        Several read routes declare their query parameters individually and then re-validate
        them through one Pydantic filter, so that a cross-field rule ("`occurred_from` must
        not be later than `occurred_to`") lives in a single place instead of being repeated in
        every route that shares it. That construction happens *inside* the handler, so its
        failure is a plain `pydantic.ValidationError` rather than FastAPI's
        `RequestValidationError` — and without this handler it reached `handle_unexpected` and
        was reported as an internal error.

        A 500 there is worse than untidy. It tells the caller the *server* is broken when the
        *request* was wrong, and on a compliance read it hides the reason the schema went to
        the trouble of stating: a reversed date range returns nothing, and "nothing" reads as
        "no such activity" in an audit review.
        """
        return _validation_response(_field_errors(exc.errors(), drop_first=False))

    @app.exception_handler(RateLimitExceeded)
    async def handle_rate_limited(request: Request, exc: RateLimitExceeded) -> JSONResponse:
        """429 with `Retry-After`, in the same error envelope as everything else.

        slowapi's default handler returns a bare string body; overriding it keeps one
        response shape across the API so a client needs one parser.
        """
        logger.warning("rate_limit.exceeded", path=request.url.path, limit=str(exc.detail))
        response = JSONResponse(
            status_code=429,
            content={
                "error": {
                    "code": "rate_limited",
                    "message": "Too many requests. Please slow down and try again.",
                }
            },
        )
        # `view_rate_limit` is attached by SlowAPIMiddleware. `_inject_headers` is what
        # slowapi's own handler calls to add `Retry-After` and the `X-RateLimit-*` family;
        # reusing it is what keeps those headers correct when the body is customized.
        window = getattr(request.state, "view_rate_limit", None)
        if window is not None:
            return limiter._inject_headers(response, window)  # noqa: SLF001
        response.headers["Retry-After"] = "60"
        return response

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "http_error", "message": str(exc.detail)}},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled.error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred.",
                }
            },
        )


def _register_routes(app: FastAPI, settings: Settings) -> None:
    prefix = settings.API_V1_PREFIX

    # Probes stay unprefixed so infrastructure can reach them without knowing the
    # API version.
    app.include_router(health.router)
    app.include_router(health.router, prefix=prefix, include_in_schema=False)

    # Auth is mounted once, under the version prefix only: an unversioned /auth/login
    # would be a second door onto the same lock.
    app.include_router(auth_routes.router, prefix=prefix)

    # Everything below requires a session. Each router carries its own role dependency —
    # `AdminUser` on the management routes, `CurrentUser` on the rest — rather than being
    # gated here, so the requirement is visible at the endpoint a reader is looking at.
    app.include_router(document_routes.router, prefix=prefix)
    app.include_router(query_routes.router, prefix=prefix)
    # `/history` is the one authenticated router without a role dependency: both roles
    # reach it, and the service narrows the rows a Standard User sees to their own.
    app.include_router(history_routes.router, prefix=prefix)
    app.include_router(analytics_routes.router, prefix=prefix)
    app.include_router(audit_routes.router, prefix=prefix)
    # Read-only: `GET /users` and nothing else. A role-change route is F-05 and needs its own
    # decision, so this router deliberately has no unsafe method.
    app.include_router(user_routes.router, prefix=prefix)
