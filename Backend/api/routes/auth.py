"""backend/api/routes/auth.py

Role: HTTP routes for authentication. Validate a Pydantic schema, call one service method,
shape cookies and a DTO. No SQL and no business branching here.

Cookie policy, in one place because the pieces only work together:

  * access and refresh tokens are `httpOnly` — JavaScript cannot read them, so an XSS
    bug cannot exfiltrate a session;
  * `Secure` outside local development, `SameSite=Strict` always, so the browser will not
    attach them to a cross-site request in the first place;
  * because the browser attaches them automatically, a double-submit CSRF token is
    mandatory — that is `csrf_middleware` in `backend/api/app.py`. The CSRF cookie is the
    one cookie here that is deliberately *readable*, since the client has to echo it in a
    header. It carries no authority on its own.
  * the refresh cookie is scoped to the refresh path: it is not sent on ordinary API calls,
    so a leak on any other endpoint cannot include it.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter, Request, Response, status

from backend.api.dependencies.auth import (
    AuthServiceDep,
    CurrentUser,
    OptionalUser,
    RequestCtx,
)
from backend.core.config import Settings, get_settings
from backend.core.exceptions import AuthenticationError
from backend.core.limiter import limiter
from backend.schemas.auth import (
    LoginRequest,
    MessageResponse,
    PasswordChangeRequest,
    RegisterRequest,
    RegisterResponse,
    SessionResponse,
    UserProfile,
)
from backend.services.auth_service import IssuedSession

router = APIRouter(prefix="/auth", tags=["auth"])

#: Per-IP limit on the credential-accepting endpoints, from `AUTH_RATE_LIMIT` (10/minute).
#: Read once at import: slowapi needs a literal or a callable at decoration time, and the
#: value must not change under a running process.
AUTH_LIMIT: Final = get_settings().AUTH_RATE_LIMIT

#: Path the refresh cookie is limited to. Must match the route below.
REFRESH_COOKIE_PATH: Final = "/api/v1/auth/refresh"


def _set_session_cookies(
    response: Response, session: IssuedSession, settings: Settings
) -> None:
    common = {
        "secure": settings.cookies_secure,
        "samesite": settings.COOKIE_SAMESITE,
        "domain": settings.COOKIE_DOMAIN,
    }
    response.set_cookie(
        settings.ACCESS_COOKIE_NAME,
        session.access_token,
        max_age=session.expires_in_seconds,
        httponly=True,
        path="/",
        **common,
    )
    response.set_cookie(
        settings.REFRESH_COOKIE_NAME,
        session.refresh_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=True,
        path=REFRESH_COOKIE_PATH,
        **common,
    )
    # Readable on purpose: the client must copy it into the X-CSRF-Token header. Its value
    # is worthless to an attacker who cannot also read it, which is exactly what the
    # same-origin policy prevents.
    response.set_cookie(
        settings.CSRF_COOKIE_NAME,
        session.csrf_token,
        max_age=settings.REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60,
        httponly=False,
        path="/",
        **common,
    )


def _clear_session_cookies(response: Response, settings: Settings) -> None:
    for name, path in (
        (settings.ACCESS_COOKIE_NAME, "/"),
        (settings.REFRESH_COOKIE_NAME, REFRESH_COOKIE_PATH),
        (settings.CSRF_COOKIE_NAME, "/"),
    ):
        # Same path and domain as when set — a delete with a different path leaves the
        # original cookie in place and the session apparently still alive.
        response.delete_cookie(
            name,
            path=path,
            domain=settings.COOKIE_DOMAIN,
            secure=settings.cookies_secure,
            samesite=settings.COOKIE_SAMESITE,
        )


def _profile(session: IssuedSession) -> UserProfile:
    return UserProfile(
        id=session.user_id,
        email=session.email,
        full_name=session.full_name,
        role=session.role,
        tenant_id=session.tenant_id,
        is_active=True,
    )


@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a Standard User account",
)
@limiter.limit(AUTH_LIMIT)
async def register(
    request: Request,
    payload: RegisterRequest,
    response: Response,
    service: AuthServiceDep,
    context: RequestCtx,
) -> RegisterResponse:
    """Register and return the profile only — no session, no password material.

    Registration does not log the user in: an account may need approval, and issuing a
    session here would make that impossible to add later without breaking clients.

    `response` is unused by the body and still required: `@limiter.limit` writes the
    `X-RateLimit-*` headers through a parameter named exactly `response`, and raises
    ``parameter `response` must be an instance of starlette.responses.Response`` when it is
    absent — *after* the account is created, so the row rolls back and the caller sees a 500
    on the one path that was supposed to succeed. Every rate-limited route in this file
    takes it for that reason; the others also happened to need it for cookies.
    """
    user = await service.register(
        email=payload.email,
        password=payload.password,
        full_name=payload.full_name,
        context=context,
    )
    return RegisterResponse(user=UserProfile.model_validate(user))


@router.post(
    "/login", response_model=SessionResponse, summary="Sign in and receive session cookies"
)
@limiter.limit(AUTH_LIMIT)
async def login(
    request: Request,
    payload: LoginRequest,
    response: Response,
    service: AuthServiceDep,
    context: RequestCtx,
) -> SessionResponse:
    session = await service.authenticate(
        email=payload.email, password=payload.password, context=context
    )
    _set_session_cookies(response, session, get_settings())
    return SessionResponse(
        user=_profile(session), expires_in_seconds=session.expires_in_seconds
    )


@router.post(
    "/refresh", response_model=SessionResponse, summary="Rotate the session cookies"
)
@limiter.limit(AUTH_LIMIT)
async def refresh(
    request: Request,
    response: Response,
    service: AuthServiceDep,
    context: RequestCtx,
) -> SessionResponse:
    """Exchange the refresh cookie for a new pair. The old refresh token is revoked.

    A missing cookie is a 401 rather than a 400: from the client's point of view there is
    no session, which is the same situation as an expired one.
    """
    settings = get_settings()
    token = request.cookies.get(settings.REFRESH_COOKIE_NAME)
    if not token:
        raise AuthenticationError()
    session = await service.refresh(refresh_token=token, context=context)
    _set_session_cookies(response, session, settings)
    return SessionResponse(
        user=_profile(session), expires_in_seconds=session.expires_in_seconds
    )


@router.post("/logout", response_model=MessageResponse, summary="End the session")
async def logout(
    request: Request,
    response: Response,
    user: OptionalUser,
    service: AuthServiceDep,
    context: RequestCtx,
) -> MessageResponse:
    """Revoke the refresh token and clear the cookies.

    Uses `OptionalUser`: logging out with an already-expired access token must still clear
    the browser's cookies, and returning 401 here would leave a dead session visible in
    the UI.
    """
    settings = get_settings()
    await service.logout(
        refresh_token=request.cookies.get(settings.REFRESH_COOKIE_NAME),
        user=user,
        context=context,
    )
    _clear_session_cookies(response, settings)
    return MessageResponse(message="Signed out.")


@router.get("/me", response_model=UserProfile, summary="The signed-in user's profile")
async def me(user: CurrentUser) -> UserProfile:
    """Identity as the server sees it. The frontend's source of truth for role display."""
    return UserProfile.model_validate(user)


@router.post(
    "/password",
    response_model=MessageResponse,
    summary="Change the signed-in user's password",
)
@limiter.limit(AUTH_LIMIT)
async def change_password(
    request: Request,
    payload: PasswordChangeRequest,
    response: Response,
    user: CurrentUser,
    service: AuthServiceDep,
    context: RequestCtx,
) -> MessageResponse:
    """Change the password and end every session, including this one."""
    await service.change_password(
        user=user,
        current_password=payload.current_password,
        new_password=payload.new_password,
        context=context,
    )
    _clear_session_cookies(response, get_settings())
    return MessageResponse(message="Password changed. Please sign in again.")
