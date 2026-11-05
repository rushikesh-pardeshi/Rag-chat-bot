"""backend/api/dependencies/auth.py

Role: Resolve the current user and tenant from the httpOnly session cookie, and enforce
role. This is the server-side authorization boundary — the frontend's role check is
cosmetic and is assumed bypassed.

Three things this file is careful about:

  * **The token is the only source of identity.** `tenant_id` and `role` come from signed
    claims, never from a header, query parameter or body. A caller who could name their own
    tenant would have read access to every tenant.
  * **The database is re-checked.** A valid token whose user is now disabled, deleted, or
    demoted must not authorize anything for up to 15 minutes, so the row is loaded and its
    current `role` and `is_active` win over the claim.
  * **Denials are audited.** An access-control failure that leaves no trace is itself an
    audit gap (`.claude/rules/rbac-enforcement.md`).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from backend.api.dependencies.database import DbSession
from backend.core.config import get_settings
from backend.core.exceptions import AuthenticationError, UnauthorizedAccess
from backend.core.logging import bind_actor_context
from backend.core.security import decode_access_token
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.user import User, UserRole
from backend.repositories.audit_repository import AuditRepository
from backend.repositories.auth_repository import AuthRepository
from backend.repositories.user_repository import UserRepository
from backend.services.audit_service import AuditService
from backend.services.auth_service import AuthService, RequestContext


# -- Wiring ------------------------------------------------------------------
def get_user_repository(session: DbSession) -> UserRepository:
    return UserRepository(session)


def get_auth_repository(session: DbSession) -> AuthRepository:
    return AuthRepository(session)


def get_audit_repository(session: DbSession) -> AuditRepository:
    return AuditRepository(session)


def get_audit_service(
    repository: Annotated[AuditRepository, Depends(get_audit_repository)],
) -> AuditService:
    return AuditService(repository)


def get_auth_service(
    users: Annotated[UserRepository, Depends(get_user_repository)],
    tokens: Annotated[AuthRepository, Depends(get_auth_repository)],
    audit: Annotated[AuditService, Depends(get_audit_service)],
) -> AuthService:
    return AuthService(users, tokens, audit)


def get_request_context(request: Request) -> RequestContext:
    """Source address and user-agent for audit rows.

    `request.client.host` is the peer address. Behind a proxy that is the proxy, and
    `X-Forwarded-For` is *not* read here: a client can set that header freely, and trusting
    it would let an attacker write any address they like into the audit trail. Resolving
    the real client IP is the reverse proxy's job (`--proxy-headers` with trusted hosts).
    """
    return RequestContext(
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
    )


RequestCtx = Annotated[RequestContext, Depends(get_request_context)]
AuthServiceDep = Annotated[AuthService, Depends(get_auth_service)]
AuditServiceDep = Annotated[AuditService, Depends(get_audit_service)]


# -- Identity ----------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TenantContext:
    """The tenant every query in this request must filter on."""

    tenant_id: uuid.UUID


async def get_current_user(
    request: Request,
    users: Annotated[UserRepository, Depends(get_user_repository)],
) -> User:
    """The authenticated user, or 401.

    Raises `AuthenticationError` for a missing, malformed, expired or wrong-type cookie,
    and for a valid token whose user has since been disabled or removed — all with the same
    message, so the caller cannot tell which.
    """
    settings = get_settings()
    token = request.cookies.get(settings.ACCESS_COOKIE_NAME)
    if not token:
        raise AuthenticationError()

    claims = decode_access_token(token)
    try:
        user_id = uuid.UUID(claims["sub"])
        tenant_id = uuid.UUID(claims["tenant_id"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AuthenticationError("Invalid authentication token.") from exc

    # Tenant-scoped lookup: a token whose `tenant_id` was somehow swapped still resolves
    # nothing, because the pair has to match a real row.
    user = await users.get_by_id(user_id, tenant_id)
    if user is None or not user.is_active:
        raise AuthenticationError("Session is no longer valid. Please sign in again.")

    # Correlate every subsequent log line in this request with the actor. The email is
    # sanitized by the logging processor before it reaches a line.
    bind_actor_context(user_id=str(user.id), tenant_id=str(user.tenant_id))
    request.state.current_user = user
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


async def get_optional_user(
    request: Request,
    users: Annotated[UserRepository, Depends(get_user_repository)],
) -> User | None:
    """Like `get_current_user` but returns None instead of raising.

    Only for endpoints that behave differently when signed in rather than refusing —
    `POST /auth/logout` is the case: it must succeed for an expired session too.
    """
    try:
        return await get_current_user(request, users)
    except AuthenticationError:
        return None


OptionalUser = Annotated[User | None, Depends(get_optional_user)]


async def get_tenant_context(user: CurrentUser) -> TenantContext:
    """Tenant scope for the request. Derived from the user row, never from the request."""
    return TenantContext(tenant_id=user.tenant_id)


TenantCtx = Annotated[TenantContext, Depends(get_tenant_context)]


# -- Role enforcement --------------------------------------------------------
async def require_admin(
    request: Request,
    user: CurrentUser,
    audit: AuditServiceDep,
    context: RequestCtx,
) -> User:
    """403 unless the caller is an Administrator, and audit the denial.

    Compliance officers hold ADMINISTRATOR; there is no third role to check for
    (`.claude/rules/rbac-enforcement.md`). Role is read from the freshly loaded row, so a
    demotion takes effect immediately rather than when the token expires.
    """
    if user.role is not UserRole.ADMINISTRATOR:
        await audit.log(
            tenant_id=user.tenant_id,
            user_id=user.id,
            action=AuditAction.ACCESS_DENIED,
            object_type=AuditObjectType.USER,
            object_id=str(user.id),
            outcome=AuditOutcome.DENIED,
            actor_email=user.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={
                "reason": "administrator_required",
                # Path and method only — never the body, which may hold a document or query.
                "path": request.url.path,
                "method": request.method,
                "role": user.role.value,
            },
        )
        raise UnauthorizedAccess()
    return user


AdminUser = Annotated[User, Depends(require_admin)]


async def audit_denied(
    *,
    audit: AuditService,
    user: User,
    context: RequestContext,
    action: AuditAction,
    object_type: AuditObjectType,
    object_id: str,
    reason: str,
) -> None:
    """Record a denial raised inside a service rather than by a dependency.

    Exported so cross-tenant 404s — which are authorization failures wearing a 404 —
    still produce an `outcome='DENIED'` row.
    """
    await audit.log(
        tenant_id=user.tenant_id,
        user_id=user.id,
        action=action,
        object_type=object_type,
        object_id=object_id,
        outcome=AuditOutcome.DENIED,
        actor_email=user.email,
        ip_address=context.ip_address,
        user_agent=context.user_agent,
        detail={"reason": reason},
    )
