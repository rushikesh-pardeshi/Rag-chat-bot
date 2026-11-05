"""backend/services/auth_service.py

Role: Business rules for authentication — registration, login, refresh rotation, logout,
password change. Owns lockout policy and the audit call for every decision.

Returns plain dataclasses, never HTTP responses: the route decides cookies and status
codes. That split is what lets `test_auth_service.py` assert on lockout and rotation
without a running server.

Two invariants this file exists to hold:
  * `role` and `tenant_id` come from the server, never from the caller. A registration
    body that contains `role` is rejected by the schema and would be ignored here anyway.
  * every outcome — success, failure, denial — produces an audit row before the caller
    hears the answer.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Final

from backend.core.config import get_settings
from backend.core.exceptions import (
    AccountLocked,
    AuthenticationError,
    EmailAlreadyRegistered,
    InvalidCredentials,
    UnauthorizedAccess,
    UserNotFound,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.core.security import (
    MAX_PASSWORD_BYTES,
    MIN_PASSWORD_LENGTH,
    TOKEN_TYPE_REFRESH,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    dummy_verify,
    generate_csrf_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.user import User, UserRole
from backend.repositories.auth_repository import AuthRepository, RefreshTokenReuse
from backend.repositories.user_repository import UserRepository
from backend.services.audit_service import AuditService
from backend.utils.ids import utc_now

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Where a request came from. Passed explicitly so the service never reads a Request."""

    ip_address: str | None = None
    user_agent: str | None = None


@dataclass(frozen=True, slots=True)
class IssuedSession:
    """What a successful authentication produces. The route turns this into cookies."""

    access_token: str
    refresh_token: str
    csrf_token: str
    expires_in_seconds: int
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    role: UserRole
    full_name: str | None = None


#: Most accounts returned in one page of `list_users`, whatever a caller asks for.
#:
#: A cap and not a default: `?limit=100000` would otherwise be a way to pull every account in
#: the tenant in one request. Fifty rows is a screenful, and the pager fetches the next one.
MAX_USER_PAGE: Final = 200


@dataclass(frozen=True, slots=True)
class UserPage:
    """One page of a tenant's accounts, plus the total behind it.

    Carries `User` entities. Mapping them to a DTO is the schema layer's job — this service
    does not know which of a user's columns a route may publish, and `hashed_password` staying
    out of the response is enforced by `TenantUser` having no such field rather than by
    anything here.
    """

    users: tuple[User, ...]
    total: int
    limit: int
    offset: int
    #: Administrators in the tenant, across every page. The UI needs it to disable the demote
    #: control on the last one; `set_role` refuses it either way.
    admins: int = 0


class AuthService:
    def __init__(
        self,
        users: UserRepository,
        tokens: AuthRepository,
        audit: AuditService,
    ) -> None:
        self._users = users
        self._tokens = tokens
        self._audit = audit

    # -- Registration --------------------------------------------------------
    async def register(
        self,
        *,
        email: str,
        password: str,
        full_name: str | None,
        context: RequestContext,
        tenant_id: uuid.UUID | None = None,
    ) -> User:
        """Create a Standard User.

        The role is not a parameter a caller can influence: every self-registration is a
        Standard User, and promotion is a separate audited operation. `tenant_id` is
        resolved from the email domain unless an Administrator supplied one.
        """
        # Account lifecycle events are audited as ROLE_CHANGE with an `event` detail:
        # the action enum is fixed by the compliance schema, and adding a member is a
        # schema decision rather than something to slip in here. `detail.event`
        # distinguishes register / password_change / promotion within it.
        email = _normalize_email(email)
        _validate_password(password)

        resolved_tenant = tenant_id or await self._resolve_tenant_for_email(email)
        existing = await self._users.get_by_email(email, resolved_tenant)
        if existing is not None:
            await self._audit.log(
                tenant_id=resolved_tenant,
                action=AuditAction.ROLE_CHANGE,
                object_type=AuditObjectType.USER,
                object_id=email,
                outcome=AuditOutcome.FAILURE,
                actor_email=email,
                ip_address=context.ip_address,
                user_agent=context.user_agent,
                detail={"reason": "duplicate_email"},
            )
            raise EmailAlreadyRegistered()

        user = await self._users.create(
            tenant_id=resolved_tenant,
            email=email,
            hashed_password=hash_password(password),
            role=UserRole.STANDARD_USER,
            full_name=full_name,
        )
        await self._audit.log(
            tenant_id=resolved_tenant,
            user_id=user.id,
            action=AuditAction.ROLE_CHANGE,
            object_type=AuditObjectType.USER,
            object_id=str(user.id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={"event": "register", "role": UserRole.STANDARD_USER.value},
        )
        return user

    async def _resolve_tenant_for_email(self, email: str) -> uuid.UUID:
        domain = email.rsplit("@", 1)[-1]
        tenant = await self._users.get_tenant_by_domain(domain)
        if tenant is None or not tenant.is_active:
            # Deliberately not "no tenant for example.com": that would turn registration
            # into a directory of which organizations use the system.
            raise ValidationError("Registration is not open for that email domain.")
        return tenant.id

    # -- Login ---------------------------------------------------------------
    async def authenticate(
        self, *, email: str, password: str, context: RequestContext
    ) -> IssuedSession:
        """Verify credentials and issue a session.

        Raises `InvalidCredentials` for an unknown email, a wrong password, and a disabled
        account alike — the caller must not be able to tell those apart. `AccountLocked` is
        the one exception: telling a locked-out user to wait is worth more than the small
        amount it reveals, and it only fires after the password already failed repeatedly.
        """
        settings = get_settings()
        email = _normalize_email(email)
        user = await self._users.get_by_email_any_tenant(email)

        if user is None:
            # Equalize timing so a 404-shaped fast path does not enumerate accounts.
            dummy_verify()
            await self._audit_login_failure(email, context, reason="unknown_email")
            raise InvalidCredentials()

        if user.locked_until is not None and user.locked_until > utc_now():
            await self._audit_login_failure(
                email, context, reason="locked", user=user
            )
            raise AccountLocked()

        if not user.is_active:
            dummy_verify()
            await self._audit_login_failure(email, context, reason="inactive", user=user)
            raise InvalidCredentials()

        if not verify_password(password, user.hashed_password):
            attempts = await self._users.record_failed_login(
                user.id, locked_until=user.locked_until
            )
            if attempts >= settings.MAX_FAILED_LOGIN_ATTEMPTS:
                await self._users.apply_lockout(
                    user.id,
                    utc_now() + timedelta(minutes=settings.ACCOUNT_LOCKOUT_MINUTES),
                )
            await self._audit_login_failure(
                email,
                context,
                reason="bad_password",
                user=user,
                detail={"failed_attempts": attempts},
            )
            raise InvalidCredentials()

        await self._users.reset_failed_logins(user.id, last_login_at=utc_now())
        session = await self._issue_session(user)
        await self._audit.log(
            tenant_id=user.tenant_id,
            user_id=user.id,
            action=AuditAction.LOGIN_SUCCESS,
            object_type=AuditObjectType.SESSION,
            object_id=str(user.id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
        )
        return session

    async def _audit_login_failure(
        self,
        email: str,
        context: RequestContext,
        *,
        reason: str,
        user: User | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Record a failed attempt.

        An unknown email has no tenant, and `audit_log.tenant_id` is NOT NULL with a FK —
        so there is nothing to chain it to. Those attempts are logged to the application
        log instead, where they still reach the SIEM, and the audit table keeps its
        referential guarantee. Every attempt against a *known* account is audited.
        """
        payload = {"reason": reason, **(detail or {})}
        if user is None:
            logger.warning(
                "auth.login_failed_unknown_account",
                email=email,
                reason=reason,
                ip_address=context.ip_address,
            )
            return
        await self._audit.log(
            tenant_id=user.tenant_id,
            user_id=user.id,
            action=AuditAction.LOGIN_FAILURE,
            object_type=AuditObjectType.SESSION,
            object_id=str(user.id),
            outcome=AuditOutcome.FAILURE,
            actor_email=email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail=payload,
        )

    async def _issue_session(self, user: User) -> IssuedSession:
        settings = get_settings()
        access = create_access_token(
            user_id=str(user.id), tenant_id=str(user.tenant_id), role=user.role.value
        )
        refresh, jti = create_refresh_token(
            user_id=str(user.id), tenant_id=str(user.tenant_id)
        )
        await self._tokens.store(
            token_id=uuid.UUID(bytes=_jti_to_uuid_bytes(jti)),
            tenant_id=user.tenant_id,
            user_id=user.id,
            token_hash=hash_refresh_token(refresh),
            expires_at=utc_now() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        return IssuedSession(
            access_token=access,
            refresh_token=refresh,
            csrf_token=generate_csrf_token(),
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            tenant_id=user.tenant_id,
            email=user.email,
            role=user.role,
            full_name=user.full_name,
        )

    # -- Refresh -------------------------------------------------------------
    async def refresh(
        self, *, refresh_token: str, context: RequestContext
    ) -> IssuedSession:
        """Rotate a refresh token.

        The presented token is revoked and a new one issued in the same transaction. A
        token that is already revoked means either a replay or a stolen cookie, and the
        response to both is the same: revoke every session for that user and make them log
        in again.
        """
        claims = decode_access_token(refresh_token, expected_type=TOKEN_TYPE_REFRESH)
        try:
            user_id = uuid.UUID(claims["sub"])
            tenant_id = uuid.UUID(claims["tenant_id"])
            token_id = uuid.UUID(bytes=_jti_to_uuid_bytes(claims["jti"]))
        except (ValueError, TypeError, UnicodeEncodeError) as exc:
            # `decode_access_token` has already proved the signature and that these claims are
            # present and non-empty — it does not prove they parse. A signed token whose `sub`
            # is not a UUID can only come from a build that issued a different claim shape, so
            # in practice this is a cookie outliving a format change rather than an attack.
            #
            # Guarded anyway, because the alternative is a 500 on an unauthenticated endpoint
            # whose whole input is a cookie. "Your session is not usable" is a 401, and a 500
            # says the opposite: that the request was fine and the server is broken. The
            # client's retry logic reads the two the same way here, which is exactly why the
            # difference would otherwise go unnoticed in the logs.
            raise AuthenticationError(
                "Session is no longer valid. Please sign in again."
            ) from exc

        # Resolve the subject **before** anything writes an audit row, because
        # `audit_log.tenant_id` is `ForeignKey("tenant.id")` and the only tenant id in hand so
        # far came out of the cookie. Auditing against an unverified claim is how this endpoint
        # answered 500 instead of 401: a refresh cookie naming a tenant that no longer exists
        # reached `audit.log`, `AuditRepository.append` flushed the INSERT, PostgreSQL refused
        # the foreign key, and `IntegrityError` left the request as an unhandled exception.
        #
        # That is not an exotic case on a long-lived deployment — it is what a browser holding
        # its cookies through a database reset or re-seed does on its next page load. The
        # signature still verifies, because `SECRET_KEY` did not change.
        user = await self._users.get_by_id(user_id, tenant_id)
        if user is None:
            # Still revoked, as the pre-reorder code did when it reached its own `user is
            # None` check: an orphaned `refresh_token` row for a subject that no longer
            # resolves is unusable but should not be left live. This is an UPDATE filtered on
            # `user_id`, so unlike the audit insert below it touches no foreign key and is
            # safe to issue for an id this database may not have.
            await self._tokens.revoke_all_for_user(user_id)
            # There is deliberately *no* audit row: `tenant_id` has nothing valid to point at
            # — writing one is the bug this ordering fixes — and `user_id` would name nobody.
            # The application log records it instead: ids only, no token material.
            logger.warning(
                "auth.refresh_unknown_subject",
                user_id=str(user_id),
                tenant_id=str(tenant_id),
            )
            raise AuthenticationError("Session is no longer valid. Please sign in again.")

        stored = await self._tokens.get_by_hash(hash_refresh_token(refresh_token))
        if stored is None or stored.revoked_at is not None:
            # Signature was valid but the row is gone or spent — treat as compromise.
            await self._tokens.revoke_all_for_user(user.id)
            await self._audit.log(
                # Both from the database now, not from the token. `user.tenant_id` is a row
                # that exists, which is what makes this insert safe.
                tenant_id=user.tenant_id,
                user_id=user.id,
                action=AuditAction.ACCESS_DENIED,
                object_type=AuditObjectType.SESSION,
                object_id=str(token_id),
                outcome=AuditOutcome.DENIED,
                # The replayed cookie does identify someone: the lookup above resolved it. The
                # trail naming them beats the `unknown` this used to record.
                actor_email=user.email,
                ip_address=context.ip_address,
                user_agent=context.user_agent,
                detail={"reason": "refresh_token_reuse", "sessions_revoked": True},
            )
            raise AuthenticationError("Session is no longer valid. Please sign in again.")

        if not user.is_active:
            await self._tokens.revoke_all_for_user(user.id)
            raise AuthenticationError("Session is no longer valid. Please sign in again.")

        settings = get_settings()
        new_refresh, new_jti = create_refresh_token(
            user_id=str(user.id), tenant_id=str(user.tenant_id)
        )
        try:
            await self._tokens.rotate(
                old_token_id=stored.id,
                new_token_id=uuid.UUID(bytes=_jti_to_uuid_bytes(new_jti)),
                tenant_id=user.tenant_id,
                user_id=user.id,
                new_token_hash=hash_refresh_token(new_refresh),
                new_expires_at=utc_now()
                + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            )
        except RefreshTokenReuse as exc:
            # Lost a race with a parallel refresh of the same token.
            await self._tokens.revoke_all_for_user(user.id)
            raise AuthenticationError(
                "Session is no longer valid. Please sign in again."
            ) from exc

        return IssuedSession(
            access_token=create_access_token(
                user_id=str(user.id),
                tenant_id=str(user.tenant_id),
                role=user.role.value,
            ),
            refresh_token=new_refresh,
            csrf_token=generate_csrf_token(),
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.id,
            tenant_id=user.tenant_id,
            email=user.email,
            role=user.role,
            full_name=user.full_name,
        )

    # -- Logout --------------------------------------------------------------
    async def logout(
        self,
        *,
        refresh_token: str | None,
        user: User | None,
        context: RequestContext,
    ) -> None:
        """Revoke the presented session. Idempotent — logging out twice is not an error."""
        if refresh_token:
            stored = await self._tokens.get_by_hash(hash_refresh_token(refresh_token))
            if stored is not None:
                await self._tokens.revoke(stored.id)
        if user is not None:
            await self._audit.log(
                tenant_id=user.tenant_id,
                user_id=user.id,
                action=AuditAction.LOGOUT,
                object_type=AuditObjectType.SESSION,
                object_id=str(user.id),
                outcome=AuditOutcome.SUCCESS,
                actor_email=user.email,
                ip_address=context.ip_address,
                user_agent=context.user_agent,
            )

    # -- Password change -----------------------------------------------------
    async def change_password(
        self,
        *,
        user: User,
        current_password: str,
        new_password: str,
        context: RequestContext,
    ) -> None:
        """Verify the current password, then set the new one and end every other session."""
        _validate_password(new_password)
        if not verify_password(current_password, user.hashed_password):
            await self._audit.log(
                tenant_id=user.tenant_id,
                user_id=user.id,
                action=AuditAction.ACCESS_DENIED,
                object_type=AuditObjectType.USER,
                object_id=str(user.id),
                outcome=AuditOutcome.DENIED,
                actor_email=user.email,
                ip_address=context.ip_address,
                user_agent=context.user_agent,
                detail={"reason": "password_change_wrong_current"},
            )
            raise InvalidCredentials("Current password is incorrect.")
        if verify_password(new_password, user.hashed_password):
            raise ValidationError("The new password must differ from the current one.")

        await self._users.set_password(user, hash_password(new_password))
        # A password change is a response to suspected compromise as often as not; every
        # existing refresh token has to die with it.
        revoked = await self._tokens.revoke_all_for_user(user.id)
        await self._audit.log(
            tenant_id=user.tenant_id,
            user_id=user.id,
            action=AuditAction.ROLE_CHANGE,
            object_type=AuditObjectType.USER,
            object_id=str(user.id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=user.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={"event": "password_change", "sessions_revoked": revoked},
        )

    # -- User administration -------------------------------------------------
    async def list_users(
        self,
        *,
        actor: User,
        limit: int = 50,
        offset: int = 0,
        context: RequestContext,
    ) -> UserPage:
        """Everyone in the actor's tenant, newest first. Administrator-only.

        Two independent checks, per `rbac-enforcement.md`: the actor must hold the
        Administrator role, and the query is scoped to `actor.tenant_id`. An Administrator in
        tenant A asking for users gets tenant A's users; there is no parameter that could ask
        for anyone else's, which is why the tenant is read from the actor rather than accepted.

        A denial is audited here as well as refused. The success path is not audited yet —
        `AuditAction` has no member for reading the user list, and adding one is an enum change
        on a native PostgreSQL type, which is a schema change and needs approval
        (`workflow.md`). Recording it as `AUDIT_VIEW` or `HISTORY_VIEW` would put a false
        action in an immutable log, which is worse than a documented gap; see F-13 in
        `docs/RELEASE.md`.
        """
        if not actor.is_administrator:
            await self._audit.log(
                tenant_id=actor.tenant_id,
                user_id=actor.id,
                action=AuditAction.ACCESS_DENIED,
                object_type=AuditObjectType.USER,
                object_id=str(actor.id),
                outcome=AuditOutcome.DENIED,
                actor_email=actor.email,
                ip_address=context.ip_address,
                user_agent=context.user_agent,
                detail={"reason": "user_list_requires_administrator"},
            )
            raise UnauthorizedAccess()

        capped = max(1, min(limit, MAX_USER_PAGE))
        users = await self._users.list_for_tenant(
            actor.tenant_id, limit=capped, offset=max(0, offset)
        )
        total = await self._users.count_for_tenant(actor.tenant_id)
        admins = await self._users.count_admins(actor.tenant_id)
        return UserPage(
            users=tuple(users),
            total=total,
            limit=capped,
            offset=max(0, offset),
            admins=admins,
        )

    # -- Role administration -------------------------------------------------
    async def set_role(
        self,
        *,
        actor: User,
        target_user_id: uuid.UUID,
        role: UserRole,
        context: RequestContext,
    ) -> None:
        """Change a user's role. Administrator-only; the route enforces that too.

        Refuses to remove the last Administrator in a tenant: a tenant with no admin can
        never grant the role back, and recovering it means a database edit.
        """
        if not actor.is_administrator:
            raise UnauthorizedAccess()
        target = await self._users.get_by_id(target_user_id, actor.tenant_id)
        if target is None:
            # Cross-tenant id is a 404, never a 403 — a 403 would confirm the id exists.
            raise UserNotFound()
        if target.role is role:
            return
        if (
            target.role is UserRole.ADMINISTRATOR
            and await self._users.count_admins(actor.tenant_id) <= 1
        ):
            raise ValidationError("A tenant must retain at least one Administrator.")

        # Read before writing: the update may refresh `target` in place, and an audit row
        # that says ADMINISTRATOR -> ADMINISTRATOR records nothing about what changed.
        previous_role = target.role
        await self._users.set_role(target.id, actor.tenant_id, role)
        # The old access token still carries the old role until it expires; killing the
        # refresh tokens bounds that window to the access-token lifetime.
        await self._tokens.revoke_all_for_user(target.id)
        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=AuditAction.ROLE_CHANGE,
            object_type=AuditObjectType.USER,
            object_id=str(target.id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={"from": previous_role.value, "to": role.value},
        )


# -- Helpers -----------------------------------------------------------------
def _normalize_email(email: str) -> str:
    """Lower-case and trim. Stored and compared in one form only."""
    cleaned = email.strip().lower()
    if not cleaned or "@" not in cleaned:
        raise ValidationError("A valid email address is required.")
    return cleaned


def _validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValidationError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValidationError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded."
        )


def _jti_to_uuid_bytes(jti: str) -> bytes:
    """Derive the 16-byte `refresh_token.id` from the token's `jti` claim.

    The `jti` is url-safe base64 of arbitrary length, and the column is a UUID. Hashing it
    gives a stable, collision-resistant mapping in both directions of the lookup — the
    token itself is never recoverable from the id.
    """
    return hashlib.sha256(jti.encode("ascii")).digest()[:16]


#: `_actor_or_unknown` used to live here, filling `actor_email` with `"unknown"` for a replayed
#: refresh cookie. It has no callers left: `refresh` now resolves the subject against the
#: database before it audits anything, so the one row that used it can name the real account.
#: An audit row that says `unknown` when the identity was recoverable is a worse record than
#: one that names the user, and nothing else in this service needed the fallback.
