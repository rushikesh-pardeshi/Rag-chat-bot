"""backend/tests/test_auth_service.py

Role: Unit tests for the Phase 2 authentication rules — password hashing, token claims,
brute-force lockout, refresh rotation, tenant isolation and RBAC.

These run without a database. The repositories are faked (`backend/tests/fakes.py`) because
none of the behaviour under test lives in SQL: lockout, rotation and the "last Administrator"
rule are service invariants. The queries themselves are covered separately by the
`requires_db` integration tests.

Read alongside `.claude/rules/rbac-enforcement.md` — the tenant-mismatch tests exist because
role and tenant are independent checks, and an Administrator of tenant A must get nothing
from tenant B.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import jwt
import pytest

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
from backend.core.security import (
    BCRYPT_ROUNDS,
    MAX_PASSWORD_BYTES,
    TOKEN_TYPE_ACCESS,
    TOKEN_TYPE_REFRESH,
    create_access_token,
    create_refresh_token,
    decode_access_token,
    generate_csrf_token,
    hash_password,
    hash_refresh_token,
    verify_csrf_token,
    verify_password,
)
from backend.models.audit_log import AuditAction, AuditOutcome
from backend.models.user import UserRole
from backend.services.audit_service import AuditService
from backend.services.auth_service import AuthService, RequestContext
from backend.tests.fakes import (
    FakeAuditRepository,
    FakeAuthRepository,
    FakeUserRepository,
)
from backend.utils.ids import utc_now

PASSWORD = "Correct-Horse-Battery-9"
CONTEXT = RequestContext(ip_address="203.0.113.10", user_agent="pytest")


@pytest.fixture
def users() -> FakeUserRepository:
    return FakeUserRepository()


@pytest.fixture
def tokens() -> FakeAuthRepository:
    return FakeAuthRepository()


@pytest.fixture
def audit_repo() -> FakeAuditRepository:
    return FakeAuditRepository()


@pytest.fixture
def service(
    users: FakeUserRepository,
    tokens: FakeAuthRepository,
    audit_repo: FakeAuditRepository,
) -> AuthService:
    return AuthService(users, tokens, AuditService(audit_repo))  # type: ignore[arg-type]


@pytest.fixture
def tenant(users: FakeUserRepository):
    return users.add_tenant(domain="examplebank.com")


# -- Password hashing --------------------------------------------------------
class TestPasswordHashing:
    def test_hash_uses_bcrypt_at_configured_cost(self) -> None:
        digest = hash_password(PASSWORD)
        assert digest.startswith(f"$2b${BCRYPT_ROUNDS:02d}$")

    def test_verify_accepts_the_right_password(self) -> None:
        assert verify_password(PASSWORD, hash_password(PASSWORD)) is True

    def test_verify_rejects_the_wrong_password(self) -> None:
        assert verify_password("not-the-password", hash_password(PASSWORD)) is False

    def test_two_hashes_of_one_password_differ(self) -> None:
        """Distinct salts. Equal hashes would make the table rainbow-table-able."""
        assert hash_password(PASSWORD) != hash_password(PASSWORD)

    def test_malformed_stored_hash_returns_false_not_an_exception(self) -> None:
        """A corrupted row must fail closed, not 500 — a 500 confirms the account exists."""
        assert verify_password(PASSWORD, "not-a-bcrypt-hash") is False

    def test_overlong_password_is_rejected_rather_than_truncated(self) -> None:
        """bcrypt ignores bytes past 72. Truncating silently would make the tail decorative."""
        with pytest.raises(ValidationError):
            hash_password("a" * (MAX_PASSWORD_BYTES + 1))


# -- Tokens ------------------------------------------------------------------
class TestTokens:
    def test_access_token_round_trips_the_authorization_claims(self) -> None:
        user_id, tenant_id = str(uuid.uuid4()), str(uuid.uuid4())
        claims = decode_access_token(
            create_access_token(
                user_id=user_id, tenant_id=tenant_id, role=UserRole.ADMINISTRATOR.value
            )
        )
        assert claims["sub"] == user_id
        assert claims["tenant_id"] == tenant_id
        assert claims["role"] == UserRole.ADMINISTRATOR.value
        assert claims["type"] == TOKEN_TYPE_ACCESS

    def test_refresh_token_carries_the_jti_it_returns(self) -> None:
        token, jti = create_refresh_token(
            user_id=str(uuid.uuid4()), tenant_id=str(uuid.uuid4())
        )
        claims = decode_access_token(token, expected_type=TOKEN_TYPE_REFRESH)
        assert claims["jti"] == jti

    def test_refresh_token_is_not_accepted_as_an_access_token(self) -> None:
        """The `type` claim is what stops a long-lived refresh cookie authorizing a request."""
        token, _ = create_refresh_token(
            user_id=str(uuid.uuid4()), tenant_id=str(uuid.uuid4())
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(token)

    def test_tampered_signature_is_rejected(self) -> None:
        token = create_access_token(
            user_id=str(uuid.uuid4()),
            tenant_id=str(uuid.uuid4()),
            role=UserRole.STANDARD_USER.value,
        )
        header, payload, _ = token.split(".")
        with pytest.raises(AuthenticationError):
            decode_access_token(f"{header}.{payload}.AAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    def test_role_escalation_by_editing_the_payload_is_rejected(self) -> None:
        """Re-signing with a different key must fail: the signature is the whole control."""
        forged = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "tenant_id": str(uuid.uuid4()),
                "role": UserRole.ADMINISTRATOR.value,
                "type": TOKEN_TYPE_ACCESS,
                "exp": utc_now() + timedelta(minutes=15),
            },
            "an-attacker-supplied-signing-key-32-chars",
            algorithm="HS256",
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(forged)

    def test_unsigned_token_is_rejected(self) -> None:
        """`alg: none` must not decode: the algorithm list is pinned, not read from
        the token header."""
        unsigned = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "tenant_id": str(uuid.uuid4()),
                "role": UserRole.ADMINISTRATOR.value,
                "type": TOKEN_TYPE_ACCESS,
                "exp": utc_now() + timedelta(minutes=15),
            },
            key="",
            algorithm="none",
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(unsigned)

    def test_expired_token_is_rejected(self) -> None:
        settings = get_settings()
        expired = jwt.encode(
            {
                "sub": str(uuid.uuid4()),
                "tenant_id": str(uuid.uuid4()),
                "role": UserRole.STANDARD_USER.value,
                "type": TOKEN_TYPE_ACCESS,
                "exp": utc_now() - timedelta(seconds=1),
            },
            settings.SECRET_KEY.get_secret_value(),
            algorithm=settings.JWT_ALGORITHM,
        )
        with pytest.raises(AuthenticationError):
            decode_access_token(expired)

    def test_stored_refresh_hash_is_not_the_token(self) -> None:
        """A stolen database yields hashes, not sessions."""
        token, _ = create_refresh_token(
            user_id=str(uuid.uuid4()), tenant_id=str(uuid.uuid4())
        )
        digest = hash_refresh_token(token)
        assert digest != token
        assert len(digest) == 64
        assert digest == hash_refresh_token(token)

    def test_csrf_tokens_compare_only_against_themselves(self) -> None:
        token = generate_csrf_token()
        assert verify_csrf_token(token, token) is True
        assert verify_csrf_token(token, generate_csrf_token()) is False
        assert verify_csrf_token(None, token) is False
        assert verify_csrf_token(token, None) is False


# -- Registration ------------------------------------------------------------
class TestRegister:
    async def test_registration_assigns_standard_user_server_side(
        self, service: AuthService, tenant
    ) -> None:
        user = await service.register(
            email="New.User@examplebank.com",
            password=PASSWORD,
            full_name="New User",
            context=CONTEXT,
        )
        assert user.role is UserRole.STANDARD_USER
        assert user.tenant_id == tenant.id
        # Normalized on the way in, so the unique constraint means what it says.
        assert user.email == "new.user@examplebank.com"

    async def test_registration_stores_a_bcrypt_hash_not_the_password(
        self, service: AuthService, tenant
    ) -> None:
        user = await service.register(
            email="hash@examplebank.com",
            password=PASSWORD,
            full_name=None,
            context=CONTEXT,
        )
        assert PASSWORD not in user.hashed_password
        assert user.hashed_password.startswith(f"$2b${BCRYPT_ROUNDS:02d}$")

    async def test_duplicate_email_in_the_same_tenant_is_refused(
        self, service: AuthService, tenant
    ) -> None:
        await service.register(
            email="dupe@examplebank.com",
            password=PASSWORD,
            full_name=None,
            context=CONTEXT,
        )
        with pytest.raises(EmailAlreadyRegistered):
            await service.register(
                email="dupe@examplebank.com",
                password=PASSWORD,
                full_name=None,
                context=CONTEXT,
            )

    async def test_same_email_in_a_second_tenant_is_allowed(
        self, service: AuthService, users: FakeUserRepository, tenant
    ) -> None:
        """Identity is `(email, tenant_id)` — the same person may exist in two banks."""
        other = users.add_tenant(domain="otherbank.com", name="Other Bank")
        first = await service.register(
            email="shared@examplebank.com",
            password=PASSWORD,
            full_name=None,
            context=CONTEXT,
        )
        second = await service.register(
            email="shared@examplebank.com",
            password=PASSWORD,
            full_name=None,
            context=CONTEXT,
            tenant_id=other.id,
        )
        assert first.id != second.id
        assert {first.tenant_id, second.tenant_id} == {tenant.id, other.id}

    async def test_unknown_email_domain_does_not_reveal_which_tenants_exist(
        self, service: AuthService, tenant
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            await service.register(
                email="outsider@unknown-domain.example",
                password=PASSWORD,
                full_name=None,
                context=CONTEXT,
            )
        assert "unknown-domain.example" not in str(excinfo.value)

    async def test_short_password_is_refused(self, service: AuthService, tenant) -> None:
        with pytest.raises(ValidationError):
            await service.register(
                email="weak@examplebank.com",
                password="short",
                full_name=None,
                context=CONTEXT,
            )

    async def test_registration_is_audited(
        self, service: AuthService, audit_repo: FakeAuditRepository, tenant
    ) -> None:
        await service.register(
            email="audited@examplebank.com",
            password=PASSWORD,
            full_name=None,
            context=CONTEXT,
        )
        assert audit_repo.entries
        entry = audit_repo.entries[-1]
        assert entry.action is AuditAction.ROLE_CHANGE
        assert entry.outcome is AuditOutcome.SUCCESS
        assert entry.detail == {
            "event": "register",
            "role": UserRole.STANDARD_USER.value,
        }


# -- Login and lockout -------------------------------------------------------
class TestAuthenticate:
    @pytest.fixture
    def account(self, users: FakeUserRepository, tenant):
        return users.add_user(
            tenant_id=tenant.id,
            email="teller@examplebank.com",
            hashed_password=hash_password(PASSWORD),
        )

    async def test_valid_credentials_issue_a_session(
        self, service: AuthService, account
    ) -> None:
        session = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        claims = decode_access_token(session.access_token)
        assert claims["sub"] == str(account.id)
        assert claims["tenant_id"] == str(account.tenant_id)
        assert claims["role"] == UserRole.STANDARD_USER.value
        assert session.expires_in_seconds == (
            get_settings().ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        assert session.csrf_token

    async def test_successful_login_persists_a_hashed_refresh_token(
        self, service: AuthService, tokens: FakeAuthRepository, account
    ) -> None:
        session = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        stored = await tokens.get_by_hash(hash_refresh_token(session.refresh_token))
        assert stored is not None
        assert stored.revoked_at is None
        assert stored.token_hash != session.refresh_token

    async def test_login_is_case_insensitive_on_email(
        self, service: AuthService, account
    ) -> None:
        session = await service.authenticate(
            email="Teller@ExampleBank.com", password=PASSWORD, context=CONTEXT
        )
        assert session.user_id == account.id

    async def test_unknown_email_and_wrong_password_raise_the_same_error(
        self, service: AuthService, account
    ) -> None:
        """Identical exception type and message, or login becomes an account oracle."""
        with pytest.raises(InvalidCredentials) as unknown:
            await service.authenticate(
                email="nobody@examplebank.com", password=PASSWORD, context=CONTEXT
            )
        with pytest.raises(InvalidCredentials) as wrong:
            await service.authenticate(
                email=account.email, password="wrong-password-here", context=CONTEXT
            )
        assert unknown.value.message == wrong.value.message

    async def test_disabled_account_looks_like_bad_credentials(
        self, service: AuthService, users: FakeUserRepository, tenant
    ) -> None:
        users.add_user(
            tenant_id=tenant.id,
            email="disabled@examplebank.com",
            hashed_password=hash_password(PASSWORD),
            is_active=False,
        )
        with pytest.raises(InvalidCredentials):
            await service.authenticate(
                email="disabled@examplebank.com", password=PASSWORD, context=CONTEXT
            )

    async def test_repeated_failures_lock_the_account(
        self, service: AuthService, account
    ) -> None:
        limit = get_settings().MAX_FAILED_LOGIN_ATTEMPTS
        for _ in range(limit):
            with pytest.raises(InvalidCredentials):
                await service.authenticate(
                    email=account.email, password="wrong", context=CONTEXT
                )
        assert account.failed_login_attempts == limit
        assert account.locked_until is not None
        # The right password now fails too — that is the point of a lockout.
        with pytest.raises(AccountLocked):
            await service.authenticate(
                email=account.email, password=PASSWORD, context=CONTEXT
            )

    async def test_a_successful_login_clears_the_failure_counter(
        self, service: AuthService, account
    ) -> None:
        with pytest.raises(InvalidCredentials):
            await service.authenticate(
                email=account.email, password="wrong", context=CONTEXT
            )
        assert account.failed_login_attempts == 1
        await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        assert account.failed_login_attempts == 0
        assert account.last_login_at is not None

    async def test_an_expired_lockout_stops_blocking(
        self, service: AuthService, users: FakeUserRepository, tenant
    ) -> None:
        users.add_user(
            tenant_id=tenant.id,
            email="waited@examplebank.com",
            hashed_password=hash_password(PASSWORD),
            failed_login_attempts=5,
            locked_until=utc_now() - timedelta(minutes=1),
        )
        session = await service.authenticate(
            email="waited@examplebank.com", password=PASSWORD, context=CONTEXT
        )
        assert session.role is UserRole.STANDARD_USER

    async def test_every_login_outcome_against_a_known_account_is_audited(
        self, service: AuthService, audit_repo: FakeAuditRepository, account
    ) -> None:
        with pytest.raises(InvalidCredentials):
            await service.authenticate(
                email=account.email, password="wrong", context=CONTEXT
            )
        await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        assert audit_repo.actions() == [
            AuditAction.LOGIN_FAILURE.value,
            AuditAction.LOGIN_SUCCESS.value,
        ]
        assert audit_repo.outcomes() == [
            AuditOutcome.FAILURE.value,
            AuditOutcome.SUCCESS.value,
        ]

    async def test_audit_rows_never_contain_the_password(
        self, service: AuthService, audit_repo: FakeAuditRepository, account
    ) -> None:
        with pytest.raises(InvalidCredentials):
            await service.authenticate(
                email=account.email, password=PASSWORD + "-typo", context=CONTEXT
            )
        serialized = repr([e.detail for e in audit_repo.entries])
        assert PASSWORD not in serialized


# -- Refresh rotation --------------------------------------------------------
class TestRefresh:
    @pytest.fixture
    def account(self, users: FakeUserRepository, tenant):
        return users.add_user(
            tenant_id=tenant.id,
            email="rotator@examplebank.com",
            hashed_password=hash_password(PASSWORD),
        )

    async def test_refresh_issues_a_new_pair_and_revokes_the_old_token(
        self, service: AuthService, tokens: FakeAuthRepository, account
    ) -> None:
        first = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        second = await service.refresh(
            refresh_token=first.refresh_token, context=CONTEXT
        )
        assert second.refresh_token != first.refresh_token
        old = await tokens.get_by_hash(hash_refresh_token(first.refresh_token))
        assert old is not None
        assert old.revoked_at is not None
        assert old.replaced_by_id is not None

    async def test_replaying_a_rotated_token_revokes_the_whole_family(
        self, service: AuthService, tokens: FakeAuthRepository, account
    ) -> None:
        """Reuse means either a replay or a stolen cookie; both end every session."""
        first = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        second = await service.refresh(
            refresh_token=first.refresh_token, context=CONTEXT
        )
        with pytest.raises(AuthenticationError):
            await service.refresh(refresh_token=first.refresh_token, context=CONTEXT)

        surviving = await tokens.get_by_hash(hash_refresh_token(second.refresh_token))
        assert surviving is not None
        assert surviving.revoked_at is not None

    async def test_reuse_is_audited_as_a_denial(
        self, service: AuthService, audit_repo: FakeAuditRepository, account
    ) -> None:
        first = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        await service.refresh(refresh_token=first.refresh_token, context=CONTEXT)
        with pytest.raises(AuthenticationError):
            await service.refresh(refresh_token=first.refresh_token, context=CONTEXT)
        denial = audit_repo.entries[-1]
        assert denial.action is AuditAction.ACCESS_DENIED
        assert denial.outcome is AuditOutcome.DENIED
        assert denial.detail is not None
        assert denial.detail["reason"] == "refresh_token_reuse"

    async def test_an_access_token_cannot_be_used_to_refresh(
        self, service: AuthService, account
    ) -> None:
        session = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        with pytest.raises(AuthenticationError):
            await service.refresh(
                refresh_token=session.access_token, context=CONTEXT
            )

    async def test_a_cookie_whose_subject_no_longer_exists_is_a_401_not_a_500(
        self, service: AuthService, audit_repo: FakeAuditRepository
    ) -> None:
        """The reported 500 on `/auth/refresh`, reproduced.

        A browser that keeps its cookies across a database reset or re-seed presents a refresh
        token whose signature still verifies — `SECRET_KEY` did not change — but whose `sub`
        and `tenant_id` name rows that are gone. The token row is gone too, so the reuse
        branch runs, and it used to audit against the **cookie's** `tenant_id`.

        `audit_log.tenant_id` is `ForeignKey("tenant.id")` and `AuditRepository.append`
        flushes, so PostgreSQL refused the insert and `IntegrityError` escaped as an unhandled
        exception: a 500 for what is only an expired session. The fix resolves the subject
        against the database first, so an unverifiable one is simply not a session.

        No audit row is written here, deliberately: there is no tenant for it to belong to and
        no account for it to name. The application log carries it instead.
        """
        settings = get_settings()
        stale = jwt.encode(
            {
                # Both ids are valid UUIDs and neither exists in this repository.
                "sub": str(uuid.uuid4()),
                "tenant_id": str(uuid.uuid4()),
                "type": TOKEN_TYPE_REFRESH,
                "jti": "a-jti-from-the-previous-database",
                "exp": utc_now() + timedelta(days=1),
            },
            settings.SECRET_KEY.get_secret_value(),
            algorithm=settings.JWT_ALGORITHM,
        )

        before = len(audit_repo.entries)
        with pytest.raises(AuthenticationError):
            await service.refresh(refresh_token=stale, context=CONTEXT)
        assert len(audit_repo.entries) == before

    async def test_a_replayed_cookie_for_a_real_user_still_audits_and_revokes(
        self, service: AuthService, audit_repo: FakeAuditRepository, account
    ) -> None:
        """The security-relevant half of the same branch, which must not have been lost.

        Resolving the subject before auditing is what fixed the 500; it must not have turned
        replay detection into a silent 401. A token whose row is spent still revokes every
        session for that account and still writes the `refresh_token_reuse` row — now naming
        the account rather than recording `unknown`, because by this point it is known.
        """
        session = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        # One legitimate rotation, which revokes the token just presented.
        await service.refresh(refresh_token=session.refresh_token, context=CONTEXT)

        before = len(audit_repo.entries)
        with pytest.raises(AuthenticationError):
            # The same cookie a second time: a replay.
            await service.refresh(refresh_token=session.refresh_token, context=CONTEXT)

        written = audit_repo.entries[before:]
        reuse = [e for e in written if (e.detail or {}).get("reason") == "refresh_token_reuse"]
        assert len(reuse) == 1
        assert reuse[0].outcome is AuditOutcome.DENIED
        assert reuse[0].action is AuditAction.ACCESS_DENIED
        assert reuse[0].actor_email == account.email
        assert reuse[0].tenant_id == account.tenant_id
        assert reuse[0].detail["sessions_revoked"] is True

    @pytest.mark.parametrize(
        ("claim", "value"),
        [
            # `sub` and `tenant_id` are converted with `uuid.UUID(...)`, which raises
            # `ValueError` on anything that is not a UUID string.
            ("sub", "not-a-uuid"),
            ("tenant_id", "not-a-uuid"),
            # `jti` is hashed with `.encode("ascii")`, which raises `UnicodeEncodeError`.
            ("jti", "jti-with-a-non-ascii-char-é"),
        ],
    )
    async def test_a_signed_refresh_token_with_unparseable_claims_is_a_401_not_a_500(
        self, service: AuthService, claim: str, value: str
    ) -> None:
        """An unauthenticated endpoint whose entire input is a cookie must never 500.

        `decode_access_token` proves the signature and that the required claims are present
        and non-empty. It does **not** prove they parse — so a validly-signed token carrying,
        say, a non-UUID `sub` used to reach `uuid.UUID(...)` unguarded and surface as an
        unhandled exception.

        Reachable in practice only from a cookie that outlived a change to the claim format,
        not from forgery: producing a token that gets this far needs the server's own signing
        key. It is still the wrong answer. A 500 tells the caller the request was fine and the
        server is broken, when the truth is the opposite — and because
        `api-client.attemptRefresh` reads any non-`ok` response as "refresh failed", the two
        are indistinguishable in the browser and the 500 would only ever show up as noise in
        the log.
        """
        settings = get_settings()
        payload = {
            "sub": str(uuid.uuid4()),
            "tenant_id": str(uuid.uuid4()),
            "type": TOKEN_TYPE_REFRESH,
            "jti": "a-normal-looking-jti",
            "exp": utc_now() + timedelta(days=1),
        }
        payload[claim] = value
        # Signed with the server's real key: this is the case the signature check passes.
        token = jwt.encode(
            payload,
            settings.SECRET_KEY.get_secret_value(),
            algorithm=settings.JWT_ALGORITHM,
        )

        with pytest.raises(AuthenticationError):
            await service.refresh(refresh_token=token, context=CONTEXT)

    async def test_refresh_fails_once_the_account_is_disabled(
        self, service: AuthService, account
    ) -> None:
        session = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        account.is_active = False
        with pytest.raises(AuthenticationError):
            await service.refresh(
                refresh_token=session.refresh_token, context=CONTEXT
            )


# -- Logout ------------------------------------------------------------------
class TestLogout:
    async def test_logout_revokes_the_presented_token_and_audits(
        self,
        service: AuthService,
        users: FakeUserRepository,
        tokens: FakeAuthRepository,
        audit_repo: FakeAuditRepository,
        tenant,
    ) -> None:
        user = users.add_user(
            tenant_id=tenant.id,
            email="leaver@examplebank.com",
            hashed_password=hash_password(PASSWORD),
        )
        session = await service.authenticate(
            email=user.email, password=PASSWORD, context=CONTEXT
        )
        await service.logout(
            refresh_token=session.refresh_token, user=user, context=CONTEXT
        )
        assert await tokens.is_revoked(hash_refresh_token(session.refresh_token))
        assert audit_repo.entries[-1].action is AuditAction.LOGOUT

    async def test_logout_without_a_session_is_not_an_error(
        self, service: AuthService
    ) -> None:
        """A dead access token must still clear the browser's cookies."""
        await service.logout(refresh_token=None, user=None, context=CONTEXT)


# -- Password change ---------------------------------------------------------
class TestChangePassword:
    @pytest.fixture
    def account(self, users: FakeUserRepository, tenant):
        return users.add_user(
            tenant_id=tenant.id,
            email="changer@examplebank.com",
            hashed_password=hash_password(PASSWORD),
        )

    async def test_change_sets_the_new_hash_and_ends_every_session(
        self, service: AuthService, tokens: FakeAuthRepository, account
    ) -> None:
        session = await service.authenticate(
            email=account.email, password=PASSWORD, context=CONTEXT
        )
        new_password = "Another-Long-Passphrase-7"
        await service.change_password(
            user=account,
            current_password=PASSWORD,
            new_password=new_password,
            context=CONTEXT,
        )
        assert verify_password(new_password, account.hashed_password)
        assert not verify_password(PASSWORD, account.hashed_password)
        assert await tokens.is_revoked(hash_refresh_token(session.refresh_token))

    async def test_wrong_current_password_is_refused_and_audited(
        self, service: AuthService, audit_repo: FakeAuditRepository, account
    ) -> None:
        with pytest.raises(InvalidCredentials):
            await service.change_password(
                user=account,
                current_password="not-the-current-one",
                new_password="Another-Long-Passphrase-7",
                context=CONTEXT,
            )
        assert audit_repo.entries[-1].outcome is AuditOutcome.DENIED

    async def test_reusing_the_same_password_is_refused(
        self, service: AuthService, account
    ) -> None:
        with pytest.raises(ValidationError):
            await service.change_password(
                user=account,
                current_password=PASSWORD,
                new_password=PASSWORD,
                context=CONTEXT,
            )


# -- RBAC and tenant isolation -----------------------------------------------
class TestRoleAdministration:
    @pytest.fixture
    def admin(self, users: FakeUserRepository, tenant):
        return users.add_user(
            tenant_id=tenant.id,
            email="admin@examplebank.com",
            hashed_password=hash_password(PASSWORD),
            role=UserRole.ADMINISTRATOR,
        )

    @pytest.fixture
    def standard(self, users: FakeUserRepository, tenant):
        return users.add_user(
            tenant_id=tenant.id,
            email="standard@examplebank.com",
            hashed_password=hash_password(PASSWORD),
        )

    async def test_administrator_can_promote_a_standard_user(
        self, service: AuthService, admin, standard
    ) -> None:
        await service.set_role(
            actor=admin,
            target_user_id=standard.id,
            role=UserRole.ADMINISTRATOR,
            context=CONTEXT,
        )
        assert standard.role is UserRole.ADMINISTRATOR

    async def test_standard_user_cannot_promote_anyone(
        self, service: AuthService, standard, admin
    ) -> None:
        with pytest.raises(UnauthorizedAccess):
            await service.set_role(
                actor=standard,
                target_user_id=standard.id,
                role=UserRole.ADMINISTRATOR,
                context=CONTEXT,
            )
        assert standard.role is UserRole.STANDARD_USER

    async def test_a_role_change_revokes_the_targets_sessions(
        self, service: AuthService, tokens: FakeAuthRepository, admin, standard
    ) -> None:
        """The old access token still carries the old role; revoking refresh tokens
        bounds that window to one access-token lifetime."""
        session = await service.authenticate(
            email=standard.email, password=PASSWORD, context=CONTEXT
        )
        await service.set_role(
            actor=admin,
            target_user_id=standard.id,
            role=UserRole.ADMINISTRATOR,
            context=CONTEXT,
        )
        assert await tokens.is_revoked(hash_refresh_token(session.refresh_token))

    async def test_the_last_administrator_cannot_be_demoted(
        self, service: AuthService, admin
    ) -> None:
        """A tenant with no Administrator can never grant the role back."""
        with pytest.raises(ValidationError):
            await service.set_role(
                actor=admin,
                target_user_id=admin.id,
                role=UserRole.STANDARD_USER,
                context=CONTEXT,
            )
        assert admin.role is UserRole.ADMINISTRATOR

    async def test_cross_tenant_target_is_a_404_not_a_403(
        self, service: AuthService, users: FakeUserRepository, admin
    ) -> None:
        """A 403 would confirm the id exists. Administrator of tenant A sees nothing in B."""
        other_tenant = users.add_tenant(domain="otherbank.com", name="Other Bank")
        victim = users.add_user(
            tenant_id=other_tenant.id,
            email="victim@otherbank.com",
            hashed_password=hash_password(PASSWORD),
        )
        with pytest.raises(UserNotFound):
            await service.set_role(
                actor=admin,
                target_user_id=victim.id,
                role=UserRole.ADMINISTRATOR,
                context=CONTEXT,
            )
        assert victim.role is UserRole.STANDARD_USER

    async def test_role_change_is_audited_with_the_transition(
        self, service: AuthService, audit_repo: FakeAuditRepository, admin, standard
    ) -> None:
        await service.set_role(
            actor=admin,
            target_user_id=standard.id,
            role=UserRole.ADMINISTRATOR,
            context=CONTEXT,
        )
        entry = audit_repo.entries[-1]
        assert entry.action is AuditAction.ROLE_CHANGE
        assert entry.user_id == admin.id
        assert entry.object_id == str(standard.id)
        assert entry.detail == {
            "from": UserRole.STANDARD_USER.value,
            "to": UserRole.ADMINISTRATOR.value,
        }


@pytest.mark.requires_db
class TestRefreshAgainstPostgres:
    """The half the fakes cannot express: the foreign key that produced the 500.

    Marked as well as fixture-guarded. The `db_available` fixture makes these skip cleanly on
    a laptop with no PostgreSQL; the marker is what gets them *selected* by
    `scripts/run_db_tests.py test`, which is the gate that has a database on purpose. Without
    it they would be deselected there and only ever run by accident.

    `FakeAuditRepository` is a list. It accepts any `tenant_id`, so against the fakes the old
    code wrote its audit row happily and the bug was invisible — the insert only fails where
    `audit_log.tenant_id REFERENCES tenant(id)` is enforced, which is PostgreSQL.

    Skipped rather than failed without a database, like the rest of the `db_available` suite.
    """

    async def test_a_stale_cookie_does_not_raise_an_integrity_error(
        self, db_available: bool
    ) -> None:
        """Reproduces the reported failure, and pins the 401 that replaced it.

        Before the fix this body raised `sqlalchemy.exc.IntegrityError` — a foreign key
        violation on a tenant id that came out of a cookie — which FastAPI's catch-all turned
        into `500 internal_error`. The assertion is therefore as much about the *type* of the
        exception as about the status: anything other than `AuthenticationError` here is a 500
        in production.
        """
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope
        from backend.repositories.audit_repository import AuditRepository
        from backend.repositories.auth_repository import AuthRepository
        from backend.repositories.user_repository import UserRepository

        settings = get_settings()
        stale = jwt.encode(
            {
                # A well-formed, correctly-signed token for a world that no longer exists.
                "sub": str(uuid.uuid4()),
                "tenant_id": str(uuid.uuid4()),
                "type": TOKEN_TYPE_REFRESH,
                "jti": "a-jti-from-the-previous-database",
                "exp": utc_now() + timedelta(days=1),
            },
            settings.SECRET_KEY.get_secret_value(),
            algorithm=settings.JWT_ALGORITHM,
        )

        async with session_scope() as session:
            service = AuthService(
                UserRepository(session),
                AuthRepository(session),
                AuditService(AuditRepository(session)),
            )
            with pytest.raises(AuthenticationError):
                await service.refresh(refresh_token=stale, context=CONTEXT)
