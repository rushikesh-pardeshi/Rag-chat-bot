"""backend/core/security.py

Role: AuthN/AuthZ primitives — password hashing, JWT encode/decode, CSRF token pair.
Called by `auth_service` and the auth dependencies; no route calls these directly.

  * passwords are hashed with bcrypt, never with a general-purpose digest — a fast hash
    over a stolen table is a weekend of GPU time;
  * tokens carry `sub` (user id), `tenant_id`, `role`, `type` and `exp`. `tenant_id` is in
    the token because every query filters on it, and re-reading it from the request body
    would let a caller choose their own tenant;
  * `decode_access_token` raises, and never returns a partially-trusted claim set.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

import bcrypt
import jwt

from backend.core.config import get_settings
from backend.core.exceptions import AuthenticationError, ValidationError

# bcrypt truncates silently at 72 bytes; reject longer input rather than hash a prefix.
MAX_PASSWORD_BYTES = 72
BCRYPT_ROUNDS = 12

#: Minimum password length. Length beats composition rules — a 12-character passphrase
#: resists guessing better than "P@ss1" ever will. Lives here rather than in the service
#: so `schemas/auth.py` can enforce it without importing a service.
MIN_PASSWORD_LENGTH = 12

# S105 reads these as hardcoded passwords. They are the two legal values of the `type`
# claim — naming them is what lets `decode_access_token` reject a refresh token presented
# as an access token.
TOKEN_TYPE_ACCESS = "access"  # noqa: S105
TOKEN_TYPE_REFRESH = "refresh"  # noqa: S105

#: Claims every access token must carry. A token missing one is rejected rather than
#: treated as a partially-trusted caller.
_REQUIRED_ACCESS_CLAIMS = ("sub", "tenant_id", "role", "type", "exp")
_REQUIRED_REFRESH_CLAIMS = ("sub", "tenant_id", "type", "exp", "jti")

#: A precomputed bcrypt hash of a value no password equals. `verify_password` runs against
#: this when the user is unknown, so a missing account costs the same time as a wrong
#: password — otherwise response timing enumerates valid emails.
_DUMMY_HASH = "$2b$12$UsUwC1IrQ6EThGcGYRp7f.FyEsh0dxLYKXgi.aWtdPYqUMO6Jm4Y2"  # noqa: S105


def _secret() -> str:
    return get_settings().SECRET_KEY.get_secret_value()


# -- Passwords ---------------------------------------------------------------
def hash_password(password: str) -> str:
    """Return a bcrypt hash (cost 12) of `password`.

    Raises `ValidationError` when the password exceeds `MAX_PASSWORD_BYTES`.
    """
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        # Truncating instead would mean two different passwords authenticate the same
        # account, which is a silent downgrade of every long passphrase.
        raise ValidationError(
            f"Password must be at most {MAX_PASSWORD_BYTES} bytes when UTF-8 encoded."
        )
    rounds = get_settings().BCRYPT_ROUNDS
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=rounds)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time verification. Returns False on a malformed hash — never raises."""
    raw = password.encode("utf-8")
    if len(raw) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(raw, password_hash.encode("ascii"))
    except (ValueError, TypeError, UnicodeEncodeError):
        # A row with a corrupted or non-bcrypt hash must not authenticate, and must not
        # turn into a 500 that tells the caller the account exists.
        return False


def dummy_verify() -> None:
    """Burn one bcrypt verification against a fixed hash.

    Called on the unknown-email path so login latency does not distinguish "no such
    account" from "wrong password".
    """
    with contextlib.suppress(ValueError):  # pragma: no cover - constant is well-formed
        bcrypt.checkpw(b"timing-equalizer", _DUMMY_HASH.encode("ascii"))


# -- Tokens ------------------------------------------------------------------
def _encode(claims: dict[str, Any]) -> str:
    return jwt.encode(claims, _secret(), algorithm=get_settings().JWT_ALGORITHM)


def create_access_token(
    *,
    user_id: str,
    tenant_id: str,
    role: str,
    expires_in: timedelta | None = None,
) -> str:
    """Signed short-lived access token carrying `sub`, `tenant_id`, `role`, `type`, `exp`."""
    settings = get_settings()
    now = datetime.now(UTC)
    lifetime = expires_in or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    return _encode(
        {
            "sub": user_id,
            "tenant_id": tenant_id,
            "role": role,
            "type": TOKEN_TYPE_ACCESS,
            "iat": now,
            "exp": now + lifetime,
            "jti": secrets.token_urlsafe(16),
        }
    )


def create_refresh_token(*, user_id: str, tenant_id: str) -> tuple[str, str]:
    """Signed long-lived refresh token. Carries no `role` — roles can be revoked.

    Returns `(token, jti)`. The caller stores only `sha256(token)`; `jti` identifies the
    row so rotation can revoke exactly the token that was presented.
    """
    settings = get_settings()
    now = datetime.now(UTC)
    jti = secrets.token_urlsafe(24)
    token = _encode(
        {
            "sub": user_id,
            "tenant_id": tenant_id,
            "type": TOKEN_TYPE_REFRESH,
            "iat": now,
            "exp": now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            "jti": jti,
        }
    )
    return token, jti


def decode_access_token(
    token: str, *, expected_type: str = TOKEN_TYPE_ACCESS
) -> dict[str, Any]:
    """Verify signature, expiry and token type; return the claims.

    Raises `AuthenticationError` on any failure — bad signature, expiry, wrong type, or a
    missing required claim. A refresh token presented as an access token is a failure.
    """
    settings = get_settings()
    try:
        claims: dict[str, Any] = jwt.decode(
            token,
            _secret(),
            # Pinned to one algorithm. Accepting a list is how "alg: none" and
            # HS256-signed-with-the-RSA-public-key confusions get in.
            algorithms=[settings.JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Session expired.") from exc
    except jwt.InvalidTokenError as exc:
        # One message for every structural failure: the caller learns nothing about
        # which part of the token was wrong.
        raise AuthenticationError("Invalid authentication token.") from exc

    if claims.get("type") != expected_type:
        raise AuthenticationError("Invalid authentication token.")

    required = (
        _REQUIRED_REFRESH_CLAIMS
        if expected_type == TOKEN_TYPE_REFRESH
        else _REQUIRED_ACCESS_CLAIMS
    )
    if any(claims.get(name) in (None, "") for name in required):
        raise AuthenticationError("Invalid authentication token.")
    return claims


def hash_refresh_token(token: str) -> str:
    """SHA-256 of a refresh token, hex.

    The token is already 190+ bits of entropy, so a slow hash buys nothing here — the
    point is that a stolen database yields digests rather than usable sessions.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


# -- CSRF --------------------------------------------------------------------
def generate_csrf_token() -> str:
    """URL-safe random token for the double-submit cookie pair."""
    return secrets.token_urlsafe(32)


def verify_csrf_token(cookie_token: str | None, header_token: str | None) -> bool:
    """Constant-time compare of the CSRF cookie against the `X-CSRF-Token` header."""
    if not cookie_token or not header_token:
        return False
    return hmac.compare_digest(cookie_token, header_token)
