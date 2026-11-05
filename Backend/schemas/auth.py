"""backend/schemas/auth.py

Role: Pydantic DTOs for the auth endpoints. The API boundary — never ORM entities.

Every request model sets `extra="forbid"`. That is the mass-assignment control: a body
carrying `role`, `tenant_id`, `is_active` or `failed_login_attempts` is rejected with 422
rather than silently ignored, so a probe for privilege escalation shows up as an error
instead of succeeding-but-having-no-effect.

No response model contains `hashed_password`, and none is built from an ORM object
implicitly — each is constructed from explicit fields.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from backend.core.security import MIN_PASSWORD_LENGTH
from backend.models.user import UserRole

#: Shared by every request model in this module.
_REQUEST_CONFIG = ConfigDict(extra="forbid", str_strip_whitespace=True)

#: bcrypt's hard limit. Enforced here too so an oversized password is a 422 rather than a
#: 400 raised deeper in the service.
MAX_PASSWORD_LENGTH = 72


class RegisterRequest(BaseModel):
    """Self-registration.

    `role` and `tenant_id` are absent on purpose and would be rejected if supplied: the
    server assigns Standard User and derives the tenant from the email domain.
    """

    model_config = _REQUEST_CONFIG

    email: EmailStr
    password: str = Field(
        min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH
    )
    full_name: str | None = Field(default=None, max_length=255)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class LoginRequest(BaseModel):
    model_config = _REQUEST_CONFIG

    email: EmailStr
    # No min_length: a length rule here would reject a short wrong password with a
    # different status than a long wrong one, which is a free oracle.
    password: str = Field(max_length=MAX_PASSWORD_LENGTH)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: str) -> str:
        return value.lower()


class PasswordChangeRequest(BaseModel):
    model_config = _REQUEST_CONFIG

    current_password: str = Field(max_length=MAX_PASSWORD_LENGTH)
    new_password: str = Field(
        min_length=MIN_PASSWORD_LENGTH, max_length=MAX_PASSWORD_LENGTH
    )


class RoleChangeRequest(BaseModel):
    """Administrator-only role assignment. The target is a path parameter, not a body field."""

    model_config = _REQUEST_CONFIG

    role: UserRole


class UserProfile(BaseModel):
    """The caller's own identity, as returned by `GET /auth/me`.

    `tenant_id` is included because the frontend displays it; it is *not* trusted on the
    way back in — the server always reads the tenant from the token.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    role: UserRole
    tenant_id: uuid.UUID
    is_active: bool
    last_login_at: datetime | None = None
    created_at: datetime | None = None


class SessionResponse(BaseModel):
    """What login and refresh return in the body.

    Deliberately contains no token. The access and refresh tokens are set as httpOnly
    cookies and are unreadable from JavaScript; putting a copy in the body would undo
    that. `expires_in_seconds` is here so the client can schedule a refresh without
    parsing a JWT it cannot read.
    """

    user: UserProfile
    expires_in_seconds: int


class RegisterResponse(BaseModel):
    """Registration confirms the account and nothing else — no session, no password echo."""

    user: UserProfile


class MessageResponse(BaseModel):
    """For endpoints whose only useful answer is "done"."""

    message: str
