"""backend/schemas/user_schemas.py

Role: Pydantic DTOs for the Administrator's user-management endpoints.

Separate from `backend/schemas/auth.py` on purpose. `UserProfile` there answers "who am I?" and
is returned to its own subject; `TenantUser` here answers "who is in this workspace?" and is
returned to an Administrator about *someone else*. Those two audiences do not have the same
claim on the same columns, and keeping one model for both is how a field added for the self
view quietly becomes visible to a third party.

What is deliberately absent from `TenantUser`:
  * `hashed_password` — never leaves the database, and cannot be omitted by accident because
    there is no field for it to fill;
  * `failed_login_attempts` and `locked_until` — a live brute-force signal about a named
    person. An Administrator troubleshooting a locked account reads the audit log, where the
    attempt is recorded with its outcome and the read itself is traceable;
  * `tenant_id` — every row in the page is the caller's own tenant by construction, so
    repeating it per row invites a client to filter on it and believe it is a control.

`created_at` is the field the request "admin can see all the users who signed up" actually
turns on: it is when the account was created, so the list reads as a registration log.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr

from backend.models.user import User, UserRole


class TenantUser(BaseModel):
    """One account, as an Administrator sees it."""

    model_config = ConfigDict(extra="forbid")

    id: uuid.UUID
    email: EmailStr
    full_name: str | None
    role: UserRole
    is_active: bool
    #: When the account was created — for a self-registration, when the person signed up.
    created_at: datetime
    #: Null until the account is used once. A signed-up-but-never-signed-in account is a real
    #: state and the UI says so rather than showing an empty cell.
    last_login_at: datetime | None = None

    @classmethod
    def from_entity(cls, user: User) -> TenantUser:
        """Explicit field mapping — never `from_attributes` on the ORM entity.

        `model_config` could enable `from_attributes` and this method would shrink to nothing,
        and then a column added to `User` would start appearing in an API response the moment
        someone also added it here. Listing the fields means publishing one is a decision.
        """
        return cls(
            id=user.id,
            email=user.email,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
        )


class TenantUserPage(BaseModel):
    """A page of accounts plus the total, for the pager.

    `admins` is sent because the UI needs it for a rule the server owns: a tenant must keep at
    least one Administrator, so the demote control on the last one has to be disabled with an
    explanation rather than offered and then refused with a 400. The server still enforces it
    (`AuthService.set_role`) — this is so the UI can be honest before the click, not instead.
    """

    model_config = ConfigDict(extra="forbid")

    items: list[TenantUser]
    total: int
    limit: int
    offset: int
    #: How many Administrators the tenant has, counted server-side across all pages.
    admins: int
