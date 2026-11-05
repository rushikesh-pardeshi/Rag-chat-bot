"""backend/models/user.py

Role: ORM entities for `user` and `refresh_token`.

Identity is `email`, unique per tenant — not `username`. There are exactly two roles;
compliance officers are granted ADMINISTRATOR (see .claude/rules/rbac-enforcement.md).

Lockout counters live on the row rather than in a cache: rate limiting is per IP, so a
distributed attempt on one account would slip past it, and a counter that vanishes on
restart is not a control.
"""

from __future__ import annotations

import enum
import uuid  # runtime import: SQLAlchemy resolves Mapped[...] annotations at mapper setup
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base
from backend.utils.ids import CreatedAt, UpdatedAt, UuidPk

if TYPE_CHECKING:
    from backend.models.tenant import Tenant


class UserRole(enum.StrEnum):
    """The complete role set. Adding a third member is a product decision, not a refactor.

    A compliance officer is a person holding ADMINISTRATOR — audit read access comes with
    that role, it is not a role of its own.
    """

    ADMINISTRATOR = "ADMINISTRATOR"
    STANDARD_USER = "STANDARD_USER"


class User(Base):
    __tablename__ = "user_account"  # "user" is reserved in PostgreSQL
    __table_args__ = (
        # Identity is (email, tenant_id): the same person may exist in two tenants.
        UniqueConstraint("email", "tenant_id", name="uq_user_email_tenant"),
    )

    id: Mapped[UuidPk]
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenant.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role", native_enum=True, validate_strings=True),
        nullable=False,
        default=UserRole.STANDARD_USER,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    # -- Brute-force state ---------------------------------------------------
    failed_login_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    last_login_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]

    tenant: Mapped[Tenant] = relationship(back_populates="users", lazy="joined")

    @property
    def is_administrator(self) -> bool:
        return self.role is UserRole.ADMINISTRATOR

    def __repr__(self) -> str:
        # Never include hashed_password: __repr__ ends up in logs and tracebacks.
        return f"<User id={self.id} role={self.role.value}>"


class RefreshToken(Base):
    """One row per issued refresh token, so rotation can revoke the one it replaces.

    Only the SHA-256 of the token is stored. A stolen database gives an attacker hashes,
    not usable sessions — the same reasoning as for passwords, minus the need for a slow
    hash since the token is already high-entropy.
    """

    __tablename__ = "refresh_token"

    id: Mapped[UuidPk]
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("tenant.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user_account.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(
        String(64), nullable=False, unique=True, index=True
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    #: Set when this token was rotated, so a replayed old token identifies its successor.
    replaced_by_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("refresh_token.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[CreatedAt]

    def __repr__(self) -> str:
        revoked = bool(self.revoked_at)
        return f"<RefreshToken id={self.id} user_id={self.user_id} revoked={revoked}>"
