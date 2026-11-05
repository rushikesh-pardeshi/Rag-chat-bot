"""backend/models/tenant.py

Role: ORM entity for `tenant` — the isolation boundary every other table is scoped to.

One banking organization per deployment today, but `tenant_id` is carried and filtered on
everywhere regardless. That is what turns a cross-tenant id into a 404 rather than a leak,
and it keeps a second organization from being a migration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.core.database import Base
from backend.utils.ids import CreatedAt, UuidPk

if TYPE_CHECKING:
    from backend.models.user import User


class Tenant(Base):
    __tablename__ = "tenant"

    id: Mapped[UuidPk]
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    #: Email domain used to route a registration to the right tenant.
    domain: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[CreatedAt]

    users: Mapped[list[User]] = relationship(back_populates="tenant", lazy="raise")

    def __repr__(self) -> str:
        return f"<Tenant id={self.id} domain={self.domain!r}>"
