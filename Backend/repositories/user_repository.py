"""backend/repositories/user_repository.py

Role: The only place `user_account` and `tenant` rows are queried. ORM statements only —
no raw SQL, no business rules, no HTTP.

Every read takes `tenant_id` as a parameter rather than reading it from anywhere ambient.
A method that could be called without a tenant is a method that will eventually be called
without one, and that is a cross-tenant leak.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.tenant import Tenant
from backend.models.user import User, UserRole


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Tenants -------------------------------------------------------------
    async def get_tenant_by_domain(self, domain: str) -> Tenant | None:
        stmt = select(Tenant).where(Tenant.domain == domain.lower())
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def get_tenant_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        return await self._session.get(Tenant, tenant_id)

    async def create_tenant(self, *, name: str, domain: str) -> Tenant:
        tenant = Tenant(name=name, domain=domain.lower())
        self._session.add(tenant)
        await self._session.flush()
        return tenant

    async def count_tenants(self) -> int:
        stmt = select(func.count()).select_from(Tenant)
        return int((await self._session.execute(stmt)).scalar_one())

    # -- Users ---------------------------------------------------------------
    async def get_by_email(self, email: str, tenant_id: uuid.UUID) -> User | None:
        """Identity lookup. Email is normalized to lower case by the service."""
        stmt = select(User).where(
            User.email == email.lower(), User.tenant_id == tenant_id
        )
        return (await self._session.execute(stmt)).unique().scalar_one_or_none()

    async def get_by_email_any_tenant(self, email: str) -> User | None:
        """Login lookup, used before a tenant is known.

        The tenant is resolved *from the row* and then carried in the token; it is never
        taken from the request. This is the one query without a tenant filter, and it is
        read-only on a globally unique-enough key.
        """
        stmt = select(User).where(User.email == email.lower())
        return (await self._session.execute(stmt)).unique().scalars().first()

    async def get_by_id(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> User | None:
        stmt = select(User).where(User.id == user_id, User.tenant_id == tenant_id)
        return (await self._session.execute(stmt)).unique().scalar_one_or_none()

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        email: str,
        hashed_password: str,
        role: UserRole,
        full_name: str | None = None,
    ) -> User:
        user = User(
            tenant_id=tenant_id,
            email=email.lower(),
            hashed_password=hashed_password,
            role=role,
            full_name=full_name,
        )
        self._session.add(user)
        await self._session.flush()
        return user

    async def set_password(self, user: User, hashed_password: str) -> None:
        user.hashed_password = hashed_password
        await self._session.flush()

    async def list_for_tenant(
        self, tenant_id: uuid.UUID, *, limit: int = 100, offset: int = 0
    ) -> list[User]:
        stmt = (
            select(User)
            .where(User.tenant_id == tenant_id)
            .order_by(User.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self._session.execute(stmt)).unique().scalars())

    async def count_for_tenant(self, tenant_id: uuid.UUID) -> int:
        """How many accounts the tenant has, for the pager over `list_for_tenant`.

        Counted rather than derived from `len(list_for_tenant(...))`: a page is capped, so the
        length of one page says nothing about the total and a pager built on it would stop at
        the first page.
        """
        stmt = (
            select(func.count()).select_from(User).where(User.tenant_id == tenant_id)
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def count_admins(self, tenant_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(User)
            .where(
                User.tenant_id == tenant_id, User.role == UserRole.ADMINISTRATOR
            )
        )
        return int((await self._session.execute(stmt)).scalar_one())

    # -- Brute-force counters ------------------------------------------------
    # Written as UPDATE statements rather than attribute assignment: two concurrent
    # failed logins must not each read 2 and write 3.
    async def record_failed_login(
        self, user_id: uuid.UUID, *, locked_until: datetime | None
    ) -> int:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(
                failed_login_attempts=User.failed_login_attempts + 1,
                locked_until=locked_until,
            )
            .returning(User.failed_login_attempts)
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def reset_failed_logins(
        self, user_id: uuid.UUID, *, last_login_at: datetime
    ) -> None:
        stmt = (
            update(User)
            .where(User.id == user_id)
            .values(
                failed_login_attempts=0,
                locked_until=None,
                last_login_at=last_login_at,
            )
        )
        await self._session.execute(stmt)

    async def apply_lockout(self, user_id: uuid.UUID, locked_until: datetime) -> None:
        stmt = update(User).where(User.id == user_id).values(locked_until=locked_until)
        await self._session.execute(stmt)

    async def set_role(
        self, user_id: uuid.UUID, tenant_id: uuid.UUID, role: UserRole
    ) -> bool:
        """Returns False when no row matched — a cross-tenant id changes nothing."""
        stmt = (
            update(User)
            .where(User.id == user_id, User.tenant_id == tenant_id)
            .values(role=role)
        )
        return (await self._session.execute(stmt)).rowcount == 1
