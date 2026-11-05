"""backend/repositories/auth_repository.py

Role: The refresh-token store. Only place `refresh_token` rows are queried.

Rotation is the reason this table exists: every refresh mints a new token and marks the
presented one revoked, pointing at its successor. A replayed old token therefore proves
theft, not just staleness — the service can revoke the whole family.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.user import RefreshToken
from backend.utils.ids import utc_now


class AuthRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def store(
        self,
        *,
        token_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime,
    ) -> RefreshToken:
        """Persist a newly issued token. `token_id` is the JWT's `jti`."""
        row = RefreshToken(
            id=token_id,
            tenant_id=tenant_id,
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self._session.add(row)
        await self._session.flush()
        return row

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        """Look up by digest, never by the token itself — the token is not stored."""
        stmt = select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def rotate(
        self,
        *,
        old_token_id: uuid.UUID,
        new_token_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        new_token_hash: str,
        new_expires_at: datetime,
    ) -> RefreshToken:
        """Revoke the presented token and issue its successor in one transaction.

        The `revoked_at IS NULL` predicate makes this the concurrency check too: two
        parallel refreshes with the same token cannot both rotate it, because the second
        UPDATE matches no row and the caller sees a reuse.
        """
        revoked = await self._session.execute(
            update(RefreshToken)
            .where(RefreshToken.id == old_token_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=utc_now(), replaced_by_id=new_token_id)
        )
        if revoked.rowcount != 1:
            raise RefreshTokenReuse(str(old_token_id))
        return await self.store(
            token_id=new_token_id,
            tenant_id=tenant_id,
            user_id=user_id,
            token_hash=new_token_hash,
            expires_at=new_expires_at,
        )

    async def revoke(self, token_id: uuid.UUID) -> bool:
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.id == token_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=utc_now())
        )
        return (await self._session.execute(stmt)).rowcount == 1

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> int:
        """Kill every live session for a user. Used on detected reuse and on role change."""
        stmt = (
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=utc_now())
        )
        return int((await self._session.execute(stmt)).rowcount)

    async def is_revoked(self, token_id: uuid.UUID) -> bool:
        """True when the token is revoked *or unknown*.

        An unknown id is treated as revoked: the safe answer to "is this token still
        good" for a row that does not exist is no.
        """
        stmt = select(RefreshToken.revoked_at).where(RefreshToken.id == token_id)
        result = (await self._session.execute(stmt)).one_or_none()
        return result is None or result[0] is not None

    async def delete_expired(self, *, before: datetime) -> int:
        """Housekeeping. Expired rows carry no authority and only grow the table."""
        stmt = delete(RefreshToken).where(RefreshToken.expires_at < before)
        return int((await self._session.execute(stmt)).rowcount)


class RefreshTokenReuse(Exception):
    """A revoked or already-rotated refresh token was presented.

    Not a `DocQueryError`: the service decides what this means (revoke the family, audit
    it) before any HTTP status is chosen.
    """
