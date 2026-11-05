"""backend/api/dependencies/database.py

Role: Database session dependency for request lifecycle management.

One session per request, with **three** outcomes rather than two:

  * the handler returns cleanly → commit;
  * the handler raises a `DocQueryError` → **commit**, then re-raise;
  * anything else raises → roll back, then re-raise.

The middle case is the one that matters, and it is not an optimization. A `DocQueryError`
is a *decided outcome* — "these credentials are wrong", "you may not do that", "that file is
not a PDF" — and the code that decided it has already written the rows that record the
decision: the failed-login counter that drives lockout, and the audit entry with
`outcome='FAILURE'` or `'DENIED'`. Rolling those back would mean:

  * a denied access leaves no trace, which `.claude/rules/rbac-enforcement.md` requires
    ("Log all access attempts (authorized and denied)") and which is itself an audit gap;
  * account lockout never engages, because every increment of the failed-attempt counter is
    undone by the very exception that recorded it.

An unexpected exception is the opposite case: nothing decided anything, the state is unknown,
and rolling back is the only safe answer.

The invariant this places on services: **leave the data in a consistent state before raising
a `DocQueryError`.** `document_service` does this explicitly — it marks a document `FAILED`
before re-raising a parse error, rather than leaving it `PROCESSING` forever.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.database import get_session_factory
from backend.core.exceptions import DocQueryError
from backend.core.logging import get_logger

logger = get_logger(__name__)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_session_factory()() as session:
        try:
            yield session
        except DocQueryError:
            # Keep the audit and lockout rows the service wrote on its way out. If the
            # commit itself fails the transaction is already unusable, so fall back to a
            # rollback and let the original domain error surface — replacing it with a
            # database error would hide what the caller actually did wrong.
            try:
                await session.commit()
            except Exception:  # noqa: BLE001 - the original error is the one to report
                logger.warning("db.commit_failed_on_domain_error")
                await session.rollback()
            raise
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


DbSession = Annotated[AsyncSession, Depends(get_db)]
