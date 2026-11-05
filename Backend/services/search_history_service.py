"""backend/services/search_history_service.py

Role: Search history — a role-filtered projection over `query`. Owns the decision about whose
history a caller may read.

There is no `search_history` table
---------------------------------
History *is* the `query` table, read with a filter. `.claude/rules/architecture.md` requires
this: duplicating question and answer text into a second table gives one audited fact two
copies free to drift, and a compliance review that finds them disagreeing cannot tell which
one the user actually saw. So this service holds no storage of its own — only the
authorization rule and the shape of the page.

The one decision this service exists to make
--------------------------------------------
`rbac-enforcement.md`: a Standard User reads their **own history only**; an Administrator reads
all rows **for their tenant**. That is expressed as a choice between two differently-named
repository methods rather than as an optional `user_id` argument, because the failure modes are
not comparable. With one method and an optional filter, forgetting the argument silently
returns everyone's history — a working feature that is a data breach. Here, reading another
user's rows requires calling `list_for_tenant`, which is visible in a diff and in a review.

`user_id` means two different things
------------------------------------
For a Standard User it is *ignored*: the scope is their own id, taken from the authenticated
principal, and a `user_id` in the request cannot widen it. For an Administrator it is an
optional *filter* — "show me what this analyst asked". A Standard User who supplies another
user's id gets their own history back, not an error and not the other user's rows, because the
parameter was never an authorization input in the first place.

Every read is audited, including the denied ones
------------------------------------------------
`HISTORY_VIEW` on each call. A Standard User's read of their own history is as audited as an
Administrator's read of the whole tenant, since "who looked at what customer's questions" is
itself a compliance question (BR-05).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.query import Query, QueryStatus
from backend.models.user import User, UserRole
from backend.repositories.query_repository import QueryRepository
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext

logger = get_logger(__name__)

#: Page size bounds. The maximum is a denial-of-service control as much as a UX one: history
#: rows carry full answer text, and an unbounded `limit` is a cheap way for an authenticated
#: user to make the server serialize the tenant's entire query history in one response.
DEFAULT_PAGE_SIZE: Final = 25
MAX_PAGE_SIZE: Final = 100

#: Longest keyword accepted. A fixed bound, and the repository uses `ILIKE` with `\`, `%` and
#: `_` escaped — never a user-supplied regex (`secure-coding-vapt.md`, ReDoS).
MAX_KEYWORD_LENGTH: Final = 200

#: Marks the `HISTORY_VIEW` row written by `history_stats` rather than by `list_history`, so a
#: reviewer can tell an aggregate read from a page read and `HISTORY_VIEW` counts stay
#: interpretable. Same device as `document_service.AUDIT_PHASE_REINDEX`: a new `AuditAction`
#: member needs `ALTER TYPE` on a native PostgreSQL enum, and `detail` is JSONB.
AUDIT_PHASE_STATS: Final = "stats"


@dataclass(frozen=True, slots=True)
class HistoryPage:
    """One page of history plus the total, so the UI can render a pager.

    `scope` says which authorization path produced this page — `"own"` or `"tenant"`. It is
    returned to the caller so the UI can label an Administrator's view honestly ("all users")
    instead of implying the list is the viewer's own activity.
    """

    items: tuple[Query, ...]
    total: int
    limit: int
    offset: int
    scope: str


@dataclass(frozen=True, slots=True)
class HistoryStats:
    """Outcome counts for a history filter, without paging any rows.

    `scope` carries the same meaning as on `HistoryPage` — `"own"` or `"tenant"` — and matters
    more here: a count with no rows beside it gives the reader nothing to infer the scope from,
    so a Standard User's total has to be labelled as theirs rather than the workspace's.

    `by_status` holds every `QueryStatus`, zeros included. SQL omits a group with no rows and a
    tile row needs them all; see `history_stats`.
    """

    scope: str
    total: int
    by_status: dict[QueryStatus, int]


class SearchHistoryService:
    def __init__(self, queries: QueryRepository, audit: AuditService) -> None:
        self._queries = queries
        self._audit = audit

    async def list_history(
        self,
        *,
        actor: User,
        context: RequestContext,
        user_id: uuid.UUID | None = None,
        status: QueryStatus | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> HistoryPage:
        """A page of history, scoped by role.

        `tenant_id` comes from `actor` in both branches and is never a parameter. Role and
        tenant are independent checks (`rbac-enforcement.md`): an Administrator in tenant A
        reading tenant B's history is the same defect as a Standard User reading a colleague's,
        and the only way to be sure neither is reachable is for the tenant never to be an
        input.
        """
        limit, offset = _normalize_page(limit, offset)
        cleaned_keyword = _normalize_keyword(keyword)
        is_admin = actor.role is UserRole.ADMINISTRATOR

        if is_admin:
            scope = "tenant"
            # `user_id` is an Administrator's filter. None means every user in the tenant.
            effective_user_id = user_id
            items = await self._queries.list_for_tenant(
                actor.tenant_id,
                user_id=effective_user_id,
                status=status,
                keyword=cleaned_keyword,
                created_from=created_from,
                created_to=created_to,
                limit=limit,
                offset=offset,
            )
        else:
            scope = "own"
            # Deliberately discards the requested `user_id`. A Standard User asking for
            # someone else's history gets their own — the parameter is not an authorization
            # input, so there is nothing here to bypass. The attempt is recorded below.
            effective_user_id = actor.id
            items = await self._queries.list_for_user(
                actor.tenant_id,
                actor.id,
                status=status,
                keyword=cleaned_keyword,
                created_from=created_from,
                created_to=created_to,
                limit=limit,
                offset=offset,
            )

        total = await self._queries.count(
            actor.tenant_id,
            user_id=effective_user_id,
            status=status,
            keyword=cleaned_keyword,
            created_from=created_from,
            created_to=created_to,
        )

        foreign_request = (
            not is_admin and user_id is not None and user_id != actor.id
        )
        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=AuditAction.HISTORY_VIEW,
            object_type=AuditObjectType.QUERY,
            # A list read has no single row, so the object is *whose history was read*: the
            # filtered user when there is one, otherwise the tenant. `audit_service.list_logs`
            # records its own collection reads the same way. `object_id` is required and
            # rejects empty, so this cannot be left blank.
            object_id=str(effective_user_id or actor.tenant_id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={
                "scope": scope,
                "returned": len(items),
                "total": total,
                "limit": limit,
                "offset": offset,
                # Whether a keyword was used, not what it was. A history keyword is user text
                # and can name a customer or an account number.
                "keyword_used": cleaned_keyword is not None,
                "status_filter": status.value if status else None,
                # A Standard User who asked for another user's history. Not an error — the
                # scope was already narrowed to their own — but it is worth a trace, because
                # it is what an IDOR probe looks like from the inside.
                "foreign_user_requested": foreign_request,
            },
        )

        if foreign_request:
            logger.warning(
                "history.foreign_user_id_ignored",
                actor_id=str(actor.id),
                # The requested id is a UUID from the query string, not free text — no CR/LF
                # can reach the log through it.
                requested_user_id=str(user_id),
            )

        logger.info(
            "history.listed", scope=scope, returned=len(items), total=total
        )
        return HistoryPage(
            items=tuple(items), total=total, limit=limit, offset=offset, scope=scope
        )

    async def history_stats(
        self,
        *,
        actor: User,
        context: RequestContext,
        user_id: uuid.UUID | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> HistoryStats:
        """How many questions fall in each outcome, under the caller's own scope.

        Exists to keep the audit trail readable. The breakdown was previously assembled in the
        browser by calling `list_history` once per outcome with `limit=1` and reading `total`,
        which wrote a `HISTORY_VIEW` row per call — four rows for one filter change, three of
        them counting reads indistinguishable from a person reading somebody's history. The
        trail is the product here, so "how often was this history viewed" has to be answerable
        from it (`.claude/rules/audit-log-immutability.md`). One request, one row, one
        `GROUP BY`.

        **Scoped exactly as `list_history` is, and for the same reason.** `tenant_id` comes
        from `actor` and is never a parameter, and a Standard User's requested `user_id` is
        discarded in favour of their own id rather than refused — the parameter is not an
        authorization input, so there is nothing here to bypass, and the attempt is recorded.
        A count is still a disclosure: telling a Standard User how many questions a colleague
        asked would leak the shape of that colleague's activity without showing a single row.

        Takes no `status` filter: the result *is* the breakdown by status, and narrowing to one
        before grouping would return that single figure, which `list_history` already gives.
        """
        cleaned_keyword = _normalize_keyword(keyword)
        is_admin = actor.role is UserRole.ADMINISTRATOR
        scope = "tenant" if is_admin else "own"
        effective_user_id = user_id if is_admin else actor.id

        counts = await self._queries.count_by_status(
            actor.tenant_id,
            user_id=effective_user_id,
            keyword=cleaned_keyword,
            created_from=created_from,
            created_to=created_to,
        )
        # Zeros filled here, not in the repository: SQL omits a group with no rows, and a
        # caller rendering tiles needs every outcome present. Which statuses exist is a
        # presentation contract, so it belongs on this side.
        by_status = {status: counts.get(status, 0) for status in QueryStatus}
        total = sum(by_status.values())

        foreign_request = not is_admin and user_id is not None and user_id != actor.id
        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=AuditAction.HISTORY_VIEW,
            object_type=AuditObjectType.QUERY,
            object_id=str(effective_user_id or actor.tenant_id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail={
                "scope": scope,
                "total": total,
                # Marks the row as the aggregate read rather than a page read, so a reviewer
                # can tell them apart and `HISTORY_VIEW` counts stay interpretable. Same
                # device as `document_service.AUDIT_PHASE_REINDEX`: a new `AuditAction` member
                # would need `ALTER TYPE` on a native enum, and `detail` is JSONB.
                "phase": AUDIT_PHASE_STATS,
                "keyword_used": cleaned_keyword is not None,
                "foreign_user_requested": foreign_request,
            },
        )

        if foreign_request:
            logger.warning(
                "history.foreign_user_id_ignored",
                actor_id=str(actor.id),
                requested_user_id=str(user_id),
            )

        logger.info("history.stats", scope=scope, total=total)
        return HistoryStats(scope=scope, total=total, by_status=by_status)


def _normalize_page(limit: int, offset: int) -> tuple[int, int]:
    """Clamp pagination into range rather than rejecting it.

    A `limit` of 0 or a negative `offset` is a client bug, not an attack, and answering it with
    a 422 makes a pager awkward to write. An oversized `limit` is *capped* rather than honoured
    — see `MAX_PAGE_SIZE`.
    """
    safe_limit = min(max(1, int(limit)), MAX_PAGE_SIZE)
    safe_offset = max(0, int(offset))
    return safe_limit, safe_offset


def _normalize_keyword(keyword: str | None) -> str | None:
    """Trim, bound, and reject nothing silently.

    Returns None for a blank keyword so the repository skips the `ILIKE` entirely — filtering
    on `%%` is a full scan that returns everything, which is the same result at more cost.
    """
    if keyword is None:
        return None
    cleaned = keyword.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_KEYWORD_LENGTH:
        raise ValidationError(
            f"A history keyword may not exceed {MAX_KEYWORD_LENGTH} characters."
        )
    return cleaned
