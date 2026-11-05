"""backend/services/audit_service.py

Role: Audit business rules — required-field validation, the per-tenant hash chain, and
chain verification. Every other service calls `log()`; nothing else writes `audit_log`.

The chain is what makes the log tamper-*evident* rather than merely append-only. Each entry
hashes its canonical fields together with the previous entry's `integrity_hash`, so:

  * editing a row breaks that row's own hash;
  * removing a row breaks the next row's `previous_hash`;
  * inserting a row cannot produce a hash that also satisfies its successor.

Canonicalization is the fragile part. Field order is fixed, timestamps are UTC ISO-8601 to
microseconds, and separators are explicit — any drift there produces verification failures
that look exactly like tampering.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from backend.core.exceptions import ValidationError
from backend.core.logging import get_logger, sanitize_log_value
from backend.models.audit_log import (
    GENESIS_HASH,
    AuditAction,
    AuditLog,
    AuditObjectType,
    AuditOutcome,
)
from backend.repositories.audit_repository import AuditRepository
from backend.utils.ids import new_id, utc_now

logger = get_logger(__name__)

#: Fixed field order for the canonical form. Changing this list invalidates every existing
#: chain, so it is versioned rather than edited — see `_CANONICAL_VERSION`.
_CANONICAL_FIELDS = (
    "log_id",
    "tenant_id",
    "user_id",
    "actor_email",
    "action",
    "object_type",
    "object_id",
    "outcome",
    "occurred_at",
    "ip_address",
    "user_agent",
    "detail",
)
_CANONICAL_VERSION = "v1"

#: Longest user-agent kept. A megabyte of attacker-chosen header should not become a
#: megabyte of audit row, and the tail carries no forensic value.
_MAX_USER_AGENT = 512

#: Substituted when a request carries no usable source address (a task, a CLI script).
#: `inet` rejects the empty string, and a NULL would break the NOT NULL the schema wants.
UNKNOWN_IP = "0.0.0.0"  # noqa: S104 - stored as a marker value, never bound to


def _normalize_ip(value: str | None) -> str:
    """Return a value PostgreSQL's `inet` will accept, or the marker.

    `audit_log.ip_address` is `inet`, so a non-address string makes the INSERT raise
    `DataError` — and that happens *after* the audited action has run, so the caller sees a
    500 on an operation that actually succeeded and the row is rolled back with it. The
    audit write must never be the reason a request fails.

    `get_request_context` reads `request.client.host` and deliberately ignores
    `X-Forwarded-For`, so this is not an attacker-controlled path today. It is still
    reachable: an ASGI harness reports the peer as `testclient`, and a reverse proxy
    configured with `--proxy-headers` can hand over whatever it likes. An unparseable
    address is recorded as unknown rather than dropped, because "this happened and the
    source was not usable" is the honest entry.
    """
    if not value:
        return UNKNOWN_IP
    candidate = value.strip()
    # A `host:port` peer is common from proxies; `inet` rejects the port.
    if candidate.count(":") == 1 and "." in candidate:
        candidate = candidate.rsplit(":", 1)[0]
    try:
        return str(ipaddress.ip_address(candidate))
    except ValueError:
        logger.warning("audit.ip_unparseable", value=sanitize_log_value(value)[:64])
        return UNKNOWN_IP


@dataclass(frozen=True, slots=True)
class AuditPage:
    """A page of audit entries plus the total, for the compliance table."""

    items: tuple[AuditLog, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class AuditQuery:
    """Filters for a compliance read. Every field is optional and every one is bound.

    A dataclass rather than a long parameter list because the route already has a validated
    Pydantic filter object and this is the shape it maps onto — and because adding a filter
    later should not change the signature of the read.
    """

    action: AuditAction | None = None
    outcome: AuditOutcome | None = None
    #: Whose entries to return — the *subject* of the rows, not the person asking.
    user_id: uuid.UUID | None = None
    object_type: str | None = None
    occurred_from: datetime | None = None
    occurred_to: datetime | None = None
    limit: int = 50
    offset: int = 0


@dataclass(frozen=True, slots=True)
class ChainVerification:
    """Result of walking one tenant's chain."""

    tenant_id: uuid.UUID
    entries_checked: int
    is_valid: bool
    #: `log_id` of the first entry that failed, and why. Empty when valid.
    failures: tuple[tuple[uuid.UUID, str], ...] = ()

    @property
    def first_failure(self) -> tuple[uuid.UUID, str] | None:
        return self.failures[0] if self.failures else None


def _canonical_value(name: str, value: Any) -> str:
    """One field, rendered to a stable string.

    `None` becomes a sentinel rather than the empty string: otherwise a null `user_id` and
    an empty `actor_email` would be indistinguishable in the digest, and two different rows
    could share a hash.
    """
    if value is None:
        return "\x00null"
    if isinstance(value, datetime):
        if value.tzinfo is None:
            # A naive timestamp would hash differently depending on the reader's locale.
            raise ValidationError(f"Audit field {name} must be timezone-aware.")
        # Normalized to UTC before rendering: the same instant must hash identically
        # whether it arrives as +05:30 from the app or as +00:00 from asyncpg.
        return value.astimezone(UTC).isoformat()
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, (AuditAction, AuditOutcome, AuditObjectType)):
        return value.value
    if isinstance(value, dict):
        # sort_keys + fixed separators: dict insertion order must not change the digest.
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return str(value)


def canonical_form(entry: AuditLog) -> str:
    """The exact string that gets hashed. Deterministic for a given row, forever."""
    parts = [_CANONICAL_VERSION]
    for name in _CANONICAL_FIELDS:
        parts.append(f"{name}={_canonical_value(name, getattr(entry, name))}")
    return "\x1f".join(parts)


def compute_integrity_hash(entry: AuditLog, previous_hash: str) -> str:
    """SHA-256 over the canonical form chained to `previous_hash`."""
    payload = f"{previous_hash}\x1e{canonical_form(entry)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class AuditService:
    """Append-only audit writer. Has no update or delete operation, by design."""

    def __init__(self, repository: AuditRepository) -> None:
        self._repo = repository

    async def log(
        self,
        *,
        tenant_id: uuid.UUID,
        action: AuditAction,
        object_type: AuditObjectType | str,
        object_id: str,
        outcome: AuditOutcome,
        actor_email: str,
        user_id: uuid.UUID | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        detail: dict[str, Any] | None = None,
        occurred_at: datetime | None = None,
    ) -> AuditLog:
        """Append one entry and return it.

        Raises `ValidationError` when a required field is missing — an entry that cannot
        answer "who did what to which object, and did it succeed" is not an audit record.
        """
        if not actor_email:
            raise ValidationError("Audit entry requires actor_email.")
        if not object_id:
            raise ValidationError("Audit entry requires object_id.")

        entry = AuditLog(
            log_id=new_id(),
            tenant_id=tenant_id,
            user_id=user_id,
            # Sanitized: an email or user-agent reaching a log line must not be able to
            # forge a second line with an embedded CR/LF.
            actor_email=sanitize_log_value(actor_email)[:320],
            action=action,
            object_type=(
                object_type.value
                if isinstance(object_type, AuditObjectType)
                else sanitize_log_value(object_type)[:32]
            ),
            object_id=sanitize_log_value(object_id)[:128],
            outcome=outcome,
            occurred_at=occurred_at or utc_now(),
            ip_address=_normalize_ip(ip_address),
            user_agent=sanitize_log_value(user_agent or "unknown")[:_MAX_USER_AGENT],
            detail=_sanitize_detail(detail),
        )

        previous_hash = await self._repo.lock_chain(tenant_id)
        entry.previous_hash = previous_hash
        entry.integrity_hash = compute_integrity_hash(entry, previous_hash)
        await self._repo.append(entry)

        logger.info(
            "audit.appended",
            action=action.value,
            outcome=outcome.value,
            object_type=entry.object_type,
        )
        return entry

    # -- Compliance reads ---------------------------------------------------
    async def list_entries(
        self,
        *,
        tenant_id: uuid.UUID,
        filters: AuditQuery,
        actor_email: str,
        actor_user_id: uuid.UUID,
        ip_address: str | None = None,
        user_agent: str | None = None,
    ) -> AuditPage:
        """A filtered page of one tenant's audit log, and an audit row recording the read.

        **Reading the audit log is itself an audited action.** "Who looked at the log, when,
        and with what filter" is a question a regulator asks, and a log that cannot answer it
        about itself has a hole in exactly the place that matters.

        The actor is passed as primitives rather than as a `User`: `auth_service` imports this
        module, so importing anything from it here would be a cycle. The filter fields are
        recorded in `detail` — they are enum values, ids and timestamps, all non-sensitive.

        Authorization happened before this call. Only `require_admin` reaches it, and
        `tenant_id` comes from the caller's own row, so there is no argument here that could
        widen the read to another organization.
        """
        items = await self._repo.list_for_tenant(
            tenant_id,
            action=filters.action,
            outcome=filters.outcome,
            user_id=filters.user_id,
            object_type=filters.object_type,
            occurred_from=filters.occurred_from,
            occurred_to=filters.occurred_to,
            limit=filters.limit,
            offset=filters.offset,
        )
        total = await self._repo.count_for_tenant(
            tenant_id,
            action=filters.action,
            outcome=filters.outcome,
            user_id=filters.user_id,
            object_type=filters.object_type,
            occurred_from=filters.occurred_from,
            occurred_to=filters.occurred_to,
        )

        await self.log(
            tenant_id=tenant_id,
            user_id=actor_user_id,
            action=AuditAction.AUDIT_VIEW,
            object_type=AuditObjectType.AUDIT_EXPORT,
            # No single object was read, so the object id is the tenant whose log it was.
            object_id=str(tenant_id),
            outcome=AuditOutcome.SUCCESS,
            actor_email=actor_email,
            ip_address=ip_address,
            user_agent=user_agent,
            detail={
                "returned": len(items),
                "total": total,
                "filter_action": filters.action.value if filters.action else None,
                "filter_outcome": filters.outcome.value if filters.outcome else None,
                "filter_user_id": str(filters.user_id) if filters.user_id else None,
                "filter_from": (
                    filters.occurred_from.isoformat() if filters.occurred_from else None
                ),
                "filter_to": (
                    filters.occurred_to.isoformat() if filters.occurred_to else None
                ),
            },
        )
        return AuditPage(
            items=tuple(items),
            total=total,
            limit=filters.limit,
            offset=filters.offset,
        )

    async def verify_chain(
        self, tenant_id: uuid.UUID, *, limit: int | None = None
    ) -> ChainVerification:
        """Recompute every hash for a tenant and report every divergence found.

        Walks the whole chain rather than stopping at the first bad entry, so a reviewer
        sees the full extent of the damage in one pass.

        Each row is checked against its own *stored* `previous_hash` and against the
        preceding row's *stored* `integrity_hash`. That localizes a finding to the rows
        actually affected instead of marking every subsequent row as damaged — an edit to
        row 2 of 100,000 should not produce 99,998 findings a reviewer has to read past.
        """
        entries = await self._repo.iter_chain(tenant_id, limit=limit)
        failures: list[tuple[uuid.UUID, str]] = []
        expected_previous = GENESIS_HASH

        for entry in entries:
            if entry.previous_hash != expected_previous:
                failures.append(
                    (entry.log_id, "previous_hash does not match the preceding entry")
                )
            recomputed = compute_integrity_hash(entry, entry.previous_hash)
            if recomputed != entry.integrity_hash:
                failures.append((entry.log_id, "integrity_hash does not match contents"))
            expected_previous = entry.integrity_hash

        return ChainVerification(
            tenant_id=tenant_id,
            entries_checked=len(entries),
            is_valid=not failures,
            failures=tuple(failures),
        )


def _sanitize_detail(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip control characters from string values in `detail`.

    Redaction of sensitive keys happens in `core/logging._redact_and_sanitize` for log
    lines; this is the equivalent for the stored JSONB. Callers are responsible for not
    putting credentials or document text in here at all — this is the second line, not the
    first.
    """
    if not detail:
        return None
    cleaned: dict[str, Any] = {}
    for key, value in detail.items():
        safe_key = sanitize_log_value(str(key))[:64]
        cleaned[safe_key] = (
            sanitize_log_value(value) if isinstance(value, str) else value
        )
    return cleaned
