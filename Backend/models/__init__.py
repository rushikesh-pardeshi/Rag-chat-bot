"""backend/models/__init__.py

Role: ORM entities (SQLAlchemy 2.0) mapped to PostgreSQL tables, and the import surface
Alembic relies on.

`alembic/env.py` imports this package for its side effect: a model that is never imported
is absent from `Base.metadata`, and autogenerate would silently emit a migration that drops
its table. Every new model must be added here.
"""

from backend.models.audit_log import (
    GENESIS_HASH,
    AuditAction,
    AuditLog,
    AuditObjectType,
    AuditOutcome,
)
from backend.models.document import (
    EMBEDDING_DIMENSIONS,
    RETRIEVABLE_STATUSES,
    Document,
    DocumentChunk,
    DocumentStatus,
)
from backend.models.query import Query, QueryStatus
from backend.models.tenant import Tenant
from backend.models.user import RefreshToken, User, UserRole

__all__ = [
    "EMBEDDING_DIMENSIONS",
    "GENESIS_HASH",
    "RETRIEVABLE_STATUSES",
    "AuditAction",
    "AuditLog",
    "AuditObjectType",
    "AuditOutcome",
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "Query",
    "QueryStatus",
    "RefreshToken",
    "Tenant",
    "User",
    "UserRole",
]
