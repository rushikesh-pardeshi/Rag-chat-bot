"""backend/utils/ids.py

Role: ID and timestamp column helpers, so every table declares them the same way.

UUIDv4 primary keys, not sequential integers: a document id appears in URLs, and an
enumerable id is an invitation to walk another tenant's documents. Tenant scoping is the
real defence, but a non-guessable id means an IDOR attempt has to be a guess.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Annotated

from sqlalchemy import DateTime, Uuid, func
from sqlalchemy.orm import mapped_column

#: Server-generated UUID primary key.
UuidPk = Annotated[
    uuid.UUID,
    mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4),
]

#: A UUID foreign-key column body. Pair it with the ForeignKey in the model itself.
UuidCol = Annotated[uuid.UUID, mapped_column(Uuid(as_uuid=True))]

#: Set by the database clock, so rows written by different processes stay comparable.
CreatedAt = Annotated[
    datetime,
    mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False),
]

UpdatedAt = Annotated[
    datetime,
    mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    ),
]


def new_id() -> uuid.UUID:
    """A fresh UUIDv4. Use this rather than calling uuid4 at each site."""
    return uuid.uuid4()


def utc_now() -> datetime:
    """Timezone-aware UTC now.

    Audit timestamps must be comparable across processes and stable when re-serialized for
    the integrity hash; a naive datetime silently breaks both.
    """
    return datetime.now(UTC)
