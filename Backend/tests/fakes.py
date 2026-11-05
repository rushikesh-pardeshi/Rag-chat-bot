"""backend/tests/fakes.py

Role: In-memory stand-ins for the repository layer.

Why these exist: the phase gate asks for lockout, rotation, tenant-isolation and RBAC
tests, and none of those behaviours live in the database — they live in the services. Faking
at the repository boundary keeps those tests runnable on a laptop with no Postgres, while
the queries themselves are covered by the `requires_db` integration tests.

Each fake mirrors the *contract* of the real repository, including the parts that matter to
a caller: `record_failed_login` increments and returns the new count, `rotate` raises
`RefreshTokenReuse` on a second attempt, and every getter is tenant-scoped. A fake that is
more permissive than the real thing would let a tenant-isolation bug pass.

`FakeS3Client` is the one exception to "fake at the repository boundary": it stands in for the
*boto3 client*, one level below `S3StorageService`, so key construction and filename
sanitization stay real code under test. See its docstring.

`FakeBedrockClient` and `FakePineconeIndex` are exceptions for the same reason, and they are
what make the RAG suite runnable with `PINECONE_API_KEY` blank and no AWS credentials.
`FakePineconeIndex` sits below `VectorStore`, so namespace construction, batching, the metadata
allowlist and match normalization all stay real code under test — the parts that, if wrong,
leak one tenant's passages into another's answer.
"""

from __future__ import annotations

import math
import re
import uuid
import zlib
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from backend.models.audit_log import GENESIS_HASH, AuditLog
from backend.models.document import Document, DocumentChunk, DocumentStatus
from backend.models.query import Query, QueryStatus
from backend.models.tenant import Tenant
from backend.models.user import RefreshToken, User, UserRole
from backend.repositories.analytics_repository import (
    ActiveUser,
    DocumentUsage,
    KeywordCount,
    QuerySummary,
    TrendGranularity,
    TrendPoint,
)
from backend.repositories.auth_repository import RefreshTokenReuse
from backend.repositories.query_repository import RetrievedChunk
from backend.utils.ids import utc_now
from backend.utils.vector_store import FORBIDDEN_METADATA_KEYS


class FakeUserRepository:
    """Dict-backed `UserRepository`. Keys mirror the real unique constraints."""

    def __init__(self) -> None:
        self.users: dict[uuid.UUID, User] = {}
        self.tenants: dict[uuid.UUID, Tenant] = {}

    # -- Test helpers (not part of the real contract) ------------------------
    def add_tenant(
        self, *, domain: str, name: str = "Test Bank", active: bool = True
    ) -> Tenant:
        tenant = Tenant(id=uuid.uuid4(), name=name, domain=domain, is_active=active)
        self.tenants[tenant.id] = tenant
        return tenant

    def add_user(
        self,
        *,
        tenant_id: uuid.UUID,
        email: str,
        hashed_password: str,
        role: UserRole = UserRole.STANDARD_USER,
        is_active: bool = True,
        failed_login_attempts: int = 0,
        locked_until: datetime | None = None,
    ) -> User:
        user = User(
            id=uuid.uuid4(),
            tenant_id=tenant_id,
            email=email,
            full_name=None,
            hashed_password=hashed_password,
            role=role,
            is_active=is_active,
            failed_login_attempts=failed_login_attempts,
            locked_until=locked_until,
        )
        self.users[user.id] = user
        return user

    # -- Tenants -------------------------------------------------------------
    async def get_tenant_by_domain(self, domain: str) -> Tenant | None:
        return next(
            (t for t in self.tenants.values() if t.domain == domain.lower()), None
        )

    async def get_tenant_by_id(self, tenant_id: uuid.UUID) -> Tenant | None:
        return self.tenants.get(tenant_id)

    async def create_tenant(self, *, name: str, domain: str) -> Tenant:
        return self.add_tenant(name=name, domain=domain)

    async def count_tenants(self) -> int:
        return len(self.tenants)

    # -- Users ---------------------------------------------------------------
    async def get_by_email(self, email: str, tenant_id: uuid.UUID) -> User | None:
        return next(
            (
                u
                for u in self.users.values()
                if u.email == email and u.tenant_id == tenant_id
            ),
            None,
        )

    async def get_by_email_any_tenant(self, email: str) -> User | None:
        return next((u for u in self.users.values() if u.email == email), None)

    async def get_by_id(self, user_id: uuid.UUID, tenant_id: uuid.UUID) -> User | None:
        user = self.users.get(user_id)
        # The tenant filter is the point of the test — a fake that ignored it would make
        # every cross-tenant assertion vacuous.
        if user is None or user.tenant_id != tenant_id:
            return None
        return user

    async def create(
        self,
        *,
        tenant_id: uuid.UUID,
        email: str,
        hashed_password: str,
        role: UserRole,
        full_name: str | None = None,
    ) -> User:
        user = self.add_user(
            tenant_id=tenant_id,
            email=email,
            hashed_password=hashed_password,
            role=role,
        )
        user.full_name = full_name
        return user

    async def set_password(self, user: User, hashed_password: str) -> User:
        user.hashed_password = hashed_password
        return user

    async def list_for_tenant(
        self, tenant_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> list[User]:
        rows = [u for u in self.users.values() if u.tenant_id == tenant_id]
        return rows[offset : offset + limit]

    async def count_for_tenant(self, tenant_id: uuid.UUID) -> int:
        """Counted over every row, not over the page — the pager needs the total.

        Deliberately not `len(await self.list_for_tenant(...))`: that would inherit the page
        cap and the pager built on it would always believe it was on the last page, which is
        the bug this method exists to make impossible in the real repository too.
        """
        return sum(1 for u in self.users.values() if u.tenant_id == tenant_id)

    async def count_admins(self, tenant_id: uuid.UUID) -> int:
        return sum(
            1
            for u in self.users.values()
            if u.tenant_id == tenant_id
            and u.role is UserRole.ADMINISTRATOR
            and u.is_active
        )

    async def record_failed_login(
        self, user_id: uuid.UUID, *, locked_until: datetime | None
    ) -> int:
        user = self.users[user_id]
        user.failed_login_attempts += 1
        user.locked_until = locked_until
        return user.failed_login_attempts

    async def reset_failed_logins(
        self, user_id: uuid.UUID, *, last_login_at: datetime | None = None
    ) -> None:
        user = self.users[user_id]
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = last_login_at

    async def apply_lockout(self, user_id: uuid.UUID, until: datetime) -> None:
        self.users[user_id].locked_until = until

    async def set_role(
        self, user_id: uuid.UUID, tenant_id: uuid.UUID, role: UserRole
    ) -> bool:
        user = await self.get_by_id(user_id, tenant_id)
        if user is None:
            return False
        user.role = role
        return True


class FakeAuthRepository:
    """Dict-backed refresh-token store with the real rotation guard."""

    def __init__(self) -> None:
        self.tokens: dict[uuid.UUID, RefreshToken] = {}

    async def store(
        self,
        *,
        token_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        token_hash: str,
        expires_at: datetime,
    ) -> RefreshToken:
        token = RefreshToken(
            id=token_id,
            tenant_id=tenant_id,
            user_id=user_id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self.tokens[token_id] = token
        return token

    async def get_by_hash(self, token_hash: str) -> RefreshToken | None:
        return next(
            (t for t in self.tokens.values() if t.token_hash == token_hash), None
        )

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
        old = self.tokens.get(old_token_id)
        # Mirrors the real conditional UPDATE: revoking an already-revoked row affects
        # zero rows, which is how a replay is detected.
        if old is None or old.revoked_at is not None:
            raise RefreshTokenReuse(str(old_token_id))
        old.revoked_at = utc_now()
        old.replaced_by_id = new_token_id
        return await self.store(
            token_id=new_token_id,
            tenant_id=tenant_id,
            user_id=user_id,
            token_hash=new_token_hash,
            expires_at=new_expires_at,
        )

    async def revoke(self, token_id: uuid.UUID) -> bool:
        token = self.tokens.get(token_id)
        if token is None or token.revoked_at is not None:
            return False
        token.revoked_at = utc_now()
        return True

    async def revoke_all_for_user(self, user_id: uuid.UUID) -> int:
        revoked = 0
        for token in self.tokens.values():
            if token.user_id == user_id and token.revoked_at is None:
                token.revoked_at = utc_now()
                revoked += 1
        return revoked

    async def is_revoked(self, token_hash: str) -> bool:
        token = await self.get_by_hash(token_hash)
        return token is None or token.revoked_at is not None

    async def delete_expired(self, *, before: datetime | None = None) -> int:
        cutoff = before or utc_now()
        stale = [k for k, t in self.tokens.items() if t.expires_at < cutoff]
        for key in stale:
            del self.tokens[key]
        return len(stale)


class FakeAuditRepository:
    """List-backed audit store. Append-only, like the real one — no update, no delete."""

    def __init__(self) -> None:
        self.entries: list[AuditLog] = []

    async def lock_chain(self, tenant_id: uuid.UUID) -> str:
        return await self.last_hash(tenant_id)

    async def last_hash(self, tenant_id: uuid.UUID) -> str:
        for entry in reversed(self.entries):
            if entry.tenant_id == tenant_id:
                return entry.integrity_hash
        return GENESIS_HASH

    async def append(self, entry: AuditLog) -> AuditLog:
        self.entries.append(entry)
        return entry

    async def iter_chain(
        self, tenant_id: uuid.UUID, *, limit: int | None = None
    ) -> list[AuditLog]:
        rows = [e for e in self.entries if e.tenant_id == tenant_id]
        return rows if limit is None else rows[:limit]

    async def count_for_tenant(self, tenant_id: uuid.UUID, **_: object) -> int:
        return len([e for e in self.entries if e.tenant_id == tenant_id])

    async def get(self, log_id: uuid.UUID, tenant_id: uuid.UUID) -> AuditLog | None:
        return next(
            (
                e
                for e in self.entries
                if e.log_id == log_id and e.tenant_id == tenant_id
            ),
            None,
        )

    # -- Test helpers --------------------------------------------------------
    def actions(self) -> list[str]:
        return [e.action.value for e in self.entries]

    def outcomes(self) -> list[str]:
        return [e.outcome.value for e in self.entries]

    async def list_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        action: Any = None,
        outcome: Any = None,
        user_id: uuid.UUID | None = None,
        object_type: str | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[AuditLog]:
        """Filtered page, newest first — the compliance read.

        Ordering is the reverse of insertion, matching the real `occurred_at DESC`. A fake
        that returned insertion order would let a test pass while the route showed the oldest
        entries on page one.
        """
        rows = [e for e in self.entries if e.tenant_id == tenant_id]
        if action is not None:
            rows = [e for e in rows if e.action is action]
        if outcome is not None:
            rows = [e for e in rows if e.outcome is outcome]
        if user_id is not None:
            rows = [e for e in rows if e.user_id == user_id]
        if object_type is not None:
            rows = [e for e in rows if e.object_type == object_type]
        if occurred_from is not None:
            rows = [e for e in rows if e.occurred_at >= occurred_from]
        if occurred_to is not None:
            rows = [e for e in rows if e.occurred_at <= occurred_to]
        rows.reverse()
        return rows[offset : offset + limit]


class FakeS3Client:
    """Dict-backed stand-in for the *boto3 client*, not for `S3StorageService`.

    Faking one level lower is deliberate. The interesting behaviour on the storage path is in
    `S3StorageService` itself — key construction, filename sanitization, refusing to overwrite
    an existing key — and a fake that replaced the whole service would replace exactly the code
    under test. Substituting the client keeps `build_key` and `sanitize_filename` real while
    still guaranteeing no test reaches AWS.

    `put_object` honours `IfNoneMatch="*"` the way S3 does, so the overwrite guard is exercised
    rather than assumed.
    """

    def __init__(self, *, supports_if_none_match: bool = True) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.presigned: list[tuple[str, int]] = []
        #: Keys passed to a delete call, in order — so a test can assert the purge reached
        #: storage rather than only the database.
        self.deleted: list[str] = []
        #: Set to a list of `{"Key", "Code"}` dicts to model S3's *partial* failure: a
        #: version it refuses to remove is reported here and the call does **not** raise.
        #: Object-lock retention and a policy denying `s3:DeleteObjectVersion` both land
        #: this way, which is why the real code has to read the response.
        self.delete_errors: list[dict[str, Any]] = []
        self._supports_if_none_match = supports_if_none_match
        #: Set to an exception instance to make the next call fail, for the StorageError path.
        self.fail_with: Exception | None = None

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        if "IfNoneMatch" in kwargs:
            if not self._supports_if_none_match:
                # What an older botocore does with an unknown parameter. `S3StorageService`
                # catches this and falls back to HEAD-then-PUT.
                raise TypeError("Unknown parameter in input: IfNoneMatch")
            if kwargs["Key"] in self.objects:
                raise RuntimeError("PreconditionFailed")
            kwargs.pop("IfNoneMatch")
        key = kwargs["Key"]
        self.objects[key] = {
            "Body": kwargs["Body"],
            "ContentType": kwargs.get("ContentType"),
            "ServerSideEncryption": kwargs.get("ServerSideEncryption"),
            "Metadata": kwargs.get("Metadata", {}),
        }
        return {"ETag": '"fake"'}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["Key"] not in self.objects:
            raise RuntimeError("404")
        return {"ContentLength": len(self.objects[kwargs["Key"]]["Body"])}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        import io

        if kwargs["Key"] not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[kwargs["Key"]]["Body"])}

    def generate_presigned_url(self, _operation: str, **kwargs: Any) -> str:
        if self.fail_with is not None:
            raise self.fail_with
        key = kwargs["Params"]["Key"]
        expires = int(kwargs["ExpiresIn"])
        self.presigned.append((key, expires))
        return f"https://fake-s3.invalid/{key}?X-Amz-Expires={expires}"

    # -- Deletion, as a *versioned* bucket does it ---------------------------
    #
    # `S3_VERSIONING_ENABLED` is on in this project, and the behaviour that matters is the
    # one that surprises people: `delete_object` on a versioned bucket writes a delete
    # marker and keeps every version. So this fake models versions, and
    # `S3StorageService.delete_object` is only correct here if it enumerates them — which
    # is the whole point of the method.
    def get_paginator(self, operation: str) -> Any:
        if operation != "list_object_versions":
            raise ValueError(f"no fake paginator for {operation}")
        client = self

        class _Paginator:
            def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
                if client.fail_with is not None:
                    raise client.fail_with
                prefix = kwargs.get("Prefix", "")
                versions = [
                    {"Key": key, "VersionId": version}
                    for key, stored in sorted(client.objects.items())
                    for version in stored.get("versions", ["v1"])
                    if key.startswith(prefix)
                ]
                return [{"Versions": versions, "DeleteMarkers": []}]

        return _Paginator()

    def delete_objects(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        if self.delete_errors:
            # Partial success, as S3 does it: the refused versions come back in `Errors`,
            # the call returns 200, and nothing was removed.
            return {"Deleted": [], "Errors": self.delete_errors}
        deleted = []
        for target in kwargs["Delete"]["Objects"]:
            self.objects.pop(target["Key"], None)
            deleted.append(target)
        self.deleted.extend(target["Key"] for target in deleted)
        return {"Deleted": deleted}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        self.objects.pop(kwargs["Key"], None)
        self.deleted.append(kwargs["Key"])
        return {}

    # -- Test helpers --------------------------------------------------------
    def keys(self) -> list[str]:
        return sorted(self.objects)


class FakeDocumentRepository:
    """Dict-backed `DocumentRepository`, tenant filter included on every method.

    The tenant filter is what makes an IDOR test meaningful, so it is implemented here rather
    than skipped for brevity — a fake that ignored `tenant_id` would make every cross-tenant
    assertion in the suite pass regardless of the service.

    `created_at` and `updated_at` are stamped on create because the real columns have server
    defaults. Leaving them None would make this fake more permissive than the database and
    hide a schema serialization bug.
    """

    def __init__(self) -> None:
        self.documents: dict[uuid.UUID, Document] = {}
        self.chunks: list[DocumentChunk] = []
        #: Every callback `after_commit` was handed, in order. Kept so a test can assert that
        #: ingestion was deferred to a commit rather than dispatched inline.
        self.after_commit_callbacks: list[Callable[[], None]] = []
        #: `(tenant_id, name)` per `lock_name` call, in order.
        self.name_locks: list[tuple[uuid.UUID, str]] = []
        #: Names of the mutating calls and of `lock_name`, in the order they happened. The
        #: lock-ordering assertion in `test_document_service.py` reads this: `lock_name` has
        #: to come before the first write or it does not prevent the deadlock it exists for.
        self.write_order: list[str] = []

    async def create(
        self,
        *,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        name: str,
        s3_key: str,
        content_type: str,
        file_size: int,
        uploaded_by: uuid.UUID,
        upload_date: Any,
        version: int,
        content_hash: str,
        status: DocumentStatus = DocumentStatus.PROCESSING,
        page_count: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Document:
        now = utc_now()
        document = Document(
            id=document_id,
            tenant_id=tenant_id,
            name=name,
            s3_key=s3_key,
            content_type=content_type,
            file_size=file_size,
            page_count=page_count,
            uploaded_by=uploaded_by,
            upload_date=upload_date,
            status=status,
            version=version,
            content_hash=content_hash,
            doc_metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self.documents[document_id] = document
        return document

    async def set_status(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID, status: DocumentStatus
    ) -> bool:
        self.write_order.append("set_status")
        document = self.documents.get(document_id)
        if document is None or document.tenant_id != tenant_id:
            return False
        document.status = status
        return True

    async def set_parse_result(
        self,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        status: DocumentStatus,
        page_count: int | None,
        metadata: dict[str, Any],
    ) -> bool:
        self.write_order.append("set_parse_result")
        document = self.documents.get(document_id)
        if document is None or document.tenant_id != tenant_id:
            return False
        document.status = status
        document.page_count = page_count
        document.doc_metadata = metadata
        return True

    async def add_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        self.write_order.append("add_chunks")
        for chunk in chunks:
            chunk.created_at = utc_now()
        self.chunks.extend(chunks)
        return len(chunks)

    async def mark_chunks_embedded(
        self,
        chunk_ids: Sequence[uuid.UUID],
        tenant_id: uuid.UUID,
        *,
        embedded_at: datetime,
    ) -> int:
        """Tenant-filtered like the real UPDATE, so a wrong-tenant id updates nothing."""
        wanted = set(chunk_ids)
        updated = 0
        for chunk in self.chunks:
            if chunk.id in wanted and chunk.tenant_id == tenant_id:
                chunk.embedded_at = embedded_at
                updated += 1
        return updated

    async def refresh(self, document: Document) -> Document:
        """A genuine no-op here, and only here.

        The real method exists because bulk `UPDATE`s leave the identity-mapped instance stale.
        This fake mutates the same object every method hands back, so it is never stale — there
        is nothing to re-read. The method exists so the service's call site is exercised.
        """
        return document

    async def get(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Document | None:
        document = self.documents.get(document_id)
        if document is None or document.tenant_id != tenant_id:
            return None
        return document

    async def list_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        status: DocumentStatus | None = None,
        statuses: Sequence[DocumentStatus] | None = None,
        name_contains: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Document]:
        rows = self._filtered(
            tenant_id, status=status, statuses=statuses, name_contains=name_contains
        )
        rows.sort(key=lambda d: (d.upload_date, str(d.id)), reverse=True)
        return rows[offset : offset + limit]

    async def count_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        status: DocumentStatus | None = None,
        statuses: Sequence[DocumentStatus] | None = None,
        name_contains: str | None = None,
    ) -> int:
        return len(
            self._filtered(
                tenant_id, status=status, statuses=statuses, name_contains=name_contains
            )
        )

    async def next_version(self, name: str, tenant_id: uuid.UUID) -> int:
        """Counts every status, deleted included — same as the real MAX(version) + 1."""
        versions = [
            d.version
            for d in self.documents.values()
            if d.tenant_id == tenant_id and d.name == name
        ]
        return (max(versions) if versions else 0) + 1

    async def list_versions(self, name: str, tenant_id: uuid.UUID) -> list[Document]:
        rows = [
            d
            for d in self.documents.values()
            if d.tenant_id == tenant_id and d.name == name
        ]
        rows.sort(key=lambda d: d.version)
        return rows

    async def lock_name(self, name: str, tenant_id: uuid.UUID) -> None:
        """Recorded, not enforced. There is no concurrency here to serialize.

        What it defends against — two pipelines ingesting two versions of one name at once —
        needs two transactions, so the assertion that matters lives in the `requires_db`
        tests. Recording the calls still pins that the pipeline *takes* the lock, and in the
        right place: `test_ingestion_to_query.py` asserts it happens before the first write.
        """
        self.name_locks.append((tenant_id, name))
        self.write_order.append("lock_name")

    async def list_earlier_versions(
        self, name: str, tenant_id: uuid.UUID, *, before_version: int
    ) -> list[Document]:
        rows = [
            d
            for d in self.documents.values()
            if d.tenant_id == tenant_id
            and d.name == name
            and d.version < before_version
            # Excluded here exactly as the SQL excludes it, which is what makes version
            # supersession idempotent — a second pass finds nothing left to retire.
            and d.status is not DocumentStatus.DELETED
        ]
        rows.sort(key=lambda d: d.version, reverse=True)
        return rows

    def after_commit(self, callback: Callable[[], None]) -> None:
        """Fired immediately: there is no transaction here to wait for.

        The real repository defers to the session's commit so the in-process ingestion pool
        does not race it. A test using this fake has no second session and no commit, so
        running the callback now is the faithful stand-in — and the few tests that care about
        the deferral assert on `calls` instead.
        """
        self.after_commit_callbacks.append(callback)
        callback()

    async def get_chunks(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[DocumentChunk]:
        rows = [
            c
            for c in self.chunks
            if c.document_id == document_id and c.tenant_id == tenant_id
        ]
        rows.sort(key=lambda c: c.chunk_index)
        return rows

    async def count_chunks(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> int:
        return len(await self.get_chunks(document_id, tenant_id))

    async def delete_chunks(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> int:
        keep = [
            c
            for c in self.chunks
            if not (c.document_id == document_id and c.tenant_id == tenant_id)
        ]
        removed = len(self.chunks) - len(keep)
        self.chunks = keep
        return removed

    async def delete_document(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> bool:
        """Drop the row, and its chunks with it — the real schema cascades.

        The tenant filter is part of the behaviour being faked, not an optimisation: a purge
        aimed at another organization's id must remove nothing and report `False`.
        """
        document = self.documents.get(document_id)
        if document is None or document.tenant_id != tenant_id:
            return False
        del self.documents[document_id]
        self.chunks = [c for c in self.chunks if c.document_id != document_id]
        return True

    async def exists_with_hash(self, content_hash: str, tenant_id: uuid.UUID) -> bool:
        return any(
            d.content_hash == content_hash and d.tenant_id == tenant_id
            for d in self.documents.values()
        )

    def _filtered(
        self,
        tenant_id: uuid.UUID,
        *,
        status: DocumentStatus | None,
        statuses: Sequence[DocumentStatus] | None,
        name_contains: str | None,
    ) -> list[Document]:
        rows = [d for d in self.documents.values() if d.tenant_id == tenant_id]
        if status is not None:
            rows = [d for d in rows if d.status is status]
        if statuses:
            allowed = set(statuses)
            rows = [d for d in rows if d.status in allowed]
        if name_contains:
            # Literal substring, case-insensitive: the real query escapes `%` and `_` so a
            # caller's wildcard matches itself. A fake using `fnmatch` would be laxer.
            needle = name_contains.lower()
            rows = [d for d in rows if needle in d.name.lower()]
        return rows


class FakeQueryRepository:
    """Dict-backed `QueryRepository`.

    `list_for_user` and `list_for_tenant` stay two separate methods here for the same reason
    they are two in the real repository: collapsing them into one with an optional `user_id`
    would let a test pass against a fake whose default is "everyone's history".

    `documents` is optional but nearly always worth passing: `load_chunks` is the citability
    re-check that `rag-answer-grounding.md` requires, and it can only *do* that check if it can
    see document statuses. Without it, `load_chunks` returns nothing — deliberately, so a
    grounding test that forgot to wire the repositories together fails loudly rather than
    passing against a fake that quotes soft-deleted documents.
    """

    def __init__(self, documents: FakeDocumentRepository | None = None) -> None:
        self.queries: dict[uuid.UUID, Query] = {}
        self.documents = documents

    async def create(
        self,
        *,
        query_id: uuid.UUID,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        question_text: str,
        status: QueryStatus = QueryStatus.PENDING,
    ) -> Query:
        query = Query(
            id=query_id,
            tenant_id=tenant_id,
            user_id=user_id,
            question_text=question_text,
            status=status,
            source_references=[],
            created_at=utc_now(),
        )
        self.queries[query_id] = query
        return query

    async def attach_answer(
        self,
        query_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        answer_text: str | None,
        source_references: list[dict[str, Any]],
        confidence_score: float | None,
        status: QueryStatus,
        latency_ms: int,
        answered_at: datetime,
    ) -> bool:
        query = self.queries.get(query_id)
        if query is None or query.tenant_id != tenant_id:
            return False
        query.answer_text = answer_text
        query.source_references = source_references
        query.confidence_score = confidence_score
        query.status = status
        query.latency_ms = latency_ms
        query.answered_at = answered_at
        return True

    async def get(self, query_id: uuid.UUID, tenant_id: uuid.UUID) -> Query | None:
        query = self.queries.get(query_id)
        if query is None or query.tenant_id != tenant_id:
            return None
        return query

    async def load_chunks(
        self, chunk_ids: Sequence[uuid.UUID], tenant_id: uuid.UUID
    ) -> list[RetrievedChunk]:
        """The citability re-check, faked with the *same* three conditions as the real join.

        A Pinecone match is only a candidate. A chunk is returned only when its row and its
        document both belong to this tenant and the document is `ACTIVE` — which is what keeps
        a soft-deleted document unquotable in the window before its vectors leave the index. A
        fake that skipped any of the three would make every vector-lifecycle test vacuous.

        Caller order is preserved, because `source_tracer` numbers passages `[S1]`, `[S2]`… by
        position and the model's citation markers refer to those numbers.
        """
        if self.documents is None:
            return []
        by_id = {c.id: c for c in self.chunks_for(tenant_id)}
        out: list[RetrievedChunk] = []
        for chunk_id in chunk_ids:
            chunk = by_id.get(chunk_id)
            if chunk is None:
                continue
            document = self.documents.documents.get(chunk.document_id)
            if document is None or document.tenant_id != tenant_id:
                continue
            if document.status is not DocumentStatus.ACTIVE:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_name=document.name,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    section_label=chunk.section_label or "",
                    chunk_index=chunk.chunk_index,
                    upload_date=document.upload_date,
                )
            )
        return out

    async def sample_chunks(
        self, tenant_id: uuid.UUID, *, documents: int, per_document: int
    ) -> list[RetrievedChunk]:
        """The suggestion sample, with the real method's four conditions kept.

        Tenant on the chunk, tenant on the document, document `ACTIVE`, and `embedded_at` set —
        the last one matters most here. A chunk with no vector is not in Pinecone, so a fake
        that returned it would let `rag_service.suggest_questions` offer a question that cannot
        possibly retrieve, which is the exact bug the verification step exists to prevent.

        Newest document first and `chunk_index < per_document`, matching the real ordering,
        so a test asserting *which* suggestion comes back is asserting something real.
        """
        if self.documents is None or documents <= 0 or per_document <= 0:
            return []

        rows: list[RetrievedChunk] = []
        for chunk in self.chunks_for(tenant_id):
            if chunk.embedded_at is None or chunk.chunk_index >= per_document:
                continue
            document = self.documents.documents.get(chunk.document_id)
            if document is None or document.tenant_id != tenant_id:
                continue
            if document.status is not DocumentStatus.ACTIVE:
                continue
            rows.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=document.id,
                    document_name=document.name,
                    text=chunk.text,
                    page_number=chunk.page_number,
                    section_label=chunk.section_label or "",
                    chunk_index=chunk.chunk_index,
                    upload_date=document.upload_date,
                )
            )
        rows.sort(key=lambda row: (-row.upload_date.timestamp(), row.chunk_index))
        return rows[: documents * per_document]

    def chunks_for(self, tenant_id: uuid.UUID) -> list[DocumentChunk]:
        """This tenant's chunk rows, read from the document fake. A helper, not a contract."""
        if self.documents is None:
            return []
        return [c for c in self.documents.chunks if c.tenant_id == tenant_id]

    async def list_for_user(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        *,
        status: QueryStatus | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Query]:
        rows = self._filtered(
            tenant_id,
            user_id=user_id,
            status=status,
            keyword=keyword,
            created_from=created_from,
            created_to=created_to,
        )
        return rows[offset : offset + limit]

    async def list_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None = None,
        status: QueryStatus | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> list[Query]:
        rows = self._filtered(
            tenant_id,
            user_id=user_id,
            status=status,
            keyword=keyword,
            created_from=created_from,
            created_to=created_to,
        )
        return rows[offset : offset + limit]

    async def count(
        self,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None = None,
        status: QueryStatus | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> int:
        return len(
            self._filtered(
                tenant_id,
                user_id=user_id,
                status=status,
                keyword=keyword,
                created_from=created_from,
                created_to=created_to,
            )
        )

    async def count_by_status(
        self,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None = None,
        keyword: str | None = None,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> dict[QueryStatus, int]:
        """Grouped counts, returning **only the statuses present** — as `GROUP BY` does.

        Omitting the empty statuses is the behaviour worth copying: a fake that helpfully
        returned zeros would hide it if `search_history_service` ever stopped filling them,
        and the tile row would then be missing an outcome rather than showing it as 0.
        """
        rows = self._filtered(
            tenant_id,
            user_id=user_id,
            status=None,
            keyword=keyword,
            created_from=created_from,
            created_to=created_to,
        )
        counts: dict[QueryStatus, int] = {}
        for query in rows:
            counts[query.status] = counts.get(query.status, 0) + 1
        return counts

    def _filtered(
        self,
        tenant_id: uuid.UUID,
        *,
        user_id: uuid.UUID | None,
        status: QueryStatus | None,
        keyword: str | None,
        created_from: datetime | None,
        created_to: datetime | None,
    ) -> list[Query]:
        rows = [q for q in self.queries.values() if q.tenant_id == tenant_id]
        if user_id is not None:
            rows = [q for q in rows if q.user_id == user_id]
        if status is not None:
            rows = [q for q in rows if q.status is status]
        if keyword:
            needle = keyword.lower()
            rows = [q for q in rows if needle in q.question_text.lower()]
        if created_from is not None:
            rows = [q for q in rows if q.created_at >= created_from]
        if created_to is not None:
            rows = [q for q in rows if q.created_at <= created_to]
        rows.sort(key=lambda q: (q.created_at, str(q.id)), reverse=True)
        return rows


class FakeAnalyticsRepository:
    """In-memory stand-in for the five analytics aggregates.

    Deliberately *not* a reimplementation of the SQL. Each method records the arguments it was
    called with and returns whatever the test queued, because what `analytics_service` decides
    is the policy around the aggregate — the role check, the stopword list, the window, the row
    cap — and none of that is visible in the numbers that come back.

    The statements themselves are covered by `test_analytics_repository.py`, which compiles
    them against the PostgreSQL dialect. Reimplementing `regexp_split_to_table` and
    `percentile_cont` here would test this file rather than the repository, and a fake that
    tokenized slightly differently from PostgreSQL would make a real disagreement invisible.
    """

    def __init__(self) -> None:
        #: `keyword_counts` is called twice per report (descending, then ascending), so the
        #: queued values are keyed by direction rather than popped from one list.
        self.keywords: dict[bool, list[KeywordCount]] = {False: [], True: []}
        self.documents: list[DocumentUsage] = []
        self.trend: list[TrendPoint] = []
        self.users: list[ActiveUser] = []
        self.query_summary = QuerySummary(
            total=0,
            answered=0,
            not_found=0,
            failed=0,
            pending=0,
            avg_latency_ms=None,
            p95_latency_ms=None,
            avg_confidence=None,
        )
        #: Every call, in order: `(method, tenant_id, kwargs)`. Tests assert on the tenant and
        #: on what the service passed down — the stopwords, the bounded limit, the window.
        self.calls: list[tuple[str, uuid.UUID, dict[str, Any]]] = []

    async def keyword_counts(
        self,
        tenant_id: uuid.UUID,
        *,
        stopwords: Sequence[str],
        min_length: int,
        limit: int,
        ascending: bool = False,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[KeywordCount]:
        self.calls.append(
            (
                "keyword_counts",
                tenant_id,
                {
                    "stopwords": list(stopwords),
                    "min_length": min_length,
                    "limit": limit,
                    "ascending": ascending,
                    "created_from": created_from,
                    "created_to": created_to,
                },
            )
        )
        return list(self.keywords[ascending])[:limit]

    async def document_usage(
        self,
        tenant_id: uuid.UUID,
        *,
        limit: int,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[DocumentUsage]:
        self.calls.append(
            (
                "document_usage",
                tenant_id,
                {
                    "limit": limit,
                    "created_from": created_from,
                    "created_to": created_to,
                },
            )
        )
        return list(self.documents)[:limit]

    async def volume_trend(
        self,
        tenant_id: uuid.UUID,
        *,
        granularity: TrendGranularity,
        limit: int,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[TrendPoint]:
        self.calls.append(
            (
                "volume_trend",
                tenant_id,
                {
                    "granularity": granularity,
                    "limit": limit,
                    "created_from": created_from,
                    "created_to": created_to,
                },
            )
        )
        return list(self.trend)[:limit]

    async def active_users(
        self,
        tenant_id: uuid.UUID,
        *,
        limit: int,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> list[ActiveUser]:
        self.calls.append(
            (
                "active_users",
                tenant_id,
                {
                    "limit": limit,
                    "created_from": created_from,
                    "created_to": created_to,
                },
            )
        )
        return list(self.users)[:limit]

    async def summary(
        self,
        tenant_id: uuid.UUID,
        *,
        created_from: datetime | None = None,
        created_to: datetime | None = None,
    ) -> QuerySummary:
        self.calls.append(
            (
                "summary",
                tenant_id,
                {"created_from": created_from, "created_to": created_to},
            )
        )
        return self.query_summary

    # -- Assertion helpers ---------------------------------------------------
    def tenants_seen(self) -> set[uuid.UUID]:
        """Every tenant id that reached the repository. Should always be just the actor's."""
        return {tenant_id for _, tenant_id, _ in self.calls}

    def kwargs_for(self, method: str) -> list[dict[str, Any]]:
        return [kwargs for name, _, kwargs in self.calls if name == method]


# -- RAG collaborators -------------------------------------------------------
#: Word boundary for the fake embedder. A fixed pattern, not a caller-supplied one
#: (`secure-coding-vapt.md`, ReDoS), and linear with no nested quantifier.
_WORD = re.compile(r"[a-z0-9]+")

#: Words carried by nearly every sentence, which would otherwise dominate the fake's cosine
#: similarity and make an unrelated passage look relevant. A real embedding model learns this
#: from training; a bag-of-words projection has to be told.
_STOPWORDS = frozenset(
    # fmt: off
    [
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "how", "in", "is", "it", "of", "on", "or", "our", "that", "the", "their", "this",
        "to", "was", "what", "when", "where", "which", "who", "why", "will", "with", "its",
    ]
    # fmt: on
)


class FakeBedrockClient:
    """Deterministic Bedrock stand-in whose similarity ordering is actually meaningful.

    `embed` is a hashed bag-of-words projection: every word lands in a fixed bucket, and the
    vector is L2-normalized, so the cosine between two texts approximates their normalized word
    overlap. That property is the point. A hash of the whole string would also be deterministic
    but would make *every* pair of texts equidistant, which would let a retrieval test pass
    while ranking passages at random — and ranking is the thing being tested.

    `generate` returns whatever the test queued and records what it was asked, so a grounding
    test can assert that the prompt contained the passages and nothing else.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions
        self.queued_answers: list[str] = []
        self.embed_calls: list[list[str]] = []
        self.generate_calls: list[dict[str, Any]] = []
        #: Set to an exception instance to make the next call raise. Used to prove that a
        #: Bedrock outage surfaces as `RetrievalUnavailable` rather than as a wrong answer.
        self.fail_with: Exception | None = None

    def queue_answer(self, answer: str) -> None:
        """Queue one `generate` response. Answers are consumed in order."""
        self.queued_answers.append(answer)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        if self.fail_with is not None:
            raise self.fail_with
        return [self._project(text) for text in texts]

    async def generate(
        self, *, system: str, messages: list[dict[str, Any]], **kwargs: object
    ) -> str:
        self.generate_calls.append(
            {"system": system, "messages": messages, "kwargs": dict(kwargs)}
        )
        if self.fail_with is not None:
            raise self.fail_with
        if self.queued_answers:
            return self.queued_answers.pop(0)
        # The refusal, not plausible-looking prose. A fake that invented an answer would make
        # an ungrounded-answer test pass by accident.
        return "Information not found in uploaded documents."

    def _project(self, text: str) -> list[float]:
        """Hashed bag of words, L2-normalized. `zlib.crc32`, never `hash()`.

        `hash()` on a `str` is salted per process (PYTHONHASHSEED), so a vector written by one
        test run would not match one written by the next — which would make the eval harness
        report a different score every time it ran.
        """
        vector = [0.0] * self.dimensions
        for word in _WORD.findall(text.lower()):
            if word in _STOPWORDS or len(word) < 2:
                continue
            digest = zlib.crc32(word.encode())
            # A sign bit from a different part of the digest, so two words colliding in one
            # bucket do not always reinforce each other.
            vector[digest % self.dimensions] += 1.0 if (digest >> 16) & 1 else -1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            # An all-stopword text. A zero vector has no cosine at all, so give it a fixed
            # direction that resembles nothing in particular.
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]


class FakePineconeIndex:
    """In-memory `AsyncIndex`, one dict per namespace, real cosine ranking.

    Namespaces are separate dicts rather than one dict with a namespace column, because that is
    the isolation model `architecture.md` mandates: a query against the wrong namespace has to
    return *nothing*. A flat store with a filter would return another tenant's chunks the
    moment the filter were forgotten, and a fake has to be able to fail that way to be worth
    testing against.

    `upsert` rejects document text in metadata. `_build_metadata` in `vector_store.py` already
    raises on the way in, so this is the second of two independent checks — the one that would
    still catch it if a future caller bypassed the builder.
    """

    def __init__(self) -> None:
        #: namespace -> vector id -> (values, metadata)
        self.namespaces: dict[str, dict[str, tuple[list[float], dict[str, Any]]]] = {}
        self.upsert_calls: list[tuple[str, int]] = []
        self.query_calls: list[tuple[str, int]] = []
        self.delete_calls: list[tuple[str, int, bool]] = []
        #: Set to an exception to make the next call raise, so a test can prove an index outage
        #: becomes `VectorStoreUnavailable` rather than an empty but confidently-cited answer.
        self.fail_with: Exception | None = None

    async def upsert(
        self, *, vectors: list[dict[str, Any]], namespace: str
    ) -> dict[str, int]:
        if self.fail_with is not None:
            raise self.fail_with
        self.upsert_calls.append((namespace, len(vectors)))
        space = self.namespaces.setdefault(namespace, {})
        for item in vectors:
            metadata = dict(item.get("metadata") or {})
            forbidden = FORBIDDEN_METADATA_KEYS & metadata.keys()
            if forbidden:
                raise AssertionError(
                    "Pinecone metadata must never carry document text; got "
                    f"{sorted(forbidden)}. See .claude/rules/architecture.md."
                )
            space[str(item["id"])] = (list(item["values"]), metadata)
        return {"upserted_count": len(vectors)}

    async def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str,
        include_metadata: bool = False,
        include_values: bool = False,
    ) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        self.query_calls.append((namespace, top_k))
        space = self.namespaces.get(namespace, {})
        scored = [
            (vector_id, _cosine(vector, values))
            for vector_id, (values, _metadata) in space.items()
        ]
        scored.sort(key=lambda match: match[1], reverse=True)
        matches: list[dict[str, Any]] = []
        for vector_id, score in scored[:top_k]:
            entry: dict[str, Any] = {"id": vector_id, "score": score}
            if include_metadata:
                entry["metadata"] = dict(space[vector_id][1])
            if include_values:
                entry["values"] = list(space[vector_id][0])
            matches.append(entry)
        return {"matches": matches, "namespace": namespace}

    async def delete(
        self,
        *,
        ids: list[str] | None = None,
        namespace: str = "",
        delete_all: bool = False,
    ) -> dict[str, Any]:
        if self.fail_with is not None:
            raise self.fail_with
        self.delete_calls.append((namespace, len(ids or []), delete_all))
        if delete_all:
            self.namespaces.pop(namespace, None)
            return {}
        space = self.namespaces.get(namespace, {})
        for vector_id in ids or []:
            space.pop(str(vector_id), None)
        return {}

    # -- Assertion helpers ---------------------------------------------------
    def ids_in(self, namespace: str) -> set[str]:
        return set(self.namespaces.get(namespace, {}))

    def count(self, namespace: str) -> int:
        return len(self.namespaces.get(namespace, {}))

    def total(self) -> int:
        """Every vector in every namespace — how a leak test proves nothing extra landed."""
        return sum(len(space) for space in self.namespaces.values())


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, clamped to the range a cosine index reports.

    Pinecone returns cosine similarity in `[-1, 1]`, and `RAG_MIN_SIMILARITY` is calibrated
    against that scale. Returning a raw dot product instead would make the threshold mean
    something different in tests than in production.
    """
    if len(left) != len(right):
        raise AssertionError(
            f"Vector width mismatch: queried {len(left)}, stored {len(right)}. "
            "The real index rejects this too."
        )
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class Corpus:
    """Puts a document, its chunk rows and its vectors in place the way ingestion would.

    Shared by `test_rag_service.py` and `test_query_service.py` rather than written twice,
    because the *order* is the point and two copies of it can drift: chunk rows first, vectors
    second, `ACTIVE` last. A helper that wrote `ACTIVE` before the vectors existed would let a
    test pass against a document that was briefly searchable and empty — the state
    `document_processor` is built to never leave behind.

    Takes the real `VectorStore` (over `FakePineconeIndex`) rather than writing to the index
    directly, so namespace construction and the metadata allowlist stay under test on the way
    in. Staging that bypassed them would set up fixtures the production path cannot produce.
    """

    def __init__(
        self,
        documents: FakeDocumentRepository,
        vectors: Any,
        bedrock: FakeBedrockClient,
    ) -> None:
        self._documents = documents
        self._vectors = vectors
        self._bedrock = bedrock

    async def add(
        self,
        tenant_id: uuid.UUID,
        name: str,
        passages: list[str],
        *,
        status: DocumentStatus = DocumentStatus.ACTIVE,
        first_page: int = 1,
    ) -> Document:
        from backend.utils.vector_store import ChunkVector

        document = await self._documents.create(
            document_id=uuid.uuid4(),
            tenant_id=tenant_id,
            name=name,
            s3_key=f"{tenant_id}/{name}",
            content_type="application/pdf",
            file_size=1_024,
            uploaded_by=uuid.uuid4(),
            upload_date=utc_now(),
            version=1,
            content_hash=f"hash-{name}",
            status=DocumentStatus.PROCESSING,
        )
        rows = [
            DocumentChunk(
                id=uuid.uuid4(),
                tenant_id=tenant_id,
                document_id=document.id,
                chunk_index=index,
                text=text,
                page_number=first_page + index,
                section_label="",
                token_count=max(1, len(text) // 4),
            )
            for index, text in enumerate(passages)
        ]
        await self._documents.add_chunks(rows)

        embeddings = await self._bedrock.embed([row.text for row in rows])
        await self._vectors.upsert_embeddings(
            tenant_id,
            [
                ChunkVector(
                    chunk_id=row.id,
                    document_id=document.id,
                    document_name=name,
                    vector=embedding,
                    page_number=row.page_number,
                    section_label=row.section_label,
                    chunk_index=row.chunk_index,
                )
                for row, embedding in zip(rows, embeddings, strict=True)
            ],
        )
        # Step 7 of the real pipeline, and it was missing here: `mark_chunks_embedded` before
        # `ACTIVE`. Without it `embedded_at` stays NULL, and `sample_chunks` — which requires
        # it, because a chunk with no vector cannot be in Pinecone — returned nothing. That
        # made `suggest_questions` untestable through this helper: every assertion about it
        # passed against an empty tuple. A helper that claims to stage documents "the way
        # ingestion would" has to include the acknowledgement ingestion writes.
        await self._documents.mark_chunks_embedded(
            [row.id for row in rows], tenant_id, embedded_at=utc_now()
        )
        await self._documents.set_status(document.id, tenant_id, status)
        return document


class FakeIngestionQueue:
    """Stands in for `IngestionQueue`, and runs the *real* pipeline against the fakes.

    Deliberately not a recorder. The upload path's interesting behaviour is what the document
    looks like *after* ingestion — `ACTIVE` with chunk rows and vectors, or `NEEDS_OCR`, or
    `FAILED` — and a queue that only recorded the call would leave every upload test asserting
    `PROCESSING` forever, which is the one state a finished upload should never be in.

    Errors are swallowed exactly as the real eager branch swallows them, because that behaviour
    is itself under test: a document that fails to parse must not fail the upload request.
    """

    def __init__(
        self,
        *,
        storage: Any = None,
        bedrock: Any = None,
        vectors: Any = None,
        settings: Any = None,
        ocr: Any = None,
        autorun: bool = True,
    ) -> None:
        self.storage = storage
        self.bedrock = bedrock
        self.vectors = vectors
        self.settings = settings
        #: Left `None` by default so the pipeline resolves it from settings — which, in the
        #: test harness, means no OCR and the pre-OCR behaviour. A test that wants a scan
        #: read passes a `FakeOcrClient`.
        self.ocr = ocr
        #: False makes this a pure recorder, for the few tests that assert on the upload
        #: response before ingestion has had any effect on it.
        self.autorun = autorun
        self.calls: list[tuple[uuid.UUID, uuid.UUID]] = []
        self.results: list[Any] = []
        self.errors: list[Exception] = []

    @property
    def eager(self) -> bool:
        return True

    async def enqueue(
        self,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        documents: Any = None,
        audit: Any = None,
    ) -> None:
        # Imported here rather than at module scope: `document_processor` pulls in the Celery
        # application, and nothing else in this file needs a broker configuration to exist.
        from backend.workers.document_processor import ingest_document_in_session

        self.calls.append((document_id, tenant_id))
        if not self.autorun or documents is None:
            return
        try:
            self.results.append(
                await ingest_document_in_session(
                    document_id,
                    tenant_id,
                    documents=documents,
                    storage=self.storage,
                    bedrock=self.bedrock,
                    vectors=self.vectors,
                    ocr=self.ocr,
                    settings=self.settings,
                    audit=audit,
                )
            )
        except Exception as exc:  # noqa: BLE001 - mirrors IngestionQueue.enqueue exactly
            self.errors.append(exc)


class FakeOcrClient:
    """Stands in for `TextractOcrClient` — the provider, not the pipeline step.

    Faked at the client boundary for the same reason `FakeS3Client` fakes boto3 rather than
    `S3StorageService`: the interesting logic is the *merge* in `_apply_ocr` — which pages get
    replaced, which keep their text layer, and whether page numbers survive — and a fake that
    replaced the merge would replace the code under test.

    `pages` maps 1-based page number → the text Textract "read". A page absent from the map is
    one Textract found nothing on, which is different from a page it was never asked about.
    """

    def __init__(
        self,
        pages: dict[int, str] | None = None,
        *,
        fail_with: Exception | None = None,
    ) -> None:
        self.pages = dict(pages or {})
        #: Set to `OcrUnavailable()` to exercise the "scan cannot be read" path: no
        #: credentials, no permission, an unreadable document, or a job that timed out — the
        #: client collapses all of them into that one exception.
        self.fail_with = fail_with
        #: Every (bucket, key) asked for, so a test can assert OCR was reached at all — and
        #: that it was *not* reached for a document whose text layer was already fine.
        self.calls: list[tuple[str, str]] = []

    async def detect_pages(self, *, bucket: str, key: str) -> dict[int, str]:
        self.calls.append((bucket, key))
        if self.fail_with is not None:
            raise self.fail_with
        return dict(self.pages)
