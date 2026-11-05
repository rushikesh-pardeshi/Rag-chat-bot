"""backend/repositories/document_repository.py

Role: The only place `document` and `document_chunk` are queried.

**Every method takes `tenant_id`, including the ones that look like they only need an id.**
`get(document_id, tenant_id)` returning `None` for another tenant's document is what turns an
IDOR attempt into a 404 rather than a leak. Making the parameter mandatory means a caller
cannot forget it — there is no overload that omits it, so the isolation is not something the
service layer has to remember on every call.

No business rules here. Version *selection* is a rule and lives in `document_service`; what
this file offers is `next_version`, which reports the current maximum and nothing more.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import Select, delete, event, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models.document import Document, DocumentChunk, DocumentStatus


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # -- Scoping helper ------------------------------------------------------
    def _scoped(self, tenant_id: uuid.UUID) -> Select[tuple[Document]]:
        """Base statement with the tenant filter already applied.

        Every read below starts here. A query that forgot the filter would be a cross-tenant
        read, so the filter is structural rather than something each method appends.
        """
        return select(Document).where(Document.tenant_id == tenant_id)

    # -- Write --------------------------------------------------------------
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
        """Insert one document row.

        `document_id` is supplied by the caller rather than generated here because the S3 key
        embeds it: the service has to know the id before it can write the object, and an id
        invented in two places would not match.
        """
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
        )
        self._session.add(document)
        await self._session.flush()
        return document

    async def set_status(
        self,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        status: DocumentStatus,
    ) -> bool:
        """Change status; returns False if the id does not belong to this tenant.

        The tenant filter is in the UPDATE itself, not checked beforehand. A read-then-write
        would leave a window in which the row could change tenant — impossible today, but the
        single statement costs nothing and cannot be got wrong later.
        """
        result = await self._session.execute(
            update(Document)
            .where(Document.id == document_id, Document.tenant_id == tenant_id)
            .values(status=status)
        )
        return bool(result.rowcount)

    async def set_parse_result(
        self,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        status: DocumentStatus,
        page_count: int | None,
        metadata: dict[str, Any],
    ) -> bool:
        """Record everything parsing produced, in one statement.

        Status, page count and extracted metadata are written together because a document
        that is `ACTIVE` with a null `page_count` is a state no reader knows how to handle.
        """
        result = await self._session.execute(
            update(Document)
            .where(Document.id == document_id, Document.tenant_id == tenant_id)
            .values(status=status, page_count=page_count, doc_metadata=metadata)
        )
        return bool(result.rowcount)

    async def add_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Persist parsed chunks. Returns how many were added.

        `add_all` rather than a bulk insert: the CHECK constraints on citability and blank
        text should reject a bad chunk with a nameable error, and the row counts involved
        (tens to low thousands per document) do not justify bypassing the ORM.
        """
        if not chunks:
            return 0
        self._session.add_all(list(chunks))
        await self._session.flush()
        return len(chunks)

    async def mark_chunks_embedded(
        self,
        chunk_ids: Sequence[uuid.UUID],
        tenant_id: uuid.UUID,
        *,
        embedded_at: datetime,
    ) -> int:
        """Record that Pinecone acknowledged these chunks' vectors. Returns rows updated.

        Written *after* the upsert returns, never before. `embedded_at` is the local record
        that a vector exists in the index; setting it first would produce chunks that claim to
        be embedded when the upsert then failed, and nothing would ever retry them.

        Tenant-filtered even though the ids came from our own read a moment earlier. The filter
        costs nothing and makes the statement safe to call from a Celery task, where the caller
        chain is harder to audit than in a request.
        """
        if not chunk_ids:
            return 0
        result = await self._session.execute(
            update(DocumentChunk)
            .where(
                DocumentChunk.id.in_(list(chunk_ids)),
                DocumentChunk.tenant_id == tenant_id,
            )
            .values(embedded_at=embedded_at)
        )
        return int(result.rowcount or 0)

    # -- Read ---------------------------------------------------------------
    async def get(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> Document | None:
        """One document, or None — including when it belongs to another tenant.

        Returning None rather than raising is deliberate: the service turns it into a 404,
        which is the same answer a genuinely missing id gets. A caller cannot distinguish
        "does not exist" from "exists, not yours", which is the point.
        """
        stmt = self._scoped(tenant_id).where(Document.id == document_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def refresh(self, document: Document) -> Document:
        """Re-read a loaded entity's columns from the database.

        Needed after the ingestion pipeline has run against the same session: it changes status
        and metadata with bulk `UPDATE` statements, and a subsequent `select()` returns the
        instance already in the identity map *without* overwriting attributes that are already
        loaded. Only an explicit refresh makes the in-memory entity agree with the row, and
        that entity is what the upload response reports.
        """
        await self._session.refresh(document)
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
        """A page of documents, newest upload first.

        `name_contains` is bound as a parameter with the wildcards added here, so a caller's
        `%` or `_` is matched literally after escaping — a filter is not a pattern language,
        and treating it as one is how a search box becomes a way to enumerate the corpus.
        """
        stmt = self._apply_filters(
            self._scoped(tenant_id),
            status=status,
            statuses=statuses,
            name_contains=name_contains,
        )
        stmt = (
            stmt.order_by(Document.upload_date.desc(), Document.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self._session.execute(stmt)).scalars())

    async def count_for_tenant(
        self,
        tenant_id: uuid.UUID,
        *,
        status: DocumentStatus | None = None,
        statuses: Sequence[DocumentStatus] | None = None,
        name_contains: str | None = None,
    ) -> int:
        stmt = select(func.count()).select_from(Document).where(
            Document.tenant_id == tenant_id
        )
        stmt = self._apply_filters(
            stmt, status=status, statuses=statuses, name_contains=name_contains
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def next_version(self, name: str, tenant_id: uuid.UUID) -> int:
        """The version a new upload of `name` should take.

        Counts *every* status, deleted included: reusing v2 because v2 was soft-deleted would
        put two different files behind one lineage identifier, and the audit trail already
        refers to the first one.
        """
        stmt = select(func.coalesce(func.max(Document.version), 0)).where(
            Document.tenant_id == tenant_id, Document.name == name
        )
        return int((await self._session.execute(stmt)).scalar_one()) + 1

    async def lock_name(self, name: str, tenant_id: uuid.UUID) -> None:
        """Serialize ingestion of one `(tenant, name)` for the rest of this transaction.

        **Must be taken before any row is written and before the audit chain lock.** It exists
        to remove a lock-order inversion that version supersession would otherwise create:

          * a pipeline writes its own document's row (`PROCESSING` → `ACTIVE`), taking a row
            lock on it;
          * it then appends its ingestion audit row, which takes
            `pg_advisory_xact_lock('audit_chain:<tenant>')` and holds it to commit;
          * then supersession writes the rows of *older versions of the same name*.

        So v4's pipeline can want v3's row while holding the chain lock, exactly as v3's
        pipeline is waiting for that chain lock while holding its own row — a cycle, which
        PostgreSQL resolves by aborting one of them. The loser rolls back a document that had
        already parsed, embedded and upserted: its chunk rows vanish while its vectors do not,
        leaving vectors no query can ever enumerate again, and `_mark_failed` then records a
        FAILURE for an ingestion that had in fact succeeded.

        Two concurrent re-uploads of one filename is all it takes, which the in-process pool
        makes ordinary. Taking this first means both transactions order their locks the same
        way, so the second simply waits.

        Serializing per *name* rather than per tenant is the point: two versions of one
        document have no business being changed at once, and documents with different names
        stay fully parallel — which is what the in-process pool exists to provide.

        **The invariant is "every document mutation claims the name first".** `upload`,
        `soft_delete`, `restore`, `reindex`, `purge` and `_run_pipeline` all take this before
        their first write, so no two of them can order a row lock and the chain lock
        differently. `upload` and `reindex` were the dangerous pair: `upload` appends its
        "accepted" audit row — and so takes the chain lock — before ingestion writes any other
        version's row, while `reindex` holds a document row from its `set_status` and asks for
        the chain lock afterwards. Keeping the exception list empty is cheaper than
        re-deriving which pairs are safe every time one of them is edited.

        It also fixes a race that predates supersession: `next_version` is `max(version) + 1`
        with no lock, so two concurrent uploads of one filename both computed the same number
        and the second died on `uq_document_name_version`.

        Transaction-scoped, like `audit_repository.lock_chain`, so it releases on commit or
        rollback with no `unlock` to forget. `hashtext` reduces the key to the `bigint` the
        advisory lock functions take; the name is a bound parameter, never interpolated.
        """
        await self._session.execute(
            text("SELECT pg_advisory_xact_lock(hashtext(:key))"),
            {"key": f"document_ingest:{tenant_id}:{name}"},
        )

    async def list_earlier_versions(
        self, name: str, tenant_id: uuid.UUID, *, before_version: int
    ) -> list[Document]:
        """Versions of `name` older than `before_version` that are not already `DELETED`.

        The read behind version supersession. `DELETED` is excluded so a second supersession
        pass has nothing to do — which is what makes the caller idempotent without it having
        to remember what it retired last time.

        Newest first, because the caller audits each one and the most recent predecessor is
        the interesting row in that list.
        """
        stmt = (
            self._scoped(tenant_id)
            .where(
                Document.name == name,
                Document.version < before_version,
                Document.status != DocumentStatus.DELETED,
            )
            .order_by(Document.version.desc())
        )
        return list((await self._session.execute(stmt)).scalars())

    def after_commit(self, callback: Callable[[], None]) -> None:
        """Run `callback` once, when this session's transaction commits.

        The session is this layer's to own, so the hook onto it belongs here rather than in a
        service reaching for `_session` (`architecture.md`).

        It exists for one caller: dispatching ingestion to the in-process pool. That pool's
        task opens a **new** session, so it can only see the `document` row once the request's
        transaction has committed — dispatching at the point of the call would race, and the
        worker would find nothing and give up. This is the precise version of the
        `DISPATCH_DELAY_SECONDS` countdown the Celery path uses, and it is available here only
        because that pool runs in this same process.

        `once=True` removes the listener after it fires. A rolled-back transaction never fires
        it at all, which is the behaviour wanted: there is no document to ingest.
        """
        # `sync_session` is the greenlet-backed `Session` the async wrapper drives. SQLAlchemy
        # emits ORM session events there, not on `AsyncSession`.
        @event.listens_for(self._session.sync_session, "after_commit", once=True)
        def _fire(_session: object) -> None:
            callback()

    async def list_versions(self, name: str, tenant_id: uuid.UUID) -> list[Document]:
        """Every version of one name, oldest first — the lineage view (BR-01)."""
        stmt = (
            self._scoped(tenant_id)
            .where(Document.name == name)
            .order_by(Document.version.asc())
        )
        return list((await self._session.execute(stmt)).scalars())

    async def get_chunks(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> list[DocumentChunk]:
        """Chunks in document order. Tenant-filtered on the chunk's own column.

        `DocumentChunk.tenant_id` is denormalized precisely so this needs no join — and so a
        chunk read cannot accidentally be scoped by nothing at all.
        """
        stmt = (
            select(DocumentChunk)
            .where(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.document_id == document_id,
            )
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return list((await self._session.execute(stmt)).scalars())

    async def count_chunks(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> int:
        stmt = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.document_id == document_id,
            )
        )
        return int((await self._session.execute(stmt)).scalar_one())

    async def delete_chunks(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> int:
        """Remove a document's chunks so re-processing does not duplicate them.

        Targets derived data only: chunks can be regenerated from the S3 object. Destroying
        the `document` row itself is `delete_document`, which exists for one caller.
        """
        result = await self._session.execute(
            delete(DocumentChunk).where(
                DocumentChunk.tenant_id == tenant_id,
                DocumentChunk.document_id == document_id,
            )
        )
        return int(result.rowcount or 0)

    async def delete_document(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> bool:
        """Destroy one document row. True when a row was actually removed.

        Irreversible, and the only statement in this project that erases a document. Three
        things make it safe to have:

          * **`tenant_id` is in the WHERE clause**, so an id guessed or leaked from another
            organization deletes nothing and reports `False` — the same shape as a 404,
            which is what `rbac-enforcement.md` requires of every query here.
          * **`document_chunk` has `ON DELETE CASCADE`** on `document_id`, so the chunks go
            with it in the same statement rather than being left as orphans pointing at a
            row that no longer exists.
          * **`audit_log` has no foreign key to `document`** — by design. The record of the
            upload, of every query that cited it, and of this deletion all survive, which is
            what makes an erasure auditable rather than a gap
            (`.claude/rules/audit-log-immutability.md`).

        The service is responsible for the S3 object and the vectors; a repository has no
        business reaching outside the database.
        """
        result = await self._session.execute(
            delete(Document).where(
                Document.id == document_id, Document.tenant_id == tenant_id
            )
        )
        return bool(result.rowcount)

    async def exists_with_hash(self, content_hash: str, tenant_id: uuid.UUID) -> bool:
        """Has this tenant already stored these exact bytes?

        Used to tell an Administrator they are re-uploading an identical file. Scoped to the
        tenant: whether *another* organization holds the same document is not information
        this tenant may have.
        """
        stmt = select(
            select(Document.id)
            .where(
                Document.tenant_id == tenant_id, Document.content_hash == content_hash
            )
            .exists()
        )
        return bool((await self._session.execute(stmt)).scalar_one())

    @staticmethod
    def _apply_filters(
        stmt: Any,
        *,
        status: DocumentStatus | None,
        statuses: Sequence[DocumentStatus] | None,
        name_contains: str | None,
    ) -> Any:
        if status is not None:
            stmt = stmt.where(Document.status == status)
        if statuses:
            stmt = stmt.where(Document.status.in_(list(statuses)))
        if name_contains:
            # `escape` plus escaping the wildcards makes the input a literal substring.
            escaped = (
                name_contains.replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            stmt = stmt.where(Document.name.ilike(f"%{escaped}%", escape="\\"))
        return stmt
