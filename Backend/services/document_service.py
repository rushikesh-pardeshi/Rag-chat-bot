"""backend/services/document_service.py

Role: The document lifecycle — upload, list, read, preview, soft-delete, restore. Owns every
rule about what may be stored and who may see it.

The upload order is deliberate and is the part most worth reading:

  1. **Sniff the bytes.** The extension and the browser's `Content-Type` are both
     caller-supplied. An `.exe` renamed `.pdf` is refused here, before any parser runs.
  2. **Size, then extension, then magic bytes, then agreement between the last two.** A file
     whose real type contradicts its extension is refused even when both are individually
     allowed — that mismatch is the signature of a deliberate attempt, not a mistake.
  3. **S3 first, database second.** The bytes are what cannot be reconstructed. A stored
     object with no row is an orphan a reconciliation job can find; a row with no object is a
     document that appears in every listing and can never be opened.
  4. **Hand off, do not parse.** The row is written as `PROCESSING` and ingestion — parse,
     chunk, embed, upsert — happens in `backend/workers/document_processor.py`. Parsing here
     would hold the HTTP connection for the length of a few hundred Bedrock round trips.

**This service does not know how ingestion runs.** It calls `IngestionQueue.enqueue`, which
dispatches to Celery in production and awaits the pipeline in-process when
`CELERY_TASK_ALWAYS_EAGER` is set. That is why the upload response is a 202 with
`status='PROCESSING'` in both modes: the contract is "accepted", not "indexed", and a client
polls or refreshes to see it become `ACTIVE`.

**Nothing here deletes a document.** `soft_delete` sets `status='DELETED'`; the row, the S3
object and the chunk rows all stay exactly as they were. What it *does* remove is the
document's vectors from Pinecone — housekeeping, not the security boundary. PostgreSQL is the
authority: `query_repository.load_chunks` refuses to cite a non-`ACTIVE` document whether or
not its vectors have gone. Restoring re-embeds. This is BR-01, and it is also the only way the
audit trail keeps its meaning — an entry that says a document was consulted is worthless if the
document can be erased.

**Version lineage is app-side.** Re-uploading a name writes v2 to its own immutable key, and
v1 stays readable at its own. Our IAM role cannot enable bucket versioning, so overwriting
would destroy the earlier bytes — which is the one thing lineage exists to prevent.

Role rules (`.claude/rules/rbac-enforcement.md`): upload, delete and restore are
Administrator-only, enforced by `require_admin` on the route. Listing, reading and previewing
are open to any authenticated user *within their own tenant*, because a Standard User must be
able to download the source document behind an answer — that is what makes a citation
checkable. A Standard User does not see soft-deleted documents; an Administrator does.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from backend.core.config import Settings, get_settings
from backend.core.exceptions import (
    DocumentDeleted,
    DocumentNotDeleted,
    DocumentNotFound,
    DocumentNotSoftDeleted,
    DocumentTooLarge,
    StorageError,
    UnauthorizedAccess,
    UnsupportedDocumentType,
    ValidationError,
)
from backend.core.logging import get_logger
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.document import RETRIEVABLE_STATUSES, Document, DocumentStatus
from backend.models.user import User, UserRole
from backend.repositories.document_repository import DocumentRepository
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.services.storage_service import (
    StorageBackend,
    compute_content_hash,
    file_extension,
    sanitize_filename,
)
from backend.utils.document_parser import (
    CONTENT_TYPE_DOCX,
    CONTENT_TYPE_PDF,
    detect_content_type,
)
from backend.utils.ids import new_id, utc_now
from backend.utils.vector_store import VectorStore

if TYPE_CHECKING:
    # Type-only: the service layer must not import the Celery application at runtime, or
    # `import backend.services.document_service` would pull in a broker configuration. The
    # instance is injected by `dependencies/services.py`.
    from backend.workers.document_processor import IngestionQueue

logger = get_logger(__name__)

#: Extension → the content type its bytes must actually have. Checked in both directions, so
#: `report.pdf` containing a DOCX is refused as firmly as `report.docx` containing a PDF.
EXTENSION_CONTENT_TYPES: Final[dict[str, str]] = {
    "pdf": CONTENT_TYPE_PDF,
    "docx": CONTENT_TYPE_DOCX,
}

#: Discriminates a reindex from an original upload in the audit trail, both of which use
#: `AuditAction.DOCUMENT_UPLOAD`. Same reasoning as `document_processor.AUDIT_PHASE_INGEST`:
#: a new enum member needs `ALTER TYPE` on a native PostgreSQL enum, and `detail` is JSONB.
AUDIT_PHASE_REINDEX: Final = "reindex"

#: Marks the `DOCUMENT_DELETE` row that retires a version because a newer one replaced it.
#:
#: It shares the action with a deliberate soft-delete on purpose — the effect on the document
#: *is* a soft-delete, and inventing a second action would need `ALTER TYPE` on a native
#: PostgreSQL enum. The phase is what lets a compliance reviewer tell "an administrator
#: removed this" from "the system retired this when v3 landed", which are different events
#: with different people answerable for them.
AUDIT_PHASE_SUPERSEDE: Final = "superseded_by_new_version"

#: Statuses a Standard User may see. `DELETED` is administratively visible only — a restore
#: candidate is document management, and that is an Administrator concern.
USER_VISIBLE_STATUSES: Final[tuple[DocumentStatus, ...]] = (
    DocumentStatus.ACTIVE,
    DocumentStatus.PROCESSING,
    DocumentStatus.NEEDS_OCR,
    DocumentStatus.FAILED,
)


@dataclass(frozen=True, slots=True)
class DocumentPage:
    """A page of results plus the total, so the UI can render a pager."""

    items: tuple[Document, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class DocumentPreview:
    """A presigned download URL and how long it lasts."""

    document_id: uuid.UUID
    name: str
    content_type: str
    url: str
    expires_in_seconds: int


@dataclass(frozen=True, slots=True)
class PurgeResult:
    """What a permanent deletion actually destroyed.

    Counts rather than a bare acknowledgement, for two readers: the Administrator, who gets
    to see that the index and the bucket were cleared and not just the row, and the audit
    record, where "deleted" without a number is a claim nobody can check later.

    `vectors_deleted` is `-1` when no vector store is configured — distinct from `0`, which
    means there was nothing in the index to remove.

    It counts **ids submitted to Pinecone**, not vectors confirmed gone: the delete API
    returns no count, so there is nothing truthful to report beyond what was asked for. The
    number is normally non-zero even though `soft_delete` already cleared the index, because
    the chunk rows it enumerates are still there until this call removes them. Treated as
    "the index was told about every chunk", which is the claim the audit record can support.
    """

    document_id: uuid.UUID
    name: str
    chunks_deleted: int
    vectors_deleted: int
    s3_versions_deleted: int


class DocumentService:
    def __init__(
        self,
        documents: DocumentRepository,
        storage: StorageBackend,
        audit: AuditService,
        queue: IngestionQueue | None = None,
        vectors: VectorStore | None = None,
    ) -> None:
        self._documents = documents
        self._storage = storage
        self._audit = audit
        #: Both optional so the Phase 3 tests, which only exercise validation, listing and the
        #: lifecycle, construct this service without a queue or an index. A `None` queue means
        #: the document is stored and left `PROCESSING` — visible, honest, and not retrievable.
        self._queue = queue
        self._vectors = vectors
        self._settings = get_settings()

    # -- Upload -------------------------------------------------------------
    async def upload(
        self,
        *,
        actor: User,
        context: RequestContext,
        filename: str,
        data: bytes,
        declared_content_type: str | None = None,
    ) -> Document:
        """Validate, store, and hand the document to the ingestion queue.

        `declared_content_type` is accepted only to be *compared* with what the bytes say.
        It is never used to choose a parser and never stored — `Document.content_type` holds
        the type we determined.

        Raises `ValidationError`, `DocumentTooLarge` or `UnsupportedDocumentType` before
        anything is written. Returns a `PROCESSING` document: the bytes are safe in S3 and the
        row exists, but nothing is retrievable until the pipeline sets `ACTIVE`. A parse
        failure downstream lands as `FAILED` on this row, so the Administrator sees a failed
        document in the list rather than a silent absence.
        """
        safe_name = sanitize_filename(filename)
        content_type = self._validate_upload(
            filename=filename,
            safe_name=safe_name,
            data=data,
            declared_content_type=declared_content_type,
        )

        # Claim the name for this transaction before reading `next_version` or writing
        # anything. Two reasons, and both are races this closes rather than tidiness:
        #
        #  * `next_version` is `max(version) + 1` with no lock, so two concurrent uploads of
        #    one filename both computed the same number and the second lost to
        #    `uq_document_name_version` — an upload that failed for no reason the uploader
        #    could see. Serialized here, they get v2 and v3.
        #  * It fixes the order in which every transaction touching this name takes its locks.
        #    Version supersession writes the rows of *other* versions while the pipeline holds
        #    `pg_advisory_xact_lock('audit_chain:<tenant>')`, and `reindex`/`restore` hold a
        #    document row while waiting for that same chain lock — a cycle PostgreSQL resolves
        #    by aborting one side. Taking the name first means they queue instead.
        #
        # `_run_pipeline` takes the same lock; advisory locks are re-entrant within a
        # transaction, so the eager path re-acquiring it costs nothing.
        await self._documents.lock_name(safe_name, actor.tenant_id)

        content_hash = compute_content_hash(data)
        duplicate = await self._documents.exists_with_hash(content_hash, actor.tenant_id)

        # The id is generated before the object is written, because the key embeds it.
        document_id = new_id()
        version = await self._documents.next_version(safe_name, actor.tenant_id)
        key = self._storage.build_key(
            document_id=document_id, version=version, filename=safe_name
        )

        # Bytes before metadata: see the module docstring. A StorageError here means nothing
        # was written at all, and `get_db` rolls the (empty) transaction back.
        stored = await self._storage.put(
            key=key,
            data=data,
            content_type=content_type,
            tenant_id=actor.tenant_id,
        )

        document = await self._documents.create(
            document_id=document_id,
            tenant_id=actor.tenant_id,
            name=safe_name,
            s3_key=stored.key,
            content_type=content_type,
            file_size=stored.size_bytes,
            uploaded_by=actor.id,
            upload_date=utc_now(),
            version=version,
            content_hash=stored.content_hash,
            status=DocumentStatus.PROCESSING,
        )

        if duplicate:
            # Recorded, not refused: a bank legitimately re-files an identical PDF under a new
            # name, and refusing would make that impossible. The flag lets the UI say so.
            # Written now rather than merged later, because the pipeline replaces `metadata`
            # wholesale with what parsing found and would otherwise drop this.
            await self._documents.set_parse_result(
                document_id,
                actor.tenant_id,
                status=DocumentStatus.PROCESSING,
                page_count=None,
                metadata={"duplicate_content": True},
            )
            document.doc_metadata = {"duplicate_content": True}

        # Audited as accepted, before ingestion runs. The upload *did* succeed — the bytes are
        # stored and the row exists — and a later parse failure is a separate fact with its own
        # audit entry. Waiting until ingestion finished would mean a Celery worker that never
        # ran left no record that anyone uploaded anything.
        await self._audit_document(
            actor,
            context,
            action=AuditAction.DOCUMENT_UPLOAD,
            document_id=document_id,
            outcome=AuditOutcome.SUCCESS,
            detail={
                # Ids, counts and types only. No filename: a document name can carry a
                # customer identifier, and `secure-coding-vapt.md` forbids PII in a log.
                "version": version,
                "content_type": content_type,
                "file_size": stored.size_bytes,
                "status": DocumentStatus.PROCESSING.value,
                "duplicate_content": duplicate,
            },
        )

        if self._queue is not None:
            # The repository goes with it so the eager path can read the row we have only
            # flushed. In eager mode this awaits the whole pipeline, so `document` may come
            # back from it already `ACTIVE` — hence the refresh below.
            # The audit service goes with it too: the pipeline records its own terminal
            # outcome, and on the eager path that row belongs in this request's transaction
            # and hash chain, immediately after the "upload accepted" row above.
            await self._queue.enqueue(
                document_id,
                actor.tenant_id,
                documents=self._documents,
                audit=self._audit,
            )
            # An explicit refresh, not another `get`: the pipeline wrote status, page count and
            # metadata with bulk UPDATEs, and a second `select()` would hand back this same
            # instance with its stale attributes intact.
            await self._documents.refresh(document)
        else:
            logger.warning(
                "document.no_ingestion_queue", document_id=str(document_id)
            )

        logger.info(
            "document.uploaded",
            document_id=str(document_id),
            version=version,
            status=document.status.value,
        )
        return document

    def _validate_upload(
        self,
        *,
        filename: str,
        safe_name: str,
        data: bytes,
        declared_content_type: str | None,
    ) -> str:
        """Every check that must pass before a byte is stored. Returns the real type."""
        if not data:
            raise ValidationError("The uploaded file is empty.")
        if len(data) > self._settings.max_upload_bytes:
            raise DocumentTooLarge(
                f"The document exceeds the {self._settings.MAX_UPLOAD_SIZE_MB} MB limit."
            )
        if not filename or not filename.strip():
            raise ValidationError("A filename is required.")

        extension = file_extension(safe_name)
        if extension not in self._settings.allowed_extensions:
            raise UnsupportedDocumentType()

        # Magic bytes are the authority. This raises `UnsupportedDocumentType` for anything
        # that is not a PDF or a ZIP container, whatever the extension claimed.
        detected = detect_content_type(data)
        if EXTENSION_CONTENT_TYPES.get(extension) != detected:
            # Both are individually allowed but they disagree — refuse. A user who genuinely
            # mislabelled a file can rename it; an attacker relies on the mismatch.
            raise UnsupportedDocumentType(
                "The file's contents do not match its extension."
            )
        if declared_content_type and declared_content_type != detected:
            # Logged, not refused: browsers send `application/octet-stream` for a PDF often
            # enough that refusing on this alone would break real uploads.
            logger.info(
                "document.content_type_mismatch",
                declared=declared_content_type[:128],
                detected=detected,
            )
        return detected

    # -- Read ---------------------------------------------------------------
    async def list_documents(
        self,
        *,
        actor: User,
        status: DocumentStatus | None = None,
        name_contains: str | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> DocumentPage:
        """A page of the actor's tenant's documents.

        A Standard User asking for `status=DELETED` gets an empty page rather than a 403:
        the filter is narrowed to what they may see *before* the query runs, so there is no
        request they can craft that widens it.
        """
        statuses = None if actor.role is UserRole.ADMINISTRATOR else USER_VISIBLE_STATUSES
        if status is not None and statuses is not None and status not in statuses:
            return DocumentPage(items=(), total=0, limit=limit, offset=offset)

        items = await self._documents.list_for_tenant(
            actor.tenant_id,
            status=status,
            statuses=None if status is not None else statuses,
            name_contains=name_contains,
            limit=limit,
            offset=offset,
        )
        total = await self._documents.count_for_tenant(
            actor.tenant_id,
            status=status,
            statuses=None if status is not None else statuses,
            name_contains=name_contains,
        )
        return DocumentPage(
            items=tuple(items), total=total, limit=limit, offset=offset
        )

    async def get_document(
        self,
        document_id: uuid.UUID,
        *,
        actor: User,
        context: RequestContext,
        denied_action: AuditAction = AuditAction.DOCUMENT_PREVIEW,
    ) -> Document:
        """One document, or `DocumentNotFound`.

        Another tenant's id produces the same 404 as an id that never existed — and an audit
        row with `outcome='DENIED'`, because a cross-tenant read attempt is an authorization
        failure wearing a 404 (`.claude/rules/rbac-enforcement.md`).

        `denied_action` is what that row records. It defaults to `DOCUMENT_PREVIEW` — this is
        a read — but a caller doing something graver passes its own, because `audit_log`'s
        contract is that denials carry *the action that was attempted*. A refused attempt to
        destroy another tenant's document logged as a denied preview is findable only by
        someone already reading preview denials, and an alert on `DOCUMENT_PURGE` would miss
        the single most alarming event this service can produce.
        """
        document = await self._documents.get(document_id, actor.tenant_id)
        if document is None:
            await self._audit_document(
                actor,
                context,
                action=denied_action,
                document_id=document_id,
                outcome=AuditOutcome.DENIED,
                detail={"reason": "not_found_or_other_tenant"},
            )
            raise DocumentNotFound()

        if (
            actor.role is not UserRole.ADMINISTRATOR
            and document.status is DocumentStatus.DELETED
        ):
            # Same 404. Telling a Standard User "this exists but is deleted" is information
            # about document management they have no part in.
            await self._audit_document(
                actor,
                context,
                action=denied_action,
                document_id=document_id,
                outcome=AuditOutcome.DENIED,
                detail={"reason": "deleted_not_visible_to_role"},
            )
            raise DocumentNotFound()
        return document

    async def preview(
        self, document_id: uuid.UUID, *, actor: User, context: RequestContext
    ) -> DocumentPreview:
        """A short-lived presigned URL, minted only after the tenant and role check.

        The order is the security property: `get_document` performs the check and raises,
        so no URL exists for a document the caller may not read. A presigned URL is a bearer
        credential that cannot be recalled once issued, which is why the TTL is minutes.
        """
        document = await self.get_document(
            document_id, actor=actor, context=context
        )
        ttl = self._settings.S3_PRESIGNED_URL_EXPIRY_SECONDS
        url = await self._storage.get_presigned_url(document.s3_key, ttl_seconds=ttl)

        await self._audit_document(
            actor,
            context,
            action=AuditAction.DOCUMENT_PREVIEW,
            document_id=document_id,
            outcome=AuditOutcome.SUCCESS,
            detail={"version": document.version, "ttl_seconds": ttl},
        )
        return DocumentPreview(
            document_id=document.id,
            name=document.name,
            content_type=document.content_type,
            url=url,
            expires_in_seconds=ttl,
        )

    async def list_versions(self, name: str, *, actor: User) -> tuple[Document, ...]:
        """Every version of one name — the lineage view. Administrator-gated at the route."""
        return tuple(await self._documents.list_versions(name, actor.tenant_id))

    # -- Lifecycle ----------------------------------------------------------
    async def soft_delete(
        self, document_id: uuid.UUID, *, actor: User, context: RequestContext
    ) -> Document:
        """Set `status='DELETED'`, then remove the document's vectors from Pinecone.

        Chunk *rows* are kept: retrieval filters on `RETRIEVABLE_STATUSES`, so a deleted
        document is unreachable by search either way, and keeping the text makes a restore a
        re-embed rather than a re-parse.

        The **order matters, and is the opposite of what it looks like it should be.** The
        status change goes first, and the vector deletion is best-effort afterwards:

          * PostgreSQL is the authority. Once `status='DELETED'`, `load_chunks` refuses to
            return the chunk at all, so the document is uncitable from that instant —
            regardless of what is still in the index.
          * Deleting vectors first and then failing to write the status would leave a document
            that is `ACTIVE`, listed, and unsearchable, with no record of why.
          * `delete_document_vectors` therefore swallows its own errors rather than raising. A
            raise here would roll back the status change and leave the document *citable*,
            which is strictly the worse outcome.

        Pinecone serverless cannot delete by metadata filter, so the chunk ids have to be read
        and passed in — see `vector_store.delete_document_vectors`.
        """
        document = await self.get_document(document_id, actor=actor, context=context)
        if document.status is DocumentStatus.DELETED:
            # Idempotent: a double-click must not produce a second audit row claiming a
            # second deletion of the same document.
            return document

        # Before the row write, for the same reason as `upload`: this transaction takes a
        # document row lock and then the audit chain lock, and an ingestion running version
        # supersession takes them in the other order.
        await self._documents.lock_name(document.name, actor.tenant_id)

        previous = document.status
        await self._documents.set_status(
            document_id, actor.tenant_id, DocumentStatus.DELETED
        )
        document.status = DocumentStatus.DELETED

        vectors_removed = await self._remove_vectors(document_id, actor.tenant_id)

        await self._audit_document(
            actor,
            context,
            action=AuditAction.DOCUMENT_DELETE,
            document_id=document_id,
            outcome=AuditOutcome.SUCCESS,
            detail={
                "version": document.version,
                "previous_status": previous.value,
                # Stated explicitly in the record, because "delete" in a compliance review
                # must not be read as "erased".
                "soft_delete": True,
                # Recorded so a compliance reviewer can see the index was cleared too, and
                # `-1` when there was no vector store configured to clear it with.
                "vectors_removed": vectors_removed,
            },
        )
        return document

    async def _remove_vectors(
        self, document_id: uuid.UUID, tenant_id: uuid.UUID
    ) -> int:
        """Best-effort removal of one document's vectors. Returns the count, or -1 if skipped.

        Never raises. See `soft_delete` for why a failure here must not undo the status change.
        """
        if self._vectors is None:
            return -1
        chunks = await self._documents.get_chunks(document_id, tenant_id)
        if not chunks:
            return 0
        return await self._vectors.delete_document_vectors(
            tenant_id, document_id, [chunk.id for chunk in chunks]
        )

    async def restore(
        self, document_id: uuid.UUID, *, actor: User, context: RequestContext
    ) -> Document:
        """Return a soft-deleted document to circulation by re-ingesting it.

        Refuses anything that is not `DELETED` — restoring an already-active document is a
        no-op that would still write an audit row saying a restore happened.

        **A restore is a re-ingest, not a status flip.** `soft_delete` removed this document's
        vectors from Pinecone, so setting `ACTIVE` directly would produce the failure mode the
        pipeline is ordered to avoid: a document that is listed, looks searchable, and matches
        nothing — answering every question about itself with "not found" while appearing
        healthy. The bytes are still in S3 — that is why soft-delete keeps them — so the
        pipeline re-parses, re-chunks and re-embeds, and sets the status the content justifies
        (`ACTIVE`, or `NEEDS_OCR` for a scan).

        With no queue configured the document is left `PROCESSING` rather than made `ACTIVE`.
        That is visibly incomplete, which is the correct way to fail.
        """
        document = await self.get_document(document_id, actor=actor, context=context)
        if document.status is not DocumentStatus.DELETED:
            raise DocumentNotDeleted()

        # Before the row write and before this request's audit row, so this transaction orders
        # its locks the same way `upload` does. See the note in `upload` for the cycle.
        await self._documents.lock_name(document.name, actor.tenant_id)

        # `PROCESSING` first, so no reader sees it as retrievable before its vectors exist.
        await self._documents.set_status(
            document_id, actor.tenant_id, DocumentStatus.PROCESSING
        )
        document.status = DocumentStatus.PROCESSING

        if self._queue is not None:
            await self._queue.enqueue(
                document_id,
                actor.tenant_id,
                documents=self._documents,
                audit=self._audit,
            )
            await self._documents.refresh(document)
        else:
            logger.warning(
                "document.restore_without_queue", document_id=str(document_id)
            )

        await self._audit_document(
            actor,
            context,
            action=AuditAction.DOCUMENT_RESTORE,
            document_id=document_id,
            outcome=AuditOutcome.SUCCESS,
            detail={
                "version": document.version,
                "restored_to": document.status.value,
                "reingested": self._queue is not None,
            },
        )
        return document

    async def reindex(
        self, document_id: uuid.UUID, *, actor: User, context: RequestContext
    ) -> Document:
        """Re-run the whole pipeline over a document that is already here.

        The bytes in S3 are the source of truth, so everything derived from them can be
        rebuilt: parse, OCR for pages with no text layer, chunk, embed, upsert. That makes
        this the repair tool for every "the document is here but the index is wrong" case —
        a scan that failed OCR because a credential had expired, a document ingested before
        OCR existed at all, a `FAILED` row left by a transient Bedrock or Pinecone outage.
        Until now the only way to do it was `scripts/reocr_documents.py`, which needs a shell
        on the host.

        **Allowed from every status except `DELETED`.** Reindexing a deleted document would
        re-embed something that must not be quotable, and `restore` already re-runs this same
        pipeline *and* returns the document to circulation — so a second route that re-ingests
        while leaving it invisible is a trap rather than a feature.

        `PROCESSING` is written first, exactly as `restore` does: no reader should see a
        document as retrievable while its vectors are being replaced. With no queue
        configured it stays `PROCESSING`, which is visibly incomplete and the correct way to
        fail.

        The audit row is a `DOCUMENT_UPLOAD` with `phase: "reindex"`, following
        `AUDIT_PHASE_INGEST` and `query_service.AUDIT_PHASE_SUBMIT`: a new `AuditAction` member
        needs `ALTER TYPE` on a native PostgreSQL enum, and `detail` is JSONB. A reindex
        changes what the system will answer with but destroys nothing, so it does not warrant
        the migration that `DOCUMENT_PURGE` did.
        """
        document = await self.get_document(document_id, actor=actor, context=context)
        if document.status is DocumentStatus.DELETED:
            await self._audit_document(
                actor,
                context,
                action=AuditAction.DOCUMENT_UPLOAD,
                document_id=document_id,
                outcome=AuditOutcome.DENIED,
                detail={"phase": AUDIT_PHASE_REINDEX, "reason": "document_deleted"},
            )
            raise DocumentDeleted()

        # Same lock, same reason as `upload` and `restore`: this transaction is about to hold
        # a document row and then ask for the audit chain lock.
        await self._documents.lock_name(document.name, actor.tenant_id)

        previous = document.status
        await self._documents.set_status(
            document_id, actor.tenant_id, DocumentStatus.PROCESSING
        )
        document.status = DocumentStatus.PROCESSING

        if self._queue is not None:
            await self._queue.enqueue(
                document_id,
                actor.tenant_id,
                documents=self._documents,
                audit=self._audit,
            )
            await self._documents.refresh(document)
        else:
            logger.warning(
                "document.reindex_without_queue", document_id=str(document_id)
            )

        await self._audit_document(
            actor,
            context,
            action=AuditAction.DOCUMENT_UPLOAD,
            document_id=document_id,
            outcome=AuditOutcome.SUCCESS,
            detail={
                "phase": AUDIT_PHASE_REINDEX,
                "version": document.version,
                # Both ends recorded: an administrator reindexing a `FAILED` document and an
                # administrator reindexing an `ACTIVE` one are different events, and after
                # the fact the row is the only place that difference survives.
                "previous_status": previous.value,
                "status": document.status.value,
                "reingested": self._queue is not None,
            },
        )
        return document

    async def purge(
        self, document_id: uuid.UUID, *, actor: User, context: RequestContext
    ) -> PurgeResult:
        """Destroy a soft-deleted document: vectors, S3 object, chunks, row. Irreversible.

        **Only a `DELETED` document can be purged.** Destruction is deliberately two steps —
        delete, then delete permanently — so that no single request can erase a document that
        is still in circulation. An Administrator who means it has already seen the row move
        to `Deleted` and chosen again.

        Administrator-only, and checked **here** as well as by the route's `AdminUser`
        dependency. `soft_delete` leans on the route alone, which is fine for something
        reversible; this is not reversible, and a service that erases customer records on
        request should not be one refactor away from doing it for a Standard User.

        ## The order, and why each step is where it is

        1. **Read the chunk ids first.** Pinecone serverless cannot delete by metadata
           filter, so the ids have to come out of PostgreSQL *before* the rows go — after the
           cascade there is nothing left to enumerate and the vectors would be unreachable
           forever.
        2. **Vectors, then S3, then the row.** The database is the authority on what may be
           cited: while the row still exists the document is already `DELETED` and therefore
           unretrievable, so a failure part-way through leaves orphaned vectors or an orphaned
           object — wasteful, but not quotable. Deleting the row first and then failing would
           leave the opposite: bytes and vectors with no record that they exist, and nothing
           in the system able to find them again.
        3. **Audit last, and it still lands.** `audit_log` has no foreign key to `document`,
           so the entry naming a row that no longer exists is valid and permanent. That is the
           point: an erasure has to leave a trace, or it is indistinguishable from a gap
           (`.claude/rules/audit-log-immutability.md`).

        The S3 failure is the one that is allowed to abort the operation, because
        `delete_object` raises `StorageError` rather than returning a partial count — the
        number goes into the compliance record, and a wrong number there is worse than a
        retry. Vector removal stays best-effort for the reason given in `soft_delete`.
        """
        if actor.role is not UserRole.ADMINISTRATOR:
            await self._audit_document(
                actor,
                context,
                action=AuditAction.DOCUMENT_PURGE,
                document_id=document_id,
                outcome=AuditOutcome.DENIED,
                detail={"reason": "role_not_administrator"},
            )
            raise UnauthorizedAccess()

        document = await self.get_document(
            document_id,
            actor=actor,
            context=context,
            denied_action=AuditAction.DOCUMENT_PURGE,
        )
        if document.status is not DocumentStatus.DELETED:
            await self._audit_document(
                actor,
                context,
                action=AuditAction.DOCUMENT_PURGE,
                document_id=document_id,
                outcome=AuditOutcome.DENIED,
                detail={"reason": "not_soft_deleted", "status": document.status.value},
            )
            raise DocumentNotSoftDeleted()

        # Kept uniform with the other mutations rather than strictly required: supersession
        # skips rows that are already `DELETED`, and nothing but a `DELETED` row can be
        # purged, so this pair cannot contend today. Taking it anyway means the invariant is
        # "every document mutation claims the name first", with no exception a later edit has
        # to remember.
        await self._documents.lock_name(document.name, actor.tenant_id)

        # Captured before anything is destroyed: in a moment there will be no row to read
        # these from, and the response still has to tell the caller what it just erased.
        name = document.name
        version = document.version
        storage_key = document.s3_key

        vectors_deleted = await self._remove_vectors(document_id, actor.tenant_id)
        try:
            s3_versions_deleted = await self._storage.delete_object(storage_key)
        except StorageError:
            # A half-finished destruction still has to leave a trace. By this point the
            # vectors are gone and the row is not, so without this entry the only record of
            # the attempt would be an application log line — and `audit-log-immutability.md`
            # is explicit that the audit trail, not the log stream, is what a compliance
            # review reads. The row survives the raise because `get_db` commits on a
            # `DocQueryError` rather than rolling back.
            await self._audit_document(
                actor,
                context,
                action=AuditAction.DOCUMENT_PURGE,
                document_id=document_id,
                outcome=AuditOutcome.FAILURE,
                detail={
                    "stage": "storage",
                    "vectors_deleted": vectors_deleted,
                    "chunks_deleted": 0,
                    "s3_versions_deleted": 0,
                },
            )
            raise
        chunks_deleted = await self._documents.delete_chunks(
            document_id, actor.tenant_id
        )
        # Belt and braces: the cascade on `document_chunk.document_id` would have taken the
        # chunks anyway. Doing it explicitly means the count in the audit record is measured
        # rather than assumed, and the behaviour does not change if that constraint ever does.
        removed = await self._documents.delete_document(document_id, actor.tenant_id)
        if not removed:
            # The row vanished between the status check and here — a concurrent purge. Not an
            # error worth raising: the caller's intent is satisfied either way, and the audit
            # entry below records what this request did.
            logger.warning("document.purge_row_already_gone", document_id=str(document_id))

        await self._audit_document(
            actor,
            context,
            action=AuditAction.DOCUMENT_PURGE,
            document_id=document_id,
            outcome=AuditOutcome.SUCCESS,
            detail={
                # **No filename, even here.** It is tempting: the row is gone, so the name is
                # about to be unrecoverable, and an id looks like a poor record. But these
                # names carry customer identifiers — a clinic and a patient surname, a party
                # to a rental agreement — and `secure-coding-vapt.md` forbids PII in a log
                # regardless of how convenient it would be. `DOCUMENT_UPLOAD` recorded no
                # name either, so the trail stays joinable by `object_id` from upload to
                # destruction, which is what a reviewer actually follows. The name does go
                # back to the caller in the response, where it is read once and not stored.
                "version": version,
                "chunks_deleted": chunks_deleted,
                "vectors_deleted": vectors_deleted,
                "s3_versions_deleted": s3_versions_deleted,
                # Stated in the record so a reviewer never has to infer it from the action.
                "irreversible": True,
            },
        )
        logger.info(
            "document.purged",
            document_id=str(document_id),
            chunks=chunks_deleted,
            vectors=vectors_deleted,
            s3_versions=s3_versions_deleted,
        )
        return PurgeResult(
            document_id=document_id,
            name=name,
            chunks_deleted=chunks_deleted,
            vectors_deleted=vectors_deleted,
            s3_versions_deleted=s3_versions_deleted,
        )

    # -- Audit helper -------------------------------------------------------
    async def _audit_document(
        self,
        actor: User,
        context: RequestContext,
        *,
        action: AuditAction,
        document_id: uuid.UUID,
        outcome: AuditOutcome,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """One place where document audit rows are shaped.

        `object_id` is the document id and nothing else. Names, extracted text and metadata
        values never enter `detail` — the audit log records *which* object was acted on, not
        what was inside it.
        """
        await self._audit.log(
            tenant_id=actor.tenant_id,
            user_id=actor.id,
            action=action,
            object_type=AuditObjectType.DOCUMENT,
            object_id=str(document_id),
            outcome=outcome,
            actor_email=actor.email,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            detail=detail,
        )


@dataclass(frozen=True, slots=True)
class RetiredVersion:
    """One version taken out of search because a newer one replaced it.

    Returned rather than audited in place, because the audit row has to be written *after*
    this work rather than during it — see `supersede_previous_versions`. It therefore carries
    `previous_status`, which stops being readable from the row the instant the status changes.
    """

    document_id: uuid.UUID
    tenant_id: uuid.UUID
    version: int
    previous_status: DocumentStatus
    uploaded_by: uuid.UUID | None
    #: Vectors submitted for deletion, or `-1` when no vector store was configured.
    vectors_removed: int


@dataclass(frozen=True, slots=True)
class SupersessionOutcome:
    """What retirement did, and what it deliberately refused to do.

    `divergent` is the half worth having. A filename is not a document identity — two
    unrelated documents can be called `invoice.pdf` — so a version sharing a name but *not*
    the bytes is not evidence that it was replaced. Those are listed rather than retired, and
    the caller surfaces them so a human can decide. Silently removing them from search would
    be the system acting on a guess about which document this is.
    """

    retired: tuple[RetiredVersion, ...] = ()
    #: Version numbers left searchable because their content differs from the new upload.
    divergent: tuple[int, ...] = ()


async def supersede_previous_versions(
    document: Document,
    *,
    documents: DocumentRepository,
    vectors: VectorStore | None = None,
    settings: Settings | None = None,
) -> SupersessionOutcome:
    """Retire the older versions of `document`'s name that hold the same bytes.

    ## Identity is the content, not the filename

    The name selects the *candidates*; `content_hash` decides. An earlier version is retired
    only when its SHA-256 matches this upload's, because that is the one case with no
    judgement in it: the same bytes under the same name are redundant by definition, and one
    of the two copies is pure cost — a passage citable twice and a retrieval slot spent.

    A version that shares the name but not the content is **left searchable** and reported in
    `SupersessionOutcome.divergent`. Two unrelated documents can both be called `invoice.pdf`,
    and nothing in a filename or in the bytes distinguishes "a corrected revision of this
    document" from "a different document that happens to share a name". Guessing wrong there
    removes a live document from search silently, which in a banking corpus is worse than
    leaving a duplicate: the duplicate is visible in the answer, the absence is not.

    Retiring a divergent version therefore stays a human decision — `soft_delete` already *is*
    that action, it is one click in the documents table, and it is reversible. Note that
    filename-keyed lineage predates this function: `next_version`, `list_versions` and the
    Versions panel all group by name. This narrows what the system is willing to *act* on
    without changing that grouping.

    Called by the ingestion pipeline once a re-upload has become retrievable — not by
    `upload`, and the difference is the whole safety property. `upload` returns while the
    document is still `PROCESSING`; retiring the predecessor there would mean a v3 that then
    failed to parse had already taken v2 out of the index, leaving a name with *no* searchable
    version and an administrator with nothing to query. Waiting until v3 is `ACTIVE` means the
    corpus always has one good copy of a name, before and after.

    The rule, not the mechanism, is why this sits in the services layer while its caller is a
    worker (`architecture.md`). The worker supplies the collaborators because it owns the
    session; everything about *when* a version stops being quotable is decided here.

    Each retirement is the same two steps as `soft_delete`, in the same order and for the same
    reason: status first — PostgreSQL is the authority on what may be cited, so the document
    stops being quotable at that instant — then a best-effort vector removal that is not
    allowed to undo it. Nothing is erased: the row, the S3 object, the chunk text and the
    audit history all survive, so `Versions` still lists it and `Restore` brings it back.

    **This writes no audit rows, and that is deliberate.** `audit_supersessions` does, and the
    caller runs it immediately afterwards. Splitting them keeps every row write and every
    Pinecone round trip on *this* side of `pg_advisory_xact_lock('audit_chain:<tenant>')`:
    appending an audit row takes that lock and holds it until commit, so auditing each
    retirement as it happened would put a network call inside it and stall every other audited
    operation for the tenant — the blast radius `INGEST_LOCAL_CONCURRENCY` otherwise removes.
    It is also half of why two concurrent ingestions of one filename cannot deadlock;
    `DocumentRepository.lock_name` is the other half, and the half that does not depend on
    this ordering being preserved by whoever edits the pipeline next.

    A no-op when the setting is off, when the document is not retrievable, or when it is v1.
    Idempotent, because `list_earlier_versions` skips what is already `DELETED`.
    """
    resolved = settings or get_settings()
    if not resolved.DOCUMENT_SUPERSEDE_PREVIOUS_VERSIONS:
        return SupersessionOutcome()
    # Guarded here as well as at the call site. This function is the one that knows why the
    # precondition exists, so it is the one that should not depend on being called correctly.
    if document.status not in RETRIEVABLE_STATUSES or document.version <= 1:
        return SupersessionOutcome()

    earlier = await documents.list_earlier_versions(
        document.name, document.tenant_id, before_version=document.version
    )
    retired: list[RetiredVersion] = []
    divergent: list[int] = []
    for stale in earlier:
        if stale.content_hash != document.content_hash:
            # Same name, different bytes. Not evidence of replacement — see the docstring.
            # Reported so the administrator can retire it deliberately if it *was* a
            # revision, which is what `soft_delete` is for.
            divergent.append(stale.version)
            continue

        # Read **before** the update, exactly as `soft_delete` does. `set_status` issues an
        # ORM-enabled UPDATE, and SQLAlchemy's default `synchronize_session` evaluates the
        # criteria against the identity map — so `stale.status` becomes `DELETED` the moment
        # the statement runs, and an audit row built afterwards would record the status it was
        # changed *to* as the status it came *from*. A wrong audit record is worse than none.
        previous_status = stale.status
        uploaded_by = stale.uploaded_by
        version = stale.version
        await documents.set_status(stale.id, stale.tenant_id, DocumentStatus.DELETED)

        removed = -1
        if vectors is not None:
            chunks = await documents.get_chunks(stale.id, stale.tenant_id)
            removed = 0
            if chunks:
                # Never raises — see `soft_delete`. A failure here leaves vectors behind for a
                # document PostgreSQL already says is `DELETED`, so nothing can cite it; the
                # opposite order would leave a document that is listed, looks healthy and
                # matches nothing.
                removed = await vectors.delete_document_vectors(
                    stale.tenant_id, stale.id, [chunk.id for chunk in chunks]
                )

        retired.append(
            RetiredVersion(
                document_id=stale.id,
                tenant_id=stale.tenant_id,
                version=version,
                previous_status=previous_status,
                uploaded_by=uploaded_by,
                vectors_removed=removed,
            )
        )

    if retired or divergent:
        logger.info(
            "document.superseded_previous_versions",
            document_id=str(document.id),
            version=document.version,
            retired=len(retired),
            # Counted, never named: the point of this log line is that a human has a decision
            # waiting, and a document name can carry a customer identifier.
            divergent=len(divergent),
        )
    return SupersessionOutcome(retired=tuple(retired), divergent=tuple(divergent))


async def audit_supersessions(
    retired: Sequence[RetiredVersion],
    *,
    audit: AuditService | None,
    superseded_by: Document,
    actor_email: str,
) -> None:
    """Record each retirement in the audit trail. A no-op when no audit service was supplied.

    Separate from the work it describes so that every row write and every Pinecone call
    happens *before* the audit chain lock is taken — see `supersede_previous_versions`. Called
    immediately after it and in the same transaction, so a committed retirement always has its
    row and a rolled-back one has neither.

    `AuditAction.DOCUMENT_DELETE` with a `phase`, rather than a new action: the effect on the
    document *is* a soft-delete, and a new enum member would need `ALTER TYPE` on a native
    PostgreSQL enum. The phase is what lets a reviewer tell "an administrator removed this"
    from "the system retired this when a newer version landed".
    """
    if audit is None:
        return
    for item in retired:
        await audit.log(
            tenant_id=item.tenant_id,
            action=AuditAction.DOCUMENT_DELETE,
            object_type=AuditObjectType.DOCUMENT,
            object_id=str(item.document_id),
            outcome=AuditOutcome.SUCCESS,
            # The subsystem, because no human is in the call stack on the worker path: this
            # runs after the upload request has already responded.
            actor_email=actor_email,
            # The uploader of the *superseded* document. `user_id` is the subject across this
            # table, not the actor (`audit_service.AuditQuery`); who *caused* the retirement is
            # `detail.caused_by_user_id` below.
            user_id=item.uploaded_by,
            detail={
                "phase": AUDIT_PHASE_SUPERSEDE,
                "version": item.version,
                "previous_status": item.previous_status.value,
                "soft_delete": True,
                "vectors_removed": item.vectors_removed,
                # No filename — same rule as everywhere else in this module. The ids of the
                # replacement and of whoever uploaded it are what a reviewer needs, and
                # neither is PII.
                "superseded_by": str(superseded_by.id),
                "superseded_by_version": superseded_by.version,
                # Recorded in the row rather than one join away: resolving it through
                # `superseded_by` works only while that document still exists, and `purge`
                # deletes the row — after which the attribution would be unrecoverable.
                "caused_by_user_id": (
                    str(superseded_by.uploaded_by)
                    if superseded_by.uploaded_by is not None
                    else None
                ),
            },
        )
