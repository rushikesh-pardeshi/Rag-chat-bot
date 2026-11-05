"""backend/workers/document_processor.py

Role: Ingest an uploaded document — fetch from S3, parse, chunk, embed, upsert to Pinecone,
then set the final status. Runs after the upload request has already returned.

The order is the whole design
----------------------------
    PROCESSING (set by upload) -> lock the name -> parse -> chunk -> write chunk rows
    -> embed -> upsert -> mark chunks embedded -> ACTIVE -> retire older versions -> audit

`ACTIVE` is set **last**, and that is the only reason the pipeline is safe to interrupt. A
document is retrievable only when its status says so, and its status says so only when its
vectors are in the index. Every intermediate crash leaves a document that is visible in the
admin list as `PROCESSING` or `FAILED` and quotable by nobody.

The reverse order would be a compliance defect rather than a bug: an `ACTIVE` document with no
vectors answers every question with "not found" while appearing healthy, and an `ACTIVE`
document with half its vectors gives *incomplete* answers with confident citations — which is
worse, because nothing looks wrong.

Idempotent by construction
--------------------------
Retryable at any point, because:
  * `delete_chunks` runs before writing new ones, so a retry does not duplicate rows;
  * the Pinecone vector id is `str(chunk.id)`, so an upsert of the same chunk overwrites
    itself; and
  * `embedded_at` is set only after the upsert is acknowledged, so a partially embedded
    document is distinguishable from a complete one.

`task_acks_late=True` in `celery_app.py` means a worker killed mid-embedding gets the task
redelivered, and the three properties above are what make that redelivery harmless.

Whose transaction runs the pipeline
-----------------------------------
`_run_pipeline` does the work against a `DocumentRepository` it is handed. Two callers supply
that repository differently, and the difference is not cosmetic:

  * `ingest_document` — the Celery path. Opens its own `session_scope`, because a worker
    process has no ambient transaction.
  * `ingest_document_in_session` — the eager path, running inside the upload request. It must
    reuse the request's repository, because at that moment the `document` row has only been
    *flushed*. A second session would not see it and the pipeline would report the document
    missing on every single upload — an eager-mode-only failure that no amount of Celery
    testing would catch.

The failure paths differ for the same reason. The Celery path writes `FAILED` from a *second*
session, opened after the pipeline's session has unwound: writing it from inside the failing
transaction would be discarded by `session_scope`'s rollback, and a second session opened while
the first still holds the row lock would block until the statement timeout — leaving the
document `PROCESSING` forever, slowly. The eager path has no such problem: it writes `FAILED`
on the request's own repository and lets the request commit it.

Idempotency is what makes both safe. See above.

Every terminal outcome is audited
---------------------------------
Ingestion decides whether a document ever becomes searchable, so its outcome is a critical
activity under `.claude/rules/audit-log-immutability.md` and gets an `audit_log` row —
`DOCUMENT_UPLOAD` with `detail.phase = "ingest"`, mirroring how `query_service` distinguishes
its two phases. Without it, a document that failed to parse would be `FAILED` in the admin list
with nothing in the audit trail explaining when or why, which is precisely the question a
compliance review asks.

`user_id` on that row is `document.uploaded_by` — the person whose upload caused this work —
while `actor_email` is `INGESTION_ACTOR_EMAIL`, because the *actor* is a background worker and
claiming the uploader's own address for a row they were not present for would be a false
record.

The two early returns are not audited, and that is not an omission. "Document row is gone" has
nobody to attribute the row to, and "document was deleted while queued" already has a
`DOCUMENT_DELETE` entry saying exactly that, from the administrator who did it.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, replace
from typing import Any, Final

from backend.core.config import Settings, get_settings
from backend.core.database import session_scope
from backend.core.exceptions import OcrUnavailable, StorageError, ValidationError
from backend.core.logging import get_logger
from backend.models.audit_log import AuditAction, AuditObjectType, AuditOutcome
from backend.models.document import Document, DocumentChunk, DocumentStatus
from backend.repositories.audit_repository import AuditRepository
from backend.repositories.document_repository import DocumentRepository
from backend.services.audit_service import AuditService
from backend.services.bedrock_client import (
    BedrockClient,
    BedrockUnavailable,
    get_bedrock_client,
)
from backend.services.document_service import (
    audit_supersessions,
    supersede_previous_versions,
)
from backend.services.storage_service import StorageBackend, get_storage_service
from backend.services.textract_client import OcrClient, get_ocr_client
from backend.utils.chunk_splitter import split_sections
from backend.utils.document_parser import (
    CONTENT_TYPE_PDF,
    ParsedDocument,
    ParsedSection,
    clean_text,
    parse_document,
)
from backend.utils.document_parser import (
    MAX_PAGES as PARSER_MAX_PAGES,
)
from backend.utils.ids import new_id, utc_now
from backend.utils.vector_store import (
    ChunkVector,
    VectorStore,
    VectorStoreUnavailable,
    get_vector_store,
)
from backend.workers.celery_app import celery_app

logger = get_logger(__name__)

#: Seconds a real (non-eager) dispatch waits before the worker may pick the task up, so the
#: upload request's transaction has committed and the `document` row is visible. See
#: `IngestionQueue.enqueue`.
DISPATCH_DELAY_SECONDS: Final = 5

#: `actor_email` on an ingestion audit row. A marker value, like `audit_service.UNKNOWN_IP`:
#: `actor_email` is NOT NULL and no human is present when a worker runs, so the field names the
#: subsystem instead. Not a real address, and not a routable domain.
INGESTION_ACTOR_EMAIL: Final = "system@docquery.ingestion"

#: Discriminates the ingestion audit row from the "upload accepted" one that
#: `document_service` writes, since both use `AuditAction.DOCUMENT_UPLOAD`. Same reasoning as
#: `query_service.AUDIT_PHASE_SUBMIT`: a new enum member needs `ALTER TYPE` on a native
#: PostgreSQL enum, and `detail` is JSONB.
AUDIT_PHASE_INGEST: Final = "ingest"


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """What the pipeline did, for logs and for the task's return value.

    Counts only. No document name, no passage text — this value is serialized into Celery's
    result backend, which is Redis, which has its own access model and no audit trail.
    """

    document_id: str
    status: str
    chunks: int = 0
    embedded: int = 0
    reason: str = ""


async def ingest_document(
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    *,
    storage: StorageBackend | None = None,
    bedrock: BedrockClient | None = None,
    vectors: VectorStore | None = None,
    ocr: OcrClient | None = None,
    settings: Settings | None = None,
) -> IngestionResult:
    """Run the pipeline in its own transaction — the Celery path.

    Guarantees the status never stays `PROCESSING`. A document that cannot be parsed is a fact
    about the document, not a failure of the system, so it ends as `NEEDS_OCR` and this returns
    normally. An unexpected exception is recorded as `FAILED` and then re-raised, so Celery can
    retry it — but the row is already correct by then, which is what an administrator watching
    the list needs.

    Collaborators are injectable so a test can run the whole pipeline against a fake S3, a fake
    Bedrock and a `FakePineconeIndex`, with no network at all.
    """
    try:
        async with session_scope() as session:
            return await _run_pipeline(
                document_id,
                tenant_id,
                documents=DocumentRepository(session),
                storage=storage or get_storage_service(),
                bedrock=bedrock or get_bedrock_client(),
                vectors=vectors or get_vector_store(),
                ocr=ocr if ocr is not None else get_ocr_client(settings),
                settings=settings or get_settings(),
                audit=AuditService(AuditRepository(session)),
            )
    except Exception as exc:
        logger.error(
            "ingest.failed",
            document_id=str(document_id),
            # Type, never the message: a botocore error names the account and the model arn,
            # and a parser error can echo document content.
            error_type=type(exc).__name__,
        )
        # Safe here and only here — the pipeline's transaction has finished unwinding, so its
        # lock on this document row is released. See the module docstring.
        await _mark_failed(document_id, tenant_id, exc)
        raise


async def ingest_document_in_session(
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    *,
    documents: DocumentRepository,
    storage: StorageBackend | None = None,
    bedrock: BedrockClient | None = None,
    vectors: VectorStore | None = None,
    ocr: OcrClient | None = None,
    settings: Settings | None = None,
    audit: AuditService | None = None,
) -> IngestionResult:
    """Run the pipeline inside the caller's transaction — the eager path.

    Used when ingestion happens during the upload request, where the `document` row exists only
    as a flush on the caller's session. `FAILED` is written on that same repository and left
    for the request to commit; there is no second session, so no lock to contend with.

    `audit` is the caller's own service, so the ingestion row joins the same hash chain in the
    same transaction as the "upload accepted" row that precedes it.

    **`OCR_ON_UPLOAD` is honoured here and nowhere else**, because this is the entry point that
    shares the request's transaction. That transaction already holds
    `pg_advisory_xact_lock('audit_chain:<tenant>')` — taken when `document_service.upload`
    appended its "upload accepted" row — and the lock is released only at commit. So an OCR
    wait on this path queues every other audited operation for the tenant behind it: logins,
    queries, previews, denials. With the setting off, a scan lands `NEEDS_OCR` exactly as it
    did before OCR existed, and `scripts/reocr_documents.py` reads it later from outside any
    request. The worker path (`ingest_document`) has its own transaction and no such
    constraint, so it always offers OCR.
    """
    resolved = settings or get_settings()
    if ocr is None and resolved.OCR_ON_UPLOAD:
        ocr = get_ocr_client(resolved)
    try:
        return await _run_pipeline(
            document_id,
            tenant_id,
            documents=documents,
            storage=storage or get_storage_service(),
            bedrock=bedrock or get_bedrock_client(),
            vectors=vectors or get_vector_store(),
            ocr=ocr,
            settings=resolved,
            audit=audit,
        )
    except Exception as exc:
        logger.error(
            "ingest.failed",
            document_id=str(document_id),
            error_type=type(exc).__name__,
        )
        try:
            await documents.set_status(document_id, tenant_id, DocumentStatus.FAILED)
            document = await documents.get(document_id, tenant_id)
            await _audit_ingest(
                audit,
                tenant_id=tenant_id,
                document_id=document_id,
                uploaded_by=document.uploaded_by if document else None,
                outcome=AuditOutcome.FAILURE,
                detail=_failure_detail(exc),
            )
        except Exception:  # noqa: BLE001 - a DB-level failure already poisoned this session
            # Nothing more to do: the caller's transaction will roll back, the document stays
            # `PROCESSING`, and the original exception is the one worth surfacing.
            logger.error("ingest.mark_failed_failed", document_id=str(document_id))
        raise


async def _run_pipeline(
    document_id: uuid.UUID,
    tenant_id: uuid.UUID,
    *,
    documents: DocumentRepository,
    storage: StorageBackend,
    bedrock: BedrockClient,
    vectors: VectorStore,
    settings: Settings,
    ocr: OcrClient | None = None,
    audit: AuditService | None = None,
) -> IngestionResult:
    """The pipeline, step by step. Raises on anything unexpected; owns no transaction.

    The order of the last three steps is a correctness property, not a sequence of
    conveniences: publish, then retire what this version replaced, then audit both. Step 9
    takes the tenant's audit-chain lock and holds it to commit, so everything that writes a
    row or calls Pinecone has to be finished by then — see `supersede_previous_versions`.
    """
    document = await documents.get(document_id, tenant_id)

    if document is None:
        # Not an error worth raising: the document may have been removed by an operator
        # between enqueue and execution. Nothing to do, and a retry would not help.
        logger.warning("ingest.document_missing", document_id=str(document_id))
        return IngestionResult(
            document_id=str(document_id), status="missing", reason="not_found"
        )

    if document.status is DocumentStatus.DELETED:
        # Deleted while queued. Embedding it would put vectors in the index for a document
        # that must not be quotable — the exact state `load_chunks` exists to catch, and
        # there is no reason to create it deliberately.
        logger.info("ingest.skipped_deleted", document_id=str(document_id))
        return IngestionResult(
            document_id=str(document_id), status="skipped", reason="deleted"
        )

    # -- 0. Claim this name for the rest of the transaction ------------------
    # Before the first write, and so before the audit chain lock. Step 8 writes the rows of
    # *other* versions of this name while holding that chain lock, which without this would
    # let two concurrent ingestions of one filename deadlock — see
    # `DocumentRepository.lock_name` for the cycle and what the loser leaves behind.
    #
    # After the read above, not before it, because the name comes from the row; the read takes
    # no lock, so nothing has been ordered yet.
    await documents.lock_name(document.name, tenant_id)

    # -- 1. Fetch the bytes back from S3 ------------------------------------
    # Deliberately re-read rather than carried through the queue. The bytes are up to
    # `MAX_UPLOAD_SIZE_MB`; a task argument that large goes through the broker and lands in
    # Redis, putting a third copy of customer content in a system with its own access model —
    # the same objection `vector_store` raises about metadata.
    data = await storage.get_bytes(document.s3_key)

    # -- 2. Parse -----------------------------------------------------------
    parsed = parse_document(data, content_type=document.content_type)

    # -- 2a. OCR the pages that have no text layer --------------------------
    # Between parsing and the `has_text` decision, so a scan becomes an ordinary document
    # from here on: the same chunking, the same embeddings, the same citations with the same
    # page numbers. A failure leaves `parsed` untouched and the document falls through to
    # `NEEDS_OCR` below — which is exactly where it would have landed without OCR at all.
    ocr_pages = 0
    if ocr is not None and document.content_type == CONTENT_TYPE_PDF:
        try:
            parsed, ocr_pages = await _apply_ocr(
                parsed,
                document,
                ocr=ocr,
                bucket=settings.S3_BUCKET_NAME,
                settings=settings,
            )
        except OcrUnavailable:
            # Already logged with its error code by the client. No credentials, no
            # permission, an unreadable scan or a job that timed out — all of them mean the
            # same thing here, and none of them is a reason to fail an upload.
            logger.info("ingest.ocr_unavailable", document_id=str(document_id))
        except asyncio.CancelledError:
            # `CancelledError` is a `BaseException`, so it unwinds straight past every
            # `except Exception` recovery handler in this module: the transaction rolls back
            # and the row stays `PROCESSING` forever — invisible as a failure and with
            # nothing in the audit trail. Worker shutdown, a proxy timeout or a SIGTERM
            # during the OCR poll all land here, so the wait degrades to "no OCR" and
            # ingestion carries on to a terminal status.
            logger.warning("ingest.ocr_cancelled", document_id=str(document_id))

    if not parsed.has_text:
        # A scan with no text layer. Not a failure — it is a real document that needs OCR,
        # which is out of scope for this phase. Recorded as such so an administrator can see
        # why it never became searchable.
        await documents.set_parse_result(
            document_id,
            tenant_id,
            status=DocumentStatus.NEEDS_OCR,
            page_count=parsed.page_count,
            metadata=_ingest_metadata(
                document, parsed, chunk_count=0, truncated=parsed.truncated
            ),
        )
        logger.info("ingest.needs_ocr", document_id=str(document_id))
        await _audit_ingest(
            audit,
            tenant_id=tenant_id,
            document_id=document_id,
            uploaded_by=document.uploaded_by,
            outcome=AuditOutcome.SUCCESS,
            detail={"status": DocumentStatus.NEEDS_OCR.value, "reason": "no_text_layer"},
        )
        return IngestionResult(
            document_id=str(document_id),
            status=DocumentStatus.NEEDS_OCR.value,
            reason="no_text_layer",
        )

    # -- 3. Chunk -----------------------------------------------------------
    chunk_set = split_sections(parsed.sections, settings=settings)
    if not chunk_set.chunks:
        # Text existed but nothing survived the minimum-length filter — a cover page, or a
        # form with only field labels. Same outcome as a scan, for the same reason: there is
        # nothing citable to retrieve.
        await documents.set_parse_result(
            document_id,
            tenant_id,
            status=DocumentStatus.NEEDS_OCR,
            page_count=parsed.page_count,
            metadata=_ingest_metadata(
                document, parsed, chunk_count=0, truncated=parsed.truncated
            ),
        )
        logger.info("ingest.no_chunks", document_id=str(document_id))
        await _audit_ingest(
            audit,
            tenant_id=tenant_id,
            document_id=document_id,
            uploaded_by=document.uploaded_by,
            outcome=AuditOutcome.SUCCESS,
            detail={"status": DocumentStatus.NEEDS_OCR.value, "reason": "no_chunks"},
        )
        return IngestionResult(
            document_id=str(document_id),
            status=DocumentStatus.NEEDS_OCR.value,
            reason="no_chunks",
        )

    metadata = _ingest_metadata(
        document,
        parsed,
        chunk_count=len(chunk_set.chunks),
        truncated=chunk_set.truncated or parsed.truncated,
    )

    # -- 4. Write chunk rows ------------------------------------------------
    # Cleared first so a retry replaces rather than duplicates — `uq_chunk_document_index`
    # would reject the second attempt otherwise. Chunks are derived data, regenerable from the
    # S3 object.
    #
    # The ids are read *before* the delete because the vectors are keyed by chunk id and the
    # replacements get fresh UUIDs. Without this list the old vectors stay in the namespace
    # forever — see the cleanup after the upsert below.
    superseded = [chunk.id for chunk in await documents.get_chunks(document_id, tenant_id)]
    await documents.delete_chunks(document_id, tenant_id)
    rows = [
        DocumentChunk(
            id=new_id(),
            tenant_id=tenant_id,
            document_id=document_id,
            chunk_index=chunk.chunk_index,
            text=chunk.text,
            page_number=chunk.page_number,
            section_label=chunk.section_label,
            token_count=chunk.token_count,
        )
        for chunk in chunk_set.chunks
    ]
    await documents.add_chunks(rows)
    await documents.set_parse_result(
        document_id,
        tenant_id,
        # Still not ACTIVE. The rows exist; the vectors do not yet.
        status=DocumentStatus.PROCESSING,
        page_count=parsed.page_count,
        metadata=metadata,
    )

    # -- 5. Embed -----------------------------------------------------------
    embeddings = await bedrock.embed([row.text for row in rows])

    # -- 6. Upsert into the tenant's namespace ------------------------------
    # `strict=True`: a length mismatch between chunks and vectors would zip them silently and
    # attach every embedding to the wrong passage from that point on, producing citations that
    # point at unrelated pages. That must be an exception, not a shrug.
    chunk_vectors = [
        ChunkVector(
            chunk_id=row.id,
            document_id=document_id,
            document_name=document.name,
            vector=vector,
            page_number=row.page_number,
            section_label=row.section_label,
            chunk_index=row.chunk_index,
        )
        for row, vector in zip(rows, embeddings, strict=True)
    ]
    upserted = await vectors.upsert_embeddings(tenant_id, chunk_vectors)

    # The previous run's vectors, now orphaned. A reindex (or a retry after a partial
    # ingest) writes chunks with new ids, so the old vectors are not overwritten — they
    # accumulate, one stale set per run.
    #
    # They cannot produce a wrong citation: `load_chunks` finds no row for a deleted chunk id
    # and drops the match, which is exactly what `PINECONE_OVERFETCH_FACTOR` pays for. What
    # they do is *crowd the candidate window* — the search asks for `RAG_TOP_K *
    # OVERFETCH` matches, and every slot an orphan takes is one a real passage cannot. Enough
    # reindexes and a document stops answering questions about itself while looking healthy.
    #
    # After the upsert, not before: the namespace then always holds at least one usable set,
    # so a failure here leaves stale-but-harmless vectors rather than none at all. Deleted
    # best-effort for the same reason `soft_delete` is — losing this cleanup is untidy, while
    # failing the ingest over it would undo work that succeeded.
    stale = [chunk_id for chunk_id in superseded if chunk_id not in {row.id for row in rows}]
    if stale:
        removed = await vectors.delete_document_vectors(tenant_id, document_id, stale)
        logger.info(
            "ingest.stale_vectors_removed",
            document_id=str(document_id),
            removed=removed,
        )

    # -- 7. Record the acknowledgement, then publish -------------------------
    await documents.mark_chunks_embedded(
        [row.id for row in rows], tenant_id, embedded_at=utc_now()
    )
    await documents.set_status(document_id, tenant_id, DocumentStatus.ACTIVE)

    logger.info(
        "ingest.completed",
        document_id=str(document_id),
        chunks=len(rows),
        embedded=upserted,
        pages=parsed.page_count,
    )
    # -- 8. Retire what this version replaced --------------------------------
    # Only on this path: the document is `ACTIVE` and its vectors are upserted, so taking the
    # previous version out of the index cannot leave the name without a searchable copy. A
    # `NEEDS_OCR` or `FAILED` return above never reaches here, which is the point — see
    # `supersede_previous_versions`.
    #
    # **Before the audit writes below, not after.** Appending an audit row takes
    # `pg_advisory_xact_lock('audit_chain:<tenant>')` and holds it to commit; doing the row
    # updates and the Pinecone deletes afterwards would put them inside that lock and stall
    # every other audited operation for the tenant. `audit_supersessions` then records them
    # once the lock is held anyway, so the trail is identical either way.
    #
    # `document.status` is still whatever was read at the top of the pipeline, so the retiring
    # rule would refuse it; align it with the status just written.
    document.status = DocumentStatus.ACTIVE
    supersession = await supersede_previous_versions(
        document,
        documents=documents,
        vectors=vectors,
        settings=settings,
    )

    # -- 9. Audit, from here on holding the tenant's audit-chain lock --------
    # Not swallowed, and deliberately before the return: "this document became searchable" and
    # "we recorded that it did" have to be one atomic fact. If the audit write fails, the
    # caller's rollback takes `ACTIVE` with it and the document is retried — an unsearchable
    # document is recoverable, an unaudited state change is not.
    await _audit_ingest(
        audit,
        tenant_id=tenant_id,
        document_id=document_id,
        uploaded_by=document.uploaded_by,
        outcome=AuditOutcome.SUCCESS,
        detail={
            "status": DocumentStatus.ACTIVE.value,
            "chunks": len(rows),
            "embedded": upserted,
            # How many pages were readable only by machine. A page count, never the text —
            # OCR output *is* the document body. Present on every row including `0`, so no
            # reader has to decide what an absent key meant.
            "ocr_pages": ocr_pages,
            # Version numbers only — no names. `divergent_versions` is the trail's record that
            # a same-named version was deliberately *not* retired because its content differs,
            # which is the difference between a considered decision and an oversight when
            # somebody asks later why two versions of one name are both answering.
            "superseded_versions": [item.version for item in supersession.retired],
            "divergent_versions": list(supersession.divergent),
        },
    )
    await audit_supersessions(
        supersession.retired,
        audit=audit,
        superseded_by=document,
        actor_email=INGESTION_ACTOR_EMAIL,
    )

    return IngestionResult(
        document_id=str(document_id),
        status=DocumentStatus.ACTIVE.value,
        chunks=len(rows),
        embedded=upserted,
    )


async def _mark_failed(
    document_id: uuid.UUID, tenant_id: uuid.UUID, exc: BaseException
) -> None:
    """Set `FAILED` and audit it in a second transaction, swallowing any further failure.

    If this cannot be written either, the original exception is the one worth surfacing —
    raising from here would replace a parse error with a database error and hide the cause.
    """
    try:
        async with session_scope() as session:
            documents = DocumentRepository(session)
            await documents.set_status(document_id, tenant_id, DocumentStatus.FAILED)
            document = await documents.get(document_id, tenant_id)
            await _audit_ingest(
                AuditService(AuditRepository(session)),
                tenant_id=tenant_id,
                document_id=document_id,
                uploaded_by=document.uploaded_by if document else None,
                outcome=AuditOutcome.FAILURE,
                detail=_failure_detail(exc),
            )
    except Exception:  # noqa: BLE001 - best effort; the caller re-raises the real cause
        logger.error("ingest.mark_failed_failed", document_id=str(document_id))


async def _audit_ingest(
    audit: AuditService | None,
    *,
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
    uploaded_by: uuid.UUID | None,
    outcome: AuditOutcome,
    detail: dict[str, Any],
) -> None:
    """Write one ingestion audit row. A no-op when no audit service was supplied.

    `audit` is optional because `_run_pipeline` is also called directly by tests that assert on
    statuses and counts rather than on the trail; making it mandatory would force every one of
    them to build a repository they do not otherwise need. Both production entry points supply
    one, so the row exists on every real ingestion.
    """
    if audit is None:
        return
    await audit.log(
        tenant_id=tenant_id,
        action=AuditAction.DOCUMENT_UPLOAD,
        object_type=AuditObjectType.DOCUMENT,
        object_id=str(document_id),
        outcome=outcome,
        actor_email=INGESTION_ACTOR_EMAIL,
        user_id=uploaded_by,
        detail={"phase": AUDIT_PHASE_INGEST, **detail},
    )


async def _apply_ocr(
    parsed: ParsedDocument,
    document: Document,
    *,
    ocr: OcrClient,
    bucket: str,
    settings: Settings,
) -> tuple[ParsedDocument, int]:
    """Fill pages that have no text layer with Textract's reading. Returns what changed.

    **Per page, not per document**, and that is the whole point. `ParsedDocument.has_text` is
    `any(...)`, so a PDF with two typed pages and two scanned ones already counted as having
    text: it became `ACTIVE`, looked healthy, and silently indexed half of itself. Answers
    then cited the typed pages and never mentioned what was only in the images. Filling the
    unread pages individually fixes that case as well as the all-scanned one.

    **A page with no text layer is missing from `sections`, not present and empty.**
    `_parse_pdf` does `if not text: continue`, so the join is on page *numbers*: whichever of
    `1..page_count` has no section is a page nothing was read from — whether because it is a
    scan or because `extract_text` raised on it, which this repairs too. A page that did
    yield text is left alone: the embedded text layer is what the document's author wrote,
    while OCR is a reading of a picture of it, and replacing the former with the latter loses
    accented characters, ligatures and table structure for no gain.

    Sections come back ordered by page, because `split_sections` assigns `chunk_index` in
    iteration order — appending OCR pages at the end would number a two-page document's
    chunks 1, 2 for page 2 and 3, 4 for page 1, and a reader following citations would find
    the passages out of order.

    PDFs only: Textract does not accept DOCX, and a DOCX has no pages to join on.

    Raises nothing of its own: `OcrUnavailable` is the caller's to handle, and any other
    exception means a bug worth surfacing rather than swallowing.
    """
    # The parser stops at its own `MAX_PAGES`; asking Textract about pages that were never
    # going to be extracted would bill for nothing.
    considered = min(parsed.page_count, PARSER_MAX_PAGES)
    present = {
        section.page_number for section in parsed.sections if section.page_number is not None
    }
    missing = [number for number in range(1, considered + 1) if number not in present]
    if not missing:
        return parsed, 0
    if parsed.truncated:
        # The parser stopped at `MAX_TEXT_CHARS` mid-document, so every page after that point
        # is absent for a reason that has nothing to do with scanning. Treating them as
        # unread would bill Textract for a text-heavy document's entire page count and then
        # append text past the cap the parser exists to enforce.
        logger.info("ocr.skipped_truncated_document", pages_absent=len(missing))
        return parsed, 0
    if considered > settings.OCR_MAX_PAGES:
        logger.warning(
            "ocr.skipped_too_many_pages",
            pages=considered,
            limit=settings.OCR_MAX_PAGES,
        )
        return parsed, 0

    pages = await ocr.detect_pages(bucket=bucket, key=document.s3_key)
    if not pages:
        return parsed, 0

    sections = list(parsed.sections)
    filled = 0
    for number in missing:
        # `clean_text`, like every other text path in this pipeline. OCR output was the one
        # body text reaching `document_chunk.text` unfiltered, and a NUL byte in a PostgreSQL
        # text column **aborts the INSERT** — which poisons the session, so the recovery
        # handler cannot even write `FAILED`, and an upload whose bytes are already in S3
        # answers 500 with the document stuck `PROCESSING`.
        recognised = clean_text(pages.get(number) or "")
        if not recognised:
            continue
        sections.append(ParsedSection(text=recognised, page_number=number))
        filled += 1

    if filled == 0:
        return parsed, 0

    # `None` sorts first and only occurs for formats without pages, which cannot reach here.
    sections.sort(key=lambda section: (section.page_number or 0))

    metadata = dict(parsed.metadata)
    # Recorded on the document so an Administrator can tell a document that was read by
    # machine from one that carried its own text — relevant when judging an answer's
    # confidence, and the sort of provenance a compliance review asks about.
    metadata["ocr"] = {
        "provider": "textract",
        "pages_recognised": filled,
        "pages_without_text_layer": len(missing),
    }
    return (
        replace(parsed, sections=tuple(sections), metadata=metadata),
        filled,
    )


def _ingest_metadata(
    document: Document, parsed: ParsedDocument, *, chunk_count: int, truncated: bool
) -> dict[str, Any]:
    """Fold this run's findings into whatever `doc_metadata` already held.

    `set_parse_result` replaces the column outright, so building this from `parsed.metadata`
    alone would erase keys the *request* wrote before the pipeline ever started — the
    `duplicate_content` flag `document_service.upload` sets is the live example, and losing it
    means an administrator reviewing a re-uploaded statement no longer sees that it was a
    duplicate. Parse output describes the file; it is not the whole record about the document.

    `chunk_count` is always present, including the two `NEEDS_OCR` paths where it is 0. A key
    that exists on some documents and is absent on others makes every reader write
    `.get("chunk_count", ...)` and guess what the absence meant.

    `truncated` is set or *removed*, never left over: `restore` re-runs this pipeline, and a
    document that no longer hits the cap must not keep claiming it does.
    """
    metadata = dict(document.doc_metadata or {})
    metadata.update(parsed.metadata)
    metadata["chunk_count"] = chunk_count
    if truncated:
        # Recorded rather than silent. A truncated document answers questions about its first
        # N pages and says nothing about the rest, and an administrator has to be able to find
        # out that is why.
        metadata["truncated"] = True
    else:
        metadata.pop("truncated", None)
    return metadata


def _failure_detail(exc: BaseException) -> dict[str, Any]:
    """Classify an ingestion failure into audit `detail`, with no message text.

    `reason` is a coarse label a compliance reviewer can filter on; `error_code` is the domain
    code where there is one. The exception *message* is never recorded — a parser error can
    quote document content and a botocore error names the account and model arn, and neither
    belongs in a table Administrators read (`secure-coding-vapt.md`, Secrets/PII).
    """
    detail: dict[str, Any] = {
        "reason": _failure_reason(exc),
        "error_type": type(exc).__name__,
    }
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        detail["error_code"] = code
    return detail


def _failure_reason(exc: BaseException) -> str:
    """Which step failed, named after the step rather than after the exception class.

    Deliberately coarse. The audit trail's question is "why did this document never become
    searchable", and "the embedding call failed" answers it; which botocore subclass was raised
    is a debugging detail and lives in `error_type` alongside.
    """
    if isinstance(exc, ValidationError):
        # Covers `UnsupportedDocumentType`, which is what a renamed archive or a corrupt PDF
        # raises out of `parse_document`.
        return "parse_failed"
    if isinstance(exc, BedrockUnavailable):
        return "embedding_failed"
    if isinstance(exc, VectorStoreUnavailable):
        return "upsert_failed"
    if isinstance(exc, StorageError):
        return "fetch_failed"
    return "ingest_failed"


@celery_app.task(
    name="documents.ingest",
    max_retries=2,
    # Retried only for infrastructure failures — a broker or endpoint that was briefly
    # unreachable. A parse error is not transient, and retrying it three times produces three
    # identical failures and three log entries.
    autoretry_for=(ConnectionError, TimeoutError),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def process_document(document_id: str, tenant_id: str) -> dict[str, object]:
    """Celery entry point. Ids arrive as strings because the serializer is JSON.

    A thin wrapper: it owns the event loop and nothing else. All the logic is in
    `ingest_document`, which is async and directly testable.

    `asyncio.run` creates a fresh loop per task. That is correct for a Celery prefork worker —
    the process is not otherwise running a loop — and it is also why this must not be called
    from inside a request, which `IngestionQueue` prevents.
    """
    result = asyncio.run(ingest_document(uuid.UUID(document_id), uuid.UUID(tenant_id)))
    return {
        "document_id": result.document_id,
        "status": result.status,
        "chunks": result.chunks,
        "embedded": result.embedded,
        "reason": result.reason,
    }


class LocalIngestionPool:
    """Concurrent ingestion inside the API process, for a host with no broker.

    This is what makes a multi-file upload genuinely parallel when
    `CELERY_TASK_ALWAYS_EAGER=true`. Each document becomes one `asyncio` task running
    `ingest_document`, which opens **its own session and its own transaction** — so four
    documents parse, OCR and embed at the same time, and none of them is holding the upload
    request open or sharing a transaction with another.

    ## Why this is not just "eager mode, but concurrent"

    Eager mode runs the pipeline inside the upload request's transaction. That transaction
    holds `pg_advisory_xact_lock('audit_chain:<tenant>')` from the moment
    `document_service.upload` appends its "upload accepted" row until the request commits, so
    two concurrent uploads could not overlap even if the browser sent them together — the
    second one's pipeline would wait on the first one's lock, and every login and query for
    the tenant would queue behind it too. Giving each document its own short transaction is
    what removes that, and it is also why `OCR_ON_UPLOAD`'s documented blast radius does not
    apply on this path.

    ## What it does not do

    The work lives in the web process, so it does not survive a restart: a document in flight
    when the API stops stays `PROCESSING` with nothing coming back for it. `Reindex` on the
    row is the one-click recovery and `scripts/reocr_documents.py` the bulk one. A real Celery
    worker is what survives a restart, and that path is untouched — set
    `CELERY_TASK_ALWAYS_EAGER=false` with a broker and nothing here runs at all.

    Bounded on purpose. Each slot is a database connection plus concurrent Bedrock and
    Textract calls, so an unbounded pool would turn a drag-and-drop of forty files into forty
    simultaneous connections and a throttling storm.
    """

    def __init__(self, concurrency: int) -> None:
        self._concurrency = max(1, concurrency)
        #: Created on first use, not here: a `Semaphore` binds to the loop that first awaits
        #: it, and this object is built while the settings are read — before the loop exists.
        self._slots: asyncio.Semaphore | None = None
        #: Strong references to the tasks in flight. Without this the event loop is the only
        #: holder, and a task it has not started yet can be garbage collected mid-ingestion.
        self._tasks: set[asyncio.Task[None]] = set()
        #: Dispatches handed to the loop by `after_commit` but not yet turned into tasks.
        #:
        #: `submit` runs one loop turn after the commit that scheduled it, so for that turn
        #: the work exists and `self._tasks` is empty. `drain` seeing zero there would let
        #: shutdown proceed to `dispose_engine`, and the task would then build a *second*
        #: engine and connection pool on its way out — leaking sockets and leaving the
        #: document `PROCESSING`. Counted at scheduling time instead, so the work is visible
        #: from the moment it is promised.
        self._scheduled = 0

    @property
    def in_flight(self) -> int:
        return len(self._tasks) + self._scheduled

    def schedule(self, loop: asyncio.AbstractEventLoop, *ids: uuid.UUID) -> None:
        """Hand a dispatch to the loop, counting it from now rather than from `submit`.

        Called from the repository's after-commit hook, which runs in SQLAlchemy's greenlet.
        `call_soon_threadsafe` is well-defined from there, where `create_task` is at best
        subtle — and it is why there is a turn of latency for `_scheduled` to cover.
        """
        self._scheduled += 1
        loop.call_soon_threadsafe(self.submit, *ids)

    def submit(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        """Start one ingestion. Returns immediately; never raises into the caller.

        Called from a loop callback (see `schedule`) rather than directly from the after-commit
        hook, so there is a running loop to attach the task to.
        """
        if self._slots is None:
            self._slots = asyncio.Semaphore(self._concurrency)
        task = asyncio.create_task(
            self._run(document_id, tenant_id),
            name=f"ingest-{document_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        # Decremented only once the task exists, so the two counts never both miss it.
        if self._scheduled > 0:
            self._scheduled -= 1

    async def _run(self, document_id: uuid.UUID, tenant_id: uuid.UUID) -> None:
        assert self._slots is not None  # noqa: S101 - set by `submit` before this is created
        async with self._slots:
            try:
                await ingest_document(document_id, tenant_id)
            except asyncio.CancelledError:
                # Shutdown, or `drain` giving up. The row stays `PROCESSING` and `Reindex`
                # is the way back; swallowing it here would be a lie, so it is re-raised.
                logger.warning("ingest.pool_cancelled", document_id=str(document_id))
                raise
            except Exception as exc:  # noqa: BLE001 - nothing above this to report to
                # `ingest_document` has already written `FAILED` and audited it by now. This
                # task has no caller, so the alternative to swallowing is an "exception was
                # never retrieved" warning at garbage-collection time and nothing else.
                logger.warning(
                    "ingest.pool_failed",
                    document_id=str(document_id),
                    error_type=type(exc).__name__,
                )

    async def drain(self, grace_seconds: float = 30.0) -> int:
        """Wait for the documents in flight, then cancel whatever is left. Returns the count.

        Called from the application's shutdown. A few seconds of patience here is the
        difference between a clean stop and a handful of rows stuck at `PROCESSING` after
        every deploy — and the grace period is what stops a wedged Textract poll from holding
        the process open indefinitely.

        Named `grace_seconds` rather than `timeout` because it is not one: this coroutine
        always returns, and the argument bounds how long it is patient before cancelling
        rather than how long before it gives up and raises.
        """
        # Let any dispatch already promised by `schedule` become a real task first, so it is
        # in `self._tasks` to be waited on rather than starting after shutdown has torn the
        # engine down. One turn is the documented latency; the loop is belt and braces for a
        # commit landing while this runs.
        for _ in range(3):
            if self._scheduled == 0:
                break
            await asyncio.sleep(0)

        pending = {task for task in self._tasks if not task.done()}
        if not pending:
            return 0
        logger.info("ingest.pool_draining", in_flight=len(pending))
        done, unfinished = await asyncio.wait(pending, timeout=grace_seconds)
        for task in unfinished:
            task.cancel()
        if unfinished:
            # Let the cancellations settle so each task's `except CancelledError` runs and
            # says which document was abandoned.
            await asyncio.gather(*unfinished, return_exceptions=True)
            logger.warning("ingest.pool_abandoned", count=len(unfinished))
        return len(done) + len(unfinished)


#: Process-wide, because `get_ingestion_queue` is a FastAPI dependency and builds a new queue
#: per request. A semaphore that is per-request bounds nothing.
_pool: LocalIngestionPool | None = None


def get_ingestion_pool(settings: Settings | None = None) -> LocalIngestionPool:
    global _pool
    if _pool is None:
        resolved = settings or get_settings()
        _pool = LocalIngestionPool(resolved.INGEST_LOCAL_CONCURRENCY)
    return _pool


async def drain_ingestion_pool(grace_seconds: float = 30.0) -> int:
    """Shutdown hook for `backend/api/app.py`. A no-op when no pool was ever built."""
    if _pool is None:
        return 0
    return await _pool.drain(grace_seconds)


class IngestionQueue:
    """How `document_service` starts ingestion, without knowing whether Celery is real.

    Injected into `DocumentService` rather than imported by it, so a test can substitute a
    queue that records calls instead of running anything — which is what keeps the upload tests
    from needing Bedrock.

    ## Three modes, chosen by configuration

    * **Celery** (`CELERY_TASK_ALWAYS_EAGER=false`). Dispatch to a broker; a separate worker
      process does the work. The only mode that survives a restart of the API.
    * **Local pool** (eager, `INGEST_LOCAL_CONCURRENCY > 0`) — the default here. Ingestion
      runs concurrently in this process, each document in its own transaction, dispatched
      *after* the upload request commits. See `LocalIngestionPool`.
    * **Inline** (eager, `INGEST_LOCAL_CONCURRENCY = 0`). The original behaviour: the pipeline
      is awaited inside the caller's transaction, so the upload response does not arrive until
      the document is indexed. Deterministic, which is why it is kept, but serial.

    The inline branch exists for a concrete reason, not for tidiness. With
    `CELERY_TASK_ALWAYS_EAGER=true`, `.delay()` runs the task body in the calling thread, and
    the task body calls `asyncio.run()`. Inside a FastAPI request that thread already has a
    running loop, so `asyncio.run` raises `RuntimeError`. Awaiting the pipeline directly is the
    same work in the same process, minus the loop conflict.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        pool: LocalIngestionPool | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._pool = pool

    @property
    def eager(self) -> bool:
        return self._settings.CELERY_TASK_ALWAYS_EAGER

    @property
    def parallel(self) -> bool:
        """True when ingestion goes to the in-process pool rather than inline or to Celery."""
        return self.eager and self._settings.INGEST_LOCAL_CONCURRENCY > 0

    async def enqueue(
        self,
        document_id: uuid.UUID,
        tenant_id: uuid.UUID,
        *,
        documents: DocumentRepository | None = None,
        audit: AuditService | None = None,
    ) -> None:
        """Start ingestion. Returns once dispatched — or once complete, in inline mode.

        `documents` is the caller's repository, and passing it is what makes the two in-process
        paths work, for opposite reasons. Inline needs it because the row it reads has only
        been flushed, not committed, so the pipeline must run on that same session. The pool
        needs it because its task runs on a *different* session, so it has to wait for this
        one to commit — `documents.after_commit` is how it knows when. `audit` travels with
        them for the inline case: that ingestion row has to be written on the session that
        will commit the status it describes.

        Both are ignored on the Celery path: that process opens its own session and builds its
        own audit service, because a repository bound to a session in the web process is not a
        thing a worker can be handed. The pool ignores `audit` for the same reason — its task
        builds one against its own session.

        Errors are logged and swallowed. The bytes are already in S3 and the row already says
        `PROCESSING`; raising here would fail the upload request for a document that *was*
        stored, and the administrator would re-upload and get a v2 they did not want. The
        pipeline has already recorded `FAILED` by this point, so the outcome is visible in the
        admin list either way.
        """
        if self.parallel and documents is not None:
            # Dispatched on the caller's commit, not now. The pool's task opens a *new*
            # session, so it can only see the `document` row once this transaction has
            # committed — submitting here would race it, the task would find nothing, and
            # ingestion would be silently skipped for a document that uploaded fine. This is
            # the exact version of the `DISPATCH_DELAY_SECONDS` guess the Celery path has to
            # make, and it is available only because the pool is in this same process.
            #
            # The loop is captured here, inside a coroutine, and the hook hands the work back
            # to it with `call_soon_threadsafe`: the after-commit callback runs in
            # SQLAlchemy's greenlet, and scheduling onto the loop explicitly is well-defined
            # from there where `create_task` is at best subtle.
            loop = asyncio.get_running_loop()
            pool = self._pool or get_ingestion_pool(self._settings)
            documents.after_commit(
                lambda: pool.schedule(loop, document_id, tenant_id)
            )
            logger.info("ingest.deferred_to_pool", document_id=str(document_id))
            return

        if self.eager:
            logger.info("ingest.eager", document_id=str(document_id))
            try:
                if documents is not None:
                    await ingest_document_in_session(
                        document_id, tenant_id, documents=documents, audit=audit
                    )
                else:
                    await ingest_document(document_id, tenant_id)
            except Exception as exc:  # noqa: BLE001 - see docstring
                logger.warning(
                    "ingest.eager_failed",
                    document_id=str(document_id),
                    error_type=type(exc).__name__,
                )
            return

        try:
            # `countdown`, not `delay()`. A worker that picked the task up instantly could
            # start before the upload request's transaction commits, read no document row, and
            # give up — ingestion silently skipped for a document that uploaded fine. A few
            # seconds is longer than any commit here takes. The rigorous fix is a
            # transactional outbox (enqueue as a row, dispatch after commit); this is the
            # honest approximation, and it is why the task treats a missing document as a
            # warning rather than an error.
            process_document.apply_async(
                args=(str(document_id), str(tenant_id)),
                countdown=DISPATCH_DELAY_SECONDS,
            )
            logger.info("ingest.enqueued", document_id=str(document_id))
        except Exception as exc:  # noqa: BLE001 - a broker outage is not an upload failure
            logger.error(
                "ingest.enqueue_failed",
                document_id=str(document_id),
                error_type=type(exc).__name__,
            )


def get_ingestion_queue() -> IngestionQueue:
    """Provider for the FastAPI dependency. Overridden in tests.

    The queue is per-request and cheap; the pool it dispatches to is process-wide, which is
    what makes `INGEST_LOCAL_CONCURRENCY` a real ceiling rather than a per-request one.
    """
    settings = get_settings()
    pool = get_ingestion_pool(settings) if settings.INGEST_LOCAL_CONCURRENCY > 0 else None
    return IngestionQueue(settings, pool=pool)
