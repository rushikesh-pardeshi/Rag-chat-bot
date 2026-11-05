"""backend/api/routes/documents.py

Role: HTTP routes for the document lifecycle. Validate, call exactly one service method,
return a DTO. No SQL, no business branching.

Role split (`.claude/rules/rbac-enforcement.md`, and the Phase 3 gate in `WORKBREAKDOWN.md`):

  * **Administrator only** — upload, delete, restore, and the version/lineage view. These are
    document *management*, and they are gated by the `AdminUser` dependency, which audits the
    denial before raising 403.
  * **Any authenticated user, own tenant only** — list, read, preview. A Standard User must be
    able to open the source document behind a cited answer; a citation nobody can open is not
    traceability. The service narrows what a Standard User may see (no soft-deleted documents)
    rather than the route doing it, so the rule holds for every future caller.

The upload body is read with a hard ceiling *before* it is handed to the service, so an
oversized file is refused without ever being fully buffered. Starlette has already spooled
large parts to a temporary file by then; the ceiling is what stops us from turning that
temporary file into 500 MB of process memory.
"""

from __future__ import annotations

import uuid
from typing import Final

from fastapi import APIRouter, File, Query, UploadFile, status

from backend.api.dependencies.auth import AdminUser, CurrentUser, RequestCtx
from backend.api.dependencies.services import DocumentServiceDep
from backend.core.config import get_settings
from backend.core.exceptions import DocumentTooLarge, ValidationError
from backend.models.document import DocumentStatus
from backend.schemas.document_schemas import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    DocumentListResponse,
    DocumentPreviewResponse,
    DocumentPurgeResponse,
    DocumentResponse,
    DocumentVersionsResponse,
)

router = APIRouter(prefix="/documents", tags=["documents"])

#: Read size per iteration while streaming the upload into memory. 1 MiB is large enough that
#: a 50 MB file is 50 reads, small enough that the ceiling check happens often.
_READ_CHUNK: Final = 1024 * 1024


async def _read_bounded(upload: UploadFile, limit: int) -> bytes:
    """Read an upload into memory, refusing to exceed `limit`.

    `upload.size` is not trusted as the only check: it comes from the multipart parser, and
    for a chunked request without a `Content-Length` it can be absent. Reading with a running
    total is the check that cannot be avoided by a client.
    """
    parts: list[bytes] = []
    total = 0
    while chunk := await upload.read(_READ_CHUNK):
        total += len(chunk)
        if total > limit:
            # Stop reading immediately. The remainder of the body is discarded by Starlette.
            raise DocumentTooLarge()
        parts.append(chunk)
    if total == 0:
        raise ValidationError("The uploaded file is empty.")
    return b"".join(parts)


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document (Administrator only)",
)
async def upload_document(
    actor: AdminUser,
    context: RequestCtx,
    service: DocumentServiceDep,
    file: UploadFile = File(..., description="PDF or DOCX, validated by magic bytes"),
) -> DocumentResponse:
    """Store a document, extract its text, and audit `DOCUMENT_UPLOAD`.

    The filename is treated as hostile: it is sanitized to a single path component before it
    is used in a storage key or persisted. `file.content_type` is passed to the service only
    so it can be compared with what the bytes actually are — the decision is made on magic
    bytes.
    """
    settings = get_settings()
    data = await _read_bounded(file, settings.max_upload_bytes)
    document = await service.upload(
        actor=actor,
        context=context,
        filename=file.filename or "",
        data=data,
        declared_content_type=file.content_type,
    )
    return DocumentResponse.from_entity(document)


@router.get("", response_model=DocumentListResponse, summary="List tenant documents")
async def list_documents(
    actor: CurrentUser,
    service: DocumentServiceDep,
    document_status: DocumentStatus | None = Query(
        default=None,
        alias="status",
        description="Filter by lifecycle status. DELETED is Administrator-visible only.",
    ),
    search: str | None = Query(default=None, max_length=255),
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    """A page of the caller's own tenant's documents, newest upload first."""
    page = await service.list_documents(
        actor=actor,
        status=document_status,
        name_contains=search,
        limit=limit,
        offset=offset,
    )
    return DocumentListResponse(
        items=[DocumentResponse.from_entity(item) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get(
    "/{document_id}", response_model=DocumentResponse, summary="Get one document"
)
async def get_document(
    document_id: uuid.UUID,
    actor: CurrentUser,
    context: RequestCtx,
    service: DocumentServiceDep,
) -> DocumentResponse:
    """One document. Another tenant's id returns 404 and audits a denial."""
    document = await service.get_document(document_id, actor=actor, context=context)
    return DocumentResponse.from_entity(document)


@router.get(
    "/{document_id}/preview",
    response_model=DocumentPreviewResponse,
    summary="Presigned download URL",
)
async def preview_document(
    document_id: uuid.UUID,
    actor: CurrentUser,
    context: RequestCtx,
    service: DocumentServiceDep,
) -> DocumentPreviewResponse:
    """A short-TTL presigned URL, issued only after the tenant and role check passes.

    The URL is not logged and is not stored. It is a bearer credential with a five-minute
    life, and the audit row records that a preview was issued rather than the URL itself.
    """
    preview = await service.preview(document_id, actor=actor, context=context)
    return DocumentPreviewResponse(
        document_id=preview.document_id,
        name=preview.name,
        content_type=preview.content_type,
        url=preview.url,
        expires_in_seconds=preview.expires_in_seconds,
    )


@router.delete(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Soft-delete a document (Administrator only)",
)
async def delete_document(
    document_id: uuid.UUID,
    actor: AdminUser,
    context: RequestCtx,
    service: DocumentServiceDep,
) -> DocumentResponse:
    """Set `status='DELETED'` and audit it.

    Returns the document rather than 204 so the UI can render the new state — and so it is
    evident from the response that the document still exists. Nothing is erased: the row, the
    chunks and the S3 object are all untouched (BR-01).
    """
    document = await service.soft_delete(document_id, actor=actor, context=context)
    return DocumentResponse.from_entity(document)


@router.post(
    "/{document_id}/reindex",
    response_model=DocumentResponse,
    summary="Re-run ingestion over a stored document (Administrator only)",
)
async def reindex_document(
    document_id: uuid.UUID,
    actor: AdminUser,
    context: RequestCtx,
    service: DocumentServiceDep,
) -> DocumentResponse:
    """Re-parse, re-OCR, re-chunk and re-embed from the file already in S3.

    The repair path for a document whose index is wrong rather than whose bytes are missing:
    a scan that failed OCR on an expired credential, a document ingested before OCR existed,
    a `FAILED` row from a transient outage. Returns the document at `PROCESSING` so the UI can
    show it reprocessing; `409` for a deleted document, which `restore` handles instead.
    """
    document = await service.reindex(document_id, actor=actor, context=context)
    return DocumentResponse.from_entity(document)


@router.delete(
    "/{document_id}/purge",
    response_model=DocumentPurgeResponse,
    summary="Permanently delete a soft-deleted document (Administrator only)",
)
async def purge_document(
    document_id: uuid.UUID,
    actor: AdminUser,
    context: RequestCtx,
    service: DocumentServiceDep,
) -> DocumentPurgeResponse:
    """Erase the row, its chunks, its vectors and every version of the stored file.

    Irreversible, and the only endpoint here that destroys anything. `409` unless the
    document is already soft-deleted: two deliberate steps, so no single request can erase a
    document that is still in circulation. `404` for another tenant's id, same as everywhere
    else. The audit entry survives the document — that is what makes an erasure auditable.
    """
    result = await service.purge(document_id, actor=actor, context=context)
    return DocumentPurgeResponse(
        document_id=result.document_id,
        name=result.name,
        chunks_deleted=result.chunks_deleted,
        vectors_deleted=result.vectors_deleted,
        s3_versions_deleted=result.s3_versions_deleted,
    )


@router.post(
    "/{document_id}/restore",
    response_model=DocumentResponse,
    summary="Restore a soft-deleted document (Administrator only)",
)
async def restore_document(
    document_id: uuid.UUID,
    actor: AdminUser,
    context: RequestCtx,
    service: DocumentServiceDep,
) -> DocumentResponse:
    """Return a deleted document to service. 409 if it was not deleted."""
    document = await service.restore(document_id, actor=actor, context=context)
    return DocumentResponse.from_entity(document)


@router.get(
    "/{document_id}/versions",
    response_model=DocumentVersionsResponse,
    summary="Version lineage for a document's name (Administrator only)",
)
async def list_versions(
    document_id: uuid.UUID,
    actor: AdminUser,
    context: RequestCtx,
    service: DocumentServiceDep,
) -> DocumentVersionsResponse:
    """Every version stored under this document's name, oldest first.

    Keyed off a document id rather than a raw name in the path: a name is user-supplied text
    and would have to be escaped into the URL, while an id the caller already holds proves
    they can see at least one version of it.
    """
    document = await service.get_document(document_id, actor=actor, context=context)
    versions = await service.list_versions(document.name, actor=actor)
    return DocumentVersionsResponse(
        name=document.name,
        versions=[DocumentResponse.from_entity(version) for version in versions],
    )
