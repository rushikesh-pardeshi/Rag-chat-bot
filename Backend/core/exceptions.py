"""backend/core/exceptions.py

Role: Domain exceptions and their HTTP mapping.

Services raise these; they know nothing about HTTP. The handler in `backend/api/app.py`
turns them into responses. Client-facing messages are deliberately plain — no stack
traces, no SQL, no internal identifiers.
"""

from __future__ import annotations

from http import HTTPStatus


class DocQueryError(Exception):
    """Base class. `status_code` and `code` drive the HTTP mapping."""

    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None, /, **context: object) -> None:
        self.message = message or self.message
        #: Non-sensitive extras for logging only. Never serialized to the client.
        self.context = context
        super().__init__(self.message)


# -- 400 ---------------------------------------------------------------------
class ValidationError(DocQueryError):
    status_code = HTTPStatus.BAD_REQUEST
    code = "validation_error"
    message = "The request was not valid."


class UnsupportedDocumentType(ValidationError):
    code = "unsupported_document_type"
    message = "Only PDF and DOCX documents are accepted."


class DocumentTooLarge(ValidationError):
    code = "document_too_large"
    message = "The document exceeds the maximum permitted size."


# -- 401 / 403 ---------------------------------------------------------------
class AuthenticationError(DocQueryError):
    status_code = HTTPStatus.UNAUTHORIZED
    code = "authentication_required"
    message = "Authentication is required."


class InvalidCredentials(AuthenticationError):
    code = "invalid_credentials"
    # Deliberately does not distinguish unknown email from wrong password.
    message = "Email or password is incorrect."


class AccountLocked(AuthenticationError):
    status_code = HTTPStatus.TOO_MANY_REQUESTS
    code = "account_locked"
    message = "Too many failed attempts. Try again later."


class UnauthorizedAccess(DocQueryError):
    status_code = HTTPStatus.FORBIDDEN
    code = "forbidden"
    message = "You do not have permission to perform this action."


class CSRFValidationError(DocQueryError):
    status_code = HTTPStatus.FORBIDDEN
    code = "csrf_failed"
    message = "Request rejected: missing or invalid CSRF token."


# -- 404 ---------------------------------------------------------------------
class NotFoundError(DocQueryError):
    status_code = HTTPStatus.NOT_FOUND
    code = "not_found"
    message = "The requested resource does not exist."


class DocumentNotFound(NotFoundError):
    code = "document_not_found"
    message = "Document not found."


class QueryNotFound(NotFoundError):
    code = "query_not_found"
    message = "Query not found."


class UserNotFound(NotFoundError):
    code = "user_not_found"
    message = "User not found."


# -- 409 ---------------------------------------------------------------------
class ConflictError(DocQueryError):
    status_code = HTTPStatus.CONFLICT
    code = "conflict"
    message = "The request conflicts with the current state."


class EmailAlreadyRegistered(ConflictError):
    code = "email_already_registered"
    message = "That email is already registered for this organization."


class DocumentNotDeleted(ConflictError):
    code = "document_not_deleted"
    message = "Only a deleted document can be restored."


class DocumentDeleted(ConflictError):
    """A soft-deleted document was asked to do something only a live one can do.

    Reindex is the case: `restore` already re-runs the pipeline, and it is the operation that
    also returns the document to circulation. Two routes that both re-ingest, one of which
    silently leaves the document unsearchable, would be a trap.
    """

    code = "document_deleted"
    message = "Restore a deleted document instead of reindexing it."


class DocumentNotSoftDeleted(ConflictError):
    """Purge refused because the document is still in circulation.

    Separate from `DocumentNotDeleted` only for its message: the two conditions are the same
    check from opposite directions, and telling an Administrator to "restore" when they asked
    to destroy something would send them the wrong way.
    """

    code = "document_not_soft_deleted"
    message = "Soft-delete the document before deleting it permanently."


# -- 422 / 500 ---------------------------------------------------------------
class RetrievalUnavailable(DocQueryError):
    status_code = HTTPStatus.SERVICE_UNAVAILABLE
    code = "retrieval_unavailable"
    message = "The document index is not available. Try again shortly."


class IntegrityChainBroken(DocQueryError):
    """Raised by audit verification. This is a compliance incident, not a bad request."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "audit_integrity_broken"
    message = "Audit log integrity verification failed."


class AuditLogImmutable(DocQueryError):
    """Raised if any code path attempts to mutate an existing audit entry."""

    status_code = HTTPStatus.INTERNAL_SERVER_ERROR
    code = "audit_log_immutable"
    message = "Audit log entries cannot be modified or deleted."


class StorageError(DocQueryError):
    code = "storage_error"
    message = "Document storage is unavailable."


class OcrUnavailable(DocQueryError):
    """Text could not be read off a scan — for any reason.

    Deliberately one exception for every failure mode (no credentials, no permission, an
    unsupported document, a job that failed or timed out). The pipeline's response is the same
    in all of them: leave the document `NEEDS_OCR`, which is the state it was already in. A
    scan that cannot be read is a fact about the scan and the environment, not a broken
    ingestion, so this should never reach a client as a 5xx.
    """

    code = "ocr_unavailable"
    message = "Text recognition is unavailable for this document."
