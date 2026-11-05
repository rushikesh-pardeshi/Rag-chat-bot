"""backend/services/storage_service.py

Role: The narrow interface between the application and S3. Nothing else calls boto3 for
object storage, and nothing writes a document to local disk.

**Keys are built here, never received.** `build_key` composes
`documents/{document_id}/v{n}/{safe_filename}` from a server-generated UUID, a
server-computed version, and a *sanitized* filename. A key assembled from a user-supplied
path is the S3 form of path traversal: `../../` in a filename would place the object outside
the tenant's prefix, and with a shared bucket that means writing over another tenant's data.
`sanitize_filename` reduces the name to a single path component before it is used.

**Lineage is app-side.** Our IAM role cannot enable bucket versioning, so every version gets
its own immutable key and nothing is ever overwritten (`ExtraArgs` includes no key reuse, and
`put` refuses a key that already exists). Soft-deleting a document does not touch S3 at all —
the object is what makes a restore possible, and what an audit reference points at.

**Presigned URLs are short-lived and role-checked before they are minted.** The URL itself
carries no identity, so its lifetime is the whole of its security; `S3_PRESIGNED_URL_EXPIRY_
SECONDS` defaults to 300. The tenant and role check happens in `document_service` *before*
this class is asked for a URL — once issued, a presigned URL cannot be recalled.

The `StorageBackend` protocol is what tests substitute. `FakeStorage` lives in
`backend/tests/fakes.py`; no test in this project makes a network call to AWS.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, BinaryIO, Final, Protocol, runtime_checkable

from backend.core.config import Settings, get_settings
from backend.core.exceptions import StorageError, ValidationError
from backend.core.logging import get_logger

logger = get_logger(__name__)

#: Characters permitted in a stored filename. An allowlist, not a denylist: enumerating what
#: is dangerous (`..`, `/`, `\`, NUL, `%2e%2e`, unicode look-alikes) is a game you lose once.
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
#: Leading dots would produce a hidden file or a traversal fragment.
_LEADING_JUNK = re.compile(r"^[._-]+")

MAX_FILENAME_LENGTH: Final = 180
FALLBACK_FILENAME: Final = "document"

#: Read in 1 MiB blocks when hashing, so a 50 MB upload does not need a second copy in memory.
_HASH_BLOCK_SIZE: Final = 1024 * 1024


def sanitize_filename(filename: str) -> str:
    """Reduce an arbitrary client string to one safe path component.

    Applied to every filename before it reaches a key. The steps, and why each is needed:

      1. NFKD normalization, so a unicode look-alike for `/` or `.` cannot survive as one;
      2. take the last component after both `/` and `\\`, defeating `../../etc/passwd` and
         its Windows spelling — the separators are removed rather than rejected, because a
         legitimate browser on Windows really does send `C:\\Users\\x\\report.pdf`;
      3. strip surrounding whitespace, *before* substitution. A trailing space would otherwise
         become `_` and turn `report.pdf ` into extension `pdf_`, which the upload allowlist
         then refuses — a real file rejected for a character the user cannot see;
      4. replace everything outside the allowlist with `_`;
      5. strip leading dots and dashes, so nothing becomes `.hidden` or looks like a flag;
      6. length-cap while preserving the extension, since the extension is what the
         allowlist check reads.

    Returns `FALLBACK_FILENAME` if nothing usable survives — an empty component would collapse
    the key to a directory prefix.
    """
    normalized = unicodedata.normalize("NFKD", filename or "")
    # Both separators, because the client's OS is unknown and either may appear.
    last_component = normalized.replace("\\", "/").rsplit("/", 1)[-1].strip()
    cleaned = _SAFE_FILENAME.sub("_", last_component)
    cleaned = _LEADING_JUNK.sub("", cleaned)
    if not cleaned or cleaned in {".", ".."}:
        return FALLBACK_FILENAME

    if len(cleaned) <= MAX_FILENAME_LENGTH:
        return cleaned
    stem, dot, extension = cleaned.rpartition(".")
    if dot and len(extension) <= 10:
        keep = MAX_FILENAME_LENGTH - len(extension) - 1
        return f"{stem[:keep]}.{extension}"
    return cleaned[:MAX_FILENAME_LENGTH]


def file_extension(filename: str) -> str:
    """Lowercase extension with no dot, from an already-sanitized name."""
    _, dot, extension = sanitize_filename(filename).rpartition(".")
    return extension.lower() if dot else ""


def _aws_error_code(exc: Exception) -> str | None:
    """The AWS error code from a botocore `ClientError`, or None.

    Worth logging on its own: `error_type=ClientError` is true of a missing bucket, a denied
    policy and an expired token alike, and the three need different fixes. The *code* is a
    fixed AWS identifier (`ExpiredToken`, `AccessDenied`, `NoSuchBucket`) — no bucket name,
    no key material, nothing from the request — so it is safe where `str(exc)` is not.
    """
    error = getattr(exc, "response", None)
    if isinstance(error, dict):
        code = error.get("Error", {}).get("Code")
        if isinstance(code, str):
            return code[:64]
    return None


def compute_content_hash(data: bytes) -> str:
    """SHA-256 of the stored bytes — the lineage fingerprint on `Document.content_hash`."""
    digest = hashlib.sha256()
    for start in range(0, len(data), _HASH_BLOCK_SIZE):
        digest.update(data[start : start + _HASH_BLOCK_SIZE])
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StoredObject:
    """What `put` wrote. `key` is the immutable identifier the document row records."""

    key: str
    size_bytes: int
    content_hash: str


@runtime_checkable
class StorageBackend(Protocol):
    """The surface `document_service` depends on.

    Deliberately narrow. Anything wider — listing a prefix, copying, changing an ACL — would
    be an operation the service does not need and a test would have to fake. `delete_object`
    is here only because permanent deletion exists: nothing else in the application removes
    an object, and soft delete explicitly does not (the bytes are what makes a restore
    possible).
    """

    def build_key(self, *, document_id: Any, version: int, filename: str) -> str: ...

    async def put(
        self, *, key: str, data: bytes, content_type: str, tenant_id: Any
    ) -> StoredObject: ...

    async def get_presigned_url(self, key: str, *, ttl_seconds: int | None = None) -> str: ...

    async def get_bytes(self, key: str) -> bytes: ...

    async def delete_object(self, key: str) -> int: ...


class S3StorageService:
    """S3-backed implementation. The only boto3 client for object storage in the codebase."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings or get_settings()
        #: Injected in tests, and created lazily otherwise — building a boto3 client at
        #: import time would make every test that imports this module need credentials.
        self._client = client

    # -- Client -------------------------------------------------------------
    def _get_client(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "s3",
                region_name=self._settings.S3_REGION,
                # Only ever from settings. A user-controlled endpoint would be SSRF with a
                # signed request attached (`.claude/rules/secure-coding-vapt.md`).
                endpoint_url=self._settings.S3_ENDPOINT_URL,
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                    connect_timeout=10,
                    read_timeout=60,
                ),
                **self._settings.bedrock_credentials(),
            )
        return self._client

    # -- Keys ---------------------------------------------------------------
    def build_key(self, *, document_id: Any, version: int, filename: str) -> str:
        """`documents/{document_id}/v{n}/{safe_filename}`.

        Every component is server-controlled: `document_id` is a UUID we generated, `version`
        is a counter we computed, and `filename` is passed through `sanitize_filename`. There
        is no argument to this method that a caller can use to escape the prefix.
        """
        if version < 1:
            raise ValidationError("Document version must be at least 1.")
        prefix = self._settings.S3_DOCUMENT_PREFIX.strip("/")
        safe_name = sanitize_filename(filename)
        return f"{prefix}/{document_id}/v{version}/{safe_name}"

    # -- Operations ---------------------------------------------------------
    async def put(
        self, *, key: str, data: bytes, content_type: str, tenant_id: Any
    ) -> StoredObject:
        """Write an object, refusing to overwrite an existing key.

        `IfNoneMatch="*"` makes the guarantee the server's rather than ours: a retried upload
        that reused a key would otherwise silently replace the bytes an audit entry refers to.
        Falls back to a HEAD-then-PUT on a backend that does not support the precondition,
        which is racy but still catches the realistic case (a retry seconds later).
        """
        client = self._get_client()
        extra: dict[str, Any] = {
            "Bucket": self._settings.S3_BUCKET_NAME,
            "Key": key,
            "Body": data,
            "ContentType": content_type,
            "ServerSideEncryption": self._settings.S3_SERVER_SIDE_ENCRYPTION,
            # Tenant on the object itself, so a bucket-level audit can attribute an object
            # without consulting our database. Not a security boundary — the database is.
            "Metadata": {"tenant_id": str(tenant_id)},
        }
        try:
            try:
                client.put_object(**extra, IfNoneMatch="*")
            except TypeError as exc:
                # Older botocore, or a fake that does not accept the parameter.
                if await self._exists(key):
                    raise StorageError(
                        "A document already exists at that storage key."
                    ) from exc
                client.put_object(**extra)
        except StorageError:
            raise
        except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
            # The key is safe to log (server-generated); the exception text is not echoed to
            # the client, because botocore messages can name the bucket.
            logger.error(
                "storage.put_failed",
                key=key,
                error_type=type(exc).__name__,
                aws_code=_aws_error_code(exc),
            )
            raise StorageError() from exc

        return StoredObject(
            key=key, size_bytes=len(data), content_hash=compute_content_hash(data)
        )

    async def get_presigned_url(
        self, key: str, *, ttl_seconds: int | None = None
    ) -> str:
        """A time-limited download URL.

        Callers must have already checked tenant and role: this URL is a bearer credential
        that cannot be revoked once handed out, which is why the TTL is minutes and why the
        check does not happen here.
        """
        ttl = ttl_seconds or self._settings.S3_PRESIGNED_URL_EXPIRY_SECONDS
        client = self._get_client()
        try:
            return str(
                client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self._settings.S3_BUCKET_NAME, "Key": key},
                    ExpiresIn=ttl,
                )
            )
        except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
            logger.error(
                "storage.presign_failed",
                key=key,
                error_type=type(exc).__name__,
                aws_code=_aws_error_code(exc),
            )
            raise StorageError() from exc

    async def get_bytes(self, key: str) -> bytes:
        """Read an object back — used by the Phase 4 processor, not by any request path."""
        client = self._get_client()
        try:
            response = client.get_object(
                Bucket=self._settings.S3_BUCKET_NAME, Key=key
            )
            body: BinaryIO = response["Body"]
            return bytes(body.read())
        except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
            logger.error(
                "storage.get_failed",
                key=key,
                error_type=type(exc).__name__,
                aws_code=_aws_error_code(exc),
            )
            raise StorageError() from exc

    async def delete_object(self, key: str) -> int:
        """Erase one object and **every version of it**. Returns how many versions went.

        `S3_VERSIONING_ENABLED` is true for document lineage, and on a versioned bucket a
        plain `delete_object` does not delete: it writes a *delete marker* and keeps every
        previous version, retrievable by anyone who can name a version id. For a feature
        whose entire promise is "this document is gone", that is not deletion — so this
        enumerates the key's versions and delete markers and removes them explicitly.

        One `list_object_versions` call per page, then a batched `delete_objects`, because
        S3 caps a batch at 1000 keys. `Prefix` is an exact object key here, not a directory,
        and the results are filtered on `Key == key` regardless: a prefix listing would
        otherwise also match `report.pdf.bak` when asked for `report.pdf`.

        Raises `StorageError` rather than returning a partial count, because the caller
        audits this number — reporting "2 versions deleted" when the second call failed
        would put a false claim in the compliance record.
        """
        client = self._get_client()
        bucket = self._settings.S3_BUCKET_NAME
        removed = 0
        try:
            paginator = client.get_paginator("list_object_versions")
            for page in paginator.paginate(Bucket=bucket, Prefix=key):
                targets = [
                    {"Key": entry["Key"], "VersionId": entry["VersionId"]}
                    for group in ("Versions", "DeleteMarkers")
                    for entry in page.get(group, [])
                    if entry.get("Key") == key
                ]
                for start in range(0, len(targets), 1000):
                    batch = targets[start : start + 1000]
                    response = client.delete_objects(
                        Bucket=bucket, Delete={"Objects": batch}
                    )
                    # **`delete_objects` is a partial-success API.** A version that cannot be
                    # removed — object-lock retention, a policy denying
                    # `s3:DeleteObjectVersion`, MFA-delete — comes back in `Errors` and does
                    # *not* raise. Counting `len(batch)` would then put a false number in an
                    # append-only audit row: bytes still retrievable in S3, no row left in
                    # PostgreSQL that can find them, and a compliance record asserting they
                    # were destroyed. So the response is inspected, and the count is what S3
                    # says it deleted rather than what we asked it to.
                    failures = response.get("Errors") or []
                    if failures:
                        logger.error(
                            "storage.delete_incomplete",
                            key=key,
                            failed=len(failures),
                            # The S3 error code only — `Message` can echo the key back.
                            aws_code=str(failures[0].get("Code", "unknown")),
                        )
                        raise StorageError()
                    removed += len(response.get("Deleted") or [])
            if removed == 0:
                # Nothing listed: an unversioned bucket whose listing came back empty, or a
                # key that is already gone. The plain call still runs so the object goes
                # either way, but the count stays 0 — S3 answers 204 for a key that was never
                # there, and reporting "1 version deleted" on the strength of that would be
                # the same unverified claim this method exists to avoid.
                client.delete_object(Bucket=bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
            logger.error(
                "storage.delete_failed",
                key=key,
                error_type=type(exc).__name__,
                aws_code=_aws_error_code(exc),
            )
            raise StorageError() from exc
        logger.info("storage.deleted", key=key, versions=removed)
        return removed

    async def _exists(self, key: str) -> bool:
        client = self._get_client()
        try:
            client.head_object(Bucket=self._settings.S3_BUCKET_NAME, Key=key)
            return True
        except Exception:  # noqa: BLE001 - a 404 arrives as ClientError
            return False


def get_storage_service() -> S3StorageService:
    """Factory for the FastAPI dependency. One client per process via boto3's own caching."""
    return S3StorageService()
