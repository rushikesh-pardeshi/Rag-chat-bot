"""backend/services/textract_client.py
Role: Read text off a scanned document with AWS Textract. The only `textract` client here.

A PDF with no text layer is a photograph of a document. `document_parser` extracts nothing
from it and the pipeline parks it as `NEEDS_OCR`; this module is what turns that into text so
the normal chunk → embed → cite path can run.

## Why the asynchronous API, and what that costs

Textract has two surfaces. `DetectDocumentText` is synchronous and accepts a **single image**
only. Anything multi-page — which every PDF here is — must go through
`StartDocumentTextDetection`, a job that is started and then polled. There is no synchronous
option for a PDF, so the wait is not a design choice; the only choices are where to wait and
for how long, and those are `OCR_TIMEOUT_SECONDS` and the caller's.

The job reads the object **from S3 directly**. The bytes are already there, so nothing is
downloaded into this process, re-uploaded, or written to a temp file — which also means no
path for a crafted filename to reach a filesystem (`.claude/rules/secure-coding-vapt.md`).

## What comes back, and what this returns

Textract answers with `Block` records. `LINE` blocks carry the text, each tagged with the
`Page` it came from; `WORD` blocks are the same text again, one word at a time. Only `LINE` is
read, grouped per page, so the result maps page number → text and the pipeline can keep every
citation pointing at the page the words are actually on (BR-03, `source-tracing`).

Nothing here logs extracted text or a document name. OCR output *is* the document body — the
thing `secure-coding-vapt.md` forbids putting in a log — and the only observable facts are
page counts, block counts and error codes.

## Failure is not an error

`OcrUnavailable` is raised for every failure mode: no credentials, no permission, an
unsupported document, a job that fails or never finishes. The caller is expected to catch it
and leave the document `NEEDS_OCR` — the state it was already in. A scan that cannot be read
today is a fact about the scan and the environment, not a broken ingestion.
"""

from __future__ import annotations

import asyncio
import functools
import time
from typing import Any, Final, Protocol, runtime_checkable

from backend.core.config import Settings, get_settings
from backend.core.exceptions import OcrUnavailable
from backend.core.logging import get_logger

logger = get_logger(__name__)

#: Textract job states. `PARTIAL_SUCCESS` means some pages were read and others were not —
#: treated as success, because half a scanned contract indexed with correct page numbers beats
#: none of it, and the page map says plainly which pages produced text.
_TERMINAL_OK = frozenset({"SUCCEEDED", "PARTIAL_SUCCESS"})
_TERMINAL_BAD = frozenset({"FAILED"})

#: Ceiling on recognised text for one document, matching `document_parser.MAX_TEXT_CHARS`.
#: Two readings of the same document should not have two different limits, and without a
#: limit a service that keeps returning pagination tokens grows the page map until the
#: process dies.
MAX_OCR_CHARS: Final = 5_000_000


@runtime_checkable
class OcrClient(Protocol):
    """What the ingestion pipeline needs from an OCR provider.

    One method, taking an S3 location rather than bytes: that is what keeps the document out
    of this process's memory and off its disk. A different provider (or a fake) implements the
    same shape.
    """

    async def detect_pages(self, *, bucket: str, key: str) -> dict[int, str]: ...


class TextractOcrClient:
    """Textract-backed OCR. Construct via `get_ocr_client()`."""

    def __init__(self, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = settings or get_settings()
        #: Injected in tests, built lazily otherwise — importing this module must never
        #: resolve AWS credentials, or the test suite fails at collection.
        self._client = client

    def _textract(self) -> Any:
        if self._client is None:
            import boto3
            from botocore.config import Config

            region = self._settings.OCR_REGION or self._settings.AWS_REGION
            self._client = boto3.client(
                "textract",
                region_name=region,
                config=Config(
                    connect_timeout=10,
                    read_timeout=30,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
                **self._settings.bedrock_credentials(),
            )
        return self._client

    async def _call(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """One boto3 call, in a worker thread. boto3 is synchronous and this code is not."""
        try:
            client = self._textract()
            method = functools.partial(getattr(client, operation), **kwargs)
            return dict(await asyncio.to_thread(method))
        except Exception as exc:  # noqa: BLE001 - botocore raises a wide family
            # The code, never the message: a botocore error names the account and the bucket.
            logger.warning(
                "ocr.call_failed",
                operation=operation,
                error_type=type(exc).__name__,
                aws_code=_error_code(exc),
            )
            raise OcrUnavailable() from exc

    async def detect_pages(self, *, bucket: str, key: str) -> dict[int, str]:
        """Text per 1-based page number, for the object at `s3://bucket/key`.

        Pages Textract read but found no text on are present with an empty string, so a caller
        can tell "nothing there" from "never looked".
        """
        # **One deadline for the whole operation**, computed before the first call and shared
        # by both loops below. Scoping it to the job wait alone was a bug: once the job
        # reported SUCCEEDED the budget vanished, and the block-pagination loop would then
        # spin for as long as the service kept handing back a token — a repeated token means
        # forever, inside a request transaction that holds the tenant's audit-chain lock.
        deadline = time.monotonic() + self._settings.OCR_TIMEOUT_SECONDS

        start = await self._call(
            "start_document_text_detection",
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": key}},
        )
        job_id = str(start.get("JobId") or "")
        if not job_id:
            raise OcrUnavailable()

        first_page = await self._await_job(job_id, deadline=deadline)
        pages = _lines_by_page(first_page.get("Blocks") or [])

        # Blocks are paginated independently of document pages: a long scan returns a
        # `NextToken` and the rest of its blocks have to be collected or the tail of the
        # document silently goes missing — a document that looks ingested and answers nothing
        # about its last pages.
        token = first_page.get("NextToken")
        characters = sum(len(text) for text in pages.values())
        while token:
            if time.monotonic() >= deadline:
                logger.warning("ocr.pagination_timed_out", pages=len(pages))
                raise OcrUnavailable()
            if characters >= MAX_OCR_CHARS:
                # The same ceiling the parser applies to an embedded text layer. Without it a
                # service returning tokens indefinitely grows this dict until the process
                # dies, and the per-page concatenation below is quadratic as it does.
                logger.warning("ocr.text_cap_reached", pages=len(pages), characters=characters)
                break
            page = await self._call(
                "get_document_text_detection", JobId=job_id, NextToken=token
            )
            for number, text in _lines_by_page(page.get("Blocks") or []).items():
                pages[number] = f"{pages[number]}\n{text}" if number in pages else text
                characters += len(text)
            token = page.get("NextToken")

        logger.info(
            "ocr.completed",
            pages=len(pages),
            # Characters, not content. A count is diagnostic; the text is the document body.
            characters=sum(len(text) for text in pages.values()),
        )
        return pages

    async def _await_job(self, job_id: str, *, deadline: float) -> dict[str, Any]:
        """Poll until the job finishes, or the caller's deadline passes.

        The deadline is passed in rather than computed here, so starting the job, waiting for
        it and collecting its blocks all share one budget. What the caller cares about is how
        long an upload can block in total, and that is a single duration.

        `asyncio.sleep` yields the event loop, so a slow OCR job does not stall every other
        request in the process — though with eager ingestion it does hold the database
        transaction, which is what `OCR_ON_UPLOAD` exists to let an operator opt out of.
        """
        interval = max(0.1, self._settings.OCR_POLL_SECONDS)
        while True:
            response = await self._call("get_document_text_detection", JobId=job_id)
            status = str(response.get("JobStatus") or "")
            if status in _TERMINAL_OK:
                return response
            if status in _TERMINAL_BAD:
                logger.warning(
                    "ocr.job_failed",
                    job_status=status,
                    # Textract's own reason, which describes the document class
                    # ("UnsupportedDocumentException"), not its contents.
                    reason=str(response.get("StatusMessage") or "")[:200],
                )
                raise OcrUnavailable()
            if time.monotonic() >= deadline:
                logger.warning(
                    "ocr.timed_out",
                    waited_seconds=self._settings.OCR_TIMEOUT_SECONDS,
                    job_status=status,
                )
                raise OcrUnavailable()
            await asyncio.sleep(interval)


def _lines_by_page(blocks: list[dict[str, Any]]) -> dict[int, str]:
    """`LINE` blocks joined per page, in the order Textract returned them.

    `WORD` blocks are skipped: they repeat the same text one word at a time, so including both
    would double every page and destroy the line breaks that make a cited passage readable.
    """
    pages: dict[int, list[str]] = {}
    for block in blocks:
        if block.get("BlockType") != "LINE":
            continue
        text = str(block.get("Text") or "")
        if not text:
            continue
        # `Page` is absent for a single-page document, where every block is on page 1.
        number = int(block.get("Page") or 1)
        pages.setdefault(number, []).append(text)
    return {number: "\n".join(lines) for number, lines in pages.items()}


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            return str(error.get("Code", "unknown"))
    return "unknown"


def get_ocr_client(settings: Settings | None = None) -> TextractOcrClient | None:
    """The configured OCR client, or `None` when OCR is switched off.

    `None` rather than a no-op implementation, so the pipeline's "is OCR available" question
    is answered by the type and a disabled deployment makes no AWS client at all.
    """
    resolved = settings or get_settings()
    if not resolved.OCR_ENABLED:
        return None
    return TextractOcrClient(resolved)
