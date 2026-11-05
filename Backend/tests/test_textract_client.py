"""Unit tests for `backend/services/textract_client.py`.

`FakeOcrClient` lets the ingestion tests exercise the *merge* without AWS, which means the
client itself — job polling, block pagination, `LINE` versus `WORD`, the timeout — would
otherwise have no coverage at all. That is the half where a mistake is invisible: a dropped
`NextToken` loses the tail of a long scan, and a document that silently indexes its first few
pages looks perfectly healthy while answering nothing about the rest.

So these tests fake one level lower, at the boto3 client, exactly as `FakeS3Client` does.
Nothing here reaches AWS.
"""

from __future__ import annotations

from typing import Any

import pytest

from backend.core.exceptions import OcrUnavailable
from backend.services.textract_client import TextractOcrClient, get_ocr_client


def _line(text: str, page: int | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"BlockType": "LINE", "Text": text}
    if page is not None:
        block["Page"] = page
    return block


class FakeTextractBoto:
    """The boto3 `textract` client, as far as this module uses it.

    `responses` is the queue `get_document_text_detection` answers from, so a test can model
    a job that is `IN_PROGRESS` twice and then succeeds — which is the normal case against the
    real service and the one a single-response fake would never exercise.
    """

    def __init__(
        self,
        responses: list[dict[str, Any]],
        *,
        job_id: str = "job-1",
        start_error: Exception | None = None,
    ) -> None:
        self.responses = list(responses)
        self.job_id = job_id
        self.start_error = start_error
        self.started: list[dict[str, Any]] = []
        self.gets: list[dict[str, Any]] = []

    def start_document_text_detection(self, **kwargs: Any) -> dict[str, Any]:
        if self.start_error is not None:
            raise self.start_error
        self.started.append(kwargs)
        return {"JobId": self.job_id}

    def get_document_text_detection(self, **kwargs: Any) -> dict[str, Any]:
        self.gets.append(kwargs)
        if not self.responses:
            raise AssertionError("the client polled more times than the test allowed for")
        return self.responses.pop(0)


@pytest.fixture
def fast_settings(settings: Any) -> Any:
    """Polling without the waiting. The interval is behaviour, not something to sit through."""
    settings.OCR_POLL_SECONDS = 0.01
    settings.OCR_TIMEOUT_SECONDS = 5
    return settings


class TestReadingBlocks:
    async def test_line_blocks_are_grouped_by_page(self, fast_settings: Any) -> None:
        boto = FakeTextractBoto(
            [
                {
                    "JobStatus": "SUCCEEDED",
                    "Blocks": [
                        _line("Interest Rate Policy", 1),
                        _line("Effective April", 1),
                        _line("Early repayment carries no fee", 2),
                    ],
                }
            ]
        )
        client = TextractOcrClient(fast_settings, client=boto)

        pages = await client.detect_pages(bucket="docs", key="documents/a/v1/scan.pdf")

        assert pages == {
            1: "Interest Rate Policy\nEffective April",
            2: "Early repayment carries no fee",
        }
        # Read from S3 by reference. Nothing was downloaded into this process.
        assert boto.started == [
            {
                "DocumentLocation": {
                    "S3Object": {"Bucket": "docs", "Name": "documents/a/v1/scan.pdf"}
                }
            }
        ]

    async def test_word_blocks_are_ignored(self, fast_settings: Any) -> None:
        """`WORD` repeats the same text one word at a time.

        Including both would double every page and destroy the line breaks that make a cited
        passage readable.
        """
        boto = FakeTextractBoto(
            [
                {
                    "JobStatus": "SUCCEEDED",
                    "Blocks": [
                        _line("Account balance", 1),
                        {"BlockType": "WORD", "Text": "Account", "Page": 1},
                        {"BlockType": "WORD", "Text": "balance", "Page": 1},
                        {"BlockType": "PAGE", "Page": 1},
                    ],
                }
            ]
        )
        client = TextractOcrClient(fast_settings, client=boto)

        assert await client.detect_pages(bucket="b", key="k") == {1: "Account balance"}

    async def test_a_block_without_a_page_number_is_page_one(
        self, fast_settings: Any
    ) -> None:
        """Textract omits `Page` for a single-page document."""
        boto = FakeTextractBoto(
            [{"JobStatus": "SUCCEEDED", "Blocks": [_line("Only page")]}]
        )
        client = TextractOcrClient(fast_settings, client=boto)

        assert await client.detect_pages(bucket="b", key="k") == {1: "Only page"}

    async def test_blocks_spanning_several_responses_are_all_collected(
        self, fast_settings: Any
    ) -> None:
        """The failure this exists to prevent: a long scan losing its tail.

        Block pagination is independent of document pages, so a dropped `NextToken` silently
        truncates the document — and the result still looks like a successful ingest.
        """
        boto = FakeTextractBoto(
            [
                {
                    "JobStatus": "SUCCEEDED",
                    "Blocks": [_line("Page one line one", 1)],
                    "NextToken": "t1",
                },
                {
                    "Blocks": [_line("Page one line two", 1), _line("Page two", 2)],
                    "NextToken": "t2",
                },
                {"Blocks": [_line("Page three", 3)]},
            ]
        )
        client = TextractOcrClient(fast_settings, client=boto)

        pages = await client.detect_pages(bucket="b", key="k")

        assert pages == {
            1: "Page one line one\nPage one line two",
            2: "Page two",
            3: "Page three",
        }
        # The tokens were actually followed, in order.
        assert [call.get("NextToken") for call in boto.gets] == [None, "t1", "t2"]


class TestBudget:
    """The deadline covers starting the job, waiting for it *and* reading its blocks.

    Scoping it to the wait alone was a real bug: once the job reported `SUCCEEDED` the budget
    was gone, so the pagination loop would spin for as long as the service kept returning a
    token — forever, if it returned the same one. With eager ingestion that loop runs inside
    the transaction holding the tenant's audit-chain lock, so "forever" means the tenant's
    API stops answering, not just that one upload.
    """

    async def test_a_repeating_pagination_token_does_not_loop_forever(
        self, settings: Any
    ) -> None:
        settings.OCR_POLL_SECONDS = 0.01
        settings.OCR_TIMEOUT_SECONDS = 1

        class EndlessTokens:
            """Always `SUCCEEDED`, always another token. The shape of the hang."""

            def __init__(self) -> None:
                self.calls = 0

            def start_document_text_detection(self, **_: Any) -> dict[str, Any]:
                return {"JobId": "job-loop"}

            def get_document_text_detection(self, **_: Any) -> dict[str, Any]:
                self.calls += 1
                return {
                    "JobStatus": "SUCCEEDED",
                    "Blocks": [_line(f"line {self.calls}", 1)],
                    "NextToken": "same-token-every-time",
                }

        boto = EndlessTokens()
        client = TextractOcrClient(settings, client=boto)

        with pytest.raises(OcrUnavailable):
            await client.detect_pages(bucket="b", key="k")
        # It really was the deadline that stopped it, not an accidental cap.
        assert boto.calls > 1

    async def test_recognised_text_is_capped(self, settings: Any) -> None:
        """A page map that grows without bound takes the process with it.

        The per-page concatenation is quadratic as it grows, so this is an OOM rather than
        merely a large response.
        """
        settings.OCR_POLL_SECONDS = 0.01
        settings.OCR_TIMEOUT_SECONDS = 30
        monkey = "x" * 100_000

        class Flood:
            def start_document_text_detection(self, **_: Any) -> dict[str, Any]:
                return {"JobId": "job-flood"}

            def get_document_text_detection(self, **_: Any) -> dict[str, Any]:
                return {
                    "JobStatus": "SUCCEEDED",
                    "Blocks": [_line(monkey, 1)],
                    "NextToken": "more",
                }

        client = TextractOcrClient(settings, client=Flood())
        pages = await client.detect_pages(bucket="b", key="k")

        # Stops at the cap and returns what it has, rather than raising: the text collected so
        # far is real, and a 5 MB document is already past anything citable.
        assert 0 < len(pages[1]) < 6_000_000


class TestJobLifecycle:
    async def test_polls_until_the_job_finishes(self, fast_settings: Any) -> None:
        boto = FakeTextractBoto(
            [
                {"JobStatus": "IN_PROGRESS"},
                {"JobStatus": "IN_PROGRESS"},
                {"JobStatus": "SUCCEEDED", "Blocks": [_line("Done", 1)]},
            ]
        )
        client = TextractOcrClient(fast_settings, client=boto)

        assert await client.detect_pages(bucket="b", key="k") == {1: "Done"}
        assert len(boto.gets) == 3

    async def test_partial_success_is_treated_as_success(
        self, fast_settings: Any
    ) -> None:
        """Half a scanned contract, with correct page numbers, beats none of it.

        The page map says plainly which pages produced text, so a `PARTIAL_SUCCESS` leaves the
        unreadable pages absent rather than wrong.
        """
        boto = FakeTextractBoto(
            [{"JobStatus": "PARTIAL_SUCCESS", "Blocks": [_line("Page two only", 2)]}]
        )
        client = TextractOcrClient(fast_settings, client=boto)

        assert await client.detect_pages(bucket="b", key="k") == {2: "Page two only"}

    async def test_a_failed_job_raises_ocr_unavailable(
        self, fast_settings: Any
    ) -> None:
        boto = FakeTextractBoto(
            [{"JobStatus": "FAILED", "StatusMessage": "UnsupportedDocumentException"}]
        )
        client = TextractOcrClient(fast_settings, client=boto)

        with pytest.raises(OcrUnavailable):
            await client.detect_pages(bucket="b", key="k")

    async def test_a_job_that_never_finishes_times_out(self, settings: Any) -> None:
        """The ceiling on how long an upload can block, with eager ingestion."""
        settings.OCR_POLL_SECONDS = 0.01
        settings.OCR_TIMEOUT_SECONDS = 0
        boto = FakeTextractBoto([{"JobStatus": "IN_PROGRESS"}])
        client = TextractOcrClient(settings, client=boto)

        with pytest.raises(OcrUnavailable):
            await client.detect_pages(bucket="b", key="k")

    async def test_a_boto_failure_becomes_ocr_unavailable(
        self, fast_settings: Any
    ) -> None:
        """No credentials, no permission, a throttle — one exception for the caller.

        The pipeline's answer is the same in every case: leave the document `NEEDS_OCR`.
        """
        denied = Exception("AccessDeniedException")
        denied.response = {"Error": {"Code": "AccessDeniedException"}}  # type: ignore[attr-defined]
        boto = FakeTextractBoto([], start_error=denied)
        client = TextractOcrClient(fast_settings, client=boto)

        with pytest.raises(OcrUnavailable):
            await client.detect_pages(bucket="b", key="k")

    async def test_a_start_response_without_a_job_id_is_refused(
        self, fast_settings: Any
    ) -> None:
        class NoJobId(FakeTextractBoto):
            def start_document_text_detection(self, **kwargs: Any) -> dict[str, Any]:
                return {}

        client = TextractOcrClient(fast_settings, client=NoJobId([]))

        with pytest.raises(OcrUnavailable):
            await client.detect_pages(bucket="b", key="k")


class TestFactory:
    def test_disabled_ocr_builds_no_client_at_all(self, settings: Any) -> None:
        """`None`, not a no-op implementation.

        The pipeline asks "is OCR available" by checking the type, and a deployment with OCR
        switched off should not construct an AWS client it will never call.
        """
        settings.OCR_ENABLED = False
        try:
            assert get_ocr_client(settings) is None
        finally:
            settings.OCR_ENABLED = True

    def test_enabled_ocr_returns_a_textract_client(self, settings: Any) -> None:
        settings.OCR_ENABLED = True
        client = get_ocr_client(settings)
        assert isinstance(client, TextractOcrClient)
