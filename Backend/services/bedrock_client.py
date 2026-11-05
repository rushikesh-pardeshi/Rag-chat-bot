"""backend/services/bedrock_client.py

Role: The only module that talks to Amazon Bedrock. Embeddings via Titan, generation via
Nova Pro. Nothing else in the codebase constructs a `bedrock-runtime` client.

Two models, two different APIs
------------------------------
`amazon.titan-embed-text-v2:0` has no batch endpoint — one `InvokeModel` call per text.
The loop here is therefore not a missed optimization; it is the API. Calls are issued
concurrently with a bounded semaphore so embedding a 200-chunk document is not 200 serial
round trips.

`us.amazon.nova-pro-v1:0` is called through **Converse**, not `InvokeModel`. Converse
normalizes the system/messages shape across model families, so the prompt-building code in
`rag_service` does not encode Nova's native request format.

boto3 is synchronous; every call is pushed to a worker thread with `asyncio.to_thread` so
the event loop is never blocked. A shared client is safe: boto3 clients are thread-safe for
API calls (only resource creation is not).

Grounding is not enforced here
------------------------------
This module has no opinion about grounding. It sends what it is given and returns what
comes back. The rules from `.claude/rules/rag-answer-grounding.md` — retrieved chunks only,
no fallback to model knowledge, mandatory citations — are enforced in `rag_service` and
`source_tracer`, because they are about *what* is in the prompt and *whether* the answer is
usable, and neither question belongs in a transport wrapper.

SSRF: no URL from any request reaches boto3. Region and model ids come from
`backend/core/config.py`; the endpoint is derived by botocore from the region. There is no
code path here that accepts a caller-supplied endpoint, and adding one would be the
`secure-coding-vapt.md` SSRF finding.
"""

from __future__ import annotations

import asyncio
import functools
import json
from functools import lru_cache
from typing import Any, Final

from backend.core.config import Settings, get_settings
from backend.core.exceptions import RetrievalUnavailable
from backend.core.logging import get_logger

logger = get_logger(__name__)

#: Concurrent in-flight Bedrock calls. Titan's per-account rate limit is the real ceiling;
#: this keeps a large document from tripping it and turning one upload into a retry storm.
MAX_CONCURRENT_CALLS: Final = 8

#: botocore error codes worth retrying. Throttling and a 5xx are transient; a validation
#: error or an access denial will fail identically on every attempt, so retrying them only
#: delays the report.
RETRYABLE_ERROR_CODES: Final = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException",
    }
)

#: Longest text sent to Titan in one call. The model's own limit is ~8k tokens; chunks are
#: built to `CHUNK_SIZE_TOKENS` (512) so this is a guard against a pathological chunk, not
#: a normal code path.
MAX_EMBED_CHARACTERS: Final = 30_000


class BedrockUnavailable(RetrievalUnavailable):
    """Bedrock could not be reached, or refused, after retries.

    A subclass of `RetrievalUnavailable` so the API answers 503 rather than 500: the request
    was valid and may well succeed on a retry. The client sees the parent's generic message
    — a botocore error string can name the account, the model arn and the region, none of
    which belongs in a response body.
    """

    code = "bedrock_unavailable"


class BedrockClient:
    """Async-facing wrapper over `bedrock-runtime`.

    Construct via `get_bedrock_client()` rather than directly, so one client is shared for
    the process. Creating a boto3 client costs a credential resolution and a session build;
    doing that per request is measurable against the 3–5 second query target.
    """

    def __init__(self, settings: Settings | None = None, client: Any = None) -> None:
        self._settings = settings or get_settings()
        self._client = client  # injected in tests; built lazily otherwise
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

    @property
    def dimensions(self) -> int:
        """Embedding width. Must match the Pinecone index, which is fixed at creation."""
        return self._settings.BEDROCK_EMBEDDING_DIMENSIONS

    # -- boto3 plumbing -----------------------------------------------------
    def _runtime(self) -> Any:
        """Build the boto3 client on first use.

        Lazy so that importing this module — which the test suite does — never resolves AWS
        credentials. With `AWS_ACCESS_KEY_ID` blanked in the test harness, an eager client
        would fail at import time and take the whole suite with it.
        """
        if self._client is None:
            import boto3
            from botocore.config import Config

            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self._settings.BEDROCK_REGION,
                config=Config(
                    read_timeout=self._settings.BEDROCK_TIMEOUT_SECONDS,
                    connect_timeout=10,
                    # botocore's own retries are disabled: this class implements the
                    # backoff, and two independent retry layers multiply into a
                    # worst-case wait far past QUERY_TIMEOUT_SECONDS.
                    retries={"max_attempts": 1, "mode": "standard"},
                ),
                **self._settings.bedrock_credentials(),
            )
        return self._client

    async def _call_with_retry(self, operation: str, **kwargs: Any) -> dict[str, Any]:
        """Invoke one boto3 operation in a thread, retrying transient failures.

        Exponential backoff without jitter. Jitter matters when many clients retry in
        lockstep against a shared service; here the concurrency is already capped at
        `MAX_CONCURRENT_CALLS` per process, and a deterministic schedule is testable —
        `Math.random`-style nondeterminism in a retry loop makes a flaky test.
        """
        attempts = max(1, self._settings.BEDROCK_MAX_RETRIES)
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                async with self._semaphore:
                    client = self._runtime()
                    # `partial` rather than a lambda: a lambda closing over a loop variable is
                    # the late-binding bug B023 exists to catch. Harmless here because
                    # `to_thread` calls it immediately, but the form should not need that
                    # argument to be safe.
                    method = functools.partial(getattr(client, operation), **kwargs)
                    return await asyncio.to_thread(method)
            except Exception as exc:  # noqa: BLE001 - classified immediately below
                last_error = exc
                code = _error_code(exc)
                if code not in RETRYABLE_ERROR_CODES or attempt == attempts - 1:
                    # Log the code, never the exception text: a botocore message carries
                    # the account id and the model arn.
                    logger.warning(
                        "bedrock.call_failed",
                        operation=operation,
                        error_code=code,
                        attempt=attempt + 1,
                    )
                    raise BedrockUnavailable() from exc
                backoff = 0.5 * (2**attempt)
                logger.info(
                    "bedrock.retrying",
                    operation=operation,
                    error_code=code,
                    attempt=attempt + 1,
                    sleep_seconds=backoff,
                )
                await asyncio.sleep(backoff)

        raise BedrockUnavailable() from last_error

    # -- Embeddings ---------------------------------------------------------
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each text with Titan. Order of the result matches order of the input.

        Order is part of the contract: the caller zips these vectors back onto the chunks
        they came from, so a reordered result would attach every embedding to the wrong
        passage and produce citations that point at unrelated pages.
        """
        if not texts:
            return []

        # `gather` preserves input order regardless of completion order.
        vectors = await asyncio.gather(*(self._embed_one(text) for text in texts))

        expected = self.dimensions
        for index, vector in enumerate(vectors):
            if len(vector) != expected:
                # A width mismatch means the model id or the dimensions setting changed
                # without the Pinecone index being recreated. Upserting anyway is rejected
                # by Pinecone with a far less obvious error, so it is caught here.
                logger.error(
                    "bedrock.embedding_dimension_mismatch",
                    expected=expected,
                    actual=len(vector),
                    index=index,
                )
                raise BedrockUnavailable()
        return list(vectors)

    async def _embed_one(self, text: str) -> list[float]:
        payload = {
            "inputText": text[:MAX_EMBED_CHARACTERS],
            # Titan v2 accepts an output width; it must equal the Pinecone index dimension.
            "dimensions": self.dimensions,
            "normalize": self._settings.BEDROCK_EMBEDDING_NORMALIZE,
        }
        response = await self._call_with_retry(
            "invoke_model",
            modelId=self._settings.BEDROCK_EMBEDDING_MODEL_ID,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(payload),
        )
        return _read_embedding(response)

    # -- Generation ---------------------------------------------------------
    async def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        """Generate an answer with Nova Pro via Converse.

        `temperature` defaults to `BEDROCK_LLM_TEMPERATURE`, which is 0.0. That is a
        compliance property, not a style preference: the same question over the same
        documents must produce the same answer, because a regulator reviewing an audit trail
        has to be able to reproduce what the system told the user.
        """
        response = await self._call_with_retry(
            "converse",
            modelId=self._settings.BEDROCK_LLM_MODEL_ID,
            system=[{"text": system}],
            messages=messages,
            inferenceConfig={
                "maxTokens": max_tokens or self._settings.BEDROCK_LLM_MAX_TOKENS,
                "temperature": (
                    self._settings.BEDROCK_LLM_TEMPERATURE
                    if temperature is None
                    else temperature
                ),
            },
        )
        return _read_converse_text(response)


def _error_code(exc: Exception) -> str:
    """Pull botocore's error code out of a ClientError without importing botocore.

    Kept import-free so this module can be reasoned about — and tested — without AWS
    libraries loaded. `response` is a plain dict on every botocore exception.
    """
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        if isinstance(error, dict):
            code = error.get("Code")
            if isinstance(code, str):
                return code
    return type(exc).__name__


def _read_embedding(response: dict[str, Any]) -> list[float]:
    """Decode Titan's response body into a vector."""
    body = response.get("body")
    raw = body.read() if hasattr(body, "read") else body
    try:
        parsed = json.loads(raw)
        embedding = parsed["embedding"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.error("bedrock.malformed_embedding_response")
        raise BedrockUnavailable() from exc
    if not isinstance(embedding, list) or not embedding:
        raise BedrockUnavailable()
    return [float(value) for value in embedding]


def _read_converse_text(response: dict[str, Any]) -> str:
    """Concatenate the text blocks of a Converse reply.

    Nova returns `output.message.content` as a list of blocks. Only `text` blocks are read;
    anything else (a tool-use block, say) is ignored rather than stringified, because a
    stringified block would end up quoted to the user as if it were part of the answer.
    """
    try:
        blocks = response["output"]["message"]["content"]
    except (KeyError, TypeError) as exc:
        logger.error("bedrock.malformed_converse_response")
        raise BedrockUnavailable() from exc
    parts = [
        block["text"]
        for block in blocks
        if isinstance(block, dict) and isinstance(block.get("text"), str)
    ]
    return "\n".join(parts).strip()


@lru_cache(maxsize=1)
def get_bedrock_client() -> BedrockClient:
    """The process-wide client. Overridden in tests via a FastAPI dependency."""
    return BedrockClient()
