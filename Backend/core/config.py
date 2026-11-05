"""backend/core/config.py

Role: Typed settings from env/.env (DB URL, secrets, Bedrock, S3 and Pinecone identifiers).
The single source of truth for configuration.

Nothing else in `backend/` may read the environment. Import `get_settings()` instead.
`scripts/` is exempt (see .claude/rules/architecture.md).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Every credential-bearing field carries `repr=False`.
#:
#: `SecretStr` already masks itself, so the five API keys are safe in a `repr`. The connection
#: URLs are not: they are plain `str`, and a deployed DSN is
#: `postgresql+asyncpg://user:PASSWORD@host/db` — the password sits in the middle of an
#: ordinary string whose *field name* (`DATABASE_URL`, `REDIS_URL`) matches nothing in
#: `logging._SENSITIVE_KEY_PATTERN`. Any `repr(settings)` that reached a log line, an exception
#: message or a traceback frame therefore printed the database password in clear.
#:
#: `repr=False` is the fix rather than converting them to `SecretStr`, because
#: `create_async_engine`, Alembic's `env.py`, Celery's config and slowapi's storage all consume
#: these as plain strings — changing the type would mean five `get_secret_value()` calls added
#: to satisfy a `repr` concern, and one of them forgotten is a startup failure.
#:
#: `backend/tests/security/test_secrets_in_logs.py` asserts this for every field the class
#: declares, so a URL added later without `repr=False` fails the suite rather than the audit.


class Settings(BaseSettings):
    """Every configurable value in the system, validated at import time.

    A missing required key raises here — before the app can serve a request with a
    half-configured security posture.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Application ---------------------------------------------------------
    APP_NAME: str = "docquery-ai-document-search"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["local", "dev", "staging", "prod"] = "local"
    #: Serve `/docs` and `/openapi.json`. `None` means "follow ENVIRONMENT", which keeps the
    #: old behaviour: on in local development, off everywhere else.
    #:
    #: It exists because `ENVIRONMENT` used to decide this *and* `cookies_secure`, two
    #: unrelated things. Wanting the interactive docs on the POC instance therefore meant
    #: setting `ENVIRONMENT=local`, which silently dropped the `Secure` flag from the session
    #: cookies on an HTTPS site. Set this to `true` instead and the cookies are unaffected.
    #:
    #: Turning it on publishes the complete API surface to anyone who can reach the origin.
    #: That is a real decision, not a formality — it is acceptable on a POC whose routes are
    #: all authenticated and RBAC-checked, and it is not acceptable in production. Reach it
    #: through an SSH tunnel instead if the instance is internet-facing and you would rather
    #: not advertise the surface (see `deploy/README.md` §6).
    ENABLE_API_DOCS: bool | None = None
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    API_V1_PREFIX: str = "/api/v1"
    # Loopback by default. Binding 0.0.0.0 would put an unauthenticated-by-default local
    # API on every interface of the developer's network; a container that needs it can set
    # HOST explicitly.
    HOST: str = "127.0.0.1"
    PORT: int = 8000

    # -- Database ------------------------------------------------------------
    DATABASE_URL: str = Field(repr=False)
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_ECHO: bool = False

    # -- Redis / Celery ------------------------------------------------------
    REDIS_URL: str = Field("redis://localhost:6379/0", repr=False)
    CELERY_BROKER_URL: str = Field("redis://localhost:6379/1", repr=False)
    CELERY_RESULT_BACKEND: str = Field("redis://localhost:6379/2", repr=False)
    CELERY_TASK_ALWAYS_EAGER: bool = False
    #: How many documents may be ingested at once *without a broker*, when
    #: `CELERY_TASK_ALWAYS_EAGER=true`.
    #:
    #: This is the setting that makes a multi-file upload actually concurrent on a host with
    #: no Redis. Above zero, `IngestionQueue` stops running the pipeline inside the upload
    #: request and instead hands the document to an in-process pool — one `asyncio` task per
    #: document, each with **its own session and its own transaction**, dispatched after the
    #: request commits. Four uploads then parse, OCR and embed at the same time instead of
    #: queueing, and the upload request returns as soon as the bytes are in S3.
    #:
    #: Two consequences worth knowing before changing it:
    #:
    #:  * It removes the `OCR_ON_UPLOAD` blast radius described below. Each document's audit
    #:    rows are written in a short transaction of their own, so OCR's wait no longer holds
    #:    `pg_advisory_xact_lock('audit_chain:<tenant>')` across the whole pipeline.
    #:  * The work lives in the API process, so a restart mid-ingestion leaves the document
    #:    `PROCESSING` with no worker coming back for it. `Reindex` in the documents table is
    #:    the recovery, and `scripts/reocr_documents.py` the bulk one. A real Celery worker
    #:    (`CELERY_TASK_ALWAYS_EAGER=false`, with a broker) is what survives a restart, and
    #:    nothing here changes that path.
    #:
    #: Zero restores the original behaviour: ingestion inline, in the caller's transaction,
    #: upload blocking until the document is indexed. Keep it bounded — each slot is a
    #: database connection plus concurrent Bedrock and Textract calls, so it must stay well
    #: under `DB_POOL_SIZE`.
    INGEST_LOCAL_CONCURRENCY: int = Field(4, ge=0, le=16)
    RATE_LIMIT_STORAGE_URI: str = Field("memory://", repr=False)

    # -- Auth / JWT / RBAC ---------------------------------------------------
    SECRET_KEY: SecretStr
    JWT_ALGORITHM: str = "HS256"
    #: 5 hours — a full working session, which is what this is set from.
    #:
    #: It also fixes the length of the *browser session*, not just the token's, and that is
    #: the part worth knowing before lowering it again. `docquery_access` is set with
    #: `max_age` equal to this, and `frontend/proxy.ts` decides whether to bounce a visitor
    #: to /login purely on that cookie being present. The refresh cookie is scoped to
    #: `/api/v1/auth/refresh`, so the proxy cannot see it and cannot know the session is
    #: still renewable. At the old 15 minutes, a user who read a page for a quarter of an
    #: hour and then clicked was redirected to /login with six days of refresh token left.
    #:
    #: The cost of a long access token is revocation latency: `POST /auth/logout` revokes the
    #: refresh token, but an access JWT already issued stays valid until it expires — there
    #: is no denylist. Disabling an account therefore takes effect for API calls within this
    #: window rather than immediately. Shortening it is a real security improvement, and it
    #: needs a session marker the proxy can read first.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 300
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    BCRYPT_ROUNDS: int = 12
    AUTH_RATE_LIMIT: str = "10/minute"
    # Rate limiting is per IP, so it does not stop a distributed attempt on one account.
    # Lockout is per account and closes that gap.
    MAX_FAILED_LOGIN_ATTEMPTS: int = 5
    ACCOUNT_LOCKOUT_MINUTES: int = 15
    CSRF_COOKIE_NAME: str = "docquery_csrf"
    ACCESS_COOKIE_NAME: str = "docquery_access"
    REFRESH_COOKIE_NAME: str = "docquery_refresh"
    CSRF_HEADER_NAME: str = "X-CSRF-Token"
    # `Secure` is derived, not configured — see the `cookies_secure` property below.
    COOKIE_SAMESITE: Literal["strict", "lax", "none"] = "strict"
    #: Left None so the cookie is host-only. Set it only when the API and UI share a
    #: registrable domain — a broader value hands the session to every subdomain.
    COOKIE_DOMAIN: str | None = None

    # -- AWS -----------------------------------------------------------------
    AWS_REGION: str = "us-east-1"
    AWS_PROFILE: str | None = None
    AWS_ACCESS_KEY_ID: SecretStr | None = None
    AWS_SECRET_ACCESS_KEY: SecretStr | None = None
    AWS_SESSION_TOKEN: SecretStr | None = None

    # -- Bedrock -------------------------------------------------------------
    BEDROCK_REGION: str = "us-east-1"
    BEDROCK_LLM_MODEL_ID: str = "us.amazon.nova-pro-v1:0"
    BEDROCK_LLM_MAX_TOKENS: int = 2048
    BEDROCK_LLM_TEMPERATURE: float = 0.0
    BEDROCK_EMBEDDING_MODEL_ID: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_EMBEDDING_DIMENSIONS: int = 1024
    BEDROCK_EMBEDDING_NORMALIZE: bool = True
    BEDROCK_MAX_RETRIES: int = 3
    BEDROCK_TIMEOUT_SECONDS: int = 30

    # -- Pinecone (the vector store) -----------------------------------------
    #: Optional so the settings object can be built without it — tests never reach Pinecone,
    #: and a missing key must fail where the client is created, with a message naming it,
    #: rather than at import of an unrelated module.
    PINECONE_API_KEY: SecretStr | None = None
    PINECONE_INDEX_NAME: str = "docquery-chunks"
    PINECONE_CLOUD: Literal["aws", "gcp", "azure"] = "aws"
    PINECONE_REGION: str = "us-east-1"
    #: Cosine, to match normalized Titan embeddings. Changing this needs a new index — the
    #: metric is fixed at index creation and cannot be altered afterwards.
    PINECONE_METRIC: Literal["cosine", "dotproduct", "euclidean"] = "cosine"
    #: One namespace per tenant: `tenant-{tenant_id}`. Isolation by namespace rather than by
    #: metadata filter, because a forgotten filter returns another tenant's chunks while a
    #: wrong namespace returns nothing. The failure modes are not symmetric, so the default
    #: has to be the safe one.
    PINECONE_NAMESPACE_PREFIX: str = "tenant-"
    PINECONE_UPSERT_BATCH_SIZE: int = 100
    PINECONE_TIMEOUT_SECONDS: int = 20
    #: Read back more matches than are cited, because retrieved chunks are re-checked against
    #: PostgreSQL for document status before use — a soft-deleted document must not be cited
    #: even if its vectors are still in the index.
    PINECONE_OVERFETCH_FACTOR: int = 3

    # -- S3 ------------------------------------------------------------------
    S3_BUCKET_NAME: str
    S3_REGION: str = "us-east-1"
    S3_DOCUMENT_PREFIX: str = "documents/"
    S3_SERVER_SIDE_ENCRYPTION: str = "AES256"
    S3_VERSIONING_ENABLED: bool = False
    S3_PRESIGNED_URL_EXPIRY_SECONDS: int = 300
    #: Override only for a local S3 stand-in (MinIO, LocalStack). Blank in `.env` means
    #: "use the real AWS endpoint" — see `_blank_is_unset`.
    S3_ENDPOINT_URL: str | None = None

    # -- Document ingestion --------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_UPLOAD_EXTENSIONS: str = "pdf,docx"
    CHUNK_SIZE_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 64
    #: Retire an earlier version of a name when a re-upload of **the same bytes** becomes
    #: searchable.
    #:
    #: Re-uploading `report.pdf` always creates a *new row* at `next_version` — that is how
    #: BR-01 lineage works and it does not change. What this decides is whether the version it
    #: replaces stays in the index.
    #:
    #: **The name selects the candidates; `content_hash` decides.** An earlier version is
    #: retired only when its SHA-256 matches the new upload's, because that is the one case
    #: with no judgement in it — the same bytes under the same name are redundant, and the
    #: second copy is pure cost: a passage citable twice and a retrieval slot spent.
    #:
    #: A version that shares the name but holds *different* content is left searchable and
    #: reported instead. Two unrelated documents can both be called `invoice.pdf`, and nothing
    #: in a filename or in the bytes tells "a corrected revision of this document" from "a
    #: different document that reused the name". Acting on that guess removes a live document
    #: from search silently, which is worse than a duplicate: the duplicate shows up in the
    #: answer, the absence does not. Retiring one of those stays an administrator's decision,
    #: and `soft_delete` — the Delete button — already is that decision, reversibly.
    #:
    #: Nothing is erased either way. The row keeps its audit history and its S3 object,
    #: `Versions` still lists it, and `Restore` re-ingests it.
    #:
    #: Set it `false` only if querying several identical copies side by side is the intent.
    DOCUMENT_SUPERSEDE_PREVIOUS_VERSIONS: bool = True

    # -- OCR (AWS Textract) --------------------------------------------------
    #: On by default, because the fallback when it cannot run *is* the old behaviour: a page
    #: with no text layer leaves the document `NEEDS_OCR`, exactly as before. Switching this
    #: off is therefore a cost or policy decision, not a safety one.
    OCR_ENABLED: bool = True
    #: `None` means "use AWS_REGION". Textract is regional and not available everywhere, so
    #: this exists for the case where documents live in a region Textract does not serve.
    #:
    #: Optional rather than an empty string so it joins `_blank_is_unset`: a bare `OCR_REGION=`
    #: in `.env`, or a line whose inline comment leaked into the value, reads as "not set"
    #: instead of being handed to boto3 as a region name.
    OCR_REGION: str | None = None
    #: Run OCR during the upload request, or leave it to `scripts/reocr_documents.py`.
    #:
    #: **This is a blast-radius setting, not a feature flag.** `AuditService.log` takes
    #: `pg_advisory_xact_lock('audit_chain:<tenant>')`, which is *transaction-scoped*, and with
    #: `CELERY_TASK_ALWAYS_EAGER=true` ingestion runs inside the upload request's transaction.
    #: So OCR's wait holds that lock, and every other audited operation for the tenant — every
    #: login, query, preview, and every denial — queues behind it. `OCR_TIMEOUT_SECONDS` caps
    #: how long that can last.
    #:
    #: `true` is right for a single-user POC, where the stall is invisible and OCR on upload is
    #: the point. Set it `false` (and backfill out-of-band) or run a real Celery worker
    #: (`CELERY_TASK_ALWAYS_EAGER=false`) before this serves more than one person at a time.
    OCR_ON_UPLOAD: bool = True
    #: How long to wait for one document's asynchronous job. Multi-page PDFs are only
    #: supported through `StartDocumentTextDetection`, which is a job to be polled.
    #:
    #: 25 seconds, not two minutes, because of `OCR_ON_UPLOAD` above: this is how long one
    #: upload can hold a tenant's audit-chain lock. A one- to three-page scan finishes well
    #: inside it; a longer one times out, lands `NEEDS_OCR`, and is picked up by
    #: `scripts/reocr_documents.py`, which runs outside any request and can afford to wait —
    #: raise the value in that script's environment rather than here.
    #:
    #: Timing out leaves the document `NEEDS_OCR`, never `FAILED`: nothing was lost, and
    #: re-running the pipeline is all that is needed.
    OCR_TIMEOUT_SECONDS: int = 25
    #: Gap between polls. Textract charges per page, not per poll, so this trades a little
    #: latency for far fewer API calls than a tight loop.
    OCR_POLL_SECONDS: float = 2.0
    #: Refuse to OCR a document longer than this. A 3000-page scan would hold an upload
    #: request open for the full timeout and bill for every page, and the parser caps text
    #: extraction anyway.
    OCR_MAX_PAGES: int = 200

    # -- RAG -----------------------------------------------------------------
    #: Passages offered to the model per question.
    #:
    #: **10, not 5, because this is a budget for the whole corpus rather than per document.**
    #: The five slots were competed for by every document a tenant has, so a question whose
    #: answer is one multi-step procedure got two or three chunks of the right SOP and the rest
    #: from elsewhere — and the model, correctly refusing to state what it was not given, wrote
    #: a summary. The reported case was a five-bullet bankruptcy procedure answered with four
    #: of its bullets: not a hallucination and not a grounding failure, simply the fifth bullet
    #: never reaching the prompt. A `CHUNK_SIZE_TOKENS` of 512 means a five-page SOP is five to
    #: ten chunks, so covering one needs roughly this many slots before anything is left for
    #: the other documents.
    #:
    #: What this does **not** loosen: `RAG_MIN_SIMILARITY` still filters every candidate, rule
    #: 2 of `SYSTEM_RULES` discards any claim that cites no passage, and
    #: `RAG_REQUIRE_SOURCE_CITATION` drops an answer that cites nothing at all. A passage
    #: retrieved and irrelevant is simply never cited, so more candidates buys completeness
    #: without buying licence — which is the whole reason the citation gate exists.
    #:
    #: What it costs, and the headroom it was chosen against: ~10 x 512 = 5k tokens of context
    #: per question, against a 2048-token answer cap (`BEDROCK_LLM_MAX_TOKENS`) and a 20s
    #: wall-clock budget (`QUERY_TIMEOUT_SECONDS`) that a measured p95 of 8.8s sits well
    #: inside. Input tokens roughly double, so this is a real per-query cost increase. Pinecone
    #: is asked for `RAG_TOP_K * PINECONE_OVERFETCH_FACTOR` = 30 candidates before the
    #: database re-check trims them.
    #:
    #: Lower it if latency matters more than completeness for a given deployment; raise it
    #: further only with the p95 in `/analytics` in front of you.
    RAG_TOP_K: int = 10
    #: Cosine floor a candidate must clear to be offered to the model.
    #:
    #: **0.20, measured — not 0.35, assumed.** The original value was a guess, and against
    #: real Titan Text Embeddings V2 with a real Pinecone index it refused answers the corpus
    #: plainly contained. Measured on 2026-10-27 against a 5-document tenant:
    #:
    #:   off-topic questions ("recipe for sourdough bread", "mortgage rates in Japan 1997",
    #:     "airspeed velocity of an unladen swallow")            0.092 - 0.122
    #:   topical queries the corpus answers ("water systems",
    #:     "lease equipment", "attention is all you need")       0.282 - 0.746
    #:
    #: So everything between 0.122 and 0.35 was being discarded, and that is exactly where a
    #: short topical phrase lands: "water systems" scored 0.282 against a document whose text
    #: is a Macke Water Systems lease, and the user was told the information was not in the
    #: corpus. 0.20 sits in the measured gap with ~1.6x margin over the worst off-topic score.
    #:
    #: Lowering a grounding threshold deserves the caution `rag-answer-grounding.md` implies,
    #: so note what still protects an answer: the model is given the retrieved passages and
    #: instructed to answer only from them and to refuse when they are insufficient, and
    #: `RAG_REQUIRE_SOURCE_CITATION` drops an answer that cites nothing. The floor is the
    #: first of three gates, not the only one — and set this high it was rejecting evidence
    #: before the other two ever saw it.
    #:
    #: Re-measure per corpus if answers start looking thin: this gap is a property of the
    #: embedding model *and* the documents, not a universal constant.
    RAG_MIN_SIMILARITY: float = 0.20
    RAG_REQUIRE_SOURCE_CITATION: bool = True
    #: Wall-clock budget for generation, enforced in `rag_service._generate`.
    #:
    #: Was 5, matching the 3-5s NFR, and that figure only ever fit a one-line answer. The
    #: prompt now asks for a structured 150-400 word answer, which is several times the tokens
    #: and several times the time; at 5s the long answers would not be terser, they would be
    #: `FAILED`. Read the NFR as covering retrieval plus first token — a fuller answer was the
    #: explicit requirement and it is not free. Stays under `BEDROCK_TIMEOUT_SECONDS` (30) so
    #: this timeout, not botocore's, is the one that fires.
    QUERY_TIMEOUT_SECONDS: int = 20

    # -- Retention -----------------------------------------------------------
    AUDIT_LOG_RETENTION_DAYS: int = 2555
    SEARCH_HISTORY_RETENTION_DAYS: int = 365
    DELETED_DOCUMENT_PURGE_AFTER_DAYS: int = 30

    # -- CORS ----------------------------------------------------------------
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    CORS_ALLOW_ORIGINS: str = "http://localhost:3000"

    # -- Validators ----------------------------------------------------------
    @field_validator(
        "S3_ENDPOINT_URL",
        "COOKIE_DOMAIN",
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "PINECONE_API_KEY",
        "OCR_REGION",
        mode="before",
    )
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat `KEY=` — or `KEY=    # a comment` — in `.env` as unset.

        dotenv cannot express "absent" — a key listed with no value parses as `""`, which
        for an optional setting is not what the file means. It matters because the empty
        string is a *value* downstream: `boto3.client("s3", endpoint_url="")` raises
        `ValueError: Invalid endpoint:`, so a blank line in `.env` turned every document
        upload into a 500. `None` is the only reading of a blank optional that behaves like
        omitting it.

        The credential fields are listed for a second reason: `SecretStr("")` is a truthy
        object, so `if not settings.PINECONE_API_KEY` does not catch a blank key and the
        failure surfaces later as an opaque provider auth error instead of "not configured".

        ## The leading `#` case, which cost a working OCR deployment

        dotenv strips an inline comment **only when there is a value in front of it**:

            KEY=us-east-1   # comment   ->  'us-east-1'
            KEY=            # comment   ->  '# comment'      <- the comment IS the value
            KEY=                        ->  ''

        So a template line that documents an empty setting with a trailing comment hands the
        comment text to the application. That is what broke Textract in this project: the
        `.env` carried `OCR_REGION=    # blank = AWS_REGION…`, boto3 was handed that sentence
        as a region name, and every scanned upload failed with `InvalidRegionError` and landed
        `NEEDS_OCR` — indistinguishable, from the UI, from a scan nothing could be read from.

        The templates no longer write comments that way, which is the real fix. This is the
        belt: none of these settings can legitimately begin with `#`, so a value that does is
        a comment that leaked through, and reading it as "unset" restores the behaviour the
        file was describing.
        """
        if isinstance(value, str) and (
            value.strip() == "" or value.lstrip().startswith("#")
        ):
            return None
        return value

    @field_validator("DATABASE_URL")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        if not value.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver: postgresql+asyncpg://..."
            )
        return value

    @field_validator("SECRET_KEY")
    @classmethod
    def _reject_weak_secret(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if len(raw) < 32:
            raise ValueError("SECRET_KEY must be at least 32 characters")
        if raw.upper() in {"CHANGEME", "SECRET", "CHANGE_ME"}:
            raise ValueError("SECRET_KEY is a placeholder — generate a real one")
        return value

    @field_validator("RAG_MIN_SIMILARITY")
    @classmethod
    def _similarity_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("RAG_MIN_SIMILARITY must be between 0.0 and 1.0")
        return value

    @field_validator("PINECONE_INDEX_NAME")
    @classmethod
    def _valid_index_name(cls, value: str) -> str:
        """Pinecone's own naming rule, checked here rather than by a 400 from the API.

        Lowercase alphanumerics and hyphens, up to 45 characters. A typo caught at startup
        costs a restart; the same typo caught at query time is a failed customer query, and
        `create_index` is idempotent enough that a bad name silently creates a *second*
        empty index rather than erroring.
        """
        if not value or len(value) > 45:
            raise ValueError("PINECONE_INDEX_NAME must be 1-45 characters")
        if not all(char.isascii() and (char.isalnum() or char == "-") for char in value):
            raise ValueError(
                "PINECONE_INDEX_NAME may contain only letters, digits and hyphens"
            )
        if value != value.lower():
            raise ValueError("PINECONE_INDEX_NAME must be lowercase")
        return value

    # -- Derived values ------------------------------------------------------
    # `repr=False` for the same reason as `DATABASE_URL` itself, and it has to be said twice:
    # a computed field is rendered in `repr(settings)` and included in `model_dump()`
    # independently of the field it derives from, so hiding the source without hiding the
    # derivative would have left the same password one attribute to the left.
    @computed_field(repr=False)  # type: ignore[prop-decorator]
    @property
    def sync_database_url(self) -> str:
        """Alembic runs sync; it needs the psycopg-free URL without the async driver."""
        return self.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins(self) -> list[str]:
        origins = {
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        }
        origins.add(self.FRONTEND_ORIGIN)
        return sorted(origins)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def allowed_extensions(self) -> frozenset[str]:
        """Allowlist, lowercase, no leading dot."""
        return frozenset(
            ext.strip().lower().lstrip(".")
            for ext in self.ALLOWED_UPLOAD_EXTENSIONS.split(",")
            if ext.strip()
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cookies_secure(self) -> bool:
        """Secure cookies everywhere except plain-HTTP local development."""
        return self.ENVIRONMENT != "local"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def api_docs_enabled(self) -> bool:
        """Whether `/docs` and `/openapi.json` are mounted.

        `ENABLE_API_DOCS` wins when set; otherwise this follows `ENVIRONMENT`, so the default
        behaviour is unchanged. Deliberately *not* derived from `cookies_secure`: the two were
        the same switch once, and the docs were the reason people flipped it — taking the
        cookies' `Secure` flag with them.
        """
        if self.ENABLE_API_DOCS is not None:
            return self.ENABLE_API_DOCS
        return self.ENVIRONMENT == "local"

    def pinecone_namespace(self, tenant_id: object) -> str:
        """The namespace a tenant's vectors live in.

        The only place this string is built. Every read and write goes through it, so tenant
        isolation in the vector store is one function rather than an argument every caller
        has to remember to pass — which is the same reason `build_key` exists in
        `storage_service`.
        """
        return f"{self.PINECONE_NAMESPACE_PREFIX}{tenant_id}"

    def bedrock_credentials(self) -> dict[str, str]:
        """Explicit credentials for boto3, or `{}` to fall back to the profile chain.

        Returned values are unwrapped secrets — pass them straight to boto3 and never
        log the result.
        """
        if not (self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY):
            return {}
        creds = {
            "aws_access_key_id": self.AWS_ACCESS_KEY_ID.get_secret_value(),
            "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY.get_secret_value(),
        }
        if self.AWS_SESSION_TOKEN:
            creds["aws_session_token"] = self.AWS_SESSION_TOKEN.get_secret_value()
        return creds


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton. Call this; never instantiate `Settings()` directly."""
    return Settings()  # type: ignore[call-arg]
