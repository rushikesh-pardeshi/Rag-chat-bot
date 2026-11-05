# Work Breakdown Plan — docquery-ai-document-search

Generated for **Claude Code**. See **README.md** for the project file layout and the
resolved stack table.

## How to use this plan with IDE agents

1. Open this workspace and skim **README.md** for agents/skills and file layout.
2. Pick **one phase** below (start at Phase 1).
3. Ask your agent: `Implement Phase N from WORKBREAKDOWN.md`.
4. Build bottom-up inside the phase: models → repositories → services → schemas →
   routes → wiring in `backend/api/app.py`. A route is the last thing you write, never
   the first.
5. Run every **Test & acceptance** bullet for that phase.
6. Mark tasks `- [x]` before moving on.

Every path in the task lists below is a real path in this repo, or is marked `(new)`.
If a task names a file that does not exist and is not marked `(new)`, the plan is wrong —
say so rather than creating a parallel tree.

## Product scope (from user documents)

DocQuery is an AI-driven document search system for banking/compliance environments. Administrators upload business-critical documents (tax reports, customer records, regulatory guidelines, compliance manuals, financial policies) into a centralized repository. Authorized users submit natural language queries and receive AI-generated answers with exact source traceability—document name, section/page, confidence level. The system maintains complete audit trails, search history, and analytics dashboards to support compliance reviews and operational insights.

### Document requirements covered

- Document Management: Administrators upload, preview, delete, restore documents with metadata (name, upload date, uploader, status) and full data lineage tracking (BR-DM, BR-01)
- AI Query Engine: Natural language queries processed in 3-5 seconds, answers generated strictly from uploaded documents, domain-specific terminology (tax, customer accounts, compliance) interpreted accurately (BR-AIQ, BR-02, NFR-9.1)
- Source Traceability: Every AI answer displays source document name, relevant section/page, and optional confidence level; prevents unsupported answer generation (BR-03, BR-AIQ)
- Search History: Store all queries with user identity, question, answer, timestamp, associated source documents; standard users view own history, administrators view system-wide (BR-SH, BR-04)
- Audit Logging: Auto-log document uploads/deletions/restorations, user logins/access attempts, query submissions, data exports; tamper-resistant logs with user identity, action, date/time, affected object (BR-AL, BR-05, R-03)
- Analytics & Reporting: Generate reports on most/least searched keywords, frequently referenced documents, usage trends; export to PDF/Excel; administrator-only access (BR-AN, AC-11.1)
- Role-Based Access Control: Exactly two roles (Administrator, Standard User); administrators manage documents, analytics, audit logs, user access; standard users submit queries and view own history. Compliance officers hold the Administrator role — there is no third role (BR-UAM, BR-06, NFR-9.2)
- Performance: 99% uptime, 3-5 second query response under normal load, support concurrent users without degradation, handle document volume growth (NFR-9.1, NFR-9.3, NFR-9.4)
- Security: Banking-grade security for customer financial data and regulatory documents; secure authentication, RBAC enforcement, encrypted storage and transmission, tamper-resistant audit logs (NFR-9.2, C-01, R-07, R-10)
- Compliance: 100% audit log availability during compliance reviews, full traceability from answer to source, data retention per organizational policies, regulatory and governance requirements met (AC-11.4, C-05)
- Usability: Clean, intuitive UI responsive across desktop and mobile; minimal training required; clear navigation and feedback for major actions (NFR-9.5, R-08)
- Data Integrity: Answers derived only from uploaded documents, no external content; document validation checks and admin review before approval; version tracking and document status management (C-04, R-01, R-09)
- Scalability: Expandable storage capacity, accommodate increasing concurrent users and query frequency over time (NFR-9.4, C-03)
- Query Accuracy: >90% accuracy rate validated through business/compliance reviews; human validation for critical queries, continuous AI performance monitoring (Problem Statement, R-04)
- Scanned/image-only documents (D-01) are accepted, marked `status='needs_ocr'`, and excluded from retrieval. OCR itself is **out of scope** for these six phases — it needs a new dependency (e.g. AWS Textract, a third AWS service) and must be agreed before anyone adds it.

## Tech stack

Resolved and pinned. `README.md` holds the same table; this is the short form:

| Layer | Choice |
|---|---|
| API | FastAPI (async) + Pydantic v2 + `pydantic-settings` |
| ORM | **SQLAlchemy 2.0 async** + `asyncpg`. Not SQLModel |
| Migrations | Alembic (async `env.py`) |
| Database | **Stock PostgreSQL 17** — no extension to compile. It holds metadata, chunk text and citations; it holds no vectors |
| Vector store | **Pinecone serverless** (`pinecone==9.1.0`, async client) — not pgvector/FAISS/Chroma. One namespace per tenant: `tenant-{tenant_id}` |
| Auth | **PyJWT** + `bcrypt` (cost 12). JWT in an httpOnly cookie + anti-CSRF token |
| Identity | **email**, unique on `(email, tenant_id)` |
| Rate limiting | `slowapi`, 10/min on `/auth/*` |
| Embeddings | Bedrock `amazon.titan-embed-text-v2:0`, 1024 dims |
| Generation | Bedrock `us.amazon.nova-pro-v1:0` via boto3 Converse, temperature 0.0 |
| Object storage | S3, AES-256 at rest, app-side lineage keys |
| Async jobs | **Celery + Redis**. Locally `CELERY_TASK_ALWAYS_EAGER=true` — no Redis needed to develop |
| Export | `reportlab` (PDF) + `openpyxl` (Excel), generated **server-side** |
| Frontend | Next.js **App Router** + TypeScript + Tailwind. There is no `frontend/pages/` |
| Tests | pytest + pytest-asyncio (backend), Jest + React Testing Library (frontend) |
| Python env | `uv venv` → `.venv/`, deps pinned in `requirements.txt` |

Config keys live in `.env.example` and are read **only** through `backend/core/config.py`.
The real names are `SECRET_KEY`, `DATABASE_URL`, `BEDROCK_LLM_MODEL_ID`,
`BEDROCK_EMBEDDING_MODEL_ID`, `S3_BUCKET_NAME`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`,
`CORS_ORIGINS`, `LOG_LEVEL`. There is no `JWT_SECRET` and no `AI_MODEL_ENDPOINT`.

## Assumptions & constraints

- One banking organization per deployment. `tenant_id` is nonetheless carried on every
  table and enforced in every query — it is the isolation boundary that makes an IDOR a
  404 instead of a leak, and it keeps a second organization from being a migration.
  **Role and tenant are independent checks**: an Administrator of tenant A has no access
  to tenant B.
- Banking-grade security: TLS in transit, AES-256 at rest, append-only audit logs with a
  per-tenant hash chain, VAPT review before release.
- AI responses limited to the uploaded corpus. No external sources, no fallback to model
  knowledge. When retrieval returns nothing above threshold the answer is
  "Information not found in uploaded documents".
- Retention: audit logs per `AUDIT_LOG_RETENTION_DAYS` (2555 = 7 years), archived to cold
  storage before any purge — never an in-place delete. Documents are soft-deleted with a
  restore window.
- Regulatory context: GDPR/CCPA for customer data, SOC 2-style access and audit controls,
  financial-sector rules (GLBA and, if payment data ever lands here, PCI-DSS).
- **Local development is the current target, not deployment.** Two AWS services are in
  scope: **Bedrock** and **S3**. RDS, CloudWatch, VPC and load balancers are deployment
  concerns and are not built or tested in these phases. S3 bucket versioning is
  unavailable to the current IAM role, so document lineage is app-side: every version
  gets its own immutable key `documents/{document_id}/v{n}/{filename}` and nothing is
  ever overwritten.
- **Pinecone is a third external service and it is not AWS.** Embeddings leave the AWS
  account, so it needs its own note in the data-flow section of any compliance review: what
  crosses the boundary is a 1024-float vector plus citation metadata (document id, document
  name, page, section) and **never document text**. Tenants are separated by namespace, and
  the API key is a credential with the same handling rules as the AWS keys — `.env` only,
  never a log, never `.env.example`. The dependency is deliberate: it keeps PostgreSQL free
  of compiled extensions and the local stack installable without Docker.

## Agents in this workspace

Invoke these from your IDE when implementing matching phases. Each one **writes code** in
its layer — none of them is the runtime feature.

- **`document-ingestion-agent`** — Uploads, validates, and manages business documents with metadata tracking and lineage
  - Skills: document-processing, metadata-extraction | Tools: Read, Write, Grep, Glob, Bash
- **`ai-query-agent`** — Processes natural language queries, generates answers from documents, provides source references
  - Skills: rag-pipeline, source-tracing, pre-vapt-secure-coding | Tools: Read, Write, Grep, Glob
- **`audit-logging-agent`** — Captures all system activities, maintains tamper-resistant logs for compliance reviews
  - Skills: audit-trail-generation, pre-vapt-secure-coding | Tools: Read, Write, Grep, Glob
- **`analytics-agent`** — Generates usage reports, query trends, document references for administrators
  - Skills: analytics-aggregation | Tools: Read, Write, Grep, Glob
- **`security-vapt-reviewer`** — Reviews code for SQL injection, XSS, log injection, command injection, IDOR, SSRF, CSRF, path traversal, mass assignment, ReDoS, secrets exposure
  - Skills: pre-vapt-secure-coding | Tools: Read, Grep, Glob (read-only by design)
- **`api-integration-agent`** — Builds FastAPI routes, dependencies, schemas for document upload, query, history, audit endpoints
  - Skills: fastapi-development, pre-vapt-secure-coding | Tools: Read, Write, Grep, Glob
- **`frontend-dashboard-agent`** — Implements Next.js UI for document management, query interface, search history, analytics dashboards
  - Skills: nextjs-development | Tools: Read, Write, Grep, Glob, Bash

## Skills in this workspace

- **`document-processing`** — Extracts text, metadata, sections from PDFs/Word docs; stores with lineage tracking
- **`rag-pipeline`** — Embeds documents, retrieves relevant chunks, generates answers with source citations using LLM
- **`source-tracing`** — Maps AI answers to exact document sections, pages, confidence scores for traceability
- **`audit-trail-generation`** — Logs user actions, timestamps, affected objects; ensures tamper-resistant storage
- **`analytics-aggregation`** — Aggregates query keywords, document usage, trends; exports to PDF/Excel formats
- **`pre-vapt-secure-coding`** — Prevents SQL injection, XSS, log injection, command injection, IDOR, SSRF, CSRF, path traversal, mass assignment, ReDoS, secrets in logs
- **`fastapi-development`** — Builds async FastAPI routes with Pydantic validation, dependency injection, error handling
- **`nextjs-development`** — Creates responsive Next.js pages, components, hooks with TypeScript and Tailwind CSS
- **`metadata-extraction`** — Captures document name, upload date, uploader, status, version for lifecycle management

## Rules in this workspace

Exported as plain markdown under `.claude/rules/`. Always-on rules are imported by `CLAUDE.md`, so they are in context every session; scoped rules state their globs in their own header.

- **`secure-coding-vapt`** — Enforce parameterized queries, input validation, output encoding, CSRF tokens, no secrets in logs (always apply)
- **`document-source-traceability`** — Every AI answer must include document name, section/page, confidence level in response schema (always apply)
- **`audit-log-immutability`** — Audit logs append-only; no updates/deletes; include user, action, timestamp, object (always apply)
- **`rbac-enforcement`** — Check user role in dependencies; Admin-only for document mgmt, analytics; users own history only (always apply)
- **`rag-answer-grounding`** — AI answers derived strictly from uploaded documents; no external knowledge; cite sources (always apply)
- **`architecture`** — Layer boundaries for docquery-ai-document-search: routes → services → repositories. (always apply)
- **`workflow`** — How to work through phases and when to delegate to agents and skills. (always apply)

Platform: **Claude Code**. IDE folders (`.cursor/`, `.windsurf/`, or `.claude/`) are listed in **README.md** — not duplicated here.

## Known scaffold artifacts

Present on disk, referenced by no phase. Leave them alone; do not build on them:

- the eleven `docquery_ai_document_search` files under `backend/` and `frontend/` — the
  generator named a domain slice after the project itself
- `backend/api/dependencies/db.py` duplicates `database.py` → use **`database.py`**
- `frontend/lib/api.ts` duplicates `api-client.ts` → use **`api-client.ts`**
- `backend/models/auth.py` and `backend/schemas/auth.py` overlap `user.py` — auth DTOs go
  in `schemas/auth.py`, the ORM entity in `models/user.py`

**`search_history` is not a table.** It is a role-filtered projection over `query`:
Standard Users see rows where `user_id = current_user.id`, Administrators see all rows for
their tenant. `search_history_service.py` exists; there is no `models/search_history.py`
and none should be created.

## Phase overview

- **Phase 1: Foundation, settings & API scaffold**
- **Phase 2: Authentication, RBAC & tenant isolation**
- **Phase 3: Documents, S3 storage & audit trail**
- **Phase 4: AI query engine — RAG, embeddings & async processing**
- **Phase 5: Frontend flows, search history & analytics**
- **Phase 6: Testing, security review & release**

The AI engine (Phase 4) is built **before** the UI that displays its answers (Phase 5).
Building the query page first would mean shipping a page with nothing to render and then
reworking it — the dependency runs one way.

## Phase 1: Foundation, settings & API scaffold

### Objectives

- Boot the FastAPI app through `main.py` → `backend/api/app.py` with health checks, CORS and structured logging
- Establish the async PostgreSQL connection and the Alembic migration chain (stock Postgres 17 — vectors live in Pinecone, so no extension is compiled)
- Configure the Next.js App Router frontend with TypeScript, Tailwind and a typed API client
- Centralize configuration in `backend/core/config.py` so nothing else reads the environment
- Fail fast: a missing required setting must stop startup with a clear message

### Tasks (actionable)

- [x] `backend/core/config.py` — Pydantic `Settings` (pydantic-settings) for `DATABASE_URL`, `SECRET_KEY`, `CORS_ORIGINS`, `LOG_LEVEL`, `AWS_REGION`, `S3_BUCKET_NAME`, `BEDROCK_LLM_MODEL_ID`, `BEDROCK_EMBEDDING_MODEL_ID`, `BEDROCK_EMBEDDING_DIMENSIONS`, the `PINECONE_*` block (`API_KEY`, `INDEX_NAME`, `CLOUD`, `REGION`, `METRIC`, `NAMESPACE_PREFIX`, `UPSERT_BATCH_SIZE`, `TIMEOUT_SECONDS`, `OVERFETCH_FACTOR`) plus `pinecone_namespace(tenant_id)`, `RAG_TOP_K`, `RAG_MIN_SIMILARITY`, `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`, `MAX_UPLOAD_SIZE_MB`, `ALLOWED_UPLOAD_EXTENSIONS`, `AUDIT_LOG_RETENTION_DAYS`; cached `get_settings()`
- [x] `backend/core/database.py` — async engine + `async_sessionmaker` with `pool_size=20, max_overflow=10`, and the declarative `Base`
- [x] `backend/api/dependencies/database.py` — `get_db()` dependency yielding one `AsyncSession` per request, committing on success and rolling back on error
- [x] `backend/core/logging.py` **(new)** — `structlog` JSON logging with `request_id`, `user_id`, `action`; **a CR/LF-stripping processor on every user-supplied value** (log injection)
- [x] `backend/core/exceptions.py` **(new)** — `DocumentNotFound`, `UnauthorizedAccess`, `ValidationError`, `IntegrityChainBroken` + a single exception handler that maps them to status codes and never echoes internals
- [x] `backend/api/app.py` — app factory: settings, logging, CORS from `CORS_ORIGINS` with `allow_credentials=True`, exception handlers, request-id middleware, router registration under `/api/v1`
- [x] `backend/api/routes/health.py` — `GET /health` → `{status, version, timestamp}`; `GET /readiness` → checks the DB with `SELECT 1`
- [x] `main.py` — import the factory and expose `app`. No logic beyond that
- [x] `alembic/env.py` — async migration environment bound to `Base.metadata`, URL read from `get_settings()`
- [x] first migration **(new, `alembic/versions/`)** — `CREATE EXTENSION IF NOT EXISTS pgcrypto` and the `alembic_version` baseline. It no longer enables `vector`: Pinecone is the vector store, so nothing in this schema needs a compiled extension. The revision keeps its `0001_pgvector` id because later revisions point at it
- [x] `backend/core/security.py` — signature-only stubs for `hash_password`, `verify_password`, `create_access_token`, `decode_access_token` (filled in Phase 2)
- [x] `frontend/lib/api-client.ts` — `fetch` wrapper on `NEXT_PUBLIC_API_BASE_URL` + `NEXT_PUBLIC_API_V1_PREFIX`, `credentials: 'include'`, error normalization
- [x] `frontend/app/layout.tsx` — root layout: Tailwind, fonts, DocQuery metadata, responsive viewport
- [x] `backend/tests/test_health.py` — health and readiness tests over `httpx.ASGITransport`

### Testing strategy

- Unit test `GET /health` returns 200 and the exact schema `{status, version, timestamp}`
- Integration test the session dependency: 50 concurrent requests complete with no leaked connections
- Test that dropping a required key from the environment raises a validation error at import time
- Test CORS: the configured origin gets `Access-Control-Allow-Credentials`, an unlisted origin does not
- Assert the log processor turns `"a\nb"` into a single line — the log-injection guard has a test, not just a comment

### Test & acceptance (phase gate — must pass before next phase)

- [x] `GET /health` returns 200 `{status: 'healthy', version, timestamp}` in under 100ms
- [x] `GET /readiness` returns 200 only when the database answers; 503 otherwise
- [x] The session dependency serves 50 concurrent async requests with no errors and no connection leak
- [ ] `alembic upgrade head` succeeds on an empty database, creates `alembic_version`, and `pgcrypto` is installed  
      **BLOCKED — no PostgreSQL on this machine** (no Docker, no native install).
      The async `env.py` and the `0001_pgvector` revision are written; this check runs the
      moment any **stock** Postgres 17 is reachable — since the pgvector swap there is no
      extension to build, so a plain installer or `postgres:17` image is enough.
- [x] `npm run dev` serves `localhost:3000` and renders the root layout with Tailwind applied
- [x] Removing `DATABASE_URL` or `SECRET_KEY` from `.env` stops startup with a message naming the missing key
- [x] Logs reach stdout as one JSON object per line with `request_id`, `timestamp`, `level`, `message`; CR/LF in input never produces a second line
- [x] `ruff check .` and `pytest -q` are clean
- [x] `grep -rn "os.environ\|os.getenv" backend/` returns nothing — config goes through `config.py` (`scripts/` is exempt)

### Phase completion checklist

- [x] All Tasks (actionable) complete
- [x] All Test & acceptance checks verified
- [x] No open High/Critical VAPT findings
- [x] Ready to ask agent: `Implement Phase 2 from WORKBREAKDOWN.md`

### Agent prompt (copy-paste)

```text
Implement Phase 1 of WORKBREAKDOWN.md for DocQuery, an AI document search system for banking/compliance.

Boot FastAPI through main.py -> backend/api/app.py: CORS from settings with credentials allowed, request-id middleware, centralized exception handlers, and GET /health plus GET /readiness under backend/api/routes/health.py.

Centralize every setting in backend/core/config.py using pydantic-settings, including the PINECONE_* block and a pinecone_namespace(tenant_id) helper. No other module may read the environment. Wire the async SQLAlchemy 2.0 engine and session factory in backend/core/database.py, expose get_db in backend/api/dependencies/database.py, and set up an async Alembic env.py with a first migration that enables pgcrypto. Do not enable pgvector and do not add an embedding column anywhere: the vector store is Pinecone, and this schema must install on a stock PostgreSQL 17.

Add backend/core/logging.py with structlog JSON output carrying request_id, user_id and action, and a processor that strips CR/LF from user-supplied values before they are logged.

Configure the Next.js App Router frontend: frontend/app/layout.tsx with Tailwind and responsive metadata, frontend/lib/api-client.ts as a fetch wrapper using NEXT_PUBLIC_API_BASE_URL with credentials included.

Do not create backend/main.py, backend/api/v1/, or frontend/pages/ — none of those exist in this repo. Leave backend/api/dependencies/db.py and frontend/lib/api.ts untouched; they are unused duplicates.
```

## Phase 2: Authentication, RBAC & tenant isolation

### Objectives

- Cookie-based JWT authentication with bcrypt (cost 12) hashing and refresh rotation
- Two roles, enforced server-side in dependencies: Administrator and Standard User
- Tenant isolation as a second, independent check on every query
- Rate limiting and brute-force protection on `/auth/*`
- Frontend auth flows with route protection that never treats the client-side check as security

### Tasks (actionable)

- [x] `backend/models/user.py` — `User`: `id`, `tenant_id` (FK), `email`, `hashed_password`, `role` (`Enum('ADMINISTRATOR','STANDARD_USER')`), `is_active`, `created_at`, `updated_at`; `UniqueConstraint(email, tenant_id)`
- [x] `backend/models/tenant.py` **(new)** — `Tenant`: `id`, `name`, `domain`, `is_active`, `created_at`; one-to-many to `User`
- [x] `backend/core/security.py` — `hash_password`/`verify_password` on `bcrypt` directly (cost 12), `create_access_token`/`create_refresh_token`/`decode_token` on **PyJWT**, plus CSRF token mint/compare. Not passlib, not python-jose
- [x] `backend/repositories/user_repository.py` — `get_by_email(email, tenant_id)`, `get_by_id`, `create`, `set_password`, `list_for_tenant`. ORM only, no raw SQL, no business rules
- [x] `backend/repositories/auth_repository.py` — refresh-token store: `store`, `rotate`, `revoke`, `is_revoked`
- [x] `backend/services/auth_service.py` — register, authenticate, refresh, logout, password change. Owns lockout and the audit call. Never builds an HTTP response
- [x] `backend/schemas/auth.py` — `RegisterRequest`, `LoginRequest`, `TokenResponse`, `UserProfile`. `model_config = ConfigDict(extra='forbid')` on every request model (mass assignment); `role` and `tenant_id` are **not** client-settable on register
- [x] `backend/api/dependencies/auth.py` — `get_current_user` (reads the httpOnly cookie, 401 on bad/expired), `require_admin` (403 unless Administrator), `get_tenant_context`; every denial writes an audit row with `outcome='DENIED'`
- [x] `backend/api/routes/auth.py` — `POST /api/v1/auth/register`, `/login`, `/refresh`, `/logout`, `GET /api/v1/auth/me`. Tokens set as `httpOnly`, `Secure`, `SameSite=Strict` cookies; CSRF token returned in a readable cookie
- [x] CSRF middleware in `backend/api/app.py` — double-submit check on every unsafe method
- [x] `slowapi` limiter in `backend/api/app.py` — 10/min per IP on `/api/v1/auth/*`, storage from `RATE_LIMIT_STORAGE_URI`
- [x] `frontend/hooks/useAuth.ts` — session hook reading `GET /auth/me`; exposes role for **display only**
- [x] `frontend/app/login/page.tsx` — email + password form, server-set cookie, role-based redirect
- [x] `backend/tests/test_auth_service.py` **(new)** — hashing, token, lockout, tenant-mismatch and RBAC tests
- [x] seed script **(new, `scripts/seed_admin.py`)** — create the first tenant and Administrator; refuses to run if one already exists

**Added during implementation** (not in the original list, each with the reason):

- [x] `backend/models/audit_log.py` + `audit_repository.py` + `audit_service.py` **pulled forward from Phase 3** — every task above says "writes an audit row", so authentication cannot be finished without the audit chain existing. Phase 3 keeps the audit *read* route and its schemas.
- [x] `backend/tests/fakes.py` **(new)** — in-memory repository fakes that preserve the tenant filter, so the RBAC and isolation tests are meaningful without a database.
- [x] `backend/tests/test_audit_service.py` — rewritten from a stub: hash determinism, chain linkage, tamper detection, append-only assertions.
- [x] `backend/tests/test_auth_routes.py` **(new)** — HTTP-level CSRF, rate-limit, mass-assignment and 401 tests. These run with no database because each control rejects before the handler touches a session.
- [x] `frontend/proxy.ts` **(new)** — the unauthenticated → `/login` redirect. **Next 16 renamed `middleware.ts` to `proxy.ts`**; do not add a `middleware.ts`.
- [x] `frontend/app/login/login-form.tsx` **(new)** — split from `page.tsx`: `useSearchParams()` requires a Suspense boundary, and a page that reads it without one fails `next build` outright.
- [x] `frontend/services/authService.ts`, `frontend/types/user.ts`, `frontend/components/ui/button.tsx` — the login page's dependencies; `types/user.ts` deliberately has no token field, since the tokens are httpOnly cookies.
- [x] `ApiError.retryAfter` in `frontend/lib/api-client.ts` — the login form's back-off countdown needs the server's `Retry-After`, and rate limiting happens in middleware so the header is the only place that value exists.
- [x] `<AuthProvider>` wired into `frontend/app/layout.tsx`.

### Testing strategy

- Unit test hashing: `verify_password` true/false, two hashes of the same password differ
- Unit test tokens: valid token decodes to `{user_id, role, tenant_id}`, expired raises, tampered signature raises
- Integration test register: user persisted with a bcrypt hash, duplicate `(email, tenant_id)` → 409, a client-supplied `role` field is rejected not honoured
- Integration test login: valid credentials set the cookie, invalid → 401 with no user-existence hint, 11th attempt in a minute → 429
- Test tenant isolation directly: a user of tenant A requesting a tenant-B document id gets 404, and the attempt is audited
- Test RBAC: Standard User on `POST /documents`, `GET /analytics`, `GET /audit` → 403; Administrator → allowed
- Test CSRF: an unsafe request with the cookie but no matching CSRF header → 403

### Test & acceptance (phase gate — must pass before next phase)

Four of these nine are verified end to end without a database, because the control being
checked rejects the request before any handler touches a session. The other five need a live
Postgres and are marked **BLOCKED**, each with the test that already covers its logic at the
service layer — so what is outstanding is the wiring, not the rule.

- [ ] **BLOCKED — no PostgreSQL on this machine.** `POST /api/v1/auth/register` stores a bcrypt hash (cost 12), assigns the role server-side, returns no password material. Verified at the service layer in `test_auth_service.py` (cost-12 hash, server-assigned role, `UserProfile` carries no password field), and the *server-side* half is verified over HTTP: `test_auth_routes.py::TestMassAssignment` proves a client-supplied `role` is rejected with a 422. Only "the row lands in Postgres" is unverified.
- [ ] **BLOCKED — no PostgreSQL on this machine.** `POST /api/v1/auth/login` sets an httpOnly `Secure` `SameSite=Strict` access cookie with `{user_id, role, tenant_id}` claims, an `ACCESS_TOKEN_EXPIRE_MINUTES` expiry (5 hours), plus a rotating refresh cookie. Login cannot reach the cookie-setting code without a user row to authenticate against. The claims and expiry are verified in `test_security.py`; the cookie flags are not.
- [x] `/auth/*` rate limiting returns 429 with `Retry-After` after 10 attempts per IP per minute — `test_auth_routes.py::TestRateLimiting`, 5 tests. Also asserts the header is present and positive, that the envelope matches every other error, and that a *failing* attempt still counts (otherwise an attacker brute-forces for free against whatever is currently erroring).
- [x] `get_current_user` → 401 on missing/invalid/expired token — `test_auth_routes.py::TestUnauthenticatedAccess`, 7 tests: no cookie, garbage, expired, a refresh token used as an access token, a tampered signature, refresh with no cookie, and two different failures returning byte-identical bodies. **`require_admin` → 403 for Standard Users is BLOCKED over HTTP** (needs a real user row) but is verified against the fakes in `test_auth_service.py`.
- [ ] **BLOCKED — no PostgreSQL on this machine.** Tenant isolation holds on every read: cross-tenant ids return 404, never another tenant's data. Verified against `backend/tests/fakes.py`, whose repositories keep the `tenant_id` filter precisely so this test means something without a DB. Phase 3 re-checks it over HTTP on documents.
- [x] Every state-changing route rejects a request without a valid CSRF token (403) — `test_auth_routes.py::TestCSRF`, 9 tests: no token, cookie-only, header-only, mismatched pair, a matching pair getting through, `/login|/register|/refresh` exempt (no session for a forgery to ride on), and safe methods unaffected.
- [ ] **BLOCKED — no PostgreSQL on this machine.** Every allowed **and** denied auth or access decision produces an audit row with `outcome` set. The append-only chain, hash linkage and tamper detection are verified in `test_audit_service.py`; that `auth_service` calls it on both paths is verified through the fakes. What is unverified is that the INSERT commits.
- [ ] **BLOCKED — needs the API and UI running together.** Frontend: login redirects Administrator and Standard User to their landing pages; an unauthenticated visit to a protected route redirects to `/login`; the server still enforces the rule when the client check is bypassed. What *is* verified: `npm run build` passes and lists `ƒ Proxy (Middleware)`, so `proxy.ts` is picked up; `landingPathFor(role)` is pure and covered; and the server-side half is exactly the `get_current_user`/`require_admin` row above — the proxy is cosmetic by design and holds no secret.
- [ ] **BLOCKED — no PostgreSQL on this machine.** Refresh happens transparently on 401 and the old refresh token is revoked (rotation, not reuse). Rotation and replay-rejection are verified in `test_auth_service.py` against the fake token store; the transparent-retry path in `api-client.ts` has no test yet — see the frontend test-runner decision below.

### Phase completion checklist

- [x] All Tasks (actionable) complete
- [ ] All Test & acceptance checks verified — **4 of 9 verified, 5 BLOCKED on Postgres/a running stack.** No check is failing; each blocked one runs the moment a stock Postgres 17 instance is reachable (no extension build needed since the Pinecone swap).
- [x] No open High/Critical VAPT findings
- [x] Ready to ask agent: `Implement Phase 3 from WORKBREAKDOWN.md`

### Verification run (2026-08-13)

```text
pytest -q      → 142 passed          (113 at the start of Phase 2)
ruff check .   → All checks passed!
npm run typecheck → clean
npm run build     → passes; ƒ Proxy (Middleware), ○ /login prerendered
```

`npm run lint` and `npm test` are **broken scaffold scripts**, not failing checks: `next lint`
was removed in Next 16, and `package.json` declares neither `eslint` nor `jest`. Fixing either
means adding dependencies, which `.claude/rules/workflow.md` says to ask about first.

### Agent prompt (copy-paste)

```text
Implement Phase 2 of WORKBREAKDOWN.md: authentication, RBAC and tenant isolation for DocQuery.

Build User (email, hashed_password, role, tenant_id, unique on (email, tenant_id)) and Tenant models. Use bcrypt directly at cost 12 and PyJWT for tokens — not passlib, not python-jose.

Put tokens in httpOnly, Secure, SameSite=Strict cookies and add a double-submit CSRF check on every unsafe method; the two decisions are coupled, so implement them together. Rate-limit /api/v1/auth/* at 10/min per IP with slowapi.

Layer it properly: user_repository and auth_repository issue the queries, auth_service holds the rules (lockout, rotation, audit calls), routes in backend/api/routes/auth.py validate a Pydantic schema and call one service method. Request schemas use extra='forbid'; role and tenant_id are never client-settable.

Dependencies live in backend/api/dependencies/auth.py: get_current_user, require_admin, get_tenant_context. There are exactly two roles — compliance officers are Administrators. Every denial writes an audit row with outcome='DENIED'. Role and tenant are independent checks: an Administrator of tenant A gets nothing from tenant B.

Frontend: frontend/app/login/page.tsx and frontend/hooks/useAuth.ts. Client-side role checks are cosmetic — the server decides.
```

## Phase 3: Documents, S3 storage & audit trail

### Objectives

- Document lifecycle: upload, list, preview, soft-delete, restore — Administrator only
- S3 from the start, with app-side immutable version keys for lineage (BR-01)
- Text and metadata extraction into `document_chunk` rows, ready for Phase 4 to embed
- The append-only audit log with a per-tenant hash chain (BR-AL, BR-05)
- The `query` table and a query submission endpoint that records the question

### Tasks (actionable)

- [x] `backend/models/document.py` — `Document`: `id`, `tenant_id`, `name`, `s3_key`, `content_type`, `file_size`, `page_count`, `uploaded_by` (FK), `upload_date`, `status` (`ACTIVE|DELETED|NEEDS_OCR|PROCESSING|FAILED`), `version`, `metadata` (JSONB), `created_at`, `updated_at`
- [x] `backend/models/document.py` — `DocumentChunk`: `id`, `tenant_id`, `document_id` (FK), `chunk_index`, `text`, `page_number`, `section_label`, `embedded_at` (nullable — set in Phase 4 when Pinecone acknowledges the upsert). **No embedding column**: the vector lives in Pinecone under the id `str(chunk.id)`. Page/section are `NOT NULL` — a chunk that cannot be cited must not exist
- [x] `backend/models/query.py` — `Query`: `id`, `tenant_id`, `user_id` (FK), `question_text`, `answer_text` (nullable), `source_references` (JSONB), `confidence_score` (nullable), `status` (`PENDING|ANSWERED|NOT_FOUND|FAILED`), `latency_ms`, `created_at`; index `(tenant_id, user_id, created_at DESC)`
- [x] `backend/models/audit_log.py` — `AuditLog` exactly as `.claude/skills/audit-trail-generation/references/audit-schema.md` specifies, including `previous_hash` and `integrity_hash`
- [x] Alembic migration **(new)** — all four tables, the indexes from the schema reference, and a `REVOKE UPDATE, DELETE ON audit_log` for the application role
- [x] `backend/repositories/document_repository.py` — `create`, `get(id, tenant_id)`, `list(tenant_id, status, limit, offset)`, `set_status`, `next_version(name, tenant_id)`, `add_chunks`. Every method takes `tenant_id`
- [x] `backend/repositories/audit_repository.py` — `append(entry)` and read-only `list(filters)` / `last_hash(tenant_id)`. **No update or delete method exists in this file**
- [x] `backend/repositories/query_repository.py` — `create`, `get(id, tenant_id)`, `list_for_user`, `list_for_tenant`, `attach_answer`
- [x] `backend/services/audit_service.py` — `log(...)`: validate required fields, strip CR/LF, mask secrets/PII, compute `integrity_hash` over the canonical fields plus the previous hash, append. Plus `verify_chain(tenant_id)`
- [x] `backend/utils/document_parser.py` — PDF via `pypdf`, DOCX via `python-docx`, per-page/per-section text with page numbers retained. **XXE off** — DOCX is a zip of XML, so parse defensively. If no extractable text: `status='NEEDS_OCR'`
- [x] `backend/services/storage_service.py` **(new)** — S3 behind a narrow interface: `put(document_id, version, filename, stream)`, `get_presigned_url(s3_key, ttl)`, `delete`. Keys are built as `documents/{document_id}/v{n}/{safe_filename}` — **never** from a user-supplied path, and the filename is sanitized before use (path traversal)
- [x] `backend/services/document_service.py` — orchestrates: validate content type against `ALLOWED_UPLOAD_EXTENSIONS` and magic bytes, enforce `MAX_UPLOAD_SIZE_MB`, store to S3, parse, persist document + chunks, call `audit_service.log`. Soft-delete and restore flip `status` and audit; nothing is erased
- [x] `backend/services/query_service.py` — `submit(question, user)`: validate length, persist a `PENDING` query, audit `QUERY_SUBMIT`. Answer generation arrives in Phase 4
- [x] `backend/schemas/document_schemas.py` — `DocumentResponse`, `DocumentListResponse` (paginated), `DocumentPreviewResponse` (presigned URL + TTL). No ORM entity ever leaves a route
- [x] `backend/schemas/query_schemas.py` — `QueryCreate` (`extra='forbid'`, length-bounded `question_text`), `QueryResponse`, `SourceReference` (`source_document_name` and `section_or_page` **required**, `confidence_level` optional)
- [x] `backend/schemas/audit_schemas.py` — `AuditLogResponse`, `AuditLogFilter` (action, user, date range, pagination)
- [x] `backend/api/routes/documents.py` — `POST /api/v1/documents` (multipart, admin), `GET /api/v1/documents`, `GET /api/v1/documents/{id}`, `DELETE /api/v1/documents/{id}` (admin), `POST /api/v1/documents/{id}/restore` (admin), `GET /api/v1/documents/{id}/preview` (presigned, short TTL, tenant-checked)
- [x] `backend/api/routes/queries.py` — `POST /api/v1/queries` → `query_service.submit(...)`; `GET /api/v1/queries/{id}`
- [x] `backend/api/routes/audit_logs.py` — `GET /api/v1/audit` (admin only), filterable, read-only. No write, update or delete route exists
- [x] `backend/tests/test_document_service.py`, `backend/tests/test_audit_service.py` — fake the S3 client; no test touches AWS

**Added during implementation** (not in the original list, each with the reason):

- [x] `backend/api/dependencies/services.py` **(new)** — one place that builds `DocumentService`, `QueryService` and `AuditService` from a session. Without it every route constructs its own service graph, and a route that assembles its own dependencies is a route that can quietly assemble a different one.
- [x] `AuditService.list_entries` + `AuditQuery`/`AuditPage` — the audit *read* the gate requires. It takes the actor's role, tenant and email as primitives rather than a `User`, because `audit_service` importing from `auth_service` would close an import cycle (`auth_service` logs).
- [x] `GET /api/v1/audit/verify` — `verify_chain` had no way to be called. A tamper-detection mechanism a compliance officer cannot run is a mechanism that will not be run.
- [x] `GET /api/v1/documents/{id}/versions` — the lineage the app-side version keys exist to provide. Without it, "v1 is still readable" is true but unobservable.
- [x] `get_db` now **commits on a `DocQueryError`** and rolls back only on unexpected exceptions. A denied request writes an audit row and then raises `UnauthorizedAccess`; the original rollback-on-any-exception discarded exactly the `outcome='DENIED'` rows the gate asks for.
- [x] `backend/tests/document_fixtures.py` **(new)** — generated PDF/DOCX/hostile-archive builders. A committed `sample.pdf` is a blob nobody can review, and adding a PDF *writer* dependency to run tests would need approval; `make_pdf` writes the structure by hand, xref table included, so `pypdf` reads it through the real path.
- [x] `backend/tests/test_document_parser.py` **(new)** — the parser's own tests: magic-byte detection, XXE against a real file on disk, a remote-entity SSRF attempt, an implausible compression ratio, and page numbers that survive a skipped blank page.
- [x] `backend/tests/test_api_surface.py` **(new)** — reads `app.openapi()` to prove the audit log exposes no state-changing method and that there is exactly one document delete. Repository-level immutability is necessary; "no route can reach a write" is the part a reviewer can check.
- [x] `sanitize_filename` fix in `backend/services/storage_service.py` — a trailing space became `_`, turning `report.pdf ` into extension `pdf_` and getting a legitimate file rejected for a character the user cannot see. Found by `TestFilenameSafety`.
- [x] `backend/tests/conftest.py` — its duplicate `FakeS3Client` now re-exports the one in `fakes.py`. The old copy accepted any `put_object`, so the no-overwrite guard was exercised by some tests and silently ignored by others.

### Testing strategy

- Unit test the document parser on a small PDF and DOCX fixture: text extracted, page numbers attached, an image-only PDF yields `NEEDS_OCR`
- Unit test the audit hash chain: modify a row → its own hash mismatches; delete a row → the next row's `previous_hash` mismatches; `verify_chain` names the first bad row
- Unit test that `audit_repository` exposes no mutating method (introspect the class) and that a raw `UPDATE audit_log` from the app role is rejected
- Integration test upload: file lands at `documents/{id}/v1/{name}`, `Document` row created, chunks persisted with page numbers, audit row `DOCUMENT_UPLOAD/SUCCESS`
- Test upload rejection: an `.exe` renamed `.pdf` is refused on magic bytes, an oversized file is refused, a filename of `../../etc/passwd` produces a sanitized key inside the document's own prefix
- Test access control: Standard User `POST /documents` → 403 and an audit row with `outcome='DENIED'`; tenant-B document id → 404
- Test re-upload of the same name: `v2` key written, `v1` still readable, both rows present (lineage without bucket versioning)

### Test & acceptance (phase gate — must pass before next phase)

Same split as Phase 2. Four of these ten are verified as written; six describe an HTTP round
trip that needs a real user row to authenticate against, so they are marked **BLOCKED — no
PostgreSQL on this machine** with the service-layer test that already covers the *rule* named
alongside. What is outstanding on those six is the wiring, not the decision. The service tests
are not a weaker substitute for one thing in particular: they drive `DocumentService` and
`QueryService` through the same methods the routes call, with the S3 *client* faked one level
below `S3StorageService`, so key construction and filename sanitization are real code under test.

- [ ] **BLOCKED — no PostgreSQL on this machine.** `POST /api/v1/documents` (admin) accepts PDF/DOCX up to `MAX_UPLOAD_SIZE_MB`, validates by magic bytes not extension, stores at `documents/{document_id}/v{n}/{filename}`, records name/type/size/page-count/uploader/tenant, and audits `DOCUMENT_UPLOAD`. Every clause is verified at the service layer in `test_document_service.py`: `TestUpload` (row, chunks, one key, SSE, tenant metadata), `TestMagicByteValidation` (11 tests — a renamed `.exe`, plain text, a DOCX named `.pdf`, four disallowed extensions, empty, oversize — each asserting nothing reached S3 *and* no row was written), `TestFilenameSafety` (16 tests), `TestVersionLineage`. Unverified: that the multipart route hands the service those bytes and the INSERT commits.
- [x] Extracted chunks carry `page_number` or `section_label`; a document with no extractable text is `NEEDS_OCR` and excluded from retrieval — `TestCitability` (PDF chunks carry pages, DOCX chunks carry heading labels, no chunk is blank or uncitable) and `test_document_parser.py::TestParsePdf` (a skipped blank page does not renumber the rest — a citation that names the wrong page is worse than no answer). `NEEDS_OCR` on a text-free PDF is asserted in `TestUpload` and on restore in `TestLifecycle`. **The "excluded from retrieval" half is re-checked in Phase 4's gate**, where retrieval exists: today the status is set correctly and there is no retrieval to exclude it from.
- [ ] **BLOCKED — no PostgreSQL on this machine.** `GET /api/v1/documents` paginates tenant documents with status filtering; Standard Users may list and preview but not upload, delete or restore. `TestVisibility` covers cross-tenant 404, `DELETED` hidden from Standard Users but visible to Administrators, and `name_contains="%"` returning nothing (a search box is not a pattern language); the role split on upload/delete/restore is asserted on every method.
- [ ] **BLOCKED — no PostgreSQL on this machine.** `DELETE /api/v1/documents/{id}` soft-deletes (`status='DELETED'`, audit `DOCUMENT_DELETE`); `POST /{id}/restore` returns it to `ACTIVE` (audit `DOCUMENT_RESTORE`); the S3 object is untouched in both cases. `TestLifecycle` asserts the bytes and chunks survive a delete, that deleting twice writes one audit row, that a restored scan returns to `NEEDS_OCR` rather than `ACTIVE`, and that restoring a live document raises rather than writing a `DOCUMENT_RESTORE` row. `test_api_surface.py` adds the half that needs no DB: there is exactly one document delete route and no purge path.
- [ ] **BLOCKED — no PostgreSQL on this machine.** `GET /api/v1/documents/{id}/preview` returns a short-TTL presigned URL only after a tenant and role check. `TestPreview` asserts an outsider gets nothing *and* that `generate_presigned_url` was never reached — the check has to happen before the URL is minted, because a presigned URL cannot be recalled. The TTL comes from settings and is recorded in the audit detail.
- [ ] **BLOCKED — no PostgreSQL on this machine.** `POST /api/v1/queries` persists a `PENDING` query with `user_id`, `tenant_id`, `created_at` and audits `QUERY_SUBMIT` — `test_query_service.py::TestSubmit`, including that tenant and user come from the actor rather than the request body, and `QueryCreate` rejecting a client-supplied `answer_text`.
- [ ] **PARTIAL — the read route is BLOCKED, the immutability is not.** `GET /api/v1/audit` (admin only) returns logs filterable by action, user and date range; Standard Users get 403; **there is no route that writes, edits or deletes a log**. The last clause is verified: `test_api_surface.py::TestAuditHasNoWriteSurface` reads the generated OpenAPI document and asserts the audit paths are exactly `GET /audit` and `GET /audit/verify`. The filtering logic is covered against `FakeAuditRepository` (newest-first, every filter applied); the 403 for a Standard User over HTTP needs a user row.
- [ ] **BLOCKED for the "direct DB tamper" clause.** `audit_service.verify_chain(tenant_id)` passes on a clean table and fails, identifying the row, after a tamper. Verified in memory in `test_audit_service.py` and again across a full document lifecycle in `TestAuditPrivacy`, which flips one entry's `outcome` and asserts `verify_chain` names *that* `log_id`. What a database adds is the `REVOKE UPDATE, DELETE` check — that the tamper has to be done as a superuser at all.
- [x] No log line or audit `detail` field contains a password, token, account number or document body text — `TestAuditPrivacy` uploads `acct-4417829930-statement.pdf`, runs preview/delete/restore, and asserts the account number, the document name and its text are all absent from every `detail` while `object_id` still points at the document. `test_query_service.py` asserts the query audit detail is exactly `{question_length, status}` — the question itself lives only in the `query` row the entry points at. `test_document_parser.py::test_metadata_carries_no_document_body` keeps body text out of the JSONB that reaches a dashboard.
- [x] `pytest -q` passes with no test making a network call to AWS — **278 passed** (264 before `test_api_surface.py`). Nothing in the suite constructs a boto3 client: `S3StorageService` takes the client by injection and every test passes `FakeS3Client`.

### Phase completion checklist

- [x] All Tasks (actionable) complete
- [ ] All Test & acceptance checks verified — **4 of 10 verified, 6 BLOCKED on Postgres.** No check is failing. Each blocked item names the service-layer test that covers its rule; what is missing is one `alembic upgrade head` against a stock Postgres 17 instance — since the Pinecone swap that no longer means building a compiled extension.
- [x] No open High/Critical VAPT findings
- [x] Ready to ask agent: `Implement Phase 4 from WORKBREAKDOWN.md`

### Verification run (2026-08-13)

```text
pytest        → 278 passed in 259s   (142 at the end of Phase 2)
ruff check .  → All checks passed!
```

New test files this phase: `test_document_service.py` (54), `test_query_service.py` (28),
`test_document_parser.py` (40), `test_api_surface.py` (14).

Two things a reader should not have to rediscover:

- **The `get_db` change is load-bearing for this phase's audit gate.** Rolling back on any
  exception meant a denial's audit row was discarded along with the denial. `DocQueryError`
  now commits; anything unexpected still rolls back.
- **`alembic upgrade head` has never been run.** The migration is written and reviewed but
  unexecuted, so the schema is a claim about SQLAlchemy metadata, not an observed table
  layout. This is the same blocker as Phase 1's migration gate item.

### Vector store change: pgvector → Pinecone (2026-08-13, after Phase 3 closed)

Decided after Phase 3's gate, applied to the Phase 1–3 code that already existed and to the
Phase 4–6 plans above. **Pinecone serverless is the vector store; there is no other.**

What changed in code:

- `backend/core/config.py` — the `PINECONE_*` block, a `PINECONE_INDEX_NAME` format
  validator (lowercase, alphanumeric-and-hyphen, ≤45 chars — Pinecone's own rule, caught at
  startup instead of at first upsert), and `pinecone_namespace(tenant_id)`. The namespace is
  built in exactly one place for the same reason `build_key` is: a tenant boundary assembled
  by string concatenation at four call sites is a boundary with four ways to be wrong.
- `backend/models/document.py` — the `embedding Vector(1024)` column is gone, replaced by
  `embedded_at`. The vector's id is `str(chunk.id)`; PostgreSQL keeps text, citation, status
  and ownership.
- `alembic/versions/20260813_0001_enable_pgvector.py` — now enables only `pgcrypto`. It
  keeps its `0001_pgvector` revision id and filename because later revisions point at that
  id; renaming it would break the chain to relabel a pointer.
- `alembic/versions/20260813_0003_documents_and_queries.py` — no `Vector` import, no
  embedding column, no vector index.
- `requirements.txt` — `pgvector==0.5.0` → `pinecone==9.1.0` (the package is `pinecone`, not
  the retired `pinecone-client`; v9 ships `PineconeAsyncio`, so retrieval can be async like
  everything else here).
- `conftest.py` / `pyproject.toml` — `PINECONE_API_KEY` is **hard-blanked** in the test
  harness, not defaulted, and there is a `requires_pinecone` marker. A test that
  accidentally built a real client would otherwise write test vectors into the live index,
  and an upsert has no dry-run.
- `backend/utils/vector_store.py` — still a Phase 4 scaffold, but its docstring now records
  the four decisions (namespace-per-tenant, id = `str(chunk.id)`, Postgres as status
  authority + overfetch, citations-only metadata) so Phase 4 implements them rather than
  re-deciding them.

Re-verified after the swap, with `pgvector` **uninstalled** from `.venv` — stronger evidence
than grep that nothing imports it:

```text
pytest        → 278 passed
ruff check .  → All checks passed!
```

One side effect worth knowing: the Postgres blocker got smaller. There is no compiled
extension to build any more, so a stock PostgreSQL 17 — plain installer or `postgres:17` —
unblocks the eleven `BLOCKED` gate items across Phases 1–3.

### Agent prompt (copy-paste)

```text
Implement Phase 3 of WORKBREAKDOWN.md: documents, S3 storage and the audit trail for DocQuery.

Build Document, DocumentChunk, Query and AuditLog models. Documents go to S3 immediately — there is no local-disk stage. Because bucket versioning is unavailable to our IAM role, lineage is app-side: every version gets an immutable key documents/{document_id}/v{n}/{filename} and nothing is overwritten. Sanitize the filename; never build a key from a user-supplied path.

AuditLog follows .claude/skills/audit-trail-generation/references/audit-schema.md exactly, hash chain included. audit_repository has append and read methods only — no update, no delete — and the migration revokes UPDATE and DELETE from the application role. audit_service computes integrity_hash over the canonical fields plus the previous entry's hash, strips CR/LF from user input, and masks secrets and PII.

Extract text with pypdf and python-docx, keeping the page or section on every chunk — an uncitable chunk must not exist. A document with no extractable text becomes status='NEEDS_OCR' and is excluded from retrieval; do not add an OCR dependency, that needs approval.

Layer it: repositories query, services decide, routes validate and delegate to exactly one service call. Every repository method takes tenant_id. Delegate the ingestion work to document-ingestion-agent and the log work to audit-logging-agent. Fake the S3 client in tests; no test hits AWS.
```

## Phase 4: AI query engine — RAG, embeddings & async processing

### Objectives

- Embed document chunks with Titan and upsert them into **Pinecone**, one namespace per tenant
- Answer natural language questions strictly from retrieved chunks via Nova Pro, with mandatory source references (BR-02, BR-03)
- Refuse to answer when retrieval is empty or below threshold — no fallback to model knowledge
- Move embedding off the request path with Celery, eager locally
- Stand up an accuracy harness so the ">90%" target is a measurement, not a claim

### Tasks (actionable)

- [x] `backend/services/bedrock_client.py` **(new)** — thin boto3 `bedrock-runtime` wrapper: `embed(texts) -> list[list[float]]` (Titan, InvokeModel) and `generate(system, messages) -> text` (Nova Pro, Converse, `temperature=0.0`). Retries with backoff on throttling. Model ids from settings; nothing hardcoded. No user-controlled URL ever reaches boto3 (SSRF)
- [x] `backend/utils/vector_store.py` — the **only** module that imports `pinecone`. Async client (`PineconeAsyncio`), index and namespace from settings. Four functions: `upsert_embeddings(tenant_id, chunks_with_vectors)` (id = `str(chunk.id)`, batches of `PINECONE_UPSERT_BATCH_SIZE`, metadata = citation fields only — **never chunk text**), `search(tenant_id, vector, top_k)` (queries `pinecone_namespace(tenant_id)` only and returns `(chunk_id, score)` pairs — no SQL, no business rules), `delete_document_vectors(tenant_id, document_id)`, and `delete_namespace(tenant_id)`. Isolation is the namespace, never a metadata filter
- [x] `scripts/setup_pinecone.py` **(new)** — create the serverless index if absent (`PINECONE_INDEX_NAME`, `dimension=BEDROCK_EMBEDDING_DIMENSIONS`, `metric=PINECONE_METRIC`, `ServerlessSpec(cloud, region)`), then wait for ready and print the host. Idempotent: an existing index whose dimension or metric disagrees with settings is a **hard error**, not a silent mismatch — Pinecone cannot change either after creation. Replaces the pgvector HNSW-index migration; there is no vector index in Alembic any more
- [x] `backend/repositories/query_repository.py` — `load_chunks(chunk_ids, tenant_id)`: fetch the chunk rows for Pinecone's hits, joined to `document`, keeping only `tenant_id` matches whose document is `ACTIVE`, and return them **in Pinecone's score order**. This is where a stale vector stops being citable
- [x] `backend/utils/chunk_splitter.py` **(new)** — split parsed pages into `CHUNK_SIZE_TOKENS` chunks with `CHUNK_OVERLAP_TOKENS` overlap, propagating page/section to every chunk
- [x] `backend/services/rag_service.py` — the pipeline: embed the question → `vector_store.search` for `RAG_TOP_K * PINECONE_OVERFETCH_FACTOR` in the tenant's namespace → `query_repository.load_chunks` to re-check tenant and `ACTIVE` status in PostgreSQL → drop below `RAG_MIN_SIMILARITY` → trim to `RAG_TOP_K` → if nothing survives return the not-found result → build a grounded prompt → `bedrock_client.generate` → parse and validate references
- [x] `backend/services/source_tracer.py` **(new)** — map each cited passage back to its chunk, producing `source_document_name` + `section_or_page` + `confidence_level`. **Drop any citation that does not resolve to a retrieved chunk**, and if that leaves none, downgrade the answer to not-found
- [x] `backend/services/query_service.py` — extend `submit` to run the RAG pipeline, record `answer_text`, `source_references`, `confidence_score`, `latency_ms`, `status`, and audit the source references for compliance
- [x] `backend/services/search_history_service.py` — role-filtered read over `query_repository`: Standard User → own rows, Administrator → tenant rows. **No new table**
- [x] `backend/api/routes/search_history.py` — `GET /api/v1/history` with pagination, date range and a **fixed-pattern** keyword filter (never a user-supplied regex — ReDoS)
- [x] `backend/workers/celery_app.py` **(new)** — Celery app configured from settings; honours `CELERY_TASK_ALWAYS_EAGER` so local development needs no Redis
- [x] `backend/workers/document_processor.py` **(new)** — task: parse → chunk → embed → upsert to Pinecone → set `document_chunk.embedded_at` and `status='ACTIVE'`; on failure `status='FAILED'` with the reason audited. Idempotent by `(document_id, version)` — because the vector id is `str(chunk.id)`, a re-run overwrites its own vectors instead of duplicating them
- [x] `backend/services/document_service.py` — enqueue the processing task after upload and return `PROCESSING` instead of blocking the request. On delete, call `vector_store.delete_document_vectors` **after** the status change lands, so the document is already unciteable while the vectors are still going; on restore, re-enqueue processing. A Pinecone failure must not roll back the status change — PostgreSQL is the authority
- [x] `backend/tests/fakes.py` — `FakePineconeIndex`: in-memory namespaces, real cosine scoring, and it **rejects metadata containing a `text` field** so the no-document-text rule is enforced by the fake rather than by review
- [x] `backend/tests/test_query_service.py` — fake Bedrock client and fake Pinecone index; cover the answer path, the empty-retrieval path, the below-threshold path, the unresolvable-citation path, and the match-points-at-a-soft-deleted-document path
- [x] `backend/tests/eval/` **(new)** — a fixture corpus plus ~30 question/expected-source pairs and a `pytest` runner that reports accuracy. This is the evidence for the >90% gate

Deviations from the task list above, all deliberate:
- `vector_store` exposes a `VectorStore` class whose methods are the four functions named above, plus module-level wrappers. A class is what lets a test substitute a fake index without patching the module, which is the same reason S3 sits behind an interface on `document_service`
- `delete_document_vectors(tenant_id, document_id, chunk_ids)` takes the chunk ids as a third argument. Pinecone deletes by vector id and the id is `str(chunk.id)`, so the caller — which has just read the rows — passes them rather than making the vector store query PostgreSQL, which would put a database read in the one module allowed to import `pinecone`
- `source_tracer` owns `NOT_FOUND_ANSWER`, the exact "Information not found in uploaded documents" string, so the not-found wording has one definition and the regression suite asserts against the same constant the service returns
- there is **no `QUERY_ANSWER` audit action**. A query's audit row is `QUERY_SUBMIT` with the source references and the pipeline phase in the JSONB `detail`. Adding an enum member is an API-contract change, which `workflow.md` says to ask about first — it is listed as an open decision in `docs/RELEASE.md` §11
- `document_processor` exposes both `ingest_document` (opens its own session, the Celery entry point) and `ingest_document_in_session` (joins the caller's transaction, what the tests and the eager path use). Without the second, an eager task would deadlock against the session that enqueued it
- `DocumentRepository.refresh` was added so the service can re-read a row the worker wrote in another session
- `restore` is implemented as a **re-ingest**, not as an un-delete of the vectors: the delete removed them from Pinecone, so restoring the status and re-embedding is the only path that leaves the index consistent with `document.status`
- the non-eager dispatch is `apply_async(countdown=5)`. The upload transaction must commit before the worker reads the row; a bare `delay()` races it
- ingestion writes its own audit row for the outcome (`ACTIVE`, `NEEDS_OCR` or `FAILED` with the reason), because the upload's audit row is written before the work happens and a failure after it would otherwise be invisible
- `backend/tests/fakes.py` also gained `FakeBedrockClient` and `Corpus`. `FakeBedrockClient` embeds with a hashed bag-of-words stand-in, which is why the eval figure is not a Titan measurement — see §4.1 of `docs/RELEASE.md` and the gate note below

### Testing strategy

- Unit test the splitter: chunk sizes and overlap correct, every chunk keeps its page
- Unit test `vector_store` against the fake index: ordering by score, the upsert id is `str(chunk.id)`, batching honours `PINECONE_UPSERT_BATCH_SIZE`, no `text` key reaches metadata, and a query for tenant A never sees a vector upserted for tenant B
- Unit test the overfetch path: a match whose document is `DELETED`, `NEEDS_OCR` or another tenant's is dropped by `load_chunks` and never reaches the prompt, and `min_similarity` is applied after the database re-check
- Unit test `rag_service` with a faked Bedrock client: a grounded question cites only retrieved documents; an unrelated question returns "Information not found in uploaded documents"; a model reply citing a document that was not retrieved has that citation dropped
- Integration test upload → processing → query end to end with Celery eager, asserting the answer carries a resolvable source
- Measure latency on the fixture corpus and record the p95 for the 3–5s target
- Run the eval harness and record the accuracy figure

### Test & acceptance (phase gate — must pass before next phase)

- [ ] **NOT VERIFIED THIS SESSION** — `python scripts/setup_pinecone.py` creates the index at 1024 dims / `cosine` and is safe to re-run; a dimension or metric mismatch fails loudly. The script is written and its mismatch branch raises rather than warns, but it was **not executed here**: it has no read-only mode, and creating a serverless index is an outward-facing, billable action on the user's Pinecone account. Run it once against the target project and keep the printed host
- [x] Uploaded documents are chunked and embedded asynchronously; the upload request returns `PROCESSING` without waiting, `document_chunk.embedded_at` is set once Pinecone acknowledges, and status reaches `ACTIVE` when embedding completes — `test_document_processor.py`, `test_upload_to_query.py`. Verified against `FakePineconeIndex`, so what is proven is the app's own sequencing, not Pinecone's acknowledgement semantics
- [x] `POST /api/v1/queries` returns an answer whose every source reference includes `source_document_name` and `section_or_page`, with `confidence_level` when available — enforced by the schema (both fields required) and asserted in `test_query_service.py`, `regression/test_query_answers.py` and `compliance/test_traceability.py`
- [x] Retrieval queries only `pinecone_namespace(tenant_id)` **and** re-checks every hit against `tenant_id` and `status='ACTIVE'` in PostgreSQL; `NEEDS_OCR` and `DELETED` documents are never cited, including in the window before their vectors are deleted — `security/test_vector_isolation.py`, `compliance/test_vector_lifecycle.py`. The soft-delete window is tested directly: the fake index keeps serving the vector while the row says `DELETED`, and `load_chunks` drops it
- [x] Pinecone metadata contains no chunk text — asserted by a test, not by inspection. Stronger than asked: `FakePineconeIndex` **raises** on any metadata carrying a `text` key, so every upsert in the whole suite is the assertion
- [x] `grep -rn "import pinecone\|from pinecone" backend/` matches only `backend/utils/vector_store.py` — verified: one match, `vector_store.py:177`, inside the lazily-imported client factory
- [x] A question with no supporting content returns "Information not found in uploaded documents" with `status='NOT_FOUND'` and no invented citation — covered for both the empty-retrieval and the all-below-`RAG_MIN_SIMILARITY` paths, and the eval harness refuses 4 of 4 deliberately off-corpus questions
- [x] A citation that does not resolve to a retrieved chunk is dropped; if none remain the answer becomes not-found rather than uncited — `source_tracer` tests plus the "model cites a document that was never retrieved" case in `test_query_service.py`
- [ ] **NOT MET** — p95 end-to-end query latency on the fixture corpus is within the 3–5 second target, measured and written down. What exists is a load run (`backend/tests/perf/test_query_load.py`, 64 requests at 16 concurrent, 0 errors, wall p50 92.0 / p95 225.3 ms) with **every network call faked** — no Titan embedding, no Pinecone query, no Nova generation, which is exactly what the 3–5 s budget is spent on. Those milliseconds are the app's own overhead and **must not be quoted as this gate**. The real number is `percentile_cont(0.95)` over `query.latency_ms` from live traffic; it needs working AWS credentials. `docs/RELEASE.md` §4.2 and §4.3
- [x] **MET, with two qualifications stated** — the eval harness in `backend/tests/eval/` reports accuracy and the figure is **93.3% top-1 (28 of 30)**, recall@k 93.3%, 0 misranked, 2 below `RAG_MIN_SIMILARITY`, 4 of 4 off-corpus questions refused, over 6 fixture documents / 28 passages. Written down here and in `docs/RELEASE.md` §4.1, not assumed. (a) It was measured against **`FakeBedrockClient`, a hashed bag-of-words stand-in — not Titan**, so it scores the pipeline's plumbing and thresholds, not embedding quality; the real figure can only be lower or higher, and this number cannot be quoted without that sentence. (b) "runs in CI" is **not** met in the literal sense: this repo has no CI pipeline at all (no `.github/workflows`, no other config), so it runs as part of `pytest -q`. Both misses are below-threshold refusals, not wrong answers — the harness prefers a refusal to a bad citation, which is the behaviour `rag-answer-grounding.md` asks for
- [x] `GET /api/v1/history` returns own rows for a Standard User and all tenant rows for an Administrator, with pagination and date filtering; the keyword filter uses a fixed pattern — `regression/test_search_history.py`, `security/test_idor.py`. It is a projection over `query`, not a second table
- [x] Every query records an audit row with its source references — as `QUERY_SUBMIT` with the references in the JSONB `detail`; see the deviation note above on why there is no separate `QUERY_ANSWER` action
- [x] No test contacts AWS or Pinecone; `pytest -q` is green offline with `PINECONE_API_KEY` blank — the whole suite runs with the key blanked in the harness, and it is green: **1,712 passed / 18 skipped** without a database, **1,727 passed / 3 skipped** with one

### Phase completion checklist

- [x] All Tasks (actionable) complete
- [ ] All Test & acceptance checks verified — **11 of 13 met**. Not met: the p95 against the 3–5 s target (needs live Bedrock and Pinecone), and `setup_pinecone.py` against a real index (not run here — billable and outward-facing). The eval gate is met at 93.3% but only against the stand-in embedder
- [x] No open High/Critical VAPT findings — none attributable to Phase 4. The one open High, F-01, is in the Phase 2 registration flow; see Phase 6 and `docs/RELEASE.md` §5.1
- [x] Ready to ask agent: `Implement Phase 5 from WORKBREAKDOWN.md` — Phase 5 is implemented

### Agent prompt (copy-paste)

```text
Implement Phase 4 of WORKBREAKDOWN.md: the AI query engine for DocQuery. Delegate to ai-query-agent and follow the rag-pipeline and source-tracing skills.

Embeddings: amazon.titan-embed-text-v2:0 at 1024 dims via boto3 InvokeModel. Generation: us.amazon.nova-pro-v1:0 via boto3 Converse at temperature 0.0. Read both ids and every RAG tuning value from backend/core/config.py.

The vector store is Pinecone serverless — not pgvector, FAISS or Chroma. PostgreSQL has no embedding column; document_chunk.embedded_at records that the upsert was acknowledged. backend/utils/vector_store.py is the only module allowed to import pinecone; use the async client. Three rules that are already decided, so implement them rather than re-deciding:
  1. One namespace per tenant, built only by Settings.pinecone_namespace(tenant_id). Isolation is the namespace, never a metadata filter — a wrong namespace returns nothing, a forgotten filter returns another tenant's chunks.
  2. The vector id is str(chunk.id), so re-processing overwrites its own vectors instead of duplicating them.
  3. Pinecone metadata carries citation fields only (document_id, document_name, page, section) and never the chunk text. The passage stays in document_chunk.text.
Run scripts/setup_pinecone.py to create the index at 1024 dims with metric=cosine; dimension and metric are fixed at creation, so a mismatch against settings must be a hard error.

Pipeline in rag_service: embed the question, search the caller's namespace for RAG_TOP_K * PINECONE_OVERFETCH_FACTOR hits, load those chunk ids through query_repository.load_chunks which re-checks tenant_id and status='ACTIVE' in PostgreSQL, discard anything below RAG_MIN_SIMILARITY, trim to RAG_TOP_K, and if nothing survives return "Information not found in uploaded documents" — never fall back to the model's own knowledge. PostgreSQL stays the authority on status: a soft-deleted document must not be citable even while its vectors are still in the index. Build a grounded prompt from the surviving chunks, generate, then have source_tracer resolve every citation back to a retrieved chunk. Drop citations that do not resolve; if that leaves none, the answer becomes not-found rather than uncited.

Move parse/chunk/embed/upsert into a Celery task in backend/workers/document_processor.py, idempotent per (document_id, version). CELERY_TASK_ALWAYS_EAGER is true locally, so this must work with no Redis running. Document delete removes that document's vectors after the status change lands; a Pinecone failure must not roll back the status change.

search_history_service is a role-filtered read over query_repository — do not create a search_history table.

Build backend/tests/eval/ with a fixture corpus and question/expected-source pairs so the >90% accuracy gate is measured. Fake Bedrock and Pinecone in every test — PINECONE_API_KEY is blanked in the test harness, so no test may hit AWS or Pinecone.
```

## Phase 5: Frontend flows, search history & analytics

### Objectives

- Document management UI for Administrators: upload, preview, delete, restore, with metadata
- Query interface showing the answer with its source references and confidence
- Search history views honouring the role split
- Analytics dashboard (Administrator only) with keyword, document-usage and trend reporting, exportable to PDF/Excel
- Responsive across desktop and mobile, with clear feedback on every action

### Tasks (actionable)

- [x] `backend/repositories/analytics_repository.py` **(new)** — parameterized aggregate queries: top/bottom keywords, most-referenced documents, query volume per period, active users. `GROUP BY` in SQL, not in Python
- [x] `backend/services/analytics_service.py` — assemble report DTOs; keyword extraction uses **fixed** patterns and a stopword list, never a user-supplied regex
- [x] `backend/utils/export_utils.py` — server-side PDF (`reportlab`) and Excel (`openpyxl`) rendering for history and analytics; escape every cell value, and never emit a masked field unmasked
- [x] `backend/schemas/analytics_schemas.py` — `KeywordTrendResponse`, `DocumentUsageResponse`, `UsageTrendResponse`, `AnalyticsFilter`
- [x] `backend/api/routes/analytics.py` — `GET /api/v1/analytics/keywords`, `/documents`, `/trends`, `POST /api/v1/analytics/export` — all `require_admin`, all audited (`ANALYTICS_VIEW`, `DATA_EXPORT`)
- [x] `frontend/app/documents/page.tsx` — Administrator document management: upload form, metadata table (name, upload date, uploader, status, version), preview modal, delete and restore actions with confirmation
- [x] `frontend/app/query/page.tsx` — natural language input, pending state while `PENDING`, answer panel, source references list, and an explicit empty state for `NOT_FOUND`
- [x] `frontend/app/history/page.tsx` — history table with pagination and date filtering; the same page serves both roles because the API decides what it returns
- [x] `frontend/app/analytics/page.tsx` — Administrator dashboard: keyword chart, document usage chart, trend line, export buttons
- [x] `frontend/app/audit/page.tsx` — Administrator audit viewer: filter by action, user, date range; read-only, no edit affordance
- [x] `frontend/components/DocumentUploader.tsx` — client-side type and size pre-check, progress, error surface. The server still validates
- [x] `frontend/components/QueryInterface.tsx` — input, submit, loading, answer rendering
- [x] `frontend/components/SourceReference.tsx` — document name, section/page, confidence, link to the presigned download
- [x] `frontend/components/SearchHistoryTable.tsx` — question, answer, timestamp, sources; user column shown only when the API returns one
- [x] `frontend/components/AnalyticsCharts.tsx` and `frontend/components/AuditLogTable.tsx` — presentational, typed from `frontend/types/`
- [x] `frontend/services/documentService.ts`, `queryService.ts`, `analyticsService.ts` — typed clients over `frontend/lib/api-client.ts`, sending the CSRF header on unsafe methods
- [x] `frontend/hooks/useQuery.ts` — submit and poll a query to completion
- [ ] **WRITTEN BUT UNRUNNABLE** — `frontend/tests/QueryInterface.test.tsx` — renders answer with sources, renders the not-found state, does not render a source without a document name. The file exists and covers those three cases, but **there is no test runner installed**: `package.json` declares `"test": "jest"` and `"lint": "next lint"` while `devDependencies` contains neither `jest`/`vitest` nor `eslint`, and there is no config file for either. So both commands fail and this test has **never executed**. Adding the tooling means adding dependencies, which `workflow.md` says to ask about first — it is an open decision in `docs/RELEASE.md` §11. `tsc --noEmit` does run and is clean, so the file at least type-checks

Deviations from the task list above, all deliberate:
- extra frontend files the list does not name: `types/analytics.ts` and `types/audit.ts` (the charts and the audit table had to be typed from somewhere), `services/auditService.ts` (the audit page needs a client and the list only named three), `components/ui/page-shell.tsx` (one nav-plus-header shell, so five pages do not each re-implement the role-aware nav), and `app/history/history-panel.tsx` + `app/documents/documents-panel.tsx` — the interactive halves split out as client components so the route files stay server components
- `lib/api-client.ts` gained `download()`, `safeFilename()` and a `DownloadedFile` type for the PDF/Excel and presigned-document responses. `safeFilename()` exists because a filename arriving in a `Content-Disposition` header is server-supplied but originally user-supplied, and it is about to become a path on the user's disk
- `lib/constants.ts` gained `MIN_QUESTION_LENGTH` / `MAX_QUESTION_LENGTH` so the client pre-check uses the same bounds the Pydantic schema enforces instead of a second opinion
- `components/SourceReference.tsx` also exports `SourceReferenceList`, which is what both the query page and the history table actually render
- `frontend/package.json` declares `"test": "jest"` and `"lint": "next lint"` but **installs neither jest nor eslint** — see the last task above; this is a gap, not a design choice

### Testing strategy

- Upload through the UI and confirm metadata appears; exercise preview, delete and restore, and confirm each shows a confirmation and a result
- Submit domain queries (tax, customer account, compliance wording) and confirm answer plus source references render, and that the not-found case renders as an empty state rather than a blank panel
- Log in as each role and confirm the history table content differs correctly, and that `/analytics`, `/documents` and `/audit` are unreachable for a Standard User both in the nav and by typing the URL
- Export history and analytics to PDF and Excel and open both files
- Check the layout at mobile, tablet and desktop widths
- Inject `<script>` and `"><img onerror>` into a document name and a question, then confirm the rendered output is escaped everywhere it appears, including exports

### Test & acceptance (phase gate — must pass before next phase)

Read this before reading the ticks below: **the backend half of every item here is covered by tests; the "in a browser" half is not.** No item was verified by clicking through a running UI in this session — there is no browser automation in this repo and no frontend test runner installed. `[x]` below means "the API behaviour is asserted by a test and the component that consumes it type-checks"; where an item is *only* observable in a browser it is left unticked and says so.

- [x] Administrators upload documents and see name, upload date, uploader, status and version; preview, delete and restore all succeed and appear in the audit log — API side asserted in `test_document_routes.py` and `regression/test_documents.py`, including the audit row for each action. The table renders those five columns and was exercised manually in an earlier session; it is not covered by an automated UI test
- [x] The query page renders the answer with each source's document name and section/page, plus confidence when present, and renders a clear empty state when the backend returns `NOT_FOUND` — `SourceReferenceList` renders name and section/page unconditionally and confidence only when present, and `QueryInterface` branches on `NOT_FOUND` to a dedicated panel rather than an empty answer box. The assertion that would prove it, `frontend/tests/QueryInterface.test.tsx`, cannot run for lack of a runner
- [x] Standard Users see only their own history; Administrators see system-wide history with the user identity column — enforced server-side and asserted in `regression/test_search_history.py` and `security/test_idor.py`; the table shows the user column only when the API returns one, so the role split is the API's decision and not the UI's
- [ ] **HALF MET** — the analytics dashboard shows most and least searched keywords, most referenced documents and usage trends; PDF and Excel export both download and open. The four endpoints are implemented and asserted in `test_analytics_routes.py` / `regression/test_analytics.py`, and `test_export_utils.py` checks the produced bytes are a valid PDF (`%PDF-` header) and a readable workbook whose cells hold the expected values. **Nobody opened either file in Acrobat or Excel** — that is the half that is missing, and a file that parses with `openpyxl` can still displease Excel
- [x] Typing `/analytics`, `/documents` or `/audit` as a Standard User yields a server-side 403, not just a hidden nav item — every admin-only route sits behind `require_admin`, and `regression/test_rbac.py` plus `security/test_idor.py` call each one as a Standard User and assert 403 with the denial audited. The page itself renders and then shows the API's error, which is the correct shape: the nav hiding is cosmetic and the 403 is the control
- [ ] **NOT VERIFIED** — the UI is usable at 375px, 768px and 1280px widths. The layouts are built with Tailwind responsive utilities and `page-shell.tsx` collapses the nav below `md`, but **no one has looked at the three widths**. This needs a browser and a human eye; it cannot be asserted from here and I am not going to tick it on the strength of the class names
- [x] Script payloads in document names, questions and answers render escaped in the UI and in exports — no execution. Exports are covered by tests (`security/test_xss.py` pushes `<script>` and `"><img onerror>` through document names and question text into both renderers and asserts the payload comes back inert), and the UI never uses `dangerouslySetInnerHTML` — verified by search: the only two matches under `frontend/` are comments in `types/query.ts` and `SourceReference.tsx` saying not to use it, and there is no `innerHTML`, `eval` or `new Function` either. That absence is what makes React's default escaping the whole story here
- [x] Every unsafe request from the frontend carries the CSRF header and succeeds; stripping it produces 403 — `api-client.ts` attaches the double-submit header on every non-GET, and `security/test_csrf.py` walks every unsafe route asserting 403 without the header, 403 on a mismatched one, and success with it. 3 tests in that module skip by design: the login and refresh endpoints are the ones that *issue* the token, so they are exempt and the skip records why
- [x] API errors surface a readable message and leave no partially written state — `api-client.ts` funnels every non-2xx into a typed `ApiError` carrying the server's `detail`, and each panel renders that string rather than a generic failure. The no-partial-state half is the backend's: every service method runs in one transaction, and a `PydanticValidationError` handler was added so a malformed body returns 422 with field errors instead of a 500 after a partial write
- [x] Analytics and export access is audited (`ANALYTICS_VIEW`, `DATA_EXPORT`) with the requesting user recorded — asserted per route in `test_analytics_routes.py`, including the denied Standard User attempts, which are audited too

### Phase completion checklist

- [ ] All Tasks (actionable) complete — **17 of 18**. Every file exists; the exception is `frontend/tests/QueryInterface.test.tsx`, which is written but has no runner to execute it
- [ ] All Test & acceptance checks verified — **8 of 10 met**, and see the note above the list about what a tick means here. Not met: the three responsive widths (needs a browser) and opening the exported PDF/XLSX in real applications
- [x] No open High/Critical VAPT findings — none attributable to Phase 5. F-01, the one open High, is in the Phase 2 registration flow
- [x] Ready to ask agent: `Implement Phase 6 from WORKBREAKDOWN.md` — Phase 6 is implemented

### Agent prompt (copy-paste)

```text
Implement Phase 5 of WORKBREAKDOWN.md: the DocQuery frontend, search history views and analytics. Delegate the UI to frontend-dashboard-agent and the aggregation to analytics-agent.

Backend first: analytics_repository does the aggregation in parameterized SQL (GROUP BY in the database, not in Python), analytics_service builds the DTOs with fixed keyword patterns and a stopword list — never a user-supplied regex — and backend/api/routes/analytics.py exposes keywords, documents, trends and export behind require_admin, auditing every call. PDF and Excel are rendered server-side in backend/utils/export_utils.py with reportlab and openpyxl.

Frontend is App Router only: pages at frontend/app/{documents,query,history,analytics,audit}/page.tsx. Use the existing component names — DocumentUploader.tsx, QueryInterface.tsx, SourceReference.tsx, SearchHistoryTable.tsx, AnalyticsCharts.tsx, AuditLogTable.tsx. Do not create frontend/pages/, frontend/src/, DocumentUpload.tsx or HistoryTable.tsx, and do not build an export utility in the frontend.

Escape every value the model, a user or a document produced — document names and query text are untrusted, in the UI and in exports. Send the CSRF header on every unsafe request. Hiding a nav item is not access control: the server enforces the role.

An answer with no resolvable source must render as "Information not found in uploaded documents", never as a bare answer.
```

## Phase 6: Testing, security review & release

### Objectives

- VAPT review across every endpoint for the classes named in `.claude/rules/secure-coding-vapt.md`
- Prove the audit log is tamper-evident and complete, including denied attempts
- Regression coverage for BR-01 to BR-09
- Compliance verification: 100% audit availability, answer-to-source traceability, RBAC enforcement
- Release readiness: benchmarks recorded, runbook and rollback written, findings closed or formally waived

### Tasks (actionable)

- [x] `backend/tests/security/test_sql_injection.py` — injection payloads through every filter and search parameter, including the analytics date and keyword filters. Its 3 `requires_db` live-driver probes **executed and passed** on 2026-08-14 via `scripts/run_db_tests.py`; they skip on a machine with no test DSN
- [x] `backend/tests/security/test_xss.py` — payloads in document names, question text and metadata; assert stored-and-rendered output is escaped
- [x] `backend/tests/security/test_log_injection.py` — CR/LF, control characters and format strings in inputs never produce a second log line or a forged audit row
- [x] `backend/tests/security/test_idor.py` — cross-user and cross-tenant ids for documents, queries, history and audit all return 404/403 and are audited
- [x] `backend/tests/security/test_vector_isolation.py` — two tenants with deliberately near-identical chunk text: tenant A's question never retrieves a vector upserted for tenant B, the query is issued against exactly one namespace, and a chunk id guessed from tenant B never resolves through `load_chunks` for tenant A. Also assert Pinecone metadata holds no document text
- [x] `backend/tests/security/test_csrf.py` — every unsafe route rejects a request without a matching CSRF token
- [x] `backend/tests/security/test_path_traversal.py` — traversal in upload filenames and preview parameters cannot escape the document's own S3 prefix
- [x] `backend/tests/security/test_mass_assignment.py` — client-supplied `role`, `tenant_id`, `status`, `integrity_hash` and `user_id` are rejected, not silently applied
- [x] `backend/tests/security/test_secrets_in_logs.py` — scan captured log output and audit `detail` for password, token, account-number and document-body patterns
- [x] `backend/tests/regression/test_document_management.py` — BR-01: upload, preview, delete, restore, metadata, version lineage
- [x] `backend/tests/regression/test_ai_query.py` — BR-02/BR-03: natural language answers, domain terminology, mandatory source references, the not-found path
- [x] `backend/tests/regression/test_audit_logs.py` — BR-05: complete capture, chain verification, Administrator-only read, no mutation path
- [x] `backend/tests/compliance/test_traceability.py` — every stored answer resolves to a live document and a real page/section; an orphaned reference fails the test
- [x] `backend/tests/compliance/test_vector_lifecycle.py` — PostgreSQL and Pinecone agree: every `ACTIVE` chunk has `embedded_at` set and a vector present, a soft-deleted document is uncitable **immediately** (before its vectors go) and its vectors are gone afterwards, and a purge leaves nothing behind in the namespace
- [x] performance run (`backend/tests/perf/test_query_load.py`) — 64 concurrent queries, 0 errors; latency recorded in `docs/RELEASE.md` **as app overhead against faked backends, not as the 3–5s gate** (that figure is `percentile_cont` over `query.latency_ms` and is still unmeasured)
- [x] VAPT pass over the whole diff, findings triaged High/Critical/Medium/Low → `docs/RELEASE.md` §5. Done **inline**: agent delegation was disabled for this session, so `security-vapt-reviewer` did not run. 5 findings fixed, 7 open (1 High, 5 Medium, 1 Low)
- [x] release notes (`docs/RELEASE.md`) — measured benchmarks with their caveats, the eval accuracy figure and its corpus, runbook, rollback plan, the waiver table (unsigned) and the open-decision list

Deviations from the task list above, all deliberate:
- the security package has **9** modules as listed, the regression package has **7** (the list names three: `test_rbac.py`, `test_search_history.py`, `test_analytics.py` and `test_coverage_map.py` were added), and the compliance package has **3** (`test_audit_immutability_db.py` was added for the DB-level controls)
- `backend/tests/harness.py` was extracted so the compliance, regression and perf packages share one stack builder; `backend/tests/security/conftest.py` thinned to match
- `scripts/run_db_tests.py` runs the `requires_db` set — 15 tests that **skip silently** without a test DSN, which is why they need their own command. They **were executed on 2026-08-14** against a migrated `docquery_test`: 15 passed, exit 0. See `docs/RELEASE.md` §3.1 for what that run establishes and the four test defects it exposed. The script derives `<db>_test` from the real DSN read through `core/config.py`, passes it to pytest in the child's **environment only** — never on the command line, never printed — and its `reset` refuses any database name not ending in `_test`
- `backend/tests/perf/` is a new package the task list did not name, holding the load run; `backend/tests/compliance/` likewise
- fixes made to production code while writing the suites, each because a test caught it: `DocumentChunk.citation` rendered the page label with the wrong casing; the module docstring on `routes/queries.py` described a response shape the route had stopped returning; and a `PydanticValidationError` handler was added to `api/app.py` so a malformed body returns 422 with field errors rather than a 500
- `repr=False` on the ORM relationship attributes, so a debugger or a failing assertion cannot print a document's whole chunk text into a log — `secure-coding-vapt.md` treats document body text as customer content
- `SSL_CERT_FILE` is set in the test bootstrap because boto3 on this Windows host otherwise fails certificate verification before any fake can intercept it
- `make_docx_with_traversal_member` was added to `document_fixtures.py`: a real `.docx` whose zip member name is `../../evil.txt`, so the path-traversal test attacks the parser with a genuinely malicious archive rather than a mocked one

### Testing strategy

- Run the security suite against a seeded two-tenant database with one Administrator and one Standard User per tenant
- Tamper with the audit table directly, then confirm `verify_chain` reports the exact row
- Run the full regression and compliance suites, then the eval harness from Phase 4, and record every number
- Load-test the query path at the expected concurrency and capture latency percentiles
- Have `security-vapt-reviewer` re-review after every fix, not once at the end

### Test & acceptance (phase gate — release gate)

- [ ] **NOT MET** — VAPT review finds no open High or Critical issue … or each remaining one carries a written, owner-signed risk acceptance. **F-01 is an open High**: `POST /auth/register` is unauthenticated and yields an immediately active account (`is_active` defaults `True`, no email verification, no approval) with query access to the tenant's whole corpus, and no route exists to deactivate it. Nothing in the waiver table is signed. See `docs/RELEASE.md` §5.1 and §6
- [x] The audit log captures every critical activity with user identity, action, timestamp, affected object and outcome, **including denied attempts**, and the hash chain verifies clean — `regression/test_audit_logs.py`, `test_audit_service.py`
- [x] **MET, with F-02 noted** — there is no code path, route or repository method that updates or deletes an audit row (verified: `AuditRepository` and `AuditService` expose no such method, and no route reaches one), and the database half is now **verified against a live PostgreSQL 17** rather than asserted on paper: the 12 tests in `compliance/test_audit_immutability_db.py` ran on 2026-08-14 and passed. They confirm both `BEFORE UPDATE`/`BEFORE DELETE` triggers exist, are enabled and are `FOR EACH ROW` (read out of `pg_get_triggerdef`, not inferred from `tgtype` bits — an earlier draft of the test misread bit 8 as TRUNCATE), that a single-row UPDATE, a single-row DELETE and a whole-table UPDATE are each refused with SQLSTATE 23001 or 42501 and the row is unchanged when re-read in a fresh session, that `relacl` grants only INSERT and SELECT, and that `verify_chain` names a forged row while sparing the legitimate ones. **F-02 stands and is now confirmed against the live schema, not inferred**: TRUNCATE has no statement-level trigger, so the revoked grant is its only control — a superuser or table owner would bypass it. `docs/RELEASE.md` §3.1, §5.1
- [x] Regression suites pass for BR-01 to BR-06 plus BR-AL, BR-AIQ and BR-AN — 303 tests. **BR-07/08/09 do not exist under those numbers** in the brief; the mapping assumption is listed as an open decision in `docs/RELEASE.md` §11 and encoded in `regression/test_coverage_map.py`
- [x] Every stored answer traces to a live document and a real section/page; the compliance suite fails on an orphan — `compliance/test_traceability.py`, 22 tests
- [ ] **NOT MET** — the eval accuracy figure **is** recorded with the corpus it came from (93.3% top-1 over 6 fixture documents / 28 passages, `docs/RELEASE.md` §4.1) **and with the caveat that it was measured against a hashed bag-of-words stand-in, not Titan**. The **p95 against the 3–5 s target is not measured**: it needs `percentile_cont` over `query.latency_ms` from real traffic, which needs working AWS credentials. The perf run's numbers are app overhead against faked backends and must not be quoted as the gate
- [x] Role enforcement verified server-side for every Administrator-only endpoint, with a bypass attempt logged — `regression/test_rbac.py`, `security/test_idor.py`
- [x] Vector isolation holds: retrieval touches one tenant namespace only, no cross-tenant chunk is ever returned, and a deleted document's vectors are removed and were never citable after the status change — `security/test_vector_isolation.py`, `compliance/test_vector_lifecycle.py`. Pinecone's own honouring of namespaces is the vendor's half and needs a live index
- [x] No secret, credential, account number or document body text appears in any log, audit `detail`, or Pinecone metadata — `security/test_secrets_in_logs.py`, `security/test_vector_isolation.py`
- [x] `docs/RELEASE.md` exists with benchmarks, runbook, rollback plan and the waiver list
- [ ] **NOT STARTED** — UAT sign-off recorded, with all Critical and High defects resolved. UAT needs the live round trip, which is blocked on AWS credentials

### Phase completion checklist

- [x] All Tasks (actionable) complete
- [ ] All Test & acceptance checks verified — **3 of 11 are not met**: the open High finding (F-01), the unmeasured production p95, and UAT. All three need a decision or credentials from outside this codebase; none is a missing implementation
- [ ] No open High/Critical VAPT findings, or each waived in writing by a named owner — **F-01 is open and unsigned**
- [ ] This is the final phase. Remaining work is deliberately out of scope and needs a decision before it starts: OCR for `NEEDS_OCR` documents (D-01), any external-system webhook or notification integration, and deployment infrastructure (RDS, CloudWatch, VPC, load balancing). None of these are started without agreement.

### Agent prompt (copy-paste)

```text
Implement Phase 6 of WORKBREAKDOWN.md: security review, regression and release readiness for DocQuery. Drive the review with security-vapt-reviewer and the log verification with audit-logging-agent.

Write the security suite under backend/tests/security/: SQL injection through every filter, XSS in document names and question text, log injection with CR/LF and control characters, IDOR across users and tenants, CSRF on every unsafe route, path traversal in upload filenames and preview parameters, mass assignment of role/tenant_id/status/integrity_hash, and a scan of captured logs and audit detail fields for secrets and PII.

Write regression suites for BR-01 to BR-09 and a compliance suite asserting every stored answer resolves to a live document and a real page or section — an orphaned reference must fail.

Prove tamper-evidence: modify an audit row directly in the database and confirm verify_chain names it. Confirm no route, service or repository method can update or delete a log, and that the application DB role has no such grant.

Run the Phase 4 eval harness and a concurrent load test; record the accuracy figure, the corpus it came from, and p50/p95 latency in docs/RELEASE.md alongside the runbook, rollback plan and any signed risk acceptance.

Do not implement OCR, webhooks or deployment infrastructure. All three are out of scope and need agreement first.
```
