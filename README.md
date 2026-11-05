


<div align="center">

# DocQuery

Ask natural-language questions over uploaded documents and get answers with source citations and audit logs.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-16-000000?logo=nextdotjs&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1?logo=postgresql&logoColor=white)

[Run locally](#run-locally-windows) · [Docker](#run-with-docker) · [Checks](#checks) · [Project structure](#project-structure) · [Runbook](docs/RUNNING.md)

</div>

## About the project

DocQuery lets users search business documents in plain language. Administrators upload PDF and Word files. The system extracts the text, splits it into passages, and stores vectors in Pinecone. When a user asks a question, the answer is generated only from matching passages, and each answer cites its document and page.

Two roles are supported:
- **Administrator:** uploads and manages documents, views analytics, the audit log, and all users' search history.
- **Standard User:** asks questions and views their own search history.

Every query and document action is written to an append-only audit log.

## Run locally (Windows)

1. Start PostgreSQL 17, using your own data directory:
```powershell
pg_ctl.exe -D "<your-pgdata-dir>" -o "-p 5432" start
```

2. Set up the backend (once):
```powershell
uv venv
uv pip install -r requirements.txt
copy .env.example .env
.venv\Scripts\alembic.exe upgrade head
```
Fill in the values in `.env`.

3. Start the API on port 8000 (terminal 1):
```powershell
.venv\Scripts\python.exe scripts\dev_server.py
```

4. Start the UI on port 3000 (terminal 2):
```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. To check the API, open http://localhost:8000/health.

## Create the first administrator

```powershell
.venv\Scripts\python.exe scripts\seed_admin.py --email <admin-email> --name "<name>" --tenant-name "<tenant>"
```

Standard users sign up with an email address on the same domain.

## Run with Docker

Build and run the API (port 8000):
```powershell
docker build -t docquery-api .
docker run --env-file .env -p 8000:8000 docquery-api
```

Build and run the UI (port 3000), passing the API URL the browser will use:
```powershell
docker build -t docquery-ui --build-arg NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 frontend
docker run -p 3000:3000 docquery-ui
```

Notes:
- The images do not include PostgreSQL. Set `DATABASE_URL` in `.env` to a reachable host, such as `host.docker.internal` instead of `localhost`.
- Remove `HOST` from `.env`, or set it to `0.0.0.0`. A value of `127.0.0.1` stops the API from being reachable from outside the container.
- The UI URL is fixed at build time. Rebuild the UI image if the API URL changes.

## Checks

| Command | What it runs |
|---|---|
| `pytest -q` | Backend tests, no database needed |
| `.venv\Scripts\python.exe scripts\run_db_tests.py test -m requires_db` | Database tests |
| `ruff check .` | Lint and security rules |
| `cd frontend && npm run typecheck` | TypeScript check |

## Tech stack

- **Backend:** FastAPI, SQLAlchemy 2.0 (async), PostgreSQL, Alembic
- **Vector search:** Pinecone
- **AI and storage:** AWS Bedrock (embeddings and generation), AWS S3
- **Frontend:** Next.js (App Router), React, TypeScript, Tailwind CSS

## Project structure

```
DocQuery/
├── alembic/              database migrations
├── backend/              FastAPI API
│   ├── api/              app factory, dependencies, routes (HTTP endpoints)
│   ├── core/             config, database, security, logging, exceptions
│   ├── models/           SQLAlchemy tables (user, tenant, document, query, audit log)
│   ├── repositories/     the only layer that runs database queries
│   ├── schemas/          Pydantic request and response models
│   ├── services/         business logic (auth, documents, RAG query, audit, analytics)
│   ├── utils/            document parsing, chunking, Pinecone vector store, exports
│   ├── workers/          background document ingestion
│   └── tests/            unit, compliance, security, and regression tests
├── frontend/             Next.js UI
│   ├── app/              pages (login, signup, query, documents, history, audit, analytics, users)
│   ├── components/       reusable UI pieces (query box, answers, uploader, tables, charts)
│   ├── hooks/            React hooks (auth, query)
│   ├── services/         typed API calls, one module per area
│   ├── lib/              API client, constants
│   ├── types/            TypeScript types matching backend schemas
│   ├── features/         feature modules
│   ├── public/           static assets
│   ├── proxy.ts          route protection (redirects visitors to login)
│   └── tests/            frontend tests
├── scripts/              setup, seeding, and maintenance scripts
├── docs/                 runbook and release notes
├── Dockerfile            backend container image
├── .dockerignore         files excluded from the backend image
├── main.py               backend entry point
└── requirements.txt      Python dependencies
```

### Backend layers

Requests flow in one direction: **routes → services → repositories → database**. Routes validate input and call services. Services hold the business rules. Repositories are the only code that queries the database.

