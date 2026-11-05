"""backend/workers/celery_app.py

Role: The Celery application. Configuration only — no tasks are defined here.

Why ingestion is a background task at all
-----------------------------------------
Embedding a document is many Bedrock round trips. A 60-page PDF is a few hundred chunks, and
even at `MAX_CONCURRENT_CALLS` that is tens of seconds — far past what an HTTP upload should
hold a connection for, and past most proxy timeouts. So `POST /documents` stores the bytes,
writes the row as `PROCESSING`, and returns; the parse/chunk/embed work happens after.

Local development runs without Redis
------------------------------------
`CELERY_TASK_ALWAYS_EAGER=true` in `.env` makes Celery run a task in the calling process
instead of dispatching it. That is why this project needs no Redis locally, and it is a
supported Celery mode rather than a workaround.

It does have one consequence worth stating: **in eager mode the "background" work runs inside
the request**, so an upload takes as long as its embedding. That is a development trade, not
production behaviour. `document_service` therefore does not call `.delay()` directly — it goes
through `IngestionQueue` in `backend/workers/document_processor.py`, which awaits the async
pipeline in-process for eager mode and dispatches for real otherwise. Calling `.delay()` from
inside a running event loop would try `asyncio.run()` in a loop that is already running, which
raises.

Serialization is JSON, not pickle
---------------------------------
`task_serializer="json"` with `accept_content=["json"]`. Celery's pickle serializer
deserializes arbitrary objects from the broker, which is the insecure-deserialization finding
in `secure-coding-vapt.md` — a broker an attacker can write to becomes remote code execution
in the worker. Task arguments here are ids and strings, so JSON costs nothing.
"""

from __future__ import annotations

from celery import Celery

from backend.core.config import get_settings

_settings = get_settings()

celery_app = Celery(
    "docquery",
    broker=_settings.CELERY_BROKER_URL,
    backend=_settings.CELERY_RESULT_BACKEND,
    # Imported by name so the task module is loaded in the worker process without this module
    # importing it — which would be a cycle, since the task decorates itself with `celery_app`.
    include=["backend.workers.document_processor"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_always_eager=_settings.CELERY_TASK_ALWAYS_EAGER,
    # In eager mode a task exception propagates to the caller instead of being captured in a
    # result. `document_service` relies on that to record the failure on the document row.
    task_eager_propagates=True,
    # Acknowledge only after the task returns. If a worker is killed mid-embedding the task is
    # redelivered — safe because the vector id is `str(chunk.id)`, so a retry overwrites its
    # own vectors rather than accumulating duplicates.
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # A ceiling, not an expectation. Past this the task is killed and the document is marked
    # FAILED rather than holding a worker slot indefinitely.
    task_time_limit=1_800,
    task_soft_time_limit=1_500,
    # Results are diagnostics; nothing polls them. The document row carries the real status.
    result_expires=3_600,
)
