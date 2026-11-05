"""backend/workers/

Role: Background work that must not run inside a request — currently document ingestion.

Nothing here holds a business rule. A task is a *scheduler* for work the service layer owns:
it resolves a session, calls into `document_processor`, and records the outcome. That keeps the
ingestion rules testable without Celery, Redis or a broker.
"""
