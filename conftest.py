"""conftest.py (repository root)

Role: Test-harness environment bootstrap. Runs before any `backend/` module is imported,
so `backend.core.config` sees a complete, test-only environment.

This file is the *only* place outside `scripts/` that touches `os.environ`, and it does so
deliberately: a test harness has to control the environment it hands to the application,
and doing it inside `backend/` would put environment reads in the package that
`architecture.md` says must read config solely through `backend/core/config.py`.
"""

from __future__ import annotations

import os

# Only defaults — a developer with a real local Postgres can override any of these by
# exporting the variable before running pytest.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://docquery:docquery@localhost:5432/docquery_test"
)
os.environ.setdefault("SECRET_KEY", "test-secret-key-only-for-tests-do-not-reuse-0123456789")
os.environ.setdefault("S3_BUCKET_NAME", "docquery-test-bucket")
os.environ.setdefault("ENVIRONMENT", "local")
os.environ.setdefault("CELERY_TASK_ALWAYS_EAGER", "true")
os.environ.setdefault("RATE_LIMIT_STORAGE_URI", "memory://")

os.environ.setdefault("PINECONE_INDEX_NAME", "docquery-test")

# Hard-set, not defaulted: no real credential may leak into a test run, even if the
# developer's shell has one exported. `PINECONE_API_KEY` is blanked for the same reason as the
# AWS keys — a test that accidentally constructed a real client would otherwise write test
# vectors into the production index, and Pinecone upserts have no dry-run mode.
os.environ["AWS_ACCESS_KEY_ID"] = ""
os.environ["AWS_SECRET_ACCESS_KEY"] = ""
os.environ["AWS_SESSION_TOKEN"] = ""
os.environ["PINECONE_API_KEY"] = ""
