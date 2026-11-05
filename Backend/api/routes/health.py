"""backend/api/routes/health.py

Role: Liveness/readiness endpoints used by deploy probes and smoke tests.

`/health` answers without touching a dependency — it says the process is up.
`/readiness` proves the database is reachable, so a probe can tell "starting" from
"serving". Neither route requires authentication, and neither leaks version detail
beyond the application version already visible in the OpenAPI document.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from backend.api.dependencies.database import DbSession
from backend.core.config import get_settings

router = APIRouter(tags=["health"])


def _now() -> str:
    return datetime.now(UTC).isoformat()


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "healthy",
        "version": settings.APP_VERSION,
        "timestamp": _now(),
    }


@router.get("/readiness", summary="Readiness probe")
async def readiness(db: DbSession, response: Response) -> dict[str, object]:
    """503 until the database answers, so traffic is not routed to a dead instance."""
    checks: dict[str, str] = {}
    try:
        await db.execute(text("SELECT 1"))
        checks["database"] = "ok"
        ready = True
    except Exception:
        # The reason belongs in the log, not in the probe response.
        checks["database"] = "unavailable"
        ready = False

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": "ready" if ready else "not_ready",
        "version": get_settings().APP_VERSION,
        "timestamp": _now(),
        "checks": checks,
    }
