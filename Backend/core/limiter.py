"""backend/core/limiter.py

Role: The process-wide slowapi limiter.

It lives here rather than in `backend/api/app.py` because route modules need the object to
decorate their handlers, and `app.py` imports those route modules — putting it there would
be a circular import.

Built at import time on purpose: slowapi resolves its storage backend inside `__init__`, so
assigning `storage_uri` after construction leaves the limiter silently on in-memory storage
while looking configured. Storage is per process by default; `RATE_LIMIT_STORAGE_URI=redis://…`
makes the counters shared, which is what a multi-worker deployment needs for a per-IP limit
to mean anything at all.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.core.config import get_settings

#: Keys on the peer address. Behind a proxy, run uvicorn with
#: `--proxy-headers --forwarded-allow-ips=<proxy>` so this is the real client and not the
#: proxy — otherwise every user shares one bucket.
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=get_settings().RATE_LIMIT_STORAGE_URI,
    # Emits `Retry-After` and the `X-RateLimit-*` family, so a client can back off
    # correctly instead of hammering.
    headers_enabled=True,
    retry_after="delta-seconds",
    key_style="endpoint",
)
