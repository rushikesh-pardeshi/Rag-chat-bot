"""backend/core/logging.py

Role: Structured JSON logging with request_id / user_id / action, and the sanitizers that
keep untrusted input and secrets out of the log stream.

Two controls live here, both required by .claude/rules/secure-coding-vapt.md:
  * log injection — CR/LF and control characters are stripped from every value, so a
    crafted document name or question cannot forge a second log line.
  * secrets/PII — known-sensitive keys are redacted before rendering.
"""

from __future__ import annotations

import logging
import re
import sys
from typing import Any

import structlog

from backend.core.config import get_settings

#: Anything matching these key names is replaced wholesale, never truncated.
#:
#: The point of this guard is to catch a *careless future log call*, not the ones written so
#: far — nothing in the codebase logs a credential today. That is the standard it has to be
#: judged against, and two alternatives were added after it failed that standard:
#:   * `access[_-]?key` — `api[_-]?key` did not match `aws_access_key_id`, the literal name of
#:     one of this project's own `Settings` fields. A guard that misses the credential we
#:     ourselves hold is not a guard.
#:   * `credential|bearer|jwt|private[_-]?key` — the other names a session token travels under.
#:
#: `pan` and `card` are anchored to token boundaries because unanchored they matched `company`,
#: `span_count`, `panel` and `expanded_pages`. Over-redaction is not a breach, but it silently
#: replaces operational counters with `[REDACTED]` and an operator cannot tell a masked field
#: from a missing one. Snake_case is this codebase's convention, so `card_number` and
#: `pan_number` still match; a hypothetical single-word `cardholder` would not.
#:
#: `email` is deliberately absent. `auth.login_failed_unknown_account` logs the attempted
#: address on purpose — `rbac-enforcement.md` requires denied access attempts to be logged, and
#: an attempt with no subject is not an audit record. That is a considered PII trade-off rather
#: than an oversight; see `backend/tests/security/test_secrets_in_logs.py`.
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(password|passwd|secret|token|authorization|api[_-]?key|access[_-]?key"
    r"|private[_-]?key|credential|bearer|jwt|cookie|session"
    r"|account[_-]?number|ssn|aadhaar"
    r"|(?:^|[_\-.])(?:pan|card)(?:$|[_\-.]))",
    re.IGNORECASE,
)

#: C0/C1 control characters, including CR and LF. Kept explicit rather than \s so that
#: ordinary spaces survive.
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f]")

REDACTED = "[REDACTED]"

#: Log values are truncated so a 50MB document name cannot flood the log.
MAX_VALUE_LENGTH = 2048

#: Libraries whose DEBUG level dumps raw HTTP traffic, floored at INFO no matter what
#: `LOG_LEVEL` says.
#:
#: `_redact_and_sanitize` redacts by *key name*, and botocore puts an entire response body
#: into `event` — a key that matches nothing. With `LOG_LEVEL=DEBUG` that printed a full
#: STS session token (`<Token-0>…`) and every signed request header into the log, which is
#: exactly what secure-coding-vapt.md's Secrets/PII rule forbids. Redaction cannot fix it
#: after the fact: the secret is embedded in prose. The only reliable control is not asking
#: these libraries for wire-level logging.
#:
#: Raising one of these to DEBUG while chasing a bug means accepting credentials in the log
#: stream — do it in a scratch process, never in a deployed one.
_WIRE_LEVEL_LOGGERS = (
    "boto3",
    "botocore",
    "s3transfer",
    "urllib3",
    "httpx",
    "httpcore",
    "pinecone",
    "pinecone_plugin_interface",
)


def sanitize_log_value(value: Any, *, max_length: int = MAX_VALUE_LENGTH) -> Any:
    """Flatten a value to something safe to write on one log line.

    Strings lose control characters (CR/LF included) and are length-capped. Containers
    are walked recursively, and a sensitive key is redacted at any depth. Non-string scalars
    pass through untouched.

    The nested redaction matters because the deepest user-influenced structure in this
    application is the audit `detail` dict, and `_redact_and_sanitize` inspects only top-level
    keys. Without this branch, `logger.info("x", detail={"authorization": token})` sanitized
    the token and logged it — the guard sat one nesting level above where the data actually is.
    """
    if isinstance(value, str):
        cleaned = _CONTROL_CHARS.sub(" ", value)
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "…[truncated]"
        return cleaned
    if isinstance(value, dict):
        return {
            k: REDACTED
            if isinstance(k, str) and _SENSITIVE_KEY_PATTERN.search(k)
            else sanitize_log_value(v, max_length=max_length)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_log_value(v, max_length=max_length) for v in value]
    return value


def _redact_and_sanitize(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """structlog processor: redact sensitive keys, then sanitize everything else."""
    for key in list(event_dict):
        if _SENSITIVE_KEY_PATTERN.search(key):
            event_dict[key] = REDACTED
        else:
            event_dict[key] = sanitize_log_value(event_dict[key])
    return event_dict


#: Processors applied to structlog calls *and*, via `foreign_pre_chain`, to plain stdlib
#: records from third-party libraries. Both paths must be sanitized: a library that logs a
#: user-supplied value is as good an injection vector as our own code.
_SHARED_PROCESSORS: list[Any] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    _redact_and_sanitize,
]


def configure_logging() -> None:
    """Install the JSON renderer and processor chain. Idempotent."""
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL)

    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            # Rendering happens in the stdlib formatter below, so structlog and uvicorn
            # records come out of the same renderer instead of two competing ones.
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=_SHARED_PROCESSORS,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
    )

    root = logging.getLogger()
    root.handlers.clear()  # idempotent: a second call must not double every line
    root.addHandler(handler)
    root.setLevel(level)

    # uvicorn installs its own plain-text handlers. Clear them and let the records reach
    # the JSON handler above, or half the stream is unparseable.
    for name in ("uvicorn", "uvicorn.error", "fastapi", "asyncio", "watchfiles.main"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    # Floor the wire-level libraries at INFO. `setLevel` on the library's own logger wins
    # over the root level, so this holds even with LOG_LEVEL=DEBUG.
    for name in _WIRE_LEVEL_LOGGERS:
        logging.getLogger(name).setLevel(max(level, logging.INFO))

    # uvicorn.access is switched off outright: the request-id middleware already logs every
    # request with the fields that matter, and uvicorn's version carries no request_id, so
    # keeping it would mean two lines per request and one of them untraceable.
    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.disabled = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_request_context(
    *, request_id: str, user_id: str | None = None, action: str | None = None
) -> None:
    """Attach request-scoped fields so every later log line carries them."""
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        user_id=user_id,
        action=action,
    )


def bind_actor_context(*, user_id: str, tenant_id: str) -> None:
    """Add the resolved actor to the already-bound request context.

    Separate from `bind_request_context` because the actor is not known when the request
    id is assigned — authentication happens later, in a dependency. Rebinding through the
    other function would require re-supplying `request_id` and risk clearing it.

    Only ids are bound. The email is deliberately not: a log aggregator holding every
    request line should not also hold a directory of user identities.
    """
    structlog.contextvars.bind_contextvars(user_id=user_id, tenant_id=tenant_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()
