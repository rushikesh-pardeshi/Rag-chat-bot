"""backend/tests/regression/conftest.py

Role: Fixtures and shared fixture data for the business-requirement regression suites, plus
the requirement-to-module map these suites are answerable to.

What a regression test is here, and what it is not
-------------------------------------------------
The unit suites (`backend/tests/test_*.py`) assert how one service behaves when called
directly. The security suite asserts that a rule cannot be bypassed. These modules assert the
third thing: that a requirement from the problem statement is *still satisfied over HTTP*, by
the shipped routes, DTOs, dependencies and middleware. So almost every test here drives the
real app through `httpx` rather than calling a service — an answer that is correct in
`QueryService` but loses its citations in `QueryResponse` is a BR-03 regression, and only a
test that goes through the DTO can see it.

On the requirement numbering — read this before adding a module
--------------------------------------------------------------
`WORKBREAKDOWN.md`'s Phase 6 gate says "Regression suites pass for BR-01 to BR-09", and
`.claude/agents/frontend-dashboard-agent.md` also refers to "BR-01 to BR-09". **Only BR-01
through BR-06 are actually numbered anywhere in this workspace.** The remaining requirement
families in `WORKBREAKDOWN.md` § "Document requirements covered" are named rather than
numbered: Analytics & Reporting is `BR-AN`, and Performance, Security, Compliance, Usability,
Data Integrity, Scalability and Query Accuracy carry NFR/C/R references instead.

Rather than invent BR-07, BR-08 and BR-09 — a mapping nobody could check against a source
document — `BR_COVERAGE` below keys the families by the identifier the repository actually
uses, and the two families beyond the six numbered ones are covered by *name*. If a numbered
BR-07..BR-09 mapping is ever agreed, it belongs in `WORKBREAKDOWN.md` first and here second.
`test_coverage_map.py` keeps this table honest: every module it names has to exist.

Where the harness lives
-----------------------
`backend/tests/harness.py`, shared with `backend/tests/security/`. See its docstring for why
the stack is two tenants and four users, and why the app is overridden at the leaves rather
than mocked.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Final

import httpx
import pytest

from backend.models.user import User
from backend.tests.document_fixtures import make_docx, make_pdf
from backend.tests.harness import (
    API,
    CONTEXT,
    ROOT,
    Api,
    Stack,
    Tenant,
    backend_modules,
    build_api,
    build_stack,
    csrf_headers,
)

__all__ = [
    "API",
    "BR_COVERAGE",
    "CONTEXT",
    "FEE_DOCX",
    "FEE_PAGES",
    "FEE_PDF",
    "REVISED_FEE_PAGES",
    "REVISED_FEE_PDF",
    "MAINTENANCE_PAGE",
    "OFF_CORPUS_QUESTION",
    "ROOT",
    "TAX_QUESTION",
    "WIRE_PAGE",
    "WIRE_QUESTION",
    "Api",
    "Stack",
    "Tenant",
    "backend_modules",
    "csrf_headers",
    "upload",
]

#: Requirement family → (identifier used in the repository, module that covers it).
#: Deliberately keyed by name, because the name is what `WORKBREAKDOWN.md` § "Document
#: requirements covered" actually states. See the module docstring on BR-07..BR-09.
BR_COVERAGE: Final[dict[str, tuple[str, str]]] = {
    "Document Management": ("BR-01", "test_document_management.py"),
    "AI Query Engine": ("BR-02", "test_ai_query.py"),
    "Source Traceability": ("BR-03", "test_ai_query.py"),
    "Search History": ("BR-04", "test_search_history.py"),
    "Audit Logging": ("BR-05", "test_audit_logs.py"),
    "Role-Based Access Control": ("BR-06", "test_rbac.py"),
    "Analytics & Reporting": ("BR-AN", "test_analytics.py"),
    "Data Integrity": ("C-04/R-01/R-09", "test_document_management.py"),
    "Query Accuracy": ("R-04", "test_ai_query.py"),
}


# -- Fixture documents -------------------------------------------------------
#: Four pages, all about fees or tax, so a citation that names the right document and the wrong
#: page still fails. Naming the document is the weak claim; BR-03 asks for the location.
FEE_PAGES: Final = [
    "The outbound domestic wire transfer fee is 25 USD for every retail payment "
    "instruction, debited from the originating account on the day of submission.",
    "The monthly account maintenance fee is 12 USD and is waived for any month in which "
    "the average daily balance stays above 5,000 USD.",
    "Withholding tax on interest credited to a resident savings account is deducted at "
    "10 percent unless a valid exemption declaration is on file.",
    "Cash withdrawals at an out-of-network ATM carry a 3 USD surcharge in addition to any "
    "amount the operator of that machine chooses to levy.",
]
FEE_PDF: Final = make_pdf(FEE_PAGES)

#: The same document after a price revision. Used where a *second version of the same name* is
#: needed: it has to be a valid PDF whose bytes differ, because re-uploading identical bytes
#: exercises the duplicate-content path instead and re-uploading a DOCX under a `.pdf` name is
#: correctly refused by the magic-byte check.
REVISED_FEE_PAGES: Final = [
    FEE_PAGES[0].replace("25 USD", "30 USD"),
    *FEE_PAGES[1:],
]
REVISED_FEE_PDF: Final = make_pdf(REVISED_FEE_PAGES)

#: The same content as a DOCX, so BR-01's "PDF or DOCX" is asserted on both formats rather
#: than on the one that happens to be easier to build.
FEE_DOCX: Final = make_docx(
    [("Retail fees", FEE_PAGES[0]), ("Maintenance", FEE_PAGES[1])],
    title="Retail Account Fees",
    author="Treasury Operations",
)

WIRE_QUESTION: Final = (
    "What is the outbound domestic wire transfer fee for a retail payment instruction?"
)
#: Page 1 of the fixture. `make_pdf` numbers from 1 in list order and the splitter propagates
#: that page onto every chunk it produces.
WIRE_PAGE: Final = "page 1"
MAINTENANCE_PAGE: Final = "page 2"

#: Banking-domain vocabulary that appears on page 3 and nowhere else — the BR-02
#: "domain-specific terminology interpreted accurately" case.
TAX_QUESTION: Final = (
    "At what rate is withholding tax deducted on interest credited to a resident "
    "savings account?"
)

#: Answerable from general knowledge and from nothing in the corpus. The grounding rule has to
#: decline it rather than answer it, which is the Data Integrity requirement (C-04, R-01).
OFF_CORPUS_QUESTION: Final = (
    "Which national football team won the 1994 FIFA World Cup final?"
)


# -- Fixtures ----------------------------------------------------------------
@pytest.fixture
async def stack() -> Stack:
    """Two populated tenants over the fakes. See `harness.build_stack`."""
    return await build_stack()


@pytest.fixture
async def api(stack: Stack) -> AsyncIterator[Api]:
    """The real app over `stack`, plus an `as_user` switch. See `harness.build_api`."""
    async with build_api(stack) as client_and_switch:
        yield client_and_switch


# -- Helpers -----------------------------------------------------------------
async def upload(
    client: httpx.AsyncClient,
    *,
    filename: str = "retail-account-fees.pdf",
    data: bytes = FEE_PDF,
    content_type: str = "application/pdf",
) -> httpx.Response:
    """`POST /api/v1/documents` as whoever `as_user` last selected.

    A helper rather than a fixture because half these tests care about the *response* — the
    403 a Standard User gets, the 413 an oversized file gets — and a fixture that asserted
    success would hide them.
    """
    return await client.post(
        f"{API}/documents",
        files={"file": (filename, data, content_type)},
        headers=csrf_headers(),
    )


def source_names(body: dict[str, Any]) -> list[str]:
    return [ref["source_document_name"] for ref in body["source_references"]]


def citations(body: dict[str, Any]) -> list[str]:
    return [ref["section_or_page"] for ref in body["source_references"]]


def audit_rows(stack: Stack, action: str, *, actor_email: str | None = None) -> list[Any]:
    """Every audit entry with this action value, oldest first.

    `actor_email` matters more than it looks. A single upload produces *two*
    `DOCUMENT_UPLOAD` rows: one attributed to the Administrator who made the request, and one
    written by the ingestion pipeline under `system@docquery.ingestion`, which records the
    parse phase. Picking "the last row" would silently assert against whichever of the two
    happens to be appended second, so tests that care about a specific row name its actor.
    """
    rows = [entry for entry in stack.audit_repo.entries if entry.action.value == action]
    if actor_email is not None:
        rows = [entry for entry in rows if entry.actor_email == actor_email]
    return rows


def switch_to(api: Api, user: User | None) -> httpx.AsyncClient:
    """Authenticate subsequent requests as `user` and hand back the client."""
    client, as_user = api
    as_user(user)
    return client


def new_uuid() -> uuid.UUID:
    """An id that belongs to nobody. Named so the intent is legible at the call site."""
    return uuid.uuid4()


Switch = Callable[[User | None], None]
