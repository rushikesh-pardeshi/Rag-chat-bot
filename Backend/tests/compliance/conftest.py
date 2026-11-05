"""backend/tests/compliance/conftest.py

Role: Fixtures and fixture documents for the compliance suite.

Thin on purpose. The stack itself is `backend/tests/harness.py`, shared with the security and
regression packages; this file adds only what a compliance check needs on top of it — a corpus
whose pages are distinguishable, and the two helpers every module here uses.

Why this package does not import the regression package's fixtures
------------------------------------------------------------------
It would work, and `FEE_PDF` would be defined once instead of twice. But it would also make the
regression suite load-bearing for the compliance suite: deleting a BR test could break the
traceability proof, and a reviewer reading a compliance failure would have to go looking in
another package to find out what the fixture was. The harness is the shared thing, by design;
the fixture data is deliberately local. `harness.py`'s own docstring records the same reasoning
for why it is not a `conftest.py`.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any, Final

import httpx
import pytest

from backend.models.user import User
from backend.tests.document_fixtures import make_pdf
from backend.tests.harness import (
    API,
    CONTEXT,
    Api,
    Stack,
    build_api,
    build_stack,
    csrf_headers,
)

__all__ = [
    "API",
    "CONTEXT",
    "PAGES",
    "SURCHARGE_QUESTION",
    "TAX_QUESTION",
    "UNANSWERABLE_QUESTION",
    "WIRE_QUESTION",
    "Api",
    "Stack",
    "csrf_headers",
    "switch_to",
    "upload",
]

#: Four pages, each about a different charge. Distinct subjects matter more here than anywhere
#: else in the suite: a traceability check asserts the citation names the page the answer
#: actually came from, and four paraphrases of one fact would let a wrong page number pass.
PAGES: Final = [
    "The outbound domestic wire transfer fee is 25 USD for every retail payment "
    "instruction, debited from the originating account on the day of submission.",
    "The monthly account maintenance fee is 12 USD and is waived for any month in which "
    "the average daily balance stays above 5,000 USD.",
    "Withholding tax on interest credited to a resident savings account is deducted at "
    "10 percent unless a valid exemption declaration is on file.",
    "Cash withdrawals at an out-of-network ATM carry a 3 USD surcharge in addition to any "
    "amount the operator of that machine chooses to levy.",
]
FEE_PDF: Final = make_pdf(PAGES)

WIRE_QUESTION: Final = (
    "What is the outbound domestic wire transfer fee for a retail payment instruction?"
)
TAX_QUESTION: Final = (
    "At what rate is withholding tax deducted on interest credited to a resident "
    "savings account?"
)
SURCHARGE_QUESTION: Final = (
    "What surcharge applies to a cash withdrawal at an out-of-network ATM?"
)

#: Answerable from general knowledge and from nothing in this corpus. Used where the required
#: outcome is a refusal — `rag-answer-grounding.md`'s "Information not found in uploaded
#: documents" — rather than an answer.
UNANSWERABLE_QUESTION: Final = (
    "Which national football team won the 1994 FIFA World Cup final?"
)


@pytest.fixture
async def stack() -> Stack:
    """Two populated tenants over the fakes. See `harness.build_stack`."""
    return await build_stack()


@pytest.fixture
async def api(stack: Stack) -> AsyncIterator[Api]:
    """The real app over `stack`, plus an `as_user` switch. See `harness.build_api`."""
    async with build_api(stack) as client_and_switch:
        yield client_and_switch


async def upload(
    client: httpx.AsyncClient,
    *,
    filename: str = "retail-account-fees.pdf",
    data: bytes = FEE_PDF,
    content_type: str = "application/pdf",
) -> dict[str, Any]:
    """Upload through the real route and return the document DTO.

    Unlike the regression package's helper this one asserts success and unwraps the body. A
    compliance module is never *about* the upload response — an upload that failed would make
    every assertion after it vacuous, so failing here is the right place to notice.
    """
    response = await client.post(
        f"{API}/documents",
        files={"file": (filename, data, content_type)},
        headers=csrf_headers(),
    )
    assert response.status_code in (200, 201, 202), response.text
    return response.json()


async def ask(
    client: httpx.AsyncClient, stack: Stack, question: str, answer: str
) -> dict[str, Any]:
    """Submit one question with a queued model answer, and return the response DTO.

    `answer` is what `FakeBedrockClient` will produce, so a test states the model's behaviour
    explicitly — including its citation markers, which is what `source_tracer` reads back. A
    marker referring to a passage that was never retrieved is how the "invented citation" cases
    are set up.
    """
    stack.bedrock.queue_answer(answer)
    response = await client.post(
        f"{API}/queries", json={"question_text": question}, headers=csrf_headers()
    )
    # 202: the route accepts the question and the answer is attached before it returns. The
    # status describes the contract, not the timing — see `routes/queries.py`.
    assert response.status_code == 202, response.text
    return response.json()


def switch_to(api: Api, user: User | None) -> httpx.AsyncClient:
    """Authenticate subsequent requests as `user` and hand back the client."""
    client, as_user = api
    as_user(user)
    return client


def new_uuid() -> uuid.UUID:
    """An id that belongs to nobody. Named so the intent is legible at the call site."""
    return uuid.uuid4()
