"""backend/tests/eval/questions.py

Role: The question/expected-source pairs the accuracy figure is computed from. Thirty questions
over the twenty-eight passages in `corpus.py`, each with the document and page that answers it.

The expected source is a *page*, not a document. "Cited the right document" is a much weaker
claim than BR-03 makes: `retail-account-fees-2026.pdf` has five pages pricing five different
events, and an answer citing page 4 for a question about wire fees sends a compliance
reviewer to the wrong paragraph. `section_or_page` exists precisely so that cannot happen,
so the harness scores it.

On wording: questions reuse the *inflected forms* that appear in the passages ("accrues", not
"accrue"; "postings", not "posting"). That is not how a user writes, and it is a concession to
the stand-in embedder — a hashed bag of words has no morphology, so "require" and "requires"
are unrelated tokens to it where Titan treats them as near-identical. The concession is
recorded here rather than hidden, and `test_eval_retrieval.py` states what it costs the
figure's meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One question and the single passage that should be cited for it."""

    question: str
    document: str
    page: int

    @property
    def section_or_page(self) -> str:
        """The citation string `source_tracer` builds for this page."""
        return f"page {self.page}"

    def __str__(self) -> str:  # pragma: no cover - report formatting only
        return f"{self.document} {self.section_or_page}"


FEES: Final = "retail-account-fees-2026.pdf"
OVERDRAFT: Final = "overdraft-and-interest-policy.pdf"
DORMANCY: Final = "dormant-account-procedure.pdf"
KYC: Final = "kyc-onboarding-standard.pdf"
LOANS: Final = "loan-underwriting-guidelines.pdf"
CUTOFFS: Final = "wire-transfer-cutoffs.pdf"

CASES: Final[tuple[EvalCase, ...]] = (
    # -- Fee schedule --------------------------------------------------------
    EvalCase("What is the fee for sending a domestic wire transfer?", FEES, 1),
    EvalCase(
        "How much is charged for an international wire, and are correspondent "
        "charges included?",
        FEES,
        2,
    ),
    EvalCase("When is the monthly account maintenance fee waived?", FEES, 3),
    EvalCase("Is there a surcharge for cash withdrawals at an out-of-network ATM?", FEES, 4),
    EvalCase("What does a mailed paper statement cost?", FEES, 5),
    EvalCase(
        "Does the bank absorb correspondent bank charges applied in the receiving country?",
        FEES,
        2,
    ),
    # -- Overdraft and interest ---------------------------------------------
    EvalCase("What is the daily overdraft interest rate on a negative balance?", OVERDRAFT, 1),
    EvalCase(
        "How much is an overdraft protection transfer from a linked savings account?",
        OVERDRAFT,
        2,
    ),
    EvalCase(
        "What fee applies to a returned item with non-sufficient funds?", OVERDRAFT, 3
    ),
    EvalCase(
        "Is there a grace period before an overdraft charge is assessed?", OVERDRAFT, 4
    ),
    EvalCase(
        "What happens if an account stays overdrawn for five consecutive business days?",
        OVERDRAFT,
        5,
    ),
    # -- Dormancy ------------------------------------------------------------
    EvalCase(
        "After how long with no customer initiated activity is an account classified "
        "as dormant?",
        DORMANCY,
        1,
    ),
    EvalCase("Do interest postings count as customer initiated activity?", DORMANCY, 1),
    EvalCase("When are funds escheated to the state as unclaimed property?", DORMANCY, 2),
    EvalCase(
        "Does reactivating a dormant account need a branch visit and photographic "
        "identification?",
        DORMANCY,
        3,
    ),
    EvalCase(
        "How long before classification is a dormancy notice mailed?", DORMANCY, 4
    ),
    # -- KYC -----------------------------------------------------------------
    EvalCase(
        "Is a driving licence acceptable proof of identity at onboarding?", KYC, 1
    ),
    EvalCase("How recent must proof of residential address be?", KYC, 2),
    EvalCase(
        "When is enhanced due diligence mandatory for a politically exposed person?",
        KYC,
        3,
    ),
    EvalCase(
        "What percentage of shares or voting rights makes a natural person a "
        "beneficial owner?",
        KYC,
        4,
    ),
    EvalCase("How often are high risk customers reviewed?", KYC, 5),
    # -- Lending -------------------------------------------------------------
    EvalCase(
        "What is the maximum debt to income ratio on a new consumer facility?", LOANS, 1
    ),
    EvalCase(
        "What minimum credit score does unsecured personal lending need?", LOANS, 2
    ),
    EvalCase(
        "What is the loan to value cap on residential mortgage lending?", LOANS, 3
    ),
    EvalCase("What income evidence must a self-employed applicant supply?", LOANS, 4),
    EvalCase("How often is commercial collateral revalued?", LOANS, 5),
    # -- Cutoff times (the distractors for every wire question above) -------
    EvalCase(
        "What is the cutoff time for same day domestic wire processing?", CUTOFFS, 1
    ),
    EvalCase(
        "By what time must international wire instructions reach the payments team?",
        CUTOFFS,
        2,
    ),
    EvalCase(
        "What happens to a wire instruction submitted after its cutoff?", CUTOFFS, 3
    ),
    EvalCase(
        "Are wire instructions received on a public holiday held until the following "
        "business day?",
        CUTOFFS,
        4,
    ),
)

#: Questions the corpus does not answer. These are scored separately, as a refusal rate: a
#: harness that only measured hits would rate a system answering everything confidently as
#: perfect. Under `rag-answer-grounding.md` the required output here is the refusal.
UNANSWERABLE: Final[tuple[str, ...]] = (
    "What is the current mortgage rate for a 30 year fixed loan in Ohio?",
    "Which branch has the shortest queue on a Friday afternoon?",
    "How do I dispute a credit card chargeback filed against a merchant?",
    "What is the bank's cryptocurrency custody policy?",
)
