"""backend/tests/eval/corpus.py

Role: The fixture corpus the accuracy figure is measured against — six documents, one passage
per page, written the way a retail bank's internal policy pages actually read.

Two properties are deliberate and load-bearing:

**Overlapping subject matter.** `retail-account-fees-2026.pdf` and `wire-transfer-cutoffs.pdf`
both talk about domestic and international wires; `overdraft-and-interest-policy.pdf` and
the fee schedule both price account events. A corpus of six documents on six unrelated
topics would score near-perfectly on any retrieval method at all, including a keyword
`LIKE` query, and would therefore measure nothing. The distractors are what make the
number mean something.

**No real customer data.** Every figure here is invented. `secure-coding-vapt.md` keeps account
numbers, balances and PII out of the repository, and a fixture corpus checked into git is
exactly the place that rule gets broken by accident.

One passage per page keeps `section_or_page` unambiguous: `Corpus.add` numbers pages from 1 in
list order, so the expected citation for `PASSAGES[name][i]` is always `f"page {i + 1}"`. A
multi-passage page would make an expected-source assertion depend on the chunker's boundary
choices, which are tested in `test_chunk_splitter.py` and are not what this harness is for.
"""

from __future__ import annotations

from typing import Final

#: document name -> passages, one per page, in page order.
PASSAGES: Final[dict[str, list[str]]] = {
    "retail-account-fees-2026.pdf": [
        # page 1
        "The outbound domestic wire transfer fee is 25 USD for every retail payment "
        "instruction, debited from the originating account on the day of submission.",
        # page 2
        "An outbound international wire transfer costs 45 USD plus any correspondent bank "
        "charges applied in the receiving country, which the bank does not absorb.",
        # page 3
        "The monthly account maintenance fee is 12 USD and is waived for any month in which "
        "the average daily balance stays above 5,000 USD.",
        # page 4
        "Cash withdrawals at an out-of-network ATM carry a 3 USD surcharge in addition to any "
        "amount the operator of that machine chooses to levy.",
        # page 5
        "Customers who continue to receive a mailed paper statement are charged 2 USD per "
        "statement; switching to electronic delivery removes the charge entirely.",
    ],
    "overdraft-and-interest-policy.pdf": [
        # page 1
        "Overdraft interest accrues daily at 14 percent per annum on the negative balance for "
        "as long as the account remains overdrawn.",
        # page 2
        "Each overdraft protection transfer from a linked savings account is charged at 10 "
        "USD, regardless of the amount moved to cover the shortfall.",
        # page 3
        "A returned item, sometimes called a non-sufficient funds item, attracts a 30 USD fee "
        "per presentment, and a single item may be presented more than once.",
        # page 4
        "No overdraft charge is assessed if the account is brought back to a positive balance "
        "within the 24 hour grace period that begins when the shortfall occurs.",
        # page 5
        "An account that stays overdrawn for five consecutive business days incurs a "
        "sustained overdraft charge of 20 USD, repeated weekly until the balance is restored.",
    ],
    "dormant-account-procedure.pdf": [
        # page 1
        "An account is classified as dormant once 24 months have passed with no customer "
        "initiated activity, where interest postings do not count as activity.",
        # page 2
        "Funds in a dormant account are escheated to the state after five years without "
        "contact from the owner, and are recoverable only through the state unclaimed "
        "property office.",
        # page 3
        "Reactivating a dormant account requires the owner to attend a branch in person and "
        "present photographic identification; it cannot be done through the mobile channel.",
        # page 4
        "A dormancy notice is mailed to the address of record 60 days before the account is "
        "classified, giving the owner an opportunity to prevent the classification.",
    ],
    "kyc-onboarding-standard.pdf": [
        # page 1
        "Acceptable proof of identity at onboarding is an unexpired passport or a national "
        "identity card; a driving licence alone is not sufficient for a new relationship.",
        # page 2
        "Proof of residential address must be a utility bill or bank statement dated within "
        "the previous three months, and a mobile telephone bill is not accepted.",
        # page 3
        "Enhanced due diligence is mandatory for any politically exposed person, including "
        "immediate family members and known close associates of that person.",
        # page 4
        "A beneficial owner is any natural person holding 25 percent or more of the shares or "
        "voting rights of a corporate customer, directly or through an intermediary.",
        # page 5
        "Customers rated high risk are subject to a periodic relationship review every 12 "
        "months; standard risk customers are reviewed every 36 months.",
    ],
    "loan-underwriting-guidelines.pdf": [
        # page 1
        "The maximum total debt to income ratio accepted on a new consumer facility is 43 "
        "percent, calculated on gross monthly income before deductions.",
        # page 2
        "Unsecured personal lending requires a minimum credit score of 660; applications "
        "below that score are declined without manual referral.",
        # page 3
        "Residential mortgage lending is capped at a loan to value of 80 percent, and "
        "anything above that requires mortgage insurance for the excess portion.",
        # page 4
        "A self-employed applicant must supply filed tax returns for the two most recent "
        "years, because bank statements alone do not evidence sustainable income.",
        # page 5
        "Commercial collateral is revalued every 12 months, and a valuation more than 18 "
        "months old may not be relied upon in a credit decision.",
    ],
    "wire-transfer-cutoffs.pdf": [
        # page 1
        "The cutoff time for same day domestic wire processing is 16:30 local branch time; "
        "the cutoff is a submission deadline and not a guarantee of settlement.",
        # page 2
        "International wire instructions must reach the payments team by 14:00 local time "
        "to be released in the current value date window.",
        # page 3
        "Any wire instruction submitted after its cutoff is queued and released on the next "
        "business day, retaining its original submission reference.",
        # page 4
        "Wire instructions received on a weekend or a public holiday are held without "
        "processing until the following business day opens.",
    ],
}


def total_passages() -> int:
    """How many chunk rows the harness stages. Reported so the corpus size is on the record."""
    return sum(len(pages) for pages in PASSAGES.values())
