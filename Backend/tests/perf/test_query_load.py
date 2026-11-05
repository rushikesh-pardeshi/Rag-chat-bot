"""backend/tests/perf/test_query_load.py

Role: Drive `POST /api/v1/queries` at concurrency through the real app, report p50/p95 and the
error rate, and assert the things that concurrency actually threatens.

READ THIS BEFORE QUOTING THE NUMBER
-----------------------------------
The p95 printed here is **not** the 3–5 second gate. It excludes every network round trip a
real request makes — the question embedding, the Pinecone query and the answer generation —
because all three are faked. What remains is the app's own cost: routing, CSRF and role checks,
PostgreSQL re-check of every match, `source_tracer`, the audit write and DTO serialization.
Quoted alone it would read as a claim that DocQuery answers in single-digit milliseconds, which
is false.

The gate's number is `AnalyticsRepository.summary().p95_latency_ms` — `percentile_cont(0.95)`
over `query.latency_ms` — measured against real traffic through real Bedrock and Pinecone.
`docs/RELEASE.md` is where that figure and this one are recorded side by side, each labelled.

What this run *does* establish
------------------------------
Concurrency is the interesting variable, not speed. Sixty-four overlapping requests share one
`RagService`, one `VectorStore`, one `SourceTracer` and one `AuditService`, and any per-request
state accidentally held on one of those objects shows up here and almost nowhere else:

  * every request gets its own answer and its own citations — no cross-wiring;
  * no request fails, and none is downgraded to not-found because another consumed its
    retrieval;
  * `latency_ms` is recorded on every query row, which is what the production percentile reads;
  * the app's own overhead is bounded, so a future edit that adds a synchronous round trip to
    the query path fails here rather than in production.

Two things it deliberately does not assert
-----------------------------------------
**Audit chain integrity under concurrency.** Two appends that read the same chain head fork the
chain. The control is `AuditRepository.lock_chain`'s advisory lock, which is a PostgreSQL
feature; `FakeAuditRepository` has no lock, so a fork here would be a property of the fake.
The chain is verified against a database that has the lock in
`backend/tests/compliance/test_audit_immutability_db.py`.

**Throughput.** These requests are I/O-bound on fakes that return immediately, so the run
measures how fast Python executes the app, not how many queries a deployment serves. A
throughput figure taken from here would be meaningless.

One more caveat on the wall-clock column: a round is sixteen coroutines on one event loop, so
each sample includes the time its request sat behind the other fifteen. It is closer to a batch
completion time than to a per-request service time, which is the other reason it cannot stand
in for the gate.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Final

import pytest

from backend.core.config import Settings, get_settings
from backend.models.query import QueryStatus
from backend.tests.document_fixtures import make_pdf
from backend.tests.harness import API, build_api, build_stack, csrf_headers

#: Overlapping requests per round, and how many rounds. 4 x 16 = 64 samples — enough for a p95
#: to mean something (the 95th of 64 is interpolated between the 60th and 61st sample) without
#: making the suite slow enough that someone starts skipping it.
CONCURRENCY: Final = 16
ROUNDS: Final = 4
TOTAL: Final = CONCURRENCY * ROUNDS

#: A tripwire on the app's own cost, not the gate. Generous by two orders of magnitude against
#: the measured figure on a developer laptop, because a CI box under load is slower and a
#: flaky perf assertion gets deleted rather than investigated. It still catches the change this
#: is here to catch: a synchronous round trip added to the query path moves this by hundreds of
#: milliseconds, not by a few.
MAX_OVERHEAD_P95_MS: Final = 750.0

#: The same four-page fixture the compliance suite uses, duplicated for the reason its
#: `conftest.py` gives: a perf module should not break because a compliance fixture was
#: reworded. Page 3 is the one queried — the harness seeds every tenant with a wire-fee
#: passage of its own, so a wire-fee question has two plausible sources and a citation
#: assertion could fail for a ranking reason that has nothing to do with load.
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
QUESTION: Final = (
    "At what rate is withholding tax deducted on interest credited to a resident "
    "savings account?"
)
#: What the fake model returns for every request. One citation marker, so each response must
#: carry exactly one reference — a request that came back with two picked up another's passage.
ANSWER: Final = "Interest is taxed at 10 percent [S1]."


@dataclass(frozen=True, slots=True)
class Sample:
    """One request: what the caller saw, and what the service recorded for itself."""

    status_code: int
    #: Client-side wall clock, so it includes the route, both dependencies and serialization.
    wall_ms: float
    #: `latency_ms` from the response body — the service's own measurement, and the column the
    #: production percentile is computed over. Recording both is the point: a gap between them
    #: is time spent outside the measured span.
    service_ms: int | None
    status: str | None
    document_ids: tuple[str, ...]
    citations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.status_code == 202


@dataclass
class LoadReport:
    """The measured figures, and the text that gets pasted into `docs/RELEASE.md`."""

    samples: list[Sample] = field(default_factory=list)
    document_id: str = ""
    expected_citation: str = ""

    @property
    def errors(self) -> list[Sample]:
        return [sample for sample in self.samples if not sample.ok]

    @property
    def error_rate(self) -> float:
        return len(self.errors) / len(self.samples) if self.samples else 1.0

    def percentile_ms(self, fraction: float, *, service: bool = False) -> float:
        """Interpolate between ranks — PostgreSQL's `percentile_cont`, not nearest-rank.

        Deliberately the same definition as `AnalyticsRepository.summary()`, so the two numbers
        in `docs/RELEASE.md` are comparable to each other rather than only to themselves.
        """
        values = sorted(
            float(sample.service_ms or 0) if service else sample.wall_ms
            for sample in self.samples
            if sample.ok
        )
        if not values:
            return 0.0
        if len(values) == 1:
            return values[0]
        position = fraction * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] + (position - low) * (values[high] - values[low])

    def render(self) -> str:
        return "\n".join(
            [
                "",
                "=" * 78,
                "DocQuery query-path load run — faked Bedrock and Pinecone (NOT the p95 gate)",
                "=" * 78,
                f"  requests       : {len(self.samples)} "
                f"({CONCURRENCY} concurrent x {ROUNDS} rounds)",
                f"  errors         : {len(self.errors)} ({self.error_rate:.1%})",
                f"  wall clock     : p50 {self.percentile_ms(0.50):.1f} ms, "
                f"p95 {self.percentile_ms(0.95):.1f} ms, "
                f"max {max((s.wall_ms for s in self.samples), default=0.0):.1f} ms",
                f"  recorded       : p50 {self.percentile_ms(0.50, service=True):.1f} ms, "
                f"p95 {self.percentile_ms(0.95, service=True):.1f} ms  "
                "(query.latency_ms, the column the production percentile reads)",
                "  target         : 3-5 s end to end — NOT measured here; no network in this "
                "run",
                f"  wall clock note: each sample waited behind the other {CONCURRENCY - 1} "
                "requests in its round on one event loop",
                "=" * 78,
            ]
        )


async def _measure(settings: Settings) -> LoadReport:
    """Upload one document, then hammer the query route as a Standard User.

    A Standard User on purpose: it is the role that submits queries, and it exercises the
    `search_history`/RBAC path an Administrator would skip. The role override is a single value
    on the app, so every concurrent request in a round is the same caller — interleaving two
    roles here would be measuring the harness, not the app.
    """
    stack = await build_stack()
    report = LoadReport()

    async with build_api(stack) as (client, as_user):
        as_user(stack.a.admin)
        upload = await client.post(
            f"{API}/documents",
            files={"file": ("perf-fees.pdf", FEE_PDF, "application/pdf")},
            headers=csrf_headers(),
        )
        assert upload.status_code in (200, 201, 202), upload.text
        document = upload.json()
        report.document_id = document["id"]
        chunks = await stack.documents.get_chunks(
            uuid.UUID(document["id"]), stack.a.id
        )
        # The page the queried fact is on, as `DocumentChunk.citation` renders it. Taken from
        # the chunk rows rather than hardcoded, so a change to how sections are labelled shows
        # up as a citation mismatch rather than as a silently weakened assertion.
        expected = [chunk for chunk in chunks if "10 percent" in chunk.text]
        assert len(expected) == 1, f"the fixture no longer has one tax page: {len(expected)}"
        report.expected_citation = expected[0].citation

        as_user(stack.a.user)
        for _ in range(TOTAL):
            stack.bedrock.queue_answer(ANSWER)

        async def one() -> Sample:
            started = time.perf_counter()
            response = await client.post(
                f"{API}/queries",
                json={"question_text": QUESTION},
                headers=csrf_headers(),
            )
            wall_ms = (time.perf_counter() - started) * 1_000
            body: dict[str, Any] = {}
            if response.status_code == 202:
                body = response.json()
            references = body.get("source_references") or []
            return Sample(
                status_code=response.status_code,
                wall_ms=wall_ms,
                service_ms=body.get("latency_ms"),
                status=body.get("status"),
                document_ids=tuple(str(item["document_id"]) for item in references),
                citations=tuple(str(item["section_or_page"]) for item in references),
            )

        for _ in range(ROUNDS):
            report.samples.extend(
                await asyncio.gather(*(one() for _ in range(CONCURRENCY)))
            )

    return report


@pytest.fixture(scope="module")
def report() -> LoadReport:
    """Run the load once for the module.

    Synchronous, driving the async run through `asyncio.run`, for the reason
    `backend/tests/eval/test_eval_retrieval.py` gives: a module-scoped async fixture would be
    bound to one event loop while every test gets its own.
    """
    measured = asyncio.run(_measure(get_settings()))
    print(measured.render())  # noqa: T201 - the harness output is the deliverable
    return measured


class TestNothingBreaksUnderConcurrency:
    """The half of this suite that is a real assertion rather than a measurement."""

    def test_the_run_produced_the_samples_it_claims(self, report: LoadReport) -> None:
        """Guards the denominator. Every figure below is a fraction of this count."""
        assert len(report.samples) == TOTAL
        assert report.document_id
        assert report.expected_citation

    def test_no_request_failed(self, report: LoadReport) -> None:
        codes = sorted({sample.status_code for sample in report.samples})
        assert codes == [202], f"{len(report.errors)} of {TOTAL} failed: {codes}"
        assert report.error_rate == 0.0

    def test_every_request_was_answered(self, report: LoadReport) -> None:
        """Not merely accepted. A not-found under load would mean a request lost its retrieval
        to a concurrent one — the corpus answers this question every time it is asked alone."""
        statuses = {sample.status for sample in report.samples}
        assert statuses == {QueryStatus.ANSWERED.value}, statuses

    def test_every_response_carries_its_own_single_citation(
        self, report: LoadReport
    ) -> None:
        """The cross-wiring check, and the reason the fake answer cites exactly one passage.

        Two references on a response means it was built from another request's retrieval; zero
        means `source_tracer` lost the marker. Both are invisible to a latency figure, and both
        are compliance defects rather than performance ones — `document-source-traceability.md`
        does not have a concurrency exemption.
        """
        for sample in report.samples:
            assert sample.document_ids == (report.document_id,), sample
            assert sample.citations == (report.expected_citation,), sample

    def test_every_query_recorded_its_own_latency(self, report: LoadReport) -> None:
        """`query.latency_ms` is the column the production p95 is computed over. A null or a
        zero there would make that percentile a fiction, and nothing else would notice."""
        for sample in report.samples:
            assert sample.service_ms is not None, sample
            assert sample.service_ms >= 0, sample


class TestTheMeasurement:
    """Figures, plus enough checking of the arithmetic that they can be quoted."""

    def test_the_app_overhead_stays_inside_the_tripwire(
        self, report: LoadReport
    ) -> None:
        """A regression tripwire on the app's own cost. Not the 3–5 second gate — see the
        module header before quoting either number."""
        assert report.percentile_ms(0.95) < MAX_OVERHEAD_P95_MS, report.render()

    def test_the_recorded_latency_never_exceeds_the_wall_clock(
        self, report: LoadReport
    ) -> None:
        """A sanity check on the two measurements rather than on the system.

        `latency_ms` covers a span inside the request, so its percentile cannot exceed the
        client-side one. If it ever does, one of the two is measuring the wrong thing and the
        figure in `docs/RELEASE.md` would be wrong in a way nobody could see.
        """
        for fraction in (0.50, 0.95):
            assert report.percentile_ms(fraction, service=True) <= (
                report.percentile_ms(fraction) + 1.0
            )

    def test_the_percentile_is_computed_the_way_postgresql_computes_it(self) -> None:
        """`percentile_cont` interpolates; nearest-rank does not, and on 64 samples the two
        differ. Checked against a hand-worked example so the reported p95 is comparable with
        `AnalyticsRepository.summary()`'s rather than merely similar to it."""
        harness = LoadReport(
            samples=[
                Sample(202, float(value), value, QueryStatus.ANSWERED.value, (), ())
                for value in range(1, 101)
            ]
        )
        # position = 0.95 * 99 = 94.05 -> values[94] + 0.05 * (values[95] - values[94])
        assert harness.percentile_ms(0.95) == pytest.approx(95.05)
        assert harness.percentile_ms(0.50) == pytest.approx(50.5)
        assert harness.percentile_ms(0.0) == 1.0
        assert harness.percentile_ms(1.0) == 100.0

    def test_an_empty_run_reports_a_total_error_rate(self) -> None:
        """The degenerate case, asserted because the alternative is worse than a crash: a run
        that collected nothing must not report 0% errors and a 0 ms p95."""
        empty = LoadReport()
        assert empty.error_rate == 1.0
        assert empty.percentile_ms(0.95) == 0.0
