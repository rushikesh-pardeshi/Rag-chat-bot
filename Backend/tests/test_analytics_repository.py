"""backend/tests/test_analytics_repository.py

Role: Tests for `backend/repositories/analytics_repository.py` — that each aggregate compiles
to the statement it claims to, with every value bound and every read scoped to one tenant.

Why these tests read compiled SQL instead of querying a database
----------------------------------------------------------------
The claims worth testing here are properties of the *statement*, not of any result set. "The
window is inclusive on both ends", "the stopword list is a bound parameter", "no read is
unscoped" — a result set can satisfy all three on a fixture and still come from a statement
that interpolates. Compiling against `postgresql.dialect()` inspects the thing that actually
reaches the database, and it needs neither PostgreSQL nor a session, so these run on a laptop
with nothing installed.

What this file does *not* cover, stated plainly: whether PostgreSQL's `regexp_split_to_table`
tokenizes as intended, and whether `percentile_cont` returns the percentile expected. Those are
the database's behaviour rather than this repository's, and asserting them needs a real
server — they belong to the `requires_db` integration tests, and the Phase 4 latency gate is
where the p95 gets a live figure.

How the statements are reached
------------------------------
`_token_subquery` is the one composable piece and is called directly. The five public methods
are `async` and end in `session.execute`, so each is driven with a session stub that captures
the statement and returns an empty result. Intercepting at `execute` is the honest way to see
what a method builds: the test sees exactly what production passes down, including anything a
later edit adds.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy.dialects import postgresql

from backend.repositories.analytics_repository import (
    KEYWORD_SPLIT_PATTERN,
    AnalyticsRepository,
    TrendGranularity,
)

TENANT = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER_TENANT = uuid.UUID("22222222-2222-2222-2222-222222222222")
FROM = datetime(2026, 1, 1, tzinfo=UTC)
TO = datetime(2026, 6, 30, tzinfo=UTC)

STOPWORDS = ["and", "the", "what"]


class _ZeroSummaryRow:
    """The row `summary()` reads when the window is empty.

    Every value 0 or None on purpose: it doubles as the check that an empty window produces
    zeros and nulls rather than an exception, which is what the method promises a dashboard on
    a tenant's first day.
    """

    total = answered = not_found = failed = pending = 0
    avg_latency_ms = p95_latency_ms = avg_confidence = None


class _Result:
    """The two accessors the repository uses on a result.

    `all()` is empty and `one()` is the zero row, which is why no test has to configure this:
    the four list methods map an empty sequence, and `summary()` maps the zero row. Returning
    rows to the list methods would mean maintaining a fake shape for each, and a wrong shape
    there is a test failure that says nothing about the SQL.
    """

    def all(self) -> list[Any]:
        return []

    def one(self) -> _ZeroSummaryRow:
        return _ZeroSummaryRow()


class CapturingSession:
    """Records every statement passed to `execute` and returns an empty result.

    A stub rather than a mock: the repository's entire contract with the session is
    `await execute(stmt)` returning something with `.all()` / `.one()`, and a stub implementing
    exactly that cannot drift from the real `AsyncSession` in a way a test would accept
    silently.
    """

    def __init__(self) -> None:
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _Result:
        self.statements.append(statement)
        return _Result()

    def sql(self, index: int = -1) -> str:
        """The captured statement as PostgreSQL would receive it, values still bound.

        No `literal_binds`: rendering the values inline is the opposite of what these tests
        check. Leaving them as placeholders is what makes an interpolated value visible — it
        would appear in the SQL text with nothing bound behind it.
        """
        return str(self.statements[index].compile(dialect=postgresql.dialect()))

    def params(self, index: int = -1) -> dict[str, Any]:
        """Every value the driver will send separately from the statement text."""
        return dict(self.statements[index].compile(dialect=postgresql.dialect()).params)


@pytest.fixture
def session() -> CapturingSession:
    return CapturingSession()


@pytest.fixture
def repo(session: CapturingSession) -> AnalyticsRepository:
    return AnalyticsRepository(session)  # type: ignore[arg-type]


class TestEveryReadIsTenantScoped:
    """The property that matters most here: no aggregate can span two tenants.

    Analytics is the read most likely to be written as "count everything", and a missing
    `WHERE` discloses another bank's search behaviour without showing a single row of its
    data — which is why this is asserted for all five methods rather than spot-checked.
    """

    async def test_all_five_statements_filter_on_the_tenant(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        await repo.keyword_counts(TENANT, stopwords=STOPWORDS, min_length=3, limit=10)
        await repo.document_usage(TENANT, limit=10)
        await repo.volume_trend(TENANT, granularity=TrendGranularity.DAY, limit=30)
        await repo.active_users(TENANT, limit=10)
        await repo.summary(TENANT)

        assert len(session.statements) == 5
        for index in range(5):
            sql = session.sql(index)
            assert "query.tenant_id = " in sql, f"statement {index} is unscoped"
            assert TENANT in session.params(index).values(), (
                f"statement {index} does not bind the tenant id"
            )

    async def test_the_tenant_is_never_rendered_into_the_sql(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """A bound id appears in the parameters, never in the statement text.

        That distinction is the whole of injection safety, and it earns an explicit assertion
        rather than being implied by "we used the ORM": `text()` and an f-string both compile
        happily through the same `execute`.
        """
        await repo.active_users(TENANT, limit=5)
        assert str(TENANT) not in session.sql()
        assert TENANT in session.params().values()

    async def test_a_second_tenant_produces_a_different_bound_value(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        await repo.active_users(TENANT, limit=5)
        await repo.active_users(OTHER_TENANT, limit=5)
        assert TENANT in session.params(0).values()
        assert OTHER_TENANT in session.params(1).values()
        # Identical SQL, different parameters: the statement is reusable, the scope is data.
        assert session.sql(0) == session.sql(1)

    async def test_active_users_scopes_both_sides_of_the_join(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """`user_account` is filtered too, not just `query`.

        The foreign key makes the join sufficient today. The method filters anyway, because
        "the join implies it" stops being true the moment someone changes the join type — and
        this assertion is what would fail then.
        """
        await repo.active_users(TENANT, limit=10)
        sql = session.sql()
        assert "query.tenant_id = " in sql
        assert "user_account.tenant_id = " in sql


class TestKeywordTokenizationIsFixed:
    """The split pattern is a constant, and the stopwords arrive as parameters."""

    def test_the_pattern_is_bound_not_embedded(self, repo: AnalyticsRepository) -> None:
        """`regexp_split_to_table`'s pattern travels as a parameter.

        `.claude/rules/secure-coding-vapt.md` forbids user-supplied regex, and inside the
        database the stakes rise: a crafted pattern would burn PostgreSQL's own CPU. The
        pattern being a bound parameter holding a module constant is two guarantees at once —
        no caller can choose it, and it is not concatenated into SQL either way.
        """
        subquery = repo._token_subquery(  # noqa: SLF001 - the composable piece under test
            TENANT, created_from=None, created_to=None
        )
        compiled = subquery.select().compile(dialect=postgresql.dialect())
        assert "regexp_split_to_table" in str(compiled)
        assert KEYWORD_SPLIT_PATTERN in compiled.params.values()
        assert KEYWORD_SPLIT_PATTERN not in str(compiled)

    def test_the_pattern_has_no_nested_quantifier(self) -> None:
        """A linear pattern, checked as text rather than assumed from review.

        `[^a-z0-9']+` is one character class with one quantifier — no alternation, no nesting,
        so there is no backtracking behaviour to exploit. This asserts the shape has not been
        "improved" into something with a nested group.
        """
        assert KEYWORD_SPLIT_PATTERN == r"[^a-z0-9']+"
        assert "(" not in KEYWORD_SPLIT_PATTERN
        assert ")" not in KEYWORD_SPLIT_PATTERN

    async def test_stopwords_and_min_length_are_parameters(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        await repo.keyword_counts(TENANT, stopwords=STOPWORDS, min_length=4, limit=7)
        params = session.params()
        # The list is expanded by the driver at execution, so it binds as one value.
        assert STOPWORDS in params.values()
        assert 4 in params.values()
        assert 7 in params.values()
        for word in STOPWORDS:
            assert f"'{word}'" not in session.sql()

    async def test_case_is_folded_in_sql(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """`lower()` runs in the database, so "Wire" and "wire" are one keyword.

        Folding in Python would mean pulling every question text into the web process, which is
        exactly the transfer the module docstring says it avoids.
        """
        await repo.keyword_counts(TENANT, stopwords=[], min_length=3, limit=5)
        assert "lower(query.question_text)" in session.sql()

    async def test_ascending_flips_only_the_order(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """Least-searched is the same aggregate read backwards, tie-break intact.

        The alphabetical tie-break is the load-bearing half: the bottom of a long tail is all
        count-1 rows, and without a second sort key the "least searched" panel would reshuffle
        on every refresh with no data having changed.
        """
        await repo.keyword_counts(TENANT, stopwords=[], min_length=3, limit=5)
        await repo.keyword_counts(
            TENANT, stopwords=[], min_length=3, limit=5, ascending=True
        )
        descending, ascending = session.sql(0), session.sql(1)
        assert "ORDER BY occurrences DESC" in descending
        assert "ORDER BY occurrences ASC" in ascending
        assert "tokens.word ASC" in descending
        assert "tokens.word ASC" in ascending
        # Same projection, grouping and filters: only the direction differs.
        assert descending.replace("DESC", "ASC") == ascending

    async def test_both_counts_are_selected(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """Occurrences and distinct queries, because they answer different questions."""
        await repo.keyword_counts(TENANT, stopwords=[], min_length=3, limit=5)
        sql = session.sql()
        assert "count(*) AS occurrences" in sql
        assert "count(DISTINCT tokens.query_id) AS queries" in sql


class TestDocumentUsageReadsCitations:
    async def test_it_unnests_source_references(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """Usage is counted from the answers' own citations, not from the document table."""
        await repo.document_usage(TENANT, limit=10)
        sql = session.sql()
        assert "jsonb_array_elements(query.source_references)" in sql
        params = session.params().values()
        # The two citation fields `document-source-traceability.md` makes mandatory.
        assert "source_document_name" in params
        assert "document_id" in params

    async def test_it_guards_against_a_json_scalar(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """`jsonb_typeof(...) = 'array'` is in the statement.

        `jsonb_array_elements` raises on a scalar, and this statement aggregates every
        answered query in the tenant — so one malformed row is the whole dashboard returning
        500, not one row reading wrong. The column is `NOT NULL` with a list default, so this
        is defence in depth rather than a live bug; asserting it means a later
        "simplification" has to delete a test to remove it.
        """
        await repo.document_usage(TENANT, limit=10)
        assert "jsonb_typeof(query.source_references)" in session.sql()
        assert "array" in session.params().values()

    async def test_only_answered_queries_are_counted(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """A citation exists only on an answered query.

        Counting `NOT_FOUND` rows would inflate usage with questions that produced no answer —
        and a refusal has no sources by definition, so it would contribute nothing but noise.
        """
        await repo.document_usage(TENANT, limit=10)
        assert "query.status = " in session.sql()
        assert "ANSWERED" in session.params().values()

    async def test_it_groups_on_id_and_name_together(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """Two grouping expressions, so a re-upload with the same name stays a separate row.

        Asserted on the two `->>` extractions rather than on the labels: PostgreSQL groups by
        the expression, and SQLAlchemy renders it that way, so the labels do not appear in the
        `GROUP BY` clause at all.
        """
        await repo.document_usage(TENANT, limit=10)
        group_by = session.sql().split("GROUP BY")[1]
        assert group_by.count("refs.reference ->>") == 2


class TestVolumeTrend:
    @pytest.mark.parametrize(
        "granularity",
        [TrendGranularity.DAY, TrendGranularity.WEEK, TrendGranularity.MONTH],
    )
    async def test_the_granularity_is_bound_not_formatted(
        self,
        repo: AnalyticsRepository,
        session: CapturingSession,
        granularity: TrendGranularity,
    ) -> None:
        """`date_trunc`'s unit is a parameter, for all three units.

        It comes from an enum, so it could not be arbitrary text even if it were formatted in.
        Both controls are real; this asserts the inner one, which is what still holds if the
        enum is ever widened.
        """
        await repo.volume_trend(TENANT, granularity=granularity, limit=30)
        sql = session.sql()
        assert "date_trunc(" in sql
        assert granularity.value in session.params().values()
        assert f"date_trunc('{granularity.value}'" not in sql

    async def test_it_takes_the_newest_periods_and_returns_them_oldest_first(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """Two orderings, and both are needed.

        The inner `DESC` + `LIMIT` chooses *which* periods — the most recent, so "the last 30
        days" is not silently the first 30 days of a two-year history. The outer `ASC` is the
        order they are drawn in. One ordering cannot do both.
        """
        await repo.volume_trend(TENANT, granularity=TrendGranularity.DAY, limit=30)
        sql = session.sql()
        assert "ORDER BY period_start DESC" in sql
        assert sql.rstrip().endswith("ORDER BY trend.period_start ASC")

    async def test_outcomes_are_counted_with_filter_clauses(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """One pass over the rows, and `not_found` kept distinct from `failed`.

        A chart folding a refusal into an error rate would send someone to fix the pipeline
        when the finding is a gap in the document repository.
        """
        await repo.volume_trend(TENANT, granularity=TrendGranularity.DAY, limit=30)
        sql = session.sql()
        assert sql.count("FILTER (WHERE") == 3
        params = session.params().values()
        for status in ("ANSWERED", "NOT_FOUND", "FAILED"):
            assert status in params


class TestSummary:
    async def test_the_p95_is_computed_by_postgresql(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """`percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms)`.

        This statement is the production source for the latency figure the Phase 4 gate asks to
        have written down. Computing it in SQL over `query.latency_ms` means the number comes
        from real traffic; a percentile over a fixture run measures the test harness.
        """
        await repo.summary(TENANT)
        sql = session.sql()
        assert "percentile_cont(" in sql
        assert "WITHIN GROUP (ORDER BY query.latency_ms ASC)" in sql
        assert 0.95 in session.params().values()

    async def test_an_empty_window_yields_zeros_rather_than_no_row(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """No `GROUP BY`, so the aggregate always returns exactly one row.

        A dashboard forced to tell "no rows came back" apart from "zero queries happened" would
        render a blank panel on a tenant's first day, which reads as a broken page.
        """
        result = await repo.summary(TENANT)
        assert "GROUP BY" not in session.sql()
        assert result.total == 0
        # An average over no rows is null, not zero: reporting zero would claim a measured
        # latency of nothing at all.
        assert result.avg_latency_ms is None
        assert result.p95_latency_ms is None

    async def test_all_four_outcomes_plus_pending_are_counted(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        await repo.summary(TENANT)
        assert session.sql().count("FILTER (WHERE") == 4
        params = session.params().values()
        for status in ("ANSWERED", "NOT_FOUND", "FAILED", "PENDING"):
            assert status in params


class TestTheDateWindow:
    """Inclusive on both ends, and identical in every method.

    `query_repository._apply_filters` uses the same boundaries. Two reads of one window
    disagreeing by a single row is the kind of discrepancy that makes a reviewer distrust both
    numbers — so the operators are asserted, not merely the presence of a filter.
    """

    async def test_both_bounds_are_inclusive_and_bound(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        await repo.active_users(TENANT, limit=10, created_from=FROM, created_to=TO)
        sql = session.sql()
        assert "query.created_at >= " in sql
        assert "query.created_at <= " in sql
        params = session.params().values()
        assert FROM in params
        assert TO in params

    async def test_an_omitted_bound_adds_no_filter(
        self, repo: AnalyticsRepository, session: CapturingSession
    ) -> None:
        """`None` means "no bound", not "the epoch".

        Defaulting an absent `created_from` to a sentinel date would quietly exclude older rows
        from a report the caller asked to cover all history.
        """
        await repo.active_users(TENANT, limit=10, created_from=FROM)
        sql = session.sql()
        assert "query.created_at >= " in sql
        assert "query.created_at <= " not in sql

    @pytest.mark.parametrize(
        "call",
        ["keyword_counts", "document_usage", "volume_trend", "active_users", "summary"],
    )
    async def test_every_method_applies_the_window(
        self, repo: AnalyticsRepository, session: CapturingSession, call: str
    ) -> None:
        kwargs: dict[str, Any] = {"created_from": FROM, "created_to": TO}
        if call == "keyword_counts":
            kwargs |= {"stopwords": [], "min_length": 3, "limit": 5}
        elif call == "volume_trend":
            kwargs |= {"granularity": TrendGranularity.DAY, "limit": 5}
        elif call != "summary":
            kwargs |= {"limit": 5}

        await getattr(repo, call)(TENANT, **kwargs)
        params = session.params().values()
        assert FROM in params, f"{call} dropped created_from"
        assert TO in params, f"{call} dropped created_to"


class TestRowLimits:
    @pytest.mark.parametrize(
        ("call", "kwargs"),
        [
            ("keyword_counts", {"stopwords": [], "min_length": 3, "limit": 13}),
            ("document_usage", {"limit": 13}),
            ("volume_trend", {"granularity": TrendGranularity.DAY, "limit": 13}),
            ("active_users", {"limit": 13}),
        ],
    )
    async def test_every_list_read_is_bounded(
        self,
        repo: AnalyticsRepository,
        session: CapturingSession,
        call: str,
        kwargs: dict[str, Any],
    ) -> None:
        """A `LIMIT` on all four list reads, with the value bound.

        The module promises "at most `limit` rows", and an unbounded aggregate over a year of
        history is both a slow page and a large response — the two things moving the grouping
        into SQL was meant to prevent.
        """
        await getattr(repo, call)(TENANT, **kwargs)
        assert "LIMIT" in session.sql()
        assert 13 in session.params().values()
