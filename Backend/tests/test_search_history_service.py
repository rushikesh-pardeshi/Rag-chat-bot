"""backend/tests/test_search_history_service.py

Role: Tests for `backend/services/search_history_service.py` — whose history a caller may read,
and the shape of the page they get.

The gate this file closes, verbatim from `WORKBREAKDOWN.md`: *"`GET /api/v1/history` returns
own rows for a Standard User and all tenant rows for an Administrator, with pagination and
date filtering; the keyword filter uses a fixed pattern."* Four claims, four test classes.

Why the authorization tests are here and not in a route test
------------------------------------------------------------
The route has no `require_admin` — deliberately, because one endpoint serves both roles and the
*service* narrows the scope (see the route's docstring). That makes this module the only place
the narrowing is checked, so it is checked from both directions: a Standard User cannot widen
their scope with a `user_id`, and an Administrator's scope stops at their own tenant.

Why the keyword test compiles SQL
---------------------------------
`FakeQueryRepository` matches a keyword as a literal substring, which is the *semantic* the
escaped `ILIKE` is supposed to implement — but a fake agreeing with the intent proves nothing
about the statement the database receives. `TestKeywordIsAFixedPattern` therefore calls the
real `QueryRepository._apply_filters` and inspects the compiled SQL. No session and no
PostgreSQL: it is a `staticmethod` over a `select()`, so the escaping is tested offline.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from backend.core.exceptions import ValidationError
from backend.models.audit_log import AuditAction, AuditObjectType
from backend.models.query import Query, QueryStatus
from backend.models.user import User, UserRole
from backend.repositories.query_repository import QueryRepository
from backend.services.audit_service import AuditService
from backend.services.auth_service import RequestContext
from backend.services.search_history_service import (
    AUDIT_PHASE_STATS,
    MAX_KEYWORD_LENGTH,
    MAX_PAGE_SIZE,
    SearchHistoryService,
)
from backend.tests.fakes import FakeAuditRepository, FakeQueryRepository
from backend.utils.ids import new_id, utc_now

CONTEXT = RequestContext(ip_address="203.0.113.10", user_agent="pytest")


def _user(tenant_id: uuid.UUID, role: UserRole, email: str) -> User:
    return User(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        email=email,
        full_name=None,
        hashed_password="unused-in-these-tests",
        role=role,
        is_active=True,
        failed_login_attempts=0,
        locked_until=None,
    )


@pytest.fixture
def tenant_id() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def analyst(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.STANDARD_USER, "analyst@bank.example")


@pytest.fixture
def colleague(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.STANDARD_USER, "colleague@bank.example")


@pytest.fixture
def admin(tenant_id: uuid.UUID) -> User:
    return _user(tenant_id, UserRole.ADMINISTRATOR, "admin@bank.example")


@pytest.fixture
def queries() -> FakeQueryRepository:
    """No document repository needed: history reads rows, it never resolves chunks."""
    return FakeQueryRepository()


@pytest.fixture
def audit_repo() -> FakeAuditRepository:
    return FakeAuditRepository()


@pytest.fixture
def service(
    queries: FakeQueryRepository, audit_repo: FakeAuditRepository
) -> SearchHistoryService:
    return SearchHistoryService(queries, AuditService(audit_repo))


async def _ask(
    queries: FakeQueryRepository,
    actor: User,
    question: str,
    *,
    minutes_ago: int = 0,
    status: QueryStatus = QueryStatus.ANSWERED,
) -> Query:
    """One history row.

    `created_at` is set explicitly rather than left at `utc_now()`. Every row created in one
    test would otherwise share a timestamp to the microsecond, and the pagination order and
    the date filter would then be asserted against ties — passing whatever the dict iteration
    order happened to be.
    """
    row = await queries.create(
        query_id=new_id(),
        tenant_id=actor.tenant_id,
        user_id=actor.id,
        question_text=question,
    )
    row.status = status
    row.created_at = utc_now() - timedelta(minutes=minutes_ago)
    return row


class TestScope:
    """`rbac-enforcement.md`: own rows for a Standard User, tenant rows for an Admin."""

    async def test_a_standard_user_sees_only_their_own_rows(
        self, service, queries, analyst, colleague
    ) -> None:
        await _ask(queries, analyst, "What is the wire fee?")
        await _ask(queries, colleague, "What is the overdraft rate?")

        page = await service.list_history(actor=analyst, context=CONTEXT)

        assert page.scope == "own"
        assert [row.question_text for row in page.items] == ["What is the wire fee?"]
        assert page.total == 1

    async def test_an_administrator_sees_every_user_in_the_tenant(
        self, service, queries, admin, analyst, colleague
    ) -> None:
        await _ask(queries, analyst, "What is the wire fee?", minutes_ago=2)
        await _ask(queries, colleague, "What is the overdraft rate?", minutes_ago=1)

        page = await service.list_history(actor=admin, context=CONTEXT)

        assert page.scope == "tenant"
        assert page.total == 2
        # Newest first, matching the repository's `created_at DESC`.
        assert [row.question_text for row in page.items] == [
            "What is the overdraft rate?",
            "What is the wire fee?",
        ]

    async def test_a_standard_user_cannot_widen_scope_with_a_user_id(
        self, service, queries, analyst, colleague
    ) -> None:
        """The IDOR probe. The response is their own history — not a 403, not the colleague's.

        A 403 would confirm the parameter selects something. Returning the caller's own rows
        says the parameter was never an authorization input, which is the truth.
        """
        await _ask(queries, analyst, "What is the wire fee?")
        await _ask(queries, colleague, "What is the overdraft rate?")

        page = await service.list_history(
            actor=analyst, context=CONTEXT, user_id=colleague.id
        )

        assert page.scope == "own"
        assert [row.question_text for row in page.items] == ["What is the wire fee?"]
        assert page.total == 1

    async def test_the_ignored_user_id_is_recorded_as_a_foreign_request(
        self, service, queries, audit_repo, analyst, colleague
    ) -> None:
        """An ignored probe still gets logged — it is what an IDOR attempt looks like."""
        await _ask(queries, analyst, "What is the wire fee?")

        await service.list_history(actor=analyst, context=CONTEXT, user_id=colleague.id)
        await service.list_history(actor=analyst, context=CONTEXT)

        flags = [entry.detail["foreign_user_requested"] for entry in audit_repo.entries]
        assert flags == [True, False]

    async def test_an_administrator_can_filter_to_one_user(
        self, service, queries, admin, analyst, colleague
    ) -> None:
        await _ask(queries, analyst, "What is the wire fee?")
        await _ask(queries, colleague, "What is the overdraft rate?")

        page = await service.list_history(
            actor=admin, context=CONTEXT, user_id=colleague.id
        )

        assert [row.question_text for row in page.items] == ["What is the overdraft rate?"]
        assert page.total == 1

    async def test_an_administrator_in_another_tenant_sees_nothing(
        self, service, queries, analyst
    ) -> None:
        """Role and tenant are independent checks, and `tenant_id` is never a parameter."""
        await _ask(queries, analyst, "What is the wire fee?")
        outsider = _user(uuid.uuid4(), UserRole.ADMINISTRATOR, "admin@other-bank.example")

        page = await service.list_history(actor=outsider, context=CONTEXT)

        assert page.items == ()
        assert page.total == 0

    async def test_a_standard_user_id_filter_cannot_reach_another_tenant(
        self, service, queries, analyst
    ) -> None:
        """Two dimensions at once: a foreign tenant's user id, requested by a Standard User."""
        await _ask(queries, analyst, "What is the wire fee?")
        stranger = _user(uuid.uuid4(), UserRole.STANDARD_USER, "user@other-bank.example")

        page = await service.list_history(
            actor=stranger, context=CONTEXT, user_id=analyst.id
        )

        assert page.items == ()
        assert page.total == 0


class TestPagination:
    async def test_pages_are_disjoint_and_the_total_is_unpaged(
        self, service, queries, analyst
    ) -> None:
        for index in range(5):
            await _ask(queries, analyst, f"Question {index}", minutes_ago=index)

        first = await service.list_history(actor=analyst, context=CONTEXT, limit=2)
        second = await service.list_history(
            actor=analyst, context=CONTEXT, limit=2, offset=2
        )

        assert [row.question_text for row in first.items] == ["Question 0", "Question 1"]
        assert [row.question_text for row in second.items] == ["Question 2", "Question 3"]
        # The count is of everything that matches, not of the page — otherwise a pager can
        # never show more than one page.
        assert first.total == second.total == 5

    async def test_an_oversized_limit_is_capped_not_honoured(
        self, service, queries, analyst
    ) -> None:
        """The route also bounds this with `le=MAX_PAGE_SIZE`; the service does not rely on it.

        A second caller — a Celery task, a script, a future route — reaches the service
        directly, and `MAX_PAGE_SIZE` is a resource control, not a validation nicety.
        """
        await _ask(queries, analyst, "What is the wire fee?")

        page = await service.list_history(actor=analyst, context=CONTEXT, limit=10_000)

        assert page.limit == MAX_PAGE_SIZE

    async def test_a_nonsense_page_is_clamped_rather_than_rejected(
        self, service, queries, analyst
    ) -> None:
        await _ask(queries, analyst, "What is the wire fee?")

        page = await service.list_history(
            actor=analyst, context=CONTEXT, limit=0, offset=-5
        )

        assert (page.limit, page.offset) == (1, 0)
        assert len(page.items) == 1


class TestFilters:
    async def test_a_date_window_excludes_rows_on_both_sides(
        self, service, queries, analyst
    ) -> None:
        await _ask(queries, analyst, "Too old", minutes_ago=120)
        await _ask(queries, analyst, "In the window", minutes_ago=60)
        await _ask(queries, analyst, "Too new", minutes_ago=1)
        now = utc_now()

        page = await service.list_history(
            actor=analyst,
            context=CONTEXT,
            created_from=now - timedelta(minutes=90),
            created_to=now - timedelta(minutes=30),
        )

        assert [row.question_text for row in page.items] == ["In the window"]
        assert page.total == 1

    async def test_a_status_filter_narrows_to_one_outcome(
        self, service, queries, analyst
    ) -> None:
        await _ask(queries, analyst, "Answered one", status=QueryStatus.ANSWERED)
        await _ask(queries, analyst, "Nothing found", status=QueryStatus.NOT_FOUND)

        page = await service.list_history(
            actor=analyst, context=CONTEXT, status=QueryStatus.NOT_FOUND
        )

        assert [row.question_text for row in page.items] == ["Nothing found"]

    async def test_a_blank_keyword_is_dropped_rather_than_matched(
        self, service, queries, analyst
    ) -> None:
        """`"   "` must not become `ILIKE '%   %'`, which matches nothing at all."""
        await _ask(queries, analyst, "What is the wire fee?")

        page = await service.list_history(actor=analyst, context=CONTEXT, keyword="   ")

        assert page.total == 1

    async def test_an_over_long_keyword_is_rejected(
        self, service, queries, analyst
    ) -> None:
        await _ask(queries, analyst, "What is the wire fee?")

        with pytest.raises(ValidationError, match=str(MAX_KEYWORD_LENGTH)):
            await service.list_history(
                actor=analyst, context=CONTEXT, keyword="x" * (MAX_KEYWORD_LENGTH + 1)
            )

    async def test_a_keyword_wildcard_matches_itself(
        self, service, queries, analyst
    ) -> None:
        """The contract the escaping exists to provide: `%` is a percent sign, not "anything".

        Asserted here against the fake's literal substring match, and against the generated SQL
        in `TestKeywordIsAFixedPattern` below. Both, because this test would pass with a fake
        that used `fnmatch` and a repository that forgot to escape.
        """
        await _ask(queries, analyst, "Is the fee 100% of the transfer?")
        await _ask(queries, analyst, "What is the overdraft rate?")

        page = await service.list_history(actor=analyst, context=CONTEXT, keyword="100%")

        assert [row.question_text for row in page.items] == [
            "Is the fee 100% of the transfer?"
        ]

    async def test_a_bare_percent_does_not_match_everything(
        self, service, queries, analyst
    ) -> None:
        await _ask(queries, analyst, "Is the fee 100% of the transfer?")
        await _ask(queries, analyst, "What is the overdraft rate?")

        page = await service.list_history(actor=analyst, context=CONTEXT, keyword="%")

        assert page.total == 1, "A bare % behaved as a wildcard and matched both rows."


class TestKeywordIsAFixedPattern:
    """The real `ILIKE` escaping, checked without a database.

    `_apply_filters` is a `staticmethod` over a `select()`, so the statement it builds can be
    compiled and read here. This is the assertion that would actually fail if someone replaced
    the escaping with an f-string, which is the change worth catching.
    """

    @staticmethod
    def _compiled(keyword: str) -> tuple[str, list[object]]:
        """The SQL text PostgreSQL receives, and the values bound to it.

        Two deliberate choices. The dialect is `postgresql`, because SQLAlchemy's default
        renders `ilike` as `lower(x) LIKE lower(y)` and the assertions below read the operator.
        And the pattern is asserted as a **bound parameter**, not inlined with
        `literal_binds` — literal rendering doubles `%` for the driver's paramstyle and
        doubles backslashes for string quoting, so `100\\%\\_x` arrives as `100\\\\%%\\\\_x`.
        That is quoting noise on top of the escaping under test, and it hides the very
        character sequence these tests exist to pin down. What the database actually gets is
        one parameter carrying the pattern, plus `ESCAPE` in the statement text.
        """
        stmt = QueryRepository._apply_filters(
            select(Query),
            status=None,
            keyword=keyword,
            created_from=None,
            created_to=None,
        )
        compiled = stmt.compile(dialect=postgresql.dialect())
        return str(compiled), list(compiled.params.values())

    def test_wildcards_in_the_keyword_are_escaped(self) -> None:
        sql, params = self._compiled("100%_x")

        assert "ESCAPE" in sql.upper()
        assert r"%100\%\_x%" in params

    def test_a_backslash_is_escaped_before_the_wildcards(self) -> None:
        r"""Order matters: escaping `%` first leaves `\%` and turns a literal into an escape.

        A user searching for `50\%` must find the text `50\%`, and the only way that works is
        doubling the backslash before the percent sign is escaped. Escaping in the other order
        yields `50\\%` — an escaped backslash followed by a live wildcard.
        """
        _, params = self._compiled(r"50\%")

        assert r"%50\\\%%" in params

    def test_no_regex_operator_reaches_the_statement(self) -> None:
        """`secure-coding-vapt.md` forbids a user-supplied regex. `~` is PostgreSQL's."""
        sql, params = self._compiled(".*(a+)+$")

        assert " ~ " not in sql
        assert "ILIKE" in sql.upper()
        # The regex is data, not syntax: it travels as a parameter, uninterpreted.
        assert "%.*(a+)+$%" in params


class TestAudit:
    async def test_every_read_is_audited_with_its_scope(
        self, service, queries, audit_repo, analyst, admin
    ) -> None:
        await _ask(queries, analyst, "What is the wire fee?")

        await service.list_history(actor=analyst, context=CONTEXT)
        await service.list_history(actor=admin, context=CONTEXT)

        assert [entry.action for entry in audit_repo.entries] == [
            AuditAction.HISTORY_VIEW,
            AuditAction.HISTORY_VIEW,
        ]
        assert [entry.detail["scope"] for entry in audit_repo.entries] == [
            "own",
            "tenant",
        ]
        # `==`, not `is`: the column round-trips a plain `str`, and `AuditObjectType` is a
        # `StrEnum` precisely so that comparison holds either way.
        assert audit_repo.entries[0].object_type == AuditObjectType.QUERY

    async def test_the_audit_records_that_a_keyword_was_used_never_the_keyword(
        self, service, queries, audit_repo, analyst
    ) -> None:
        """A history keyword is user text and can name a customer or an account number."""
        await _ask(queries, analyst, "What is the balance on account 4111111111111111?")

        await service.list_history(
            actor=analyst, context=CONTEXT, keyword="account 4111111111111111"
        )

        detail = audit_repo.entries[0].detail
        assert detail["keyword_used"] is True
        assert "4111111111111111" not in repr(detail)

    async def test_the_object_id_names_whose_history_was_read(
        self, service, queries, audit_repo, analyst, admin, colleague
    ) -> None:
        """A list read has no single row, so the object is the subject of the read."""
        await _ask(queries, analyst, "What is the wire fee?")

        await service.list_history(actor=analyst, context=CONTEXT)
        await service.list_history(actor=admin, context=CONTEXT)
        await service.list_history(actor=admin, context=CONTEXT, user_id=colleague.id)

        assert [entry.object_id for entry in audit_repo.entries] == [
            str(analyst.id),
            # No filter: the whole tenant was read, so the tenant is the object.
            str(admin.tenant_id),
            str(colleague.id),
        ]

    async def test_a_read_that_returns_nothing_is_still_audited(
        self, service, audit_repo, analyst
    ) -> None:
        """An empty history is a read of that user's history, and BR-05 wants it recorded."""
        page = await service.list_history(actor=analyst, context=CONTEXT)

        assert page.total == 0
        assert len(audit_repo.entries) == 1


class TestPageShape:
    async def test_the_page_reports_the_limit_and_offset_it_actually_used(
        self, service, queries, analyst
    ) -> None:
        """Not the ones given — a clamped page must say so, or the pager desynchronizes."""
        for index in range(3):
            await _ask(queries, analyst, f"Question {index}", minutes_ago=index)

        page = await service.list_history(
            actor=analyst, context=CONTEXT, limit=MAX_PAGE_SIZE + 50, offset=1
        )

        assert page.limit == MAX_PAGE_SIZE
        assert page.offset == 1
        assert len(page.items) == 2

    async def test_items_are_a_tuple_so_a_caller_cannot_mutate_the_page(
        self, service, queries, analyst
    ) -> None:
        await _ask(queries, analyst, "What is the wire fee?")

        page = await service.list_history(actor=analyst, context=CONTEXT)

        assert isinstance(page.items, tuple)

    async def test_the_scope_is_one_of_the_two_documented_values(
        self, service, analyst, admin
    ) -> None:
        """The UI labels the view from this string, so a third value renders as nothing."""
        own = await service.list_history(actor=analyst, context=CONTEXT)
        tenant = await service.list_history(actor=admin, context=CONTEXT)

        assert {own.scope, tenant.scope} == {"own", "tenant"}


def test_the_history_row_carries_no_column_the_query_table_lacks() -> None:
    """`search_history` is a projection over `query`, not a second table.

    `architecture.md` forbids duplicating question and answer text into a history table, so
    there must be no `SearchHistory` ORM model. This asserts the absence — the cheapest
    guard against someone adding one in a later phase and reintroducing the drift.
    """
    from backend.models import query as query_module

    assert not hasattr(query_module, "SearchHistory")


class TestHistoryStats:
    """`history_stats` — the outcome breakdown, and why it is one request and one audit row.

    The tiles on the history page used to be built in the browser by calling `list_history`
    once per outcome with `limit=1` and reading `total`. That worked, and it wrote a
    `HISTORY_VIEW` row per call: four for one filter change, three of them counting reads that
    a compliance reviewer cannot tell apart from a person reading somebody's history. In a
    product whose premise is a trustworthy trail, "how often was this history viewed" has to be
    answerable from the trail — so the breakdown moved to the server, where it is one
    `GROUP BY` and one row.

    These tests therefore assert the audit cost as carefully as the numbers.
    """

    async def test_every_status_is_present_including_the_zeros(
        self, service: SearchHistoryService, queries: FakeQueryRepository, admin: User
    ) -> None:
        """SQL omits a group with no rows; a tile row needs all of them.

        Asserted because the zero-filling lives in the service while the fake, like
        PostgreSQL, returns only the statuses present — so if the service ever stopped filling
        them this fails instead of the UI quietly losing an outcome.
        """
        await _ask(queries, admin, "answered one", status=QueryStatus.ANSWERED)
        await _ask(queries, admin, "answered two", status=QueryStatus.ANSWERED)
        await _ask(queries, admin, "nothing found", status=QueryStatus.NOT_FOUND)

        stats = await service.history_stats(actor=admin, context=CONTEXT)

        assert set(stats.by_status) == set(QueryStatus)
        assert stats.by_status[QueryStatus.ANSWERED] == 2
        assert stats.by_status[QueryStatus.NOT_FOUND] == 1
        assert stats.by_status[QueryStatus.FAILED] == 0
        assert stats.by_status[QueryStatus.PENDING] == 0
        assert stats.total == 3

    async def test_it_costs_exactly_one_audit_row(
        self,
        service: SearchHistoryService,
        queries: FakeQueryRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        """The whole reason this endpoint exists. One row, marked as the aggregate read."""
        await _ask(queries, admin, "a question")

        before = len(audit_repo.entries)
        await service.history_stats(actor=admin, context=CONTEXT)
        written = audit_repo.entries[before:]

        assert len(written) == 1
        entry = written[0]
        assert entry.action is AuditAction.HISTORY_VIEW
        assert entry.object_type == AuditObjectType.QUERY.value
        assert entry.actor_email == admin.email
        # `phase` is what lets a reviewer separate aggregate reads from page reads, so
        # `HISTORY_VIEW` counts stay interpretable.
        assert entry.detail["phase"] == AUDIT_PHASE_STATS
        assert entry.detail["scope"] == "tenant"
        assert entry.detail["total"] == 1

    async def test_a_standard_user_is_counted_over_their_own_history_only(
        self,
        service: SearchHistoryService,
        queries: FakeQueryRepository,
        analyst: User,
        colleague: User,
    ) -> None:
        """A count is still a disclosure.

        Telling a Standard User how many questions a colleague asked would leak the shape of
        that colleague's activity without showing a single row — so the scoping here has to be
        exactly `list_history`'s, not merely similar to it.
        """
        await _ask(queries, analyst, "mine")
        await _ask(queries, colleague, "theirs one")
        await _ask(queries, colleague, "theirs two")

        stats = await service.history_stats(actor=analyst, context=CONTEXT)

        assert stats.scope == "own"
        assert stats.total == 1

    async def test_a_standard_user_asking_for_a_colleague_gets_their_own_count(
        self,
        service: SearchHistoryService,
        queries: FakeQueryRepository,
        audit_repo: FakeAuditRepository,
        analyst: User,
        colleague: User,
    ) -> None:
        """`user_id` is a filter, not an authorization input — so it is discarded, not refused.

        Refusing would be the wrong shape: there is nothing to bypass, because the scope was
        never the caller's to choose. The attempt is recorded, because from the inside it is
        what an IDOR probe looks like.
        """
        await _ask(queries, analyst, "mine")
        await _ask(queries, colleague, "theirs")

        stats = await service.history_stats(
            actor=analyst, context=CONTEXT, user_id=colleague.id
        )

        assert stats.total == 1
        assert stats.by_status[QueryStatus.ANSWERED] == 1
        entry = audit_repo.entries[-1]
        assert entry.detail["foreign_user_requested"] is True

    async def test_an_administrator_may_count_one_user(
        self,
        service: SearchHistoryService,
        queries: FakeQueryRepository,
        admin: User,
        analyst: User,
        colleague: User,
    ) -> None:
        await _ask(queries, analyst, "one")
        await _ask(queries, analyst, "two")
        await _ask(queries, colleague, "three")

        everyone = await service.history_stats(actor=admin, context=CONTEXT)
        just_analyst = await service.history_stats(
            actor=admin, context=CONTEXT, user_id=analyst.id
        )

        assert everyone.total == 3
        assert just_analyst.total == 2

    async def test_another_tenants_questions_are_never_counted(
        self,
        service: SearchHistoryService,
        queries: FakeQueryRepository,
        admin: User,
    ) -> None:
        """`tenant_id` comes from the actor, never a parameter (`rbac-enforcement.md`)."""
        outsider = _user(uuid.uuid4(), UserRole.ADMINISTRATOR, "admin@other.example")
        await _ask(queries, admin, "ours")
        await _ask(queries, outsider, "theirs one")
        await _ask(queries, outsider, "theirs two")

        stats = await service.history_stats(actor=admin, context=CONTEXT)

        assert stats.total == 1

    async def test_the_keyword_filter_applies_and_is_not_echoed_into_the_trail(
        self,
        service: SearchHistoryService,
        queries: FakeQueryRepository,
        audit_repo: FakeAuditRepository,
        admin: User,
    ) -> None:
        """A history keyword is user text and can name a customer or an account number.

        So the row records *that* one was used, never what it was — same rule as
        `list_history` (`secure-coding-vapt.md`).
        """
        await _ask(queries, admin, "bankruptcy chapter 7")
        await _ask(queries, admin, "infestation policy")

        stats = await service.history_stats(
            actor=admin, context=CONTEXT, keyword="bankruptcy"
        )

        assert stats.total == 1
        entry = audit_repo.entries[-1]
        assert entry.detail["keyword_used"] is True
        assert "bankruptcy" not in str(entry.detail)
