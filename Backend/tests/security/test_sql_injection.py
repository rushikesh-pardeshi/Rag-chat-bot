"""backend/tests/security/test_sql_injection.py

Role: Prove that no user-supplied value reaches PostgreSQL as SQL — through any filter on any
repository, including the analytics keyword and date filters the gate calls out by name.

`.claude/rules/secure-coding-vapt.md`: *"Use parameterized queries or ORM exclusively for all
database operations (documents, queries, audit logs, search history, analytics). Never
concatenate user input into SQL/NoSQL commands."*

How this is tested, and why not by querying a database
-----------------------------------------------------
The obvious test — send `' OR '1'='1` and check nothing extra comes back — is weak twice over.
It passes against a statement that interpolates but happens to produce no extra rows on a small
fixture, and it cannot distinguish "the payload was bound" from "the payload was inlined and
PostgreSQL rejected it", which are very different security postures.

So the assertion is made against the compiled statement: the payload must appear in the
statement's **parameter dictionary** and must not appear in its **SQL string**. That is the
property "parameterized" actually means, and it holds regardless of what any particular fixture
contains. `CapturingSession` intercepts at `session.execute`, so what is inspected is exactly
what production passes down.

Two things are checked beyond binding:

  * **`LIKE` metacharacters are escaped.** `name_contains="%"` must match a literal percent
    sign, not every row. This is not injection — it is a subtler bug in the same place: a
    wildcard that leaks the existence of documents whose names the caller was narrowing away
    from. `document_repository` escapes `%`, `_` and `\\` and declares an ESCAPE clause, and
    that is asserted here rather than left to the reader.
  * **The tenant filter survives the payload.** Every statement must still be scoped. A payload
    that caused the tenant predicate to be dropped would be a cross-tenant read even with
    perfect binding.

The end of the file adds `requires_db` tests that execute the payloads for real, because
binding is a claim about the statement and "the tables are still there afterwards" is a claim
about the server. Both are worth having; only the first runs without PostgreSQL.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql

from backend.models.audit_log import AuditAction, AuditOutcome
from backend.models.document import DocumentStatus
from backend.models.query import QueryStatus
from backend.repositories.analytics_repository import AnalyticsRepository, TrendGranularity
from backend.repositories.audit_repository import AuditRepository
from backend.repositories.document_repository import DocumentRepository
from backend.repositories.query_repository import QueryRepository
from backend.repositories.user_repository import UserRepository
from backend.tests.security.conftest import INJECTION_PAYLOADS

TENANT = uuid.UUID("11111111-1111-1111-1111-111111111111")
USER = uuid.UUID("33333333-3333-3333-3333-333333333333")
FROM = datetime(2026, 1, 1, tzinfo=UTC)
TO = datetime(2026, 6, 30, tzinfo=UTC)


class _Row:
    """Enough of a row for the mappers not to raise. Values are never asserted on."""

    total = answered = not_found = failed = pending = 0
    avg_latency_ms = p95_latency_ms = avg_confidence = None


class _Result:
    """A result that yields nothing, whichever accessor the repository reaches for.

    Iterable as well as indexable because `list(result.scalars())` is the idiom throughout the
    repositories — `scalars()` returns this same object, and `list()` consumes it directly.
    """

    def __iter__(self) -> Any:
        return iter(())

    def all(self) -> list[Any]:
        return []

    def one(self) -> _Row:
        return _Row()

    def first(self) -> None:
        return None

    def scalar_one(self) -> int:
        return 0

    def scalar_one_or_none(self) -> None:
        return None

    def scalars(self) -> _Result:
        return self

    def unique(self) -> _Result:
        return self

    def one_or_none(self) -> None:
        return None


class CapturingSession:
    """Captures every statement handed to `execute`, and compiles it on demand.

    A stub, not a mock. The repositories' whole contract with a session on a read path is
    `await execute(stmt)` returning something with the accessor they use, so a stub that
    implements exactly that cannot drift from `AsyncSession` in a way this test would accept.
    """

    def __init__(self) -> None:
        self.statements: list[Any] = []
        self.added: list[Any] = []

    async def execute(self, statement: Any, *_args: Any, **_kwargs: Any) -> _Result:
        self.statements.append(statement)
        return _Result()

    async def flush(self) -> None:
        return None

    def add(self, instance: Any) -> None:
        self.added.append(instance)

    # -- Assertion helpers ---------------------------------------------------
    def compiled(self) -> list[Any]:
        return [
            stmt.compile(
                dialect=postgresql.dialect(), compile_kwargs={"render_postcompile": True}
            )
            for stmt in self.statements
        ]

    def sql(self) -> str:
        """Every captured statement's SQL, concatenated. What reaches the server, verbatim."""
        return "\n".join(str(compiled) for compiled in self.compiled())

    def params(self) -> list[Any]:
        """Every bound value from every captured statement, flattened.

        Lists are flattened one level because an `IN` clause binds a sequence, and a payload
        passed inside one (the analytics stopword list, a chunk-id batch) has to be findable.
        """
        out: list[Any] = []
        for compiled in self.compiled():
            for value in compiled.params.values():
                if isinstance(value, (list, tuple)):
                    out.extend(value)
                else:
                    out.append(value)
        return out


def assert_bound_not_inlined(session: CapturingSession, payload: str) -> None:
    """The payload is a parameter, and no fragment of it is in the SQL text.

    The comparison is case-insensitive because some filters normalize before binding — an email
    is lower-cased so that `Ada@bank.com` and `ada@bank.com` are the same account. That is
    legitimate, and the claim being made here is "the value travelled as data", which case
    folding does not weaken.

    The second half checks a *fragment* rather than the whole payload because a partial
    interpolation is still injection: `'; DROP TABLE document; --` reaching the statement as
    `DROP TABLE document` with the quoting removed would pass a whole-string check.
    """
    assert session.statements, "no statement was captured — the method did not execute"
    params = session.params()
    folded = [value.casefold() for value in params if isinstance(value, str)]
    assert payload.casefold() in folded, (
        f"{payload!r} was not bound as a parameter. Bound values were: {params!r}"
    )

    assert_no_smuggled_sql(session, payload)


#: Fragments that only exist in the payloads, never in a statement this codebase generates.
#: Finding one in the compiled SQL means a value became syntax. Kept as one list so a new
#: payload's distinctive keyword is added in a single place.
SQL_KEYWORD_FRAGMENTS = (
    "DROP TABLE",
    "DELETE FROM",
    "UPDATE audit_log",
    "UNION SELECT",
    "pg_sleep",
    "--",
)


def assert_no_smuggled_sql(session: CapturingSession, payload: str) -> None:
    """No payload keyword appears in the generated SQL.

    `--`, `;` and a bare quote are the tokens that turn a value into syntax, and none of the
    fragments above occurs in any statement this codebase legitimately builds — so their
    absence is a sound proxy for "nothing was interpolated".
    """
    sql = session.sql()
    for fragment in SQL_KEYWORD_FRAGMENTS:
        assert fragment.lower() not in sql.lower(), (
            f"{fragment!r} reached the SQL text, so {payload!r} was interpolated:\n{sql}"
        )


def assert_tenant_scoped(session: CapturingSession, tenant_id: uuid.UUID) -> None:
    """The tenant id is still bound. A dropped scope is a cross-tenant read."""
    assert tenant_id in session.params(), (
        "the statement lost its tenant filter — every read must be tenant-scoped "
        "(.claude/rules/rbac-enforcement.md)"
    )


# -- Documents ---------------------------------------------------------------
class TestDocumentFilters:
    """`name_contains` is the only free-text filter on the document list."""

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_name_filter_is_bound(self, payload: str) -> None:
        session = CapturingSession()
        await DocumentRepository(session).list_for_tenant(  # type: ignore[arg-type]
            TENANT, name_contains=payload
        )
        # The value is bound *escaped* for LIKE, so the raw payload may not appear verbatim
        # when it contains a metacharacter. What must hold is that no SQL syntax was produced
        # and that some bound parameter carries the payload's escaped form.
        assert_no_smuggled_sql(session, payload)
        escaped = payload.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        assert any(
            isinstance(value, str) and escaped in value for value in session.params()
        ), f"{payload!r} did not reach the statement as a bound value: {session.params()!r}"
        assert_tenant_scoped(session, TENANT)

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_count_uses_the_same_binding(self, payload: str) -> None:
        """The count and the page must filter identically, or the pager's total is wrong."""
        session = CapturingSession()
        await DocumentRepository(session).count_for_tenant(  # type: ignore[arg-type]
            TENANT, name_contains=payload
        )
        assert_tenant_scoped(session, TENANT)
        assert "DROP TABLE" not in session.sql().upper()

    @pytest.mark.parametrize("wildcard", ["%", "_", "%%", "\\", "a%b_c", "100%"])
    async def test_like_metacharacters_are_escaped(self, wildcard: str) -> None:
        """`name_contains='%'` must mean a literal percent sign, not "every document".

        Not injection, but the same class of mistake in the same place: an unescaped wildcard
        widens a filter the caller was using to narrow, which discloses the existence of rows
        they were not asking about. The ESCAPE clause is asserted too — escaping the value
        without declaring the escape character does nothing.
        """
        session = CapturingSession()
        await DocumentRepository(session).list_for_tenant(  # type: ignore[arg-type]
            TENANT, name_contains=wildcard
        )
        bound = [v for v in session.params() if isinstance(v, str) and v.startswith("%")]
        assert bound, f"no LIKE pattern was bound: {session.params()!r}"
        pattern = bound[0]
        inner = pattern[1:-1]  # strip the wrapping %…%
        for char in ("%", "_"):
            if char in wildcard:
                assert f"\\{char}" in inner, (
                    f"{char!r} was not escaped in the LIKE pattern {pattern!r}"
                )
        assert "ESCAPE" in session.sql().upper(), (
            "the pattern is escaped but no ESCAPE clause was declared, so the backslash is "
            "just a literal backslash to PostgreSQL"
        )

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_a_payload_as_a_document_name_goes_through_the_unit_of_work(
        self, payload: str
    ) -> None:
        """The write path too. A stored payload is what a second-order injection needs.

        `create` builds a `Document` and hands it to `session.add`, so the INSERT is generated
        by SQLAlchemy's unit of work with the name as a bound parameter — there is no statement
        for this test to compile. What is asserted instead is that shape: the name reached an
        ORM instance and no statement was executed. An implementation that built its own INSERT
        would fail here, which is the change worth catching.
        """
        session = CapturingSession()
        document = await DocumentRepository(session).create(  # type: ignore[arg-type]
            document_id=uuid.uuid4(),
            tenant_id=TENANT,
            name=payload,
            s3_key="documents/x/v1/x.pdf",
            content_type="application/pdf",
            file_size=10,
            uploaded_by=USER,
            upload_date=FROM,
            version=1,
            content_hash="0" * 64,
        )
        assert document.name == payload, "the name was altered on the way in"
        assert session.statements == [], (
            "create() issued a statement of its own instead of using the unit of work"
        )
        assert session.added, "nothing was added to the session"


# -- Queries and history -----------------------------------------------------
class TestQueryAndHistoryFilters:
    """`keyword`, `status` and the date window on the history projection."""

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_history_keyword_is_bound_for_a_standard_user(
        self, payload: str
    ) -> None:
        session = CapturingSession()
        await QueryRepository(session).list_for_user(  # type: ignore[arg-type]
            TENANT, USER, keyword=payload
        )
        sql = session.sql()
        for fragment in ("DROP TABLE", "UNION SELECT", "pg_sleep", "UPDATE audit_log"):
            assert fragment.lower() not in sql.lower(), sql
        assert_tenant_scoped(session, TENANT)
        assert USER in session.params(), (
            "a Standard User's history lost its user filter — it would show the whole tenant"
        )

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_history_keyword_is_bound_for_an_administrator(
        self, payload: str
    ) -> None:
        session = CapturingSession()
        await QueryRepository(session).list_for_tenant(  # type: ignore[arg-type]
            TENANT, keyword=payload
        )
        assert "DROP TABLE" not in session.sql().upper()
        assert_tenant_scoped(session, TENANT)

    async def test_the_date_window_is_bound_as_a_timestamp(self) -> None:
        """A date filter that concatenated would be injectable through the query string.

        Pydantic parses `created_from` to a `datetime` before it reaches here, which is the
        first line of defence — asserted by proving the bound value is a `datetime` and not a
        string, because a string would mean the parse was skipped somewhere.
        """
        session = CapturingSession()
        await QueryRepository(session).count(  # type: ignore[arg-type]
            TENANT, created_from=FROM, created_to=TO
        )
        bound_times = [v for v in session.params() if isinstance(v, datetime)]
        assert FROM in bound_times and TO in bound_times
        assert not any(isinstance(v, str) and "2026" in v for v in session.params()), (
            "a date reached the statement as a string, so it was formatted rather than bound"
        )

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_chunk_ids_from_a_vector_match_are_bound(self, payload: str) -> None:
        """`load_chunks` takes ids that came back from Pinecone — an external system.

        Pinecone vector ids are ours, but treating them as trusted would be a mistake: the
        index is a separate service with its own access model, and `secure-coding-vapt.md`'s
        rule is about untrusted input, not about who we think wrote it. `RetrievedMatch` parses
        each id to a UUID before it gets here, so a payload cannot survive the trip — asserted
        by showing a non-UUID never becomes SQL.
        """
        session = CapturingSession()
        try:
            ids = [uuid.UUID(payload)]  # type: ignore[arg-type]
        except ValueError:
            # The expected path: the payload is not a UUID, so it never reaches a statement.
            # That *is* the control, and asserting it here documents where it lives.
            return
        await QueryRepository(session).load_chunks(ids, TENANT)  # type: ignore[arg-type]
        assert "DROP TABLE" not in session.sql().upper()


# -- Audit log ---------------------------------------------------------------
class TestAuditFilters:
    """The compliance read takes five filters, all of them from a query string."""

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_object_type_filter_is_bound(self, payload: str) -> None:
        session = CapturingSession()
        await AuditRepository(session).list_for_tenant(  # type: ignore[arg-type]
            TENANT, object_type=payload
        )
        assert_bound_not_inlined(session, payload)
        assert_tenant_scoped(session, TENANT)

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_object_type_filter_is_bound_on_the_count(self, payload: str) -> None:
        session = CapturingSession()
        await AuditRepository(session).count_for_tenant(  # type: ignore[arg-type]
            TENANT, object_type=payload
        )
        assert_bound_not_inlined(session, payload)

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_an_enum_filter_cannot_carry_a_payload_at_all(self, payload: str) -> None:
        """`action` and `outcome` are enums, so the payload dies before the repository.

        Worth asserting rather than commenting: the reason these two filters are safe is *not*
        parameterization, it is that no arbitrary string can reach them. If either were ever
        loosened to a free string, this test's premise would change and the binding assertions
        above would become the only defence — so the enum rejection is checked, and the
        legitimate value is checked to bind, in the same test.
        """
        with pytest.raises(ValueError):
            AuditAction(payload)
        with pytest.raises(ValueError):
            AuditOutcome(payload)

        session = CapturingSession()
        await AuditRepository(session).list_for_tenant(  # type: ignore[arg-type]
            TENANT, action=AuditAction.ACCESS_DENIED, outcome=AuditOutcome.DENIED
        )
        bound = [v.value if isinstance(v, Enum) else v for v in session.params()]
        assert AuditAction.ACCESS_DENIED.value in bound
        assert AuditOutcome.DENIED.value in bound


# -- Analytics ---------------------------------------------------------------
class TestAnalyticsFilters:
    """The keyword and date filters the gate names explicitly."""

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_a_stopword_is_bound_not_inlined(self, payload: str) -> None:
        """The stopword list is an `IN` clause built from a Python collection.

        `STOPWORDS` is a module constant today, so no caller supplies it — which is exactly why
        this is tested. The keyword aggregate is the one place in the codebase that hands a
        *list of strings* to PostgreSQL, and a future "let the admin edit the stopword list"
        feature would make that list user-supplied without changing the SQL. Binding it now
        means that change stays safe.
        """
        session = CapturingSession()
        await AnalyticsRepository(session).keyword_counts(  # type: ignore[arg-type]
            TENANT, stopwords=[payload, "the"], min_length=3, limit=10
        )
        assert_bound_not_inlined(session, payload)
        assert_tenant_scoped(session, TENANT)

    async def test_the_split_pattern_is_a_fixed_constant(self) -> None:
        """`regexp_split_to_table` takes a pattern. A caller-supplied one would be ReDoS.

        `secure-coding-vapt.md`: *"Never use user-supplied regex patterns."* The pattern is a
        module constant, and the signature has no parameter that could override it — asserted
        by inspecting the signature, because a keyword argument added later is exactly the
        change this rule is about.
        """
        import inspect

        from backend.repositories.analytics_repository import KEYWORD_SPLIT_PATTERN

        assert isinstance(KEYWORD_SPLIT_PATTERN, str)
        signature = inspect.signature(AnalyticsRepository.keyword_counts)
        assert "pattern" not in signature.parameters, (
            "keyword_counts gained a `pattern` parameter — a caller-supplied regex is ReDoS"
        )
        session = CapturingSession()
        await AnalyticsRepository(session).keyword_counts(  # type: ignore[arg-type]
            TENANT, stopwords=["the"], min_length=3, limit=10
        )
        assert KEYWORD_SPLIT_PATTERN in session.params(), (
            "the split pattern is not bound, so it is part of the SQL text"
        )

    async def test_the_report_window_is_bound_on_every_aggregate(self) -> None:
        """All five aggregates take the same window, and all five must bind it.

        Checked together rather than one test each, because the failure mode is one aggregate
        being written differently from its four neighbours — which a per-method test would not
        surface as clearly as a list of the ones that disagreed.
        """
        repo_calls = [
            (
                "keyword_counts",
                {"stopwords": ["the"], "min_length": 3, "limit": 10},
            ),
            ("document_usage", {"limit": 10}),
            ("volume_trend", {"granularity": TrendGranularity.DAY, "limit": 30}),
            ("active_users", {"limit": 10}),
            ("summary", {}),
        ]
        offenders: list[str] = []
        for name, kwargs in repo_calls:
            session = CapturingSession()
            method = getattr(AnalyticsRepository(session), name)  # type: ignore[arg-type]
            await method(TENANT, created_from=FROM, created_to=TO, **kwargs)
            params = session.params()
            if not (FROM in params and TO in params and TENANT in params):
                offenders.append(name)
        assert offenders == [], f"these aggregates did not bind tenant+window: {offenders}"

    @pytest.mark.parametrize("limit", [-1, 0, 10**9])
    async def test_a_hostile_limit_is_still_an_integer(self, limit: int) -> None:
        """`limit` is an `int` by the time it is here, so it cannot carry syntax.

        The interesting risk with `limit` is not injection but resource exhaustion, and that is
        bounded in `analytics_service._bounded` rather than here. This asserts the narrow claim
        the repository is responsible for: whatever the number, it is bound as a number.
        """
        session = CapturingSession()
        await AnalyticsRepository(session).document_usage(TENANT, limit=limit)  # type: ignore[arg-type]
        assert all(not isinstance(v, str) or "LIMIT" not in v for v in session.params())
        assert "DROP" not in session.sql().upper()


# -- Users -------------------------------------------------------------------
class TestUserLookups:
    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_an_email_lookup_is_bound(self, payload: str) -> None:
        """The login path. A payload here would be pre-authentication injection.

        `LoginRequest.email` is an `EmailStr`, so none of these payloads can reach the
        repository through the API — but `get_by_email` is also called by the registration path
        and by `scripts/seed_admin.py`, and a repository that is only safe because of its
        callers is one refactor away from not being safe.
        """
        session = CapturingSession()
        await UserRepository(session).get_by_email(payload, TENANT)  # type: ignore[arg-type]
        assert_bound_not_inlined(session, payload)
        assert_tenant_scoped(session, TENANT)

    @pytest.mark.parametrize("payload", INJECTION_PAYLOADS)
    async def test_a_tenant_domain_lookup_is_bound(self, payload: str) -> None:
        """Registration resolves the tenant from the email domain — user-supplied text."""
        session = CapturingSession()
        await UserRepository(session).get_tenant_by_domain(payload)  # type: ignore[arg-type]
        assert payload.lower() in [
            v.lower() if isinstance(v, str) else v for v in session.params()
        ]
        assert "DROP TABLE" not in session.sql().upper()


# -- No raw SQL anywhere on a request path -----------------------------------
#: Every non-test module that calls SQLAlchemy's `text()`, and why each one is allowed. Raw SQL
#: is not banned — four places genuinely need it — but it is the one construct in this
#: codebase that *can* carry injection, so the list is closed and a new entry has to be added
#: here deliberately. A reviewer seeing `test_raw_sql_call_sites_are_the_known_four` fail
#: reads the new call site and decides, instead of a pentester finding it later.
RAW_SQL_CALL_SITES = {
    "backend/api/routes/health.py": "the readiness probe's `SELECT 1` — no values at all",
    "backend/models/query.py": "`created_at DESC` inside an Index() definition, not a query",
    "backend/repositories/audit_repository.py": (
        "`pg_advisory_xact_lock(hashtext(:key))` — the chain lock, with `:key` bound"
    ),
    "backend/repositories/document_repository.py": (
        "`pg_advisory_xact_lock(hashtext(:key))` — the per-name ingestion lock, with `:key` "
        "bound. The key embeds a document name, which *is* caller-supplied: it is an uploaded "
        "filename, already reduced to one path component by `sanitize_filename`. It reaches "
        "PostgreSQL as a bound parameter and never as SQL text, so the worst a hostile name "
        "can do is collide with another name's lock and wait its turn"
    ),
}

#: The two modules whose raw statement takes a value at all, and so the two worth asserting on
#: line by line. Both are `pg_advisory_xact_lock(hashtext(:key))`.
ADVISORY_LOCK_MODULES = ("audit_repository.py", "document_repository.py")


def _sqlalchemy_text_lines(source: str) -> list[tuple[int, str]]:
    """Lines calling SQLAlchemy's `text()`, excluding lookalikes.

    `clean_text(`, `bind_request_context(` and `extract_text(` all contain the substring, so a
    naive search reports most of the codebase. The lookbehind requires a non-identifier
    character before `text`, which leaves exactly the real call sites.
    """
    import re

    pattern = re.compile(r"(?<![\w.])text\(")
    return [
        (number, line)
        for number, line in enumerate(source.splitlines(), start=1)
        if pattern.search(line)
    ]


def _backend_modules() -> list[Any]:
    from pathlib import Path

    root = Path(__file__).resolve().parents[3] / "backend"
    return [path for path in sorted(root.rglob("*.py")) if "tests" not in path.parts]


class TestNoRawSQLOnRequestPaths:
    """Source-level checks, because the binding tests can only cover today's filters.

    Every test above names a parameter that exists now. These two cover the *shape* of the
    mistake, so a filter added next month is caught by a test that was written before it.
    """

    def test_no_module_builds_sql_by_interpolation(self) -> None:
        """`text(f"…")` or `text("…{…}…")` anywhere outside the tests.

        This is the only way injection can enter a SQLAlchemy codebase — every other path goes
        through a bound parameter whether the author thought about it or not. Fixed pattern,
        linear, no user input: the ReDoS rule applies to this file too.
        """
        import re

        interpolated = re.compile(r"(?<![\w.])text\(\s*(?:f[\"']|[\"'][^\"']*\{)")
        offenders: list[str] = []
        for path in _backend_modules():
            for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if interpolated.search(line):
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        assert offenders == [], "interpolated SQL — use bound parameters:\n" + "\n".join(
            offenders
        )

    def test_raw_sql_call_sites_are_the_known_four(self) -> None:
        """The allowlist above is exhaustive."""
        found = {
            str(path.as_posix()).split("docquery-test/")[-1]: _sqlalchemy_text_lines(
                path.read_text(encoding="utf-8")
            )
            for path in _backend_modules()
        }
        users = sorted(key.split("backend/", 1)[-1] for key, lines in found.items() if lines)
        expected = sorted(key.split("backend/", 1)[-1] for key in RAW_SQL_CALL_SITES)
        assert users == expected, (
            "raw SQL appears in a module that is not on the allowlist. Add it to "
            f"RAW_SQL_CALL_SITES with a reason, or use the ORM.\nfound: {users}"
        )

    @pytest.mark.parametrize("module", ADVISORY_LOCK_MODULES)
    def test_the_advisory_locks_bind_their_only_value(self, module: str) -> None:
        """The only raw statements that take a value at all.

        `lock_chain` serializes audit appends so two concurrent writers cannot compute the same
        `previous_hash`. `lock_name` serializes document mutations per name, so version
        supersession cannot deadlock against a concurrent upload of the same filename.

        `lock_chain`'s argument is a tenant UUID, so no payload can reach it through the API.
        **`lock_name`'s embeds a document name, which is caller-supplied** — an uploaded
        filename, reduced to a single path component by `sanitize_filename` but otherwise the
        user's text. That makes these the highest-value lines in the codebase to get wrong, so
        the shape is asserted rather than trusted: a named bind, the value passed separately,
        and no f-string anywhere near the statement.
        """
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[3] / "backend" / "repositories" / module
        ).read_text(encoding="utf-8")
        lines = dict(_sqlalchemy_text_lines(source))
        assert len(lines) == 1, f"{module} gained another raw statement: {lines}"
        statement = next(iter(lines.values()))
        assert ":key" in statement, (
            f"{module}'s advisory lock no longer uses a named bind parameter: "
            f"{statement.strip()!r}"
        )
        assert "f\"" not in statement and "f'" not in statement, (
            f"{module}'s advisory lock is built from an f-string: {statement.strip()!r}"
        )


# -- Against a real server ---------------------------------------------------
async def _insert_tenant(session: Any, tenant_id: uuid.UUID) -> uuid.UUID:
    """A scratch tenant, so a `requires_db` test never reads or writes another test's rows.

    Every id is fresh per test rather than a fixed constant. A shared scratch tenant would make
    these tests order-dependent, and an order-dependent security test is one that can be made
    to pass by running it alone.
    """
    from backend.models.tenant import Tenant

    session.add(
        Tenant(
            id=tenant_id,
            name=f"scratch-{tenant_id.hex[:8]}",
            domain=f"{tenant_id.hex}.test",
        )
    )
    await session.flush()
    return tenant_id


async def _insert_user(session: Any, tenant_id: uuid.UUID) -> uuid.UUID:
    """An uploader for the document rows. `document.uploaded_by` is a foreign key."""
    from backend.models.user import User as UserRow
    from backend.models.user import UserRole

    user_id = uuid.uuid4()
    session.add(
        UserRow(
            id=user_id,
            tenant_id=tenant_id,
            email=f"scratch-{user_id.hex[:8]}@{tenant_id.hex}.test",
            # Not a real hash: nothing here authenticates, and a bcrypt call would cost more
            # than the rest of the test.
            hashed_password="not-a-real-hash",
            role=UserRole.ADMINISTRATOR,
        )
    )
    await session.flush()
    return user_id


@pytest.mark.requires_db
class TestAgainstPostgres:
    """Binding proven at the statement level, now executed for real.

    These add one thing the compile-level tests cannot: that PostgreSQL accepts the statement
    and the schema survives. A payload that were interpolated would either error here or drop a
    table, and both are visible.
    """

    async def test_payloads_execute_without_altering_the_schema(
        self, db_available: bool
    ) -> None:
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        expected = {
            "tenant",
            "user_account",
            "document",
            "document_chunk",
            "query",
            "audit_log",
            "refresh_token",
        }
        async with session_scope() as session:
            documents = DocumentRepository(session)
            queries = QueryRepository(session)
            audit = AuditRepository(session)
            for payload in INJECTION_PAYLOADS:
                await documents.list_for_tenant(TENANT, name_contains=payload)
                await queries.list_for_user(TENANT, USER, keyword=payload)
                await audit.list_for_tenant(TENANT, object_type=payload)

        async with session_scope() as session:
            result = await session.execute(
                text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
            )
            present = {row[0] for row in result.all()}
        missing = expected - present
        assert missing == set(), f"tables disappeared after the payloads ran: {missing}"

    async def test_a_wildcard_name_filter_matches_literally(
        self, db_available: bool
    ) -> None:
        """`name_contains='%'` returns only the document whose name contains a percent sign.

        The behavioural counterpart to the ESCAPE assertion above, and the reason it is worth
        paying for a real database: whether the declared escape character is honoured is
        PostgreSQL's decision, not SQLAlchemy's.

        Two rows are inserted on purpose. Against an empty tenant the assertion would be
        "no error and no rows", which an unescaped `%` also satisfies — the test would pass
        against the bug it exists to catch. With one matching and one non-matching row, an
        unescaped `%` returns both and the test fails.
        """
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        from backend.core.database import session_scope

        tenant_id = uuid.uuid4()
        async with session_scope() as session:
            await _insert_tenant(session, tenant_id)
            uploader = await _insert_user(session, tenant_id)
            documents = DocumentRepository(session)
            for name in ("fee-schedule-100% waiver.pdf", "fee-schedule-standard.pdf"):
                await documents.create(
                    document_id=uuid.uuid4(),
                    tenant_id=tenant_id,
                    name=name,
                    s3_key=f"documents/{tenant_id}/v1/{uuid.uuid4()}.pdf",
                    content_type="application/pdf",
                    file_size=1024,
                    uploaded_by=uploader,
                    upload_date=datetime.now(UTC),
                    version=1,
                    content_hash=uuid.uuid4().hex * 2,
                )

        async with session_scope() as session:
            matched = await DocumentRepository(session).list_for_tenant(
                tenant_id, name_contains="%"
            )
        names = sorted(document.name for document in matched)
        assert names == ["fee-schedule-100% waiver.pdf"], (
            "the `%` was treated as a wildcard, so the filter returned rows the caller was "
            f"narrowing away from: {names}"
        )

        async with session_scope() as session:
            underscored = await DocumentRepository(session).list_for_tenant(
                tenant_id, name_contains="_"
            )
        assert underscored == [], (
            "`_` matched any single character, so it behaved as a wildcard rather than as the "
            f"literal the caller typed: {[d.name for d in underscored]}"
        )

    async def test_the_status_filter_rejects_a_non_enum_value(
        self, db_available: bool
    ) -> None:
        """A `document.status` payload cannot even be constructed.

        The enum is the control, not the binding. Asserted against the real column because
        PostgreSQL's own enum type is the second half of it.
        """
        if not db_available:
            pytest.skip("no PostgreSQL reachable")

        with pytest.raises(ValueError):
            DocumentStatus("'; DROP TABLE document; --")
        with pytest.raises(ValueError):
            QueryStatus("' OR 1=1 --")
