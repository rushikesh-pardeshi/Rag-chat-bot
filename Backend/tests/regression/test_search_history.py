"""backend/tests/regression/test_search_history.py

Role: BR-04 (Search History) asserted over HTTP.

The requirement, quoted from `WORKBREAKDOWN.md`:

    Search History: Store all queries with user identity, question, answer, timestamp,
    associated source documents; standard users view own history, administrators view
    system-wide (BR-SH, BR-04)

"System-wide" is scoped, and that is not a liberty taken here
------------------------------------------------------------
An Administrator sees every row **in their own tenant**, never another organization's.
`rbac-enforcement.md` is explicit that role and tenant are independent dimensions and both
apply on every request, so "system-wide" in a multi-tenant deployment can only mean
tenant-wide. The tests below assert both halves: an Administrator sees a colleague's rows, and
does not see tenant B's.

The `total` is part of the authorization boundary
-------------------------------------------------
A page of 25 rows with `total: 400` tells a Standard User how much history exists that they
cannot see. `search_history_service` counts with the same scope it lists with, and
`test_the_total_is_scoped_like_the_page` is the test that keeps it that way — it is the one
assertion here that a reviewer is most likely to think is cosmetic.

There is no second table
------------------------
History is the `query` table read with a filter (`architecture.md`), so a row appearing in
history and a row read through `GET /queries/{id}` are the same row. `test_history_rows_are_the
_query_rows` asserts that identity rather than trusting it.
"""

from __future__ import annotations

import uuid
from typing import Any

from backend.models.query import QueryStatus
from backend.services.search_history_service import (
    DEFAULT_PAGE_SIZE,
    MAX_KEYWORD_LENGTH,
    MAX_PAGE_SIZE,
)
from backend.tests.regression.conftest import (
    API,
    OFF_CORPUS_QUESTION,
    TAX_QUESTION,
    WIRE_QUESTION,
    Api,
    Stack,
    audit_rows,
    csrf_headers,
    new_uuid,
    switch_to,
    upload,
)

HISTORY = f"{API}/history"


async def ask(client: Any, question: str) -> Any:
    """Submit one question. Queue the answer on `stack.bedrock` first for an `ANSWERED` row."""
    return await client.post(
        f"{API}/queries", json={"question_text": question}, headers=csrf_headers()
    )


def stored(stack: Stack, row: dict[str, Any]) -> Any:
    """The `Query` behind a published history row.

    The API publishes ids as strings and the repository is keyed by `UUID`, and the conversion
    appears often enough below that the intent gets lost in it.
    """
    return stack.queries.queries[uuid.UUID(row["id"])]


async def seed_history(api: Api, stack: Stack) -> dict[str, Any]:
    """One answered query for tenant A's Standard User and one for its Administrator.

    Returns both response bodies, because most tests here need an id belonging to *someone
    else* — the interesting case for BR-04 is not "can I see my own row" but "can I see a
    colleague's", and only an Administrator may.
    """
    admin_client = switch_to(api, stack.a.admin)
    await upload(admin_client)

    user_client = switch_to(api, stack.a.user)
    stack.bedrock.queue_answer("Withholding tax is 10 percent [S1].")
    mine = (await ask(user_client, TAX_QUESTION)).json()

    admin_client = switch_to(api, stack.a.admin)
    stack.bedrock.queue_answer("The wire fee is 25 USD [S1].")
    theirs = (await ask(admin_client, WIRE_QUESTION)).json()

    return {"user_row": mine, "admin_row": theirs}


def ids(body: dict[str, Any]) -> list[str]:
    return [item["id"] for item in body["items"]]


class TestAStandardUserSeesOwnHistoryOnly:
    """"standard users view own history" — the clause with a breach behind it."""

    async def test_own_answered_query_appears(self, api: Api, stack: Stack) -> None:
        seeded = await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        response = await client.get(HISTORY)

        assert response.status_code == 200
        assert seeded["user_row"]["id"] in ids(response.json())

    async def test_a_colleagues_query_does_not_appear(
        self, api: Api, stack: Stack
    ) -> None:
        seeded = await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        body = (await client.get(HISTORY)).json()

        assert seeded["admin_row"]["id"] not in ids(body)

    async def test_the_scope_is_labelled_own(self, api: Api, stack: Stack) -> None:
        """The UI needs this to say "your searches" rather than implying the list is
        tenant-wide. It is returned, so it is asserted."""
        client = switch_to(api, stack.a.user)

        assert (await client.get(HISTORY)).json()["scope"] == "own"

    async def test_the_total_is_scoped_like_the_page(
        self, api: Api, stack: Stack
    ) -> None:
        """`total` counts what the caller may see, not what exists.

        A count taken with a wider scope than the list leaks the size of the tenant's activity
        to someone authorized for one user's slice of it.
        """
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        body = (await client.get(HISTORY)).json()

        assert body["total"] == len(body["items"])
        own = [
            query
            for query in stack.queries.queries.values()
            if query.user_id == stack.a.user.id
        ]
        assert body["total"] == len(own)

    async def test_asking_for_another_users_history_returns_own(
        self, api: Api, stack: Stack
    ) -> None:
        """`user_id` was never an authorization input, so there is nothing to bypass — the
        Standard User's own rows come back. Not a 403: a 403 would confirm the parameter does
        something for someone."""
        seeded = await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        body = (await client.get(f"{HISTORY}?user_id={stack.a.admin.id}")).json()

        assert body["scope"] == "own"
        assert seeded["admin_row"]["id"] not in ids(body)
        assert seeded["user_row"]["id"] in ids(body)

    async def test_the_ignored_filter_is_recorded_in_the_audit_trail(
        self, api: Api, stack: Stack
    ) -> None:
        """Not an error, but it is what an IDOR probe looks like from the inside, so the audit
        row carries `foreign_user_requested`."""
        client = switch_to(api, stack.a.user)

        await client.get(f"{HISTORY}?user_id={stack.a.admin.id}")

        rows = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.user.email)
        assert rows
        assert rows[-1].detail["foreign_user_requested"] is True

    async def test_a_plain_read_is_not_flagged_as_a_probe(
        self, api: Api, stack: Stack
    ) -> None:
        """The other half of the previous test. A flag that is always true says nothing."""
        client = switch_to(api, stack.a.user)

        await client.get(HISTORY)

        rows = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.user.email)
        assert rows[-1].detail["foreign_user_requested"] is False

    async def test_supplying_own_id_is_not_a_probe(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        await client.get(f"{HISTORY}?user_id={stack.a.user.id}")

        rows = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.user.email)
        assert rows[-1].detail["foreign_user_requested"] is False


class TestAnAdministratorSeesTheTenant:
    """"administrators view system-wide" — bounded by tenant, per `rbac-enforcement.md`."""

    async def test_a_colleagues_query_appears(self, api: Api, stack: Stack) -> None:
        seeded = await seed_history(api, stack)
        client = switch_to(api, stack.a.admin)

        listed = ids((await client.get(HISTORY)).json())

        assert seeded["user_row"]["id"] in listed
        assert seeded["admin_row"]["id"] in listed

    async def test_the_scope_is_labelled_tenant(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(HISTORY)).json()["scope"] == "tenant"

    async def test_another_tenants_history_never_appears(
        self, api: Api, stack: Stack
    ) -> None:
        """The seeded stack answers a query for tenant B too, so this is a real row belonging
        to a real user in another organization — the id an attacker actually obtains."""
        assert stack.b.query is not None
        await seed_history(api, stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(HISTORY)).json()

        assert str(stack.b.query.id) not in ids(body)
        assert all(stored(stack, item).tenant_id == stack.a.id for item in body["items"])

    async def test_the_user_id_filter_narrows_to_one_user(
        self, api: Api, stack: Stack
    ) -> None:
        seeded = await seed_history(api, stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(f"{HISTORY}?user_id={stack.a.user.id}")).json()

        assert seeded["user_row"]["id"] in ids(body)
        assert seeded["admin_row"]["id"] not in ids(body)

    async def test_filtering_by_another_tenants_user_returns_nothing(
        self, api: Api, stack: Stack
    ) -> None:
        """An Administrator naming tenant B's user id gets an empty page, not tenant B's rows —
        the filter is applied *within* a tenant-scoped query, never instead of one."""
        await seed_history(api, stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(f"{HISTORY}?user_id={stack.b.user.id}")).json()

        assert body["items"] == []
        assert body["total"] == 0


class TestEveryRowCarriesWhatTheRequirementNames:
    """"user identity, question, answer, timestamp, associated source documents"."""

    async def test_the_question_and_answer_are_both_stored(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        row = next(
            item
            for item in (await client.get(HISTORY)).json()["items"]
            if item["question_text"] == TAX_QUESTION
        )

        assert row["answer_text"]
        assert row["status"] == QueryStatus.ANSWERED.value

    async def test_the_row_carries_its_source_documents(
        self, api: Api, stack: Stack
    ) -> None:
        """"associated source documents" — an answered history row that lost its citations
        would be untraceable months later, which is exactly when it is read."""
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        row = (await client.get(HISTORY)).json()["items"][0]

        assert row["source_references"]
        for reference in row["source_references"]:
            assert reference["source_document_name"]
            assert reference["section_or_page"]

    async def test_the_timestamp_is_present(self, api: Api, stack: Stack) -> None:
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        row = (await client.get(HISTORY)).json()["items"][0]

        assert row["created_at"]
        assert row["answered_at"]

    async def test_user_identity_is_the_scope_not_a_published_field(
        self, api: Api, stack: Stack
    ) -> None:
        """Identity is recorded — on the row, and in the `HISTORY_VIEW` audit entry — but
        `QueryResponse` forbids extra fields and publishes no `user_id`. An Administrator who
        needs to know whose row it is filters by `user_id`; the answer text itself does not
        carry a colleague's identity into a shared screen."""
        await seed_history(api, stack)
        client = switch_to(api, stack.a.admin)

        row = (await client.get(HISTORY)).json()["items"][0]

        assert "user_id" not in row
        assert stored(stack, row).user_id in {stack.a.user.id, stack.a.admin.id}

    async def test_history_rows_are_the_query_rows(
        self, api: Api, stack: Stack
    ) -> None:
        """No second table (`architecture.md`): the same id read two ways is the same content.
        Two copies of an audited fact are free to drift, and a compliance review that finds
        them disagreeing cannot tell which one the user saw."""
        seeded = await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        listed = next(
            item
            for item in (await client.get(HISTORY)).json()["items"]
            if item["id"] == seeded["user_row"]["id"]
        )
        fetched = (await client.get(f"{API}/queries/{seeded['user_row']['id']}")).json()

        assert listed == fetched


class TestFilteringAndPaging:
    """The filters the UI ships with. Each one is asserted to *exclude* something, because a
    filter that returns everything looks like it works."""

    async def test_a_status_filter_narrows_the_page(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)
        await ask(client, OFF_CORPUS_QUESTION)  # a NOT_FOUND row

        answered = (await client.get(f"{HISTORY}?status=ANSWERED")).json()
        not_found = (await client.get(f"{HISTORY}?status=NOT_FOUND")).json()

        assert answered["items"]
        assert not_found["items"]
        assert set(ids(answered)).isdisjoint(ids(not_found))
        assert all(
            item["status"] == QueryStatus.ANSWERED.value for item in answered["items"]
        )

    async def test_an_unknown_status_is_422(self, api: Api, stack: Stack) -> None:
        """Typed as `QueryStatus`, so FastAPI rejects it before a string reaches the
        repository."""
        client = switch_to(api, stack.a.user)

        assert (await client.get(f"{HISTORY}?status=DELETED")).status_code == 422

    async def test_a_keyword_matches_the_question_text(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("The wire fee is 25 USD [S1].")
        await ask(client, WIRE_QUESTION)

        body = (await client.get(f"{HISTORY}?keyword=withholding")).json()

        assert body["items"]
        assert all(
            "withholding" in item["question_text"].lower() for item in body["items"]
        )

    async def test_a_keyword_that_matches_nothing_returns_an_empty_page(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        body = (await client.get(f"{HISTORY}?keyword=zzznotpresentzzz")).json()

        assert body["items"] == []
        assert body["total"] == 0

    async def test_a_wildcard_keyword_matches_itself(
        self, api: Api, stack: Stack
    ) -> None:
        """`%` is escaped in the repository's `ILIKE`, so it is a literal percent sign and not
        "match everything" (`secure-coding-vapt.md`, SQL injection)."""
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        body = (await client.get(f"{HISTORY}?keyword=%")).json()

        assert body["items"] == []

    async def test_an_overlong_keyword_is_422(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)
        oversized = "a" * (MAX_KEYWORD_LENGTH + 1)

        assert (await client.get(f"{HISTORY}?keyword={oversized}")).status_code == 422

    async def test_the_keyword_text_is_not_written_to_the_audit_trail(
        self, api: Api, stack: Stack
    ) -> None:
        """A history keyword is user text and can name a customer or an account number. The
        audit row records *that* a keyword was used, never which one."""
        client = switch_to(api, stack.a.user)

        await client.get(f"{HISTORY}?keyword=ACCT-99881122")

        written = stack.audit_details()
        assert "ACCT-99881122" not in written
        rows = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.user.email)
        assert rows[-1].detail["keyword_used"] is True

    async def test_the_default_page_size_is_the_documented_one(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        assert (await client.get(HISTORY)).json()["limit"] == DEFAULT_PAGE_SIZE

    async def test_an_oversized_limit_is_refused_at_the_boundary(
        self, api: Api, stack: Stack
    ) -> None:
        """`MAX_PAGE_SIZE` is a denial-of-service control: history rows carry full answer text,
        so an unbounded `limit` makes the server serialize the tenant's whole history. The
        route declares `le=MAX_PAGE_SIZE`, so this never reaches the service's clamp."""
        client = switch_to(api, stack.a.user)

        response = await client.get(f"{HISTORY}?limit={MAX_PAGE_SIZE + 1}")

        assert response.status_code == 422

    async def test_a_zero_limit_is_refused(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        assert (await client.get(f"{HISTORY}?limit=0")).status_code == 422

    async def test_a_negative_offset_is_refused(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        assert (await client.get(f"{HISTORY}?offset=-1")).status_code == 422

    async def test_paging_walks_without_repeating_a_row(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        await upload(client)
        client = switch_to(api, stack.a.user)
        for _ in range(4):
            stack.bedrock.queue_answer("Withholding tax is 10 percent [S1].")
            await ask(client, TAX_QUESTION)

        first = (await client.get(f"{HISTORY}?limit=2&offset=0")).json()
        second = (await client.get(f"{HISTORY}?limit=2&offset=2")).json()

        assert len(first["items"]) == 2
        assert set(ids(first)).isdisjoint(ids(second))
        assert first["total"] == second["total"]
        assert second["offset"] == 2

    async def test_an_offset_past_the_end_returns_an_empty_page_with_the_total(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        body = (await client.get(f"{HISTORY}?offset=500")).json()

        assert body["items"] == []
        assert body["total"] >= 1, "the total describes the filter, not the page"


class TestReadingHistoryIsItselfAudited:
    """"who looked at what customer's questions" is a compliance question in its own right
    (BR-05)."""

    async def test_a_standard_users_read_is_audited(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        await client.get(HISTORY)

        rows = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.user.email)
        assert rows
        assert rows[-1].detail["scope"] == "own"
        assert rows[-1].outcome.value == "SUCCESS"

    async def test_an_administrators_read_is_audited_with_its_wider_scope(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        await client.get(HISTORY)

        rows = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.admin.email)
        assert rows[-1].detail["scope"] == "tenant"

    async def test_the_audited_object_is_whose_history_was_read(
        self, api: Api, stack: Stack
    ) -> None:
        """A list read has no single row, so `object_id` records the subject: the filtered user
        when there is one, the tenant otherwise."""
        client = switch_to(api, stack.a.admin)

        await client.get(f"{HISTORY}?user_id={stack.a.user.id}")
        filtered = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.admin.email)[-1]
        await client.get(HISTORY)
        unfiltered = audit_rows(stack, "HISTORY_VIEW", actor_email=stack.a.admin.email)[-1]

        assert filtered.object_id == str(stack.a.user.id)
        assert unfiltered.object_id == str(stack.a.id)

    async def test_no_audit_detail_quotes_a_question_or_an_answer(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_history(api, stack)
        client = switch_to(api, stack.a.user)

        await client.get(HISTORY)

        written = stack.audit_details()
        assert TAX_QUESTION not in written
        assert "Withholding tax is 10 percent" not in written


class TestTheEndpointIsAuthenticatedAndReadOnly:
    async def test_an_unauthenticated_read_is_401(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, None)

        assert (await client.get(HISTORY)).status_code == 401

    async def test_there_is_no_write_route(self, api: Api, stack: Stack) -> None:
        """History is a projection. A POST, PUT, PATCH or DELETE here would be a second way to
        write an audited fact."""
        client = switch_to(api, stack.a.admin)

        for request in (
            client.post(HISTORY, json={}, headers=csrf_headers()),
            client.put(HISTORY, json={}, headers=csrf_headers()),
            client.patch(HISTORY, json={}, headers=csrf_headers()),
            client.delete(HISTORY, headers=csrf_headers()),
        ):
            assert (await request).status_code == 405

    async def test_a_nonexistent_user_filter_is_an_empty_page_not_an_error(
        self, api: Api, stack: Stack
    ) -> None:
        """A 404 here would confirm which user ids exist."""
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{HISTORY}?user_id={new_uuid()}")

        assert response.status_code == 200
        assert response.json()["items"] == []

    async def test_a_malformed_user_filter_is_422(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(f"{HISTORY}?user_id=not-a-uuid")).status_code == 422
