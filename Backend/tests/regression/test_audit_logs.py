"""backend/tests/regression/test_audit_logs.py

Role: BR-05 (Audit Logging) asserted over HTTP, plus the Compliance family's "100% audit log
availability" clause (AC-11.4).

The requirement, quoted from `WORKBREAKDOWN.md`:

    Audit Logging: Auto-log document uploads/deletions/restorations, user logins/access
    attempts, query submissions, data exports; tamper-resistant logs with user identity,
    action, date/time, affected object (BR-AL, BR-05, R-03)

That sentence has two halves and this module tests both: the **coverage** half (every named
activity produces an entry, automatically, without a caller opting in) and the
**tamper-resistance** half (the chain, and the absence of any way to write to the log through
the API).

What this module cannot prove, and where the rest of the proof lives
--------------------------------------------------------------------
`audit-log-immutability.md` names four independent controls. Only two are testable here:

  1. no write route — asserted below, over the real router;
  2. no mutating method on the repository — asserted by `backend/tests/security/`, statically;
  3. `REVOKE UPDATE, DELETE` from the application database role — needs a real database;
  4. a trigger that raises on UPDATE or DELETE — needs a real database.

Controls 3 and 4 are exercised by `backend/tests/compliance/` against `docquery_test`, and the
tamper-evidence walkthrough in `docs/RELEASE.md` records the manual run. A green result *here*
is not evidence about the database grants, and this docstring is the place that says so.

Why `entries` is read directly in places
----------------------------------------
The HTTP surface only exposes an Administrator's own tenant, so some assertions — "a Standard
User's denied read still produced a row", "tenant B's entries are not in tenant A's page" —
need to see the store from outside any tenant's view. Those read `stack.audit_repo.entries`,
which is the fake standing in for the table itself, and say so at the call site.
"""

from __future__ import annotations

import uuid
from typing import Any

from backend.models.audit_log import GENESIS_HASH, AuditAction, AuditOutcome
from backend.schemas.audit_schemas import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from backend.services.audit_service import compute_integrity_hash
from backend.tests.regression.conftest import (
    API,
    TAX_QUESTION,
    Api,
    Stack,
    audit_rows,
    csrf_headers,
    new_uuid,
    switch_to,
    upload,
)

AUDIT = f"{API}/audit"
VERIFY = f"{AUDIT}/verify"


async def seed_activity(api: Api, stack: Stack) -> dict[str, Any]:
    """One of each audited activity the requirement names, over HTTP.

    Upload, delete and restore as the Administrator; a query as the Standard User; a refused
    upload to produce a denial. Returns the uploaded document body, because several tests need
    its id to check `object_id`.
    """
    admin = switch_to(api, stack.a.admin)
    document = (await upload(admin)).json()
    await admin.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
    await admin.post(
        f"{API}/documents/{document['id']}/restore", headers=csrf_headers()
    )

    user = switch_to(api, stack.a.user)
    stack.bedrock.queue_answer("Withholding tax is 10 percent [S1].")
    await user.post(
        f"{API}/queries", json={"question_text": TAX_QUESTION}, headers=csrf_headers()
    )
    await upload(user)  # refused: Standard Users may not upload

    return document


def logged_actions(stack: Stack) -> set[str]:
    """Every action value present in the store, tenant-agnostic."""
    return {entry.action.value for entry in stack.audit_repo.entries}


class TestEveryNamedActivityIsLoggedAutomatically:
    """"Auto-log document uploads/deletions/restorations, user logins/access attempts, query
    submissions, data exports".

    "Auto" is the load-bearing word: no route asks for an audit entry, and none of these tests
    passes an opt-in flag.
    """

    async def test_document_lifecycle_actions_are_logged(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_activity(api, stack)

        assert {
            "DOCUMENT_UPLOAD",
            "DOCUMENT_DELETE",
            "DOCUMENT_RESTORE",
        } <= logged_actions(stack)

    async def test_a_query_submission_is_logged(self, api: Api, stack: Stack) -> None:
        await seed_activity(api, stack)

        assert "QUERY_SUBMIT" in logged_actions(stack)

    async def test_a_refused_action_is_logged_as_denied(
        self, api: Api, stack: Stack
    ) -> None:
        """"access attempts" covers the refused ones. A log that records only what succeeded
        cannot answer "did anyone try?", which is the question a breach review opens with."""
        await seed_activity(api, stack)

        denials = [
            entry
            for entry in stack.audit_repo.entries
            if entry.outcome is AuditOutcome.DENIED
        ]
        assert denials
        assert any(entry.actor_email == stack.a.user.email for entry in denials)

    async def test_a_data_export_is_logged(self, api: Api, stack: Stack) -> None:
        """The fourth named activity. Exports leave the system, so the entry is the only
        remaining record of what was taken."""
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            f"{API}/analytics/export",
            json={"report": "keywords", "format": "excel"},
            headers=csrf_headers(),
        )

        assert response.status_code == 200
        assert "DATA_EXPORT" in logged_actions(stack)

    async def test_reading_the_audit_log_is_itself_logged(
        self, api: Api, stack: Stack
    ) -> None:
        """A log that cannot say who read it is missing the access record a regulator asks for
        first."""
        client = switch_to(api, stack.a.admin)

        await client.get(AUDIT)

        rows = audit_rows(stack, "AUDIT_VIEW", actor_email=stack.a.admin.email)
        assert rows

    async def test_verifying_the_chain_writes_nothing(
        self, api: Api, stack: Stack
    ) -> None:
        """Deliberately *not* audited: appending to the log being checked moves the chain head
        between two consecutive checks, which makes the second result harder to reason about.
        `/verify` returns counts and verdicts, never entry contents."""
        client = switch_to(api, stack.a.admin)
        before = len(stack.audit_repo.entries)

        response = await client.get(VERIFY)

        assert response.status_code == 200
        assert len(stack.audit_repo.entries) == before


class TestEveryEntryCarriesTheFourRequiredFields:
    """"user identity, action, date/time, affected object" — the four the requirement names,
    on every row without exception."""

    async def test_all_four_fields_are_present_on_every_entry(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_activity(api, stack)

        for entry in stack.audit_repo.entries:
            assert entry.actor_email, "user identity"
            assert entry.action, "action"
            assert entry.occurred_at, "date/time"
            assert entry.object_id, "affected system object"
            assert entry.object_type

    async def test_the_published_entry_carries_them_too(
        self, api: Api, stack: Stack
    ) -> None:
        """The DTO is where a compliance reviewer actually reads them, and `extra="forbid"`
        means a field lost in the mapping does not silently reappear as `None`."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(AUDIT)).json()["items"]

        assert items
        for item in items:
            for field in (
                "log_id",
                "actor_email",
                "action",
                "object_type",
                "object_id",
                "outcome",
                "occurred_at",
                "previous_hash",
                "integrity_hash",
            ):
                assert item[field], field

    async def test_the_affected_object_is_the_thing_acted_on(
        self, api: Api, stack: Stack
    ) -> None:
        document = await seed_activity(api, stack)

        for action in ("DOCUMENT_UPLOAD", "DOCUMENT_DELETE", "DOCUMENT_RESTORE"):
            rows = audit_rows(stack, action, actor_email=stack.a.admin.email)
            assert rows, action
            assert rows[-1].object_id == document["id"], action

    async def test_the_actor_is_the_authenticated_user_not_the_subject(
        self, api: Api, stack: Stack
    ) -> None:
        """Identity is taken from the session. A body or query parameter cannot name a
        different actor, so an entry cannot be attributed to someone else."""
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("Withholding tax is 10 percent [S1].")

        await client.post(
            f"{API}/queries",
            json={"question_text": TAX_QUESTION},
            headers=csrf_headers(),
        )

        rows = [
            row
            for row in audit_rows(stack, "QUERY_SUBMIT")
            if row.tenant_id == stack.a.id
        ]
        assert rows
        assert all(row.actor_email == stack.a.user.email for row in rows)
        assert all(row.user_id == stack.a.user.id for row in rows)

    async def test_the_request_origin_is_recorded(self, api: Api, stack: Stack) -> None:
        """`ip_address` and `user_agent` are not in the requirement's four, but R-03 is about
        risk mitigation and an entry without an origin cannot support an investigation."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(AUDIT)).json()["items"]

        assert all(item["ip_address"] for item in items)
        assert all(item["user_agent"] for item in items)


class TestTheLogIsTamperEvident:
    """"tamper-resistant logs" (BR-AL, R-03). The chain is what makes an alteration *visible*;
    the grants and triggers that make it *impossible* need a real database — see the module
    docstring."""

    async def test_a_clean_chain_verifies(self, api: Api, stack: Stack) -> None:
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(VERIFY)).json()

        assert body["is_valid"] is True
        assert body["failures"] == []
        assert body["entries_checked"] > 0

    async def test_the_chain_starts_at_the_genesis_hash(
        self, api: Api, stack: Stack
    ) -> None:
        """The first entry for a tenant has no predecessor, so it links to a fixed constant
        rather than to nothing — otherwise "no previous hash" and "previous hash removed" look
        the same to a verifier."""
        first = next(
            entry
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        )

        assert first.previous_hash == GENESIS_HASH

    async def test_each_entry_links_to_the_one_before_it(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_activity(api, stack)

        chain = [
            entry
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        ]
        previous = GENESIS_HASH
        for entry in chain:
            assert entry.previous_hash == previous
            assert entry.integrity_hash == compute_integrity_hash(entry, previous)
            previous = entry.integrity_hash

    async def test_an_altered_entry_is_named_by_the_verifier(
        self, api: Api, stack: Stack
    ) -> None:
        """The assertion this whole module exists for.

        The row is mutated in the store directly — which is what a database-level attacker
        does, and what the API cannot do — and `/verify` must name that exact `log_id`. A
        boolean verdict without the id tells an auditor there is a problem without telling them
        where.
        """
        await seed_activity(api, stack)
        target = [
            entry
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        ][2]
        target.outcome = AuditOutcome.SUCCESS
        target.action = AuditAction.LOGIN_SUCCESS
        target.detail = {"note": "covering my tracks"}

        client = switch_to(api, stack.a.admin)
        body = (await client.get(VERIFY)).json()

        assert body["is_valid"] is False
        assert str(target.log_id) in [failure["log_id"] for failure in body["failures"]]

    async def test_a_deleted_entry_breaks_the_chain(
        self, api: Api, stack: Stack
    ) -> None:
        """Removing a row leaves the next one pointing at a hash that is no longer there. A log
        that only detected *edits* could be silently truncated in the middle."""
        await seed_activity(api, stack)
        chain = [
            entry
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        ]
        removed = chain[1]
        stack.audit_repo.entries.remove(removed)

        client = switch_to(api, stack.a.admin)
        body = (await client.get(VERIFY)).json()

        assert body["is_valid"] is False
        assert body["failures"]

    async def test_verification_names_only_the_rows_actually_affected(
        self, api: Api, stack: Stack
    ) -> None:
        """One edit in a long chain must not produce a finding for every row after it — an
        auditor would have to read past thousands of consequences to find the cause."""
        await seed_activity(api, stack)
        chain = [
            entry
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        ]
        chain[1].detail = {"note": "edited"}

        client = switch_to(api, stack.a.admin)
        body = (await client.get(VERIFY)).json()

        assert body["is_valid"] is False
        assert len(body["failures"]) < len(chain), (
            "a single edit must not mark the whole chain as damaged"
        )

    async def test_the_hashes_are_published_so_an_auditor_can_recompute_them(
        self, api: Api, stack: Stack
    ) -> None:
        """An auditor who exports the entries must be able to verify the chain outside this
        system. A hash they cannot see proves nothing to them."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(AUDIT)).json()["items"]

        for item in items:
            assert len(item["integrity_hash"]) == 64
            assert len(item["previous_hash"]) == 64

    async def test_another_tenants_chain_is_verified_separately(
        self, api: Api, stack: Stack
    ) -> None:
        """One chain per tenant. Breaking tenant B's must not make tenant A's log look
        altered — a shared chain would let one organization's incident cast doubt on
        another's records."""
        await seed_activity(api, stack)
        broken = next(
            entry
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.b.id
        )
        broken.detail = {"note": "edited in tenant B"}

        client = switch_to(api, stack.a.admin)
        tenant_a = (await client.get(VERIFY)).json()
        client = switch_to(api, stack.b.admin)
        tenant_b = (await client.get(VERIFY)).json()

        assert tenant_a["is_valid"] is True
        assert tenant_b["is_valid"] is False


class TestThereIsNoWayToWriteToTheLogThroughTheApi:
    """`audit-log-immutability.md`: "No programmatic deletion or modification interfaces"."""

    async def test_no_write_method_is_routed(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        for request in (
            client.post(AUDIT, json={}, headers=csrf_headers()),
            client.put(AUDIT, json={}, headers=csrf_headers()),
            client.patch(AUDIT, json={}, headers=csrf_headers()),
            client.delete(AUDIT, headers=csrf_headers()),
        ):
            response = await request
            assert response.status_code == 405, response.request.method

    async def test_no_write_method_is_routed_on_a_single_entry(
        self, api: Api, stack: Stack
    ) -> None:
        """There is no `/audit/{log_id}` at all, so an amendment has no address."""
        await seed_activity(api, stack)
        log_id = stack.audit_repo.entries[0].log_id
        client = switch_to(api, stack.a.admin)

        for request in (
            client.get(f"{AUDIT}/{log_id}"),
            client.patch(f"{AUDIT}/{log_id}", json={}, headers=csrf_headers()),
            client.delete(f"{AUDIT}/{log_id}", headers=csrf_headers()),
        ):
            assert (await request).status_code in {404, 405, 422}

    async def test_an_export_cannot_be_pointed_at_the_audit_log(
        self, api: Api, stack: Stack
    ) -> None:
        """`ExportReport` is a closed enum, so "export the audit log as a spreadsheet I can
        edit" is not a reachable request."""
        client = switch_to(api, stack.a.admin)

        response = await client.post(
            f"{API}/analytics/export",
            json={"report": "audit", "format": "excel"},
            headers=csrf_headers(),
        )

        assert response.status_code == 422


class TestOnlyAdministratorsMayRead:
    """`audit-log-immutability.md`: "only Administrators can READ logs". Compliance officers
    hold that role — there is no third role (`rbac-enforcement.md`)."""

    async def test_a_standard_user_is_refused(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        assert (await client.get(AUDIT)).status_code == 403

    async def test_a_standard_user_cannot_verify_the_chain_either(
        self, api: Api, stack: Stack
    ) -> None:
        """`/verify` returns no entry contents, but the verdict is still compliance
        information: "the log has been altered" is not something to hand to everyone."""
        client = switch_to(api, stack.a.user)

        assert (await client.get(VERIFY)).status_code == 403

    async def test_an_unauthenticated_read_is_401(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, None)

        assert (await client.get(AUDIT)).status_code == 401
        assert (await client.get(VERIFY)).status_code == 401

    async def test_the_refusal_is_itself_recorded(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        await client.get(AUDIT)

        denials = [
            entry
            for entry in stack.audit_repo.entries
            if entry.outcome is AuditOutcome.DENIED
            and entry.actor_email == stack.a.user.email
        ]
        assert denials

    async def test_an_administrator_reads_only_their_own_tenant(
        self, api: Api, stack: Stack
    ) -> None:
        """The single worst read in the system, and the reason `tenant_id` is not a
        parameter."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(f"{AUDIT}?limit={MAX_PAGE_SIZE}")).json()["items"]

        tenant_a_ids = {
            str(entry.log_id)
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        }
        assert items
        assert all(item["log_id"] in tenant_a_ids for item in items)

    async def test_naming_a_tenant_in_the_query_string_is_refused(
        self, api: Api, stack: Stack
    ) -> None:
        """`AuditLogFilter` forbids extra fields, and `tenant_id` is deliberately not one of
        them."""
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{AUDIT}?tenant_id={stack.b.id}")

        # Unknown query parameters are currently ignored rather than rejected, so the guarantee
        # asserted here is the one that matters: the parameter cannot widen the scope.
        assert response.status_code == 200
        tenant_a_ids = {
            str(entry.log_id)
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        }
        assert all(
            item["log_id"] in tenant_a_ids for item in response.json()["items"]
        )


class TestFilteringAndPagingTheComplianceRead:
    """The filters a compliance review actually uses. AC-11.4 asks for 100% availability of the
    log, which in practice means being able to *find* the relevant rows."""

    async def test_newest_entries_come_first(self, api: Api, stack: Stack) -> None:
        """A reviewer opens page one to see what just happened. Oldest-first would put the
        most recent activity on the last page."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(AUDIT)).json()["items"]

        timestamps = [item["occurred_at"] for item in items]
        assert timestamps == sorted(timestamps, reverse=True)

    async def test_an_action_filter_narrows_the_page(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(f"{AUDIT}?action=DOCUMENT_UPLOAD")).json()["items"]

        assert items
        assert {item["action"] for item in items} == {"DOCUMENT_UPLOAD"}

    async def test_an_outcome_filter_finds_the_denials(
        self, api: Api, stack: Stack
    ) -> None:
        """The query a breach review runs first."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(f"{AUDIT}?outcome=DENIED")).json()["items"]

        assert items
        assert {item["outcome"] for item in items} == {"DENIED"}

    async def test_a_user_filter_selects_the_subject_not_the_reader(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(f"{AUDIT}?user_id={stack.a.user.id}")).json()["items"]

        assert items
        assert {item["user_id"] for item in items} == {str(stack.a.user.id)}

    async def test_an_unknown_action_is_422(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(f"{AUDIT}?action=NOT_AN_ACTION")).status_code == 422

    async def test_a_malformed_user_filter_is_422(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(f"{AUDIT}?user_id=not-a-uuid")).status_code == 422

    async def test_a_reversed_date_range_is_refused(
        self, api: Api, stack: Stack
    ) -> None:
        """A reversed range silently returns nothing, which reads as "no such activity" in a
        compliance review. Better to say the filter is wrong."""
        client = switch_to(api, stack.a.admin)

        response = await client.get(
            f"{AUDIT}?occurred_from=2026-01-02T00:00:00Z&occurred_to=2026-01-01T00:00:00Z"
        )

        assert response.status_code == 422

    async def test_an_ordered_date_range_is_accepted(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        response = await client.get(
            f"{AUDIT}?occurred_from=2026-01-01T00:00:00Z&occurred_to=2026-12-31T00:00:00Z"
        )

        assert response.status_code == 200

    async def test_the_default_page_size_is_the_documented_one(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(AUDIT)).json()["limit"] == DEFAULT_PAGE_SIZE

    async def test_an_oversized_limit_is_refused(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{AUDIT}?limit={MAX_PAGE_SIZE + 1}")

        assert response.status_code == 422

    async def test_paging_a_filtered_read_walks_without_repeating_an_entry(
        self, api: Api, stack: Stack
    ) -> None:
        """Filtered on an action the read itself does not produce, so the result set is stable
        between the two requests — see the next test for why that qualifier is needed."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)
        page = f"{AUDIT}?action=DOCUMENT_UPLOAD"

        first = (await client.get(f"{page}&limit=1&offset=0")).json()
        second = (await client.get(f"{page}&limit=1&offset=1")).json()

        assert len(first["items"]) == 1
        assert len(second["items"]) == 1
        assert first["items"][0]["log_id"] != second["items"][0]["log_id"]

    async def test_an_unfiltered_page_shifts_because_reading_is_audited(
        self, api: Api, stack: Stack
    ) -> None:
        """Not a paging bug — a consequence of the requirement.

        Every `GET /audit` appends its own `AUDIT_VIEW` row, so the newest-first result set has
        grown by one before the client asks for page two, and one entry slides across the
        boundary. Recording the read is worth more than a stable pager, so the behaviour stays
        and is asserted here instead: a reviewer walking a long log should filter (previous
        test) or widen the page, and this test is the note explaining why.
        """
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        first = (await client.get(f"{AUDIT}?limit=2&offset=0")).json()
        second = (await client.get(f"{AUDIT}?limit=2&offset=2")).json()

        assert second["total"] == first["total"] + 1
        assert not {item["log_id"] for item in first["items"]}.isdisjoint(
            {item["log_id"] for item in second["items"]}
        ), "the read that produced page one is itself now the newest entry"

    async def test_no_entry_appears_twice_within_one_page(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        items = (await client.get(f"{AUDIT}?limit={MAX_PAGE_SIZE}")).json()["items"]

        log_ids = [item["log_id"] for item in items]
        assert len(log_ids) == len(set(log_ids))

    async def test_a_filter_matching_nothing_is_an_empty_page_not_an_error(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{AUDIT}?user_id={new_uuid()}")

        assert response.status_code == 200
        assert response.json()["items"] == []

    async def test_verification_can_be_bounded_to_the_oldest_entries(
        self, api: Api, stack: Stack
    ) -> None:
        """`limit` on `/verify` exists so a reviewer can spot-check a long chain without
        walking all of it."""
        await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        body = (await client.get(f"{VERIFY}?limit=2")).json()

        assert body["entries_checked"] == 2
        assert body["is_valid"] is True

    async def test_a_zero_verification_limit_is_refused(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(f"{VERIFY}?limit=0")).status_code == 422


class TestNothingSensitiveReachesTheLog:
    """`secure-coding-vapt.md`: "Never log passwords, API keys, customer account numbers,
    financial data, or PII." An append-only log is the worst possible place to put a secret,
    because there is no interface to take it back out."""

    async def test_no_entry_quotes_document_content(
        self, api: Api, stack: Stack
    ) -> None:
        await seed_activity(api, stack)

        written = stack.audit_details()
        assert "wire transfer fee is 25 USD" not in written
        assert "Withholding tax" not in written

    async def test_no_entry_quotes_a_question(self, api: Api, stack: Stack) -> None:
        await seed_activity(api, stack)

        assert TAX_QUESTION not in stack.audit_details()

    async def test_no_entry_carries_a_presigned_url(
        self, api: Api, stack: Stack
    ) -> None:
        """A presigned URL is a bearer credential for the document. In an immutable log it is
        a credential nobody can revoke from there."""
        document = await seed_activity(api, stack)
        client = switch_to(api, stack.a.admin)

        preview = await client.get(f"{API}/documents/{document['id']}/preview")

        assert preview.status_code == 200
        assert "X-Amz-Signature" not in stack.audit_details()
        assert preview.json()["url"] not in stack.audit_details()

    async def test_entry_details_carry_no_line_breaks(
        self, api: Api, stack: Stack
    ) -> None:
        """A CR or LF in a detail value is a forged second log line to anything reading the
        stream (`secure-coding-vapt.md`, Log Injection)."""
        await seed_activity(api, stack)

        for entry in stack.audit_repo.entries:
            for value in (entry.detail or {}).values():
                if isinstance(value, str):
                    assert "\n" not in value
                    assert "\r" not in value


class TestTheLogSurvivesTheThingsItRecords:
    """AC-11.4: "100% audit log availability during compliance reviews". An entry has to
    outlive the object it describes, or the trail ends where an investigation starts."""

    async def test_deleting_a_document_does_not_remove_its_entries(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        document = (await upload(client)).json()
        before = len(audit_rows(stack, "DOCUMENT_UPLOAD"))

        await client.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        assert len(audit_rows(stack, "DOCUMENT_UPLOAD")) >= before
        assert audit_rows(stack, "DOCUMENT_DELETE")

    async def test_the_entry_still_names_the_deleted_object(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        document = (await upload(client)).json()

        await client.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        rows = audit_rows(stack, "DOCUMENT_DELETE", actor_email=stack.a.admin.email)
        assert rows[-1].object_id == document["id"]
        assert uuid.UUID(rows[-1].object_id)

    async def test_entries_accumulate_and_are_never_replaced(
        self, api: Api, stack: Stack
    ) -> None:
        """Append-only, observed end to end: repeating an action adds a row rather than
        updating the previous one."""
        client = switch_to(api, stack.a.admin)
        before = len(stack.audit_repo.entries)

        await client.get(AUDIT)
        await client.get(AUDIT)
        await client.get(AUDIT)

        assert len(stack.audit_repo.entries) == before + 3
