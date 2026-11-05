"""backend/tests/security/test_idor.py

Role: Prove that holding a valid object id gets you nothing you were not already entitled to —
across tenants, and across users inside a tenant.

`.claude/rules/secure-coding-vapt.md`: *"**IDOR/Broken AuthZ**: Enforce RBAC checks server-side
for every document, search history, audit log, and analytics access. Verify user role and
ownership before data retrieval."*

Why the ids here are real
-------------------------
The weak version of this test asks for `uuid4()` and checks for a 404. That proves the route
handles an unknown id; it says nothing about authorization, because a missing row 404s whether
or not anyone checked. The interesting request is one where **the row exists and the caller is
the wrong person** — which is also the realistic one: an attacker gets an id from a citation, a
shared screenshot, an export, or their own earlier session after a role change.

So every test here uses `stack.b.document.id` or `stack.b.query.id` — rows that genuinely
exist, in a tenant the caller does not belong to.

Why 404 and not 403
-------------------
A 403 on an existing id and a 404 on a missing one are together an existence oracle: an
attacker who can tell the two apart can enumerate which document ids are real without ever
reading one. So an unauthorized *object* reference is 404, uniformly.

Role gates are different. A Standard User calling an Administrator-only route gets 403, because
the route's existence is already public — it is in `/openapi.json` — and there is nothing to
disclose. Both conventions are asserted below, because "always 404" and "always 403" are each
wrong in one direction.

What this module proves, and what it does not
---------------------------------------------
The enforcement under test is real: the route, `require_admin`, the CSRF middleware, the
service and the exception handlers are the production ones. What the fakes supply is the
repository's `WHERE` clause. So these tests prove that **the service passes the caller's tenant
and role down, refuses in the right direction, and audits the refusal** — see
`document_service.get_document`, which raises `DocumentNotFound` and writes a DENIED row before
`preview` can mint a URL.

They do *not* prove the SQL is tenant-scoped. That is a separate claim, made against compiled
statements in `test_sql_injection.py::assert_tenant_scoped`, because a behavioural test cannot
distinguish "the repository filtered" from "the fake filtered".

Verified by mutation, not by assumption: with `FakeDocumentRepository.get` patched to ignore
`tenant_id`, `test_reading_another_tenants_document_is_404` gets a **200** — they bite.

What the audit trail has to say
-------------------------------
Every denial writes a row with `outcome='DENIED'` (`rbac-enforcement.md`: *"Log all access
attempts (authorized and denied) to audit trail"*). Silent refusal is the failure mode that
matters here — an attacker enumerating ids one per second leaves no trace, and the first anyone
knows of it is the breach report. So the denial rows are asserted, not just the status codes.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable

import httpx
import pytest

from backend.models.user import User
from backend.tests.security.conftest import API, CONTEXT, Stack, csrf_headers

Api = tuple[httpx.AsyncClient, Callable[[User | None], None]]


def denied_count(stack: Stack) -> int:
    return len(stack.denials())


def denial_details(stack: Stack) -> str:
    return repr([entry.detail for entry in stack.denials()])


class TestTheHarnessItself:
    """If these fail, nothing else in the module means anything.

    A security suite whose fixture is subtly broken reports safety it never tested — the exact
    failure this package's docstring warns about. So the harness is asserted first: the two
    tenants are distinct, both have real rows, and an authorized request actually succeeds.
    """

    async def test_the_two_tenants_are_distinct_and_populated(self, stack: Stack) -> None:
        assert stack.a.id != stack.b.id
        assert stack.a.document is not None and stack.b.document is not None
        assert stack.a.document.id != stack.b.document.id
        assert stack.a.query is not None and stack.b.query is not None
        assert stack.a.query.id != stack.b.query.id
        assert stack.a.user.tenant_id == stack.a.id
        assert stack.b.user.tenant_id == stack.b.id

    async def test_an_authorized_request_succeeds(self, stack: Stack, api: Api) -> None:
        """The positive control. Without it, a wall of 404s proves only that nothing works."""
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.get(f"{API}/documents/{stack.a.document.id}")
        assert response.status_code == 200, response.text
        assert response.json()["id"] == str(stack.a.document.id)

    async def test_an_unauthenticated_request_is_401(self, stack: Stack, api: Api) -> None:
        client, as_user = api
        as_user(None)
        response = await client.get(f"{API}/documents/{stack.a.document.id}")
        assert response.status_code == 401, response.text


class TestCrossTenantDocuments:
    """Tenant A's people asking for tenant B's document. Both roles, every route."""

    @pytest.mark.parametrize("role", ["admin", "user"])
    async def test_reading_another_tenants_document_is_404(
        self, stack: Stack, api: Api, role: str
    ) -> None:
        """An Administrator is not a superuser. Role and tenant are independent dimensions.

        This is the single most important assertion in the module. `rbac-enforcement.md`:
        *"An Administrator in tenant A must not see tenant B's data."* An implementation that
        checked only the role would pass every other admin test in this repository and fail
        exactly here.
        """
        client, as_user = api
        as_user(getattr(stack.a, role))
        before = denied_count(stack)

        response = await client.get(f"{API}/documents/{stack.b.document.id}")

        assert response.status_code == 404, response.text
        assert stack.b.document.name not in response.text, (
            "the 404 body leaked the document's name"
        )
        assert denied_count(stack) > before, (
            f"a cross-tenant read was refused but not audited: {denial_details(stack)}"
        )

    async def test_a_preview_url_is_not_issued_across_tenants(
        self, stack: Stack, api: Api
    ) -> None:
        """A presigned URL is a bearer credential — anyone holding it can fetch the bytes.

        S3 does not re-check who asked for it, so this is the one denial where a mistake cannot
        be walked back: the URL stays valid for its whole TTL even after the account is locked.
        """
        client, as_user = api
        as_user(stack.a.admin)

        response = await client.get(f"{API}/documents/{stack.b.document.id}/preview")

        assert response.status_code == 404, response.text
        # Asserted against the whole body, not against one key. The error envelope is
        # `{"error": {"code", "message"}}`, so reading a top-level "message" would find nothing
        # and pass no matter what the route returned.
        assert "http" not in response.text.lower(), f"a URL appeared: {response.text}"
        assert stack.b.document.s3_key not in response.text, "the S3 key leaked"
        assert response.json()["error"]["code"], "the envelope is missing its error code"

    async def test_the_version_lineage_of_another_tenants_document_is_404(
        self, stack: Stack, api: Api
    ) -> None:
        """Lineage discloses names and upload dates even when no bytes are returned."""
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.get(f"{API}/documents/{stack.b.document.id}/versions")
        assert response.status_code == 404, response.text
        assert stack.b.document.name not in response.text

    async def test_deleting_another_tenants_document_is_404_and_changes_nothing(
        self, stack: Stack, api: Api
    ) -> None:
        """The write path. A refused delete must also be a delete that did not happen.

        Asserting the status code alone would miss the worst version of this bug: a service
        that soft-deletes the row and *then* discovers the tenant mismatch. The document's
        status is re-read afterwards, because "404" and "unchanged" are two claims.
        """
        client, as_user = api
        as_user(stack.a.admin)
        before = stack.b.document.status

        response = await client.request(
            "DELETE",
            f"{API}/documents/{stack.b.document.id}",
            headers=csrf_headers(),
        )

        assert response.status_code == 404, response.text
        still_there = await stack.documents.get(stack.b.document.id, stack.b.id)
        assert still_there is not None, "the row was removed by a request that returned 404"
        assert still_there.status is before, (
            f"the status changed to {still_there.status} on a refused delete"
        )

    async def test_restoring_another_tenants_document_is_404(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.post(
            f"{API}/documents/{stack.b.document.id}/restore", headers=csrf_headers()
        )
        assert response.status_code == 404, response.text

    async def test_the_document_list_never_crosses_tenants(
        self, stack: Stack, api: Api
    ) -> None:
        """No id is needed for this one — the list is where a missing filter shows up.

        A per-id check that passes while the collection endpoint is unscoped is a common shape:
        the id routes get the security review and the list route gets `SELECT *`.
        """
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.get(f"{API}/documents")
        assert response.status_code == 200, response.text
        ids = {item["id"] for item in response.json()["items"]}
        assert str(stack.a.document.id) in ids
        assert str(stack.b.document.id) not in ids
        assert stack.b.document.name not in response.text


class TestCrossTenantQueries:
    @pytest.mark.parametrize("role", ["admin", "user"])
    async def test_reading_another_tenants_query_is_404(
        self, stack: Stack, api: Api, role: str
    ) -> None:
        """A query row holds the question *and* the answer — the most sensitive pair here.

        A banking question ("what is the balance on account 4012…") is itself customer data
        even before the answer is attached.
        """
        client, as_user = api
        as_user(getattr(stack.a, role))
        before = denied_count(stack)

        response = await client.get(f"{API}/queries/{stack.b.query.id}")

        assert response.status_code == 404, response.text
        assert "Tenant B" not in response.text, "tenant B's corpus leaked through the error"
        assert denied_count(stack) > before

    async def test_the_history_projection_never_crosses_tenants(
        self, stack: Stack, api: Api
    ) -> None:
        """`search_history` is a role-filtered projection over `query`, so it has two filters.

        An Administrator legitimately reads every row *for the tenant*, which means the user
        filter is dropped for them — and that is exactly the code path where the tenant filter
        still has to be there. Asserted from the admin's side for that reason.
        """
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.get(f"{API}/history")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"] == "tenant"
        ids = {item["id"] for item in body["items"]}
        assert str(stack.a.query.id) in ids
        assert str(stack.b.query.id) not in ids


class TestOwnershipInsideATenant:
    """The other axis. Same tenant, wrong user — what a role check alone does not catch."""

    async def test_a_standard_user_cannot_read_a_colleagues_query(
        self, stack: Stack, api: Api
    ) -> None:
        """`rbac-enforcement.md`: *"Access **own search history only** (never other users')."*

        The colleague here is the tenant's Administrator, whose query the fixture submitted —
        an id a Standard User could plausibly obtain, since both appear in the same analytics
        aggregate.
        """
        client, as_user = api
        colleague = await stack.query_service.submit(
            question="What is the monthly account maintenance fee?",
            actor=stack.a.admin,
            context=CONTEXT,
        )
        as_user(stack.a.user)
        before = denied_count(stack)

        response = await client.get(f"{API}/queries/{colleague.id}")

        assert response.status_code == 404, response.text
        denials = stack.denials()
        assert len(denials) > before
        assert any("not_owner" in repr(entry.detail) for entry in denials), (
            f"the denial was audited without saying why: {denial_details(stack)}"
        )

    async def test_a_standard_user_reads_their_own_query(
        self, stack: Stack, api: Api
    ) -> None:
        """The positive control for the ownership check.

        Without it, `get_query` could refuse *everyone* and the test above would still pass.
        """
        client, as_user = api
        as_user(stack.a.user)
        response = await client.get(f"{API}/queries/{stack.a.query.id}")
        assert response.status_code == 200, response.text
        assert response.json()["id"] == str(stack.a.query.id)

    async def test_history_scope_is_own_for_a_standard_user(
        self, stack: Stack, api: Api
    ) -> None:
        """The list counterpart: the projection filters by `user_id`, not only by tenant."""
        client, as_user = api
        colleague = await stack.query_service.submit(
            question="What is the outbound wire fee for an internal transfer?",
            actor=stack.a.admin,
            context=CONTEXT,
        )
        as_user(stack.a.user)

        response = await client.get(f"{API}/history")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"] == "own", (
            "a Standard User's history reported a tenant-wide scope"
        )
        ids = {item["id"] for item in body["items"]}
        assert str(stack.a.query.id) in ids
        assert str(colleague.id) not in ids, "a colleague's query appeared in own-scope"


    async def test_an_administrator_sees_both_rows(self, stack: Stack, api: Api) -> None:
        """The negative of the negative: the user filter must be *dropped* for an admin.

        Otherwise `scope: "tenant"` would be a label with no behaviour behind it, and the
        system-wide history requirement in `rbac-enforcement.md` would be unmet — a functional
        failure that a suite of only-deny tests would never notice.
        """
        client, as_user = api
        colleague = await stack.query_service.submit(
            question="Which fee applies to a same-day domestic payment?",
            actor=stack.a.admin,
            context=CONTEXT,
        )
        as_user(stack.a.admin)

        response = await client.get(f"{API}/history")

        assert response.status_code == 200, response.text
        ids = {item["id"] for item in response.json()["items"]}
        assert {str(stack.a.query.id), str(colleague.id)} <= ids


class TestRoleGatesAre403NotSilent:
    """Route-level role checks. 403 here is correct — there is no object to disclose."""

    ADMIN_ONLY_GETS = (
        "/analytics/keywords",
        "/analytics/documents",
        "/analytics/trends",
        "/audit",
        "/audit/verify",
    )

    @pytest.mark.parametrize("path", ADMIN_ONLY_GETS)
    async def test_a_standard_user_is_403_on_every_admin_read(
        self, stack: Stack, api: Api, path: str
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)
        before = denied_count(stack)

        response = await client.get(f"{API}{path}")

        assert response.status_code == 403, f"{path} -> {response.status_code}"
        assert response.json()["error"]["code"] == "forbidden", response.text
        denials = stack.denials()
        assert len(denials) > before, f"{path} refused without an audit row"
        assert any("administrator_required" in repr(entry.detail) for entry in denials), (
            f"{path}: the denial does not record the reason: {denial_details(stack)}"
        )

    async def test_a_standard_user_cannot_export(self, stack: Stack, api: Api) -> None:
        """The export is the one admin-only *write*, and it is the bulk-disclosure route."""
        client, as_user = api
        as_user(stack.a.user)
        response = await client.post(
            f"{API}/analytics/export",
            json={"format": "PDF"},
            headers=csrf_headers(),
        )
        assert response.status_code == 403, response.text

    async def test_a_standard_user_cannot_upload(self, stack: Stack, api: Api) -> None:
        client, as_user = api
        as_user(stack.a.user)
        response = await client.post(
            f"{API}/documents",
            files={"file": ("fee.pdf", b"%PDF-1.4 not a real pdf", "application/pdf")},
            headers=csrf_headers(),
        )
        assert response.status_code == 403, response.text

    async def test_an_administrator_reaches_the_same_routes(
        self, stack: Stack, api: Api
    ) -> None:
        """The positive control for the whole class.

        A `require_admin` that raised for everybody would satisfy every 403 assertion above.
        """
        client, as_user = api
        as_user(stack.a.admin)
        for path in self.ADMIN_ONLY_GETS:
            response = await client.get(f"{API}{path}")
            assert response.status_code == 200, (
                f"{path} -> {response.status_code}: {response.text}"
            )


class TestUnknownIdsLookIdenticalToForbiddenOnes:
    """The existence oracle, stated as a test.

    A 404 for a nonexistent id and a 404 for someone else's id are only equivalent if the
    *whole response* matches — status, error code, and message. A different message body is
    still an oracle, just a quieter one.
    """

    async def test_a_nonexistent_document_and_another_tenants_are_indistinguishable(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)

        missing = await client.get(f"{API}/documents/{uuid.uuid4()}")
        forbidden = await client.get(f"{API}/documents/{stack.b.document.id}")

        assert missing.status_code == forbidden.status_code == 404
        assert missing.json() == forbidden.json(), (
            "the two 404s differ, so the response distinguishes 'absent' from 'not yours':\n"
            f"missing:   {missing.json()}\nforbidden: {forbidden.json()}"
        )

    async def test_a_nonexistent_query_and_a_colleagues_are_indistinguishable(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        colleague = await stack.query_service.submit(
            question="What is the maintenance fee waiver threshold?",
            actor=stack.a.admin,
            context=CONTEXT,
        )
        as_user(stack.a.user)

        missing = await client.get(f"{API}/queries/{uuid.uuid4()}")
        forbidden = await client.get(f"{API}/queries/{colleague.id}")

        assert missing.status_code == forbidden.status_code == 404
        assert missing.json() == forbidden.json()

    async def test_a_malformed_id_is_422_not_500(self, stack: Stack, api: Api) -> None:
        """Path validation happens before any lookup, so a non-UUID never reaches the service.

        Worth asserting because a 500 here would be an information leak of a different kind: a
        stack trace, and confirmation that the value got far enough to break something.
        """
        client, as_user = api
        as_user(stack.a.admin)
        response = await client.get(f"{API}/documents/not-a-uuid")
        assert response.status_code == 422, response.text
        assert "traceback" not in response.text.lower()


class TestEveryDenialIsAudited:
    """The audit trail's own coverage, checked in aggregate rather than per route.

    One test per route would pass with a per-route audit call and no rule; this asks the
    question the compliance reviewer asks — *for this run, was every refusal recorded?* — and
    reports the routes that were not.
    """

    async def test_a_sweep_of_forbidden_requests_leaves_a_denial_for_each(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)
        attempts = [
            ("GET", f"{API}/analytics/keywords"),
            ("GET", f"{API}/analytics/documents"),
            ("GET", f"{API}/analytics/trends"),
            ("GET", f"{API}/audit"),
            ("GET", f"{API}/audit/verify"),
            ("GET", f"{API}/documents/{stack.b.document.id}"),
            ("GET", f"{API}/queries/{stack.b.query.id}"),
        ]

        recorded: list[tuple[str, int, int]] = []
        for method, path in attempts:
            before = denied_count(stack)
            response = await client.request(method, path, headers=csrf_headers())
            assert response.status_code in (403, 404), f"{path} -> {response.status_code}"
            recorded.append((path, response.status_code, denied_count(stack) - before))

        unaudited = [path for path, _, added in recorded if added == 0]
        assert unaudited == [], (
            "these refusals left no DENIED audit row, so enumeration against them is "
            f"invisible: {unaudited}"
        )

    async def test_every_denial_names_the_actor_and_the_object(
        self, stack: Stack, api: Api
    ) -> None:
        """`audit-log-immutability.md` requires who, what, when and which object on every row.

        Checked over the denials a real request sweep produces, rather than over one hand-built
        entry, so a row written by a code path nobody thought to test is still covered.
        """
        client, as_user = api
        as_user(stack.a.user)
        for path in (f"{API}/audit", f"{API}/documents/{stack.b.document.id}"):
            await client.get(path)
        assert stack.denials(), "the sweep produced no denials, so this test asserts nothing"

        missing: list[str] = []
        for entry in stack.denials():
            if entry.user_id is None:
                missing.append(f"{entry.action}: no user_id")
            if entry.occurred_at is None:
                missing.append(f"{entry.action}: no timestamp")
            if not entry.object_type:
                missing.append(f"{entry.action}: no object_type")
            if entry.tenant_id is None:
                missing.append(f"{entry.action}: no tenant_id")
        assert missing == [], f"incomplete audit rows: {missing}"
