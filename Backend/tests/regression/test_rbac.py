"""backend/tests/regression/test_rbac.py

Role: BR-06 (Role-Based Access Control) asserted over HTTP, plus the Security family's
server-side authorization clause (NFR-9.2, C-01, R-07).

The requirement, quoted from `WORKBREAKDOWN.md`:

    Role-Based Access Control: Two roles — Administrator (document management, analytics,
    system-wide history, user access) and Standard User (query, own history, download
    referenced documents); enforced server-side on every route (BR-06, R-07)

Two dimensions, not one
-----------------------
`.claude/rules/rbac-enforcement.md`: "Role is not the only check. Every query must also be
scoped by `tenant_id`. An Administrator in tenant A must not see tenant B's data." So every
route has to be wrong in two independent ways before it is safe, and this module tests both —
`TestAdministratorOnlyOperations` for the role axis, `TestRoleAndTenantAreIndependent` for the
other. An Administrator of the wrong tenant is the more dangerous of the two, because the role
check passes.

How this module differs from `backend/tests/security/test_rbac_bypass.py`
------------------------------------------------------------------------
The security module asks whether the control can be *evaded* — forged claims, path tricks,
parameter injection. This one asks whether the control is still *there*, route by route, as the
requirement describes it. The overlap is deliberate: a route added later without a role
dependency is invisible to a bypass test (there is nothing to bypass) and fails here.

Where this module reports a gap rather than a guarantee
------------------------------------------------------
`rbac-enforcement.md` lists "User access management" as an Administrator-only operation.
`AuthService.set_role` exists and `RoleChangeRequest` is defined, but **no route reaches
either**, so the operation cannot be performed over HTTP at all. That is recorded as a finding
in `docs/RELEASE.md`, and `TestUserAccessManagementIsNotReachableYet` asserts the current state
so that whoever adds the route is told, by a failing test, to bring the role check with it.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from backend.models.audit_log import AuditAction
from backend.models.document import DocumentStatus
from backend.models.user import UserRole
from backend.tests.regression.conftest import (
    API,
    TAX_QUESTION,
    Api,
    Stack,
    csrf_headers,
    switch_to,
    upload,
)

#: Every Administrator-only endpoint, as `(method, path template, body)`. `{document}` is
#: substituted with an id in the caller's own tenant, so a refusal here is about role and about
#: nothing else — a nonexistent id would 404 for both roles and prove nothing.
#:
#: Kept as one table because the requirement is a list of operations, and a table is the only
#: shape in which "did we cover all of them?" is answerable by reading.
ADMIN_ONLY: list[tuple[str, str, dict[str, Any] | None]] = [
    ("DELETE", f"{API}/documents/{{document}}", None),
    ("POST", f"{API}/documents/{{document}}/restore", None),
    ("GET", f"{API}/documents/{{document}}/versions", None),
    ("GET", f"{API}/users", None),
    ("GET", f"{API}/audit", None),
    ("GET", f"{API}/audit/verify", None),
    ("GET", f"{API}/analytics/keywords", None),
    ("GET", f"{API}/analytics/documents", None),
    ("GET", f"{API}/analytics/trends", None),
    ("POST", f"{API}/analytics/export", {"report": "keywords", "format": "excel"}),
]

#: Endpoints any authenticated user may reach. `POST /documents` is absent on purpose: it is
#: document management, and it gets its own test rather than a row here, since it needs
#: multipart rather than JSON.
USER_ALLOWED: list[tuple[str, str, dict[str, Any] | None]] = [
    ("GET", f"{API}/documents", None),
    ("GET", f"{API}/documents/{{document}}", None),
    ("GET", f"{API}/documents/{{document}}/preview", None),
    ("GET", f"{API}/history", None),
    ("GET", f"{API}/auth/me", None),
    ("POST", f"{API}/queries", {"question_text": TAX_QUESTION}),
]


async def call(
    client: Any, method: str, path: str, body: dict[str, Any] | None
) -> Any:
    """One request, with a CSRF token on the unsafe methods.

    The token is supplied so that a 403 in these tests is always the *role* refusing, never the
    CSRF middleware — `backend/tests/security/test_csrf.py` owns the other question.
    """
    kwargs: dict[str, Any] = {"headers": csrf_headers()}
    if body is not None:
        kwargs["json"] = body
    return await client.request(method, path, **kwargs)


def fill(path: str, stack: Stack) -> str:
    assert stack.a.document is not None
    return path.format(document=stack.a.document.id)


class TestTheTwoRolesAreTheWholeModel:
    """"Two roles" is a closed statement, and `rbac-enforcement.md` says so twice: "Do not add
    a third role to the enum without an explicit decision — it would need its own permission
    set"."""

    def test_there_are_exactly_two_roles(self) -> None:
        assert {role.value for role in UserRole} == {"ADMINISTRATOR", "STANDARD_USER"}

    def test_there_is_no_compliance_officer_role(self) -> None:
        """Several documents in this workspace say "administrators and compliance officers".
        That is one role and one job title, not two roles: a compliance officer is a person
        holding ADMINISTRATOR. A separate enum member would need its own permission set — audit
        read without document management — and nobody has decided that."""
        assert not [role for role in UserRole if "COMPLIANCE" in role.name]

    async def test_the_role_is_published_on_the_profile(
        self, api: Api, stack: Stack
    ) -> None:
        """The UI needs it to choose a navigation, and a reviewer needs it to see that role is
        something the server states rather than something the client assumes."""
        client = switch_to(api, stack.a.user)

        body = (await client.get(f"{API}/auth/me")).json()

        assert body["role"] == "STANDARD_USER"


class TestAdministratorOnlyOperations:
    """"Document management, analytics, system-wide history, user access" — refused to a
    Standard User on every route that performs one."""

    @pytest.mark.parametrize(("method", "path", "body"), ADMIN_ONLY)
    async def test_a_standard_user_is_refused(
        self, api: Api, stack: Stack, method: str, path: str, body: dict[str, Any] | None
    ) -> None:
        client = switch_to(api, stack.a.user)

        response = await call(client, method, fill(path, stack), body)

        assert response.status_code == 403, f"{method} {path}"

    @pytest.mark.parametrize(("method", "path", "body"), ADMIN_ONLY)
    async def test_an_administrator_is_not_refused(
        self, api: Api, stack: Stack, method: str, path: str, body: dict[str, Any] | None
    ) -> None:
        """The other half of the assertion, and the half that catches a route locked so hard
        nobody can use it. Only "not 403" is claimed — `POST /restore` on a document that was
        never deleted is a legitimate 409, and this test is about role, not about outcome."""
        client = switch_to(api, stack.a.admin)

        response = await call(client, method, fill(path, stack), body)

        assert response.status_code != 403, f"{method} {path}"

    async def test_a_standard_user_cannot_upload(self, api: Api, stack: Stack) -> None:
        """Upload is document management. Separate from the table because it is multipart, and
        the payload is a real PDF so the refusal cannot be a validation error in disguise."""
        client = switch_to(api, stack.a.user)

        assert (await upload(client)).status_code == 403

    async def test_an_administrator_can_upload(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await upload(client)).status_code == 201

    async def test_a_standard_user_cannot_read_another_users_history(
        self, api: Api, stack: Stack
    ) -> None:
        """"System-wide search history viewing" is the Administrator's. A Standard User's
        `user_id` filter is discarded rather than honoured, so the parameter cannot widen the
        scope — see `test_search_history.py` for the full BR-04 treatment.

        Ownership is checked against the stored row, because `QueryResponse` publishes no
        `user_id`: an Administrator who needs to know whose row it is filters by user, and the
        row itself does not carry the answer.
        """
        client = switch_to(api, stack.a.user)

        body = (await client.get(f"{API}/history?user_id={stack.a.admin.id}")).json()

        assert body["scope"] == "own"
        owners = {
            stack.queries.queries[uuid.UUID(item["id"])].user_id
            for item in body["items"]
        }
        assert owners == {stack.a.user.id}

    async def test_an_administrator_reads_the_whole_tenants_history(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        body = (await client.get(f"{API}/history")).json()

        assert body["scope"] == "tenant"


class TestStandardUserOperations:
    """"Submit queries, view answers with source references, own history, download referenced
    documents" — available, not merely un-refused."""

    @pytest.mark.parametrize(("method", "path", "body"), USER_ALLOWED)
    async def test_a_standard_user_may_reach_it(
        self, api: Api, stack: Stack, method: str, path: str, body: dict[str, Any] | None
    ) -> None:
        client = switch_to(api, stack.a.user)

        response = await call(client, method, fill(path, stack), body)

        assert response.status_code < 400, f"{method} {path}"

    async def test_a_standard_user_can_ask_and_read_the_answer_back(
        self, api: Api, stack: Stack
    ) -> None:
        """The whole point of the Standard User role in one request pair."""
        client = switch_to(api, stack.a.admin)
        await upload(client)
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("Withholding tax is deducted at 10 percent [S1].")

        submitted = await client.post(
            f"{API}/queries",
            json={"question_text": TAX_QUESTION},
            headers=csrf_headers(),
        )
        read_back = await client.get(f"{API}/queries/{submitted.json()['id']}")

        assert submitted.status_code == 202
        assert read_back.status_code == 200
        assert read_back.json()["source_references"]

    async def test_a_standard_user_can_download_a_referenced_document(
        self, api: Api, stack: Stack
    ) -> None:
        """"Download referenced source documents" is a Standard User operation, which is why
        `GET /documents/{id}/preview` takes `CurrentUser` and not `AdminUser`.

        That is a deliberate reading of a conflict in the rules: `rbac-enforcement.md` lists
        "preview" among the Administrator-only operations *and* lists downloading a referenced
        document among the Standard User's. A citation the user cannot open is not traceable to
        them, so the narrower reading would defeat BR-03; the tenant and status checks still
        apply, and a deleted document is still refused (see below).
        """
        client = switch_to(api, stack.a.admin)
        document = (await upload(client)).json()
        client = switch_to(api, stack.a.user)

        response = await client.get(f"{API}/documents/{document['id']}/preview")

        assert response.status_code == 200
        assert response.json()["url"]

    async def test_a_standard_user_cannot_read_another_users_query(
        self, api: Api, stack: Stack
    ) -> None:
        """Same tenant, different person. A 404 rather than a 403, because a 403 would confirm
        the query exists — the id came from somewhere."""
        client = switch_to(api, stack.a.admin)
        stack.bedrock.queue_answer("The fee is stated in the schedule [S1].")
        theirs = await client.post(
            f"{API}/queries",
            json={"question_text": TAX_QUESTION},
            headers=csrf_headers(),
        )
        client = switch_to(api, stack.a.user)

        response = await client.get(f"{API}/queries/{theirs.json()['id']}")

        assert response.status_code == 404

    async def test_an_administrator_may_read_any_query_in_the_tenant(
        self, api: Api, stack: Stack
    ) -> None:
        """The asymmetry is the requirement, not an oversight: an Administrator reviewing what
        the organization asked is exactly what "system-wide search history" means."""
        client = switch_to(api, stack.a.user)
        stack.bedrock.queue_answer("The fee is stated in the schedule [S1].")
        theirs = await client.post(
            f"{API}/queries",
            json={"question_text": TAX_QUESTION},
            headers=csrf_headers(),
        )
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{API}/queries/{theirs.json()['id']}")

        assert response.status_code == 200


class TestRoleAndTenantAreIndependent:
    """"An Administrator in tenant A must not see tenant B's data." The role check passes here,
    which is what makes these the dangerous cases."""

    async def test_an_administrator_cannot_read_another_tenants_document(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.b.document is not None
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{API}/documents/{stack.b.document.id}")

        assert response.status_code == 404

    async def test_an_administrator_cannot_delete_another_tenants_document(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.b.document is not None
        client = switch_to(api, stack.a.admin)

        response = await client.delete(
            f"{API}/documents/{stack.b.document.id}", headers=csrf_headers()
        )

        assert response.status_code == 404
        assert stack.b.document.status is not DocumentStatus.DELETED

    async def test_an_administrator_cannot_read_another_tenants_query(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.b.query is not None
        client = switch_to(api, stack.a.admin)

        response = await client.get(f"{API}/queries/{stack.b.query.id}")

        assert response.status_code == 404

    async def test_an_administrators_audit_page_holds_one_tenant(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        items = (await client.get(f"{API}/audit?limit=200")).json()["items"]

        own = {
            str(entry.log_id)
            for entry in stack.audit_repo.entries
            if entry.tenant_id == stack.a.id
        }
        assert items
        assert all(item["log_id"] in own for item in items)

    async def test_an_administrators_history_holds_one_tenant(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.b.query is not None
        client = switch_to(api, stack.a.admin)

        body = (await client.get(f"{API}/history")).json()

        assert str(stack.b.query.id) not in [item["id"] for item in body["items"]]

    async def test_a_document_list_holds_one_tenant(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.b.document is not None
        client = switch_to(api, stack.a.admin)

        body = (await client.get(f"{API}/documents?limit=100")).json()

        assert str(stack.b.document.id) not in [item["id"] for item in body["items"]]


class TestRoleIsTakenFromTheServer:
    """`dependencies/auth.py` states it plainly: the token is the only source of identity, and
    `tenant_id` and `role` come from signed claims — never from a header, query parameter or
    body. A caller who could name their own role would hold every permission."""

    async def test_a_role_in_the_body_is_rejected(self, api: Api, stack: Stack) -> None:
        """`extra='forbid'` on every request DTO, so this is a 422 rather than a silently
        ignored field — mass assignment cannot even be attempted."""
        client = switch_to(api, stack.a.user)

        response = await client.post(
            f"{API}/queries",
            json={"question_text": TAX_QUESTION, "role": "ADMINISTRATOR"},
            headers=csrf_headers(),
        )

        assert response.status_code == 422

    async def test_a_role_in_the_query_string_changes_nothing(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        response = await client.get(f"{API}/audit?role=ADMINISTRATOR")

        assert response.status_code == 403

    async def test_a_role_in_a_header_changes_nothing(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        response = await client.get(
            f"{API}/audit", headers={"X-Role": "ADMINISTRATOR", "X-User-Role": "admin"}
        )

        assert response.status_code == 403

    async def test_a_tenant_in_the_query_string_changes_nothing(
        self, api: Api, stack: Stack
    ) -> None:
        assert stack.b.document is not None
        client = switch_to(api, stack.a.admin)

        response = await client.get(
            f"{API}/documents/{stack.b.document.id}?tenant_id={stack.b.id}"
        )

        assert response.status_code == 404


class TestUnauthenticatedIsRefusedBeforeRole:
    """401, not 403. An anonymous caller learning that a path exists and needs a *role* is a
    small disclosure, and the status codes are the one place it would leak."""

    @pytest.mark.parametrize(("method", "path", "body"), ADMIN_ONLY + USER_ALLOWED)
    async def test_no_session_is_401(
        self, api: Api, stack: Stack, method: str, path: str, body: dict[str, Any] | None
    ) -> None:
        client = switch_to(api, None)

        response = await call(client, method, fill(path, stack), body)

        assert response.status_code == 401, f"{method} {path}"


class TestDeletedDocumentsAreAdministrativelyVisibleOnly:
    """A restore candidate is document management, so a Standard User's view of the repository
    stops at the statuses they can act on."""

    async def test_a_standard_user_asking_for_deleted_gets_an_empty_page(
        self, api: Api, stack: Stack
    ) -> None:
        """Empty rather than 403, deliberately. A 403 on `status=DELETED` would confirm that
        deleted documents exist, and the count is itself information about the repository."""
        client = switch_to(api, stack.a.admin)
        document = (await upload(client)).json()
        await client.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
        client = switch_to(api, stack.a.user)

        response = await client.get(f"{API}/documents?status=DELETED")

        assert response.status_code == 200
        assert response.json()["items"] == []

    async def test_a_standard_user_does_not_see_a_deleted_document_in_the_default_list(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        document = (await upload(client)).json()
        await client.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
        client = switch_to(api, stack.a.user)

        body = (await client.get(f"{API}/documents?limit=100")).json()

        assert document["id"] not in [item["id"] for item in body["items"]]

    async def test_an_administrator_can_list_deleted_documents(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        document = (await upload(client)).json()
        await client.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())

        body = (await client.get(f"{API}/documents?status=DELETED")).json()

        assert document["id"] in [item["id"] for item in body["items"]]

    async def test_a_standard_user_cannot_fetch_a_deleted_document(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        document = (await upload(client)).json()
        await client.delete(f"{API}/documents/{document['id']}", headers=csrf_headers())
        client = switch_to(api, stack.a.user)

        response = await client.get(f"{API}/documents/{document['id']}")

        assert response.status_code == 404


class TestEveryAccessAttemptIsAudited:
    """`rbac-enforcement.md`: "Log all access attempts (authorized and denied) to audit
    trail"."""

    async def test_a_denial_is_recorded_with_its_reason(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        await client.get(f"{API}/audit")

        denials = stack.denials()
        assert denials
        assert any(
            (entry.detail or {}).get("reason") == "administrator_required"
            for entry in denials
        )

    async def test_a_denial_records_the_path_and_method_and_nothing_else(
        self, api: Api, stack: Stack
    ) -> None:
        """Path and method are enough to investigate. The body is not recorded because on
        `POST /analytics/export` or `POST /queries` it would put a question or a report
        specification into an append-only log (`secure-coding-vapt.md`)."""
        client = switch_to(api, stack.a.user)

        await client.post(
            f"{API}/analytics/export",
            json={"report": "keywords", "format": "excel"},
            headers=csrf_headers(),
        )

        entry = next(
            e
            for e in stack.denials()
            if (e.detail or {}).get("path", "").endswith("/analytics/export")
        )
        assert entry.detail is not None
        assert entry.detail["method"] == "POST"
        assert entry.detail["role"] == "STANDARD_USER"
        assert set(entry.detail) == {"reason", "path", "method", "role"}

    async def test_a_denial_names_who_attempted_it(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.user)

        await client.get(f"{API}/analytics/keywords")

        assert any(
            entry.actor_email == stack.a.user.email
            and entry.user_id == stack.a.user.id
            and entry.tenant_id == stack.a.id
            for entry in stack.denials()
        )

    async def test_an_authorized_administrator_action_is_recorded_too(
        self, api: Api, stack: Stack
    ) -> None:
        """Both halves of "authorized and denied". A log of refusals alone cannot answer "who
        exported the data?", which is the question that follows a leak."""
        client = switch_to(api, stack.a.admin)
        before = len(stack.audit_repo.entries)

        await client.get(f"{API}/analytics/keywords")

        assert len(stack.audit_repo.entries) > before
        assert "ANALYTICS_VIEW" in stack.audit_repo.actions()

    async def test_a_cross_tenant_404_is_still_recorded_as_a_denial(
        self, api: Api, stack: Stack
    ) -> None:
        """A 404 that is really an authorization failure must not look like a missing row in
        the audit trail — `audit_denied` exists for exactly this."""
        assert stack.b.document is not None
        client = switch_to(api, stack.a.admin)
        before = len(stack.denials())

        await client.get(f"{API}/documents/{stack.b.document.id}")

        assert len(stack.denials()) > before


class TestUserAccessManagementIsNotReachableYet:
    """A recorded gap, asserted so that closing it cannot happen quietly.

    `rbac-enforcement.md` lists "User access management" as an Administrator-only operation.
    The *read* half of it now exists — `GET /users` lists the tenant's accounts, and it is in
    `ADMIN_ONLY` above — but `AuthService.set_role` and `RoleChangeRequest` are still unrouted,
    so an Administrator cannot promote or demote anyone over HTTP. These tests assert the
    *current* state, not a desirable one; the finding is triaged as F-05 in `docs/RELEASE.md`,
    and whoever adds the route will see them fail and know to add `AdminUser`, a CSRF-protected
    unsafe method, and a `ROLE_CHANGE` audit row with it.

    `PATCH /users` answers 405 rather than 404 now that the read router is mounted. Both are
    accepted below: the assertion is that no route *handles* it, and which of the two FastAPI
    picks depends on whether the path exists for another method.
    """

    async def test_no_role_change_route_exists(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        for path in (
            f"{API}/users/{stack.a.user.id}/role",
            f"{API}/auth/users/{stack.a.user.id}/role",
            f"{API}/users",
        ):
            response = await client.request(
                "PATCH", path, json={"role": "ADMINISTRATOR"}, headers=csrf_headers()
            )
            assert response.status_code in {404, 405}, path

    def test_the_audit_action_for_it_is_nonetheless_defined(self) -> None:
        """`ROLE_CHANGE` is already in the enum and in the database type, so the route can be
        added without a migration when the decision is made."""
        assert AuditAction.ROLE_CHANGE.value == "ROLE_CHANGE"
