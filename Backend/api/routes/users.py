"""backend/api/routes/users.py

Role: Administrator-only read access to the accounts in the caller's own tenant.

`rbac-enforcement.md` lists user access management as an Administrator operation, and the first
half of it is being able to see who is in the workspace at all — including everyone who created
their own account at `/auth/register`, which is otherwise invisible until they ask a question.

**This router is read-only.** The role-change endpoint is a separate decision, still open
as F-05 in `docs/RELEASE.md`, and
`regression/test_rbac.py::TestUserAccessManagementIsNotReachableYet` asserts that no route
answers `PATCH /users/{id}/role` yet. Adding a write here means bringing CSRF on the unsafe
method and the `ROLE_CHANGE` audit row with it, and updating that test in the same change so
the gap closes visibly rather than by accident.

Two checks apply and neither is optional: `AdminUser` refuses a Standard User with a 403 and
audits the denial, and the tenant is read from the caller's row rather than from a parameter,
so there is no request an Administrator in one organization can make that returns another's
staff.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from backend.api.dependencies.auth import AdminUser, AuthServiceDep, RequestCtx
from backend.schemas.user_schemas import TenantUser, TenantUserPage
from backend.services.auth_service import MAX_USER_PAGE

router = APIRouter(prefix="/users", tags=["users"])

#: One screenful. The pager asks for the next page rather than the client asking for everyone.
DEFAULT_PAGE_SIZE = 50


@router.get(
    "",
    response_model=TenantUserPage,
    summary="List the workspace's accounts (Administrator only)",
)
async def list_users(
    actor: AdminUser,
    context: RequestCtx,
    auth: AuthServiceDep,
    limit: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_USER_PAGE),
    offset: int = Query(default=0, ge=0),
) -> TenantUserPage:
    """A page of the caller's own tenant's accounts, newest first.

    Newest first because the question this answers is "who has signed up?", and the answer a
    reader wants is at the top. `created_at` on each row is the sign-up time.

    There is no `tenant_id` parameter, by design — see the module docstring. The route
    validates, delegates once and maps the result; the role check, the tenant scope, the page
    cap and the audit of a denial all live in the layers below (`architecture.md`).
    """
    page = await auth.list_users(
        actor=actor, limit=limit, offset=offset, context=context
    )
    return TenantUserPage(
        items=[TenantUser.from_entity(user) for user in page.users],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
        admins=page.admins,
    )
