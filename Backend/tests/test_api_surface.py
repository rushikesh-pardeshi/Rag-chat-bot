"""backend/tests/test_api_surface.py

Role: Assertions about the *shape* of the HTTP API — which paths exist and which methods they
accept — read from the generated OpenAPI document rather than from any handler.

Two things are checked here that cannot be checked anywhere else.

**The audit log has no write surface.** `.claude/rules/audit-log-immutability.md` requires "no
programmatic deletion or modification interfaces". `audit_repository` having no UPDATE method
is necessary but not sufficient: the guarantee a reviewer can act on is that the deployed API
exposes no way to reach one. That is a property of the router table, so it is tested against
the router table. A `PATCH /api/v1/audit/{id}` added by a well-meaning future change fails
here immediately, at the cheapest test in the suite.

**The routers are actually wired.** `include_router` is easy to forget, and a missing router
fails as a 404 in a manual click-through rather than as a test failure. Note that this FastAPI
version wraps included routers in objects with no `.path`, so `app.routes` cannot be inspected
directly — `app.openapi()['paths']` is the reliable source.

No database is needed: generating the schema never calls a handler.
"""

from __future__ import annotations

import pytest

from backend.api.app import create_app

#: Methods that cannot alter server state. Everything else is a write as far as this file is
#: concerned — including `POST`, which is how a "just an export endpoint" would arrive.
READ_ONLY_METHODS = {"get", "head", "options"}


@pytest.fixture(scope="module")
def paths() -> dict[str, list[str]]:
    """`{path: [lowercase methods]}` from the generated OpenAPI document."""
    spec = create_app().openapi()
    return {path: sorted(operations) for path, operations in spec["paths"].items()}


# -- Audit immutability at the HTTP surface -----------------------------------
class TestAuditHasNoWriteSurface:
    def test_every_audit_path_is_read_only(self, paths: dict[str, list[str]]) -> None:
        audit_paths = {
            path: methods
            for path, methods in paths.items()
            if path.startswith("/api/v1/audit")
        }
        assert audit_paths, "the audit router is not registered at all"

        writable = {
            path: sorted(set(methods) - READ_ONLY_METHODS)
            for path, methods in audit_paths.items()
            if set(methods) - READ_ONLY_METHODS
        }
        assert writable == {}, f"audit log exposes state-changing methods: {writable}"

    def test_the_audit_paths_are_exactly_the_two_reads(
        self, paths: dict[str, list[str]]
    ) -> None:
        """Asserted exactly, so a third audit path has to be justified by a reviewer.

        `/audit` is the compliance read and `/audit/verify` reports chain integrity. Neither
        accepts a body.
        """
        audit = {p: m for p, m in paths.items() if p.startswith("/api/v1/audit")}
        assert audit == {"/api/v1/audit": ["get"], "/api/v1/audit/verify": ["get"]}


# -- Wiring ------------------------------------------------------------------
class TestPhaseThreeRoutesAreRegistered:
    @pytest.mark.parametrize(
        ("path", "method"),
        [
            ("/api/v1/documents", "post"),
            ("/api/v1/documents", "get"),
            ("/api/v1/documents/{document_id}", "get"),
            ("/api/v1/documents/{document_id}", "delete"),
            ("/api/v1/documents/{document_id}/restore", "post"),
            ("/api/v1/documents/{document_id}/preview", "get"),
            ("/api/v1/documents/{document_id}/versions", "get"),
            ("/api/v1/queries", "post"),
            ("/api/v1/queries/{query_id}", "get"),
            ("/api/v1/audit", "get"),
        ],
    )
    def test_the_route_exists(
        self, paths: dict[str, list[str]], path: str, method: str
    ) -> None:
        assert method in paths.get(path, []), f"{method.upper()} {path} is not registered"

    def test_destruction_has_exactly_one_route_and_it_is_not_the_delete_route(
        self, paths: dict[str, list[str]]
    ) -> None:
        """Two delete routes, and the difference between them is the point.

        **This assertion used to forbid the second one.** It read "a separate purge or
        permanent route would erase the object an audit entry refers to, breaking the
        traceability the delete route deliberately preserves" — and it was right about the
        risk and wrong about the remedy. Administrators needed a way to honour an erasure
        request, so the capability was added deliberately, with the traceability concern
        answered rather than ignored:

          * `audit_log` has **no foreign key** to `document`, so every entry — the upload,
            each query that cited it, and the `DOCUMENT_PURGE` itself — outlives the row. An
            erasure leaves a trail; it does not leave a gap.
          * Purge **refuses anything that is not already `DELETED`** (409), so no single
            request can destroy a document that is still in circulation.
          * It is Administrator-only, checked in the service as well as by the route's
            dependency, because this is the one operation with nothing to undo it.

        What this test now pins is the shape that keeps those properties true: destruction
        lives at exactly one path, that path is explicit about what it does, and the plain
        `DELETE /documents/{id}` stays a soft delete. A third destructive route appearing —
        or `purge` turning up on some other resource — is what should fail here.
        """
        deletes = sorted(
            path
            for path, methods in paths.items()
            if "delete" in methods and path.startswith("/api/v1/documents")
        )
        assert deletes == [
            "/api/v1/documents/{document_id}",
            "/api/v1/documents/{document_id}/purge",
        ]
        # One destructive route in the whole API, and it says so in its path. `permanent`
        # and `hard-delete` stay forbidden so a second spelling of the same capability
        # cannot be added without this test being read again.
        destructive = sorted(path for path in paths if "purge" in path)
        assert destructive == ["/api/v1/documents/{document_id}/purge"]
        assert not any(
            token in path for path in paths for token in ("permanent", "hard-delete")
        )


class TestPhaseFourAndFiveRoutesAreRegistered:
    @pytest.mark.parametrize(
        ("path", "method"),
        [
            ("/api/v1/history", "get"),
            ("/api/v1/analytics/keywords", "get"),
            ("/api/v1/analytics/documents", "get"),
            ("/api/v1/analytics/trends", "get"),
            ("/api/v1/analytics/export", "post"),
        ],
    )
    def test_the_route_exists(
        self, paths: dict[str, list[str]], path: str, method: str
    ) -> None:
        assert method in paths.get(path, []), f"{method.upper()} {path} is not registered"

    def test_the_export_is_the_only_write_under_analytics(
        self, paths: dict[str, list[str]]
    ) -> None:
        """One unsafe method, and it is the export.

        The export is a `POST` because it *records* state — every export appends a
        `DATA_EXPORT` audit row — and because an unsafe method is what carries the CSRF token.
        A second write appearing here would mean analytics grew a mutation, which it has no
        reason to have: every other endpoint is a read over `query`.
        """
        writable = {
            path: sorted(set(methods) - READ_ONLY_METHODS)
            for path, methods in paths.items()
            if path.startswith("/api/v1/analytics") and set(methods) - READ_ONLY_METHODS
        }
        assert writable == {"/api/v1/analytics/export": ["post"]}

    def test_history_is_a_projection_not_a_resource(
        self, paths: dict[str, list[str]]
    ) -> None:
        """`architecture.md`: `search_history` is a role-filtered view over `query`.

        So history is **read-only**: no way to write, delete or edit a row. A
        `DELETE /history/{id}` would let a user erase the record of what they asked, which is
        the audited fact, and a `PATCH` would let them edit it — the same defect wearing a
        different verb.

        Asserted as "every method is a GET" rather than against a literal list of one path.
        The invariant is the absence of mutation, not the count of read routes: `/history` and
        `/history/stats` are both reads of the same projection, and pinning the path list made
        adding the second one look like a violation when nothing about the guarantee changed.
        A new *verb* still fails this, which is the part that matters.
        """
        history = {p: m for p, m in paths.items() if p.startswith("/api/v1/history")}
        assert history != {}, "the history projection has no routes at all"
        mutating = {
            path: methods
            for path, methods in history.items()
            if any(method != "get" for method in methods)
        }
        assert mutating == {}, (
            f"history gained a non-GET route: {mutating}. It is a projection over `query` — a "
            f"write here would let a user alter or erase the record of what they asked."
        )
        # And no `/{id}` resource: a single history row is not addressable, because the thing
        # worth addressing is the query, which `/queries/{query_id}` already owns.
        assert not any("{" in path for path in history), (
            f"history gained a path parameter: {sorted(history)}"
        )

    def test_everything_is_under_the_versioned_prefix(
        self, paths: dict[str, list[str]]
    ) -> None:
        """Except the two probes, which exist for a load balancer that knows no versions."""
        unversioned = {
            path for path in paths if not path.startswith("/api/v1/")
        }
        assert unversioned == {"/health", "/readiness"}
