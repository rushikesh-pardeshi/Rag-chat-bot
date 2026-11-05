"""backend/tests/regression/test_document_management.py

Role: BR-01 (Document Management) and the Data Integrity family (C-04, R-01, R-09) asserted
over HTTP.

The requirement, quoted from `WORKBREAKDOWN.md`:

    Document Management: Administrators upload, preview, delete, restore documents with
    metadata (name, upload date, uploader, status) and full data lineage tracking
    (BR-DM, BR-01)

Every clause of that sentence is a test below, and each is driven through the shipped route
rather than through `DocumentService`. That distinction earns its keep: "with metadata" is a
claim about `DocumentResponse`, "Administrators" is a claim about the `AdminUser` dependency,
and "full data lineage tracking" is a claim about what `DELETE` does *not* erase. None of the
three is visible from inside the service.

The two clauses worth reading twice
-----------------------------------
**Delete is a status change and nothing else.** `TestSoftDeletePreservesEverything` asserts the
row, the chunk rows and the S3 object all survive a delete, because the object is what makes a
restore possible and what an old citation points at. A delete that removed bytes would satisfy
the word "delete" and break BR-01's "restore".

**Restore is a re-ingest, not a status flip.** `document_service.restore` re-runs the
pipeline, so the chunks and vectors an answer needs come back rather than being assumed still
present. `TestRestoreReturnsDocumentToService` asserts the document is quotable again, not
merely that the status string changed.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest

from backend.models.document import DocumentStatus
from backend.schemas.document_schemas import PUBLIC_METADATA_KEYS
from backend.tests.document_fixtures import PLAIN_TEXT, WINDOWS_EXECUTABLE
from backend.tests.regression.conftest import (
    API,
    FEE_DOCX,
    FEE_PAGES,
    FEE_PDF,
    REVISED_FEE_PDF,
    Api,
    Stack,
    audit_rows,
    csrf_headers,
    new_uuid,
    switch_to,
    upload,
)

DOCX_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class TestAdministratorUploadsADocument:
    """"Administrators upload ... documents with metadata (name, upload date, uploader,
    status)". Both accepted formats, and the role gate on the verb."""

    async def test_a_pdf_upload_returns_201_and_becomes_retrievable(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        response = await upload(client)

        assert response.status_code == 201
        body = response.json()
        assert body["name"] == "retail-account-fees.pdf"
        assert body["content_type"] == "application/pdf"
        assert body["status"] == DocumentStatus.ACTIVE.value
        assert body["is_retrievable"] is True
        # Four pages in, four pages recorded: the count is read back from the parser, not
        # echoed from the request.
        assert body["page_count"] == len(FEE_PAGES)

    async def test_a_docx_upload_is_accepted_too(self, api: Api, stack: Stack) -> None:
        """BR-01 says PDF *or* DOCX. Asserting only the format that is easier to build is how
        a DOCX-specific parser regression ships unnoticed."""
        client = switch_to(api, stack.a.admin)

        response = await upload(
            client,
            filename="retail-fees.docx",
            data=FEE_DOCX,
            content_type=DOCX_CONTENT_TYPE,
        )

        assert response.status_code == 201
        assert response.json()["status"] == DocumentStatus.ACTIVE.value

    async def test_the_response_names_the_uploader_and_the_upload_date(
        self, api: Api, stack: Stack
    ) -> None:
        """"metadata (name, upload date, uploader, status)" — all four, on the DTO.

        `uploaded_by` is the session's user id, never a body field: `POST /documents` takes a
        file and nothing else, so there is no way for a caller to attribute an upload to
        someone else.
        """
        client = switch_to(api, stack.a.admin)

        body = (await upload(client)).json()

        assert body["uploaded_by"] == str(stack.a.admin.id)
        assert body["upload_date"]
        assert body["status"]
        assert body["name"]

    async def test_a_standard_user_cannot_upload(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        response = await upload(client)

        assert response.status_code == 403
        assert not stack.s3.keys(), "a refused upload must not have written bytes"

    async def test_an_unauthenticated_upload_is_401(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, None)

        assert (await upload(client)).status_code == 401

    async def test_a_renamed_executable_is_refused(self, api: Api, stack: Stack) -> None:
        """"document validation checks ... before approval" (C-04, R-01).

        The decision is made on magic bytes, so a `.pdf` extension over a PE header does not
        get through. Kept in the regression suite as well as the unit suite because it is a
        *requirement*, not only an implementation detail.
        """
        client = switch_to(api, stack.a.admin)

        response = await upload(client, filename="statement.pdf", data=WINDOWS_EXECUTABLE)

        assert response.status_code == 400

    async def test_a_plain_text_file_is_refused(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        response = await upload(client, filename="notes.pdf", data=PLAIN_TEXT)

        assert response.status_code == 400

    async def test_an_empty_file_is_refused(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        response = await upload(client, filename="empty.pdf", data=b"")

        assert response.status_code == 400


class TestListingAndReadingDocuments:
    """Any authenticated user, own tenant only. A citation nobody can open is not
    traceability, so reading is *not* Administrator-gated."""

    async def test_an_uploaded_document_appears_in_the_list(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        page = (await client.get(f"{API}/documents")).json()

        assert uploaded["id"] in [item["id"] for item in page["items"]]
        assert page["total"] >= 1

    async def test_a_standard_user_sees_their_own_tenants_documents(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        client = switch_to(api, stack.a.user)
        response = await client.get(f"{API}/documents/{uploaded['id']}")

        assert response.status_code == 200
        assert response.json()["id"] == uploaded["id"]

    async def test_the_list_never_crosses_a_tenant_boundary(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        ids = [item["id"] for item in (await client.get(f"{API}/documents")).json()["items"]]

        assert stack.b.document is not None
        assert str(stack.b.document.id) not in ids

    async def test_another_tenants_document_id_is_404(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)
        assert stack.b.document is not None

        response = await client.get(f"{API}/documents/{stack.b.document.id}")

        assert response.status_code == 404

    async def test_an_unknown_id_is_404(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)

        assert (await client.get(f"{API}/documents/{new_uuid()}")).status_code == 404

    async def test_the_status_filter_narrows_the_page(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)
        await upload(client)

        page = (await client.get(f"{API}/documents?status=DELETED")).json()

        assert page["items"] == []

    async def test_published_metadata_is_restricted_to_the_allowlist(
        self, api: Api, stack: Stack
    ) -> None:
        """The parser stores diagnostics the client has no business seeing. `metadata` is an
        allowlist, so a new parser key is not published by default."""
        client = switch_to(api, stack.a.admin)

        body = (await upload(client)).json()

        assert set(body["metadata"]).issubset(PUBLIC_METADATA_KEYS)

    async def test_the_storage_key_is_never_published(self, api: Api, stack: Stack) -> None:
        """`s3_key` is on the ORM entity and absent from the DTO. A bucket path in a response
        tells an attacker exactly what to try against the bucket directly."""
        client = switch_to(api, stack.a.admin)

        body = (await upload(client)).json()

        assert "s3_key" not in body
        assert "documents/" not in repr(body)


class TestPreviewIssuesAShortLivedUrl:
    """"Administrators ... preview documents" — and Standard Users preview the documents a
    citation points at, which is why the route is not Administrator-gated."""

    async def test_preview_returns_a_presigned_url_with_a_ttl(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        body = (await client.get(f"{API}/documents/{uploaded['id']}/preview")).json()

        assert body["url"].startswith("https://")
        assert body["expires_in_seconds"] > 0
        assert body["document_id"] == uploaded["id"]

    async def test_a_standard_user_can_preview_a_cited_document(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        client = switch_to(api, stack.a.user)
        response = await client.get(f"{API}/documents/{uploaded['id']}/preview")

        assert response.status_code == 200

    async def test_the_presigned_url_is_not_written_to_the_audit_trail(
        self, api: Api, stack: Stack
    ) -> None:
        """A presigned URL is a bearer credential that cannot be recalled. The audit row says
        a preview was issued; it does not carry the URL."""
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        body = (await client.get(f"{API}/documents/{uploaded['id']}/preview")).json()

        assert "X-Amz-Expires" not in stack.audit_details()
        assert body["url"] not in stack.audit_details()

    async def test_preview_of_another_tenants_document_is_404(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        assert stack.b.document is not None

        response = await client.get(f"{API}/documents/{stack.b.document.id}/preview")

        assert response.status_code == 404


class TestSoftDeletePreservesEverything:
    """"delete ... documents" with "full data lineage tracking". Nothing is erased.

    This is the Data Integrity requirement in its sharpest form: a delete that removed the
    bytes would satisfy the verb and break the restore, and it would leave every previously
    issued citation pointing at nothing.
    """

    async def test_delete_returns_the_document_in_its_new_state(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        response = await client.delete(
            f"{API}/documents/{uploaded['id']}", headers=csrf_headers()
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == DocumentStatus.DELETED.value
        assert body["is_retrievable"] is False

    async def test_the_row_the_chunks_and_the_bytes_all_survive(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        document_id = uuid.UUID(uploaded["id"])
        keys_before = stack.s3.keys()
        chunks_before = await stack.documents.count_chunks(document_id, stack.a.id)

        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())

        assert stack.s3.keys() == keys_before, "soft delete must not touch S3"
        assert document_id in stack.documents.documents, "the row must survive"
        assert chunks_before > 0
        assert await stack.documents.count_chunks(document_id, stack.a.id) == chunks_before

    async def test_the_vectors_are_removed_so_it_stops_being_quotable(
        self, api: Api, stack: Stack
    ) -> None:
        """The chunk *rows* stay (lineage); the *vectors* go (retrieval). Two different
        stores, two different obligations."""
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        namespace = stack.namespace(stack.a.id)
        vectors_before = stack.index.count(namespace)

        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())

        assert stack.index.count(namespace) < vectors_before

    async def test_a_standard_user_cannot_delete(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        client = switch_to(api, stack.a.user)
        response = await client.delete(
            f"{API}/documents/{uploaded['id']}", headers=csrf_headers()
        )

        assert response.status_code == 403
        surviving = stack.documents.documents[uuid.UUID(uploaded["id"])]
        assert surviving.status is DocumentStatus.ACTIVE

    async def test_a_deleted_document_disappears_for_a_standard_user(
        self, api: Api, stack: Stack
    ) -> None:
        """A Standard User's visible statuses exclude DELETED, so this is a 404 for them and a
        200 for an Administrator — the same row, two answers, both correct."""
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())

        assert (await client.get(f"{API}/documents/{uploaded['id']}")).status_code == 200

        client = switch_to(api, stack.a.user)
        assert (await client.get(f"{API}/documents/{uploaded['id']}")).status_code == 404

    async def test_deleting_twice_is_idempotent(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        first = await client.delete(
            f"{API}/documents/{uploaded['id']}", headers=csrf_headers()
        )
        second = await client.delete(
            f"{API}/documents/{uploaded['id']}", headers=csrf_headers()
        )

        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["status"] == DocumentStatus.DELETED.value


class TestRestoreReturnsDocumentToService:
    """"restore documents". A re-ingest, so what comes back is a *searchable* document."""

    async def test_restore_makes_the_document_active_and_quotable_again(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())

        response = await client.post(
            f"{API}/documents/{uploaded['id']}/restore", headers=csrf_headers()
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == DocumentStatus.ACTIVE.value
        assert body["is_retrievable"] is True

    async def test_restore_puts_the_vectors_back(self, api: Api, stack: Stack) -> None:
        """The point of re-ingesting rather than flipping a status. A restored document whose
        vectors never came back would look healthy and answer nothing."""
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        namespace = stack.namespace(stack.a.id)
        after_upload = stack.index.count(namespace)

        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())
        await client.post(
            f"{API}/documents/{uploaded['id']}/restore", headers=csrf_headers()
        )

        assert stack.index.count(namespace) == after_upload

    async def test_restoring_a_live_document_is_409(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        response = await client.post(
            f"{API}/documents/{uploaded['id']}/restore", headers=csrf_headers()
        )

        assert response.status_code == 409

    async def test_a_standard_user_cannot_restore(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())

        client = switch_to(api, stack.a.user)
        response = await client.post(
            f"{API}/documents/{uploaded['id']}/restore", headers=csrf_headers()
        )

        assert response.status_code == 403


class TestVersionLineage:
    """"full data lineage tracking" — every upload of a name is its own immutable version."""

    async def test_re_uploading_a_name_creates_a_second_version(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)

        first = (await upload(client)).json()
        second = (await upload(client, data=REVISED_FEE_PDF)).json()

        assert first["version"] == 1
        assert second["version"] == 2
        assert first["id"] != second["id"]

    async def test_both_versions_keep_their_own_bytes(self, api: Api, stack: Stack) -> None:
        """Our IAM role cannot enable bucket versioning, so lineage is app-side: one immutable
        key per version and nothing is ever overwritten."""
        client = switch_to(api, stack.a.admin)

        await upload(client)
        await upload(client, data=REVISED_FEE_PDF)

        keys = stack.s3.keys()
        assert len(keys) == 2, "each version gets its own immutable key"
        assert any("/v1/" in key for key in keys)
        assert any("/v2/" in key for key in keys)

    async def test_the_versions_endpoint_lists_them_oldest_first(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        await upload(client)
        second = (await upload(client, data=REVISED_FEE_PDF)).json()

        body = (await client.get(f"{API}/documents/{second['id']}/versions")).json()

        assert body["name"] == "retail-account-fees.pdf"
        assert [item["version"] for item in body["versions"]] == [1, 2]

    async def test_the_versions_endpoint_is_administrator_only(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()

        client = switch_to(api, stack.a.user)
        response = await client.get(f"{API}/documents/{uploaded['id']}/versions")

        assert response.status_code == 403

    async def test_identical_bytes_are_recorded_as_a_duplicate_not_refused(
        self, api: Api, stack: Stack
    ) -> None:
        """Two uploads of the same file are a real operational event — a re-send after a
        timeout. The second is accepted as version 2 and *flagged*, because refusing it would
        lose the evidence that someone tried."""
        client = switch_to(api, stack.a.admin)

        first = (await upload(client)).json()
        second = (await upload(client)).json()

        assert first["content_hash"] == second["content_hash"]
        assert second["version"] == 2
        uploads = audit_rows(
            stack, "DOCUMENT_UPLOAD", actor_email=stack.a.admin.email
        )
        assert uploads[-1].detail["duplicate_content"] is True

    async def test_the_content_hash_is_the_lineage_fingerprint(
        self, api: Api, stack: Stack
    ) -> None:
        """Published so an Administrator can verify a download is what was ingested."""
        client = switch_to(api, stack.a.admin)

        body = (await upload(client)).json()

        assert body["content_hash"] == hashlib.sha256(FEE_PDF).hexdigest()


class TestEveryLifecycleActionIsAudited:
    """BR-01 meets BR-05: "Auto-log document uploads/deletions/restorations"."""

    async def test_upload_delete_and_restore_each_write_a_row(
        self, api: Api, stack: Stack
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())
        await client.post(
            f"{API}/documents/{uploaded['id']}/restore", headers=csrf_headers()
        )

        actions = stack.audit_repo.actions()
        assert "DOCUMENT_UPLOAD" in actions
        assert "DOCUMENT_DELETE" in actions
        assert "DOCUMENT_RESTORE" in actions

    async def test_every_row_names_the_document_it_is_about(
        self, api: Api, stack: Stack
    ) -> None:
        """`object_id` is the document id — never the name, which is user-supplied text."""
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        await client.delete(f"{API}/documents/{uploaded['id']}", headers=csrf_headers())

        for action in ("DOCUMENT_UPLOAD", "DOCUMENT_DELETE"):
            rows = audit_rows(stack, action, actor_email=stack.a.admin.email)
            assert rows, f"no {action} row attributed to the acting Administrator"
            assert rows[-1].object_id == uploaded["id"]

    async def test_no_audit_detail_repeats_the_document_text(
        self, api: Api, stack: Stack
    ) -> None:
        """A log that quotes a banking document is a second, less-protected copy of customer
        content (`secure-coding-vapt.md`, Secrets/PII)."""
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        await client.get(f"{API}/documents/{uploaded['id']}/preview")

        written = stack.audit_details()
        for page in FEE_PAGES:
            assert page[:60] not in written

    async def test_a_refused_upload_leaves_a_denial(self, api: Api, stack: Stack) -> None:
        client = switch_to(api, stack.a.user)

        await upload(client)

        denials = stack.denials()
        assert denials, "a 403 with no audit row is itself an audit gap"
        assert denials[-1].action.value == "ACCESS_DENIED"


@pytest.mark.parametrize(
    "method,path_suffix",
    [
        ("post", ""),
        ("delete", "/{document_id}"),
        ("post", "/{document_id}/restore"),
    ],
)
class TestStateChangingRoutesRequireCsrf:
    """The JWT lives in an httpOnly cookie, so the browser attaches it to cross-site requests
    too. That coupling is what makes CSRF protection mandatory here rather than optional —
    `secure-coding-vapt.md` states the two decisions change together."""

    async def test_without_a_token_the_request_is_refused(
        self, api: Api, stack: Stack, method: str, path_suffix: str
    ) -> None:
        client = switch_to(api, stack.a.admin)
        uploaded = (await upload(client)).json()
        path = f"{API}/documents{path_suffix.format(document_id=uploaded['id'])}"

        response = await client.request(method.upper(), path)

        assert response.status_code == 403
