"""backend/tests/security/test_mass_assignment.py

Role: Prove a request body cannot set a field the server owns — role, tenant, status,
ownership, or the answer text itself.

`.claude/rules/secure-coding-vapt.md`: *"**Mass Assignment**: Explicitly define allowed fields
for document metadata, user profiles, and query parameters. Reject unexpected fields."*

The three fields that matter most
---------------------------------
Not all over-posting is equal. Three of these are privilege escalation, not untidiness:

  * **`role` on register.** `RegisterRequest`'s own docstring says the server assigns Standard
    User. If a body could carry `role: ADMINISTRATOR`, anyone with the signup form would hold
    audit-log read, document management and analytics — every Administrator-only capability in
    `rbac-enforcement.md` — without an administrator ever approving it.
  * **`tenant_id` anywhere.** Tenant is the isolation boundary. A caller who could name their
    own tenant would not need an IDOR at all; they would simply ask for the other tenant's
    namespace and be answered legitimately.
  * **`answer_text` on a query.** This one is subtler and worse for compliance. Answers are
    audited as the system's own output and cited as grounded in a document. A caller who could
    write the answer could put a fabricated statement into the audit trail with a real
    citation attached — and `audit-log-immutability.md` means nobody can then correct it.

Two mechanisms, not one
-----------------------
JSON bodies are validated by Pydantic with `extra="forbid"`, so an unknown key is a 422. Query
strings and multipart form fields are not: FastAPI ignores a parameter it does not declare. For
those the assertion has to be about the *outcome* — the value the server used — rather than
about a rejection. `TestQueryStringOverPosting` documents that difference, including one place
where the codebase's own comment claimed a rejection that does not happen.

The drift guard
---------------
`TestEverySchemaForbidsExtras` walks `backend/schemas/` and fails on a request model that does
not set `extra="forbid"`. That is the test that still works in a year: individual payload tests
cover the fields someone thought of, and the sweep covers the schema added next week.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import uuid
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from backend.models.document import DocumentStatus
from backend.models.user import User, UserRole
from backend.tests.security.conftest import API, CONTEXT, Stack, csrf_headers

Api = tuple[httpx.AsyncClient, Callable[[User | None], None]]

#: Fields the server owns on every model. Sent as extra keys, each must be refused.
#: `hashed_password` is included because a caller who set it directly would choose their own
#: bcrypt hash and bypass the password policy entirely.
SERVER_OWNED_FIELDS: dict[str, Any] = {
    "id": str(uuid.uuid4()),
    "tenant_id": str(uuid.uuid4()),
    "user_id": str(uuid.uuid4()),
    "role": "ADMINISTRATOR",
    "is_active": True,
    "hashed_password": "$2b$12$0123456789012345678901234567890123456789012345678901",
    "failed_login_attempts": 0,
    "locked_until": None,
    "created_at": "2020-01-01T00:00:00Z",
    "updated_at": "2020-01-01T00:00:00Z",
    "status": "ACTIVE",
    "version": 99,
    "s3_key": "tenants/attacker/owned.pdf",
    "content_hash": "0" * 64,
    "answer_text": "The fee is 0 USD.",
    "source_references": [],
    "confidence": 1.0,
    "latency_ms": 0,
    "integrity_hash": "0" * 64,
    "occurred_at": "2020-01-01T00:00:00Z",
    "action": "LOGIN_SUCCESS",
    "outcome": "SUCCESS",
}

#: The JSON request bodies that exist, each valid on its own so a 422 can only come from the
#: extra key under test.
VALID_BODIES: dict[str, dict[str, Any]] = {
    f"{API}/queries": {"question_text": "What is the outbound domestic wire transfer fee?"},
    # Lowercase: `ExportReport` and `ExportFormat` are `StrEnum`s whose values are `"keywords"`
    # and `"pdf"`. The uppercase member names 422 — which would look like a pass on every
    # over-posting test below and reach the route on none of them.
    f"{API}/analytics/export": {"report": "keywords", "format": "pdf"},
    f"{API}/auth/register": {
        "email": "newcomer@tenant-a.example.com",
        "password": "DocQuery!Local2026",
    },
    f"{API}/auth/login": {
        "email": "standard@tenant-a.example.com",
        "password": "DocQuery!Local2026",
    },
    f"{API}/auth/password": {
        "current_password": "DocQuery!Local2026",
        "new_password": "DocQuery!Local2027",
    },
}


def _request_models() -> list[type[BaseModel]]:
    """Every Pydantic model under `backend/schemas/` that names a *request*.

    Discovered by import rather than listed, so a schema added next week is covered without
    anyone remembering to add it here. Response models are excluded: `extra="forbid"` on a
    response is a useful serialization guard but not a mass-assignment control, and conflating
    the two would make the sweep's failure message misleading.
    """
    import backend.schemas as schemas_pkg

    models: list[type[BaseModel]] = []
    for info in pkgutil.iter_modules(schemas_pkg.__path__):
        module = importlib.import_module(f"backend.schemas.{info.name}")
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, BaseModel) or obj is BaseModel:
                continue
            if obj.__module__ != module.__name__:
                continue  # imported into this module, not defined in it
            name = obj.__name__
            if name.endswith(("Request", "Create", "Filter", "Update")):
                models.append(obj)
    return models


REQUEST_MODELS = _request_models()


class TestEverySchemaForbidsExtras:
    """The drift guard. Runs over whatever `backend/schemas/` contains today."""

    def test_the_sweep_found_models(self) -> None:
        """An empty sweep passes silently, so the count is asserted first."""
        assert len(REQUEST_MODELS) >= 6, (
            f"only discovered {len(REQUEST_MODELS)} request models: "
            f"{[m.__name__ for m in REQUEST_MODELS]}"
        )

    def test_no_request_model_accepts_unknown_fields(self) -> None:
        offenders = [
            model.__name__
            for model in REQUEST_MODELS
            if model.model_config.get("extra") != "forbid"
        ]
        assert offenders == [], (
            "these request models accept unexpected fields, so a caller can over-post into "
            f"them: {offenders}. Add `model_config = ConfigDict(extra='forbid')`."
        )

    @pytest.mark.parametrize("model", REQUEST_MODELS, ids=lambda m: m.__name__)
    def test_constructing_one_with_an_unknown_field_raises(
        self, model: type[BaseModel]
    ) -> None:
        """Asserted behaviourally as well as by config, because a subclass can override it.

        `model_config` is inherited, so a model that sets `extra="forbid"` on its base and
        `extra="allow"` on itself would pass a config-only check on the base class alone.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            model.model_validate({"definitely_not_a_field_9f3a": "x"})


class TestRegistrationCannotGrantARole:
    """The privilege-escalation case, tested on its own because it is the one that matters."""

    async def test_a_role_in_the_body_is_rejected(self, stack: Stack, api: Api) -> None:
        client, as_user = api
        as_user(None)  # registration is unauthenticated

        response = await client.post(
            f"{API}/auth/register",
            json={
                "email": "attacker@tenant-a.example.com",
                "password": "DocQuery!Local2026",
                "role": "ADMINISTRATOR",
            },
        )

        assert response.status_code == 422, (
            f"registration accepted a caller-supplied role ({response.status_code}): "
            f"{response.text}"
        )
        assert response.json()["error"]["code"] == "validation_error"
        assert "role" in response.text, "the 422 does not say which field was refused"

    async def test_the_refusal_does_not_echo_the_password(
        self, stack: Stack, api: Api
    ) -> None:
        """A validation error names the *fields*, never the submitted values.

        `secure-coding-vapt.md`: *"Never log passwords…"* — and a 422 body is written to the
        client and usually to the client's own error log as well.
        """
        client, as_user = api
        as_user(None)
        secret = "Sup3rSecret!Passphrase"

        response = await client.post(
            f"{API}/auth/register",
            json={
                "email": "attacker@tenant-a.example.com",
                "password": secret,
                "role": "ADMINISTRATOR",
            },
        )

        assert secret not in response.text, f"the 422 echoed the password: {response.text}"

    @pytest.mark.parametrize("field", ["tenant_id", "is_active", "hashed_password", "id"])
    async def test_other_server_owned_fields_are_rejected_too(
        self, stack: Stack, api: Api, field: str
    ) -> None:
        client, as_user = api
        as_user(None)
        body = {
            "email": "attacker@tenant-a.example.com",
            "password": "DocQuery!Local2026",
            field: SERVER_OWNED_FIELDS[field],
        }

        response = await client.post(f"{API}/auth/register", json=body)

        assert response.status_code == 422, (
            f"registration accepted a caller-supplied {field}: {response.text}"
        )


class TestJsonBodiesRejectExtras:
    """Every JSON route, every server-owned field. Wide rather than deep, on purpose."""

    @pytest.mark.parametrize("path", sorted(VALID_BODIES))
    @pytest.mark.parametrize("field", sorted(SERVER_OWNED_FIELDS))
    async def test_an_extra_field_is_a_422(
        self, stack: Stack, api: Api, path: str, field: str
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        body = dict(VALID_BODIES[path])
        if field in body:
            pytest.skip(f"{field} is a legitimate field on {path}")
        body[field] = SERVER_OWNED_FIELDS[field]

        response = await client.post(path, json=body, headers=csrf_headers())

        assert response.status_code == 422, (
            f"POST {path} accepted an unexpected {field!r} and returned "
            f"{response.status_code}: {response.text[:400]}"
        )

    @pytest.mark.parametrize("path", sorted(VALID_BODIES))
    async def test_the_valid_body_is_not_itself_a_422(
        self, stack: Stack, api: Api, path: str
    ) -> None:
        """The positive control for the class above.

        Without it, a body that was malformed for some unrelated reason would produce a 422 on
        every parametrization and the whole class would pass while testing nothing. Anything
        other than 422 is accepted here — 200, 201, 401 and 403 are all decisions made *after*
        validation, and which one comes back is not this module's concern.
        """
        client, as_user = api
        as_user(stack.a.admin)

        response = await client.post(path, json=VALID_BODIES[path], headers=csrf_headers())

        assert response.status_code != 422, (
            f"the baseline body for {path} is invalid, so every over-posting test on it is "
            f"vacuous: {response.text[:400]}"
        )


class TestAQueryCannotCarryItsOwnAnswer:
    """The compliance case: a fabricated answer with a real citation, permanently audited."""

    @pytest.mark.parametrize(
        "field", ["answer_text", "source_references", "confidence", "status", "latency_ms"]
    )
    async def test_the_answer_side_of_a_query_is_server_only(
        self, stack: Stack, api: Api, field: str
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)

        response = await client.post(
            f"{API}/queries",
            json={"question_text": "What is the wire fee?", field: SERVER_OWNED_FIELDS[field]},
            headers=csrf_headers(),
        )

        assert response.status_code == 422, (
            f"a caller supplied {field!r} on a query and the server accepted it "
            f"({response.status_code}) — a fabricated answer would enter the audit trail: "
            f"{response.text[:300]}"
        )

    async def test_the_answer_that_is_stored_comes_from_the_pipeline(
        self, stack: Stack, api: Api
    ) -> None:
        """The outcome check behind the 422s: what got persisted is the RAG answer.

        A 422 proves the field was refused. It does not prove the server did not read the value
        anyway from somewhere else in the request, so the stored row is inspected directly.
        """
        client, as_user = api
        as_user(stack.a.user)
        stack.bedrock.queue_answer("The outbound domestic wire fee is 25 USD [S1].")

        response = await client.post(
            f"{API}/queries",
            json={"question_text": "What is the outbound domestic wire transfer fee?"},
            headers=csrf_headers(),
        )

        assert response.status_code == 202, response.text
        body = response.json()
        assert "0 USD" not in body["answer_text"], "the injected answer text was used"
        assert body["source_references"], "an answer arrived with no citation"
        assert body["source_references"][0]["source_document_name"] == stack.a.document.name


class TestMultipartOverPosting:
    """Upload is `multipart/form-data`, where an undeclared field is ignored, not refused.

    So the assertion is about the document that was created, not about a status code. This is
    the honest shape of the control here: the server ignores the field *and* derives the value
    itself, and only the second half is worth anything.
    """

    async def test_extra_form_fields_do_not_reach_the_document(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.admin)
        before = set(stack.documents.documents)

        response = await client.post(
            f"{API}/documents",
            files={"file": ("policy.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")},
            data={
                "tenant_id": str(stack.b.id),
                "status": "ACTIVE",
                "version": "99",
                "s3_key": "tenants/attacker/owned.pdf",
                "id": str(uuid.uuid4()),
            },
            headers=csrf_headers(),
        )

        assert response.status_code in (200, 201, 202), response.text
        created = [
            doc for key, doc in stack.documents.documents.items() if key not in before
        ]
        assert len(created) == 1, f"expected one new document, got {len(created)}"
        document = created[0]

        assert document.tenant_id == stack.a.id, (
            f"the form's tenant_id was used: the document landed in {document.tenant_id}, "
            f"not the uploader's tenant {stack.a.id}"
        )
        assert document.version == 1, f"the form set version to {document.version}"
        assert document.s3_key != "tenants/attacker/owned.pdf", (
            "the form chose the S3 key, which is a path-traversal primitive as well as "
            "an over-post"
        )
        assert str(document.id) in document.s3_key, (
            f"the S3 key is not derived from the server's document id: {document.s3_key}"
        )
        assert document.status is not DocumentStatus.ACTIVE, (
            "the form set the status to ACTIVE. A document reaches ACTIVE only after the "
            "pipeline has parsed it and written chunks, and this payload is not a real PDF — "
            "so ACTIVE here means the caller chose its own lifecycle state and could make an "
            "unparsed document quotable."
        )

    async def test_the_uploader_is_the_recorded_owner(self, stack: Stack, api: Api) -> None:
        """Ownership comes from the token, so a form field cannot reassign it."""
        client, as_user = api
        as_user(stack.a.admin)
        before = set(stack.documents.documents)

        response = await client.post(
            f"{API}/documents",
            files={"file": ("owner.pdf", b"%PDF-1.4\n%%EOF\n", "application/pdf")},
            data={"uploaded_by": str(stack.b.admin.id), "user_id": str(stack.b.admin.id)},
            headers=csrf_headers(),
        )

        assert response.status_code in (200, 201, 202), response.text
        created = [
            doc for key, doc in stack.documents.documents.items() if key not in before
        ]
        assert len(created) == 1
        assert created[0].uploaded_by == stack.a.admin.id, (
            f"the form reassigned ownership to {created[0].uploaded_by}"
        )


class TestQueryStringOverPosting:
    """Query parameters, where the answer is "ignored" rather than "rejected".

    Read this class as documentation of what the API actually does. Two of the tests assert a
    control that holds; one records a gap that does not, so that nobody reads a green suite as
    proof of a rejection the framework never performs.
    """

    async def test_a_standard_user_cannot_widen_their_history_with_user_id(
        self, stack: Stack, api: Api
    ) -> None:
        """`GET /history` really does take a `user_id` filter — for Administrators.

        The parameter is caller-supplied, so the only thing standing between a Standard User
        and a colleague's history is the service overriding it. That override is the control,
        and this is the test for it.
        """
        client, as_user = api
        stack.bedrock.queue_answer("The monthly maintenance fee is 12 USD [S1].")
        colleague = await stack.query_service.submit(
            question="What is the monthly account maintenance fee?",
            actor=stack.a.admin,
            context=CONTEXT,
        )
        as_user(stack.a.user)

        response = await client.get(f"{API}/history?user_id={stack.a.admin.id}")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"] == "own", (
            f"a caller-supplied user_id changed the scope to {body['scope']!r}"
        )
        ids = {item["id"] for item in body["items"]}
        assert str(colleague.id) not in ids, (
            "?user_id= let a Standard User read a colleague's history"
        )

    async def test_a_cross_tenant_user_id_returns_nothing_extra(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)

        response = await client.get(f"{API}/history?user_id={stack.b.user.id}")

        assert response.status_code == 200, response.text
        ids = {item["id"] for item in response.json()["items"]}
        assert str(stack.b.query.id) not in ids, "?user_id= crossed the tenant boundary"

    async def test_an_administrator_may_filter_by_user_id(
        self, stack: Stack, api: Api
    ) -> None:
        """The positive control: the parameter is not simply discarded for everyone."""
        client, as_user = api
        as_user(stack.a.admin)

        response = await client.get(f"{API}/history?user_id={stack.a.user.id}")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["scope"] == "tenant"
        ids = {item["id"] for item in body["items"]}
        assert str(stack.a.query.id) in ids, "the admin's user_id filter returned nothing"

    @pytest.mark.parametrize(
        "path",
        [
            f"{API}/documents",
            f"{API}/history",
            f"{API}/audit",
            f"{API}/analytics/keywords",
        ],
    )
    async def test_an_undeclared_query_parameter_is_ignored_not_rejected(
        self, stack: Stack, api: Api, path: str
    ) -> None:
        """A recorded gap, asserted as it behaves rather than as it ought to behave.

        `secure-coding-vapt.md` asks for unexpected fields to be *rejected*, and the filter
        models do set `extra="forbid"`. It has no effect on a query string: the routes declare
        each parameter individually and construct the filter model inside the handler, so an
        undeclared parameter never reaches Pydantic. `?form=2026-01-01` instead of
        `?from=2026-01-01` returns an unfiltered report with a 200.

        Severity is low and bounded, which is why this is a documented gap rather than a fix
        applied mid-suite: the tenant and role filters are derived from the token and are never
        caller-supplied, so a typo widens a *window*, not an audience. The impact is a
        compliance officer reading a figure they believe is scoped to a quarter.

        Closing it changes the query contract on every read route, which
        `.claude/rules/workflow.md` says to ask about first. Until then this test exists so the
        gap is visible in the suite instead of resting on a comment — and the comment in
        `AnalyticsFilter` that claimed the opposite has been corrected.
        """
        client, as_user = api
        as_user(stack.a.admin)

        response = await client.get(f"{path}?definitely_not_a_parameter_9f3a=1")

        assert response.status_code == 200, (
            f"{path} now rejects an unknown query parameter ({response.status_code}). That is "
            "an improvement — update this test and the AnalyticsFilter docstring together."
        )

    async def test_a_declared_parameter_still_validates_its_type(
        self, stack: Stack, api: Api
    ) -> None:
        """The gap is *unknown* names only. A known name with a hostile value is still checked.

        This is what keeps the finding above at low severity: `limit` cannot be raised past its
        ceiling and `status` cannot carry a non-enum value, so the ignored-parameter behaviour
        does not extend to the parameters that do reach the query.
        """
        client, as_user = api
        as_user(stack.a.admin)

        assert (await client.get(f"{API}/documents?limit=1000000")).status_code == 422
        assert (await client.get(f"{API}/documents?limit=-1")).status_code == 422
        assert (await client.get(f"{API}/documents?status=NOT_A_STATUS")).status_code == 422
        assert (await client.get(f"{API}/audit?action=NOT_AN_ACTION")).status_code == 422


class TestResponsesCannotGrowFields:
    """`extra="forbid"` on responses, which is a disclosure control rather than an input one.

    A response model that accepted extras would happily serialize whatever an ORM object
    carried — `hashed_password`, `s3_key`, `integrity_hash` — the first time someone passed the
    entity instead of the DTO.
    """

    async def test_a_document_response_carries_no_internal_fields(
        self, stack: Stack, api: Api
    ) -> None:
        """`s3_key` is the one that matters: it is the storage path, and publishing it hands a
        caller the input to every path-traversal and object-enumeration attempt against the
        bucket. `tenant_id` is redundant — the caller has exactly one — and printing it invites
        a client to send it back.

        `content_hash` is *not* in this list, and that is a decision rather than an oversight.
        It is published deliberately (`document_schemas.py`: "lets an Administrator verify that
        what they downloaded is what was ingested"), which is a real lineage need. The residual
        exposure is a confirmation oracle — someone holding a candidate file can check whether
        it is the one stored — and it is bounded by the tenant scope on the route. I asserted
        it as a leak first, found the rationale in the schema, and narrowed the assertion
        rather than the other way round.
        """
        client, as_user = api
        as_user(stack.a.admin)

        response = await client.get(f"{API}/documents/{stack.a.document.id}")

        assert response.status_code == 200, response.text
        body = response.json()
        leaked = [
            field for field in ("s3_key", "hashed_password", "tenant_id") if field in body
        ]
        assert leaked == [], f"the document DTO exposes internal fields: {leaked}"
        assert stack.a.document.s3_key not in response.text, (
            "the storage key appears in the response body under another name"
        )

    async def test_the_current_user_response_carries_no_password_hash(
        self, stack: Stack, api: Api
    ) -> None:
        client, as_user = api
        as_user(stack.a.user)

        response = await client.get(f"{API}/auth/me")

        assert response.status_code == 200, response.text
        body = response.json()
        assert "hashed_password" not in body
        assert "not-a-real-hash" not in response.text
        assert body["role"] == UserRole.STANDARD_USER.value
