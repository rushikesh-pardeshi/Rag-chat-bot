"""backend/tests/security/

Role: The Phase 6 VAPT suite — one module per vulnerability class named in
`.claude/rules/secure-coding-vapt.md`.

These tests are adversarial by construction. Every other suite in this project asks "does the
feature work"; these ask "can the control be walked around", and each one is written so that
*removing the control makes it fail*. A security test that passes against unprotected code is
worse than no test, because it reports safety it never checked — so several modules here assert
the negative case too (a deliberately-lax fake, a bypassed sanitizer) to prove the assertion
has teeth.

Layout, one file per class:
    test_sql_injection.py    parameterization through every filter that reaches SQL test_xss.py
    stored payloads survive as data, and are escaped on the way out test_log_injection.py
    CR/LF and control characters cannot forge a log line or audit row test_idor.py
    cross-user and cross-tenant ids, and the audit row each denial writes
    test_vector_isolation.py one namespace per tenant, and no document text in metadata
    test_csrf.py             every unsafe route rejects a request with no matching token
    test_path_traversal.py   an upload filename cannot escape the document's own S3 prefix
    test_mass_assignment.py  role, tenant_id, status and integrity_hash are never
    client-supplied test_secrets_in_logs.py  no credential, PII or document body in a log line
    or audit detail

No test here contacts a network service, and only the ones marked `requires_db` need
PostgreSQL.
"""
