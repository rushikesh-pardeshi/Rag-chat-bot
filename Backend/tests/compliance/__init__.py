"""backend/tests/compliance/

Role: The compliance verification suite — the checks a regulatory reviewer would run rather
than the ones a developer would.

Three claims are made here, and each one is the subject of a module:

  * `test_traceability.py` — every stored answer resolves to a *live* document and a *real*
    location. An orphaned reference fails, and the module proves it by making one.
  * `test_vector_lifecycle.py` — PostgreSQL and Pinecone agree about what is citable, at every
    point in a document's life including the window where they disagree.
  * `test_audit_immutability_db.py` — the two immutability controls that only exist in the
    database: the append-only triggers and the revoked `UPDATE`/`DELETE`/`TRUNCATE` grants.
    Marked `requires_db` and skipped when no PostgreSQL is reachable.

A green run of the first two modules says nothing about the third. That is why the third exists
as its own file with its own marker, and why `backend/tests/regression/test_audit_logs.py`
points at it by name.
"""
