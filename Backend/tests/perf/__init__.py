"""backend/tests/perf/

Role: The Phase 6 load run — the query path under concurrency, with p50/p95 and an error rate.

One module, `test_query_load.py`, and one honest claim: this measures **the application's own
cost and its behaviour under concurrency**, not the 3–5 second end-to-end target. Every network
call a real request makes — two Bedrock invocations and a Pinecone query — is faked here, and
those calls are what the target is about. What is genuinely measurable offline is whether
sixty-odd overlapping requests still each get their own answer, their own citations and their
own recorded latency, and whether the app adds milliseconds or hundreds of them.

The production p95 comes from `AnalyticsRepository.summary()` — `percentile_cont(0.95)` over
`query.latency_ms` — against real traffic. `docs/RELEASE.md` records that figure; this suite
records the figure a laptop can produce and says which one it is.
"""
