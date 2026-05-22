# Streaming 01 — Stream error propagation — PR #4792

- **Area:** AGUI streaming error propagation
- **PR:** [#4792](https://github.com/opensearch-project/ml-commons/pull/4792)
- **Status:** BLOCKED (cluster does not support HTTP streaming under security plugin)
- **Date:** 2026-05-22

## Goal

Validate that errors during a streaming agent execute are unwrapped and emitted as a
structured `RunErrorEvent` (for AG-UI agents) or as a `data: {"error":"…"}` SSE chunk (for
non-AG-UI streaming agents). The fix:

- Unwraps `ex.getCause()` so the actual root-cause message is surfaced (was wrapping it in
  generic "Error processing request:" prefix).
- For `OpenSearchStatusException`, uses the status exception message verbatim.
- For AG-UI agents, emits a `RunErrorEvent` JSON event (structured, parseable by clients).

## Steps attempted

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/nonexistent_id/_execute/stream" \
   -H "Content-Type: application/json" -H "Accept: text/event-stream" \
   -d '{"input":"hi","parameters":{}}'
```

Response:

```
HTTP 500
{"error":{"root_cause":[{"type":"illegal_state_exception",
  "reason":"The engine does not support HTTP streaming,
  unable to serve uri [/_plugins/_ml/agents/nonexistent_id/_execute/stream]
  and method [POST]"}],...}
```

## Why blocked

The cluster runs `org.opensearch.security.http.SecurityHttpServerTransport` (security plugin
override of the default Netty4 HTTP transport). That transport implementation does not
support OpenSearch's HTTP-streaming primitive, so `_execute/stream` cannot be reached at all
on a security-enabled cluster.

```
http.type = org.opensearch.security.http.SecurityHttpServerTransport
http.type.default = netty4
```

## What can/should be done

1. **Unit-test side**: PR #4792 includes 112 lines of new tests in
   `plugin/src/test/java/org/opensearch/ml/rest/RestMLExecuteStreamActionTests.java` — these
   cover the `buildStreamErrorChunk(...)` method directly. Source-level review confirms the
   method correctly:
   - Unwraps `ex.getCause()`.
   - Special-cases `IOException` for parse-failure messages.
   - Special-cases `OpenSearchStatusException` to use its message verbatim.
   - For AGUI: emits `RunErrorEvent.toJsonString()` wrapped in `data: …\n\n`.
   - Otherwise: emits `data: {"error":"…"}\n\n`.

2. **End-to-end side**: To exercise this on a real cluster, either:
   - Disable the security plugin and use the default Netty4 HTTP transport, OR
   - Use a streaming-capable HTTP transport (newer OpenSearch core ships one).

   The OpenSearch core team is tracking streaming support under security in a separate
   issue; see also `MLCommonsSettings.ML_COMMONS_STREAM_ENABLED`.

## Result

BLOCKED in the current test cluster. Source-level fix verified to be correct; recommend
running the e2e test on a non-security cluster or in CI where streaming is enabled before the
3.7 release ships.

## Notes / observations

- `plugins.ml_commons.stream_enabled: true` is set, but is not sufficient on its own — the
  HTTP transport itself must support streaming.
- This is not a regression in 3.7: streaming has always required the streaming-capable HTTP
  transport. The PR is validated by unit tests only on this test cluster.
