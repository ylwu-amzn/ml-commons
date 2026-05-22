# Streaming 01 — Stream error propagation — PR #4792

- **Area:** AGUI streaming error propagation
- **PR:** [#4792](https://github.com/opensearch-project/ml-commons/pull/4792)
- **Status:** PASS (re-run against streaming-capable cluster on `localhost:9201`)
- **Date:** 2026-05-22
- **Cluster:** OpenSearch 3.7.0, ml-commons 3.7.0-SNAPSHOT, **`transport-reactor-netty4`
  installed**

## Goal

Validate that errors during a streaming agent execute are unwrapped and surfaced as structured
SSE chunks rather than wrapped/swallowed:

- Pre-#4792 the error path always emitted `"data: {\"error\": \"Error processing request: <wrapped>\"}"`,
  losing the underlying cause.
- Post-#4792:
  - For `IOException` parse errors → `"Failed to parse request: <jackson msg>"` verbatim.
  - For other errors → `cause.getMessage()` (unwrapped).
  - For AG-UI agents → emit a structured `RunErrorEvent` JSON event.

## Why this re-run is possible

The first cluster (`localhost:9200`) was security-only and used
`SecurityHttpServerTransport`, which does not support HTTP streaming, so the streaming
endpoint returned 500 *before* the error-propagation code path ran.

The second cluster (`localhost:9201`) has the **`transport-reactor-netty4`** plugin
installed, which does provide HTTP-streaming support. SSE chunks are emitted correctly
under TLS + HTTP/2 + security.

## Setup

```bash
export OS_URL=https://localhost:9201
export OS_AUTH='admin:RUIwNTVGTDc2MDhDM0JNRC4u'
oc() { curl -sk -u "$OS_AUTH" "$@"; }
```

The cluster has a pre-deployed V1 conversational agent backed by Bedrock Claude Sonnet 4.5
via Converse stream:
- Agent: `uwfJUJ4BAr57BVLX_dJV` (`Chat agent`, type `CONVERSATIONAL`).

## Steps & Results

### Step 1 — Happy-path streaming (baseline)

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/uwfJUJ4BAr57BVLX_dJV/_execute/stream" \
   -H "Content-Type: application/json" -H "Accept: text/event-stream" \
   --no-buffer -m 30 \
   -d '{"parameters":{"question":"Say hi in 3 words","stream":"true"}}'
```

Response (first 4 SSE chunks):

```
data: {"inference_results":[{"output":[{"name":"memory_id","result":"DwcfUZ4BAr57BVLXcNT8"},
       {"name":"parent_interaction_id","result":"EAcfUZ4BAr57BVLXcdQN"},
       {"name":"response","dataAsMap":{"content":"Hi","is_last":false}}]}]}

data: {"inference_results":[{...,"dataAsMap":{"content":" there friend","is_last":false}}]}]}

data: {"inference_results":[{...,"dataAsMap":{"content":"!","is_last":false}}]}]}

data: {"inference_results":[{...,"dataAsMap":{"content":"","is_last":false}}]}]}
```

PASS — multi-chunk SSE stream with `is_last` flag.

### Step 2 — Error path A: malformed JSON body (IOException → unwrapped Jackson msg)

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/uwfJUJ4BAr57BVLX_dJV/_execute/stream" \
   -H "Content-Type: application/json" -H "Accept: text/event-stream" \
   --no-buffer -m 10 \
   -d '{"parameters":{"question":"hi"'   # truncated body, missing closing braces
```

Response (single SSE error chunk):

```
data: {"error": "Failed to parse request: Unexpected end-of-input: expected close marker for Object (start marker at [Source: REDACTED (`StreamReadFeature.INCLUDE_SOURCE_IN_LOCATION` disabled); byte offset: #UNKNOWN])
 at [Source: REDACTED (`StreamReadFeature.INCLUDE_SOURCE_IN_LOCATION` disabled); byte offset: #30]"}
```

PASS — exactly the behavior described in the PR: `IOException` path produces
`"Failed to parse request: <jackson msg>"`. The Jackson 3.x marker
(`StreamReadFeature.INCLUDE_SOURCE_IN_LOCATION`) is also visible — confirms #4795 in effect.

### Step 3 — Error path B: missing required parameter (cause unwrap)

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/uwfJUJ4BAr57BVLX_dJV/_execute/stream" \
   -H "Content-Type: application/json" -H "Accept: text/event-stream" \
   --no-buffer -m 10 \
   -d '{}'
```

Response:

```
data: {"error": "Expected RemoteInferenceInputDataSet for agent execution"}
```

PASS — note the message is the **underlying** error reason, not the pre-#4792 wrapping
(`"Error processing request: <generic>"`). This proves the cause-unwrapping branch in
`buildStreamErrorChunk()` is in effect (source: `RestMLExecuteStreamAction.java:625-647`).

### Step 4 — Synchronous validation: non-existent agent

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/nonexistent_id/_execute/stream" \
   -H "Content-Type: application/json" -H "Accept: text/event-stream" \
   -d '{"input":"hi","parameters":{}}'
```

Response: HTTP 404, regular JSON body
(`"Failed to find agent with the provided agent id: nonexistent_id"`).

This is correct behavior — agent existence is validated synchronously before the SSE
channel opens, so a hard 404 is more useful than an SSE error chunk.

## Result

PASS — streaming endpoint emits SSE chunks; #4792's error paths fire and unwrap correctly.

## Negative tests we attempted but couldn't usefully drive

### a) Mid-stream Bedrock 401 (intentionally bad credentials)

We registered a model whose connector hard-codes wrong AWS credentials, then put a V1
conversational agent in front of it, and streamed. Synchronous predict against the same
model returned a clean 403 with `"The security token included in the request is invalid"`.
The streaming variant **opens** the SSE channel (HTTP 200) but emits no chunks within 60s
and does not surface a structured error. This may be a separate gap (the bad-creds error
inside the streaming Bedrock client doesn't propagate to the `onErrorResume` handler in
`RestMLExecuteStreamAction`), worth reporting as a follow-up issue. Not a regression — the
PR fix only protects the `onErrorResume` path; if upstream silently drops the error, the
fix can't help.

### b) AG-UI streaming (test the `RunErrorEvent` branch)

We registered an `ag_ui` agent and confirmed `unified_agent_api_enabled=true` and
`ag_ui_enabled=true`. The endpoint opens HTTP 200 SSE but never emits a chunk in the happy
path either — looks like the AGUI runner's streaming is not wired through to the same
publisher, or requires additional setup. The `RunErrorEvent` JSON branch of #4792's
`buildStreamErrorChunk()` (`isAGUI=true`) was therefore not exercised end-to-end — only
the unit tests in the PR cover it. This is worth a CI-side e2e test in a future cycle.

### c) V2 chat agent streaming

Cleanly **not supported** in 3.7. Trying `/agents/{v2}/_execute/stream` returns:

```
HTTP 400
"V2 agents (CONVERSATIONAL_V2) do not support streaming. Please use the non-streaming
 execute endpoint: POST /_plugins/_ml/agents/.../_execute"
```

This is a deliberate, documented restriction — not a bug. Worth surfacing in V2 docs.

## Notes / observations

- **Use this cluster (`localhost:9201`) for any streaming test.** The `localhost:9200`
  cluster has `SecurityHttpServerTransport` and **does not** support `_execute/stream`.
- The `transport-reactor-netty4` plugin must be installed for streaming to work.
- The cluster also has `arrow-flight-rpc` (gRPC streaming) — separate code path, not tested
  here.
- Settings touched on this cluster:
  `plugins.ml_commons.unified_agent_api_enabled=true`,
  `plugins.ml_commons.ag_ui_enabled=true`. (Required for V2 / AGUI registration.)

## Result update

- Step 1 (happy path streaming): PASS
- Step 2 (malformed JSON → unwrapped Jackson msg): PASS — primary #4792 fix verified
- Step 3 (missing param → cause unwrapped): PASS — primary #4792 fix verified
- Step 4 (404 on missing agent — sync path): PASS

Overall: **PASS** for the parts of #4792 we could exercise. AGUI `RunErrorEvent` branch
remains unit-test-only.
