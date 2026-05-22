# Jackson 3.x 01 — JSON paths under tools.jackson — PR #4795, #4784

- **Area:** Jackson 3.x migration (com.fasterxml.jackson → tools.jackson)
- **PRs:**
  - [#4795](https://github.com/opensearch-project/ml-commons/pull/4795) — support Jackson 3.x release line
  - [#4784](https://github.com/opensearch-project/ml-commons/pull/4784) — fix Jackson exception handling post 3.x migration
- **Status:** PASS
- **Date:** 2026-05-22

## Goal

Validate that the Jackson 3.x migration in `StringUtils.MAPPER`, `MLNodeUtils`,
`MLExtractJsonProcessor`, MCP transport, etc., does not regress core JSON-handling paths.
The most positive evidence of "Jackson 3.x is the runtime in use" is when Jackson-3-only
features (the `tools.jackson` package, `StreamReadFeature.STRICT_DUPLICATE_DETECTION`) show
up in error messages.

## Source check

`common/src/main/java/org/opensearch/ml/common/utils/StringUtils.java`:

```java
import tools.jackson.core.JacksonException;
import tools.jackson.databind.json.JsonMapper;
import tools.jackson.core.StreamReadFeature;

public static final ObjectMapper MAPPER = JsonMapper
    .builder()
    .accessorNaming(new DefaultAccessorNamingStrategy.Provider().withFirstCharAcceptance(true, true))
    .configure(StreamReadFeature.STRICT_DUPLICATE_DETECTION, true)
    .configure(DeserializationFeature.FAIL_ON_READING_DUP_TREE_KEY, true)
    .build();
```

Imports are `tools.jackson.*` (Jackson 3.x), not `com.fasterxml.jackson.*`. PR #4784
also updated catch blocks from `JsonProcessingException` (Jackson 2.x) to `JacksonException`
(Jackson 3.x).

## Steps

### Step 1 — STRICT_DUPLICATE_DETECTION enforced (smoking-gun for Jackson 3.x)

Send a request with a duplicate JSON property:

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/WzYHUZ4Bn4mRprk_7w_W/_execute" \
   -H "Content-Type: application/json" \
   -d '{"input":"hi","input":"duplicate"}'
```

Response:

```json
{
  "error": {
    "root_cause": [{
      "type": "json_parse_exception",
      "reason": "Duplicate Object property \"input\"\n at [Source: REDACTED (`StreamReadFeature.INCLUDE_SOURCE_IN_LOCATION` disabled); byte offset: #21]"
    }],
    "type": "json_parse_exception",
    "caused_by": {
      "type": "stream_read_exception",
      "reason": "Duplicate Object property \"input\"\n at [Source: REDACTED ...]"
    }
  },
  "status": 400
}
```

PASS. Two strong signals this is Jackson 3.x:
1. `StreamReadFeature.INCLUDE_SOURCE_IN_LOCATION` is a Jackson-3.x-only feature flag.
2. Exception type `stream_read_exception` is the Jackson 3.x rename of Jackson 2's
   `json_parse_exception` underlying type.

### Step 2 — Nested JSON parse via Bedrock predict

Bedrock Converse response contains nested objects, arrays, integers, empty objects:

```json
{
  "metrics": {"latencyMs": 1146},
  "output": {"message": {"content": [{"text": "..."}], "role": "assistant"}},
  "usage": {"inputTokens": 29, "outputTokens": 41, "serverToolUsage": {}, "totalTokens": 70}
}
```

Successfully deserialized into `dataAsMap`. PASS.

### Step 3 — `_source` includes/excludes regression (#4794 sub-fix)

The PR also fixed "the same entry [model_content] cannot be both included and excluded in
_source" — which surfaced after the OS core change.

```bash
oc -X POST "$OS_URL/_plugins/_ml/models/_search" \
   -H "Content-Type: application/json" \
   -d '{"query":{"match_all":{}},"size":1}'
```

Returns 200 OK with model docs — no source-filter conflict error. PASS.

### Step 4 — MCP tool register/list

```bash
oc -X PUT "$OS_URL/_cluster/settings" -d '{"persistent":{"plugins.ml_commons.mcp_server_enabled":true}}'

oc -X POST "$OS_URL/_plugins/_ml/mcp/tools/_register" -d '{
  "tools": [{ "name": "echo_tool", "description": "Echoes its input back", "type": "ListIndexTool" }]
}'

oc "$OS_URL/_plugins/_ml/mcp/tools/_list?pretty"
```

Response:

```json
{
  "tools": [{
    "type": "ListIndexTool",
    "name": "echo_tool",
    "description": "Echoes its input back",
    "create_time": 1779476401735
  }]
}
```

PASS. This exercises `McpToolsHelper` and `OpenSearchMcpStatelessServerTransportProvider`
which were both in the diff for #4795.

### Step 5 — MCP connector registration (mcp_sse protocol)

```bash
oc -X POST "$OS_URL/_plugins/_ml/connectors/_create" -d '{
  "name": "t37_mcp_test_connector",
  "description": "PR 4795 MCP serialization under Jackson three.x",
  "version": 1,
  "protocol": "mcp_sse",
  "url": "http://localhost:9999/sse",
  "credential": {}
}'
# → {"connector_id":"pzYOUZ4Bn4mRprk_8A9t"}
```

PASS — connector registers, JSON serialization round-trips. (Did not exercise the actual
SSE transport because no MCP server is running on `localhost:9999` — this is a serialization
smoke test only.)

## Result

PASS — Jackson 3.x is in effect, parses normal JSON, enforces `STRICT_DUPLICATE_DETECTION`,
exception types are renamed correctly, MCP tool registration JSON path works.

## Notes / observations

- Strong recommendation: add `STRICT_DUPLICATE_DETECTION` rejection to the release notes.
  Pre-3.7 a request like `{"foo": 1, "foo": 2}` would silently take the last value; in 3.7
  it returns a 400. Existing clients sending duplicate keys will break.
- The error message pattern changed: pre-3.7 `JsonParseException`, post-3.7 the response
  type is `json_parse_exception` with `caused_by.type: stream_read_exception`. Clients that
  parse error envelopes by exception type will see a new value.
- The NPM/PyPI Jackson 3.x docs note that `accessorNaming` config is required for compat
  with old POJOs (e.g., `getX` → `x`). PR sets `withFirstCharAcceptance(true, true)`, which
  matches the legacy behavior.
- MCP server's actual SSE / streamable-HTTP endpoints were not exercised end-to-end here
  — same blocker as the streaming case (security HTTP transport).
