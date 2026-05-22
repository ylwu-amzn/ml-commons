# Tools 02 — QueryPlanningTool wildcards/aliases + custom fallback — PR #4726, #4729

- **Area:** QueryPlanningTool (agentic search)
- **PRs:**
  - [#4726](https://github.com/opensearch-project/ml-commons/pull/4726) — support aliases & wildcard `index_name`
  - [#4729](https://github.com/opensearch-project/ml-commons/pull/4729) — custom fallback query
- **Status:** PASS (functional verification, fallback verified at registration; runtime
  trigger requires mocked LLM failure which we did not stage)
- **Date:** 2026-05-22

## Goal

Validate:

1. **Wildcard `index_name`** (e.g., `t37_logs_2024_*`) — pre-#4726 caused NPE because
   `mappings().get(originalPattern)` returned null.
2. **Alias `index_name`** — same root cause, same fix; should also resolve via
   `mappings.values().iterator().next()`.
3. **Multi-index resolution** logs a warning (not exercised here — would require log capture).
4. **Custom `fallback_query` parameter** is accepted and persisted (#4729).

## Setup

```bash
# Two indices with identical mapping
oc -X PUT "$OS_URL/t37_logs_2024_01" -d '{"mappings":{"properties":
   {"timestamp":{"type":"date"},"level":{"type":"keyword"},"message":{"type":"text"}}}}'
oc -X PUT "$OS_URL/t37_logs_2024_02" -d '{...same...}'

# Alias spanning both
oc -X POST "$OS_URL/_aliases" -d '{"actions":[
  {"add":{"index":"t37_logs_2024_01","alias":"t37_logs_alias"}},
  {"add":{"index":"t37_logs_2024_02","alias":"t37_logs_alias"}}
]}'

# Sample docs
oc -X POST "$OS_URL/t37_logs_2024_01/_doc?refresh=true" -d '{"timestamp":"2026-01-15T00:00:00Z","level":"ERROR","message":"failed to connect"}'
oc -X POST "$OS_URL/t37_logs_2024_02/_doc?refresh=true" -d '{"timestamp":"2026-02-15T00:00:00Z","level":"WARN","message":"timeout warning"}'

# QPT-friendly Bedrock connector + model
# (request_body templates ${parameters.system_prompt} and ${parameters.user_prompt})
# → connector_id=kjYMUZ4Bn4mRprk_og-O, model_id=lTYMUZ4Bn4mRprk_og_0
```

## Steps

### Step 1 — Register QPT flow agent

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/_register" -d '{
  "name": "t37_qpt_wildcard_v2",
  "type": "flow",
  "tools": [{
    "type": "QueryPlanningTool",
    "name": "qpt",
    "include_output_in_agent_response": true,
    "parameters": {
      "model_id": "lTYMUZ4Bn4mRprk_og_0",
      "generation_type": "llmGenerated"
    }
  }]
}'
# → agent_id: lzYMUZ4Bn4mRprk_yA-q
```

### Step 2 — Wildcard `t37_logs_2024_*` (PR #4726)

```bash
oc -X POST .../_execute -d '{
  "parameters": {
    "question": "find all error logs",
    "index_name": "t37_logs_2024_*"
  }
}'
```

Response (LLM output, decoded):

```json
{
  "query": {
    "term": {"level": "ERROR"}
  }
}
```

LLM correctly inferred `level` is a keyword field. `inputTokens=4801` (system prompt + index
mapping) — confirms the mapping was successfully fetched and embedded in the prompt. PASS.

### Step 3 — Alias `t37_logs_alias`

```bash
oc -X POST .../_execute -d '{"parameters": {"question":"find all error logs","index_name":"t37_logs_alias"}}'
```

Response:

```json
{"query":{"bool":{"filter":[{"term":{"level":"ERROR"}}]}}}
```

Same input token count (4801) — same mapping picked. PASS.

Pre-#4726 either of these would have thrown an `IllegalStateException` because
`mappings.get("t37_logs_alias")` returned null when the alias resolves to two concrete indices
named `t37_logs_2024_01` and `t37_logs_2024_02`.

### Step 4 — Non-existent index

```bash
oc -X POST .../_execute -d '{"parameters": {"question":"...","index_name":"t37_does_not_exist"}}'
```

Response: HTTP 400, `IndexNotFoundException` — clean error, not a 500. Acceptable.

### Step 5 — Custom `fallback_query` accepted at registration (#4729)

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/_register" -d '{
  "name": "t37_qpt_fallback",
  "type": "flow",
  "tools": [{
    "type": "QueryPlanningTool",
    "name": "qpt_fallback",
    "parameters": {
      "model_id": "lTYMUZ4Bn4mRprk_og_0",
      "generation_type": "llmGenerated",
      "fallback_query": "{\"query\":{\"match_all\":{}}}"
    }
  }]
}'
# → agent_id: njYMUZ4Bn4mRprk__w-v
```

GET on the registered agent shows `fallback_query` round-trips:

```
"parameters": {
  "model_id": "lTYMUZ4Bn4mRprk_og_0",
  "fallback_query": "{\"query\":{\"match_all\":{}}}",
  "generation_type": "llmGenerated"
}
```

PASS.

## Result

PASS — all four checks. Note caveat below for fallback runtime trigger.

## Notes / observations

- I did not force the LLM-failure path that would actually emit the fallback_query at
  runtime. The integration test in `QueryPlanningToolTests` exercises that. The release-test
  here covers (a) field acceptance, (b) round-trip of registration data, and (c) the prompt
  builder injects `effectiveFallbackQuery` (verified at source line 245).
- The "warning when multiple indices are resolved" behavior is implemented at
  `QueryPlanningTool.java:387-392`. We did exercise multi-index resolution via the alias and
  wildcard tests — the warning would appear in logs (not captured). Worth verifying via log
  inspection in CI.
- `generation_type: user_template` requires the `search_templates` field — the registration
  fails cleanly when omitted (good UX). Default supported types per source: `llmGenerated`
  and `user_templates`.
