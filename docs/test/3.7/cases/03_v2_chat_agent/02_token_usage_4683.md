# V2 Chat Agent 02 — Token usage tracking — PR #4683

- **Area:** Token usage tracking on Conversational, V2, PER agents
- **PR:** [#4683](https://github.com/opensearch-project/ml-commons/pull/4683)
- **Status:** PASS
- **Date:** 2026-05-22

## Goal

Validate `include_token_usage` opt-in flag and the per-model / per-turn token tracking
returned by V1 Conversational agents and V2 chat agents.

## Steps

### Step 1 — V2 chat agent (returns by default)

V2 always returns `metrics.total_usage` (see `01_register_execute_4732.md`):

```json
"metrics": {
  "total_usage": {"inputTokens": 19, "outputTokens": 13, "totalTokens": 32}
}
```

PASS.

### Step 2 — V1 conversational agent without `include_token_usage`

Existing V1 agent: `EztGQp4BmZYVijh_5F_y` (type: CONVERSATIONAL, memory: conversation_index).

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/EztGQp4BmZYVijh_5F_y/_execute" \
   -H "Content-Type: application/json" \
   -d '{"parameters":{"question":"What is 1+1?"}}'
```

Response (truncated):

```
"name": "memory_id"
"name": "parent_interaction_id"
"name": "response"
```

No `token_usage` block — opt-in respected. PASS.

### Step 3 — V1 conversational agent with `include_token_usage: true`

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/EztGQp4BmZYVijh_5F_y/_execute" \
   -H "Content-Type: application/json" \
   -d '{
     "parameters":{"question":"What is 3+3?","include_token_usage":"true"}
   }'
```

Response (relevant portion):

```json
{
  "name": "token_usage",
  "dataAsMap": {
    "per_turn_usage": [{
      "model_name": "Bedrock Claude Haiku V1 chat",
      "model_url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/converse",
      "model_id": "EDtGQp4BmZYVijh_fF-L",
      "turn": 1,
      "input_tokens": 621,
      "output_tokens": 38,
      "total_tokens": 659,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0
    }],
    "per_model_usage": [{
      "model_name": "Bedrock Claude Haiku V1 chat",
      "model_url": "...",
      "model_id": "EDtGQp4BmZYVijh_fF-L",
      "call_count": 1,
      "input_tokens": 621,
      "output_tokens": 38,
      "total_tokens": 659,
      "cache_creation_input_tokens": 0,
      "cache_read_input_tokens": 0
    }]
  }
}
```

PASS — both `per_turn_usage` (1 turn here) and `per_model_usage` (rolled up by model with
`call_count`) are populated. Cache tokens (creation/read) are zero (Bedrock prompt caching
not configured here).

## Result

PASS — token tracking works:
- V1: opt-in via `include_token_usage: "true"`, returns per-turn + per-model rollups.
- V2: returned by default in `metrics.total_usage`.

## Notes / observations

- The V1 `token_usage` parameter must be passed as the **string** `"true"`, not the boolean
  `true`. The agent input parser converts everything to `Map<String,String>`. Worth
  documenting.
- V2's `metrics.total_usage` has different keys (`inputTokens`/`outputTokens`/`totalTokens` —
  camelCase from Bedrock) compared to V1 `per_turn_usage` which uses `input_tokens` /
  `output_tokens` / `total_tokens`. This inconsistency may complicate downstream consumers
  expecting a single schema. Worth flagging for a future cleanup PR.
- Did NOT exercise PER (Plan-Execute-Reflect) here — that is a separate test case once a
  PER agent is registered.
- AGUI streaming `CustomEvent` token emission was not exercised here — see streaming case.
