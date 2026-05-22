# V2 Chat Agent 01 — Register and execute — PR #4732

- **Area:** V2 Chat Agent (CONVERSATIONAL_V2)
- **PR:** [#4732](https://github.com/opensearch-project/ml-commons/pull/4732)
- **Status:** PASS
- **Date:** 2026-05-22

## Goal

Validate the V2 Chat Agent's simplified registration and execution:

- New agent type `conversational_v2`.
- Simplified registration via top-level `model` block (no pre-registered remote model
  required — credentials and provider live with the agent).
- Single-turn and multi-turn execution.
- Memory persisted to a memory container (agentic_memory).
- V2-specific validation: requires `agentic_memory` or `remote_agentic_memory` (not
  `conversation_index`).

## Setup

Reuses the memory container from `02_memory_container/02_semantic_hybrid_search_4658.md`:
- Memory container ID: `LDYDUZ4Bn4mRprk_Sw8s`
- Embedding model: Bedrock Titan Embed v2 (1024-dim)
- LLM (for fact extraction): Claude Haiku 4.5

## Steps

### Step 1 — Register V2 chat agent

Note: V2 takes `model` (not `llm`). `model.model_id` is the **provider's model ID**, not an
ml-commons model ID.

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/_register" -H "Content-Type: application/json" -d '{
  "name": "t37_v2_chat_agent",
  "type": "conversational_v2",
  "description": "PR #4732 V2 Chat Agent test",
  "model": {
    "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "model_provider": "bedrock/converse",
    "credential": {
      "access_key": "...", "secret_key": "...", "session_token": "...", "region": "us-west-2"
    },
    "model_parameters": { "max_tokens": "1024", "temperature": "0.0" }
  },
  "memory": {
    "type": "agentic_memory",
    "memory_container_id": "LDYDUZ4Bn4mRprk_Sw8s"
  }
}'
# → {"agent_id":"WzYHUZ4Bn4mRprk_7w_W"}
```

PASS — agent registered.

### Step 2 — Single-turn execute

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/WzYHUZ4Bn4mRprk_7w_W/_execute" \
   -H "Content-Type: application/json" -d '{
     "input": "What is 2+2?",
     "parameters": { "namespace": {"user_id": "alice"} }
   }'
```

Response:

```json
{
  "stop_reason": "end_turn",
  "message": {
    "content": [{"text": "2 + 2 = 4"}],
    "role": "assistant"
  },
  "memory_id": "XDYHUZ4Bn4mRprk__g9x",
  "metrics": {
    "total_usage": {"inputTokens": 19, "outputTokens": 13, "totalTokens": 32}
  }
}
```

PASS. Note `metrics.total_usage` is returned by default — exercises PR #4683 token tracking.

### Step 3 — Multi-turn execute

Pass `memory_id` from the previous response:

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/WzYHUZ4Bn4mRprk_7w_W/_execute" \
   -H "Content-Type: application/json" -d '{
     "input": "What was my previous question?",
     "parameters": {
       "namespace": {"user_id": "alice"},
       "memory_id": "XDYHUZ4Bn4mRprk__g9x"
     }
   }'
```

Response:

```json
{
  "stop_reason": "end_turn",
  "message": {"content":[{"text": "Your previous question was \"What is 2+2?\""}], "role":"assistant"},
  "memory_id": "XDYHUZ4Bn4mRprk__g9x",
  "metrics": {"total_usage": {"inputTokens": 41, "outputTokens": 15, "totalTokens": 56}}
}
```

PASS. The agent correctly recalled the prior turn — session memory works. `inputTokens`
grew 19 → 41 (+22), reflecting the prior turn injected as context.

### Step 4 — Validation: bad memory type

```bash
"memory": {"type": "conversation_index"}
```

Response:

```
HTTP 400
"V2 agents (CONVERSATIONAL_V2) are not compatible with conversation_index memory.
 Found memory type: conversation_index. Please use 'agentic_memory' or
 'remote_agentic_memory' instead."
```

PASS.

### Step 5 — Validation: missing memory

```bash
# No "memory" block at all
```

Response:

```
HTTP 400
"V2 agents (CONVERSATIONAL_V2) require memory configuration. Please configure 'memory'
 with type 'agentic_memory' or 'remote_agentic_memory'."
```

PASS.

## Result

PASS — all 5 sub-tests succeed.

## Notes / observations

- The legacy `parameters.messages` envelope is **not** the V2 input format. V2 uses
  `{"input": "text"}` or `{"input": [{"role":"user","content":[{"type":"text","text":"..."}]}]}`.
  Sending `parameters.messages` to a V2 agent yields:
  > `"V2 agents require executor-provided memory. Use runV2() instead."`
  This is a confusing error — at the protocol level, the user's actual mistake was using the
  wrong input field name. Worth a doc/UX improvement.
- The `model_id` field overlaps in name with the ml-commons remote-model `model_id`. For V2,
  it is the **provider's** native model ID (e.g., `us.anthropic.claude-haiku-4-5-…`). This is
  a meaningful behavior change vs V1 and should be highlighted in the V2 docs.
- `model_parameters` values must be **strings** (the parser uses `getParameterMap` which is
  `Map<String,String>`). `"max_tokens": 1024` (int) would fail; must be `"max_tokens": "1024"`.
  Worth documenting.
- After Step 3, the memory container's long-term index still contains only the original 3
  facts — neither "What is 2+2?" nor "Your previous question was …" was extracted as a
  long-term fact. The SEMANTIC strategy reasonably filters trivial Q&A out. Not a bug.
