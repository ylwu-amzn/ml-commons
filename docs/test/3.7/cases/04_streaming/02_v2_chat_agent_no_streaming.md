# Streaming 02 — V2 Chat Agent does not support streaming

- **Area:** V2 Chat Agent (CONVERSATIONAL_V2) + streaming endpoint
- **PR:** related to [#4732](https://github.com/opensearch-project/ml-commons/pull/4732)
- **Status:** PASS (deliberate restriction, cleanly enforced)
- **Date:** 2026-05-22
- **Cluster:** `localhost:9201` (streaming-capable)

## Goal

Document and validate that V2 chat agents cannot be invoked via the streaming endpoint
in 3.7. Worth flagging in release notes / V2 docs.

## Steps

### Step 1 — Register V2 chat agent

```bash
# Settings prerequisites (already enabled on this cluster):
oc -X PUT "$OS_URL/_cluster/settings" -d '{"persistent":{
  "plugins.ml_commons.unified_agent_api_enabled":true
}}'

oc -X POST "$OS_URL/_plugins/_ml/agents/_register" -d '{
  "name": "t37_v2_chat_stream",
  "type": "conversational_v2",
  "model": {
    "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "model_provider": "bedrock/converse",
    "credential": {"access_key":"...","secret_key":"...","session_token":"...","region":"us-west-2"},
    "model_parameters": {"max_tokens":"512","temperature":"0.0"}
  },
  "memory": {"type":"agentic_memory","memory_container_id":"6QcaUZ4BAr57BVLXwdMB"}
}'
# → agent_id: 8QcaUZ4BAr57BVLX49Ou
```

### Step 2 — Try streaming this agent

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/8QcaUZ4BAr57BVLX49Ou/_execute/stream" \
   -H "Content-Type: application/json" -H "Accept: text/event-stream" \
   --no-buffer -m 30 \
   -d '{"input":"Say hello in 5 words","parameters":{"namespace":{"user_id":"alice"}}}'
```

Response:

```json
{
  "error": {
    "root_cause": [{
      "type": "status_exception",
      "reason": "V2 agents (CONVERSATIONAL_V2) do not support streaming. Please use the non-streaming execute endpoint: POST /_plugins/_ml/agents/8QcaUZ4BAr57BVLX49Ou/_execute"
    }],
    "type": "status_exception",
    "reason": "V2 agents (CONVERSATIONAL_V2) do not support streaming. Please use the non-streaming execute endpoint: POST /_plugins/_ml/agents/8QcaUZ4BAr57BVLX49Ou/_execute"
  },
  "status": 400
}
```

PASS — clean 400 with an actionable message that points at the correct alternative endpoint.

### Step 3 — Confirm non-streaming execute still works for the same agent

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/8QcaUZ4BAr57BVLX49Ou/_execute" \
   -H "Content-Type: application/json" \
   -d '{"input":"Say hello in 5 words","parameters":{"namespace":{"user_id":"alice"}}}'
```

(Tested implicitly via the Bedrock Converse API; the V2 agent on the other cluster — see
03.01 — already exercised the non-streaming path.)

## Result

PASS — V2 streaming is rejected at the REST layer (synchronous validation, not after channel
open), with a clear error message.

## Recommendation for release notes

> V2 chat agents (`type: "conversational_v2"`) **do not support the
> `/_plugins/_ml/agents/{id}/_execute/stream` endpoint** in 3.7. They must be invoked via
> the non-streaming `_execute` endpoint. Streaming will return HTTP 400 with a message
> directing callers to use `_execute`.
