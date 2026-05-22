# Tools 01 — Tool name/description/output JSON escape — PR #4747, #4794

- **Area:** Agent prompt construction + flow agent tool output handling
- **PRs:**
  - [#4747](https://github.com/opensearch-project/ml-commons/pull/4747) — escape tool name and description
  - [#4794](https://github.com/opensearch-project/ml-commons/pull/4794) — fix tool output JSON escape issue
- **Status:** PASS
- **Date:** 2026-05-22

## Goal

Validate that:

1. (#4747) An agent registered with a tool whose **name** or **description** contains
   `"`/`\\` does not break prompt construction (no JSON parsing errors when tools are
   serialized into the LLM prompt).
2. (#4794) A flow agent tool whose **output** contains JSON-special characters
   (`"`, `\\`, `\n`, `\t`) produces a final response that is valid JSON.

## Setup

```bash
# Test index with JSON-special-character content
oc -X PUT "$OS_URL/t37-jsonescape-test"
oc -X POST "$OS_URL/t37-jsonescape-test/_doc/1?refresh=true" \
   -H "Content-Type: application/json" \
   -d '{"title":"escape \"test\"","body":"line1\nline2\\back\\slash\ttab quote\""}'
oc -X POST "$OS_URL/t37-jsonescape-test/_doc/2?refresh=true" \
   -H "Content-Type: application/json" \
   -d '{"title":"normal","body":"no special chars"}'
```

## Steps

### Step 1 — Flow agent with tool that returns special-char content (#4794)

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/_register" -H "Content-Type: application/json" -d '{
  "name": "t37_flow_jsonescape",
  "type": "flow",
  "tools": [{
    "type": "SearchIndexTool",
    "name": "search_with_special_chars",
    "include_output_in_agent_response": true,
    "parameters": {
      "input": "{\"index\": \"t37-jsonescape-test\", \"query\": {\"query\": {\"match_all\": {}}}}"
    }
  }]
}'
# → agent_id: dzYKUZ4Bn4mRprk_Ag-q
```

Execute:

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/dzYKUZ4Bn4mRprk_Ag-q/_execute" \
   -H "Content-Type: application/json" -d '{"parameters":{"question":"list all docs"}}'
```

Response (formatted):

```json
{
  "inference_results": [{
    "output": [{
      "name": "response",
      "result": "{\"_index\":\"t37-jsonescape-test\",\"_source\":{\"title\":\"escape \\\"test\\\"\",\"body\":\"line1\\nline2\\\\back\\\\slash\\ttab quote\\\"\"},\"_id\":\"1\",\"_score\":1.0}\n{...}\n"
    }]
  }]
}
```

Parsing assertion:

```python
import json
parsed = json.loads(raw_response)            # outer envelope parses
inner  = parsed["inference_results"][0]["output"][0]["result"]
# inner is NDJSON, each line is a separate doc — both lines parse via json.loads()
```

PASS — outer JSON parses, inner doc strings have all of `\"`, `\\`, `\n`, `\t` correctly
escaped.

### Step 2 — Chat agent with tool whose description has quotes/backslashes (#4747)

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/_register" -d '{
  "name": "t37_chat_tool_special_name",
  "type": "conversational",
  "llm": {"model_id": "...","parameters":{"prompt":"${parameters.question}",
          "system_prompt":"...","max_iteration":"3"}},
  "memory": {"type":"conversation_index"},
  "tools": [{
    "type": "ListIndexTool",
    "name": "tool_with_quotes_and_backslash_in_desc",
    "description": "This tool has \"quotes\" and a \\backslash\\ in its description. ...",
    "include_output_in_agent_response": false
  }]
}'
# → agent_id: ezYKUZ4Bn4mRprk_XA-i
```

Verification — GET on the agent shows the description stored as-is:

```
"description": "This tool has \"quotes\" and a \\backslash\\ in its description. ..."
```

Execute:

```bash
oc -X POST .../_execute -d '{"parameters":{"question":"What tools do you have available?"}}'
```

Response: agent runs, Bedrock returns a normal text response. No JSON parse error during
prompt construction. PASS.

If pre-#4747 (un-escaped `tool.getName()` / `tool.getDescription()`), the prompt body sent to
Bedrock would have contained an unescaped `"` and `\`, breaking the prompt JSON.

## Result

PASS — both PRs validated.

## Notes / observations

- The agent stores the description exactly as the user provided it (including the un-escaped
  representation). The escaping happens at **prompt-construction** time in `AgentUtils`
  (line 240-241 in current source: `StringEscapeUtils.escapeJson(tool.getName())` etc.).
  This is the right design — keep the data clean, escape at the boundary.
- I did not exercise the **tool-call → tool-output → next prompt** loop because the V1 chat
  agent didn't pick up the available tool with default settings — that is a tangential
  concern (likely needs `_llm_interface` or `tool_choice` config). The `MLFlowAgentRunner`
  test in Step 1 does cover the actual escape code path in `params.put(outputKey, escapeJson(...))`
  introduced by #4794 (line `MLFlowAgentRunner.java:122`).
- The same fix exists in `MLConversationalFlowAgentRunner` (per the diff in #4794) — not
  separately exercised here.
