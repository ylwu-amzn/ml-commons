# Streaming 03 — V1 conversational agent streaming, happy path

- **Area:** V1 conversational agent + Bedrock Converse streaming
- **PR:** N/A (regression coverage; exercises BedrockStreamingHandler under Jackson 3.x #4795)
- **Status:** PASS
- **Date:** 2026-05-22
- **Cluster:** `localhost:9201` (with `transport-reactor-netty4`)

## Goal

End-to-end smoke for streaming via a V1 conversational agent. Verifies:

- Multi-chunk SSE delivery (`data: {...}\n\n` framing).
- Stable `memory_id` / `parent_interaction_id` across chunks (session preserved).
- `is_last` flag — exactly one chunk should carry `is_last: true`, indicating clean stream
  termination.
- Bedrock Converse streaming response parsed correctly under Jackson 3.x (#4795 changed
  `BedrockStreamingHandler` to use `tools.jackson`).

## Setup

Pre-deployed on this cluster:
- Agent `uwfJUJ4BAr57BVLX_dJV` (V1 `CONVERSATIONAL`, `memory: conversation_index`)
- LLM model `uAfJUJ4BAr57BVLX2dIf` (Bedrock Claude Sonnet 4.5 via Converse)

## Steps

```bash
oc -X POST "$OS_URL/_plugins/_ml/agents/uwfJUJ4BAr57BVLX_dJV/_execute/stream" \
   -H "Content-Type: application/json" -H "Accept: text/event-stream" \
   --no-buffer -m 30 \
   -d '{"parameters":{"question":"Count from 1 to 5 in words","stream":"true"}}'
```

Response (raw, all 5 SSE chunks):

```
data: {"inference_results":[{"output":[{"name":"memory_id","result":"GAcgUZ4BAr57BVLXs9Qn"},{"name":"parent_interaction_id","result":"GQcgUZ4BAr57BVLXs9Q3"},{"name":"response","dataAsMap":{"content":"One","is_last":false}}]}]}

data: {"inference_results":[{"output":[...,"dataAsMap":{"content":",","is_last":false}}]}]}

data: {"inference_results":[{"output":[...,"dataAsMap":{"content":" two, three, four, five.","is_last":false}}]}]}

data: {"inference_results":[{"output":[...,"dataAsMap":{"content":"","is_last":false}}]}]}

data: {"inference_results":[{"output":[...,"dataAsMap":{"content":"","is_last":true}}]}]}
```

## Verification

- `Total chunks: 5`
- `is_last:true chunks: 1`
- `memory_id` is identical across all chunks (`GAcgUZ4BAr57BVLXs9Qn`).
- Final response text reconstructed: `"One, two, three, four, five."` ✓
- Each chunk is itself parseable JSON (every `data: {…}` line is valid).

## Result

PASS — Bedrock Converse streaming is fully functional under Jackson 3.x. Stream framing
matches SSE convention (`data: <json>\n\n` blocks). Clean termination via single `is_last:true`
chunk.

## Notes / observations

- The HTTP/2 + TLS + security combination works here because of `transport-reactor-netty4`.
- `parameters.stream` should be set to `"true"` (string) — the input parser converts
  everything to `Map<String,String>`.
- Streaming chunks each include the entire envelope (`memory_id`, `parent_interaction_id`,
  one `response` block). This is more verbose than strictly necessary but lets clients
  treat each line as a self-contained event.
