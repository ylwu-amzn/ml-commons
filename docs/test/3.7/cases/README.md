# 3.7 Test Cases

One markdown file per test case. Folders group related cases (e.g., everything against a memory container).

## File naming

`<seq>_<short-slug>.md` — e.g., `01_create_bedrock_connector.md`.

## Per-case template

Every case file has this structure:

```
# <Title> — PR #<N>

- **Area:** <feature area>
- **PR:** #<N> (link)
- **Status:** PASS | FAIL | BLOCKED | PARTIAL
- **Date:** YYYY-MM-DD
- **Cluster:** OpenSearch x.y, ml-commons x.y

## Goal
What this case validates and why it matters.

## Setup
Preconditions and any fixtures needed.

## Steps
Numbered steps with the exact request and expected response.

## Result
Observed response and whether it matches expectations.

## Notes / Bugs / Follow-ups
Anything surprising; bug summary if applicable.
```

## Folders

- `00_smoke/` — cluster reachability, ml-commons settings, baseline.
- `01_connector/` — connector + remote model lifecycle, timeout defaults, predict.
- `02_memory_container/` — memory containers, error codes, search APIs.
- `03_v2_chat_agent/` — V2 chat agent register/execute.
- `04_streaming/` — AGUI streaming, error propagation.
- `05_tools_and_query_planning/` — QueryPlanningTool wildcards/fallback, tool JSON escape.
- `06_jackson_mcp/` — Jackson 3.x JSON paths and MCP transports.
- `07_pooling_modes/` — LAST_TOKEN / NONE pooling.
- `08_user_preference/` — Agentic memory USER_PREFERENCE extraction.

See `../run-log.md` for an aggregated pass/fail summary.
