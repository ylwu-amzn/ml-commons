# OpenSearch ml-commons 3.7 Release — All Findings

Aggregated findings from the 3.5 → 3.7 release test run. Sourced from the case files under
`cases/` and the `run-log.md` summary.

- **Branch:** `test`
- **Tip commit at finalize:** `74531a8eb`
- **Date:** 2026-05-22

## Test environment

Two clusters used for different scenarios:

| Cluster | Endpoint | HTTP transport | Streaming? | Used for |
|---------|----------|---------------|:----------:|----------|
| A — primary | `https://localhost:9200` | `SecurityHttpServerTransport` | ✗ | Cases 00–03, 05–09 |
| B — streaming | `https://localhost:9201` | `transport-reactor-netty4` (security still on) | ✓ | Case 04 (streaming + AGUI + V2-no-streaming) |

Both run OpenSearch 3.7.0 / ml-commons 3.7.0-SNAPSHOT, single-node, security-enabled.
Cluster B additionally has `arrow-flight-rpc` (gRPC streaming, not exercised) and requires
`plugins.ml_commons.unified_agent_api_enabled=true` and `plugins.ml_commons.ag_ui_enabled=true`
for V2/AGUI tests.

## Test results: 13 PASS · 1 REVIEW · 1 PARTIAL · 0 BLOCKED · 0 FAIL

| # | Case | PR(s) | Status |
|---|------|-------|:------:|
| 00.01 | Cluster reachable | — | PASS |
| 01.01 | `connection_timeout` / `read_timeout` defaults (seconds) | #4759 | PASS |
| 01.02 | Bedrock predict (Jackson 3.x) | #4795, #4784 | PASS |
| 02.01 | Memory container 404 error codes | #4723, #4701, #4725 | PASS |
| 02.02 | Semantic & hybrid search APIs (default + tuned weights, `k` boundaries) | #4658 | PASS |
| 03.01 | V2 chat agent register/execute (single + multi-turn) | #4732 | PASS |
| 03.02 | Token usage tracking (V1 opt-in, V2 default) | #4683 | PASS |
| 04.01 | Stream error propagation | #4792 | PASS (cluster B) |
| 04.02 | V2 chat agent does not support streaming | #4732 | PASS |
| 04.03 | V1 conversational streaming, happy path | (regression) | PASS |
| 05.01 | Tool name/description/output JSON escape | #4747, #4794 | PASS |
| 05.02 | QueryPlanningTool wildcards/aliases + custom fallback | #4726, #4729 | PASS |
| 06.01 | Jackson 3.x JSON paths (incl. STRICT_DUPLICATE_DETECTION, MCP) | #4795, #4784 | PASS |
| 07.01 | Pooling modes LAST_TOKEN/NONE | #4711, #4710 | PARTIAL |
| 08.01 | USER_PREFERENCE simplified prompt | #4798 | PASS |
| 09.01 | V2 chat agent interface review | #4732 | REVIEW (13 issues, 3 P0) |

---

## V2 Chat Agent interface review — bugs found

The happy path of V2 (TEXT / CONTENT_BLOCKS / MESSAGES / image multi-modal / tools /
multi-turn) works. The issues below are concrete, source-located, and reproducible.

### P0 — release blockers (3)

#### P0-1. `SourceType` error message lies about `BYTES`

**Where:** `common/src/main/java/org/opensearch/ml/common/input/execute/agent/AgentInput.java:605, 642, 679` (three identical instances in `createImageContent`, `createVideoContent`, `createDocumentContent`).

**What:** Error says `"Supported types: BYTES, URL"` but `SourceType` enum is:

```java
public enum SourceType { URL, BASE64 }
```

**Why "BYTES" leaked in:** Bedrock Converse's wire format uses `"bytes"` as the JSON
field name. `BedrockConverseModelProvider.mapSourceTypeToBedrock()` translates
`BASE64 → "bytes"`. The error author confused the wire field name with the enum name.

**Reproduction:**

```bash
curl -d '{"input":[{"type":"image","source":{"type":"BYTES","format":"png","data":"AAAA"}}],...}'
# → 400 "Invalid source type. Supported types: BYTES, URL"
# User retypes "BYTES" → same error → loop
```

**Why P0:** Multi-modal is a marquee feature of V2. The first thing a user does with
multi-modal is send an image. The error directs them to the exact value that fails. No
escape without reading source.

**Fix:** three-line change, zero risk:

```diff
- throw new IllegalArgumentException("Invalid source type. Supported types: BYTES, URL", e);
+ throw new IllegalArgumentException("Invalid source type. Supported types: BASE64, URL", e);
```

#### P0-2. `model_parameters` (max_tokens, temperature) silently ignored

**Where:** `common/src/main/java/org/opensearch/ml/common/agent/BedrockConverseModelProvider.java:61-63`.

**What:** The auto-generated connector stores `model_parameters` in connector
`parameters`:

```
"parameters" : {
  "max_tokens" : "10",
  "temperature" : "0.0",
  ...
}
```

…but `REQUEST_BODY_TEMPLATE` doesn't reference any of them:

```java
private static final String REQUEST_BODY_TEMPLATE = "{\"system\": [{\"text\": \"${parameters.system_prompt}\"}], "
    + "\"messages\": [${parameters._chat_history:-}${parameters.body}${parameters._interactions:-}]"
    + "${parameters.tool_configs:-} }";
```

No `inferenceConfig: {maxTokens, temperature, topP}` block — Bedrock falls back to its
model defaults (~4096 tokens for Claude Haiku).

**Reproduction:** registered V2 with `"max_tokens": "10"`, asked for a 1000-word essay,
received 1316 output tokens.

**Fix:** add an `inferenceConfig` block to the template, e.g.:

```json
"inferenceConfig": {
  "maxTokens": ${parameters.max_tokens:-4096},
  "temperature": ${parameters.temperature:-1.0}
}
```

#### P0-3. `AgentInputProcessor.validateInput()` is dead code in V2

**Where:** `common/src/main/java/org/opensearch/ml/common/input/execute/agent/AgentInputProcessor.java`.

**What:** The class implements detailed validation:
- Last message must be `user` or `tool`.
- Assistant messages must have content or tool calls.
- Tool messages must have `toolCallId` and content.
- Content blocks must be non-empty per type.
- Tool calls must have `id`, `type`, `function.name`, `function.arguments`.

`grep -rn AgentInputProcessor.validateInput src/main` → **0 hits**. Only
`extractQuestionText` is called (which calls `validateInput` internally, but only for
text-extraction, not for the V2 message-handling path).

**Reproduction:**

```bash
# Last message is assistant — should be rejected per validateMessages
curl -d '{"input":[
  {"role":"user","content":[{"type":"text","text":"hi"}]},
  {"role":"assistant","content":[{"type":"text","text":"hello"}]}
]}'
# → 200 OK with hallucinated "! how can i help you today?"
```

```bash
# First and only message is assistant
curl -d '{"input":[{"role":"assistant","content":[{"type":"text","text":"hi"}]}]}'
# → 200 OK with weird hallucinated response
```

**Fix:** call `AgentInputProcessor.validateInput(agentInput)` once in
`MLAgentExecutor` after parsing. One line change.

### P1 — release blockers if a customer demos V2 (4)

#### P1-1. Output schema asymmetric — input requires `"type"`, output omits it

**Where:** `common/src/main/java/org/opensearch/ml/common/output/execute/agent/AgentV2Output.java:113-124`.

Comment on line 113: `// Write content blocks (simplified format without "type" field)`.

But the input parser at `AgentInput.parseArrayItem():344` requires `"type"` to identify
a `ContentBlock`:

```java
if (itemMap.containsKey("type")) {
    return createContentBlock(itemMap);
}
throw new IllegalArgumentException("Invalid item format. Must have 'role' (for messages) or 'type' (for content blocks).");
```

A reasonable client (e.g., a chat UI) would echo `output.message` into the next
request's input — that's broken without the `type` field.

**Fix:** emit `"type":"text"` in the output writer (or accept ContentBlocks without
`type` when they have a single non-null content field).

#### P1-2. Output drops non-text content blocks

Same `toXContent` only writes `text`:

```java
for (var contentBlock : message.getContent()) {
    builder.startObject();
    if (contentBlock.getText() != null) {
        builder.field("text", contentBlock.getText());
    }
    builder.endObject();
}
```

If a model returns image/document/video content in `assistant.content`, it is silently
dropped. Today, Claude can return `tool_use` blocks — those are dropped too (see P1-3).
Future Bedrock vision-output models would also be dropped.

**Fix:** mirror `BedrockConverseModelProvider.buildContentArrayFromBlocks` writer logic.

#### P1-3. Output drops `toolCalls` and `toolCallId`

`Message` has `toolCalls` and `toolCallId` fields. `AgentV2Output.toXContent` doesn't
emit either. A client running an external tool loop can't see what the model wanted to
call. The V2 internal ReAct loop runs tools server-side, but client-side tool execution
needs `toolCalls` in the output.

**Fix:** emit both fields when present.

#### P1-4. `region` does NOT propagate from `model.credential`

**Where:** `BedrockConverseModelProvider.createConnector():87`.

```java
parameters.put("region", DEFAULT_REGION); // "us-east-1"
...
if (modelParameters != null) {
    parameters.putAll(modelParameters);  // region from model_parameters wins
}
```

`credential` is passed only to `.credential(...)`. The URL template
`bedrock-runtime.${parameters.region}.…` picks up `us-east-1` unless the user puts
`region` in `model_parameters`.

**Reproduction:** users typically write `"credential": {"access_key": "...", "region":
"us-west-2"}`. Agent registers; predict path uses us-east-1; fails or routes wrong.

**Fix:** promote `credential.region → parameters.region` if not already set, or document
explicitly that `region` belongs in `model.model_parameters`, not `model.credential`.

### P2 — polish (6)

#### P2-1. Cryptic Bedrock error when no-tools agent loads tool-bearing memory

**Where:** session/memory loading flow.

**Original concern (downgraded):** I initially thought passing Agent A's `memory_id`
into Agent B (different user, no tools) leaked A's tool history. Drilled in by creating
two security users (`alice_user`, `bob_user`) with shared backend role on the same
memory container, both executing against the same `session_id`:

| Querier | Hits | inputTokens for "show me everything" |
|---------|------|--------------------------------------|
| alice_user | 4 (own only) | retrieves SSN |
| bob_user | 8 (own only) | 331 — only Bob's |
| admin | 12 (all) | 1238 — sees Alice's SSN |

**Why isolation works:** `TransportSearchMemoriesAction.searchMemories()` lines 120-123:

```java
if (user != null && !ConnectorAccessControlHelper.isAdmin(user)) {
    memoryContainerHelper.addOwnerIdFilter(user, input.getSearchSourceBuilder());
}
```

For non-admin users, an `owner_id` filter is automatically appended. Memories are stamped
with `owner_id` at write time. Cross-user memory access is blocked even when `session_id`
and memory container are shared.

**Remaining UX issue:** when the same user previously used a tool-equipped agent and
then continues the same session via a no-tools agent, the session contains tool messages
and Bedrock rejects the request with `"toolConfig field must be defined when using
toolUse and toolResult content blocks"`. ml-commons should detect this and reject with
a clear message, not let Bedrock's cryptic error bubble up.

**Note on admin visibility:** `all_access` users (admin) can read all memories
regardless of `owner_id`. Consistent with OpenSearch security defaults but worth
documenting for compliance/audit.

#### P2-2. Misleading error for legacy `parameters.messages` envelope

```bash
curl -d '{"parameters":{"messages":[{"role":"user","content":[{"text":"hi"}]}]}}'
# → 500 "V2 agents require executor-provided memory. Use runV2() instead."
```

User mistake: V1 envelope `{"parameters":{"messages":...}}` instead of V2 simplified
`{"input":...}`. Error talks about an internal Java method name (`runV2()`).

**Fix:** detect this case in `MLAgentExecutor` and emit `"V2 agents use the simplified
input format. Send {\"input\": <text or array>}, not parameters.messages."`.

#### P2-3. Empty-string text input passes ml-commons, fails at Bedrock

```
curl -d '{"input":""}'
# → 400 from Bedrock: "The text field in the ContentBlock object at messages.0.content.0 is blank."
```

`AgentInputProcessor.validateTextInput` would catch with `"Text input cannot be null or
empty"` — but `validateInput` isn't called (P0-3).

#### P2-4. Bad/non-existent `memory_id` silently ignored

```
curl -d '{"input":"hi","memory_id":"nonexistent_mem_id"}'
# → 200 OK; new session created; old memory_id discarded
```

No warning, no error. User thinks they're continuing a session, actually starts new.

**Fix:** at minimum log a warning. Better: return 404.

#### P2-5. `system` role accepted by parser, rejected by Claude

```bash
curl -d '{"input":[
  {"role":"system","content":[{"type":"text","text":"You are helpful"}]},
  {"role":"user","content":[{"type":"text","text":"hi"}]}
]}'
# → 400 from Bedrock: "This model doesn't support system messages."
```

Claude Bedrock Converse takes system prompts via a separate top-level `system` field,
not the messages array. Real system prompt path is the agent's `system_prompt` parameter
(or `parameters.system_prompt` override). Confusing dual mechanism.

**Fix:** detect `role: system` in messages and either (a) hoist to `system_prompt`, or
(b) reject with "use system_prompt parameter, not message role".

#### P2-6. `model_parameters` values must be strings

```
"model_parameters": { "max_tokens": 1024 }   # int → 400
"model_parameters": { "max_tokens": "1024" } # string → registers (but ignored — see P0-2)
```

Driven by `Map<String,String>` typing in `MLAgentModelSpec`. Should either widen the
type or convert at parse time.

---

## Behavior changes worth release-noting (8)

These aren't bugs; they're intentional 3.5 → 3.7 changes that may break existing client
integrations.

1. **`connection_timeout` / `read_timeout` are seconds, not milliseconds** (#4759).
   Pre-3.7 defaults were `30000` interpreted as seconds → ~8.3h timeouts. Post-3.7
   default is `30`. Users with explicit values in the thousands need to divide by 1000.

2. **Memory container / context management DELETE & GET return 404 (was 500)** for
   missing resources (#4723, #4701, #4725). Clients that retry on 5xx should not retry
   on 404.

3. **Jackson 3.x `STRICT_DUPLICATE_DETECTION` is enforced.** `{"foo":1,"foo":2}` now
   yields `400 json_parse_exception` ("Duplicate Object property"). Pre-3.7 silently
   took the second value. Error envelope's nested `caused_by.type` is
   `stream_read_exception` (Jackson 3 rename of `JsonParseException`).

4. **V2 Chat Agent input format is `{"input": ...}`**, not
   `{"parameters":{"messages":...}}`. Wrong format produces the confusing "Use runV2()
   instead" error.

5. **V2 `model.model_id` is the provider's native model ID** (e.g.,
   `us.anthropic.claude-haiku-4-5-…`), NOT an ml-commons remote-model ID. Different
   semantics from V1's `llm.model_id`.

6. **V2 `model_parameters` values must be strings** (not numbers).

7. **V2 vs V1 token-usage schema differs.** V2 uses
   `metrics.total_usage.{inputTokens, outputTokens, totalTokens}` (camelCase from
   Bedrock); V1 uses
   `token_usage.per_turn_usage[].{input_tokens, output_tokens, total_tokens}`
   (snake_case). Worth aligning in a future PR.

8. **V2 chat agents do not support `_execute/stream`** in 3.7 (case 04.02). Returns 400
   with "Please use the non-streaming execute endpoint". Document explicitly.

---

## Known cluster limitations (not bugs in 3.7)

1. **HTTP streaming + security plugin** requires `transport-reactor-netty4`. Default
   `SecurityHttpServerTransport` (cluster A) doesn't support streaming.

2. **Mid-stream Bedrock errors don't reach `onErrorResume`.** When the Bedrock streaming
   client encounters bad credentials/401, the SSE channel opens (HTTP 200) but no chunks
   emit and no error chunk is delivered. Not a regression — #4792's fix only protects the
   `onErrorResume` path; if upstream silently hangs, the fix can't help. Worth a separate
   follow-up issue.

3. **AGUI streaming opens HTTP 200 but emits no chunks** even on the happy path. Setup
   may be incomplete, or there's a real defect. The `RunErrorEvent` branch of #4792's
   `buildStreamErrorChunk()` is exercised only by unit tests on this cluster. Worth a
   CI-side e2e test.

4. **No decoder-only / pre-pooled sentence-transformer model fixtures** on this host —
   affects 07.01 PARTIAL. Parser-level validation passed; full embedding-correctness
   validation needs model fixtures.

---

## Items deliberately not exercised

| Item | PR | Reason |
|------|----|--------|
| EncryptorImpl async multi-tenant scaling | #3919 | Needs load-test harness |
| SdkAsyncHttpClient resource leak fix | #4716 | Soak test, hours of traffic + JVM profiling |
| FIPS-by-default build/run | #4719 | Build-system check, not runtime |
| PER (Plan-Execute-Reflect) token tracking | #4683 | Would need a PER agent + tool fixtures |
| AGUI streaming + CustomEvent token emission | #4683 | See limitation #3 above |
| Yandex Cloud blueprint walkthrough | #4810, #4469 | Doc-only; needs live Yandex AI Studio account |
| MCP SSE / streamable HTTP transports e2e | #4795 | Needs external MCP server. In-process MCP tool register/list IS covered by 06.01 |
| `ConnectorAccessControlHelper` coverage | #4689 | Test-only PR, not a runtime feature |

---

## Recommended next steps

### Before 3.7 ships

1. **P0-1 fix** — 3-line error-message change in `AgentInput.java:605, 642, 679`.
2. **P0-2 fix** — wire `max_tokens`/`temperature` into
   `BedrockConverseModelProvider.REQUEST_BODY_TEMPLATE`.
3. **P0-3 fix** — call `AgentInputProcessor.validateInput()` once in `MLAgentExecutor`
   after parsing.
4. Add the 8 behavior changes above to release notes / migration guide.

### Investigation follow-ups (separate issues)

1. Why does mid-stream Bedrock-401 not reach `onErrorResume` in the streaming agent?
2. Why does AGUI streaming open HTTP 200 but emit no chunks?

### Eventual e2e coverage

1. Stage decoder-only and pre-pooled model fixtures, then complete 07.01 e2e.
2. 1-hour soak test of `_predict` against Bedrock to validate #4716 leak fix.

---

## Cross-reference index

| Topic | Case file |
|-------|-----------|
| Cluster baseline | [cases/00_smoke/01_cluster_reachable.md](cases/00_smoke/01_cluster_reachable.md) |
| Connector timeout defaults | [cases/01_connector/01_default_timeouts_4759.md](cases/01_connector/01_default_timeouts_4759.md) |
| Bedrock predict / Jackson 3 | [cases/01_connector/02_bedrock_predict_jackson3.md](cases/01_connector/02_bedrock_predict_jackson3.md) |
| Memory container error codes | [cases/02_memory_container/01_error_codes_4723_4701_4725.md](cases/02_memory_container/01_error_codes_4723_4701_4725.md) |
| Semantic & hybrid search | [cases/02_memory_container/02_semantic_hybrid_search_4658.md](cases/02_memory_container/02_semantic_hybrid_search_4658.md) |
| V2 chat agent register/execute | [cases/03_v2_chat_agent/01_register_execute_4732.md](cases/03_v2_chat_agent/01_register_execute_4732.md) |
| Token usage tracking | [cases/03_v2_chat_agent/02_token_usage_4683.md](cases/03_v2_chat_agent/02_token_usage_4683.md) |
| Stream error propagation | [cases/04_streaming/01_stream_error_propagation_4792.md](cases/04_streaming/01_stream_error_propagation_4792.md) |
| V2 streaming rejected | [cases/04_streaming/02_v2_chat_agent_no_streaming.md](cases/04_streaming/02_v2_chat_agent_no_streaming.md) |
| V1 streaming happy path | [cases/04_streaming/03_v1_chat_streaming_happy_path.md](cases/04_streaming/03_v1_chat_streaming_happy_path.md) |
| Tool name/output JSON escape | [cases/05_tools_and_query_planning/01_tool_json_escape_4747_4794.md](cases/05_tools_and_query_planning/01_tool_json_escape_4747_4794.md) |
| QueryPlanningTool wildcards | [cases/05_tools_and_query_planning/02_qpt_wildcards_aliases_4726_4729.md](cases/05_tools_and_query_planning/02_qpt_wildcards_aliases_4726_4729.md) |
| Jackson 3.x JSON paths | [cases/06_jackson_mcp/01_jackson3_paths_4795_4784.md](cases/06_jackson_mcp/01_jackson3_paths_4795_4784.md) |
| Pooling LAST_TOKEN/NONE | [cases/07_pooling_modes/01_pooling_modes_4710_4711.md](cases/07_pooling_modes/01_pooling_modes_4710_4711.md) |
| USER_PREFERENCE prompt | [cases/08_user_preference/01_user_preference_simplified_prompt_4798.md](cases/08_user_preference/01_user_preference_simplified_prompt_4798.md) |
| **V2 interface review (this is where the 13 issues live)** | [cases/09_v2_interface_review/01_v2_chat_agent_interface_review.md](cases/09_v2_interface_review/01_v2_chat_agent_interface_review.md) |
| Aggregated run-log | [run-log.md](run-log.md) |
| Per-PR change list (3.6 wave) | [changes-3.6.md](changes-3.6.md) |
| Per-PR change list (3.7 wave) | [changes-3.7.md](changes-3.7.md) |
| Original test plan | [test-plan.md](test-plan.md) |
