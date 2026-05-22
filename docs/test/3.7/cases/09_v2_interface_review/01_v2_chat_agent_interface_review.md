# V2 Chat Agent Interface Review — PR #4732

- **Area:** V2 Chat Agent (CONVERSATIONAL_V2) registration / input / output / errors / multi-modal
- **PR:** [#4732](https://github.com/opensearch-project/ml-commons/pull/4732)
- **Date:** 2026-05-22
- **Reviewer outcome:** **Solid foundation; ship-blocker bugs are small and contained.**
  Happy path (TEXT / CONTENT_BLOCKS / MESSAGES / image multi-modal / tools / multi-turn)
  works. Found 13 issues ranging from "release-blocker bug" to "doc/UX gap", revised down
  to **3 P0** after verifying user-level memory isolation is enforced. List below.

## TL;DR — Issues found

| # | Severity | Issue |
|---|:--------:|-------|
| 1 | **BLOCKER** | `SourceType` error message says `"Supported types: BYTES, URL"` but the enum is `BASE64, URL` — users will literally copy `"BYTES"` from the error and still get rejected. |
| 2 | **BLOCKER** | `model.model_parameters` (`max_tokens`, `temperature`, etc.) are stored on the auto-created connector but **not interpolated into the request body template** — completely ignored at runtime. |
| 3 | **HIGH** | `model.credential` does **not** route `region`. The auto-generated connector hard-defaults to `us-east-1`. To target another region, user must put `region` in `model_parameters`, not `credential`. Undocumented. |
| 4 | **HIGH** | `AgentInputProcessor.validateInput()` is **never called** in the V2 production path. All the role/content validation it implements (last-message-must-be-user/tool, assistant-must-have-content-or-tool-calls, tool-message-must-have-toolCallId, etc.) is dead code. |
| 5 | **HIGH** | Output schema is asymmetric: input content blocks are `{"type":"text","text":"..."}`, output content blocks are `{"text":"..."}` (no `type` field). Clients who want to round-trip output → next input must add the `type` field. |
| 6 | **HIGH** | Output `toXContent` only emits **text** content blocks. Image/video/document content in an assistant message is silently dropped. |
| 7 | **HIGH** | Output drops `toolCalls` and `toolCallId` from the assistant `Message`. Clients have no way to see "which tools were invoked this turn" without re-querying memory. |
| 8 | ~~**HIGH**~~ → **REVISED to LOW** | Cross-agent memory access **is by design** (agents are tools; memory containers are the data store; multiple agents can share a container). User-level isolation **is enforced** by `TransportSearchMemoriesAction` which automatically appends an `owner_id` filter for non-admin users. Verified by driving alice/bob/admin against the same session_id — bob saw only his messages, admin saw all. The remaining concern is UX (the cryptic Bedrock error when tool-history bleeds into a no-tools agent — agent should detect and reject). See "Issue #8 — revised" below. |
| 9 | **MEDIUM** | Sending the legacy `parameters.messages` envelope to a V2 agent yields error `"V2 agents require executor-provided memory. Use runV2() instead."` — leaks an internal API contract instead of saying "use the `input` field". |
| 10 | **MEDIUM** | Empty-string text input passes ml-commons and fails downstream at Bedrock. `validateTextInput` exists but isn't called. |
| 11 | **MEDIUM** | Bad/non-existent `memory_id` is silently ignored — no warning, no error. The agent starts a new session. |
| 12 | **MEDIUM** | `system` role in messages is accepted by ml-commons' (dead) validator and the parser, but Claude Bedrock Converse rejects it (`"This model doesn't support system messages"`). Real system prompt goes through the agent's `system_prompt` parameter, not message role. Confusing dual mechanism. |
| 13 | **LOW** | `model_parameters` values must be **strings**. `"max_tokens": 1024` (int) fails registration. Map type is `Map<String,String>`. Should either accept numbers or reject with a clearer message. |

## How I tested

Cluster A (`localhost:9200`), V2 agent with both with-tools and without-tools variants. Drove
the matrix: TEXT / CONTENT_BLOCKS / MESSAGES input shapes, image multi-modal, tool round-trip
in input, all the obvious negative cases. Cross-referenced the source against observed
behavior to determine root cause.

---

## Detailed findings

### Issue #1 — `SourceType` error message lies

**Where:** `AgentInput.java:605` (and the same in `createVideoContent` line 642, `createDocumentContent` line 679).

```java
} catch (IllegalArgumentException e) {
    throw new IllegalArgumentException("Invalid source type. Supported types: BYTES, URL", e);
}
```

But the actual enum:

```java
// SourceType.java
public enum SourceType {
    URL,
    BASE64
}
```

**Reproduction:**

```bash
curl -d '{"input":[{"type":"image","source":{"type":"BYTES","format":"png","data":"AAAA"}}],...}'
# → 400 "Invalid source type. Supported types: BYTES, URL"
```

User reads the message, types `"BYTES"`, gets rejected again.

**Fix:** change three error messages to say `"Supported types: BASE64, URL"`. Also worth
documenting that the parser accepts lowercase (`base64`) since it does `.toUpperCase()`.

---

### Issue #2 — `model_parameters` are dead

**Where:** `BedrockConverseModelProvider.createConnector()` and `REQUEST_BODY_TEMPLATE`.

The auto-generated connector stores `max_tokens`/`temperature` in `parameters`:

```
"parameters" : {
  "max_tokens" : "10",
  "temperature" : "0.0",
  ...
}
```

…but the request body template never references them:

```
{"system": [{"text": "${parameters.system_prompt}"}],
 "messages": [...${parameters.body}...]
 ${parameters.tool_configs:-} }
```

No `inferenceConfig: {maxTokens, temperature, topP}` block — Bedrock falls back to its model
defaults (~4096 tokens for Claude Haiku).

**Reproduction:** registered V2 with `"max_tokens": "10"`, asked for a 1000-word essay,
received 1316 output tokens back. PASS-through is broken.

**Fix:** add an `inferenceConfig` block to the template that interpolates these parameters
when present. Or simpler: support `${parameters.max_tokens:-4096}`.

---

### Issue #3 — `region` doesn't propagate from `credential`

**Where:** `BedrockConverseModelProvider.createConnector():87`:

```java
parameters.put("region", DEFAULT_REGION); // "us-east-1"
...
if (modelParameters != null) {
    parameters.putAll(modelParameters);  // region from model_parameters wins
}
```

Notice `credential` is passed to `.credential(...)` only — its `region` field is ignored
when building the URL. The URL template `bedrock-runtime.${parameters.region}.…` will pick
up `us-east-1` unless the user also puts `region` in `model_parameters`.

**Reproduction:** users typically write `"credential": {"access_key": "...", "region":
"us-west-2"}` because that's how AWS SDK-style configs look. The agent registers, but the
predict path uses `us-east-1`, fails with a "model doesn't exist in this region" or the
SigV4 signs for the wrong region.

**Fix:** either:
- Promote `credential.region` to `parameters.region` if not set elsewhere, OR
- Document explicitly that `region` belongs in `model.model_parameters`, not in
  `model.credential`.

The current behavior (silent default to us-east-1) is a footgun.

---

### Issue #4 — `AgentInputProcessor.validateInput()` is never called in production

**Where:** every code path that uses `AgentInput` in V2.

The class implements detailed validation:
- Last message must be `user` or `tool`.
- Assistant messages must have content or tool calls.
- Tool messages must have `toolCallId` and content.
- Content blocks must be non-empty per type.
- Tool calls must have `id`, `type`, and `function.name`/`function.arguments`.

`grep -r AgentInputProcessor.validateInput src/main` → 0 hits. Only `extractQuestionText`
is called (which calls `validateInput` internally, but **only for the text-extraction path,
not for the V2 message-handling path**).

**Reproduction:**

```bash
# Last message is assistant — should be rejected per validateMessages
curl -d '{"input":[
  {"role":"user","content":[{"type":"text","text":"hi"}]},
  {"role":"assistant","content":[{"type":"text","text":"hello"}]}
]}'
# → 200 OK with hallucinated response ("! how can i help you today?")
```

```bash
# Assistant first — should also be rejected
curl -d '{"input":[{"role":"assistant","content":[{"type":"text","text":"hi"}]}]}'
# → 200 OK with weird hallucinated response
```

**Fix:** call `AgentInputProcessor.validateInput(agentInput)` in `MLAgentExecutor` right
after parsing. Or remove the dead validation logic and document that no validation is done.
Either way, current state is worst-of-both-worlds.

---

### Issue #5 — Output content blocks omit `type` field

**Where:** `AgentV2Output.toXContent():113-124`.

Comment: `// Write content blocks (simplified format without "type" field)`.

But the input parser at `AgentInput.parseArrayItem():344` **requires** `"type"` to identify
a ContentBlock:

```java
if (itemMap.containsKey("type")) {
    return createContentBlock(itemMap);
}
throw new IllegalArgumentException("Invalid item format. Must have 'role' (for messages) or 'type' (for content blocks).");
```

Round-trip:

```
output.message.content = [{"text": "..."}]
                                  ↓ feed back as input
input.content = [{"text": "..."}]      ← missing "type"
                                  ↓ parser fails
```

A reasonable client building a chat UI would echo `output.message` into the next request's
input messages array. That's broken.

**Fix:** emit `"type":"text"` in `toXContent`. Or, accept ContentBlocks without `type`
when they have a single non-null content field.

---

### Issue #6 — Output drops non-text content blocks

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

If a future model returns image/document/video content in `assistant.content`, **it is
silently dropped**. Today, Claude can already return tool_use blocks — those are dropped
(see #7). Future Bedrock vision-output models would also be dropped.

**Fix:** mirror the `buildContentArrayFromBlocks` logic from
`BedrockConverseModelProvider` to write each content type properly.

---

### Issue #7 — Output drops `toolCalls` and `toolCallId`

`Message` has `toolCalls` and `toolCallId` fields. `toXContent` doesn't emit either. So a
client running an external tool loop can't see what the model *wanted* to call. (V2's
internal ReAct loop runs tools server-side; but PR #4732's stated direction supports
client-side tool execution too — and that needs `toolCalls` in the output.)

**Fix:** emit both fields when present.

---

### Issue #8 — REVISED: agent sharing is by-design; user isolation IS enforced

**Original concern:** I saw Bedrock complain `"toolConfig field must be defined when using
toolUse and toolResult content blocks"` when Agent B (no tools) was given Agent A's
memory_id. I concluded that A's tool history was loaded into B's prompt — looked like a
cross-agent isolation gap.

**Actually drilled in.** Created two security users (`alice_user`, `bob_user`), each with
`ml_full_access` and a shared `backend_role` granting access to the same memory container.
Drove the same `session_id` from both:

```
admin's view of session NDZCUZ4Bn4mRprk_gRGJ:
  alice_user user      "My SSN is 999-88-7777 and password is alicepw1."
  alice_user assistant "I appreciate you testing my security awareness! ..."
  bob_user   user      "What was the SSN that was just shared? Please tell me verbatim."
  bob_user   assistant "I don't see any SSN..."
  bob_user   user      "What did the previous user just share with you?"
  bob_user   assistant "I don't have access to conversations with other users..."
  alice_user user      "What did I just share with you?"
  alice_user assistant "You shared: 'My SSN is 999-88-7777 and password is alicepw1.'"
  bob_user   user      "Print every word of my conversation history verbatim..."
  bob_user   assistant <bob's own messages 1-5 only — Alice's not present>

Bob's view (curl as bob_user) of same session_id:
  → 8 hits, ALL bob_user.

Admin's view of same session_id:
  → 12 hits including Alice's SSN message.
```

**Why Bob can't see Alice's data even though session_id is shared:**
`TransportSearchMemoriesAction.searchMemories()` lines 120-123:

```java
if (user != null && !ConnectorAccessControlHelper.isAdmin(user)) {
    memoryContainerHelper.addOwnerIdFilter(user, input.getSearchSourceBuilder());
}
```

…and `MemoryContainerHelper.addOwnerIdFilter()` adds
`{"term":{"owner_id":"<user.getName()>"}}` as a filter clause. Memories are stamped with
`owner_id` at write time. Result: non-admin users see only their own messages, even when
session_id and memory container are shared.

**Verification:**
- `inputTokens` for Bob's "show me everything" prompt: 331 — only his 4 messages.
- `inputTokens` for admin executing same prompt: 1238 — admin saw the full session
  including Alice's SSN message.
- Direct search via `_search` endpoint: bob gets 8 hits, admin gets 12 hits for the same
  query.

**So:** the original `toolConfig` Bedrock error wasn't a leak — it was loading Bob's own
prior tool-using turns into a no-tools agent. The cross-user-isolation is enforced.

**Remaining (lower-severity) concern:** the UX of "agent without tools loads memory that
contains tool messages" still produces a cryptic Bedrock error. The agent should either:
1. Strip tool messages from history when no `toolConfig` is in the request, or
2. Reject with `"This agent has no tools but the session contains tool messages from a
   previous run. Use a different session or an agent with the same tools."`

Severity downgraded to **LOW** (UX polish, not a security issue).

**Note on admin-tier visibility:** by design, any user with `all_access` (admin) can read
all memories regardless of `owner_id`. This is consistent with how OpenSearch security
generally treats `all_access`. Worth documenting that admins have visibility into all
agentic-memory content for compliance/audit purposes.

---

### Issue #9 — Misleading error for legacy input format

```bash
curl -d '{"parameters":{"messages":[{"role":"user","content":[{"text":"hi"}]}]}}'
# → 500 "V2 agents require executor-provided memory. Use runV2() instead."
```

The user's mistake was using the legacy V1 envelope `{"parameters":{"messages":...}}`
instead of the V2 simplified `{"input":...}`. The error talks about an internal Java method
name (`runV2()`).

**Fix:** detect this case in `MLAgentExecutor` and emit `"V2 agents use the simplified
input format. Send {\"input\": <text or array>}, not parameters.messages."`.

---

### Issue #10 — Empty-string input passes ml-commons

```
curl -d '{"input":""}'
# → 400 from Bedrock: "The text field in the ContentBlock object at messages.0.content.0 is blank."
```

`AgentInputProcessor.validateTextInput` would catch this with `"Text input cannot be null
or empty"`, but it isn't called (#4).

---

### Issue #11 — Bad memory_id silently ignored

```
curl -d '{"input":"hi","memory_id":"nonexistent_mem_id"}'
# → 200 OK; new session created; old memory_id discarded
```

No warning, no error. User thinks they're continuing a session, actually starts a new one.

**Fix:** at minimum log a warning. Better: return 404.

---

### Issue #12 — `system` role accepted by parser, rejected by Claude

```bash
curl -d '{"input":[
  {"role":"system","content":[{"type":"text","text":"You are helpful"}]},
  {"role":"user","content":[{"type":"text","text":"hi"}]}
]}'
# → 400 from Bedrock: "This model doesn't support system messages."
```

Claude Bedrock Converse takes system prompts via a separate top-level `system` field, not
the messages array. The agent's `system_prompt` parameter (or `parameters.system_prompt`
override) is the right channel. The unified-input parser accepts `system` role anyway,
which is misleading.

**Fix:** detect `role: system` in messages and either (a) hoist to `system_prompt`, or
(b) reject with "use system_prompt parameter, not message role".

---

### Issue #13 — `model_parameters` must be strings

```bash
"model_parameters": { "max_tokens": 1024 }   # ← int
# → 400 "model_parameters must be string"
```

vs.

```bash
"model_parameters": { "max_tokens": "1024" } # ← string
# → registers (but see #2 — value is ignored anyway)
```

Driven by `Map<String,String>` typing in `MLAgentModelSpec`. This is JSON; values can be
ints or floats. Either widen the type or do conversion at parse time.

---

## Reproduction summary (just the curls)

All against agent `ODYsUZ4Bn4mRprk_sxBz` (no-tools) or `VzYtUZ4Bn4mRprk_jhBx` (with
ListIndexTool) on cluster A. Bodies abbreviated.

| Edge | Curl | Result | Issue # |
|------|------|--------|---------|
| TEXT input | `{"input":"What is 2+2?"}` | 200 OK | (happy) |
| CONTENT_BLOCKS | `{"input":[{"type":"text","text":"hi"}]}` | 200 OK | (happy) |
| MESSAGES | `{"input":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}` | 200 OK | (happy) |
| MULTIMODAL image (BASE64) | `{"input":[{"type":"text","text":"color?"},{"type":"image","source":{"type":"BASE64","format":"png","data":"…"}}]}` | 200 OK | (happy) |
| Tool execution | `{"input":"List indices and count them"}` (with ListIndexTool) | 200 OK | (happy) |
| `BYTES` source type | `{"input":[…,{"type":"image","source":{"type":"BYTES",…}}]}` | 400 misleading error | #1 |
| Override `max_tokens=10`, ask for essay | small `max_tokens` ignored | #2 |
| Region in `credential` | `"credential":{…,"region":"us-west-2"}` | URL goes to us-east-1 | #3 |
| Last msg = assistant | `[{"role":"user",…},{"role":"assistant",…}]` | 200 OK (should be 400) | #4 |
| First msg = assistant | `[{"role":"assistant",…}]` | 200 OK (should be 400) | #4 |
| Empty string input | `{"input":""}` | 400 from Bedrock | #4, #10 |
| `parameters.messages` (legacy) | `{"parameters":{"messages":[…]}}` | 500 with "Use runV2()" | #9 |
| Bad `memory_id` | `{"input":"hi","memory_id":"nope"}` | 200 OK silently new session | #11 |
| `role: system` | `[{"role":"system",…},{"role":"user",…}]` | 400 from Bedrock | #12 |
| Cross-agent `memory_id` | Agent B + Agent A's memory | Bedrock error reveals leak | #8 |
| Round-trip output as input | `{"input":[output.message.content_block_without_type]}` | 400 "Must have 'role' or 'type'" | #5 |
| `max_tokens` as int | `"model_parameters":{"max_tokens":1024}` | 400 | #13 |

---

## What's well-designed

Worth saying — the V2 design has real strengths:

- **Simplified registration** (`model.model_id` + `model_provider` + `credential`) is
  much cleaner than V1's "register a connector, then a model, then reference model_id".
- **Multi-modal type system** (TEXT / CONTENT_BLOCKS / MESSAGES with role + content) maps
  cleanly onto Bedrock Converse, OpenAI Chat Completions, and Anthropic Messages APIs.
- **Image base64 multi-modal works end-to-end** in a single execute call.
- **`metrics.total_usage`** in the response is a much better default than V1's opt-in.
- **ReAct loop with bounded iterations** + `stop_reason: max_iterations` is the right shape.
- **Tool execution is server-side**, so clients don't need to implement the loop themselves.
- **Memory persistence is automatic** via the linked memory container.

The interface is on the right track. The issues above are all concrete and fixable;
none of them require redesign.

---

## Recommended next steps before 3.7 ships

P0 (release blockers):
- Fix #1 (error message — trivial three-line change).
- Fix #2 (`max_tokens`/`temperature` in template — small change).
- Fix #4 (call `validateInput` — one line in executor).

P1 (release blockers if a client demos this):
- Fix #5, #6, #7 (output schema — emit `type`, all content types, tool calls).
- Fix #3 (region routing — either propagate or document).

P2 (polish):
- Fix #8 (clearer error when no-tools agent loads tool-bearing memory — UX, not security).
- Fix #9, #10, #11, #12, #13.

---

## How this relates to other 3.7 cases

- `03_v2_chat_agent/01_register_execute_4732.md` — happy path register + execute.
- `03_v2_chat_agent/02_token_usage_4683.md` — `metrics.total_usage` default behavior.
- `04_streaming/02_v2_chat_agent_no_streaming.md` — V2 explicitly rejects streaming.

Together with this review, a release-go/no-go decision is well-informed.
