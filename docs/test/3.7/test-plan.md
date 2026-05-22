# 3.7 Test Plan

Concrete test items for ml-commons 3.7. Items are grouped by feature area and ordered by
priority (P0 = blocker for release; P1 = must-test; P2 = nice-to-have / regression coverage).

Legend: `[wave]` is `3.6` (shipped in 3.6 release) or `3.7` (new on main).

---

## P0 — Critical path

### 1. Jackson 3.x migration `[3.7]` (#4795, #4784)
**Why critical:** dependency upgrade with broad blast radius across JSON parsing,
streaming, MCP, Nova preprocessing, and ingest processors.

Test items:
- [ ] Register / predict on every embedding model type (text-embedding, sparse, multi-modal Nova).
- [ ] Register / execute every connector type (Bedrock, OpenAI, Cohere, Anthropic, SageMaker, Yandex).
- [ ] All four streaming paths: Bedrock Converse stream, Bedrock InvokeModelWithResponseStream,
      OpenAI chat completions stream, AGUI streaming agent. Confirm chunked JSON parses correctly.
- [ ] MCP server: SSE transport, streamable HTTP transport, stateless transport — list tools,
      call tools, error path with malformed JSON.
- [ ] `MLExtractJsonProcessor` and `MLInferenceIngestProcessor`: malformed input, deeply nested
      JSON, unicode strings.
- [ ] Index Insight `LogRelatedIndexCheckTask` with realistic mappings.
- [ ] Confirm error types thrown on bad JSON match expectations (#4784 changed exception types).

### 2. V2 Chat Agent `[3.6]` (#4732)
- [ ] Register V2 chat agent with simplified API (no model_id autofill).
- [ ] Single-turn execute.
- [ ] Multi-turn execute with conversational memory.
- [ ] Multi-modal input (image + text).
- [ ] Tool use end-to-end; tool output stored in memory.
- [ ] Streaming response through AGUI; CustomEvent emission for token usage.
- [ ] MCP-connector tools respected on V2 agent (#4739).
- [ ] AGUI stream error paths — upstream LLM 5xx surfaces as AGUI error event with cause (#4792).

### 3. Agentic memory — semantic & hybrid search `[3.6]` (#4658)
- [ ] `POST /_plugins/_ml/memory_containers/{id}/_semantic_search` returns top-k matches.
- [ ] `POST /_plugins/_ml/memory_containers/{id}/_hybrid_search`:
  - default weights (`bm25_weight=0.5`, `neural_weight=0.5`).
  - tuned weights (e.g., 0.2/0.8 and 0.8/0.2).
  - `k` boundary: `k=1`, `k=10000` valid; `k=0` and `k=10001` rejected.
- [ ] Hybrid search **without** neural-search plugin → clear error message.
- [ ] Filter on `tenant_id` works in both endpoints.
- [ ] No persistent pipeline created in cluster state by hybrid search.
- [ ] Concurrent hybrid searches → no `UnsupportedOperationException` on processor config map (Map.of vs HashMap fix).

### 4. EncryptorImpl async / multi-tenant `[3.6]` (#3919)
- [ ] Concurrent registration of remote models for **different tenants**: each gets a unique master key, no duplicates.
- [ ] Concurrent requests for the **same tenant** wait on a single key generation (no thrash).
- [ ] No thread starvation or deadlock under load (e.g., 50 parallel `register_model` calls).
- [ ] Master key cache expiry still works (`#4543` from 3.5 + this change).

### 5. Encrypt/decrypt with FIPS-by-default `[3.6]` (#4719)
- [ ] Plugin starts in FIPS mode (default in 3.7).
- [ ] All crypto paths (master key gen, model artifact decryption) work under bouncy-castle FIPS.
- [ ] Non-FIPS opt-out still functional.

---

## P1 — High priority

### 6. Streaming error propagation `[3.7]` (#4792)
- [ ] Bedrock stream → simulate 429 / 5xx / malformed chunk → AGUI error event carries unwrapped cause, not generic wrapper.
- [ ] Same for OpenAI chat completions stream.
- [ ] `Propagate stream error` regression: error during early stream (before first chunk) is surfaced.

### 7. Tool output JSON escaping `[3.7]` (#4794, [3.6] #4747)
- [ ] Tool whose name contains `"` and `\\` registers and is referenced cleanly in agent prompts.
- [ ] Flow-agent tool output containing `"`, `\\`, newlines produces valid JSON in final response.
- [ ] Search response with `model_content` no longer fails with “same entry … included and excluded”.

### 8. USER_PREFERENCE extraction prompt `[3.7]` (#4798)
- [ ] Run agentic memory with smaller LLM (Claude Haiku 3 or equivalent) → USER_PREFERENCE facts no longer silently discarded.
- [ ] Extracted facts are plain sentences (no `Context: …. Categories: …` prefix).
- [ ] Run with Claude Sonnet for regression — overall recall remains within ±2% of 3.6 numbers.

### 9. ConnectorClientConfig timeout defaults `[3.6/3.7]` (#4759)
- [ ] **Migration risk:** any user explicitly setting `connection_timeout: 30000` or
      `read_timeout: 30000` previously got 8h timeouts. After 3.7, those values become 30000s ≈ 8h still.
      Document this as a behavior change, communicate to users that the unit is **seconds** and they
      should reduce their values by 1000x.
- [ ] New cluster default: 30s (was previously misinterpreted but happened to work).
- [ ] Log message displays unit "seconds".

### 10. Token usage tracking `[3.6]` (#4683)
- [ ] `include_token_usage=true` returns per-model and per-turn usage.
- [ ] AGUI streaming emits `CustomEvent` with token usage payload.
- [ ] PER agent suppresses sub-agent token logging (no double-logging).
- [ ] AGUI legacy agent + conversational memory: no NPE.
- [ ] Token usage works for Bedrock, OpenAI, Anthropic, Cohere connectors.

### 11. SdkAsyncHttpClient resource leak fix `[3.6]` (#4716)
- [ ] Soak test: thousands of `_predict` calls across a mix of remote connectors. Confirm:
  - Netty event-loop count stable.
  - No connection-pool-acquire timeouts.
  - JVM thread count steady-state.
- [ ] `RemoteModel.close()` cleans up executor before nulling reference.
- [ ] `ExecuteConnectorTransportAction`, `GetTaskTransportAction`, `CancelBatchJobTransportAction`,
      `RemoteAgenticConversationMemory` all clean up short-lived executors.

### 12. New pooling modes — LAST_TOKEN & NONE `[3.6]` (#4711, #4710)
- [ ] Register a decoder-only embedding model (e.g., Qwen3 / GPT-style) with `pooling_mode=lasttoken`.
  - ONNX path.
  - TorchScript / HuggingFace path.
- [ ] Register a sentence-transformer model that ships pre-pooled output with `pooling_mode=none`.
  - ONNX uses second output (`sentence_embedding`).
  - HuggingFace path uses fallback chain.
- [ ] Sanity-check embedding dimensions and search relevance.

### 13. QueryPlanningTool — wildcards/aliases + custom fallback `[3.6]` (#4726, #4729)
- [ ] `index_name=logs_*` works (no NPE).
- [ ] `index_name=<alias_to_2_indices>` works; warning logged when multi-index resolves.
- [ ] Custom fallback query path triggered when LLM cannot generate plan.

### 14. AGUI agent — messages array + chat history `[3.6]` (#4645, #4720)
- [ ] AGUI agent receives MessagesSnapshot event.
- [ ] Messages array works for all memory types (conversational, agentic, remote agentic).
- [ ] Legacy interface agent restores AGUI context.

### 15. Memory container / context management error codes `[3.6]` (#4723, #4701, #4725)
- [ ] DELETE non-existent memory container → 404 (not 500).
- [ ] DELETE non-existent context-management template → 404.
- [ ] Client-side errors during `add_memory`, `create_session`, `predict` surface as 4xx (not 500).

### 16. AgentTool infinite-loop fix `[3.6/3.7]` (#4762, #4733)
- [ ] AgentTool nesting: agent A calls AgentTool → agent B → agent B's logger no longer
      mutates `agent_id`, no infinite loop.
- [ ] AgentTool with parameters supplied as immutable map: no `UnsupportedOperationException`.

---

## P2 — Regression / supporting coverage

### 17. Yandex Cloud blueprints `[3.7]` (#4810, #4469)
- [ ] Walk through both blueprints with a real Yandex AI Studio account; embedding endpoint
      returns expected vectors.

### 18. Misc 3.6 fixes (regression sweep)
- [ ] `Tags.addTag()` — metric tags actually appear in stats output (#4712).
- [ ] Numeric type preservation in ML inference query template substitution (#4656).
- [ ] Stats collector job: completes for models with built-in connector (#4560).
- [ ] `ValidatingObjectInputStream.resolveClass()` plugin classloader fallback works for ModelSerDeSer (#4692).
- [ ] Inline-create context-management overwrite during agent register (#4637).
- [ ] Post-memory hook with structured message fires after memory write (#4687).

### 19. Build / CI infra
- [ ] Clean build under Gradle 9.4.1 (#4811).
- [ ] FIPS-mode build & integ tests pass (#4719, #4654, #4659).
- [ ] Jacoco 0.8.14 coverage reports generate.
- [ ] Optimized IT setup runtime (#4667) — verify ~50% improvement holds.

### 20. Security / access plugin
- [ ] Resource access levels yaml is recognized after rename (#4737).

---

## Areas to call out in the release notes / migration guide

1. **`connection_timeout` / `read_timeout` are seconds, not millis.** Anyone with explicit
   values in the thousands needs to divide by 1000 (#4759).
2. **FIPS is the default build/run mode.** Documented but worth a release-notes note (#4719).
3. **Jackson 3.x.** No user-facing API change, but external code wrapping ml-commons exception
   types may need adjustment (#4784).
4. **Memory container / context management error codes** changed from 500 to 4xx (#4723, #4701,
   #4725) — clients that key off 5xx for retry need review.
5. **HTTP method on memory container delete:** error path returns 404 instead of 500 — clients
   should not retry on 404.

---

## Suggested tracking

For each P0 / P1 item above, create a Jira/GitHub issue keyed to the PR number and assign an
owner. Use this file as the master checklist; tick items as `[x]` as they are validated.
