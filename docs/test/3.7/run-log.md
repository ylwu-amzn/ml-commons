# 3.7 Run Log — Aggregated Results

Snapshot of test execution against a local OpenSearch 3.7.0 cluster running ml-commons
3.7.0-SNAPSHOT. One row per test case file under `cases/`.

## Clusters under test

Two clusters on the same host, used for different scenarios:

| Cluster | Endpoint | HTTP transport | Streaming? | Used for |
|---------|----------|---------------|:----------:|----------|
| A — primary | `https://localhost:9200` | `SecurityHttpServerTransport` | ✗ | Cases 00–08 except 04 |
| B — streaming | `https://localhost:9201` | `transport-reactor-netty4` (security plugin still on) | ✓ | Case 04 (streaming + AGUI + V2 streaming behavior) |

Both run OpenSearch 3.7.0 / ml-commons 3.7.0-SNAPSHOT, single-node, admin / RUIw….
Cluster B additionally has `arrow-flight-rpc` (gRPC streaming, not exercised) and
requires `plugins.ml_commons.unified_agent_api_enabled=true` and
`plugins.ml_commons.ag_ui_enabled=true` for V2/AGUI tests.

## Results

| # | Case | PR(s) | Status | Notes |
|---|------|-------|:------:|-------|
| 00.01 | [Cluster reachable](./cases/00_smoke/01_cluster_reachable.md) | — | PASS | Baseline |
| 01.01 | [Default timeouts (seconds)](./cases/01_connector/01_default_timeouts_4759.md) | #4759 | PASS | Source default = 30; explicit `client_config` round-trips; predict completes in ~862ms |
| 01.02 | [Bedrock predict (Jackson 3 path)](./cases/01_connector/02_bedrock_predict_jackson3.md) | #4795, #4784 | PASS | Converse JSON deserialize end-to-end |
| 02.01 | [Memory container 404 error codes](./cases/02_memory_container/01_error_codes_4723_4701_4725.md) | #4723, #4701, #4725 | PASS | DELETE/GET on missing → 404 |
| 02.02 | [Semantic & hybrid search APIs](./cases/02_memory_container/02_semantic_hybrid_search_4658.md) | #4658 | PASS | Default + tuned weights; k boundary [1,10000]; query validation |
| 03.01 | [V2 chat agent register/execute](./cases/03_v2_chat_agent/01_register_execute_4732.md) | #4732 | PASS | Single + multi-turn; bad memory rejected |
| 03.02 | [Token usage tracking](./cases/03_v2_chat_agent/02_token_usage_4683.md) | #4683 | PASS | V1 opt-in via `include_token_usage`; V2 default; per-turn + per-model |
| 04.01 | [Stream error propagation](./cases/04_streaming/01_stream_error_propagation_4792.md) | #4792 | PASS | Re-run on cluster B; malformed JSON + missing param → unwrapped error in SSE chunk; AG-UI / mid-stream Bedrock-401 paths still unit-test only |
| 04.02 | [V2 chat agent does not support streaming](./cases/04_streaming/02_v2_chat_agent_no_streaming.md) | #4732 (related) | PASS | Deliberate restriction — clean 400 with actionable message |
| 04.03 | [V1 conversational streaming, happy path](./cases/04_streaming/03_v1_chat_streaming_happy_path.md) | (regression) | PASS | 5 SSE chunks, single `is_last:true`, Bedrock Converse stream parsed under Jackson 3.x |
| 05.01 | [Tool name/output JSON escape](./cases/05_tools_and_query_planning/01_tool_json_escape_4747_4794.md) | #4747, #4794 | PASS | Tool desc with `"`/`\`; flow agent output with `"`/`\\`/`\n`/`\t` valid JSON |
| 05.02 | [QueryPlanningTool wildcards/aliases + fallback](./cases/05_tools_and_query_planning/02_qpt_wildcards_aliases_4726_4729.md) | #4726, #4729 | PASS | Wildcard `t37_logs_2024_*` + alias resolve; `fallback_query` registers |
| 06.01 | [Jackson 3.x JSON paths](./cases/06_jackson_mcp/01_jackson3_paths_4795_4784.md) | #4795, #4784 | PASS | `tools.jackson` in effect; STRICT_DUPLICATE_DETECTION enforced; MCP tool register/list |
| 07.01 | [Pooling modes LAST_TOKEN/NONE](./cases/07_pooling_modes/01_pooling_modes_4710_4711.md) | #4711, #4710 | PARTIAL | Parser accepts both; e2e blocked by lack of decoder-only / pre-pooled model fixtures |
| 08.01 | [USER_PREFERENCE simplified prompt](./cases/08_user_preference/01_user_preference_simplified_prompt_4798.md) | #4798 | PASS | All extracted facts plain sentences; no `Context:` / `Categories:` |

**Summary:** 13 PASS · 0 BLOCKED · 1 PARTIAL · 0 FAIL

## Bugs / regressions found

None observed in the executed cases. All checked behaviors match the PR descriptions.

## Behavior changes worth release-noting

These are not bugs; they are intentional changes from PRs in the 3.5 → 3.7 delta that may
affect existing client integrations. Each links to the case that exercised it.

1. **`connection_timeout` / `read_timeout` are seconds, not milliseconds.** Pre-3.7, defaults
   were `30000` but interpreted as seconds → ~8.3h timeouts. Post-3.7, default is `30`. Any
   user who set values in the thousands needs to divide by 1000.
   *Source: `ConnectorClientConfig.java:42-43`. Case: 01.01.*

2. **Memory container / context management DELETE & GET return 404 (was 500) for missing
   resources.** Clients that retry on 5xx should not retry on 404.
   *Cases: 02.01.*

3. **Jackson 3.x duplicate-key detection is enforced.** Sending `{"foo":1,"foo":2}` now
   yields `400 json_parse_exception` ("Duplicate Object property"). Previously the second
   value was silently used. The exception envelope's nested type is `stream_read_exception`
   (Jackson 3 rename). Clients that key off exception type strings should review.
   *Case: 06.01.*

4. **V2 Chat Agent input format is `{"input": ...}`, not `{"parameters":{"messages":...}}`.**
   The wrong format yields a confusing `"V2 agents require executor-provided memory. Use
   runV2() instead."` error. Documentation should be explicit.
   *Case: 03.01.*

5. **V2 Chat Agent `model.model_id` is the provider's native model ID** (e.g.,
   `us.anthropic.claude-haiku-4-5-20251001-v1:0`), **not** an ml-commons-registered model
   ID. This is a meaningful semantic change versus V1's `llm.model_id`.
   *Case: 03.01.*

6. **V2 `model_parameters` values must be strings.** `"max_tokens": 1024` (int) fails;
   must be `"max_tokens": "1024"`.
   *Case: 03.01.*

7. **V2 vs V1 token-usage schema differs.** V2 uses `metrics.total_usage.{inputTokens,
   outputTokens, totalTokens}` (camelCase from Bedrock); V1 uses `token_usage.per_turn_usage[].
   {input_tokens, output_tokens, total_tokens}` (snake_case). Worth aligning in a future PR.
   *Case: 03.02.*

8. **V2 chat agents do not support `_execute/stream` in 3.7.** Calling the streaming
   endpoint on a `conversational_v2` agent returns HTTP 400 with a clear message pointing
   to the non-streaming `_execute` endpoint. Document explicitly in V2 docs.
   *Case: 04.02.*

## Known cluster limitations

- **HTTP streaming + security**: requires `transport-reactor-netty4` plugin (cluster B).
  The default `SecurityHttpServerTransport` (cluster A) does not support `_execute/stream`.
  All streaming tests should run on cluster B.
- **Mid-stream Bedrock errors not propagated to SSE error chunk**: when the underlying
  Bedrock streaming client encounters bad credentials / 401, the SSE channel opens (HTTP 200)
  but no chunks emit and no error chunk is delivered before client timeout. This may be a
  separate gap in the streaming Bedrock client; it is **not a regression** — #4792's
  `onErrorResume` only fires when the upstream Flux errors out, and if the upstream silently
  hangs, the fix can't help. Worth a follow-up issue.
- **AGUI streaming**: the AGUI streaming runner appears to need additional setup beyond
  `ag_ui_enabled=true` to deliver chunks. The `RunErrorEvent` branch of #4792's
  `buildStreamErrorChunk()` is therefore exercised only by the unit tests in the PR. Worth a
  CI-side e2e in a future cycle.
- **Local model fixtures**: no decoder-only or pre-pooled sentence-transformer model is
  staged on this host. Affects: 07.01 (PARTIAL). Parser-level validation passed; full
  embedding-correctness validation needs model fixtures.

## What was NOT exercised in this run

(Items in `test-plan.md` deliberately not run, with rationale.)

- **EncryptorImpl async multi-tenant scaling (#3919)** — needs a load-test harness, not
  feasible in interactive testing.
- **SdkAsyncHttpClient resource leak fix (#4716)** — soak test, needs hours of traffic +
  JVM thread/memory profiling.
- **FIPS-by-default build/run (#4719)** — build-system check, not a runtime test.
- **PER (Plan-Execute-Reflect) token tracking (#4683)** — would require a PER agent + tool
  fixtures.
- **AGUI streaming agent + CustomEvent token emission** — endpoint reachable on cluster B
  but no chunks emit; needs investigation (see "Known cluster limitations" above).
- **Yandex Cloud blueprint walkthrough (#4810, #4469)** — doc-only; verifying the
  blueprints requires a live Yandex AI Studio account.
- **MCP SSE / streamable HTTP transports end-to-end (#4795)** — needs a running external
  MCP server. The MCP server's tool register/list (which is in-process) is covered by 06.01.
- **`ConnectorAccessControlHelper` coverage (#4689)** — test-only PR, not a runtime feature.

## Next steps

1. Investigate why mid-stream Bedrock errors (e.g., 401 inside the streaming client) do not
   reach the `onErrorResume` handler and therefore aren't surfaced as SSE error chunks.
   Likely outside the scope of #4792; file as a separate issue.
2. Investigate why AGUI agent streaming opens HTTP 200 but emits no chunks on cluster B.
   Could be a setup item we missed, or a real defect.
3. Stage decoder-only and pre-pooled model fixtures, then complete 07.01 e2e.
4. Schedule a 1-hour soak test of `_predict` against a Bedrock connector to validate the
   #4716 leak fix.
5. Add the eight "behavior changes worth release-noting" items (incl. V2-no-streaming) to
   the 3.7 release notes / migration guide.
