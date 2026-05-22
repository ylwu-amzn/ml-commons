# 3.7 Run Log — Aggregated Results

Snapshot of test execution against a local OpenSearch 3.7.0 cluster running ml-commons
3.7.0-SNAPSHOT. One row per test case file under `cases/`.

## Cluster under test

- OpenSearch 3.7.0 (`build_hash 8f2d05879…`, build_date 2026-05-19)
- ml-commons 3.7.0-SNAPSHOT
- Plugins: opensearch-knn, opensearch-neural-search, opensearch-security,
  opensearch-skills, opensearch-flow-framework, etc. (full list in
  `cases/00_smoke/01_cluster_reachable.md`)
- Single-node, security-enabled (admin / RUIw…), HTTP transport =
  `SecurityHttpServerTransport` (no streaming support)

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
| 04.01 | [Stream error propagation](./cases/04_streaming/01_stream_error_propagation_4792.md) | #4792 | BLOCKED | Cluster's HTTP transport (security plugin) does not support streaming |
| 05.01 | [Tool name/output JSON escape](./cases/05_tools_and_query_planning/01_tool_json_escape_4747_4794.md) | #4747, #4794 | PASS | Tool desc with `"`/`\`; flow agent output with `"`/`\\`/`\n`/`\t` valid JSON |
| 05.02 | [QueryPlanningTool wildcards/aliases + fallback](./cases/05_tools_and_query_planning/02_qpt_wildcards_aliases_4726_4729.md) | #4726, #4729 | PASS | Wildcard `t37_logs_2024_*` + alias resolve; `fallback_query` registers |
| 06.01 | [Jackson 3.x JSON paths](./cases/06_jackson_mcp/01_jackson3_paths_4795_4784.md) | #4795, #4784 | PASS | `tools.jackson` in effect; STRICT_DUPLICATE_DETECTION enforced; MCP tool register/list |
| 07.01 | [Pooling modes LAST_TOKEN/NONE](./cases/07_pooling_modes/01_pooling_modes_4710_4711.md) | #4711, #4710 | PARTIAL | Parser accepts both; e2e blocked by lack of decoder-only / pre-pooled model fixtures |
| 08.01 | [USER_PREFERENCE simplified prompt](./cases/08_user_preference/01_user_preference_simplified_prompt_4798.md) | #4798 | PASS | All extracted facts plain sentences; no `Context:` / `Categories:` |

**Summary:** 11 PASS · 1 BLOCKED · 1 PARTIAL · 0 FAIL

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

## Known cluster limitations

These prevented full end-to-end validation of two cases. Recommend re-running them on a
non-security or streaming-enabled cluster, or in CI.

- **HTTP streaming** (`/_execute/stream`): not supported by `SecurityHttpServerTransport`.
  Affects: 04.01 (BLOCKED). Source-level review confirms the fix is correct.
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
- **AGUI streaming agent + CustomEvent token emission** — same blocker as 04.01.
- **Yandex Cloud blueprint walkthrough (#4810, #4469)** — doc-only; verifying the
  blueprints requires a live Yandex AI Studio account.
- **MCP SSE / streamable HTTP transports end-to-end (#4795)** — same blocker as 04.01;
  also needs a running external MCP server.
- **`ConnectorAccessControlHelper` coverage (#4689)** — test-only PR, not a runtime feature.

## Next steps

1. Run cases 04.01 and the PER / AGUI streaming items on an HTTP-streaming-capable cluster.
2. Stage decoder-only and pre-pooled model fixtures, then complete 07.01 e2e.
3. Schedule a 1-hour soak test of `_predict` against a Bedrock connector to validate the
   #4716 leak fix.
4. Add the seven "behavior changes worth release-noting" items to the 3.7 release notes /
   migration guide.
