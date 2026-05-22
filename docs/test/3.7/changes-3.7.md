# Changes shipped in the 3.6 → 3.7 wave (new in 3.7)

These are the commits on `main` after the `3.6.0.0` tag (i.e., commits that 3.6 users will get
fresh in 3.7).

Range: `3.6.0.0..upstream/main` — 23 commits.

## Features / new content

| PR | Title | Notes for testing |
|----|-------|-------------------|
| [#4810](https://github.com/opensearch-project/ml-commons/pull/4810) | **Yandex Cloud AI Studio standard embedding blueprint** | New blueprint at `docs/remote_inference_blueprints/standard_blueprints/yandexcloud_connector_embedding_standard_blueprint.md`. Doc-only — sanity-check by running through the blueprint with a Yandex AI Studio account. README updated. |
| [#4469](https://github.com/opensearch-project/ml-commons/pull/4469) | **Yandex Cloud embeddings legacy blueprint** | Companion legacy-style blueprint. |

## Bug fixes — Agent / Streaming / Memory

| PR | Title | Notes for testing |
|----|-------|-------------------|
| [#4798](https://github.com/opensearch-project/ml-commons/pull/4798) | **Simplify USER_PREFERENCE extraction prompt** to plain sentences | Previously prompt embedded `Context: …. Categories: …` metadata which caused malformed JSON on smaller LLMs (Haiku 3) and degraded embedding quality. Now matches SEMANTIC strategy format. Validate USER_PREFERENCE fact extraction with both small and large LLMs. Resolves [#4780](https://github.com/opensearch-project/ml-commons/issues/4780). |
| [#4792](https://github.com/opensearch-project/ml-commons/pull/4792) | **Propagate stream error** through AGUI events / unwrap cause | Validate that streaming errors from upstream (Bedrock, OpenAI, etc.) are surfaced as AGUI error events with the underlying cause. Tests added. |
| [#4794](https://github.com/opensearch-project/ml-commons/pull/4794) | **Fix tool output JSON escape** + duplicate `_source include/exclude` issue | Validate (a) flow-agent tool outputs containing `"`/`\` produce valid JSON, (b) `model_content` is no longer both included and excluded in `_source`. |
| [#4762](https://github.com/opensearch-project/ml-commons/pull/4762) | `agent_id` → `agent_id_log` rename to avoid AgentTool infinite loop | Same as 3.6 PR (this is the cherry-picked fix landing on main). |

## Bug fixes — Build / Dependencies

| PR | Title | Notes for testing |
|----|-------|-------------------|
| [#4811](https://github.com/opensearch-project/ml-commons/pull/4811) | **Update Gradle to 9.4.1, Jacoco to 0.8.14** | OpenSearch core requires Gradle 9.4.1+. Validate clean build, jacoco reports. |
| [#4795](https://github.com/opensearch-project/ml-commons/pull/4795) | **Support Jackson 3.x release line** | Jackson upgrade. Touches `StringUtils`, `MLNodeUtils`, `MLExtractJsonProcessor`, `BedrockStreamingHandler`, MCP transport, Nova preprocess, Index Insight. Bumps `io.modelcontextprotocol.sdk:mcp` to 1.1.1. **Highest-risk dependency change in 3.7.** Validate JSON parsing across all major paths. |
| [#4784](https://github.com/opensearch-project/ml-commons/pull/4784) | **Fix Jackson exception handling** post 3.x migration | Touches `MLCommonsClassLoader`, `MLAgentExecutorTest`, search response processors, generative QA. Validate exception types raised on bad JSON input. |

## Bug fixes — CI / Infra (smaller blast radius)

| PR | Title | Notes |
|----|-------|-------|
| [#4781](https://github.com/opensearch-project/ml-commons/pull/4781) | Fix flaky `RestMLInferenceSearchResponseProcessorIT` connection pool timeout | Reduces per-test connector creation; reduces `max_connection` 200 → 50; enables `max_retry_times: 3` for Bedrock. CI-only. |
| [#4767](https://github.com/opensearch-project/ml-commons/pull/4767) | Increase Cohere IT timeout to 120s + reachability check | CI-only. |
| [#4759](https://github.com/opensearch-project/ml-commons/pull/4759) | Fix `connection_timeout` / `read_timeout` defaults to seconds | Same as 3.6 PR landing on main. |
| [#4742](https://github.com/opensearch-project/ml-commons/pull/4742) | Upgrade Bedrock Claude models in IT for higher rate limits | CI-only. |
| [#4772](https://github.com/opensearch-project/ml-commons/pull/4772) | `RestChatAgentIT` teardown failure when AWS credentials absent | CI-only. |
| [#4689](https://github.com/opensearch-project/ml-commons/pull/4689) | Increase code coverage in `ConnectorAccessControlHelper` (to 99.30%) | Test-only. |
| [#4737](https://github.com/opensearch-project/ml-commons/pull/4737) | `resource-action-groups.yml` → `resource-access-levels.yml` | Security plugin alignment. |
| [#4827](https://github.com/opensearch-project/ml-commons/pull/4827) | Add issues:write permission to untriaged label workflow | CI-only. |
| [#4788](https://github.com/opensearch-project/ml-commons/pull/4788) | Add `akolarkunnu` as maintainer | Doc-only. |

## Bookkeeping

| PR | Title |
|----|-------|
| [#4751](https://github.com/opensearch-project/ml-commons/pull/4751) | Increment version to 3.7.0-SNAPSHOT |
| [#4777](https://github.com/opensearch-project/ml-commons/pull/4777) (cherry-picked as #4778) | Add 3.6.0 release notes |

## Risk summary

The 3.6 → 3.7 wave is dominated by **dependency / build-tooling upgrades** (Jackson 3.x, Gradle
9.4.1) plus targeted bug fixes on the agent + streaming + memory path. The two areas to focus
QA effort:

1. **Jackson 3.x migration** (#4795, #4784) — broad surface area, test JSON-heavy paths in:
   - `StringUtils.fromJson` / `toJson`
   - `MLExtractJsonProcessor`
   - Bedrock streaming handler (Converse / InvokeModelWithResponseStream)
   - MCP transport (SSE, streamable HTTP, stateless server)
   - Nova multi-modal embedding preprocess
   - Index Insight `LogRelatedIndexCheckTask`
   - ML inference ingest processor
2. **Streaming error propagation** (#4792, #4794) — every stream-capable connector + agent
   should be exercised against an LLM error (rate limit, 5xx, malformed payload).
