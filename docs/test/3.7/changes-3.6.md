# Changes shipped in the 3.5 → 3.6 wave

These changes are all already on the `3.6` release branch and tagged `3.6.0.0`. They are part
of the 3.7 delta from a 3.5 user's point of view, so QA must cover them.

Section labels match the 3.6 release notes (`release-notes/opensearch-ml-commons.release-notes-3.6.0.0.md`).

## Features

| PR | Title | Notes for testing |
|----|-------|-------------------|
| [#4732](https://github.com/opensearch-project/ml-commons/pull/4732) | **Introduce V2 Chat Agent** with unified interface, multi-modal support, simplified registration | New agent type. Validate registration, single-turn / multi-turn execution, tool use, multi-modal input, memory integration, streaming, error paths. |
| [#4658](https://github.com/opensearch-project/ml-commons/pull/4658) | **Semantic and hybrid search APIs for long-term memory retrieval** in agentic memory | New endpoints `_semantic_search` and `_hybrid_search`. Hybrid uses inline temporary pipeline (no pre-create needed). Tunable `bm25_weight` / `neural_weight` (default 0.5/0.5). Need neural-search plugin; validate clear error when missing. Validate `k` range [1, 10000]. |
| [#4683](https://github.com/opensearch-project/ml-commons/pull/4683) | **Token usage tracking** for Conversational, AG-UI, Plan-Execute-Reflect agents | Per-model and per-turn tracking. Opt-in via `include_token_usage`. AGUI emits CustomEvent. Sub-agent token logging suppressed in PER. |
| [#4711](https://github.com/opensearch-project/ml-commons/pull/4711) | **LAST_TOKEN pooling** for text embedding (decoder-only / GPT-style / Qwen3) | New pooling option in PoolingMode. Validate model registration with `pooling_mode=lasttoken` for both ONNX and TorchScript. |
| [#4710](https://github.com/opensearch-project/ml-commons/pull/4710) | **NONE pooling mode** for pre-pooled model outputs | Avoids redundant pooling. Validate sentence-transformer models that ship pre-pooled `sentence_embedding`. ONNX uses 2nd output; HuggingFace translator has fallback chain. |
| [#4729](https://github.com/opensearch-project/ml-commons/pull/4729) | **Custom fallback query** in QueryPlanningTool for agentic search | Validate fallback path when LLM fails to generate a plan. |
| [#4645](https://github.com/opensearch-project/ml-commons/pull/4645) | **Messages array in all memory types + chat history in AGUI agent** | Memory interface extension. Validate remote agentic memory and MessagesSnapshot AGUI event. |
| [#4687](https://github.com/opensearch-project/ml-commons/pull/4687) | **Post-memory hook with structured message** for context managers | Validate that hook runs after memory write with structured payload. |
| [#3919](https://github.com/opensearch-project/ml-commons/pull/3919) | **EncryptorImpl async handling for scalability**, fix duplicate master key generation | Removed `CountDownLatch`. Per-tenant in-flight tracking. Validate concurrent multi-tenant key generation does not duplicate keys, no thread starvation. |
| [#4726](https://github.com/opensearch-project/ml-commons/pull/4726) | **Aliases and wildcard index patterns in QueryPlanningTool** | Previously NPE on alias/wildcard. Validate `index_name=logs_*` and alias `logs` resolve to mappings. Multi-index match logs a warning. |

## Enhancements

| PR | Title | Notes for testing |
|----|-------|-------------------|
| [#4681](https://github.com/opensearch-project/ml-commons/pull/4681) | More detailed logging in Agent Workflow | Verify new log lines appear; useful for debug. |
| [#4637](https://github.com/opensearch-project/ml-commons/pull/4637) | Allow overwrite during execute for inline create context management during agent register | Validate inline-creation overwrite path. |
| [#4676](https://github.com/opensearch-project/ml-commons/pull/4676) | Helper method for Nova clean request | Sanity test Nova invocations. |
| [#4692](https://github.com/opensearch-project/ml-commons/pull/4692) | Override `ValidatingObjectInputStream.resolveClass()` — plugin classloader fallback | Validate model serde across plugin boundaries (ModelSerDeSer). |
| [#4720](https://github.com/opensearch-project/ml-commons/pull/4720) | Restore AGUI context for legacy interface agent | Validate legacy agents still receive AGUI context. |
| [#4747](https://github.com/opensearch-project/ml-commons/pull/4747) | Escape tool name and description (handle quotation marks) | Validate tools whose names/descriptions contain `"`/`\` are rendered correctly in prompts. |

## Bug fixes

| PR | Title | Notes for testing |
|----|-------|-------------------|
| [#4716](https://github.com/opensearch-project/ml-commons/pull/4716) | Fix `SdkAsyncHttpClient` resource leak in connector executors | Long-running test: invoke many connectors, watch Netty event loop / connection pool. `RemoteConnectorExecutor` is now `AutoCloseable`. |
| [#4712](https://github.com/opensearch-project/ml-commons/pull/4712) | `Tags.addTag()` return value not captured (immutable Tags) | Validate metric tags actually appear after recent immutable Tags refactor. |
| [#4759](https://github.com/opensearch-project/ml-commons/pull/4759) | `connection_timeout` / `read_timeout` defaults `30000` → `30` (seconds, not millis) | Important: any caller passing `30000` previously got 8h timeouts. Validate defaults; check log unit shown as seconds. |
| [#4656](https://github.com/opensearch-project/ml-commons/pull/4656) | Numeric type preservation in ML inference query template substitution | Validate range queries with numeric substitution (e.g., embedding length). Bedrock connector raw response. |
| [#4560](https://github.com/opensearch-project/ml-commons/pull/4560) | Early exit in stats collector job when fetching connector for model details | Validate stats job no longer aborts. |
| [#4772](https://github.com/opensearch-project/ml-commons/pull/4772) | `RestChatAgentIT` teardown failure when AWS credentials absent | CI-only fix. |
| [#4762](https://github.com/opensearch-project/ml-commons/pull/4762) | `agent_id` → `agent_id_log` rename to avoid AgentTool infinite loop | Validate AgentTool nested invocation path. |
| [#4739](https://github.com/opensearch-project/ml-commons/pull/4739) | Respect MCP connector setting for Agent V2 | Validate `plugins.ml_commons.mcp_connector_enabled` (or equivalent) gate honored by V2. |
| [#4723](https://github.com/opensearch-project/ml-commons/pull/4723) | Wrong error code when deleting memory containers / context management templates | Validate DELETE returns proper 4xx (404 for missing). |
| [#4725](https://github.com/opensearch-project/ml-commons/pull/4725) | `OpenSearchStatusException` → `OpenSearchException` for broader 4XX coverage | Validate client-side errors during memory create / session create / predict no longer get masked as 500. |
| [#4701](https://github.com/opensearch-project/ml-commons/pull/4701) | Delete context management template returns 404 (was 500) | Validate. |
| [#4730](https://github.com/opensearch-project/ml-commons/pull/4730) | Context restoration bug — user info missing | Validate user identity propagation through async paths. |
| [#4733](https://github.com/opensearch-project/ml-commons/pull/4733) | `UnsupportedOperationException` putting agent_id into immutable map | Validate AgentTool path again after this fix. |
| [#4767](https://github.com/opensearch-project/ml-commons/pull/4767) | Cohere IT timeout 120s + reachability check | Test infra. |

## Infrastructure

| PR | Title | Notes |
|----|-------|-------|
| [#4654](https://github.com/opensearch-project/ml-commons/pull/4654) | Adapt to Gradle shadow plugin v9, FIPS build param aware | Build-only. |
| [#4719](https://github.com/opensearch-project/ml-commons/pull/4719) | FIPS flag enabled by default | Validate FIPS-mode build & integ tests. |
| [#4666](https://github.com/opensearch-project/ml-commons/pull/4666) | Code diff analyzer / reviewer | CI tooling. |
| [#4667](https://github.com/opensearch-project/ml-commons/pull/4667) | Optimize IT setup (~50% faster) | CI perf. |
| [#4659](https://github.com/opensearch-project/ml-commons/pull/4659) | Quote FIPS crypto-standard param | CI. |
| [#4668](https://github.com/opensearch-project/ml-commons/pull/4668) | Skip unreachable OpenAI tests, fix flaky `IndexUtilsTests` | CI. |
| [#4665](https://github.com/opensearch-project/ml-commons/pull/4665) | `SearchModelGroupITTests` timeout, Bedrock connection pool | CI. |
| [#4742](https://github.com/opensearch-project/ml-commons/pull/4742) | Bedrock Claude model bumps for higher RPM | CI. |
| [#4737](https://github.com/opensearch-project/ml-commons/pull/4737) | Rename `resource-action-groups.yml` → `resource-access-levels.yml` | Security plugin compatibility. |
