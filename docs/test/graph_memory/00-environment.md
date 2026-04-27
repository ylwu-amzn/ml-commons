# Graph Memory E2E — Test Environment

## Cluster

| Property | Value |
|---|---|
| Distribution | OpenSearch 3.5.0 (tar) |
| Path | `/home/ubuntu/os/opensearch-3.5.0` |
| HTTPS endpoint | `https://localhost:9200` |
| Auth | `admin` / `RUIwNTVGTDc2MDhDM0JNRC4u` |
| Security plugin | enabled |

## Plugin built and installed

| Property | Value |
|---|---|
| Plugin zip | `/home/ubuntu/code/ylwu/ml-commons-dev/plugin/build/distributions/opensearch-ml-3.5.0.0.zip` |
| Branch | `local-test/3.5` (fork of `feature/memory_graph`) |
| Plugin version | `3.5.0.0` |
| Plugin descriptor `opensearch.version` | `3.5.0` |

### In-session fixes required to make the graph feature reachable

1. Added `@Inject` to `GraphSearchService` so Guice can construct it.
2. Registered `TransportGraphSearchAction`, `TransportListGraphEntitiesAction`, `TransportDeleteGraphDataAction` in `MachineLearningPlugin.getActions()`.
3. Registered `RestMLGraphSearchAction`, `RestMLListGraphEntitiesAction`, `RestMLDeleteGraphDataAction` in `MachineLearningPlugin.getRestHandlers()`.
4. Fixed `TransportCreateMemoryContainerAction.createGraphIndices` to pass `{"index.knn": true}` when creating the `lpg-nodes` index (the `knn_vector` mapping requires this setting).

Without fixes 1–3, no graph REST route was registered — every request to `/memory_containers/{id}/memories/graph/*` fell through to other handlers (or 404) and no transport action was bound. Without fix 4, the graph-nodes/edges indices silently failed to create on container creation.

## ML Commons cluster settings

```json
PUT /_cluster/settings
{
  "persistent": {
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99",
    "plugins.ml_commons.allow_registering_model_via_url": "true",
    "plugins.ml_commons.allow_registering_model_via_local_file": "true"
  }
}
```

`plugins.ml_commons.agentic_memory_enabled` is already `true` by default on this build.

## Registered models

| Role | Model ID | Model | Status |
|---|---|---|---|
| Text embedding | `BWxf0J0BTxnG51e8Pch9` | `huggingface/sentence-transformers/all-MiniLM-L6-v2` (384-dim, TORCH_SCRIPT) | DEPLOYED |
| LLM (graph extraction) | `3qRp0J0BQ4F7Y_V3OOKT` | Bedrock Claude Sonnet 4.6 (`us.anthropic.claude-sonnet-4-6`) via AWS SigV4 connector `26Rp0J0BQ4F7Y_V3BOI8` | DEPLOYED |

### Bedrock connector (credentials redacted)

```json
POST /_plugins/_ml/connectors/_create
{
  "name": "Bedrock Claude Sonnet 4.6 Connector",
  "version": "1",
  "protocol": "aws_sigv4",
  "parameters": {
    "region": "us-west-2",
    "service_name": "bedrock",
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 4096,
    "model": "us.anthropic.claude-sonnet-4-6"
  },
  "credential": { "access_key": "…", "secret_key": "…" },
  "actions": [{
    "action_type": "predict",
    "method": "POST",
    "url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/invoke",
    "headers": { "content-type": "application/json" },
    "request_body": "{\"anthropic_version\":\"${parameters.anthropic_version}\",\"max_tokens\":${parameters.max_tokens},\"messages\":[{\"role\":\"user\",\"content\":\"${parameters.prompt}\"}]}"
  }]
}
```

## Test container

| Property | Value |
|---|---|
| Memory container ID | `4aRp0J0BQ4F7Y_V3tOLD` |
| Name | `graph-container-e2e` |
| `enable_graph` | `true` |
| Index prefix | `e2e` |

Indices created:

```
.plugins-ml-am-e2e-memory-sessions
.plugins-ml-am-e2e-memory-working
.plugins-ml-am-e2e-memory-lpg-nodes       # knn_vector enabled
.plugins-ml-am-e2e-memory-lpg-edges
```
