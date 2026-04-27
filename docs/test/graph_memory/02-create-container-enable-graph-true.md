# Test 2 — Create memory container with `enable_graph: true`

**Goal:** Verify that creating a memory container with `enable_graph: true` provisions the graph-nodes and graph-edges indices in addition to the regular memory indices.

**Status:** ✅ PASS (after in-session fix to `createGraphIndices`)

## Request

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/_create" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "graph-container-e2e",
    "description": "Graph-enabled container for e2e testing with Bedrock Sonnet 4.6 + MiniLM",
    "configuration": {
      "embedding_model_type": "TEXT_EMBEDDING",
      "embedding_model_id": "BWxf0J0BTxnG51e8Pch9",
      "embedding_dimension": 384,
      "llm_id": "3qRp0J0BQ4F7Y_V3OOKT",
      "index_prefix": "e2e",
      "enable_graph": true
    }
  }'
```

## Response

```json
{ "memory_container_id": "4aRp0J0BQ4F7Y_V3tOLD", "status": "created" }
```

## Verification

### Indices created

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  "https://localhost:9200/_cat/indices?v&h=index&expand_wildcards=all" | grep e2e
```

```
.plugins-ml-am-e2e-memory-sessions
.plugins-ml-am-e2e-memory-working
.plugins-ml-am-e2e-memory-lpg-nodes
.plugins-ml-am-e2e-memory-lpg-edges
```

All four indices are present, including the two graph indices.

### Graph-nodes mapping has `knn_vector`

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XGET "https://localhost:9200/.plugins-ml-am-e2e-memory-lpg-nodes/_mapping"
```

```json
{
  ".plugins-ml-am-e2e-memory-lpg-nodes": {
    "mappings": {
      "properties": {
        "entity_embedding": {
          "type": "knn_vector",
          "dimension": 384,
          "method": { "engine": "faiss", "space_type": "cosinesimil", "name": "hnsw", "parameters": {} }
        },
        "entity_id": { "type": "keyword" },
        "entity_name": { "type": "text", "analyzer": "standard" },
        "entity_type": { "type": "keyword" },
        "confidence": { "type": "float" },
        "memory_container_id": { "type": "keyword" },
        "owner_id": { "type": "keyword" },
        "tenant_id": { "type": "keyword" },
        "created_time": { "type": "long" },
        "updated_time": { "type": "long" },
        "mention_count": { "type": "integer" }
      }
    }
  }
}
```

## Bugs found and fixed during this test

### Bug 2.1 — `createGraphIndices` did not enable `index.knn`

**Symptom:** First attempt to create a graph-enabled container succeeded with HTTP 200, but only the sessions + working indices were created. Server log showed:

```
[ERROR] failed on parsing mappings on index creation [.plugins-ml-am-e2e-memory-lpg-nodes]
Caused by: java.lang.IllegalArgumentException: Cannot set modelId or method parameters when index.knn setting is false
```

The `GRAPH_NODES_INDEX_MAPPING` uses `knn_vector` with an `hnsw` method, which requires `index.knn=true`. `createGraphIndices` was calling the 4-arg `initIndexIfAbsent` which passes `null` settings, causing OpenSearch's KNN plugin to reject the mapping. The outer handler in `TransportCreateMemoryContainerAction` then logs the failure and **returns success anyway** (graph indices are treated as optional) — so the user sees a `201 created` even though the feature silently broke.

**Fix:** Switched to the 5-arg overload and passed `{"index.knn": true}`:

```java
Map<String, Object> nodesSettings = Map.of("index.knn", true);
mlIndicesHandler.initIndexIfAbsent(graphNodesIndex, nodesMapping, nodesSettings, 1, …, useSystemIndex);
```

Bug is structural: the "continue on graph-index failure" behavior hides real misconfiguration from the caller. Graph-related API calls then fail with `index_not_found`.

### Bug 2.2 — `embedding_dimension` field name mismatch in error message

**Symptom:** First create attempt used `dimension: 384`, rejected with `"Dimension is required for TEXT_EMBEDDING"`. The JSON field is actually `embedding_dimension` (per `MemoryContainerConstants.DIMENSION_FIELD`). The error message should have been clearer: `"embedding_dimension is required"`.

Not fixed in this session (minor usability issue, not a correctness bug).
