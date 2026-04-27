# Test 3 — Create container with `enable_graph: false` and verify rejection

**Goal:** Verify that when `enable_graph` is not enabled on a container, the graph REST endpoints reject requests with a 400 and a clear error message.

**Status:** ✅ PASS

## Step 1 — Create plain container

### Request

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/_create" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "plain-container-e2e",
    "description": "Graph-disabled container",
    "configuration": {
      "embedding_model_type": "TEXT_EMBEDDING",
      "embedding_model_id": "BWxf0J0BTxnG51e8Pch9",
      "embedding_dimension": 384,
      "index_prefix": "plain",
      "enable_graph": false
    }
  }'
```

### Response

```json
{ "memory_container_id": "7qRr0J0BQ4F7Y_V3p-Jo", "status": "created" }
```

### Indices created (no lpg-nodes/lpg-edges)

```
.plugins-ml-am-plain-memory-sessions
.plugins-ml-am-plain-memory-working
```

## Step 2 — Attempt to list graph entities

### Request

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XGET "https://localhost:9200/_plugins/_ml/memory_containers/7qRr0J0BQ4F7Y_V3p-Jo/memories/graph/entities?query=*&top_k=10"
```

### Response (HTTP 400)

```json
{
  "error": {
    "root_cause": [{
      "type": "status_exception",
      "reason": "Graph functionality is not enabled for this memory container"
    }],
    "type": "status_exception",
    "reason": "Graph functionality is not enabled for this memory container"
  },
  "status": 400
}
```

The rejection comes from `TransportGraphSearchAction`'s `enableGraph` guard (also covered by the unit test `testDoExecute_GraphDisabled`).
