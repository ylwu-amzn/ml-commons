# Test 9 — Unknown memory container ID

**Goal:** Verify graph endpoints return HTTP 404 when the container id does not exist.

**Status:** ✅ PASS

## Request A — GET list entities

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XGET "https://localhost:9200/_plugins/_ml/memory_containers/does-not-exist/memories/graph/entities?query=*"
```

### Response (HTTP 404)

```json
{
  "error": {
    "root_cause": [{ "type": "status_exception", "reason": "Memory container not found" }],
    "type": "status_exception",
    "reason": "Memory container not found"
  },
  "status": 404
}
```

## Request B — DELETE graph data

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XDELETE "https://localhost:9200/_plugins/_ml/memory_containers/does-not-exist/memories/graph"
```

### Response (HTTP 404)

Identical to Request A.

## Observations

- The error is produced by `MemoryContainerHelper.getMemoryContainer` before any graph-specific logic runs, so the behaviour is consistent with non-graph memory endpoints.
