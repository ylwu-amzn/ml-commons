# Test 8 — Delete graph data (stub)

**Goal:** Verify `DELETE /_plugins/_ml/memory_containers/{id}/memories/graph` responds with a 200 success and a `DeleteResponse` body.

**Status:** ✅ PASS (stub path) — **real deletion not implemented**

## Request

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XDELETE "https://localhost:9200/_plugins/_ml/memory_containers/4aRp0J0BQ4F7Y_V3tOLD/memories/graph"
```

## Response (HTTP 200)

```json
{
  "_index": ".plugins-ml-am-e2e-memory-lpg-nodes",
  "_id": "graph_data_delete",
  "_version": 1,
  "result": "deleted",
  "_shards": null,
  "_seq_no": 0,
  "_primary_term": 0
}
```

## Observations & caveats

- This endpoint is **only a stub**. `TransportDeleteGraphDataAction.deleteGraphIndices` hardcodes a synthetic `DeleteResponse` and does not actually delete-by-query against the graph indices.
- The `_index` field in the response is the graph-nodes index name only. No delete-by-query is fired against the edges index either.
- **Earlier bug (fixed in this session):** the stub was constructing `DeleteResponse` with a `null` `ShardId`, which NPE'd for every caller (see `TransportDeleteGraphDataActionTests.testDoExecute_Success`). Fixed by passing a valid `ShardId(nodesIndex, "_na_", 0)`.

## Follow-up

Before shipping: replace the stub with real delete-by-query calls filtered by `memory_container_id` on both `*-lpg-nodes` and `*-lpg-edges` indices, aggregate the two `BulkByScrollResponse`s, and return a proper response shape (either a `BulkByScrollResponse` or a purpose-built result type).
