# Test 7 — List graph entities

**Goal:** Verify `GET /_plugins/_ml/memory_containers/{id}/memories/graph/entities` returns an empty result set for a graph-enabled but empty container.

**Status:** ✅ PASS

## Request

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XGET "https://localhost:9200/_plugins/_ml/memory_containers/4aRp0J0BQ4F7Y_V3tOLD/memories/graph/entities?query=*&top_k=10"
```

## Response (HTTP 200)

```json
{ "total_count": 0, "query": "*" }
```

## Observations

- `GET` on `.../graph/entities` is a query-param interface (no body required), unlike the POST `_search` endpoint.
- Internally `TransportListGraphEntitiesAction` delegates to `TransportGraphSearchAction` with `search_type="text"` and the `*` query fed to the neural search.
- With no entities in `.plugins-ml-am-e2e-memory-lpg-nodes`, the neural query returns an empty hits array and the response carries `total_count=0`.

## Notes / open questions

- **No field for entity_type filter in response** — caller can pass `entity_type` as a query param, but the handler does not echo it back in the response or surface it in logs when used. Not blocking, but making `entity_type` show in the response would help verification.
- The `entity_type` query parameter is accepted but currently is **not** forwarded into the downstream neural query filter (it's set on `MLGraphSearchInput.entityType` but never applied inside `GraphSearchService.searchEntitiesByText`). Worth a follow-up.
