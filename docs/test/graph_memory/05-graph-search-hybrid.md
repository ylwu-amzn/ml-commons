# Test 5 — Graph search `search_type=hybrid` (default)

**Goal:** Verify hybrid search blends neural text-similarity hits with graph relationship expansion.

**Status:** ✅ PASS

Hybrid is the **default** `search_type` when the field is absent.

## Seed data

Same 4-node / 4-edge graph as [Test 4](./04-graph-search-text.md).

## Request A — hybrid default (no `search_type`): "alice pasta"

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4Rh60J0B8jwm23PNlvB4/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"alice pasta","top_k":5}'
```

### Response (HTTP 200)

```json
{
  "entities": [
    { "entity_id": "e-pasta", "name": "pasta",     "type": "food",    "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-alice", "name": "Alice",     "type": "person",  "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-acme",  "name": "Acme Corp", "type": "company", "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-bob",   "name": "Bob",       "type": "person",  "confidence": 0.95, "mention_count": 1 }
  ],
  "search_results": [
    { "entity": { "entity_id": "e-pasta", "name": "pasta", "type": "food",    "confidence": 0.95, "mention_count": 1 }, "score": 1.0, "match_type": "text_similarity",        "query_text": "alice pasta" },
    { "entity": { "entity_id": "e-alice", "name": "Alice", "type": "person",  "confidence": 0.95, "mention_count": 1 }, "score": 1.0, "match_type": "text_similarity",        "query_text": "alice pasta" },
    { "entity": { "entity_id": "e-acme",  "name": "Acme Corp", "type": "company", "confidence": 0.95, "mention_count": 1 }, "score": 0.6, "match_type": "relationship_expansion", "query_text": "alice pasta", "relationship_count": 1 },
    { "entity": { "entity_id": "e-bob",   "name": "Bob",   "type": "person",  "confidence": 0.95, "mention_count": 1 }, "score": 0.6, "match_type": "relationship_expansion", "query_text": "alice pasta", "relationship_count": 1 }
  ],
  "total_count": 4,
  "query": "alice pasta"
}
```

- `match_type="text_similarity"` on Alice and pasta — both match the neural query directly (both terms in the query).
- `match_type="relationship_expansion"` on Acme and Bob — reached via graph edges from the text-similarity hits (Alice-WORKS_AT→Acme, Alice-COLLEAGUE_OF→Bob).

## Request B — explicit `search_type=hybrid`: "pasta"

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4Rh60J0B8jwm23PNlvB4/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"pasta","search_type":"hybrid","top_k":3}'
```

### Response (HTTP 200)

```json
{
  "entities": [
    { "entity_id": "e-pasta", "name": "pasta",     "type": "food",    "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-alice", "name": "Alice",     "type": "person",  "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-acme",  "name": "Acme Corp", "type": "company", "confidence": 0.95, "mention_count": 1 }
  ],
  "search_results": [
    { "entity": { "entity_id": "e-pasta", "name": "pasta",     "type": "food",    "confidence": 0.95, "mention_count": 1 }, "score": 1.0, "match_type": "text_similarity",        "query_text": "pasta" },
    { "entity": { "entity_id": "e-alice", "name": "Alice",     "type": "person",  "confidence": 0.95, "mention_count": 1 }, "score": 0.9, "match_type": "relationship_expansion", "query_text": "pasta", "relationship_count": 4 },
    { "entity": { "entity_id": "e-acme",  "name": "Acme Corp", "type": "company", "confidence": 0.95, "mention_count": 1 }, "score": 0.6, "match_type": "relationship_expansion", "query_text": "pasta", "relationship_count": 1 }
  ],
  "total_count": 3,
  "query": "pasta"
}
```

Pasta is the direct text-similarity hit (score 1.0). Alice reaches score 0.9 because she's connected via multiple relationships (`relationship_count=4`). Acme is reached via 2-hop through Alice (`relationship_count=1`, score 0.6).

## Request C — hybrid on empty graph

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4aRp0J0BQ4F7Y_V3tOLD/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"alice pasta","top_k":5}'
```

### Response (HTTP 200)

```json
{ "total_count": 0, "query": "alice pasta" }
```
