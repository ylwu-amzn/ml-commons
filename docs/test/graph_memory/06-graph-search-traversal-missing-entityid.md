# Test 6 — Graph search `search_type=traversal`

**Goal:** Verify traversal search (a) rejects requests lacking `entity_id`, and (b) performs BFS from a known entity on a populated graph.

**Status:** ✅ PASS (both cases); one minor issue noted

## Request A — missing `entity_id` (negative)

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4Rh60J0B8jwm23PNlvB4/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"root","search_type":"traversal"}'
```

### Response (HTTP 400)

```json
{
  "error": {
    "root_cause": [{ "type": "status_exception", "reason": "Entity ID is required for traversal search" }],
    "type": "status_exception",
    "reason": "Entity ID is required for traversal search"
  },
  "status": 400
}
```

## Request B — traversal from `e-alice`, depth 2

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4Rh60J0B8jwm23PNlvB4/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"alice","search_type":"traversal","entity_id":"e-alice","max_depth":2}'
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
  "relationships": [
    { "relationship_id": "r1", "source_entity": "e-alice", "target_entity": "e-pasta", "relationship_type": "LIKES",        "confidence": 0.9 },
    { "relationship_id": "r2", "source_entity": "e-alice", "target_entity": "e-acme",  "relationship_type": "WORKS_AT",     "confidence": 0.9 },
    { "relationship_id": "r4", "source_entity": "e-alice", "target_entity": "e-bob",   "relationship_type": "COLLEAGUE_OF", "confidence": 0.9 },
    { "relationship_id": "r1", "source_entity": "e-alice", "target_entity": "e-pasta", "relationship_type": "LIKES",        "confidence": 0.9 },
    { "relationship_id": "r2", "source_entity": "e-alice", "target_entity": "e-acme",  "relationship_type": "WORKS_AT",     "confidence": 0.9 },
    { "relationship_id": "r3", "source_entity": "e-bob",   "target_entity": "e-acme",  "relationship_type": "WORKS_AT",     "confidence": 0.9 },
    { "relationship_id": "r4", "source_entity": "e-alice", "target_entity": "e-bob",   "relationship_type": "COLLEAGUE_OF", "confidence": 0.9 }
  ],
  "total_count": 4,
  "query": "alice"
}
```

BFS starting at Alice reaches all 4 entities and returns all 4 relationships that bridge them — correct behavior.

## Issue — duplicate edges in traversal output

**Bug 6.1 (minor)** — The `relationships` array contains `r1`, `r2`, `r4` twice (once at depth 1 from Alice, once at depth 2 from Bob or as back-edges). The response should dedupe by `relationship_id`.

Source: `GraphSearchService.findRelatedEntities` uses a `visitedEntities` set but appends to a plain `List<GraphRelationship>` across BFS levels without checking membership. Recommend switching to `Map<String, GraphRelationship>` keyed on `relationshipId`, or filter when serializing the response.

Not fixed in this session.
