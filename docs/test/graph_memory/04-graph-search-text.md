# Test 4 — Graph search `search_type=text`

**Goal:** Verify neural text search against the graph-nodes index on both empty and populated graphs.

**Status:** ✅ PASS (empty and populated)

## Seed data (populated graph)

4 entities + 4 relationships on container `4Rh60J0B8jwm23PNlvB4`:

| Entity | Type | |
|---|---|---|
| Alice | person | |
| Bob | person | |
| pasta | food | |
| Acme Corp | company | |

Relationships: Alice-LIKES→pasta, Alice-WORKS_AT→Acme, Bob-WORKS_AT→Acme, Alice-COLLEAGUE_OF→Bob.

Seeded by directly indexing into `e2e2-memory-lpg-nodes` / `e2e2-memory-lpg-edges` with 384-dim embeddings from the MiniLM model.

## Request A — text search "alice"

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4Rh60J0B8jwm23PNlvB4/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"alice","search_type":"text","top_k":5}'
```

### Response (HTTP 200)

```json
{
  "entities": [
    { "entity_id": "e-alice", "name": "Alice",     "type": "person",  "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-bob",   "name": "Bob",       "type": "person",  "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-acme",  "name": "Acme Corp", "type": "company", "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-pasta", "name": "pasta",     "type": "food",    "confidence": 0.95, "mention_count": 1 }
  ],
  "total_count": 4,
  "query": "alice"
}
```

Ordering is the natural KNN similarity order: Alice first, Bob second (both people, close to "alice" embedding), then company, then food.

## Request B — text search "food"

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4Rh60J0B8jwm23PNlvB4/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"food","search_type":"text","top_k":3}'
```

### Response (HTTP 200)

```json
{
  "entities": [
    { "entity_id": "e-pasta", "name": "pasta", "type": "food",   "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-bob",   "name": "Bob",   "type": "person", "confidence": 0.95, "mention_count": 1 },
    { "entity_id": "e-alice", "name": "Alice", "type": "person", "confidence": 0.95, "mention_count": 1 }
  ],
  "total_count": 3,
  "query": "food"
}
```

`top_k=3` honored (out of 4 total entities); pasta ranks first as expected.

## Request C — text search on empty graph (earlier container)

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/4aRp0J0BQ4F7Y_V3tOLD/memories/graph/_search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"alice","search_type":"text","top_k":5}'
```

### Response (HTTP 200)

```json
{ "total_count": 0, "query": "alice" }
```

## Bug uncovered during this test — already fixed

### Bug 4.1 — `MLGraphSearchRequest.parse` did not advance parser

**Before fix:** any POST to `/memories/graph/_search` failed with HTTP 400:

```
"Failed to parse object: expecting token of type [START_OBJECT] but found [null]"
```

**Cause:** `MLGraphSearchRequest.parse` called `ensureExpectedToken(START_OBJECT, parser.currentToken(), parser)` against a fresh `XContentParser`, whose `currentToken()` is `null`. Other memory REST handlers call `parser.nextToken()` before parsing.

**Fix:**

```java
public static MLGraphSearchRequest parse(XContentParser parser, String tenantId) throws IOException {
    if (parser.currentToken() == null) {
        parser.nextToken();
    }
    ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
    ...
}
```

## Environmental note — JVM crash from KNN+DJL libstdc++ mismatch (resolved)

An earlier run crashed the JVM on first graph-nodes flush:

```
UnsatisfiedLinkError: libopensearchknn_faiss_avx512.so: …ml_cache/pytorch/…/libstdc++.so.6: version `GLIBCXX_3.4.20' not found
```

**Not a graph-feature bug** — DJL's bundled libstdc++ (max `GLIBCXX_3.4.19`) was pre-empting the system libstdc++ after the MiniLM model was loaded. Mitigated by starting OpenSearch with `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6`. All test requests above ran cleanly after the preload fix.
