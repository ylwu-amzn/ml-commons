# Integration Proposal: OpenSearch Graph Plugin for Memory Graph

> Generated: 2026-04-13
> Prerequisite: [memory-gap-analysis-vs-mem0.md](./memory-gap-analysis-vs-mem0.md)
> Graph plugin source: `/Users/ylwu/code/others/xinl/opensearch/graph`

---

## 1. Executive Summary

**Yes, the OpenSearch Graph plugin is an excellent fit to close the P0 gap.** It provides everything mem0 gets from Neo4j -- and more -- as a native OpenSearch plugin, with zero external dependencies. The LPG (Labeled Property Graph) backend maps almost 1:1 to mem0's graph memory model, and the plugin adds capabilities mem0 doesn't have: Cypher read/write, ACID transactions, hybrid retrieval (vector+text+graph fusion), variable-length path traversal, and distributed execution.

**Key advantage over mem0's approach**: mem0 requires users to deploy and manage a separate Neo4j instance. With the Graph plugin, everything stays inside the OpenSearch cluster -- same security model, same access control, same operational tooling.

---

## 2. Capability Mapping: mem0 Graph Memory -> OpenSearch Graph Plugin

| mem0 Feature | mem0 Implementation | Graph Plugin Equivalent | Match |
|-------------|---------------------|------------------------|-------|
| **Node storage** | Neo4j nodes with `__Entity__` label | `lpg-nodes` index with labels array | Direct |
| **Edge storage** | Neo4j relationships with type | `lpg-edges` index with type, source, target | Direct |
| **Node properties** | Neo4j node properties (name, user_id, etc.) | `properties` map on lpg-nodes | Direct |
| **Edge properties** | Neo4j edge properties (valid, created_at, mentions) | `properties` map on lpg-edges | Direct |
| **Node embeddings** | `vector.similarity.cosine()` on Neo4j | `knn_vector` field on lpg-nodes (HNSW/FAISS/Lucene) | Direct (better) |
| **Cosine similarity search** | Cypher `vector.similarity.cosine()` | Native KNN search on embedding field | Direct (faster) |
| **MERGE semantics** | Cypher `MERGE ... ON CREATE SET ... ON MATCH SET` | Cypher `MERGE` via Graph plugin Cypher engine | Direct |
| **Soft-delete** | `SET r.valid = false` | `SET r.valid = false` via Cypher write | Direct |
| **Bidirectional traversal** | Two UNION Cypher queries | `MATCH (a)-[r]-(b)` undirected pattern | Direct (simpler) |
| **Variable-length paths** | Not used in mem0 | `(a)-[:knows*1..3]->(b)` supported | **Upgrade** |
| **Scope filtering** | WHERE clauses on user_id/agent_id/run_id | Same WHERE clauses in Cypher | Direct |
| **BM25 reranking** | Python `rank_bm25.BM25Okapi` on results | Full-text search on `properties.*` fields | Direct |
| **Entity type labels** | Stored as node property or dynamic label | Multiple labels per node (`labels` array) | Direct (better) |
| **Mention counting** | `SET n.mentions = coalesce(n.mentions, 0) + 1` | Same Cypher SET expression | Direct |
| **Transaction support** | Not used (single-statement) | Full ACID with BEGIN/COMMIT/ROLLBACK | **Upgrade** |
| **Hybrid retrieval** | Not available | `_plugins/_retrieval` endpoint (vector+text+graph) | **Upgrade** |
| **Graph expansion** | Not available | `BoundedGraphExpander` with configurable hops/frontier | **Upgrade** |
| **Temporal queries** | Soft-delete only | `valid_at` parameter for temporal filtering | **Upgrade** |
| **Distributed execution** | Single Neo4j instance | Distributed across OpenSearch cluster | **Upgrade** |

**Match summary**: 14 Direct matches, 5 Upgrades (capabilities mem0 doesn't have), 0 Gaps.

---

## 3. Architecture Design

### 3.1 Data Model

Map mem0's entity-relationship model to the Graph plugin's LPG format:

**Entity Node (lpg-nodes)**:
```json
{
  "id": "entity_alice",
  "labels": ["__Entity__", "person"],
  "properties": {
    "name": "alice",
    "user_id": "user_123",
    "agent_id": "agent_456",
    "container_id": "mem_container_789",
    "mentions": "3",
    "created_at": "2026-04-13T10:00:00Z",
    "updated_at": "2026-04-13T14:30:00Z"
  },
  "embedding": [0.12, -0.45, 0.78, ...]
}
```

**Relationship Edge (lpg-edges)**:
```json
{
  "id": "rel_alice_works_at_opensearch",
  "type": "WORKS_AT",
  "source": "entity_alice",
  "target": "entity_opensearch",
  "properties": {
    "valid": "true",
    "created_at": "2026-04-13T10:00:00Z",
    "updated_at": "2026-04-13T10:00:00Z",
    "invalidated_at": "",
    "mentions": "2",
    "container_id": "mem_container_789"
  }
}
```

### 3.2 Database Per Memory Container

Each ML Commons `MemoryContainer` with graph enabled would create a dedicated Graph database:

```
Database name: "memory-graph-{container_id}"
Indices created:
  - memory-graph-{container_id}-lpg-nodes   (with knn_vector embedding)
  - memory-graph-{container_id}-lpg-edges
```

This provides:
- Namespace isolation per memory container (no WHERE filtering needed for container scope)
- Independent index lifecycle (delete container = delete graph)
- Configurable sharding per container

### 3.3 Integration Points in ML Commons

```
MemoryProcessingService (existing)
  ├── extractFactsFromConversation()          (existing - fact extraction)
  ├── makeMemoryDecisions()                   (existing - ADD/UPDATE/DELETE)
  ├── extractEntitiesFromConversation()       (NEW - entity + type extraction)
  ├── extractRelationshipsFromEntities()      (NEW - relationship extraction)
  └── makeGraphDecisions()                    (NEW - graph conflict resolution)
         ↓
MemoryGraphService (NEW)
  ├── addEntitiesAndRelationships()           (Cypher MERGE via Graph plugin)
  ├── softDeleteRelationships()               (Cypher SET valid=false)
  ├── searchByEntitySimilarity()              (KNN on node embeddings)
  ├── getRelationshipsForEntity()             (Cypher MATCH traversal)
  └── hybridGraphSearch()                     (Graph plugin _retrieval API)
         ↓
Graph Plugin (external)
  ├── POST _plugins/_cypher                   (write operations)
  ├── POST _plugins/_retrieval                (hybrid search)
  └── POST _plugins/_graph/database           (lifecycle)
```

### 3.4 Memory Container Configuration Extension

Add graph configuration to `MemoryConfiguration`:

```java
// New fields in MemoryConfiguration
private Boolean enableGraph;                    // default: false
private String graphEmbeddingModelId;           // can reuse embeddingModelId
private Integer graphEmbeddingDimension;        // can reuse embeddingDimension
private String graphLlmId;                      // can reuse llmId (for entity extraction)
private Double graphSimilarityThreshold;        // default: 0.7
private Integer graphMaxTraversalHops;          // default: 3
private Integer graphMaxFrontierPerHop;         // default: 100
private Map<String, Object> graphSettings;      // custom graph index settings
```

### 3.5 New REST APIs

```
# Graph Memory Operations (under existing memory container path)
POST   /_plugins/_ml/memory_containers/{id}/memories/graph/_search
  → Search graph by text query (entity extraction + embedding similarity + graph expansion)

POST   /_plugins/_ml/memory_containers/{id}/memories/graph/_hybrid_search
  → Hybrid search combining vector + text + graph (delegates to _plugins/_retrieval)

GET    /_plugins/_ml/memory_containers/{id}/memories/graph/entities
  → List all entities (with optional filtering by label, user_id, etc.)

GET    /_plugins/_ml/memory_containers/{id}/memories/graph/relationships
  → List all relationships (with optional filtering by type, valid status)

DELETE /_plugins/_ml/memory_containers/{id}/memories/graph
  → Delete all graph data for this container
```

---

## 4. Processing Pipeline

### 4.1 Write Path (When Messages Are Added)

Augment the existing `TransportAddMemoriesAction` pipeline:

```
User adds messages to memory container
         ↓
1. Store messages in working memory (EXISTING)
         ↓
2. Extract facts via LLM (EXISTING - MemoryProcessingService)
         ↓
3. Make memory decisions + store in long-term (EXISTING)
         ↓
4. [NEW] Extract entities from messages via LLM
   Prompt: "Extract all entities and their types from this conversation"
   Output: [{entity: "alice", type: "person"}, {entity: "opensearch", type: "technology"}]
         ↓
5. [NEW] Extract relationships from entities + messages via LLM
   Prompt: "Given these entities and conversation, establish relationships"
   Output: [{source: "alice", relationship: "WORKS_ON", destination: "opensearch"}]
         ↓
6. [NEW] Search existing graph for conflicting relationships
   Query: Cypher MATCH for existing relationships involving extracted entities
         ↓
7. [NEW] LLM decides which existing relationships to soft-delete
         ↓
8. [NEW] Execute graph mutations via Cypher
   - MERGE entities (with embeddings)
   - MERGE new relationships
   - SET valid=false on conflicting relationships
```

### 4.2 Read Path (Search)

```
User searches graph memory
         ↓
Option A: Entity-based search
  1. Extract entities from query via LLM
  2. Embed entity names
  3. KNN search on lpg-nodes to find similar entities
  4. Cypher traversal to expand relationships
  5. Return entity-relationship results
         ↓
Option B: Hybrid retrieval (recommended)
  1. Call Graph plugin _plugins/_retrieval endpoint
     - query_text: user's search query
     - query_vector: pre-computed embedding of query
     - database: "memory-graph-{container_id}"
     - hops: 2 (configurable)
     - node_labels: ["__Entity__"]
     - edge_types: [] (all types)
     - weights: {vector: 0.5, text: 0.3, graph: 0.2}
  2. Graph plugin handles:
     - Vector seed retrieval (KNN on node embeddings)
     - Lexical seed retrieval (BM25 on properties.name)
     - Bounded graph expansion (traverse N hops from seeds)
     - Weighted fusion of all signals
  3. Return ranked results
```

### 4.3 Cypher Query Templates

**MERGE entity with embedding**:
```cypher
MERGE (n:__Entity__ {id: $entityId})
ON CREATE SET
  n.name = $name,
  n.labels = $labels,
  n.user_id = $userId,
  n.container_id = $containerId,
  n.mentions = 1,
  n.created_at = $now
ON MATCH SET
  n.mentions = n.mentions + 1,
  n.updated_at = $now
RETURN n.id, n.name
```

Note: Embeddings set separately via index update (Graph plugin stores them in `knn_vector` field).

**MERGE relationship**:
```cypher
MATCH (src:__Entity__ {id: $sourceId})
MATCH (dst:__Entity__ {id: $destId})
MERGE (src)-[r:$relType]->(dst)
ON CREATE SET
  r.valid = true,
  r.created_at = $now,
  r.updated_at = $now,
  r.mentions = 1,
  r.container_id = $containerId
ON MATCH SET
  r.valid = true,
  r.mentions = r.mentions + 1,
  r.updated_at = $now,
  r.invalidated_at = null
RETURN src.name, type(r), dst.name
```

**Soft-delete relationship**:
```cypher
MATCH (src:__Entity__ {name: $sourceName, user_id: $userId})
-[r:$relType]->
(dst:__Entity__ {name: $destName, user_id: $userId})
WHERE r.valid = true
SET r.valid = false, r.invalidated_at = $now
RETURN src.name, type(r), dst.name
```

**Search entity neighbors**:
```cypher
MATCH (n:__Entity__ {name: $entityName, user_id: $userId})-[r]-(m:__Entity__)
WHERE r.valid = true OR r.valid IS NULL
RETURN n.name AS source, type(r) AS relationship, m.name AS target, r.mentions AS weight
ORDER BY r.mentions DESC
LIMIT $limit
```

**Multi-hop traversal**:
```cypher
MATCH path = (start:__Entity__ {name: $startEntity, user_id: $userId})
  -[:*1..3]->
  (end:__Entity__)
WHERE ALL(r IN relationships(path) WHERE r.valid = true OR r.valid IS NULL)
RETURN [n IN nodes(path) | n.name] AS entities,
       [r IN relationships(path) | type(r)] AS relationships
LIMIT $limit
```

---

## 5. Comparison: Graph Plugin vs mem0's Neo4j Approach

| Dimension | mem0 + Neo4j | ML Commons + Graph Plugin |
|-----------|-------------|--------------------------|
| **Deployment** | Separate Neo4j instance needed | Same OpenSearch cluster (no new infra) |
| **Security** | Separate auth system | Unified OpenSearch security (TLS, RBAC, tenant isolation) |
| **Multi-tenancy** | Manual user_id filtering | Native tenant isolation + container-level database isolation |
| **Vector search** | Neo4j's native vector (limited) | OpenSearch native KNN (HNSW/FAISS/Lucene, battle-tested) |
| **Hybrid search** | Not available (vector OR graph, not both) | `_plugins/_retrieval` fuses vector+text+graph in one call |
| **Transaction support** | Single-statement only | Full ACID (BEGIN/COMMIT/ROLLBACK) |
| **Path traversal** | Not used in mem0 (1-hop only) | Variable-length paths with bounded depth |
| **Query language** | Raw Cypher strings constructed in Python | Native Cypher endpoint with query optimization (Calcite) |
| **Scalability** | Single Neo4j instance | Distributed across OpenSearch cluster shards |
| **Operations** | Two systems to monitor/backup/upgrade | Single system |
| **Cost** | Neo4j license (Enterprise features) | Included with OpenSearch |
| **Graph expansion** | Manual BFS in Python | Native `BoundedGraphExpander` with configurable frontier |
| **Temporal queries** | Manual via `valid_at` property | Native `valid_at` parameter support |

**The Graph plugin approach is strictly superior in every dimension.**

---

## 6. Implementation Plan

### Phase 1: Foundation (Week 1-2)

1. **MemoryConfiguration extension** - Add graph config fields (`enableGraph`, `graphSimilarityThreshold`, etc.)
2. **Graph database lifecycle** - Create/delete graph database when container is created/deleted with `enableGraph=true`
3. **MemoryGraphService** - New service class with basic Cypher execution via Graph plugin transport actions
4. **Entity extraction prompt + LLM call** - Reuse `MemoryProcessingService` pattern for entity extraction

### Phase 2: Write Path (Week 2-3)

5. **Entity extraction pipeline** - LLM extracts entities with types from conversation messages
6. **Relationship extraction pipeline** - LLM establishes relationships between extracted entities
7. **Graph write operations** - MERGE entities, MERGE relationships, soft-delete conflicts
8. **Embedding computation** - Use configured embedding model to embed entity names, store on nodes
9. **Integration with TransportAddMemoriesAction** - Hook graph processing after fact extraction

### Phase 3: Read Path (Week 3-4)

10. **Graph search transport action** - `TransportGraphSearchMemoriesAction`
11. **Entity-based search** - KNN on entity embeddings + Cypher traversal
12. **Hybrid retrieval integration** - Delegate to Graph plugin `_plugins/_retrieval` endpoint
13. **REST API endpoints** - Expose graph search, entity listing, relationship listing

### Phase 4: Polish (Week 4-5)

14. **Conflict resolution** - LLM-driven graph conflict detection and soft-delete decisions
15. **Mention tracking** - Increment counters on entity/relationship access
16. **Testing** - Unit tests, integration tests with Graph plugin
17. **Documentation** - API docs, configuration guide

### Estimated Total: 4-5 weeks (down from 4-6 weeks in original estimate, because Graph plugin eliminates the need to build storage/query layers from scratch)

---

## 7. LLM Prompts for Graph Operations

### 7.1 Entity Extraction Prompt (New)

```
<ROLE>You are an entity extraction agent. Extract all named entities and their semantic types from the conversation.</ROLE>

<SCOPE>
- Extract entities from both USER and ASSISTANT messages
- Include: people, organizations, products, technologies, locations, concepts, events
- Map self-references ("I", "me", "my") to the provided user identifier
</SCOPE>

<RULES>
- Normalize entity names to lowercase
- Replace spaces with underscores
- Classify each entity with a semantic type
- Do NOT extract generic/vague entities ("things", "stuff", "it")
</RULES>

<OUTPUT>
Return ONLY a JSON object: {"entities": [{"entity": "name", "entity_type": "type"}]}
</OUTPUT>
```

### 7.2 Relationship Extraction Prompt (New)

```
<ROLE>You are a relationship extraction agent. Given entities and conversation text, establish meaningful relationships between entities.</ROLE>

<RULES>
- Use consistent, timeless relationship names (e.g., "WORKS_AT" not "STARTED_WORKING_AT")
- Relationships are directed: source -> relationship -> destination
- Each relationship should be a single, clear semantic connection
- Do NOT create relationships between entities that have no meaningful connection in the text
</RULES>

<OUTPUT>
Return ONLY a JSON object: {"relationships": [{"source": "entity_a", "relationship": "REL_TYPE", "destination": "entity_b"}]}
</OUTPUT>
```

### 7.3 Graph Conflict Resolution Prompt (New)

```
<ROLE>You are a graph memory conflict resolver. Compare new relationships against existing ones and decide which old relationships should be invalidated.</ROLE>

<RULES>
- Do NOT delete if the same relationship type exists with DIFFERENT destinations (they can coexist)
  Example: "alice LIKES pizza" and "alice LIKES burger" should BOTH exist
- DELETE only when new information CONTRADICTS existing relationships
  Example: "alice WORKS_AT google" should invalidate "alice WORKS_AT amazon" (if it means she switched)
- When unsure, preserve existing relationships
</RULES>

<OUTPUT>
Return ONLY a JSON object: {"deletions": [{"source": "entity_a", "relationship": "REL_TYPE", "destination": "entity_b"}]}
If nothing to delete, return: {"deletions": []}
</OUTPUT>
```

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Graph plugin not released when ML Commons ships | Medium | High | Feature-flag graph memory (`enableGraph`); core memory works without it |
| Cypher write performance under high concurrency | Low | Medium | Graph plugin has transaction support + thread pools; batch writes when possible |
| LLM entity extraction quality | Medium | Medium | Allow custom prompts; validate extraction results; graceful fallback to no-graph |
| Graph size growth unbounded | Medium | Medium | Configurable max entities per container; periodic consolidation; TTL on old entities |
| Cross-plugin dependency complexity | Low | Low | Communication via OpenSearch transport actions (standard inter-plugin pattern) |

---

## 9. What the Graph Plugin Gives Us Beyond mem0

These are features we get "for free" that mem0 doesn't have:

1. **Hybrid Retrieval Endpoint** (`_plugins/_retrieval`) - Single API call that fuses vector similarity + lexical search + graph expansion with configurable weights. mem0 does vector OR graph, never both in one query.

2. **Multi-hop Path Traversal** - `(a)-[:KNOWS*1..5]->(b)` finds connections up to 5 hops away. mem0 only does 1-hop neighbor lookups.

3. **ACID Transactions** - Multi-statement transactions with BEGIN/COMMIT/ROLLBACK. mem0's Neo4j integration is single-statement only.

4. **Query Optimization** - Apache Calcite-based cost optimizer with filter pushdown, join reordering, cardinality estimation. mem0 runs raw Cypher with no optimization.

5. **Temporal Queries** - `valid_at` parameter for point-in-time graph queries. mem0 only has soft-delete, no temporal query semantics.

6. **Async Queries** - Long-running graph queries can be submitted async with cursor-based pagination. Essential for large graph traversals.

7. **SPARQL** - Optional RDF/semantic web support for more expressive knowledge representation. Could enable ontology-based memory organization in the future.

8. **Circuit Breaker** - Memory-bounded query execution prevents graph queries from OOM-killing the cluster. mem0 has no memory protection.

9. **Distributed Execution** - Graph queries execute across cluster shards, not limited to single-node memory. Essential for production-scale memory graphs.

10. **Fine-Grained Authorization** - Layer-2 element-level access control on graph nodes/edges. Could enable per-user entity visibility within shared memory containers.

---

## 10. Conclusion

The OpenSearch Graph plugin is the ideal backend for ML Commons memory graph. It provides a **100% feature match** with mem0's Neo4j-based graph memory, plus significant upgrades in hybrid retrieval, multi-hop traversal, transactions, temporal queries, and distributed execution. The integration is natural since both are OpenSearch plugins communicating via transport actions.

The recommended approach:
1. Use the LPG (Labeled Property Graph) backend exclusively (not RDF)
2. Create one graph database per memory container
3. Leverage the `_plugins/_retrieval` endpoint for hybrid search (the most unique advantage)
4. Keep graph memory behind a feature flag until the Graph plugin is released
5. Reuse existing `MemoryProcessingService` patterns for LLM-based entity/relationship extraction

This reduces the estimated implementation effort from 4-6 weeks to **4-5 weeks**, primarily because we skip building the entire storage/query layer and get enterprise features (transactions, security, distribution) for free.
