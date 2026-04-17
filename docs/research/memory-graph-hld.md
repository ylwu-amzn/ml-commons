# Memory Graph: High-Level Design

> Date: 2026-04-13 (v2 -- incorporates review feedback)
> Status: Draft
> Branch: `feature/memory_graph`
> References: [Gap Analysis vs mem0](./memory-gap-analysis-vs-mem0.md), [Graph Plugin Integration Proposal](./graph-plugin-integration-proposal.md)

---

## 1. Problem Statement

ML Commons agentic memory stores facts as flat text entries in a vector index. This works well for semantic recall ("What does the user prefer?") but cannot answer **relationship questions** ("Who does Alice work with?", "What tools are used by Project X?", "How are these teams connected?").

mem0 solves this with a **knowledge graph layer** (backed by Neo4j) that extracts entities and relationships from conversations. ML Commons currently has no equivalent.

The unreleased **OpenSearch Graph plugin** provides native graph storage (LPG), Cypher queries, vector embeddings on graph nodes, hybrid retrieval, and a purpose-built **memory-aware index schema** (`LpgMemoryIndexSchema`) -- all within the same OpenSearch cluster, with zero external dependencies.

**Goal**: Add a graph memory layer to ML Commons that extracts entities and relationships from conversations and enables structured relationship reasoning, using the Graph plugin's memory-aware schemas.

---

## 2. Architecture Options Analysis

### 2.1 How Should ML Commons Interact with the Graph Plugin?

#### Option A: Direct Index Operations + REST Client (Selected)

ML Commons writes to `lpg-nodes`/`lpg-edges` indices using standard OpenSearch `IndexRequest`/`UpdateRequest`, following the `LpgMemoryIndexSchema` mapping conventions. For complex queries (multi-hop traversal, hybrid retrieval), it calls the Graph plugin's REST endpoints via internal HTTP client.

| Pros | Cons |
|------|------|
| No compile-time dependency on graph plugin JAR | Must replicate index mapping logic from `LpgMemoryIndexSchema` |
| Graceful degradation if graph plugin not installed | Two communication styles (index API + REST) |
| Leverages existing OpenSearch client patterns | Must track schema changes across plugin versions |
| Write path has minimal overhead (direct index) | No Cypher for writes (raw index operations) |

**Schema drift mitigation**: Store `schema_version` on every document. Validate index mappings at container creation against expected field set. Log warnings if mappings diverge.

#### Option B: Library Dependency on Graph Plugin

Add `graph-sparql` module as compile dependency. Use `LpgStoreWriter`/`LpgStoreReader` directly.

| Pros | Cons |
|------|------|
| Type-safe API, guaranteed schema alignment | Hard JAR dependency -- breaks if graph plugin absent |
| Full Cypher capability for both reads and writes | Version coupling between ML Commons and Graph plugin |
| `LpgMemoryIndexSchema` used directly | Cannot ship ML Commons without graph plugin JAR |

#### Option C: Enhanced Transport Actions

Collaborate with Graph plugin team to make transport actions functional (currently stubs for security authorization).

| Pros | Cons |
|------|------|
| Clean API boundary, standard plugin pattern | Requires cross-team changes first (blocking) |
| Async-native via ActionListener | String-based query interface (no type safety) |
| Security intercepted automatically | Cannot start until graph team delivers |

**Decision**: Option A. No dependency, graceful degradation, can start immediately.

---

### 2.2 Where Should Graph Processing Fit in the Pipeline?

Currently, when messages are added:
```
index working memory --> return response --> [async] extract facts --> search similar --> LLM decisions --> write
```

#### Option A: Parallel with Fact Extraction (Selected)

```
                         [async thread pool]
                        ┌──────────────────────┐
                        │                      │
                        ▼                      ▼
              Existing Pipeline         Graph Pipeline
              ─────────────────         ──────────────
              Extract facts (LLM)       Extract entities (LLM)
              Search similar            Extract relationships (LLM)
              LLM decisions             Dedup entities
              Write long-term           Resolve conflicts
              Write history             Write nodes + edges
```

| Pros | Cons |
|------|------|
| Faster end-to-end (parallel LLM calls) | Cannot reuse extracted facts for entity extraction |
| Graph failure does not block fact storage | Up to 4 LLM calls in graph pipeline (see cost analysis below) |
| Independent evolution of both pipelines | More complex error handling (two independent paths) |

#### Option B: Sequential After Fact Extraction

```
              Extract facts --> LLM decisions --> Write long-term
                                                        │
                                                        ▼
              Extract entities from facts --> Dedup --> Write graph
```

| Pros | Cons |
|------|------|
| Can reuse extracted facts as entity extraction input | Adds latency (sequential) |
| Fewer LLM calls (facts feed entity extraction) | Graph blocked by fact extraction failures |
| Simpler error handling | Tighter coupling between pipelines |

#### Option C: Single Combined LLM Call

One prompt extracts facts + entities + relationships together.

| Pros | Cons |
|------|------|
| Most efficient (1 LLM call instead of 3+) | Complex prompt, harder to debug |
| Lowest latency and cost | Breaks existing prompt design |
| | Cannot disable graph without affecting facts |

**Decision**: Option A (Parallel). Fault isolation is critical -- graph failures must never break the existing memory pipeline.

**LLM Cost Analysis (per addMemories request)**:

| Pipeline | LLM Calls | Estimated Latency | Estimated Cost |
|----------|-----------|-------------------|----------------|
| Existing (facts) | 2 (extraction + decisions) | 2-5s | $0.02-0.05 |
| Graph (entities + relationships) | 2 (extraction + relationships) | 2-5s | $0.02-0.05 |
| Graph (dedup, conditional) | 0-1 (only for ambiguous entities) | 0-3s | $0-0.03 |
| Graph (conflict resolution, conditional) | 0-1 (only if conflicts found) | 0-3s | $0-0.03 |
| **Total** | **4-6** | **4-10s** (parallel) | **$0.04-0.16** |

**Backpressure**: When LLM service is overloaded, graph pipeline should be shed first (existing fact pipeline has priority). Implement via a configurable circuit breaker on graph LLM calls.

---

### 2.3 How Should Graph Databases Be Scoped?

#### Option A: One Graph DB Per Memory Container (Selected)

```
Container A (indexPrefix="proj-a")
  ──→ mlgraph-proj-a
        ├── mlgraph-proj-a-lpg-nodes
        └── mlgraph-proj-a-lpg-edges

Container B (indexPrefix="proj-b")
  ──→ mlgraph-proj-b
        ├── mlgraph-proj-b-lpg-nodes
        └── mlgraph-proj-b-lpg-edges
```

| Pros | Cons |
|------|------|
| Clean isolation (no cross-container leakage) | Index proliferation with many containers |
| Simple lifecycle (delete container = delete indices) | Cannot link entities across containers |
| No namespace filtering needed in queries | |

#### Option B: Shared Graph with Namespace Filtering

Single index pair shared across containers, filtered by `container_id` property.

| Pros | Cons |
|------|------|
| Fewer indices to manage | Every query needs container_id filter |
| Could enable cross-container entity linking | Deletion requires delete-by-query |
| | Cross-tenant risk if filter missing |

#### Option C: One Graph DB Per Tenant

Graph shared across a tenant's containers, isolated between tenants.

| Pros | Cons |
|------|------|
| Entity reuse across conversations | Complex lifecycle management |
| Natural multi-tenant isolation | Selective cleanup on container delete |

**Decision**: Option A with mitigations:
- **Lazy index creation**: Graph indices created only when the first entity is extracted, not at container creation. Avoids empty indices.
- **Index naming**: Derive graph database name from the container's existing `indexPrefix` field (not containerId). Use `mlgraph-{indexPrefix}` as the database name, which produces indices `mlgraph-{indexPrefix}-lpg-nodes` and `mlgraph-{indexPrefix}-lpg-edges`. The `indexPrefix` defaults to `"default"` or a random UUID -- both pass `LpgIndexSchema.validateDatabaseName()` (lowercase alphanumeric + hyphens/underscores, max 200 chars, no dots). Note: system index prefix `.plugins-ml-am-` contains dots which are NOT valid in graph database names, so graph indices cannot use system index naming -- they use plain index names.
- **Constraints index**: Not created (only needed for Cypher `CREATE CONSTRAINT` operations, which ML Commons does not use).
- **_meta marker**: ML Commons must set `_meta.memory_managed: true` in the index mapping when creating graph indices directly (since it's not going through the Graph plugin's database creation API). This enables the Graph plugin to recognize and apply scope enforcement when it is present.

---

### 2.4 How Should Entity Deduplication Work?

#### Option A: Embedding Similarity Only

Embed entity name, KNN search, merge if similarity > 0.7.

| Pros | Cons |
|------|------|
| No extra LLM call (fast, cheap) | Single threshold = false merges or false splits |
| Proven by mem0 (~80-85% accuracy) | Cannot leverage conversation context |
| Deterministic | Short names may have noisy embeddings |

#### Option B: LLM-Based Entity Resolution

Send all extracted + existing entity names to LLM.

| Pros | Cons |
|------|------|
| Most context-aware (~90-95% accuracy) | Extra LLM call per request (1-3s) |
| Handles aliases, nicknames | Entity list grows with graph size (context limit) |
| | Non-deterministic, may hallucinate |

#### Option C: Normalize + Embedding Fallback

String normalization first, embedding similarity if no exact match.

| Pros | Cons |
|------|------|
| Fast for obvious cases | Two stages, more code complexity |
| No LLM cost (~85-90% accuracy) | Normalization rules need maintenance |

#### Option D: Embedding + LLM Verification (Selected)

Embedding similarity as first pass. LLM only for ambiguous cases (configurable thresholds).

| Pros | Cons |
|------|------|
| Best accuracy (~95%+) | Most complex implementation |
| LLM only for ambiguous cases (controlled cost) | Two thresholds to tune per embedding model |
| Clear matches and non-matches skip LLM | Unpredictable LLM cost in ambiguous-heavy data |

**Decision**: Option D.

```
For each extracted entity:
  ┌─────────────────────────────────┐
  │  Normalize name                 │
  │  Compute embedding              │
  │  KNN search existing nodes      │
  └───────────────┬─────────────────┘
                  │
           ┌──────┴──────┐
           │ Top score?  │
           └──────┬──────┘
       ┌──────────┼──────────┐
       ▼          ▼          ▼
    > 0.8      0.6-0.8     < 0.6
       │          │          │
       ▼          ▼          ▼
  Auto-merge   LLM verify  Create new
  (increment   (same        (new node
   mentions)    entity?)     with embedding)
```

**Threshold tuning note**: Defaults (0.6/0.8) assume a general-purpose embedding model (e.g., 384-dim text-embedding). Domain-specific or smaller models may need adjustment. Recommend a calibration step during initial deployment.

---

## 3. System Architecture

### 3.1 Overall Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      ML Commons Plugin                          │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    REST API Layer                          │  │
│  │  POST /memories   POST /graph/_search  GET /graph/entities│  │
│  │                   POST /graph/_hybrid_search               │  │
│  │                   GET/PUT/DELETE /graph/entities/{id}       │  │
│  │                   GET/DELETE /graph/relationships/{id}      │  │
│  └──────────┬──────────────────────┬─────────────────────────┘  │
│             │                      │                             │
│  ┌──────────▼────────────┐  ┌──────▼─────────────────────────┐  │
│  │  Memory Write         │  │  Graph Search Service           │  │
│  │  Pipeline             │  │                                 │  │
│  │                       │  │  Mode A: KNN on lpg-nodes       │  │
│  │  ┌─────────────────┐  │  │          + edge expansion       │  │
│  │  │ Fact Extraction  │  │  │          (always available)     │  │
│  │  │ (existing)       │  │  │                                 │  │
│  │  └─────────────────┘  │  │  Mode B: Hybrid retrieval ──────┼──┼──→ Graph Plugin
│  │  ┌─────────────────┐  │  │          (REST client)          │  │   _plugins/_retrieval
│  │  │ Graph Extraction │  │  │                                 │  │
│  │  │ (new, parallel)  │  │  │  Mode C: Cypher traversal ─────┼──┼──→ Graph Plugin
│  │  │                  │  │  │          (REST client)          │  │   _plugins/_cypher
│  │  │ • Entity LLM     │  │  └─────────────────────────────────┘  │
│  │  │ • Relationship   │  │                                       │
│  │  │   LLM            │  │                                       │
│  │  │ • Entity Dedup   │  │                                       │
│  │  │ • Conflict Res.  │  │                                       │
│  │  └────────┬────────┘  │                                       │
│  └───────────┼───────────┘                                       │
│              │ Direct Index Operations                            │
└──────────────┼────────────────────────────────────────────────────┘
               ▼
┌───────────────────────────────────────────────────────────────────┐
│                     OpenSearch Cluster                              │
│                                                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ .plugins-ml-am-  │  │ mlgraph-{prefix} │  │ mlgraph-{prefix}│  │
│  │ {prefix}-memory-*│  │ -lpg-nodes       │  │ -lpg-edges      │  │
│  │ (existing:       │  │                  │  │                  │  │
│  │  -memory-working │  │ _meta:           │  │ _meta:           │  │
│  │  -memory-long-   │  │  memory_managed: │  │  memory_managed: │  │
│  │  term            │  │  true            │  │  true            │  │
│  │  -memory-sessions│  │                  │  │                  │  │
│  │  -memory-history)│  │                  │  │                  │  │
│  │                  │  │ Top-level fields: │  │ Top-level fields:│  │
│  │                  │  │  id, labels,      │  │  id, type,       │  │
│  │                  │  │  tenant_id,       │  │  source, target, │  │
│  │                  │  │  user_id,         │  │  tenant_id,      │  │
│  │                  │  │  agent_id,        │  │  user_id,        │  │
│  │                  │  │  session_id,      │  │  namespace,      │  │
│  │                  │  │  namespace,       │  │  confidence,     │  │
│  │                  │  │  confidence,      │  │  tombstone,      │  │
│  │                  │  │  tombstone,       │  │  valid_from,     │  │
│  │                  │  │  embedding (knn)  │  │  valid_to, ...   │  │
│  │                  │  │  properties {}    │  │  properties {}   │  │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘  │
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              Graph Plugin (optional)                         │    │
│  │  • Recognizes memory-managed indices via _meta marker       │    │
│  │  • Cypher engine (read/write) with scope enforcement        │    │
│  │  • Hybrid retrieval (vector + text + graph fusion)          │    │
│  │  • Multi-hop traversal with bounded expansion               │    │
│  └────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────┘
```

### 3.2 Write Path

```
Messages arrive at POST /_plugins/_ml/memory_containers/{id}/memories
              │
              ▼
    TransportAddMemoriesAction.processAndIndexMemory()
    ┌─────────────────────┐
    │ Index working memory │ ──→ Return MLAddMemoriesResponse
    └─────────┬───────────┘      (sessionId + workingMemoryId)
              │
              ▼ threadPool.executor("opensearch_ml_agentic_memory").execute(...)
                [fire-and-forget, line 249 of TransportAddMemoriesAction]
    ┌─────────┴──────────┐
    │                    │
    ▼                    ▼
┌─────────────┐    ┌─────────────────────────────────┐
│  EXISTING   │    │  GRAPH PIPELINE (NEW)            │
│  PIPELINE   │    │  (skipped if enable_graph=false  │
│             │    │   or circuit breaker open)       │
│ extractLong │    │                                  │
│ TermMemory()│    │  1. Extract entities +           │
│             │    │     relationships (LLM)          │
│ 1. For each │    │                                  │
│    strategy:│    │  2. For each entity:             │
│    extract  │    │     a. Normalize + embed          │
│    facts    │    │     b. KNN search lpg-nodes      │
│    (LLM)    │    │     c. Dedup decision             │
│             │    │        (auto-merge / LLM / new)  │
│ 2. Search   │    │                                  │
│    similar  │    │  3. For each relationship:       │
│    (KNN,    │    │     a. Search existing edges      │
│    sequen-  │    │     b. Conflict resolution (LLM)  │
│    tial per │    │                                  │
│    fact)    │    │  4. Bulk write (single request):  │
│             │    │     - Index new nodes             │
│ 3. LLM     │    │     - Update merged nodes          │
│    memory   │    │     - Index new edges              │
│    decisions│    │     - Update tombstoned edges      │
│             │    │                                  │
│ 4. Bulk     │    │  On failure: log + skip           │
│    write    │    │  (no retry, no dead-letter)       │
│    long-term│    └─────────────────────────────────┘
│             │
│ 5. Write    │
│    history  │
└─────────────┘
```

### 3.3 Read Path

```
Search request arrives
         │
         ▼
  ┌────────────────────────────────────────────┐
  │ Extract entities from query (LLM)          │
  │ (or skip if query is a direct entity name) │
  └──────────────────┬─────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
  ┌──────────────────┐   ┌────────────────────────────┐
  │ Mode A:          │   │ Mode B:                    │
  │ Entity-based     │   │ Hybrid Retrieval           │
  │ (always works)   │   │ (requires Graph plugin)    │
  │                  │   │                            │
  │ 1. Embed query   │   │ POST _plugins/_retrieval   │
  │    entities      │   │ {                          │
  │ 2. KNN search    │   │   query_text, query_vector,│
  │    lpg-nodes     │   │   database: "mlgraph-{prefix}",│
  │ 3. Expand edges  │   │   hops: 2,                 │
  │    (both dirs)   │   │   weights: {               │
  │ 4. Filter:       │   │     vector: 0.5,           │
  │    tombstone     │   │     text: 0.3,             │
  │    !=true        │   │     graph: 0.2             │
  │                  │   │   }                        │
  └────────┬─────────┘   │ }                          │
           │              └──────────────┬─────────────┘
           │                             │
           ▼                             ▼
  ┌────────────────────────────────────────────┐
  │ Return entities + relationships             │
  │ (ranked by relevance / mentions)           │
  └────────────────────────────────────────────┘
```

---

## 4. Data Layer Design

### 4.1 Schema Alignment with Graph Plugin

The data model uses the Graph plugin's **`LpgMemoryIndexSchema`** conventions. All scope, temporal, provenance, and lifecycle fields are **top-level** (not inside `properties{}`), enabling proper typed queries, date range filtering, and the Graph plugin's memory-managed detection.

Index metadata marker:
```json
{ "_meta": { "memory_managed": true } }
```

This allows the Graph plugin to recognize these indices and enforce scope rules via `isMemoryManagedMapping()`.

### 4.2 Entity Node Schema (lpg-nodes)

| Field | Type | Category | Description |
|-------|------|----------|-------------|
| `id` | keyword | LPG core | Node identifier |
| `labels` | keyword[] | LPG core | `["__Entity__", "{type}"]` where type is: person, organization, technology, location, concept, event, product |
| `properties` | flat_object | LPG core | Arbitrary metadata (display_name, custom attributes) |
| `embedding` | knn_vector | LPG core | Entity name embedding for similarity search |
| `tenant_id` | keyword | Scope | Multi-tenant isolation |
| `user_id` | keyword | Scope | User ownership |
| `agent_id` | keyword | Scope | Agent that extracted this entity |
| `session_id` | keyword | Scope | Session context |
| `namespace` | keyword | Scope | Logical grouping (e.g., container ID) |
| `key` | keyword | Identity | Original entity key for reverse lookup from hashed `_id` |
| `event_time` | date | Temporal | When the entity was first mentioned in conversation |
| `ingest_time` | date | Temporal | When the entity was ingested into the graph |
| `valid_from` | date | Temporal | Start of validity window |
| `valid_to` | date | Temporal | End of validity window |
| `expires_at` | date | Temporal | TTL expiration timestamp |
| `last_access_time` | date | Temporal | Last time entity was accessed in a search |
| `tombstone` | boolean | Lifecycle | Soft-delete flag (`true` = logically deleted) |
| `supersedes` | keyword | Lifecycle | ID of the entity this one supersedes |
| `source_id` | keyword | Provenance | Source identifier (e.g., working memory doc ID) |
| `source_type` | keyword | Provenance | `"conversation"`, `"manual"`, `"backfill"` |
| `extractor_id` | keyword | Provenance | Extraction pipeline identifier |
| `model_id` | keyword | Provenance | LLM model used for extraction |
| `confidence` | float | Provenance | Extraction confidence (0.0-1.0) |
| `schema_version` | keyword | Meta | Schema version for backward compatibility |

### 4.3 Relationship Edge Schema (lpg-edges)

| Field | Type | Category | Description |
|-------|------|----------|-------------|
| `id` | keyword | LPG core | Edge identifier |
| `type` | keyword | LPG core | Relationship type: `WORKS_AT`, `KNOWS`, `USES`, `PART_OF`, etc. |
| `source` | keyword | LPG core | Source entity node ID |
| `target` | keyword | LPG core | Target entity node ID |
| `properties` | flat_object | LPG core | Arbitrary metadata (mentions count, custom attributes) |
| `tenant_id` | keyword | Scope | Multi-tenant isolation |
| `user_id` | keyword | Scope | User ownership |
| `agent_id` | keyword | Scope | Agent that extracted this relationship |
| `session_id` | keyword | Scope | Session context |
| `namespace` | keyword | Scope | Logical grouping |
| `event_time` | date | Temporal | When the relationship was mentioned |
| `ingest_time` | date | Temporal | When ingested into the graph |
| `valid_from` | date | Temporal | Start of validity |
| `valid_to` | date | Temporal | End of validity (set on soft-delete) |
| `expires_at` | date | Temporal | TTL expiration |
| `last_access_time` | date | Temporal | Last search access |
| `tombstone` | boolean | Lifecycle | Soft-delete flag |
| `supersedes` | keyword | Lifecycle | ID of the edge this supersedes |
| `source_id` | keyword | Provenance | Source identifier |
| `source_type` | keyword | Provenance | Source type |
| `extractor_id` | keyword | Provenance | Extraction pipeline ID |
| `model_id` | keyword | Provenance | LLM model used |
| `confidence` | float | Provenance | Extraction confidence |
| `schema_version` | keyword | Meta | Schema version |

### 4.4 ID Generation

Following `LpgMemoryIndexSchema` conventions with deterministic SHA-256 hashing:

**Entity Node ID** (follows `MemorySchema.memoryNodeId()` convention):
```
_id = "mn:" + sha256hex(tenant_id + "\0" + scope + "\0" + normalized_entity_name)
key = normalized_entity_name
```
Result: 67-character string (3-char prefix `mn:` + 64-char hex SHA-256 digest).

The `scope` component determines the entity identity boundary:
- Default: `scope = user_id` -- entities are per-user (same as current behavior)
- Configurable via `graph.entity_scope` (see Section 6.2)
- Future options: `scope = container_id` (shared across users within a container) or `scope = tenant_id` (shared across all containers in a tenant)

> **One-way door note**: The scope component is baked into every entity ID. Changing it after entities exist requires reindexing all nodes and edges (edges reference node IDs). This is why `scope` is configurable from day one rather than hardcoded to `user_id` -- it avoids permanently closing the door on cross-user or cross-container entity resolution.

`user_id` is always stored as a top-level scope field on the document regardless of the `entity_scope` setting, ensuring queries can still filter by user.

**Relationship Edge ID** (follows `MemorySchema.memoryEdgeId()` convention):
```
_id = "me:" + sha256hex(source_node_id + "\0" + relationship_type + "\0" + target_node_id)
```
Result: 67-character string (3-char prefix `me:` + 64-char hex SHA-256 digest).

Properties:
- Deterministic: same entity/relationship always produces same `_id` (enables upsert)
- Null-byte (`\0`) separator prevents delimiter collision
- `mn:` / `me:` prefixes distinguish node IDs from edge IDs
- `key` field stores the original entity name for reverse lookup from hashed `_id`
- `namespace` intentionally excluded from node ID to enable cross-namespace linking within a container

**Symmetric relationship handling**: For undirected relationship types (e.g., `WORKS_WITH`), canonicalize by sorting source/target alphabetically before hashing. This prevents duplicate edges when "Alice WORKS_WITH Bob" and "Bob WORKS_WITH Alice" are extracted.

### 4.5 Index Naming

Derived from the container's `indexPrefix` field (same field used for existing memory indices like `.plugins-ml-am-{indexPrefix}-memory-working`):

```
Graph database name:  mlgraph-{indexPrefix}
Nodes index:          mlgraph-{indexPrefix}-lpg-nodes
Edges index:          mlgraph-{indexPrefix}-lpg-edges
```

- `indexPrefix` defaults to `"default"` or a generated UUID (both valid for `LpgIndexSchema.validateDatabaseName()`)
- Pattern: `[a-z0-9][a-z0-9_-]*`, max 200 chars, no dots/uppercase
- Graph indices are **not** system indices (system index prefix `.plugins-ml-am-` contains dots which fail graph database validation). They are regular OpenSearch indices with `_meta.memory_managed: true` marker.
- No `lpg-constraints` index created (only needed for Cypher `CREATE CONSTRAINT`)
- Containers that share the same `indexPrefix` (rare, guarded by safety checks) would share graph indices -- consistent with existing memory index behavior

### 4.6 Soft-Delete vs Tombstone

The `LpgMemoryIndexSchema` uses `tombstone` (boolean) instead of mem0's `valid` (string) pattern:

```
Active relationship:     tombstone = false (or absent)
Deleted relationship:    tombstone = true, valid_to = <deletion timestamp>
```

All read queries include `tombstone != true` filter by default. Optional `include_deleted=true` parameter shows full history.

The `supersedes` field creates a version chain: when updating a relationship, the new edge's `supersedes` points to the old edge's ID. This enables rollback and lineage tracking.

### 4.7 Index Lifecycle

Existing ML Commons index lifecycle (for reference):
- `TransportCreateMemoryContainerAction.createMemoryDataIndices()` creates up to 4 indices (session, working, long-term, history) via `mlIndicesHandler`
- `TransportDeleteMemoryContainerAction.deleteSelectiveMemoryIndices()` deletes indices by MemoryType, with safety checks for shared indexPrefix

Graph index lifecycle:

| Event | Action |
|-------|--------|
| Create container with `enable_graph=true` | No graph indices created yet (lazy creation). Validate that embedding model is configured. |
| First entity extracted | Create `mlgraph-{indexPrefix}-lpg-nodes` with `LpgMemoryIndexSchema` memory mapping (knn_vector + all scope/temporal/provenance fields + `_meta.memory_managed: true`). Create `mlgraph-{indexPrefix}-lpg-edges` with memory edge mapping. |
| Delete memory container | Delete both graph indices (add to existing `deleteSelectiveMemoryIndices()` logic). Same shared-indexPrefix safety check applies. |
| Disable graph on existing container | Keep indices (no data loss). Stop graph write pipeline. Reads continue on existing data. |
| Embedding model changes | Block if graph indices exist with different knn_vector dimension. Require explicit graph data reset first. |

---

## 5. REST API Design

### 5.1 Graph Search (Entity-Based)

Always available (no Graph plugin dependency).

```http
POST /_plugins/_ml/memory_containers/{container_id}/memories/graph/_search
{
  "query": "Who does Alice work with?",
  "top_k": 10,
  "include_deleted": false
}
```

**Response:**
```json
{
  "entities": [
    {
      "id": "mn:a1b2c3...",
      "name": "alice_smith",
      "labels": ["__Entity__", "person"],
      "confidence": 0.95,
      "properties": { "display_name": "Alice Smith" },
      "event_time": "2026-04-10T09:00:00Z"
    },
    {
      "id": "mn:d4e5f6...",
      "name": "bob_jones",
      "labels": ["__Entity__", "person"],
      "confidence": 0.88,
      "properties": { "display_name": "Bob Jones" },
      "event_time": "2026-04-11T14:00:00Z"
    }
  ],
  "relationships": [
    {
      "source": "mn:a1b2c3...",
      "type": "WORKS_WITH",
      "target": "mn:d4e5f6...",
      "confidence": 0.90,
      "event_time": "2026-04-10T09:15:00Z",
      "properties": { "mentions": "3" }
    }
  ]
}
```

### 5.2 Graph Hybrid Search (Requires Graph Plugin)

```http
POST /_plugins/_ml/memory_containers/{container_id}/memories/graph/_hybrid_search
{
  "query": "What projects are connected to Alice's team?",
  "top_k": 10,
  "hops": 2,
  "max_frontier_per_hop": 100,
  "weights": { "vector": 0.5, "text": 0.3, "graph": 0.2 }
}
```

**Response:**
```json
{
  "results": [
    {
      "entity": {
        "id": "mn:abc123...",
        "name": "project_x",
        "labels": ["__Entity__", "concept"],
        "properties": { "display_name": "Project X" }
      },
      "score": 0.87,
      "path": ["alice_smith", "WORKS_ON", "project_x"]
    }
  ],
  "graph_plugin_available": true
}
```

### 5.3 Get Single Entity

```http
GET /_plugins/_ml/memory_containers/{container_id}/memories/graph/entities/{entity_id}
```

### 5.4 List Entities

```http
GET /_plugins/_ml/memory_containers/{container_id}/memories/graph/entities?label=person&top_k=50&from=0
```

### 5.5 Update Entity

```http
PUT /_plugins/_ml/memory_containers/{container_id}/memories/graph/entities/{entity_id}
{
  "labels": ["__Entity__", "person"],
  "properties": { "display_name": "Dr. Alice Smith" }
}
```

### 5.6 Delete Single Entity

```http
DELETE /_plugins/_ml/memory_containers/{container_id}/memories/graph/entities/{entity_id}
```

Soft-deletes: sets `tombstone=true`, `valid_to=now`. Also soft-deletes all connected edges.

### 5.7 List Relationships

```http
GET /_plugins/_ml/memory_containers/{container_id}/memories/graph/relationships?type=WORKS_AT&top_k=50&from=0
```

### 5.8 Delete Single Relationship

```http
DELETE /_plugins/_ml/memory_containers/{container_id}/memories/graph/relationships/{relationship_id}
```

### 5.9 Manual Add Entity

```http
POST /_plugins/_ml/memory_containers/{container_id}/memories/graph/entities
{
  "name": "project_alpha",
  "labels": ["concept"],
  "properties": { "display_name": "Project Alpha", "status": "active" }
}
```

### 5.10 Manual Add Relationship

```http
POST /_plugins/_ml/memory_containers/{container_id}/memories/graph/relationships
{
  "source_id": "mn:abc123...",
  "type": "PART_OF",
  "target_id": "mn:def456...",
  "properties": {}
}
```

### 5.11 Delete All Graph Data

```http
DELETE /_plugins/_ml/memory_containers/{container_id}/memories/graph
```

---

## 6. Configuration

### 6.1 Example Container Creation

```json
{
  "name": "Agent Memory with Knowledge Graph",
  "configuration": {
    "embedding_model_id": "embed-model-1",
    "embedding_model_type": "TEXT_EMBEDDING",
    "embedding_dimension": 384,
    "llm_id": "llm-model-1",
    "enable_graph": true,
    "graph": {
      "entity_scope": "user_id",
      "similarity_high_threshold": 0.8,
      "similarity_low_threshold": 0.6,
      "max_traversal_hops": 3,
      "max_entities_per_extraction": 20,
      "custom_entity_extraction_prompt": null,
      "custom_relationship_extraction_prompt": null
    },
    "strategies": [
      { "type": "SEMANTIC", "namespace": ["user_id"] }
    ]
  }
}
```

### 6.2 Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_graph` | `false` | Master switch for graph memory |
| `graph.entity_scope` | `"user_id"` | Scope component for entity ID generation. Determines the identity boundary for entity dedup. Options: `"user_id"` (per-user entities), `"container_id"` (shared within container), `"tenant_id"` (shared across tenant). **Immutable after first entity is created** -- changing requires graph data reset. See Section 4.4. |
| `graph.similarity_high_threshold` | `0.8` | Above this: auto-merge entities |
| `graph.similarity_low_threshold` | `0.6` | Below this: create new entity |
| `graph.max_traversal_hops` | `3` | Max hops for graph expansion |
| `graph.max_entities_per_extraction` | `20` | Max entities per LLM extraction |
| `graph.custom_entity_extraction_prompt` | `null` | Override default entity extraction prompt |
| `graph.custom_relationship_extraction_prompt` | `null` | Override default relationship extraction prompt |
| `graph.index_settings` | `{}` | Custom OpenSearch index settings |

Graph reuses the container-level `embedding_model_id`, `embedding_dimension`, and `llm_id`. No separate model configuration needed.

---

## 7. Error Handling & Failure Modes

### 7.1 Failure Handling Matrix

| Step | Failure | Behavior |
|------|---------|----------|
| Entity extraction (LLM) | LLM call fails or times out | Skip entire graph pipeline for this request. Log error. Existing fact pipeline unaffected. |
| Relationship extraction (LLM) | LLM call fails | Write extracted entities (without relationships). Log warning. |
| Entity dedup (embedding) | Embedding model call fails | Create entity as new (skip dedup). Log warning. |
| Entity dedup (LLM verify) | LLM call fails for ambiguous case | Fall back to embedding-only decision (use threshold midpoint: if sim > 0.7, merge; else create new). |
| Conflict resolution (LLM) | LLM call fails | Skip soft-deletes, keep all existing relationships (safe default: preserve data). |
| Graph bulk write | Partial write failure | Log failed items. Successfully written items persist. Graph is eventually consistent, not transactionally consistent. |
| Graph indices not found | Indices deleted externally | Re-create indices on next write (lazy creation). |

### 7.2 Circuit Breaker

A dedicated circuit breaker for the graph pipeline:
- Opens after N consecutive LLM failures (configurable, default: 5)
- When open: entire graph pipeline is skipped, fact pipeline continues
- Half-open after cooldown period (configurable, default: 60s)
- Closes after successful LLM call in half-open state

### 7.3 Concurrency

**Race condition**: Two concurrent requests may extract the same entity and both create it (before OpenSearch near-real-time refresh makes the first write visible).

**Mitigation**:
- Deterministic IDs ensure that concurrent creates for the same entity result in an upsert (last write wins), not duplicates
- For similar but not identical entity names (e.g., "alice" vs "alice_smith"), duplicates may be created
- **Background consolidation** (future): periodic job scans for high-similarity entity pairs and merges them

---

## 8. Security

### 8.1 Index Protection

Graph indices follow the same security model as existing memory indices:
- All graph REST APIs enforce the same user/role-based access controls as memory container APIs
- Owner validation: only the container owner (or users with matching backend roles) can access graph data
- Tenant isolation: `tenant_id` on every node and edge, enforced at query time

### 8.2 Graph Plugin Integration

When the Graph plugin is present:
- Graph indices use `_meta.memory_managed: true` marker
- The Graph plugin's `isMemoryManagedMapping()` detects these indices and enforces scope rules (`SEC-MEM-003`)
- Cypher and retrieval queries through the Graph plugin automatically apply tenant/user scope filtering

---

## 9. Scaling Considerations

### 9.1 Entity Count Growth

| Entity Count | Concern | Mitigation |
|-------------|---------|------------|
| < 10K | No issues | Standard configuration |
| 10K-100K | KNN search latency may increase | HNSW handles this well. Tune `ef_search` parameter. |
| 100K+ | Conflict resolution LLM context overflow | Only send top-N nearest entities to LLM (not all). Configurable via `max_entities_per_extraction`. |
| 1M+ | Index size, shard sizing | Configure multiple shards in `graph.index_settings`. Consider ILM policies for TTL-based cleanup via `expires_at`. |

### 9.2 Index Proliferation

With per-container isolation, 1000 containers = 2000 graph indices. Mitigations:
- **Lazy creation**: Indices only created when first entity extracted
- **Monitoring**: Log warning when graph index count exceeds configurable threshold
- **Future optimization**: Offer shared-index mode (Option B from scoping analysis) for high-container-count deployments

---

## 10. Migration

### 10.1 Enabling Graph on Existing Containers

When `enable_graph` is toggled to `true` on an existing container:
1. Graph indices are not created immediately (lazy creation)
2. New messages trigger graph extraction going forward
3. Existing conversation history is not backfilled automatically

### 10.2 Backfill API (Future)

Optional endpoint to process existing conversations through the graph pipeline:

```http
POST /_plugins/_ml/memory_containers/{container_id}/memories/graph/_backfill
{
  "max_messages": 1000,
  "batch_size": 50
}
```

This is a non-blocking, async operation. Progress can be tracked via a task API.

---

## 11. Graceful Degradation

| Scenario | Behavior |
|----------|----------|
| Graph plugin not installed, `enable_graph=false` | No graph operations. Standard memory works normally. |
| Graph plugin not installed, `enable_graph=true` | Warning logged at first entity extraction. Graph write pipeline skips silently. Mode A search works (standard OpenSearch queries on graph indices). Mode B/C return empty with `graph_plugin_available: false`. |
| Graph plugin installed, `enable_graph=true` | Full functionality. |
| Graph plugin removed after data written | Existing graph indices remain readable via Mode A. Mode B/C stop working. No data loss. |

---

## 12. Comparison with mem0

| Capability | mem0 + Neo4j | ML Commons + Graph Plugin |
|-----------|-------------|--------------------------|
| Node/Edge storage | Neo4j native | OpenSearch LPG indices (memory-managed) |
| Vector on nodes | Neo4j vector (limited) | OpenSearch knn_vector (HNSW/FAISS/Lucene) |
| Hybrid search | Not available | Vector + text + graph fusion in one call |
| Multi-hop traversal | 1-hop only | Variable-length paths, configurable depth |
| Transactions | Single-statement | Full ACID (via Graph plugin) |
| Entity dedup | Embedding only (0.7) | Embedding + LLM verification (two thresholds) |
| Soft-delete | `valid=false` property | `tombstone` boolean + `valid_to` date + `supersedes` chain |
| Mention tracking | `mentions` counter | `mentions` in properties + `last_access_time` |
| Temporal queries | Manual property filter | Native `valid_from`/`valid_to`/`valid_at` support |
| Confidence tracking | Not available | `confidence` float on every node and edge |
| Provenance | Not available | `model_id`, `extractor_id`, `source_id`, `source_type` |
| Schema versioning | Not available | `schema_version` field |
| Custom prompts | `config.graph_store.custom_prompt` | `graph.custom_entity_extraction_prompt` + `graph.custom_relationship_extraction_prompt` |
| Deployment | Separate Neo4j instance | Same OpenSearch cluster |
| Security | Separate auth | Unified OpenSearch security + memory-managed scope enforcement |
| Scalability | Single Neo4j instance | Distributed across cluster shards |
