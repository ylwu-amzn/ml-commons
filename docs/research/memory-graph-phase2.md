# Memory Graph: Phase 2 Improvements

> Date: 2026-04-13
> Status: Draft
> Prerequisite: [Memory Graph HLD](./memory-graph-hld.md) (Phase 1)
> Source: Findings from security, performance, and scalability review of the Phase 1 HLD

This document captures improvements identified during the Phase 1 HLD review that are **not one-way doors** -- they can be layered on after the initial implementation without breaking existing data or APIs.

---

## 1. Security Hardening

### 1.1 Tenant Isolation Enforcement

**Problem**: The write pipeline relies on code discipline to set `tenant_id`/`user_id` on every graph document. A bug could omit these fields, breaking tenant isolation.

**Fix**:
- Add `enforceSecurityFields()` validation that rejects writes missing `tenant_id`/`user_id`
- Add an ingest pipeline on graph indices that rejects documents without required scope fields (defense-in-depth, same pattern as `MemoryContainerPipelineHelper.createLongTermMemoryIngestPipeline`)

### 1.2 LLM Input/Output Sanitization

**Problem**: User conversation text flows directly to entity extraction prompts (prompt injection risk). LLM-returned JSON is indexed directly (output injection risk).

**Fix**:
- Sanitize conversation input before LLM calls (strip prompt injection patterns)
- Validate LLM output JSON schema before indexing
- Reject LLM output that contains forbidden keys (`tenant_id`, `user_id`, `_id`)
- HTML-escape string values in properties

### 1.3 Manual API Input Validation

**Problem**: Manual entity/relationship creation APIs (HLD 5.9, 5.10) accept arbitrary user input.

**Fix**:
- Validate entity name format (alphanumeric + underscore/hyphen, max 100 chars)
- Validate labels against configured entity types
- Reject forbidden property keys
- Size-limit properties (max 50 keys)

### 1.4 Shared indexPrefix Guard

**Problem**: Two containers with the same `indexPrefix` share graph indices -- only document-level filtering prevents cross-tenant leakage.

**Fix**:
- Block graph index creation if `mlgraph-{indexPrefix}-lpg-nodes` already exists for a different container
- Require unique `indexPrefix` for graph-enabled containers

### 1.5 RBAC for Graph Operations

**Problem**: All graph APIs use the same coarse-grained container ownership check. Destructive operations (delete all graph data, backfill) need stronger authorization.

**Fix**:
- Add fine-grained permissions: `memory_graph_read`, `memory_graph_write`, `memory_graph_delete`, `memory_graph_admin`
- Require `memory_graph_admin` for bulk delete and backfill

### 1.6 Audit Logging

**Problem**: No audit trail for graph mutations.

**Fix**:
- Log all graph operations (create entity, delete relationship, backfill) to ML Commons audit index
- Include: timestamp, user, tenant_id, operation, container_id, entity/edge ID

---

## 2. Performance Optimization

### 2.1 Thread Pool Sizing

**Problem**: `AGENTIC_MEMORY_THREAD_POOL` is `4 * CPU cores` with queue depth 10,000. Adding graph pipeline (2-4 LLM calls per request) doubles the load per task.

**Fix options** (choose one):
- Increase pool size to `8 * CPU cores` for graph-enabled deployments
- Create a separate `GRAPH_MEMORY_THREAD_POOL` so graph failures don't starve fact extraction
- Implement queue-depth-based backpressure: circuit breaker opens at 70% queue capacity

### 2.2 Batch LLM Calls

**Problem**: 4-6 LLM calls per `addMemories` request. At 1000 msg/hr, this is $29K-$115K/month in LLM costs.

**Fix**:
- Combine entity + relationship extraction into a single LLM call (save 1 call per request)
- Raise auto-merge threshold from 0.8 to 0.85 to reduce ambiguous-range LLM verification calls
- Cache embeddings by entity name within same session (5 min TTL)

### 2.3 Parallelize Entity Dedup

**Problem**: Sequential KNN search per entity. 10 entities = 10 serial KNN calls (~200ms each = 2s total).

**Fix**:
- Fire all KNN searches concurrently via `CompletableFuture.allOf()`
- Or use OpenSearch multi-search API for batch KNN in single round-trip
- Reduces 2s to ~200ms for 10 entities

### 2.4 Read Path Latency

**Problem**: Mode A search takes 1.5-3s (LLM entity extraction + N KNN searches + edge expansion). Interactive search target is <500ms.

**Fix**:
- Cache entity extraction results for repeated queries within same session
- Skip LLM extraction for direct entity name queries (detect via simple heuristic)
- Parallelize KNN searches and edge expansion queries

### 2.5 Bulk Write Optimization

**Problem**: Graph writes to 2 indices (lpg-nodes + lpg-edges) require 2 separate bulk requests with `IMMEDIATE` refresh.

**Fix**:
- Submit node + edge bulk requests concurrently
- Switch from `IMMEDIATE` to `WAIT_UNTIL` refresh policy to batch refreshes
- Monitor 99th percentile bulk latency, alert if >500ms

### 2.6 Circuit Breaker

**Problem**: No circuit breaker for graph pipeline. LLM service outage blocks all threads.

**Fix**:
- Dedicated circuit breaker for graph LLM calls
- Opens after 5 consecutive failures (configurable)
- Half-open after 60s cooldown
- When open: skip graph pipeline entirely, fact pipeline continues

---

## 3. Scalability

### 3.1 Per-Tenant Shared Index Mode

**Problem**: Per-container isolation = 6 indices per container. Default cluster limit ~1000 indices = ~166 containers.

**Fix**:
- Add shared-index mode: one graph index pair per tenant, with `namespace` field for container scoping
- Make this the default for new deployments
- Per-container isolation remains available as opt-in for strict data isolation requirements
- Add cluster setting `plugins.ml_commons.memory.graph.max_graph_indices` with warning at 80% and hard limit at 100%

### 3.2 Tombstone Garbage Collection

**Problem**: Soft-deleted entities/edges accumulate forever. Tombstoned nodes with embeddings still consume HNSW memory.

**Fix**:
- Add `GraphMaintenanceService` that runs on configurable schedule (default: daily)
- Issues `delete_by_query` for `tombstone=true AND valid_to < now - retention_period`
- Add `graph.tombstone_retention_period` config (default: 30 days)
- Set `expires_at = valid_to + retention_period` whenever `tombstone=true` is set in write pipeline

### 3.3 KNN Memory Management

**Problem**: 384-dim HNSW vectors at 100K entities = ~400-600MB off-heap per shard. At 1M entities = ~4-6GB.

**Fix**:
- Document explicit memory budget in operational guide
- Add HNSW tuning params to config: `graph.knn_engine`, `graph.knn_ef_construction`, `graph.knn_m`
- Recommend FAISS on-disk mode for >100K entities per container
- Future: int8 vector quantization when Graph plugin supports it (4x memory reduction)

### 3.4 Graph Size Limits

**Problem**: No cap on entity/edge count per container.

**Fix**:
- Add cluster settings: `ml_commons.max_graph_nodes_per_container` (default: 100,000), `ml_commons.max_graph_edges_per_container` (default: 500,000)
- Check limits before writes, return 429 if exceeded
- Add per-tenant limit on graph-enabled containers: `ml_commons.max_graph_containers_per_tenant` (default: 50)

---

## 4. Extensibility

### 4.1 Separate Graph Embedding Model

**Problem**: Entity names (1-5 words) have different embedding characteristics than conversation text. A model fine-tuned for entity resolution would improve dedup accuracy.

**Fix**:
- Add optional `graph.embedding_model_id` and `graph.embedding_dimension` to config
- Default: inherit from container-level `embedding_model_id`
- Override: use a specialized model for entity name embeddings

### 4.2 Configurable Entity Types

**Problem**: Entity labels (person, organization, technology, etc.) are implicit in prompts. No mechanism for domain-specific types.

**Fix**:
- Add `graph.entity_types` config: list of allowed entity type labels
- Pipeline validates LLM output against this list, falls back to `"concept"` for unrecognized types
- Default: `["person", "organization", "technology", "location", "concept", "event", "product"]`

### 4.3 Configurable Relationship Types

**Problem**: Relationship types are unconstrained. LLM may produce `WORKS_AT` vs `EMPLOYED_BY` for the same meaning.

**Fix**:
- Add optional `graph.relationship_types` config
- When specified, LLM prompt includes these as the allowed set
- When not specified, accept any type (current behavior)
- Add `graph.undirected_relationship_types` for symmetric relationship canonicalization (default: `["WORKS_WITH", "KNOWS", "COLLABORATES_WITH"]`)

### 4.4 Schema Evolution

**Problem**: `schema_version` field exists but no migration logic.

**Fix**:
- Define schema evolution rules: additive-only changes are backward-compatible, type changes require reindex
- Implement `GraphSchemaManager` that checks index mapping at first write, adds missing fields via `putMapping`
- Store schema version in index `_meta` alongside `memory_managed: true`

### 4.5 Monitoring and Observability

**Problem**: No metrics, health checks, or dashboard guidance.

**Fix**:
- Add `GET /_plugins/_ml/memory_containers/{id}/memories/graph/_stats` endpoint (entity count, edge count, index size, last extraction time, circuit breaker state)
- Expose graph pipeline metrics via `_nodes/stats`: extraction latency, failure rate, dedup hit rate
- Define alerting thresholds: circuit breaker open >5 min, failure rate >20%, index count >80% max

### 4.6 Bulk Write Retry and Dead-Letter

**Problem**: Graph bulk write failures silently discard extracted data ("log + skip").

**Fix**:
- Add 1-2 retry attempts with exponential backoff for bulk write failures
- On final failure, write extracted entities/relationships to dead-letter location (history index with `graph_extraction_failed` event type)
- Enable manual or automated recovery

### 4.7 Entity Dedup Fallback Safety

**Problem**: When dedup LLM verification fails, fallback merges at 0.7 threshold -- risking false merges.

**Fix**:
- Change fallback to "create new entity" (no merge) on LLM failure. False duplicates are recoverable; false merges corrupt data.

### 4.8 Soft-Delete Enforcement via Alias

**Problem**: Read queries must manually include `tombstone != true` filter. Developer forgetting this exposes deleted data.

**Fix**:
- Create index alias with pre-applied filter: `mlgraph-{prefix}-lpg-nodes-active` with `{"must_not": {"term": {"tombstone": true}}}`
- All read queries use the alias by default. Only admin/debug queries use the direct index.

---

## 5. Priority Matrix

| ID | Category | Effort | Impact | Recommended Phase |
|----|----------|--------|--------|-------------------|
| 1.1 | Security | Medium | Critical | 2a (pre-GA) |
| 1.2 | Security | Medium | Critical | 2a (pre-GA) |
| 1.3 | Security | Low | High | 2a (pre-GA) |
| 1.4 | Security | Low | Critical | 2a (pre-GA) |
| 1.5 | Security | High | Medium | 2b (post-GA) |
| 1.6 | Security | Medium | Medium | 2b (post-GA) |
| 2.1 | Performance | Low | High | 2a (pre-GA) |
| 2.2 | Performance | Medium | High | 2a (pre-GA) |
| 2.3 | Performance | Low | Medium | 2a (pre-GA) |
| 2.4 | Performance | Medium | Medium | 2b (post-GA) |
| 2.5 | Performance | Low | Low | 2b (post-GA) |
| 2.6 | Performance | Medium | High | 2a (pre-GA) |
| 3.1 | Scalability | High | Critical | 2b (post-GA) |
| 3.2 | Scalability | Medium | High | 2a (pre-GA) |
| 3.3 | Scalability | Low | Medium | 2b (post-GA) |
| 3.4 | Scalability | Low | Medium | 2b (post-GA) |
| 4.1 | Extensibility | Low | Medium | 2b (post-GA) |
| 4.2 | Extensibility | Low | Low | 2b (post-GA) |
| 4.3 | Extensibility | Low | Low | 2b (post-GA) |
| 4.4 | Extensibility | Medium | Medium | 2b (post-GA) |
| 4.5 | Extensibility | Medium | Medium | 2a (pre-GA) |
| 4.6 | Extensibility | Medium | Medium | 2b (post-GA) |
| 4.7 | Extensibility | Low | Medium | 2a (pre-GA) |
| 4.8 | Extensibility | Low | Low | 2b (post-GA) |

**Phase 2a (pre-GA)**: Security hardening (1.1-1.4), circuit breaker (2.6), thread pool tuning (2.1), LLM batching (2.2), dedup parallelization (2.3), tombstone GC (3.2), monitoring (4.5), dedup fallback safety (4.7)

**Phase 2b (post-GA)**: RBAC (1.5), audit logging (1.6), read path optimization (2.4), shared-index mode (3.1), KNN tuning (3.3), size limits (3.4), separate embedding model (4.1), configurable types (4.2-4.3), schema evolution (4.4), dead-letter (4.6), soft-delete alias (4.8)
