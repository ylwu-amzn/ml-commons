# Memory Container 02 — Semantic & Hybrid search APIs — PR #4658

- **Area:** Long-term memory retrieval
- **PR:** [#4658](https://github.com/opensearch-project/ml-commons/pull/4658)
- **Status:** PASS
- **Date:** 2026-05-22
- **Cluster:** OpenSearch 3.7.0, ml-commons 3.7.0-SNAPSHOT, neural-search plugin present

## Goal

Validate the new long-term memory search APIs:

- `POST /_plugins/_ml/memory_containers/{id}/memories/long-term/_semantic_search`
- `POST /_plugins/_ml/memory_containers/{id}/memories/long-term/_hybrid_search`

Including: weight tuning, `k` boundary validation [1, 10000], `query` validation, no
persistent pipeline created.

## Setup

Embedding model (Titan Embed v2, 1024-dim, normalized): `3Tu2QZ4BmZYVijh_4FoW`.
LLM (Claude Haiku 4.5 Converse): `oTsCQp4BmZYVijh_El1Y`.

### Step 1 — Create memory container

```bash
oc -X POST "$OS_URL/_plugins/_ml/memory_containers/_create" -H "Content-Type: application/json" -d '{
  "name": "t37_hybrid_search_test",
  "description": "PR #4658 hybrid/semantic search test",
  "configuration": {
    "embedding_model_id": "3Tu2QZ4BmZYVijh_4FoW",
    "embedding_dimension": 1024,
    "embedding_model_type": "TEXT_EMBEDDING",
    "llm_id": "oTsCQp4BmZYVijh_El1Y",
    "index_prefix": "t37-mc-hybrid",
    "strategies": [{ "type": "SEMANTIC", "namespace": ["user_id"], "enabled": true }],
    "max_infer_size": 5
  }
}'
# → {"memory_container_id":"LDYDUZ4Bn4mRprk_Sw8s","status":"created"}
```

### Step 2 — Add 3 user messages

Three POSTs to `/memory_containers/{id}/memories` with `"infer": true`:

| Message | Extracted long-term memory |
|---------|----------------------------|
| "I love programming in Python." | "The user loves programming in Python." |
| "Java is my favorite for backend development." | "The user's favorite programming language for backend development is Java." |
| "I prefer the OpenSearch query DSL over SQL for log analytics." | "The user prefers the OpenSearch query DSL over SQL for log analytics." |

Note: extractions are plain sentences (no `Context: …. Categories: …` prefix) — confirms #4798
prompt-simplification fix is active.

## Steps & Results

### Step 3 — Semantic search (happy path)

```bash
oc -X POST ".../_semantic_search" -d '{"namespace":{"user_id":"alice"},"query":"Python programming","k":5}'
```

Response (3 hits, ordered by relevance):

| _score | memory |
|--------|--------|
| 0.7174479 | "The user loves programming in Python." |
| 0.6213856 | "The user's favorite programming language for backend development is Java." |
| 0.5380882 | "The user prefers the OpenSearch query DSL over SQL for log analytics." |

PASS — Python ranks #1.

### Step 4 — Hybrid search, default weights (0.5/0.5)

```bash
oc -X POST ".../_hybrid_search" -d '{"namespace":{"user_id":"alice"},"query":"Python programming","k":5}'
```

Top hits (max_score: 1.0):

| _score | memory |
|--------|--------|
| 1.0 | "The user loves programming in Python." |
| 0.23270763 | "The user's favorite programming language for backend development is Java." |
| (third dropped) | "The user prefers the OpenSearch query DSL over SQL for log analytics." |

PASS.

### Step 5 — Hybrid search with tuned weights

`bm25_weight: 0.2, neural_weight: 0.8` — Java score `0.3717` (semantic dominates).
`bm25_weight: 0.8, neural_weight: 0.2` — Java score `0.0937` (BM25 dominates; "Java" doesn't
contain the keyword "Python", so its keyword score is near-zero).

PASS — weights tunable per-request.

### Step 6 — `k` boundary validation

```
k=0       → 400 "k must be between 1 and 10000"
k=10001   → 400 "k must be between 1 and 10000"
k=10000   → 200 (3 hits)
k=1       → 200 (1 hit)
```

PASS.

### Step 7 — `query` validation

```
missing query   → 400 "query cannot be null or blank"
empty query "   " → 400 "query cannot be null or blank"
```

PASS.

### Step 8 — No persistent pipeline created

```bash
oc "$OS_URL/_search/pipeline?pretty"
```

Returned only the pre-existing `bedrock-titan-v2-search-pipeline` and
`bedrock-titan-v2-hybrid-pipeline` fixtures (not created by this test). No new
`_hybrid_search`-specific pipeline persisted in cluster state. PASS.

## Result

PASS — all 7 checks (incl. weight tuning, k validation, query validation, no persistent
pipeline) work as designed.

## Notes / observations

- Hybrid score normalization is `min_max` with `arithmetic_mean` combination (visible in the
  pre-existing `bedrock-titan-v2-hybrid-pipeline` config; the `_hybrid_search` API uses an
  inline equivalent, not this pipeline).
- Field name for the search input is `query`, not `query_text` (this caught me — worth
  documenting clearly in API docs / examples).
- Long-term index naming convention: `.plugins-ml-am-{index_prefix}-memory-long-term` —
  e.g., `.plugins-ml-am-t37-mc-hybrid-memory-long-term`.
- Did not test "neural-search plugin missing → clear error" because the test cluster has
  neural-search installed. Should be covered by an integration test environment without it.
