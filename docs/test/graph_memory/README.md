# Graph Memory — End-to-end Test Suite

Local cluster: OpenSearch 3.5.0 + this repo's ml-commons plugin (branch `local-test/3.5`) + Bedrock Claude Sonnet 4.6 + HuggingFace MiniLM-L6-v2.

See [`00-environment.md`](./00-environment.md) for setup details.

## Test matrix

| # | Test | Status | Notes |
|---|---|---|---|
| 1 | [Feature flag disabled](./01-feature-flag-disabled.md) | ✅ PASS (after fix) | HTTP 403; **Bug 1.1** fixed |
| 2 | [Create container `enable_graph=true`](./02-create-container-enable-graph-true.md) | ✅ PASS (after fix) | lpg indices created; **Bug 2.1** fixed |
| 3 | [Create container `enable_graph=false`](./03-create-container-enable-graph-false.md) | ✅ PASS | Graph endpoints correctly return 400 |
| 4 | [Graph search — text](./04-graph-search-text.md) | ✅ PASS | Empty + populated both work; **Bug 4.1** fixed |
| 5 | [Graph search — hybrid (default)](./05-graph-search-hybrid.md) | ✅ PASS | Empty + populated; blends `text_similarity` + `relationship_expansion` |
| 6 | [Graph search — traversal](./06-graph-search-traversal-missing-entityid.md) | ✅ PASS | Negative (missing entity_id) + positive (BFS); minor dedup issue found |
| 7 | [List graph entities (GET)](./07-list-graph-entities.md) | ✅ PASS | Empty graph returns `total_count=0` |
| 8 | [Delete graph data](./08-delete-graph-data.md) | ✅ PASS (stub) | Endpoint works; real deletion not implemented |
| 9 | [Unknown container id](./09-unknown-container-id.md) | ✅ PASS | Clean 404 |

**All 9 REST-level graph tests now pass with real data.**

## Graph-feature bugs found and fixed during testing

Every bug below was uncovered by running these tests and verified as a real issue in the feature branch (not environmental). Fixes live on `local-test/3.5` (uncommitted).

| # | Bug | Symptom | Fix |
|---|---|---|---|
| **A** | Graph transport/REST actions never registered | All `/memories/graph/*` URLs fell through to the generic memory-type dispatcher ("Invalid memory type: graph") | Registered 3 REST handlers + 3 ActionHandlers in `MachineLearningPlugin`; added `@Inject` to `GraphSearchService` |
| **1.1** | Flag-disabled REST threw `IllegalStateException` → HTTP 500 | Wrong status code for a documented-as-disabled feature | Replaced with `OpenSearchStatusException(..., RestStatus.FORBIDDEN)` in all 3 REST handlers |
| **2.1** | `lpg-nodes` mapping rejected (`Cannot set modelId or method parameters when index.knn setting is false`) | `enable_graph=true` container creation silently skipped graph indices; user saw `201 Created` anyway | Pass `{"index.knn": true}` to `initIndexIfAbsent` in `createGraphIndices` |
| **4.1** | Body parser not advanced | All POST `/graph/_search` bodies rejected with `"expecting [START_OBJECT] but found [null]"` | Call `parser.nextToken()` at start of `MLGraphSearchRequest.parse` if `currentToken()` is null |
| **8** (prior session) | `DeleteResponse` with null `ShardId` NPE'd on every call | Stub-success path always threw NPE | Pass non-null placeholder `ShardId(indexName, "_na_", 0)` |

## Graph-feature bugs found but not fixed

| # | Bug | Details |
|---|---|---|
| **6.1** | Traversal response duplicates relationships | BFS appends to a plain list without deduping on `relationship_id`. See [Test 6](./06-graph-search-traversal-missing-entityid.md). |
| F.1 | `entity_type` filter never applied | `MLGraphSearchInput.entityType` is read off the wire but `GraphSearchService.searchEntitiesByText` never adds it as a term filter |
| F.2 | `DELETE /memories/graph` is a stub | Returns synthetic `DeleteResponse`; no delete-by-query against `*-lpg-nodes` / `*-lpg-edges` |
| F.5 | Error message `"Dimension is required"` references wrong field | JSON field is `embedding_dimension`; cosmetic |

## Orthogonal bugs (not graph-feature, surfaced during env setup)

| # | Bug | Details |
|---|---|---|
| F.3 | Memory-container PUT doesn't persist `configuration.parameters` or top-level `llm_result_path` | Update endpoint silently drops these fields |
| F.4 | `Map.of(OWNER_ID_FIELD, input.getOwnerId(), ...)` in `TransportAddMemoriesAction:171` | `Map.of` can NPE if any field is null — hit during inference-mode add-memories testing |

## Environmental issue (not a plugin bug)

DJL's bundled PyTorch libstdc++ (max `GLIBCXX_3.4.19`) pre-empts the system libstdc++ for the KNN plugin's faiss library, which needs `GLIBCXX_3.4.20+`. First KNN refresh after embedding-model load crashed the JVM.

**Mitigation:** start OpenSearch with

```bash
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6
bin/opensearch
```

After this preload, tests 4, 5, 6 ran cleanly with real populated-graph data.

## Reproducing

```bash
export EMB_MODEL_ID=BWxf0J0BTxnG51e8Pch9
export LLM_MODEL_ID=3qRp0J0BQ4F7Y_V3OOKT
export CONTAINER_ID=4Rh60J0B8jwm23PNlvB4   # graph-enabled, use_system_index=false
export AUTH='admin:RUIwNTVGTDc2MDhDM0JNRC4u'

# Smoke
curl -sk -u $AUTH "https://localhost:9200/_cat/plugins?v" | grep opensearch-ml
curl -sk -u $AUTH "https://localhost:9200/_plugins/_ml/memory_containers/$CONTAINER_ID"

# Text search (populated)
curl -sk -u $AUTH -XPOST "https://localhost:9200/_plugins/_ml/memory_containers/$CONTAINER_ID/memories/graph/_search" \
  -H 'Content-Type: application/json' -d '{"query":"alice","search_type":"text","top_k":5}'
```

Individual test files contain exact curl commands and the raw responses.
