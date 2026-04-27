# Memory Graph Compile Fix — Part 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `./gradlew :opensearch-ml-plugin:compileJava` succeed end-to-end. Part 1 (commit `63b489282`) moved the 3 graph DTOs into `common/`; ~100 errors remain in 8 plugin files plus Lombok-cascade errors in 3 collateral files.

**Architecture:** Two commits with a clean review boundary.

- **Commit 1 — Mechanical renames.** The 8 graph-feature files were authored against OpenSearch 2.x import paths that moved in 3.x: `client.node.NodeClient`, `client.Client`, `action.ActionListener`, `ml.settings.MLFeatureEnabledSetting`, `ml.helper.TenantAwareHelper`, `ml.action.prediction.MLPredictionTask{Action,Request,Response}`, `common.xcontent.XContentBuilder`, static constants `ML_BASE_URI` and `PARAMETER_MEMORY_CONTAINER_ID`. No logic changes — purely import rewrites plus the two response-type swaps (`MLPredictionTaskResponse` → `MLTaskResponse`) that follow mechanically.

- **Commit 2 — Neural/wrapper rewrite.** The feature currently calls two phantom classes (`TextEmbeddingMLRemoteInferenceInput`, `MLAlgoParams.TEXT_EMBEDDING`) and one unavailable class (`KNNQueryBuilder` from `org.opensearch.knn.*`, which the plugin doesn't depend on). All four embedding/KNN call sites are rewritten to use the established `neural` / `wrapperQuery` pattern from `MemorySearchQueryBuilder.buildSemanticSearchQuery` — the neural-search plugin handles embedding server-side, eliminating the client-side embedding round-trip entirely. The graceful fallback behavior already in `EntityDeduplicationService.deduplicateEntity` (lines 100–118) is preserved.

**Tech Stack:** Java 21, Gradle multi-module, Lombok, OpenSearch 3.x, neural-search integration via JSON-built `wrapperQuery`.

**Verification command (final):** `./gradlew :opensearch-ml-plugin:compileJava` must exit 0.

---

## File Structure

**Files modified in Commit 1 (mechanical renames only — 8 files):**

| File | Import rewrites |
|---|---|
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java` | `action.ActionListener` → `core.action.ActionListener`; `ml.helper.TenantAwareHelper` → `ml.utils.TenantAwareHelper`; `ml.settings.MLFeatureEnabledSetting` → `ml.common.settings.MLFeatureEnabledSetting`; add `org.opensearch.ml.helper.MemoryContainerHelper` |
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportDeleteGraphDataAction.java` | Same three package moves; `client.Client` → `transport.client.Client` |
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportListGraphEntitiesAction.java` | `action.ActionListener` → `core.action.ActionListener`; `ml.settings.MLFeatureEnabledSetting` → `ml.common.settings.MLFeatureEnabledSetting` |
| `plugin/src/main/java/org/opensearch/ml/rest/RestMLGraphSearchAction.java` | `client.node.NodeClient` → `transport.client.node.NodeClient`; `ml.helper.TenantAwareHelper` → `ml.utils.TenantAwareHelper`; `ml.settings.MLFeatureEnabledSetting` → `ml.common.settings.MLFeatureEnabledSetting`; `ml.common.CommonValue.ML_BASE_URI` → `ml.plugin.MachineLearningPlugin.ML_BASE_URI`; `ml.utils.RestActionUtils.PARAMETER_MEMORY_CONTAINER_ID` → `ml.common.memorycontainer.MemoryContainerConstants.PARAMETER_MEMORY_CONTAINER_ID` |
| `plugin/src/main/java/org/opensearch/ml/rest/RestMLDeleteGraphDataAction.java` | Same set as above |
| `plugin/src/main/java/org/opensearch/ml/rest/RestMLListGraphEntitiesAction.java` | Same set as above |
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java` | `action.ActionListener` → `core.action.ActionListener`; `client.Client` → `transport.client.Client`; `ml.action.prediction.MLPredictionTask{Action,Request,Response}` → `ml.common.transport.prediction.MLPredictionTask{Action,Request}` + `ml.common.transport.MLTaskResponse` |
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java` | Same set as above |

Commit 1 leaves the still-broken references to `TextEmbeddingMLRemoteInferenceInput`, `MLAlgoParams.TEXT_EMBEDDING`, and `KNNQueryBuilder` untouched — the plugin still won't compile after Commit 1 alone. That's by design: Commit 1 is a reviewable unit of mechanical change, and Commit 2 rewrites the methods that own those references.

**Files modified in Commit 2 (logic rewrites — 2 files):**

| File | Rewrite |
|---|---|
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java` | Rewrite `searchEntitiesByText` and `searchEntitiesByEmbedding` as one combined `searchEntitiesByText` using neural `wrapperQuery`. Delete `generateQueryEmbedding`, `parseEmbeddingFromResponse`, `parseEmbeddingVector`. Drop now-dead imports: `knn.index.query.KNNQueryBuilder`, `ml.common.transport.prediction.MLPredictionTaskAction`, `ml.common.transport.prediction.MLPredictionTaskRequest`, `ml.common.transport.MLTaskResponse`, `ml.common.input.MLInput`, `ml.common.input.parameter.MLAlgoParams`, `ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput`, `com.jayway.jsonpath.JsonPath`. |
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java` | Rewrite `deduplicateEntity` to skip embedding generation — call `searchSimilarEntities` directly with `entity.getName() + " [" + entity.getType() + "]"`. Rewrite `searchSimilarEntities` to take a `String entityText` instead of `String embedding` and use neural `wrapperQuery`. Delete `generateEntityEmbedding`, `parseEmbeddingFromResponse`, `parseEmbeddingVector`. Drop `MLModelManager` field + constructor parameter (was never used). Drop now-dead imports matching the list above. |

Three files in the last compile run (`AccessController.java`, `MLTaskRunner.java`, `MachineLearningPlugin.java`) show "cannot find symbol: log" and "constructor ToolFactoryWrapper cannot be applied to given types" errors. These are **Lombok annotation-processor cascade failures** that disappear once the graph files compile (verified: those three files have `@Log4j2` on their class declarations and `ToolFactoryWrapper` has `@AllArgsConstructor`; they only fail because the graph-feature compile errors short-circuit Lombok's processor pass). Do not modify them.

---

## Commit 1 — Mechanical Import Renames

### Task 1: Rewrite imports in the 3 transport actions

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java`
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportDeleteGraphDataAction.java`
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportListGraphEntitiesAction.java`

- [ ] **Step 1: `TransportGraphSearchAction.java`**

Open the file. The current import block (lines 11–28, 32) reads:

```java
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
import org.opensearch.ml.common.transport.memory.MLGraphSearchAction;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.ml.helper.TenantAwareHelper;
import org.opensearch.ml.settings.MLFeatureEnabledSetting;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
import org.opensearch.commons.authuser.User;
```

Replace it with:

```java
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchAction;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.ml.helper.MemoryContainerHelper;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
```

Leave the `com.google.common.collect.ImmutableMap` and `lombok.extern.log4j.Log4j2` imports untouched. Do not change any method bodies in this task.

- [ ] **Step 2: `TransportDeleteGraphDataAction.java`**

Current import block (lines 10–25, 27):

```java
import org.opensearch.action.ActionListener;
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.client.Client;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataAction;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataRequest;
import org.opensearch.ml.helper.TenantAwareHelper;
import org.opensearch.ml.settings.MLFeatureEnabledSetting;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
import org.opensearch.commons.authuser.User;
```

Replace with:

```java
import org.opensearch.action.delete.DeleteResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.memorycontainer.MLMemoryContainer;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataAction;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataRequest;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.Client;
```

- [ ] **Step 3: `TransportListGraphEntitiesAction.java`**

Current import block (lines 8–17):

```java
import org.opensearch.action.ActionListener;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.ml.common.transport.memory.MLListGraphEntitiesAction;
import org.opensearch.ml.settings.MLFeatureEnabledSetting;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
```

Replace with:

```java
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.common.inject.Inject;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLGraphSearchResponse;
import org.opensearch.ml.common.transport.memory.MLListGraphEntitiesAction;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
```

- [ ] **Step 4: Do not build yet**

The REST actions, `GraphSearchService`, and `EntityDeduplicationService` still have broken imports. Move to Task 2. A mid-commit compile attempt would produce noise, not signal.

---

### Task 2: Rewrite imports in the 3 REST actions

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/rest/RestMLGraphSearchAction.java`
- Modify: `plugin/src/main/java/org/opensearch/ml/rest/RestMLDeleteGraphDataAction.java`
- Modify: `plugin/src/main/java/org/opensearch/ml/rest/RestMLListGraphEntitiesAction.java`

All three files need the same five import path updates. `ML_BASE_URI` moves to `MachineLearningPlugin`; `PARAMETER_MEMORY_CONTAINER_ID` moves to `MemoryContainerConstants`; `NodeClient`, `TenantAwareHelper`, and `MLFeatureEnabledSetting` move to their 3.x homes. `getAllNodes` and `returnContent` stay on `RestActionUtils`.

- [ ] **Step 1: `RestMLGraphSearchAction.java`**

Current import block (lines 8–10, 17–25):

```java
import static org.opensearch.ml.common.CommonValue.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MEMORY_CONTAINER_ID;
import static org.opensearch.ml.utils.RestActionUtils.getAllNodes;
import static org.opensearch.ml.utils.RestActionUtils.returnContent;
...
import org.opensearch.client.node.NodeClient;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.transport.memory.MLGraphSearchAction;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.helper.TenantAwareHelper;
import org.opensearch.ml.settings.MLFeatureEnabledSetting;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
```

Replace with:

```java
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.PARAMETER_MEMORY_CONTAINER_ID;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.getAllNodes;
import static org.opensearch.ml.utils.RestActionUtils.returnContent;
...
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchAction;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;
```

Leave `java.io.IOException`, `java.util.List`, `java.util.Locale`, and `com.google.common.collect.ImmutableList` untouched.

- [ ] **Step 2: `RestMLDeleteGraphDataAction.java`**

Current import block (lines 8–9, 15–22):

```java
import static org.opensearch.ml.common.CommonValue.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MEMORY_CONTAINER_ID;
...
import org.opensearch.client.node.NodeClient;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataAction;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataRequest;
import org.opensearch.ml.helper.TenantAwareHelper;
import org.opensearch.ml.settings.MLFeatureEnabledSetting;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
```

Replace with:

```java
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.PARAMETER_MEMORY_CONTAINER_ID;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
...
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataAction;
import org.opensearch.ml.common.transport.memory.MLDeleteGraphDataRequest;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;
```

- [ ] **Step 3: `RestMLListGraphEntitiesAction.java`**

Current import block (lines 8–9, 15–23):

```java
import static org.opensearch.ml.common.CommonValue.ML_BASE_URI;
import static org.opensearch.ml.utils.RestActionUtils.PARAMETER_MEMORY_CONTAINER_ID;
...
import org.opensearch.client.node.NodeClient;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLListGraphEntitiesAction;
import org.opensearch.ml.helper.TenantAwareHelper;
import org.opensearch.ml.settings.MLFeatureEnabledSetting;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
```

Replace with:

```java
import static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.PARAMETER_MEMORY_CONTAINER_ID;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
...
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.memory.MLGraphSearchInput;
import org.opensearch.ml.common.transport.memory.MLGraphSearchRequest;
import org.opensearch.ml.common.transport.memory.MLListGraphEntitiesAction;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;
```

- [ ] **Step 4: Do not build yet**

Move to Task 3.

---

### Task 3: Rewrite imports in `GraphSearchService` and `EntityDeduplicationService`

Both services share the same import changes. `MLPredictionTaskAction` and `MLPredictionTaskRequest` moved to `common.transport.prediction`; `MLPredictionTaskResponse` was removed in 3.x and replaced by `MLTaskResponse` at `org.opensearch.ml.common.transport.MLTaskResponse`. `ActionListener` and `Client` moved. The broken imports for `TextEmbeddingMLRemoteInferenceInput`, `MLAlgoParams` (`parameter.MLAlgoParams`), `KNNQueryBuilder`, and `MLModelManager` stay in this task — they are logic-level casualties that Commit 2 removes.

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java`
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java`

- [ ] **Step 1: `GraphSearchService.java`**

Current import block (lines 27–52):

```java
import org.opensearch.action.ActionListener;
import org.opensearch.action.search.MultiSearchAction;
import org.opensearch.action.search.MultiSearchRequest;
import org.opensearch.action.search.MultiSearchResponse;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Client;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.core.common.Strings;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.ml.action.prediction.MLPredictionTaskAction;
import org.opensearch.ml.action.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.action.prediction.MLPredictionTaskResponse;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
import org.opensearch.ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.commons.authuser.User;
```

Replace with:

```java
import org.opensearch.action.search.MultiSearchAction;
import org.opensearch.action.search.MultiSearchRequest;
import org.opensearch.action.search.MultiSearchResponse;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.commons.authuser.User;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.Strings;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.transport.client.Client;
```

Inside the method bodies, change the one `parseEmbeddingFromResponse(MLPredictionTaskResponse response)` method signature to `parseEmbeddingFromResponse(MLTaskResponse response)`. This is a find/replace on the type name only; the method body (`response.getOutput().toString()` followed by `JsonPath.read(...)`) stays the same. Use the Edit tool with `old_string="MLPredictionTaskResponse response"` and `new_string="MLTaskResponse response"` — there is only one occurrence in the file.

- [ ] **Step 2: `EntityDeduplicationService.java`**

Current import block (lines 24–45):

```java
import org.opensearch.action.ActionListener;
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.client.Client;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.ml.action.prediction.MLPredictionTaskAction;
import org.opensearch.ml.action.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.action.prediction.MLPredictionTaskResponse;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.memorycontainer.MemoryType;
import org.opensearch.ml.common.model.MLModelManager;
import org.opensearch.ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
```

Replace with:

```java
import org.opensearch.action.search.SearchAction;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.memorycontainer.MemoryConfiguration;
import org.opensearch.ml.common.memorycontainer.MemoryType;
import org.opensearch.ml.common.model.MLModelManager;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.transport.client.Client;
```

(`XContentBuilder` moves from `common.xcontent` to `core.xcontent`; Commit 2 removes it entirely as unused — leaving it in the correct 3.x location is fine here.)

Method-body change: rename `MLPredictionTaskResponse` to `MLTaskResponse` at each call site (there are three: one in `parseEmbeddingFromResponse`, one in `parseVerificationResponse`, and in the method parameter of both). Use the Edit tool with `old_string="MLPredictionTaskResponse"` and `new_string="MLTaskResponse"`, `replace_all=true`.

Reminder: `MLModelManager` is imported, stored as a field, and required by the constructor, but is never actually invoked. Leave it alone in Commit 1; Commit 2 deletes it.

- [ ] **Step 3: Do not build yet — still logic errors incoming**

After Task 3, the two services still won't compile because `TextEmbeddingMLRemoteInferenceInput`, `MLAlgoParams.TEXT_EMBEDDING`, and `KNNQueryBuilder` remain unavailable. Commit 1 intentionally leaves these broken — they are logic-level fixes, not import fixes. Move to Task 4.

---

### Task 4: Commit 1

**Files:** (none modified)

- [ ] **Step 1: Review the changeset**

Run: `git status` and `git diff --stat`

Expected: 8 modified files, all under `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/` and `plugin/src/main/java/org/opensearch/ml/rest/`. No new files, no deleted files.

- [ ] **Step 2: Sanity-check — confirm the plugin still doesn't compile (this is expected)**

Run: `./gradlew :opensearch-ml-plugin:compileJava 2>&1 | grep -c 'error:'`

Expected: a non-zero count (roughly 40–60 errors remaining), all clustered in `GraphSearchService.java` and `EntityDeduplicationService.java` on the four symbols `TextEmbeddingMLRemoteInferenceInput`, `MLAlgoParams.TEXT_EMBEDDING`, `KNNQueryBuilder`, and `MLModelManager`. If errors appear in any of the other 6 files, the imports were not rewritten correctly — fix the imports before continuing.

- [ ] **Step 3: Commit**

Run:

```bash
git add plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportDeleteGraphDataAction.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportListGraphEntitiesAction.java \
        plugin/src/main/java/org/opensearch/ml/rest/RestMLGraphSearchAction.java \
        plugin/src/main/java/org/opensearch/ml/rest/RestMLDeleteGraphDataAction.java \
        plugin/src/main/java/org/opensearch/ml/rest/RestMLListGraphEntitiesAction.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java

git commit -m "$(cat <<'EOF'
refactor: align memory-graph imports with OpenSearch 3.x package layout

The memory-graph feature was authored against 2.x import paths. Rewrite
imports in 8 files to match 3.x: ActionListener/NodeClient/Client moved
to core/transport.client; MLFeatureEnabledSetting moved to common.settings;
TenantAwareHelper moved from ml.helper to ml.utils; MLPredictionTask*
moved into common.transport.prediction; MLPredictionTaskResponse was
replaced by MLTaskResponse. Static constants ML_BASE_URI and
PARAMETER_MEMORY_CONTAINER_ID moved to MachineLearningPlugin and
MemoryContainerConstants respectively.

No logic changes. GraphSearchService and EntityDeduplicationService
still reference TextEmbeddingMLRemoteInferenceInput, MLAlgoParams, and
KNNQueryBuilder — the follow-up commit rewrites those methods to use
the neural/wrapper query pattern.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify the commit**

Run: `git log -1 --stat`

Expected: one commit, 8 files changed, insertions roughly match deletions (pure import shuffle).

---

## Commit 2 — Neural/Wrapper Rewrite

The neural/wrapper pattern replaces this two-phase flow:

1. Call `MLPredictionTaskAction` to compute an embedding vector client-side.
2. Build a `KNNQueryBuilder` with that vector and issue a `SearchAction`.

...with a single flow:

1. Build a `neural` query as JSON with `XContentBuilder` (query text + model ID + k), wrap it in `QueryBuilders.wrapperQuery(...)`, and issue a `SearchAction`. The neural-search plugin handles embedding server-side.

Reference implementation: `plugin/src/main/java/org/opensearch/ml/utils/MemorySearchQueryBuilder.java:190-234` (`buildSemanticSearchQuery`).

### Task 5: Rewrite `GraphSearchService` embedding flow

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java`

- [ ] **Step 1: Add the new static imports at the top of the import block**

After the existing `static org.opensearch.ml.common.memorycontainer.MemoryContainerConstants.*` static imports (lines 8–17), add:

```java
import static org.opensearch.common.xcontent.json.JsonXContent.jsonXContent;
```

Also add (to the class imports section):

```java
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.ml.common.FunctionName;
```

- [ ] **Step 2: Delete now-dead imports**

Remove these lines from the import block (they were fixed up in Commit 1, but the classes they reference are no longer used after this task):

```java
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput;
import com.jayway.jsonpath.JsonPath;
```

- [ ] **Step 3: Replace `searchEntitiesByText` and `searchEntitiesByEmbedding` with a single neural-query implementation**

Locate by content: the first target is `public void searchEntitiesByText(String queryText, MemoryConfiguration config, Map<String, String> namespace, User user, int topK, ActionListener<List<GraphEntity>> listener)`. The second target is immediately below it: `@VisibleForTesting void searchEntitiesByEmbedding(String embedding, ...)`. Replace both methods (from the start of `searchEntitiesByText`'s doc-comment down to the closing brace of `searchEntitiesByEmbedding`) with:

```java
    /**
     * Search for entities by text query using neural (semantic) search.
     * The neural-search plugin handles embedding generation server-side,
     * so no client-side embedding round-trip is required.
     */
    public void searchEntitiesByText(
        String queryText,
        MemoryConfiguration config,
        Map<String, String> namespace,
        User user,
        int topK,
        ActionListener<List<GraphEntity>> listener
    ) {
        try {
            String graphNodesIndex = config.getGraphNodesIndexName();
            int k = Math.min(topK, 100);

            if (config.getEmbeddingModelType() != FunctionName.TEXT_EMBEDDING
                && config.getEmbeddingModelType() != FunctionName.SPARSE_ENCODING) {
                listener.onFailure(
                    new IllegalStateException("Unsupported embedding model type for entity search: " + config.getEmbeddingModelType())
                );
                return;
            }
            String neuralType = config.getEmbeddingModelType() == FunctionName.TEXT_EMBEDDING ? "neural" : "neural_sparse";

            String neuralQuery;
            try (XContentBuilder builder = XContentBuilder.builder(jsonXContent)) {
                neuralQuery = builder
                    .startObject()
                    .startObject(neuralType)
                    .startObject(ENTITY_EMBEDDING_FIELD)
                    .field("query_text", queryText)
                    .field("model_id", config.getEmbeddingModelId())
                    .field("k", k)
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
            }

            BoolQueryBuilder boolQuery = QueryBuilders.boolQuery()
                .must(QueryBuilders.wrapperQuery(neuralQuery))
                .filter(QueryBuilders.termQuery(MEMORY_CONTAINER_ID_FIELD, namespace.get(MEMORY_CONTAINER_ID_FIELD)));

            if (namespace.containsKey(TENANT_ID_FIELD)) {
                boolQuery.filter(QueryBuilders.termQuery(TENANT_ID_FIELD, namespace.get(TENANT_ID_FIELD)));
            }
            if (user != null && !Strings.isNullOrEmpty(user.getName())) {
                boolQuery.filter(QueryBuilders.termQuery(OWNER_ID_FIELD, user.getName()));
            }

            SearchRequest searchRequest = new SearchRequest(graphNodesIndex)
                .source(new SearchSourceBuilder()
                    .query(boolQuery)
                    .size(Math.min(topK, DEFAULT_TOP_K))
                    .fetchSource(true)
                );

            client.execute(SearchAction.INSTANCE, searchRequest, ActionListener.wrap(
                searchResponse -> listener.onResponse(parseGraphEntities(searchResponse)),
                error -> {
                    log.error("Entity search failed for query: {}", queryText, error);
                    listener.onFailure(error);
                }
            ));
        } catch (IOException e) {
            log.error("Failed to build neural query for: {}", queryText, e);
            listener.onFailure(e);
        } catch (Exception e) {
            log.error("Error in entity text search for query: {}", queryText, e);
            listener.onFailure(e);
        }
    }
```

The new method keeps all tenant/owner/container filter logic, the DEFAULT_TOP_K cap, the graceful listener.onFailure on errors, and the @VisibleForTesting boundary has shifted (the old embedding-aware method is gone — callers now invoke only `searchEntitiesByText`). The `FunctionName` check matches `MemorySearchQueryBuilder.buildSemanticSearchQuery:205-208`.

- [ ] **Step 4: Delete `generateQueryEmbedding`, `parseEmbeddingFromResponse`, `parseEmbeddingVector`**

Locate by content, not line number (Commit 1 shifted some lines). Use the Edit tool to delete these three methods:

- `void generateQueryEmbedding(String queryText, MemoryConfiguration config, ActionListener<String> listener)` and its preceding doc-comment
- `private String parseEmbeddingFromResponse(MLTaskResponse response)` and its preceding doc-comment
- `private float[] parseEmbeddingVector(String embeddingJson)` and its preceding doc-comment

Leave `parseGraphEntities`, `parseGraphRelationships`, `mapToGraphEntity`, and `mapToGraphRelationship` intact — those are still used. Sanity check with: `grep -nE 'generateQueryEmbedding|parseEmbeddingFromResponse|parseEmbeddingVector' plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java` → expect no matches.

- [ ] **Step 5: Remove any remaining references to `searchEntitiesByEmbedding`**

Run: `grep -n 'searchEntitiesByEmbedding' plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java`

Expected: no matches. If matches appear (e.g., in `hybridGraphSearch` or `traverseGraph`), replace each call with a call to `searchEntitiesByText` passing the same parameters. The embedding parameter in the old signature becomes the text the caller already has.

---

### Task 6: Rewrite `EntityDeduplicationService` embedding flow

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java`

- [ ] **Step 1: Update imports**

Add (new static + new class imports):

```java
import static org.opensearch.common.xcontent.json.JsonXContent.jsonXContent;
...
import org.opensearch.ml.common.FunctionName;
```

Delete (now-dead imports from Commit 1's list):

```java
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.input.parameter.MLAlgoParams;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.ml.common.model.MLModelManager;
import org.opensearch.ml.engine.algorithms.remote.TextEmbeddingMLRemoteInferenceInput;
import org.opensearch.common.xcontent.XContentFactory;
```

Keep `com.jayway.jsonpath.JsonPath` — `parseVerificationResponse` (the LLM dedup helper, still used by `verifyEntitySimilarityWithLLM`) calls `JsonPath.read(...)`. Also keep `org.opensearch.core.xcontent.XContentBuilder` (used by the new `searchSimilarEntities` body) and `org.opensearch.core.xcontent.XContentParser` (still used elsewhere). Sanity check with: `grep -n 'JsonPath\|XContentBuilder\|XContentParser' plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java`.

- [ ] **Step 2: Drop the `MLModelManager` field and constructor parameter**

Current constructor (line 76):

```java
public EntityDeduplicationService(Client client, MLModelManager mlModelManager) {
    this.client = client;
    this.mlModelManager = mlModelManager;
}
```

Replace with:

```java
public EntityDeduplicationService(Client client) {
    this.client = client;
}
```

Delete the field declaration (line 59):

```java
private final MLModelManager mlModelManager;
```

`MLModelManager` was imported, stored, and never invoked. `grep -rn 'new EntityDeduplicationService' plugin/ common/` to check callers — there are none in the current tree, so signature change is safe.

- [ ] **Step 3: Rewrite `deduplicateEntity` to skip client-side embedding generation**

Current body (lines 84–120) wraps `generateEntityEmbedding` → `searchSimilarEntities` → `processSimilarityResults`. Replace with:

```java
    /**
     * Deduplicate an extracted entity against existing entities.
     * Uses neural search to delegate embedding to the neural-search plugin.
     * Gracefully falls back to creating a new entity on search failure.
     */
    public void deduplicateEntity(
        ExtractedEntity entity,
        MemoryConfiguration config,
        Map<String, String> namespace,
        ActionListener<ProcessedEntity> listener
    ) {
        try {
            String entityText = entity.getName() + " [" + entity.getType() + "]";
            searchSimilarEntities(entity, entityText, config, namespace, ActionListener.wrap(
                similarEntities -> processSimilarityResults(entity, similarEntities, config, namespace, listener),
                error -> {
                    log.error("Failed to search similar entities for entity: {}", entity.getName(), error);
                    ProcessedEntity newEntity = createNewEntity(entity, config, namespace);
                    listener.onResponse(newEntity);
                }
            ));
        } catch (Exception e) {
            log.error("Error in entity deduplication for entity: {}", entity.getName(), e);
            ProcessedEntity newEntity = createNewEntity(entity, config, namespace);
            listener.onResponse(newEntity);
        }
    }
```

The graceful fallback-to-new-entity behavior on any error is preserved — callers see the same semantics.

- [ ] **Step 4: Rewrite `searchSimilarEntities` to take text and use neural query**

Current signature (line 176): `void searchSimilarEntities(ExtractedEntity entity, String embedding, MemoryConfiguration config, Map<String, String> namespace, ActionListener<List<SimilarEntity>> listener)`.

Replace the entire method with:

```java
    @VisibleForTesting
    void searchSimilarEntities(
        ExtractedEntity entity,
        String entityText,
        MemoryConfiguration config,
        Map<String, String> namespace,
        ActionListener<List<SimilarEntity>> listener
    ) {
        try {
            String graphNodesIndex = config.getGraphNodesIndexName();

            if (config.getEmbeddingModelType() != FunctionName.TEXT_EMBEDDING
                && config.getEmbeddingModelType() != FunctionName.SPARSE_ENCODING) {
                listener.onFailure(
                    new IllegalStateException("Unsupported embedding model type for dedup: " + config.getEmbeddingModelType())
                );
                return;
            }
            String neuralType = config.getEmbeddingModelType() == FunctionName.TEXT_EMBEDDING ? "neural" : "neural_sparse";

            String neuralQuery;
            try (XContentBuilder builder = XContentBuilder.builder(jsonXContent)) {
                neuralQuery = builder
                    .startObject()
                    .startObject(neuralType)
                    .startObject(ENTITY_EMBEDDING_FIELD)
                    .field("query_text", entityText)
                    .field("model_id", config.getEmbeddingModelId())
                    .field("k", 10)
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
            }

            BoolQueryBuilder boolQuery = QueryBuilders.boolQuery()
                .must(QueryBuilders.wrapperQuery(neuralQuery))
                .filter(QueryBuilders.termQuery(MEMORY_CONTAINER_ID_FIELD, namespace.get(MEMORY_CONTAINER_ID_FIELD)))
                .filter(QueryBuilders.termQuery(ENTITY_TYPE_FIELD, entity.getType()));

            if (namespace.containsKey(TENANT_ID_FIELD)) {
                boolQuery.filter(QueryBuilders.termQuery(TENANT_ID_FIELD, namespace.get(TENANT_ID_FIELD)));
            }

            SearchRequest searchRequest = new SearchRequest(graphNodesIndex)
                .source(new SearchSourceBuilder().query(boolQuery).size(10));

            client.execute(SearchAction.INSTANCE, searchRequest, ActionListener.wrap(
                searchResponse -> listener.onResponse(parseSimilarEntities(searchResponse)),
                error -> {
                    log.error("Similar-entity search failed for entity: {}", entity.getName(), error);
                    listener.onFailure(error);
                }
            ));
        } catch (IOException e) {
            log.error("Failed to build neural query for entity: {}", entity.getName(), e);
            listener.onFailure(e);
        } catch (Exception e) {
            log.error("Error in similar-entity search for entity: {}", entity.getName(), e);
            listener.onFailure(e);
        }
    }
```

Same filter semantics as before (container + entity-type + optional tenant). Top-10 cap preserved.

- [ ] **Step 5: Delete `generateEntityEmbedding`, `parseEmbeddingFromResponse`, `parseEmbeddingVector`**

Locate by content (line numbers will have shifted from Commit 1 + constructor edit). Use Edit to remove these three methods and their preceding doc-comments:

- `void generateEntityEmbedding(ExtractedEntity entity, MemoryConfiguration config, ActionListener<String> listener)`
- `private String parseEmbeddingFromResponse(MLTaskResponse response)`
- `private float[] parseEmbeddingVector(String embeddingJson)`

Do **not** touch `parseVerificationResponse` — it also reads via `JsonPath` and is still used by `verifyEntitySimilarityWithLLM`.

- [ ] **Step 6: Verify no residual callers of the removed methods**

Run: `grep -nE 'generateEntityEmbedding|parseEmbeddingFromResponse|parseEmbeddingVector' plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java`

Expected: no matches.

---

### Task 7: Compile and commit

**Files:** (none modified)

- [ ] **Step 1: Plugin compile**

Run: `./gradlew :opensearch-ml-plugin:compileJava`

Expected: `BUILD SUCCESSFUL`. If any errors remain:
1. If they're `cannot find symbol: log` in `AccessController`, `MLTaskRunner`, or `MachineLearningPlugin`: these should have been Lombok-cascade. If they persist, run `./gradlew :opensearch-ml-plugin:compileJava --rerun-tasks` to force a clean annotation-processor pass.
2. If they're in graph files, the neural-query rewrite is likely missing a filter, import, or type. Check the error line against `MemorySearchQueryBuilder.buildSemanticSearchQuery` as the reference.
3. If they're in the REST or transport layer, revisit Tasks 1–2.

- [ ] **Step 2: Common compile (regression check)**

Run: `./gradlew :opensearch-ml-common:compileJava`

Expected: `BUILD SUCCESSFUL`. This catches any accidental common-module imports added during the plan.

- [ ] **Step 3: Existing tests still pass**

Run: `./gradlew :opensearch-ml-common:test --tests '*MemoryConfigurationTests*'`

Expected: `BUILD SUCCESSFUL`, all tests pass. `MemoryConfigurationTests` is the only test file in the feature commit; passing it confirms neighboring common-module surface isn't broken.

- [ ] **Step 4: Review the Commit 2 changeset**

Run: `git status` and `git diff --stat`

Expected: 2 modified files (`GraphSearchService.java` and `EntityDeduplicationService.java`). Deletions should exceed insertions (we removed ~150 lines of embedding plumbing, added ~100 lines of neural-query building).

- [ ] **Step 5: Commit**

Run:

```bash
git add plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/EntityDeduplicationService.java

git commit -m "$(cat <<'EOF'
refactor: use neural/wrapper query for memory-graph entity search

GraphSearchService and EntityDeduplicationService previously computed
embeddings client-side (via TextEmbeddingMLRemoteInferenceInput + an
MLPredictionTaskAction round-trip) and then issued KNNQueryBuilder
queries against the k-NN plugin. Both approaches failed to compile:
TextEmbeddingMLRemoteInferenceInput does not exist, and the plugin does
not depend on the k-NN plugin.

Rewrite both services to use the neural/wrapper-query pattern already
established in MemorySearchQueryBuilder.buildSemanticSearchQuery: build
a neural (or neural_sparse) query as JSON and wrap it in
QueryBuilders.wrapperQuery. The neural-search plugin performs embedding
server-side, removing the client-side round-trip and dropping ~150
lines of embedding plumbing.

Preserve graceful fallback: deduplicateEntity still creates a new
entity on search failure. Drop unused MLModelManager dependency from
EntityDeduplicationService's constructor — it was stored but never
invoked.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Final verification**

Run: `./gradlew :opensearch-ml-plugin:compileJava && git log --oneline -3`

Expected: `BUILD SUCCESSFUL`. `git log` should show the two new commits on top of `63b489282 fix: move graph DTOs to common module`.

The feature branch now compiles. Phase 2a hardening (tenant-isolation, dedup-fallback safety, LLM circuit breaker) and test coverage for the 3 services + 3 transport actions are separate follow-up plans.
