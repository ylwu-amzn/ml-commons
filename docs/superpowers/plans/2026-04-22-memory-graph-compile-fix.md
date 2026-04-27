# Memory Graph Compile Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make branch `feature/memory_graph` compile by resolving the module-boundary violation where `common/MLGraphSearchResponse.java` imports three classes from `plugin/`.

**Architecture:** The `common/` module must not depend on `plugin/` (plugin already depends on common — a reverse import creates a cycle and fails compilation). Three Lombok-annotated data classes — `GraphEntity`, `GraphRelationship`, `GraphSearchResult` — are currently in `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/` but are part of the serialized wire response in `common/`. Move them to `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/` alongside existing common memory types, update import sites in the 6 plugin-side files that reference them, and verify the plugin module compiles.

**Tech Stack:** Java 21, Gradle multi-module build, Lombok (`@Data`, `@Builder`), OpenSearch 3.x.

---

## File Structure

**Classes to relocate (move, don't copy):**

| From | To |
|---|---|
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphEntity.java` | `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphEntity.java` |
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphRelationship.java` | `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphRelationship.java` |
| `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchResult.java` | `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphSearchResult.java` |

After the move, the class bodies are unchanged; only `package` and (for `GraphSearchResult`) the reference to `GraphEntity` via same-package resolution remain. All three classes live in the same new package, so `GraphSearchResult` does not need an import for `GraphEntity`.

**Plugin files that must have imports updated** (they lose same-package resolution after the move):
- `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java`
- `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphTraversalResult.java`
- `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java`

**Common file that must have imports fixed** (currently reaches across modules — the bug being fixed):
- `common/src/main/java/org/opensearch/ml/common/transport/memory/MLGraphSearchResponse.java`

**Verification command:** `./gradlew :opensearch-ml-plugin:compileJava` must succeed with exit code 0.

---

## Task 1: Move GraphEntity to common module

**Files:**
- Create: `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphEntity.java`
- Delete: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphEntity.java`

- [ ] **Step 1: Create the new file in the common module**

Write `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphEntity.java` with exactly this content:

```java
/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.memorycontainer.graph;

import lombok.Builder;
import lombok.Data;

/**
 * Represents an entity in the knowledge graph
 */
@Data
@Builder
public class GraphEntity {
    private String entityId;
    private String name;
    private String type;
    private Double confidence;
    private Integer mentionCount;
}
```

- [ ] **Step 2: Delete the old file**

Run: `git rm plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphEntity.java`

Expected: the file is staged as deleted.

- [ ] **Step 3: Do not build yet**

The other two classes still reference the old location indirectly through the plugin-side consumers. Skip the build — move on to Task 2. A mid-task compile attempt will fail and is not informative.

---

## Task 2: Move GraphRelationship to common module

**Files:**
- Create: `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphRelationship.java`
- Delete: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphRelationship.java`

- [ ] **Step 1: Create the new file**

Write `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphRelationship.java`:

```java
/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.memorycontainer.graph;

import lombok.Builder;
import lombok.Data;

/**
 * Represents a relationship between entities in the knowledge graph
 */
@Data
@Builder
public class GraphRelationship {
    private String relationshipId;
    private String sourceEntityId;
    private String targetEntityId;
    private String relationshipType;
    private Double confidence;
}
```

- [ ] **Step 2: Delete the old file**

Run: `git rm plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphRelationship.java`

Expected: the file is staged as deleted.

---

## Task 3: Move GraphSearchResult to common module

**Files:**
- Create: `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphSearchResult.java`
- Delete: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchResult.java`

- [ ] **Step 1: Create the new file**

Write `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/GraphSearchResult.java`. The `GraphEntity` reference resolves via same-package (both classes now live in `org.opensearch.ml.common.memorycontainer.graph`), so no explicit import is needed:

```java
/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.memorycontainer.graph;

import lombok.Builder;
import lombok.Data;

/**
 * Individual search result from hybrid graph search operations
 */
@Data
@Builder
public class GraphSearchResult {
    private GraphEntity entity;
    private Float score; // Relevance score (0.0-1.0+)
    private String matchType; // "text_similarity", "relationship_expansion", etc.
    private String queryText; // Original query for context
    private Integer relationshipCount; // Number of relationships for this entity
}
```

- [ ] **Step 2: Delete the old file**

Run: `git rm plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchResult.java`

Expected: the file is staged as deleted.

---

## Task 4: Fix imports in MLGraphSearchResponse (the root cause)

**Files:**
- Modify: `common/src/main/java/org/opensearch/ml/common/transport/memory/MLGraphSearchResponse.java`

- [ ] **Step 1: Replace the three cross-module imports**

Open `common/src/main/java/org/opensearch/ml/common/transport/memory/MLGraphSearchResponse.java`.

Replace this block (lines 16–18):

```java
import org.opensearch.ml.action.memorycontainer.memory.GraphEntity;
import org.opensearch.ml.action.memorycontainer.memory.GraphRelationship;
import org.opensearch.ml.action.memorycontainer.memory.GraphSearchResult;
```

With:

```java
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
```

No other changes to this file. The class-name references throughout the body (fields, method signatures, helper methods) already resolve through these imports.

---

## Task 5: Fix imports in GraphTraversalResult

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphTraversalResult.java`

- [ ] **Step 1: Read the file to locate existing imports**

Read the file. It currently resolves `GraphEntity` and `GraphRelationship` via same-package (both were in the same `memorycontainer.memory` package). After Tasks 1–2 those classes moved, so it needs explicit imports.

- [ ] **Step 2: Add the two imports**

Inside the existing `import` block, add:

```java
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
```

Place them in alphabetical order with the other `org.opensearch.ml.*` imports, following the file's existing ordering.

---

## Task 6: Fix imports in GraphSearchService

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java`

- [ ] **Step 1: Add the three imports**

Open the file. Find the `org.opensearch.ml.common.memorycontainer.MemoryConfiguration` import (currently line 45). Immediately below it (to keep alphabetical grouping of `org.opensearch.ml.common.memorycontainer.*` lines), add:

```java
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
```

No other edits needed — the class-name usages throughout the file already resolve through these imports.

---

## Task 7: Fix imports in TransportGraphSearchAction

**Files:**
- Modify: `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java`

- [ ] **Step 1: Read the file to locate the `GraphEntity` / `GraphSearchResult` usages**

Run: `grep -n 'GraphEntity\|GraphSearchResult\|GraphRelationship' plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java`

Expected matches around lines 229–230 referencing `GraphEntity` and `GraphSearchResult`.

- [ ] **Step 2: Add the imports it needs**

Based on the grep results, add imports for every class it references (at minimum `GraphEntity` and `GraphSearchResult`; add `GraphRelationship` only if grep shows a reference). Inside the existing `import` block, add the following (omit any the file does not reference):

```java
import org.opensearch.ml.common.memorycontainer.graph.GraphEntity;
import org.opensearch.ml.common.memorycontainer.graph.GraphRelationship;
import org.opensearch.ml.common.memorycontainer.graph.GraphSearchResult;
```

Place them in alphabetical order with the other `org.opensearch.ml.common.memorycontainer.*` imports.

---

## Task 8: Verify clean compile

**Files:** (none modified)

- [ ] **Step 1: Run the plugin compile**

Run: `./gradlew :opensearch-ml-plugin:compileJava`

Expected: `BUILD SUCCESSFUL`. If there are "cannot find symbol" errors referring to `GraphEntity`, `GraphRelationship`, or `GraphSearchResult` in any `.java` file:

1. Run `grep -rn 'GraphEntity\|GraphRelationship\|GraphSearchResult' plugin/src/main/java/ common/src/main/java/ | grep -v 'org.opensearch.ml.common.memorycontainer.graph'` to find files that reference the classes without the correct import.
2. Add the missing `import org.opensearch.ml.common.memorycontainer.graph.<ClassName>;` in alphabetical order with the other `org.opensearch.ml.common.memorycontainer.*` imports.
3. Re-run the compile.

- [ ] **Step 2: Run the common compile to confirm the module is still clean**

Run: `./gradlew :opensearch-ml-common:compileJava`

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: Run the common module tests**

Run: `./gradlew :opensearch-ml-common:test --tests '*MemoryConfigurationTests*'`

Expected: `BUILD SUCCESSFUL`, tests pass. This is the only test file in the feature commit and it exercises the neighboring `MemoryConfiguration` surface — verifying it still passes confirms nothing in the move broke adjacent code.

---

## Task 9: Commit

**Files:** (none modified)

- [ ] **Step 1: Review what will be committed**

Run: `git status`

Expected: 3 new files under `common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/`, 3 deleted files under `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/`, and modified files: `common/src/main/java/org/opensearch/ml/common/transport/memory/MLGraphSearchResponse.java`, `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphTraversalResult.java`, `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java`, `plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java`.

- [ ] **Step 2: Stage and commit**

Run:

```bash
git add common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/ \
        common/src/main/java/org/opensearch/ml/common/transport/memory/MLGraphSearchResponse.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphTraversalResult.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchService.java \
        plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/TransportGraphSearchAction.java
git rm plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphEntity.java \
       plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphRelationship.java \
       plugin/src/main/java/org/opensearch/ml/action/memorycontainer/memory/GraphSearchResult.java 2>/dev/null || true
```

(`git rm` may no-op if Tasks 1–3 already ran it; that is fine.)

Then:

```bash
git commit -m "$(cat <<'EOF'
fix: move graph DTOs to common module to resolve compile failure

MLGraphSearchResponse lives in common/ but imported GraphEntity,
GraphRelationship, and GraphSearchResult from plugin/ — a reverse
module dependency that breaks the build. Relocate the three data
classes to common/src/main/java/org/opensearch/ml/common/memorycontainer/graph/
and update all importers.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Verify the commit landed cleanly**

Run: `git log -1 --stat`

Expected: one commit listing 3 new files in `common/.../graph/`, 3 deleted files in `plugin/.../memorycontainer/memory/`, and the 4 modified files.

- [ ] **Step 4: Final build sanity check**

Run: `./gradlew :opensearch-ml-plugin:compileJava`

Expected: `BUILD SUCCESSFUL`. The feature branch now compiles and is ready for the follow-up test-coverage and Phase 2a hardening plans.
