# Agentic Memory Gap Analysis: ML Commons vs mem0

> Generated: 2026-04-13
> ML Commons branch: `feature/memory_graph`
> mem0 source: `/Users/ylwu/code/public/mem0`

---

## Executive Summary

ML Commons has a solid foundation for agentic memory with multi-tier storage (sessions, working, long-term, history), LLM-driven fact extraction, semantic/hybrid search, and comprehensive REST APIs built on OpenSearch. However, compared to mem0, the **biggest gap is the complete absence of Graph Memory (Knowledge Graph)**. Secondary gaps include lack of episodic/procedural memory types, no reranking pipeline, and no agent-side memory extraction.

---

## 1. Feature Comparison Matrix

| Feature | ML Commons | mem0 | Gap Severity |
|---------|-----------|------|-------------|
| **Graph Memory / Knowledge Graph** | None | Full (Neo4j, Memgraph, Kuzu, Apache AGE, Neptune) | **CRITICAL** |
| **Entity Extraction** | None | LLM-based with type classification | **CRITICAL** |
| **Relationship Management** | None | Source-Relationship-Destination triplets with soft-delete | **CRITICAL** |
| **Episodic Memory** | None (working memory is closest) | Dedicated type with role/actor tracking | **HIGH** |
| **Procedural Memory** | None | Dedicated type for agent execution history | **HIGH** |
| **Reranking Pipeline** | None | 5 implementations (Cohere, SentenceTransformer, LLM, HF, ZeroEntropy) | **HIGH** |
| **Agent Memory Extraction** | None | Separate prompt for extracting assistant characteristics | **MEDIUM** |
| **Memory Scoring/Relevance** | Similarity score from KNN | Similarity + multi-stage reranking + BM25 for graph | **MEDIUM** |
| **Mention/Frequency Tracking** | None | Entity mention count tracking | **MEDIUM** |
| **Vision Support** | None | Image content in messages | **LOW** |
| **Semantic Search** | Yes (KNN + sparse) | Yes (vector similarity) | Comparable |
| **Hybrid Search** | Yes (BM25 + KNN) | Varies by vector store | Comparable |
| **LLM Fact Extraction** | Yes (3 strategy types) | Yes (user + agent extraction) | Comparable |
| **Memory Decisions (ADD/UPDATE/DELETE/NONE)** | Yes (MemoryDecision) | Yes (identical pattern) | Comparable |
| **Custom Prompts** | Yes (strategy config) | Yes (custom_update_memory_prompt) | Comparable |
| **Multi-tenancy / Scoping** | Yes (namespace, owner, tenantId) | Yes (user_id, agent_id, run_id) | ML Commons stronger |
| **History/Audit Trail** | Yes (OpenSearch HISTORY index) | Yes (SQLite changelog) | Comparable |
| **Session Management** | Yes (dedicated SESSIONS type) | Basic (run_id scoping) | ML Commons stronger |
| **Conversation Summarization** | Yes (SESSION_SUMMARY_PROMPT) | Yes (PROCEDURAL_MEMORY_SYSTEM_PROMPT) | Comparable |
| **Access Control** | Yes (user/backend roles, tenant isolation) | Basic (API key + user_id) | ML Commons stronger |
| **Binary Data Support** | Yes (MLWorkingMemory.binaryData) | No | ML Commons stronger |

---

## 2. CRITICAL Gap: Graph Memory / Knowledge Graph

This is the single largest feature gap. mem0's graph memory enables **structured relationship reasoning** that flat vector search cannot replicate.

### What mem0 Has

**Architecture** (`mem0/memory/graph_memory.py`):
- `MemoryGraph` class backed by Neo4j (with Memgraph, Kuzu, Apache AGE, Neptune alternatives)
- Entity nodes with embeddings stored directly on graph nodes
- Directed relationships with metadata (valid, created_at, updated_at, mentions)

**Entity Extraction Pipeline**:
1. LLM extracts entities with semantic type classification (person, place, concept, etc.)
2. Entities normalized: lowercase, spaces replaced with underscores
3. Self-references (I, me, my) mapped to `user_id`
4. Entity embeddings computed and stored on graph nodes

**Relationship Management**:
- Source -> Relationship -> Destination triplets
- Relationship names are timeless and consistent ("professor" not "became_professor")
- MERGE semantics: creates or updates nodes/relationships
- Soft-delete: `valid=false` + `invalidated_at` timestamp (enables temporal reasoning)
- Mention counting on both nodes and relationships

**Graph Search**:
- Cosine similarity on node embeddings to find relevant entities
- Configurable similarity threshold (default 0.7)
- Bidirectional traversal (incoming + outgoing relationships)
- BM25 Okapi reranking of relationship triplets against query
- Scope filtering (user_id, agent_id, run_id)

**Conflict Resolution**:
- LLM decides which existing relationships to soft-delete when new data contradicts
- Does NOT delete when same relationship type exists with different destinations (additive)
- Example: "alice -- loves_to_eat -- pizza" + new "loves to eat burger" -> keeps BOTH

**Cypher Query Examples** (from `graph_memory.py`):
```cypher
-- Entity search with embedding similarity
MATCH (n :`__Entity__` {user_id: $user_id})
WHERE n.embedding IS NOT NULL
WITH n, round(2 * vector.similarity.cosine(n.embedding, $n_embedding) - 1, 4) AS similarity
WHERE similarity >= $threshold
...

-- Bidirectional relationship traversal
MATCH (n)-[r]->(m :`__Entity__` {user_id: $user_id})
WHERE r.valid IS NULL OR r.valid = true
RETURN n.name AS source, type(r) AS relationship, m.name AS destination
UNION
MATCH (n)<-[r]-(m :`__Entity__` {user_id: $user_id})
WHERE r.valid IS NULL OR r.valid = true
RETURN m.name AS source, type(r) AS relationship, n.name AS destination
```

### What ML Commons Needs

To close this gap, ML Commons would need:

1. **Graph Store Abstraction**: Interface for graph operations (add entity, add relationship, search, delete)
2. **Entity Extraction Service**: LLM-based entity extraction with type classification, leveraging existing `MemoryProcessingService` pattern
3. **Relationship Management**: CRUD for entity relationships stored in OpenSearch (or external graph DB)
4. **Graph Search API**: New transport action for graph-based memory search
5. **Soft-Delete Semantics**: Temporal validity tracking for relationships
6. **Integration with Existing Memory Pipeline**: Graph extraction as an additional strategy type or parallel processing step during `addMemories`

**Recommended Approach**: Leverage the **OpenSearch Graph Plugin** (unreleased, in development). It provides native LPG storage (`lpg-nodes`/`lpg-edges` indices), Cypher read/write, kNN embeddings on graph nodes, hybrid retrieval (vector+text+graph fusion), ACID transactions, and distributed execution -- all within the same OpenSearch cluster. This is strictly superior to mem0's Neo4j dependency. See full analysis: **[graph-plugin-integration-proposal.md](./graph-plugin-integration-proposal.md)**.

---

## 3. HIGH Gap: Episodic Memory

### What mem0 Has (`MemoryType.EPISODIC`)

- Event-based, time-specific memory
- Raw message preservation from conversations (not extracted/summarized)
- Tracks: role (user/assistant), actor_id, created_at, updated_at
- Stored in vector store alongside semantic memories but with different type marker
- Useful for "what did the user say last Tuesday?" queries

### ML Commons Current State

- `MLWorkingMemory` stores conversation messages but is designed as short-term/ephemeral
- No dedicated episodic type for long-term event preservation
- Interactions track conversations but in a separate legacy system

### Recommendation

Add `EPISODIC` to `MemoryStrategyType` (or `MemoryType`). Episodic memories would be raw message snapshots stored in the long-term index with a strategy marker, preserving role, timestamp, and actor information without LLM summarization.

---

## 4. HIGH Gap: Procedural Memory

### What mem0 Has (`MemoryType.PROCEDURAL`)

- Agent execution history and task workflows
- Requires `agent_id` presence
- Uses `PROCEDURAL_MEMORY_SYSTEM_PROMPT` for comprehensive step-by-step summarization
- Records: agent actions, results, findings, navigation history, errors
- Preserves all agent outputs verbatim
- Structure: Task Objective, Progress Status, Sequential Agent Actions (numbered)

### ML Commons Current State

- Agent traces tracked via `parentInteractionId` and `traceNum` in Interaction model
- `GetTracesAction` retrieves trace trees
- But no LLM-driven summarization of agent execution into reusable procedural knowledge

### Recommendation

Add procedural memory as a new strategy type. When an agent completes a task, summarize the execution trace into procedural memory that can be recalled for similar future tasks.

---

## 5. HIGH Gap: Reranking Pipeline

### What mem0 Has

5 reranker implementations in `mem0/reranker/`:

| Reranker | Description | File |
|----------|-------------|------|
| **Cohere** | Cloud-based reranking API | `cohere_reranker.py` |
| **Sentence Transformer** | Local cross-encoder model | `sentence_transformer_reranker.py` |
| **LLM Reranker** | Uses LLM for relevance scoring | `llm_reranker.py` |
| **HuggingFace** | Community model-based | `huggingface_reranker.py` |
| **Zero Entropy** | Uncertainty-based ranking | `zero_entropy_reranker.py` |

Reranking is applied after initial vector search to improve precision. Configurable via `MemoryConfig.reranker`.

### ML Commons Current State

- Semantic search returns raw KNN scores
- Hybrid search combines BM25 + KNN but no post-retrieval reranking
- No reranker abstraction or pipeline

### Recommendation

ML Commons already has a model serving infrastructure. A reranking step could use an existing cross-encoder model registered in ML Commons. Add a reranking phase to `MemorySearchService` that optionally re-scores results using a configured reranker model.

---

## 6. MEDIUM Gap: Agent Memory Extraction

### What mem0 Has

Two separate extraction prompts:
- `USER_MEMORY_EXTRACTION_PROMPT` - Extracts facts ONLY from user messages
- `AGENT_MEMORY_EXTRACTION_PROMPT` - Extracts assistant characteristics ONLY from assistant messages

The system auto-selects which prompt to use based on `_should_use_agent_memory_extraction()`.

### ML Commons Current State

- `SEMANTIC_FACTS_EXTRACTION_PROMPT` extracts from both user and assistant messages (with nuance about assistant conclusions)
- `USER_PREFERENCE_FACTS_EXTRACTION_PROMPT` extracts from user messages only
- No dedicated agent/assistant memory extraction

### Recommendation

Consider adding an `AGENT_BEHAVIOR` or `AGENT_CHARACTERISTIC` strategy type that captures assistant patterns and capabilities from conversations. This enables building a profile of how different agents behave.

---

## 7. MEDIUM Gap: Memory Scoring & Frequency Tracking

### What mem0 Has

- **Mention counting**: Both entity nodes and relationships track `mentions` count
- **Multi-stage ranking**: Initial vector search -> optional reranking -> BM25 for graph results
- **Threshold filtering**: Configurable similarity threshold with score-based cutoff

### ML Commons Current State

- KNN scores from semantic search
- BM25 scores from hybrid search
- No mention/frequency tracking
- No multi-stage ranking pipeline

### Recommendation

Add a `mention_count` or `access_count` field to `MLLongTermMemory`. Increment on each retrieval or reference. This enables importance-weighted recall and can inform memory consolidation/pruning.

---

## 8. What ML Commons Does Better

| Feature | Advantage |
|---------|-----------|
| **Multi-tenancy** | Native tenant isolation, backend roles, system indices |
| **Access Control** | User/role-based access with OpenSearch security |
| **Binary Data** | MLWorkingMemory supports binary payloads |
| **Session Management** | Dedicated SESSIONS type with LLM summarization |
| **Namespace Flexibility** | Arbitrary key-value namespace maps vs. fixed user_id/agent_id/run_id |
| **Index Configuration** | Per-index settings (shards, replicas, mappings) |
| **Sparse Encoding** | Supports SPARSE_ENCODING in addition to TEXT_EMBEDDING |
| **Distributed Architecture** | Built for distributed cluster deployment |
| **History Disabling** | Configurable audit trail for performance tuning |
| **Strategy-level LLM Override** | Each strategy can use a different LLM |
| **Prompt Validation** | Validates custom prompts contain required format |

---

## 9. Prompts Comparison

### Fact Extraction

| Aspect | ML Commons | mem0 |
|--------|-----------|------|
| Prompt quality | Well-structured XML tags, specific rules | Simpler, few-shot example based |
| JSON enforcement | Separate enforcement message appended | Inline in prompt |
| Language detection | Mentioned but not enforced | Explicit instruction |
| Custom prompts | Supported with validation | Supported via parameter |

ML Commons prompts are arguably more structured and production-ready (XML-tagged sections, specific rules about named entities, time references, etc.).

### Memory Decision

Both systems use a remarkably similar LLM-driven decision pattern:
- ML Commons: `DEFAULT_UPDATE_MEMORY_PROMPT` with `memory_decision` array -> `MemoryDecision` (ADD/UPDATE/DELETE/NONE)
- mem0: `DEFAULT_UPDATE_MEMORY_PROMPT` with `memory` array -> same events (ADD/UPDATE/DELETE/NONE)

ML Commons adds:
- Similarity score awareness ("Respect similarity scores")
- Duplicate ID handling in old_memory
- Specificity preservation rules ("Named entities must NEVER be generalized")
- More detailed guidelines about when NOT to merge

---

## 10. Implementation Priority

| Priority | Gap | Effort | Impact |
|----------|-----|--------|--------|
| **P0** | Graph Memory (Knowledge Graph) | Large (4-6 weeks) | Enables relationship reasoning, entity-based recall |
| **P1** | Reranking Pipeline | Medium (1-2 weeks) | Improves retrieval precision significantly |
| **P1** | Episodic Memory Type | Small (1 week) | Enables time-based event recall |
| **P2** | Procedural Memory Type | Medium (2-3 weeks) | Enables agent learning from past executions |
| **P2** | Agent Memory Extraction | Small (1 week) | Enables assistant profiling |
| **P3** | Mention/Frequency Tracking | Small (few days) | Enables importance-based ranking |
| **P3** | Vision Support | Medium (1-2 weeks) | Multi-modal memory |

---

## 11. Key Source Files Reference

### ML Commons Memory Files

```
# Data Models
common/.../memorycontainer/MLMemoryContainer.java
common/.../memorycontainer/MLLongTermMemory.java
common/.../memorycontainer/MLWorkingMemory.java
common/.../memorycontainer/MemoryConfiguration.java
common/.../memorycontainer/MemoryStrategyType.java      # SEMANTIC, USER_PREFERENCE, SUMMARY
common/.../memorycontainer/MemoryStrategy.java
common/.../memorycontainer/MemoryDecision.java           # ADD/UPDATE/DELETE/NONE
common/.../memorycontainer/MemoryContainerConstants.java # All prompts

# Processing & Search
plugin/.../memorycontainer/memory/MemoryProcessingService.java  # LLM extraction + decisions
plugin/.../memorycontainer/memory/MemorySearchService.java      # Semantic + hybrid search
plugin/.../memorycontainer/memory/MemoryOperationsService.java  # CRUD operations

# Engine
ml-algorithms/.../memory/AgenticConversationMemory.java
ml-algorithms/.../memory/MLMemoryManager.java
ml-algorithms/.../memory/ConversationIndexMemory.java
```

### mem0 Key Files

```
# Core
mem0/memory/main.py              # Memory + AsyncMemory classes
mem0/memory/base.py              # MemoryBase abstract
mem0/configs/enums.py            # MemoryType: SEMANTIC, EPISODIC, PROCEDURAL
mem0/configs/prompts.py          # All extraction + decision prompts

# Graph Memory (THE BIG GAP)
mem0/memory/graph_memory.py      # MemoryGraph (Neo4j)
mem0/memory/kuzu_memory.py       # Kuzu graph
mem0/memory/memgraph_memory.py   # Memgraph
mem0/memory/apache_age_memory.py # Apache AGE
mem0/graphs/tools.py             # Entity extraction tools
mem0/graphs/utils.py             # Graph prompts

# Reranking (ANOTHER GAP)
mem0/reranker/base.py
mem0/reranker/cohere_reranker.py
mem0/reranker/sentence_transformer_reranker.py
mem0/reranker/llm_reranker.py
mem0/reranker/huggingface_reranker.py
mem0/reranker/zero_entropy_reranker.py

# Storage (25+ vector stores)
mem0/vector_stores/opensearch.py  # They support OpenSearch too!
mem0/vector_stores/...            # 25+ implementations
```

---

## 12. Conclusion

ML Commons has a **production-grade memory infrastructure** that exceeds mem0 in areas like multi-tenancy, access control, distributed architecture, and binary data support. The extraction and decision prompts are more sophisticated. However, the absence of **Graph Memory** means ML Commons cannot perform relationship-based reasoning (e.g., "Who does Alice work with?" or "What companies are connected to Project X?"). This is the most impactful feature gap to address.

The branch name `feature/memory_graph` suggests this is already a recognized priority. The recommended approach is to leverage OpenSearch as the graph store (using entity/relationship indices) while designing the abstraction to support external graph databases later.
