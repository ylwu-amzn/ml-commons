# Graph Memory — Leadership Demo

## 1-slide summary

> Vanilla agentic memory stores facts as **text blobs**. When an LLM retrieves them, it gets text back — good for "what did we discuss?" but weak for "who, what, and how are they connected?"
>
> Graph memory adds a second, structured layer: entities (people, companies, products) and typed relationships between them. That layer lets the agent answer **multi-hop, filtered, and aggregated** questions that vanilla memory cannot.

## The scenario

Daisy is an Account Executive. Her AI assistant has been taking notes from her calls for a quarter. Leadership wants to know what her assistant can do that a vanilla memory-backed assistant can't.

The two memory stores were populated with the **same 18 CRM facts** (e.g. "Jordan Lee is CTO at Globex", "Globex runs on MongoDB"). The only difference is how they're stored:

| Store | What it contains |
|---|---|
| **Vanilla memory** (baseline) | 18 sentences, each embedded via the same MiniLM model, searched via `neural` KNN — this is what the production agentic-memory `long-term` index looks like. |
| **Graph memory** (this feature) | 16 entities (Daisy, Sarah, Elena, …, Postgres, MongoDB, …) and 17 typed relationships (`REPORTS_TO`, `WORKS_AT`, `USES_TECH`, `COMPETES_WITH`, `INTERESTED_IN`, `KNOWS`). |

## The 5 demo questions

### Q1 — "Who's on Sarah's team?"

```
[VANILLA]   neural search: "who is on Sarah Kim team"
  0.823   Sarah Kim is my sales manager.                   ← Sarah herself, not a teammate
  0.783   Daisy Chen is an AE who reports to Sarah Kim.    ✓
  0.763   Marcus Webb is an SDR who reports to Sarah Kim.  ✓
  0.733   Ravi Patel is a sales engineer who reports to Sarah Kim. ✓
  0.664   Mia Rossi is the Platform Lead at Initech.       ← wrong, unrelated

[GRAPH]     incoming REPORTS_TO edges where target = Sarah
  - Marcus Webb
  - Daisy Chen
  - Ravi Patel
```

The agent has to read and filter 5 text blobs; graph returns exactly the 3 people.

### Q2 — "Which customers run on PostgreSQL?"

```
[VANILLA]   neural search: "which customers use PostgreSQL"
  0.768   Umbrella Corp runs on PostgreSQL.                ✓
  0.750   Northwind Traders runs on PostgreSQL.            ✓
  0.648   Initech competes with Northwind Traders...       ← wrong
  0.624   Sarah Kim is my sales manager.                   ← wrong
  0.614   Elena Torres is researching vector databases...  ← wrong

[GRAPH]     USES_TECH edges where target = t-postgres
  - Northwind Traders
  - Umbrella Corp
```

Vanilla returned 2 correct answers mixed with 3 irrelevant ones. Graph returned exactly 2.

### Q3 — "Which people I already know work at companies using MongoDB?"

This is a **2-hop** question: person → company → tech.

```
[VANILLA]   neural search
  0.806   Initech runs on MongoDB.                         ← not a person
  0.791   Globex runs on MongoDB.                          ← not a person
  0.696   Umbrella Corp runs on PostgreSQL.                ← wrong tech
  0.679   Elena Torres is researching vector databases...  ← wrong topic
  0.651   Initech competes with Northwind Traders...       ← wrong

[GRAPH]     KNOWS(daisy, ?p) ∧ WORKS_AT(?p, ?c) ∧ USES_TECH(?c, t-mongo)
  - Mia Rossi @ Initech
```

**Vanilla cannot answer this.** It returns only companies, and the LLM consumer has no way to cross-reference them with Daisy's contacts. Graph does it in a single query.

### Q4 — "Who are Daisy's contacts at companies competing with our customers?"

A **3-hop** question: `KNOWS → WORKS_AT → COMPETES_WITH → customer`.

```
[VANILLA]   neural search
  0.641   Umbrella Corp runs on PostgreSQL.                ← wrong
  0.629   Daisy Chen is an AE who reports to Sarah Kim.    ← wrong
  0.612   Initech competes with Northwind Traders...       ← partial context only
  0.590   Globex runs on MongoDB.                          ← wrong
  0.590   Elena Torres is researching vector databases...  ← wrong

[GRAPH]
  - Elena Torres @ Northwind Traders  (competes with Initech)
  - Mia Rossi     @ Initech           (competes with Northwind Traders)
```

Vanilla's top result isn't even close to the intent. Graph finds the 2 exact people and names the competitor pair — leadership's "who should we brief about this competitive risk?" question is now answerable.

### Q5 — "Who's interested in vector databases? (for our new product pitch)"

```
[VANILLA]   neural search
  0.872   Elena Torres is researching vector databases...  ✓
  0.674   Tom Becker is the Data Architect at Umbrella.    ← wrong (not vector-interested)
  0.674   Umbrella Corp runs on PostgreSQL.                ← wrong
  0.673   Jordan Lee wants to add vector search to Globex's product. ✓
  0.665   Globex runs on MongoDB.                          ← wrong

[GRAPH]     INTERESTED_IN edges where target = t-vectordb
  - Elena Torres (at Northwind Traders)
  - Jordan Lee   (at Globex)
```

Vanilla does find the 2 leads but buries them among 3 irrelevant hits. Graph returns **exactly the 2 leads with their employer for free** (one more hop) — ready-to-go target list.

## Scoreboard

| Question | Hops | Vanilla precision | Graph precision |
|---|---|---|---|
| Q1. Sarah's team | 1 | 3/5 correct, 2 wrong | 3/3 |
| Q2. Postgres customers | 1 | 2/5 correct, 3 wrong | 2/2 |
| Q3. People I know at MongoDB cos | 2 | **0/5 correct** | 1/1 |
| Q4. Contacts at competitor cos | 3 | **0/5 correct** | 2/2 |
| Q5. Interested in vector DBs | 1 | 2/5 correct, 3 wrong | 2/2 |

The gap grows with each additional hop. Even at 1 hop, graph precision is perfect vs vanilla ~50%. At 2+ hops vanilla essentially can't answer.

## What the REST API looks like

### Seeding graph memory

When the agent observes a fact, it writes an entity and/or a relationship. Example (direct write — LLM extraction from conversation text is also supported):

```bash
POST /e2e2-memory-lpg-edges/_doc/r15?refresh=true
{
  "relationship_id": "r15",
  "source_entity": "p-daisy",
  "target_entity":  "p-elena",
  "relationship_type": "KNOWS",
  "confidence": 0.9,
  "memory_container_id": "4Rh60J0B8jwm23PNlvB4",
  "owner_id": "admin"
}
```

### Agent-facing `/memories/graph/_search` endpoint

Three modes, all behind a single REST surface:

1. **text** — neural entity lookup by name

    ```bash
    POST /_plugins/_ml/memory_containers/{id}/memories/graph/_search
    { "query": "Elena", "search_type": "text", "top_k": 5 }
    ```

2. **hybrid** (default) — blend neural similarity with 1-hop relationship expansion

    ```bash
    POST /_plugins/_ml/memory_containers/{id}/memories/graph/_search
    { "query": "pasta", "top_k": 5 }
    ```

    ```json
    {
      "search_results": [
        {"entity": {"entity_id": "e-pasta"}, "score": 1.0, "match_type": "text_similarity"},
        {"entity": {"entity_id": "e-alice"}, "score": 0.9, "match_type": "relationship_expansion", "relationship_count": 4}
      ]
    }
    ```

3. **traversal** — BFS from a known entity (great for "tell me about Elena's network")

    ```bash
    POST /_plugins/_ml/memory_containers/{id}/memories/graph/_search
    { "query": "Elena Torres", "search_type": "traversal", "entity_id": "p-elena", "max_depth": 2 }
    ```

    Returns Elena + all entities reachable within 2 hops, plus the connecting edges.

## How it fits into the existing agentic-memory story

- **Same container, same container ID** as regular memory — graph is opt-in per container via `enable_graph: true`.
- **Same embedding model** for entity names (no extra model required).
- **Same owner/tenant isolation** — every entity and edge is filtered by `memory_container_id` + `owner_id` + `tenant_id`.
- **Same REST tree**: `memories/` for facts, `memories/graph/` for structure.

## Demo logistics (for re-running)

1. OpenSearch 3.5 local cluster, `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6`
2. ml-commons plugin 3.5.0.0 from `local-test/3.5` branch
3. HuggingFace MiniLM-L6-v2 registered (384-dim) + optional Bedrock Sonnet 4.6 LLM
4. `python3 /tmp/seed_demo.py` — seeds 18 vanilla facts + 16 entities + 17 edges (~15 s)
5. `python3 /tmp/run_demo.py` — runs all 5 Q&A pairs

The two scripts are self-contained and reproducible.

## The punch line

> If the question is *"what did we talk about?"* — vanilla memory is fine.
>
> If the question is *"given what we talked about, who should I call, and why?"* — you need graph memory.
