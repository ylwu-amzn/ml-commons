# Graph Memory — Leadership Demo

## TL;DR

> **Vanilla agentic memory stores extracted facts.** Semantic search returns well-formed sentences
> — good for "what did we discuss?", weak for "who is who, and how are they connected?"
>
> **Graph memory adds a structured layer in the same `POST /memories` call.** Claude Sonnet also
> produces entities and typed relationships, stored in a parallel knowledge graph. The agent gets
> multi-hop queries that vanilla cannot answer.
>
> Enabling graph is one line: `"enable_graph": true` plus a strategy block.

## The scenario

Daisy is an AE. Her AI assistant has been taking notes from 3 of her calls. Same `POST /memories`
call populates both stores:

| Store | How | What ended up in it |
|---|---|---|
| **Vanilla** | `SEMANTIC` strategy → Sonnet extracts facts → written to `demo-memory-long-term` | **22 facts** like "Northwind Traders runs all their analytics on PostgreSQL." |
| **Graph** | `GraphProcessingService` → Sonnet extracts entities + typed relationships → written to `demo-memory-lpg-nodes` / `-lpg-edges` | **18 entities + 29 relationships**, including `Initech -[COMPETES_WITH]-> Northwind Traders`, `Daisy Chen -[KNOWS]-> Elena Torres`, etc. |

Both are retrieved via the agentic-memory API — **no custom indexing, no hand-crafted schema**.

## The 5 demo questions — live output

### Q1. Who's on Sarah Kim's team?

```
VANILLA  POST /memories/long-term/_semantic_search   query="who is on Sarah Kim team", k=5
  0.807  Sarah Kim is the manager of Daisy Chen.                                ← mentions Daisy
  0.718  Marcus Webb is one of Sarah Kim's SDRs and booked a discovery call…   ✓ Marcus
  0.688  Sarah Kim said Daisy Chen should loop in Ravi Patel from the SE team. ← Ravi not on team
  0.664  Mia Rossi is the Platform Lead at Initech.                             ✗
  0.655  Elena Torres is VP of Engineering at Northwind Traders.                ✗

GRAPH  MANAGES edges from Sarah OR REPORTS_TO edges to Sarah
  - Marcus Webb
  - Daisy Chen
```

### Q2. Which customers run on PostgreSQL?

```
VANILLA  query="which customers use PostgreSQL", k=5
  0.794  Umbrella Corp uses PostgreSQL.                               ✓
  0.751  Northwind Traders runs all their analytics on PostgreSQL.    ✓
  0.651  Initech competes with Northwind Traders in retail analytics. ✗
  0.616  Globex also runs on MongoDB.                                 ✗
  0.614  Elena Torres is researching vector databases…                ✗

GRAPH  USES edges to PostgreSQL, filter to ORGANIZATION
  - Northwind Traders
  - Umbrella Corp
```

### Q3. People Daisy knows at companies using MongoDB  (2 hops)

```
VANILLA  query="people Daisy knows at companies using MongoDB", k=5
  0.783  Globex also runs on MongoDB.                             ← not a person
  0.782  Initech runs on MongoDB.                                  ← not a person
  0.727  Jordan Lee is the CTO of Globex…                          ← no "I know" signal
  0.720  Mia Rossi ran into Daisy Chen at the MongoDB conference.  ← adjacent
  0.715  Umbrella Corp uses PostgreSQL.                            ← wrong tech

GRAPH  KNOWS(Daisy, ?p) ∧ WORKS_AT(?p, ?c) ∧ USES(?c, MongoDB)
  - Jordan Lee @ Globex
  - Mia Rossi @ Initech
```

### Q4. Daisy's contacts at companies that compete with our customers  (3 hops)

```
VANILLA  query="Daisy contacts at companies competing with our customers", k=5
  0.660  Sarah Kim is the manager of Daisy Chen.                          ✗
  0.655  Daisy Chen knows Mia Rossi from Daisy Chen's previous job at Acme. ~ adjacent
  0.645  Daisy Chen had a great call with Elena Torres today.             ✗
  0.644  Umbrella Corp uses PostgreSQL.                                   ✗
  0.633  Mia Rossi ran into Daisy Chen at the MongoDB conference.          ~ adjacent

GRAPH  KNOWS(Daisy,?p) ∧ WORKS_AT(?p, ?c) ∧ COMPETES_WITH(?c, ?our_customer)
  - Elena Torres @ Northwind Traders  (competes with Initech)
  - Mia Rossi    @ Initech            (competes with Northwind Traders)
```

### Q5. Who is interested in vector databases?

```
VANILLA  query="who is interested in vector databases", k=5
  0.872  Elena Torres is researching vector databases for their next platform. ✓
  0.813  Jordan Lee emailed Daisy Chen about vector-search capabilities.       ✓
  0.808  Tom Becker runs the data platform at Umbrella Corp and is evaluating vector search. ✓
  0.703  Umbrella Corp uses PostgreSQL.                                         ✗
  0.701  Northwind Traders runs all their analytics on PostgreSQL.              ✗

GRAPH  hybrid search around "vector databases"
  - Jordan Lee     (relationship_expansion, score 1.10)
  - Elena Torres   (text_similarity,        score 1.00)
  - Tom Becker     (relationship_expansion, score 1.00)
  - Marcus Webb    (relationship_expansion, score 0.60)
```

## Scoreboard

| # | Question | Hops | Vanilla | Graph |
|---|---|---|---|---|
| Q1 | Sarah's team | 1 | 1/5 + 2 adjacent | **2/2** |
| Q2 | PostgreSQL customers | 1 | 2/5 correct | **2/2** |
| Q3 | Known contacts at MongoDB cos | 2 | **0/5** | **2/2** |
| Q4 | Contacts at competitors | 3 | **0/5** | **2/2 with competitor named** |
| Q5 | Vector-DB leads | 1 | 3/5 buried | **3/3 ranked + Marcus via network** |

- **1-hop gap is real but narrow** — an LLM consumer can usually sift vanilla's 2/5 correct.
- **2- and 3-hop gap is categorical** — vanilla returns only passages that the LLM cannot join.

## The honest production-readiness picture

| Capability | Status |
|---|---|
| Container creation with `enable_graph`, strategies, ingest pipeline, KNN settings | ✅ |
| Automatic LLM extraction on `POST /memories?infer=true` (both vanilla + graph in one call) | ✅ |
| Real vanilla retrieval via `/memories/long-term/_semantic_search` | ✅ |
| Graph text + hybrid + traversal search | ✅ |
| Deterministic entity IDs (same name+type mentioned again → updates not duplicates) | ✅ |
| Custom relationship vocabulary via `custom_relationship_extraction_prompt` | ✅ (used here for COMPETES_WITH) |
| Tenant + owner isolation | ✅ |
| Unit tests (22) + end-to-end REST tests (9) | ✅ |
| Reusing `EntityDeduplicationService` for neural-based dedup (beyond deterministic IDs) | 🟡 service exists, not wired in by default |
| Native multi-hop DSL (client stitches 2-3 calls today) | 🟡 |
| Real `DELETE /memories/graph` (stub today) | 🟡 |

## Re-run

```bash
# With container already created (see ingest_demo.py for config)
python3 docs/demo/ingest_demo.py   # 3 POST /memories calls, ~30s total
python3 docs/demo/run_demo.py      # 5 Q&A pairs side-by-side
```
