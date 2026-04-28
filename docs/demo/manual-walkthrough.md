# Graph Memory — Manual Walkthrough

Step-by-step script for demo'ing the graph memory feature by hand. Every block
below is a single OpenSearch Dashboards → **Dev Tools** request: paste, hit the
green play arrow, watch the response. Order matters — each step depends on
state from the previous ones.

## Prerequisites

- OpenSearch 3.5 cluster running with ml-commons plugin from `local-test/3.5`
- Cluster launched with `LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6`
  (so KNN/faiss links against a modern libstdc++)
- `admin` user with password `RUIwNTVGTDc2MDhDM0JNRC4u`
- For LLM: an AWS account with Bedrock Claude Sonnet 4.6 access in `us-west-2`
  (credentials at `~/.aws/credentials`)

If you prefer curl over Dev Tools, prepend every request with:

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -X <METHOD> "https://localhost:9200<path>" \
  -H 'Content-Type: application/json' \
  -d '<body>'
```

---

## Step 1. Enable the agentic memory feature flag

```
PUT /_cluster/settings
{
  "persistent": {
    "plugins.ml_commons.agentic_memory_enabled": true,
    "plugins.ml_commons.only_run_on_ml_node": "false",
    "plugins.ml_commons.model_access_control_enabled": "true",
    "plugins.ml_commons.native_memory_threshold": "99",
    "plugins.ml_commons.allow_registering_model_via_url": "true",
    "plugins.ml_commons.allow_registering_model_via_local_file": "true"
  }
}
```

**Expect:** `{"acknowledged": true, ...}`

> 如果这步返回 403/500，先把 plugin 装对：zip 路径在
> `plugin/build/distributions/opensearch-ml-3.5.0.0.zip`。

---

## Step 2. Register & deploy the text embedding model

```
POST /_plugins/_ml/models/_register?deploy=true
{
  "name": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
  "version": "1.0.2",
  "model_format": "TORCH_SCRIPT"
}
```

**Expect:** `{"task_id": "...", "status": "CREATED"}`

Poll the task until `COMPLETED`:

```
GET /_plugins/_ml/tasks/<task_id_from_above>
```

When done the task response contains `model_id`. **Save it** (e.g. `BWxf0J0BTxnG51e8Pch9`).

Sanity-check the embedding model:

```
POST /_plugins/_ml/_predict/text_embedding/BWxf0J0BTxnG51e8Pch9
{
  "text_docs": ["hello"],
  "target_response": ["sentence_embedding"]
}
```

**Expect:** `inference_results[0].output[0].data` is a 384-element float array.

---

## Step 3. Create the Bedrock Claude Sonnet 4.6 connector

Replace `<ACCESS_KEY>` and `<SECRET_KEY>` with the values from
`~/.aws/credentials`.

```
POST /_plugins/_ml/connectors/_create
{
  "name": "Bedrock Claude Sonnet 4.6 Connector",
  "version": "1",
  "protocol": "aws_sigv4",
  "parameters": {
    "region": "us-west-2",
    "service_name": "bedrock",
    "max_tokens": 4096,
    "system_prompt": "",
    "user_prompt": ""
  },
  "credential": {
    "access_key": "<ACCESS_KEY>",
    "secret_key": "<SECRET_KEY>"
  },
  "actions": [
    {
      "action_type": "predict",
      "method": "POST",
      "url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.anthropic.claude-sonnet-4-6/converse",
      "headers": { "content-type": "application/json" },
      "request_body": "{\"system\":[{\"text\":\"${parameters.system_prompt}\"}],\"inferenceConfig\":{\"maxTokens\":${parameters.max_tokens}},\"messages\":[{\"role\":\"user\",\"content\":[{\"text\":\"${parameters.user_prompt}\"}]}]}"
    }
  ]
}
```

**Expect:** `{"connector_id": "..."}` — save it (e.g. `26Rp0J0BQ4F7Y_V3BOI8`).

> **Why the Converse API (`/converse`)?** Its response shape is
> `output.message.content[0].text` — matches ml-commons' default
> `llm_result_path`. Using the older `/invoke` endpoint works too but would
> require a custom `llm_result_path` like `$.output[0].dataAsMap.content[0].text`.

---

## Step 4. Register & deploy the LLM model

```
POST /_plugins/_ml/models/_register?deploy=true
{
  "name": "bedrock-claude-sonnet-4-6",
  "function_name": "remote",
  "description": "Bedrock Claude Sonnet 4.6",
  "connector_id": "26Rp0J0BQ4F7Y_V3BOI8"
}
```

**Expect:** `{"task_id": "...", "status": "COMPLETED", "model_id": "..."}` — save the
`model_id` (e.g. `3qRp0J0BQ4F7Y_V3OOKT`).

Smoke test:

```
POST /_plugins/_ml/models/3qRp0J0BQ4F7Y_V3OOKT/_predict
{
  "parameters": {
    "system_prompt": "You are a JSON generator.",
    "user_prompt": "Return {\"ok\":true}"
  }
}
```

**Expect:** `inference_results[0].output[0].dataAsMap.output.message.content[0].text`
contains JSON.

---

## Step 5. Create the demo memory container

This is where the graph feature gets wired in: `enable_graph: true`, a
`SEMANTIC` strategy so long-term memory gets populated in parallel, and a
custom relationship extraction prompt that includes `COMPETES_WITH`,
`INTERESTED_IN`, etc.

```
POST /_plugins/_ml/memory_containers/_create
{
  "name": "demo-graph-crm",
  "description": "Live CRM demo with vanilla + graph memory",
  "configuration": {
    "embedding_model_type": "TEXT_EMBEDDING",
    "embedding_model_id": "BWxf0J0BTxnG51e8Pch9",
    "embedding_dimension": 384,
    "llm_id": "3qRp0J0BQ4F7Y_V3OOKT",
    "index_prefix": "demo",
    "enable_graph": true,
    "use_system_index": false,
    "parameters": { "llm_result_path": "$.output.message.content[0].text" },
    "strategies": [
      { "type": "SEMANTIC", "enabled": true, "namespace": ["owner_id"] }
    ],
    "custom_relationship_extraction_prompt": "<ROLE>You are a relationship extraction agent for a sales CRM.</ROLE>\n\n<SCOPE>Extract relationships between entities mentioned in the conversation. Use clear, standardized relationship types.</SCOPE>\n\n<OUTPUT>\nReturn ONLY a single JSON object exactly as {\"relationships\": [{\"source\": \"entity1\", \"target\": \"entity2\", \"type\": \"RELATIONSHIP_TYPE\", \"confidence\": 0.8}]}.\nRelationship types (choose the MOST specific one): WORKS_AT, KNOWS, MANAGES, REPORTS_TO, USES, INTERESTED_IN, EVALUATING, COMPETES_WITH, PART_OF, LOCATED_IN, COLLABORATES_WITH, MET_AT, PREVIOUSLY_WORKED_AT, DEVELOPED_BY, OTHER.\nUse COMPETES_WITH when one company is described as competing with another.\nUse INTERESTED_IN when a person is researching or interested in a technology.\nUse EVALUATING when a person is actively evaluating a technology for adoption.\nUse PREVIOUSLY_WORKED_AT for past employment.\nUse MET_AT for events where people met.\nUse exact entity names as they appear in entity extraction.\nConfidence: 0.0-1.0.\nNo code fences, no extra text, one line only.\nIf no relationships found, return {\"relationships\": []}.\n</OUTPUT>"
  }
}
```

**Expect:** `{"memory_container_id": "...", "status": "created"}` — save
(e.g. `hV4x0Z0BrnKV8q3htQqt`). The examples below all use this id.

Verify all 6 indices were created and the ingest pipeline is attached:

```
GET /_cat/indices/demo-*?v&expand_wildcards=all&h=index,docs.count,status
```

**Expect:**

```
demo-memory-sessions
demo-memory-working
demo-memory-long-term
demo-memory-history
demo-memory-lpg-nodes
demo-memory-lpg-edges
```

```
GET /_ingest/pipeline/demo-memory-lpg-nodes-embedding
```

**Expect:** a `text_embedding` processor mapping `entity_name → entity_embedding`.

```
GET /demo-memory-lpg-nodes/_settings
```

**Expect:** `index.default_pipeline: "demo-memory-lpg-nodes-embedding"` and
`index.knn: "true"`.

---

## Step 6. Ingest the first call note with `infer: true`

This is the "money shot" — **one** REST call triggers four LLM calls on the
server (session summary, fact extraction, entity extraction, relationship
extraction) and populates all the memory indices.

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories
{
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Daisy Chen had a great call with Elena Torres today. Elena is VP of Engineering at Northwind Traders. Northwind runs all their analytics on PostgreSQL. Elena is researching vector databases for their next platform. Daisy met Elena at KubeCon last quarter. Sarah Kim is the manager of Daisy Chen, and Sarah said Daisy should loop in Ravi Patel from the SE team. Ravi used to work with Tom Becker at Acme five years ago; Tom is now Data Architect at Umbrella Corp."
        }
      ]
    }
  ],
  "infer": true,
  "namespace": { "owner_id": "admin" }
}
```

**Expect (returns in ~2 seconds):**

```json
{ "session_id": "...", "working_memory_id": "..." }
```

> Graph + long-term extraction runs in the background — **wait ~8 seconds**
> before running the next verification calls so Sonnet finishes.

### Step 6a. Verify long-term memory was populated

```
POST /demo-memory-long-term/_search
{
  "size": 10,
  "_source": ["memory"],
  "query": { "match_all": {} }
}
```

**Expect:** sentence-level facts like `"Elena Torres is VP of Engineering at
Northwind Traders."`

### Step 6b. Verify graph entities were populated

```
POST /demo-memory-lpg-nodes/_search
{
  "size": 20,
  "_source": { "excludes": ["entity_embedding"] },
  "query": { "match_all": {} }
}
```

**Expect:** entities like `{"entity_name": "Elena Torres", "entity_type":
"PERSON", "confidence": 0.99}`. Each doc also has a populated
`entity_embedding` (384 dims) — confirm by removing the `_source.excludes`.

### Step 6c. Verify graph relationships were populated

```
POST /demo-memory-lpg-edges/_search
{
  "size": 20,
  "query": { "match_all": {} }
}
```

**Expect:** edges like `elena-torres -[WORKS_AT]-> northwind-traders`,
`ravi-patel -[PREVIOUSLY_WORKED_AT]-> acme`, etc.

---

## Step 7. Ingest the other two call notes

Same endpoint, just different text. Run both and wait ~8 seconds after each.

**Call note 2 (Jordan / Globex):**

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories
{
  "messages": [
    { "role": "user", "content": [{ "type": "text",
      "text": "Jordan Lee from Globex emailed Daisy Chen about vector-search capabilities. Globex is a fintech firm that runs on MongoDB. Jordan is their CTO and wants to add semantic search to their trading product. Jordan mentioned Initech is doing something similar; Initech competes with Northwind Traders in retail analytics. Marcus Webb, one of Sarah Kim SDRs, booked a discovery call with Jordan for next Tuesday."
    }]}
  ],
  "infer": true,
  "namespace": { "owner_id": "admin" }
}
```

**Call note 3 (Mia / Initech):**

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories
{
  "messages": [
    { "role": "user", "content": [{ "type": "text",
      "text": "Mia Rossi at Initech ran into Daisy Chen at the MongoDB conference. Mia is Platform Lead at Initech and Daisy Chen knows her from Daisy Chen previous job at Acme. Initech runs on MongoDB like Globex does. Mia mentioned Tom Becker at Umbrella Corp is evaluating vector search; Tom runs the data platform there and Umbrella uses PostgreSQL."
    }]}
  ],
  "infer": true,
  "namespace": { "owner_id": "admin" }
}
```

After both, check counts:

```
GET /demo-memory-long-term,demo-memory-lpg-nodes,demo-memory-lpg-edges/_count
```

**Expect:** approximately `{"count": 22}` for long-term, `{"count": 18}` for
lpg-nodes, `{"count": 29}` for lpg-edges (±2 each — Sonnet isn't perfectly
deterministic).

---

## Step 8. Look at what Sonnet produced

### All distinct relationship types (shows that COMPETES_WITH fired)

```
POST /demo-memory-lpg-edges/_search
{
  "size": 0,
  "aggs": {
    "by_type": { "terms": { "field": "relationship_type", "size": 30 } }
  }
}
```

**Expect:** buckets for `WORKS_AT`, `USES`, `KNOWS`, `EVALUATING`,
`PREVIOUSLY_WORKED_AT`, `INTERESTED_IN`, `MET_AT`, `REPORTS_TO`, `MANAGES`,
`COMPETES_WITH`, etc.

### The single COMPETES_WITH edge

```
POST /demo-memory-lpg-edges/_search
{
  "size": 10,
  "query": { "term": { "relationship_type": "COMPETES_WITH" } }
}
```

**Expect:** one hit, `initech -[COMPETES_WITH]-> northwind-traders`.

### All entities grouped by type

```
POST /demo-memory-lpg-nodes/_search
{
  "size": 30,
  "_source": ["entity_name", "entity_type"],
  "query": { "match_all": {} },
  "sort": [{ "entity_type": "asc" }, { "entity_name.keyword": "asc" }]
}
```

**Expect:** PERSON × 7 (Daisy Chen, Sarah Kim, Ravi Patel, Marcus Webb, Elena
Torres, Jordan Lee, Mia Rossi, Tom Becker), ORGANIZATION × 5 (Northwind Traders,
Globex, Initech, Umbrella Corp, Acme), TECHNOLOGY × 4 (PostgreSQL, MongoDB,
Vector Database, ...), EVENT × 2 (KubeCon, MongoDB Conference), CONCEPT × 1-2
(Fintech).

---

## Step 9. The 5 demo queries — vanilla vs graph side-by-side

For each question, run the vanilla query first, then the graph query, and
observe the difference.

### Q1. "Who's on Sarah Kim's team?"

**Vanilla (real agentic-memory long-term semantic search):**

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/long-term/_semantic_search
{ "query": "who is on Sarah Kim team", "k": 5 }
```

Returns 5 sentences; top 3 are related but include Elena & Mia as noise.

**Graph: MANAGES-from-Sarah OR REPORTS_TO-Sarah**

First find Sarah's entity id:

```
POST /demo-memory-lpg-nodes/_search
{
  "size": 1,
  "query": { "bool": { "must": [
    { "match_phrase": { "entity_name": "Sarah Kim" } },
    { "term": { "entity_type": "PERSON" } }
  ]}}
}
```

Grab `entity_id` from the hit (e.g.
`ent:hV4x0Z0BrnKV8q3htQqt:person:sarah-kim`), then:

```
POST /demo-memory-lpg-edges/_search
{
  "size": 10,
  "query": {
    "bool": {
      "should": [
        { "bool": { "must": [
          { "term": { "source_entity": "ent:hV4x0Z0BrnKV8q3htQqt:person:sarah-kim" } },
          { "term": { "relationship_type": "MANAGES" } }
        ]}},
        { "bool": { "must": [
          { "term": { "target_entity": "ent:hV4x0Z0BrnKV8q3htQqt:person:sarah-kim" } },
          { "term": { "relationship_type": "REPORTS_TO" } }
        ]}}
      ],
      "minimum_should_match": 1
    }
  }
}
```

**Expect:** exactly 2 hits — Marcus Webb and Daisy Chen.

### Q2. "Which customers run on PostgreSQL?"

**Vanilla:**

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/long-term/_semantic_search
{ "query": "which customers use PostgreSQL", "k": 5 }
```

**Graph:**

```
POST /demo-memory-lpg-edges/_search
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        { "term": { "target_entity": "ent:hV4x0Z0BrnKV8q3htQqt:technology:postgresql" } },
        { "term": { "relationship_type": "USES" } }
      ]
    }
  }
}
```

**Expect:** two `source_entity` values — Northwind Traders and Umbrella Corp.

### Q3. "People Daisy knows at companies using MongoDB" (2-hop)

Vanilla returns 0/5 useful hits — it can't join.

**Graph — three chained calls:**

```
# 1. Who does Daisy KNOW (either direction)?
POST /demo-memory-lpg-edges/_search
{
  "size": 30,
  "query": {
    "bool": {
      "must": [{ "term": { "relationship_type": "KNOWS" } }],
      "should": [
        { "term": { "source_entity": "ent:hV4x0Z0BrnKV8q3htQqt:person:daisy-chen" } },
        { "term": { "target_entity": "ent:hV4x0Z0BrnKV8q3htQqt:person:daisy-chen" } }
      ],
      "minimum_should_match": 1
    }
  }
}
```

Collect the "other side" of each hit: call this set `DAISY_KNOWS`.

```
# 2. Those people's employers
POST /demo-memory-lpg-edges/_search
{
  "size": 20,
  "query": {
    "bool": {
      "must": [
        { "terms": { "source_entity": [ <DAISY_KNOWS> ] } },
        { "term": { "relationship_type": "WORKS_AT" } }
      ]
    }
  }
}
```

Build a map `person → company`.

```
# 3. Which of those companies USE MongoDB?
POST /demo-memory-lpg-edges/_search
{
  "size": 10,
  "query": {
    "bool": {
      "must": [
        { "term": { "target_entity": "ent:hV4x0Z0BrnKV8q3htQqt:technology:mongodb" } },
        { "term": { "relationship_type": "USES" } }
      ]
    }
  }
}
```

Intersect the companies with the map from step 2.

**Expect:** Jordan Lee @ Globex, Mia Rossi @ Initech.

### Q4. "Daisy's contacts at competitor companies" (3-hop)

Extend Q3's chain with one more step:

```
# 4. COMPETES_WITH pairs
POST /demo-memory-lpg-edges/_search
{
  "size": 10,
  "query": { "term": { "relationship_type": "COMPETES_WITH" } }
}
```

Intersect the companies from Q3 step 2 with any company in a competes-with
pair.

**Expect:** Elena Torres @ Northwind Traders (competes with Initech) and
Mia Rossi @ Initech (competes with Northwind Traders).

### Q5. "Who is interested in vector databases?" (the full graph search endpoint)

**Vanilla:**

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/long-term/_semantic_search
{ "query": "who is interested in vector databases", "k": 5 }
```

**Graph hybrid (one call — this shows the real agent-facing API shines):**

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/graph/_search
{ "query": "vector databases", "top_k": 10 }
```

**Expect:** `search_results[]` blending `text_similarity` matches with
`relationship_expansion` matches. Jordan Lee, Elena Torres, Tom Becker, Marcus
Webb appear with different `match_type` and `score`.

---

## Step 10. Traversal — "show me Elena's network"

The third `search_type`, great for "tell me about this person":

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/graph/_search
{
  "query": "Elena",
  "search_type": "traversal",
  "entity_id": "ent:hV4x0Z0BrnKV8q3htQqt:person:elena-torres",
  "max_depth": 2
}
```

**Expect:** `entities` array with Elena + everyone reachable within 2 hops
(Northwind Traders, Daisy Chen, PostgreSQL, Vector Database, KubeCon, …) and
`relationships` listing the bridging edges.

---

## Step 11. Text-only entity lookup

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/graph/_search
{ "query": "Elena", "search_type": "text", "top_k": 5 }
```

**Expect:** entity neighbors by embedding-space proximity to "Elena" — Elena
Torres first, then other PERSON entities.

---

## Step 12. List all entities (GET convenience endpoint)

```
GET /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/graph/entities?query=*&top_k=50
```

**Expect:** same shape as step 11 but via GET, useful for a "show me everything"
admin view.

---

## Step 13. Sanity-check tenant isolation

Try searching graph data **from the wrong container**:

```
POST /_plugins/_ml/memory_containers/does-not-exist/memories/graph/_search
{ "query": "alice", "search_type": "text", "top_k": 5 }
```

**Expect:** HTTP 404 `"Memory container not found"`.

Try with the flag off:

```
PUT /_cluster/settings
{ "persistent": { "plugins.ml_commons.agentic_memory_enabled": false } }
```

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/graph/_search
{ "query": "alice", "top_k": 5 }
```

**Expect:** HTTP 403 `"Agentic memory feature is not enabled"`.

Turn it back on:

```
PUT /_cluster/settings
{ "persistent": { "plugins.ml_commons.agentic_memory_enabled": true } }
```

---

## Step 14. Clean up (optional)

```
POST /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt/memories/graph
POST /_plugins/_ml/memory_containers/_search
{ "size": 20, "query": { "match_all": {} } }
DELETE /_plugins/_ml/memory_containers/hV4x0Z0BrnKV8q3htQqt
```

> **Note:** `DELETE /memories/graph` today returns a stub success — real
> delete-by-query on `lpg-nodes`/`lpg-edges` is on the TODO list.
> Container-level `DELETE` fully cleans up indices and pipeline.

---

## Summary: what you just demoed

| Step | Demonstrates |
|---|---|
| 1–4 | Standard ml-commons setup — nothing graph-specific |
| 5 | **One config knob** (`enable_graph: true` + strategies + custom prompt) wires in the entire graph feature |
| 6–7 | **One API call** (`POST /memories`) populates both vanilla + graph memory, automatically via Sonnet |
| 8 | Verify what the LLM extracted — entities, relationships, typed vocabulary |
| 9 Q1–Q5 | Vanilla vs graph side-by-side — 1-hop, 2-hop, 3-hop, hybrid |
| 10–12 | All three graph-search modes + convenience endpoints |
| 13 | Access control + tenant isolation still work |

**Runtime of the full script:** ~2–3 minutes of LLM time (steps 6–7) plus
whatever you spend reading results.
