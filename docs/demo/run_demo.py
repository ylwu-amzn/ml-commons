#!/usr/bin/env python3
"""Run the 5 graph-memory demo queries against vanilla vs graph stores.

Vanilla uses a naive neural-text index as a stand-in for agentic-memory's long_term store.
Graph uses the real /_plugins/_ml/memory_containers/{id}/memories/graph/_search endpoint plus
a few stitched edge-index queries for 2- and 3-hop questions (a native multi-hop DSL is
out of scope for this release).
"""
import json
import subprocess

AUTH = "admin:RUIwNTVGTDc2MDhDM0JNRC4u"
BASE = "https://localhost:9200"
EMB_MODEL = "BWxf0J0BTxnG51e8Pch9"
CID = "_YWw0J0BFKsTOMMlBkoK"
VANILLA = "demo-vanilla-memory"
NODES = "demo-memory-lpg-nodes"
EDGES = "demo-memory-lpg-edges"


def curl(method, path, body=None):
    cmd = ["curl", "-sk", "-u", AUTH, "-X", method, f"{BASE}{path}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)


def vanilla(q, k=5):
    r = curl("POST", f"/{VANILLA}/_search", {
        "size": k,
        "query": {"neural": {"embedding": {"query_text": q, "model_id": EMB_MODEL, "k": k}}},
        "_source": ["text"],
    })
    return [(h["_score"], h["_source"]["text"]) for h in r.get("hits", {}).get("hits", [])]


def graph_search(body):
    return curl("POST", f"/_plugins/_ml/memory_containers/{CID}/memories/graph/_search", body)


def edges(query):
    return curl("POST", f"/{EDGES}/_search", {"size": 30, "query": query})


def nodes(query):
    return curl("POST", f"/{NODES}/_search",
                {"size": 30, "_source": {"excludes": ["entity_embedding"]}, "query": query})


def name_of(entity_id):
    r = curl("POST", f"/{NODES}/_search",
             {"size": 1, "query": {"term": {"entity_id": entity_id}},
              "_source": ["entity_name", "entity_type"]})
    h = r.get("hits", {}).get("hits", [])
    return (h[0]["_source"]["entity_name"], h[0]["_source"]["entity_type"]) if h else (None, None)


def id_for(name, etype=None):
    must = [{"match_phrase": {"entity_name": name}}]
    if etype:
        must.append({"term": {"entity_type": etype}})
    r = nodes({"bool": {"must": must}})
    h = r.get("hits", {}).get("hits", [])
    return h[0]["_source"]["entity_id"] if h else None


def sep(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def q1():
    sep("Q1: Who's on Sarah Kim's team?")
    print("\n[VANILLA] neural: 'who is on Sarah Kim team'")
    for s, t in vanilla("who is on Sarah Kim team", 5):
        print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] MANAGES edges from Sarah Kim")
    sarah = id_for("Sarah Kim", "PERSON")
    r = edges({"bool": {"must": [{"term": {"source_entity": sarah}},
                                  {"term": {"relationship_type": "MANAGES"}}]}})
    for h in r["hits"]["hits"]:
        n, _ = name_of(h["_source"]["target_entity"])
        print(f"  - {n}")


def q2():
    sep("Q2: Which customers run on PostgreSQL?")
    print("\n[VANILLA] neural: 'which customers use PostgreSQL'")
    for s, t in vanilla("which customers use PostgreSQL", 5):
        print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] USES edges to PostgreSQL, filter to ORGANIZATION")
    pg = id_for("PostgreSQL", "TECHNOLOGY")
    r = edges({"bool": {"must": [{"term": {"target_entity": pg}},
                                  {"term": {"relationship_type": "USES"}}]}})
    for h in r["hits"]["hits"]:
        n, t = name_of(h["_source"]["source_entity"])
        if t == "ORGANIZATION":
            print(f"  - {n}")


def q3():
    sep("Q3: People I have a connection to, at companies using MongoDB")
    print("\n[VANILLA] neural: 'people I know at companies using MongoDB'")
    for s, t in vanilla("people I know at companies using MongoDB", 5):
        print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] 2-hop: USES(company, Mongo) AND WORKS_AT(person, company) AND known")
    mongo = id_for("MongoDB", "TECHNOLOGY")
    r = edges({"bool": {"must": [{"term": {"target_entity": mongo}},
                                  {"term": {"relationship_type": "USES"}}]}})
    cos = [h["_source"]["source_entity"] for h in r["hits"]["hits"]]
    r = edges({"bool": {"must": [{"terms": {"target_entity": cos}},
                                  {"term": {"relationship_type": "WORKS_AT"}}]}})
    people = [(h["_source"]["source_entity"], h["_source"]["target_entity"])
              for h in r["hits"]["hits"]]
    r = edges({"bool": {"must": [{"term": {"relationship_type": "KNOWS"}}]}})
    known = set()
    for h in r["hits"]["hits"]:
        known.add(h["_source"]["source_entity"])
        known.add(h["_source"]["target_entity"])
    for p, c in people:
        p_n, p_t = name_of(p)
        c_n, _ = name_of(c)
        if p_t == "PERSON" and p in known:
            print(f"  - {p_n} @ {c_n}")


def q4():
    sep("Q4: Any contacts at companies that compete with our customers?")
    print("\n[VANILLA] neural: 'contacts at companies that compete with our customers'")
    for s, t in vanilla("contacts at companies that compete with our customers", 5):
        print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] hybrid entity lookup for 'competitor'")
    g = graph_search({"query": "competitor", "search_type": "text", "top_k": 5})
    for e in g.get("entities", []):
        print(f"  - {e['name']} ({e['type']})")
    print("\n  (Graph doesn't have a typed COMPETES_WITH edge — Sonnet's default vocabulary")
    print("   labeled the competition as OTHER. Fix: custom relationship extraction prompt.)")


def q5():
    sep("Q5: Who is interested in vector databases?")
    print("\n[VANILLA] neural: 'who is interested in vector databases'")
    for s, t in vanilla("who is interested in vector databases", 5):
        print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] hybrid search around 'vector databases'")
    g = graph_search({"query": "vector databases", "top_k": 10})
    seen = set()
    for sr in g.get("search_results", []):
        e = sr["entity"]
        if e["type"] == "PERSON" and e["entity_id"] not in seen:
            seen.add(e["entity_id"])
            print(f"  - {e['name']}  ({sr['match_type']}, score {sr['score']:.2f})")


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, q5):
        fn()
