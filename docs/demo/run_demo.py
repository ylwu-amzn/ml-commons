#!/usr/bin/env python3
"""Run demo queries against both vanilla and graph memory; print results."""
import json
import subprocess
import sys

AUTH = "admin:RUIwNTVGTDc2MDhDM0JNRC4u"
BASE = "https://localhost:9200"
EMB_MODEL = "BWxf0J0BTxnG51e8Pch9"
CID = "4Rh60J0B8jwm23PNlvB4"
VANILLA = "demo-vanilla-memory"
NODES = "e2e2-memory-lpg-nodes"
EDGES = "e2e2-memory-lpg-edges"


def curl(method, path, body=None):
    cmd = ["curl", "-sk", "-u", AUTH, "-X", method, f"{BASE}{path}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(r.stdout)


def vanilla_search(query, k=5):
    """Neural top-k against the vanilla text index."""
    body = {
        "size": k,
        "query": {
            "neural": {
                "embedding": {
                    "query_text": query,
                    "model_id": EMB_MODEL,
                    "k": k,
                }
            }
        },
        "_source": ["text"],
    }
    r = curl("POST", f"/{VANILLA}/_search", body)
    hits = r.get("hits", {}).get("hits", [])
    return [(h["_score"], h["_source"]["text"]) for h in hits]


def graph_search(body):
    return curl(
        "POST",
        f"/_plugins/_ml/memory_containers/{CID}/memories/graph/_search",
        body,
    )


def by_entity_lookup(entity_id):
    body = {"query": {"term": {"entity_id": entity_id}}, "size": 1}
    return curl("POST", f"/{NODES}/_search", body)


def edge_lookup(query):
    return curl("POST", f"/{EDGES}/_search", {"size": 20, "query": query})


def node_lookup(query):
    return curl("POST", f"/{NODES}/_search", {"size": 20, "query": query})


def entities_in(ids):
    """Fetch entity_name for a list of entity_ids via mget."""
    if not ids:
        return {}
    body = {"docs": [{"_index": NODES, "_id": i} for i in ids]}
    r = curl("POST", "/_mget", body)
    out = {}
    for d in r.get("docs", []):
        if d.get("found"):
            s = d["_source"]
            out[s["entity_id"]] = (s["entity_name"], s["entity_type"])
    return out


# --------- demo queries ---------

def q1_sarahs_team():
    print("\n" + "=" * 70)
    print("Q1: Who's on Sarah Kim's team?")
    print("=" * 70)
    print("\n[VANILLA] neural search: 'who is on Sarah Kim team'")
    for score, text in vanilla_search("who is on Sarah Kim team", k=5):
        print(f"  {score:.3f}  {text}")

    print("\n[GRAPH] lookup Sarah's entity_id, then find incoming REPORTS_TO edges")
    body = {
        "bool": {
            "must": [
                {"term": {"target_entity": "p-sarah"}},
                {"term": {"relationship_type": "REPORTS_TO"}},
            ]
        }
    }
    r = edge_lookup(body)
    hits = r.get("hits", {}).get("hits", [])
    src_ids = [h["_source"]["source_entity"] for h in hits]
    names = entities_in(src_ids)
    print(f"  (found {len(src_ids)} direct reports)")
    for sid in src_ids:
        n, _t = names.get(sid, (sid, "?"))
        print(f"  - {n}  (entity_id={sid})")


def q2_postgres_customers():
    print("\n" + "=" * 70)
    print("Q2: Which customers run on PostgreSQL?")
    print("=" * 70)
    print("\n[VANILLA] neural search: 'which customers use PostgreSQL'")
    for score, text in vanilla_search("which customers use PostgreSQL", k=5):
        print(f"  {score:.3f}  {text}")

    print("\n[GRAPH] edges of type USES_TECH with target = t-postgres")
    body = {
        "bool": {
            "must": [
                {"term": {"target_entity": "t-postgres"}},
                {"term": {"relationship_type": "USES_TECH"}},
            ]
        }
    }
    r = edge_lookup(body)
    src_ids = [h["_source"]["source_entity"] for h in r.get("hits", {}).get("hits", [])]
    names = entities_in(src_ids)
    for sid in src_ids:
        n, _t = names.get(sid, (sid, "?"))
        print(f"  - {n}")


def q3_people_i_know_at_mongo_companies():
    print("\n" + "=" * 70)
    print("Q3: Find people I (Daisy) know at companies using MongoDB")
    print("=" * 70)
    print("\n[VANILLA] neural search: 'people I know at companies using MongoDB'")
    for score, text in vanilla_search("people I know at companies using MongoDB", k=5):
        print(f"  {score:.3f}  {text}")

    print("\n[GRAPH] 2-hop: USES_TECH=mongo -> company -> WORKS_AT <- person <- KNOWS <- daisy")
    # 1. companies that use mongo
    r = edge_lookup({
        "bool": {"must": [
            {"term": {"target_entity": "t-mongo"}},
            {"term": {"relationship_type": "USES_TECH"}},
        ]}
    })
    mongo_cos = [h["_source"]["source_entity"] for h in r["hits"]["hits"]]

    # 2. people who WORK_AT any of those companies
    r = edge_lookup({
        "bool": {"must": [
            {"terms": {"target_entity": mongo_cos}},
            {"term": {"relationship_type": "WORKS_AT"}},
        ]}
    })
    workers = {h["_source"]["source_entity"]: h["_source"]["target_entity"]
               for h in r["hits"]["hits"]}

    # 3. Of those, ones Daisy KNOWS
    r = edge_lookup({
        "bool": {"must": [
            {"term": {"source_entity": "p-daisy"}},
            {"term": {"relationship_type": "KNOWS"}},
            {"terms": {"target_entity": list(workers.keys())}},
        ]}
    })
    matches = [h["_source"]["target_entity"] for h in r["hits"]["hits"]]
    names = entities_in(matches + mongo_cos)
    for pid in matches:
        p_name, _ = names.get(pid, (pid, "?"))
        co = workers[pid]
        co_name, _ = names.get(co, (co, "?"))
        print(f"  - {p_name} @ {co_name}")


def q4_alice_colleagues_at_competitors():
    print("\n" + "=" * 70)
    print("Q4: Who are Daisy's contacts at companies that compete with our customers?")
    print("=" * 70)
    print("\n[VANILLA] neural search")
    for score, text in vanilla_search("Daisy contacts at competing companies", k=5):
        print(f"  {score:.3f}  {text}")

    print("\n[GRAPH] 3-hop: Daisy KNOWS -> person WORKS_AT -> company COMPETES_WITH -> our customer")
    # people daisy knows
    r = edge_lookup({
        "bool": {"must": [
            {"term": {"source_entity": "p-daisy"}},
            {"term": {"relationship_type": "KNOWS"}},
        ]}
    })
    daisy_knows = [h["_source"]["target_entity"] for h in r["hits"]["hits"]]

    # their employers
    r = edge_lookup({
        "bool": {"must": [
            {"terms": {"source_entity": daisy_knows}},
            {"term": {"relationship_type": "WORKS_AT"}},
        ]}
    })
    person_to_company = {h["_source"]["source_entity"]: h["_source"]["target_entity"]
                         for h in r["hits"]["hits"]}

    # companies that compete with those
    r = edge_lookup({
        "bool": {"should": [
            {"terms": {"source_entity": list(person_to_company.values())}},
            {"terms": {"target_entity": list(person_to_company.values())}},
        ],
        "must": [{"term": {"relationship_type": "COMPETES_WITH"}}],
        "minimum_should_match": 1}
    })
    competing_pairs = [(h["_source"]["source_entity"], h["_source"]["target_entity"])
                       for h in r["hits"]["hits"]]
    competing_companies = set()
    for a, b in competing_pairs:
        competing_companies.update([a, b])

    # restrict to contacts whose employer is in a competing pair
    matches = [(p, c) for p, c in person_to_company.items() if c in competing_companies]
    names = entities_in([p for p, _ in matches] + [c for _, c in matches])
    for p, c in matches:
        p_name, _ = names.get(p, (p, "?"))
        c_name, _ = names.get(c, (c, "?"))
        # find the competitor
        comp = next((x for a, b in competing_pairs
                     for x in (a, b) if x != c and c in (a, b)), "?")
        c2_name, _ = names.get(comp, (comp, "?"))
        print(f"  - {p_name} @ {c_name}  (competes with {c2_name})")


def q5_vectordb_interest():
    print("\n" + "=" * 70)
    print("Q5: Who is interested in vector databases? (for our new pitch)")
    print("=" * 70)
    print("\n[VANILLA] neural search: 'who is interested in vector databases'")
    for score, text in vanilla_search("who is interested in vector databases", k=5):
        print(f"  {score:.3f}  {text}")

    print("\n[GRAPH] INTERESTED_IN -> t-vectordb")
    r = edge_lookup({
        "bool": {"must": [
            {"term": {"target_entity": "t-vectordb"}},
            {"term": {"relationship_type": "INTERESTED_IN"}},
        ]}
    })
    src_ids = [h["_source"]["source_entity"] for h in r["hits"]["hits"]]

    # enrich with their employer
    r = edge_lookup({
        "bool": {"must": [
            {"terms": {"source_entity": src_ids}},
            {"term": {"relationship_type": "WORKS_AT"}},
        ]}
    })
    employer = {h["_source"]["source_entity"]: h["_source"]["target_entity"]
                for h in r["hits"]["hits"]}

    names = entities_in(src_ids + list(employer.values()))
    for pid in src_ids:
        p_name, _ = names.get(pid, (pid, "?"))
        co = employer.get(pid, "")
        co_name = names.get(co, (co, ""))[0] if co else ""
        suffix = f" (at {co_name})" if co_name else ""
        print(f"  - {p_name}{suffix}")


def main():
    q1_sarahs_team()
    q2_postgres_customers()
    q3_people_i_know_at_mongo_companies()
    q4_alice_colleagues_at_competitors()
    q5_vectordb_interest()


if __name__ == "__main__":
    main()
