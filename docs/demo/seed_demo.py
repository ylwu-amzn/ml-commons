#!/usr/bin/env python3
"""Seed the sales-CRM demo dataset into both vanilla and graph memory."""
import json
import subprocess
import sys
import time

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
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return {"raw": r.stdout, "err": r.stderr}


def embed(text):
    r = curl(
        "POST",
        f"/_plugins/_ml/_predict/text_embedding/{EMB_MODEL}",
        {"text_docs": [text], "target_response": ["sentence_embedding"]},
    )
    return r["inference_results"][0]["output"][0]["data"]


# ---- Domain: a sales rep named Daisy's CRM notes ----
# Entities: people (our sales team + prospects), companies, technologies
people = [
    ("p-daisy",   "Daisy Chen",     "person",   "AE, owns Enterprise West"),
    ("p-sarah",   "Sarah Kim",      "person",   "Sales Manager, Daisy's boss"),
    ("p-marcus",  "Marcus Webb",    "person",   "SDR on Sarah's team"),
    ("p-ravi",    "Ravi Patel",     "person",   "SE, supports Daisy"),
    ("p-elena",   "Elena Torres",   "person",   "VP Eng at Northwind, prospect"),
    ("p-jordan",  "Jordan Lee",     "person",   "CTO at Globex, prospect"),
    ("p-mia",     "Mia Rossi",      "person",   "Platform Lead at Initech"),
    ("p-tom",     "Tom Becker",     "person",   "Data Architect at Umbrella"),
]
companies = [
    ("c-northwind", "Northwind Traders", "company", "Retail, 2000 employees"),
    ("c-globex",    "Globex",            "company", "Fintech, series C"),
    ("c-initech",   "Initech",           "company", "SaaS competitor"),
    ("c-umbrella",  "Umbrella Corp",     "company", "Pharma, enterprise prospect"),
]
techs = [
    ("t-postgres", "PostgreSQL",        "tech",    "relational DB"),
    ("t-mongo",    "MongoDB",           "tech",    "document DB"),
    ("t-vectordb", "vector databases",  "tech",    "VDB category"),
    ("t-opensearch","OpenSearch",       "tech",    "search + vector"),
]
all_entities = people + companies + techs

# Relationships: (rid, source, target, type, note)
relationships = [
    ("r01", "p-marcus",  "p-sarah",    "REPORTS_TO",     "Marcus is on Sarah's team"),
    ("r02", "p-daisy",   "p-sarah",    "REPORTS_TO",     "Daisy is on Sarah's team"),
    ("r03", "p-ravi",    "p-sarah",    "REPORTS_TO",     "Ravi is on Sarah's team"),
    ("r04", "p-elena",   "c-northwind", "WORKS_AT",       "Elena is VP Eng at Northwind"),
    ("r05", "p-jordan",  "c-globex",   "WORKS_AT",       "Jordan is CTO at Globex"),
    ("r06", "p-mia",     "c-initech",  "WORKS_AT",       "Mia is Platform Lead at Initech"),
    ("r07", "p-tom",     "c-umbrella", "WORKS_AT",       "Tom is Data Architect at Umbrella"),
    ("r08", "c-northwind","t-postgres","USES_TECH",      "Northwind runs on Postgres"),
    ("r09", "c-globex",  "t-mongo",    "USES_TECH",      "Globex runs on MongoDB"),
    ("r10", "c-initech", "t-mongo",    "USES_TECH",      "Initech runs on MongoDB"),
    ("r11", "c-umbrella","t-postgres", "USES_TECH",      "Umbrella runs on Postgres"),
    ("r12", "c-initech", "c-northwind","COMPETES_WITH",  "Initech competes with Northwind"),
    ("r13", "p-elena",   "t-vectordb", "INTERESTED_IN",  "Elena is researching vector databases"),
    ("r14", "p-jordan",  "t-vectordb", "INTERESTED_IN",  "Jordan wants to add vector search"),
    ("r15", "p-daisy",   "p-elena",    "KNOWS",          "Daisy met Elena at KubeCon"),
    ("r16", "p-daisy",   "p-mia",      "KNOWS",          "Daisy knows Mia from previous job"),
    ("r17", "p-ravi",    "p-tom",      "KNOWS",          "Ravi and Tom worked together at Acme"),
]

# The exact same facts as plain sentences (what vanilla memory would store)
facts = [
    "Sarah Kim is my sales manager.",
    "Marcus Webb is an SDR who reports to Sarah Kim.",
    "Daisy Chen is an AE who reports to Sarah Kim.",
    "Ravi Patel is a sales engineer who reports to Sarah Kim.",
    "Elena Torres is VP of Engineering at Northwind Traders.",
    "Jordan Lee is CTO at Globex.",
    "Mia Rossi is the Platform Lead at Initech.",
    "Tom Becker is the Data Architect at Umbrella Corp.",
    "Northwind Traders runs on PostgreSQL.",
    "Globex runs on MongoDB.",
    "Initech runs on MongoDB.",
    "Umbrella Corp runs on PostgreSQL.",
    "Initech competes with Northwind Traders in the retail analytics space.",
    "Elena Torres is researching vector databases for their next platform.",
    "Jordan Lee wants to add vector search to Globex's product.",
    "I met Elena Torres at KubeCon last quarter.",
    "I know Mia Rossi from my previous job at Acme.",
    "Ravi Patel worked with Tom Becker at Acme five years ago.",
]


def main():
    print("---- seeding graph entities ----")
    for eid, name, etype, _note in all_entities:
        emb = embed(name)
        doc = {
            "entity_id": eid,
            "entity_name": name,
            "entity_type": etype,
            "confidence": 0.95,
            "memory_container_id": CID,
            "owner_id": "admin",
            "created_time": 1777320000,
            "updated_time": 1777320000,
            "mention_count": 1,
            "entity_embedding": emb,
        }
        r = curl("POST", f"/{NODES}/_doc/{eid}?refresh=true", doc)
        assert r.get("result") in ("created", "updated"), f"node fail {eid}: {r}"
    print(f"  {len(all_entities)} entities indexed")

    print("---- seeding graph edges ----")
    for rid, src, tgt, rtype, _note in relationships:
        doc = {
            "relationship_id": rid,
            "source_entity": src,
            "target_entity": tgt,
            "relationship_type": rtype,
            "confidence": 0.9,
            "memory_container_id": CID,
            "owner_id": "admin",
            "created_time": 1777320000,
            "updated_time": 1777320000,
            "is_active": True,
        }
        r = curl("POST", f"/{EDGES}/_doc/{rid}?refresh=true", doc)
        assert r.get("result") in ("created", "updated"), f"edge fail {rid}: {r}"
    print(f"  {len(relationships)} edges indexed")

    print("---- seeding vanilla memory (same facts as text) ----")
    for i, fact in enumerate(facts):
        emb = embed(fact)
        doc = {
            "text": fact,
            "embedding": emb,
            "owner_id": "admin",
            "memory_container_id": CID,
        }
        r = curl("POST", f"/{VANILLA}/_doc/vanilla-{i}?refresh=true", doc)
        assert r.get("result") in ("created", "updated"), f"vanilla fail {i}: {r}"
    print(f"  {len(facts)} vanilla-memory documents indexed")

    print("---- done ----")


if __name__ == "__main__":
    main()
