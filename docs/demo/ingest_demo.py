#!/usr/bin/env python3
"""
Graph-memory demo — ingest 3 call notes through the full LLM extraction pipeline.

Expects:
 - OpenSearch cluster on https://localhost:9200 with admin creds below
 - Text embedding model registered & deployed: EMB_MODEL
 - LLM model (Bedrock Claude Sonnet 4.6 via Converse API) registered: LLM_MODEL
 - Memory container created with enable_graph=true: CID
 - Parallel vanilla-memory index for the baseline comparison: VANILLA

The same 3 call notes are used for both stores:
 - Graph store: server-side LLM extraction via POST /memories?infer=true
 - Vanilla store: sentence-chunked + embedded for neural search
"""
import json
import subprocess

AUTH = "admin:RUIwNTVGTDc2MDhDM0JNRC4u"
BASE = "https://localhost:9200"
EMB_MODEL = "BWxf0J0BTxnG51e8Pch9"
LLM_MODEL = "3qRp0J0BQ4F7Y_V3OOKT"
CID = "_YWw0J0BFKsTOMMlBkoK"
VANILLA = "demo-vanilla-memory"


def curl(method, path, body=None):
    cmd = ["curl", "-sk", "-u", AUTH, "-X", method, f"{BASE}{path}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)


def embed(text):
    r = curl(
        "POST",
        f"/_plugins/_ml/_predict/text_embedding/{EMB_MODEL}",
        {"text_docs": [text], "target_response": ["sentence_embedding"]},
    )
    return r["inference_results"][0]["output"][0]["data"]


# The three realistic call notes (what Daisy's assistant captured from her calls)
call_notes = [
    "Had a great call with Elena Torres today. She is VP of Engineering at Northwind Traders. "
    "Northwind runs all their analytics on PostgreSQL. Elena is researching vector databases for "
    "their next platform. I met her at KubeCon last quarter. My manager Sarah Kim said I should "
    "loop in Ravi Patel from our SE team. Ravi used to work with Tom Becker at Acme five years "
    "ago; Tom is now Data Architect at Umbrella Corp.",

    "Jordan Lee from Globex emailed about our new vector-search capabilities. Globex is a fintech "
    "firm and they run on MongoDB. Jordan is their CTO and wants to add semantic search to their "
    "trading product. He mentioned Initech is doing something similar — Initech competes with "
    "Northwind Traders in retail analytics. Marcus Webb, one of Sarah Kim's SDRs, booked a "
    "discovery call with Jordan for next Tuesday.",

    "Mia Rossi at Initech ran into me at the MongoDB conference. Mia is Platform Lead at Initech "
    "and I know her from my previous job at Acme. Initech runs on MongoDB like Globex does. Mia "
    "mentioned Tom Becker at Umbrella Corp is evaluating vector search — Tom runs the data "
    "platform there and Umbrella uses PostgreSQL.",
]


def ingest_graph():
    print("---- graph memory: ingest 3 call notes via LLM extraction ----")
    for i, note in enumerate(call_notes, 1):
        r = curl(
            "POST",
            f"/_plugins/_ml/memory_containers/{CID}/memories",
            {"messages": [{"role": "user", "content": [{"type": "text", "text": note}]}],
             "infer": True},
        )
        print(f"  call-note {i}: session_id={r.get('session_id')}")


def ingest_vanilla():
    print("---- vanilla memory: sentence-chunk + embed the same notes ----")
    # Split on '. ' to mimic a naive sentence-chunk retriever
    sentences = []
    for note in call_notes:
        for s in note.split(". "):
            s = s.strip().rstrip(".")
            if s:
                sentences.append(s + ".")
    for i, s in enumerate(sentences):
        curl("POST", f"/{VANILLA}/_doc/v-{i}?refresh=true", {"text": s, "embedding": embed(s)})
    print(f"  {len(sentences)} vanilla-memory documents indexed")


def report_graph_state():
    nodes = curl("POST", f"/demo-memory-lpg-nodes/_count", None)["count"]
    edges = curl("POST", f"/demo-memory-lpg-edges/_count", None)["count"]
    vanilla = curl("POST", f"/{VANILLA}/_count", None)["count"]
    print("\n---- resulting state ----")
    print(f"  graph entities:       {nodes}")
    print(f"  graph relationships:  {edges}")
    print(f"  vanilla documents:    {vanilla}")


if __name__ == "__main__":
    ingest_graph()
    ingest_vanilla()
    report_graph_state()
