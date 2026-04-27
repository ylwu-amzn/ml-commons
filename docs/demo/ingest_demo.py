#!/usr/bin/env python3
"""Ingest 3 realistic call notes through the agentic-memory API so the LLM builds
both long-term memory (vanilla) AND the graph in one call.

Prerequisites:
  - OpenSearch 3.5 cluster with ml-commons plugin (local-test/3.5 branch)
  - Text embedding model deployed:      EMB_MODEL
  - LLM model (Bedrock Claude Sonnet):  LLM_MODEL
  - Memory container CID created with:
      * enable_graph: true
      * at least one SEMANTIC strategy with namespace ['owner_id']
      * custom_relationship_extraction_prompt including COMPETES_WITH
      * llm_result_path:  $.output.message.content[0].text  (Bedrock Converse)

Each POST /memories with infer=true returns quickly; in the background the server:
  1) Summarizes and writes session + working memory
  2) Runs each SEMANTIC strategy to extract facts into long-term memory
  3) (Parallel) Runs GraphProcessingService to extract entities + typed edges
     into the lpg-nodes/lpg-edges indices

Both memory layers come from the same Sonnet call chain over the same 3 notes.
"""
import json, subprocess, time

AUTH = "admin:RUIwNTVGTDc2MDhDM0JNRC4u"
BASE = "https://localhost:9200"
CID  = "hV4x0Z0BrnKV8q3htQqt"  # your container id

NOTES = [
    "Daisy Chen had a great call with Elena Torres today. Elena is VP of Engineering at "
    "Northwind Traders. Northwind runs all their analytics on PostgreSQL. Elena is researching "
    "vector databases for their next platform. Daisy met Elena at KubeCon last quarter. Sarah Kim "
    "is the manager of Daisy Chen, and Sarah said Daisy should loop in Ravi Patel from the SE team. "
    "Ravi used to work with Tom Becker at Acme five years ago; Tom is now Data Architect at Umbrella Corp.",

    "Jordan Lee from Globex emailed Daisy Chen about vector-search capabilities. Globex is a fintech "
    "firm that runs on MongoDB. Jordan is their CTO and wants to add semantic search to their trading "
    "product. Jordan mentioned Initech is doing something similar; Initech competes with Northwind "
    "Traders in retail analytics. Marcus Webb, one of Sarah Kim SDRs, booked a discovery call with "
    "Jordan for next Tuesday.",

    "Mia Rossi at Initech ran into Daisy Chen at the MongoDB conference. Mia is Platform Lead at "
    "Initech and Daisy Chen knows her from Daisy Chen previous job at Acme. Initech runs on MongoDB "
    "like Globex does. Mia mentioned Tom Becker at Umbrella Corp is evaluating vector search; Tom runs "
    "the data platform there and Umbrella uses PostgreSQL.",
]


def curl(method, path, body=None):
    cmd = ["curl", "-sk", "-u", AUTH, "-X", method, f"{BASE}{path}"]
    if body is not None:
        cmd += ["-H", "Content-Type: application/json", "-d", json.dumps(body)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)


def ingest():
    for i, note in enumerate(NOTES, 1):
        r = curl("POST", f"/_plugins/_ml/memory_containers/{CID}/memories", {
            "messages": [{"role": "user", "content": [{"type": "text", "text": note}]}],
            "infer": True,
            "namespace": {"owner_id": "admin"},
        })
        print(f"  call-note {i}: session={r.get('session_id')}")
        # Give the background graph + long-term extraction time to finish before next POST
        time.sleep(6)


def report():
    for idx, label in [
        ("demo-memory-long-term", "long-term facts"),
        ("demo-memory-lpg-nodes", "graph entities"),
        ("demo-memory-lpg-edges", "graph edges"),
    ]:
        c = curl("POST", f"/{idx}/_count")
        print(f"  {label:20s}: {c.get('count')}")


if __name__ == "__main__":
    print("---- ingest 3 call notes via POST /memories ?infer=true ----")
    ingest()
    time.sleep(8)
    print("\n---- result state ----")
    report()
