#!/usr/bin/env python3
"""Run 5 demo queries using the real /memories/long-term/_semantic_search endpoint."""
import json, subprocess

AUTH="admin:RUIwNTVGTDc2MDhDM0JNRC4u"
BASE="https://localhost:9200"
CID="hV4x0Z0BrnKV8q3htQqt"
NODES="demo-memory-lpg-nodes"
EDGES="demo-memory-lpg-edges"

def curl(method, path, body=None):
    cmd=["curl","-sk","-u",AUTH,"-X",method,f"{BASE}{path}"]
    if body is not None: cmd+=["-H","Content-Type: application/json","-d",json.dumps(body)]
    return json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)

# Real vanilla = agentic-memory's native semantic search endpoint over long-term index
def vanilla(q, k=5):
    r=curl("POST",f"/_plugins/_ml/memory_containers/{CID}/memories/long-term/_semantic_search",
           {"query":q,"k":k})
    return [(h["_score"],h["_source"]["memory"]) for h in r.get("hits",{}).get("hits",[])][:k]

def graph_search(body):
    return curl("POST",f"/_plugins/_ml/memory_containers/{CID}/memories/graph/_search",body)

def edges(query):
    return curl("POST",f"/{EDGES}/_search",{"size":30,"query":query})

def nodes(query):
    return curl("POST",f"/{NODES}/_search",{"size":30,"_source":{"excludes":["entity_embedding"]},"query":query})

def name_of(eid):
    r=curl("POST",f"/{NODES}/_search",{"size":1,"query":{"term":{"entity_id":eid}},"_source":["entity_name","entity_type"]})
    h=r.get("hits",{}).get("hits",[])
    return (h[0]["_source"]["entity_name"], h[0]["_source"]["entity_type"]) if h else (None, None)

def id_for(name, etype=None):
    must=[{"match_phrase":{"entity_name":name}}]
    if etype: must.append({"term":{"entity_type":etype}})
    r=nodes({"bool":{"must":must}})
    h=r.get("hits",{}).get("hits",[])
    return h[0]["_source"]["entity_id"] if h else None

DAISY=id_for("Daisy Chen","PERSON")

def sep(t):
    print("\n"+"="*70); print(t); print("="*70)

def q1():
    sep("Q1: Who's on Sarah Kim's team?")
    print("\n[VANILLA] POST /memories/long-term/_semantic_search  query='who is on Sarah Kim team'")
    for s,t in vanilla("who is on Sarah Kim team",5): print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] MANAGES-from-Sarah OR REPORTS_TO-Sarah")
    sarah=id_for("Sarah Kim","PERSON")
    r=edges({"bool":{"should":[
        {"bool":{"must":[{"term":{"source_entity":sarah}},{"term":{"relationship_type":"MANAGES"}}]}},
        {"bool":{"must":[{"term":{"target_entity":sarah}},{"term":{"relationship_type":"REPORTS_TO"}}]}},
    ],"minimum_should_match":1}})
    members=set()
    for h in r["hits"]["hits"]:
        s=h["_source"]
        members.add(s["target_entity"] if s["source_entity"]==sarah else s["source_entity"])
    for m in members:
        n,_=name_of(m); print(f"  - {n}")

def q2():
    sep("Q2: Which customers run on PostgreSQL?")
    print("\n[VANILLA] semantic_search  query='which customers use PostgreSQL'")
    for s,t in vanilla("which customers use PostgreSQL",5): print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] USES edges to PostgreSQL, filtered to ORGANIZATION")
    pg=id_for("PostgreSQL","TECHNOLOGY")
    r=edges({"bool":{"must":[{"term":{"target_entity":pg}},{"term":{"relationship_type":"USES"}}]}})
    for h in r["hits"]["hits"]:
        n,t=name_of(h["_source"]["source_entity"])
        if t=="ORGANIZATION": print(f"  - {n}")

def q3():
    sep("Q3: People Daisy knows at companies using MongoDB")
    print("\n[VANILLA] semantic_search  query='people Daisy knows at companies using MongoDB'")
    for s,t in vanilla("people Daisy knows at companies using MongoDB",5): print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] 2-hop: Daisy KNOWS ?p ; ?p WORKS_AT ?c ; ?c USES MongoDB")
    # Daisy KNOWS (both directions)
    r=edges({"bool":{"must":[{"term":{"relationship_type":"KNOWS"}}],
                     "should":[{"term":{"source_entity":DAISY}},{"term":{"target_entity":DAISY}}],
                     "minimum_should_match":1}})
    known=set()
    for h in r["hits"]["hits"]:
        s=h["_source"]
        other = s["target_entity"] if s["source_entity"]==DAISY else s["source_entity"]
        known.add(other)
    # Their employers
    r=edges({"bool":{"must":[{"terms":{"source_entity":list(known)}},{"term":{"relationship_type":"WORKS_AT"}}]}})
    person_co=[(h["_source"]["source_entity"],h["_source"]["target_entity"]) for h in r["hits"]["hits"]]
    # Which of those companies USE mongo
    mongo=id_for("MongoDB","TECHNOLOGY")
    r=edges({"bool":{"must":[{"term":{"target_entity":mongo}},{"term":{"relationship_type":"USES"}}]}})
    mongo_cos={h["_source"]["source_entity"] for h in r["hits"]["hits"]}
    for p,c in person_co:
        if c in mongo_cos:
            p_n,_=name_of(p); c_n,_=name_of(c)
            print(f"  - {p_n} @ {c_n}")

def q4():
    sep("Q4: Daisy's contacts at companies that compete with our customers")
    print("\n[VANILLA] semantic_search  query='Daisy contacts at companies competing with our customers'")
    for s,t in vanilla("Daisy contacts at companies competing with our customers",5): print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] 3-hop: Daisy KNOWS ?p ; ?p WORKS_AT ?c ; ?c COMPETES_WITH ?our_customer")
    # Daisy KNOWS
    r=edges({"bool":{"must":[{"term":{"relationship_type":"KNOWS"}}],
                     "should":[{"term":{"source_entity":DAISY}},{"term":{"target_entity":DAISY}}],
                     "minimum_should_match":1}})
    known=set()
    for h in r["hits"]["hits"]:
        s=h["_source"]
        other = s["target_entity"] if s["source_entity"]==DAISY else s["source_entity"]
        known.add(other)
    # Their companies
    r=edges({"bool":{"must":[{"terms":{"source_entity":list(known)}},{"term":{"relationship_type":"WORKS_AT"}}]}})
    person_to_co={h["_source"]["source_entity"]:h["_source"]["target_entity"] for h in r["hits"]["hits"]}
    # Competition edges (either direction)
    r=edges({"term":{"relationship_type":"COMPETES_WITH"}})
    comp_pairs=[(h["_source"]["source_entity"],h["_source"]["target_entity"]) for h in r["hits"]["hits"]]
    comp_cos={c for pair in comp_pairs for c in pair}
    for p,c in person_to_co.items():
        if c in comp_cos:
            p_n,_=name_of(p); c_n,_=name_of(c)
            # find the competitor pair entry to report
            other=None
            for a,b in comp_pairs:
                if c==a: other=b
                elif c==b: other=a
            o_n,_=name_of(other) if other else (None,None)
            print(f"  - {p_n} @ {c_n}  (competes with {o_n})")

def q5():
    sep("Q5: Who is interested in vector databases?")
    print("\n[VANILLA] semantic_search  query='who is interested in vector databases'")
    for s,t in vanilla("who is interested in vector databases",5): print(f"  {s:.3f}  {t}")
    print("\n[GRAPH] hybrid search around 'vector databases'")
    g=graph_search({"query":"vector databases","top_k":10})
    seen=set()
    for sr in g.get("search_results",[]):
        e=sr["entity"]
        if e["type"]=="PERSON" and e["entity_id"] not in seen:
            seen.add(e["entity_id"])
            print(f"  - {e['name']}  ({sr['match_type']}, score {sr['score']:.2f})")

for f in [q1,q2,q3,q4,q5]: f()
