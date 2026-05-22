# Smoke 01 — Cluster reachable, ml-commons present

- **Area:** baseline
- **PR:** N/A
- **Status:** PASS
- **Date:** 2026-05-22
- **Cluster:** OpenSearch 3.7.0, ml-commons 3.7.0-SNAPSHOT (single-node, security enabled)

## Goal

Confirm the test cluster is up, reachable with the provided admin credentials, and that
ml-commons + key dependencies (neural-search, knn, skills, flow-framework, security) are
installed at expected versions.

## Setup

```bash
export OS_URL=https://localhost:9200
export OS_AUTH='admin:RUIwNTVGTDc2MDhDM0JNRC4u'
oc() { curl -sk -u "$OS_AUTH" "$@"; }
```

## Steps

### Step 1 — Cluster root

```bash
oc "$OS_URL/"
```

Response:

```json
{
  "name" : "ip-172-31-18-42",
  "cluster_name" : "opensearch",
  "cluster_uuid" : "san4qxWbSPmiJ4KzV34TMA",
  "version" : {
    "distribution" : "opensearch",
    "number" : "3.7.0",
    "build_type" : "tar",
    "build_hash" : "8f2d05879bb4e88163d93cdb0c4326e9bb350747",
    "build_date" : "2026-05-19T01:54:03.477325764Z",
    "build_snapshot" : false,
    "lucene_version" : "10.4.0",
    "minimum_wire_compatibility_version" : "2.19.0",
    "minimum_index_compatibility_version" : "2.0.0"
  }
}
```

### Step 2 — Cluster health

```bash
oc "$OS_URL/_cluster/health?pretty"
```

`status: yellow`, single data node, `active_shards_percent: 96.77%`. Yellow is expected on
a single-node deployment (no replicas). Acceptable.

### Step 3 — Plugin inventory (filtered)

```
opensearch-flow-framework            3.7.0.0
opensearch-knn                       3.7.0.0
opensearch-ml                        3.7.0.0-SNAPSHOT
opensearch-neural-search             3.7.0.0
opensearch-security                  3.7.0.0
opensearch-skills                    3.7.0.0
```

### Step 4 — ml-commons stats

```bash
oc "$OS_URL/_plugins/_ml/stats?pretty"
```

Pre-existing in this cluster (left over from prior testing):
- `ml_model_count: 33`
- `ml_connector_count: 27`
- `ml_deployed_model_count: 26`
- All ml indices green.

## Result

PASS — cluster is healthy, ml-commons 3.7.0-SNAPSHOT is the build under test, neural-search
present (required for hybrid memory search tests).

## Notes

- The cluster already has many models/connectors from earlier work. We will create new ones
  with `t37_` prefix to avoid colliding with existing fixtures.
- `master_key_cache_ttl_minutes: 5` (default) — keep in mind for any tenant/encryption tests.
- `mcp_connector_enabled: false` (default) — must be enabled before MCP testing.
- `agentic_memory_enabled: true` (default).
- `unified_agent_api_enabled: true`.
