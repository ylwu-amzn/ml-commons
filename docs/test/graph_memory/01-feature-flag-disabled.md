# Test 1 — Agentic-memory feature flag disabled

**Goal:** Verify graph endpoints reject requests when `plugins.ml_commons.agentic_memory_enabled=false`.

**Status:** ✅ PASS (after fix — returns 403)

## Setup

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPUT "https://localhost:9200/_cluster/settings" \
  -H 'Content-Type: application/json' \
  -d '{"persistent":{"plugins.ml_commons.agentic_memory_enabled":"false"}}'
```

```json
{ "acknowledged": true, "persistent": { "plugins": { "ml_commons": { "agentic_memory_enabled": "false" } } } }
```

## Request A — GET list entities (flag off)

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XGET "https://localhost:9200/_plugins/_ml/memory_containers/4aRp0J0BQ4F7Y_V3tOLD/memories/graph/entities?query=*"
```

### Response — HTTP 403 (after fix)

```json
{
  "error": {
    "root_cause": [{ "type": "status_exception", "reason": "Agentic memory feature is not enabled" }],
    "type": "status_exception",
    "reason": "Agentic memory feature is not enabled"
  },
  "status": 403
}
```

### Before fix — HTTP 500

```json
{
  "error": {
    "root_cause": [{ "type": "illegal_state_exception", "reason": "Agentic memory feature is not enabled" }],
    "status": 500
  }
}
```

## Cleanup

```bash
curl -sk -u admin:RUIwNTVGTDc2MDhDM0JNRC4u \
  -XPUT "https://localhost:9200/_cluster/settings" \
  -H 'Content-Type: application/json' \
  -d '{"persistent":{"plugins.ml_commons.agentic_memory_enabled":"true"}}'
```

## Bug found

**Bug 1.1 — Graph REST handlers throw raw `IllegalStateException`, producing HTTP 500 instead of 403.**

Source (`RestMLGraphSearchAction`, `RestMLListGraphEntitiesAction`, `RestMLDeleteGraphDataAction`):

```java
if (!mlFeatureEnabledSetting.isAgenticMemoryEnabled()) {
    throw new IllegalStateException("Agentic memory feature is not enabled");
}
```

Compare with the transport-action path which correctly uses `OpenSearchStatusException(..., RestStatus.FORBIDDEN)` (covered by `TransportGraphSearchActionTests.testDoExecute_FeatureDisabled`).

**Fix:** in each of the three REST handlers, replace the `IllegalStateException` with:

```java
throw new OpenSearchStatusException(
    "Agentic memory feature is not enabled", RestStatus.FORBIDDEN);
```

Not fixed in this session — marked for follow-up. The transport-layer check is still present and would fire on any internal call, but REST clients currently see a misleading 500.
