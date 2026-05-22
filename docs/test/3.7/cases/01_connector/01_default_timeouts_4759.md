# Connector 01 — Default `connection_timeout` / `read_timeout` semantics — PR #4759

- **Area:** ConnectorClientConfig
- **PR:** [#4759](https://github.com/opensearch-project/ml-commons/pull/4759)
- **Status:** PASS
- **Date:** 2026-05-22
- **Cluster:** OpenSearch 3.7.0, ml-commons 3.7.0-SNAPSHOT

## Goal

Before #4759 the defaults were `30000` but the code interpreted the value as **seconds** via
`Duration.ofSeconds()`, so the effective timeout was ~8.3 hours. The fix changes the defaults
to `30` (= 30 seconds) and updates log units. Verify:

1. Defaults are `30` not `30000` (source-level).
2. A connector with explicit `client_config: {connection_timeout: 30, read_timeout: 30}`
   round-trips correctly.
3. A real predict call against a Bedrock connector completes within seconds (i.e. defaults
   don't break valid usage).

## Source check

`common/src/main/java/org/opensearch/ml/common/connector/ConnectorClientConfig.java:42-43`

```java
public static final Integer CONNECTION_TIMEOUT_DEFAULT_VALUE = Integer.valueOf(30);
public static final Integer READ_TIMEOUT_DEFAULT_VALUE = Integer.valueOf(30);
```

Confirmed — defaults are `30` (seconds).

## Setup

```bash
export OS_URL=https://localhost:9200
export OS_AUTH='admin:RUIwNTVGTDc2MDhDM0JNRC4u'
oc() { curl -sk -u "$OS_AUTH" "$@"; }

# Per-test STS creds (this is a build host with iam user/role)
CREDS=$(aws sts get-session-token --output json)
AK=$(echo "$CREDS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["Credentials"]["AccessKeyId"])')
SK=$(echo "$CREDS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["Credentials"]["SecretAccessKey"])')
ST=$(echo "$CREDS" | python3 -c 'import sys, json; print(json.load(sys.stdin)["Credentials"]["SessionToken"])')
```

## Steps

### Step 1 — Connector without `client_config`

Register a Bedrock Converse connector with no `client_config` block:

```bash
cat > /tmp/conn.json << EOF
{
  "name": "t37_bedrock_claude_haiku_default_timeouts",
  "version": 1,
  "protocol": "aws_sigv4",
  "parameters": { "region": "us-west-2", "service_name": "bedrock",
                   "model": "us.anthropic.claude-haiku-4-5-20251001-v1:0" },
  "credential": { "access_key": "$AK", "secret_key": "$SK", "session_token": "$ST" },
  "actions": [{
    "action_type": "predict",
    "method": "POST",
    "url": "https://bedrock-runtime.us-west-2.amazonaws.com/model/\${parameters.model}/converse",
    "headers": { "content-type": "application/json" },
    "request_body": "{ \"messages\": \${parameters.messages} }"
  }]
}
EOF
oc -X POST "$OS_URL/_plugins/_ml/connectors/_create" -H "Content-Type: application/json" -d @/tmp/conn.json
# → {"connector_id":"HzYAUZ4Bn4mRprk_ww-O"}
```

GET shows no `client_config` echoed (the connector inherits defaults at runtime). The fact
that defaults aren't serialized into the index doc is fine — runtime uses
`CONNECTION_TIMEOUT_DEFAULT_VALUE = 30`.

### Step 2 — Connector with explicit `client_config`

```json
"client_config": {
  "max_connection": 50,
  "connection_timeout": 30,
  "read_timeout": 30
}
```

GET response shows the values round-trip correctly:

```
"client_config" : {
  "max_connection" : 50,
  "connection_timeout" : 30,
  "read_timeout" : 30,
  ...
}
```

### Step 3 — Predict actually works

Register and deploy a remote model on the default-timeout connector
(`connector_id=HzYAUZ4Bn4mRprk_ww-O`). Deploy completed `state: COMPLETED`.

Predict:

```bash
oc -X POST "$OS_URL/_plugins/_ml/models/IzYBUZ4Bn4mRprk_GQ_C/_predict" \
   -H "Content-Type: application/json" \
   -d '{"parameters":{"messages":[{"role":"user","content":[{"text":"Say hello in 5 words"}]}]}}'
```

Response (truncated):

```json
{"inference_results":[{"output":[{"name":"response","dataAsMap":{
  "metrics":{"latencyMs":862},
  "output":{"message":{"content":[{"text":"Hello, how are you today?"}],"role":"assistant"}},
  "stopReason":"end_turn",
  "usage":{"inputTokens":14,"outputTokens":10,"totalTokens":24}
}}],"status_code":200}]}
```

`latencyMs: 862` — much less than 30s; defaults produce a normal request without false
timeouts. (Pre-fix this would have either timed out at 30000ms = 30s wrongly interpreted as
30000s, or worked by accident.)

## Result

PASS — defaults are correct; explicit `client_config` round-trips; real Bedrock predict works.

## Migration callout for the release notes

> The `connection_timeout` and `read_timeout` parameters in `client_config` are interpreted as
> **seconds**, not milliseconds. Users who previously set `30000` to mean "30 seconds" will
> now get a 30000 second (~8.3h) timeout. Reduce these values by a factor of 1000.
