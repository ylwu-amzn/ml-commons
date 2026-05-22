# Connector 02 — Bedrock Converse predict (covers Jackson 3.x #4795)

- **Area:** Connector + Jackson 3.x JSON parsing
- **PRs:** [#4795](https://github.com/opensearch-project/ml-commons/pull/4795), [#4784](https://github.com/opensearch-project/ml-commons/pull/4784)
- **Status:** PASS
- **Date:** 2026-05-22

## Goal

Exercise the most common ml-commons remote-inference path under Jackson 3.x: register
connector → deploy remote model → predict → parse JSON response. If Jackson 3.x serialization
or exception handling were broken, this end-to-end path is where it would surface first.

## Steps

Reuses the connector from
[01_default_timeouts_4759.md](./01_default_timeouts_4759.md):

- Connector ID: `HzYAUZ4Bn4mRprk_ww-O`
- Model ID: `IzYBUZ4Bn4mRprk_GQ_C` (deploy state: `COMPLETED`)

### Predict

```bash
oc -X POST "$OS_URL/_plugins/_ml/models/IzYBUZ4Bn4mRprk_GQ_C/_predict" \
   -H "Content-Type: application/json" \
   -d '{"parameters":{"messages":[{"role":"user","content":[{"text":"Say hello in 5 words"}]}]}}'
```

### Response

```json
{
  "inference_results": [{
    "output": [{
      "name": "response",
      "dataAsMap": {
        "metrics": {"latencyMs": 862},
        "output": {
          "message": {
            "content": [{"text": "Hello, how are you today?"}],
            "role": "assistant"
          }
        },
        "stopReason": "end_turn",
        "usage": {
          "cacheReadInputTokenCount": 0,
          "cacheReadInputTokens": 0,
          "cacheWriteInputTokenCount": 0,
          "cacheWriteInputTokens": 0,
          "inputTokens": 14,
          "outputTokens": 10,
          "serverToolUsage": {},
          "totalTokens": 24
        }
      }
    }],
    "status_code": 200
  }]
}
```

## Result

PASS — Bedrock Converse JSON response is correctly deserialized to `dataAsMap`. Nested object
(`output.message.content[0].text`), arrays, integers (`inputTokens: 14`), and the empty
object (`serverToolUsage: {}`) all parse cleanly.

This validates `StringUtils.fromJson` and `StringUtils.toJson` in the Jackson 3.x line for the
predict path. Streaming-specific Jackson paths (`BedrockStreamingHandler`) need a separate
test — see `04_streaming/`.
