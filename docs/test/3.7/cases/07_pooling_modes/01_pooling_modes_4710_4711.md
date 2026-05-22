# Pooling 01 — LAST_TOKEN and NONE pooling modes — PR #4710, #4711

- **Area:** Local sentence-transformer / HuggingFace embedding model pooling
- **PRs:**
  - [#4711](https://github.com/opensearch-project/ml-commons/pull/4711) — LAST_TOKEN pooling
  - [#4710](https://github.com/opensearch-project/ml-commons/pull/4710) — NONE pooling
- **Status:** PARTIAL (parser accepts both new modes; full e2e requires a decoder-only or
  pre-pooled model URL we don't have on this host)
- **Date:** 2026-05-22

## Goal

Validate that:

1. `pooling_mode: LAST_TOKEN` is accepted by `BaseModelConfig.PoolingMode.from(...)`.
2. `pooling_mode: NONE` is accepted similarly.
3. `pooling_mode: <invalid>` is rejected with a clean 400.
4. End-to-end: a model registered with one of these modes produces correct embeddings
   (decoder-only model for LAST_TOKEN; pre-pooled model for NONE).

## Source check

`common/src/main/java/org/opensearch/ml/common/model/BaseModelConfig.java:244-251`:

```java
public enum PoolingMode {
    MEAN("mean"),
    MEAN_SQRT_LEN("mean_sqrt_len"),
    MAX("max"),
    WEIGHTED_MEAN("weightedmean"),
    CLS("cls"),
    LAST_TOKEN("lasttoken"),     // ← #4711
    NONE("none");                // ← #4710
    ...
    public static PoolingMode from(String value) {
        try {
            return PoolingMode.valueOf(value.toUpperCase(Locale.ROOT));
        } catch (Exception e) {
            throw new IllegalArgumentException("Wrong pooling method");
        }
    }
}
```

ONNX & HuggingFace translators were updated:
- `ONNXSentenceTransformerTextEmbeddingTranslator.java:92-93` — when `poolingMode == NONE
  && list.size() > 1`, use the **second output** (`sentence_embedding`) directly.
- `ONNXSentenceTransformerTextEmbeddingTranslator.java:124` — `case LAST_TOKEN`: extract the
  embedding of the last non-padding token (decoder-only models like Qwen3).
- `HuggingfaceTextEmbeddingTranslator.java:83` — NONE pooling fallback chain.

## Steps

### Step 1 — Invalid pooling mode rejected

```bash
oc -X POST "$OS_URL/_plugins/_ml/models/_register" -d '{
  "name":"t37_bad_pool",
  "version":"1.0.0",
  "model_format":"TORCH_SCRIPT",
  "function_name":"TEXT_EMBEDDING",
  "model_content_hash_value":"deadbeef",
  "url":"https://example.com/notreal.zip",
  "model_config":{
    "model_type":"bert",
    "embedding_dimension":384,
    "framework_type":"SENTENCE_TRANSFORMERS",
    "pooling_mode":"INVALID_POOLING"
  }
}'
```

Response:

```
HTTP 400
{"type":"illegal_argument_exception","reason":"Wrong pooling method"}
```

PASS.

### Step 2 — LAST_TOKEN parses successfully

Same payload with `"pooling_mode": "LAST_TOKEN"`:

```
{"task_id":"yzYRUZ4Bn4mRprk_lA_c","status":"CREATED"}
```

Task registered (then later fails at download step because URL is fake — but pooling-mode
parsing succeeded). Task state: `FAILED`, `error: "https://example.com/notreal.zip"` —
download failure, NOT a pooling-mode failure.

PASS — parser accepts `LAST_TOKEN`.

### Step 3 — NONE parses successfully

Same payload with `"pooling_mode": "NONE"`:

```
{"task_id":"zjYRUZ4Bn4mRprk_lQ8E","status":"CREATED"}
```

Same outcome — task fails at URL download, not at pooling-mode parsing.

PASS.

## Result

PARTIAL — parser-level validation works for both new modes. End-to-end embedding production
not validated because:

1. The test cluster has no pre-staged decoder-only model (Qwen3 / GPT-style) we could use to
   test LAST_TOKEN.
2. The test cluster has no pre-staged sentence-transformer that ships pre-pooled output to
   test NONE (the existing `all-MiniLM-L6-v2` uses MEAN pooling and outputs only
   `token_embeddings` + `input_ids`, no `sentence_embedding`).

To complete this test:

- **LAST_TOKEN**: stage a decoder-only embedding model (e.g., `Qwen/Qwen2.5-0.5B`) traced to
  TorchScript or ONNX with `lasttoken` registered as a valid pooling. Register and predict;
  expect a 384-dim (or whatever hidden_size) vector that comes from the last non-padding
  token.
- **NONE**: stage a sentence-transformer model (e.g., `all-mpnet-base-v2` from
  `sentence-transformers/`) that exports a `sentence_embedding` second output. Register
  with `pooling_mode: NONE`; expect the predict to use the pre-pooled output and skip
  recomputation.

## Notes / observations

- The unit tests in PR #4710 / #4711 cover both ONNX and TorchScript translators using
  fixture model files. The release-test cluster does not ship those fixtures, so
  end-to-end validation falls back to either CI or a dedicated test cluster with model
  fixtures.
- Strong recommendation: include the LAST_TOKEN/NONE registration steps in the next
  release's tutorial/integration test fixtures so future release-tests can validate this
  end-to-end without needing to stage models separately.
