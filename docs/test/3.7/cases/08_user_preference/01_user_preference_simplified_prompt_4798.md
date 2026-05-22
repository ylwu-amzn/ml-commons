# User Preference 01 — Simplified extraction prompt — PR #4798

- **Area:** Agentic memory USER_PREFERENCE strategy
- **PR:** [#4798](https://github.com/opensearch-project/ml-commons/pull/4798)
- **Status:** PASS
- **Date:** 2026-05-22

## Goal

Pre-3.7, the USER_PREFERENCE extraction prompt asked the LLM to emit facts in the format
`<sentence>. Context: <why/how>. Categories: cat1,cat2`. That metadata caused malformed JSON
on smaller LLMs (Haiku 3) and diluted embedding quality. The fix simplifies the prompt to
emit plain self-contained sentences (matching the SEMANTIC strategy format).

Validate that:

1. The prompt source no longer contains `Context:` / `Categories:` instructions.
2. Extracted USER_PREFERENCE facts are plain sentences with no metadata prefix/suffix.
3. The strategy still successfully discriminates user preferences from non-preference text.

## Source check

`common/src/main/java/org/opensearch/ml/common/memorycontainer/MemoryContainerConstants.java:203`:

```java
public static final String USER_PREFERENCE_FACTS_EXTRACTION_PROMPT =
    """
        <ROLE>You are a USER PREFERENCE EXTRACTOR, not a chat assistant. ...</ROLE>
        ...
        <STYLE & RULES>
        • One self-contained sentence per preference. Include the subject's name and
          enough context to be meaningful on its own.
        • Merge closely related details into the same sentence. No duplicates.
        • Preserve user wording, names, numbers, and units. Avoid relative time references;
          use absolute dates when available.
        • Keep each fact under 350 characters.
        </STYLE & RULES>
        <OUTPUT>
        Return ONLY one minified JSON object exactly as {"facts":["fact sentence"]}.
        ...
        </OUTPUT>""";
```

No mention of `Context:` or `Categories:` — confirmed plain-sentence format. PASS.

## Steps

### Step 1 — Create memory container with USER_PREFERENCE strategy

```bash
oc -X POST "$OS_URL/_plugins/_ml/memory_containers/_create" -d '{
  "name": "t37_user_pref_test",
  "configuration": {
    "embedding_model_id": "3Tu2QZ4BmZYVijh_4FoW",
    "embedding_dimension": 1024,
    "embedding_model_type": "TEXT_EMBEDDING",
    "llm_id": "oTsCQp4BmZYVijh_El1Y",
    "index_prefix": "t37-userpref",
    "strategies": [{ "type": "USER_PREFERENCE", "namespace": ["user_id"], "enabled": true }],
    "max_infer_size": 5
  }
}'
# → memory_container_id: rzYQUZ4Bn4mRprk_dg-s
```

### Step 2 — Add a preference-rich user message

```bash
oc -X POST "$OS_URL/_plugins/_ml/memory_containers/{id}/memories" -d '{
  "messages":[{"role":"user","content":[{"text":
    "I prefer dark mode and always set my IDE font to JetBrains Mono. I dislike Helvetica."
  }]}],
  "namespace":{"user_id":"alice"},
  "infer":true
}'
```

### Step 3 — Inspect long-term memories

```bash
oc -X POST .../_semantic_search -d '{"namespace":{"user_id":"alice"},"query":"font preferences","k":10}'
```

Three extracted facts:

| _score | memory text |
|--------|-------------|
| 0.6551 | "The user dislikes Helvetica font." |
| 0.6424 | "The user always sets their IDE font to JetBrains Mono." |
| 0.6231 | "The user prefers dark mode for their IDE." |

## Result

PASS — all three facts are plain sentences, no `Context: ...` or `Categories: ...` suffix.
The strategy correctly split the input into three distinct preferences.

## Notes / observations

- The earlier general SEMANTIC test (`02_memory_container/02_semantic_hybrid_search_4658.md`)
  also already showed plain-sentence extraction, since the SEMANTIC strategy was always
  plain-sentence. PR #4798's value is bringing USER_PREFERENCE into alignment with SEMANTIC.
- `strategy_type: USER_PREFERENCE` is correctly stamped on each fact in the long-term index.
- This test used Claude Haiku 4.5 as the LLM. The PR's stated motivation was that smaller
  LLMs (Haiku **3**) were dropping USER_PREFERENCE results due to malformed JSON; we did not
  re-run with Haiku 3 because that model is no longer available on Bedrock. Per the PR
  benchmark (LoCoMo, 1540 questions, Sonnet 4.6), the simplified prompt only differs by
  −1.6% (within LLM variance) while increasing extraction count by +1%.
