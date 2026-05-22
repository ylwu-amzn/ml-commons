# V2 Chat Agent — Postman Collection

Manual test cases for the V2 chat agent (`type: conversational_v2`, PR #4732).

## Files

- `v2_chat_agent.postman_collection.json` — collection with 9 folders, ~30 requests.
- `v2_chat_agent.environment.json` — environment with all variables you'll edit.

## Import

In Postman: **Import** → both JSON files → select `ml-commons V2 chat agent` as the
active environment.

## Variables to fill in (environment)

| Variable | Description |
|----------|-------------|
| `endpoint` | OpenSearch URL, default `https://localhost:9200` |
| `username`, `password` | Cluster credentials (Basic auth at collection level) |
| `aws_region` | Bedrock region, default `us-west-2` |
| `aws_access_key`, `aws_secret_key`, `aws_session_token` | AWS creds for Bedrock SigV4 |
| `bedrock_model_id` | Provider model id (default Claude Haiku 4.5) |
| `embedding_dimension` | Match the embedding model (1024 for Titan Embed v2) |
| `user_id` | Default namespace user_id (default `alice`) |
| `tiny_png_base64` | Pre-filled with a 1x1 PNG for image tests |
| `sample_s3_image_uri` | Set to a real `s3://bucket/key.png` if running 5.2 |
| `sample_pdf_base64` | Optional; for 5.3 document content block test |

These are populated by the Setup folder's **Tests** scripts (don't edit by hand):

- `embed_connector_id`, `embedding_model_id`
- `llm_connector_id`, `fact_extraction_llm_id`
- `memory_container_id`
- `agent_id`, `agent_id_with_tools`
- `memory_id` (set after first agent execute, used for multi-turn)

## How to run

1. **Folder 0. Setup** — run requests 0.1 → 0.7 in order. Each writes the next ID
   to the environment.
2. **Folder 1. Register V2 chat agent** — run 1.a (no tools), then 1.b (with tools)
   if you want to exercise the Tools folder.
3. **Folders 2–7** — happy-path tests. Each is independent; you can run any one.
4. **Folder 8. Negative / edge cases** — each is **expected to fail** (or behave
   surprisingly). Useful for confirming validation and reproducing the open issues
   tracked in [#4829](https://github.com/opensearch-project/ml-commons/issues/4829)
   and [#4830](https://github.com/opensearch-project/ml-commons/issues/4830).
5. **Folder 9. Cleanup** — optional teardown.

## Coverage at a glance

| Folder | What it covers |
|--------|----------------|
| 0. Setup | Cluster settings, embed connector + model, LLM connector + model, memory container |
| 1. Register V2 chat agent | Simplified registration (`model` block) with and without tools |
| 2. Input shape: TEXT | Plain string under `input` |
| 3. Input shape: CONTENT_BLOCKS | Array of content blocks (no `role`) |
| 4. Input shape: MESSAGES | Array of `{role, content}` messages, including a multi-turn-in-one-call test and a tool round-trip injection |
| 5. Multi-modal | Image (BASE64), Image (S3 URL), Document (BASE64 PDF) |
| 6. Multi-turn / memory | `memory_id` continuation across two turns + long-term semantic & hybrid search |
| 7. Tools | Server-side ReAct tool execution + bounded iterations |
| 8. Negative / edge cases | 15 cases: missing/empty/null input, mixed array, bad source type, system role, max_tokens silently ignored, bad memory_id, V2 streaming reject, V2 missing/wrong memory at register |
| 9. Cleanup | Teardown |

## Notes

- Cluster prerequisites: `opensearch-ml`, `opensearch-knn`, `opensearch-neural-search`,
  `opensearch-security`. Setup folder enables `unified_agent_api_enabled` and
  `agentic_memory_enabled`.
- All requests use the collection-level Basic auth; no per-request auth needed.
- `aws_session_token` is optional for long-lived IAM users. Leave blank if you don't
  use STS session credentials.
- The `tests` script on each request that produces an ID will populate the matching
  environment variable automatically — chain runs in order.
