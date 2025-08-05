# Topic

This tutorial works for 3.1 and above. For more details, see the OpenSearch document [Agents and tools](https://opensearch.org/docs/latest/ml-commons-plugin/agents-tools/index/).

Read this AWS [retrieval-augmented-generation](https://aws.amazon.com/what-is/retrieval-augmented-generation/) doc to learn more details about RAG concepts.

One of the known limitations of large language models (LLMs) is that their knowledge base only contains information up to their training cutoff date. 
LLMs have no knowledge of recent events or your internal data. 
You can augment the LLM knowledge base by using retrieval-augmented generation (RAG).

This tutorial explains how to build Agentic RAG solutions with OpenSearch in two approaches:

- Basic Implementation: Creating a RAG system with a predefined knowledge base in OpenSearch
- Advanced Implementation: Building a dynamic RAG system that automatically selects appropriate indexes and search

By following both implementations, you'll learn how to progress from a simple RAG setup to a more flexible, intelligent system that can adapt to different data sources.

Note: Replace the placeholders that start with `your_` with your own values.

# Approach 1: Agentic RAG with Predefined Knowledge Base

## 1. Preparation Knowledge Base

### 1.1 Create Embedding Model

We will use Bedrock Titan Embedding V2 in this tutorial.
```
POST /_plugins/_ml/models/_register
{
  "name": "Bedrock embedding model",
  "function_name": "remote",
  "description": "Bedrock Titan Embedding Model V2",
  "connector": {
    "name": "Amazon Bedrock Connector: embedding",
    "description": "The connector to bedrock Titan embedding model",
    "version": 1,
    "protocol": "aws_sigv4",
    "parameters": {
      "region": "your_aws_region",
      "service_name": "bedrock",
      "model": "amazon.titan-embed-text-v2:0",
      "dimensions": 1024,
      "normalize": true,
      "embeddingTypes": [
        "float"
      ]
    },
    "credential": {
      "access_key": "your_aws_access_key",
      "secret_key": "your_aws_secret_key",
      "session_token": "your_aws_session_token"
    },
    "actions": [
      {
        "action_type": "predict",
        "method": "POST",
        "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/invoke",
        "headers": {
          "content-type": "application/json",
          "x-amz-content-sha256": "required"
        },
        "request_body": "{ \"inputText\": \"${parameters.inputText}\", \"dimensions\": ${parameters.dimensions}, \"normalize\": ${parameters.normalize}, \"embeddingTypes\": ${parameters.embeddingTypes} }",
        "pre_process_function": "connector.pre_process.bedrock.embedding",
        "post_process_function": "connector.post_process.bedrock.embedding"
      }
    ]
  }
}
```

Sample output
```
{
  "task_id": "-OVdX5gB_sp0V9yyvtqh",
  "status": "CREATED",
  "model_id": "6uVbX5gB_sp0V9yyy9ro"
}
```

Test model by running predict API
```
POST _plugins/_ml/models/6uVbX5gB_sp0V9yyy9ro/_predict
{
  "parameters": {
    "inputText": "What is the meaning of life?"
  }
}
```

Sample predict output
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "sentence_embedding",
          "data_type": "FLOAT32",
          "shape": [
            1024
          ],
          "data": [
            -0.04134807735681534,
            ....
          ]
        }
      ],
      "status_code": 200
    }
  ]
}
```

### 1.2 Create Population Knowledge Base

Create ingest pipeline
```
PUT _ingest/pipeline/test_population_data_pipeline
{
  "processors": [
    {
      "text_embedding": {
        "model_id": "your_embedding_model_id_from_step_1.1",
        "field_map": {
          "population_description": "population_description_embedding"
        }
      }
    }
  ]
}
```
Create K-NN index 
```
PUT test_population_data
{
  "mappings": {
    "properties": {
      "population_description": {
        "type": "text"
      },
      "population_description_embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "engine": "lucene"
        }
      }
    }
  },
  "settings": {
    "index": {
      "default_pipeline": "test_population_data_pipeline",
      "knn": true
    }
  }
}
```
Ingest test data 
```
POST _bulk
{"index": {"_index": "test_population_data"}}
{"population_description": "The current metro area population of Ogden-Layton in 2025 is 771,000, a 1.31% increase from 2024.\nThe metro area population of Ogden-Layton in 2024 was 761,000, a 1.47% increase from 2023.\nThe metro area population of Ogden-Layton in 2023 was 750,000, a 1.63% increase from 2022.\nThe metro area population of Ogden-Layton in 2022 was 738,000, a 1.79% increase from 2021."}
{"index": {"_index": "test_population_data"}}
{"population_description": "The current metro area population of New York City in 2025 is 19,154,000, a 0.63% increase from 2024.\nThe metro area population of New York City in 2024 was 19,034,000, a 0.51% increase from 2023.\nThe metro area population of New York City in 2023 was 18,937,000, a 0.37% increase from 2022.\nThe metro area population of New York City in 2022 was 18,867,000, a 0.23% increase from 2021."}
{"index": {"_index": "test_population_data"}}
{"population_description": "The current metro area population of Austin in 2025 is 2,313,000, a 1.72% increase from 2024.\nThe metro area population of Austin in 2024 was 2,274,000, a 2.06% increase from 2023.\nThe metro area population of Austin in 2023 was 2,228,000, a 2.39% increase from 2022.\nThe metro area population of Austin in 2022 was 2,176,000, a 2.79% increase from 2021."}
{"index": {"_index": "test_population_data"}}
{"population_description": "The current metro area population of Seattle in 2025 is 3,581,000, a 0.9% increase from 2024.\nThe metro area population of Seattle in 2024 was 3,549,000, a 0.85% increase from 2023.\nThe metro area population of Seattle in 2023 was 3,519,000, a 0.86% increase from 2022.\nThe metro area population of Seattle in 2022 was 3,489,000, a 0.81% increase from 2021."}

```
### 1.3 Create Tech News Knowledge Base

Create ingest pipeline
```
PUT _ingest/pipeline/test_tech_news_pipeline
{
  "processors": [
    {
      "text_embedding": {
        "model_id": "your_embedding_model_id_from_step_1.1",
        "field_map": {
          "passage": "passage_embedding"
        }
      }
    }
  ]
}
```
Create K-NN index
```
PUT test_tech_news
{
  "mappings": {
    "properties": {
      "passage": {
        "type": "text"
      },
      "passage_embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "engine": "lucene"
        }
      }
    }
  },
  "settings": {
    "index": {
      "default_pipeline": "test_tech_news_pipeline",
      "knn": true
    }
  }
}
```

Ingest test data
```
POST _bulk
{"index":{"_index":"test_tech_news"}}
{"passage":"OpenSearch 3.1 release highlights: \nMakes GPU acceleration for vector index builds generally available. Introduces memory-optimized search for Faiss indexes using Lucene HNSW, semantic field type for streamlined semantic search, and Search Relevance Workbench for search quality optimization. Makes star-tree indexes generally available with support for comprehensive query types. Enhances observability with ML Commons metrics integration, custom index support for OpenTelemetry data, and new PPL commands for JSON manipulation. Improves agent management with Update Agent API and persistent MCP tools. Includes security enhancements with immutable user objects and new resource sharing framework. For a full list of release highlights, see the Release Notes."}
{"index":{"_index":"test_tech_news"}}
{"passage":"OpenSearch 3.0 release highlights:\nUpgrades to Lucene 10 for improved indexing and vector search. Adds experimental gRPC support and pull-based ingestion from Kafka and Kinesis. Introduces GPU acceleration for vector operations and semantic sentence highlighting. Improves range query performance and hybrid search with z-score normalization. Adds plan-execute-reflect agents and native MCP protocol support for agentic workflows. Enhances security with a new Java agent replacing the Security Manager. Includes PPL query improvements with lookup, join, and subsearch commands. For a full list of release highlights, see the Release Notes."}
{"index":{"_index":"test_tech_news"}}
{"passage":"Announcing Amazon S3 Vectors (Preview)—First cloud object storage with native support for storing and querying vectors:\nAmazon S3 Vectors delivers purpose-built, cost-optimized vector storage for AI agents, AI inference, and semantic search of your content stored in Amazon S3. By reducing the cost of uploading, storing, and querying vectors by up to 90%, S3 Vectors makes it cost-effective to create and use large vector datasets to improve the memory and context of AI agents as well as semantic search results of your S3 data. Designed to provide the same elasticity, scale, and durability as Amazon S3, S3 Vectors lets you store and search data with sub-second query performance. It's ideal for applications that need to build and maintain vector indexes so you can organize and search through massive amounts of information. S3 Vectors provides a simple and flexible API for operations such as finding similar scenes in petabyte-scale video archives, identifying collections of related business documents, or detecting rare patterns in diagnostic collections including millions of medical images.\nS3 Vectors is natively integrated with Amazon Bedrock Knowledge Bases so that you can reduce the cost of using large vector datasets for retrieval-augmented generation (RAG). You can also use S3 Vectors with Amazon OpenSearch Service to lower storage costs for infrequent queried vectors, and then quickly move them to OpenSearch as demands increase or to enhance search capabilities.\nS3 Vectors introduces a new bucket type optimized for durable, low-cost vector storage. It includes a dedicated set of APIs for you to store, access, and query vectors without provisioning any infrastructure. Within a vector bucket, you can organize vector data within vector indexes and elastically scale up to 10,000 indexes per bucket. When creating a Knowledge Base in Amazon Bedrock or Amazon SageMaker Unified Studio, you can select an S3 vector index as your vector store or use the Quick Create workflow to set one up. Within OpenSearch, you can adopt a tiered strategy to store large vector datasets in S3 for near real-time access while effortlessly activating the vector data with the highest performance requirements in OpenSearch."}

```

## 2. Create LLM
We will use Bedrock Claude 3.7 model in this tutorial.
```
POST _plugins/_ml/models/_register
{
  "name": "Bedrock Claude 3.7 model",
  "function_name": "remote",
  "description": "test model",
  "connector": {
    "name": "Bedrock Claude3.7 connector",
    "description": "test connector",
    "version": 1,
    "protocol": "aws_sigv4",
    "parameters": {
      "region": "your_aws_region",
      "service_name": "bedrock",
      "model": "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    },
    "credential": {
      "access_key": "your_aws_access_key",
      "secret_key": "your_aws_secret_key",
      "session_token": "your_aws_session_token"
    },
    "actions": [
      {
        "action_type": "predict",
        "method": "POST",
        "url": "https://bedrock-runtime.${parameters.region}.amazonaws.com/model/${parameters.model}/converse",
        "headers": {
          "content-type": "application/json"
        },
        "request_body": "{ \"system\": [{\"text\": \"${parameters.system_prompt}\"}], \"messages\": [${parameters._chat_history:-}{\"role\":\"user\",\"content\":[{\"text\":\"${parameters.prompt}\"}]}${parameters._interactions:-}]${parameters.tool_configs:-} }"
      }
    ]
  }
}
```

Sample output
```
{
  "task_id": "sOV8X5gB_sp0V9yyjdvB",
  "status": "CREATED",
  "model_id": "seV8X5gB_sp0V9yyjdvf"
}
```

Test model with predict API
```
POST _plugins/_ml/models/seV8X5gB_sp0V9yyjdvf/_predict
{
    "parameters": {
        "system_prompt": "You are a helpful assistant.",
        "prompt": "hello"
    }
}
```

Sample output
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "response",
          "dataAsMap": {
            "metrics": {
              "latencyMs": 1587
            },
            "output": {
              "message": {
                "content": [
                  {
                    "text": "Hello! How can I assist you today? Whether you have a question, need information, or just want to chat, I'm here to help. What's on your mind?"
                  }
                ],
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
              "outputTokens": 39,
              "totalTokens": 53
            }
          }
        }
      ],
      "status_code": 200
    }
  ]
}
```

## 3. Create Agent
Create an agent of the `conversational` type.

In this tutorial, the agent includes two tools that provide latest population data and tech news.

```
POST _plugins/_ml/agents/_register
{
  "name": "RAG Agent",
  "type": "conversational",
  "description": "this is a test agent",
  "app_type": "rag",
  "llm": {
    "model_id": "your_llm_id_from_step_2",
    "parameters": {
      "max_iteration": 10,
      "system_prompt": "You are a helpful assistant. You are able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.\nIf the question is complex, you will split it into several smaller questions, and solve them one by one. For example, the original question is:\nhow many orders in last three month? Which month has highest?\nYou will spit into several smaller questions:\n1.Calculate total orders of last three month.\n2.Calculate monthly total order of last three month and calculate which month's order is highest.",
      "prompt": "${parameters.question}"
    }
  },
  "memory": {
    "type": "conversation_index"
  },
  "parameters": {
    "_llm_interface": "bedrock/converse/claude"
  },
  "tools": [
    {
      "type": "SearchIndexTool",
      "name": "retrieve_population_data",
      "description": "This tool provides recent tech news.",
      "parameters": {
        "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
        "index": "test_population_data",
        "query": {
          "query": {
            "neural": {
              "population_description_embedding": {
                "query_text": "${parameters.question}",
                "model_id": "your_embedding_model_id_from_step_1.1"
              }
            }
          },
          "size": 2,
          "_source": "population_description"
        }
      },
      "attributes": {
        "input_schema": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string",
              "description": "Natural language question"
            }
          },
          "required": [ "question" ],
          "additionalProperties": false
        },
        "strict": false
      }
    },
    {
      "type": "SearchIndexTool",
      "name": "retrieve_tech_news",
      "description": "This tool provides recent tech news.",
      "parameters": {
        "input": "{\"index\": \"${parameters.index}\", \"query\": ${parameters.query} }",
        "index": "test_tech_news",
        "query": {
          "query": {
            "neural": {
              "passage_embedding": {
                "query_text": "${parameters.question}",
                "model_id": "your_embedding_model_id_from_step_1.1"
              }
            }
          },
          "size": 2,
          "_source": "passage"
        }
      },
      "attributes": {
        "input_schema": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string",
              "description": "Natural language question"
            }
          },
          "required": [ "question" ],
          "additionalProperties": false
        },
        "strict": false
      }
    }
  ]
}
```

Sample output
```
{
  "agent_id": "tbPvX5gBc62yu84LsNzh"
}
```

## 4. Test Agent

The `conversational` agent supports a `verbose` option. You can set `verbose` to `true` to obtain detailed steps.

Alternatively, you can use the Get Trace Data API:
```
GET _plugins/_ml/memory/message/your_message_id/traces
```

### 4.1 Simple Query Example
- Example 1: Ask a question related to tech news:
```
POST _plugins/_ml/agents/your_agent_id/_execute
{
  "parameters": {
    "question": "What's the population of Seattle 2025",
    "verbose": true
  }
}
```
In the response, note that the agent runs the `retrieve_population_data` tool to retrieve data.
The LLM uses the retrieved data to answer the question correctly.
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "memory_id",
          "result": "hbMPYJgBc62yu84LGN3I"
        },
        {
          "name": "parent_interaction_id",
          "result": "hrMPYJgBc62yu84LGN3Z"
        },
        {
          "name": "response",
          "result": """{"metrics":{"latencyMs":3290.0},"output":{"message":{"content":[{"text":"I need to provide information about Seattle\u0027s population for 2025. Since we\u0027re looking at a future date, I\u0027ll need to retrieve the most recent population data and any projections that might be available."},{"toolUse":{"input":{"question":"What is the projected population of Seattle for 2025?"},"name":"retrieve_population_data","toolUseId":"tooluse_W-EGMztFRzakU4Y_Yz_GpA"}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"cacheReadInputTokenCount":0.0,"cacheReadInputTokens":0.0,"cacheWriteInputTokenCount":0.0,"cacheWriteInputTokens":0.0,"inputTokens":597.0,"outputTokens":109.0,"totalTokens":706.0}}"""
        },
        {
          "name": "response",
          "result": """{"_index":"test_population_data","_source":{"population_description":"The current metro area population of Seattle in 2025 is 3,581,000, a 0.9% increase from 2024.\nThe metro area population of Seattle in 2024 was 3,549,000, a 0.85% increase from 2023.\nThe metro area population of Seattle in 2023 was 3,519,000, a 0.86% increase from 2022.\nThe metro area population of Seattle in 2022 was 3,489,000, a 0.81% increase from 2021."},"_id":"XLMMYJgBc62yu84LOd1K","_score":0.70810187}
{"_index":"test_population_data","_source":{"population_description":"The current metro area population of New York City in 2025 is 19,154,000, a 0.63% increase from 2024.\nThe metro area population of New York City in 2024 was 19,034,000, a 0.51% increase from 2023.\nThe metro area population of New York City in 2023 was 18,937,000, a 0.37% increase from 2022.\nThe metro area population of New York City in 2022 was 18,867,000, a 0.23% increase from 2021."},"_id":"WrMMYJgBc62yu84LOd1K","_score":0.4465592}
"""
        },
        {
          "name": "response",
          "result": """Based on the data I retrieved, the projected population of Seattle in 2025 is 3,581,000. This represents a 0.9% increase from 2024.

For context, here's how Seattle's population has grown in recent years:
- 2022: 3,489,000
- 2023: 3,519,000 (0.86% increase from 2022)
- 2024: 3,549,000 (0.85% increase from 2023)
- 2025: 3,581,000 (0.9% increase from 2024)

This data refers to the Seattle metropolitan area population rather than just the city limits of Seattle itself."""
        }
      ]
    }
  ]
}
```

### 4.2 Multi-Topic Question
```
POST _plugins/_ml/agents/your_agent_id/_execute
{
  "parameters": {
    "question": "Any new features released in OpenSearch 3.1? What's the population of Seattle in 2025",
    "verbose": true
  }
}
```
In the response, note that the agent runs `retrieve_tech_news` first , then run `retrieve_population_data` tool to retrieve latest data.  
Then LLM can answer the question correctly.
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "memory_id",
          "result": "kLMQYJgBc62yu84LcN1d"
        },
        {
          "name": "parent_interaction_id",
          "result": "kbMQYJgBc62yu84LcN1v"
        },
        {
          "name": "response",
          "result": """{"metrics":{"latencyMs":3540.0},"output":{"message":{"content":[{"text":"I\u0027ll help you answer both of your questions by breaking them down and using the appropriate tools.\n\nFirst, let\u0027s check for new features in OpenSearch 3.1:"},{"toolUse":{"input":{"question":"What are the new features released in OpenSearch 3.1?"},"name":"retrieve_tech_news","toolUseId":"tooluse_lDYhyUJxQiqd3cy9Grx3Zw"}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"cacheReadInputTokenCount":0.0,"cacheReadInputTokens":0.0,"cacheWriteInputTokenCount":0.0,"cacheWriteInputTokens":0.0,"inputTokens":610.0,"outputTokens":105.0,"totalTokens":715.0}}"""
        },
        {
          "name": "response",
          "result": """{"_index":"test_tech_news","_source":{"passage":"OpenSearch 3.1 release highlights: \nMakes GPU acceleration for vector index builds generally available. Introduces memory-optimized search for Faiss indexes using Lucene HNSW, semantic field type for streamlined semantic search, and Search Relevance Workbench for search quality optimization. Makes star-tree indexes generally available with support for comprehensive query types. Enhances observability with ML Commons metrics integration, custom index support for OpenTelemetry data, and new PPL commands for JSON manipulation. Improves agent management with Update Agent API and persistent MCP tools. Includes security enhancements with immutable user objects and new resource sharing framework. For a full list of release highlights, see the Release Notes."},"_id":"b7MMYJgBc62yu84L_93y","_score":0.7947274}
{"_index":"test_tech_news","_source":{"passage":"OpenSearch 3.0 release highlights:\nUpgrades to Lucene 10 for improved indexing and vector search. Adds experimental gRPC support and pull-based ingestion from Kafka and Kinesis. Introduces GPU acceleration for vector operations and semantic sentence highlighting. Improves range query performance and hybrid search with z-score normalization. Adds plan-execute-reflect agents and native MCP protocol support for agentic workflows. Enhances security with a new Java agent replacing the Security Manager. Includes PPL query improvements with lookup, join, and subsearch commands. For a full list of release highlights, see the Release Notes."},"_id":"cLMMYJgBc62yu84L_93y","_score":0.6915282}
"""
        },
        {
          "name": "response",
          "result": """{"metrics":{"latencyMs":2723.0},"output":{"message":{"content":[{"text":"Now, let\u0027s get information about Seattle\u0027s population for 2025:"},{"toolUse":{"input":{"question":"What\u0027s the population of Seattle in 2025?"},"name":"retrieve_population_data","toolUseId":"tooluse_E-9ER5MfTO6KNM39AYOrnw"}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"cacheReadInputTokenCount":0.0,"cacheReadInputTokens":0.0,"cacheWriteInputTokenCount":0.0,"cacheWriteInputTokens":0.0,"inputTokens":1118.0,"outputTokens":82.0,"totalTokens":1200.0}}"""
        },
        {
          "name": "response",
          "result": """{"_index":"test_population_data","_source":{"population_description":"The current metro area population of Seattle in 2025 is 3,581,000, a 0.9% increase from 2024.\nThe metro area population of Seattle in 2024 was 3,549,000, a 0.85% increase from 2023.\nThe metro area population of Seattle in 2023 was 3,519,000, a 0.86% increase from 2022.\nThe metro area population of Seattle in 2022 was 3,489,000, a 0.81% increase from 2021."},"_id":"XLMMYJgBc62yu84LOd1K","_score":0.7388486}
{"_index":"test_population_data","_source":{"population_description":"The current metro area population of New York City in 2025 is 19,154,000, a 0.63% increase from 2024.\nThe metro area population of New York City in 2024 was 19,034,000, a 0.51% increase from 2023.\nThe metro area population of New York City in 2023 was 18,937,000, a 0.37% increase from 2022.\nThe metro area population of New York City in 2022 was 18,867,000, a 0.23% increase from 2021."},"_id":"WrMMYJgBc62yu84LOd1K","_score":0.45074782}
"""
        },
        {
          "name": "response",
          "result": """Here are the answers to your questions:

### New Features in OpenSearch 3.1

OpenSearch 3.1 includes several significant new features and improvements:

1. **GPU Acceleration**: Made generally available for vector index builds
2. **Memory-Optimized Search**: Introduced for Faiss indexes using Lucene HNSW
3. **Semantic Field Type**: Added for streamlined semantic search
4. **Search Relevance Workbench**: New tool for search quality optimization
5. **Star-Tree Indexes**: Now generally available with support for comprehensive query types
6. **Observability Enhancements**:
   - ML Commons metrics integration
   - Custom index support for OpenTelemetry data
   - New PPL commands for JSON manipulation
7. **Agent Management Improvements**:
   - Update Agent API
   - Persistent MCP tools
8. **Security Enhancements**:
   - Immutable user objects
   - New resource sharing framework

### Population of Seattle in 2025

According to the data, the metro area population of Seattle in 2025 is 3,581,000, which represents a 0.9% increase from 2024."""
        }
      ]
    }
  ]
}
```


### 4.3 Continue a conversation

You can continue the conversation by providing `memory_id`.
```
POST _plugins/_ml/agents/your_agent_id/_execute
{
  "parameters": {
    "question": "What's the population of Ogden-Layton 2025, compare with Seattle",
    "memory_id": "l7VUxI0B8vrNLhb9sRuQ",
    "verbose": true
  }
}
```
 
# Approach 2: Agentic RAG with Automatic Index Selection

In this approach, we will use two agents:
- A Flow Agent which will translate human natural language question to OpenSearch query, then search index
- A Chat Agent which use ListIndexTool and the Flow Agent to automatically select index and query data.

## 1 Create Flow Agent
```
POST _plugins/_ml/agents/_register
{
  "name": "Query DSL Translator Agent",
  "type": "flow",
  "description": "This is a demo agent for translating NLQ to OpenSearcdh DSL",
  "tools": [
    {
      "type": "IndexMappingTool",
      "include_output_in_agent_response": false
    },
    {
      "type": "SearchIndexTool",
      "parameters": {
        "input": "{\"index\": \"${parameters.index_name}\", \"query\": ${parameters.query_dsl} }",
        "query_dsl": "{\"size\":2,\"query\":{\"match_all\":{}}}"
      },
      "include_output_in_agent_response": false
    },
    {
      "type": "MLModelTool",
      "name": "generate_os_query_dsl",
      "description": "A tool to generate OpenSearch query DSL with natrual language question.",
      "parameters": {
        "response_filter": "$.output.message.content[0].text",
        "model_id": "your_llm_id_from_approach1_step_2",
        "embedding_model_id": "your_embedding_model_id_from_approach1_step1.1",
        "system_prompt": "You are an OpenSearch query generator that converts natural language questions into precise OpenSearch Query DSL JSON. Your ONLY response should be the valid JSON DSL query without any explanations or additional text.\n\nFollow these rules:\n1. Analyze the index mapping and sample document first, use the exact field name, DON'T use non-existing field name in generated query DSL\n2. Analyze the question to identify search criteria, filters, sorting, and result limits\n3. Extract specific parameters (fields, values, operators, size) mentioned in the question\n4. Apply the most appropriate query type (match, match_all, term, range, bool, etc.)\n5. Return ONLY the JSON query DSL with proper formatting\n\nNEURAL SEARCH GUIDANCE:\n1. OpenSearch KNN index can be identified index settings with `\"knn\": \"true\",`; or in index mapping with any field with `\"type\": \"knn_vector\"`\n2. If search KNN index, prefer to use OpenSearch neural search query which is semantic based search, and has better accuracy.\n3. OpenSearch neural search needs embedding model id, please always use this model id \"${parameters.embedding_model_id}\"\n4. In KNN indices, embedding fields follow the pattern: {text_field_name}_embedding. For example, the raw text input is \"description\", then the generated embedding for this field will be saved into KNN field \"description_embedding\". \n5. Always exclude embedding fields from search results as they contain vector arrays that clutter responses\n6. Embedding fields can be identified in index mapping with \"type\": \"knn_vector\"\n7. OpenSearch neural search query will use embedding field (knn_vector type) and embedding model id. \n\nQUERY CONSTRUCTION:\nWhen constructing neural search queries, follow this pattern:\n{\n  \"_source\": {\n    \"excludes\": [\n      \"{field_name}_embedding\"\n    ]\n  },\n  \"query\": {\n    \"neural\": {\n      \"{field_name}_embedding\": {\n        \"query_text\": \"your query here\",\n        \"model_id\": \"${parameters.embedding_model_id}\"\n      }\n    }\n  }\n}\n\nExamples:\n\nQuestion: retrieve 5 documents from index test_data\n{\"query\":{\"match_all\":{}},\"size\":5}\n\nQuestion: find documents where the field title contains machine learning\n{\"query\":{\"match\":{\"title\":\"machine learning\"}}}\n\nQuestion: search for documents with the phrase artificial intelligence in the content field and return top 10 results\n{\"query\":{\"match_phrase\":{\"content\":\"artificial intelligence\"}},\"size\":10}\n\nQuestion: get documents where price is greater than 100 and category is electronics\n{\"query\":{\"bool\":{\"must\":[{\"range\":{\"price\":{\"gt\":100}}},{\"term\":{\"category\":\"electronics\"}}]}}}\n\nQuestion: find the average rating of products in the electronics category\n{\"query\":{\"term\":{\"category\":\"electronics\"}},\"aggs\":{\"avg_rating\":{\"avg\":{\"field\":\"rating\"}}},\"size\":0}\n\nQuestion: return documents sorted by date in descending order, limit to 15 results\n{\"query\":{\"match_all\":{}},\"sort\":[{\"date\":{\"order\":\"desc\"}}],\"size\":15}\n\nQuestion: which book has the introduction of AWS AgentCore\n{\"_source\":{\"excludes\":[\"book_content_embedding\"]},\"query\":{\"neural\":{\"book_content_embedding\":{\"query_text\":\"which book has the introduction of AWS AgentCore\"}}}}\n",
        "prompt": "The index mappoing of ${parameters.index_name}:\n${parameters.IndexMappingTool.output:-}\n\nThe sample documents of ${parameters.index_name}:\n${parameters.SearchIndexTool.output:-}\n\nPlease generate the OpenSearch query dsl for the question:\n${parameters.question}"
      },
      "include_output_in_agent_response": false
    },
    {
      "type": "SearchIndexTool",
      "name": "search_index_with_llm_generated_dsl",
      "include_output_in_agent_response": false,
      "parameters": {
        "input": "{\"index\": \"${parameters.index_name}\", \"query\": ${parameters.query_dsl} }",
        "query_dsl": "${parameters.generate_os_query_dsl.output}",
        "return_full_response": true
      },
      "attributes": {
        "required_parameters": [
          "index_name",
          "query_dsl",
          "generate_os_query_dsl.output"
        ]
      }
    }
  ]
}
```
Sample response
```
{
"agent_id": "NM3Md5gBdXuU4dFd8GPC"
}
```
Note the agent id , will use it to create chat agent

Test
```
POST _plugins/_ml/agents/NM3Md5gBdXuU4dFd8GPC/_execute
{
  "parameters": {
    "question": "What's the population of Seattle?",
    "index_name": "test_population_data"
  }
}
```
Sample response
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "search_index_with_llm_generated_dsl",
          "dataAsMap": {
            "_shards": {
              "total": 1,
              "failed": 0,
              "successful": 1,
              "skipped": 0
            },
            "hits": {
              "hits": [
                {
                  "_index": "test_population_data",
                  "_source": {
                    "population_description": """The current metro area population of Seattle in 2025 is 3,581,000, a 0.9% increase from 2024.
The metro area population of Seattle in 2024 was 3,549,000, a 0.85% increase from 2023.
The metro area population of Seattle in 2023 was 3,519,000, a 0.86% increase from 2022.
The metro area population of Seattle in 2022 was 3,489,000, a 0.81% increase from 2021."""
                  },
                  "_id": "XHYHc5gB7m8g_uZgFJD8",
                  "_score": 0.6568403
                },
                {
                  "_index": "test_population_data",
                  "_source": {
                    "population_description": """The current metro area population of New York City in 2025 is 19,154,000, a 0.63% increase from 2024.
The metro area population of New York City in 2024 was 19,034,000, a 0.51% increase from 2023.
The metro area population of New York City in 2023 was 18,937,000, a 0.37% increase from 2022.
The metro area population of New York City in 2022 was 18,867,000, a 0.23% increase from 2021."""
                  },
                  "_id": "WnYHc5gB7m8g_uZgFJD8",
                  "_score": 0.4017819
                },
                {
                  "_index": "test_population_data",
                  "_source": {
                    "population_description": """The current metro area population of Austin in 2025 is 2,313,000, a 1.72% increase from 2024.
The metro area population of Austin in 2024 was 2,274,000, a 2.06% increase from 2023.
The metro area population of Austin in 2023 was 2,228,000, a 2.39% increase from 2022.
The metro area population of Austin in 2022 was 2,176,000, a 2.79% increase from 2021."""
                  },
                  "_id": "W3YHc5gB7m8g_uZgFJD8",
                  "_score": 0.38802224
                },
                {
                  "_index": "test_population_data",
                  "_source": {
                    "population_description": """The current metro area population of Ogden-Layton in 2025 is 771,000, a 1.31% increase from 2024.
The metro area population of Ogden-Layton in 2024 was 761,000, a 1.47% increase from 2023.
The metro area population of Ogden-Layton in 2023 was 750,000, a 1.63% increase from 2022.
The metro area population of Ogden-Layton in 2022 was 738,000, a 1.79% increase from 2021."""
                  },
                  "_id": "WXYHc5gB7m8g_uZgFJD8",
                  "_score": 0.3659583
                }
              ],
              "total": {
                "value": 4,
                "relation": "eq"
              },
              "max_score": 0.6568403
            },
            "took": 313,
            "timed_out": false
          }
        }
      ]
    }
  ]
}
```
You can use selected tools parameter to run specific tool. For example, only run first three tools to check LLM generated OpenSearch query.

This example use the OpenSearch Sample Flights data set index `opensearch_dashboards_sample_data_flights`. https://docs.opensearch.org/latest/dashboards/quickstart/
```
POST /_plugins/_ml/agents/NM3Md5gBdXuU4dFd8GPC/_execute
{
  "parameters": {
    "question": "How many flights from Canada to USA?",
    "index_name": "opensearch_dashboards_sample_data_flights",
    "selected_tools": ["IndexMappingTool", "SearchIndexTool", "generate_os_query_dsl"]
  }
}
```
Sample output
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "generate_os_query_dsl",
          "result": """{
    "size": 0.0,
    "query": {
        "bool": {
            "must": [{
                "term": {
                    "OriginCountry": "CA"
                }
            }, {
                "term": {
                    "DestCountry": "US"
                }
            }]
        }
    },
    "aggs": {
        "flight_count": {
            "value_count": {
                "field": "FlightNum"
            }
        }
    }
}"""
        }
      ]
    }
  ]
}
```
## 2. Create Chat Agent

```
POST _plugins/_ml/agents/_register
{
  "name": "RAG Agent",
  "type": "conversational",
  "description": "this is a test agent",
  "app_type": "rag",
  "llm": {
    "model_id": "your_llm_id_from_approach1_step_2",
    "parameters": {
      "max_iteration": 10,
      "system_prompt": "You are an expert RAG (Retrieval Augmented Generation) assistant with access to OpenSearch indices. Your primary purpose is to help users find and understand information by retrieving relevant content from the OpenSearch indices.\n\nSEARCH GUIDELINES:\n1. When a user asks for information:\n   - First determine which index to search (use ListIndexTool if unsure)\n   - For complex questions, break them down into smaller sub-questions\n   - Be specific and targeted in your search queries\n\n\nRESPONSE GUIDELINES:\n1. Always cite which index the information came from\n2. Synthesize search results into coherent, well-structured responses\n3. Use formatting (headers, bullet points) to organize information when appropriate\n4. If search results are insufficient, acknowledge limitations and suggest alternatives\n5. Maintain context from the conversation history when formulating responses\n\nPrioritize accuracy over speculation. Be professional, helpful, and concise in your interactions.",
      "prompt": "${parameters.question}"
    }
  },
  "memory": {
    "type": "conversation_index"
  },
  "parameters": {
    "_llm_interface": "bedrock/converse/claude"
  },
  "tools": [
    {
      "type": "ListIndexTool"
    },
    {
      "type": "AgentTool",
      "name": "search_opensearch_index_with_nlq",
      "include_output_in_agent_response": true,
      "description": "This tool accepts one OpenSearch index and one natrual language question and generate OpenSearch query DSL. Then query the index with generated query DSL. If the question if complex, suggest split it into smaller questions then query one by one.",
      "parameters": {
        "agent_id": "your_flow_agent_id_created_in_appoach2_step1",
        "output_filter": "$.mlModelOutputs[0].mlModelTensors[2].dataAsMap"
      },
      "attributes": {
        "required_parameters": [
          "index_name",
          "question"
        ],
        "input_schema": {
          "type": "object",
          "properties": {
            "question": {
              "type": "string",
              "description": "Natural language question"
            },
            "index_name": {
              "type": "string",
              "description": "Name of the index to query"
            }
          },
          "required": [
            "question"
          ],
          "additionalProperties": false
        },
        "strict": false
      }
    }
  ]
}
```

Sample response
```
{
  "agent_id": "5s3gd5gBdXuU4dFdymMg"
}
```
Test
```
POST _plugins/_ml/agents/5s3gd5gBdXuU4dFdymMg/_execute
{
  "parameters": {
    "question": "Any new features released in OpenSearch 3.1? What's the population of Seattle in 2025",
    "verbose": true
  }
}
```

Sample response
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "memory_id",
          "result": "883hd5gBdXuU4dFdvGMM"
        },
        {
          "name": "parent_interaction_id",
          "result": "9M3hd5gBdXuU4dFdvGMd"
        },
        {
          "name": "response",
          "result": """{"metrics":{"latencyMs":2639.0},"output":{"message":{"content":[{"text":"I\u0027ll help you with information about OpenSearch 3.1 features. For your second question about Seattle\u0027s population in 2025, I\u0027ll need to check if that information is available in our indices.\n\nFirst, let\u0027s see what indices are available to search:"},{"toolUse":{"input":{"indices":[]},"name":"ListIndexTool","toolUseId":"tooluse_QvJ-aU8nTxed4A1GU8Nzyw"}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"cacheReadInputTokenCount":0.0,"cacheReadInputTokens":0.0,"cacheWriteInputTokenCount":0.0,"cacheWriteInputTokens":0.0,"inputTokens":925.0,"outputTokens":111.0,"totalTokens":1036.0}}"""
        },
        {
          "name": "response",
          "result": """row,health,status,index,uuid,pri(number of primary shards),rep(number of replica shards),docs.count(number of available documents),docs.deleted(number of deleted documents),store.size(store size of primary and replica shards),pri.store.size(store size of primary shards)
1,green,open,.plugins-ml-model-group,HU6XuHpjRp6mPe4GYn0qqQ,1,0,3,0,6.2kb,6.2kb
2,green,open,otel-v1-apm-service-map-sample,KYlj9cclQXCFWKTsNGV3Og,1,0,49,0,17.7kb,17.7kb
3,green,open,.plugins-ml-memory-meta,CCnEW7ixS_KrJdmEl330lw,1,0,428,28,161.8kb,161.8kb
4,green,open,.ql-datasources,z4aXBn1oQoSXNiCMvwapkw,1,0,0,0,208b,208b
5,yellow,open,test_tech_news,2eA8yCDIRcaKoUhDwnD4uQ,1,1,3,0,23.1kb,23.1kb
6,yellow,open,test_mem_container,VS57K2JvQmyXlJxDwJ9nhA,1,1,0,0,208b,208b
7,green,open,.plugins-ml-task,GTX8717YRtmw0nkl1vnXMg,1,0,293,29,111.5kb,111.5kb
8,green,open,ss4o_metrics-otel-sample,q5W0oUlZRuaEVtfL1dknKA,1,0,39923,0,4.3mb,4.3mb
9,green,open,.opendistro_security,3AmSFRp1R42BSFMPfXXVqw,1,0,9,0,83.7kb,83.7kb
10,yellow,open,test_population_data,fkgI4WqQSv2qHSaXRJv_-A,1,1,4,0,22.5kb,22.5kb
11,green,open,.plugins-ml-model,Ksvz5MEwR42RmrOZAg6RTQ,1,0,10,0,182.1kb,182.1kb
12,green,open,.plugins-ml-memory-container,TYUOuxfMT5i4DdOMfYe45w,1,0,2,0,9.9kb,9.9kb
13,green,open,.kibana_-969161597_admin123_1,dlCY0iSQR9aTtRJsJx9cIg,1,0,1,0,5.3kb,5.3kb
14,green,open,opensearch_dashboards_sample_data_ecommerce,HdLqBN2QQCC4nUhE-5mOkA,1,0,4675,0,4.1mb,4.1mb
15,green,open,security-auditlog-2025.08.04,Ac34ImxkQSeonGeW8RSYcw,1,0,985,0,1mb,1mb
16,green,open,.plugins-ml-memory-message,osbPSzqeQd2sCPF2yqYMeg,1,0,1862,5,2.8mb,2.8mb
17,green,open,security-auditlog-2025.08.05,hBi-5geLRqml4mDixXMPYQ,1,0,83,0,230.2kb,230.2kb
18,green,open,otel-v1-apm-span-sample,mewMjWmHSdOlGNzgz-OnXg,1,0,13061,0,6.2mb,6.2mb
19,green,open,security-auditlog-2025.08.01,lvwEW0_VQnq88o7XjJkoPw,1,0,57,0,216.9kb,216.9kb
20,green,open,.kibana_92668751_admin_1,w45HvTqcTQ-eVTWK77lx7A,1,0,242,0,125.2kb,125.2kb
21,green,open,.plugins-ml-agent,Q-kWqw9_RdW3SfhUcKBcfQ,1,0,99,0,330.6kb,330.6kb
22,green,open,security-auditlog-2025.08.03,o_iRgiWZRryXrhQ-y36JQg,1,0,847,0,575.9kb,575.9kb
23,green,open,.kibana_1,tn9JwuiSSe6gR9sInHFxDg,1,0,4,0,16.7kb,16.7kb
24,green,open,ss4o_logs-otel-sample,wIaGQx8wT1ijhxDjjbux6A,1,0,16286,0,5.6mb,5.6mb
25,green,open,.plugins-ml-config,uh1sYBiTRYy_lK61Rx0fkQ,1,0,1,0,4kb,4kb
26,green,open,opensearch_dashboards_sample_data_logs,jDDgZ_PfTkKHfaDqBl72lQ,1,0,14074,0,8.3mb,8.3mb
27,green,open,opensearch_dashboards_sample_data_flights,7dcplp7MRc2C9WTPyLRRIg,1,0,13059,0,5.8mb,5.8mb
"""
        },
        {
          "name": "response",
          "result": """{"metrics":{"latencyMs":2729.0},"output":{"message":{"content":[{"text":"Now let\u0027s search for information about OpenSearch 3.1 features:"},{"toolUse":{"input":{"index_name":"test_tech_news","question":"What are the new features in OpenSearch 3.1?"},"name":"search_opensearch_index_with_nlq","toolUseId":"tooluse_aQ7_1eYDSSePxX-RF-olFQ"}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"cacheReadInputTokenCount":0.0,"cacheReadInputTokens":0.0,"cacheWriteInputTokenCount":0.0,"cacheWriteInputTokens":0.0,"inputTokens":2577.0,"outputTokens":114.0,"totalTokens":2691.0}}"""
        },
        {
          "name": "response",
          "result": """{"_shards":{"total":1,"failed":0,"successful":1,"skipped":0},"hits":{"hits":[{"_index":"test_tech_news","_source":{"passage":"OpenSearch 3.1 release highlights: \nMakes GPU acceleration for vector index builds generally available. Introduces memory-optimized search for Faiss indexes using Lucene HNSW, semantic field type for streamlined semantic search, and Search Relevance Workbench for search quality optimization. Makes star-tree indexes generally available with support for comprehensive query types. Enhances observability with ML Commons metrics integration, custom index support for OpenTelemetry data, and new PPL commands for JSON manipulation. Improves agent management with Update Agent API and persistent MCP tools. Includes security enhancements with immutable user objects and new resource sharing framework. For a full list of release highlights, see the Release Notes."},"_id":"c3YIc5gB7m8g_uZgPpAA","_score":0.7774649},{"_index":"test_tech_news","_source":{"passage":"OpenSearch 3.0 release highlights:\nUpgrades to Lucene 10 for improved indexing and vector search. Adds experimental gRPC support and pull-based ingestion from Kafka and Kinesis. Introduces GPU acceleration for vector operations and semantic sentence highlighting. Improves range query performance and hybrid search with z-score normalization. Adds plan-execute-reflect agents and native MCP protocol support for agentic workflows. Enhances security with a new Java agent replacing the Security Manager. Includes PPL query improvements with lookup, join, and subsearch commands. For a full list of release highlights, see the Release Notes."},"_id":"dHYIc5gB7m8g_uZgPpAA","_score":0.6750423},{"_index":"test_tech_news","_source":{"passage":"Announcing Amazon S3 Vectors (Preview)—First cloud object storage with native support for storing and querying vectors:\nAmazon S3 Vectors delivers purpose-built, cost-optimized vector storage for AI agents, AI inference, and semantic search of your content stored in Amazon S3. By reducing the cost of uploading, storing, and querying vectors by up to 90%, S3 Vectors makes it cost-effective to create and use large vector datasets to improve the memory and context of AI agents as well as semantic search results of your S3 data. Designed to provide the same elasticity, scale, and durability as Amazon S3, S3 Vectors lets you store and search data with sub-second query performance. It\u0027s ideal for applications that need to build and maintain vector indexes so you can organize and search through massive amounts of information. S3 Vectors provides a simple and flexible API for operations such as finding similar scenes in petabyte-scale video archives, identifying collections of related business documents, or detecting rare patterns in diagnostic collections including millions of medical images.\nS3 Vectors is natively integrated with Amazon Bedrock Knowledge Bases so that you can reduce the cost of using large vector datasets for retrieval-augmented generation (RAG). You can also use S3 Vectors with Amazon OpenSearch Service to lower storage costs for infrequent queried vectors, and then quickly move them to OpenSearch as demands increase or to enhance search capabilities.\nS3 Vectors introduces a new bucket type optimized for durable, low-cost vector storage. It includes a dedicated set of APIs for you to store, access, and query vectors without provisioning any infrastructure. Within a vector bucket, you can organize vector data within vector indexes and elastically scale up to 10,000 indexes per bucket. When creating a Knowledge Base in Amazon Bedrock or Amazon SageMaker Unified Studio, you can select an S3 vector index as your vector store or use the Quick Create workflow to set one up. Within OpenSearch, you can adopt a tiered strategy to store large vector datasets in S3 for near real-time access while effortlessly activating the vector data with the highest performance requirements in OpenSearch."},"_id":"dXYIc5gB7m8g_uZgPpAA","_score":0.39090624}],"total":{"value":3,"relation":"eq"},"max_score":0.7774649},"took":113,"timed_out":false}"""
        },
        {
          "name": "response",
          "result": """{"metrics":{"latencyMs":3873.0},"output":{"message":{"content":[{"text":"For your second question about Seattle\u0027s population in 2025, let me check the population data index:"},{"toolUse":{"input":{"question":"What is the population of Seattle in 2025?","index_name":"test_population_data"},"name":"search_opensearch_index_with_nlq","toolUseId":"tooluse_lIKrohT0SAumqMpHk2jkQQ"}}],"role":"assistant"}},"stopReason":"tool_use","usage":{"cacheReadInputTokenCount":0.0,"cacheReadInputTokens":0.0,"cacheWriteInputTokenCount":0.0,"cacheWriteInputTokens":0.0,"inputTokens":3750.0,"outputTokens":117.0,"totalTokens":3867.0}}"""
        },
        {
          "name": "response",
          "result": """{"_shards":{"total":1,"failed":0,"successful":1,"skipped":0},"hits":{"hits":[{"_index":"test_population_data","_source":{"population_description":"The current metro area population of Seattle in 2025 is 3,581,000, a 0.9% increase from 2024.\nThe metro area population of Seattle in 2024 was 3,549,000, a 0.85% increase from 2023.\nThe metro area population of Seattle in 2023 was 3,519,000, a 0.86% increase from 2022.\nThe metro area population of Seattle in 2022 was 3,489,000, a 0.81% increase from 2021."},"_id":"XHYHc5gB7m8g_uZgFJD8","_score":0.74888927},{"_index":"test_population_data","_source":{"population_description":"The current metro area population of New York City in 2025 is 19,154,000, a 0.63% increase from 2024.\nThe metro area population of New York City in 2024 was 19,034,000, a 0.51% increase from 2023.\nThe metro area population of New York City in 2023 was 18,937,000, a 0.37% increase from 2022.\nThe metro area population of New York City in 2022 was 18,867,000, a 0.23% increase from 2021."},"_id":"WnYHc5gB7m8g_uZgFJD8","_score":0.4503471},{"_index":"test_population_data","_source":{"population_description":"The current metro area population of Austin in 2025 is 2,313,000, a 1.72% increase from 2024.\nThe metro area population of Austin in 2024 was 2,274,000, a 2.06% increase from 2023.\nThe metro area population of Austin in 2023 was 2,228,000, a 2.39% increase from 2022.\nThe metro area population of Austin in 2022 was 2,176,000, a 2.79% increase from 2021."},"_id":"W3YHc5gB7m8g_uZgFJD8","_score":0.42577574},{"_index":"test_population_data","_source":{"population_description":"The current metro area population of Ogden-Layton in 2025 is 771,000, a 1.31% increase from 2024.\nThe metro area population of Ogden-Layton in 2024 was 761,000, a 1.47% increase from 2023.\nThe metro area population of Ogden-Layton in 2023 was 750,000, a 1.63% increase from 2022.\nThe metro area population of Ogden-Layton in 2022 was 738,000, a 1.79% increase from 2021."},"_id":"WXYHc5gB7m8g_uZgFJD8","_score":0.37380075}],"total":{"value":4,"relation":"eq"},"max_score":0.74888927},"took":86,"timed_out":false}"""
        },
        {
          "name": "response",
          "result": """# OpenSearch 3.1 New Features

Based on information retrieved from the `test_tech_news` index, OpenSearch 3.1 includes the following new features and improvements:

## Vector Search Enhancements
- **GPU Acceleration**: Now generally available for vector index builds
- **Memory-optimized Search**: For Faiss indexes using Lucene HNSW
- **Semantic Field Type**: New feature for streamlined semantic search

## Performance and Data Management
- **Star-tree Indexes**: Now generally available with support for comprehensive query types
- **Search Relevance Workbench**: New tool for search quality optimization

## Observability Improvements
- **ML Commons Metrics Integration**: Enhanced metrics tracking
- **Custom Index Support**: For OpenTelemetry data
- **New PPL Commands**: Added capabilities for JSON manipulation

## Agent Management
- **Update Agent API**: For improved agent management
- **Persistent MCP Tools**: Enhanced management capabilities

## Security Enhancements
- **Immutable User Objects**: Strengthened security controls
- **New Resource Sharing Framework**: Improved resource management

# Seattle Population in 2025

According to data from the `test_population_data` index, the metro area population of Seattle in 2025 is 3,581,000, which represents a 0.9% increase from 2024.

For context, the population in previous years was:
- 2024: 3,549,000 (0.85% increase from 2023)
- 2023: 3,519,000 (0.86% increase from 2022)
- 2022: 3,489,000 (0.81% increase from 2021)

The data shows Seattle has maintained a steady growth rate of around 0.8-0.9% annually in recent years."""
        }
      ]
    }
  ]
}
```