# Topic

> Agent Framework is an experimental feature released in OpenSearch 2.12 and not recommended for use in a production environment. For updates on the progress of the feature or if you want to leave feedback, see the associated [GitHub issue](https://github.com/opensearch-project/ml-commons/issues/1161).

> This tutorial doesn't explain what's retrieval-augmented generation(RAG).

This tutorial explains how to use conversational agent to build intelligent RAG application by leveraging your 
OpenSearch data as knowledge base.

Note: You should replace the placeholders with prefix `your_` with your own value

# Steps

## 0. Preparation

To build RAG application, we need to have some OpenSearch index as knowledge base. In this tutorial, we
are going to use [k-NN index](https://opensearch.org/docs/latest/search-plugins/knn/knn-index/) and 
[semantic search](https://opensearch.org/docs/latest/search-plugins/semantic-search/). You can read more 
details on their document and this [tutorial](https://opensearch.org/docs/latest/search-plugins/neural-search-tutorial/).
It's totally fine to just follow below steps to quick start.

### update cluster setting

If you have dedicated ML node, you don't need to set `"plugins.ml_commons.only_run_on_ml_node": false`.

We set `"plugins.ml_commons.native_memory_threshold"` as 100% to avoid triggering native memory circuit breaker.
```
PUT _cluster/settings
{
    "persistent": {
        "plugins.ml_commons.only_run_on_ml_node": false,
        "plugins.ml_commons.native_memory_threshold": 100,
        "plugins.ml_commons.agent_framework_enabled": true,
        "plugins.ml_commons.memory_feature_enabled": true
    }
}
```

## 1. Prepare knowledge base

### 1.1 register text embedding model

Find more details on [pretrained model](https://opensearch.org/docs/latest/ml-commons-plugin/pretrained-models/)

1. Upload model:
```
POST /_plugins/_ml/models/_register
{
  "name": "huggingface/sentence-transformers/all-MiniLM-L12-v2",
  "version": "1.0.1",
  "model_format": "TORCH_SCRIPT"
}
```
Find model id by calling get task API.

Copy the text embedding model id, will use it in following steps.
```
GET /_plugins/_ml/tasks/your_task_id
```
2. Deploy model
```
POST /_plugins/_ml/models/your_text_embedding_model_id/_deploy
```
3. Test predict
```
POST /_plugins/_ml/models/your_text_embedding_model_id/_predict
{
  "text_docs":[ "today is sunny"],
  "return_number": true,
  "target_response": ["sentence_embedding"]
}
```

### 1.2 create test population ingest pipeline and k-NN index

1. Create ingest pipeline 

Create pipeline with text embedding processor which can invoke model created in step1.1 to translate text
field to embedding.

```
PUT /_ingest/pipeline/test_population_data_pipeline
{
    "description": "text embedding pipeline",
    "processors": [
        {
            "text_embedding": {
                "model_id": "your_text_embedding_model_id",
                "field_map": {
                    "population_description": "population_description_embedding"
                }
            }
        }
    ]
}
```

2. create k-NN index with the ingest pipeline.
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
        "dimension": 384
      }
    }
  },
  "settings": {
    "index": {
      "knn.space_type": "cosinesimil",
      "default_pipeline": "test_population_data_pipeline",
      "knn": "true"
    }
  }
}
```

3. Ingest test data
```
POST _bulk
{"index": {"_index": "test_population_data"}}
{"population_description": "Chart and table of population level and growth rate for the Ogden-Layton metro area from 1950 to 2023. United Nations population projections are also included through the year 2035.\nThe current metro area population of Ogden-Layton in 2023 is 750,000, a 1.63% increase from 2022.\nThe metro area population of Ogden-Layton in 2022 was 738,000, a 1.79% increase from 2021.\nThe metro area population of Ogden-Layton in 2021 was 725,000, a 1.97% increase from 2020.\nThe metro area population of Ogden-Layton in 2020 was 711,000, a 2.16% increase from 2019."}
{"index": {"_index": "test_population_data"}}
{"population_description": "Chart and table of population level and growth rate for the New York City metro area from 1950 to 2023. United Nations population projections are also included through the year 2035.\\nThe current metro area population of New York City in 2023 is 18,937,000, a 0.37% increase from 2022.\\nThe metro area population of New York City in 2022 was 18,867,000, a 0.23% increase from 2021.\\nThe metro area population of New York City in 2021 was 18,823,000, a 0.1% increase from 2020.\\nThe metro area population of New York City in 2020 was 18,804,000, a 0.01% decline from 2019."}
{"index": {"_index": "test_population_data"}}
{"population_description": "Chart and table of population level and growth rate for the Chicago metro area from 1950 to 2023. United Nations population projections are also included through the year 2035.\\nThe current metro area population of Chicago in 2023 is 8,937,000, a 0.4% increase from 2022.\\nThe metro area population of Chicago in 2022 was 8,901,000, a 0.27% increase from 2021.\\nThe metro area population of Chicago in 2021 was 8,877,000, a 0.14% increase from 2020.\\nThe metro area population of Chicago in 2020 was 8,865,000, a 0.03% increase from 2019."}
{"index": {"_index": "test_population_data"}}
{"population_description": "Chart and table of population level and growth rate for the Miami metro area from 1950 to 2023. United Nations population projections are also included through the year 2035.\\nThe current metro area population of Miami in 2023 is 6,265,000, a 0.8% increase from 2022.\\nThe metro area population of Miami in 2022 was 6,215,000, a 0.78% increase from 2021.\\nThe metro area population of Miami in 2021 was 6,167,000, a 0.74% increase from 2020.\\nThe metro area population of Miami in 2020 was 6,122,000, a 0.71% increase from 2019."}
{"index": {"_index": "test_population_data"}}
{"population_description": "Chart and table of population level and growth rate for the Austin metro area from 1950 to 2023. United Nations population projections are also included through the year 2035.\\nThe current metro area population of Austin in 2023 is 2,228,000, a 2.39% increase from 2022.\\nThe metro area population of Austin in 2022 was 2,176,000, a 2.79% increase from 2021.\\nThe metro area population of Austin in 2021 was 2,117,000, a 3.12% increase from 2020.\\nThe metro area population of Austin in 2020 was 2,053,000, a 3.43% increase from 2019."}
{"index": {"_index": "test_population_data"}}
{"population_description": "Chart and table of population level and growth rate for the Seattle metro area from 1950 to 2023. United Nations population projections are also included through the year 2035.\\nThe current metro area population of Seattle in 2023 is 3,519,000, a 0.86% increase from 2022.\\nThe metro area population of Seattle in 2022 was 3,489,000, a 0.81% increase from 2021.\\nThe metro area population of Seattle in 2021 was 3,461,000, a 0.82% increase from 2020.\\nThe metro area population of Seattle in 2020 was 3,433,000, a 0.79% increase from 2019."}

```

### 1.3 create test stock price ingest pipeline and k-NN index

1. Create ingest pipeline

Create pipeline with text embedding processor which can invoke model created in step1.1 to translate text
field to embedding.

```
PUT /_ingest/pipeline/test_stock_price_data_pipeline
{
    "description": "text embedding pipeline",
    "processors": [
        {
            "text_embedding": {
                "model_id": "your_text_embedding_model_id",
                "field_map": {
                    "stock_price_history": "stock_price_history_embedding"
                }
            }
        }
    ]
}
```

2. create k-NN index with the ingest pipeline.
```
PUT test_stock_price_data
{
  "mappings": {
    "properties": {
      "stock_price_history": {
        "type": "text"
      },
      "stock_price_history_embedding": {
        "type": "knn_vector",
        "dimension": 384
      }
    }
  },
  "settings": {
    "index": {
      "knn.space_type": "cosinesimil",
      "default_pipeline": "test_stock_price_data_pipeline",
      "knn": "true"
    }
  }
}
```

3. Ingest test data
```
POST _bulk
{"index": {"_index": "test_stock_price_data"}}
{"stock_price_history": "This is the historical montly stock price record for Amazon.com, Inc. (AMZN) with CSV format.\nDate,Open,High,Low,Close,Adj Close,Volume\n2023-03-01,93.870003,103.489998,88.120003,103.290001,103.290001,1349240300\n2023-04-01,102.300003,110.860001,97.709999,105.449997,105.449997,1224083600\n2023-05-01,104.949997,122.919998,101.150002,120.580002,120.580002,1432891600\n2023-06-01,120.690002,131.490005,119.930000,130.360001,130.360001,1242648800\n2023-07-01,130.820007,136.649994,125.919998,133.679993,133.679993,1058754800\n2023-08-01,133.550003,143.630005,126.410004,138.009995,138.009995,1210426200\n2023-09-01,139.460007,145.860001,123.040001,127.120003,127.120003,1120271900\n2023-10-01,127.279999,134.479996,118.349998,133.089996,133.089996,1224564700\n2023-11-01,133.960007,149.259995,133.710007,146.089996,146.089996,1025986900\n2023-12-01,146.000000,155.630005,142.809998,151.940002,151.940002,931128600\n2024-01-01,151.539993,161.729996,144.050003,155.199997,155.199997,953344900\n2024-02-01,155.869995,175.000000,155.619995,174.449997,174.449997,437720800\n"}
{"index": {"_index": "test_stock_price_data"}}
{"stock_price_history": "This is the historical montly stock price record for Apple Inc. (AAPL) with CSV format.\nDate,Open,High,Low,Close,Adj Close,Volume\n2023-03-01,146.830002,165.000000,143.899994,164.899994,164.024475,1520266600\n2023-04-01,164.270004,169.850006,159.779999,169.679993,168.779099,969709700\n2023-05-01,169.279999,179.350006,164.309998,177.250000,176.308914,1275155500\n2023-06-01,177.699997,194.479996,176.929993,193.970001,193.207016,1297101100\n2023-07-01,193.779999,198.229996,186.600006,196.449997,195.677261,996066400\n2023-08-01,196.240005,196.729996,171.960007,187.869995,187.130997,1322439400\n2023-09-01,189.490005,189.979996,167.619995,171.210007,170.766846,1337586600\n2023-10-01,171.220001,182.339996,165.669998,170.770004,170.327972,1172719600\n2023-11-01,171.000000,192.929993,170.119995,189.949997,189.458313,1099586100\n2023-12-01,190.330002,199.619995,187.449997,192.529999,192.284637,1062774800\n2024-01-01,187.149994,196.380005,180.169998,184.399994,184.164993,1187219300\n2024-02-01,183.990005,191.050003,179.250000,188.850006,188.609329,420063900\n"}
{"index": {"_index": "test_stock_price_data"}}
{"stock_price_history": "This is the historical montly stock price record for NVIDIA Corporation (NVDA) with CSV format.\nDate,Open,High,Low,Close,Adj Close,Volume\n2023-03-01,231.919998,278.339996,222.970001,277.769989,277.646820,1126373100\n2023-04-01,275.089996,281.100006,262.200012,277.489990,277.414032,743592100\n2023-05-01,278.399994,419.380005,272.399994,378.339996,378.236420,1169636000\n2023-06-01,384.890015,439.899994,373.559998,423.019989,422.904175,1052209200\n2023-07-01,425.170013,480.880005,413.459991,467.290009,467.210449,870489500\n2023-08-01,464.600006,502.660004,403.109985,493.549988,493.465942,1363143600\n2023-09-01,497.619995,498.000000,409.799988,434.989990,434.915924,857510100\n2023-10-01,440.299988,476.089996,392.299988,407.799988,407.764130,1013917700\n2023-11-01,408.839996,505.480011,408.690002,467.700012,467.658905,914386300\n2023-12-01,465.250000,504.329987,450.100006,495.220001,495.176453,740951700\n2024-01-01,492.440002,634.929993,473.200012,615.270020,615.270020,970385300\n2024-02-01,621.000000,721.849976,616.500000,721.330017,721.330017,355346500\n"}

```
## 2. Prepare LLM

Find more details on [Remote model](https://opensearch.org/docs/latest/ml-commons-plugin/remote-models/index/)

We use [Bedrock Claude model](https://aws.amazon.com/bedrock/claude/) in this tutorial. You can also use OpenAI or other LLM.

1. Create connector
```
POST /_plugins/_ml/connectors/_create
{
  "name": "BedRock Claude instant-v1 Connector ",
  "description": "The connector to BedRock service for claude model",
  "version": 1,
  "protocol": "aws_sigv4",
  "parameters": {
    "region": "us-east-1",
    "service_name": "bedrock",
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens_to_sample": 8000,
    "temperature": 0.0001,
    "response_filter": "$.completion"
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
      "url": "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-instant-v1/invoke",
      "headers": {
        "content-type": "application/json",
        "x-amz-content-sha256": "required"
      },
      "request_body": "{\"prompt\":\"${parameters.prompt}\", \"max_tokens_to_sample\":${parameters.max_tokens_to_sample}, \"temperature\":${parameters.temperature},  \"anthropic_version\":\"${parameters.anthropic_version}\" }"
    }
  ]
}
```

Copy the connector id from the response. 

2. register model

```
POST /_plugins/_ml/models/_register
{
    "name": "Bedrock Claude Instant model",
    "function_name": "remote",
    "description": "Bedrock Claude instant-v1 model",
    "connector_id": "your_LLM_connector_id"
}
```
Copy the LLM model id from the response, will use it in following steps.

3. Deploy model
```
POST /_plugins/_ml/models/your_LLM_model_id/_deploy
```

4. Test predict
```
POST /_plugins/_ml/models/your_LLM_model_id/_predict
{
  "parameters": {
    "prompt": "\n\nHuman: how are you? \n\nAssistant:"
  }
}
```

## 3. Create Agent
Agent framework provides several agent types: `flow`, `conversational_flow` and `conversational`.

We will use `conversational` agent in this tutorial. This agent uses Reason-Action(ReAct) to solve complex problems step by step. 

The agent consists of:
1. meta info: `name`, `type`, `description`
2. `app_type`: this is to differentiate different application type
3. `memory`: this is to store agent execution result, so user can retrieve memory later and continue one conversation.
4. `llm`: large language model (LLM) which functions as reasoning engine.
5. `tools`: define a list of tools to use. Agent will use `llm` to reason which tool to use.
```
POST /_plugins/_ml/agents/_register
{
    "name": "Data analysis agent",
    "type": "conversational",
    "description": "This is a demo agent for data analysis",
    "app_type": "data_analysis",
    "memory": {
        "type": "conversation_index"
    },
    "llm": {
        "model_id": "Pw7qlI0BHcHmo_czPKdW",
        "parameters": {
            "max_iteration": 5,
            "stop_when_no_tool_found": true,
            "response_filter": "$.completion"
        }
    },
    "tools": [
        {
            "type": "VectorDBTool",
            "name": "population_knowledge_base",
            "description": "This tool provides population data.",
            "parameters": {
                "model_id": "Ow7RlI0BHcHmo_cz5aYk",
                "index": "test_population_data",
                "embedding_field": "population_description_embedding",
                "source_field": [
                    "population_description"
                ],
                "input": "${parameters.question}"
            }
        },
        {
            "type": "VectorDBTool",
            "name": "stock_price_knowledge_base",
            "description": "This tool provides historical stock price data",
            "parameters": {
                "model_id": "Ow7RlI0BHcHmo_cz5aYk",
                "index": "test_stock_price_data",
                "embedding_field": "stock_price_history_embedding",
                "source_field": [
                    "stock_price_history"
                ],
                "input": "${parameters.question}"
            }
        }
    ]
}
```

Sample response
```
{
  "agent_id": "cQ5ilY0BHcHmo_czbqmq"
}
```

Copy the agent id, will use it in next step. 

## 4. Execute Agent

The agent will reason which tool to use based on the question. 

### 4.1 ask questions related to population

Run the agent to analyze Seattle population increase.

When run this agent, it will create a new conversation. 
Later you can continue the conversation by asking other questions.

```
POST /_plugins/_ml/agents/cQ5ilY0BHcHmo_czbqmq/_execute
{
  "parameters": {
    "question": "What's the increased population of Seattle from 2021 to 2023? List the detail number and calculation process"
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
          "result": "kw5llY0BHcHmo_czCqnP"
        },
        {
          "name": "parent_interaction_id",
          "result": "lA5llY0BHcHmo_czCqna"
        },
        {
          "name": "response",
          "dataAsMap": {
            "response": "The increased population of Seattle from 2021 to 2023 was 58,000. The population in 2021 was 3,461,000. The population in 2023 was 3,519,000. The growth rate was 0.86%.",
            "additional_info": {}
          }
        }
      ]
    }
  ]
}
```
Explanation of the output:
1. `memory_id` means the conversation id, copy it as we will use in Step4.2
2. `parent_message_id` means the current interaction (one round of question/answer), one conversation can have multiple interactions

Check more details of conversation by calling get memory API.
```
GET /_plugins/_ml/memory/kw5llY0BHcHmo_czCqnP

GET /_plugins/_ml/memory/kw5llY0BHcHmo_czCqnP/messages
```
Check more details of interaction by calling get message API.
```
GET /_plugins/_ml/memory/message/lA5llY0BHcHmo_czCqna
```
For debugging purpose, each interaction/message has its own trace data, you can find trace data by calling
```
GET /_plugins/_ml/memory/message/lA5llY0BHcHmo_czCqna/traces
```
### 4.2 ask questions related to stock price
```
POST /_plugins/_ml/agents/QA5WlY0BHcHmo_czQqmg/_execute
{
  "parameters": {
    "question": "What's the stock price increase of Amazon from April 2023 to Jan 2024?"
  }
}
```
sample response
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "memory_id",
          "result": "kw5llY0BHcHmo_czCqnP"
        },
        {
          "name": "parent_interaction_id",
          "result": "qw5olY0BHcHmo_cz4Kkg"
        },
        {
          "name": "response",
          "dataAsMap": {
            "response": "The population of New York City in 2023 is approximately 18,937,000 people, which is a slight increase from 2022. The population of Seattle in 2023 is approximately 3,519,000 people, also a slight increase from the previous year. While New York City has a significantly larger population, both cities have experienced small increases in recent years.",
            "additional_info": {}
          }
        }
      ]
    }
  ]
}
```

### 4.3 Continue a conversation by asking new question

Continue last conversation by providing memory id from step4.1

Explanation of the input:
1. `message_history_limit`: specify how many historical messages included in this interaction.
2. `prompt`: use can customize prompt. For example, this example adds a new instruction `always learn useful information from chat history` 
and a new parameter `next_action`.

```
POST /_plugins/_ml/agents/cQ5ilY0BHcHmo_czbqmq/_execute
{
  "parameters": {
    "question": "What's the population of New York City in 2023? Compare with Seattle population of 2023",
    "memory_id": "kw5llY0BHcHmo_czCqnP",
    "message_history_limit": 3
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
          "result": "kw5llY0BHcHmo_czCqnP"
        },
        {
          "name": "parent_interaction_id",
          "result": "qw5olY0BHcHmo_cz4Kkg"
        },
        {
          "name": "response",
          "dataAsMap": {
            "response": "The population of New York City in 2023 is approximately 18,937,000 people, which is a slight increase from 2022. The population of Seattle in 2023 is approximately 3,519,000 people, also a slight increase from the previous year. While New York City has a significantly larger population, both cities have experienced small increases in recent years.",
            "additional_info": {}
          }
        }
      ]
    }
  ]
}
```
