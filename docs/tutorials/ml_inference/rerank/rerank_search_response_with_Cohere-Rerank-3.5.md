## Step1 Create model

Note: You can also create connector first, then create model with connector id.

Create model
```
PUT _plugins/_ml/models/_register
{
  "name": "Bedrock Cohere Rerank model",
  "function_name": "remote",
  "description": "Bedrock Cohere Rerank model",
  "connector": {
    "name": "Amazon Bedrock Connector: Cohere Rerank 3.5",
    "description": "The connector to Bedrock Cohere Rerank 3.5 model",
    "version": 1,
    "protocol": "aws_sigv4",
    "parameters": {
      "region": "us-west-2",
      "service_name": "bedrock",
      "model": "cohere.rerank-v3-5:0",
      "api_version": 2
    },
    "credential": {
      "access_key": "{{ _.access_key }}",
      "secret_key": "{{ _.secret_key }}",
      "session_token": "{{ _.session_token }}"
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
        "request_body": "{\"query\":\"${parameters.query}\", \"documents\": ${parameters.documents}, \"top_n\": ${parameters.top_n}, \"api_version\": ${parameters.api_version}}",
        "post_process_function": "connector.post_process.cohere.rerank"
      }
    ]
  }
}
```
Sample response
```
{
  "task_id": "wD_EsZUB9v7eY1pFN_-P",
  "status": "CREATED",
  "model_id": "wT_EsZUB9v7eY1pFN__H"
}
```
Test model by running predict
```
POST _plugins/_ml/models/wT_EsZUB9v7eY1pFN__H/_predict
{
  "parameters": {
    "top_n": 10,
    "query": "What is the capital of the United States?",
    "documents": [
      "Carson City is the capital city of the American state of Nevada.",
      "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
      "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
      "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
      "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."
    ]
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
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            0.17279942
          ]
        },
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            0.08885999
          ]
        },
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            0.0865164
          ]
        },
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            0.8742601
          ]
        },
        {
          "name": "similarity",
          "data_type": "FLOAT32",
          "shape": [
            1
          ],
          "data": [
            0.10787861
          ]
        }
      ],
      "status_code": 200
    }
  ]
}
```
## Step2 Load test data
```
POST _bulk
{ "index": { "_index": "test_data" } }
{ "test_text": "Carson City is the capital city of the American state of Nevada." }
{ "index": { "_index": "test_data" } }
{ "test_text":  "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan."}
{ "index": { "_index": "test_data" } }
{ "test_text":  "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages."}
{ "index": { "_index": "test_data" } }
{ "test_text":  "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district."}
{ "index": { "_index": "test_data" } }
{ "test_text":  "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."}

```

## Step3 Create search pipeline
```
PUT _search/pipeline/cohere_pipeline
{
  "response_processors": [
    {
      "ml_inference": {
        "model_id": "wT_EsZUB9v7eY1pFN__H",
        "input_map": [
          {
            "documents": "test_text",
            "query": "_request.ext.query_context.query_text",
            "top_n": "_request.ext.query_context.top_n"
          }
        ],
        "output_map": [
          {
            "relevance_score": "$.inference_results[*].output[*].data[0]"
          }
        ],
        "full_response_path": true,
        "ignore_missing": false,
        "ignore_failure": false,
        "one_to_one": false,
        "override": false,
        "model_config": {}
      }
    },
    {
      "rerank": {
        "by_field": {
          "target_field": "relevance_score",
          "remove_target_field": true,
          "keep_previous_score": true,
          "ignore_failure": false
        }
      }
    }
  ]
}
```

## Step4 Test

### Step 4.1 Test without search pipeline
```
GET test_data/_search
{
  "query": {
    "match": {
      "test_text": "What is the capital of the United States?"
    }
  }
}
```
Search response:

Note : we can see the correct answer ranked at second.
```
{
  "took": 13,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 5,
      "relation": "eq"
    },
    "max_score": 2.9798057,
    "hits": [
      {
        "_index": "test_data",
        "_id": "yD_IsZUB9v7eY1pFsf-I",
        "_score": 2.9798057,
        "_source": {
          "test_text": "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states."
        }
      },
      {
        "_index": "test_data",
        "_id": "xz_IsZUB9v7eY1pFsf-I",
        "_score": 2.1746304,
        "_source": {
          "test_text": "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district."
        }
      },
      {
        "_index": "test_data",
        "_id": "xD_IsZUB9v7eY1pFsf-I",
        "_score": 0.6310567,
        "_source": {
          "test_text": "Carson City is the capital city of the American state of Nevada."
        }
      },
      {
        "_index": "test_data",
        "_id": "xT_IsZUB9v7eY1pFsf-I",
        "_score": 0.62163705,
        "_source": {
          "test_text": "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan."
        }
      },
      {
        "_index": "test_data",
        "_id": "xj_IsZUB9v7eY1pFsf-I",
        "_score": 0.5046487,
        "_source": {
          "test_text": "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages."
        }
      }
    ]
  }
}
```

### Step 4.1 Test with search pipeline
```
GET test_data/_search?search_pipeline=cohere_pipeline
{
  "query": {
    "match": {
      "test_text": "What is the capital of the United States?"
    }
  },
  "size": 5,
  "ext": {
    "rerank": {
      "query_context": {
        "query_text": "What is the capital of the United States?",
        "top_n": "10"
      }
    }
  }
}
```
Search response: 

We can see the correct answer ranked at first.
```
{
  "took": 1,
  "timed_out": false,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  },
  "hits": {
    "total": {
      "value": 5,
      "relation": "eq"
    },
    "max_score": 0.8742601,
    "hits": [
      {
        "_index": "test_data",
        "_id": "xz_IsZUB9v7eY1pFsf-I",
        "_score": 0.8742601,
        "_source": {
          "test_text": "Washington, D.C. (also known as simply Washington or D.C., and officially as the District of Columbia) is the capital of the United States. It is a federal district.",
          "previous_score": 2.1746304
        }
      },
      {
        "_index": "test_data",
        "_id": "xD_IsZUB9v7eY1pFsf-I",
        "_score": 0.17279942,
        "_source": {
          "test_text": "Carson City is the capital city of the American state of Nevada.",
          "previous_score": 0.6310567
        }
      },
      {
        "_index": "test_data",
        "_id": "yD_IsZUB9v7eY1pFsf-I",
        "_score": 0.10787861,
        "_source": {
          "test_text": "Capital punishment has existed in the United States since beforethe United States was a country. As of 2017, capital punishment is legal in 30 of the 50 states.",
          "previous_score": 2.9798057
        }
      },
      {
        "_index": "test_data",
        "_id": "xT_IsZUB9v7eY1pFsf-I",
        "_score": 0.08885999,
        "_source": {
          "test_text": "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean. Its capital is Saipan.",
          "previous_score": 0.62163705
        }
      },
      {
        "_index": "test_data",
        "_id": "xj_IsZUB9v7eY1pFsf-I",
        "_score": 0.0865164,
        "_source": {
          "test_text": "Capitalization or capitalisation in English grammar is the use of a capital letter at the start of a word. English usage varies from capitalization in other languages.",
          "previous_score": 0.5046487
        }
      }
    ]
  },
  "profile": {
    "shards": []
  }
}
```