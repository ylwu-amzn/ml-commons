# Topic

This doc introduces new interface to make it easier to create model and agent.

# Design

## Model interface

```
POST /_plugins/_ml/models/_register
{
  "model_provider": "bedrock/converse", 
  "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
  "credential": { // mandarory
    "access_key": "{{access_key}}",
    "secret_key": "{{secret_key}}",
    "session_token": "{{session_token}}"
  },
  "parameters": {
    "region": "us-west-2" // optional, default value is us-east-1
  },
  "request_body": ".."
}
```
1. `model_provider`:  This decides URL and request body in connector configuration
2. `model_id`: This decides URL or model id in request body
3. `credential`: This will be used as connector credential
4. [optional] `parameters`: This will be used as connector parameters.
5. [optional] `name`: model name stored in model system index. If not provided, will use `model_id` as name.
6. [optional] `description`: model description stored in model system index. Default value is null.
7. [optional] `version`: model description stored in model system index. Default value is 1.
8. [optional] `request_body`: custom request body, will override the default request body decided by `model_provider` 


## Agent interface

### Register agent
```
POST _plugins/_ml/agents/_register
{
  "name": "RAG Agent",
  "type": "chat",
  "model": {
    "model_id": "us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    "model_provider": "bedrock/converse",
    "credential": {
      "access_key": "{{access_key}}",
      "secret_key": "{{secret_key}}",
      "session_token": "{{session_token}}"
    }
  },
  "tools": [
    {
      "type": "ListIndexTool"
    }
  ]
}
```

1. `name`: agent name
2. `type`: agent type
3. `model`: the model configuration, refer to "Model interface"
4. [optional] `tools`: a list of tools
 
### Execute agent request 

Agent execute API accepts these inputs
1. `system_prompt`: String, LLM system prompt
2. `question`: String, text question
3. `content_blocks`: List<ContentBlock>, For example 
```
[ 
    {
        "text": "hello"
    }, 
    {
        "image": {"format": "png", "source": "...."}
    } 
]
```
4. `messages`: List<Message>, the message list for example
```
[
    {
        "role": "user",
        "content": [ 
            {
                "text": "hello"
            }, 
            {
                "image": {"format": "png", "source": "...."}
            } 
        ]
    }
]
```

Agent will convert these input into LLM input format.

### Execute agent response

Different LLMs may have different formats. Agent will convert them into standard format

1. stop_reason: stop reason like `end_turn`, `tool_use`
2. message: one message consists of role and List<ContentBlock>
3. metrics: metrics data
4. state: additional information
