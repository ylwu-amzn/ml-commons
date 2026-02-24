# Topic

This tutorial works for OpenSearch 3.0 and above. To read more details, see the OpenSearch document [Agents and tools](https://opensearch.org/docs/latest/ml-commons-plugin/agents-tools/index/).

Skills are reusable instruction sets that guide agent behavior for specific tasks. When an agent is configured with multiple skills, the LLM can intelligently select the most appropriate skill based on the user's query. Skills enable domain expertise to be encapsulated and shared across multiple agents, promoting consistency and best practices.

This tutorial explains how to:
- Create specialized skills with detailed instructions
- Configure agents to use multiple skills
- Allow the LLM to automatically select the appropriate skill based on query context

Common use cases include:
- **Flight Data Analysis**: Following strict data retrieval protocols (count before fetch, field filtering)
- **Code Review**: Applying consistent security and performance checks
- **Data Science Workflows**: Standardizing data preprocessing and analysis steps

Note: Replace the placeholders that start with `your_` with your own values.

# Steps

## 1. Preparation: Load Sample Data

This tutorial uses the OpenSearch Dashboards sample flight data. If you haven't loaded it yet, follow these steps:

### 1.1 Load Sample Flight Data

In OpenSearch Dashboards:
1. Navigate to the home page
2. Click "Add sample data"
3. Click "Add data" for "Sample flight data"

Alternatively, you can verify the data exists:
```
GET opensearch_dashboards_sample_data_flights/_count
```

Sample output:
```
{
  "count": 13059,
  "_shards": {
    "total": 1,
    "successful": 1,
    "skipped": 0,
    "failed": 0
  }
}
```

### 1.2 Explore the Data Structure

Check the index mapping to understand available fields:
```
GET opensearch_dashboards_sample_data_flights/_mapping
```

Key fields include:
- `timestamp`: Flight date/time
- `DestCityName`: Destination city
- `Carrier`: Airline carrier
- `FlightDelay`: Boolean indicating if delayed
- `FlightDelayMin`: Delay duration in minutes
- `FlightDelayType`: Type of delay (Carrier, Security, Weather, etc.)
- `Cancelled`: Boolean indicating if cancelled
- `AvgTicketPrice`: Average ticket price

## 2. Create Skills

Skills define specialized behaviors for your agent. Each skill contains:
- **name**: Unique identifier for the skill
- **description**: Helps the LLM decide when to use this skill
- **instructions**: Detailed step-by-step guidance for the agent

### 2.1 Create Flight Expert Skill

This skill teaches the agent to analyze flight data using best practices like counting documents before fetching and using field filtering to minimize data transfer.

```
POST _plugins/_ml/skills/_create
{
  "name": "flight-expert",
  "description": "Expert which could query flight statistics from OpenSearch flight index and do analysis.",
  "instructions": "When analyzing flight data, follow these steps strictly:\n\n## Step 1: Discover Index\nList all indices in the OpenSearch cluster to identify the index containing flight data.\n\n## Step 2: Get Index Mapping\nRetrieve the mapping of the flight index to understand the available fields and their data types.\n\n## Step 3: Get Sample Data\nFetch 2 sample documents from the flight index to understand the data structure.\n\n## Step 4: Count Matching Documents (CRITICAL)\n**IMPORTANT**: Before fetching data for analysis, you MUST first count how many documents match your intended query.\n\nUse the search API with `size: 0` and `track_total_hits: true` to get the count:\n```json\nGET /<index_name>/_search\n{\n  \"query\": { <your_filter_query> },\n  \"size\": 0,\n  \"track_total_hits\": true\n}\n```\n\nFor example, if you want to filter by date range:\n```json\nGET /flight_index/_search\n{\n  \"query\": {\n    \"range\": {\n      \"timestamp\": {\n        \"gte\": \"2024-01-01\",\n        \"lte\": \"2024-01-31\"\n      }\n    }\n  },\n  \"size\": 0,\n  \"track_total_hits\": true\n}\n```\n\nThe response will contain `hits.total.value` which is the document count.\n\n## Step 5: Determine Query Size\nBased on the count result from Step 4:\n- Calculate the appropriate size: `min(2 * matched_document_count, 10000)`\n- If matched_document_count > 10000, STOP the analysis and inform the user: \"The query matches more than 10,000 documents. Please provide additional filters to narrow down the scope.\"\n\n## Step 6: Execute Analysis Query\nRe-run the same query from Step 4, but now with:\n- The calculated `size` parameter\n- **`_source` filtering to retrieve ONLY the fields necessary for your analysis**\n\n```json\nGET /<index_name>/_search\n{\n  \"_source\": {\n    \"includes\": [\"field1\", \"field2\", \"field3\"]\n  },\n  \"query\": { <same_filter_query_as_step_4> },\n  \"size\": <calculated_size_from_step_5>\n}\n```\n\nExample - analyzing flight delays by carrier:\n```json\nGET /flight_index/_search\n{\n  \"_source\": {\n    \"includes\": [\"carrier\", \"flight_number\", \"delay_minutes\", \"origin\", \"destination\"]\n  },\n  \"query\": {\n    \"range\": {\n      \"timestamp\": {\n        \"gte\": \"2024-01-01\",\n        \"lte\": \"2024-01-31\"\n      }\n    }\n  },\n  \"size\": 120\n}\n```\n\n## Important Rules:\n- NEVER assume default size is sufficient\n- ALWAYS count matching documents BEFORE fetching data, using the SAME query filters\n- ALWAYS use `size: 0` with `track_total_hits: true` for counting\n- ALWAYS set explicit `size` parameter based on the formula: min(2 * count, 10000)\n- **ALWAYS use `_source.includes` to retrieve ONLY the fields needed for your analysis**\n- **NEVER fetch all fields when only a subset is required**\n- **Exclude large text fields, binary data, or nested objects unless specifically needed**\n- If aggregations can answer the question, prefer aggregations over fetching raw documents (aggregations do not require fetching all docs)"
}
```

Sample output:
```
{
  "skill_id": "flight-expert",
  "status": "CREATED"
}
```

Verify the skill was created:
```
GET _plugins/_ml/skills/flight-expert
```

Sample output:
```
{
  "name": "flight-expert",
  "description": "Expert which could query flight statistics from OpenSearch flight index and do analysis.",
  "instructions": "When analyzing flight data, follow these steps strictly:\n\n## Step 1: Discover Index...",
  "owner": {
    "name": "admin",
    "backend_roles": ["admin"],
    "roles": ["own_index", "all_access"],
    "user_requested_tenant": null,
    "user_requested_tenant_access": "WRITE",
    "custom_attribute_names": []
  },
  "created_time": 1771375656061,
  "last_updated_time": 1771375656061
}
```

### 2.2 Create Code Review Expert Skill (Optional)

This skill demonstrates skill selection - the agent should NOT use this skill for flight queries.

```
POST _plugins/_ml/skills/_create
{
  "name": "code-review-expert",
  "description": "Expert knowledge for conducting thorough code reviews with focus on security, performance, and best practices",
  "instructions": "When reviewing code, systematically check:\n\n1. **Security Issues**\n   - SQL injection vulnerabilities\n   - Cross-site scripting (XSS)\n   - Authentication and authorization flaws\n\n2. **Performance**\n   - N+1 query problems\n   - Unnecessary loops\n   - Memory leaks\n\n3. **Code Quality**\n   - Naming conventions\n   - Error handling\n   - Logging practices\n\nProvide specific line numbers and actionable suggestions."
}
```

Sample output:
```
{
  "skill_id": "code-review-expert",
  "status": "CREATED"
}
```

## 3. Create LLM Model

We will use Bedrock Claude 3.7 Sonnet model in this tutorial. Note that skills require a model that supports tool calling and complex reasoning.

```
POST _plugins/_ml/models/_register
{
  "name": "Bedrock Claude 3.7 Sonnet model",
  "function_name": "remote",
  "description": "Claude 3.7 model for agent with skills",
  "connector": {
    "name": "Bedrock Claude 3.7 connector",
    "description": "Connector for Bedrock Converse API",
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

Sample output:
```
{
  "task_id": "abc123",
  "status": "CREATED",
  "model_id": "def456"
}
```

Deploy the model (not required, but recommended for production):
```
POST _plugins/_ml/models/your_model_id_from_above/_deploy
```

## 4. Create Memory Container

Agents require a memory container to store conversation history and trace data.

```
POST _plugins/_ml/memory_containers/_create
{
  "name": "agent-skills-demo-memory"
}
```

Sample output:
```
{
  "memory_container_id": "xyz789",
  "status": "created"
}
```

## 5. Create Agent with Skills

Create a conversational agent that has access to both skills. The agent will automatically select the appropriate skill based on the user's query.

```
POST _plugins/_ml/agents/_register
{
  "name": "Multi-Skill Agent",
  "type": "conversational",
  "description": "Agent with multiple specialized skills",
  "model": {
    "model_id": "your_model_id_from_step_3",
    "model_provider": "bedrock/converse"
  },
  "parameters": {
    "max_iteration": 30
  },
  "skills": [
    "flight-expert",
    "code-review-expert"
  ],
  "tools": [
    {
      "type": "ListIndexTool"
    },
    {
      "type": "SearchIndexTool",
      "parameters": {
        "return_raw_response": true
      }
    },
    {
      "type": "IndexMappingTool"
    }
  ],
  "memory": {
    "type": "agentic_memory",
    "memory_container_id": "your_memory_container_id_from_step_4"
  }
}
```

Sample output:
```
{
  "agent_id": "agent123"
}
```

Key configuration notes:
- **skills**: Array of skill names to make available to the agent
- **tools**: The agent needs data access tools (ListIndex, SearchIndex, IndexMapping) to execute the skill instructions
- **max_iteration**: Set higher (20-30) for complex multi-step skills
- **memory type**: Use `agentic_memory` for full trace support

## 6. Test Agent with Flight Query

### 6.1 Execute Flight Analysis Query

Ask the agent to analyze flight data. The agent should automatically select the `flight-expert` skill.

```
POST _plugins/_ml/agents/your_agent_id/_execute
{
  "input": "Do you see any abnormal for flights to Seattle in last 4 weeks"
}
```

Sample output (abbreviated):
```
{
  "inference_results": [
    {
      "output": [
        {
          "name": "memory_id",
          "result": "mem456"
        },
        {
          "name": "parent_interaction_id",
          "result": "interaction789"
        },
        {
          "name": "response",
          "dataAsMap": {
            "response": "Based on my analysis of flights to Seattle in the last 4 weeks, here are the key observations and abnormalities:\n\n### Flight Statistics Overview\n- Total flights to Seattle: 29 flights\n- Date range: January 21, 2026 to February 18, 2026\n\n### Abnormalities Identified:\n\n1. **High Disruption Rate (45%)**:\n   - 9 flights delayed (31%)\n   - 4 flights cancelled (13.8%)\n   - Industry average is typically 15-20% delays and 2-3% cancellations\n\n2. **Severe Delays**:\n   - 360 minutes (Ottawa - Security Delay)\n   - 330 minutes (Palermo - Carrier Delay)\n   - 315 minutes (Oslo - Carrier Delay)\n   - Average delay: 225 minutes\n\n3. **Delay Patterns by Type**:\n   - Carrier Delays: 3 incidents\n   - Security Delays: 3 incidents\n   - Late Aircraft Delays: 2 incidents\n   - Weather Delays: 1 incident\n\n4. **Weather Impact**:\n   - Multiple flights affected by adverse weather conditions\n   - Thunder & Lightning: Several flights\n   - Heavy Fog: 2 destinations\n\n5. **Price Anomalies**:\n   - Wide range: $172.49 (Milwaukee) to $1,125.08 (Osaka)\n   - Several flights with unusually high prices (>$900)\n\nThese patterns suggest operational issues affecting flights to Seattle, potentially related to weather conditions, security protocols, or carrier-specific problems.",
            "final_answer": true
          }
        }
      ]
    }
  ]
}
```

### 6.2 Verify Skill Selection

Check that the agent selected the correct skill by retrieving trace data:

```
GET _plugins/_ml/memory_containers/your_memory_container_id/memories/working/_search
{
  "query": {
    "bool": {
      "must": [
        {
          "term": {
            "namespace.session_id": "your_session_id_from_step_6.1_response"
          }
        },
        {
          "term": {
            "metadata.type": "trace"
          }
        }
      ]
    }
  },
  "sort": [
    {
      "message_id": {
        "order": "asc"
      }
    }
  ],
  "size": 50
}
```

In the trace data, you should see:
1. **Trace 1**: LLM recognizes flight query and loads `flight-expert` skill
2. **Trace 2**: Skill instructions injected with "SKILL LOADED: flight-expert"
3. **Trace 3-13**: Agent follows the 6-step process defined in the skill

Example trace showing skill selection (Trace 1):
```json
{
  "trace_number": 1,
  "origin": "LLM",
  "response": "{\"output\":{\"message\":{\"content\":[{\"text\":\"I'll help you analyze flights to Seattle. This seems to be related to flight statistics, so I'll use the flight-expert skill.\"},{\"toolUse\":{\"input\":{\"skill_id\":\"flight-expert\"},\"name\":\"Skill_Tool\"}}]}}}"
}
```

Example trace showing skill loaded (Trace 2):
```json
{
  "trace_number": 2,
  "origin": "Skill_Tool",
  "response": "SKILL LOADED: flight-expert\nInstructions:\nWhen analyzing flight data, follow these steps strictly:\n\n## Step 1: Discover Index..."
}
```

### 6.3 Verify Skill Instructions Were Followed

The agent should execute these steps in order:

| Step | Action | Tool Used | Expected Result |
|------|--------|-----------|-----------------|
| 1 | Discover Index | ListIndexTool | Found `opensearch_dashboards_sample_data_flights` |
| 2 | Get Mapping | IndexMappingTool | Retrieved field definitions |
| 3 | Sample Data | SearchIndexTool (size=2) | Fetched 2 sample documents |
| 4 | Count Documents | SearchIndexTool (size=0, track_total_hits=true) | Found 29 matching flights |
| 5 | Calculate Size | N/A | Calculated: min(2*29, 10000) = 58 |
| 6 | Fetch with Filtering | SearchIndexTool (size=58, _source filtering) | Retrieved 29 flights with only necessary fields |

You can verify Step 4 (critical counting step) in the trace:
```json
{
  "trace_number": 10,
  "origin": "SearchIndexTool",
  "input": "{\"query\":{\"bool\":{\"must\":[{\"match\":{\"DestCityName\":\"Seattle\"}},{\"range\":{\"timestamp\":{\"gte\":\"2026-01-21\",\"lte\":\"2026-02-18\"}}}]}},\"size\":0,\"track_total_hits\":true}",
  "response": "{\"hits\":{\"total\":{\"value\":29,\"relation\":\"eq\"}}}"
}
```

And Step 6 (fetching with field filtering) in the trace:
```json
{
  "trace_number": 12,
  "origin": "SearchIndexTool",
  "input": "{\"_source\":{\"includes\":[\"timestamp\",\"Carrier\",\"FlightDelay\",\"FlightDelayMin\",\"FlightDelayType\",\"Cancelled\",\"OriginCityName\",\"AvgTicketPrice\",\"FlightTimeMin\",\"OriginWeather\",\"DestWeather\"]},\"query\":{...},\"size\":58}"
}
```

Notice the agent only requested 11 fields instead of all 27 available fields, following the skill's efficiency guidelines.

## 7. Test Skill Selection with Different Query Types

### 7.1 Code Review Query (Tests Skill Selection)

To verify the agent selects the correct skill based on query context, try a code review query:

```
POST _plugins/_ml/agents/your_agent_id/_execute
{
  "input": "Please review this Python function for security issues: def query_db(user_input): return db.execute('SELECT * FROM users WHERE name = ' + user_input)"
}
```

The agent should select the `code-review-expert` skill instead of `flight-expert`, demonstrating intelligent skill selection.

### 7.2 Different Flight Query

Test with another flight-related query to ensure consistent skill selection:

```
POST _plugins/_ml/agents/your_agent_id/_execute
{
  "input": "Which carrier has the most delays in the flight data?"
}
```

The agent should again select `flight-expert` and follow the same 6-step process.

## 8. Advanced: Update a Skill

Skills can be updated to refine instructions without recreating the agent.

```
PUT _plugins/_ml/skills/flight-expert
{
  "name": "flight-expert",
  "description": "Expert which could query flight statistics from OpenSearch flight index and do analysis.",
  "instructions": "... updated instructions ..."
}
```

After updating a skill, the agent will automatically use the new instructions in future executions.

## 9. Best Practices

### 9.1 Writing Effective Skill Instructions

1. **Be Specific**: Use numbered steps and clear action verbs
2. **Include Examples**: Show JSON query examples inline
3. **Set Constraints**: Define limits (e.g., max documents, required parameters)
4. **Explain Why**: Help the LLM understand the reasoning behind each step
5. **Handle Edge Cases**: Include instructions for error scenarios

### 9.2 Skill Descriptions

The `description` field is critical for skill selection:
- Keep it concise (1-2 sentences)
- Use domain-specific keywords (e.g., "flight statistics", "code review")
- Describe **what** the skill does, not **how** it does it

Good example:
```
"Expert which could query flight statistics from OpenSearch flight index and do analysis."
```

Bad example:
```
"A skill that lists indices, gets mappings, counts documents, and fetches data."
```

### 9.3 Agent Configuration

- Set `max_iteration` to 20-30 for complex multi-step skills
- Use `agentic_memory` for full trace support and debugging
- Include all necessary tools (ListIndexTool, SearchIndexTool, IndexMappingTool)
- Use `return_raw_response: true` for SearchIndexTool to get full OpenSearch responses

### 9.4 Testing and Debugging

1. **Use Trace Data**: Always retrieve trace data to verify skill selection and step execution
2. **Test Skill Selection**: Create queries that should trigger different skills
3. **Verify Efficiency**: Check that field filtering and size calculations are applied correctly
4. **Monitor Token Usage**: Review LLM token consumption in trace data

## 10. Common Issues and Solutions

### Issue: Agent doesn't select the right skill

**Solution**: Improve skill descriptions to be more distinctive:
```json
{
  "name": "flight-expert",
  "description": "ONLY use for queries about flights, airlines, airports, and aviation data in OpenSearch flight indices."
}
```

### Issue: Agent skips steps in skill instructions

**Solution**: Make steps more explicit and add validation:
```
## Step 4: Count Documents (CRITICAL - DO NOT SKIP)
Before proceeding to Step 5, you MUST count documents.
If you skip this step, STOP and go back to Step 4.
```

### Issue: Skill instructions too long, hitting token limits

**Solution**:
- Split into multiple smaller skills
- Remove verbose examples and keep only essential guidance
- Use reference-style instructions: "Follow the same pattern as Step 2"

### Issue: Agent uses wrong tool parameters

**Solution**: Include exact parameter formats in instructions:
```json
Example query format:
{
  "_source": {"includes": ["field1", "field2"]},
  "query": {...},
  "size": 100,
  "track_total_hits": true
}
```

## Summary

This tutorial demonstrated:
- ✅ Creating specialized skills with detailed multi-step instructions
- ✅ Configuring an agent with multiple skills
- ✅ Automatic skill selection based on query context
- ✅ Verifying skill execution through trace data
- ✅ Following complex data retrieval protocols (count, calculate, filter, fetch)

Skills provide a powerful way to encode domain expertise and best practices into reusable components that can be shared across multiple agents. The LLM's ability to select the appropriate skill and follow detailed instructions enables sophisticated agentic workflows.

For more examples, see:
- [Agentic RAG Tutorial](./Agentic_RAG.md) - Using agents for retrieval-augmented generation
- [Agent Framework Documentation](https://opensearch.org/docs/latest/ml-commons-plugin/agents-tools/index/)
