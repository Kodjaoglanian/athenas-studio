# Function Calling / Tool Use

Athenas Studio supports OpenAI-compatible function calling (tool use), allowing models to invoke external functions, APIs, and tools during generation.

## Overview

Function calling enables the model to:
1. Recognize when a user's request requires calling an external function
2. Generate structured JSON arguments for the function call
3. Return a `tool_calls` array in the response
4. Accept `tool` role messages with function results to continue the conversation

## Request Format

```json
{
  "model": "llama-3.1-8b-instruct",
  "messages": [
    {"role": "user", "content": "What's the weather in San Francisco?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name, e.g. 'San Francisco, CA'"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}
```

### `tool_choice` Options

| Value | Description |
|-------|-------------|
| `"auto"` | Model decides whether to call a tool (default) |
| `"none"` | Model will not call any tools |
| `"required"` | Model must call at least one tool |
| `{"type": "function", "function": {"name": "get_weather"}}` | Force a specific function |

## Response with Tool Calls

```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "model": "llama-3.1-8b-instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"San Francisco, CA\", \"unit\": \"fahrenheit\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }],
  "usage": {
    "prompt_tokens": 85,
    "completion_tokens": 25,
    "total_tokens": 110
  }
}
```

## Continuing the Conversation

After executing the function, send the result back with the `tool` role:

```json
{
  "model": "llama-3.1-8b-instruct",
  "messages": [
    {"role": "user", "content": "What's the weather in San Francisco?"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"San Francisco, CA\"}"}}]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temperature\": 62, \"unit\": \"fahrenheit\", \"condition\": \"foggy\"}"}
  ],
  "tools": [/* same tools as before */]
}
```

The model will then generate a natural language response incorporating the tool result.

## Streaming with Tool Calls

When streaming with `stream: true`, tool calls are delivered as deltas:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ation\": \"San"}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":" Francisco, CA\"}"}}]}}]}

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}
```

## Full Example (Python)

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

# Step 1: Initial request
response = client.chat.completions.create(
    model="llama-3.1-8b-instruct",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

# Step 2: Check if model wants to call a tool
if message.tool_calls:
    # Execute the function (your implementation)
    tool_call = message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    
    # Call your actual function
    result = get_weather(args["location"])  # your function
    
    # Step 3: Send result back
    response = client.chat.completions.create(
        model="llama-3.1-8b-instruct",
        messages=[
            {"role": "user", "content": "What's the weather in Tokyo?"},
            message,
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            }
        ],
        tools=tools
    )
    
    print(response.choices[0].message.content)
else:
    print(message.content)
```

## Supported Models

Function calling works best with models fine-tuned for tool use:

- **Llama 3.1/3.2/3.3** — Native function calling support
- **Qwen 2.5 / Qwen 3** — Excellent tool use capabilities
- **Mistral / Mixtral** — Function calling with specific templates
- **DeepSeek V3 / R1** — Tool use with reasoning
- **Phi-3 / Phi-4** — Function calling support

The llama.cpp backend uses the model's built-in chat template (with `--jinja` flag) to format tool calls. Models without native function calling support may produce unreliable results.

## Implementation Details

- Tool definitions are passed through to the llama.cpp server's `/v1/chat/completions` endpoint
- The `--jinja` flag is enabled on llama-server startup, ensuring proper chat template rendering
- Tool call arguments are returned as JSON strings (matching OpenAI format)
- The `finish_reason` is set to `"tool_calls"` when the model generates a tool call
- Streaming tool calls are assembled incrementally via delta merging
