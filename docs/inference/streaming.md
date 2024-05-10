# Streaming

> Important: Currently streaming is only supported in the Empower API

Streaming is fully supported for tool using for both tool only (`"tool_choice" = "any"`)
mode or mixed mode (`"auto" = "any"`). The format is fully compatible with OpenAI streaming.

## How to Use

All you need to do is to set `stream=true` when
calling the chat completions api. This returns an object that streams
the response as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).
To extract chunks,
use the delta field instead of the message field.

Specifically, the streaming mode of tool use will guarantee to have the function name return as a
single chunk then streaming the arguments. (See the code example)

## Code Example

In this example, we demonstrate how to use the streaming mode for calling tools, which will trigger two functions, and how to process their responses.

```python python
from openai import OpenAI

client = OpenAI(
    base_url="https://app.empower.dev/api/v1",
    api_key="YOUR_API_KEY"
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="empower-functions",
    messages=[{"role": "user",
               "content": "What's the weather in San Francisco and Los Angles in Celsius?"}],
    temperature=0,
    tools=tools,
    tool_choice="auto",
    stream=True
)

functions = []
for chunk in response:
    if (chunk.choices[0].delta.tool_calls):
        tool_call = chunk.choices[0].delta.tool_calls[0]
        # Can use the index to differentiate different tool calls if there're more than one
        # index = tool_call.index
        function = tool_call.function
        if function.name:
            # Function name return as one single trunk
            # Optionally you can call the function here
            functions.append({
                "index": index,
                "name": function.name,
                "arguments": ""
            })
        else:
            functions[tool_call.index]['arguments'] += function.arguments

        print(functions)

```

Output

```
[{'name': 'get_current_weather', 'arguments': ''}]
[{'name': 'get_current_weather', 'arguments': '{'}]
[{'name': 'get_current_weather', 'arguments': '{\n'}]
[{'name': 'get_current_weather', 'arguments': '{\n '}]
[{'name': 'get_current_weather', 'arguments': '{\n  "'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location":'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San'}]
....
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San Francisco, CA",\n  "unit": "celsius"\n  }'}, {'name': 'get_current_weather', 'arguments': ''}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San Francisco, CA",\n  "unit": "celsius"\n  }'}, {'name': 'get_current_weather', 'arguments': '{'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San Francisco, CA",\n  "unit": "celsius"\n  }'}, {'name': 'get_current_weather', 'arguments': '{\n'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San Francisco, CA",\n  "unit": "celsius"\n  }'}, {'name': 'get_current_weather', 'arguments': '{\n '}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San Francisco, CA",\n  "unit": "celsius"\n  }'}, {'name': 'get_current_weather', 'arguments': '{\n  "'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San Francisco, CA",\n  "unit": "celsius"\n  }'}, {'name': 'get_current_weather', 'arguments': '{\n  "location'}]
[{'name': 'get_current_weather', 'arguments': '{\n  "location": "San Francisco, CA",\n  "unit": "celsius"\n  }'}, {'name': 'get_current_weather', 'arguments': '{\n  "location": "Los Angeles, CA",\n  "unit": "celsius"\n  }'}]
```
