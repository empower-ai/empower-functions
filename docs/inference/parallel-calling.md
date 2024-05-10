# Parallel Calling

In certain scenarios, the model may need to invoke multiple tools. This could involve triggering
a single function multiple times with different parameters, activating multiple functions, or
a combination of both. Such situations typically arise when fulfilling a user request requires
triggering multiple tools simultaneously. We refer to this process as "parallel calling," and
we have optimized the model to support this functionality effectively.

## How to Use

Similar as [this example](introduction.md#code-example), but in the parallel calling scenario,
the model will include multiple `"tool_calls"` in its response.
These tool_calls should all be executed, and their responses should be sent back as multiple `"tool"`
messages.

## Code Example

```python python
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://app.empower.dev/api/v1", # Replace with localhost if running in Llama.cpp server
    api_key="YOUR_API_KEY"
)

# Function definitions
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": unit})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "22"})

def get_capital(country):
    """Get the capital city of a given country"""
    capitals = {
        "japan": "Tokyo",
        "united states": "Washington D.C.",
        "france": "Paris",
        "united kingdom": "London",
        "germany": "Berlin",
        "india": "New Delhi"
    }

    country_lower = country.lower()
    if country_lower in capitals:
        capital = capitals[country_lower]
        return json.dumps({"country": country, "capital": capital})
    else:
        return json.dumps({"country": country, "capital": "Unknown"})

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
    },
    {
        "type": "function",
        "function": {
                "name": "get_capital",
                "description": "Get the capital city of a given country",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "Name of the country",
                        }
                    },
                    "required": ["country"],
                },
        },
    }
]
# End function definitions

# #1 Ask the model for the function to call
messages = [
    {"role": "user", "content": "What's the current weather in San Francisco, Paris and Beijing? Also can you help me check the capital of Germany and India?"},
]
# #2 Model responds with the function to call the the arguments to use
# In this case it should respond with 5 tool_calls
response = client.chat.completions.create(
    model="empower-functions",
    messages=messages,
    temperature=0.0,
    tools=tools,
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls
messages.append(response_message)

# #3 Execute all the functions based on the model response
for tool_call in tool_calls:
    function = tool_call.function
    function_name = function.name
    function_arguments = json.loads(function.arguments)
    function_response = globals()[function_name](**function_arguments)

    print(f"function name: {function_name}, arguments: {function_arguments}")

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(function_response)
    })

# #4 Invoke the model again including the first model response and the response function executions
response = client.chat.completions.create(
    model="empower-functions",
    messages=messages,
    temperature=0.0,
    tools=tools,
)

print(response.choices[0].message.content)
```

Output:

```
function name: get_current_weather, arguments: {'location': 'San Francisco, CA', 'unit': 'fahrenheit'}
function name: get_current_weather, arguments: {'location': 'Paris, France', 'unit': 'celsius'}
function name: get_current_weather, arguments: {'location': 'Beijing, China', 'unit': 'celsius'}
function name: get_capital, arguments: {'country': 'Germany'}
function name: get_capital, arguments: {'country': 'India'}
The current weather in San Francisco is 72 degrees Fahrenheit. In Paris, it's 22 degrees Celsius. And in Beijing, it's also 22 degrees Celsius. The capital of Germany is Berlin and the capital of India is New Delhi.
```
