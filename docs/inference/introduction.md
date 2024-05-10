# Introduction and General Flow

Similar to the OpenAI [tool use (a.k.a function calling)](<(https://platform.openai.com/docs/guides/function-calling)>),
Empower offers models that are fine-tuned for working with functions, capable of determining when
and how a function should be called. If your request includes one or more functions, the model
assesses whether any of them should be activated based on the prompt's context. When the model
decides that a function should be called, it responds with a JSON object containing the arguments
for that function.

> To ensure optimal outcomes from the model, we acknowledge the **limitations** of our current version.
> Please note that we are actively working on iterating and improving the model, [contact us](../contact.md) if any of them blocks your use case:
>
> - Nested function calling: This refers to scenarios where the result of one function depends on another,
>   for example, getWeather(getCurrentLocation()). The model does not currently specialize in handling nested function calls.
> - Number of functions: We optimized the model for up to 10 functions in the `"tool_call"`, expect some performance
>   degrade if there are more than 10 functions.
> - Deeply Nested Argument Schema: We have optimized the model for up to one layer of nesting in the arguments.
>   This means if your arguments include an object or dictionary, we recommend that its value be of a scalar type (number, string, boolean, etc.).
>   Expect some performance degradation if the arguments contain objects within objects.
> - Set a very low temperature as possible for the best performance.

### General flow

A general process for invoking the tool's capabilities involves the following steps:

1. Invoke the model with the user's query and a collection of functions in the `tools` parameter and optionally set the `tool_choice` parameter to set the mode.
2. The model may decide to execute one or more functions; in such instances, the output will be a stringified JSON object that follows your specified schema (note: the model might introduce fictional parameters).
3. Convert the string back into JSON in your code, and if provided, execute your function with the given arguments.
4. Invoke the model again, including the response from the function as a new message, allowing the model to summarize and present the results to the user.

### `tools` parameter

This parameter specifies a list of tools the model can utilize. Currently, only functions are recognized as tools.
Use it to define a list of functions for which the model may generate JSON inputs.
A maximum of 128 functions are supported. See the [api reference](https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools) for detailed schema.

### `tool_choice` parameter

This parameter governs the model's function-calling behavior, with the following three possible values:

> Currently local model only supports "auto"

- `"auto"`: This means the model can choose between generating a message or calling a function.
- `"none"`: The model will not call a function and will instead generate a message.
- `"any"`: The model is compelled to trigger functions, even when it may not be relevant. In such cases, the most relevant function available will be activated.

## Code Example

Below is a code example of the full flow described above with with a single function triggered:

```python python
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://app.empower.dev/api/v1", # Replace with localhost if running in Llama.cpp server
    api_key="YOUR_API_KEY"
)

def get_current_weather(location: str, unit: str = "fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return {"location": "Tokyo", "temperature": "10", "unit": unit}
    elif "san francisco" in location.lower():
        return {"location": "San Francisco", "temperature": "72", "unit": unit}
    elif "paris" in location.lower():
        return {"location": "Paris", "temperature": "22", "unit": unit}
    else:
        return {"location": location, "temperature": "22"}

# #1 Ask the model for the function to call
response = client.chat.completions.create(
    model="empower-functions",
    messages=[{"role": "user",
               "content": "What's the weather in San Francisco and Los Angles in Celsius?"}],
    temperature=0,
    tools=[{
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
    }],
    tool_choice="auto"  # optional, auto is default
)

# #2 Model responds with the function to call the the arguments to use
response_message = response.choices[0].message
tool = response_message.tool_calls[0]
function = tool.function
function_name = function.name
function_arguments = json.loads(function.arguments)
print(f"function name: {function_name}, arguments: {function_arguments}")

# #3 Execute the function based on the model response
function_response = globals()[function_name](
    location=function_arguments['location'],
    unit=function_arguments['unit']
)

# #4 Invoke the model again including the first model response and the response function execution
response = client.chat.completions.create(
    model="empower-functions",
    messages=[
        {"role": "user",
         "content": "What's the weather in San Francisco and Los Angles in Celsius?"},
        # Append the response message from the first call
        response_message,
        # Append the function response, please make sure to have the tool_call_id
        {
            "role": "tool",
            'tool_call_id':  tool.id,
            "content": json.dumps(function_response)
        }
    ],
    temperature=0
)
print(response.choices[0].message.content)
```

Output

```
function name: get_current_weather, arguments: {'location': 'San Francisco, CA', 'unit': 'celsius'}
The current temperature in San Francisco is 72 degrees Celsius.
```

### Advanced Use Cases

Besides the basic single turn use case as the example above. Empower tool use models also supports more advanced use case such
as [tools streaming](/inference/tool-use/streaming), [multi-turn tool use](/inference/tool-use/multi-turn) and [parallel tool use](/inference/tool-use/parallel-calling). See other docs under "Tool Using" section for more details.
