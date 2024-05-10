# Sequential Calling

Sequential calling refers to a scenario where more than one function needs to be triggered to fulfill the user request, but they have dependencies among them. For instance, for a load inquiry agent, it may first need to ask the user for their specific requirements or criteria. Once the user's input is received, the agent can then call another function to check the availability of loads that match the criteria. After finding the available loads, a third function might be called to provide the user with detailed information about the matching loads. Each function in this sequence relies on the output of the previous function to proceed, ensuring a coherent and logical flow of operations to achieve the desired outcome.

## How to Use

This is similar as the multi-turn use case, just instead of responding plain text message, the model will keep responding the function calls based on the function response until it finishes the call chain.

## Code Example:

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://app.empower.dev/api/v1", # Replace with localhost if running in Llama.cpp server
    api_key="YOU_API_KEY"
)


tools = [
    {
        "type": "function",
        "function": {
                "name": "get_vehicle_diagnostics",
                "description": "Retrieves diagnostic data from the user's vehicle",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vin_number": {
                            "type": "string"
                        }
                    },
                    "required": ["vin_number"]
                }
        }
    },
    {
        "type": "function",
        "function": {
                "name": "analyze_diagnostics",
                "description": "Analyzes vehicle diagnostic data to suggest maintenance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "diagnostic_data": {
                            "type": "object",
                            "description": "The diagnostic data to analyze"
                        }
                    },
                    "required": [
                        "diagnostic_data"
                    ]
                }
        }
    }
]


def call_model(messages):
    user_message = {
        "role": "system",
        "content": """
            Hey, can you check what maintenance my car might need soon? The vin number is 1234567
        """
    }
    return client.chat.completions.create(
        model="empower-functions",
        messages=[user_message] + messages,
        tools=tools,
        temperature=0
    )


def call_function_and_get_response(function_name, _):

    if function_name == "get_vehicle_diagnostics":
        function_response = {"diagnostic_data": {
            "engine_status": "warning", "tire_pressure": "low"}}
    elif function_name == "analyze_diagnostics":
        function_response = {
            "maintenance_recommendations": "Consider checking the engine and inflating your tires."}
    return {
        "role": "tool",
        "tool_call_id": function_name,
        "content": json.dumps(function_response)
    }


messages = []
while True:
    response = call_model(messages)
    messages.append(response.choices[0].message)

    if response.choices[0].message.content:
        print(f"assistant: {response.choices[0].message.content}")

        break

    else:
        response_message = response.choices[0].message
        tool = response_message.tool_calls[0]
        function = tool.function
        function_name = function.name
        function_arguments = json.loads(function.arguments)
        print(
            f"assistant: call function {function_name}, with arguments {function_arguments}")
        tool_message = call_function_and_get_response(
            function_name, function_arguments)
        print(f"function response: {tool_message['content']}")

        messages.append(tool_message)

```

Output:

```
assistant: call function get_vehicle_diagnostics, with arguments {'vin_number': '1234567'}
function response: {"diagnostic_data": {"engine_status": "warning", "tire_pressure": "low"}}
assistant: call function analyze_diagnostics, with arguments {'diagnostic_data': {'engine_status': 'warning', 'tire_pressure': 'low'}}
function response: {"maintenance_recommendations": "Consider checking the engine and inflating your tires."}
assistant: Based on the diagnostic data, it seems like your car needs some maintenance. The engine is showing a warning and the tire pressure is low. I recommend you to check the engine and inflate your tires.
```
