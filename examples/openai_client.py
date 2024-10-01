import openai
import json

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_API_KEY"
)

messages = [
    {"role": "user", "content": "Hi, can you tell me the current weather in San Francisco and New York City in Fahrenheit?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "get_current_weather_san_francisco",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": json.dumps({
                        "location": "San Francisco, CA",
                        "unit": "fahrenheit"
                    })
                }
            },
            {
                "id": "get_current_weather_new_york",
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "arguments": json.dumps({
                        "location": "New York City, NY",
                        "unit": "fahrenheit"
                    })
                }
            }
        ]},
    {
        "role": "tool",
        "tool_call_id": "get_current_weather_san_francisco",
        "content": json.dumps({"temperature": 75})

    },
    {
        "role": "tool",
        "tool_call_id": "get_current_weather_new_york",
        "content":  json.dumps({"temperature": 82})
    }
]

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g., San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

chat_completion = client.chat.completions.create(
    model="does_not_matter",
    messages=messages,
    tools=tools,
    temperature=0,
    tool_choice="auto",
)

print(chat_completion)
