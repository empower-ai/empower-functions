import openai
import json

client = openai.OpenAI(
    base_url = "http://localhost:8000/v1",
    api_key = "YOUR_API_KEY"
)

messages = [
    {"role": "user", "content": "What's the weather in San Francisco?"}
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
    tool_choice="auto"
)

print(chat_completion)
