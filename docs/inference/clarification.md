# Clarification

When the model is about to trigger a function but cannot infer the value of any required function parameter from the context, or if it needs more information to determine which function to call, it will respond with messages to clarify with the user. This is important behavior, we explicitly train model in this way to minimize the chance of the model hallucinating the argument value or function names.

## Code Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://app.empower.dev/api/v1", # Replace with localhost if running in Llama.cpp server
    api_key="sk_8OntboZEhjrAUr3R7QZoJqMvAuHWtlOTyqIF3GhNFiI="
)

response = client.chat.completions.create(
    model="empower-functions",
    messages=[{"role": "user",
               "content": "Hey can you help me to check the weather in my city?"}],
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
)
print(response.choices[0].message.content)
```

Output

```
Of course! Could you please tell me the name of your city and whether you prefer the temperature in Celsius or Fahrenheit?
```
