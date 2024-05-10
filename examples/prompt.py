from transformers import AutoModelForCausalLM, AutoTokenizer
from empower_functions.prompt import prompt_messages
import json

device = "cuda"

model_path = "empower-dev/empower-functions-small"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

functions = [
    {
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
    }
]

messages = [
    {"role": "user", "content": "Hi, can you tell me the current weather in San Francisco and New York City in Fahrenheit?"},
    {"role": "assistant", "tool_calls": [
        {
            "id": "get_current_weather_san_francisco",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": {
                    "location": "San Francisco, CA",
                    "unit": "fahrenheit"
                }
            }
        },
        {
            "id": "get_current_weather_new_york",
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "arguments": {
                    "location": "New York City, NY",
                    "unit": "fahrenheit"
                }
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

messages = prompt_messages(messages, functions)
model_inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt").to(model.device)

generated_ids = model.generate(model_inputs, max_new_tokens=128)
decoded = tokenizer.batch_decode(generated_ids)

print(decoded[0])
