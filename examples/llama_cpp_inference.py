import json
from empower_functions import EmpowerFunctionsCompletionHandler
from llama_cpp.llama_tokenizer import LlamaHFTokenizer
from llama_cpp import Llama


llm = Llama.from_pretrained(
    repo_id="empower-dev/llama3-empower-functions-small-gguf",
    filename="ggml-model-Q4_K_M.gguf",
    chat_format="llama-3",
    chat_handler=EmpowerFunctionsCompletionHandler(),
    tokenizer=LlamaHFTokenizer.from_pretrained("empower-dev/llama3-empower-functions-small-gguf"),
    n_gpu_layers=0
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

result = llm.create_chat_completion(
      messages = messages,
      tools=tools,
      tool_choice="auto",
      max_tokens=128
)
print(json.dumps(result["choices"][0], indent=2))
