# Built in Chain-of-Thought

[Chain-of-thought (CoT)](https://www.promptingguide.ai/techniques/cot) enables complex reasoning capabilities through intermediate reasoning steps.

Empower functions models have been trained with built-in CoT that can be enabled with a special prompt. When CoT is enabled, Empower functions models will respond with their thought process within <thinking></thinking> tags before the actual response. Due to the nature of the causal inference process, including the thought process before the actual response improves the accuracy of the response. Additionally, in some specific use cases, displaying the thought process to end-users enhances transparency.

Below is an example of the thought process output by the model with CoT enabled:

Function:

```json
{
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
```

Prompt:

```
What's the weather in San Francisco?
```

Thinking:

```
<thinking>The user asked for the weather in San Francisco. The relevant tool to use is "get_current_weather," which requires the "location" parameter. The user provided the location directly as "San Francisco," so all required parameters are present, leading to the tool call with the argument "location" set to "San Francisco."</thinking>
```

## How to Use

#### Using API:

The most straightforward way to use CoT mode is via the API. Both the Empower platform and the llama.cpp server support this feature and can toggle it at the request level with a parameter: `include_thinking`.

```python
import openai


def _separate_thinking_if_present(text):
    tag = "</thinking>"
    tag_position = text.find(tag)

    if tag_position != -1:
        # Split the string into two parts
        part1 = text[: tag_position + len(tag)]
        part2 = text[tag_position + len(tag):]
        return part2, part1
    else:
        return text, None


def print_response(chat_completion):
    print("Response:")
    (content, thinking) = _separate_thinking_if_present(
        chat_completion.choices[0].message.content)

    if thinking:
        print(f"Thinking: {thinking}")
    if content:
        print(f"Content: {content}")
    if chat_completion.choices[0].message.tool_calls:
        print("Tool calls:")
        for tool_call in chat_completion.choices[0].message.tool_calls:
            print(
                f"name: {tool_call.function.name}, arguments: {tool_call.function.arguments}")


client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="YOUR_API_KEY"
)

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
    messages=[
        {"role": "user", "content": "What's the weather in San Francisco and Tokyo?"}
    ],
    tools=tools,
    temperature=0,
    tool_choice="auto",
    extra_body={
        "include_thinking": True
    }
)

print_response(chat_completion)
print()


chat_completion = client.chat.completions.create(
    model="does_not_matter",
    messages=[
        {"role": "user", "content": "Can you help me order a pizza"}
    ],
    tools=tools,
    temperature=0,
    tool_choice="auto",
    extra_body={
        "include_thinking": True
    }
)

print_response(chat_completion)

```

#### Prompt the model directly:

The `empower-functions` package has the support to prompt the raw model to enable the CoT. See the example at [prompt.py](/examples/prompt.py) for details