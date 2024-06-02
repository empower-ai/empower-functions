# Built in Chain-of-Thought

Chain-of-Thought (CoT) is a prompting technique that enhances complex reasoning in AI models by breaking down the reasoning process into intermediate steps. This method allows models to handle tasks that require multi-step thinking by explicitly generating and following a thought process before arriving at a final response. By doing so, CoT improves the accuracy and transparency of the model's outputs, because of the nature of casual inference of LLMs.
In function-calling use cases, CoT is typically utilized to analyze the intent of the user input to determine whether it’s appropriate to trigger functions or continue the conversation as usual. If it’s suitable to trigger functions, the model identifies the most appropriate function(s) to invoke. It checks if any required parameters are missing and cannot be inferred from the conversation context. Based on this analysis, the model triggers the functions or asks the user for follow-up information.

Below is a quick example of prompt used for the model to do CoT for function calling and a sample model response on the thought process:

**Prompt:**

```
To respond to the user's request, use relevant tools if available. Follow these steps:
Analyze the request to identify the appropriate tool to use.
Review the required parameters for the selected tool.
Determine if the user has provided all necessary parameters or if they can be inferred from the context. Carefully consider all provided information to support any inferred values.
If all required parameters are present or can be reasonably inferred, proceed to call the tool.
If any required parameter is missing, do not call the tool. Instead, ask the user for the missing information.
Do not request additional details for optional parameters if they are not provided.
```

**Thinking response:**

```
The user asked for the weather in San Francisco. The relevant tool to use is "get_current_weather," which requires the "location" parameter. The user provided the location directly as "San Francisco," so all required parameters are present, leading to the tool call with the argument "location" set to "San Francisco."
```

## Model Level Chain-of-Thought Support

While it’s typical to implement CoT at the prompt level, this approach has two main drawbacks:

- Performance: Additional instructions and tokens are needed to guide the CoT process, introducing overhead in terms of both cost and latency.
- Reliability: Ensuring the model follows the correct format is challenging, especially for function calling, which involves a mix of JSON (function calls) and free text (thinking). This complexity makes streaming extremely difficult. There are tricks to mitigate this, such as [adding an additional "explanation" parameter to the function definition](https://pierce-lamb.medium.com/improving-gpt-4-function-calling-with-an-explanation-parameter-4fba06a4c6bb), but this has limitations. When the explanation is generated, the model has already decided to trigger functions and which exact function(s) to trigger, so the improvement in accuracy is limited.

To address these drawbacks, we decided to enable CoT at the model level. Empower functions models have been trained with built-in CoT that can be enabled with a special prompt (less than 10 tokens in the internal system prompt). When CoT is enabled, Empower functions models will respond with their thought process within tags before the actual response (which will be a set of function calls or regular conversations). This approach provides the model with a full “thought process” before deciding whether to trigger any functions and which function(s) to trigger. We have fully supported streaming with CoT. Additionally, the model can function without CoT if the special prompt is not added.

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
    base_url="https://app.empower.dev/api/v1", # Replace with localhost if running in Llama.cpp server
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
    model="empower-functions",
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
    model="empower-functions",
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

Output:

```
Response:                                                                                                                                                                                                                                                                       [279/279]
Thinking: <thinking>The user asked for the weather in both San Francisco and Tokyo. The relevant tool is "get_current_weather," which requires the "location" parameter. Both required locations are directly provided by the user, so the tool call is made with the arguments "San Fran
cisco, CA" and "Tokyo."</thinking>
Tool calls:
name: get_current_weather arguments: {"location": "San Francisco, CA"}
name: get_current_weather arguments: {"location": "Tokyo"}

Response:
Thinking: <thinking>The user's request to order a pizza cannot be fulfilled with the available tool, which is designed to provide current weather information. The response explains the limitation of the assistant's capabilities and redirects the user to ask for something within it
s scope, such as providing the current weather. This ensures clarity about the assistant's functions and sets appropriate expectations for the user.</thinking>
Content: I'm sorry, but I don't have the capability to perform external tasks like ordering a pizza. My current function allows me to provide information about the current weather in a specified location. Is there anything else you would like to know within my capabilities?
```

#### Prompt the model directly:

The `empower-functions` package has the support to prompt the raw model to enable the CoT. See the example at [prompt.py](/examples/prompt.py) for details
