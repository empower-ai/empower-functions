# Model Prompt

The Empower functions model has only two roles: **"user"** and **"assistant."** This setup ensures compatibility with various chat templates from different foundation models, such as the Llama and Mistal families. We use a special format, described below, to encapsulate different types of information, including function call responses, regular conversation responses from the assistant, regular user messages, and function responses in the user role.

## Prompt Format

#### User Message

**First message**
FIrst user message will include the system prompt, json encoded functions and user message. In the follow format:

> To ensure the best performance, we recommend to json encode functions with indent=2 and new lines

```
In this environment you have access to a set of functions defined in the JSON format you can use to address user's requests, use them if needed.
Functions: [json encoded functions]

<u>[user_message]
```

Format of the json functions:
`[{name: string, description: string, parameters: object}]` # object should be a valid json schema, see [OpenAI format](https://platform.openai.com/docs/guides/function-calling) for more details.

Example:

```
In this environment you have access to a set of functions defined in the JSON format you can use to address user's requests, use them if needed.
Functions: [{
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
}]

<u>How's the weather in San Francisco?
```

**Rest user messages**
There're two types of formats for the rest of user messages:

- Function response: function response should be json encoded with a `<r>` tag in the beginning.
  Format of json: `[{value: string, tool_call_id: string}]`
  Example: `<r>[{"value": "{\"tempeature"\:\"70 fahrenheit\"}", "tool_call_id":"get_current_weather_dsfef123"}]`
- Plain text message from user: just add `<u>` in the beginning, for example:
  ```
  <u>Can you also help me find the weather in New York City?
  ```

#### Assistant Message

There're two types of assistant messages:

- Function calls response: function calls response include a set of json encoded function calls with a `<f>` prepend in the beginning.
  Format of the json: `[{name: string, arguments: string}]`
  Example:
  ```
  <f>[{
    "name": "get_current_weather",
    "arguments": "{\"location\":\"San Francisco\"}"
  }]
  ```
- Plain text conversation response: just add `<c>` in the beginning, for example:
  ```
  <c>The weather in San Francisco is 70 degrees Fahrenheit.
  ```

## Using empower-functions Library

To make it easier, to use, we have encapusilate the prompting logic into our `empower-functions` pip package. See the example at [prompt.py](/examples/prompt.py) for details.
