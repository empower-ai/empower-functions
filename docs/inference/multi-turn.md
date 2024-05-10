# Multi-Turn Tool Calling

In real-world applications, the need for precise function invocation becomes especially crucial within the context
of multi-turn conversations. These scenarios demand not only the accurate identification and execution of functions
based on user inputs but also the preservation of context across multiple interactions.

Typical use cases, for example, include scenarios such as engaging with a support agent or implementing workflow automation.
These applications benefit significantly from a system's ability to understand and respond to a sequence of user inputs coherently,
ensuring a smooth and effective user experience.

## How to Use

Multi-turn tool calling is similar to single-turn tool calling, as illustrated in [this example](introduction.md#code-example), but it can involve multiple rounds.
It utilizes the results of previous conversations—including the assistant's messages, user messages, or tool calls and their
responses—to infer the next response.

## Code Example

In this example, we create a simplified customer support bot to demonstrate a basic multi-turn tool calling scenario.
Specifically, we instruct the bot to first verify the user's identity and then check the ticket status for the user.
The model is required to automatically decide whether to call functions or send messages.
Additionally, it must understand the flow and current status from the conversation context.

```python
from openai import OpenAI
import json

client = OpenAI(
    base_url="https://app.empower.dev/api/v1", # Replace with localhost if running in Llama.cpp server
    api_key="YOUR_API_KEY"
)

### Function definitions

def get_ticket_status(ticket_id: str):
    """Returns mock ticket status based on the ticket ID."""
    status_data = {
        "12345": {"status": "Open", "issue": "Billing Query"},
        "67890": {"status": "Closed", "issue": "Technical Support"},
    }
    return status_data.get(ticket_id, {"status": "Unknown", "issue": "Unknown"})

def verify_user(username: str, api_key: str):
    """Verify user based on username and API key."""
    return {
        "success": True,
    }

# Tools definitions based on the function definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_ticket_status",
            "description": "Returns mock ticket status based on the ticket ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "The unique identifier for the ticket."
                    }
                },
                "required": ["ticket_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_user",
            "description": "Verifies the user based on the provided username and API key.",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {
                        "type": "string",
                        "description": "The username of the user to be verified."
                    },
                    "api_key": {
                        "type": "string",
                        "description": "The API key provided by the user for verification."
                    }
                },
                "required": ["username", "api_key"]
            }
        }
    }
]

### End function definitions

### Utils
def call_model(messages):
    """ Call the model used the provided messages """
    system_message = {
        "role": "system",
        "content": """
            You are a customer support agent you job is to help users check ticket status.
            Make sure you verify the user before checking the ticket status.
            Start by greeting the user.
        """
    }
    return client.chat.completions.create(
        model="empower-functions",
        messages=[system_message] + messages,
        tools=tools,
        temperature=0
    )


def call_function_and_get_response(function_name, function_arguments):
    """ Run the "function_name" using "function_argument" and convert the response into a "tool" message
    function_response = globals()[function_name](**function_arguments)

    return {
        "role": "tool",
        "tool_call_id": tool.id,
        "content": json.dumps(function_response)
    }

### End Utils

user_messages = [
    {
        'role': 'user',
        'content': 'Hi there, my username is john_doe, can you help me check the status for my ticket?'
    },

    {
        'role': 'user',
        'content': 'Sure, my username is "john_doe" and my API key is: "key"'
    },
    {
        'role': 'user',
        'content': 'Yea it\'s: 12345'
    },
    {
        'role': 'user',
        'content': 'That\'s it, thanks.'
    }
]

messages = []
while True:
    # First call the model to get the response
    response = call_model(messages)
    messages.append(response.choices[0].message)

    if response.choices[0].message.content:
        # If the response is a message, append the assistant message to the "messages", then fetch
        # a user message from the "user_messages" and append to the "messages"
        print(f"assistant: {response.choices[0].message.content}")

        # Break if there's no user message left
        if not user_messages:
            break
        user_message = user_messages.pop(0)
        print(f"user: {user_message['content']}")
        messages.append(user_message)

    else:
        # Otherwise it's a function call response. We fetch the function name and argument from the
        # response, append the response to the "messages" then call the function accordingly and
        # convert its response into a "tool" message and append into the "messages".
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
assistant: Hello! Welcome to our customer support. I'm here to assist you with your ticket status. Could you please provide me with your username and API key for verification?
user: Hi there, my username is john_doe, can you help me check the status for my ticket?
assistant: Sure, I can help with that. But first, could you please provide me with your API key for verification?
user: Sure, my username is "john_doe" and my API key is: "key"
assistant: call function verify_user, with arguments {'username': 'john_doe', 'api_key': 'key'}
function response: {"success": true}
assistant: Thank you for providing your API key. I have successfully verified your identity. Now, could you please provide me with the ticket ID that you want to check the status for?
user: Yea it's: 12345
assistant: call function get_ticket_status, with arguments {'ticket_id': '12345'}
function response: {"status": "Open", "issue": "Billing Query"}
assistant: Your ticket with ID 12345 is currently open and it's a billing query. Is there anything else I can assist you with?
user: That's it, thanks.
assistant: You're welcome! Don't hesitate to reach out if you have any other questions or concerns. Have a great day!
```
