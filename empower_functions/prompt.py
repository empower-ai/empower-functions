import json

SYSTEM_INSTRUCTION = "In this environment you have access to a set of functions defined in the JSON format you can use to address user's requests, use them if needed."


def _check_functions_def(functions_def):
    """Check if the functions definition is valid."""

    for function in functions_def:
        if 'name' not in function:
            raise 'Function name must be provided'
        if 'description' not in function:
            raise 'Function description must be provided'
        if 'parameters' not in function:
            raise 'Function parameters must be provided'

        parameters = function['parameters']
        if 'type' not in parameters:
            raise 'Function parameters type must be provided'
        if parameters['type'] != 'object':
            raise 'Function parameters type must be object'
        if 'properties' not in parameters:
            raise 'Function parameters properties must be provided'

        properties = parameters['properties']
        if not isinstance(properties, dict):
            raise 'Function parameters properties must be an object'
        if 'required' in parameters and not isinstance(parameters['required'], list):
            raise 'Function parameters required must be an array'


def _check_and_merge_messages(messages):
    """Check if the messages are valid."""
    if len(messages) == 0:
        raise 'Messages cannot be empty'

    first_message = messages[0]
    updated_messages = []
    if first_message['role'] == 'system':
        messages = messages[1:]
        updated_messages.append(first_message)

    if len(messages) == 0:
        raise 'At least user message must be provided'

    previous_role = ''
    for message in messages:
        if 'role' not in message or message['role'] not in ['user', 'assistant', 'tool']:
            raise Exception('Invalid role')

        if message['role'] == 'tool':
            if 'content' not in message:
                raise Exception(
                    '"content" must be provided for message with role "tool"')
            if 'tool_call_id' not in message or not message['tool_call_id']:
                raise Exception(
                    '"tool_call_id" must be provided for message with role "tool"')

            try:
                content = json.loads(message['content'])
            except:
                raise Exception(
                    'Content of a message with role "tool" must be a valid JSON string')

            if previous_role == 'tool':
                updated_messages[-1]['content'].append({
                    "value": content,
                    'tool_call_id': message['tool_call_id']
                })
            else:
                updated_messages.append({
                    "role": "tool",
                    "content": [{
                        "value": content,
                        'tool_call_id': message['tool_call_id']
                    }]
                })

        elif message['role'] == 'user':
            if 'content' not in message:
                raise Exception(
                    '"content" must be provided for message with role "user"')
            if previous_role == 'user':
                updated_messages[-1]['content'] += "\n\n" + message['content']
            else:
                updated_messages.append(message)
        elif message['role'] == 'assistant':
            if previous_role == 'assistant':
                raise Exception(
                    'Consecutive assistant messages are not allowed')

            if 'content' not in message and 'tool_calls' not in message:
                raise Exception(
                    'Either "content" or "tool_calls" must be provided message with role "assistant"')

            if 'tool_calls' in message:
                tool_calls = message['tool_calls']
                if not isinstance(tool_calls, list):
                    raise Exception('Tool calls must be an array')
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        raise Exception('Tool call must be an object')
                    if 'id' not in tool_call:
                        raise Exception(
                            '"id" must be provided for tool call')
                    if 'function' not in tool_call:
                        raise Exception(
                            '"function" must be provided for tool call')

                    function = tool_call['function']
                    if not isinstance(function, dict):
                        raise Exception('Tool call function must be an object')
                    if 'name' not in function:
                        raise Exception(
                            '"name" must be provided for the function in tool call')
                    if 'arguments' not in function:
                        raise Exception(
                            '"arguments" must be provided for the function in tool call')

            updated_messages.append(message)

        previous_role = message['role']

    return updated_messages


def prompt_messages(messages, functions_def, include_thinking=False):
    if not functions_def:
        functions_def = []

    if len(functions_def) == 0 and include_thinking:
        raise Exception(
            'Currently thinking mode is only supported with tools. Please provide functions_def to enable thinking mode.')

    _check_functions_def(functions_def)
    messages = _check_and_merge_messages(messages)

    system_instruction = SYSTEM_INSTRUCTION
    if include_thinking:
        system_instruction += "\nMake sure to include your thinking inside < thinking > </thinking > before response."

    first_user_message = messages[0]
    starting_index = 1
    if messages[0]['role'] == 'system':
        system_instruction = messages[0]['content']
        first_user_message = messages[1]
        starting_index = 2

    if len(functions_def) == 0:
        prompted_messages = [first_user_message]
    else:
        prompted_messages = [{'role': 'user', 'content': (
            system_instruction
            + "\n"
            + "Functions:\n"
            + json.dumps(functions_def, indent=2, ensure_ascii=False)
            + "\n\n"
            + "User Message:\n"
            + first_user_message['content']
        )}]

    for message in messages[starting_index:]:
        if message['role'] == 'tool':
            prompted_messages.append({
                'role': 'user',
                'content': '<r>' + json.dumps(message['content'], indent=2, ensure_ascii=True)
            })
        elif message['role'] == 'user':
            prompted_messages.append({
                'role': 'user',
                'content': '<u>' + message['content']
            })
        elif message['role'] == 'assistant':
            if 'content' in message and message['content'] and len(message['content']) > 0:
                prompted_messages.append({
                    'role': 'assistant',
                    'content': '<c>' + message['content']
                })
            else:
                functions = [tool_call['function']
                             for tool_call in message['tool_calls']]
                prompted_messages.append({
                    'role': 'assistant', 'content': '<f>' + json.dumps(functions, indent=2, ensure_ascii=False)
                })

    return prompted_messages
