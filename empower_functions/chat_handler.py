import json

from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, Protocol, cast

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

import llama_cpp.llama as llama
import llama_cpp.llama_types as llama_types
from llama_cpp.llama_chat_format import LlamaChatCompletionHandler
from empower_functions.prompt import prompt_messages

class EmpowerFunctionsCompletionHandler(LlamaChatCompletionHandler):
    SYSTEM_INSTRUCTION = "In this environment you have access to a set of functions defined in the JSON format you can use to address user's requests, use them if needed."

    def __call__(
        self,
        llama: llama.Llama,
        messages: List[llama_types.ChatCompletionRequestMessage],
        functions: Optional[List[llama_types.ChatCompletionFunction]] = None,
        function_call: Optional[llama_types.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_types.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_types.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        response_format: Optional[llama_types.ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama.LogitsProcessorList] = None,
        grammar: Optional[llama.LlamaGrammar] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs,  # type: ignore
    ) -> Union[
        llama_types.CreateChatCompletionResponse,
        Iterator[llama_types.CreateChatCompletionStreamResponse],
    ]:        
        template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = '<|begin_of_text|>' + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        template_renderer = ImmutableSandboxedEnvironment(
            autoescape=False,
            undefined=jinja2.StrictUndefined,
        ).from_string(template)

        # Convert legacy function_call to tool_choice
        if function_call is not None:
            if isinstance(function_call, str) and (
                function_call == "none" or function_call == "auto"
            ):
                tool_choice = function_call
            if isinstance(function_call, dict) and "name" in function_call:
                tool_choice = {
                    "type": "function",
                    "function": {
                        "name": function_call["name"],
                    },
                }
        if not tool_choice:
            tool_choice = "auto"

        assert tool_choice != "any"

        if functions is None:
            functions = [ tool.get('function') for tool in tools ]
        stop = [stop, "<|eot_id|>"] if isinstance(stop, str) else stop + ["<|eot_id|>"] if stop else ["<|eot_id|>"]

        if tool_choice == "none":
            functions = []
        prompted_messages = prompt_messages(messages, functions)
        prompt = template_renderer.render(messages=prompted_messages, add_generation_prompt=True)

        # Case 1: No tool choice by user
        generated = llama.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                typical_p=typical_p,
                stream=stream,
                stop=stop,
                max_tokens=max_tokens,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repeat_penalty=repeat_penalty,
                tfs_z=tfs_z,
                mirostat_mode=mirostat_mode,
                mirostat_tau=mirostat_tau,
                mirostat_eta=mirostat_eta,
                model=model,
                logits_processor=logits_processor,
                grammar=grammar,
                logprobs=top_logprobs if logprobs else None,
            )
        generated_text = generated['choices'][0]['text']
        if generated_text.startswith('<f>'):
            json_object = json.loads(generated_text[3:])
            return _convert_completion_to_chat_function(
                completion_or_chunks=generated
            )
        elif generated_text.startswith('<c>'):
            generated['choices'][0]['text'] = generated_text[3:]
            return _convert_completion_to_chat(generated, stream=stream)

        return _convert_completion_to_chat(generated, stream=stream)

def _convert_completion_to_chat(
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool = False,
) -> Union[
    llama_types.CreateChatCompletionResponse, Iterator[llama_types.ChatCompletionChunk]
]:
    if stream:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore
        return _convert_text_completion_chunks_to_chat(chunks)
    else:
        completion: llama_types.Completion = completion_or_chunks  # type: ignore
        return _convert_text_completion_to_chat(completion)

def _maybe_json_dumps(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _convert_completion_to_chat_function(
    completion_or_chunks: llama_types.CreateCompletionResponse,
):
    completion: llama_types.CreateCompletionResponse = completion_or_chunks  # type: ignore
    assert "usage" in completion
    # tool_id = "call_" + "_0_" + tool_name + "_" + completion["id"]
    # TODO: Fix for legacy function calls
    json_object = json.loads(completion["choices"][0]["text"][3:])

    print(json.dumps(json_object, indent=2))

    tool_calls = [
        {
            "id": "call_" + "_0_" + tool['name'] + "_" + completion["id"] + "_" + str(i),
            "type": "function",
            "function": {
                "name": tool['name'],
                "arguments": _maybe_json_dumps(tool['arguments']),
            },
        }

        for (i, tool) in enumerate(json_object)
    ]
    
    json_object = json.loads(completion["choices"][0]["text"][3:])
    
    chat_completion: llama_types.CreateChatCompletionResponse = {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls
                },
                "logprobs": completion["choices"][0]["logprobs"],
                "finish_reason": "tool_calls",
            }
        ],
        "usage": completion["usage"],
    }
    return chat_completion

def _convert_text_completion_chunks_to_chat(
    chunks: Iterator[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.ChatCompletionChunk]:
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": (
                        {
                            "content": chunk["choices"][0]["text"],
                        }
                        if chunk["choices"][0]["finish_reason"] is None
                        else {}
                    ),
                    "logprobs": chunk["choices"][0]["logprobs"],
                    "finish_reason": chunk["choices"][0]["finish_reason"],
                }
            ],
        }


def _convert_completion_to_chat(
    completion_or_chunks: Union[
        llama_types.CreateCompletionResponse,
        Iterator[llama_types.CreateCompletionStreamResponse],
    ],
    stream: bool = False,
) -> Union[
    llama_types.CreateChatCompletionResponse, Iterator[llama_types.ChatCompletionChunk]
]:
    if stream:
        chunks: Iterator[llama_types.CreateCompletionStreamResponse] = completion_or_chunks  # type: ignore
        return _convert_text_completion_chunks_to_chat(chunks)
    else:
        completion: llama_types.Completion = completion_or_chunks  # type: ignore
        return _convert_text_completion_to_chat(completion)

def _convert_text_completion_to_chat(
    completion: llama_types.Completion,
) -> llama_types.ChatCompletion:
    assert "usage" in completion
    return {
        "id": "chat" + completion["id"],
        "object": "chat.completion",
        "created": completion["created"],
        "model": completion["model"],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion["choices"][0]["text"],
                },
                "logprobs": completion["choices"][0]["logprobs"],
                "finish_reason": completion["choices"][0]["finish_reason"],
            }
        ],
        "usage": completion["usage"],
    }


def _convert_text_completion_chunks_to_chat(
    chunks: Iterator[llama_types.CreateCompletionStreamResponse],
) -> Iterator[llama_types.ChatCompletionChunk]:
    for i, chunk in enumerate(chunks):
        if i == 0:
            yield {
                "id": "chat" + chunk["id"],
                "model": chunk["model"],
                "created": chunk["created"],
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                        },
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            }
        yield {
            "id": "chat" + chunk["id"],
            "model": chunk["model"],
            "created": chunk["created"],
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": (
                        {
                            "content": chunk["choices"][0]["text"],
                        }
                        if chunk["choices"][0]["finish_reason"] is None
                        else {}
                    ),
                    "logprobs": chunk["choices"][0]["logprobs"],
                    "finish_reason": chunk["choices"][0]["finish_reason"],
                }
            ],
        }
