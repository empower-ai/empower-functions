from __future__ import annotations
from typing import Any, Dict, Iterator, List, Optional, Union

import anyio
from fastapi.concurrency import run_in_threadpool
import llama_cpp
from llama_cpp.server.model import LlamaProxy
from llama_cpp.server.app import (
    get_llama_proxy,
    create_chat_completion as _create_chat_completion,
    router,
    authenticate,
    openai_v1_tag,
    _logit_bias_tokens_to_input_ids,
    get_event_publisher,
    _ping_message_factory
)
from llama_cpp.server.types import (
    ChatCompletionRequestMessage,
)

import llama_cpp

from fastapi import Depends, Request, Body

from llama_cpp.server.model import (
    LlamaProxy,
)
from sse_starlette import EventSourceResponse
from .types import (
    CreateChatCompletionRequestPatched,
)
import llama_cpp.llama_chat_format as llama_chat_format


def _create_chat_completion_patched(
        llama: llama_cpp.Llama,
        messages: List[ChatCompletionRequestMessage],
        functions: Optional[List[llama_cpp.ChatCompletionFunction]] = None,
        function_call: Optional[llama_cpp.ChatCompletionRequestFunctionCall] = None,
        tools: Optional[List[llama_cpp.ChatCompletionTool]] = None,
        tool_choice: Optional[llama_cpp.ChatCompletionToolChoiceOption] = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k: int = 40,
        min_p: float = 0.05,
        typical_p: float = 1.0,
        stream: bool = False,
        stop: Optional[Union[str, List[str]]] = [],
        seed: Optional[int] = None,
        response_format: Optional[llama_cpp.ChatCompletionRequestResponseFormat] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        repeat_penalty: float = 1.1,
        tfs_z: float = 1.0,
        mirostat_mode: int = 0,
        mirostat_tau: float = 5.0,
        mirostat_eta: float = 0.1,
        model: Optional[str] = None,
        logits_processor: Optional[llama_cpp.LogitsProcessorList] = None,
        grammar: Optional[llama_cpp.LlamaGrammar] = None,
        logit_bias: Optional[Dict[str, float]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[
        llama_cpp.CreateChatCompletionResponse, Iterator[llama_cpp.CreateChatCompletionStreamResponse]
]:
    handler = llama.chat_handler or llama._chat_handlers.get(llama.chat_format) or llama_chat_format.get_chat_completion_handler(
        llama.chat_format
    )
    return handler(
        llama=llama,
        messages=messages,
        functions=functions,
        function_call=function_call,
        tools=tools,
        tool_choice=tool_choice,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        typical_p=typical_p,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
        stream=stream,
        stop=stop,
        seed=seed,
        response_format=response_format,
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
        logit_bias=logit_bias,
        **kwargs,
    )


@router.post(
    "/v1/chat/completions",
    summary="Chat",
    dependencies=[Depends(authenticate)],
    response_model=Union[llama_cpp.ChatCompletion, str],
    responses={
        "200": {
            "description": "Successful Response",
            "content": {
                "application/json": {
                    "schema": {
                        "anyOf": [
                            {
                                "$ref": "#/components/schemas/CreateChatCompletionResponse"
                            }
                        ],
                        "title": "Completion response, when stream=False",
                    }
                },
                "text/event-stream": {
                    "schema": {
                        "type": "string",
                        "title": "Server Side Streaming response, when stream=True"
                        + "See SSE format: https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format",  # noqa: E501
                        "example": """data: {... see CreateChatCompletionResponse ...} \\n\\n data: ... \\n\\n ... data: [DONE]""",
                    }
                },
            },
        }
    },
    tags=[openai_v1_tag],
    name="create_chat_completion_patched",
)
async def create_chat_completion(
    request: Request,
    body: CreateChatCompletionRequestPatched = Body(),
    llama_proxy: LlamaProxy = Depends(get_llama_proxy),
):
    exclude = {
        "n",
        "logit_bias_type",
        "user",
        "min_tokens",
    }
    kwargs = body.model_dump(exclude=exclude)
    llama = llama_proxy(body.model)
    if body.logit_bias is not None:
        kwargs["logit_bias"] = (
            _logit_bias_tokens_to_input_ids(llama, body.logit_bias)
            if body.logit_bias_type == "tokens"
            else body.logit_bias
        )

    if body.grammar is not None:
        kwargs["grammar"] = llama_cpp.LlamaGrammar.from_string(body.grammar)

    if body.min_tokens > 0:
        _min_tokens_logits_processor = llama_cpp.LogitsProcessorList(
            [llama_cpp.MinTokensLogitsProcessor(
                body.min_tokens, llama.token_eos())]
        )
        if "logits_processor" not in kwargs:
            kwargs["logits_processor"] = _min_tokens_logits_processor
        else:
            kwargs["logits_processor"].extend(_min_tokens_logits_processor)

    kwargs["llama"] = llama
    iterator_or_completion: Union[
        llama_cpp.ChatCompletion, Iterator[llama_cpp.ChatCompletionChunk]
    ] = await run_in_threadpool(_create_chat_completion_patched, **kwargs)

    if isinstance(iterator_or_completion, Iterator):
        # EAFP: It's easier to ask for forgiveness than permission
        first_response = await run_in_threadpool(next, iterator_or_completion)

        # If no exception was raised from first_response, we can assume that
        # the iterator is valid and we can use it to stream the response.
        def iterator() -> Iterator[llama_cpp.ChatCompletionChunk]:
            yield first_response
            yield from iterator_or_completion

        send_chan, recv_chan = anyio.create_memory_object_stream(10)
        return EventSourceResponse(
            recv_chan,
            data_sender_callable=partial(  # type: ignore
                get_event_publisher,
                request=request,
                inner_send_chan=send_chan,
                iterator=iterator(),
            ),
            sep="\n",
            ping_message_factory=_ping_message_factory,
        )
    else:
        return iterator_or_completion
    # return await _create_chat_completion(request, body, llama_proxy)


def patch_app():
    for route in router.routes:
        if route.name == "create_chat_completion":
            router.routes.remove(route)
            break
