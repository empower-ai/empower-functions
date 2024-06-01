
from llama_cpp.server.types import CreateChatCompletionRequest


class CreateChatCompletionRequestPatched(CreateChatCompletionRequest):
    include_thinking: bool = False
