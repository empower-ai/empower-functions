import llama_cpp
from llama_cpp.server.model import LlamaProxy
from llama_cpp.server.settings import ModelSettings
from llama_cpp.server.__main__ import main
from empower_functions.chat_handler import EmpowerFunctionsCompletionHandler
# Monkey pacthing the LlamaProxy class

def load_llama_from_model_settings(settings: ModelSettings) -> llama_cpp.Llama:
    chat_handler = None
    if settings.chat_format == "empower-functions":
        chat_handler = EmpowerFunctionsCompletionHandler()
    elif settings.chat_format == "llava-1-5":
        assert settings.clip_model_path is not None, "clip model not found"
        if settings.hf_model_repo_id is not None:
            chat_handler = (
                llama_cpp.llama_chat_format.Llava15ChatHandler.from_pretrained(
                    repo_id=settings.hf_model_repo_id,
                    filename=settings.clip_model_path,
                    verbose=settings.verbose,
                )
            )
        else:
            chat_handler = llama_cpp.llama_chat_format.Llava15ChatHandler(
                clip_model_path=settings.clip_model_path, verbose=settings.verbose
            )
    elif settings.chat_format == "obsidian":
        assert settings.clip_model_path is not None, "clip model not found"
        if settings.hf_model_repo_id is not None:
            chat_handler = (
                llama_cpp.llama_chat_format.ObsidianChatHandler.from_pretrained(
                    repo_id=settings.hf_model_repo_id,
                    filename=settings.clip_model_path,
                    verbose=settings.verbose,
                )
            )
        else:
            chat_handler = llama_cpp.llama_chat_format.ObsidianChatHandler(
                clip_model_path=settings.clip_model_path, verbose=settings.verbose
            )
    elif settings.chat_format == "llava-1-6":
        assert settings.clip_model_path is not None, "clip model not found"
        if settings.hf_model_repo_id is not None:
            chat_handler = (
                llama_cpp.llama_chat_format.Llava16ChatHandler.from_pretrained(
                    repo_id=settings.hf_model_repo_id,
                    filename=settings.clip_model_path,
                    verbose=settings.verbose,
                )
            )
        else:
            chat_handler = llama_cpp.llama_chat_format.Llava16ChatHandler(
                clip_model_path=settings.clip_model_path, verbose=settings.verbose
            )
    elif settings.chat_format == "moondream":
        assert settings.clip_model_path is not None, "clip model not found"
        if settings.hf_model_repo_id is not None:
            chat_handler = (
                llama_cpp.llama_chat_format.MoondreamChatHandler.from_pretrained(
                    repo_id=settings.hf_model_repo_id,
                    filename=settings.clip_model_path,
                    verbose=settings.verbose,
                )
            )
        else:
            chat_handler = llama_cpp.llama_chat_format.MoondreamChatHandler(
                clip_model_path=settings.clip_model_path, verbose=settings.verbose
            )
    elif settings.chat_format == "nanollava":
        assert settings.clip_model_path is not None, "clip model not found"
        if settings.hf_model_repo_id is not None:
            chat_handler = (
                llama_cpp.llama_chat_format.NanoLlavaChatHandler.from_pretrained(
                    repo_id=settings.hf_model_repo_id,
                    filename=settings.clip_model_path,
                    verbose=settings.verbose,
                )
            )
        else:
            chat_handler = llama_cpp.llama_chat_format.NanoLlavaChatHandler(
                clip_model_path=settings.clip_model_path, verbose=settings.verbose
            )
    elif settings.chat_format == "llama-3-vision-alpha":
        assert settings.clip_model_path is not None, "clip model not found"
        if settings.hf_model_repo_id is not None:
            chat_handler = (
                llama_cpp.llama_chat_format.Llama3VisionAlpha.from_pretrained(
                    repo_id=settings.hf_model_repo_id,
                    filename=settings.clip_model_path,
                    verbose=settings.verbose,
                )
            )
        else:
            chat_handler = llama_cpp.llama_chat_format.Llama3VisionAlpha(
                clip_model_path=settings.clip_model_path, verbose=settings.verbose
            )
    elif settings.chat_format == "hf-autotokenizer":
        assert (
            settings.hf_pretrained_model_name_or_path is not None
        ), "hf_pretrained_model_name_or_path must be set for hf-autotokenizer"
        chat_handler = (
            llama_cpp.llama_chat_format.hf_autotokenizer_to_chat_completion_handler(
                settings.hf_pretrained_model_name_or_path
            )
        )
    elif settings.chat_format == "hf-tokenizer-config":
        assert (
            settings.hf_tokenizer_config_path is not None
        ), "hf_tokenizer_config_path must be set for hf-tokenizer-config"
        chat_handler = llama_cpp.llama_chat_format.hf_tokenizer_config_to_chat_completion_handler(
            json.load(open(settings.hf_tokenizer_config_path))
        )

    tokenizer: Optional[llama_cpp.BaseLlamaTokenizer] = None
    if settings.hf_pretrained_model_name_or_path is not None:
        tokenizer = llama_tokenizer.LlamaHFTokenizer.from_pretrained(
            settings.hf_pretrained_model_name_or_path
        )

    draft_model = None
    if settings.draft_model is not None:
        draft_model = llama_speculative.LlamaPromptLookupDecoding(
            num_pred_tokens=settings.draft_model_num_pred_tokens
        )

    kv_overrides: Optional[Dict[str, Union[bool, int, float]]] = None
    if settings.kv_overrides is not None:
        assert isinstance(settings.kv_overrides, list)
        kv_overrides = {}
        for kv in settings.kv_overrides:
            key, value = kv.split("=")
            if ":" in value:
                value_type, value = value.split(":")
                if value_type == "bool":
                    kv_overrides[key] = value.lower() in ["true", "1"]
                elif value_type == "int":
                    kv_overrides[key] = int(value)
                elif value_type == "float":
                    kv_overrides[key] = float(value)
                else:
                    raise ValueError(f"Unknown value type {value_type}")

    import functools

    kwargs = {}

    if settings.hf_model_repo_id is not None:
        create_fn = functools.partial(
            llama_cpp.Llama.from_pretrained,
            repo_id=settings.hf_model_repo_id,
            filename=settings.model,
        )
    else:
        create_fn = llama_cpp.Llama
        kwargs["model_path"] = settings.model

    _model = create_fn(
        **kwargs,
        # Model Params
        n_gpu_layers=settings.n_gpu_layers,
        main_gpu=settings.main_gpu,
        tensor_split=settings.tensor_split,
        vocab_only=settings.vocab_only,
        use_mmap=settings.use_mmap,
        use_mlock=settings.use_mlock,
        kv_overrides=kv_overrides,
        # Context Params
        seed=settings.seed,
        n_ctx=settings.n_ctx,
        n_batch=settings.n_batch,
        n_threads=settings.n_threads,
        n_threads_batch=settings.n_threads_batch,
        rope_scaling_type=settings.rope_scaling_type,
        rope_freq_base=settings.rope_freq_base,
        rope_freq_scale=settings.rope_freq_scale,
        yarn_ext_factor=settings.yarn_ext_factor,
        yarn_attn_factor=settings.yarn_attn_factor,
        yarn_beta_fast=settings.yarn_beta_fast,
        yarn_beta_slow=settings.yarn_beta_slow,
        yarn_orig_ctx=settings.yarn_orig_ctx,
        mul_mat_q=settings.mul_mat_q,
        logits_all=settings.logits_all,
        embedding=settings.embedding,
        offload_kqv=settings.offload_kqv,
        flash_attn=settings.flash_attn,
        # Sampling Params
        last_n_tokens_size=settings.last_n_tokens_size,
        # LoRA Params
        lora_base=settings.lora_base,
        lora_path=settings.lora_path,
        # Backend Params
        numa=settings.numa,
        # Chat Format Params
        chat_format=settings.chat_format,
        chat_handler=chat_handler,
        # Speculative Decoding
        draft_model=draft_model,
        # KV Cache Quantization
        type_k=settings.type_k,
        type_v=settings.type_v,
        # Tokenizer
        tokenizer=tokenizer,
        # Misc
        verbose=settings.verbose,
    )
    if settings.cache:
        if settings.cache_type == "disk":
            if settings.verbose:
                print(f"Using disk cache with size {settings.cache_size}")
            cache = llama_cpp.LlamaDiskCache(capacity_bytes=settings.cache_size)
        else:
            if settings.verbose:
                print(f"Using ram cache with size {settings.cache_size}")
            cache = llama_cpp.LlamaRAMCache(capacity_bytes=settings.cache_size)
        _model.set_cache(cache)
    return _model

LlamaProxy.load_llama_from_model_settings = staticmethod(load_llama_from_model_settings)

if __name__ == "__main__":
    main()
