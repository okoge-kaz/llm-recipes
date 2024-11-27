import time

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    Gemma2ForCausalLM,
    AutoModelForCausalLM,
)
from llm_recipes.core.fsdp.distributed import is_rank_0
import torch
from megatron_lm.megatron.global_vars import get_args


def get_model(
    model_name: str, use_cache: bool = False
) -> LlamaForCausalLM | MistralForCausalLM | Phi3ForCausalLM | Gemma2ForCausalLM:
    """return CausalLM model

    Args:
        model_name: str
        use_cache (bool, optional):

    Raises:
        NotImplementedError: currently only supports LlamaForCausalLM and MistralForCausalLM

    Returns:
        LlamaForCausalLM | MistralForCausalLM: PyTorch model
    """
    args = get_args()
    if is_rank_0():
        print("Instantiating Model ...", flush=True)
        init_time = time.perf_counter()

    if "Llama" in model_name or "Swallow" in model_name:
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if args.quantization else None,
            device_map="auto" if args.quantization else None,
            use_cache=use_cache,
            max_position_embeddings=args.seq_length,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    elif "Mistral" in model_name or "mistral" in model_name or "Codestral" in model_name:
        # If using torch.device("meta"), FSDP training hang
        # FYI: https://github.com/iwiwi/epochraft-hf-fsdp/pull/10#issuecomment-1803360147
        # https://github.com/pytorch/pytorch/issues/105840 are maybe helpful
        mistral_max_length: int = args.seq_length
        sliding_window: int = args.sliding_window_size
        assert sliding_window == 4096

        model = MistralForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if args.quantization else None,
            device_map="auto" if args.quantization else None,
            use_cache=use_cache,
            sliding_window=sliding_window,
            max_position_embeddings=mistral_max_length,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    elif "Phi-3" in model_name:

        model = Phi3ForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if args.quantization else None,
            device_map="auto" if args.quantization else None,
            use_cache=use_cache,
            max_position_embeddings=args.seq_length,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    elif "Yi-1.5" in model_name:
        # https://huggingface.co/01-ai/Yi-1.5-9B/blob/main/config.json

        model = LlamaForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if args.quantization else None,
            device_map="auto" if args.quantization else None,
            use_cache=use_cache,
            max_position_embeddings=args.seq_length,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    elif "gemma-2" in model_name:
        model = Gemma2ForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True if args.quantization else None,
            device_map="auto" if args.quantization else None,
            use_cache=use_cache,
            max_position_embeddings=args.seq_length,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )

    else:
        raise NotImplementedError("model not implemented")

    if is_rank_0():
        print(f"Model instantiation took {time.perf_counter() - init_time:.2f} secs")

    return model  # type: ignore
