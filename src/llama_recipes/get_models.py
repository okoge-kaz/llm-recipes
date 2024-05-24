from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    MistralForCausalLM,
    Phi3ForCausalLM,
    AutoModelForCausalLM,
)
from llama_recipes.utils.distributed import is_rank_0
import torch
from megatron_lm.megatron.global_vars import get_args


def get_model(
    model_name: str, use_cache: bool = False
) -> LlamaForCausalLM | MistralForCausalLM | AutoModelForCausalLM:
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

    if "Llama" in model_name or "Swallow" in model_name:
        if args.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models like llama 70B, in which case
            model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some communications
            overhead.
            """
            if is_rank_0():
                model = LlamaForCausalLM.from_pretrained(
                    model_name,
                    load_in_8bit=True if args.quantization else None,
                    device_map="auto" if args.quantization else None,
                    use_cache=use_cache,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True if args.quantization else None,
                device_map="auto" if args.quantization else None,
                use_cache=use_cache,
            )

        return model  # type: ignore

    elif "Mistral" in model_name or "mistral" in model_name:
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

        return model  # type: ignore

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

        return model  # type: ignore

    else:
        raise NotImplementedError("model not implemented")
