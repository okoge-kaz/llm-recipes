import argparse
import os
import sys

current_path: str = os.getcwd()
sys.path.append(f"{current_path}/src")
sys.path.append(current_path)

import torch
import torch.distributed._shard.checkpoint as dist_cp
from torch.distributed._shard.checkpoint import FileSystemReader
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_sharded_model_single_gpu(model, model_path: str):
    state_dict = {
        "model": model.state_dict()
    }
    dist_cp.load(
        state_dict=state_dict,
        storage_reader= FileSystemReader(model_path),
    )
    model.load_state_dict(state_dict["model"])

    print(f"Sharded state checkpoint loaded from {model_path}", flush=True)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Llama-Recipes convert")

    parser.add_argument("--hf-base-model-path", type=str, default=None, help="huggingface checkpoint path")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="tokenizer path")
    parser.add_argument("--fsdp-checkpoint-path", type=str, default=None, help="FSDP checkpoint path")
    parser.add_argument("--checkpoint-output-path", type=str, default=None, help="output checkpoint path")
    parser.add_argument("--sequence-length", type=int, required=True)

    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()

    model = AutoModelForCausalLM.from_pretrained(
        args.hf_base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_position_embeddings=args.sequence_length,
    )
    print(f"Loaded HF model: {args.hf_base_model_path}\n", flush=True)

    print(f"Loading sharded checkpoint from {args.fsdp_checkpoint_path}", flush=True)
    model = load_sharded_model_single_gpu(model, args.fsdp_checkpoint_path)
    print(f"Loaded sharded checkpoint from {args.fsdp_checkpoint_path}\n", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
    )
    tokenizer.save_pretrained(args.checkpoint_output_path)

    model.save_pretrained(args.checkpoint_output_path, safe_serialization=True)
    print(f"Saved checkpoint to {args.checkpoint_output_path}", flush=True)


if __name__ == "__main__":
    main()
