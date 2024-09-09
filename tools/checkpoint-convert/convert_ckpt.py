import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-base-model-checkpoint-path", type=str,
        required=True, help="HuggingFace transformers model name"
    )
    parser.add_argument("--hf-tokenizer-path", type=str, required=True)
    parser.add_argument(
        "--pytorch-model-checkpoint-path", type=str,
        required=True, help="Path to checkpoint (`model.pth`)"
    )
    parser.add_argument("--out", type=str, required=True, help="Path to output directory")
    parser.add_argument("--sequence-length", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading HF model: {args.hf_base_model_checkpoint_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_base_model_checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        max_position_embeddings=args.sequence_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.hf_tokenizer_path)

    print(f"Loading CKPT: {args.pytorch_model_checkpoint_path}", flush=True)
    state_dict = torch.load(args.pytorch_model_checkpoint_path, map_location="cpu")

    print("Loading state dict into HF model", flush=True)
    model.load_state_dict(state_dict)

    print("Saving HF model", flush=True)
    model.save_pretrained(args.out, safe_serialization=True)
    tokenizer.save_pretrained(args.out)


if __name__ == "__main__":
    main()
