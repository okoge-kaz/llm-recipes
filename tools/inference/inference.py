import argparse

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser(description="Generation")
parser.add_argument("--model-path", type=str)
parser.add_argument("--tokenizer-path", type=str)
parser.add_argument("--prompt", type=str, default=None)
args = parser.parse_args()


print(f"Loading model {args.model_path}")

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=args.tokenizer_path,
)
model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    device_map="auto", torch_dtype=torch.bfloat16
)

input_ids: torch.Tensor = tokenizer.encode(  # type: ignore
    args.prompt,
    add_special_tokens=False,
    return_tensors="pt"
)
outputs = model.generate(  # type: ignore
    input_ids.to(device=model.device),  # type: ignore
    max_new_tokens=1024,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
