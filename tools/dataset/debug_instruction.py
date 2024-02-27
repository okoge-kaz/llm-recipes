from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
import torch
import numpy as np
import copy

IGNORE_INDEX = -100

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore
    "/groups/gaf51275/hf-checkpoints/Llama-2-13b-hf/"
)
torch.set_printoptions(threshold=4096)

prompt: str = "こんにちは"

example: str = prompt + "世界"  # type: ignore

encoded_prompt: torch.Tensor = torch.tensor(
    tokenizer.encode(prompt, add_special_tokens=False),
    dtype=torch.int64
)
encoded_example: list[int] = tokenizer.encode(
    example, add_special_tokens=False
)
encoded_example.append(tokenizer.eos_token_id)  # type: ignore
encoded_tensor_example: torch.Tensor = torch.tensor(encoded_example, dtype=torch.int64)

padding: int = 4096 - encoded_tensor_example.shape[0]
if padding > 0:  # pad_token_id = 0 (substitute unk_token)
    encoded_tensor_example = torch.cat((encoded_tensor_example, torch.zeros(padding, dtype=torch.int64) - 1))
elif padding < 0:
    encoded_tensor_example = encoded_tensor_example[: 4096]

labels = copy.deepcopy(encoded_tensor_example)
# promptの長さ分だけ -1 で埋める -> 損失関数で無視するようになる
labels[: len(encoded_prompt)] = -1
# 0より大きい(ge)かどうかの真偽値でmaskを作成
example_mask = encoded_tensor_example.ge(0)
label_mask = labels.ge(0)

# ~example_mask -> paddingの部分を 0 で埋める
encoded_tensor_example[~example_mask] = 0
# ~label_mask -> prompt の部分を ignore_index で埋める
labels[~label_mask] = IGNORE_INDEX

print(
    {
        "input_ids": encoded_tensor_example,
        "labels": labels,
        "attention_mask": example_mask.float(),
    }
)
