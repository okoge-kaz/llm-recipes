import copy
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as torch_distributed
from transformers.tokenization_utils import PreTrainedTokenizer
from pathlib import Path
from llama_recipes.utils.distributed import print_rank_0

from megatron_lm.megatron.global_vars import get_args, set_sampler


class InstructDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_path: str,
    ) -> None:
        args = get_args()

        self.data_path: str = data_path
        self.max_tokens: int = args.seq_length
        self.tokenizer = tokenizer
        self.debug_mode = args.instruct_debug

        # system prompt
        self.system_prompt_role = args.system_prompt_role
        self.system_prompt_content = args.system_prompt_content

        # index file
        dataset_dir = Path(self.data_path).parent
        index_cache_dir = dataset_dir / ".index_cache"
        os.makedirs(index_cache_dir, exist_ok=True)
        index_file_path = index_cache_dir / str(os.path.basename(self.data_path)).replace(".jsonl", ".idx")
        self.index_file_path: str = str(index_file_path)

        try:
            with open(self.index_file_path, "r", encoding="utf-8") as f:
                self.indexes: list[int] = [int(line.strip()) for line in f]
        except Exception as e:
            print(f"index file error: {e}")
            exit(1)

    def __len__(self) -> int:
        return len(self.indexes)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        with open(self.data_path, "r", encoding="utf-8") as file:
            offset: int = self.indexes[index]
            file.seek(offset)
            try:
                line = file.readline()
            except Exception as e:
                print(f"index={index}, offset={offset}, error={e}")
                exit(1)

            try:
                conversations: dict[str, list[dict[str, str]] | str] = json.loads(line)
            except Exception as e:
                print(f"index={index}, offset={offset}, line={line}, error={e}")
                exit(1)

        eod_token_id: int = self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0]

        if "role" in conversations and conversations["role"] == "next_token_prediction":
            prompt = [self.tokenizer.bos_token_id]
            example = self.tokenizer.encode(
                conversations["content"], add_special_tokens=True  # type: ignore  # <bos>text<eos> + <pad>
            )
            example += [eod_token_id]
            tensor_example = torch.tensor(example, dtype=torch.int64)
        else:
            SYSTEM_PROMPT: list[dict[str, str]] = [
                {
                    "role": self.system_prompt_role,
                    "content": self.system_prompt_content,
                }
            ]
            # chat template
            prompt = self.tokenizer.apply_chat_template(
                conversation=SYSTEM_PROMPT + conversations["input"],  # type: ignore
                tokenize=True,
            )

            example = self.tokenizer.apply_chat_template(
                conversation=SYSTEM_PROMPT + conversations["input"] + [conversations["output"]],  # type: ignore
                tokenize=True,
            )
            tensor_example: torch.Tensor = torch.tensor(example, dtype=torch.int64)

        if self.debug_mode:
            print(
                f"prompt: {self.tokenizer.decode(prompt, skip_special_tokens=False)}\n\nexample: {self.tokenizer.decode(example, skip_special_tokens=False)}\n\n",
                flush=True,
            )

        if len(example) > self.max_tokens:
            print(f"\n\nWARNING: example={self.tokenizer.decode(example)}\n\n")

        padding_length: int = self.max_tokens - len(example)
        pad_token_id: int = self.tokenizer.pad_token_id  # type: ignore
        assert pad_token_id is not None
        if padding_length > 0:
            pad_tensor = torch.full((padding_length,), pad_token_id, dtype=torch.int64)
            tensor_example = torch.cat((tensor_example, pad_tensor))
        elif padding_length < 0:
            tensor_example = tensor_example[: self.max_tokens]

        labels = copy.deepcopy(tensor_example)
        # promptの長さ分だけ -1 で埋める -> 損失関数で無視するようになる
        labels[: len(prompt)] = -1
        label_mask = labels.ge(0)

        if torch.all(label_mask == 0):  # 予測部分がない
            random_index: int = np.random.randint(0, len(self.indexes))
            self.__getitem__(random_index)

        # ~label_mask -> prompt の部分を ignore_index で埋める
        labels[~label_mask] = IGNORE_INDEX
        labels[labels == pad_token_id] = IGNORE_INDEX
        # mask out pad token
        attention_mask = (tensor_example != pad_token_id).float()

        # assert
        if self.debug_mode:
            # padding
            pad_ignore_count = torch.sum((tensor_example == pad_token_id) & (labels == IGNORE_INDEX)).item()
            assert (
                pad_ignore_count == padding_length
            ), f"Number of IGNORE_INDEX due to padding ({pad_ignore_count}) does not match padding_length ({padding_length})"

            # prompt
            non_pad_ignore_count = torch.sum(
                (tensor_example != pad_token_id) & (labels == IGNORE_INDEX)).item()
            assert non_pad_ignore_count == len(
                prompt
            ), f"Number of IGNORE_INDEX not due to padding ({non_pad_ignore_count}) does not match prompt length ({len(prompt)})"

            # labels' non ignore index
            if "output" in conversations:
                non_ignore_labels = labels[labels != IGNORE_INDEX]

                chat_template = [conversations["output"]]
                expected_tokens = self.tokenizer.apply_chat_template(
                    chat_template, return_tensors="pt", tokenize=True  # type: ignore
                ).squeeze()  # type: ignore
                if expected_tokens[0] == self.tokenizer.bos_token_id:
                    expected_tokens = expected_tokens[1:]

                assert torch.all(
                    non_ignore_labels == expected_tokens
                ), "Non-ignored labels do not match the tokenized last assistant message"

        return {
            "input_ids": tensor_example,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def worker_init_fn(worker_id: int) -> None:
    import random

    args = get_args()

    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_instruction_tuning_dataloader(
    tokenizer: PreTrainedTokenizer,
    data_path: str,
    train: bool = False,
) -> DataLoader:
    from llama_recipes.utils.sequence_length_warmup import CustomDistributedSampler
    from llama_recipes.utils.checkpoint import load_sampler_state_dict

    args = get_args()

    instruction_dataset = InstructDataset(
        tokenizer=tokenizer,
        data_path=data_path,
    )

    if train:
        args.instruction_dataset_size = len(instruction_dataset)
        print_rank_0(f"Instruction dataset size: {args.instruction_dataset_size}")

    train_sampler = CustomDistributedSampler(
        dataset=instruction_dataset,
        rank=torch_distributed.get_rank(),
        num_replicas=torch_distributed.get_world_size(),
        shuffle=True,
        seed=args.seed,
    )

    if args.load:
        load_sampler_state_dict(sampler=train_sampler, path=args.load)

    set_sampler(sampler=train_sampler)

    return DataLoader(
        instruction_dataset,
        batch_size=args.micro_batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
