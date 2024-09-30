import copy
import os
import sys
current_path: str = os.getcwd()
sys.path.append(f"{current_path}/src")

from datetime import timedelta

import torch
import torch.distributed as torch_distributed
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload  # type: ignore
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.optim.lr_scheduler import StepLR
import wandb

from llm_recipes.core.fsdp.checkpointing import apply_fsdp_checkpointing
from llm_recipes.training.training import (
    clear_gpu_cache,
    freeze_transformer_layers,
    get_policies,
    print_model_size,
    setup_environ_flags,
    train,
)
from llm_recipes.core.optimizer.scheduler import WarmupCosineAnnealingLR
from llm_recipes.training.random import set_seed
from llm_recipes.core.fsdp.distributed import (
    print_rank_0,
    is_rank_0,
    set_mpi_env,
    get_rank,
    get_local_rank,
)
from llm_recipes.training.get_models import get_model
from llm_recipes.core.checkpoint.checkpoint import (
    load_model_state_dict,
    load_optimizer_state_dict,
    load_dist_model_state_dict,
    load_dist_optimizer_state_dict,
    load_scheduler_state_dict,
    load_rng_state_dict,
    get_latest_iteration,
)

from llm_recipes.training.arguments import parse_args
from llm_recipes.core.fsdp.get_fsdp import get_sharding_strategy
from llm_recipes.core.precision.precision import preserve_fp32_buffers
from megatron_lm.megatron.global_vars import set_global_variables
from llm_recipes.core.distributed.distributed import get_distributed_optimizer


def main() -> None:
    # initialize
    args = parse_args()
    is_pretraining = not (args.instruction_tuning or args.direct_preference_optimization)
    set_global_variables(args=args, build_tokenizer=is_pretraining)

    # Set the seeds for reproducibility
    set_seed(seed=args.seed)  # TODO add dp_mesh

    # Distributed args.
    if args.use_mpi:
        set_mpi_env()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    args.rank = rank
    args.world_size = world_size
    args.gradient_accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    assert args.gradient_accumulation_steps >= 1

    timeout = timedelta(minutes=args.distributed_timeout_minutes)
    torch_distributed.init_process_group(
        backend="nccl", world_size=world_size, rank=rank, timeout=timeout,
    )

    # wandb setting
    if args.wandb_name is not None and is_rank_0():
        import datetime

        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d-%H-%M-%S")
        wandb_setting: dict = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.wandb_name,
            "config": vars(args),
        }
        wandb.require("core")  # type: ignore
        wandb.init(**wandb_setting)

    if torch_distributed.is_initialized():
        torch.cuda.set_device(get_local_rank())  # type: ignore
        clear_gpu_cache(get_local_rank())  # type: ignore
        setup_environ_flags(get_rank())  # type: ignore

    iteration: int = get_latest_iteration(args.load)
    args.iteration = iteration
    torch_distributed.barrier()

    # random seed
    if args.load:
        load_rng_state_dict(args.load)
        torch_distributed.barrier()

    use_cache = False
    model = get_model(
        model_name=args.base_model, use_cache=use_cache
    )
    if args.direct_preference_optimization:
        reference_model = copy.deepcopy(model)
        for param in reference_model.parameters():
            param.requires_grad = False

    if args.load:
        if args.use_dist_ckpt:
            load_dist_model_state_dict(model, args.load)  # type: ignore
        else:
            load_model_state_dict(model, args.load)  # type: ignore

    print_model_size(model, args.base_model, rank)  # type: ignore

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    # RoPE inv_freq etc. are stored in fp32, so we need to preserve them
    with preserve_fp32_buffers(model):  # type: ignore
        if args.bf16:
            model.to(torch.bfloat16)  # type: ignore
        elif args.fp16:
            model.to(torch.float16)  # type: ignore

    if args.direct_preference_optimization:
        with preserve_fp32_buffers(reference_model):
            if args.bf16:
                reference_model.to(torch.bfloat16)  # type: ignore
            elif args.fp16:
                reference_model.to(torch.float16)  # type: ignore

    if args.use_freeze_layers:
        print_rank_0("NOTE: freeze transformer layers")
        freeze_transformer_layers(model=model, layer_ranges=args.freeze_layers)

    mixed_precision_policy, wrapping_policy = get_policies(
        rank=get_rank(),
        model_name=args.base_model,
    )

    from torch.distributed._tensor.device_mesh import init_device_mesh  # type: ignore
    device_mesh = init_device_mesh(device_type="cuda", mesh_shape=(world_size, ))

    model = FSDP(
        model,  # type: ignore
        auto_wrap_policy=wrapping_policy,
        cpu_offload=CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=get_sharding_strategy(),
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        sync_module_states=args.low_cpu_fsdp,
        param_init_fn=lambda module: module.to_empty(  # type: ignore
            device=torch.cuda.current_device(), recurse=False,  # type: ignore
        )
        if args.low_cpu_fsdp and rank != 0
        else None,
        device_mesh=device_mesh,
    )
    if args.fsdp_activation_checkpointing:
        # ref: https://github.com/meta-llama/llama-recipes/blob/778e31e35cfbe385a31b3a94b794e3f75e276d1a/src/llm_recipes/finetuning.py#L193-L195
        # model.enable_input_require_grads()
        # model.gradient_checkpointing_enable()
        apply_fsdp_checkpointing(model=model, model_name=args.base_model)

    if args.direct_preference_optimization:
        reference_model = FSDP(
            reference_model,  # type: ignore
            auto_wrap_policy=wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=get_sharding_strategy(),
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=args.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(  # type: ignore
                device=torch.cuda.current_device(), recurse=False,  # type: ignore
            )
            if args.low_cpu_fsdp and rank != 0
            else None,
        )

    if not args.instruction_tuning and not args.direct_preference_optimization:
        args.continual_pretraining = True

    dpo_loss_fn = None
    if args.continual_pretraining:
        from llm_recipes.core.dataset.pretrain_dataset import build_train_valid_test_datasets
        from megatron_lm.megatron.data.data_samplers import build_pretraining_data_loader

        train_dataset, validation_dataset, test_dataset = build_train_valid_test_datasets()

        args.consumed_train_samples = args.global_batch_size * args.iteration
        args.consumed_valid_samples = args.global_batch_size * (
            args.iteration // args.eval_interval) * args.eval_iters

        train_dataloader = build_pretraining_data_loader(
            dataset=train_dataset,
            consumed_samples=args.consumed_train_samples,
        )
        validation_dataloader = build_pretraining_data_loader(
            dataset=validation_dataset,
            consumed_samples=args.consumed_valid_samples,
        )

    else:
        from transformers import AutoTokenizer
        from llm_recipes.core.dataset.instruction_tuning_dataset import get_instruction_tuning_dataloader
        from llm_recipes.core.dataset.dpo_dataset import get_dpo_dataloader

        hf_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.hf_transformer_model_dir
        )

        if args.instruction_tuning:
            train_dataloader = get_instruction_tuning_dataloader(
                tokenizer=hf_tokenizer,  # type: ignore
                data_path=args.instruction_train_data_path,
                train=True,
            )
            validation_dataloader = get_instruction_tuning_dataloader(
                tokenizer=hf_tokenizer,  # type: ignore
                data_path=args.instruction_valid_data_path,
            )

            args.train_iters = args.instruction_dataset_size // args.global_batch_size * args.epoch
            args.lr_decay_iters = args.train_iters
            args.lr_warmup_iters = args.lr_decay_iters // 10
            args.save_sampler_state = True
            if rank == 0:
                from llm_recipes.core.logs.wandb_utils import update_iter_info
                update_iter_info()

        elif args.direct_preference_optimization:
            from llm_recipes.core.dpo.dpo_loss import DPOLoss

            dpo_loss_fn = DPOLoss(
                beta=args.dpo_beta,
                label_smoothing=args.dpo_label_smoothing,
            )

            train_dataloader = get_dpo_dataloader(
                tokenizer=hf_tokenizer,  # type: ignore
                data_path=args.dpo_train_data_path,
                train=True
            )
            validation_dataloader = get_dpo_dataloader(
                tokenizer=hf_tokenizer,  # type: ignore
                data_path=args.dpo_valid_data_path
            )

            args.train_iters = args.dpo_dataset_size // args.global_batch_size * args.epoch
            args.lr_decay_iters = args.train_iters
            args.lr_warmup_iters = args.lr_decay_iters // 10
            args.save_sampler_state = True
            if rank == 0:
                from llm_recipes.core.logs.wandb_utils import update_iter_info
                update_iter_info()
        else:
            raise ValueError("unknown training mode")

    if args.use_distributed_optimizer:
        assert args.use_3d_parallelism is True, "3D parallelism must be enabled for distributed optimizer"
        assert args.use_fsdp is False, "FSDP must be disabled for distributed optimizer"

        optimizer = get_distributed_optimizer(
            model=model,
            lr=args.lr,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_epsilon=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.AdamW(  # type: ignore
            model.parameters(),  # type: ignore
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )

    if args.load:
        if args.use_dist_ckpt:
            load_dist_optimizer_state_dict(model=model, optimizer=optimizer, path=args.load)  # type: ignore
        else:
            load_optimizer_state_dict(model=model, optimizer=optimizer, path=args.load)  # type: ignore

    if args.lr_decay_style == "cosine":
        scheduler = WarmupCosineAnnealingLR(
            optimizer=optimizer,
            warmup_iterations=args.lr_warmup_iters,
            decay_iterations=args.lr_decay_iters,
            max_iterations=args.train_iters,
            eta_min=args.min_lr,
        )
    else:
        scheduler = StepLR(optimizer, step_size=1, gamma=0.85)

    if args.load:
        load_scheduler_state_dict(scheduler, args.load)  # type: ignore

    # Start the training process
    train(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=validation_dataloader,
        optimizer=optimizer,  # type: ignore
        lr_scheduler=scheduler,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        local_rank=get_local_rank(),
        rank=get_rank(),
        dpo_loss_fn=dpo_loss_fn,
        reference_model=reference_model if args.direct_preference_optimization else None,
    )


if __name__ == "__main__":
    main()
