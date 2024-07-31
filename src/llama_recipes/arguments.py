import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("llm-recipes")

    parser = _add_fsdp_args(parser=parser)
    parser = _add_data_args(parser=parser)
    parser = _add_training_args(parser=parser)
    parser = _add_regularization_args(parser=parser)
    parser = _add_instruction_tuning_args(parser=parser)

    args = parser.parse_args()

    # Args from environment
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))

    # validate
    if args.use_freeze_layers:
        assert args.no_save_optimizer_state is True
    # adam epsilon is very sensitive value so don't change
    assert args.adam_eps == 1e-8

    return args


def _print_args(title: str, args: argparse.Namespace) -> None:
    """Print arguments."""
    if args.rank == 0:
        print(f'------------------------ {title} ------------------------', flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (48 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print(f'-------------------- end of {title} ---------------------', flush=True)


def _add_fsdp_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="FSDP setting")

    group.add_argument(
        "--sharding-strategy", default="FULL_SHARD",
        choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"],
        help="which ShardingStrategy to use."
    )
    group.add_argument(
        "--checkpoint-type", default="LOCAL_STATE_DICT",
        choices=["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"],
        help="which checkpoint StateDictType to use."
    )
    group.add_argument(
        "--fsdp-activation-checkpointing", action="store_true"
    )
    group.add_argument(
        "--fsdp-cpu-offload", action="store_true"
    )
    group.add_argument(
        "--low-cpu-fsdp", action="store_true"
    )
    group.add_argument(
        "--no-meta-device", action="store_true"
    )

    return parser


def _add_data_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument(
        '--data-path', nargs='*', default=None,
        help='Path to the training dataset. Accepted format:'
        '1) a single data path, 2) multiple datasets in the'
        'form: dataset1-weight dataset1-path dataset2-weight '
        'dataset2-path ... It is used with --split when a '
        'single dataset used for all three: train, valid '
        'and test. It is exclusive to the other '
        '--*-data-path args'
    )
    group.add_argument(
        '--split', type=str, default='969, 30, 1',
        help='Comma-separated list of proportions for training,'
        ' validation, and test split. For example the split '
        '`90,5,5` will use 90%% of data for training, 5%% for '
        'validation and 5%% for test.'
    )
    group.add_argument(
        '--train-data-path', nargs='*', default=None,
        help='Path to the training dataset. Accepted format:'
        '1) a single data path, 2) multiple datasets in the'
        'form: dataset1-weight dataset1-path dataset2-weight '
        'dataset2-path ...'
    )
    group.add_argument(
        '--valid-data-path', nargs='*', default=None,
        help='Path to the validation dataset. Accepted format:'
        '1) a single data path, 2) multiple datasets in the'
        'form: dataset1-weight dataset1-path dataset2-weight '
        'dataset2-path ...'
    )
    group.add_argument(
        '--test-data-path', nargs='*', default=None,
        help='Path to the test dataset. Accepted format:'
        '1) a single data path, 2) multiple datasets in the'
        'form: dataset1-weight dataset1-path dataset2-weight '
        'dataset2-path ...'
    )
    group.add_argument(
        '--data-cache-path', default=None,
        help='Path to a directory to hold cached index files.'
    )

    group.add_argument('--vocab-size', type=int, default=None, help='Size of vocab before EOD or padding.')
    group.add_argument('--vocab-file', type=str, default=None, help='Path to the vocab file.')
    group.add_argument('--merge-file', type=str, default=None, help='Path to the BPE merge file.')
    group.add_argument('--seq-length', type=int, default=None, help='Maximum sequence length to process.')
    group.add_argument('--num-workers', type=int, default=2, help="Dataloader number of workers.")
    group.add_argument(
        '--tokenizer-type', type=str, default=None,
        choices=['SentencePieceTokenizer', 'GPTSentencePieceTokenizer', 'Llama2Tokenizer', 'Llama3Tokenizer', 'NullTokenizer'],
        help='What type of tokenizer to use.'
    )
    group.add_argument('--tokenizer-model', type=str, default=None, help='Sentencepiece tokenizer model.')
    group.add_argument(
        '--reset-position-ids', action='store_true',
        help='Reset position ids after end-of-document token.'
    )
    group.add_argument(
        '--reset-attention-mask', action='store_true',
        help='Reset self attention mask after end-of-document token.'
    )
    group.add_argument(
        '--eod-mask-loss', action='store_true',
        help='Mask loss for the end of document tokens.'
    )
    group.add_argument(
        "--retro-return-doc-ids", action="store_true",
        help="Turn this on when preprocessing retro data."
    )
    group.add_argument(
        '--short-seq-prob', type=float, default=0.1,
        help='Probability of producing a short sequence.'
    )
    group.add_argument(
        '--vocab-extra-ids', type=int, default=0,
        help='Number of additional vocabulary tokens. They are used for span masking in the T5 model'
    )
    # instruction tuning
    group.add_argument(
        '--system-prompt-role', type=str, default="system"
    )
    group.add_argument(
        '--system-prompt-content', type=str, default='あなたは誠実で優秀な日本人のアシスタントです。'
    )

    return parser


def _add_training_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title="training")

    group.add_argument('--seed', type=int, default=1234, help='Random seed used for python, numpy, pytorch, and cuda.')
    group.add_argument("--use-mpi", action="store_true")

    # wandb
    group.add_argument("--wandb-entity", type=str, default=None)
    group.add_argument("--wandb-name", type=str, default=None)
    group.add_argument("--wandb-project", type=str, default=None)

    # PEFT
    group.add_argument("--quantization", action="store_true")
    group.add_argument("--use-freeze-layers", action="store_true")
    group.add_argument(
        "--freeze-layers", type=str, help="freezed layers (ex: 1-5,7-9) 0-indexed", default=None
    )

    # precision
    group.add_argument("--bf16", action="store_true")
    group.add_argument("--fp16", action="store_true")
    group.add_argument("--mixed-precision", action="store_true")
    group.add_argument(
        "--param-dtype", type=str, default=None, choices=["fp16", "bf16", "fp32"]
    )

    # checkpoint
    group.add_argument("--load", type=str, default=None)
    group.add_argument("--save", type=str, default=None)
    group.add_argument("--base-model", type=str, default=None)

    group.add_argument("--grad-clip-norm", type=float, default=1.0)

    # interval
    group.add_argument("--eval-interval", type=int, default=100)
    group.add_argument("--save-interval", type=int, default=500)
    group.add_argument("--eval-iters", type=int, default=10)

    # optimizer
    group.add_argument(
        '--optimizer', type=str, default='adam',
        choices=['adam', 'anyprecision'],
        help='Optimizer function'
    )
    group.add_argument(
        '--lr', type=float, default=None,
        help='Initial learning rate. Depending on decay style '
        'and initial warmup, the learning rate at each iteration would be different.'
    )
    group.add_argument(
        '--lr-decay-style', type=str, default='linear',
        choices=['cosine', 'step'],
        help='Learning rate decay function.'
    )
    group.add_argument(
        '--lr-decay-iters', type=int, default=None,
        help='number of iterations to decay learning rate over,If None defaults to `--train-iters`'
    )
    group.add_argument(
        '--lr-warmup-iters', type=int, default=0,
        help='number of iterations to linearly warmup learning rate over.'
    )
    group.add_argument(
        '--min-lr', type=float, default=0.0,
        help='Minimum value for learning rate. The scheduler clip values below this threshold.'
    )

    # training iteration
    group.add_argument(
        '--train-iters', type=int, default=None,
        help='Total number of iterations to train over all '
        'training runs. Note that either train-iters or '
        'train-samples should be provided.'
    )
    group.add_argument("--train-samples", type=int, default=None)

    # batch size
    group.add_argument(
        '--global-batch-size', type=int, default=None,
        help='Training batch size. If set, it should be a multiple of micro-batch-size times data-parallel-size'
        'If this value is None, then use micro-batch-size * data-parallel-size as the '
        'global batch size. This choice will result in 1 for number of micro-batches.'
    )
    group.add_argument(
        '--micro-batch-size', type=int, default=None,
        help='Batch size per model instance (local batch size). '
        'Global batch size is local batch size times data parallel size times number of micro batches.'
    )

    # position embedding
    group.add_argument(
        '--make-vocab-size-divisible-by', type=int, default=128,
        help='Pad the vocab size to be divisible by this value.This is added for computational efficiency reasons.'
    )

    # model
    group.add_argument("--sliding-window-size", type=int, default=4096)

    # loss spike
    group.add_argument("--skip-batch", nargs='*', default=None)

    # checkpoint
    group.add_argument("--no-save-optimizer-state", action="store_true")

    # continual pre-training
    group.add_argument("--continual-pretraining", action="store_true")
    # instruction tuning
    group.add_argument("--instruction-tuning", action="store_true")
    # DPO
    group.add_argument("--direct-preference-optimization", action="store_true")

    return parser


def _add_regularization_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title='regularization')

    group.add_argument(
        '--attention-dropout', type=float, default=0.1,
        help='Post attention dropout probability.'
    )
    group.add_argument(
        '--hidden-dropout', type=float, default=0.1,
        help='Dropout probability for hidden state transformer.'
    )
    group.add_argument(
        '--weight-decay', type=float, default=0.01,
        help='Weight decay coefficient for L2 regularization.'
    )
    group.add_argument(
        '--adam-beta1', type=float, default=0.9,
        help='First coefficient for computing running averages of gradient and its square'
    )
    group.add_argument(
        '--adam-beta2', type=float, default=0.95,
        help='Second coefficient for computing running averages of gradient and its square'
    )
    group.add_argument(
        '--adam-eps', type=float, default=1e-08,
        help='Term added to the denominator to improve numerical stability'
    )

    return parser


def _add_instruction_tuning_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group(title='instruction tuning')

    group.add_argument(
        "--hf-transformer-model-dir", type=str, default=None,
    )
    group.add_argument(
        "--instruction-train-data-path", type=str, default=None,
    )
    group.add_argument(
        "--instruction-valid-data-path", type=str, default=None,
    )
    group.add_argument(
        "--epoch", type=int, default=2,
    )
    group.add_argument(
        "--instruction-dataset-size", type=int, default=None,
    )
    group.add_argument(
        "--save-sampler-state", action="store_true",
    )

    return parser
