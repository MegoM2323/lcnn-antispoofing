"""
Average the weights of several checkpoints of a run into a single model and
measure its BatchNorm statistics again.

The result is one LCNN with the very same architecture, so it is scored by
'scripts/predict_eval.py' like any other checkpoint. See
src/model/weight_averaging.py for why the running statistics of the BatchNorm
layers cannot simply be averaged along with the weights.

Usage:
    python3 scripts/average_checkpoints.py \\
        saved/lcnn_stft_600f/checkpoint-epoch{15,20,25,30}.pth \\
        -o saved/swa/swa_15_30.pth
"""

import argparse
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.checkpoint_config import (  # noqa: E402
    SIDECAR_NAME,
    load_checkpoint_config,
    resolve_device,
)
from scripts.predict_eval import DEFAULT_DATA_DIR, SEED  # noqa: E402
from src.datasets.collate import DEFAULT_MAX_LEN, get_collate_fn  # noqa: E402
from src.datasets.data_utils import (  # noqa: E402
    move_batch_transforms_to_device,
)
from src.model.weight_averaging import (  # noqa: E402
    average_state_dicts,
    recalibrate_batchnorm,
)
from src.utils.init_utils import set_random_seed, set_worker_seed  # noqa: E402

DEFAULT_BN_BATCHES = 400
BN_PARTITION = "train"


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        args (Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Average several checkpoints into one model and "
        "recalibrate its BatchNorm statistics."
    )
    parser.add_argument(
        "checkpoints", type=Path, nargs="+", help="checkpoints of one run to average"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="checkpoint to write, together with a copy of the run config",
    )
    parser.add_argument(
        "-n",
        "--bn-batches",
        type=int,
        default=DEFAULT_BN_BATCHES,
        help="train batches the BatchNorm statistics are measured on "
        f"(default: {DEFAULT_BN_BATCHES})",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=32, help="batch size (default: 32)"
    )
    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        help="'auto', 'cpu' or a cuda device (default: auto)",
    )
    parser.add_argument(
        "-j",
        "--num-workers",
        type=int,
        default=4,
        help="dataloader workers (default: 4)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help=f"LA root directory (default: {DEFAULT_DATA_DIR})",
    )
    return parser.parse_args()


def load_state_dicts(paths: list[Path]) -> list[dict]:
    """
    Read the weights of every checkpoint.

    Args:
        paths (list[Path]): checkpoints to read.
    Returns:
        state_dicts (list[dict]): weights of every checkpoint.
    """
    state_dicts = []
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Checkpoint '{path}' does not exist")
        try:
            stored = torch.load(path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError) as e:
            raise SystemExit(f"Cannot read the checkpoint '{path}': {e}")

        if not isinstance(stored, dict) or "state_dict" not in stored:
            raise SystemExit(f"'{path}' holds no 'state_dict'")
        print(f"{path}: epoch {stored.get('epoch', '?')}")
        state_dicts.append(stored["state_dict"])
    return state_dicts


def build_train_dataloader(
    config: DictConfig,
    device: str,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
):
    """
    Build the train dataloader of the run, for the BatchNorm recalibration.

    The statistics have to be measured on the data the model was trained on,
    with the front-end of the training run: another partition or another window
    would describe another input distribution.

    Args:
        config (DictConfig): config of the training run.
        device (str): device the front-end runs on.
        data_dir (Path): LA root directory.
        batch_size (int): utterances per batch.
        num_workers (int): dataloader workers.
    Returns:
        dataloader (DataLoader): shuffled dataloader over the train partition.
        transforms (nn.Module | None): front-end applied to 'data_object'.
    """
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)
    train_transforms = (batch_transforms or {}).get(BN_PARTITION) or {}

    dataset_config = config.datasets.get(BN_PARTITION)
    if dataset_config is None:
        raise SystemExit(
            "The run config defines no train partition, the BatchNorm "
            "statistics of the averaged weights cannot be measured."
        )
    with open_dict(dataset_config):
        dataset_config.data_dir = str(data_dir)
    dataset = instantiate(dataset_config)

    dataloader = instantiate(
        config.dataloader,
        dataset=dataset,
        collate_fn=get_collate_fn(config.get("collate_max_len", DEFAULT_MAX_LEN)),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        shuffle=True,
        worker_init_fn=set_worker_seed,
        _convert_="partial",
    )
    return dataloader, train_transforms.get("data_object")


def frontend_batches(dataloader, transform, device: str):
    """
    Yield model input tensors: waveforms moved to the device and passed
    through the front-end of the training run.

    Args:
        dataloader (DataLoader): dataloader over the train partition.
        transform (nn.Module | None): front-end applied to the waveforms.
        device (str): device to run on.
    Yields:
        data_object (Tensor): batch of model input.
    """
    for batch in dataloader:
        data_object = batch["data_object"].to(device)
        yield data_object if transform is None else transform(data_object)


def main() -> int:
    """
    Average the checkpoints and write the resulting model.

    Returns:
        exit_code (int): 0 on success.
    """
    args = parse_args()
    if len(args.checkpoints) < 2:
        raise SystemExit("Averaging needs at least two checkpoints")
    if args.bn_batches <= 0:
        raise SystemExit(f"--bn-batches must be positive, got {args.bn_batches}")

    config = load_checkpoint_config(args.checkpoints[0])
    device = resolve_device(args.device)
    print(f"device: {device}")

    set_random_seed(SEED, cudnn_benchmark=False)

    state_dicts = load_state_dicts(args.checkpoints)
    averaged = average_state_dicts(state_dicts)

    model = instantiate(config.model).to(device)
    model.load_state_dict(averaged)
    print(f"averaged {len(state_dicts)} checkpoints")

    dataloader, transform = build_train_dataloader(
        config, device, args.data_dir, args.batch_size, args.num_workers
    )
    print(
        f"recalibrating BatchNorm on up to {args.bn_batches} batches of "
        f"{args.batch_size} train utterances"
    )
    seen = recalibrate_batchnorm(
        model,
        frontend_batches(dataloader, transform, device),
        max_batches=args.bn_batches,
    )
    print(f"BatchNorm statistics measured on {seen} batches")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arch": type(model).__name__,
        "epoch": None,
        "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
        "config": config,
        "averaged_from": [str(path) for path in args.checkpoints],
        "bn_batches": seen,
    }
    try:
        torch.save(payload, args.output)
        OmegaConf.save(config, args.output.parent / SIDECAR_NAME)
    except OSError as e:
        raise SystemExit(f"Cannot write '{args.output}': {e}")

    print(f"\naveraged model saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
