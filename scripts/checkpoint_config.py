"""
Rebuild the input pipeline a checkpoint was trained with.

A checkpoint is only meaningful together with its front-end: 'load_state_dict'
checks the shapes of the weights and nothing else, so a foreign waveform
length, window or hop is accepted silently and quietly corrupts the scores (a
measurement on the trained model is quoted in src/trainer/config_check.py).
The scoring tools therefore take the model and everything that shapes its input
from the config of the training run, and only report how far the current
configs of the project have drifted away from it.
"""

import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.datasets.collate import LEGACY_MAX_LEN  # noqa: E402
from src.trainer.config_check import (  # noqa: E402
    config_mismatches,
    format_mismatch_note,
)

CONFIG_DIR = PROJECT_ROOT / "src" / "configs"
CURRENT_CONFIG_NAME = "inference"
SIDECAR_NAME = "config.yaml"


def load_checkpoint_config(checkpoint: Path) -> DictConfig:
    """
    Read the config the checkpoint was trained with.

    The copy saved next to the checkpoint is preferred over the one stored
    inside it: it is the file a human can read and, if the run was resumed,
    correct.

    Args:
        checkpoint (Path): path to the checkpoint.
    Returns:
        config (DictConfig): config of the training run.
    """
    sidecar = checkpoint.parent / SIDECAR_NAME
    if sidecar.exists():
        loaded = OmegaConf.load(sidecar)
        if not isinstance(loaded, DictConfig):
            raise SystemExit(f"'{sidecar}' is not a config: expected a mapping")
        print(f"input pipeline taken from {sidecar}")
        return loaded

    try:
        stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as e:
        raise SystemExit(f"Cannot read the checkpoint '{checkpoint}': {e}")

    config = stored.get("config") if isinstance(stored, dict) else None
    if config is None:
        raise SystemExit(
            f"Neither '{sidecar}' nor the checkpoint itself stores a config, "
            "so the front-end of the training run is unknown. Scoring with a "
            "guessed input pipeline would produce meaningless scores."
        )
    print(f"input pipeline taken from the config stored in {checkpoint}")
    return config


def report_config_drift(saved_config: DictConfig, checkpoint: Path) -> None:
    """
    Compare the config of the checkpoint with the current configs of the
    project and report every difference.

    The scoring run uses the checkpoint config regardless, so this is a note
    and not a warning: it only shows how far the project configs have moved
    away from the trained model. The LFCC systems of the final result always
    hit this branch, since 'inference.yaml' describes the spectrogram one.

    Args:
        saved_config (DictConfig): config of the training run.
        checkpoint (Path): path of the checkpoint, for the text of the note.
    """
    try:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            current_config = compose(config_name=CURRENT_CONFIG_NAME)
    except Exception as e:  # hydra raises a whole family of composition errors
        print(f"warning: cannot read '{CURRENT_CONFIG_NAME}.yaml' ({e})")
        return

    mismatches = config_mismatches(saved_config, current_config)
    if mismatches:
        print(format_mismatch_note(mismatches, str(checkpoint)))
    else:
        print("Checkpoint config matches the current configs of the project.")


def resolve_device(name: str) -> str:
    """
    Turn the requested device into a concrete one.

    Args:
        name (str): 'auto', 'cpu' or a cuda device.
    Returns:
        device (str): device to run on.
    """
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_run_config(
    saved_config: DictConfig,
    checkpoint: Path,
    partition: str,
    device: str,
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    limit: int | None = None,
) -> DictConfig:
    """
    Assemble the config of a scoring run.

    Everything that shapes the input of the model (the waveform length, the
    front-end, the model itself) comes from the training run; only the way the
    data is fed (batch size, workers, device) is taken from the caller.

    Args:
        saved_config (DictConfig): config of the training run.
        checkpoint (Path): weights to load.
        partition (str): partition to score, "eval" for the submission.
        device (str): device to run on.
        data_dir (Path): LA root directory.
        batch_size (int): utterances per batch.
        num_workers (int): dataloader workers.
        limit (int | None): score only the first N utterances, for a smoke run.
    Returns:
        config (DictConfig): config accepted by get_dataloaders and Inferencer.
    """
    # a config without the key comes from a run older than the key itself
    collate_max_len = int(saved_config.get("collate_max_len", LEGACY_MAX_LEN))
    transforms = saved_config.get("transforms", {})
    batch_transforms = transforms.get("batch_transforms", {})
    instance_transforms = transforms.get("instance_transforms", {})

    if batch_transforms.get("inference") is None:
        raise SystemExit(
            "The checkpoint config defines no inference front-end "
            "(transforms.batch_transforms.inference), the input of the model "
            "cannot be reproduced."
        )

    return OmegaConf.create(
        {
            "model": saved_config.get("model"),
            "transforms": {
                "batch_transforms": {"inference": batch_transforms.get("inference")},
                "instance_transforms": instance_transforms,
            },
            "collate_max_len": collate_max_len,
            "datasets": {
                partition: {
                    "_target_": "src.datasets.ASVspoofDataset",
                    "part": partition,
                    "data_dir": str(data_dir),
                    "max_len": collate_max_len,
                    "random_crop": False,
                    "limit": limit,
                    "instance_transforms": instance_transforms.get("inference"),
                }
            },
            "dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": batch_size,
                "num_workers": num_workers,
                "pin_memory": device != "cpu",
            },
            "inferencer": {
                "device_tensors": ["data_object", "labels"],
                "device": device,
                # fp32 scores for the submission, bf16 only changes the speed
                "use_amp": False,
                "from_pretrained": str(checkpoint),
            },
        }
    )
