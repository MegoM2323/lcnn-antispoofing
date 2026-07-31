"""
Score the whole eval partition with a trained checkpoint and build the
submission file.

The input pipeline is taken from the config saved next to the checkpoint, not
from the current configs of the project. A checkpoint is only meaningful
together with the front-end it was trained with: 'load_state_dict' checks the
shapes of the weights and nothing else, so a foreign waveform length or a
foreign window is accepted silently and quietly corrupts the scores (see
src/trainer/config_check.py). Everything that the current configs would set
differently is reported, but the checkpoint always wins.

Usage:
    python3 scripts/predict_eval.py saved/lcnn_stft_600f/model_best.pth
"""

import argparse
import os
import sys
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.checkpoint_config import (  # noqa: E402
    build_run_config,
    load_checkpoint_config,
    report_config_drift,
    resolve_device,
)
from scripts.make_submission import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_SUBMISSION_NAME,
    validate_submission,
)
from scripts.score_files import load_score_file, write_score_csv  # noqa: E402
from src.datasets.data_utils import get_dataloaders  # noqa: E402
from src.trainer import Inferencer  # noqa: E402
from src.utils.init_utils import set_random_seed  # noqa: E402
from src.utils.protocol import read_protocol_entries  # noqa: E402

DEFAULT_SAVE_DIR = PROJECT_ROOT / "data" / "saved" / "eval"
DEFAULT_DATA_DIR = os.environ.get("ASVSPOOF_DIR", str(PROJECT_ROOT / "data" / "LA"))
PARTITION = "eval"
SEED = 1


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        args (Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run a checkpoint over the eval partition and build the "
        "submission file."
    )
    parser.add_argument("checkpoint", type=Path, help="path to the '.pth' checkpoint")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_SUBMISSION_NAME),
        help=f"submission file to write (default: {DEFAULT_SUBMISSION_NAME})",
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
        "-p",
        "--protocol",
        type=Path,
        default=Path(DEFAULT_PROTOCOL),
        help="eval protocol the predictions are checked against",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(DEFAULT_DATA_DIR),
        help=f"LA root directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help=f"directory for the raw predictions (default: {DEFAULT_SAVE_DIR})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="score only the first N utterances of the protocol, for a smoke "
        "test: the result is not a valid submission",
    )
    return parser.parse_args()


def run_inference(config: DictConfig, device: str, save_dir: Path) -> Path:
    """
    Score the eval partition and write the raw predictions.

    Args:
        config (DictConfig): config built by build_run_config.
        device (str): device to run on.
        save_dir (Path): directory for '<part>_scores.csv' and
            '<part>_outputs.pth'.
    Returns:
        scores_path (Path): csv with the raw predictions.
    """
    set_random_seed(SEED, cudnn_benchmark=False)

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)

    save_dir.mkdir(parents=True, exist_ok=True)
    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_dir,
        metrics=None,
        skip_model_load=False,
    )
    inferencer.run_inference()

    return save_dir / f"{PARTITION}_scores.csv"


def build_submission(scores_path: Path, protocol_path: Path, output: Path) -> int:
    """
    Check that the predictions cover the protocol and write the submission.

    The file is written only after the check: a csv with a hole in it is a
    KeyError in the grading script and a zero for the whole homework, so it is
    better not to have such a file on disk at all.

    Args:
        scores_path (Path): csv with the raw predictions.
        protocol_path (Path): eval protocol.
        output (Path): submission file to write.
    Returns:
        exit_code (int): 0 if the submission is gradeable, 1 otherwise.
    """
    try:
        scores = load_score_file(scores_path)
        entries = read_protocol_entries(protocol_path)
    except (OSError, ValueError) as e:
        print(f"Cannot check the predictions: {e}")
        return 1

    missing = [entry.utt_id for entry in entries if entry.utt_id not in scores]
    if missing:
        print(
            f"{len(missing)} of {len(entries)} protocol utterances were not "
            f"scored, e.g. {missing[:5]}. The submission is not written."
        )
        return 1

    extra = len(scores) - len(entries)
    if extra > 0:
        print(f"warning: {extra} scored utterances are not in the protocol")

    # the rows follow the protocol, so two runs produce byte-identical files
    ordered = {entry.utt_id: scores[entry.utt_id] for entry in entries}
    try:
        write_score_csv(output, ordered)
    except OSError as e:
        print(f"Cannot write the submission '{output}': {e}")
        return 1

    print(f"\nsubmission written to {output.resolve()}\n")
    return 0 if validate_submission(output, protocol_path) is not None else 1


def main() -> int:
    """
    Score the eval partition and prepare the submission.

    Returns:
        exit_code (int): 0 on a gradeable submission, 1 otherwise.
    """
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint '{args.checkpoint}' does not exist")

    saved_config = load_checkpoint_config(args.checkpoint)
    report_config_drift(saved_config, args.checkpoint)

    device = resolve_device(args.device)
    print(f"device: {device}")

    config = build_run_config(
        saved_config,
        checkpoint=args.checkpoint,
        partition=PARTITION,
        device=device,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit=args.limit,
    )
    scores_path = run_inference(config, device, args.save_dir)

    if args.limit is not None:
        print(
            f"\n--limit={args.limit} was given: the predictions in "
            f"{scores_path} cover a part of the protocol and are not a "
            "submission."
        )
        return 0

    return build_submission(scores_path, args.protocol, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
