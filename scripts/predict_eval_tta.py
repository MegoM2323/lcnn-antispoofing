"""
Score the eval partition with several segments per utterance and build the
submission file.

The trained model reads 600 spectrogram frames, that is the first 4.87 s of an
utterance, and the rest of the recording is never seen (arXiv:1904.05576 uses
"only the first 600 features for each file"). Here every utterance is cut into
several windows of the very same length, each window is scored on its own and
the scores are pooled into one. Nothing is retrained and the architecture is
untouched: this is a property of the scoring pass only. Utterances shorter than
one window are scored exactly as before, on a single repeat-padded segment.

Usage:
    python3 scripts/predict_eval_tta.py saved/lcnn_stft_600f/model_best.pth
    python3 scripts/predict_eval_tta.py model_best.pth -n 5 -a min
"""

import argparse
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
)
from scripts.predict_eval import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_SAVE_DIR,
    PARTITION,
    SEED,
    build_submission,
)
from src.datasets.data_utils import get_multicrop_dataloader  # noqa: E402
from src.datasets.multicrop import (  # noqa: E402
    DEFAULT_N_SEGMENTS,
    DEFAULT_SHORT_SHIFTS,
)
from src.metrics.segment_pooling import AGGREGATIONS, DEFAULT_AGGREGATION  # noqa: E402
from src.trainer.multicrop_inferencer import MultiCropInferencer  # noqa: E402
from src.utils.init_utils import set_random_seed  # noqa: E402
from src.utils.protocol import read_utt_ids  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        args (Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Score the eval partition on several segments per "
        "utterance and build the submission file."
    )
    parser.add_argument("checkpoint", type=Path, help="path to the '.pth' checkpoint")
    parser.add_argument(
        "-n",
        "--n-segments",
        type=int,
        default=DEFAULT_N_SEGMENTS,
        help=f"segments per utterance (default: {DEFAULT_N_SEGMENTS}). 1 "
        "reproduces the single-window scoring of predict_eval.py",
    )
    parser.add_argument(
        "--short-shifts",
        type=int,
        default=DEFAULT_SHORT_SHIFTS,
        help="windows per utterance shorter than one segment, cut at evenly "
        f"spread phases of its cyclic repetition (default: {DEFAULT_SHORT_SHIFTS}, "
        "the single repeat-padded segment of the ordinary pipeline)",
    )
    parser.add_argument(
        "-a",
        "--aggregation",
        choices=AGGREGATIONS,
        default=DEFAULT_AGGREGATION,
        help=f"how the segment scores are pooled (default: {DEFAULT_AGGREGATION})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_SUBMISSION_NAME),
        help=f"submission file to write (default: {DEFAULT_SUBMISSION_NAME})",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=32,
        help="utterances per batch, not segments (default: 32)",
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
        "-s",
        "--subset",
        type=Path,
        default=None,
        help="score only the utterance ids listed in this file, one per line, "
        "for a measurement run: the result is not a valid submission",
    )
    return parser.parse_args()


def read_subset(subset_path: Path | None) -> set[str] | None:
    """
    Read the utterance ids of a measurement run.

    Args:
        subset_path (Path | None): file with the ids, one per line.
    Returns:
        utt_ids (set[str] | None): ids to score, None for the whole partition.
    """
    if subset_path is None:
        return None

    try:
        return set(read_utt_ids(subset_path))
    except OSError as e:
        raise SystemExit(f"Cannot read the subset '{subset_path}': {e}")


def run_inference(
    config: DictConfig,
    device: str,
    save_dir: Path,
    n_segments: int,
    aggregation: str,
    utt_ids: set[str] | None,
    short_shifts: int = DEFAULT_SHORT_SHIFTS,
) -> Path:
    """
    Score the eval partition and write the raw predictions.

    Args:
        config (DictConfig): config built by build_run_config.
        device (str): device to run on.
        save_dir (Path): directory for the predictions of the partition.
        n_segments (int): segments per utterance, at most.
        aggregation (str): pooling rule for the segment scores.
        utt_ids (set[str] | None): utterances to score, None scores all.
        short_shifts (int): windows per utterance that is shorter than one
            segment, see src.datasets.multicrop.shifted_segments.
    Returns:
        scores_path (Path): csv with the pooled predictions.
    """
    set_random_seed(SEED, cudnn_benchmark=False)

    try:
        dataloader, batch_transforms = get_multicrop_dataloader(
            config, PARTITION, device, n_segments, utt_ids, short_shifts
        )
    except ValueError as e:
        raise SystemExit(f"Cannot build the dataloader: {e}")

    n_utterances = len(dataloader.dataset)
    print(f"utterances to score: {n_utterances}")
    if utt_ids is not None and len(utt_ids) != n_utterances:
        print(
            f"warning: {len(utt_ids) - n_utterances} of the requested ids are "
            f"not in the '{PARTITION}' partition"
        )
    model = instantiate(config.model).to(device)

    save_dir.mkdir(parents=True, exist_ok=True)
    inferencer = MultiCropInferencer(
        model=model,
        config=config,
        device=device,
        dataloaders={PARTITION: dataloader},
        batch_transforms=batch_transforms,
        save_path=save_dir,
        metrics=None,
        skip_model_load=False,
        aggregation=aggregation,
        n_segments=n_segments,
    )
    inferencer.run_inference()

    return save_dir / f"{PARTITION}_scores.csv"


def main() -> int:
    """
    Score the eval partition with multi-crop inference.

    Returns:
        exit_code (int): 0 on a gradeable submission, 1 otherwise.
    """
    args = parse_args()
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint '{args.checkpoint}' does not exist")
    if args.n_segments <= 0:
        raise SystemExit(f"--n-segments must be positive, got {args.n_segments}")
    if args.short_shifts <= 0:
        raise SystemExit(f"--short-shifts must be positive, got {args.short_shifts}")

    saved_config = load_checkpoint_config(args.checkpoint)
    report_config_drift(saved_config, args.checkpoint)

    device = resolve_device(args.device)
    print(
        f"device: {device}, segments per utterance: {args.n_segments}, "
        f"aggregation: {args.aggregation}"
    )

    config = build_run_config(
        saved_config,
        checkpoint=args.checkpoint,
        partition=PARTITION,
        device=device,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    scores_path = run_inference(
        config,
        device,
        args.save_dir,
        args.n_segments,
        args.aggregation,
        read_subset(args.subset),
        args.short_shifts,
    )

    if args.subset is not None:
        print(
            f"\n--subset was given: the predictions in {scores_path} cover a "
            "part of the protocol and are not a submission."
        )
        return 0

    return build_submission(scores_path, args.protocol, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
