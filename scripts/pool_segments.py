"""
Pool the per-segment scores of a multi-crop run into utterance scores.

'scripts/predict_eval_tta.py' writes '<part>_segments.pth' with the score of
every single segment next to its submission csv. Comparing the pooling rules
one against another therefore costs no GPU at all: the segments are scored
once, and every rule is a reduction of the numbers in that dump.

Usage:
    python3 scripts/pool_segments.py data/saved/eval/eval_segments.pth
    python3 scripts/pool_segments.py eval_segments.pth -a min -o min.csv
"""

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.make_submission import DEFAULT_PROTOCOL  # noqa: E402
from scripts.score_files import write_score_csv  # noqa: E402
from src.metrics.attack_eer import pooled_eer  # noqa: E402
from src.metrics.segment_pooling import (  # noqa: E402
    AGGREGATIONS,
    aggregate_segment_scores,
)
from src.utils.protocol import filter_entries, read_protocol_entries  # noqa: E402


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        args (Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pool the segment scores of a multi-crop run and report "
        "the EER of every pooling rule."
    )
    parser.add_argument(
        "segments", type=Path, help="'<part>_segments.pth' of a multi-crop run"
    )
    parser.add_argument(
        "-a",
        "--aggregation",
        choices=("all",) + AGGREGATIONS,
        default="all",
        help="pooling rule to apply, 'all' compares every rule (default: all)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="csv with the pooled scores, written only for a single rule",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        type=Path,
        default=Path(DEFAULT_PROTOCOL),
        help="protocol used to report the EER",
    )
    return parser.parse_args()


def load_segments(path: Path) -> tuple[list[str], torch.Tensor, list[int]]:
    """
    Read the dump written by the multi-crop inferencer.

    Args:
        path (Path): '<part>_segments.pth' of a multi-crop run.
    Returns:
        utt_ids (list[str]): utterance ids, one per utterance.
        scores (Tensor): 1D tensor with the score of every segment.
        sizes (list[int]): number of segments of every utterance.
    """
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as e:
        raise SystemExit(f"Cannot read '{path}': {e}")

    try:
        utt_ids = list(payload["utt_id"])
        scores = payload["segment_scores"].reshape(-1)
        sizes = [int(size) for size in payload["segment_sizes"]]
    except (KeyError, TypeError, AttributeError) as e:
        raise SystemExit(f"'{path}' is not a segment dump: {e}")

    if len(utt_ids) != len(sizes):
        raise SystemExit(
            f"'{path}' holds {len(utt_ids)} utterance ids for {len(sizes)} "
            "segment counts"
        )

    print(
        f"{path}: {len(utt_ids)} utterances, {scores.numel()} segments "
        f"({scores.numel() / max(len(utt_ids), 1):.2f} per utterance)"
    )
    return utt_ids, scores, sizes


def main() -> int:
    """
    Pool the segment scores and report the EER.

    Returns:
        exit_code (int): 0 on success.
    """
    args = parse_args()
    utt_ids, scores, sizes = load_segments(args.segments)

    rules = AGGREGATIONS if args.aggregation == "all" else (args.aggregation,)
    if args.output is not None and len(rules) > 1:
        raise SystemExit("--output needs a single --aggregation, not 'all'")

    try:
        entries = read_protocol_entries(args.protocol)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Cannot read the protocol '{args.protocol}': {e}")
    scored = filter_entries(entries, set(utt_ids))
    print(f"trials: {len(scored)} of {len(entries)} of the protocol\n")

    print(f"{'aggregation':<14}{'EER':>10}")
    for rule in rules:
        try:
            pooled = aggregate_segment_scores(scores, sizes, rule)
        except ValueError as e:
            raise SystemExit(f"Cannot pool the segment scores: {e}")

        by_id = {utt_id: float(score) for utt_id, score in zip(utt_ids, pooled)}
        eer = pooled_eer(by_id, scored) if scored else None
        print(f"{rule:<14}{'n/a' if eer is None else f'{eer:>9.4f}%'}")

        if args.output is not None:
            try:
                write_score_csv(args.output, by_id)
            except OSError as e:
                raise SystemExit(f"Cannot write '{args.output}': {e}")
            print(f"\npooled scores saved to {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
