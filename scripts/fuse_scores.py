"""
Fuse the scores of several runs into one submission csv.

Score-level fusion of systems that look at the signal differently (another
front-end, another seed) is the standard last step of this task and stays
inside the assignment: every member is still an LCNN, only their scores are
combined. See src/metrics/score_fusion.py for why the scores are normalized
before being averaged.

Usage:
    python3 scripts/fuse_scores.py \\
        data/saved/stft/eval_scores.csv data/saved/lfcc/eval_scores.csv \\
        -o data/saved/fused_scores.csv
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.make_submission import DEFAULT_PROTOCOL, validate_submission  # noqa: E402
from scripts.score_files import load_score_file, write_score_csv  # noqa: E402
from src.metrics.attack_eer import pooled_eer  # noqa: E402
from src.metrics.score_fusion import (  # noqa: E402
    DEFAULT_NORMALIZATION,
    NORMALIZATIONS,
    fuse_scores,
)
from src.utils.protocol import filter_entries, read_protocol_entries  # noqa: E402

DEFAULT_OUTPUT = "data/saved/fused_scores.csv"


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        args (Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Fuse the scores of several systems into one submission."
    )
    parser.add_argument(
        "scores",
        type=Path,
        nargs="+",
        help="score files to fuse: submission csv or '<part>_outputs.pth'",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"csv with the fused scores (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "-m",
        "--method",
        choices=NORMALIZATIONS,
        default=DEFAULT_NORMALIZATION,
        help=f"normalization applied to every system (default: {DEFAULT_NORMALIZATION})",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="weight of every system, equal weights by default",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        type=Path,
        default=Path(DEFAULT_PROTOCOL),
        help="protocol used to report the EER and to check the result",
    )
    return parser.parse_args()


def load_systems(paths: list[Path]) -> list[dict[str, float]]:
    """
    Read every score file given on the command line.

    Args:
        paths (list[Path]): score files.
    Returns:
        systems (list[dict[str, float]]): utt_id -> score of every system.
    """
    systems = []
    for path in paths:
        try:
            scores = load_score_file(path)
        except (OSError, ValueError) as e:
            raise SystemExit(f"Cannot read '{path}': {e}")
        print(f"{path}: {len(scores)} scores")
        systems.append(scores)
    return systems


def report_eers(
    systems: list[dict[str, float]],
    fused: dict[str, float],
    paths: list[Path],
    protocol_path: Path,
) -> None:
    """
    Print the EER of every system and of the fusion.

    Without those numbers the fusion is taken on faith: a member that is much
    weaker than the rest, or that scores with the opposite sign, makes the
    result worse than the best single system, and the only way to see it is to
    compare them side by side.

    Args:
        systems (list[dict[str, float]]): scores of every system.
        fused (dict[str, float]): fused scores.
        paths (list[Path]): score files, for the labels of the report.
        protocol_path (Path): protocol with the ground truth.
    """
    try:
        entries = read_protocol_entries(protocol_path)
    except (OSError, ValueError) as e:
        print(f"warning: the EER is not reported, cannot read the protocol: {e}")
        return

    scored = filter_entries(entries, set(fused))
    if not scored:
        print(
            f"warning: none of the {len(fused)} scored utterances is in "
            f"{protocol_path}, the EER is not reported"
        )
        return

    print(f"\nEER on {len(scored)} of {len(entries)} trials of the protocol:")
    for path, system in zip(paths, systems):
        eer = pooled_eer(system, scored)
        print(f"  {path}: {'n/a' if eer is None else f'{eer:.4f}%'}")

    fusion_eer = pooled_eer(fused, scored)
    print(f"  fusion: {'n/a' if fusion_eer is None else f'{fusion_eer:.4f}%'}")


def main() -> int:
    """
    Fuse the score files and write the result.

    Returns:
        exit_code (int): 0 if the fused csv is gradeable, 1 otherwise.
    """
    args = parse_args()

    systems = load_systems(args.scores)
    try:
        fused = fuse_scores(systems, args.weights, args.method)
    except ValueError as e:
        raise SystemExit(f"Cannot fuse the scores: {e}")

    weights = args.weights or [1.0] * len(systems)
    print(
        f"\nfusion of {len(systems)} systems, {args.method} normalization, weights {weights}"
    )
    report_eers(systems, fused, args.scores, args.protocol)

    try:
        write_score_csv(args.output, fused)
    except OSError as e:
        raise SystemExit(f"Cannot write '{args.output}': {e}")
    print(f"\nfused scores saved to {args.output.resolve()}")

    if not args.protocol.exists():
        print(f"warning: '{args.protocol}' is not there, the result is not checked")
        return 0

    print()
    return 0 if validate_submission(args.output, args.protocol) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
