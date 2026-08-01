import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.make_submission import (  # noqa: E402
    DEFAULT_PROTOCOL,
    load_score_file,
    validate_submission,
    write_score_csv,
)
from src.metrics.score_fusion import fuse_scores  # noqa: E402

DEFAULT_OUTPUT = "data/saved/fused_scores.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse the scores of several systems into one submission."
    )
    parser.add_argument(
        "scores", type=Path, nargs="+", help="submission csv files to fuse"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_OUTPUT),
        help=f"csv with the fused scores (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    systems = [load_score_file(path) for path in args.scores]
    write_score_csv(args.output, fuse_scores(systems))

    if validate_submission(args.output, DEFAULT_PROTOCOL) is None:
        return 1

    print(f"fusion of {len(systems)} systems saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
