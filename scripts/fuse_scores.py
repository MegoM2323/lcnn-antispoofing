"""
Объединение скоров нескольких прогонов в один csv посылки.

Объединение на уровне скоров систем, которые смотрят на сигнал по-разному
(другой фронт-энд, другая эпоха), не выходит за рамки задания: каждый участник
по-прежнему LCNN, складываются только их скоры. Почему скоры нормируются перед
усреднением, написано в src/metrics/score_fusion.py.

    python3 scripts/fuse_scores.py data/saved/{lfcc21,stft15}/eval_scores.csv \\
        -o mppanin.csv
"""

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

    systems = []
    for path in args.scores:
        try:
            systems.append(load_score_file(path))
        except (OSError, ValueError) as e:
            raise SystemExit(f"Cannot read '{path}': {e}")

    try:
        fused = fuse_scores(systems)
    except ValueError as e:
        raise SystemExit(f"Cannot fuse the scores: {e}")

    try:
        write_score_csv(args.output, fused)
    except OSError as e:
        raise SystemExit(f"Cannot write '{args.output}': {e}")
    print(f"fusion of {len(systems)} systems saved to {args.output.resolve()}")

    return 0 if validate_submission(args.output, DEFAULT_PROTOCOL) is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
