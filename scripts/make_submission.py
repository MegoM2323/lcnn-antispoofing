"""
Validate the eval scores produced by 'inference.py' and prepare the submission
file for the course grading script.

The checks below mirror 'grading.py': the grader builds a dict from the csv,
skips malformed rows and then looks up *every* utterance of the protocol, so a
missing (or malformed) row raises a KeyError and zeroes the grade. Everything
that would break it is reported here instead.

Usage:
    python3 scripts/make_submission.py data/saved/eval/eval_scores.csv
"""

import argparse
import csv
import math
import os
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().resolve().parent.parent))

from src.metrics.eer_utils import compute_eer_percent  # noqa: E402

DEFAULT_PROTOCOL = os.environ.get(
    "ASVSPOOF_EVAL_PROTOCOL",
    "/home/mego/data/Homework/Practice/repo/hw/ASVspoof2019.LA.cm.eval.trl.txt",
)
DEFAULT_SUBMISSION_NAME = "mppanin.csv"  # must match the university login

# grading thresholds from the homework description (EER in %, 0-100 scale)
EER_ZERO_GRADE = 10.9
EER_FULL_GRADE = 5.3
MAX_GRADE = 10.0
MIN_LINEAR_GRADE = 2.0


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        args (Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Check the eval scores and prepare the submission file."
    )
    parser.add_argument(
        "scores",
        type=Path,
        help="csv with the model scores (no header, 'utterance_id,score')",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        type=Path,
        default=Path(DEFAULT_PROTOCOL),
        help="ASVspoof2019 LA eval protocol (default: $ASVSPOOF_EVAL_PROTOCOL)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_SUBMISSION_NAME),
        help=f"path of the submission file (default: {DEFAULT_SUBMISSION_NAME})",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="only run the checks, do not write the submission file",
    )
    return parser.parse_args()


def read_protocol(protocol_path: Path) -> list[tuple[str, int]]:
    """
    Read the evaluation protocol.

    Args:
        protocol_path (Path): path to the ASVspoof2019 LA eval protocol.
    Returns:
        trials (list[tuple[str, int]]): (utterance_id, label) pairs in the
            order they appear in the protocol. The label is 1 for bonafide
            and 0 for spoof.
    """
    trials = []
    try:
        with protocol_path.open("r") as protocol:
            for line_number, line in enumerate(protocol, start=1):
                fields = line.strip().split()
                if len(fields) != 5:
                    raise ValueError(
                        f"{protocol_path}:{line_number}: expected 5 fields, "
                        f"got {len(fields)}"
                    )
                _, key, _, _, label = fields
                trials.append((key, 1 if label == "bonafide" else 0))
    except OSError as e:
        raise SystemExit(f"Cannot read the protocol '{protocol_path}': {e}")
    return trials


def read_scores(scores_path: Path) -> tuple[dict[str, float], list[str]]:
    """
    Read the csv with the model scores, exactly as strictly as the grading
    script reads it (and a bit stricter: malformed rows are reported instead
    of being silently skipped).

    Args:
        scores_path (Path): path to the csv with the scores.
    Returns:
        scores (dict[str, float]): utterance_id -> score.
        errors (list[str]): human-readable descriptions of the problems found.
    """
    scores: dict[str, float] = {}
    errors: list[str] = []
    try:
        with scores_path.open("r", newline="") as file:
            for line_number, row in enumerate(csv.reader(file), start=1):
                if len(row) != 2:
                    errors.append(
                        f"line {line_number}: expected 2 columns, got {len(row)}"
                    )
                    continue

                key, raw_score = row
                key = key.strip()
                try:
                    score = float(raw_score)
                except ValueError:
                    errors.append(f"line {line_number}: '{raw_score}' is not a float")
                    continue

                if not math.isfinite(score):
                    errors.append(f"line {line_number}: score is {score}")
                    continue
                if key in scores:
                    errors.append(f"line {line_number}: duplicated id '{key}'")
                    continue

                scores[key] = score
    except OSError as e:
        raise SystemExit(f"Cannot read the scores '{scores_path}': {e}")

    return scores, errors


def compute_grade(eer: float) -> float:
    """
    Compute the expected performance grade for a given EER.

    Args:
        eer (float): equal error rate in percents (0-100).
    Returns:
        grade (float): grade in [0, 10].
    """
    if eer > EER_ZERO_GRADE:
        return 0.0
    if eer < EER_FULL_GRADE:
        return MAX_GRADE
    return MIN_LINEAR_GRADE + (EER_ZERO_GRADE - eer) * (
        (MAX_GRADE - MIN_LINEAR_GRADE) / (EER_ZERO_GRADE - EER_FULL_GRADE)
    )


def main() -> int:
    """
    Run the checks and write the submission file.

    Returns:
        exit_code (int): 0 if the submission is valid, 1 otherwise.
    """
    args = parse_args()

    trials = read_protocol(args.protocol)
    scores, errors = read_scores(args.scores)

    print(f"protocol: {len(trials)} trials from {args.protocol}")
    print(f"scores:   {len(scores)} unique ids from {args.scores}")

    missing = [key for key, _ in trials if key not in scores]
    if missing:
        errors.append(
            f"{len(missing)} protocol ids are missing from the csv, "
            f"e.g. {missing[:5]}"
        )

    protocol_keys = {key for key, _ in trials}
    extra = [key for key in scores if key not in protocol_keys]
    if extra:
        # the grader ignores them, but they usually mean a wrong partition
        print(f"warning: {len(extra)} ids are not in the protocol, e.g. {extra[:5]}")

    if errors:
        print("\nThe submission is INVALID:")
        for error in errors[:20]:
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more problems")
        return 1

    # the same computation as in grading.py: the official compute_eer over the
    # scores ordered by the protocol
    ordered_scores = [scores[key] for key, _ in trials]
    labels = [label for _, label in trials]
    eer = compute_eer_percent(ordered_scores, labels)
    grade = compute_grade(eer)

    bonafide_count = sum(labels)
    print(
        f"\nbonafide trials: {bonafide_count}, spoof trials: {len(labels) - bonafide_count}"
    )
    print(f"EER: {eer:.4f}%")
    print(f"expected performance grade: {grade:.2f} / 10")

    if not args.no_copy:
        try:
            shutil.copyfile(args.scores, args.output)
        except OSError as e:
            print(f"Cannot write the submission file '{args.output}': {e}")
            return 1
        print(f"submission saved to {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
