"""
Read, check and write the csv with the eval scores.

The checks mirror 'grading.py': the grader builds a dict from the csv and then
looks up *every* utterance of the protocol, so a missing or malformed row is a
KeyError there and a zero for the homework. The reading and writing helpers are
shared with the other two scripts.

    python3 scripts/make_submission.py data/saved/lfcc21/eval_scores.csv -o mppanin.csv
"""

import argparse
import csv
import math
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.metrics.attack_eer import attack_breakdown  # noqa: E402
from src.metrics.eer_utils import compute_eer_percent  # noqa: E402
from src.utils.protocol import read_protocol_entries  # noqa: E402

# the protocol of the LA corpus, laid out as described in the README
EVAL_PROTOCOL = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
DEFAULT_PROTOCOL = os.environ.get(
    "ASVSPOOF_EVAL_PROTOCOL", str(PROJECT_ROOT / "data" / "LA" / EVAL_PROTOCOL)
)
DEFAULT_SUBMISSION_NAME = "mppanin.csv"  # must match the university login

# grading thresholds from the homework description (EER in %, 0-100 scale)
EER_ZERO_GRADE = 10.9
EER_FULL_GRADE = 5.3
MAX_GRADE = 10.0
MIN_LINEAR_GRADE = 2.0


def read_scores(scores_path: str | Path) -> tuple[dict[str, float], list[str]]:
    """
    Read the csv with the model scores and return them together with the
    problems found. Empty lines are ignored, the grader ignores them too.
    """
    scores: dict[str, float] = {}
    errors: list[str] = []
    with Path(scores_path).open("r", newline="") as file:
        for line, row in enumerate(csv.reader(file), start=1):
            if not row:
                continue
            if len(row) != 2:
                errors.append(f"line {line}: {len(row)} columns instead of 2")
                continue

            key, raw_score = row
            if key != key.strip():
                # the grader looks the id up as is, so a padded id is a
                # KeyError there: it must not be silently repaired here
                errors.append(f"line {line}: id '{key}' is padded")
                continue

            try:
                score = float(raw_score)
            except ValueError:
                errors.append(f"line {line}: '{raw_score}' is not a float")
                continue

            if not math.isfinite(score):
                errors.append(f"line {line}: score is {score}")
            elif key in scores:
                errors.append(f"line {line}: duplicated id '{key}'")
            else:
                scores[key] = score

    return scores, errors


def load_score_file(path: str | Path) -> dict[str, float]:
    """Read a set of scores, refusing a malformed file when it is loaded."""
    scores, errors = read_scores(path)
    if errors:
        listed = "\n".join(f"  - {error}" for error in errors[:10])
        raise ValueError(f"'{path}' is malformed:\n{listed}")
    if not scores:
        raise ValueError(f"'{path}' holds no scores")
    return scores


def write_score_csv(path: str | Path, scores: Mapping[str, float]) -> None:
    """
    Write the scores in the submission format. 'repr' is used instead of a
    fixed format: rounding a score creates ties between utterances that the
    model separated, and ties move the EER.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        for utt_id, score in scores.items():
            writer.writerow([utt_id, repr(float(score))])


def compute_grade(eer: float) -> float:
    """Expected performance grade in [0, 10] for a given EER in percents."""
    if eer > EER_ZERO_GRADE:
        return 0.0
    if eer < EER_FULL_GRADE:
        return MAX_GRADE
    return MIN_LINEAR_GRADE + (EER_ZERO_GRADE - eer) * (
        (MAX_GRADE - MIN_LINEAR_GRADE) / (EER_ZERO_GRADE - EER_FULL_GRADE)
    )


def validate_submission(
    scores_path: str | Path, protocol_path: str | Path
) -> float | None:
    """
    Run every check the grading script would trip over and report the EER, the
    expected grade and the EER of every spoofing algorithm (each of them scored
    against the whole bonafide pool, as in the official evaluation plan).
    Returns None if the file is not gradeable; the problems are printed.
    """
    try:
        entries = read_protocol_entries(protocol_path)
        scores, errors = read_scores(scores_path)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Cannot read the protocol or the scores: {e}")

    missing = [entry.utt_id for entry in entries if entry.utt_id not in scores]
    if missing:
        errors.append(f"{len(missing)} protocol ids are missing, e.g. {missing[:5]}")

    if errors:
        print(f"\n{scores_path} is INVALID:")
        for error in errors[:20]:
            print(f"  - {error}")
        return None

    # the same computation as in grading.py: the official compute_eer over the
    # scores ordered by the protocol
    labels = [entry.label for entry in entries]
    eer = compute_eer_percent([scores[entry.utt_id] for entry in entries], labels)

    print(f"\n{len(entries)} trials, {sum(labels)} of them bonafide")
    print(f"EER: {eer:.4f}%")
    print(f"expected performance grade: {compute_grade(eer):.2f} / 10")

    print(f"\n{'attack':<8}{'trials':>8}{'EER':>12}")
    for stats in sorted(attack_breakdown(scores, entries), key=lambda st: -st.eer):
        print(f"{stats.attack_id:<8}{stats.n_trials:>8}{stats.eer:>11.4f}%")

    return eer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the eval scores and prepare the submission file."
    )
    parser.add_argument("scores", type=Path, help="csv with the model scores")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_SUBMISSION_NAME),
        help=f"path of the submission file (default: {DEFAULT_SUBMISSION_NAME})",
    )
    args = parser.parse_args()

    if validate_submission(args.scores, DEFAULT_PROTOCOL) is None:
        return 1

    try:
        shutil.copyfile(args.scores, args.output)
    except OSError as e:
        print(f"Cannot write the submission file '{args.output}': {e}")
        return 1

    print(f"\nsubmission saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
