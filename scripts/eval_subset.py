"""
Score an existing set of scores on a subset of the protocol and break the
result down by spoofing algorithm.

A full eval pass over 71237 utterances takes minutes, which is too slow for
comparing checkpoints one against another; scoring a fixed subset of a few
thousand trials answers the same question much faster. The per-attack table is
the other half of the tool: the pooled EER of this system is made almost
entirely of one attack, and only the breakdown shows it.

Usage:
    python3 scripts/eval_subset.py data/saved/eval/eval_scores.csv
    python3 scripts/eval_subset.py data/saved/eval/eval_scores.csv -s subset.txt
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.make_submission import DEFAULT_PROTOCOL  # noqa: E402
from scripts.score_files import load_score_file  # noqa: E402
from src.metrics.attack_eer import attack_breakdown, pooled_eer  # noqa: E402
from src.utils.protocol import (  # noqa: E402
    filter_entries,
    read_protocol_entries,
    read_utt_ids,
)

SORT_KEYS = ("eer", "attack")


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        args (Namespace): parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="EER and per-attack breakdown on a subset of the protocol."
    )
    parser.add_argument(
        "scores",
        type=Path,
        help="scores to evaluate: submission csv or '<part>_outputs.pth'",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=Path,
        default=None,
        help="file with the utterance ids to score, one per line "
        "(default: every trial of the protocol that has a score)",
    )
    parser.add_argument(
        "-p",
        "--protocol",
        type=Path,
        default=Path(DEFAULT_PROTOCOL),
        help="protocol with the ground truth labels and attack ids",
    )
    parser.add_argument(
        "--sort",
        choices=SORT_KEYS,
        default="eer",
        help="order of the per-attack table (default: worst attack first)",
    )
    return parser.parse_args()


def select_entries(
    protocol_path: Path, subset_path: Path | None, scored_ids: set[str]
) -> list:
    """
    Choose the trials to evaluate.

    Args:
        protocol_path (Path): protocol with the ground truth.
        subset_path (Path | None): file with the ids to keep, None keeps every
            trial that has a score.
        scored_ids (set[str]): ids present in the score file.
    Returns:
        entries (list[ProtocolEntry]): trials to evaluate.
    """
    try:
        entries = read_protocol_entries(protocol_path)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Cannot read the protocol '{protocol_path}': {e}")

    if subset_path is None:
        selected = filter_entries(entries, scored_ids)
    else:
        try:
            requested = read_utt_ids(subset_path)
        except OSError as e:
            raise SystemExit(f"Cannot read the subset '{subset_path}': {e}")

        selected = filter_entries(entries, set(requested))
        unknown = len(set(requested)) - len(selected)
        if unknown > 0:
            print(f"warning: {unknown} ids of '{subset_path}' are not in the protocol")

        without_scores = [
            entry.utt_id for entry in selected if entry.utt_id not in scored_ids
        ]
        if without_scores:
            raise SystemExit(
                f"{len(without_scores)} requested utterances have no score, "
                f"e.g. {without_scores[:5]}"
            )

    print(f"protocol: {len(entries)} trials from {protocol_path}")
    if not selected:
        raise SystemExit("None of the scored utterances is in the protocol")
    return selected


def print_breakdown(scores: dict[str, float], entries: list, sort_key: str) -> None:
    """
    Print the EER of every spoofing algorithm.

    Args:
        scores (dict[str, float]): utt_id -> score.
        entries (list[ProtocolEntry]): trials to evaluate.
        sort_key (str): "eer" for the worst attack first, "attack" for the
            order of the attack ids.
    """
    breakdown = attack_breakdown(scores, entries)
    if not breakdown:
        print("\nno bonafide trials in the subset, the breakdown is undefined")
        return

    if sort_key == "eer":
        breakdown = sorted(breakdown, key=lambda stats: -stats.eer)

    print(f"\n{'attack':<8}{'trials':>8}{'EER':>12}")
    for stats in breakdown:
        print(f"{stats.attack_id:<8}{stats.n_trials:>8}{stats.eer:>11.4f}%")


def main() -> int:
    """
    Evaluate the scores on the requested subset.

    Returns:
        exit_code (int): 0 if the EER could be computed, 1 otherwise.
    """
    args = parse_args()

    try:
        scores = load_score_file(args.scores)
    except (OSError, ValueError) as e:
        raise SystemExit(f"Cannot read '{args.scores}': {e}")

    entries = select_entries(args.protocol, args.subset, set(scores))
    bonafide = sum(entry.label for entry in entries)
    print(
        f"subset:   {len(entries)} trials "
        f"({bonafide} bonafide, {len(entries) - bonafide} spoof)"
    )

    eer = pooled_eer(scores, entries)
    if eer is None:
        print("\nEER: n/a, one of the two classes is missing from the subset")
        return 1
    print(f"\nEER: {eer:.4f}%")

    print_breakdown(scores, entries, args.sort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
