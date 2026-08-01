from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from src.metrics.eer_utils import compute_eer_percent
from src.utils.protocol import ProtocolEntry


class AttackStats(NamedTuple):
    attack_id: str
    n_trials: int
    eer: float


def ordered_scores(
    scores: Mapping[str, float], entries: Sequence[ProtocolEntry]
) -> tuple[np.ndarray, np.ndarray]:
    values = np.array([scores[entry.utt_id] for entry in entries], dtype=np.float64)
    labels = np.array([entry.label for entry in entries], dtype=np.float64)
    return values, labels


def attack_breakdown(
    scores: Mapping[str, float], entries: Sequence[ProtocolEntry]
) -> list[AttackStats]:
    bonafide = [entry for entry in entries if entry.label == 1]

    attacks: dict[str, list[ProtocolEntry]] = {}
    for entry in entries:
        if entry.label == 0:
            attacks.setdefault(entry.attack_id, []).append(entry)

    breakdown = []
    for attack_id in sorted(attacks):
        spoof = attacks[attack_id]
        values, labels = ordered_scores(scores, bonafide + spoof)
        breakdown.append(
            AttackStats(attack_id, len(spoof), compute_eer_percent(values, labels))
        )

    return breakdown
