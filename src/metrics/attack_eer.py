"""
Breakdown of the EER by spoofing algorithm.

A pooled EER hides where the system actually fails: on ASVspoof2019 LA the
error is concentrated in a couple of eval attacks (A17 in particular, a
waveform-filtering voice conversion whose artifacts differ from everything in
the train partition), while the remaining ones are solved almost perfectly.
The per-attack numbers follow the convention of the official evaluation plan:
every attack is scored against the whole bonafide pool of the partition, so the
numbers are comparable with the published ones.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from src.metrics.eer_utils import compute_eer_percent
from src.utils.protocol import ProtocolEntry


class AttackStats(NamedTuple):
    """
    EER of a single spoofing algorithm.

    Attributes:
        attack_id (str): spoofing algorithm, e.g. "A17".
        n_trials (int): number of spoof trials of this attack.
        eer (float): EER in percents of this attack against the bonafide pool.
    """

    attack_id: str
    n_trials: int
    eer: float


def ordered_scores(
    scores: Mapping[str, float], entries: Sequence[ProtocolEntry]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Lay the scores out in the order of the protocol, together with the
    matching labels (1 = bonafide).
    """
    missing = [entry.utt_id for entry in entries if entry.utt_id not in scores]
    if missing:
        raise ValueError(
            f"{len(missing)} of {len(entries)} trials have no score, "
            f"e.g. {missing[:5]}"
        )

    values = np.array([scores[entry.utt_id] for entry in entries], dtype=np.float64)
    labels = np.array([entry.label for entry in entries], dtype=np.float64)
    return values, labels


def pooled_eer(
    scores: Mapping[str, float], entries: Sequence[ProtocolEntry]
) -> float | None:
    """
    EER over a set of trials, None if one of the two classes is not
    represented and the metric is undefined.
    """
    values, labels = ordered_scores(scores, entries)
    bonafide_count = int(labels.sum())
    if labels.size == 0 or bonafide_count in (0, labels.size):
        return None
    return compute_eer_percent(values, labels)


def attack_breakdown(
    scores: Mapping[str, float], entries: Sequence[ProtocolEntry]
) -> list[AttackStats]:
    """
    Compute the EER of every spoofing algorithm against the bonafide pool,
    one AttackStats per attack, sorted by attack id. Empty if the trials carry
    no bonafide utterance, since without them no attack has an EER.
    """
    bonafide = [entry for entry in entries if entry.label == 1]
    if not bonafide:
        return []

    attacks: dict[str, list[ProtocolEntry]] = {}
    for entry in entries:
        if entry.label == 0:
            attacks.setdefault(entry.attack_id, []).append(entry)

    breakdown = []
    for attack_id in sorted(attacks):
        spoof = attacks[attack_id]
        # both classes are in the pool by construction, so the EER is defined
        values, labels = ordered_scores(scores, bonafide + spoof)
        breakdown.append(
            AttackStats(attack_id, len(spoof), compute_eer_percent(values, labels))
        )

    return breakdown
