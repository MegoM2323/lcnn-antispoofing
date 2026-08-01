"""
Разбивка EER по алгоритмам атак.

Общий EER скрывает, где система на самом деле ошибается: на ASVspoof2019 LA
ошибка сосредоточена в паре эвалюационных атак (прежде всего A17, преобразование
голоса с фильтрацией формы волны, чьи артефакты не похожи ни на что из
train-партиции), тогда как остальные решаются почти идеально. Числа по атакам
считаются по соглашению официального плана оценки: каждая атака сравнивается
со всем пулом bonafide своей партиции, поэтому они сопоставимы с
опубликованными.
"""

from collections.abc import Mapping, Sequence
from typing import NamedTuple

import numpy as np

from src.metrics.eer_utils import compute_eer_percent
from src.utils.protocol import ProtocolEntry


class AttackStats(NamedTuple):
    """
    EER одного алгоритма атаки.

    Поля:
        attack_id (str): алгоритм атаки, например "A17".
        n_trials (int): число поддельных испытаний этой атаки.
        eer (float): EER этой атаки против пула bonafide, в процентах.
    """

    attack_id: str
    n_trials: int
    eer: float


def ordered_scores(
    scores: Mapping[str, float], entries: Sequence[ProtocolEntry]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Раскладывает скоры в порядке протокола вместе с отвечающими им метками
    (1 = bonafide).
    """
    values = np.array([scores[entry.utt_id] for entry in entries], dtype=np.float64)
    labels = np.array([entry.label for entry in entries], dtype=np.float64)
    return values, labels


def attack_breakdown(
    scores: Mapping[str, float], entries: Sequence[ProtocolEntry]
) -> list[AttackStats]:
    """
    Считает EER каждого алгоритма атаки против пула bonafide и возвращает по
    одному AttackStats на атаку, отсортированных по идентификатору атаки.
    """
    bonafide = [entry for entry in entries if entry.label == 1]

    attacks: dict[str, list[ProtocolEntry]] = {}
    for entry in entries:
        if entry.label == 0:
            attacks.setdefault(entry.attack_id, []).append(entry)

    breakdown = []
    for attack_id in sorted(attacks):
        spoof = attacks[attack_id]
        # оба класса в пуле по построению, поэтому EER определён
        values, labels = ordered_scores(scores, bonafide + spoof)
        breakdown.append(
            AttackStats(attack_id, len(spoof), compute_eer_percent(values, labels))
        )

    return breakdown
