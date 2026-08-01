"""
Объединение нескольких контрмер на уровне скоров.

Объединение систем, которые видят вход по-разному (другой фронт-энд, другой
seed), это стандартный способ выжать из задачи последние доли процента: система
STC из arXiv:1904.05576 даёт 1,84 % EER для объединения против 4,53 % для своей
лучшей одиночной модели. Складываются только скоры, поэтому архитектура каждого
участника остаётся тем самым LCNN, которого требует задание.

Скоры двух прогонов несравнимы как есть: LLR одной модели может лежать
в [-30, 20], а LLR другой в [-4, 6], и при обычном усреднении решала бы одна
модель с самым широким диапазоном. Поэтому перед взвешенным усреднением каждый
набор нормируется. Обе нормировки монотонны, так что EER отдельной системы,
к которой их применили, ни одна из них не меняет.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from src.metrics.eer_utils import as_float_array

NORMALIZATIONS = ("rank", "zscore")
DEFAULT_NORMALIZATION = "rank"


def z_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Центрирует скоры и приводит их к единичному стандартному отклонению.
    Постоянный набор скоров не несёт информации и обращается в нули.
    """
    scores = as_float_array(scores)
    std = float(scores.std())
    if std == 0.0:
        return np.zeros_like(scores)
    return (scores - float(scores.mean())) / std


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Заменяет скоры их рангами, отмасштабированными в [0, 1].

    Ранги не зависят от формы распределения скоров, поэтому одна система
    с тяжёлым хвостом (горстка записей со скором -60 при остальных около нуля)
    не может утянуть объединение за собой так, как это позволяет z-нормировка.
    Одинаковые скоры получают средний ранг, иначе порядок внутри такой группы
    определялся бы порядком строк в файле.
    """
    scores = as_float_array(scores)
    if scores.size < 2:
        return np.zeros_like(scores)

    values, inverse = np.unique(scores, return_inverse=True)
    counts = np.bincount(inverse, minlength=values.size)
    first_rank = np.concatenate((np.zeros(1), np.cumsum(counts)[:-1]))
    average_rank = first_rank + (counts - 1) / 2.0

    return average_rank[inverse] / (scores.size - 1)


def normalize_scores(
    scores: np.ndarray, method: str = DEFAULT_NORMALIZATION
) -> np.ndarray:
    if method == "rank":
        return rank_normalize(scores)
    if method == "zscore":
        return z_normalize(scores)
    raise ValueError(f"Unknown normalization '{method}', expected {NORMALIZATIONS}")


def check_same_keys(systems: Sequence[Mapping[str, float]]) -> list[str]:
    """
    Проверяет, что все системы оценили ровно один и тот же набор записей, и
    возвращает идентификаторы в порядке первой системы.

    Объединение по частичному пересечению молча дало бы csv с дырой, а дыра это
    KeyError в проверяющем скрипте.
    """
    reference = list(systems[0])
    reference_set = set(reference)
    for position, system in enumerate(systems[1:], start=2):
        keys = set(system)
        if keys != reference_set:
            missing = sorted(reference_set - keys)
            extra = sorted(keys - reference_set)
            raise ValueError(
                f"System {position} scores another set of utterances: "
                f"{len(missing)} ids are missing (e.g. {missing[:3]}), "
                f"{len(extra)} are new (e.g. {extra[:3]})"
            )

    return reference


def fuse_scores(
    systems: Sequence[Mapping[str, float]],
    weights: Sequence[float] | None = None,
    method: str = DEFAULT_NORMALIZATION,
) -> dict[str, float]:
    """
    Сводит скоры нескольких систем в один набор, в порядке первой системы.
    Веса нормируются к единичной сумме, так что важна только их пропорция;
    по умолчанию веса равные.
    """
    utt_ids = check_same_keys(systems)

    if weights is None:
        weights = [1.0] * len(systems)
    if len(weights) != len(systems):
        raise ValueError(
            f"Got {len(weights)} weights for {len(systems)} systems: "
            "every system needs exactly one weight"
        )

    weight_array = as_float_array(weights)
    total_weight = float(weight_array.sum())
    if np.any(weight_array < 0) or total_weight == 0.0:
        raise ValueError(f"Weights must be non-negative and not all zero: {weights}")

    fused = np.zeros(len(utt_ids), dtype=np.float64)
    for system, weight in zip(systems, weight_array):
        ordered = as_float_array([system[utt_id] for utt_id in utt_ids])
        fused += (weight / total_weight) * normalize_scores(ordered, method)

    return {utt_id: float(score) for utt_id, score in zip(utt_ids, fused)}
