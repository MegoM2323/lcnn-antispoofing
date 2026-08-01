from collections.abc import Mapping, Sequence

import numpy as np

from src.metrics.eer_utils import as_float_array

DEFAULT_NORMALIZATION = "rank"


def z_normalize(scores: np.ndarray) -> np.ndarray:
    scores = as_float_array(scores)
    return (scores - float(scores.mean())) / float(scores.std())


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    scores = as_float_array(scores)
    values, inverse = np.unique(scores, return_inverse=True)
    counts = np.bincount(inverse, minlength=values.size)
    first_rank = np.concatenate((np.zeros(1), np.cumsum(counts)[:-1]))
    average_rank = first_rank + (counts - 1) / 2.0

    return average_rank[inverse] / (scores.size - 1)


NORMALIZERS = {"rank": rank_normalize, "zscore": z_normalize}


def fuse_scores(
    systems: Sequence[Mapping[str, float]],
    weights: Sequence[float] | None = None,
    method: str = DEFAULT_NORMALIZATION,
) -> dict[str, float]:
    utt_ids = list(systems[0])

    weight_array = as_float_array([1.0] * len(systems) if weights is None else weights)
    total_weight = float(weight_array.sum())

    normalize = NORMALIZERS[method]
    fused = np.zeros(len(utt_ids), dtype=np.float64)
    for system, weight in zip(systems, weight_array):
        ordered = as_float_array([system[utt_id] for utt_id in utt_ids])
        fused += (weight / total_weight) * normalize(ordered)

    return {utt_id: float(score) for utt_id, score in zip(utt_ids, fused)}
