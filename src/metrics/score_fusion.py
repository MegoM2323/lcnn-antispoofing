"""
Score-level fusion of several countermeasure systems.

Fusing systems that see the input differently (another front-end, another seed)
is the standard way of squeezing the last points out of this task: the STC
system of arXiv:1904.05576 reports 1.84% EER for the fusion against 4.53% for
its best single model. Only the scores are combined, so the architecture of
every member stays the LCNN required by the assignment.

The scores of two runs are not comparable as they are: the LLR of one model may
live in [-30, 20] and the LLR of another in [-4, 6], so a plain average would
let the model with the widest range decide alone. Every set is therefore
normalized before the weighted average. Both normalizations are monotone, hence
neither changes the EER of the single system it is applied to.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from src.metrics.eer_utils import as_float_array

NORMALIZATIONS = ("rank", "zscore")
DEFAULT_NORMALIZATION = "rank"


def z_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Center the scores and bring them to a unit standard deviation.

    Args:
        scores (np.ndarray): 1D array of detection scores.
    Returns:
        normalized (np.ndarray): 1D array with zero mean and unit std. A
            constant set of scores carries no information and becomes zeros.
    """
    scores = as_float_array(scores)
    std = float(scores.std())
    if std == 0.0:
        return np.zeros_like(scores)
    return (scores - float(scores.mean())) / std


def rank_normalize(scores: np.ndarray) -> np.ndarray:
    """
    Replace the scores by their ranks, scaled to [0, 1].

    Ranks ignore the shape of the score distribution, so a single system with a
    heavy tail (a handful of utterances scored -60 while the rest sits around
    zero) cannot drag the fusion after itself the way z-normalization lets it.
    Tied scores share the average rank, otherwise the order inside a tie would
    be decided by the order of the file.

    Args:
        scores (np.ndarray): 1D array of detection scores.
    Returns:
        normalized (np.ndarray): 1D array of ranks in [0, 1].
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
    """
    Apply the requested normalization.

    Args:
        scores (np.ndarray): 1D array of detection scores.
        method (str): "rank" or "zscore".
    Returns:
        normalized (np.ndarray): normalized scores.
    """
    if method == "rank":
        return rank_normalize(scores)
    if method == "zscore":
        return z_normalize(scores)
    raise ValueError(f"Unknown normalization '{method}', expected {NORMALIZATIONS}")


def check_same_keys(systems: Sequence[Mapping[str, float]]) -> list[str]:
    """
    Check that every system scored exactly the same utterances.

    A fusion built on a partial intersection would silently produce a csv with
    a hole in it, and a hole is a KeyError in the grading script.

    Args:
        systems (Sequence[Mapping]): utt_id -> score of every system.
    Returns:
        utt_ids (list[str]): ids in the order of the first system.
    """
    if not systems:
        raise ValueError("Nothing to fuse: no scores were given")

    reference = list(systems[0])
    reference_set = set(reference)
    for position, system in enumerate(systems[1:], start=2):
        keys = set(system)
        if keys == reference_set:
            continue
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
    Combine the scores of several systems into one set.

    Args:
        systems (Sequence[Mapping]): utt_id -> score of every system.
        weights (Sequence[float] | None): weight of every system, equal
            weights by default. They are normalized to sum to one, so only
            their proportion matters.
        method (str): normalization applied to every system, see
            NORMALIZATIONS.
    Returns:
        fused (dict[str, float]): utt_id -> fused score, in the order of the
            first system.
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
    if np.any(weight_array < 0):
        raise ValueError(f"Weights must be non-negative, got {list(weights)}")
    total_weight = float(weight_array.sum())
    if total_weight == 0.0:
        raise ValueError("The weights sum to zero, the fusion is undefined")

    fused = np.zeros(len(utt_ids), dtype=np.float64)
    for system, weight in zip(systems, weight_array):
        ordered = as_float_array([system[utt_id] for utt_id in utt_ids])
        fused += (weight / total_weight) * normalize_scores(ordered, method)

    return {utt_id: float(score) for utt_id, score in zip(utt_ids, fused)}
