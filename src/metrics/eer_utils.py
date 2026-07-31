"""
Reference implementation of the Equal Error Rate (EER) computation.

``compute_det_curve`` and ``compute_eer`` are taken verbatim (logic-wise) from
the official ASVspoof2019 evaluation package (``calculate_eer.py`` shipped with
the course). They are kept unchanged on purpose: the final submission is graded
with exactly these functions, so any deviation would make local numbers
incomparable with the leaderboard ones.

Only cosmetic changes were applied (formatting + type hints). The rest of this
module contains thin numpy-2.x-safe wrappers used by ``src.metrics.eer``.
"""

import numpy as np

DEGENERATE_EER = 50.0
"""EER (in %) reported when one of the two classes is missing from the buffer.

A chance-level value is used instead of ``nan`` because ``MetricTracker``
aggregates metrics with pandas, and a single ``nan`` would poison the running
average of the whole epoch.
"""


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Detection Error Tradeoff (DET) curve.

    Args:
        target_scores (np.ndarray): scores of the target (bonafide) trials.
        nontarget_scores (np.ndarray): scores of the nontarget (spoof) trials.
    Returns:
        frr (np.ndarray): false rejection rates.
        far (np.ndarray): false acceptance rates.
        thresholds (np.ndarray): thresholds corresponding to frr/far.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # false rejection rates
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # false acceptance rates
    # Thresholds are the sorted scores
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(
    bonafide_scores: np.ndarray, other_scores: np.ndarray
) -> tuple[float, float]:
    """
    Returns equal error rate (EER) and the corresponding threshold.

    Note that the EER is returned as a fraction in [0, 1], not in percents.

    Args:
        bonafide_scores (np.ndarray): scores of the bonafide trials.
        other_scores (np.ndarray): scores of the spoofed trials.
    Returns:
        eer (float): equal error rate in [0, 1].
        threshold (float): threshold at which frr and far are the closest.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def as_float_array(values) -> np.ndarray:
    """
    Convert an arbitrary sequence of scores/labels into a contiguous float64
    numpy array. Required because ``compute_det_curve`` relies on ``.size`` and
    on numpy broadcasting semantics.

    Args:
        values (Sequence | np.ndarray | Tensor): values to convert.
    Returns:
        array (np.ndarray): 1D float64 array.
    """
    array = np.asarray(values, dtype=np.float64)
    return np.atleast_1d(array).reshape(-1)


def compute_eer_percent(scores, labels) -> float:
    """
    Compute the EER (in percents) for a flat collection of scores and labels.

    The score convention is the same as in the official grading script: a
    higher score means "more likely bonafide", and bonafide trials are the ones
    with ``label == 1``. Feeding scores with the opposite sign yields
    ``100 - EER`` instead of ``EER``.

    Args:
        scores (Sequence | np.ndarray): detection scores, higher = bonafide.
        labels (Sequence | np.ndarray): ground-truth labels (1 = bonafide,
            0 = spoof).
    Returns:
        eer (float): equal error rate in percents (0-100). Returns
            ``DEGENERATE_EER`` if either class is not present.
    """
    scores_array = as_float_array(scores)
    labels_array = as_float_array(labels)

    bonafide_mask = labels_array == 1
    bonafide_scores = scores_array[bonafide_mask]
    spoof_scores = scores_array[~bonafide_mask]

    if bonafide_scores.size == 0 or spoof_scores.size == 0:
        return DEGENERATE_EER

    eer, _ = compute_eer(bonafide_scores, spoof_scores)
    return float(eer) * 100
