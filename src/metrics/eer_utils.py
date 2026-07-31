"""
Reference implementation of the Equal Error Rate (EER) computation.

compute_det_curve and compute_eer are taken verbatim (logic-wise) from the
official ASVspoof2019 evaluation package (calculate_eer.py shipped with the
course). They are kept unchanged on purpose: the final submission is graded
with exactly these functions, so any deviation would make local numbers
incomparable with the leaderboard ones.

Only cosmetic changes were applied (formatting + type hints). The rest of this
module contains thin numpy-2.x-safe wrappers shared by the metric, the trainer
and the inferencer.
"""

from collections.abc import Callable

import numpy as np
import torch

# EER (in %) reported when one of the two classes is missing from the buffer.
# The official calculate_eer.py has no such case: it is always given both
# classes and would divide by zero otherwise. A chance-level value is used
# instead of nan because MetricTracker aggregates metrics with pandas, and a
# single nan would poison the running average of the whole epoch.
#
# The value never reaches a reported EER: the epoch-level number comes from
# 'epoch_eer', which returns None for a degenerate partition, and both the
# trainer and the inferencer log the EER only when it is not None. Every
# partition that is scored has both classes in it, so the branch is only ever
# taken by per-batch metrics.
DEGENERATE_EER = 50.0

SCORE_TYPES = ("llr", "bonafide_logit", "softmax")


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

    The EER is a fraction in [0, 1], not a percentage.

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
    numpy array. Required because compute_det_curve relies on .size and on
    numpy broadcasting semantics.

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
    with label == 1. Feeding scores with the opposite sign yields 100 - EER
    instead of EER.

    Args:
        scores (Sequence | np.ndarray): detection scores, higher = bonafide.
        labels (Sequence | np.ndarray): ground-truth labels (1 = bonafide,
            0 = spoof).
    Returns:
        eer (float): equal error rate in percents (0-100). Returns
            DEGENERATE_EER if either class is not present.
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


def validate_score_type(score_type: str) -> None:
    """
    Check that the way of turning logits into a score is a supported one.

    Args:
        score_type (str): score type to check, see logits_to_scores.
    """
    if score_type not in SCORE_TYPES:
        raise ValueError(
            f"Unknown score_type '{score_type}', expected one of {SCORE_TYPES}"
        )


def logits_to_scores(logits: torch.Tensor, score_type: str = "llr") -> torch.Tensor:
    """
    Reduce model outputs to a 1D detection score, using the same convention as
    the official grading script: a higher score means "more likely bonafide"
    (label 1).

    Args:
        logits (Tensor): model output of shape (B, n_classes) or (B,). A 1D
            tensor is treated as a ready-to-use score.
        score_type (str): how to reduce the logits.
            * "llr" (default): logits[:, 1] - logits[:, 0], the log-likelihood
              ratio of bonafide vs spoof.
            * "bonafide_logit": logits[:, 1].
            * "softmax": posterior probability of the bonafide class.
            All three are monotonically related, hence they give exactly the
            same EER; "softmax" is only useful for logging bounded scores.
    Returns:
        scores (Tensor): 1D float32 tensor with detection scores.
    """
    validate_score_type(score_type)

    logits = logits.detach().float()
    if logits.ndim == 1:
        return logits
    if logits.ndim != 2 or logits.shape[-1] < 2:
        raise ValueError(
            f"Expected logits of shape (B, n_classes>=2), got {tuple(logits.shape)}"
        )

    if score_type == "bonafide_logit":
        return logits[:, 1]
    if score_type == "softmax":
        return torch.softmax(logits, dim=-1)[:, 1]
    return logits[:, 1] - logits[:, 0]


def epoch_eer(
    scores: torch.Tensor | None,
    labels: torch.Tensor | None,
    warn: Callable[[str], None] | None = None,
) -> float | None:
    """
    Compute the EER over the scores of a whole partition.

    Unlike compute_eer_percent, a degenerate partition gives None instead of
    DEGENERATE_EER: a chance-level number logged as the epoch EER would look
    like a real result.

    Args:
        scores (Tensor | None): 1D tensor with the detection scores.
        labels (Tensor | None): 1D tensor with the ground truth labels.
        warn (Callable | None): called with an explanation when the EER cannot
            be computed although the scores are there.
    Returns:
        eer (float | None): equal error rate in percents (0-100), or None if
            the scores are missing, the labels are missing, or one of the
            classes is absent.
    """
    if scores is None or labels is None or scores.numel() == 0:
        return None

    bonafide_count = int((labels == 1).sum())
    if bonafide_count == 0 or bonafide_count == labels.numel():
        if warn is not None:
            warn(
                "Cannot compute the EER: one of the classes is missing "
                "in the partition."
            )
        return None

    return compute_eer_percent(scores.numpy(), labels.numpy())
