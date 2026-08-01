"""
Функции compute_det_curve и compute_eer взяты из официального пакета оценки
ASVspoof2019 (calculate_eer.py из материалов курса); изменены только
форматирование и аннотации типов.
"""

import numpy as np
import torch


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(
    bonafide_scores: np.ndarray, other_scores: np.ndarray
) -> tuple[float, float]:
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def as_float_array(values) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.atleast_1d(array).reshape(-1)


def compute_eer_percent(scores, labels) -> float:
    scores_array = as_float_array(scores)
    labels_array = as_float_array(labels)

    eer, _ = compute_eer(
        scores_array[labels_array == 1], scores_array[labels_array != 1]
    )
    return float(eer) * 100


def logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.detach().float()
    return logits[:, 1] - logits[:, 0]
