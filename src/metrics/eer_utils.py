"""
Эталонная реализация подсчёта Equal Error Rate (EER).

Функции compute_det_curve и compute_eer по логике дословно взяты из
официального пакета оценки ASVspoof2019 (calculate_eer.py из материалов
курса). Они намеренно оставлены без изменений: итоговая посылка проверяется
ровно этими функциями, поэтому любое расхождение сделало бы локальные числа
несравнимыми с числами лидерборда.

Изменения носят косметический характер (форматирование, аннотации типов и
перевод комментариев). Остальная часть модуля это тонкие обёртки, безопасные
для numpy 2.x, которые общие для метрики, тренера и инференсера.
"""

import numpy as np
import torch


def compute_det_curve(
    target_scores: np.ndarray, nontarget_scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Считает кривую Detection Error Tradeoff (DET).

    Аргументы:
        target_scores (np.ndarray): скоры целевых (bonafide) испытаний.
        nontarget_scores (np.ndarray): скоры нецелевых (spoof) испытаний.
    Возвращает:
        frr (np.ndarray): доли ложных отказов.
        far (np.ndarray): доли ложных пропусков.
        thresholds (np.ndarray): пороги, отвечающие frr/far.
    """
    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate(
        (np.ones(target_scores.size), np.zeros(nontarget_scores.size))
    )

    # Сортировка меток по скорам
    indices = np.argsort(all_scores, kind="mergesort")
    labels = labels[indices]

    # Подсчёт долей ложных отказов и ложных пропусков
    tar_trial_sums = np.cumsum(labels)
    nontarget_trial_sums = nontarget_scores.size - (
        np.arange(1, n_scores + 1) - tar_trial_sums
    )

    # доли ложных отказов
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))
    far = np.concatenate(
        (np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size)
    )  # доли ложных пропусков
    # Пороги это отсортированные скоры
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices])
    )

    return frr, far, thresholds


def compute_eer(
    bonafide_scores: np.ndarray, other_scores: np.ndarray
) -> tuple[float, float]:
    """
    Возвращает equal error rate (EER) и отвечающий ему порог.

    EER это доля из [0, 1], а не проценты.

    Аргументы:
        bonafide_scores (np.ndarray): скоры испытаний bonafide.
        other_scores (np.ndarray): скоры поддельных испытаний.
    Возвращает:
        eer (float): equal error rate из [0, 1].
        threshold (float): порог, при котором frr и far ближе всего.
    """
    frr, far, thresholds = compute_det_curve(bonafide_scores, other_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)
    eer = np.mean((frr[min_index], far[min_index]))
    return eer, thresholds[min_index]


def as_float_array(values) -> np.ndarray:
    """
    Превращает произвольную последовательность скоров или меток в плоский
    массив float64: compute_det_curve опирается на .size и на семантику
    броадкастинга numpy.
    """
    array = np.asarray(values, dtype=np.float64)
    return np.atleast_1d(array).reshape(-1)


def compute_eer_percent(scores, labels) -> float:
    """
    Считает EER (в процентах) для плоского набора скоров и меток.

    Соглашение о скорах то же, что в официальном проверяющем скрипте: чем выше
    скор, тем вероятнее bonafide, а испытания bonafide это те, у которых
    label == 1. Скоры с обратным знаком дадут 100 - EER вместо EER.
    """
    scores_array = as_float_array(scores)
    labels_array = as_float_array(labels)

    eer, _ = compute_eer(
        scores_array[labels_array == 1], scores_array[labels_array != 1]
    )
    return float(eer) * 100


def logits_to_scores(logits: torch.Tensor) -> torch.Tensor:
    """
    Сводит выход модели к детекционному скору проверяющего скрипта: отношению
    правдоподобий bonafide (класс 1) против spoof (класс 0) в логарифме, так
    что чем выше скор, тем вероятнее bonafide.
    """
    logits = logits.detach().float()
    return logits[:, 1] - logits[:, 0]
