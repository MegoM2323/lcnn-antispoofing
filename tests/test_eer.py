"""
The EER must follow the official grader: a higher score means "more likely
bonafide" (label 1), the value is computed over the whole partition instead of
being averaged over batches, and a missing class gives 50% rather than a NaN.
The metric is stateful, hence it must be resettable between epochs.
"""

import numpy as np
import pytest
import torch

from src.metrics.eer import EERMetric
from src.metrics.eer_utils import (
    DEGENERATE_EER,
    compute_eer_percent,
    logits_to_scores,
)


def test_perfect_separation_gives_zero():
    scores = [3.0, 2.0, 1.0, -1.0, -2.0, -3.0]
    labels = [1, 1, 1, 0, 0, 0]

    assert compute_eer_percent(scores, labels) == pytest.approx(0.0)


def test_inverted_scores_give_hundred():
    scores = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    labels = [1, 1, 1, 0, 0, 0]

    assert compute_eer_percent(scores, labels) == pytest.approx(100.0)


def test_indistinguishable_classes_give_fifty():
    scores = [1.0, 0.0, 1.0, 0.0]
    labels = [1, 1, 0, 0]

    assert compute_eer_percent(scores, labels) == pytest.approx(50.0)


def test_single_class_is_not_nan():
    assert compute_eer_percent([1.0, 2.0], [1, 1]) == DEGENERATE_EER
    assert compute_eer_percent([1.0, 2.0], [0, 0]) == DEGENERATE_EER


def test_one_error_out_of_ten():
    # 10 bonafide and 10 spoof trials, exactly one bonafide falls below the
    # whole spoof distribution: 10% of the target trials are rejected
    scores = list(np.arange(10, 20, dtype=float)) + list(np.arange(0, 10, dtype=float))
    scores[9] = -1.0
    labels = [1] * 10 + [0] * 10

    assert compute_eer_percent(scores, labels) == pytest.approx(10.0, abs=1e-6)


def test_eer_is_invariant_to_monotone_rescaling():
    scores = np.array([2.0, -1.0, 0.5, -3.0, 1.5, -0.2])
    labels = [1, 0, 1, 0, 1, 0]

    assert compute_eer_percent(scores, labels) == pytest.approx(
        compute_eer_percent(scores * 7.0 + 3.0, labels)
    )


def test_metric_matches_the_reference_over_the_whole_buffer():
    torch.manual_seed(0)
    logits = torch.randn(64, 2)
    labels = torch.randint(0, 2, (64,))

    metric = EERMetric(score_type="llr", name="EER")
    for start in range(0, 64, 16):  # fed batch by batch
        value = metric(
            logits=logits[start : start + 16], labels=labels[start : start + 16]
        )

    reference = compute_eer_percent(
        (logits[:, 1] - logits[:, 0]).numpy(), labels.numpy()
    )
    assert value == pytest.approx(reference)


def test_metric_reset_drops_the_history():
    metric = EERMetric(score_type="llr", name="EER")
    metric(logits=torch.tensor([[0.0, 5.0], [5.0, 0.0]]), labels=torch.tensor([1, 0]))
    metric.reset()

    assert metric.compute() == DEGENERATE_EER


def test_llr_and_softmax_scores_give_the_same_eer():
    torch.manual_seed(1)
    logits = torch.randn(32, 2)
    labels = torch.randint(0, 2, (32,))

    llr = EERMetric(score_type="llr", name="a")(logits=logits, labels=labels)
    softmax = EERMetric(score_type="softmax", name="b")(logits=logits, labels=labels)

    assert llr == pytest.approx(softmax)


def test_unknown_score_type_is_rejected():
    with pytest.raises(ValueError):
        EERMetric(score_type="cosine", name="EER")


def test_single_logit_output_is_rejected():
    metric = EERMetric(name="EER")

    with pytest.raises(ValueError):
        metric(logits=torch.zeros(4, 1), labels=torch.zeros(4, dtype=torch.long))


def test_higher_score_means_bonafide():
    # class 1 is bonafide, so the score must grow with the bonafide logit
    logits = torch.tensor([[0.0, 5.0], [5.0, 0.0]])

    scores = logits_to_scores(logits)

    assert scores[0] > scores[1]
    assert scores.tolist() == [5.0, -5.0]


def test_ready_made_scores_pass_through():
    scores = logits_to_scores(torch.tensor([0.3, -0.7]))

    assert scores.tolist() == pytest.approx([0.3, -0.7])
