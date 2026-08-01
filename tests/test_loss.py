"""
Requirements covered:

* the LA train partition holds 22800 spoof against 2580 bonafide utterances, so
  an unweighted objective is minimized by answering "spoof" almost everywhere.
  Class weights must make an error on the rare class cost proportionally more;
* the "balanced" weights derived from the counts reproduce the 8.84 ratio the
  configs use explicitly;
* invalid hyperparameters fail at construction, not after an epoch of training.
"""

import pytest
import torch
from torch.nn.functional import cross_entropy

from src.loss import CELoss
from src.loss.ce_loss import ASVSPOOF19_LA_TRAIN_COUNTS, balanced_class_weights

# ratio of the two classes of the LA train partition, see src/configs/lcnn.yaml
BONAFIDE_WEIGHT = 8.84

# one spoof and one bonafide utterance, so that the two cases below share the
# normalizer of the weighted mean and only the misclassified class differs
LABELS = torch.tensor([0, 1])
BOTH_RIGHT = torch.tensor([[3.0, -3.0], [-3.0, 3.0]])
BONAFIDE_WRONG = torch.tensor([[3.0, -3.0], [3.0, -3.0]])
SPOOF_WRONG = torch.tensor([[-3.0, 3.0], [-3.0, 3.0]])


def test_rare_class_error_costs_more():
    loss = CELoss(class_weights=[1.0, BONAFIDE_WEIGHT])

    on_bonafide = loss(BONAFIDE_WRONG, LABELS)["loss"]
    on_spoof = loss(SPOOF_WRONG, LABELS)["loss"]

    assert on_bonafide > on_spoof
    assert on_spoof > loss(BOTH_RIGHT, LABELS)["loss"]


def test_class_weights_scale_the_per_utterance_loss():
    weights = torch.tensor([1.0, BONAFIDE_WEIGHT])
    loss = CELoss(class_weights=weights.tolist())

    per_utterance = cross_entropy(BONAFIDE_WRONG, LABELS, reduction="none")
    sample_weights = weights[LABELS]
    expected = (per_utterance * sample_weights).sum() / sample_weights.sum()

    assert torch.isclose(loss(BONAFIDE_WRONG, LABELS)["loss"], expected)


def test_unweighted_loss_treats_the_classes_alike():
    loss = CELoss()

    assert torch.isclose(
        loss(BONAFIDE_WRONG, LABELS)["loss"], loss(SPOOF_WRONG, LABELS)["loss"]
    )


def test_balanced_weights_reproduce_the_class_ratio():
    spoof_weight, bonafide_weight = balanced_class_weights(ASVSPOOF19_LA_TRAIN_COUNTS)

    assert bonafide_weight / spoof_weight == pytest.approx(BONAFIDE_WEIGHT, abs=1e-2)


def test_auto_weight_matches_the_explicit_weights():
    auto = CELoss(auto_weight=True)
    explicit = CELoss(class_weights=balanced_class_weights(ASVSPOOF19_LA_TRAIN_COUNTS))

    assert torch.allclose(auto.class_weights, explicit.class_weights)


def test_weights_follow_the_loss_to_the_device():
    loss = CELoss(class_weights=[1.0, BONAFIDE_WEIGHT])

    # registered as a buffer, so it is part of the state and moves with .to()
    assert "class_weights" in loss.state_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label_smoothing": 1.0},
        {"class_weights": [1.0, -1.0]},
        {"auto_weight": True, "class_weights": [1.0, 8.84]},
        {"auto_weight": True, "class_counts": [0, 2580]},
    ],
)
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        CELoss(**kwargs)
