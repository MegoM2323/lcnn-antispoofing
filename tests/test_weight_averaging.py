"""
Averaging checkpoints is only correct if two things hold. The weights must be
the plain arithmetic mean of the checkpoints, entry by entry, and the counters
that are not weights must not be averaged into something that describes no
model. The BatchNorm statistics must be measured again on real data: the
running values of the averaged checkpoints belong to other weights, and a model
scored with them produces garbage. The recalibration is therefore checked
against the statistics of the data it was shown, including the case that is
easy to get wrong -- a dropout layer in front of the BatchNorm, which at
training time inflates the variance the layer would measure.
"""

import pytest
import torch
from torch import nn

from src.model.weight_averaging import (
    average_state_dicts,
    batchnorm_modules,
    recalibrate_batchnorm,
)

FEATURES = 3


class TinyNet(nn.Module):
    """Dropout in front of a BatchNorm, the layout of the LCNN head."""

    def __init__(self, p: float = 0.75):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.bn = nn.BatchNorm1d(FEATURES)

    def forward(self, data_object: torch.Tensor, **batch) -> dict:
        return {"logits": self.bn(self.dropout(data_object))}


def state_dict(value: float) -> dict:
    return {
        "weight": torch.full((2, 2), value),
        "bn.running_mean": torch.full((FEATURES,), value),
        "bn.num_batches_tracked": torch.tensor(int(value), dtype=torch.long),
    }


def test_weights_are_the_arithmetic_mean_of_the_checkpoints():
    averaged = average_state_dicts([state_dict(1.0), state_dict(2.0), state_dict(6.0)])

    assert torch.allclose(averaged["weight"], torch.full((2, 2), 3.0))
    assert torch.allclose(averaged["bn.running_mean"], torch.full((FEATURES,), 3.0))


def test_integer_counters_are_not_averaged():
    averaged = average_state_dicts([state_dict(2.0), state_dict(5.0)])

    assert averaged["bn.num_batches_tracked"].dtype == torch.long
    assert int(averaged["bn.num_batches_tracked"]) == 2


def test_one_checkpoint_is_returned_unchanged():
    averaged = average_state_dicts([state_dict(4.0)])

    assert torch.allclose(averaged["weight"], torch.full((2, 2), 4.0))


def test_averaging_does_not_touch_the_inputs():
    first = state_dict(1.0)
    average_state_dicts([first, state_dict(3.0)])

    assert torch.allclose(first["weight"], torch.full((2, 2), 1.0))


def test_averaging_nothing_is_rejected():
    with pytest.raises(ValueError):
        average_state_dicts([])


def test_checkpoints_of_another_architecture_are_rejected():
    other = state_dict(1.0)
    del other["weight"]

    with pytest.raises(ValueError):
        average_state_dicts([state_dict(1.0), other])


def test_mismatched_shapes_are_rejected():
    other = state_dict(1.0)
    other["weight"] = torch.zeros(3, 3)

    with pytest.raises(ValueError):
        average_state_dicts([state_dict(1.0), other])


def test_dtype_is_preserved():
    half = {"weight": torch.ones(2, dtype=torch.float16)}
    zeros = {"weight": torch.zeros(2, dtype=torch.float16)}
    averaged = average_state_dicts([half, zeros])

    assert averaged["weight"].dtype == torch.float16


def test_batchnorm_layers_are_found():
    assert len(batchnorm_modules(TinyNet())) == 1


def test_recalibration_measures_the_statistics_of_the_data():
    model = TinyNet()
    batches = [torch.randn(64, FEATURES) * 3.0 + 5.0 for _ in range(8)]

    recalibrate_batchnorm(model, iter(batches), progress=False)

    data = torch.cat(batches)
    assert torch.allclose(model.bn.running_mean, data.mean(0), atol=0.2)
    assert torch.allclose(model.bn.running_var, data.var(0, unbiased=True), atol=1.0)


def test_recalibration_ignores_the_stale_statistics():
    model = TinyNet()
    model.bn.running_mean.fill_(100.0)
    model.bn.running_var.fill_(100.0)

    recalibrate_batchnorm(
        model, (torch.zeros(16, FEATURES) for _ in range(4)), progress=False
    )

    assert torch.allclose(model.bn.running_mean, torch.zeros(FEATURES), atol=1e-5)


def test_dropout_does_not_inflate_the_measured_variance():
    model = TinyNet(p=0.9)
    batches = [torch.randn(256, FEATURES) for _ in range(8)]

    recalibrate_batchnorm(model, iter(batches), progress=False)

    assert torch.allclose(model.bn.running_var, torch.cat(batches).var(0), atol=0.3)


def test_the_result_does_not_depend_on_the_order_of_the_batches():
    batches = [torch.randn(32, FEATURES) + shift for shift in range(6)]

    forward, backward = TinyNet(), TinyNet()
    recalibrate_batchnorm(forward, iter(batches), progress=False)
    recalibrate_batchnorm(backward, iter(batches[::-1]), progress=False)

    assert torch.allclose(forward.bn.running_mean, backward.bn.running_mean, atol=1e-5)


def test_only_the_requested_number_of_batches_is_used():
    model = TinyNet()
    batches = [torch.zeros(8, FEATURES), torch.full((8, FEATURES), 10.0)]

    seen = recalibrate_batchnorm(model, iter(batches), max_batches=1, progress=False)

    assert seen == 1
    assert torch.allclose(model.bn.running_mean, torch.zeros(FEATURES), atol=1e-5)


def test_the_model_is_left_in_evaluation_mode():
    model = TinyNet()
    model.eval()

    recalibrate_batchnorm(
        model, (torch.zeros(8, FEATURES) for _ in range(2)), progress=False
    )

    assert not model.training
    assert not model.bn.training


def test_the_momentum_of_the_layers_is_restored():
    model = TinyNet()
    momentum = model.bn.momentum

    recalibrate_batchnorm(
        model, (torch.zeros(8, FEATURES) for _ in range(2)), progress=False
    )

    assert model.bn.momentum == momentum


def test_a_model_without_batchnorm_is_rejected():
    model = nn.Sequential(nn.Linear(FEATURES, FEATURES))

    with pytest.raises(ValueError):
        recalibrate_batchnorm(model, iter([]), progress=False)


def test_an_empty_stream_of_batches_is_rejected():
    with pytest.raises(ValueError):
        recalibrate_batchnorm(TinyNet(), iter([]), progress=False)
