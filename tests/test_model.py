"""
Requirements covered:

* the countermeasure is a binary classifier: whatever the front-end, the model
  answers with two logits per utterance, and the detection score of the
  submission is their difference;
* the head is sized from the actual backbone output, so the same class fits
  both front-ends of the project. The sizes are pinned: the flattened head
  makes the parameter count depend on the input resolution, and a silent
  change there means the checkpoints of the two runs stop being comparable;
* an input arriving with or without the channel dimension is accepted, a
  malformed one is rejected instead of being reshaped by broadcasting;
* the embedding is exported only when it is asked for.
"""

import pytest
import torch

from src.model import LCNN

# input of the two front-ends of the project, see src/configs/model/lcnn.yaml
# and src/configs/lcnn_lfcc.yaml
STFT_SHAPE = (863, 600)
LFCC_SHAPE = (60, 750)

STFT_PARAMETERS = 10198818
LFCC_PARAMETERS = 865058


def n_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("in_freq, in_frames", [STFT_SHAPE, LFCC_SHAPE])
def test_forward_returns_two_logits_per_utterance(in_freq, in_frames):
    model = LCNN(in_freq=in_freq, in_frames=in_frames).eval()

    output = model(torch.randn(3, in_freq, in_frames))

    assert output["logits"].shape == (3, 2)
    assert torch.isfinite(output["logits"]).all()


def test_stft_model_size_is_fixed():
    model = LCNN(in_freq=STFT_SHAPE[0], in_frames=STFT_SHAPE[1])

    assert n_parameters(model) == STFT_PARAMETERS


def test_lfcc_model_size_is_fixed():
    model = LCNN(in_freq=LFCC_SHAPE[0], in_frames=LFCC_SHAPE[1])

    assert n_parameters(model) == LFCC_PARAMETERS


def test_channel_dimension_is_optional():
    model = LCNN(in_freq=60, in_frames=750).eval()
    features = torch.randn(2, 60, 750)

    with torch.no_grad():
        without_channel = model(features)["logits"]
        with_channel = model(features.unsqueeze(1))["logits"]

    assert torch.equal(without_channel, with_channel)


def test_malformed_input_is_rejected():
    model = LCNN(in_freq=60, in_frames=750).eval()

    with pytest.raises(ValueError):
        model(torch.randn(60, 750, 2, 1, 1))


def test_embedding_is_returned_only_on_request():
    features = torch.randn(2, 60, 750)

    assert "embedding" not in LCNN(in_freq=60, in_frames=750).eval()(features)

    model = LCNN(in_freq=60, in_frames=750, return_embedding=True).eval()
    output = model(features)

    assert output["embedding"].shape == (2, 80)  # embedding_dim // 2 after MFM
