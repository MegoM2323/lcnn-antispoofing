"""
The front-end feeds the model a tensor of a fixed size, and the size is the one
the model was built for: 863 x 600 for the log spectrogram of arXiv:1904.05576
and 60 x 750 for the LFCC of arXiv:2103.11326. A drift here is not caught by
'load_state_dict', which only validates weight shapes.
"""

import pytest
import torch

from src.model import LCNN
from src.trainer.base_trainer import BaseTrainer
from src.transforms import LFCC, LogSpectrogram

# input of the two front-ends of the project, with the number of parameters of
# the model built for it: the flattened head makes the count depend on the
# input resolution
FRONT_ENDS = [
    (LogSpectrogram, (863, 600), 10198818),
    (LFCC, (60, 750), 865058),
]

# the length every waveform of a batch is brought to, see collate_max_len
WAVEFORM_LEN = 77870


class _Loader(BaseTrainer):
    """Minimal holder that exposes only the checkpoint loading logic."""

    def __init__(self, model):
        self.model = model
        self.device = "cpu"


@pytest.mark.parametrize("front_end, shape, _", FRONT_ENDS)
def test_front_end_matches_the_model_input(front_end, shape, _):
    features = front_end()(torch.randn(2, WAVEFORM_LEN))

    assert features.shape == (2, *shape)
    assert torch.isfinite(features).all()
    # a 0.5 s utterance, far below the window, still fills the same tensor
    assert front_end()(torch.randn(1, 8000)).shape == (1, *shape)


@pytest.mark.parametrize("_, shape, n_expected", FRONT_ENDS)
def test_forward_returns_two_logits_per_utterance(_, shape, n_expected):
    in_freq, in_frames = shape
    model = LCNN(in_freq=in_freq, in_frames=in_frames).eval()

    output = model(torch.randn(3, in_freq, in_frames))

    assert output["logits"].shape == (3, 2)
    assert torch.isfinite(output["logits"]).all()
    assert sum(p.numel() for p in model.parameters()) == n_expected


def test_saved_checkpoint_is_loaded_back(tmp_path):
    # a checkpoint is the only path from a trained model to a submission
    torch.manual_seed(0)
    model = LCNN(in_freq=60, in_frames=750, dropout=0.0).eval()
    path = tmp_path / "model_best.pth"
    torch.save({"state_dict": model.state_dict()}, path)

    loader = _Loader(LCNN(in_freq=60, in_frames=750, dropout=0.0).eval())
    loader._from_pretrained(path)

    features = torch.randn(2, 60, 750)
    with torch.no_grad():
        assert torch.equal(
            model(data_object=features)["logits"],
            loader.model(data_object=features)["logits"],
        )
