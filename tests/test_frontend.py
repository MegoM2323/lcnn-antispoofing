"""
Requirements covered:

* the front-end feeds the model a tensor of a fixed size, and the size is the
  one the model was built for: 863 x 600 for the log spectrogram of
  arXiv:1904.05576 and 60 x 750 for the LFCC of arXiv:2103.11326. A drift here
  is not caught by 'load_state_dict', which only validates weight shapes;
* an utterance shorter than the window is repeated cyclically, not zero-padded:
  trailing silence carries no spoofing cues and the network learns it as a
  shortcut;
* scoring is deterministic. 'crop=first' always takes the leading frames, so a
  submission does not depend on the random state, while 'crop=random' is a
  train-time augmentation and does move;
* an unknown crop or padding mode fails at construction, not silently at the
  first batch.
"""

import pytest
import torch

from src.transforms import LFCC, LogSpectrogram
from src.transforms.frontend import fix_frames

SR = 16000
# the length every waveform of a batch is brought to, see collate_max_len
WAVEFORM_LEN = 77870


def test_log_spectrogram_matches_the_model_input():
    spec = LogSpectrogram()(torch.randn(2, WAVEFORM_LEN))

    assert spec.shape == (2, 863, 600)
    assert torch.isfinite(spec).all()


def test_lfcc_matches_the_model_input():
    lfcc = LFCC()(torch.randn(2, WAVEFORM_LEN))

    assert lfcc.shape == (2, 60, 750)
    assert torch.isfinite(lfcc).all()


def test_front_ends_keep_their_size_on_a_short_utterance():
    short = torch.randn(1, SR // 2)  # 0.5 s, far below the window

    assert LogSpectrogram()(short).shape == (1, 863, 600)
    assert LFCC()(short).shape == (1, 60, 750)


def test_short_sequence_is_repeated_not_zero_padded():
    spec = torch.tensor([[1.0, 2.0, 3.0]])

    padded = fix_frames(spec, n_frames=8, crop="first", pad_mode="repeat")

    assert torch.equal(padded, torch.tensor([[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]]))
    assert (padded != 0).all()


def test_zero_padding_is_available_but_not_the_default():
    spec = torch.tensor([[1.0, 2.0, 3.0]])

    padded = fix_frames(spec, n_frames=5, crop="first", pad_mode="zero")

    assert torch.equal(padded, torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]]))


def test_first_crop_takes_the_leading_frames():
    spec = torch.arange(10, dtype=torch.float32).reshape(1, 10)

    cropped = fix_frames(spec, n_frames=4, crop="first", pad_mode="repeat")

    assert torch.equal(cropped, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))


def test_inference_front_end_is_deterministic():
    waveform = torch.randn(1, WAVEFORM_LEN)
    front_end = LogSpectrogram(crop="first")

    torch.manual_seed(0)
    first = front_end(waveform)
    torch.manual_seed(1)
    second = front_end(waveform)

    assert torch.equal(first, second)


def test_random_crop_moves_across_the_utterance():
    spec = torch.arange(100, dtype=torch.float32).reshape(1, 100)

    torch.manual_seed(0)
    crops = {
        tuple(fix_frames(spec, 4, crop="random", pad_mode="repeat")[0].tolist())
        for _ in range(20)
    }

    assert len(crops) > 1
    # whatever the offset, a crop is a contiguous slice of the input
    for crop in crops:
        assert crop == tuple(float(crop[0] + i) for i in range(4))


@pytest.mark.parametrize("kwargs", [{"crop": "middle"}, {"pad_mode": "wrap"}])
def test_unknown_frame_mode_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LogSpectrogram(**kwargs)
    with pytest.raises(ValueError):
        LFCC(**kwargs)
