"""
Requirements covered:

* a checkpoint is only meaningful together with the front-end it was trained
  with: 'load_state_dict' validates the shapes of the weights and nothing
  else, so a different waveform length, window, hop or number of frames is
  accepted silently and quietly corrupts the scores. Every such difference
  must be reported;
* checkpoints written before 'collate_max_len' existed have to be compared
  against the default that 'collate_fn' used back then;
* a matching config produces no false alarm.
"""

from omegaconf import OmegaConf

from src.datasets.collate import DEFAULT_MAX_LEN
from src.trainer.config_check import config_mismatches, format_mismatch_warning


def make_config(collate_max_len=77870, n_frames=600, crop="first", in_freq=863):
    config = {
        "model": {
            "_target_": "src.model.LCNN",
            "in_freq": in_freq,
            "in_frames": n_frames,
        },
        "transforms": {
            "batch_transforms": {
                "inference": {
                    "data_object": {
                        "_target_": "torch.nn.Sequential",
                        "_args_": [
                            {
                                "_target_": "src.transforms.LogSpectrogram",
                                "n_fft": 1724,
                                "win_length": 1724,
                                "hop_length": 130,
                                "window": "blackman",
                                "n_frames": n_frames,
                                "crop": crop,
                            }
                        ],
                    }
                }
            }
        },
    }
    if collate_max_len is not None:
        config["collate_max_len"] = collate_max_len
    return OmegaConf.create(config)


def test_identical_configs_have_no_mismatches():
    assert config_mismatches(make_config(), make_config()) == []


def test_different_collate_max_len_is_reported():
    mismatches = config_mismatches(make_config(64600), make_config(77870))

    assert len(mismatches) == 1
    assert "collate_max_len" in mismatches[0]
    assert "64600" in mismatches[0] and "77870" in mismatches[0]


def test_absent_collate_max_len_falls_back_to_the_collate_default():
    saved = make_config(collate_max_len=None)

    assert config_mismatches(saved, make_config(DEFAULT_MAX_LEN)) == []

    mismatches = config_mismatches(saved, make_config(77870))
    assert str(DEFAULT_MAX_LEN) in mismatches[0]
    assert "absent" in mismatches[0]


def test_frontend_difference_is_reported():
    mismatches = config_mismatches(
        make_config(n_frames=400, in_freq=863), make_config(n_frames=600, in_freq=863)
    )

    reported = " ".join(mismatches)
    assert "n_frames" in reported
    assert "in_frames" in reported  # the model input changes together with it


def test_crop_mode_difference_is_reported():
    mismatches = config_mismatches(
        make_config(crop="random"), make_config(crop="first")
    )

    assert len(mismatches) == 1
    assert "crop" in mismatches[0]
    assert "random" in mismatches[0] and "first" in mismatches[0]


def test_model_class_difference_is_reported():
    other = make_config()
    other.model._target_ = "src.model.BaselineModel"

    mismatches = config_mismatches(other, make_config())

    assert any("model._target_" in mismatch for mismatch in mismatches)


def test_missing_config_is_not_a_mismatch():
    assert config_mismatches(None, make_config()) == []
    assert config_mismatches(make_config(), None) == []


def test_warning_lists_every_difference():
    mismatches = config_mismatches(make_config(64600, n_frames=400), make_config())

    text = format_mismatch_warning(mismatches, "/tmp/model_best.pth")

    assert "/tmp/model_best.pth" in text
    for mismatch in mismatches:
        assert mismatch in text
