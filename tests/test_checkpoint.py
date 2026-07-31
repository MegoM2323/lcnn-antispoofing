"""
A checkpoint is the only path from a trained model to a submission, so the
inference run must load back both what '_save_checkpoint' writes (the hydra
config object included) and a bare state dict from an external source. Loading
weights that were trained with another input pipeline has to be reported: the
shapes still match, so nothing else catches it. The small LFCC-sized LCNN is
built on CPU, so the test takes a second.
"""

import pytest
import torch
from omegaconf import OmegaConf

from src.model import LCNN
from src.trainer.base_trainer import BaseTrainer

CONFIG = OmegaConf.create(
    {
        "model": {"_target_": "src.model.LCNN", "in_freq": 60, "in_frames": 750},
        "optimizer": {"_target_": "torch.optim.Adam", "lr": 3e-4},
        "lr_scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR"},
    }
)


class _Loader(BaseTrainer):
    """Minimal holder that exposes only the checkpoint loading logic."""

    def __init__(self, model, config=None):
        self.model = model
        self.device = "cpu"
        if config is not None:
            self.config = config


def make_model():
    return LCNN(in_freq=60, in_frames=750, n_class=2, dropout=0.0)


def test_saved_checkpoint_can_be_loaded_back(tmp_path):
    model = make_model()
    checkpoint = {
        "arch": type(model).__name__,
        "epoch": 3,
        "state_dict": model.state_dict(),
        "optimizer": torch.optim.Adam(model.parameters()).state_dict(),
        "lr_scheduler": {},
        "monitor_best": 4.2,
        "config": CONFIG,  # _save_checkpoint stores the hydra config as is
    }
    path = tmp_path / "model_best.pth"
    torch.save(checkpoint, path)

    loader = _Loader(make_model())
    loader._from_pretrained(path)

    for saved, loaded in zip(
        model.state_dict().values(), loader.model.state_dict().values()
    ):
        assert torch.equal(saved, loaded)


def test_bare_state_dict_can_be_loaded(tmp_path):
    model = make_model()
    path = tmp_path / "weights.pth"
    torch.save(model.state_dict(), path)

    loader = _Loader(make_model())
    loader._from_pretrained(path)

    assert torch.equal(
        model.state_dict()["classifier.bias"],
        loader.model.state_dict()["classifier.bias"],
    )


def test_loaded_model_reproduces_the_predictions(tmp_path):
    torch.manual_seed(0)
    model = make_model().eval()
    features = torch.randn(2, 60, 750)
    with torch.no_grad():
        expected = model(data_object=features)["logits"]

    path = tmp_path / "model_best.pth"
    torch.save({"state_dict": model.state_dict(), "config": CONFIG}, path)

    loader = _Loader(make_model())
    loader._from_pretrained(path)
    loader.model.eval()
    with torch.no_grad():
        actual = loader.model(data_object=features)["logits"]

    assert torch.allclose(expected, actual)


def test_missing_checkpoint_is_reported(tmp_path):
    loader = _Loader(make_model())

    with pytest.raises((FileNotFoundError, OSError)):
        loader._from_pretrained(tmp_path / "does_not_exist.pth")


def save_checkpoint(tmp_path, config):
    path = tmp_path / "model_best.pth"
    torch.save({"state_dict": make_model().state_dict(), "config": config}, path)
    return path


def test_foreign_input_pipeline_is_reported(tmp_path, capsys):
    path = save_checkpoint(
        tmp_path, OmegaConf.merge(CONFIG, {"collate_max_len": 64600})
    )
    current = OmegaConf.merge(CONFIG, {"collate_max_len": 77870})

    _Loader(make_model(), current)._from_pretrained(path)

    report = capsys.readouterr().out
    assert "CONFIG MISMATCH" in report
    assert "collate_max_len" in report
    assert "64600" in report and "77870" in report


def test_matching_config_does_not_warn(tmp_path, capsys):
    config = OmegaConf.merge(CONFIG, {"collate_max_len": 77870})
    path = save_checkpoint(tmp_path, config)

    _Loader(make_model(), config)._from_pretrained(path)

    assert "CONFIG MISMATCH" not in capsys.readouterr().out


def test_checkpoint_without_a_config_is_reported(tmp_path, capsys):
    path = tmp_path / "weights.pth"
    torch.save({"state_dict": make_model().state_dict()}, path)

    _Loader(make_model(), CONFIG)._from_pretrained(path)

    assert "stores no config" in capsys.readouterr().out
