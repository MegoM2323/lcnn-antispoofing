"""
Requirements covered:

* a checkpoint written by the training run must be loadable by the inference
  run: this is the only path from a trained model to a submission, and it is
  checked here on a checkpoint with exactly the same content as the one
  '_save_checkpoint' writes (including the hydra config object);
* a bare state dict (a checkpoint from an external source) is also accepted.

The test builds the small LFCC-sized LCNN on CPU, so it takes a second.
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

    def __init__(self, model):
        self.model = model
        self.device = "cpu"


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
