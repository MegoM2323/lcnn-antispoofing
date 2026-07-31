"""
Requirements covered:

* the best checkpoint is the one with the best validation metric: an epoch
  that only repeats the current best value is not an improvement. On
  ASVspoof2019 LA the dev EER saturates at 0.0 within a few epochs, and with a
  non-strict comparison every later epoch counted as "best", so model_best.pth
  was simply the last epoch and early stopping could never trigger;
* the ties of a saturated metric can be resolved by a secondary metric
  ('trainer.monitor_tiebreak'), which is off by default;
* early stopping fires after 'early_stop' epochs without an improvement;
* the criterion the best epoch won by is available for the log.
"""

import logging

import pytest
from omegaconf import OmegaConf

from src.trainer.base_trainer import BaseTrainer


class _Monitor(BaseTrainer):
    """Only the monitoring part of the trainer, without model or data."""

    def __init__(self, **trainer_config):
        self.config = OmegaConf.create({"trainer": trainer_config})
        self.cfg_trainer = self.config.trainer
        self.logger = logging.getLogger("test_monitor")

        self.monitor = self.cfg_trainer.get("monitor", "off")
        BaseTrainer._setup_monitoring(self)

    def feed(self, *epoch_logs):
        """
        Run the monitor over a sequence of epochs.

        Args:
            *epoch_logs (dict): logs of every epoch, in order.
        Returns:
            best_epochs (list[int]): 1-based numbers of the epochs that were
                declared the best ones.
            stopped_at (int | None): epoch that triggered early stopping.
        """
        best_epochs = []
        stopped_at = None
        not_improved_count = 0
        for epoch, logs in enumerate(epoch_logs, start=1):
            best, stop_process, not_improved_count = self._monitor_performance(
                logs, not_improved_count
            )
            if best:
                best_epochs.append(epoch)
            if stop_process:
                stopped_at = epoch
                break
        return best_epochs, stopped_at


def make_monitor(**trainer_config):
    trainer_config.setdefault("early_stop", 100)
    return _Monitor(monitor="min dev_EER", **trainer_config)


def test_repeated_metric_value_is_not_an_improvement():
    monitor = make_monitor()

    best_epochs, _ = monitor.feed(
        {"dev_EER": 1.0},
        {"dev_EER": 0.0},
        {"dev_EER": 0.0},
        {"dev_EER": 0.0},
    )

    assert best_epochs == [1, 2]


def test_worse_metric_value_is_not_an_improvement():
    monitor = make_monitor()

    best_epochs, _ = monitor.feed(
        {"dev_EER": 0.5},
        {"dev_EER": 0.7},
        {"dev_EER": 0.4},
    )

    assert best_epochs == [1, 3]


def test_max_mode_compares_strictly():
    monitor = _Monitor(monitor="max dev_Accuracy", early_stop=100)

    best_epochs, _ = monitor.feed(
        {"dev_Accuracy": 0.9},
        {"dev_Accuracy": 0.9},
        {"dev_Accuracy": 0.95},
    )

    assert best_epochs == [1, 3]


def test_early_stopping_fires_on_a_saturated_metric():
    monitor = make_monitor(early_stop=2)

    best_epochs, stopped_at = monitor.feed(*([{"dev_EER": 0.0}] * 5))

    assert best_epochs == [1]
    assert stopped_at == 3


def test_tiebreak_ranks_the_epochs_with_the_same_metric():
    monitor = make_monitor(monitor_tiebreak="min dev_loss")

    best_epochs, _ = monitor.feed(
        {"dev_EER": 0.0, "dev_loss": 0.5},
        {"dev_EER": 0.0, "dev_loss": 0.4},
        {"dev_EER": 0.0, "dev_loss": 0.6},
        {"dev_EER": 0.0, "dev_loss": 0.3},
    )

    assert best_epochs == [1, 2, 4]


def test_tiebreak_does_not_override_the_main_metric():
    monitor = make_monitor(monitor_tiebreak="min dev_loss")

    best_epochs, _ = monitor.feed(
        {"dev_EER": 0.0, "dev_loss": 0.5},
        {"dev_EER": 0.3, "dev_loss": 0.1},
    )

    assert best_epochs == [1]


def test_tiebreak_restarts_after_the_main_metric_improves():
    monitor = make_monitor(monitor_tiebreak="min dev_loss")

    best_epochs, _ = monitor.feed(
        {"dev_EER": 1.0, "dev_loss": 0.1},
        {"dev_EER": 0.0, "dev_loss": 0.9},
        {"dev_EER": 0.0, "dev_loss": 0.8},
    )

    assert best_epochs == [1, 2, 3]


def test_tiebreak_is_off_by_default():
    monitor = make_monitor()

    best_epochs, _ = monitor.feed(
        {"dev_EER": 0.0, "dev_loss": 0.5},
        {"dev_EER": 0.0, "dev_loss": 0.1},
    )

    assert best_epochs == [1]


def test_missing_tiebreak_metric_is_tolerated():
    monitor = make_monitor(monitor_tiebreak="min dev_loss")

    best_epochs, _ = monitor.feed({"dev_EER": 0.0}, {"dev_EER": 0.0})

    assert best_epochs == [1]


def test_best_criterion_names_the_deciding_metric():
    monitor = make_monitor(monitor_tiebreak="min dev_loss")

    monitor.feed({"dev_EER": 0.0, "dev_loss": 0.5})
    assert "dev_EER" in monitor.best_criterion

    monitor.feed({"dev_EER": 0.0, "dev_loss": 0.2})
    assert "dev_loss" in monitor.best_criterion


def test_missing_monitored_metric_disables_monitoring():
    monitor = make_monitor()

    best_epochs, _ = monitor.feed({"dev_Accuracy": 0.9}, {"dev_EER": 0.0})

    assert best_epochs == []
    assert monitor.mnt_mode == "off"


def test_malformed_tiebreak_is_rejected():
    with pytest.raises((AssertionError, ValueError)):
        make_monitor(monitor_tiebreak="lower dev_loss")
