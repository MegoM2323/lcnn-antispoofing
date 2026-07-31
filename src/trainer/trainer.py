from contextlib import nullcontext

import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import compute_eer_percent
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

SCORE_HIST_BINS = 64


class Trainer(BaseTrainer):
    """
    Trainer for the anti-spoofing countermeasure.

    Adds two things on top of the base trainer:

    1. Mixed precision (bf16 autocast). The LCNN input is a 863x600 spectrogram,
       so activations dominate the memory footprint; bf16 halves it and speeds
       the forward pass up. bf16 has the same exponent range as fp32, hence no
       GradScaler is required.
    2. Correct epoch-level EER. EER is a property of the whole score
       distribution and is not decomposable over batches: the average of
       per-batch EERs is not the EER of the partition. Scores of the whole
       evaluation partition are therefore accumulated and the EER is computed
       once per epoch, exactly like the official grading script does.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            *args: positional arguments of BaseTrainer.
            **kwargs: keyword arguments of BaseTrainer.
        """
        super().__init__(*args, **kwargs)

        self.use_amp = bool(self.cfg_trainer.get("use_amp", False))
        self.amp_dtype = getattr(
            torch, str(self.cfg_trainer.get("amp_dtype", "bfloat16"))
        )
        self.amp_device_type = torch.device(self.device).type
        if self.use_amp and self.amp_device_type != "cuda":
            self.logger.warning(
                f"AMP is requested but the device is '{self.device}'. "
                "Autocast is disabled."
            )
            self.use_amp = False
        if self.use_amp:
            self.logger.info(f"Training with autocast, dtype={self.amp_dtype}.")

        # buffers for the epoch-level EER (filled during evaluation only)
        self._epoch_scores = []
        self._epoch_labels = []

    def _autocast(self):
        """
        Context manager for the forward pass.

        Returns:
            context (AbstractContextManager): autocast context if AMP is
                enabled, a no-op context otherwise.
        """
        if not self.use_amp:
            return nullcontext()
        return torch.autocast(device_type=self.amp_device_type, dtype=self.amp_dtype)

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

            # margin-based losses (AMSoftmax, OCSoftmax, P2SGrad) may replace
            # "logits" with their own scores, so the scores are read after this
            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            self._accumulate_scores(batch)

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch. Drops the state accumulated inside the
        metrics (e.g. the score buffer of the running EER) so that metrics of
        the current epoch are not contaminated by the previous ones.

        Args:
            epoch (int): current training epoch.
        Returns:
            logs (dict): logs that contain the average loss and metric in
                this epoch.
        """
        self._reset_metric_state("train")
        return super()._train_epoch(epoch)

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Repeats the logic of the base method and additionally computes the EER
        over the whole partition (see the class docstring). The value is logged
        as the "EER" scalar and put into the returned logs, so that it appears
        as "{part}_EER" in the common logs and can be monitored.

        Args:
            epoch (int): current training epoch.
            part (str): partition to evaluate on.
            dataloader (DataLoader): dataloader for the partition.
        Returns:
            logs (dict): logs that contain the information about evaluation.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
        self._reset_metric_state("inference")
        self._epoch_scores = []
        self._epoch_labels = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.epoch_len, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_batch(
                batch_idx, batch, part
            )  # log only the last batch during inference

        logs = self.evaluation_metrics.result()

        eer = self._compute_epoch_eer()
        if eer is not None:
            # logged after _log_scalars on purpose: if a running EER metric is
            # logged under the same name, the exact epoch value must win
            logs["EER"] = eer
            if self.writer is not None:
                self.writer.add_scalar("EER", eer)

        return logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        For evaluation partitions the score distributions of the two classes
        are logged as histograms: their overlap is exactly what the EER
        measures, so the plot shows how separable the classes are.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train" or self.writer is None:
            # nothing heavy on the train partition: the method is called
            # every log_step steps and images/histograms slow training down
            return

        scores, labels = self._collected_scores()
        if scores is None:
            return

        bonafide_scores = scores[labels == 1]
        spoof_scores = scores[labels == 0]
        if bonafide_scores.numel() > 0:
            self.writer.add_histogram(
                "scores_bonafide", bonafide_scores, bins=SCORE_HIST_BINS
            )
        if spoof_scores.numel() > 0:
            self.writer.add_histogram(
                "scores_spoof", spoof_scores, bins=SCORE_HIST_BINS
            )

    def _accumulate_scores(self, batch):
        """
        Store detection scores and labels of the batch for the epoch-level EER.

        Args:
            batch (dict): batch after the forward pass, contains "logits"
                and "labels".
        """
        logits = batch.get("logits")
        labels = batch.get("labels")
        if logits is None or labels is None:
            return
        self._epoch_scores.append(self.logits_to_scores(logits).cpu())
        self._epoch_labels.append(labels.detach().reshape(-1).cpu())

    def _collected_scores(self):
        """
        Concatenate the scores and labels accumulated during the epoch.

        Returns:
            scores (Tensor | None): 1D float tensor with detection scores,
                None if nothing was accumulated.
            labels (Tensor | None): 1D tensor with ground truth labels,
                None if nothing was accumulated.
        """
        if not self._epoch_scores:
            return None, None
        return torch.cat(self._epoch_scores), torch.cat(self._epoch_labels)

    def _compute_epoch_eer(self):
        """
        Compute the EER over all the scores accumulated during the epoch.

        Returns:
            eer (float | None): equal error rate in percents (0-100), or None
                if the scores are missing or one of the classes is absent.
        """
        scores, labels = self._collected_scores()
        if scores is None:
            return None

        bonafide_count = int((labels == 1).sum())
        if bonafide_count == 0 or bonafide_count == labels.numel():
            self.logger.warning(
                "Cannot compute the EER: one of the classes is missing "
                "in the partition."
            )
            return None

        return compute_eer_percent(scores.numpy(), labels.numpy())

    def _reset_metric_state(self, part):
        """
        Reset the internal state of stateful metrics (e.g. the score buffer of
        the running EER) for the given group of metrics.

        Args:
            part (str): "train" or "inference".
        """
        for met in self.metrics.get(part, []):
            reset = getattr(met, "reset", None)
            if callable(reset):
                reset()

    @staticmethod
    def logits_to_scores(logits):
        """
        Reduce model outputs to a 1D detection score, using the same convention
        as the official grading script: a higher score means "more likely
        bonafide" (label 1).

        Args:
            logits (Tensor): model output of shape (B, n_classes) or (B,).
        Returns:
            scores (Tensor): 1D float32 tensor with detection scores.
        """
        logits = logits.detach().float()
        if logits.ndim == 1:
            return logits
        if logits.ndim != 2 or logits.shape[-1] < 2:
            raise ValueError(
                f"Expected logits of shape (B, n_classes>=2), got {tuple(logits.shape)}"
            )
        return logits[:, 1] - logits[:, 0]
