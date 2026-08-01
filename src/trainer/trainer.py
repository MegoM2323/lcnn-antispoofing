import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import epoch_eer, logits_to_scores
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

SCORE_HIST_BINS = 64


class Trainer(BaseTrainer):
    """
    Trainer for the anti-spoofing countermeasure.

    Adds two things on top of the base trainer:

    1. Mixed precision around the forward pass, see BaseTrainer._setup_amp.
    2. Correct epoch-level EER. EER is a property of the whole score
       distribution and is not decomposable over batches: the average of
       per-batch EERs is not the EER of the partition. Scores of the whole
       evaluation partition are therefore accumulated and the EER is computed
       once per epoch, exactly like the official grading script does.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._setup_amp()

        # buffers for the epoch-level EER (filled during evaluation only)
        self._epoch_scores = []
        self._epoch_labels = []

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.
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

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Evaluate model on the partition after training for an epoch.

        Repeats the logic of the base method and additionally computes the EER
        over the whole partition (see the class docstring). The value is logged
        as the "EER" scalar and put into the returned logs, so that it appears
        as "{part}_EER" in the common logs and can be monitored.
        """
        self.is_train = False
        self.model.eval()
        self.evaluation_metrics.reset()
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

        scores, labels = self._collected_scores()
        eer = epoch_eer(scores, labels, warn=self.logger.warning)
        if eer is not None:
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
        """
        self._epoch_scores.append(logits_to_scores(batch["logits"]).cpu())
        self._epoch_labels.append(batch["labels"].detach().reshape(-1).cpu())

    def _collected_scores(self):
        """
        Concatenate the scores and labels accumulated during the epoch,
        (None, None) if nothing was accumulated.
        """
        if not self._epoch_scores:
            return None, None
        return torch.cat(self._epoch_scores), torch.cat(self._epoch_labels)
