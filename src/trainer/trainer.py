import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import compute_eer_percent, logits_to_scores
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer

SCORE_HIST_BINS = 64


class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._setup_amp()

        self._epoch_scores = []
        self._epoch_labels = []

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

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
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        else:
            self._accumulate_scores(batch)

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
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
            self._log_batch(batch_idx, batch, part)

        logs = self.evaluation_metrics.result()

        scores, labels = self._collected_scores()
        logs["EER"] = compute_eer_percent(scores.numpy(), labels.numpy())
        self.writer.add_scalar("EER", logs["EER"])

        return logs

    def _log_batch(self, batch_idx, batch, mode="train"):
        if mode == "train":
            return

        scores, labels = self._collected_scores()
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
        self._epoch_scores.append(logits_to_scores(batch["logits"]).cpu())
        self._epoch_labels.append(batch["labels"].detach().reshape(-1).cpu())

    def _collected_scores(self):
        return torch.cat(self._epoch_scores), torch.cat(self._epoch_labels)
