import csv

import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import compute_eer_percent, logits_to_scores
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}
        self.save_path = save_path

        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        self._setup_amp()

        self._utt_ids = []
        self._scores = []
        self._labels = []

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        return {
            part: self._inference_part(part, dataloader)
            for part, dataloader in self.evaluation_dataloaders.items()
        }

    def process_batch(self, batch, metrics):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        self._scores.append(logits_to_scores(batch["logits"]).cpu())
        self._utt_ids.extend(batch["utt_id"])
        self._labels.append(batch["labels"].detach().reshape(-1).cpu())

        return batch

    def _inference_part(self, part, dataloader):
        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        self._utt_ids = []
        self._scores = []
        self._labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=part, total=len(dataloader)):
                self.process_batch(batch=batch, metrics=self.evaluation_metrics)

        scores = torch.cat(self._scores)
        labels = torch.cat(self._labels)

        self._save_scores(part, scores)

        logs = (
            self.evaluation_metrics.result()
            if self.evaluation_metrics is not None
            else {}
        )
        logs["EER"] = compute_eer_percent(scores.numpy(), labels.numpy())

        return logs

    def _save_scores(self, part, scores):
        self.save_path.mkdir(exist_ok=True, parents=True)
        with (self.save_path / f"{part}_scores.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            for utt_id, score in zip(self._utt_ids, scores.tolist()):
                writer.writerow([utt_id, repr(float(score))])
