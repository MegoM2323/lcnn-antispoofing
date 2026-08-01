import csv

import torch
from tqdm.auto import tqdm

from src.metrics.eer_utils import epoch_eer, logits_to_scores
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.

    The predictions are saved in the format expected by the official grading
    script: a headerless csv with the "utterance_id,score" rows, where the
    score is the log-likelihood ratio of the bonafide class.
    """

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
        """
        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders to score.
            save_path (Path): directory the predictions are written to.
            metrics (dict | None): metrics[inference], each of them an
                instance of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms applied on
                the whole batch, depending on the tensor name.
            skip_model_load (bool): if False, the checkpoint of
                config.inferencer.from_pretrained is loaded into the model.
        """
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

        # buffers with the predictions of the current partition
        self._utt_ids = []
        self._scores = []
        self._labels = []

        if not skip_model_load:
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition and return its logs by partition name.
        """
        return {
            part: self._inference_part(part, dataloader)
            for part, dataloader in self.evaluation_dataloaders.items()
        }

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Unlike the base implementation, tensors that are not present in the
        batch are silently skipped: partitions without ground truth labels
        should not break the inference.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            if tensor_for_device in batch:
                batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def process_batch(self, batch, metrics):
        """
        Run batch through the model, compute metrics, and accumulate
        predictions. Everything is written to disk once per partition
        in '_inference_part': one file per utterance would mean 71237
        files for the eval partition.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        self._scores.append(logits_to_scores(batch["logits"]).cpu())
        self._utt_ids.extend(batch["utt_id"])
        if batch.get("labels") is not None:
            self._labels.append(batch["labels"].detach().reshape(-1).cpu())

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition, save the predictions and compute
        the EER if the ground truth labels are available.
        """
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

        scores = torch.cat(self._scores) if self._scores else torch.empty(0)
        labels = torch.cat(self._labels) if self._labels else None

        self._save_scores(part, scores)

        logs = (
            self.evaluation_metrics.result()
            if self.evaluation_metrics is not None
            else {}
        )
        eer = epoch_eer(scores, labels, warn=print)
        if eer is not None:
            logs["EER"] = eer
            print(f"{part} EER: {eer:.4f}%")

        return logs

    def _save_scores(self, part, scores):
        """
        Write '{part}_scores.csv' in the submission format: no header,
        "utterance_id,score", one row per utterance of the partition.
        """
        if self.save_path is None:
            return

        self.save_path.mkdir(exist_ok=True, parents=True)
        csv_path = self.save_path / f"{part}_scores.csv"
        try:
            with csv_path.open("w", newline="") as file:
                writer = csv.writer(file)
                for utt_id, score in zip(self._utt_ids, scores.tolist()):
                    writer.writerow([utt_id, repr(float(score))])
        except OSError as e:
            print(f"Failed to write scores to {csv_path}: {e}")
        else:
            print(f"Saved {scores.numel()} scores to {csv_path}")
