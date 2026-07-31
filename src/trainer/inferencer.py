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
    score is the log-likelihood ratio of the bonafide class. Raw logits of the
    whole partition are additionally saved into a single .pth file, which is
    useful for the score-level fusion of several systems and for the analysis
    of the score distribution.
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
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
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
        self._logits = []
        self._labels = []

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def move_batch_to_device(self, batch):
        """
        Move all necessary tensors to the device.

        Unlike the base implementation, tensors that are not present in the
        batch are silently skipped: partitions without ground truth labels
        should not break the inference.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader with some of the tensors on the device.
        """
        for tensor_for_device in self.cfg_trainer.device_tensors:
            if tensor_for_device in batch:
                batch[tensor_for_device] = batch[tensor_for_device].to(self.device)
        return batch

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and accumulate
        predictions. Everything is written to disk once per partition
        in '_inference_part': one file per utterance would mean 71237
        files for the eval partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

        if metrics is not None:
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch))

        logits = batch["logits"].detach().float().cpu()
        self._logits.append(logits)
        self._scores.append(logits_to_scores(logits))
        if batch.get("utt_id") is not None:
            self._utt_ids.extend(batch["utt_id"])
        if batch.get("labels") is not None:
            self._labels.append(batch["labels"].detach().reshape(-1).cpu())

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition, save predictions and compute the
        EER if the ground truth labels are available.

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()
        for met in (self.metrics or {}).get("inference", []):
            reset = getattr(met, "reset", None)
            if callable(reset):
                reset()

        self._utt_ids = []
        self._scores = []
        self._logits = []
        self._labels = []

        # create Save dir
        if self.save_path is not None:
            self.save_path.mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        scores = torch.cat(self._scores) if self._scores else torch.empty(0)
        labels = torch.cat(self._labels) if self._labels else None

        self._save_predictions(part, scores, labels)

        logs = (
            self.evaluation_metrics.result()
            if self.evaluation_metrics is not None
            else {}
        )
        eer = self._compute_eer(scores, labels)
        if eer is not None:
            logs["EER"] = eer
            print(f"{part} EER: {eer:.4f}%")

        return logs

    def _save_predictions(self, part, scores, labels):
        """
        Write the scores of the partition to disk.

        Two files are written: '{part}_scores.csv' in the submission format
        (no header, "utterance_id,score") and '{part}_outputs.pth' with the raw
        logits of the whole partition.

        Args:
            part (str): name of the partition.
            scores (Tensor): 1D tensor with the detection scores.
            labels (Tensor | None): 1D tensor with the ground truth labels.
        """
        if self.save_path is None:
            return

        if len(self._utt_ids) == scores.numel():
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
        else:
            print(
                "The dataset does not provide 'utt_id', the submission csv is "
                "not written."
            )

        output = {
            "utt_id": self._utt_ids,
            "logits": torch.cat(self._logits) if self._logits else torch.empty(0),
            "scores": scores,
            "labels": labels,
        }
        pth_path = self.save_path / f"{part}_outputs.pth"
        try:
            torch.save(output, pth_path)
        except OSError as e:
            print(f"Failed to write logits to {pth_path}: {e}")

    def _compute_eer(self, scores, labels):
        """
        Compute the EER over the whole partition. Averaging per-batch EERs is
        incorrect, hence the metric is computed once, over all the scores.

        Args:
            scores (Tensor): 1D tensor with the detection scores.
            labels (Tensor | None): 1D tensor with the ground truth labels.
        Returns:
            eer (float | None): equal error rate in percents (0-100), or None
                if the labels are missing or one of the classes is absent.
        """
        return epoch_eer(scores, labels, warn=print)
