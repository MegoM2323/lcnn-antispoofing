"""
Inference with several segments per utterance.

Everything except the shape of a batch is inherited from 'Inferencer': the
checkpoint loading, the front-end, the writing of the submission csv and the
EER of the partition. A batch here holds the segments of several utterances
stacked together, so the forward pass is as wide as an ordinary one, and only
the reduction of the logits differs: the scores of the segments of an utterance
are pooled into the single score the protocol expects.
"""

import torch

from src.datasets.multicrop import DEFAULT_N_SEGMENTS
from src.metrics.eer_utils import logits_to_scores
from src.metrics.segment_pooling import (
    DEFAULT_AGGREGATION,
    aggregate_segment_scores,
    validate_aggregation,
)
from src.trainer.inferencer import Inferencer


class MultiCropInferencer(Inferencer):
    """
    Inferencer that scores every utterance on several segments.

    Besides the two files of the base class it writes '<part>_segments.pth'
    with the score of every single segment. Repeating a run only to try another
    pooling rule costs another full pass over the partition, while re-reading
    that dump costs milliseconds.
    """

    def __init__(
        self,
        *args,
        aggregation: str = DEFAULT_AGGREGATION,
        n_segments: int = DEFAULT_N_SEGMENTS,
        **kwargs,
    ):
        """
        Args:
            aggregation (str): pooling rule for the segment scores, see
                src.metrics.segment_pooling.AGGREGATIONS.
            n_segments (int): number of segments per utterance the dataloader
                was built with. Only stored, for the header of the dump.
        """
        validate_aggregation(aggregation)
        super().__init__(*args, **kwargs)

        self.aggregation = aggregation
        self.n_segments = n_segments
        self._segment_scores: list[torch.Tensor] = []
        self._segment_sizes: list[int] = []

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Score every segment of the batch and pool the scores per utterance.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): batch from the multi-crop collate function, with
                'segment_sizes' telling which rows belong to which utterance.
            metrics (MetricTracker | None): ignored, the per-batch metrics of
                the base class expect one row per utterance.
            part (str): name of the partition.
        Returns:
            batch (dict): the batch with the model outputs and the pooled
                scores added.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        with self._autocast():
            outputs = self.model(**batch)
            batch.update(outputs)

        segment_sizes = batch["segment_sizes"]
        segment_scores = logits_to_scores(batch["logits"].detach().float().cpu())
        scores = aggregate_segment_scores(
            segment_scores, segment_sizes, self.aggregation
        )
        batch["scores"] = scores

        self._scores.append(scores)
        # the dump of the base class stays one row per utterance: the raw
        # per-segment outputs go to the segment dump instead
        self._logits.append(scores)
        self._segment_scores.append(segment_scores)
        self._segment_sizes.extend(int(size) for size in segment_sizes)
        if batch.get("utt_id") is not None:
            self._utt_ids.extend(batch["utt_id"])
        if batch.get("labels") is not None:
            self._labels.append(batch["labels"].detach().reshape(-1).cpu())

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a partition, resetting the per-segment buffers first.

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader built with the multi-crop
                collate function.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """
        self._segment_scores = []
        self._segment_sizes = []
        return super()._inference_part(part, dataloader)

    def _save_predictions(self, part, scores, labels):
        """
        Write the pooled predictions and the scores of the single segments.

        Args:
            part (str): name of the partition.
            scores (Tensor): 1D tensor with the pooled scores.
            labels (Tensor | None): 1D tensor with the ground truth labels.
        """
        super()._save_predictions(part, scores, labels)

        if self.save_path is None:
            return

        segments = (
            torch.cat(self._segment_scores) if self._segment_scores else torch.empty(0)
        )
        payload = {
            "utt_id": self._utt_ids,
            "segment_sizes": self._segment_sizes,
            "segment_scores": segments,
            "labels": labels,
            "aggregation": self.aggregation,
            "n_segments": self.n_segments,
        }
        path = self.save_path / f"{part}_segments.pth"
        try:
            torch.save(payload, path)
        except OSError as e:
            print(f"Failed to write the segment scores to {path}: {e}")
        else:
            print(f"Saved {segments.numel()} segment scores to {path}")
