from collections import deque

import torch

from src.metrics.base_metric import BaseMetric
from src.metrics.eer_utils import (
    DEGENERATE_EER,
    compute_eer_percent,
    logits_to_scores,
    validate_score_type,
)


class EERMetric(BaseMetric):
    """
    Equal Error Rate for voice anti-spoofing countermeasures.

    EER is not decomposable over batches: averaging per-batch EERs is not the
    same as the EER of the whole partition. To keep the value logged by
    MetricTracker meaningful, this metric keeps an internal buffer of all
    scores seen so far and returns the running EER (computed over the whole
    buffer) on every call. The last value logged in an epoch is therefore the
    EER over the full partition, and the arithmetic average reported by the
    tracker is a smoothed approximation of it.

    Score convention (must match the official grading script): a higher score
    means "more likely bonafide", bonafide is label == 1, spoof is label == 0.
    Flipping the sign of the score turns EER into 100 - EER, so keep an eye on
    the score_type and on the label mapping of the dataset.
    """

    def __init__(
        self,
        score_type: str = "llr",
        max_buffer_size: int | None = 200_000,
        *args,
        **kwargs,
    ):
        """
        Args:
            score_type (str): how to turn model logits into a scalar score,
                see logits_to_scores.
            max_buffer_size (int | None): maximum number of trials kept in the
                buffer. The buffer behaves like a sliding window, so old scores
                are dropped once the limit is reached. None means unbounded.
                The default covers the full ASVspoof2019 LA eval partition
                (71237 utterances).
        """
        super().__init__(*args, **kwargs)
        validate_score_type(score_type)
        if max_buffer_size is not None and max_buffer_size <= 0:
            raise ValueError("max_buffer_size must be positive or None")

        self.score_type = score_type
        self.max_buffer_size = max_buffer_size
        self._scores: deque[float] = deque(maxlen=max_buffer_size)
        self._labels: deque[int] = deque(maxlen=max_buffer_size)

    def reset(self) -> None:
        """
        Drop all accumulated scores. Should be called at the beginning of every
        epoch/partition; otherwise the running EER mixes trials from different
        model states.
        """
        self._scores.clear()
        self._labels.clear()

    def compute(self) -> float:
        """
        Compute the EER over everything currently stored in the buffer.

        Returns:
            eer (float): equal error rate in percents (0-100), or
                DEGENERATE_EER (50.0) if one of the classes is missing.
        """
        if not self._scores:
            return DEGENERATE_EER
        return compute_eer_percent(self._scores, self._labels)

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch) -> float:
        """
        Add the batch to the buffer and return the running EER.

        Args:
            logits (Tensor): model output of shape (B, n_classes). A 1D tensor
                of shape (B,) is also accepted and treated as a ready-to-use
                score (higher = bonafide).
            labels (Tensor): ground-truth labels of shape (B,), 1 = bonafide.
        Returns:
            eer (float): running equal error rate in percents (0-100).
        """
        scores = logits_to_scores(logits, self.score_type)
        self._scores.extend(scores.cpu().tolist())
        self._labels.extend(labels.detach().cpu().reshape(-1).tolist())
        return self.compute()
