import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    """
    Plain classification accuracy (argmax over logits).

    Accuracy is a poor target metric for anti-spoofing because of the strong
    class imbalance (a constant "spoof" prediction already gives ~90% on the
    ASVspoof2019 LA train partition), but it is cheap and decomposable over
    batches, which makes it a convenient sanity check during training.
    """

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch) -> float:
        """
        Args:
            logits (Tensor): model output of shape (B, n_classes).
            labels (Tensor): ground-truth labels of shape (B,).
        Returns:
            accuracy (float): share of correct predictions in the batch.
        """
        if logits.ndim != 2:
            raise ValueError(
                f"Expected logits of shape (B, n_classes), got {tuple(logits.shape)}"
            )
        predictions = logits.detach().argmax(dim=-1)
        return (predictions == labels.detach()).float().mean().item()
