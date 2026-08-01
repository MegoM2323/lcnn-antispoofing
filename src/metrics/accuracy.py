import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch) -> float:
        predictions = logits.detach().argmax(dim=-1)
        return (predictions == labels.detach()).float().mean().item()
