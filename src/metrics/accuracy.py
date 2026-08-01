import torch

from src.metrics.base_metric import BaseMetric


class AccuracyMetric(BaseMetric):
    """
    Обычная точность классификации (argmax по логитам).

    Для антиспуфинга точность плохая целевая метрика из-за сильного перекоса
    классов (постоянный ответ "spoof" уже даёт около 90 % на train-партиции
    ASVspoof2019 LA), но она дёшева и раскладывается по батчам, что делает её
    удобной проверкой вменяемости во время обучения.
    """

    def __call__(self, logits: torch.Tensor, labels: torch.Tensor, **batch) -> float:
        predictions = logits.detach().argmax(dim=-1)
        return (predictions == labels.detach()).float().mean().item()
