"""
Кросс-энтропия с весами классов, компенсирующими перекос bonafide/spoof
в train-партиции ASVspoof2019 LA.
"""

import torch
from torch import nn


class CELoss(nn.Module):
    """
    Взвешенная кросс-энтропия, основной лосс антиспуфинг-классификатора.

    Порядок классов такой же, как в датасете: индекс 0 = spoof, индекс 1 =
    bonafide. В train-партиции LA 22800 подделок против 2580 записей bonafide,
    поэтому конфиги дают классу bonafide вес 22800 / 2580 = 8.84, и ошибка на
    редком классе стоит во столько же раз дороже.
    """

    # аннотация проставлена явно: после одного `register_buffer` тайпчекер
    # видит атрибут как `Tensor | Module`
    class_weights: torch.Tensor | None

    def __init__(self, class_weights: list[float] | None = None):
        """
        Аргументы:
            class_weights (list[float] | None): веса классов в порядке
                [spoof, bonafide]. None означает невзвешенную кросс-энтропию.
        """
        super().__init__()

        weight_tensor = (
            None
            if class_weights is None
            else torch.tensor(list(class_weights), dtype=torch.float32)
        )
        # регистрируется как буфер, чтобы `loss_function.to(device)` его переносил
        self.register_buffer("class_weights", weight_tensor)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        """
        Аргументы:
            logits (Tensor): выход модели формы (B, n_classes).
            labels (Tensor): истинные метки формы (B,).
        Возвращает:
            losses (dict): словарь с ключом 'loss'.
        """
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        return {"loss": loss}
