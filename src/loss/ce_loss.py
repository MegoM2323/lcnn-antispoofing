import torch
from torch import nn


class CELoss(nn.Module):
    class_weights: torch.Tensor | None

    def __init__(self, class_weights: list[float] | None = None):
        super().__init__()

        weight_tensor = (
            None
            if class_weights is None
            else torch.tensor(list(class_weights), dtype=torch.float32)
        )
        self.register_buffer("class_weights", weight_tensor)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        return {"loss": loss}
