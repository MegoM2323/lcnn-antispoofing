"""
Cross-entropy with class weights that compensate the bonafide/spoof imbalance
of the ASVspoof2019 LA train partition.
"""

import torch
from torch import nn


class CELoss(nn.Module):
    """
    Weighted cross-entropy, the main loss for the anti-spoofing classifier.

    Class order follows the dataset convention: index 0 = spoof,
    index 1 = bonafide. The LA train partition holds 22800 spoof against 2580
    bonafide utterances, so the configs give the bonafide class the weight
    22800 / 2580 = 8.84 and an error on the rare class costs that much more.
    """

    # annotated explicitly: `register_buffer` alone leaves the attribute typed
    # as `Tensor | Module` for the type checker
    class_weights: torch.Tensor | None

    def __init__(self, class_weights: list[float] | None = None):
        """
        Args:
            class_weights (list[float] | None): per-class weights ordered as
                [spoof, bonafide]. None means an unweighted cross-entropy.
        """
        super().__init__()

        weight_tensor = (
            None
            if class_weights is None
            else torch.tensor(list(class_weights), dtype=torch.float32)
        )
        # registered as a buffer so that `loss_function.to(device)` moves it
        self.register_buffer("class_weights", weight_tensor)

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor, **batch
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits (Tensor): model output of shape (B, n_classes).
            labels (Tensor): ground-truth labels of shape (B,).
        Returns:
            losses (dict): dict with the 'loss' key.
        """
        loss = nn.functional.cross_entropy(logits, labels, weight=self.class_weights)
        return {"loss": loss}
