import torch
from torch import nn

# ASVspoof2019 LA train partition: 22800 spoof (label 0) vs 2580 bonafide
# (label 1), i.e. a ~1:8.84 imbalance.
ASVSPOOF19_LA_TRAIN_COUNTS: list[float] = [22800, 2580]


def balanced_class_weights(class_counts: list[float]) -> list[float]:
    """
    Compute inverse-frequency class weights (the "balanced" scheme used by
    sklearn)::

        w_c = N / (n_classes * count_c),    N = sum(count_c)

    With the ASVspoof2019 LA train counts [22800, 2580] this gives
    [0.5566, 4.9186]: a bonafide error costs ~8.84x more than a spoof error,
    which compensates the imbalance exactly.

    Args:
        class_counts (list[float]): number of samples per class, ordered by
            class index ([spoof, bonafide]).
    Returns:
        weights (list[float]): per-class weights.
    """
    if not class_counts or any(count <= 0 for count in class_counts):
        raise ValueError(f"class_counts must be positive, got {class_counts}")
    total = float(sum(class_counts))
    n_classes = len(class_counts)
    return [total / (n_classes * float(count)) for count in class_counts]


class CELoss(nn.Module):
    """
    Weighted cross-entropy, the main loss for the anti-spoofing classifier.

    Class order follows the dataset convention: index 0 = spoof,
    index 1 = bonafide.
    """

    # annotated explicitly: `register_buffer` alone leaves the attribute typed
    # as `Tensor | Module` for the type checker
    class_weights: torch.Tensor | None

    def __init__(
        self,
        class_weights: list[float] | None = None,
        label_smoothing: float = 0.0,
        auto_weight: bool = False,
        class_counts: list[float] | None = None,
    ):
        """
        Args:
            class_weights (list[float] | None): explicit per-class weights
                ordered as [spoof, bonafide]. ``None`` (default) means an
                unweighted cross-entropy.
            label_smoothing (float): label smoothing factor in [0, 1).
            auto_weight (bool): if True, the weights are derived from
                ``class_counts`` via ``balanced_class_weights`` instead of
                being taken from ``class_weights``. Mutually exclusive with
                a non-None ``class_weights``.
            class_counts (list[float] | None): per-class sample counts used by
                ``auto_weight``. Defaults to the ASVspoof2019 LA train counts
                [22800, 2580].
        """
        super().__init__()
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {label_smoothing}"
            )
        if auto_weight and class_weights is not None:
            raise ValueError("Pass either class_weights or auto_weight, not both")

        if auto_weight:
            counts = (
                class_counts if class_counts is not None else ASVSPOOF19_LA_TRAIN_COUNTS
            )
            class_weights = balanced_class_weights(counts)

        if class_weights is None:
            weight_tensor = None
        else:
            weight_tensor = torch.tensor(list(class_weights), dtype=torch.float32)
            if (weight_tensor < 0).any():
                raise ValueError(
                    f"class_weights must be non-negative, got {class_weights}"
                )

        # registered as a buffer so that `loss_function.to(device)` moves it
        self.register_buffer("class_weights", weight_tensor)
        self.label_smoothing = label_smoothing

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
        loss = nn.functional.cross_entropy(
            logits,
            labels,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )
        return {"loss": loss}
