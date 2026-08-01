"""
Light CNN countermeasure: nine convolutional blocks with Max-Feature-Map
activations and a fully connected head over the whole feature map.
"""

import torch
from torch import nn

from src.model.mfm import MFM


def init_weights(module: nn.Module) -> None:
    """
    Kaiming normal initialization for convolutional and linear layers
    (arXiv:1904.05576, Sec. 2.3). Biases are set to zero.
    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class LCNNBackbone(nn.Module):
    """
    Convolutional part of the Light CNN (layers 1-28 of Table 1 in
    arXiv:1904.05576).

    The block sequence is Conv -> MFM -> (MaxPool) -> (BatchNorm), and the
    network is fully convolutional, so the output resolution depends on the
    input one only through the four 2x2 max-pooling layers.
    """

    def __init__(self, in_channels: int = 1):
        super().__init__()

        # The layer numbers below are the row numbers of Table 1 in
        # arXiv:1904.05576, so the block can be read against the paper directly.
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),  # Conv 1
            MFM(32),  # MFM 2
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 3
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # Conv 4
            MFM(32),  # MFM 5
            nn.BatchNorm2d(32),  # BatchNorm 6
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),  # Conv 7
            MFM(48),  # MFM 8
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 9
            nn.BatchNorm2d(48),  # BatchNorm 10
            nn.Conv2d(48, 96, kernel_size=1, stride=1),  # Conv 11
            MFM(48),  # MFM 12
            nn.BatchNorm2d(48),  # BatchNorm 13
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),  # Conv 14
            MFM(64),  # MFM 15
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 16
            nn.Conv2d(64, 128, kernel_size=1, stride=1),  # Conv 17
            MFM(64),  # MFM 18
            nn.BatchNorm2d(64),  # BatchNorm 19
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),  # Conv 20
            MFM(32),  # MFM 21
            nn.BatchNorm2d(32),  # BatchNorm 22
            nn.Conv2d(32, 64, kernel_size=1, stride=1),  # Conv 23
            MFM(32),  # MFM 24
            nn.BatchNorm2d(32),  # BatchNorm 25
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Conv 26
            MFM(32),  # MFM 27
            nn.MaxPool2d(kernel_size=2, stride=2),  # MaxPool 28
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LCNN(nn.Module):
    """
    Light CNN for voice anti-spoofing, exactly as in Table 1 of
    arXiv:1904.05576: the backbone feature map is flattened and goes through
    FC_29 -> MFM_30 -> Dropout -> BatchNorm_31 -> FC_32.
    """

    def __init__(
        self,
        in_freq: int = 863,
        in_frames: int = 600,
        n_class: int = 2,
        dropout: float = 0.75,
        embedding_dim: int = 160,
    ):
        """
        Args:
            in_freq (int): number of frequency bins of the input feature.
            in_frames (int): number of time frames of the input feature.
            n_class (int): number of classes (2 for bonafide/spoof).
            dropout (float): dropout probability applied before the final
                BatchNorm, as required by the task statement.
            embedding_dim (int): number of FC_29 output features. The final
                embedding is twice smaller because of MFM_30.
        """
        super().__init__()

        self.backbone = LCNNBackbone()
        pooled_dim = self._flattened_dim(in_freq, in_frames)

        # Rows 29-32 of Table 1 in arXiv:1904.05576, with the dropout inserted
        # where the task statement asks for it (Table 1 has no dropout row).
        self.fc = nn.Linear(pooled_dim, embedding_dim)  # FC 29
        self.mfm_fc = MFM(embedding_dim // 2)  # MFM 30
        self.dropout = nn.Dropout(p=dropout)  # Dropout, before the final BatchNorm
        self.bn = nn.BatchNorm1d(embedding_dim // 2)  # BatchNorm 31
        self.classifier = nn.Linear(embedding_dim // 2, n_class)  # FC 32

        self.apply(init_weights)

    def _flattened_dim(self, in_freq: int, in_frames: int) -> int:
        """
        Number of features FC_29 gets, measured by running a dummy tensor
        through the backbone.

        A real pass (instead of a formula) keeps the head size correct for any
        front-end and, unlike lazy layers, is available before the first
        forward, so checkpoints can be loaded right after construction.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            shape = self.backbone(torch.zeros(1, 1, in_freq, in_frames)).shape
        self.backbone.train(was_training)

        return int(shape[1] * shape[2] * shape[3])

    def forward(self, data_object: torch.Tensor, **batch) -> dict:
        """
        Turn the input feature, (B, F, T) or (B, 1, F, T), into the logits of
        the two classes.
        """
        if data_object.dim() == 3:
            data_object = data_object.unsqueeze(1)

        features = self.backbone(data_object)
        embedding = self.mfm_fc(self.fc(torch.flatten(features, start_dim=1)))
        embedding = self.bn(self.dropout(embedding))

        return {"logits": self.classifier(embedding)}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
