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

    Args:
        module (nn.Module): module to initialize (used with apply).
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
        """
        Args:
            in_channels (int): number of input channels (1 for a spectrogram).
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2),
            MFM(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            MFM(32),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(48),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, stride=1),
            MFM(48),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            MFM(64),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(32),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            MFM(32),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): input of shape (B, in_channels, F, T).
        Returns:
            output (Tensor): feature map of shape (B, 32, F', T').
        """
        return self.net(x)


class LCNNBase(nn.Module):
    """
    Base class for LCNN-based anti-spoofing models.

    It owns the convolutional backbone and the classification head
    (FC_29 -> MFM_30 -> Dropout -> BatchNorm_31 -> FC_32 of Table 1).
    Subclasses only define how the backbone feature map is pooled into
    a single vector via build_pooling and pool_features.
    """

    def __init__(
        self,
        in_freq: int = 863,
        in_frames: int = 600,
        n_class: int = 2,
        dropout: float = 0.75,
        embedding_dim: int = 160,
        return_embedding: bool = False,
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
            return_embedding (bool): if True, the forward pass additionally
                returns the embedding (needed by margin-based losses).
        """
        super().__init__()

        self.return_embedding = return_embedding

        self.backbone = LCNNBackbone()
        channels, freq, frames = self._conv_output_shape(in_freq, in_frames)

        pooled_dim = self.build_pooling(channels, freq, frames)

        self.fc = nn.Linear(pooled_dim, embedding_dim)
        self.mfm_fc = MFM(embedding_dim // 2, mfm_type=2)
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(embedding_dim // 2)
        self.classifier = nn.Linear(embedding_dim // 2, n_class)

        self.apply(init_weights)

    def _conv_output_shape(self, in_freq: int, in_frames: int) -> tuple[int, int, int]:
        """
        Compute the backbone output shape by running a dummy tensor through it.

        A real pass (instead of a formula) keeps the head size correct for any
        front-end and, unlike lazy layers, is available before the first
        forward, so checkpoints can be loaded right after construction.

        Args:
            in_freq (int): number of input frequency bins.
            in_frames (int): number of input time frames.
        Returns:
            shape (tuple): (channels, freq, frames) of the feature map.
        """
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_freq, in_frames)
            shape = self.backbone(dummy).shape
        self.backbone.train(was_training)

        return int(shape[1]), int(shape[2]), int(shape[3])

    def build_pooling(self, channels: int, freq: int, frames: int) -> int:
        """
        Create the layers that turn the backbone feature map into a vector.

        Args:
            channels (int): number of backbone output channels.
            freq (int): backbone output frequency resolution.
            frames (int): backbone output time resolution.
        Returns:
            pooled_dim (int): number of features fed into FC_29.
        """
        raise NotImplementedError()

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): backbone output of shape (B, C, F', T').
        Returns:
            pooled (Tensor): tensor of shape (B, pooled_dim).
        """
        raise NotImplementedError()

    @staticmethod
    def prepare_input(data_object: torch.Tensor) -> torch.Tensor:
        """
        Add the channel dimension if the input comes as (B, F, T).

        Args:
            data_object (Tensor): input feature, (B, F, T) or (B, 1, F, T).
        Returns:
            output (Tensor): input of shape (B, 1, F, T).
        """
        if data_object.dim() == 3:
            return data_object.unsqueeze(1)
        if data_object.dim() == 4:
            return data_object
        raise ValueError(
            f"Expected a 3D or 4D input tensor, got shape {tuple(data_object.shape)}"
        )

    def get_embedding(self, data_object: torch.Tensor) -> torch.Tensor:
        """
        Compute the speaker-independent utterance embedding
        (everything up to and including BatchNorm_31).

        Args:
            data_object (Tensor): input feature, (B, F, T) or (B, 1, F, T).
        Returns:
            embedding (Tensor): embedding of shape (B, embedding_dim // 2).
        """
        features = self.backbone(self.prepare_input(data_object))
        pooled = self.pool_features(features)

        embedding = self.mfm_fc(self.fc(pooled))
        embedding = self.dropout(embedding)

        return self.bn(embedding)

    def forward(self, data_object: torch.Tensor, **batch) -> dict:
        """
        Model forward method.

        Args:
            data_object (Tensor): input feature, (B, F, T) or (B, 1, F, T).
        Returns:
            output (dict): output dict containing logits and, if
                return_embedding is set, the embedding.
        """
        embedding = self.get_embedding(data_object)
        output = {"logits": self.classifier(embedding)}

        if self.return_embedding:
            output["embedding"] = embedding

        return output

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


class LCNN(LCNNBase):
    """
    Light CNN for voice anti-spoofing, exactly as in Table 1 of
    arXiv:1904.05576: the backbone feature map is flattened and fed
    into FC_29.
    """

    def build_pooling(self, channels: int, freq: int, frames: int) -> int:
        """
        Flatten the whole feature map, as in Table 1 of the paper.
        """
        return channels * freq * frames

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): backbone output of shape (B, C, F', T').
        Returns:
            pooled (Tensor): flattened feature map, (B, C * F' * T').
        """
        return torch.flatten(features, start_dim=1)
