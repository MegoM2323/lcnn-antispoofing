import torch
from torch import nn

from src.model.lcnn import LCNNBase


class LCNNAttention(LCNNBase):
    """
    LCNN with single-head attentive pooling instead of the huge FC_29 input
    (arXiv:2103.11326, Sec. 3.1).

    The backbone feature map is averaged over the frequency axis and the
    resulting frame-level sequence is aggregated over time with learned
    attention weights, which makes the head independent of the input length.
    """

    def __init__(self, *args, attention_dim: int = 64, **kwargs):
        """
        Args:
            attention_dim (int): hidden size of the attention scoring network.
        """
        self.attention_dim = attention_dim
        super().__init__(*args, **kwargs)

    def build_pooling(self, channels: int, freq: int, frames: int) -> int:
        """
        Attention pooling over time: the head stops depending on the input
        length.
        """
        self.attention = nn.Sequential(
            nn.Linear(channels, self.attention_dim),
            nn.Tanh(),
            nn.Linear(self.attention_dim, 1),
        )

        return channels

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): backbone output of shape (B, C, F', T').
        Returns:
            pooled (Tensor): attention-pooled features, (B, C).
        """
        # (B, C, F', T') -> (B, T', C): average over frequency, time last
        sequence = features.mean(dim=2).transpose(1, 2)

        weights = torch.softmax(self.attention(sequence), dim=1)

        return (weights * sequence).sum(dim=1)


class LCNNLSTMSum(LCNNBase):
    """
    LCNN followed by residual Bi-LSTM layers and average pooling over time
    (arXiv:2103.11326, Sec. 3.1), the best-performing variant of the paper.

    The backbone feature map is averaged over the frequency axis, projected
    to the recurrent width (so that the skip connections are dimension-wise
    valid), processed by two Bi-LSTM layers with residual connections and
    summarized by an average over time.
    """

    def __init__(self, *args, hidden_size: int = 64, num_layers: int = 2, **kwargs):
        """
        Args:
            hidden_size (int): hidden size of each Bi-LSTM direction. The
                recurrent width is 2 * hidden_size because of bidirectionality.
            num_layers (int): number of residual Bi-LSTM layers.
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        super().__init__(*args, **kwargs)

    def build_pooling(self, channels: int, freq: int, frames: int) -> int:
        """
        Two residual Bi-LSTM layers, then an average over time.
        """
        recurrent_dim = 2 * self.hidden_size

        self.input_proj = nn.Linear(channels, recurrent_dim)

        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=recurrent_dim,
                    hidden_size=self.hidden_size,
                    batch_first=True,
                    bidirectional=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        return recurrent_dim

    def pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features (Tensor): backbone output of shape (B, C, F', T').
        Returns:
            pooled (Tensor): time-averaged recurrent features,
                (B, 2 * hidden_size).
        """
        # (B, C, F', T') -> (B, T', C): average over frequency, time last
        sequence = features.mean(dim=2).transpose(1, 2)
        sequence = self.input_proj(sequence)

        for lstm in self.lstm_layers:
            output, _ = lstm(sequence)
            sequence = sequence + output

        return sequence.mean(dim=1)
