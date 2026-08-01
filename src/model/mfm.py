import torch
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map activation (MFM 2/1).

    Splits the input into two equal halves along the channel (feature)
    dimension and takes the element-wise maximum of them:
    y^k = max(x^k, x^{k + N}), where N is the number of output channels.
    See Eq. 1 of arXiv:1511.02683. The same operation covers both types of the
    original paper: type 1 after a convolution (4D input, B x 2N x F x T) and
    type 2 after a fully connected layer (2D input, B x 2N).
    """

    def __init__(self, out_channels: int):
        """
        Args:
            out_channels (int): number of output channels/features (N).
                The layer expects 2 * out_channels input channels.
        """
        super().__init__()

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # a wrong number of channels is otherwise a silent halving of the layer
        if x.shape[1] != 2 * self.out_channels:
            raise ValueError(
                f"MFM expects {2 * self.out_channels} channels, got {x.shape[1]}"
            )

        first, second = torch.chunk(x, 2, dim=1)
        return torch.max(first, second)

    def extra_repr(self) -> str:
        return f"out_channels={self.out_channels}"
