import torch
from torch import nn


class MFM(nn.Module):
    """
    Max-Feature-Map activation (MFM 2/1).

    Splits the input into two equal halves along the channel (feature)
    dimension and takes the element-wise maximum of them:
    y^k = max(x^k, x^{k + N}), where N is the number of output channels.
    See Eq. 1 of arXiv:1511.02683.

    Two types are supported (as in the original paper):
        * type 1: applied after a convolution, input is 4D (B, 2N, F, T);
        * type 2: applied after a fully-connected layer, input is 2D (B, 2N).
    """

    def __init__(self, out_channels: int, mfm_type: int = 1):
        """
        Args:
            out_channels (int): number of output channels/features (N).
                The layer expects 2 * out_channels input channels.
            mfm_type (int): 1 for convolutional MFM (4D input),
                2 for fully-connected MFM (2D input).
        """
        super().__init__()

        if mfm_type not in (1, 2):
            raise ValueError(f"mfm_type must be 1 or 2, got {mfm_type}")

        self.out_channels = out_channels
        self.mfm_type = mfm_type
        self.expected_ndim = 4 if mfm_type == 1 else 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the max-feature-map activation.

        Args:
            x (Tensor): input tensor with 2 * out_channels channels.
        Returns:
            output (Tensor): tensor with out_channels channels.
        """
        if x.dim() != self.expected_ndim:
            raise ValueError(
                f"MFM of type {self.mfm_type} expects a {self.expected_ndim}D "
                f"tensor, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != 2 * self.out_channels:
            raise ValueError(
                f"MFM expects {2 * self.out_channels} channels, got {x.shape[1]}"
            )

        first, second = torch.chunk(x, 2, dim=1)
        return torch.max(first, second)

    def extra_repr(self) -> str:
        return f"out_channels={self.out_channels}, mfm_type={self.mfm_type}"
