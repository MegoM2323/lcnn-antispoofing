import torch
from torch import nn


class MFM(nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != 2 * self.out_channels:
            raise ValueError(
                f"MFM expects {2 * self.out_channels} channels, got {x.shape[1]}"
            )

        first, second = torch.chunk(x, 2, dim=1)
        return torch.max(first, second)

    def extra_repr(self) -> str:
        return f"out_channels={self.out_channels}"
