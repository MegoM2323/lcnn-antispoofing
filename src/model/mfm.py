import torch
from torch import nn


class MFM(nn.Module):
    """
    Активация Max-Feature-Map (MFM 2/1).

    Вход делится пополам по канальному (признаковому) измерению, и берётся
    поэлементный максимум половин: y^k = max(x^k, x^{k + N}), где N это число
    выходных каналов. См. формулу 1 в arXiv:1511.02683. Одна и та же операция
    покрывает оба типа из исходной статьи: тип 1 после свёртки (4D вход,
    B x 2N x F x T) и тип 2 после полносвязного слоя (2D вход, B x 2N).
    """

    def __init__(self, out_channels: int):
        """
        Аргументы:
            out_channels (int): число выходных каналов/признаков (N).
                На входе слой ожидает 2 * out_channels каналов.
        """
        super().__init__()

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # иначе неверное число каналов молча уполовинит слой
        if x.shape[1] != 2 * self.out_channels:
            raise ValueError(
                f"MFM expects {2 * self.out_channels} channels, got {x.shape[1]}"
            )

        first, second = torch.chunk(x, 2, dim=1)
        return torch.max(first, second)

    def extra_repr(self) -> str:
        return f"out_channels={self.out_channels}"
