import torch
import torchaudio
from torch import nn


class SpecAugment(nn.Module):
    """
    SpecAugment-style masking applied to a batch of spectrograms.

    Neither of the reference anti-spoofing papers uses augmentation, so this
    module is disabled by default in the configs and is only meant as a
    regularizer for the ~10M parameter LCNN trained on 25k utterances.
    Masking is a no-op outside of training mode.
    """

    def __init__(
        self,
        freq_mask_param: int = 24,
        time_mask_param: int = 40,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        p: float = 0.5,
    ) -> None:
        """
        Args:
            freq_mask_param (int): maximum width of a frequency mask.
            time_mask_param (int): maximum width of a time mask.
            n_freq_masks (int): number of frequency masks to apply.
            n_time_masks (int): number of time masks to apply.
            p (float): probability of applying the augmentation to a batch.
        """
        super().__init__()

        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        self.p = p
        self.freq_masks = nn.ModuleList(
            torchaudio.transforms.FrequencyMasking(freq_mask_param, iid_masks=True)
            for _ in range(n_freq_masks)
        )
        self.time_masks = nn.ModuleList(
            torchaudio.transforms.TimeMasking(time_mask_param, iid_masks=True)
            for _ in range(n_time_masks)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): spectrogram of shape (B, F, T).
        Returns:
            x (Tensor): augmented spectrogram of the same shape.
        """
        if not self.training or self.p == 0.0:
            return x
        if torch.rand(1).item() >= self.p:
            return x

        # iid_masks requires an explicit channel dimension: (B, 1, F, T)
        x = x.unsqueeze(1)
        for mask in self.freq_masks:
            x = mask(x)
        for mask in self.time_masks:
            x = mask(x)
        return x.squeeze(1)


class RandomGain(nn.Module):
    """
    Multiply each waveform in a batch by a random gain.

    Applied to raw audio before the front-end. Disabled by default in the
    configs to keep the baseline run faithful to the reference papers.
    """

    def __init__(self, min_db: float = -6.0, max_db: float = 6.0, p: float = 1.0):
        """
        Args:
            min_db (float): lower bound of the gain in decibels.
            max_db (float): upper bound of the gain in decibels.
            p (float): probability of applying the augmentation to a batch.
        """
        super().__init__()

        if min_db > max_db:
            raise ValueError(f"min_db ({min_db}) must not exceed max_db ({max_db})")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        self.min_db = min_db
        self.max_db = max_db
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): waveform of shape (B, T).
        Returns:
            x (Tensor): scaled waveform of the same shape.
        """
        if not self.training or self.p == 0.0:
            return x
        if torch.rand(1).item() >= self.p:
            return x

        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        gain_db = torch.empty(shape, device=x.device, dtype=x.dtype)
        gain_db.uniform_(self.min_db, self.max_db)
        return x * torch.pow(10.0, gain_db / 20.0)
