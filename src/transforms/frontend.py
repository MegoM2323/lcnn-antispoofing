import math
from typing import Optional

import torch
import torchaudio
import torchaudio.functional as F
from torch import nn

CROP_MODES = ("first", "random")
PAD_MODES = ("repeat", "zero")


def fix_frames(
    spec: torch.Tensor, n_frames: int, crop: str, pad_mode: str
) -> torch.Tensor:
    """
    Bring a feature sequence to the fixed number of frames.

    Sequences shorter than n_frames are padded, longer ones are cropped.
    Repeat padding (cyclic repetition of the utterance) is used in the
    ASVspoof literature instead of zero padding, because silence carries
    no spoofing cues and biases the network.

    Args:
        spec (Tensor): features of shape (..., F, T).
        n_frames (int): required number of frames.
        crop (str): "first" to take the leading frames, "random" to take
            a random slice (useful as a train-time augmentation).
        pad_mode (str): "repeat" for cyclic padding, "zero" for zeros.
    Returns:
        spec (Tensor): features of shape (..., F, n_frames).
    """
    n_cur = spec.shape[-1]
    if n_cur == n_frames:
        return spec

    if n_cur < n_frames:
        if pad_mode == "repeat":
            n_repeat = math.ceil(n_frames / n_cur)
            spec = spec.repeat(*([1] * (spec.dim() - 1)), n_repeat)
        else:
            spec = torch.nn.functional.pad(spec, (0, n_frames - n_cur))
        return spec[..., :n_frames]

    start = 0
    if crop == "random":
        start = int(torch.randint(0, n_cur - n_frames + 1, (1,)).item())
    return spec[..., start : start + n_frames]


def as_batched_waveform(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the waveform layout to (B, T).

    Args:
        x (Tensor): waveform of shape (T,), (B, T) or (B, 1, T).
    Returns:
        x (Tensor): waveform of shape (B, T).
    """
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 3 and x.shape[1] == 1:
        return x.squeeze(1)
    if x.dim() != 2:
        raise ValueError(f"Expected waveform of shape (B, T), got {tuple(x.shape)}")
    return x


def autocast_dtype(device_type: str) -> Optional[torch.dtype]:
    """
    Get the dtype the surrounding autocast region expects, if any.

    Args:
        device_type (str): device type of the input tensor.
    Returns:
        dtype (torch.dtype | None): autocast dtype or None if disabled.
    """
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return None


class LogSpectrogram(nn.Module):
    """
    Log power magnitude spectrum front-end for the FFT-LCNN system.

    Follows the STC ASVspoof2019 submission (arXiv:1904.05576, Sec. 2.1):
    1724-point FFT with a Blackman window, 863 frequency bins, no CMVN
    (mean/variance normalization degraded EER in the original study), and
    only the first 600 frames fed to the network.
    """

    def __init__(
        self,
        n_fft: int = 1724,
        win_length: int = 1724,
        hop_length: int = 130,
        window: str = "blackman",
        power: float = 2.0,
        eps: float = 1e-8,
        n_frames: int = 600,
        crop: str = "first",
        pad_mode: str = "repeat",
        center: bool = True,
        stft_pad_mode: str = "reflect",
    ) -> None:
        """
        Args:
            n_fft (int): FFT size, defines n_fft // 2 + 1 frequency bins.
            win_length (int): analysis window length in samples.
            hop_length (int): frame step in samples (0.0081 s at 16 kHz).
            window (str): "blackman", "hann" or "hamming".
            power (float): exponent of the magnitude spectrum (2.0 = power).
            eps (float): additive constant under the logarithm.
            n_frames (int): fixed number of output frames.
            crop (str): "first" or "random", see fix_frames.
            pad_mode (str): "repeat" or "zero", see fix_frames.
            center (bool): whether STFT pads the signal on both sides.
            stft_pad_mode (str): padding mode used by STFT when center is True.
        """
        super().__init__()

        if crop not in CROP_MODES:
            raise ValueError(f"crop must be one of {CROP_MODES}, got {crop}")
        if pad_mode not in PAD_MODES:
            raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode}")
        window_fns = {
            "blackman": torch.blackman_window,
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
        }
        if window not in window_fns:
            raise ValueError(f"window must be one of {tuple(window_fns)}, got {window}")

        self.eps = eps
        self.n_frames = n_frames
        self.crop = crop
        self.pad_mode = pad_mode
        self.n_freqs = n_fft // 2 + 1

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fns[window],
            power=power,
            center=center,
            pad_mode=stft_pad_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): waveform of shape (B, T).
        Returns:
            spec (Tensor): log power spectrum of shape (B, n_freqs, n_frames).
        """
        x = as_batched_waveform(x)
        out_dtype = autocast_dtype(x.device.type)

        # STFT is numerically unstable (and partially unsupported) in bf16/fp16,
        # so the front-end always runs in float32 regardless of the autocast state
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self.spectrogram(x.float())
            spec = torch.log(spec + self.eps)

        spec = fix_frames(spec, self.n_frames, self.crop, self.pad_mode)
        if out_dtype is not None:
            spec = spec.to(out_dtype)
        return spec


class LFCC(nn.Module):
    """
    Linear frequency cepstral coefficients front-end.

    Follows the ASVspoof2019 baseline recipe as described in
    arXiv:2103.11326 (Sec. 3.1): 20 ms frames with 10 ms shift, 512-point
    FFT, 20 linearly spaced triangular filters, the zeroth cepstral
    coefficient replaced by the log spectral energy, plus delta and
    delta-delta features (60 dimensions in total).
    """

    fbanks: torch.Tensor
    dct_mat: torch.Tensor

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 20,
        n_lfcc: int = 20,
        win_length: int = 320,
        hop_length: int = 160,
        n_fft: int = 512,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        with_delta: bool = True,
        with_energy: bool = True,
        delta_win_length: int = 5,
        eps: float = 1e-8,
        n_frames: int = 750,
        crop: str = "first",
        pad_mode: str = "repeat",
        center: bool = True,
    ) -> None:
        """
        Args:
            sample_rate (int): waveform sample rate.
            n_filter (int): number of linearly spaced triangular filters.
            n_lfcc (int): number of cepstral coefficients kept after DCT.
            win_length (int): analysis window length in samples (20 ms).
            hop_length (int): frame step in samples (10 ms).
            n_fft (int): FFT size.
            f_min (float): lowest filterbank frequency in Hz.
            f_max (float | None): highest filterbank frequency, defaults to
                the Nyquist frequency.
            with_delta (bool): append delta and delta-delta coefficients.
            with_energy (bool): replace c0 with the log spectral energy.
            delta_win_length (int): window used by compute_deltas.
            eps (float): additive constant under the logarithms.
            n_frames (int): fixed number of output frames.
            crop (str): "first" or "random", see fix_frames.
            pad_mode (str): "repeat" or "zero", see fix_frames.
            center (bool): whether STFT pads the signal on both sides.
        """
        super().__init__()

        if crop not in CROP_MODES:
            raise ValueError(f"crop must be one of {CROP_MODES}, got {crop}")
        if pad_mode not in PAD_MODES:
            raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode}")
        if n_lfcc > n_filter:
            raise ValueError(f"n_lfcc ({n_lfcc}) cannot exceed n_filter ({n_filter})")

        self.eps = eps
        self.n_frames = n_frames
        self.crop = crop
        self.pad_mode = pad_mode
        self.with_delta = with_delta
        self.with_energy = with_energy
        self.delta_win_length = delta_win_length
        self.n_features = n_lfcc * (3 if with_delta else 1)

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
            center=center,
        )
        fbanks = F.linear_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=float(sample_rate // 2) if f_max is None else f_max,
            n_filter=n_filter,
            sample_rate=sample_rate,
        )
        self.register_buffer("fbanks", fbanks)  # (n_freqs, n_filter)
        self.register_buffer("dct_mat", F.create_dct(n_lfcc, n_filter, norm="ortho"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): waveform of shape (B, T).
        Returns:
            lfcc (Tensor): features of shape (B, n_features, n_frames).
        """
        x = as_batched_waveform(x)
        out_dtype = autocast_dtype(x.device.type)

        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self.spectrogram(x.float())  # (B, n_freqs, T')
            filtered = torch.log(spec.transpose(-1, -2) @ self.fbanks + self.eps)
            lfcc = filtered @ self.dct_mat  # (B, T', n_lfcc)
            lfcc = lfcc.transpose(-1, -2)

            if self.with_energy:
                energy = torch.log(spec.sum(dim=-2) + self.eps)
                lfcc = torch.cat([energy.unsqueeze(-2), lfcc[..., 1:, :]], dim=-2)

            if self.with_delta:
                delta = F.compute_deltas(lfcc, win_length=self.delta_win_length)
                delta2 = F.compute_deltas(delta, win_length=self.delta_win_length)
                lfcc = torch.cat([lfcc, delta, delta2], dim=-2)

        lfcc = fix_frames(lfcc, self.n_frames, self.crop, self.pad_mode)
        if out_dtype is not None:
            lfcc = lfcc.to(out_dtype)
        return lfcc
