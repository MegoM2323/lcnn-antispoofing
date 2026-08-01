"""
Front-ends that turn a batch of waveforms into the input of the model: the log
STFT spectrogram and the LFCC, both computed on the device of the batch.
"""

import math

import torch
import torchaudio
import torchaudio.functional as F
from torch import nn

CROP_MODES = ("first", "random")
PAD_MODES = ("repeat", "zero")

EPS = 1e-8  # additive constant under the logarithms
DELTA_WIN_LENGTH = 5  # window of the delta/delta-delta filter


def fix_frames(
    spec: torch.Tensor, n_frames: int, crop: str, pad_mode: str
) -> torch.Tensor:
    """
    Bring a feature sequence to the fixed number of frames.

    Sequences shorter than n_frames are padded, longer ones are cropped.
    Repeat padding (cyclic repetition of the utterance) is used in the
    ASVspoof literature instead of zero padding, because silence carries
    no spoofing cues and biases the network. 'crop' is "first" (the leading
    frames, used at inference) or "random" (a train-time augmentation),
    'pad_mode' is "repeat" or "zero".
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


def validate_frame_modes(crop: str, pad_mode: str) -> None:
    # a typo here does not fail, it silently changes the input of the model
    if crop not in CROP_MODES:
        raise ValueError(f"crop must be one of {CROP_MODES}, got {crop}")
    if pad_mode not in PAD_MODES:
        raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode}")


def autocast_dtype(device_type: str) -> torch.dtype | None:
    """
    Dtype the surrounding autocast region expects, None if it is disabled.
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
        eps: float = EPS,
        n_frames: int = 600,
        crop: str = "first",
        pad_mode: str = "repeat",
    ) -> None:
        super().__init__()

        validate_frame_modes(crop, pad_mode)
        window_fns = {
            "blackman": torch.blackman_window,
            "hann": torch.hann_window,
            "hamming": torch.hamming_window,
        }

        self.eps = eps
        self.n_frames = n_frames
        self.crop = crop
        self.pad_mode = pad_mode

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window_fn=window_fns[window],
            power=power,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Turn a batch of waveforms (B, T) into (B, n_freqs, n_frames).
        """
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
        with_delta: bool = True,
        with_energy: bool = True,
        n_frames: int = 750,
        crop: str = "first",
        pad_mode: str = "repeat",
    ) -> None:
        super().__init__()

        validate_frame_modes(crop, pad_mode)

        self.n_frames = n_frames
        self.crop = crop
        self.pad_mode = pad_mode
        self.with_delta = with_delta
        self.with_energy = with_energy

        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
        )
        fbanks = F.linear_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=0.0,
            f_max=float(sample_rate // 2),
            n_filter=n_filter,
            sample_rate=sample_rate,
        )
        self.register_buffer("fbanks", fbanks)  # (n_freqs, n_filter)
        self.register_buffer("dct_mat", F.create_dct(n_lfcc, n_filter, norm="ortho"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Turn a batch of waveforms (B, T) into (B, n_features, n_frames).
        """
        out_dtype = autocast_dtype(x.device.type)

        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self.spectrogram(x.float())  # (B, n_freqs, T')
            filtered = torch.log(spec.transpose(-1, -2) @ self.fbanks + EPS)
            lfcc = filtered @ self.dct_mat  # (B, T', n_lfcc)
            lfcc = lfcc.transpose(-1, -2)

            if self.with_energy:
                energy = torch.log(spec.sum(dim=-2) + EPS)
                lfcc = torch.cat([energy.unsqueeze(-2), lfcc[..., 1:, :]], dim=-2)

            if self.with_delta:
                delta = F.compute_deltas(lfcc, win_length=DELTA_WIN_LENGTH)
                delta2 = F.compute_deltas(delta, win_length=DELTA_WIN_LENGTH)
                lfcc = torch.cat([lfcc, delta, delta2], dim=-2)

        lfcc = fix_frames(lfcc, self.n_frames, self.crop, self.pad_mode)
        if out_dtype is not None:
            lfcc = lfcc.to(out_dtype)
        return lfcc
