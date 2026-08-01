"""
Фронт-энды, превращающие батч сигналов во вход модели: логарифмическая
STFT-спектрограмма и LFCC, оба считаются на устройстве батча.
"""

import math

import torch
import torchaudio
import torchaudio.functional as F
from torch import nn

CROP_MODES = ("first", "random")
PAD_MODES = ("repeat", "zero")

EPS = 1e-8  # аддитивная константа под логарифмами
DELTA_WIN_LENGTH = 5  # окно фильтра delta/delta-delta


def fix_frames(
    spec: torch.Tensor, n_frames: int, crop: str, pad_mode: str
) -> torch.Tensor:
    """
    Приводит последовательность признаков к фиксированному числу фреймов.

    Всё, что короче n_frames, дополняется, всё, что длиннее, обрезается.
    В работах по ASVspoof вместо дополнения нулями принято циклически повторять
    запись: тишина не несёт следов спуфинга и смещает сеть. Параметр crop это
    "first" (первые фреймы, режим инференса) или "random" (аугментация на
    обучении), pad_mode это "repeat" или "zero".
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
    # опечатка здесь не ломает запуск, а молча меняет вход модели
    if crop not in CROP_MODES:
        raise ValueError(f"crop must be one of {CROP_MODES}, got {crop}")
    if pad_mode not in PAD_MODES:
        raise ValueError(f"pad_mode must be one of {PAD_MODES}, got {pad_mode}")


def autocast_dtype(device_type: str) -> torch.dtype | None:
    """
    Тип данных, которого ждёт окружающая область autocast; None, если он
    выключен.
    """
    if torch.is_autocast_enabled(device_type):
        return torch.get_autocast_dtype(device_type)
    return None


class LogSpectrogram(nn.Module):
    """
    Фронт-энд системы FFT-LCNN: логарифм спектра мощности.

    Повторяет решение STC для ASVspoof2019 (arXiv:1904.05576, разд. 2.1):
    1724-точечное FFT с окном Блэкмана, 863 частотных бина, без CMVN
    (нормировка по среднему и дисперсии в исходной работе ухудшала EER) и
    только первые 600 фреймов на вход сети.
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
        Превращает батч сигналов (B, T) в (B, n_freqs, n_frames).
        """
        out_dtype = autocast_dtype(x.device.type)

        # в bf16/fp16 STFT численно неустойчиво (и поддержано не полностью),
        # поэтому фронт-энд всегда считается в float32 независимо от autocast
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = self.spectrogram(x.float())
            spec = torch.log(spec + self.eps)

        spec = fix_frames(spec, self.n_frames, self.crop, self.pad_mode)
        if out_dtype is not None:
            spec = spec.to(out_dtype)
        return spec


class LFCC(nn.Module):
    """
    Фронт-энд на кепстральных коэффициентах в линейной шкале частот.

    Повторяет baseline-рецепт ASVspoof2019 в изложении arXiv:2103.11326
    (разд. 3.1): окно 20 мс с шагом 10 мс, 512-точечное FFT, 20 равномерно
    расставленных треугольных фильтров, нулевой кепстральный коэффициент
    заменён логарифмом энергии спектра, плюс признаки delta и delta-delta
    (всего 60 измерений).
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
        Превращает батч сигналов (B, T) в (B, n_features, n_frames).
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
