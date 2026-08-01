"""
Cutting an utterance into several fixed-length segments for multi-crop
inference.

The FFT-LCNN system takes exactly 600 spectrogram frames, which is 77870
samples of waveform, and the scoring path of arXiv:1904.05576 keeps "only the
first 600 features for each file". Everything after the first 4.87 s is thus
thrown away, although the synthesis artifacts are not distributed uniformly
over an utterance. This module produces the segments of one utterance; scoring
them separately and pooling the scores is a test-time trick that needs no
retraining and leaves the architecture untouched.

Utterances shorter than one window keep the current behaviour exactly: a single
repeat-padded segment, the very tensor the plain collate function builds.
"""

from collections.abc import Callable
from functools import partial

import torch

from src.datasets.collate import DEFAULT_MAX_LEN, pad_or_crop

DEFAULT_N_SEGMENTS = 3
DEFAULT_SHORT_SHIFTS = 1


def segment_starts(length: int, window: int, n_segments: int) -> list[int]:
    """
    Choose where the segments of an utterance begin.

    The starts are spread evenly over the whole utterance, so the first segment
    begins at sample 0 and the last one ends at the last sample: no part of the
    recording is systematically preferred. On a short utterance the requested
    number of windows does not fit without overlap, and the segments simply
    overlap, which is still useful because a shifted window gives the front-end
    another set of frames. Coinciding starts are dropped: identical segments
    would only cost a forward pass.

    Args:
        length (int): number of samples in the utterance.
        window (int): number of samples in one segment.
        n_segments (int): number of segments to place, at most.
    Returns:
        starts (list[int]): sorted first sample of every segment. A single
            zero for an utterance not longer than the window.
    """
    if length <= 0:
        raise ValueError(f"length should be positive, got {length}")
    if window <= 0:
        raise ValueError(f"window should be positive, got {window}")
    if n_segments <= 0:
        raise ValueError(f"n_segments should be positive, got {n_segments}")

    if length <= window or n_segments == 1:
        return [0]

    span = length - window
    starts = {round(index * span / (n_segments - 1)) for index in range(n_segments)}
    return sorted(starts)


def shifted_segments(audio: torch.Tensor, window: int, n_shifts: int) -> torch.Tensor:
    """
    Build several windows of an utterance shorter than one window.

    Such an utterance is repeated cyclically until the window is full, so the
    single segment the scoring pass sees is one point of a periodic signal.
    Starting the repetition at another sample gives another set of spectrogram
    frames of the very same recording, without inventing a sample that is not
    in it: the model sees the same audio content, cut at another phase. The
    number of the seams where the end of the recording meets its beginning does
    not change, only their position does.

    Args:
        audio (Tensor): waveform of shape (T,), not longer than the window.
        window (int): number of samples in one segment.
        n_shifts (int): number of starting points, evenly spread over the
            utterance.
    Returns:
        segments (Tensor): float tensor of shape (n, window), n <= n_shifts.
    """
    length = audio.shape[0]
    # rounding can land on the length itself, which is the unshifted window
    # again: a coinciding offset would only cost another forward pass
    offsets = sorted(
        {round(index * length / n_shifts) % length for index in range(n_shifts)}
    )
    return torch.stack(
        [pad_or_crop(torch.roll(audio, -offset), window) for offset in offsets]
    )


def split_segments(
    audio: torch.Tensor,
    window: int,
    n_segments: int,
    short_shifts: int = DEFAULT_SHORT_SHIFTS,
) -> torch.Tensor:
    """
    Cut a waveform into fixed-length segments.

    Args:
        audio (Tensor): waveform of shape (T,) or (1, T).
        window (int): number of samples in one segment.
        n_segments (int): number of segments to cut, at most.
        short_shifts (int): windows to build for an utterance that is not
            longer than one window, see 'shifted_segments'. The default of 1
            keeps the single repeat-padded segment of the ordinary pipeline.
    Returns:
        segments (Tensor): float tensor of shape (n, window), where n is
            'short_shifts' at most for an utterance not longer than the window.
    """
    if short_shifts <= 0:
        raise ValueError(f"short_shifts should be positive, got {short_shifts}")

    audio = audio.reshape(-1)
    if audio.shape[0] <= window:
        if short_shifts == 1:
            return pad_or_crop(audio, window).unsqueeze(0)
        return shifted_segments(audio, window, short_shifts)

    starts = segment_starts(audio.shape[0], window, n_segments)
    return torch.stack([audio[start : start + window] for start in starts])


def multicrop_collate_fn(
    dataset_items: list[dict],
    max_len: int = DEFAULT_MAX_LEN,
    n_segments: int = DEFAULT_N_SEGMENTS,
    short_shifts: int = DEFAULT_SHORT_SHIFTS,
) -> dict:
    """
    Collate a batch in which every utterance contributes several segments.

    The segments of the whole batch are stacked into one tensor, so a forward
    pass costs the same as on an ordinary batch of the same number of windows;
    'segment_sizes' says how many rows belong to each utterance and is what the
    pooling of the scores is driven by. The labels and the ids stay per
    utterance, one entry per element of 'dataset_items'.

    Args:
        dataset_items (list[dict]): objects from dataset.__getitem__, with
            waveforms that were not cut to the model input length.
        max_len (int): number of samples in one segment.
        n_segments (int): number of segments per utterance, at most.
        short_shifts (int): windows per utterance that is shorter than one
            segment, see 'shifted_segments'.
    Returns:
        result_batch (dict): "data_object" of shape (sum(segment_sizes),
            max_len), "labels" of shape (B,), "segment_sizes" (list[int]) and
            the utterance ids if the dataset provides them.
    """
    if len(dataset_items) == 0:
        raise ValueError("Cannot collate an empty list of dataset items")

    segments = [
        split_segments(elem["data_object"], max_len, n_segments, short_shifts).float()
        for elem in dataset_items
    ]

    result_batch = {
        "data_object": torch.cat(segments),
        "labels": torch.tensor(
            [elem["labels"] for elem in dataset_items], dtype=torch.long
        ),
        "segment_sizes": [int(item.shape[0]) for item in segments],
    }

    if "utt_id" in dataset_items[0]:
        result_batch["utt_id"] = [elem["utt_id"] for elem in dataset_items]

    return result_batch


def get_multicrop_collate_fn(
    max_len: int = DEFAULT_MAX_LEN,
    n_segments: int = DEFAULT_N_SEGMENTS,
    short_shifts: int = DEFAULT_SHORT_SHIFTS,
) -> Callable[[list[dict]], dict]:
    """
    Create a multi-crop collate_fn with the given segment length and count.

    Args:
        max_len (int): number of samples in one segment.
        n_segments (int): number of segments per utterance, at most.
        short_shifts (int): windows per utterance that is shorter than one
            segment, see 'shifted_segments'.
    Returns:
        collate_fn (Callable): collate function with fixed segmentation.
    """
    if max_len <= 0:
        raise ValueError(f"max_len should be positive, got {max_len}")
    if n_segments <= 0:
        raise ValueError(f"n_segments should be positive, got {n_segments}")
    if short_shifts <= 0:
        raise ValueError(f"short_shifts should be positive, got {short_shifts}")
    return partial(
        multicrop_collate_fn,
        max_len=max_len,
        n_segments=n_segments,
        short_shifts=short_shifts,
    )
