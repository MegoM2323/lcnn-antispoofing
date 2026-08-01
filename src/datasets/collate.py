"""
Assembly of dataset items into a batch: every waveform of the batch is brought
to the same number of samples, the remaining fields are stacked as they are.
"""

from collections.abc import Callable
from functools import partial

import torch

# what the configs of the project set 'collate_max_len' to: 77870 samples,
# 4.87 s at 16 kHz
DEFAULT_MAX_LEN = 77870

# the length the waveforms were padded to before 'collate_max_len' appeared in
# the configs (64600 samples, 4.04 s at 16 kHz, the value of the ASVspoof
# recipes). Nothing pads to it any more, it is only needed to reconstruct the
# input of a checkpoint trained back then, see src/trainer/config_check.py
LEGACY_MAX_LEN = 64600


def pad_or_crop(audio: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Bring a waveform to the fixed length of max_len samples.

    Short utterances are repeated cyclically instead of being zero-padded:
    silence at the end of an utterance is an artifact that the model learns
    to use as a shortcut, which is the standard practice for ASVspoof.

    Args:
        audio (Tensor): waveform of shape (T,) or (1, T).
        max_len (int): required number of samples.
    Returns:
        audio (Tensor): waveform of shape (max_len,).
    """
    audio = audio.reshape(-1)

    length = audio.shape[0]
    if length == 0:
        raise ValueError("Cannot collate an empty waveform")

    if length < max_len:
        audio = torch.tile(audio, (max_len // length + 1,))

    return audio[:max_len]


def collate_fn(dataset_items: list[dict], max_len: int = DEFAULT_MAX_LEN) -> dict:
    """
    Collate fields in the dataset items and bring the waveforms to the
    same length. Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
        max_len (int): number of samples in each waveform of the batch.
            The default is DEFAULT_MAX_LEN, 4.87 seconds at 16 kHz.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    if len(dataset_items) == 0:
        raise ValueError("Cannot collate an empty list of dataset items")

    result_batch = {
        "data_object": torch.stack(
            [
                pad_or_crop(elem["data_object"], max_len).float()
                for elem in dataset_items
            ]
        ),
        "labels": torch.tensor(
            [elem["labels"] for elem in dataset_items], dtype=torch.long
        ),
    }

    if "utt_id" in dataset_items[0]:
        result_batch["utt_id"] = [elem["utt_id"] for elem in dataset_items]

    return result_batch


def get_collate_fn(max_len: int = DEFAULT_MAX_LEN) -> Callable[[list[dict]], dict]:
    """
    Create a collate_fn with the given waveform length.

    Used when the length has to be configured from the experiment config,
    since collate_fn itself is passed to the dataloader without arguments.

    Args:
        max_len (int): number of samples in each waveform of the batch.
    Returns:
        collate_fn (Callable): collate function with a fixed max_len.
    """
    if max_len <= 0:
        raise ValueError(f"max_len should be positive, got {max_len}")
    return partial(collate_fn, max_len=max_len)
