from collections.abc import Callable
from functools import partial

import torch

DEFAULT_MAX_LEN = 77870


def pad_or_crop(audio: torch.Tensor, max_len: int) -> torch.Tensor:
    audio = audio.reshape(-1)

    length = audio.shape[0]
    if length < max_len:
        audio = torch.tile(audio, (max_len // length + 1,))

    return audio[:max_len]


def collate_fn(dataset_items: list[dict], max_len: int = DEFAULT_MAX_LEN) -> dict:
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
        "utt_id": [elem["utt_id"] for elem in dataset_items],
    }

    return result_batch


def get_collate_fn(max_len: int = DEFAULT_MAX_LEN) -> Callable[[list[dict]], dict]:
    return partial(collate_fn, max_len=max_len)
