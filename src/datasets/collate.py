"""
Сборка элементов датасета в батч: каждый сигнал приводится к одному и тому же
числу отсчётов, остальные поля складываются как есть.
"""

from collections.abc import Callable
from functools import partial

import torch

# значение, которое конфиги проекта ставят в 'collate_max_len': 77870 отсчётов,
# то есть 4,87 с при 16 кГц
DEFAULT_MAX_LEN = 77870


def pad_or_crop(audio: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Приводит сигнал к фиксированной длине в max_len отсчётов.

    Короткие записи повторяются циклически, а не дополняются нулями: тишина
    в конце записи это артефакт, который модель научается использовать как
    подсказку. Так принято делать в работах по ASVspoof.

    Аргументы:
        audio (Tensor): сигнал формы (T,) или (1, T).
        max_len (int): нужное число отсчётов.
    Возвращает:
        audio (Tensor): сигнал формы (max_len,).
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
    Складывает поля элементов датасета и приводит сигналы к одной длине,
    превращая отдельные элементы в батч.

    Аргументы:
        dataset_items (list[dict]): список объектов из dataset.__getitem__.
        max_len (int): число отсчётов в каждом сигнале батча.
    Возвращает:
        result_batch (dict[Tensor]): словарь с батчевыми версиями тензоров.
    """
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
    """
    Создаёт collate_fn с заданной длиной сигнала, поскольку сама collate_fn
    передаётся в даталоадер без аргументов.
    """
    return partial(collate_fn, max_len=max_len)
