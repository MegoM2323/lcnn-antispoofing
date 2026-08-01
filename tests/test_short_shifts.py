"""
Nine of ten eval utterances are shorter than the 4.87 s window of the model, so
cutting the long ones into several segments leaves most of the partition scored
exactly once. A short utterance is repeated cyclically to fill the window, and
starting that repetition at another sample gives the model another view of the
very same recording. The requirement is that this is a view and not a new
recording: every sample of the window must come from the utterance, the multi-
set of samples must not change, and the default must keep the single
repeat-padded segment the ordinary scoring pass produces.
"""

import pytest
import torch

from src.datasets.collate import pad_or_crop
from src.datasets.multicrop import (
    get_multicrop_collate_fn,
    multicrop_collate_fn,
    shifted_segments,
    split_segments,
)

WINDOW = 8


def audio(length):
    return torch.arange(1, length + 1, dtype=torch.float32)


def item(length, label=0, utt_id="LA_E_1000000"):
    return {"data_object": audio(length), "labels": label, "utt_id": utt_id}


def test_default_keeps_the_single_repeat_padded_segment():
    segments = split_segments(audio(3), WINDOW, n_segments=5)

    assert segments.shape == (1, WINDOW)
    assert torch.equal(segments[0], pad_or_crop(audio(3), WINDOW))


def test_shifts_produce_one_window_each():
    segments = split_segments(audio(4), WINDOW, n_segments=1, short_shifts=4)

    assert segments.shape == (4, WINDOW)


def test_the_first_shift_is_the_ordinary_segment():
    segments = split_segments(audio(5), WINDOW, n_segments=1, short_shifts=3)

    assert torch.equal(segments[0], pad_or_crop(audio(5), WINDOW))


def test_every_shift_is_a_different_view():
    segments = split_segments(audio(5), WINDOW, n_segments=1, short_shifts=5)

    views = {tuple(segment.tolist()) for segment in segments}
    assert len(views) == len(segments)


def test_shifts_invent_no_samples():
    source = set(audio(5).tolist())

    segments = split_segments(audio(5), WINDOW, n_segments=1, short_shifts=5)

    assert set(segments.reshape(-1).tolist()) <= source


def test_a_shift_is_a_rotation_of_the_padded_window():
    segments = shifted_segments(audio(4), WINDOW, n_shifts=4)
    ordinary = pad_or_crop(audio(4), WINDOW)

    for shift, segment in enumerate(segments):
        assert torch.equal(segment, torch.roll(ordinary, -shift))


def test_more_shifts_than_samples_produce_no_duplicates():
    segments = shifted_segments(audio(3), WINDOW, n_shifts=10)

    assert segments.shape[0] <= 3


def test_long_utterances_ignore_the_shifts():
    long = split_segments(audio(4 * WINDOW), WINDOW, n_segments=3, short_shifts=5)

    assert long.shape == (3, WINDOW)
    assert torch.equal(long[0], audio(4 * WINDOW)[:WINDOW])


def test_non_positive_shift_count_is_rejected():
    with pytest.raises(ValueError):
        split_segments(audio(3), WINDOW, n_segments=1, short_shifts=0)


def test_configured_collate_is_rejected_without_shifts():
    with pytest.raises(ValueError):
        get_multicrop_collate_fn(max_len=WINDOW, n_segments=2, short_shifts=-1)


def test_batch_counts_the_shifted_windows_of_short_utterances():
    batch = multicrop_collate_fn(
        [item(4), item(3 * WINDOW)], max_len=WINDOW, n_segments=3, short_shifts=4
    )

    assert batch["segment_sizes"] == [4, 3]
    assert batch["data_object"].shape == (7, WINDOW)


def test_labels_and_ids_stay_per_utterance_with_shifts():
    batch = multicrop_collate_fn(
        [item(4, 1, "LA_E_0000001"), item(4, 0, "LA_E_0000002")],
        max_len=WINDOW,
        n_segments=1,
        short_shifts=4,
    )

    assert torch.equal(batch["labels"], torch.tensor([1, 0]))
    assert batch["utt_id"] == ["LA_E_0000001", "LA_E_0000002"]
