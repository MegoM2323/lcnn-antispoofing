"""
Every batch must have a fixed waveform length whatever the utterances are, and
short ones are repeated cyclically instead of being zero-padded: trailing
silence is a shortcut the countermeasure must not learn. Labels and utterance
ids survive collation in the original order, degenerate input is rejected
instead of producing a silently broken batch.
"""

import pytest
import torch

from src.datasets.collate import collate_fn, get_collate_fn, pad_or_crop


def item(length, label=0, utt_id="LA_T_1000000"):
    return {
        "data_object": torch.arange(1, length + 1, dtype=torch.float32),
        "labels": label,
        "utt_id": utt_id,
    }


def test_short_waveform_is_repeated_not_zero_padded():
    audio = pad_or_crop(torch.tensor([1.0, 2.0, 3.0]), max_len=8)

    assert torch.equal(audio, torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]))
    assert (audio != 0).all()


def test_long_waveform_is_cropped():
    audio = pad_or_crop(torch.arange(10, dtype=torch.float32), max_len=4)

    assert torch.equal(audio, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_exact_length_is_untouched():
    original = torch.arange(5, dtype=torch.float32)

    assert torch.equal(pad_or_crop(original, max_len=5), original)


def test_single_sample_waveform_is_padded():
    assert torch.equal(
        pad_or_crop(torch.tensor([2.0]), max_len=3), torch.full((3,), 2.0)
    )


def test_channel_dimension_is_accepted():
    audio = pad_or_crop(torch.ones(1, 3), max_len=6)

    assert audio.shape == (6,)


def test_empty_waveform_is_rejected():
    with pytest.raises(ValueError):
        pad_or_crop(torch.empty(0), max_len=4)


def test_batch_is_rectangular_for_mixed_lengths():
    batch = collate_fn([item(3), item(100), item(64600)], max_len=64600)

    assert batch["data_object"].shape == (3, 64600)
    assert batch["data_object"].dtype == torch.float32


def test_labels_and_ids_keep_their_order():
    batch = collate_fn(
        [item(10, 1, "LA_T_0000001"), item(10, 0, "LA_T_0000002")], max_len=16
    )

    assert torch.equal(batch["labels"], torch.tensor([1, 0]))
    assert batch["utt_id"] == ["LA_T_0000001", "LA_T_0000002"]
    assert batch["labels"].dtype == torch.long


def test_utt_id_is_optional():
    batch = collate_fn([{"data_object": torch.ones(4), "labels": 0}], max_len=4)

    assert "utt_id" not in batch


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError):
        collate_fn([], max_len=4)


def test_configured_collate_uses_its_length():
    batch = get_collate_fn(max_len=128)([item(3)])

    assert batch["data_object"].shape == (1, 128)


def test_non_positive_configured_length_is_rejected():
    with pytest.raises(ValueError):
        get_collate_fn(max_len=0)
