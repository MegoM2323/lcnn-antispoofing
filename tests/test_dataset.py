"""The dataset is built from the CM protocol, the batch has a fixed length."""

import pytest
import torch

from src.datasets import ASVspoofDataset
from src.datasets.collate import collate_fn, pad_or_crop

TRAIN_PROTOCOL = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"


def make_dataset(la_root, tmp_path, part="train"):
    return ASVspoofDataset(
        part=part, data_dir=str(la_root), index_dir=str(tmp_path / "index")
    )


def test_index_follows_the_protocol(la_root, tmp_path):
    dataset = make_dataset(la_root, tmp_path)
    lines = (la_root / TRAIN_PROTOCOL).read_text().splitlines()

    assert len(dataset) == len(lines)


def test_labels_follow_the_protocol(la_root, tmp_path):
    dataset = make_dataset(la_root, tmp_path)

    expected = {}
    for line in (la_root / TRAIN_PROTOCOL).read_text().splitlines():
        _, utt_id, _, _, label = line.split()
        expected[utt_id] = 1 if label == "bonafide" else 0

    assert {1, 0} == set(expected.values())
    for i in range(len(dataset)):
        item = dataset[i]
        assert item["labels"] == expected[item["utt_id"]]
        assert item["data_object"].dim() == 1


def test_missing_eval_audio_is_fatal(la_root, tmp_path):
    # the grading script looks up every id of the protocol in the submission,
    # so an incomplete eval index means a rejected submission
    removed = sorted((la_root / "ASVspoof2019_LA_eval" / "flac").glob("*.flac"))[0]
    removed.unlink()

    with pytest.raises(FileNotFoundError, match=removed.stem):
        make_dataset(la_root, tmp_path, part="eval")


def test_malformed_protocol_line_is_rejected(la_root, tmp_path):
    protocol = la_root / TRAIN_PROTOCOL
    protocol.write_text(protocol.read_text() + "LA_0000 LA_T_9999999 - A01\n")

    with pytest.raises(ValueError, match="expected 5 fields"):
        make_dataset(la_root, tmp_path)


def test_short_waveform_is_repeated_not_zero_padded():
    # trailing silence is a shortcut the countermeasure must not learn
    audio = pad_or_crop(torch.tensor([1.0, 2.0, 3.0]), max_len=8)

    assert torch.equal(audio, torch.tensor([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0]))


def test_long_waveform_is_cropped():
    audio = pad_or_crop(torch.arange(10, dtype=torch.float32), max_len=4)

    assert torch.equal(audio, torch.tensor([0.0, 1.0, 2.0, 3.0]))


def test_batch_is_rectangular_for_mixed_lengths():
    items = [
        {"data_object": torch.ones(3), "labels": 1, "utt_id": "LA_T_0000001"},
        {"data_object": torch.ones(77870), "labels": 0, "utt_id": "LA_T_0000002"},
    ]

    batch = collate_fn(items, max_len=77870)

    assert batch["data_object"].shape == (2, 77870)
    # labels and ids keep the order of the items
    assert torch.equal(batch["labels"], torch.tensor([1, 0]))
    assert batch["utt_id"] == ["LA_T_0000001", "LA_T_0000002"]
