"""
Requirements covered:

* the index of a partition is built from the official CM protocol, every
  utterance keeps its id and its binary label (bonafide = 1, spoof = 0);
* utterances listed in the protocol but missing on disk are skipped on
  train/dev, and are fatal on eval: the grading script looks up every id of
  the protocol in the submission, so an incomplete eval index means a rejected
  submission;
* a malformed protocol (wrong number of fields, unknown label) is rejected
  loudly instead of silently producing a wrong index;
* the index is cached and re-read, and the cache is rebuilt whenever the
  corpus location or the protocol changes: the index stores absolute paths and
  labels, so a stale cache silently feeds the wrong data;
* an unknown partition name is rejected.
"""

import json
import shutil

import pytest
import torch

from src.datasets import ASVspoofDataset


def make_dataset(la_root, index_dir, part="train", **kwargs):
    return ASVspoofDataset(
        part=part, data_dir=str(la_root), index_dir=str(index_dir), **kwargs
    )


def test_index_matches_protocol(la_root, index_dir):
    dataset = make_dataset(la_root, index_dir)

    protocol = (
        (la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt")
        .read_text()
        .splitlines()
    )

    assert len(dataset) == len(protocol)


def test_labels_follow_the_protocol(la_root, index_dir):
    dataset = make_dataset(la_root, index_dir)

    expected = {}
    for line in (
        (la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt")
        .read_text()
        .splitlines()
    ):
        _, utt_id, _, _, label = line.split()
        expected[utt_id] = 1 if label == "bonafide" else 0

    for i in range(len(dataset)):
        item = dataset[i]
        assert item["labels"] == expected[item["utt_id"]]


def test_item_is_a_mono_float_waveform(la_root, index_dir):
    item = make_dataset(la_root, index_dir)[0]

    assert item["data_object"].dim() == 1
    assert item["data_object"].dtype == torch.float32
    assert item["data_object"].numel() > 0


def test_long_utterance_is_cut_to_max_len(la_root, index_dir):
    dataset = make_dataset(la_root, index_dir, max_len=8000, random_crop=False)

    for i in range(len(dataset)):
        assert dataset[i]["data_object"].numel() <= 8000


def test_missing_audio_is_skipped(la_root, index_dir):
    removed = sorted((la_root / "ASVspoof2019_LA_train" / "flac").glob("*.flac"))[0]
    removed.unlink()

    dataset = make_dataset(la_root, index_dir)

    assert removed.stem not in {entry["utt_id"] for entry in dataset._index}
    assert len(dataset) > 0


def test_missing_eval_audio_is_fatal(la_root, index_dir):
    removed = sorted((la_root / "ASVspoof2019_LA_eval" / "flac").glob("*.flac"))[0]
    removed.unlink()

    with pytest.raises(FileNotFoundError, match=removed.stem):
        make_dataset(la_root, index_dir, part="eval")


def test_malformed_protocol_line_is_rejected(la_root, index_dir):
    protocol = (
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    protocol.write_text(protocol.read_text() + "LA_0000 LA_T_9999999 - A01\n")

    with pytest.raises(ValueError, match="expected 5 fields"):
        make_dataset(la_root, index_dir)


def test_unknown_label_is_rejected(la_root, index_dir):
    protocol = (
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    protocol.write_text(protocol.read_text() + "LA_0000 LA_T_9999999 - A01 deepfake\n")

    with pytest.raises(ValueError, match="Unknown label"):
        make_dataset(la_root, index_dir)


def test_missing_protocol_is_reported(tmp_path, index_dir):
    (tmp_path / "LA" / "ASVspoof2019_LA_train" / "flac").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        ASVspoofDataset(
            part="train", data_dir=str(tmp_path / "LA"), index_dir=str(index_dir)
        )


def test_unknown_partition_is_rejected(la_root, index_dir):
    with pytest.raises(ValueError, match="Unknown partition"):
        make_dataset(la_root, index_dir, part="test")


def test_non_positive_max_len_is_rejected(la_root, index_dir):
    with pytest.raises(ValueError, match="max_len"):
        make_dataset(la_root, index_dir, max_len=0)


def test_index_is_cached_and_reused(la_root, index_dir):
    first = make_dataset(la_root, index_dir)
    cache = index_dir / "asvspoof_la_train.json"

    assert cache.exists()
    assert len(json.loads(cache.read_text())["index"]) == len(first)

    # the second dataset must not re-scan the protocol
    (
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    ).unlink()
    assert len(make_dataset(la_root, index_dir)) == len(first)


def test_cache_is_rebuilt_when_the_protocol_changes(la_root, index_dir):
    protocol = (
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    first = make_dataset(la_root, index_dir)

    lines = protocol.read_text().splitlines(keepends=True)
    protocol.write_text("".join(lines[:-2]))
    second = make_dataset(la_root, index_dir)

    assert len(second) == len(first) - 2


def test_cache_is_rebuilt_when_the_labels_change(la_root, index_dir):
    protocol = (
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    make_dataset(la_root, index_dir)

    protocol.write_text(protocol.read_text().replace("bonafide", "spoof"))
    labels = {entry["label"] for entry in make_dataset(la_root, index_dir)._index}

    assert labels == {0}


def test_cache_is_rebuilt_when_the_corpus_moves(la_root, index_dir, tmp_path):
    make_dataset(la_root, index_dir)

    moved_root = tmp_path / "LA_moved"
    shutil.copytree(la_root, moved_root)
    dataset = make_dataset(moved_root, index_dir)

    assert all(entry["path"].startswith(str(moved_root)) for entry in dataset._index)


def test_legacy_cache_without_provenance_is_rebuilt(la_root, index_dir):
    cache = index_dir / "asvspoof_la_train.json"
    dataset = make_dataset(la_root, index_dir)
    # the format used before the cache carried the data it was built from
    cache.write_text(json.dumps(json.loads(cache.read_text())["index"][:3]))

    rebuilt = make_dataset(la_root, index_dir)

    assert len(rebuilt) == len(dataset)
    assert "fingerprint" in json.loads(cache.read_text())


def test_limit_and_shuffle_are_deterministic(la_root, index_dir):
    first = make_dataset(la_root, index_dir, limit=4, shuffle_index=True)
    second = make_dataset(la_root, index_dir, limit=4, shuffle_index=True)

    assert len(first) == 4
    assert [item["utt_id"] for item in first._index] == [
        item["utt_id"] for item in second._index
    ]


def test_empty_partition_is_reported(la_root, index_dir):
    protocol = (
        la_root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    protocol.write_text("")

    with pytest.raises(RuntimeError, match="empty"):
        make_dataset(la_root, index_dir)
