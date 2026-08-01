"""
Multi-crop scoring must see the whole recording without inventing samples that
are not in it, and it must leave the short utterances exactly where they were:
anything shorter than one window is scored on a single repeat-padded segment,
the very tensor the ordinary collate function builds. The pooling rules are
checked on numbers whose mean, median and extremes are known by hand, and the
two symmetric ones are checked not to depend on the order of the segments.
"""

import pytest
import torch

from src.datasets import ASVspoofDataset
from src.datasets.collate import collate_fn
from src.datasets.data_utils import select_utterances
from src.datasets.multicrop import (
    get_multicrop_collate_fn,
    multicrop_collate_fn,
    segment_starts,
    split_segments,
)
from src.metrics.segment_pooling import aggregate_segment_scores

WINDOW = 8


def item(length, label=0, utt_id="LA_E_1000000"):
    return {
        "data_object": torch.arange(1, length + 1, dtype=torch.float32),
        "labels": label,
        "utt_id": utt_id,
    }


@pytest.mark.parametrize("length", [WINDOW + 1, 3 * WINDOW, 7 * WINDOW - 3])
@pytest.mark.parametrize("n_segments", [2, 3, 5])
def test_segments_stay_inside_the_utterance(length, n_segments):
    starts = segment_starts(length, WINDOW, n_segments)

    assert starts[0] == 0
    assert all(start >= 0 for start in starts)
    assert all(start + WINDOW <= length for start in starts)


@pytest.mark.parametrize("length", [WINDOW + 1, 3 * WINDOW, 7 * WINDOW - 3])
@pytest.mark.parametrize("n_segments", [2, 3, 5])
def test_segments_reach_the_end_of_the_utterance(length, n_segments):
    starts = segment_starts(length, WINDOW, n_segments)

    assert starts[-1] + WINDOW == length


def test_enough_segments_cover_every_sample():
    length, n_segments = 5 * WINDOW, 5

    starts = segment_starts(length, WINDOW, n_segments)
    covered = {sample for start in starts for sample in range(start, start + WINDOW)}

    assert covered == set(range(length))


def test_starts_are_sorted_and_unique():
    # more windows than fit into the utterance: the extra ones coincide
    starts = segment_starts(WINDOW + 2, WINDOW, n_segments=9)

    assert starts == sorted(set(starts))


@pytest.mark.parametrize("length", [1, WINDOW // 2, WINDOW])
def test_short_utterance_gives_exactly_one_segment(length):
    segments = split_segments(torch.arange(1, length + 1).float(), WINDOW, n_segments=5)

    assert segments.shape == (1, WINDOW)


def test_short_utterance_is_scored_as_before():
    short = item(3)

    multicrop = multicrop_collate_fn([short], max_len=WINDOW, n_segments=5)
    plain = collate_fn([short], max_len=WINDOW)

    assert multicrop["segment_sizes"] == [1]
    assert torch.equal(multicrop["data_object"], plain["data_object"])


def test_single_segment_reproduces_the_plain_batch():
    items = [item(3), item(WINDOW), item(5 * WINDOW)]

    multicrop = multicrop_collate_fn(items, max_len=WINDOW, n_segments=1)
    plain = collate_fn(items, max_len=WINDOW)

    assert multicrop["segment_sizes"] == [1, 1, 1]
    assert torch.equal(multicrop["data_object"], plain["data_object"])


def test_segments_are_the_slices_of_the_waveform():
    audio = torch.arange(1, 3 * WINDOW + 1, dtype=torch.float32)

    segments = split_segments(audio, WINDOW, n_segments=3)

    assert torch.equal(segments[0], audio[:WINDOW])
    assert torch.equal(segments[-1], audio[-WINDOW:])


def test_batch_stacks_the_segments_of_every_utterance():
    batch = multicrop_collate_fn(
        [item(3), item(3 * WINDOW), item(2 * WINDOW)], max_len=WINDOW, n_segments=3
    )

    assert batch["segment_sizes"] == [1, 3, 3]
    assert batch["data_object"].shape == (7, WINDOW)
    assert batch["data_object"].dtype == torch.float32


def test_labels_and_ids_stay_per_utterance():
    batch = multicrop_collate_fn(
        [item(3 * WINDOW, 1, "LA_E_0000001"), item(3, 0, "LA_E_0000002")],
        max_len=WINDOW,
        n_segments=3,
    )

    assert torch.equal(batch["labels"], torch.tensor([1, 0]))
    assert batch["utt_id"] == ["LA_E_0000001", "LA_E_0000002"]


def test_empty_batch_is_rejected():
    with pytest.raises(ValueError):
        multicrop_collate_fn([], max_len=WINDOW, n_segments=3)


@pytest.mark.parametrize("n_segments", [0, -1])
def test_non_positive_segment_count_is_rejected(n_segments):
    with pytest.raises(ValueError):
        get_multicrop_collate_fn(max_len=WINDOW, n_segments=n_segments)


def test_configured_collate_uses_its_segmentation():
    batch = get_multicrop_collate_fn(max_len=WINDOW, n_segments=2)([item(4 * WINDOW)])

    assert batch["data_object"].shape == (2, WINDOW)


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("mean", 2.0), ("max", 5.0), ("min", -1.0), ("median", 2.0)],
)
def test_aggregations_on_known_numbers(aggregation, expected):
    scores = torch.tensor([-1.0, 2.0, 5.0])

    pooled = aggregate_segment_scores(scores, [3], aggregation)

    assert pooled.tolist() == pytest.approx([expected])


def test_median_of_two_segments_is_not_the_minimum():
    pooled = aggregate_segment_scores(torch.tensor([-4.0, 2.0]), [2], "median")

    assert pooled.tolist() == pytest.approx([-1.0])


@pytest.mark.parametrize("aggregation", ["mean", "median"])
def test_symmetric_aggregations_ignore_the_order(aggregation):
    scores = torch.tensor([3.0, -7.0, 0.5, 11.0])
    shuffled = scores[torch.tensor([2, 0, 3, 1])]

    pooled = aggregate_segment_scores(scores, [4], aggregation)
    reordered = aggregate_segment_scores(shuffled, [4], aggregation)

    assert pooled.tolist() == pytest.approx(reordered.tolist())


def test_utterances_are_pooled_independently():
    scores = torch.tensor([1.0, 3.0, -5.0, 10.0])

    pooled = aggregate_segment_scores(scores, [2, 1, 1], "mean")

    assert pooled.tolist() == pytest.approx([2.0, -5.0, 10.0])


def test_single_segment_is_returned_unchanged():
    scores = torch.tensor([1.5, -2.5])

    for aggregation in ("mean", "max", "min", "median"):
        pooled = aggregate_segment_scores(scores, [1, 1], aggregation)
        assert pooled.tolist() == pytest.approx(scores.tolist())


def test_score_count_must_match_the_segment_sizes():
    with pytest.raises(ValueError):
        aggregate_segment_scores(torch.zeros(3), [2, 2], "mean")


def test_utterance_without_segments_is_rejected():
    with pytest.raises(ValueError):
        aggregate_segment_scores(torch.zeros(2), [2, 0], "mean")


def test_unknown_aggregation_is_rejected():
    with pytest.raises(ValueError):
        aggregate_segment_scores(torch.zeros(2), [2], "geometric")


def eval_dataset(la_root, index_dir):
    return ASVspoofDataset(
        part="eval", data_dir=str(la_root), index_dir=str(index_dir), max_len=None
    )


def test_measurement_subset_keeps_the_requested_utterances(la_root, index_dir):
    dataset = eval_dataset(la_root, index_dir)
    requested = {"LA_E_1000001", "LA_E_1000004"}

    selected = select_utterances(dataset, requested)

    assert len(selected) == 2
    assert {item["utt_id"] for item in selected} == requested


def test_measurement_subset_ignores_unknown_ids(la_root, index_dir):
    dataset = eval_dataset(la_root, index_dir)

    selected = select_utterances(dataset, {"LA_E_1000000", "LA_E_9999999"})

    assert len(selected) == 1


def test_empty_measurement_subset_is_rejected(la_root, index_dir):
    dataset = eval_dataset(la_root, index_dir)

    with pytest.raises(ValueError):
        select_utterances(dataset, {"LA_E_9999999"})
