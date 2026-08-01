"""
Pooling of the per-segment scores of one utterance into a single detection
score.

The four rules are not interchangeable, they encode different decisions. The
mean is the usual variance reduction of test-time augmentation. The maximum
accepts an utterance as soon as one of its segments looks bonafide, the minimum
rejects it as soon as one segment looks spoofed, and for a countermeasure the
second one is the natural reading: a partially spoofed recording is a spoofed
recording. The median throws away a single outlying segment. Which of them wins
is an empirical question, the score distributions of the attacks differ too
much for a general answer.
"""

from collections.abc import Sequence

import torch

AGGREGATIONS = ("mean", "max", "min", "median")
DEFAULT_AGGREGATION = "mean"


def validate_aggregation(aggregation: str) -> None:
    """
    Check that the pooling rule is a supported one.

    Args:
        aggregation (str): rule to check, see AGGREGATIONS.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}', expected one of {AGGREGATIONS}"
        )


def pool_scores(scores: torch.Tensor, aggregation: str) -> torch.Tensor:
    """
    Reduce the scores of the segments of one utterance to a single score.

    Args:
        scores (Tensor): 1D tensor with the score of every segment.
        aggregation (str): pooling rule, see AGGREGATIONS.
    Returns:
        score (Tensor): scalar tensor with the score of the utterance.
    """
    if scores.numel() == 0:
        raise ValueError("Cannot pool the scores of an utterance without segments")

    if aggregation == "mean":
        return scores.mean()
    if aggregation == "max":
        return scores.max()
    if aggregation == "min":
        return scores.min()
    # torch.median returns the lower of the two middle values, which for two
    # segments would silently turn the median into the minimum
    return torch.quantile(scores, 0.5)


def aggregate_segment_scores(
    scores: torch.Tensor,
    segment_sizes: Sequence[int],
    aggregation: str = DEFAULT_AGGREGATION,
) -> torch.Tensor:
    """
    Pool the scores of a batch of segments into one score per utterance.

    Args:
        scores (Tensor): 1D tensor with the score of every segment, the
            segments of one utterance lying next to each other.
        segment_sizes (Sequence[int]): number of segments of every utterance.
        aggregation (str): pooling rule, see AGGREGATIONS.
    Returns:
        pooled (Tensor): 1D tensor with one score per utterance, in the order
            of 'segment_sizes'.
    """
    validate_aggregation(aggregation)

    scores = scores.detach().float().reshape(-1)
    sizes = [int(size) for size in segment_sizes]
    if any(size <= 0 for size in sizes):
        raise ValueError(f"Every utterance needs at least one segment, got {sizes}")

    total = sum(sizes)
    if total != scores.numel():
        raise ValueError(
            f"Got {scores.numel()} segment scores for {total} segments "
            f"of {len(sizes)} utterances"
        )

    return torch.stack(
        [pool_scores(chunk, aggregation) for chunk in torch.split(scores, sizes)]
    )
