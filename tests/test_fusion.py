"""
Requirements covered:

* normalizing the scores of a system must not change what that system decides:
  both normalizations are monotone, so the EER of a single run is the same
  before and after them;
* fusing a system with itself gives the same EER, whatever the weights are:
  a fusion that moves the number in this case is not a fusion but a bug;
* a fusion is only defined on the same set of utterances, otherwise the
  resulting csv has a hole in it and the grading script dies with a KeyError.
  Mismatched inputs must be refused;
* fusing two systems that see different parts of the problem is better than
  either of them, which is the whole point of the tool.
"""

import numpy as np
import pytest

from src.metrics.eer_utils import compute_eer_percent
from src.metrics.score_fusion import (
    check_same_keys,
    fuse_scores,
    normalize_scores,
    rank_normalize,
    z_normalize,
)

METHODS = ("rank", "zscore")


def make_system(seed, n_bonafide=40, n_spoof=120, separation=1.0, scale=1.0):
    """
    Build one synthetic run: utt_id -> score, plus the labels.

    Bonafide scores are drawn around '+separation', spoof ones around zero, so
    the EER is somewhere between 0 and 50 percents. 'scale' stretches the
    scores of the run, which is what makes two runs incomparable without
    normalization.
    """
    rng = np.random.default_rng(seed)
    scores = {}
    labels = {}
    for i in range(n_bonafide + n_spoof):
        utt_id = f"LA_E_{1000000 + i}"
        is_bonafide = i < n_bonafide
        value = rng.normal(separation if is_bonafide else 0.0, 1.0)
        scores[utt_id] = float(value * scale)
        labels[utt_id] = int(is_bonafide)
    return scores, labels


def eer_of(scores, labels):
    utt_ids = list(labels)
    return compute_eer_percent(
        [scores[utt_id] for utt_id in utt_ids], [labels[utt_id] for utt_id in utt_ids]
    )


@pytest.mark.parametrize("method", METHODS)
def test_normalization_keeps_the_eer_of_a_system(method):
    scores, labels = make_system(seed=0)
    utt_ids = list(scores)

    normalized = normalize_scores([scores[utt_id] for utt_id in utt_ids], method)
    normalized_scores = dict(zip(utt_ids, normalized))

    assert eer_of(normalized_scores, labels) == pytest.approx(eer_of(scores, labels))


@pytest.mark.parametrize("method", METHODS)
def test_normalization_is_monotone(method):
    values = np.array([-7.0, -0.5, 0.0, 0.5, 3.0, 11.0])

    normalized = normalize_scores(values, method)

    assert np.all(np.diff(normalized) > 0)


def test_rank_normalization_maps_to_the_unit_interval():
    normalized = rank_normalize(np.array([5.0, -1.0, 3.0, 100.0]))

    assert normalized.min() == pytest.approx(0.0)
    assert normalized.max() == pytest.approx(1.0)


def test_rank_normalization_gives_tied_scores_the_same_rank():
    normalized = rank_normalize(np.array([1.0, 2.0, 2.0, 3.0]))

    assert normalized[1] == pytest.approx(normalized[2])


def test_z_normalization_of_constant_scores_does_not_divide_by_zero():
    normalized = z_normalize(np.array([2.0, 2.0, 2.0]))

    assert np.all(normalized == 0.0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("weights", [None, [1.0, 1.0], [3.0, 1.0]])
def test_fusing_a_system_with_itself_keeps_its_eer(method, weights):
    scores, labels = make_system(seed=1)

    fused = fuse_scores([scores, dict(scores)], weights, method)

    assert set(fused) == set(scores)
    assert eer_of(fused, labels) == pytest.approx(eer_of(scores, labels))


@pytest.mark.parametrize("method", METHODS)
def test_fusion_ignores_the_scale_of_a_system(method):
    # the same run, once with LLRs in [-3, 3] and once in [-300, 300]
    scores, labels = make_system(seed=2)
    stretched = {utt_id: value * 100 for utt_id, value in scores.items()}

    fused = fuse_scores([scores, stretched], None, method)

    assert eer_of(fused, labels) == pytest.approx(eer_of(scores, labels))


@pytest.mark.parametrize("method", METHODS)
def test_fusion_of_complementary_systems_beats_both(method):
    # 800 trials: with a few dozen of them the EER moves in steps of percents
    # and two systems of the same strength cannot be told apart
    first, labels = make_system(seed=3, n_bonafide=200, n_spoof=600, separation=0.7)
    second, _ = make_system(
        seed=4, n_bonafide=200, n_spoof=600, separation=0.7, scale=25.0
    )
    # the second run scores the same utterances, independently of the first one
    second = dict(zip(first, second.values()))

    fused = fuse_scores([first, second], None, method)

    assert eer_of(fused, labels) < min(eer_of(first, labels), eer_of(second, labels))


def test_fusion_refuses_a_missing_utterance():
    first, _ = make_system(seed=5)
    second = dict(first)
    second.pop(next(iter(second)))

    with pytest.raises(ValueError, match="another set of utterances"):
        fuse_scores([first, second])


def test_fusion_refuses_an_unknown_utterance():
    first, _ = make_system(seed=6)
    second = dict(first)
    second["LA_E_9999999"] = 0.0

    with pytest.raises(ValueError, match="another set of utterances"):
        fuse_scores([first, second])


def test_check_same_keys_returns_the_order_of_the_first_system():
    first = {"b": 1.0, "a": 2.0}
    second = {"a": 3.0, "b": 4.0}

    assert check_same_keys([first, second]) == ["b", "a"]


def test_fusion_refuses_a_wrong_number_of_weights():
    scores, _ = make_system(seed=7)

    with pytest.raises(ValueError, match="every system needs exactly one weight"):
        fuse_scores([scores, dict(scores)], [1.0])


def test_fusion_refuses_degenerate_weights():
    scores, _ = make_system(seed=8)

    with pytest.raises(ValueError, match="sum to zero"):
        fuse_scores([scores, dict(scores)], [0.0, 0.0])

    with pytest.raises(ValueError, match="non-negative"):
        fuse_scores([scores, dict(scores)], [-1.0, 2.0])


def test_unknown_normalization_is_refused():
    with pytest.raises(ValueError, match="Unknown normalization"):
        normalize_scores(np.array([1.0, 2.0]), "minmax")
