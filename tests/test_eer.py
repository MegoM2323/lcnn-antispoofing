"""The EER must follow the official grader, see src/metrics/eer_utils.py."""

import numpy as np
import pytest
import torch

from src.metrics.attack_eer import attack_breakdown
from src.metrics.eer_utils import DEGENERATE_EER, compute_eer_percent, logits_to_scores
from src.utils.protocol import read_protocol_entries

PROTOCOL_LINES = [
    "LA_0079 LA_E_1000000 - - bonafide",
    "LA_0079 LA_E_1000001 - - bonafide",
    "LA_0080 LA_E_1000002 - A07 spoof",
    "LA_0081 LA_E_1000003 - A17 spoof",
]

# A07 is separated from the bonafide pool, A17 overlaps with it
SCORES = {
    "LA_E_1000000": 3.0,
    "LA_E_1000001": 2.0,
    "LA_E_1000002": -5.0,
    "LA_E_1000003": 2.5,
}


def test_perfect_separation_gives_zero():
    assert compute_eer_percent([3.0, 2.0, -1.0, -2.0], [1, 1, 0, 0]) == pytest.approx(0)


def test_inverted_scores_give_hundred():
    # feeding the scores with the wrong sign gives 100 - EER, not EER
    assert compute_eer_percent([-3.0, -2.0, 1.0, 2.0], [1, 1, 0, 0]) == pytest.approx(
        100.0
    )


def test_single_class_is_not_nan():
    # a nan would poison the running average of the whole epoch
    assert compute_eer_percent([1.0, 2.0], [1, 1]) == DEGENERATE_EER


def test_one_error_out_of_ten():
    # 10 bonafide and 10 spoof trials, exactly one bonafide falls below the
    # whole spoof distribution: 10% of the target trials are rejected
    scores = list(np.arange(10, 20, dtype=float)) + list(np.arange(0, 10, dtype=float))
    scores[9] = -1.0

    assert compute_eer_percent(scores, [1] * 10 + [0] * 10) == pytest.approx(10.0)


def test_higher_score_means_bonafide():
    # class 1 is bonafide, so the score must grow with the bonafide logit
    scores = logits_to_scores(torch.tensor([[0.0, 5.0], [5.0, 0.0]]))

    assert scores.tolist() == [5.0, -5.0]


def test_every_attack_is_scored_against_the_bonafide_pool(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("\n".join(PROTOCOL_LINES) + "\n")

    breakdown = {
        stats.attack_id: stats
        for stats in attack_breakdown(SCORES, read_protocol_entries(path))
    }

    assert set(breakdown) == {"A07", "A17"}
    assert breakdown["A07"].eer == pytest.approx(0.0)
    # the whole error of the system comes from A17, the pooled EER hides it
    assert breakdown["A17"].eer > breakdown["A07"].eer
