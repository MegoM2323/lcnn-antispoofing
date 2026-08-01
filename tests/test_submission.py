"""
The submission csv is accepted only if it can be graded: one finite score for
every utterance of the eval protocol, no duplicates, no header. Everything the
grading script would choke on has to be reported instead of silently producing
a wrong grade.
"""

import csv

import numpy as np
import pytest

from scripts.make_submission import (
    compute_grade,
    read_scores,
    validate_submission,
    write_score_csv,
)
from src.metrics.eer_utils import compute_eer_percent
from src.metrics.score_fusion import fuse_scores

PROTOCOL_LINES = [
    "LA_0079 LA_E_1000000 - - bonafide",
    "LA_0079 LA_E_1000001 - - bonafide",
    "LA_0080 LA_E_1000002 - A07 spoof",
    "LA_0081 LA_E_1000003 - A17 spoof",
]

SCORES = {
    "LA_E_1000000": 3.25,
    "LA_E_1000001": 1.5,
    "LA_E_1000002": -4.75,
    "LA_E_1000003": 0.5,
}


@pytest.fixture
def protocol(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("\n".join(PROTOCOL_LINES) + "\n")
    return path


def write_rows(tmp_path, rows):
    path = tmp_path / "scores.csv"
    with path.open("w", newline="") as file:
        csv.writer(file).writerows(rows)
    return path


def make_system(seed, n_bonafide=40, n_spoof=120, scale=1.0):
    """One synthetic run: utt_id -> score, plus the labels."""
    rng = np.random.default_rng(seed)
    scores, labels = {}, {}
    for i in range(n_bonafide + n_spoof):
        utt_id = f"LA_E_{1000000 + i}"
        is_bonafide = i < n_bonafide
        scores[utt_id] = float(rng.normal(is_bonafide, 1.0) * scale)
        labels[utt_id] = int(is_bonafide)
    return scores, labels


def eer_of(scores, labels):
    utt_ids = list(labels)
    return compute_eer_percent(
        [scores[utt_id] for utt_id in utt_ids], [labels[utt_id] for utt_id in utt_ids]
    )


def test_valid_scores_are_accepted(tmp_path):
    path = write_rows(tmp_path, [["LA_E_1000000", "2.5"], ["LA_E_1000001", "-1e-9"]])

    scores, errors = read_scores(path)

    assert errors == []
    assert scores == {"LA_E_1000000": 2.5, "LA_E_1000001": -1e-9}


@pytest.mark.parametrize(
    "rows",
    [
        [["LA_E_1000000", "1.0", "extra"]],  # a third column
        [["LA_E_1000000", "bonafide"]],  # not a number
        [["LA_E_1000000", "nan"]],  # the grader would rank it arbitrarily
        # the grader looks the id up as is and dies with a KeyError on a padded
        # one, so the whitespace must not be silently stripped away
        [[" LA_E_1000000 ", "1.0"]],
        [["LA_E_1000000", "1.0"], ["LA_E_1000000", "2.0"]],  # duplicated id
    ],
)
def test_malformed_rows_are_reported(tmp_path, rows):
    _, errors = read_scores(write_rows(tmp_path, rows))

    assert errors


def test_csv_round_trip_keeps_every_digit(tmp_path):
    # a rounded score creates ties between utterances the model separated,
    # and ties move the EER
    path = tmp_path / "scores.csv"
    write_score_csv(path, {"LA_E_1000000": 0.1234567890123456})

    assert read_scores(path)[0]["LA_E_1000000"] == 0.1234567890123456


def test_submission_covering_the_protocol_is_graded(tmp_path, protocol):
    path = tmp_path / "scores.csv"
    write_score_csv(path, SCORES)

    assert validate_submission(path, protocol) == pytest.approx(0.0)


def test_missing_protocol_id_is_refused(tmp_path, protocol, capsys):
    path = tmp_path / "scores.csv"
    write_score_csv(path, {k: v for k, v in SCORES.items() if k != "LA_E_1000003"})

    assert validate_submission(path, protocol) is None
    assert "missing" in capsys.readouterr().out


@pytest.mark.parametrize("eer, expected", [(5.29, 10.0), (10.9, 2.0), (11.0, 0.0)])
def test_grade_thresholds(eer, expected):
    assert compute_grade(eer) == pytest.approx(expected)


def test_fusion_ignores_the_scale_of_a_system():
    # the same run, once with LLRs in [-3, 3] and once in [-300, 300]
    scores, labels = make_system(seed=2)
    stretched = {utt_id: value * 100 for utt_id, value in scores.items()}

    fused = fuse_scores([scores, stretched])

    assert eer_of(fused, labels) == pytest.approx(eer_of(scores, labels))


def test_fusion_of_complementary_systems_beats_both():
    first, labels = make_system(seed=3, n_bonafide=200, n_spoof=600)
    second, _ = make_system(seed=4, n_bonafide=200, n_spoof=600, scale=25.0)
    # the second run scores the same utterances, independently of the first one
    second = dict(zip(first, second.values()))

    fused = fuse_scores([first, second])

    assert eer_of(fused, labels) < min(eer_of(first, labels), eer_of(second, labels))


def test_fusion_refuses_a_different_set_of_utterances():
    # a fusion built on a partial intersection is a csv with a hole in it
    first, _ = make_system(seed=5)
    second = dict(first)
    second.pop(next(iter(second)))

    with pytest.raises(ValueError, match="another set of utterances"):
        fuse_scores([first, second])
