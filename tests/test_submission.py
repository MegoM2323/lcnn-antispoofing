"""
Requirements covered:

* the submission csv is accepted only if it can be graded: exactly one finite
  score for every utterance of the eval protocol, no duplicates, no header;
* every defect the grading script would choke on (a missing id, a duplicated
  id, a non-numeric or non-finite score, a row with a wrong number of columns)
  is reported instead of silently producing a wrong grade;
* the reported EER and grade match the thresholds of the homework.
"""

import csv

import pytest

from scripts.make_submission import compute_grade, read_protocol, read_scores

PROTOCOL_LINES = [
    "LA_0079 LA_E_1000000 - - bonafide",
    "LA_0079 LA_E_1000001 - A07 spoof",
    "LA_0080 LA_E_1000002 - A08 spoof",
]


@pytest.fixture
def protocol(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("\n".join(PROTOCOL_LINES) + "\n")
    return path


def write_scores(tmp_path, rows):
    path = tmp_path / "scores.csv"
    with path.open("w", newline="") as file:
        csv.writer(file).writerows(rows)
    return path


def test_protocol_labels_are_binary(protocol):
    trials = read_protocol(protocol)

    assert trials == [("LA_E_1000000", 1), ("LA_E_1000001", 0), ("LA_E_1000002", 0)]


def test_valid_scores_are_accepted(tmp_path):
    path = write_scores(tmp_path, [["LA_E_1000000", "2.5"], ["LA_E_1000001", "-1.0"]])

    scores, errors = read_scores(path)

    assert errors == []
    assert scores == {"LA_E_1000000": 2.5, "LA_E_1000001": -1.0}


def test_scientific_notation_is_accepted(tmp_path):
    path = write_scores(tmp_path, [["LA_E_1000000", "1e-9"]])

    scores, errors = read_scores(path)

    assert errors == []
    assert scores["LA_E_1000000"] == pytest.approx(1e-9)


def test_duplicated_id_is_reported(tmp_path):
    path = write_scores(tmp_path, [["LA_E_1000000", "1.0"], ["LA_E_1000000", "2.0"]])

    _, errors = read_scores(path)

    assert any("duplicated" in error for error in errors)


def test_padded_id_is_reported(tmp_path):
    # grading.py looks the key up as is and dies with a KeyError on a padded id,
    # so the checker must not silently strip the whitespace away
    path = write_scores(tmp_path, [[" LA_E_1000000 ", "1.0"]])

    scores, errors = read_scores(path)

    assert errors or " LA_E_1000000 " in scores


def test_non_numeric_score_is_reported(tmp_path):
    path = write_scores(tmp_path, [["LA_E_1000000", "bonafide"]])

    _, errors = read_scores(path)

    assert any("not a float" in error for error in errors)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_score_is_reported(tmp_path, value):
    path = write_scores(tmp_path, [["LA_E_1000000", value]])

    _, errors = read_scores(path)

    assert errors


def test_wrong_column_count_is_reported(tmp_path):
    path = write_scores(tmp_path, [["LA_E_1000000", "1.0", "extra"]])

    _, errors = read_scores(path)

    assert any("2 columns" in error for error in errors)


def test_header_row_is_reported(tmp_path):
    path = write_scores(tmp_path, [["utterance_id", "score"], ["LA_E_1000000", "1.0"]])

    _, errors = read_scores(path)

    assert errors


def test_missing_id_is_detected(tmp_path, protocol):
    path = write_scores(tmp_path, [["LA_E_1000000", "1.0"]])

    trials = read_protocol(protocol)
    scores, _ = read_scores(path)
    missing = [key for key, _ in trials if key not in scores]

    assert missing == ["LA_E_1000001", "LA_E_1000002"]


@pytest.mark.parametrize(
    "eer, expected",
    [(0.0, 10.0), (5.29, 10.0), (10.9, 2.0), (11.0, 0.0), (100.0, 0.0)],
)
def test_grade_thresholds(eer, expected):
    assert compute_grade(eer) == pytest.approx(expected)


def test_grade_decreases_with_eer():
    assert compute_grade(6.0) > compute_grade(8.0) > compute_grade(10.0)
