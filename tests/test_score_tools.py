"""
Requirements covered:

* the score files the tools exchange survive a round trip: the csv keeps
  enough digits to reproduce the EER exactly, and the '.pth' dump of the
  inferencer is read back into the same scores;
* a malformed csv is refused when it is loaded, not when it is graded;
* 'fuse_scores.py' and 'eval_subset.py' run end to end on a synthetic
  protocol: the fusion writes a csv that covers the whole protocol, and the
  subset tool reports the EER of the requested utterances only.
"""

import sys

import pytest
import torch

from scripts.eval_subset import main as eval_subset_main
from scripts.fuse_scores import main as fuse_main
from scripts.make_submission import read_scores
from scripts.score_files import load_score_file, write_score_csv

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


@pytest.fixture
def scores_csv(tmp_path):
    path = tmp_path / "eval_scores.csv"
    write_score_csv(path, SCORES)
    return path


def run_script(monkeypatch, entry_point, argv):
    monkeypatch.setattr(sys, "argv", argv)
    return entry_point()


def test_csv_round_trip_is_exact(scores_csv):
    assert load_score_file(scores_csv) == SCORES


def test_csv_keeps_every_digit(tmp_path):
    # a rounded score creates ties between utterances the model separated,
    # and ties move the EER
    path = tmp_path / "scores.csv"
    write_score_csv(path, {"LA_E_1000000": 0.1234567890123456})

    assert load_score_file(path)["LA_E_1000000"] == 0.1234567890123456


def test_logits_dump_is_read_back(tmp_path):
    path = tmp_path / "eval_outputs.pth"
    torch.save(
        {
            "utt_id": list(SCORES),
            "logits": torch.zeros(len(SCORES), 2),
            "scores": torch.tensor(list(SCORES.values())),
            "labels": torch.tensor([1, 1, 0, 0]),
        },
        path,
    )

    loaded = load_score_file(path)

    assert loaded == pytest.approx(SCORES)


def test_logits_dump_without_scores_uses_the_logits(tmp_path):
    path = tmp_path / "eval_outputs.pth"
    torch.save(
        {"utt_id": ["LA_E_1000000"], "logits": torch.tensor([[1.0, 3.0]])},
        path,
    )

    assert load_score_file(path) == {"LA_E_1000000": 2.0}


def test_malformed_csv_is_refused(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("LA_E_1000000,bonafide\n")

    with pytest.raises(ValueError, match="malformed"):
        load_score_file(path)


def test_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_score_file(tmp_path / "absent.csv")


def test_fusion_script_writes_a_complete_submission(
    monkeypatch, tmp_path, protocol, scores_csv, capsys
):
    second = tmp_path / "other_scores.csv"
    write_score_csv(second, {key: -value for key, value in SCORES.items()})
    output = tmp_path / "fused.csv"

    exit_code = run_script(
        monkeypatch,
        fuse_main,
        [
            "fuse_scores.py",
            str(scores_csv),
            str(second),
            "-o",
            str(output),
            "-p",
            str(protocol),
            "-w",
            "3",
            "1",
        ],
    )

    assert exit_code == 0
    scores, errors = read_scores(output)
    assert errors == []
    assert set(scores) == set(SCORES)
    assert "EER" in capsys.readouterr().out


def test_fusion_script_refuses_incompatible_inputs(
    monkeypatch, tmp_path, protocol, scores_csv
):
    partial = tmp_path / "partial.csv"
    write_score_csv(partial, {"LA_E_1000000": 1.0})

    with pytest.raises(SystemExit):
        run_script(
            monkeypatch,
            fuse_main,
            [
                "fuse_scores.py",
                str(scores_csv),
                str(partial),
                "-o",
                str(tmp_path / "fused.csv"),
                "-p",
                str(protocol),
            ],
        )


def test_subset_script_reports_the_requested_utterances(
    monkeypatch, tmp_path, protocol, scores_csv, capsys
):
    subset = tmp_path / "subset.txt"
    subset.write_text("LA_E_1000000\nLA_E_1000002\n")

    exit_code = run_script(
        monkeypatch,
        eval_subset_main,
        ["eval_subset.py", str(scores_csv), "-s", str(subset), "-p", str(protocol)],
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "2 trials (1 bonafide, 1 spoof)" in output
    assert "EER: 0.0000%" in output
    assert "A07" in output and "A17" not in output


def test_subset_script_refuses_unscored_utterances(monkeypatch, tmp_path, protocol):
    # a subset that asks for more than the score file holds must stop the tool:
    # silently dropping the trials makes two checkpoints incomparable
    partial = tmp_path / "partial.csv"
    write_score_csv(partial, {"LA_E_1000000": 1.0})
    subset = tmp_path / "subset.txt"
    subset.write_text("LA_E_1000000\nLA_E_1000002\n")

    with pytest.raises(SystemExit):
        run_script(
            monkeypatch,
            eval_subset_main,
            ["eval_subset.py", str(partial), "-s", str(subset), "-p", str(protocol)],
        )
