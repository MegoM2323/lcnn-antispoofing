"""
Requirements covered:

* the per-attack breakdown scores every spoofing algorithm against the whole
  bonafide pool, exactly as the official evaluation plan does, so a single
  attack cannot be made to look good by the trials of another one;
* the breakdown of a subset uses the trials of that subset only;
* the pooled EER of a subset equals the EER computed on the same trials by
  hand, and it is undefined (None) when one of the two classes is missing;
* a trial without a score is reported instead of being dropped: a silently
  shrinking subset would make two checkpoints incomparable.
"""

import pytest

from src.metrics.attack_eer import attack_breakdown, ordered_scores, pooled_eer
from src.metrics.eer_utils import compute_eer_percent
from src.utils.protocol import (
    ProtocolEntry,
    filter_entries,
    read_protocol_entries,
    read_utt_ids,
)

PROTOCOL_LINES = [
    "LA_0079 LA_E_1000000 - - bonafide",
    "LA_0079 LA_E_1000001 - - bonafide",
    "LA_0080 LA_E_1000002 - A07 spoof",
    "LA_0080 LA_E_1000003 - A07 spoof",
    "LA_0081 LA_E_1000004 - A17 spoof",
    "LA_0081 LA_E_1000005 - A17 spoof",
]

# A07 is separated from the bonafide pool, A17 overlaps with it completely
SCORES = {
    "LA_E_1000000": 3.0,
    "LA_E_1000001": 2.0,
    "LA_E_1000002": -5.0,
    "LA_E_1000003": -4.0,
    "LA_E_1000004": 2.5,
    "LA_E_1000005": 2.5,
}


@pytest.fixture
def protocol(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("\n".join(PROTOCOL_LINES) + "\n")
    return path


@pytest.fixture
def entries(protocol):
    return read_protocol_entries(protocol)


def test_protocol_entries_carry_the_attack_id(entries):
    assert entries[0] == ProtocolEntry("LA_E_1000000", "-", 1)
    assert entries[4] == ProtocolEntry("LA_E_1000004", "A17", 0)


def test_blank_lines_are_skipped(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text(PROTOCOL_LINES[0] + "\n\n" + PROTOCOL_LINES[2] + "\n")

    assert len(read_protocol_entries(path)) == 2


def test_malformed_protocol_line_is_reported(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("LA_0079 LA_E_1000000 bonafide\n")

    with pytest.raises(ValueError, match="expected 5 fields"):
        read_protocol_entries(path)


def test_breakdown_scores_every_attack_against_the_bonafide_pool(entries):
    breakdown = {stats.attack_id: stats for stats in attack_breakdown(SCORES, entries)}

    assert set(breakdown) == {"A07", "A17"}
    assert breakdown["A07"].n_trials == 2
    assert breakdown["A07"].eer == pytest.approx(0.0)
    # the whole error of the system comes from A17, the pooled EER hides it
    assert breakdown["A17"].eer > breakdown["A07"].eer


def test_breakdown_matches_the_eer_of_the_attack_subset(entries):
    subset = [entry for entry in entries if entry.attack_id in ("-", "A17")]
    expected = compute_eer_percent(
        [SCORES[entry.utt_id] for entry in subset],
        [entry.label for entry in subset],
    )

    breakdown = {stats.attack_id: stats for stats in attack_breakdown(SCORES, entries)}

    assert breakdown["A17"].eer == pytest.approx(expected)


def test_breakdown_of_a_subset_uses_its_trials_only(entries):
    subset = filter_entries(
        entries, {"LA_E_1000000", "LA_E_1000001", "LA_E_1000002", "LA_E_1000003"}
    )

    breakdown = attack_breakdown(SCORES, subset)

    assert [stats.attack_id for stats in breakdown] == ["A07"]
    assert breakdown[0].n_trials == 2


def test_breakdown_without_bonafide_is_empty(entries):
    spoof_only = [entry for entry in entries if entry.label == 0]

    assert attack_breakdown(SCORES, spoof_only) == []


def test_pooled_eer_of_a_subset(entries):
    subset = filter_entries(entries, {"LA_E_1000000", "LA_E_1000002"})

    assert pooled_eer(SCORES, subset) == pytest.approx(0.0)


def test_pooled_eer_is_undefined_for_one_class(entries):
    bonafide_only = [entry for entry in entries if entry.label == 1]

    assert pooled_eer(SCORES, bonafide_only) is None


def test_filter_entries_keeps_the_protocol_order(entries):
    subset = filter_entries(entries, {"LA_E_1000004", "LA_E_1000000"})

    assert [entry.utt_id for entry in subset] == ["LA_E_1000000", "LA_E_1000004"]


def test_filter_entries_without_a_subset_keeps_everything(entries):
    assert filter_entries(entries, None) == entries


def test_missing_score_is_reported(entries):
    incomplete = {key: value for key, value in SCORES.items() if key != "LA_E_1000005"}

    with pytest.raises(ValueError, match="have no score"):
        ordered_scores(incomplete, entries)


def test_read_utt_ids_skips_blank_lines(tmp_path):
    path = tmp_path / "subset.txt"
    path.write_text("LA_E_1000000\n\n  LA_E_1000004  \n")

    assert read_utt_ids(path) == ["LA_E_1000000", "LA_E_1000004"]
