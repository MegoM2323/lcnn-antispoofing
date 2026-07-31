"""
Reader of the ASVspoof2019 CM protocol files.

Every submission tool needs the same three fields of a trial: the utterance id,
the attack it was produced by and its class. ASVspoofDataset parses the very
same file, but it builds an index of audio paths and refuses to work without
the waveforms; scoring an existing csv must stay possible with the corpus
unmounted, hence a standalone reader.
"""

from pathlib import Path
from typing import NamedTuple

PROTOCOL_FIELDS = 5
BONAFIDE_LABEL = "bonafide"
BONAFIDE_ATTACK = "-"


class ProtocolEntry(NamedTuple):
    """
    One trial of the CM protocol.

    Attributes:
        utt_id (str): utterance id, the key of the submission csv.
        attack_id (str): spoofing algorithm ("A07" ... "A19"), "-" for
            bonafide trials.
        label (int): 1 for bonafide, 0 for spoof.
    """

    utt_id: str
    attack_id: str
    label: int


def read_protocol_entries(protocol_path: str | Path) -> list[ProtocolEntry]:
    """
    Read a CM protocol file.

    Args:
        protocol_path (str | Path): path to an ASVspoof2019 LA protocol.
    Returns:
        entries (list[ProtocolEntry]): trials in the order of the protocol.
    """
    protocol_path = Path(protocol_path)

    entries: list[ProtocolEntry] = []
    with protocol_path.open("r") as protocol:
        for line_number, line in enumerate(protocol, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != PROTOCOL_FIELDS:
                raise ValueError(
                    f"{protocol_path}:{line_number}: expected "
                    f"{PROTOCOL_FIELDS} fields, got {len(fields)}"
                )

            _, utt_id, _, attack_id, label = fields
            entries.append(
                ProtocolEntry(utt_id, attack_id, int(label == BONAFIDE_LABEL))
            )

    return entries


def filter_entries(
    entries: list[ProtocolEntry], utt_ids: set[str] | None
) -> list[ProtocolEntry]:
    """
    Keep the trials of a subset of utterances, in the order of the protocol.

    Args:
        entries (list[ProtocolEntry]): trials of the whole protocol.
        utt_ids (set[str] | None): ids to keep, None keeps everything.
    Returns:
        entries (list[ProtocolEntry]): selected trials.
    """
    if utt_ids is None:
        return list(entries)
    return [entry for entry in entries if entry.utt_id in utt_ids]


def read_utt_ids(path: str | Path) -> list[str]:
    """
    Read a list of utterance ids, one per line.

    Args:
        path (str | Path): text file with the ids, empty lines are skipped.
    Returns:
        utt_ids (list[str]): ids in the order of the file.
    """
    with Path(path).open("r") as file:
        return [line.strip() for line in file if line.strip()]
