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
    Read a CM protocol file into the trials, in the order of the protocol.
    """
    protocol_path = Path(protocol_path)

    entries: list[ProtocolEntry] = []
    with protocol_path.open("r") as protocol:
        for line_number, line in enumerate(protocol, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(
                    f"{protocol_path}:{line_number}: expected 5 fields, "
                    f"got {len(fields)}"
                )

            _, utt_id, _, attack_id, label = fields
            entries.append(ProtocolEntry(utt_id, attack_id, int(label == "bonafide")))

    return entries


def filter_entries(
    entries: list[ProtocolEntry], utt_ids: set[str] | None
) -> list[ProtocolEntry]:
    """
    Keep the trials of a subset of utterances, in the order of the protocol.
    None keeps everything.
    """
    if utt_ids is None:
        return list(entries)
    return [entry for entry in entries if entry.utt_id in utt_ids]
