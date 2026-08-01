from pathlib import Path
from typing import NamedTuple


class ProtocolEntry(NamedTuple):
    utt_id: str
    attack_id: str
    label: int


def read_protocol_entries(protocol_path: str | Path) -> list[ProtocolEntry]:
    protocol_path = Path(protocol_path)

    entries: list[ProtocolEntry] = []
    with protocol_path.open("r") as protocol:
        for line in protocol:
            fields = line.split()
            if not fields:
                continue

            _, utt_id, _, attack_id, label = fields
            entries.append(ProtocolEntry(utt_id, attack_id, int(label == "bonafide")))

    return entries
