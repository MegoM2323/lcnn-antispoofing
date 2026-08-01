"""
Чтение файлов CM-протокола ASVspoof2019.

Всем инструментам работы с посылкой нужны одни и те же три поля испытания:
идентификатор записи, атака, которой она порождена, и класс. ASVspoofDataset
разбирает тот же самый файл, но строит индекс путей к аудио и без самих
сигналов работать отказывается, а проверять готовый csv надо и с
неподключённым корпусом. Отсюда отдельный ридер.
"""

from pathlib import Path
from typing import NamedTuple


class ProtocolEntry(NamedTuple):
    """
    Одно испытание CM-протокола.

    Поля:
        utt_id (str): идентификатор записи, ключ csv с посылкой.
        attack_id (str): алгоритм атаки ("A07" ... "A19"), "-" для испытаний
            bonafide.
        label (int): 1 для bonafide, 0 для spoof.
    """

    utt_id: str
    attack_id: str
    label: int


def read_protocol_entries(protocol_path: str | Path) -> list[ProtocolEntry]:
    """
    Читает файл CM-протокола в список испытаний, в порядке протокола.
    """
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
