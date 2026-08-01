"""
Чтение, проверка и запись csv с эвалюационными скорами.

Проверки повторяют 'grading.py': проверяющий скрипт строит из csv словарь
и затем ищет в нём *каждую* запись протокола, поэтому пропущенная или битая
строка это KeyError у него и ноль за работу. Функции чтения и записи общие
с двумя другими скриптами.

    python3 scripts/make_submission.py data/saved/lfcc21/eval_scores.csv -o mppanin.csv
"""

import argparse
import csv
import math
import os
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.metrics.attack_eer import attack_breakdown  # noqa: E402
from src.metrics.eer_utils import compute_eer_percent  # noqa: E402
from src.utils.protocol import read_protocol_entries  # noqa: E402

# протокол корпуса LA, разложенного так, как описано в README
EVAL_PROTOCOL = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
DEFAULT_PROTOCOL = os.environ.get(
    "ASVSPOOF_EVAL_PROTOCOL", str(PROJECT_ROOT / "data" / "LA" / EVAL_PROTOCOL)
)
DEFAULT_SUBMISSION_NAME = "mppanin.csv"  # должно совпадать с университетским логином


def read_scores(scores_path: str | Path) -> tuple[dict[str, float], list[str]]:
    """
    Читает csv со скорами модели и возвращает их вместе со списком найденных
    проблем. Пустые строки пропускаются, проверяющий скрипт тоже их пропускает.
    """
    scores: dict[str, float] = {}
    errors: list[str] = []
    with Path(scores_path).open("r", newline="") as file:
        for line, row in enumerate(csv.reader(file), start=1):
            if not row:
                continue
            if len(row) != 2:
                errors.append(f"line {line}: {len(row)} columns instead of 2")
                continue

            key, raw_score = row
            if key != key.strip():
                # проверяющий скрипт ищет идентификатор как есть, поэтому
                # лишние пробелы дают у него KeyError: чинить их молча нельзя
                errors.append(f"line {line}: id '{key}' is padded")
                continue

            try:
                score = float(raw_score)
            except ValueError:
                errors.append(f"line {line}: '{raw_score}' is not a float")
                continue

            if not math.isfinite(score):
                errors.append(f"line {line}: score is {score}")
            elif key in scores:
                errors.append(f"line {line}: duplicated id '{key}'")
            else:
                scores[key] = score

    return scores, errors


def load_score_file(path: str | Path) -> dict[str, float]:
    """Читает набор скоров, сразу отвергая битый файл."""
    scores, errors = read_scores(path)
    if errors:
        raise ValueError(f"'{path}' is malformed: {errors[0]}")
    return scores


def write_score_csv(path: str | Path, scores: Mapping[str, float]) -> None:
    """
    Пишет скоры в формате посылки. Вместо фиксированного формата используется
    'repr': округление создаёт совпадающие скоры у записей, которые модель
    различила, а совпадения сдвигают EER.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        for utt_id, score in scores.items():
            writer.writerow([utt_id, repr(float(score))])


def validate_submission(
    scores_path: str | Path, protocol_path: str | Path
) -> float | None:
    """
    Прогоняет все проверки, на которых споткнулся бы проверяющий скрипт, и
    печатает EER вместе с EER каждого алгоритма атаки (каждая атака
    сравнивается со всем пулом bonafide, как в официальном плане оценки).
    Возвращает None, если файл непроверяем; найденные проблемы печатаются.
    """
    entries = read_protocol_entries(protocol_path)
    scores, errors = read_scores(scores_path)

    missing = [entry.utt_id for entry in entries if entry.utt_id not in scores]
    if missing:
        errors.append(f"{len(missing)} protocol ids are missing, e.g. {missing[:5]}")

    if errors:
        print(f"{scores_path} is INVALID:")
        for error in errors[:20]:
            print(f"  - {error}")
        return None

    # тот же расчёт, что в grading.py: официальный compute_eer по скорам,
    # упорядоченным по протоколу
    labels = [entry.label for entry in entries]
    eer = compute_eer_percent([scores[entry.utt_id] for entry in entries], labels)

    by_attack = ", ".join(
        f"{stats.attack_id} {stats.eer:.2f}%"
        for stats in sorted(attack_breakdown(scores, entries), key=lambda st: -st.eer)
    )
    print(f"{scores_path}: {len(entries)} trials, EER {eer:.4f}%")
    print(f"EER by attack: {by_attack}")

    return eer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the eval scores and prepare the submission file."
    )
    parser.add_argument("scores", type=Path, help="csv with the model scores")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(DEFAULT_SUBMISSION_NAME),
        help=f"path of the submission file (default: {DEFAULT_SUBMISSION_NAME})",
    )
    args = parser.parse_args()

    if validate_submission(args.scores, DEFAULT_PROTOCOL) is None:
        return 1

    shutil.copyfile(args.scores, args.output)
    print(f"submission saved to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
