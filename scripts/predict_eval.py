"""
Прогон всей эвалюационной партиции обученным чекпоинтом и сборка файла
посылки.

Входной конвейер берётся из конфига обучающего запуска, а не из текущих
конфигов проекта: чекпоинт осмыслен только вместе с тем фронт-эндом, с которым
его обучали. Конфиг читается из 'config.yaml' рядом с чекпоинтом или из копии,
сохранённой внутри самого '.pth', поэтому скачанный из релиза чекпоинт
работает сам по себе.

    python3 scripts/predict_eval.py checkpoints/lfcc_epoch21.pth -o mppanin.csv
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.make_submission import (  # noqa: E402
    DEFAULT_PROTOCOL,
    DEFAULT_SUBMISSION_NAME,
    load_score_file,
    validate_submission,
    write_score_csv,
)
from src.datasets.collate import DEFAULT_MAX_LEN  # noqa: E402
from src.datasets.data_utils import get_dataloaders  # noqa: E402
from src.trainer import Inferencer  # noqa: E402
from src.utils.init_utils import set_random_seed  # noqa: E402
from src.utils.protocol import read_protocol_entries  # noqa: E402

DEFAULT_SAVE_DIR = PROJECT_ROOT / "data" / "saved" / "eval"
DATA_DIR = os.environ.get("ASVSPOOF_DIR", str(PROJECT_ROOT / "data" / "LA"))
PARTITION = "eval"
NUM_WORKERS = 4
SEED = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score the eval partition.")
    parser.add_argument("checkpoint", type=Path, help="path to the '.pth' checkpoint")
    parser.add_argument(
        "-o", "--output", type=Path, default=Path(DEFAULT_SUBMISSION_NAME)
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32)
    parser.add_argument("-d", "--device", default="auto", help="'auto', 'cpu', 'cuda'")
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=DEFAULT_SAVE_DIR,
        help="directory for the raw predictions, one per checkpoint",
    )
    return parser.parse_args()


def load_checkpoint_config(checkpoint: Path) -> DictConfig:
    """
    Читает конфиг, с которым обучался чекпоинт: копию, сохранённую рядом
    с ним, если она есть, иначе копию внутри самого '.pth'.
    """
    sidecar = checkpoint.parent / "config.yaml"
    if sidecar.exists():
        return OmegaConf.load(sidecar)

    return torch.load(checkpoint, map_location="cpu", weights_only=False)["config"]


def build_run_config(
    saved_config: DictConfig, checkpoint: Path, device: str, batch_size: int
) -> DictConfig:
    """
    Собирает конфиг прогона: всё, что определяет вход модели, берётся из
    обучающего запуска, из командной строки приходят только размер батча
    и устройство.
    """
    collate_max_len = int(saved_config.get("collate_max_len", DEFAULT_MAX_LEN))
    transforms = saved_config.get("transforms", {})
    batch_transforms = transforms.get("batch_transforms", {})
    instance_transforms = transforms.get("instance_transforms", {})

    # запуски, сделанные до отказа от эмбеддинг-головы, хранят флаг, который
    # модель больше не принимает, поэтому сохранённый конфиг копируется без него
    model = OmegaConf.create(
        {
            key: value
            for key, value in OmegaConf.to_container(
                saved_config.get("model"), resolve=True
            ).items()
            if key != "return_embedding"
        }
    )

    return OmegaConf.create(
        {
            "model": model,
            "transforms": {
                "batch_transforms": {"inference": batch_transforms.get("inference")},
                "instance_transforms": instance_transforms,
            },
            "collate_max_len": collate_max_len,
            "datasets": {
                PARTITION: {
                    "_target_": "src.datasets.ASVspoofDataset",
                    "part": PARTITION,
                    "data_dir": DATA_DIR,
                    "max_len": collate_max_len,
                    "random_crop": False,
                    "instance_transforms": instance_transforms.get("inference"),
                }
            },
            "dataloader": {
                "_target_": "torch.utils.data.DataLoader",
                "batch_size": batch_size,
                "num_workers": NUM_WORKERS,
                "pin_memory": device != "cpu",
            },
            "inferencer": {
                "device_tensors": ["data_object", "labels"],
                "device": device,
                # скоры посылки считаются в fp32, bf16 меняет только скорость
                "use_amp": False,
                "from_pretrained": str(checkpoint),
            },
        }
    )


def run_inference(config: DictConfig, device: str, save_dir: Path) -> Path:
    """Прогоняет эвалюационную партицию и возвращает csv с сырыми скорами."""
    set_random_seed(SEED, cudnn_benchmark=False)

    dataloaders, batch_transforms = get_dataloaders(config, device)
    model = instantiate(config.model).to(device)

    save_dir.mkdir(parents=True, exist_ok=True)
    Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_dir,
        metrics=None,
        skip_model_load=False,
    ).run_inference()

    return save_dir / f"{PARTITION}_scores.csv"


def build_submission(scores_path: Path, protocol_path: str, output: Path) -> int:
    """
    Проверяет предсказания и пишет файл посылки.

    Файл создаётся только после проверки: csv, покрывающий протокол не
    полностью, даёт KeyError в проверяющем скрипте и ноль за всю работу,
    поэтому такого файла лучше не иметь на диске вовсе. Возвращает код
    возврата.
    """
    if validate_submission(scores_path, protocol_path) is None:
        return 1

    scores = load_score_file(scores_path)
    entries = read_protocol_entries(protocol_path)
    # строки идут в порядке протокола, поэтому два прогона дают побайтово
    # одинаковые файлы
    write_score_csv(output, {entry.utt_id: scores[entry.utt_id] for entry in entries})

    print(f"submission written to {output.resolve()}")
    return 0


def main() -> int:
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = build_run_config(
        load_checkpoint_config(args.checkpoint),
        args.checkpoint,
        device,
        args.batch_size,
    )
    scores_path = run_inference(config, device, args.save_dir)

    return build_submission(scores_path, DEFAULT_PROTOCOL, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
