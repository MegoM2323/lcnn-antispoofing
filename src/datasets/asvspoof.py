import logging
import random
from pathlib import Path

import soundfile as sf
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.collate import DEFAULT_MAX_LEN
from src.utils.io_utils import ROOT_PATH, read_json, write_json

logger = logging.getLogger(__name__)

PROTOCOL_FILES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}

AUDIO_DIRS = {
    "train": "ASVspoof2019_LA_train",
    "dev": "ASVspoof2019_LA_dev",
    "eval": "ASVspoof2019_LA_eval",
}

PROTOCOL_DIR = "ASVspoof2019_LA_cm_protocols"

LABELS = {"bonafide": 1, "spoof": 0}


class ASVspoofDataset(BaseDataset):
    def __init__(
        self,
        part: str,
        data_dir: str,
        max_len: int | None = DEFAULT_MAX_LEN,
        random_crop: bool | None = None,
        index_dir: str | None = None,
        *args,
        **kwargs,
    ):
        self.part = part
        self.data_dir = Path(data_dir).absolute().resolve()
        self.max_len = max_len
        self.random_crop = (part == "train") if random_crop is None else random_crop

        self.protocol_path = self.data_dir / PROTOCOL_DIR / PROTOCOL_FILES[part]
        self.audio_dir = self.data_dir / AUDIO_DIRS[part] / "flac"

        index_dir_path = (
            ROOT_PATH / "data" / "index" if index_dir is None else Path(index_dir)
        )
        index_path = index_dir_path / f"asvspoof_la_{part}.json"

        index = self._load_cached_index(index_path)
        if index is None:
            index = self._create_index(index_path)

        super().__init__(index, *args, **kwargs)

    def _load_cached_index(self, index_path: Path) -> list[dict] | None:
        if not index_path.exists():
            return None

        cached = read_json(str(index_path))
        if cached["data_dir"] != str(self.data_dir):
            return None

        return cached["index"]

    def _create_index(self, index_path: Path) -> list[dict]:
        logger.info(f"Creating ASVspoof2019 LA index for '{self.part}' partition")

        index: list[dict] = []
        with self.protocol_path.open("rt") as protocol_file:
            for line_number, line in enumerate(protocol_file, start=1):
                fields = line.split()
                if not fields:
                    continue

                speaker_id, utt_id, _, attack_id, label = fields
                audio_path = self.audio_dir / f"{utt_id}.flac"
                if self.part == "eval" and not audio_path.exists():
                    raise FileNotFoundError(
                        f"Audio file {audio_path} is missing, but it is listed "
                        f"at line {line_number} of {self.protocol_path}"
                    )

                index.append(
                    {
                        "path": str(audio_path),
                        "label": LABELS[label],
                        "utt_id": utt_id,
                        "attack_id": attack_id,
                        "speaker_id": speaker_id,
                    }
                )

        index_path.parent.mkdir(exist_ok=True, parents=True)
        write_json({"data_dir": str(self.data_dir), "index": index}, str(index_path))

        return index

    def __getitem__(self, ind: int) -> dict:
        data_dict = self._index[ind]

        instance_data = {
            "data_object": self.load_object(data_dict["path"]).squeeze(0),
            "labels": data_dict["label"],
            "utt_id": data_dict["utt_id"],
        }

        return self.preprocess_data(instance_data)

    def load_object(self, path: str) -> torch.Tensor:
        with sf.SoundFile(path) as audio_file:
            offset = self._get_offset(audio_file.frames)
            if offset > 0:
                audio_file.seek(offset)
            frames = -1 if self.max_len is None else self.max_len
            audio = audio_file.read(frames=frames, dtype="float32", always_2d=True)

        return torch.from_numpy(audio).T

    def _get_offset(self, frames: int) -> int:
        if self.max_len is None or frames <= self.max_len or not self.random_crop:
            return 0
        return random.randint(0, frames - self.max_len)
