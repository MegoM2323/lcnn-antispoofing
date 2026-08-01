"""
Индекс партиций ASVspoof2019 LA, собранный по официальным CM-протоколам,
вместе с чтением и обрезкой отдельной записи.
"""

import json
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
    """
    Датасет ASVspoof2019 Logical Access (LA) для голосового антиспуфинга.

    Каждый элемент это моно-запись 16 кГц с бинарной меткой: 1 для bonafide
    (настоящая речь) и 0 для подделки.

    Индекс строится по официальным файлам CM-протокола и кэшируется на диск
    в виде JSON, чтобы не разбирать протокол при каждом запуске.
    """

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
        """
        Аргументы:
            part (str): партиция датасета, одна из "train", "dev", "eval".
            data_dir (str): путь к корню LA, то есть к каталогу, в котором
                лежат ASVspoof2019_LA_train, ASVspoof2019_LA_cm_protocols
                и остальное.
            max_len (int | None): если не None, записи длиннее max_len
                отсчётов обрезаются до max_len прямо при загрузке. Это
                экономит память в воркерах даталоадера, окончательная длина
                задаётся в collate_fn.
            random_crop (bool | None): если True, длинная запись обрезается
                со случайной позиции, иначе берутся первые max_len отсчётов.
                None означает True для партиции "train" и False для
                остальных.
            index_dir (str | None): каталог для кэша индекса. По умолчанию
                ROOT_PATH / "data" / "index".
        """
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
        """
        Читает кэш индекса, если он не был собран для другого корпуса: в кэше
        лежат абсолютные пути, поэтому при смене ASVSPOOF_DIR данные молча
        поехали бы из устаревшего индекса. None означает, что индекс надо
        построить заново.
        """
        if not index_path.exists():
            return None

        try:
            cached = read_json(str(index_path))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Cannot read the cached index {index_path} ({e})")
            return None

        if cached.get("data_dir") != str(self.data_dir):
            logger.info(f"{index_path} was built for another data_dir, rebuilding it")
            return None

        return cached["index"]

    def _create_index(self, index_path: Path) -> list[dict]:
        """
        Разбирает CM-протокол партиции и строит индекс датасета.

        Каждый элемент получает путь к аудио, бинарную метку и метаданные
        записи; результат кэшируется в 'index_path'.
        """
        logger.info(f"Creating ASVspoof2019 LA index for '{self.part}' partition")

        index: list[dict] = []
        missing: list[str] = []
        with self.protocol_path.open("rt") as protocol_file:
            for line_number, line in enumerate(protocol_file, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5 or fields[4] not in LABELS:
                    raise ValueError(
                        f"{self.protocol_path}:{line_number}: expected 5 fields "
                        f"ending with a known label, got '{line.strip()}'"
                    )

                speaker_id, utt_id, _, attack_id, label = fields
                audio_path = self.audio_dir / f"{utt_id}.flac"
                if not audio_path.exists():
                    if self.part == "eval":
                        # проверяющий скрипт ищет в посылке каждый
                        # идентификатор из протокола, поэтому один
                        # пропущенный файл это KeyError и отклонённая посылка
                        raise FileNotFoundError(
                            f"Audio file {audio_path} is missing, but it is "
                            f"listed at line {line_number} of "
                            f"{self.protocol_path}. The eval index has to cover "
                            "the whole protocol: an incomplete submission is "
                            "not gradeable. Check that the corpus is unpacked."
                        )
                    missing.append(utt_id)
                    continue

                index.append(
                    {
                        "path": str(audio_path),
                        "label": LABELS[label],
                        "utt_id": utt_id,
                        "attack_id": attack_id,
                        "speaker_id": speaker_id,
                    }
                )

        if missing:
            logger.warning(
                f"Skipped {len(missing)} utterances of the '{self.part}' "
                f"partition: audio files are missing (e.g. {missing[:5]})"
            )

        index_path.parent.mkdir(exist_ok=True, parents=True)
        write_json({"data_dir": str(self.data_dir), "index": index}, str(index_path))

        return index

    def __getitem__(self, ind: int) -> dict:
        """
        Загружает запись и объединяет её с метаданными в словарь с аудио
        ("data_object"), меткой ("labels") и идентификатором ("utt_id").
        """
        data_dict = self._index[ind]

        instance_data = {
            "data_object": self.load_object(data_dict["path"]).squeeze(0),
            "labels": data_dict["label"],
            "utt_id": data_dict["utt_id"],
        }

        return self.preprocess_data(instance_data)

    def load_object(self, path: str) -> torch.Tensor:
        """
        Загружает моно-сигнал с диска, обрезая его до max_len отсчётов.

        Начиная с torchaudio 2.9 функция torchaudio.load требует бэкенд
        torchcodec, поэтому здесь напрямую используется soundfile. Чтение
        только нужного куска файла держит потребление памяти воркерами
        низким.
        """
        try:
            with sf.SoundFile(path) as audio_file:
                offset = self._get_offset(audio_file.frames)
                if offset > 0:
                    audio_file.seek(offset)
                frames = -1 if self.max_len is None else self.max_len
                audio = audio_file.read(frames=frames, dtype="float32", always_2d=True)
        except (sf.LibsndfileError, RuntimeError, OSError) as e:
            logger.error(f"Failed to read audio file {path}: {e}")
            raise RuntimeError(f"Cannot read audio file {path}") from e

        data_object = torch.from_numpy(audio).T
        if data_object.shape[0] > 1:
            data_object = data_object.mean(dim=0, keepdim=True)

        return data_object

    def _get_offset(self, frames: int) -> int:
        """
        Позиция начала обрезки для записи длиной 'frames' отсчётов.
        """
        if self.max_len is None or frames <= self.max_len or not self.random_crop:
            return 0
        return random.randint(0, frames - self.max_len)
