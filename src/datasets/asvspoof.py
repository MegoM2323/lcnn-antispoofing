"""
Index of the ASVspoof2019 LA partitions built from the official CM protocols,
with the reading and the cropping of a single utterance.
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
    ASVspoof2019 Logical Access (LA) dataset for voice anti-spoofing.

    Each element is a mono 16 kHz utterance with a binary label:
    1 for bonafide (genuine) speech and 0 for spoofed speech.

    The index is built from the official CM protocol files and cached
    on disk as a JSON file to avoid re-parsing on every run.
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
        Args:
            part (str): dataset partition, one of "train", "dev", "eval".
            data_dir (str): path to the LA root directory, the one that
                contains ASVspoof2019_LA_train, ASVspoof2019_LA_cm_protocols,
                etc.
            max_len (int | None): if not None, utterances longer than max_len
                samples are cut to max_len during loading. It saves memory in
                the dataloader workers, the final length is set in collate_fn.
            random_crop (bool | None): if True, the cut of a long utterance
                starts at a random position, otherwise the first max_len
                samples are taken. If None, defaults to True for the
                "train" partition and to False otherwise.
            index_dir (str | None): directory for the cached index. Defaults
                to ROOT_PATH / "data" / "index".
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
        Read the cached index, unless it was built for another corpus: it
        stores absolute paths, so a changed ASVSPOOF_DIR would silently be
        served from a stale cache. None means the index has to be rebuilt.
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
        Parse the CM protocol of the partition and build the dataset index.

        Every element gets its audio path, its binary label and the metadata
        of the utterance; the result is cached in 'index_path'.
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
                        # the grading script indexes the submission by every
                        # utterance id of the protocol, so a single missing
                        # file means a KeyError and a rejected submission
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
        Load an utterance and combine it with its metadata into a dict with
        the audio ("data_object"), the label ("labels") and the id ("utt_id").
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
        Load a mono waveform from disk, cutting it to max_len samples.

        torchaudio.load requires the torchcodec backend since torchaudio 2.9,
        so soundfile is used directly. Reading only the required part of the
        file keeps the memory footprint of the workers low.
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
        Start position of the cut for an utterance of 'frames' samples.
        """
        if self.max_len is None or frames <= self.max_len or not self.random_crop:
            return 0
        return random.randint(0, frames - self.max_len)
