"""
Index of the ASVspoof2019 LA partitions built from the official CM protocols,
with the reading and the cropping of a single utterance.
"""

import hashlib
import json
import logging
import random
from collections.abc import Mapping
from pathlib import Path

import soundfile as sf
import torch

from src.datasets.base_dataset import BaseDataset
from src.datasets.collate import DEFAULT_MAX_LEN
from src.utils.io_utils import ROOT_PATH, read_json, write_json

logger = logging.getLogger(__name__)

PARTS = ("train", "dev", "eval")

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

# bumped whenever the layout of the cached index changes
INDEX_FORMAT_VERSION = 1


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
        if part not in PARTS:
            raise ValueError(f"Unknown partition '{part}', expected one of {PARTS}")
        if max_len is not None and max_len <= 0:
            raise ValueError(f"max_len should be positive, got {max_len}")

        self.part = part
        self.data_dir = Path(data_dir)
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

    def _fingerprint(self) -> dict | None:
        """
        Identity of the data the index is built from.

        The index stores absolute paths and labels, so it is only valid for
        one corpus location and one revision of the protocol: without this
        check a changed ASVSPOOF_DIR or an updated protocol would be served
        from a stale cache.

        Returns:
            fingerprint (dict | None): format version, resolved data_dir,
                protocol name and the md5 of its content. None if the protocol
                cannot be read.
        """
        try:
            protocol_md5 = hashlib.md5(self.protocol_path.read_bytes()).hexdigest()
        except OSError:
            return None

        return {
            "version": INDEX_FORMAT_VERSION,
            "data_dir": str(self.data_dir.absolute().resolve()),
            "protocol": PROTOCOL_FILES[self.part],
            "protocol_md5": protocol_md5,
        }

    def _load_cached_index(self, index_path: Path) -> list[dict] | None:
        """
        Read the cached index if it was built from the very same data.

        Args:
            index_path (Path): path of the cached index.
        Returns:
            index (list[dict] | None): cached index, None if it is absent,
                unreadable or stale, in which case it has to be rebuilt.
        """
        if not index_path.exists():
            return None

        try:
            cached = read_json(str(index_path))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(
                f"Cannot read the cached index {index_path} ({e}), rebuilding it"
            )
            return None

        if not isinstance(cached, Mapping) or "index" not in cached:
            logger.info(
                f"The cached index {index_path} carries no provenance data "
                "and cannot be verified, rebuilding it"
            )
            return None

        fingerprint = self._fingerprint()
        if fingerprint is None:
            logger.warning(
                f"Protocol {self.protocol_path} is not readable: the cached "
                f"index {index_path} is used without verification"
            )
            return cached["index"]

        if cached.get("fingerprint") != fingerprint:
            logger.info(
                f"The cached index {index_path} was built for another data_dir "
                "or another revision of the protocol, rebuilding it"
            )
            return None

        return cached["index"]

    def _create_index(self, index_path: Path) -> list[dict]:
        """
        Parse the CM protocol of the partition and build the dataset index.

        Args:
            index_path (Path): path where the created index is cached.
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset with path, label and utterance metadata.
        """
        protocol_path = self.protocol_path
        audio_dir = self.audio_dir

        if not protocol_path.exists():
            raise FileNotFoundError(f"Protocol file is not found: {protocol_path}")
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory is not found: {audio_dir}")

        logger.info(f"Creating ASVspoof2019 LA index for '{self.part}' partition")

        index: list[dict] = []
        missing: list[str] = []
        total = 0
        with protocol_path.open("rt") as protocol_file:
            for line_number, line in enumerate(protocol_file, start=1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(
                        f"Malformed protocol line {line_number} in {protocol_path}: "
                        f"expected 5 fields, got {len(fields)}"
                    )

                speaker_id, utt_id, _, attack_id, label = fields
                if label not in LABELS:
                    raise ValueError(
                        f"Unknown label '{label}' at line {line_number} "
                        f"in {protocol_path}"
                    )

                total += 1
                audio_path = audio_dir / f"{utt_id}.flac"
                if not audio_path.exists():
                    if self.part == "eval":
                        # the grading script indexes the submission by every
                        # utterance id of the protocol, so a single missing
                        # file means a KeyError and a rejected submission
                        raise FileNotFoundError(
                            f"Audio file {audio_path} is missing, but it is "
                            f"listed at line {line_number} of {protocol_path}. "
                            "The eval index has to cover the whole protocol: "
                            "an incomplete submission is not gradeable. Check "
                            "that the corpus is fully unpacked."
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
            examples = ", ".join(missing[:5])
            logger.warning(
                f"Skipped {len(missing)} of {total} utterances of the "
                f"'{self.part}' partition: audio files are missing "
                f"(e.g. {examples})"
            )
        if len(index) == 0:
            raise RuntimeError(
                f"Index for '{self.part}' partition is empty, "
                f"no audio files found in {audio_dir}"
            )

        index_path.parent.mkdir(exist_ok=True, parents=True)
        write_json(
            {"fingerprint": self._fingerprint(), "index": index}, str(index_path)
        )

        return index

    def __getitem__(self, ind: int) -> dict:
        """
        Load an utterance and combine it with its metadata into a dict.

        Args:
            ind (int): index in the self._index list.
        Returns:
            instance_data (dict): dict with the audio ("data_object"),
                the binary label ("labels") and the utterance id ("utt_id").
        """
        data_dict = self._index[ind]

        audio = self.load_object(data_dict["path"])

        instance_data = {
            "data_object": audio.squeeze(0),
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

        Args:
            path (str): path to the audio file.
        Returns:
            data_object (Tensor): float32 waveform of shape (1, T).
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
        Get the start position of the cut for an utterance of 'frames' samples.

        Args:
            frames (int): total number of samples in the audio file.
        Returns:
            offset (int): index of the first sample to read.
        """
        if self.max_len is None or frames <= self.max_len:
            return 0
        if not self.random_crop:
            return 0
        return random.randint(0, frames - self.max_len)
