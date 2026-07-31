"""
Shared fixtures for the test suite.

The tests never touch the real ASVspoof2019 LA corpus: every fixture builds a
tiny synthetic dataset with the same layout and protocol format in a temporary
directory, so the whole suite runs on CPU in a few seconds.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SR = 16000

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
PREFIXES = {"train": "LA_T", "dev": "LA_D", "eval": "LA_E"}


def write_utterance(path: Path, duration: float, is_bonafide: bool, seed: int) -> None:
    """
    Write a single mono 16 kHz flac file whose class is encoded in its spectrum.

    Args:
        path (Path): destination file.
        duration (float): length of the utterance in seconds.
        is_bonafide (bool): class of the utterance.
        seed (int): seed of the noise generator.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration * SR)
    t = np.arange(n_samples, dtype=np.float32) / SR
    tone = 1000.0 if is_bonafide else 3200.0

    audio = 0.05 * rng.standard_normal(n_samples).astype(np.float32)
    audio += 0.3 * np.sin(2 * np.pi * tone * t).astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, SR, format="FLAC", subtype="PCM_16")


def build_partition(
    root: Path,
    part: str,
    n_utterances: int = 10,
    durations: tuple = (1.0, 5.0),
) -> list:
    """
    Create the audio and the CM protocol of one partition.

    Args:
        root (Path): LA root directory (the one holding ASVspoof2019_LA_*).
        part (str): "train", "dev" or "eval".
        n_utterances (int): number of utterances to generate.
        durations (tuple): durations cycled through, in seconds. Values below
            4.04 s exercise the padding branch of the collate function.
    Returns:
        entries (list[tuple[str, int]]): (utterance_id, label) pairs, where the
            label is 1 for bonafide and 0 for spoof.
    """
    audio_dir = root / AUDIO_DIRS[part] / "flac"
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    lines = []
    for i in range(n_utterances):
        # every third utterance is bonafide: both classes are always present,
        # otherwise the EER is undefined
        is_bonafide = i % 3 == 0
        utt_id = f"{PREFIXES[part]}_{1000000 + i}"
        attack_id = "-" if is_bonafide else f"A{i % 6 + 1:02d}"
        label = "bonafide" if is_bonafide else "spoof"

        write_utterance(
            audio_dir / f"{utt_id}.flac",
            durations[i % len(durations)],
            is_bonafide,
            seed=i,
        )
        lines.append(f"LA_{i:04d} {utt_id} - {attack_id} {label}\n")
        entries.append((utt_id, int(is_bonafide)))

    (protocol_dir / PROTOCOL_FILES[part]).write_text("".join(lines))
    return entries


@pytest.fixture
def la_root(tmp_path):
    """
    A synthetic ASVspoof2019 LA root with all three partitions.

    Returns:
        root (Path): directory that contains ASVspoof2019_LA_train, etc.
    """
    root = tmp_path / "LA"
    for part in ("train", "dev", "eval"):
        build_partition(root, part)
    return root


@pytest.fixture
def index_dir(tmp_path):
    """Directory for the cached dataset index, never the repository one."""
    path = tmp_path / "index"
    path.mkdir()
    return path
