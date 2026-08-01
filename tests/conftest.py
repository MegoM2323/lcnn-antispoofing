"""
The tests never touch the real ASVspoof2019 LA corpus: the fixture builds a
tiny synthetic one with the same layout and protocol format in a temporary
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
# audio directory, protocol file and utterance id prefix of every partition
PARTITIONS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", "LA_T"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt", "LA_E"),
}


def build_partition(root: Path, part: str, n_utterances: int = 6) -> None:
    """Write the audio and the CM protocol of one partition."""
    audio_dir, protocol_name, prefix = PARTITIONS[part]
    rng = np.random.default_rng(0)

    lines = []
    for i in range(n_utterances):
        # every third utterance is bonafide: both classes are always present,
        # otherwise the EER is undefined
        is_bonafide = i % 3 == 0
        utt_id = f"{prefix}_{1000000 + i}"
        # 1 s exercises the padding branch of the collate function, 5 s the
        # cropping one
        duration = 1.0 if i % 2 else 5.0

        path = root / audio_dir / "flac" / f"{utt_id}.flac"
        path.parent.mkdir(parents=True, exist_ok=True)
        audio = rng.standard_normal(int(duration * SR)).astype(np.float32) * 0.1
        sf.write(path, audio, SR, format="FLAC", subtype="PCM_16")

        attack_id = "-" if is_bonafide else f"A{i % 6 + 1:02d}"
        label = "bonafide" if is_bonafide else "spoof"
        lines.append(f"LA_{i:04d} {utt_id} - {attack_id} {label}\n")

    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    (protocol_dir / protocol_name).write_text("".join(lines))


@pytest.fixture
def la_root(tmp_path):
    """A synthetic LA root: the directory holding ASVspoof2019_LA_*."""
    root = tmp_path / "LA"
    for part in PARTITIONS:
        build_partition(root, part)
    return root
