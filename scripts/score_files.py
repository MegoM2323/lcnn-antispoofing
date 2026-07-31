"""
Reading and writing of the score files the inference tools exchange.

Two formats are in use. The csv is the submission format itself, headerless
"utterance_id,score" rows. The .pth is the dump 'Inferencer' writes next to it,
with the raw logits of the whole partition; it keeps the full precision of the
forward pass, which matters when the scores are fused rather than graded.
"""

import csv
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).absolute().resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.make_submission import read_scores  # noqa: E402
from src.metrics.eer_utils import logits_to_scores  # noqa: E402

LOGITS_SUFFIX = ".pth"


def load_logits_file(path: Path) -> dict[str, float]:
    """
    Read the scores from a dump of the raw model outputs.

    Args:
        path (Path): '<part>_outputs.pth' written by the Inferencer.
    Returns:
        scores (dict[str, float]): utt_id -> score.
    """
    # weights_only=False: the dump also carries the list of the utterance ids
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "utt_id" not in payload:
        raise ValueError(f"'{path}' is not an inference dump: no 'utt_id' in it")

    utt_ids = list(payload["utt_id"])
    scores = payload.get("scores")
    if scores is None:
        if payload.get("logits") is None:
            raise ValueError(f"'{path}' carries neither scores nor logits")
        scores = logits_to_scores(payload["logits"])

    scores = scores.detach().float().reshape(-1).tolist()
    if len(scores) != len(utt_ids):
        raise ValueError(
            f"'{path}' holds {len(scores)} scores for {len(utt_ids)} utterances"
        )

    loaded = {utt_id: float(score) for utt_id, score in zip(utt_ids, scores)}
    if len(loaded) != len(utt_ids):
        raise ValueError(f"'{path}' scores some utterances more than once")
    return loaded


def load_score_file(path: str | Path) -> dict[str, float]:
    """
    Read a set of scores, whatever of the two formats it is stored in.

    Args:
        path (str | Path): csv with the scores or a '.pth' inference dump.
    Returns:
        scores (dict[str, float]): utt_id -> score.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Score file '{path}' does not exist")

    if path.suffix == LOGITS_SUFFIX:
        return load_logits_file(path)

    scores, errors = read_scores(path)
    if errors:
        listed = "\n".join(f"  - {error}" for error in errors[:10])
        raise ValueError(f"'{path}' is malformed:\n{listed}")
    if not scores:
        raise ValueError(f"'{path}' holds no scores")
    return scores


def write_score_csv(path: str | Path, scores: Mapping[str, float]) -> None:
    """
    Write the scores in the submission format.

    'repr' is used instead of a fixed format: rounding a score to a few digits
    creates ties between utterances that the model separated, and ties move
    the EER.

    Args:
        path (str | Path): destination csv.
        scores (Mapping[str, float]): utt_id -> score.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        for utt_id, score in scores.items():
            writer.writerow([utt_id, repr(float(score))])
