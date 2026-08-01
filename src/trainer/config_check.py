"""
Check that a checkpoint is loaded with the input pipeline it was trained with.

'load_state_dict' validates the shapes of the weights and nothing else, so a
foreign waveform length, window or hop is accepted silently: a measurement on
the trained model showed that a foreign 'collate_max_len' alone moves the eval
EER by 0.33 points. Hence the front-end of the checkpoint is compared with the
current one and every difference is reported.
"""

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

# front-end settings that change the input of the model without changing the
# shapes of its weights
FRONTEND_KEYS = ("n_fft", "win_length", "hop_length", "window", "n_frames", "crop")

FRONTEND_PATH = ("transforms", "batch_transforms", "inference")


def _collect(node: Any, params: dict[str, Any]) -> None:
    """
    Depth-first search of FRONTEND_KEYS in a transform definition: the
    front-end is a Sequential with a list of submodules, so the whole subtree
    has to be walked. 'params' is the accumulator, modified in place.
    """
    if isinstance(node, (DictConfig, ListConfig)):
        node = OmegaConf.to_container(node, resolve=False)

    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in FRONTEND_KEYS and not isinstance(value, (Mapping, list)):
                params.setdefault(key, value)
            else:
                _collect(value, params)
    elif isinstance(node, list):
        for item in node:
            _collect(item, params)


def frontend_params(config: Mapping | None) -> dict[str, Any]:
    """
    Collect everything that shapes the input of the model: the length every
    waveform is brought to and the parameters of the inference front-end.
    """
    if not isinstance(config, Mapping):
        return {}

    params: dict[str, Any] = {"collate_max_len": config.get("collate_max_len")}
    node: Any = config
    for key in FRONTEND_PATH:
        node = node.get(key) if isinstance(node, Mapping) else None
    _collect(node, params)
    return params


def config_mismatches(
    saved_config: Mapping | None, current_config: Mapping | None
) -> list[str]:
    """
    Compare the input pipelines of two runs, one line per differing parameter.
    """
    saved = frontend_params(saved_config)
    current = frontend_params(current_config)
    return [
        f"{key}: trained with {saved.get(key)}, now {current.get(key)}"
        for key in sorted(set(saved) | set(current))
        if saved.get(key) != current.get(key)
    ]


def format_mismatch_warning(mismatches: list[str], checkpoint_path: str) -> str:
    """
    Warning printed when the two pipelines disagree.
    """
    return (
        f"'{checkpoint_path}' was trained with another input pipeline "
        f"({'; '.join(mismatches)}), so its scores are not comparable with the "
        "ones of the training run."
    )
