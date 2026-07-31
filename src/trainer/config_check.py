"""
Consistency check between the config a checkpoint was trained with and the
config of the run that loads it.

'load_state_dict' only validates the shapes of the weights, so everything that
shapes the input without changing them passes silently: the length the
waveforms are padded to, the window and the hop of the STFT, the number of
frames, the way a long utterance is cropped. A measurement on the trained
model showed that a foreign 'collate_max_len' alone moves the eval EER by
0.33 points and flips the sign of the score for part of the utterances, which
is exactly the kind of error a submission cannot recover from. Hence the
loaded checkpoint is compared with the current config and every difference is
reported.
"""

from collections.abc import Mapping, Sequence

from omegaconf import DictConfig, ListConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from src.datasets.collate import DEFAULT_MAX_LEN

# parameters of the inference front-end that change the input of the model
FRONTEND_KEYS = ("n_fft", "win_length", "hop_length", "window", "n_frames", "crop")

FRONTEND_PATH = ("transforms", "batch_transforms", "inference")

ABSENT = "<absent>"


def to_plain(node):
    """
    Convert an omegaconf node into plain python containers.

    Args:
        node (Any): config node or an already plain value.
    Returns:
        node (Any): dict/list/scalar with the interpolations resolved when
            possible (an unresolvable interpolation is kept as is).
    """
    if not isinstance(node, (DictConfig, ListConfig)):
        return node
    try:
        return OmegaConf.to_container(node, resolve=True)
    except OmegaConfBaseException:
        # a checkpoint config may reference keys that the current run does not
        # define; an unresolved value is still better than no comparison at all
        return OmegaConf.to_container(node, resolve=False)


def select(config, path):
    """
    Follow a chain of keys in a config, tolerating missing intermediate nodes.

    Args:
        config (Mapping | None): config to read.
        path (tuple[str]): chain of keys.
    Returns:
        node (Any | None): the value at the path, None if it is not there.
    """
    node = config
    for key in path:
        if not isinstance(node, Mapping) or key not in node:
            return None
        node = node[key]
    return node


def collate_max_len(config):
    """
    Number of samples every waveform of a batch is brought to.

    Args:
        config (Mapping | None): run config.
    Returns:
        value (int): configured length or the collate_fn default.
        is_default (bool): True if the config does not set the key. Old
            checkpoints were trained before the key existed, and back then
            'collate_fn' padded to DEFAULT_MAX_LEN.
    """
    value = select(config, ("collate_max_len",))
    if value is None:
        return DEFAULT_MAX_LEN, True
    return int(value), False


def frontend_params(config):
    """
    Collect the front-end parameters of the inference batch transform.

    The transform is a nested structure (a Sequential with a list of
    submodules), so the whole subtree is walked and the first value found for
    every key of FRONTEND_KEYS is taken.

    Args:
        config (Mapping | None): run config.
    Returns:
        params (dict): found parameters, missing ones are simply not there.
    """
    params: dict = {}
    _collect_params(to_plain(select(config, FRONTEND_PATH)), params)
    return params


def _is_container(value):
    """
    Check whether a config value has children to descend into. Strings are
    scalars here: 'window: blackman' and 'crop: first' are front-end values.

    Args:
        value (Any): value from a plain config subtree.
    Returns:
        is_container (bool): True for mappings and non-string sequences.
    """
    if isinstance(value, Mapping):
        return True
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _collect_params(node, params):
    """
    Depth-first search of FRONTEND_KEYS in a transform definition.

    Args:
        node (Any): current node of the plain config subtree.
        params (dict): accumulator, modified in place.
    """
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key in FRONTEND_KEYS and not _is_container(value):
                params.setdefault(key, value)
            else:
                _collect_params(value, params)
    elif _is_container(node):
        for item in node:
            _collect_params(item, params)


def _describe(value):
    """
    Format a config value for the report.

    Args:
        value (Any): value to format.
    Returns:
        text (str): value or the ABSENT placeholder.
    """
    return ABSENT if value is None else str(value)


def _diff_mappings(saved, current, prefix):
    """
    Compare two flat mappings key by key.

    Args:
        saved (Mapping): values from the checkpoint.
        current (Mapping): values from the current config.
        prefix (str): text put in front of every key in the report.
    Returns:
        mismatches (list[str]): one line per differing key.
    """
    mismatches = []
    for key in sorted(set(saved) | set(current)):
        if saved.get(key) != current.get(key):
            mismatches.append(
                f"{prefix}{key}: trained with {_describe(saved.get(key))}, "
                f"now {_describe(current.get(key))}"
            )
    return mismatches


def config_mismatches(saved_config, current_config):
    """
    Compare the parts of the two configs that define the input of the model.

    Args:
        saved_config (Mapping | None): config stored in the checkpoint.
        current_config (Mapping | None): config of the current run.
    Returns:
        mismatches (list[str]): description of every difference, empty if the
            two runs feed the model with the same kind of input.
    """
    if saved_config is None or current_config is None:
        return []

    mismatches = []

    saved_len, saved_is_default = collate_max_len(saved_config)
    current_len, current_is_default = collate_max_len(current_config)
    if saved_len != current_len:
        default_note = " (the key is absent, collate_fn default)"
        mismatches.append(
            "collate_max_len: trained with "
            f"{saved_len}{default_note if saved_is_default else ''}, "
            f"now {current_len}{default_note if current_is_default else ''}"
        )

    mismatches += _diff_mappings(
        frontend_params(saved_config),
        frontend_params(current_config),
        prefix="front-end ",
    )

    saved_model = to_plain(select(saved_config, ("model",)))
    current_model = to_plain(select(current_config, ("model",)))
    if isinstance(saved_model, Mapping) and isinstance(current_model, Mapping):
        mismatches += _diff_mappings(saved_model, current_model, prefix="model.")
    elif saved_model != current_model:
        mismatches.append(
            f"model: trained with {_describe(saved_model)}, "
            f"now {_describe(current_model)}"
        )

    return mismatches


def format_mismatch_warning(mismatches, checkpoint_path):
    """
    Build the text of the warning printed when the configs disagree.

    Args:
        mismatches (list[str]): output of 'config_mismatches'.
        checkpoint_path (str): path of the loaded checkpoint.
    Returns:
        text (str): multiline warning.
    """
    header = (
        "=" * 72 + "\nCONFIG MISMATCH: the checkpoint "
        f"'{checkpoint_path}' was trained with another input pipeline"
    )
    body = "\n".join(f"  * {mismatch}" for mismatch in mismatches)
    footer = (
        "The weights fit a different input, so the scores are not comparable "
        "with the ones of the training run.\nAlign the config with "
        "saved/<run_name>/config.yaml before making a submission.\n" + "=" * 72
    )
    return f"{header}\n{body}\n{footer}"
