"""
Averaging the weights of several checkpoints of one run into a single model.

The last epochs of a run walk around the same basin of the loss surface, and
the point in the middle of that walk usually generalizes better than any of the
points themselves (arXiv:1803.05407). The average is one model with the
architecture of the original: nothing is added to the LCNN, only its weights
are replaced, so the result is still a single system and not an ensemble.

The catch is BatchNorm. Its running mean and variance are not parameters that
were optimized, they are statistics of the activations of a particular set of
weights, and averaging them across checkpoints describes no model at all. They
have to be measured again, by pushing data through the averaged weights; see
'recalibrate_batchnorm'.
"""

from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import nn

# the base class of BatchNorm1d/2d/3d, the same handle torch.optim.swa_utils
# uses to find the layers whose statistics have to be measured again
from torch.nn.modules.batchnorm import _BatchNorm

# integer buffers (num_batches_tracked) are counters, not weights: averaging
# them is meaningless, and they are overwritten by the recalibration anyway
AVERAGED_DTYPES = (torch.float16, torch.float32, torch.float64, torch.bfloat16)


def average_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Average the tensors of several state dicts entry by entry.

    Args:
        state_dicts (Sequence[Mapping[str, Tensor]]): state dicts of the
            checkpoints to average. They must hold the same keys, otherwise the
            models are not the same architecture.
    Returns:
        averaged (dict[str, Tensor]): state dict with the averaged weights.
            Integer entries are taken from the first state dict.
    """
    if not state_dicts:
        raise ValueError("Nothing to average: no state dicts were given")

    reference = state_dicts[0]
    for position, state_dict in enumerate(state_dicts[1:], start=2):
        if set(state_dict) != set(reference):
            missing = sorted(set(reference) - set(state_dict))
            extra = sorted(set(state_dict) - set(reference))
            raise ValueError(
                f"Checkpoint {position} has another set of weights: "
                f"{len(missing)} are missing (e.g. {missing[:3]}), "
                f"{len(extra)} are new (e.g. {extra[:3]})"
            )

    averaged = {}
    for key, value in reference.items():
        if value.dtype not in AVERAGED_DTYPES:
            averaged[key] = value.clone()
            continue

        total = value.detach().float().clone()
        for state_dict in state_dicts[1:]:
            other = state_dict[key]
            if other.shape != value.shape:
                raise ValueError(
                    f"'{key}' has shape {tuple(other.shape)} in one checkpoint "
                    f"and {tuple(value.shape)} in another"
                )
            total += other.detach().float()
        averaged[key] = (total / len(state_dicts)).to(value.dtype)

    return averaged


def batchnorm_modules(model: nn.Module) -> list[_BatchNorm]:
    """
    Collect the BatchNorm layers of a model that keep running statistics.

    Args:
        model (nn.Module): model to inspect.
    Returns:
        modules (list[_BatchNorm]): BatchNorm layers with track_running_stats.
    """
    return [
        module
        for module in model.modules()
        if isinstance(module, _BatchNorm) and module.track_running_stats
    ]


@torch.no_grad()
def recalibrate_batchnorm(
    model: nn.Module,
    batches: Iterable[torch.Tensor],
    max_batches: int | None = None,
    progress: bool = True,
) -> int:
    """
    Measure the BatchNorm statistics of the averaged weights on real data.

    The momentum of every layer is set to None, so the running values become
    the plain cumulative average over the batches that are pushed through and
    do not depend on the order in which they arrive. Only the BatchNorm layers
    are put into training mode: dropout with p=0.75 in front of the BatchNorm
    of the head would inflate the variance it measures, and the inference pass
    the statistics are meant for runs without it.

    Args:
        model (nn.Module): model with the averaged weights.
        batches (Iterable[Tensor]): batches of model input, already on the
            device of the model and passed through the front-end.
        max_batches (int | None): stop after that many batches, None to use the
            whole iterable.
        progress (bool): print how many batches were processed.
    Returns:
        seen (int): number of batches the statistics were measured on.
    """
    layers = batchnorm_modules(model)
    if not layers:
        raise ValueError("The model has no BatchNorm layers to recalibrate")

    was_training = model.training
    momenta = [layer.momentum for layer in layers]

    model.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()

    seen = 0
    try:
        for batch in batches:
            if max_batches is not None and seen >= max_batches:
                break
            model(data_object=batch)
            seen += 1
            if progress and seen % 50 == 0:
                print(f"  batchnorm recalibration: {seen} batches")
    finally:
        for layer, momentum in zip(layers, momenta):
            layer.momentum = momentum
        model.train(was_training)

    if seen == 0:
        raise ValueError(
            "No batch reached the model: the BatchNorm statistics would stay "
            "uninitialized and the scores would be meaningless"
        )

    return seen
