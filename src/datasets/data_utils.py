from itertools import repeat

from hydra.utils import instantiate
from torch.utils.data import Subset

from src.datasets.collate import DEFAULT_MAX_LEN, get_collate_fn
from src.datasets.multicrop import (
    DEFAULT_N_SEGMENTS,
    DEFAULT_SHORT_SHIFTS,
    get_multicrop_collate_fn,
)
from src.utils.init_utils import set_worker_seed


def inf_loop(dataloader):
    """
    Wrapper function for endless dataloader.
    Used for iteration-based training scheme.

    Args:
        dataloader (DataLoader): classic finite dataloader.
    """
    for loader in repeat(dataloader):
        yield from loader


def move_batch_transforms_to_device(batch_transforms, device):
    """
    Move batch_transforms to device.

    Notice that batch transforms are applied on the batch
    that may be on GPU. Therefore, it is required to put
    batch transforms on the device. We do it here.

    Batch transforms are required to be an instance of nn.Module.
    If several transforms are applied sequentially, use nn.Sequential
    in the config (not torchvision.Compose).

    Args:
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
        device (str): device to use for batch transforms.
    """
    for transform_type in batch_transforms.keys():
        transforms = batch_transforms.get(transform_type)
        if transforms is not None:
            for transform_name in transforms.keys():
                transforms[transform_name] = transforms[transform_name].to(device)


def get_dataloaders(config, device):
    """
    Create dataloaders for each of the dataset partitions.
    Also creates instance and batch transforms.

    Args:
        config (DictConfig): hydra experiment config.
        device (str): device to use for batch transforms.
    Returns:
        dataloaders (dict[DataLoader]): dict containing dataloader for a
            partition defined by key.
        batch_transforms (dict[Callable] | None): transforms that
            should be applied on the whole batch. Depend on the
            tensor name.
    """
    # transforms or augmentations init
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    # dataset partitions init
    datasets = instantiate(config.datasets)  # instance transforms are defined inside

    # the waveform length of a batch is shared by all the partitions:
    # the model input size depends on it
    collate = get_collate_fn(config.get("collate_max_len", DEFAULT_MAX_LEN))

    # dataloaders init
    dataloaders = {}
    for dataset_partition in config.datasets.keys():
        dataset = datasets[dataset_partition]

        assert config.dataloader.batch_size <= len(dataset), (
            f"The batch size ({config.dataloader.batch_size}) cannot "
            f"be larger than the dataset length ({len(dataset)})"
        )

        partition_dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate,
            drop_last=(dataset_partition == "train"),
            shuffle=(dataset_partition == "train"),
            worker_init_fn=set_worker_seed,
        )
        dataloaders[dataset_partition] = partition_dataloader

    return dataloaders, batch_transforms


def select_utterances(dataset, utt_ids):
    """
    Keep only the requested utterances of a partition.

    Used by the measurement runs: comparing two ways of scoring on a few
    thousand trials answers the same question as a full pass over 71237 of
    them, and does it in minutes.

    Args:
        dataset (Dataset): dataset of the whole partition.
        utt_ids (set[str]): utterance ids to keep.
    Returns:
        dataset (Subset): view of the dataset with the selected utterances,
            in the order of the partition.
    """
    indices = [
        position
        for position, entry in enumerate(dataset._index)
        if entry["utt_id"] in utt_ids
    ]
    if not indices:
        raise ValueError("None of the requested utterance ids is in the partition")

    return Subset(dataset, indices)


def get_multicrop_dataloader(
    config,
    part,
    device,
    n_segments=DEFAULT_N_SEGMENTS,
    utt_ids=None,
    short_shifts=DEFAULT_SHORT_SHIFTS,
):
    """
    Create a dataloader that yields several segments per utterance.

    The waveforms are read in full instead of being cut to the model input
    length: every segment after the first one is exactly the part of the
    recording the ordinary pipeline discards.

    Args:
        config (DictConfig): config of a scoring run, as built by
            scripts.checkpoint_config.build_run_config.
        part (str): partition to score.
        device (str): device the batch transforms run on.
        n_segments (int): segments per utterance, at most.
        utt_ids (set[str] | None): utterances to score, None scores the whole
            partition.
        short_shifts (int): windows per utterance that is shorter than one
            segment, see src.datasets.multicrop.shifted_segments.
    Returns:
        dataloader (DataLoader): dataloader over the partition.
        batch_transforms (dict[Callable] | None): transforms applied to the
            whole batch.
    """
    batch_transforms = instantiate(config.transforms.batch_transforms)
    move_batch_transforms_to_device(batch_transforms, device)

    config.datasets[part].max_len = None
    dataset = instantiate(config.datasets[part])
    if utt_ids is not None:
        dataset = select_utterances(dataset, utt_ids)

    max_len = config.get("collate_max_len", DEFAULT_MAX_LEN)
    dataloader = instantiate(
        config.dataloader,
        dataset=dataset,
        collate_fn=get_multicrop_collate_fn(max_len, n_segments, short_shifts),
        drop_last=False,
        shuffle=False,
        worker_init_fn=set_worker_seed,
    )
    return dataloader, batch_transforms
