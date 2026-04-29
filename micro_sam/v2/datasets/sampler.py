import random
from typing import Dict, Iterator, List, Optional

from torch.utils.data import Sampler

from torch_em.data import ConcatDataset


class UniBatchSampler(Sampler[List[int]]):
    """Yields batches where all indices share the same group key.

    Batch sampler that groups indices by an arbitrary key (e.g. source
    dimensionality, or a ``(ndim, z_slices)`` tuple) and optionally uses
    a different batch size per group.

    Args:
        group_per_index: List of length ``len(dataset)`` mapping each global
            index to its group key (any hashable value).
        batch_size: Default number of samples per batch (used when no
            per-group override is provided).
        batch_size_per_group: Optional dict mapping group keys to their
            batch size.  Groups not present fall back to *batch_size*.
        shuffle: Whether to shuffle indices within each group and the
            batch order across groups each epoch.
        drop_last: Whether to drop the last incomplete batch per group.
    """

    def __init__(
        self,
        group_per_index: List,
        batch_size: int = 1,
        batch_size_per_group: Optional[Dict] = None,
        shuffle: bool = True,
        drop_last: bool = False,
    ):
        self.group_per_index = group_per_index
        self.batch_size = batch_size
        self.batch_size_per_group = batch_size_per_group or {}
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Partition indices into groups keyed by group key.
        self._groups: Dict = {}
        for idx, key in enumerate(group_per_index):
            self._groups.setdefault(key, []).append(idx)

    def _get_batch_size(self, group_key) -> int:
        return self.batch_size_per_group.get(group_key, self.batch_size)

    def __iter__(self) -> Iterator[List[int]]:
        all_batches: List[List[int]] = []
        for group_key, indices in self._groups.items():
            bs = self._get_batch_size(group_key)
            if self.shuffle:
                indices = indices.copy()
                random.shuffle(indices)

            for i in range(0, len(indices), bs):
                batch = indices[i: i + bs]
                if self.drop_last and len(batch) < bs:
                    continue
                all_batches.append(batch)

        # Interleave groups rather than emitting all of one group first.
        if self.shuffle:
            random.shuffle(all_batches)

        yield from all_batches

    def __len__(self) -> int:
        total = 0
        for group_key, indices in self._groups.items():
            n = len(indices)
            bs = self._get_batch_size(group_key)
            if self.drop_last:
                total += n // bs
            else:
                total += (n + bs - 1) // bs
        return total

    def set_epoch(self, epoch: int) -> None:
        """Seed the RNG so shuffling varies across epochs."""
        random.seed(epoch)


class DistributedUniBatchSampler(Sampler[List[int]]):
    """Distributed version of UniBatchSampler for DDP training.

    Shards batches **within each group** across ranks, so every rank receives
    the same proportion of 2D and 3D batches each epoch.  This avoids the load
    imbalance that arises when a globally-shuffled list is interleaved across
    ranks, because 2D and 3D batches have very different compute costs.

    All ranks use the same shuffle seed (set via :meth:`set_epoch`) so they
    independently produce identical per-group batch lists, then each takes its
    own non-overlapping slice.

    Call :meth:`set_epoch` before each epoch so the shuffle seed changes and
    all ranks stay in sync.

    Args:
        group_per_index: Same as :class:`UniBatchSampler`.
        batch_size: Default batch size.
        batch_size_per_group: Per-group batch size overrides.
        shuffle: Whether to shuffle batches each epoch.
        drop_last: Whether to drop the last incomplete batch per group.
        rank: This process's rank (0-based).
        world_size: Total number of DDP processes.
    """

    def __init__(
        self,
        group_per_index: List,
        batch_size: int = 1,
        batch_size_per_group: Optional[Dict] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ):
        self._base = UniBatchSampler(
            group_per_index, batch_size, batch_size_per_group, shuffle, drop_last
        )
        self.rank = rank
        self.world_size = world_size

    def set_epoch(self, epoch: int) -> None:
        """Seed the shuffle for *epoch* — must be called identically on all ranks."""
        self._base.set_epoch(epoch)

    def __iter__(self):
        per_rank_batches: List[List[int]] = []

        for group_key, indices in self._base._groups.items():
            bs = self._base._get_batch_size(group_key)
            group_indices = list(indices)

            if self._base.shuffle:
                random.shuffle(group_indices)

            group_batches: List[List[int]] = []
            for i in range(0, len(group_indices), bs):
                batch = group_indices[i:i + bs]
                if self._base.drop_last and len(batch) < bs:
                    continue
                group_batches.append(batch)

            # Shard this group's batches across ranks before merging.
            per_rank_batches.extend(group_batches[self.rank::self.world_size])

        if self._base.shuffle:
            random.shuffle(per_rank_batches)

        yield from per_rank_batches

    def __len__(self) -> int:
        total = 0
        for group_key, indices in self._base._groups.items():
            bs = self._base._get_batch_size(group_key)
            n = len(indices)
            n_batches = n // bs if self._base.drop_last else (n + bs - 1) // bs
            total += n_batches // self.world_size
        return total


def _build_group_map(concat_dataset: ConcatDataset) -> List:
    """Build a flat list mapping each global index to its group key.

    The group key is read from ``group_key`` first, falling back to
    ``source_ndim`` for backward compatibility.

    Args:
        concat_dataset: A ``ConcatDataset`` whose sub-datasets are
            ``UniDataWrapper`` instances.

    Returns:
        List of length ``len(concat_dataset)`` with the group key for each index.
    """
    group_per_index: List = []
    for ds in concat_dataset.datasets:
        key = getattr(ds, "group_key", None)
        if key is None:
            key = getattr(ds, "source_ndim", None)
        if key is None:
            raise ValueError(
                f"Dataset {ds} does not have a valid group_key or source_ndim attribute. "
                "Wrap it with UniDataWrapper(ds, source_ndim=2) or UniDataWrapper(ds, source_ndim=3)."
            )
        group_per_index.extend([key] * len(ds))
    return group_per_index
