"""Count the generalist training data (train + val) per domain and dimensionality.

The counter builds the datasets in the same way as ``generalist_loader.py``. Then it reads the label sources (file,
internal key and region of interest) from the torch-em leaf datasets. These leaves hold the outputs of the torch-em
``get_*_paths`` functions after every split rule of the loader. So the counter does not re-implement any split.

Each label source is one counting unit:

- An image file of an ``ImageCollectionDataset`` is one 2D image.
- A ``SegmentationDataset`` of a 3D source is one volume. A time point of a 3D time-lapse is one volume. A volume
  that train and val share through different regions of interest is one volume, and its objects are the union of
  the ids in these regions.
- A ``SegmentationDataset`` of a 2D source over a stacked file (movie frames, tile stacks) is one image per entry of
  the first axis.
- An image with several label targets, such as the Xenium nuclei and cells of one slide, is one image. The objects
  of both targets add up.

The objects of a unit are its unique non-zero label ids after the label pre-processing of the loader. This maps
background ids to zero, converts semantic masks to instances and drops oversized annotation artefacts. The ignore
label of the EM cell datasets is not an object. The counter does not run connected components. So an id with
several disconnected parts is one object, and the counts describe the labels as annotated.

The counter writes ``generalist_data_counts.csv`` to the current directory with the columns ``domain``, ``ndim``,
``n_datasets`` (top-level data folders), ``n_images_or_volumes``, ``n_objects`` and ``datasets``. With
``--per_dataset`` it also writes one row per dataset. A JSON cache in the current directory holds the counts per
label source, so a rerun after a loader change only counts new sources. Delete the cache after you regenerate
labels on disk.

Each domain builds its datasets first, which takes about ten minutes. The EM volumes dominate the counting time
(about half an hour with 6 workers). Use 6 to 8 workers on a 64 GB allocation. More workers ran out of memory.

Run ``python -m micro_sam.v2.datasets.data_counter --help`` for the options.
"""

import os
import csv
import json
import argparse
import warnings
from functools import partial
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

import torch

from torch_em.util.image import load_data
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets.electron_microscopy.platynereis import CELL_IGNORE_LABEL

from .wrapper import _leaves
from ..transforms.raw import _identity
from .generalist_loader import _get_em_datasets, _get_hp_datasets, _get_lm_datasets, _resize_then_em_label_trafo

DEFAULT_INPUT_PATH = "/mnt/vast-nhr/projects/cidas/cca/data"
DOMAIN_BUILDERS = {"lm": _get_lm_datasets, "em": _get_em_datasets, "hp": _get_hp_datasets}
DOMAIN_NAMES = {"lm": "light microscopy", "em": "electron microscopy", "hp": "histopathology"}
# The background and the ignore label of the EM cell datasets are never objects.
NON_OBJECT_IDS = (0, CELL_IGNORE_LABEL)
# The largest number of label elements that one slab of a streamed unit holds.
MAX_SLAB_ELEMENTS = 2 ** 26


@dataclass(frozen=True)
class Unit:
    """One label source of the training data."""
    domain: str
    dataset: str
    ndim: int
    label_path: str
    label_key: Optional[str]
    roi: Optional[Tuple[Tuple[Optional[int], Optional[int]], ...]]
    # A 2D source with the patch shape (1, H, W) is a stack. Its first axis holds the frames or tiles.
    stacked: bool
    splits: Tuple[str, ...]

    @property
    def key(self):
        return (self.label_path, self.label_key, self.roi, self.stacked)


@dataclass
class UnitCount:
    """The counting result of one unit. ``ids`` holds the object ids of a unit that shares its file with others."""
    n_images: int
    n_objects: int
    shape: Tuple[int, ...]
    ids: Optional[List[int]] = None


def _roi_to_tuple(roi):
    """Convert a region of interest of slices into a tuple of (start, stop) pairs, which is hashable."""
    if roi is None:
        return None
    return tuple((s.start, s.stop) for s in roi)


def _roi_to_slices(roi, shape):
    """Convert a region of interest of (start, stop) pairs into full slices for an array of ``shape``."""
    if roi is None:
        return tuple(slice(0, s) for s in shape)
    slices = tuple(
        slice(0 if start is None else start, dim if stop is None else stop) for (start, stop), dim in zip(roi, shape)
    )
    return slices + tuple(slice(0, s) for s in shape[len(slices):])


def _dataset_name(label_path, input_path):
    """Return the top-level data folder of a label file. Files outside the data root use their parent folder."""
    rel = os.path.relpath(os.path.abspath(label_path), os.path.abspath(input_path))
    if rel.startswith(os.pardir):
        return os.path.basename(os.path.dirname(label_path))
    return rel.split(os.sep)[0]


def _label_transform(leaf):
    """Return the label transform of a leaf without the in-plane resize of the small-volume datasets.

    The counter builds the datasets with ``label_transform2=None``. So the dataset-specific transforms of the loader
    wrap ``None`` and return the mapped instance ids instead of distance targets.
    """
    trafo = getattr(leaf, "label_transform2", None)
    if isinstance(trafo, partial) and trafo.func is _resize_then_em_label_trafo:
        trafo = trafo.keywords.get("em_trafo_fn")
    return trafo


def build_units(domain, input_path):
    """Build the datasets of one domain in the same way as the training and collect their label sources.

    Args:
        domain: The domain, one of ``"lm"``, ``"em"`` or ``"hp"``.
        input_path: The root path of the training data.

    Returns:
        The units, deduplicated over train and val, and the label transforms of each unit
        (``pre_label_transform``, ``label_transform2``) keyed by the unit key.
    """
    kwargs = {
        "raw_transform": _identity,
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.int64,
        "label_transform2": None,
    }
    train_ds, val_ds = DOMAIN_BUILDERS[domain](input_path, (512, 512), [8], kwargs, None)

    units: Dict[tuple, Unit] = {}
    transforms: Dict[tuple, Tuple[Optional[Callable], Optional[Callable]]] = {}
    for split, wrappers in [("train", train_ds), ("val", val_ds)]:
        for wrapper in wrappers:
            for leaf in _leaves(wrapper.ds):
                if hasattr(leaf, "label_images"):
                    sources = [(p, None, None) for p in leaf.label_images]
                    stacked = False
                else:
                    sources = [(leaf.label_path, leaf.label_key, _roi_to_tuple(leaf.roi))]
                    stacked = wrapper.source_ndim == 2 and len(leaf.patch_shape) == 3
                for label_path, label_key, roi in sources:
                    unit = Unit(
                        domain=domain, dataset=_dataset_name(label_path, input_path), ndim=wrapper.source_ndim,
                        label_path=label_path, label_key=label_key, roi=roi, stacked=stacked, splits=(split,),
                    )
                    if unit.key in units:
                        previous = units[unit.key]
                        if split not in previous.splits:
                            warnings.warn(f"{label_path} ({label_key}, {roi}) is used for training and validation.")
                            units[unit.key] = Unit(**{**asdict(previous), "splits": previous.splits + (split,)})
                        continue
                    units[unit.key] = unit
                    transforms[unit.key] = (getattr(leaf, "pre_label_transform", None), _label_transform(leaf))
    return list(units.values()), transforms


def _object_ids(labels, pre_label_transform, label_transform):
    """Return the object ids of a label array after the transforms of the loader."""
    labels = np.asarray(labels)
    if pre_label_transform is not None:
        labels = pre_label_transform(labels)
    if label_transform is not None:
        labels = label_transform(labels)
    ids = np.unique(labels)
    return ids[(ids > 0) & ~np.isin(ids, NON_OBJECT_IDS)]


def _iter_slabs(labels, slices):
    """Yield the label data of ``slices`` in slabs along the first axis. Each slab fits ``MAX_SLAB_ELEMENTS``."""
    per_slice = int(np.prod([s.stop - s.start for s in slices[1:]], dtype=np.int64)) if len(slices) > 1 else 1
    step = max(1, MAX_SLAB_ELEMENTS // max(per_slice, 1))
    for start in range(slices[0].start, slices[0].stop, step):
        stop = min(start + step, slices[0].stop)
        yield np.asarray(labels[(slice(start, stop),) + slices[1:]])


def count_unit(unit: Unit, pre_label_transform, label_transform, keep_ids=False) -> UnitCount:
    """Count the images or volumes and the objects of one unit.

    Args:
        unit: The label source.
        pre_label_transform: The ``pre_label_transform`` of the leaf dataset, or None.
        label_transform: The id-preserving label transform of the leaf dataset, or None.
        keep_ids: Whether to return the object ids of a volume or image next to their number.

    Returns:
        The count of the unit.
    """
    labels = load_data(unit.label_path, unit.label_key)
    slices = _roi_to_slices(unit.roi, labels.shape)
    shape = tuple(s.stop - s.start for s in slices)
    needs_transform = pre_label_transform is not None or label_transform is not None

    if unit.ndim == 3 or not unit.stacked:
        # A unit without a transform streams in slabs, so large files fit into memory.
        if needs_transform:
            ids = _object_ids(labels[slices], pre_label_transform, label_transform)
        else:
            ids = np.unique(np.concatenate([_object_ids(slab, None, None) for slab in _iter_slabs(labels, slices)]))
        return UnitCount(n_images=1, n_objects=len(ids), shape=shape, ids=ids.tolist() if keep_ids else None)

    if len(shape) != 3:
        raise ValueError(
            f"The stacked source {unit.label_path}, {unit.label_key} has the shape {shape}. "
            "Expected the shape (frames, height, width)."
        )

    # Every entry of the first axis of a stack is one image.
    n_objects = 0
    for slab in _iter_slabs(labels, slices):
        n_objects += sum(len(_object_ids(frame, pre_label_transform, label_transform)) for frame in slab)
    return UnitCount(n_images=shape[0], n_objects=n_objects, shape=shape)


def _file_key(unit):
    """Return the key that the units of one label file and key share, such as the train and val slabs of a volume."""
    return (unit.label_path, unit.label_key, unit.stacked)


def _count_units(units, transforms, n_workers, cache):
    """Count all units in parallel and reuse the cached results."""
    units_per_file = defaultdict(int)
    for unit in units:
        units_per_file[_file_key(unit)] += 1

    results = {}
    pending = []
    for unit in units:
        keep_ids = not unit.stacked and units_per_file[_file_key(unit)] > 1
        cache_key = json.dumps([unit.label_path, unit.label_key, unit.roi, unit.ndim, unit.stacked])
        if cache_key in cache and (cache[cache_key].get("ids") is not None or not keep_ids):
            results[unit.key] = UnitCount(**cache[cache_key])
        else:
            pending.append((unit, cache_key, keep_ids))

    if pending:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(count_unit, unit, *transforms[unit.key], keep_ids=keep_ids): (unit, cache_key)
                for unit, cache_key, keep_ids in pending
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Count the label sources"):
                unit, cache_key = futures[future]
                count = future.result()
                results[unit.key] = count
                cache[cache_key] = asdict(count)
    return results


def _load_cache(path):
    """Load the JSON cache of the counts per label source. A missing cache is empty."""
    if path is None or not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _save_cache(path, cache):
    """Write the cache to a temporary file, then move it into place, so an interrupted write leaves no partial cache."""
    if path is None:
        return
    tmp_path = f"{path}.tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(cache, f)
    os.replace(tmp_path, path)


def summarize(units: List[Unit], counts: Dict[tuple, UnitCount]):
    """Sum the unit counts per (domain, ndim) and per (domain, ndim, dataset).

    Args:
        units: The label sources.
        counts: The count of each unit, keyed by the unit key.

    Returns:
        The summary rows and the per-dataset rows. Each is a list of dicts.
    """
    per_dataset = defaultdict(
        lambda: {"n_units": 0, "n_train_units": 0, "n_val_units": 0, "n_images": 0, "n_objects": 0}
    )
    # The units of one file and key are the regions of interest of one volume: their ids are joined.
    ids_per_file = defaultdict(set)
    for unit in units:
        count = counts[unit.key]
        if count.ids is not None:
            ids_per_file[_file_key(unit)].update(count.ids)

    # An image with several label targets, such as the Xenium nuclei and cells, is one image. Its objects add up.
    seen_images, seen_files = set(), set()
    for unit in units:
        row = per_dataset[(unit.domain, unit.ndim, unit.dataset)]
        count = counts[unit.key]
        row["n_units"] += 1
        row["n_train_units"] += "train" in unit.splits
        row["n_val_units"] += "val" in unit.splits
        file_key = _file_key(unit)
        if file_key in ids_per_file:
            if file_key not in seen_files:
                seen_files.add(file_key)
                row["n_objects"] += len(ids_per_file[file_key])
        else:
            row["n_objects"] += count.n_objects
        image_key = (unit.label_path, unit.roi if unit.stacked else None)
        if image_key not in seen_images:
            seen_images.add(image_key)
            row["n_images"] += count.n_images

    dataset_rows = []
    summary = defaultdict(lambda: {"datasets": [], "n_images": 0, "n_objects": 0})
    for (domain, ndim, dataset), row in sorted(per_dataset.items()):
        dataset_rows.append({"domain": domain, "ndim": ndim, "dataset": dataset, **row})
        agg = summary[(domain, ndim)]
        agg["datasets"].append(dataset)
        agg["n_images"] += row["n_images"]
        agg["n_objects"] += row["n_objects"]

    summary_rows = []
    for (domain, ndim), agg in sorted(summary.items()):
        summary_rows.append({
            "domain": domain, "ndim": ndim, "n_datasets": len(agg["datasets"]),
            "n_images_or_volumes": agg["n_images"], "n_objects": agg["n_objects"],
            "datasets": ";".join(agg["datasets"]),
        })
    return summary_rows, dataset_rows


def _write_csv(path, rows):
    """Write the rows to a CSV file with the keys of the first row as header."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(summary_rows):
    """Print one sentence per (domain, ndim) row."""
    unit_name = {2: "images", 3: "volumes"}
    for row in summary_rows:
        print(
            f"{row['n_objects']:,} objects in {row['n_images_or_volumes']:,} {DOMAIN_NAMES[row['domain']]} "
            f"({row['ndim']}d) {unit_name[row['ndim']]} from {row['n_datasets']} datasets"
        )


def run(input_path, domains, n_workers, output_prefix, per_dataset, cache_path):
    """Build, count and summarize the training data of the requested domains.

    Args:
        input_path: The root path of the training data.
        domains: The domains to count, a subset of ``"lm"``, ``"em"`` and ``"hp"``.
        n_workers: The number of processes that count label sources in parallel.
        output_prefix: The prefix of the CSV files in the current directory.
        per_dataset: Whether to also write the breakdown per dataset.
        cache_path: The path of the JSON cache of the counts per label source, or None to disable the cache.
    """
    units, transforms = [], {}
    for domain in domains:
        domain_units, domain_transforms = build_units(domain, input_path)
        print(f"{domain}: {len(domain_units)} label sources from {len({u.dataset for u in domain_units})} datasets")
        units.extend(domain_units)
        transforms.update(domain_transforms)

    cache = _load_cache(cache_path)
    try:
        counts = _count_units(units, transforms, n_workers, cache)
    finally:
        _save_cache(cache_path, cache)

    summary_rows, dataset_rows = summarize(units, counts)
    _write_csv(f"{output_prefix}.csv", summary_rows)
    print(f"Wrote {output_prefix}.csv")
    if per_dataset:
        _write_csv(f"{output_prefix}_per_dataset.csv", dataset_rows)
        print(f"Wrote {output_prefix}_per_dataset.csv")
    _print_summary(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Count the generalist training data (train + val).")
    parser.add_argument("--input_path", default=DEFAULT_INPUT_PATH, help="The root path of the training data.")
    parser.add_argument("--domains", nargs="+", default=list(DOMAIN_BUILDERS), choices=list(DOMAIN_BUILDERS))
    parser.add_argument("--n_workers", type=int, default=len(os.sched_getaffinity(0)), help="The number of processes.")
    parser.add_argument("--output_prefix", default="generalist_data_counts", help="The prefix of the CSV files.")
    parser.add_argument("--per_dataset", action="store_true", help="Also write the breakdown per dataset.")
    parser.add_argument(
        "--cache_path", default="generalist_data_counts_cache.json",
        help="The JSON cache of the counts per label source. Pass '' to disable it."
    )
    args = parser.parse_args()
    run(
        args.input_path, args.domains, args.n_workers, args.output_prefix, args.per_dataset,
        args.cache_path or None,
    )


if __name__ == "__main__":
    main()
