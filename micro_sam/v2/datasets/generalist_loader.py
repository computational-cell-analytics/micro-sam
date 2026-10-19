import os
import re
import json
import random
import shutil
from glob import glob
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split

import torch

from elf.io import open_file

import torch_em
from torch_em.transform import get_augmentations
from torch_em.transform.generic import Compose
from torch_em.data import datasets, MinInstanceSampler, ConcatDataset

from . import cochleanet_data
from .wrapper import UniDataWrapper
from .sampler import UniBatchSampler, RejectBlankSlices, _build_group_map
from ..transforms.raw import (
    _identity, _cellpose_raw_trafo, _to_8bit, _normalize_percentile, _resize_raw_to_512, _resize_to_512,
    _prepare_identity, _drop_alpha_channel,
    _enseg_green_channel, _xenium_cell_channels, _pan_multiplex_tissuenet_order, _cvz_cell_channels,
    _minmax_raw_trafo,
    get_random_percentile_normalization,
)
from ..transforms.labels import (
    _em_cell_label_trafo, _joint_em_cell_label_trafo, _background_id_label_trafo,
    _plantseg_label_trafo, _astih_pre_label_transform, _instance_labels,
    _ignore_missing_raw_trafo, _ignore_unlabelled_blobs_trafo, _labels_to_uint32, _drop_oversized_label_trafo,
    _compact_label_ids, _ignore_label_trafo, _connected_components_pre_label_trafo,
    _JointLabelTransform, _JointGeodesicLabelTransform,
)

# The training domains. 'all' is LM + EM + HP; the mitochondria EM generalist ('em_mito') and the nucleus EM
# generalist ('em_nucleus') are separate domains and are not part of 'all'.
DATASET_CHOICES = ("lm", "em", "hp", "em_mito", "em_nucleus", "all")

# Cap on validation samples drawn per dataset, to keep the per-epoch validation pass cheap.
# Each access is a random crop (see UniDataWrapper.max_samples), so this is N random samples.
N_SAMPLES_VAL = 50

# Crops the sampler may reject before a file gives up. Sparse image sets (the worm and CTC HSC frames, the
# Tsakiroglou crops) reject up to 85 crops per accepted one. A file that gives up is replaced by another draw in
# UniDataWrapper, so the budget only bounds the time lost on a file whose labels never satisfy the sampler.
MAX_SAMPLING_ATTEMPTS = 1000

# Fixed seed for deterministic validation. The same value is used to seed the main process
# (prompt sampling in SAM2Train, object subsampling in ConvertToSam2VideoBatch) in
# Sam2Trainer._validate_impl, so the validation metric is comparable across epochs.
VALIDATION_SEED = 42

# Train with uniformly sampled symmetric percentiles. Validate deterministically with the 2nd and 98th percentiles
# to match the inference-time normalization in normalize_raw.
TRAIN_LOWER_PERCENTILE_BOUNDS = (0.0, 5.0)
VALIDATION_LOWER_PERCENTILE_BOUNDS = (2.0, 2.0)


def seed_worker(worker_id):
    """DataLoader worker_init_fn that pins per-worker RNG for deterministic validation crops.

    The torch_em datasets draw a fresh random crop and random object subset on every
    __getitem__ inside the worker process. Seeding each worker deterministically (and using
    non-persistent workers so this runs every epoch) makes those crops identical across epochs.
    """
    seed = VALIDATION_SEED + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_native_byte_order(y):
    # tifffile.memmap returns big-endian >f4 for some TIFFs. Byteswap to native so that
    # Kornia augmentation and skimage or vigra C extensions receive correctly ordered bytes.
    return y.byteswap().view(y.dtype.newbyteorder()) if not y.dtype.isnative else y


def _compact_native_label_ids(y, keep=()):
    return _compact_label_ids(_ensure_native_byte_order(y), keep=keep)


def _set_percentile_normalization(dataset, lower_percentile_bounds):
    """Replace fixed normalization in all torch-em leaves of a dataset tree."""
    if isinstance(dataset, (list, tuple)):
        for ds in dataset:
            _set_percentile_normalization(ds, lower_percentile_bounds)
        return

    if isinstance(dataset, UniDataWrapper):
        _set_percentile_normalization(dataset.ds, lower_percentile_bounds)
        return

    if isinstance(dataset, torch.utils.data.Subset):
        _set_percentile_normalization(dataset.dataset, lower_percentile_bounds)
        return

    children = getattr(dataset, "datasets", None)
    if children is not None:
        for ds in children:
            _set_percentile_normalization(ds, lower_percentile_bounds)
        return

    if not hasattr(dataset, "raw_transform"):
        raise TypeError(f"Cannot configure raw normalization for dataset of type {type(dataset).__name__}.")

    if dataset.raw_transform is _prepare_identity:
        # The missing-tile transform tests the stored intensities and normalizes the raw itself afterwards.
        normalizer = get_random_percentile_normalization(_identity, lower_percentile_bounds).normalizer
        transform = dataset.transform
        dataset.transform = partial(transform.func, *transform.args, **{**transform.keywords, "normalizer": normalizer})
        return

    dataset.raw_transform = get_random_percentile_normalization(
        dataset.raw_transform, lower_percentile_bounds=lower_percentile_bounds
    )


def _set_max_sampling_attempts(dataset, n_attempts):
    """Raise the sampler attempt limit in all torch-em leaves of a dataset tree."""
    if isinstance(dataset, (list, tuple)):
        for ds in dataset:
            _set_max_sampling_attempts(ds, n_attempts)
    elif isinstance(dataset, UniDataWrapper):
        _set_max_sampling_attempts(dataset.ds, n_attempts)
    elif isinstance(dataset, torch.utils.data.Subset):
        _set_max_sampling_attempts(dataset.dataset, n_attempts)
    elif getattr(dataset, "datasets", None) is not None:
        for ds in dataset.datasets:
            _set_max_sampling_attempts(ds, n_attempts)
    elif hasattr(dataset, "max_sampling_attempts"):
        dataset.max_sampling_attempts = n_attempts


def _reject_blank_slices(dataset):
    """Wrap the sampler of every 3d torch-em leaf so a crop with a constant raw plane is rejected."""
    if isinstance(dataset, (list, tuple)):
        for ds in dataset:
            _reject_blank_slices(ds)
    elif isinstance(dataset, UniDataWrapper):
        _reject_blank_slices(dataset.ds)
    elif isinstance(dataset, torch.utils.data.Subset):
        _reject_blank_slices(dataset.dataset)
    elif getattr(dataset, "datasets", None) is not None:
        for ds in dataset.datasets:
            _reject_blank_slices(ds)
    elif getattr(dataset, "ndim", 2) == 3 and getattr(dataset, "sampler", None) is not None:
        dataset.sampler = RejectBlankSlices(dataset.sampler)


def _configure_training_normalization(train_datasets, val_datasets):
    """Enable random percentile augmentation for training and deterministic 2nd/98th validation."""
    _set_percentile_normalization(
        train_datasets, lower_percentile_bounds=TRAIN_LOWER_PERCENTILE_BOUNDS,
    )
    _set_percentile_normalization(
        val_datasets, lower_percentile_bounds=VALIDATION_LOWER_PERCENTILE_BOUNDS,
    )
    _set_max_sampling_attempts([train_datasets, val_datasets], MAX_SAMPLING_ATTEMPTS)
    _reject_blank_slices([train_datasets, val_datasets])


def _prepare_data_loader(dataset, batch_size, shuffle, batch_size_per_group=None, num_workers=32, deterministic=False):
    # For deterministic validation, re-seed workers every epoch via worker_init_fn.
    # This requires non-persistent workers, since persistent workers run worker_init_fn only once.
    # Persistent workers also require num_workers > 0.
    persistent = (num_workers > 0) and not deterministic
    worker_init = seed_worker if deterministic else None
    if isinstance(dataset, ConcatDataset) and (batch_size > 1 or batch_size_per_group):
        batch_sampler = UniBatchSampler(
            group_per_index=_build_group_map(dataset),
            batch_size=batch_size,
            batch_size_per_group=batch_size_per_group,
            shuffle=shuffle,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=True, persistent_workers=persistent, worker_init_fn=worker_init,
        )
        # Monkey-patch shuffle attribute for torch_em DefaultTrainer compatibility.
        loader.shuffle = shuffle
    else:
        loader = torch_em.get_data_loader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
            persistent_workers=persistent, worker_init_fn=worker_init,
        )

    return loader


def _resize_then_em_label_trafo(y, em_trafo_fn):
    """Resize small label patch to 512×512 then apply the EM label transform."""
    y = _resize_to_512(y, is_label=True)
    return em_trafo_fn(y)


def _compute_platy_rois(root, sample_ids, ignore_label, file_template, label_key, cache_prefix="_roi_cache"):
    """The bounding box of the valid (not ignore_label) voxels of each Platynereis volume, the only region that trains.

    The boxes are cached in a json file in *root*, named by *cache_prefix* and the sample ids.
    """
    cache_path = os.path.join(root, f"{cache_prefix}_{'_'.join(map(str, sample_ids))}.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        return {int(k): tuple(slice(s[0], s[1]) for s in v) for k, v in data.items()}

    rois = {}
    for sample_id in sample_ids:
        path = os.path.join(root, (file_template % sample_id))
        with open_file(path, "r") as f:
            labels = f[label_key][:]
        valid_coordinates = np.where(labels != ignore_label)
        roi = tuple(slice(int(coord.min()), int(coord.max()) + 1) for coord in valid_coordinates)
        rois[sample_id] = roi

    tmp_path = cache_path + f".tmp{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump({str(k): [(s.start, s.stop) for s in v] for k, v in rois.items()}, f)
    os.replace(tmp_path, cache_path)

    return rois


def _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo):
    """Get all light microscopy (LM) datasets for generalist training.

    Returns:
        Tuple of (train_ds, val_ds) lists of UniDataWrapper instances.
    """
    train_ds, val_ds = [], []
    n_z = len(z_slices)

    # 1. CellPose (cell segmentation in (2d) fluoroscence microscopy imaging modalities)
    # NOTE: Training uses both 'cyto' (540) and 'cyto2' (256 additional, disjoint) images. 'cyto2' has no test
    # split, so the 68-image 'cyto' test split validates; CellPose has no blind in-domain test.
    cellpose_kwargs = {
        "path": os.path.join(input_path, "cellpose"),
        "patch_shape": patch_shape,
        "raw_transform": _cellpose_raw_trafo,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }
    train_ds.append(
        UniDataWrapper(
            datasets.get_cellpose_dataset(split="train", choice=None, n_samples=600, **cellpose_kwargs), source_ndim=2
        )
    )
    val_ds.append(
        UniDataWrapper(datasets.get_cellpose_dataset(split="test", choice="cyto", **cellpose_kwargs), source_ndim=2)
    )

    # 2. CVZ Fluo (cell and nucleus segmentation in (2d) fluorescence CODEX images)
    # NOTE: Cell and DAPI crops of one field of view are paired, so both stains share one split by patient or
    # slide (see CVZ_VAL_GROUPS and CVZ_TEST_GROUPS); the test groups are blind.
    def _get_cvz_dataset(stain_choice, split_choice):
        raw_paths, label_paths = datasets.cvz_fluo.get_cvz_fluo_paths(
            path=os.path.join(input_path, "cvz"), stain_choice=stain_choice,
        )
        groups = [cvz_group(p) for p in raw_paths]
        if split_choice == "train":
            keep = [g not in CVZ_VAL_GROUPS and g not in CVZ_TEST_GROUPS for g in groups]
        else:
            keep = [g in CVZ_VAL_GROUPS for g in groups]
        ds = torch_em.default_segmentation_dataset(
            raw_paths=[p for p, k in zip(raw_paths, keep) if k],
            raw_key=None,
            label_paths=[p for p, k in zip(label_paths, keep) if k],
            label_key=None,
            is_seg_dataset=False,
            patch_shape=patch_shape,
            raw_transform=_cvz_cell_channels if stain_choice == "cell" else _to_8bit,
            n_samples=200 if split_choice == "train" else 100,
            **{k: v for k, v in kwargs.items() if k != "raw_transform"}
        )
        return ds

    train_ds.append(UniDataWrapper(_get_cvz_dataset("cell", "train"), source_ndim=2))
    train_ds.append(UniDataWrapper(_get_cvz_dataset("dapi", "train"), source_ndim=2))
    val_ds.append(UniDataWrapper(_get_cvz_dataset("cell", "val"), source_ndim=2))
    val_ds.append(UniDataWrapper(_get_cvz_dataset("dapi", "val"), source_ndim=2))

    # 3. DSB dataset (nucleus segmentation in fluorescence images)
    # NOTE: The 554 fluorescence images of the 'full' source (all Kaggle stage-1 training images) minus the 50 images
    # of the StarDist test split, which is the blind test. A random 10 % of the rest (seed 42) validates. The 107
    # histopathology images train with the histopathology datasets.
    dsb_raw, dsb_labels = dsb_fluorescence_training_paths(os.path.join(input_path, "dsb"))
    dsb_train_r, dsb_val_r, dsb_train_l, dsb_val_l = train_test_split(
        dsb_raw, dsb_labels, test_size=0.1, random_state=42
    )
    dsb_kwargs = {"patch_shape": patch_shape, "is_seg_dataset": False, "raw_key": None, "label_key": None, **kwargs}
    for raws, labs, n_samples, ds_list in [
        (dsb_train_r, dsb_train_l, 600, train_ds), (dsb_val_r, dsb_val_l, 50, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **dsb_kwargs
                ), source_ndim=2,
            )
        )

    # 4. EmbedSeg (cell and nucleus segmentation in fluorescence microscopy images)
    # Anisotropy factors (z/xy) from file metadata or EmbedSeg paper (Table 3, arXiv:2101.10033).
    # Mouse-Organoid: z=1.0µm, xy=0.1733µm -> ~5.8x -> (6, 1, 1)
    # Mouse-Skull: z≈0.5µm, xy≈0.1µm -> ~5x -> (5, 1, 1)
    # Platynereis-ISH: confirmed isotropic from TIFF metadata (z≈xy≈0.45µm)
    # Platynereis-Nuclei: confirmed from TIFF metadata (z=2.031µm, xy=0.406µm -> ~5x)
    embedseg_sampling = {
        "Mouse-Organoid-Cells-CBG": (6, 1, 1),
        "Mouse-Skull-Nuclei-CBG": (5, 1, 1),
        "Platynereis-ISH-Nuclei-CBG": None,
        "Platynereis-Nuclei-CBG": (5, 1, 1),
    }

    def _embedseg_kwargs(name, z):
        return {
            "patch_shape": (z, *patch_shape),
            "raw_transform": _to_8bit,
            "n_samples": max(1, 200 // n_z),
            "label_transform2": (
                label_trafo(sampling=embedseg_sampling[name])
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2"]},
        }

    # Mouse-Skull and Platynereis-ISH train on the official train volumes; a slab of the official test volume
    # validates and the whole test volume is the blind test. Platynereis-Nuclei and the organoid cells are
    # time-lapses and are split along time.
    embedseg_root = os.path.join(input_path, "embedseg")
    platy_nuclei_raw, platy_nuclei_labels = datasets.embedseg_data.get_embedseg_paths(
        embedseg_root, name="Platynereis-Nuclei-CBG", split="train",
    )
    organoid_raw, organoid_labels = datasets.embedseg_data.get_embedseg_paths(
        embedseg_root, name="Mouse-Organoid-Cells-CBG", split="train",
    )

    def _embedseg_paths_dataset(name, raw, labels, z):
        return torch_em.default_segmentation_dataset(
            raw_paths=raw, raw_key=None, label_paths=labels, label_key=None, is_seg_dataset=True,
            **_embedseg_kwargs(name, z),
        )

    for z in z_slices:
        for name in ["Mouse-Skull-Nuclei-CBG", "Platynereis-ISH-Nuclei-CBG"]:
            train_ds.append(
                UniDataWrapper(
                    datasets.get_embedseg_dataset(embedseg_root, name=name, split="train", **_embedseg_kwargs(name, z)),
                    source_ndim=3, group_key=(3, z),
                )
            )
            val_ds.append(
                UniDataWrapper(
                    datasets.get_embedseg_dataset(
                        embedseg_root, name=name, split="test", rois=[(EMBEDSEG_VAL_Z[name],)],
                        **_embedseg_kwargs(name, z)
                    ), source_ndim=3, group_key=(3, z),
                )
            )
        for name, raw, labels, splits in [
            ("Platynereis-Nuclei-CBG", platy_nuclei_raw, platy_nuclei_labels,
             (EMBEDSEG_PLATY_NUCLEI_TRAIN_TIMEPOINTS, EMBEDSEG_PLATY_NUCLEI_VAL_TIMEPOINTS)),
            ("Mouse-Organoid-Cells-CBG", organoid_raw, organoid_labels,
             (EMBEDSEG_ORGANOID_TRAIN_TIMEPOINTS, EMBEDSEG_ORGANOID_VAL_TIMEPOINTS)),
        ]:
            for timepoints, ds_list in zip(splits, (train_ds, val_ds)):
                ds_list.append(
                    UniDataWrapper(
                        _embedseg_paths_dataset(name, raw[timepoints], labels[timepoints], z),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 5. NIS3D (nucleus segmentation in light-sheet microscopy images)
    # NOTE: Only the Drosophila volumes are used, the others carry giant unannotated-region instances. Drosophila_2
    # (official train) trains, the last slices of Drosophila_1 (official test) validate and the whole volume is blind.
    nis3d_kwargs = {"path": os.path.join(input_path, "nis3d"), "split_type": "cross-image"}

    train_raw_paths, train_label_paths = datasets.nis3d.get_nis3d_paths(split="train", **nis3d_kwargs)
    val_raw_paths, val_label_paths = datasets.nis3d.get_nis3d_paths(split="test", **nis3d_kwargs)

    def _update_paths(paths):
        return [p for p in paths if "Drosophila" in p]

    train_raw_paths, train_label_paths = _update_paths(train_raw_paths), _update_paths(train_label_paths)
    val_raw_paths, val_label_paths = _update_paths(val_raw_paths), _update_paths(val_label_paths)

    for z in z_slices:
        nis3d_kwargs = {
            "patch_shape": (z, *patch_shape),
            "raw_transform": _to_8bit,
            "n_samples": max(1, 200 // n_z),
            # NIS3D Drosophila: isotropic 1µm x 1µm x 1µm
            "label_transform2": (
                label_trafo(sampling=None)
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2"]},
        }

        train_ds.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=train_raw_paths, raw_key=None, label_paths=train_label_paths, label_key=None,
                    **nis3d_kwargs,
                ), source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=val_raw_paths, raw_key=None, label_paths=val_label_paths, label_key=None,
                    rois=[(NIS3D_VAL_Z,)] * len(val_raw_paths), **nis3d_kwargs,
                ), source_ndim=3, group_key=(3, z),
            )
        )

    # 6. PlantSeg (cell segmentation in confocal microscopy images)
    # NOTE: Root trains on the official split, the test split stays blind. Label 1 is the background and label 0
    # the unannotated deeper tissue, which is mapped to the ignore value. Ovules is held out for OOD evaluation
    # (its 'label_with_ignore' key marks unannotated regions as -1); nuclei is 98-99 % background and redundant
    # with gonuclear.
    for z in z_slices:
        plantseg_kwargs = {
            "path": os.path.join(input_path, "plantseg"),
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 200 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["sampler", "label_transform2"]}
        }

        for ds_name, exclude_ids in [("root", [0, 1])]:
            _plantseg_trafo = partial(
                _plantseg_label_trafo, data=ds_name,
                label_trafo=label_trafo() if label_trafo is not None else kwargs.get("label_transform2"),
            )
            for split, ds_list in [("train", train_ds), ("val", val_ds)]:
                ds_list.append(
                    UniDataWrapper(
                        datasets.get_plantseg_dataset(
                            name=ds_name, split=split, with_ignore=ds_name == "ovules",
                            label_transform2=_plantseg_trafo,
                            sampler=MinInstanceSampler(min_num_instances=3, exclude_ids=exclude_ids),
                            **plantseg_kwargs
                        ), source_ndim=3, group_key=(3, z),
                    )
                )

    # 7. TissueNet (cell segmentation in tissue images)
    tissuenet_kwargs = {
        "path": os.path.join(input_path, "tissuenet"),
        "raw_channel": "rgb",
        "label_channel": "cell",
        "patch_shape": patch_shape,
        "raw_transform": partial(_normalize_percentile, axis=(1, 2)),  # TissueNet 'rgb' is (3, H, W)
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    train_ds.append(
        UniDataWrapper(datasets.get_tissuenet_dataset(split="train", n_samples=1000, **tissuenet_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_tissuenet_dataset(split="val", n_samples=100, **tissuenet_kwargs), source_ndim=2)
    )

    # 8. LIVECell (cell segmentation in phase contrast microscopy images)
    livecell_kwargs = {
        "path": os.path.join(input_path, "livecell"),
        "patch_shape": patch_shape,
        "sampler": MinInstanceSampler(min_num_instances=6, exclude_ids=[0]),
        **{k: v for k, v in kwargs.items() if k != "sampler"}
    }
    train_ds.extend(
        [
            UniDataWrapper(
                datasets.get_livecell_dataset(split="train", cell_types=[ctype], n_samples=400, **livecell_kwargs),
                source_ndim=2,
            ) for ctype in datasets.livecell.CELL_TYPES
        ]
    )
    val_ds.extend(
        [
            UniDataWrapper(
                datasets.get_livecell_dataset(split="val", cell_types=[ctype], n_samples=100, **livecell_kwargs),
                source_ndim=2
            ) for ctype in datasets.livecell.CELL_TYPES
        ]
    )

    # 9. DeepBacs (bacteria segmentation in label-free microscopy images)
    # NOTE: The 'mixed' archive pools S. aureus, E. coli and B. subtilis; a random 10 % of its official train images
    # (seed 42) validate and its official test split is blind. 'e_coli_stationary' is a separate acquisition of
    # stationary-phase cells with a train split only, which trains as well.
    deepbacs_kwargs = {
        "patch_shape": patch_shape,
        "raw_transform": _to_8bit,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }
    mixed_images, mixed_labels = datasets.deepbacs.get_deepbacs_paths(
        os.path.join(input_path, "deepbacs"), bac_type="mixed", split="train",
    )
    mixed_raw = sorted(glob(os.path.join(mixed_images, "*.tif")))
    mixed_lab = sorted(glob(os.path.join(mixed_labels, "*.tif")))
    assert len(mixed_raw) == len(mixed_lab) and mixed_raw
    mixed_train_r, mixed_val_r, mixed_train_l, mixed_val_l = train_test_split(
        mixed_raw, mixed_lab, test_size=0.1, random_state=42
    )
    for raws, labs, n_samples, ds_list in [
        (mixed_train_r, mixed_train_l, 400, train_ds), (mixed_val_r, mixed_val_l, 100, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, raw_key=None, label_paths=labs, label_key=None, is_seg_dataset=False, ndim=2,
                    n_samples=n_samples, **deepbacs_kwargs,
                ), source_ndim=2,
            )
        )
    train_ds.append(
        UniDataWrapper(
            datasets.get_deepbacs_dataset(
                path=os.path.join(input_path, "deepbacs"), bac_type="e_coli_stationary", split="train", n_samples=200,
                **deepbacs_kwargs,
            ), source_ndim=2,
        )
    )

    # 10. OrgaSegment (organoid segmentation in bright field images)
    orgasegment_kwargs = {
        "path": os.path.join(input_path, "orgasegment"), "patch_shape": patch_shape, **kwargs
    }

    train_ds.append(
        UniDataWrapper(
            datasets.get_orgasegment_dataset(split="train", n_samples=300, **orgasegment_kwargs), source_ndim=2,
        )
    )
    val_ds.append(
        UniDataWrapper(
            datasets.get_orgasegment_dataset(split="val", n_samples=150, **orgasegment_kwargs), source_ndim=2,
        )
    )

    # 11. OrganoidNet (pancreatic organoid segmentation)
    organoidnet_kwargs = {
        "path": os.path.join(input_path, "organoidnet"), "patch_shape": patch_shape, **kwargs
    }

    train_ds.append(
        UniDataWrapper(
            datasets.get_organoidnet_dataset(split="Training", n_samples=700, **organoidnet_kwargs), source_ndim=2,
        )
    )
    val_ds.append(
        UniDataWrapper(
            datasets.get_organoidnet_dataset(split="Validation", n_samples=200, **organoidnet_kwargs), source_ndim=2,
        )
    )

    # 12. Omnipose (bacteria and worm segmentation in mixed modality microscopy images)
    # NOTE: All four subsets are used. The official train split gives up a random 10 % of images (seed 42) for
    # validation, the official test split is blind. The worm images hold few objects, so the default sampler's three
    # instances apply there as well.
    omnipose_root = os.path.join(input_path, "omnipose")
    omnipose_kwargs = {
        "patch_shape": patch_shape,
        "raw_transform": _to_8bit,
        "is_seg_dataset": False,
        "raw_key": None,
        "label_key": None,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }
    for data_choice, n_train in OMNIPOSE_TRAIN_SAMPLES.items():
        raw, labels = datasets.omnipose.get_omnipose_paths(omnipose_root, split="train", data_choice=data_choice)
        train_raw, val_raw, train_labels, val_labels = train_test_split(raw, labels, test_size=0.1, random_state=42)
        for raw_paths, label_paths, n_samples, ds_list in [
            (train_raw, train_labels, n_train, train_ds), (val_raw, val_labels, n_train // 5, val_ds),
        ]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=raw_paths, label_paths=label_paths, n_samples=n_samples, **omnipose_kwargs
                    ), source_ndim=2,
                )
            )

    # 13. CTC (cell segmentation from Cell Tracking Challenge)
    # NOTE: The same eight datasets as the v1 generalist (GOWT1 and N2DL-HeLa are held out), both movies of each.
    # Only the training data has labels, so a random 10 % of the labelled frames (seed 42) validate; there is no
    # blind test.
    ctc_kwargs = {
        "patch_shape": patch_shape,
        "raw_transform": _to_8bit,
        "is_seg_dataset": False,
        "raw_key": None,
        "label_key": None,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }
    for name in CTC_DATASETS:
        image_dirs, label_dirs = datasets.ctc.get_ctc_segmentation_paths(
            os.path.join(input_path, "ctc"), dataset_name=name, split="train",
        )
        raw = sorted(p for d in image_dirs for p in glob(os.path.join(d, "*.tif")))
        labels = sorted(p for d in label_dirs for p in glob(os.path.join(d, "*.tif")))
        assert len(raw) == len(labels) and raw
        train_raw, val_raw, train_labels, val_labels = train_test_split(raw, labels, test_size=0.1, random_state=42)
        for raw_paths, label_paths, ds_list in [(train_raw, train_labels, train_ds), (val_raw, val_labels, val_ds)]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(raw_paths=raw_paths, label_paths=label_paths, **ctc_kwargs),
                    source_ndim=2,
                )
            )

    # 14. YeaZ (yeast cell segmentation in brightfield microscopy images)
    # NOTE: Only the brightfield subset is used. 14 of the 28 phase contrast train files are 2D+t stacks and the
    # torch-em loader cannot handle them mixed with 2D crops. Images are ~400x450 uint16 and are zero-padded
    # to the patch shape by the image collection dataset. The split file 'yeaz_bf_splits.json' lives with the data.
    yeaz_kwargs = {"path": os.path.join(input_path, "yeaz"), "patch_shape": patch_shape, "choice": "bf", **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_yeaz_dataset(split="train", n_samples=200, **yeaz_kwargs), source_ndim=2)
    )
    val_ds.append(UniDataWrapper(datasets.get_yeaz_dataset(split="val", n_samples=50, **yeaz_kwargs), source_ndim=2))

    # 15. BCCD (blood cell segmentation in brightfield blood smear images)
    # NOTE: No native val split exists: the 146-image test split is blind and a random 10 % of the 1063 train images
    # (seed 42) validate. The instance labels are connected components of a binary mask, but the cells are
    # separated cleanly in practice (only 0.8 % of the labelled area sits in objects larger than 2.5x the median).
    bccd_paths = datasets.bccd.get_bccd_paths(path=os.path.join(input_path, "bccd"), split="train")
    bccd_train, bccd_val = train_test_split(bccd_paths, test_size=0.1, random_state=42)
    bccd_kwargs = {"patch_shape": patch_shape, "with_channels": True, "ndim": 2, **kwargs}
    for paths, ds_list, n_samples in [(bccd_train, train_ds, 400), (bccd_val, val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=paths, raw_key="raw", label_paths=paths, label_key="labels/instances",
                    n_samples=n_samples, **bccd_kwargs,
                ), source_ndim=2,
            )
        )

    # 16. NeurIPS CellSeg 2022 (cell segmentation in brightfield, phase contrast, DIC and fluorescence images)
    # NOTE: The raw images mix grayscale and RGB layouts and uint8, uint16, int32 and float64 dtypes. The default
    # raw transform covers all of them (channel triplication and per-image percentile normalization). Many images
    # are only sparsely annotated, so the MinInstanceSampler is essential here. 'val' is the challenge 'Tuning' set.
    neurips_kwargs = {"root": os.path.join(input_path, "neurips_cellseg"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(
            datasets.get_neurips_cellseg_supervised_dataset(split="train", n_samples=500, **neurips_kwargs),
            source_ndim=2,
        )
    )
    val_ds.append(
        UniDataWrapper(
            datasets.get_neurips_cellseg_supervised_dataset(split="val", n_samples=100, **neurips_kwargs),
            source_ndim=2,
        )
    )

    # 17. BitDepth NucSeg (nucleus segmentation in DAPI fluorescence images at four magnifications)
    # NOTE: 70 images over four magnifications, split 80 / 10 / 10 at random (seed 42) within each magnification, so
    # that every magnification validates and is tested; the test images are blind.
    bitdepth_kwargs = {"patch_shape": patch_shape, **kwargs}
    bd_train_r, bd_val_r, bd_train_l, bd_val_l = [], [], [], []
    for magnification in BITDEPTH_MAGNIFICATIONS:
        bitdepth_raw, bitdepth_labels = datasets.bitdepth_nucseg.get_bitdepth_nucseg_paths(
            path=os.path.join(input_path, "bitdepth_nucseg"), magnification=magnification,
        )
        (train_r, val_r, _), (train_l, val_l, _) = _train_val_test_split(bitdepth_raw, bitdepth_labels)
        bd_train_r.extend(train_r)
        bd_val_r.extend(val_r)
        bd_train_l.extend(train_l)
        bd_val_l.extend(val_l)
    for raws, labs, ds_list, n_samples in [
        (bd_train_r, bd_train_l, train_ds, 200), (bd_val_r, bd_val_l, val_ds, 50),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, raw_key=None, label_paths=labs, label_key=None,
                    is_seg_dataset=False, ndim=2, n_samples=n_samples, **bitdepth_kwargs,
                ), source_ndim=2,
            )
        )

    # 18. BMGD (nucleus segmentation in DAPI fluorescence images on four substrate stiffnesses)
    # NOTE: Native images are only 345x382, so they are randomly upscaled and padded to the patch shape. The 819
    # crops are split 80 / 10 / 10 at random (seed 42); the test crops are blind.
    bmgd_paths = datasets.bmgd.get_bmgd_paths(path=os.path.join(input_path, "bmgd"))
    ((bmgd_train, bmgd_val, _),) = _train_val_test_split(bmgd_paths)
    bmgd_kwargs = {
        "patch_shape": (345, 382), "with_channels": False, "ndim": 2,
        **{**kwargs, "transform": _with_flips(_random_resize_and_pad_trafo, patch_shape)},
    }
    for paths, ds_list, n_samples in [(bmgd_train, train_ds, 300), (bmgd_val, val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=paths, raw_key="raw", label_paths=paths, label_key="labels/instances",
                    n_samples=n_samples, **bmgd_kwargs,
                ), source_ndim=2,
            )
        )

    # 20. Cell-ACDC (yeast cell segmentation in phase contrast time-lapse)
    # NOTE: Native fields are only ~200x300, so they are randomly upscaled and padded to the patch shape. The seven
    # movies are split by movie: one control position validates, one treated position is blind.
    acdc_raw, acdc_labels = datasets.cell_acdc.get_cell_acdc_paths(os.path.join(input_path, "cell_acdc"))
    acdc_movies = [cell_acdc_movie(p) for p in acdc_raw]
    cell_acdc_kwargs = {
        "patch_shape": (1, 200, 200), "raw_key": None, "label_key": None, "is_seg_dataset": True, "ndim": 2,
        **{**kwargs, "transform": _with_flips(_random_resize_and_pad_trafo, patch_shape)},
    }
    for keep, n_samples, ds_list in [
        ([m not in CELL_ACDC_VAL_MOVIES and m not in CELL_ACDC_TEST_MOVIES for m in acdc_movies], 200, train_ds),
        ([m in CELL_ACDC_VAL_MOVIES for m in acdc_movies], 50, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=[p for p, k in zip(acdc_raw, keep) if k],
                    label_paths=[p for p, k in zip(acdc_labels, keep) if k],
                    n_samples=n_samples, **cell_acdc_kwargs,
                ), source_ndim=2,
            )
        )

    # 22. CellBinDB (nucleus segmentation in DAPI, ssDNA and mIF images; the H&E tiles train with histopathology)
    # NOTE: The tiles are 256x256, so they are randomly upscaled and padded to the patch shape. Each stain is split
    # 80 / 10 / 10 at random (seed 42); the test tiles are blind.
    cellbindb_kwargs = {
        "patch_shape": (256, 256), "is_seg_dataset": False, "ndim": 2,
        **{**kwargs, "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape)},
    }
    for stain in CELLBINDB_STAINS:
        cellbindb_raw, cellbindb_labels = datasets.cellbindb.get_cellbindb_paths(
            path=os.path.join(input_path, "cellbindb"), data_choice=stain,
        )
        (cb_train_r, cb_val_r, _), (cb_train_l, cb_val_l, _) = _train_val_test_split(cellbindb_raw, cellbindb_labels)
        for raws, labs, ds_list, n_samples in [
            (cb_train_r, cb_train_l, train_ds, 150), (cb_val_r, cb_val_l, val_ds, 30),
        ]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=raws, raw_key=None, label_paths=labs, label_key=None,
                        n_samples=n_samples, **cellbindb_kwargs,
                    ), source_ndim=2,
                )
            )

    # 23. CELLULAR (cell segmentation in Drosophila cells)
    # NOTE: Only the brightfield channel is used. The two fluorescence channels light up a few strongly
    # expressing cells, while the labels cover every cell in the field, which only brightfield resolves.
    # The 53 fields come from six wells and are split by well: I04 validates, C06 is blind.
    cellular_paths = datasets.cellular.get_cellular_paths(path=os.path.join(input_path, "cellular"))
    cellular_wells = [os.path.basename(p).split("_")[3] for p in cellular_paths]
    cellular_train = [p for p, w in zip(cellular_paths, cellular_wells) if w in CELLULAR_TRAIN_WELLS]
    cellular_val = [p for p, w in zip(cellular_paths, cellular_wells) if w in CELLULAR_VAL_WELLS]
    cellular_kwargs = {"patch_shape": patch_shape, "ndim": 2, **kwargs}
    for paths, ds_list, n_samples in [(cellular_train, train_ds, 400), (cellular_val, val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=paths, raw_key="raw/brightfield", label_paths=paths,
                    label_key="labels/instances", n_samples=n_samples, **cellular_kwargs,
                ), source_ndim=2,
            )
        )

    # 24. CISD (urothelial cell segmentation in brightfield urine cytology)
    # NOTE: Only 2-3 cells per image, so the shared 3-instance sampler would reject nearly every patch. The 3911
    # crops come from 30 slides and are split by slide (see CISD_VAL_SLIDES, CISD_TEST_SLIDES); the test slides
    # are blind.
    cisd_raw, cisd_labels = datasets.cisd.get_cisd_paths(os.path.join(input_path, "cisd"), mode="center_slice")
    cisd_slides = [os.path.basename(p).split("_")[0] for p in cisd_raw]
    cisd_kwargs = {
        "patch_shape": patch_shape, "raw_key": None, "label_key": None, "is_seg_dataset": False, "ndim": 2,
        **{**kwargs, "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0])},
    }
    for keep, n_samples, ds_list in [
        ([sl not in CISD_VAL_SLIDES and sl not in CISD_TEST_SLIDES for sl in cisd_slides], 200, train_ds),
        ([sl in CISD_VAL_SLIDES for sl in cisd_slides], 50, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=[p for p, k in zip(cisd_raw, keep) if k],
                    label_paths=[p for p, k in zip(cisd_labels, keep) if k],
                    n_samples=n_samples, **cisd_kwargs,
                ), source_ndim=2,
            )
        )

    # 26. DeMemSeg (prospore membrane segmentation in fluorescence yeast crops)
    # NOTE: Native crops are 200x200, so they are randomly upscaled and padded to the patch shape.
    # NOTE: Each 200x200 crop holds a single yeast cell with only a handful of prospore membranes, so the
    # shared 3-instance sampler rejects many patches outright.
    dememseg_kwargs = {
        "path": os.path.join(input_path, "dememseg"), "patch_shape": (200, 200),
        **{
            **kwargs,
            "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape),
            "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
        },
    }
    train_ds.append(
        UniDataWrapper(datasets.get_dememseg_dataset(split="train", n_samples=200, **dememseg_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_dememseg_dataset(split="val", n_samples=50, **dememseg_kwargs), source_ndim=2)
    )

    # 28. DynamicNuclearNet (nucleus segmentation in fluorescence time-lapse frames)
    dnn_kwargs = {"path": os.path.join(input_path, "dynamicnuclearnet"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_dynamicnuclearnet_dataset(split="train", n_samples=600, **dnn_kwargs),
                       source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_dynamicnuclearnet_dataset(split="val", n_samples=50, **dnn_kwargs), source_ndim=2)
    )

    # 29. FlyWing (epithelial cell segmentation in fluorescence membrane images)
    # NOTE: The train and val splits are native 128x128 tiles, so they are randomly upscaled and padded.
    flywing_kwargs = {
        "path": os.path.join(input_path, "flywing"), "patch_shape": (128, 128),
        **{**kwargs, "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape)},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_flywing_dataset(split="train", n_samples=400, **flywing_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_flywing_dataset(split="val", n_samples=50, **flywing_kwargs), source_ndim=2)
    )

    # 30. PNAS Arabidopsis (3D cell segmentation in confocal shoot apical meristem, acyl-YFP membranes)
    # NOTE: The labels have no id 0. Background is numbered 1 and covers about three quarters of a slice, so it
    # is mapped to 0 first, exactly as for PlantSeg root.
    for z in z_slices:
        pnas_trafo = partial(
            _background_id_label_trafo, background_id=1,
            label_trafo=label_trafo() if label_trafo is not None else kwargs.get("label_transform2"),
        )
        pnas_kwargs = {
            "path": os.path.join(input_path, "pnas_arabidopsis"),
            "patch_shape": (z, *patch_shape),
            "label_transform2": pnas_trafo,
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0, 1]),
            "n_samples": max(1, 400 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        train_ds.append(
            UniDataWrapper(
                datasets.get_pnas_arabidopsis_dataset(plants=PNAS_TRAIN_PLANTS, **pnas_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_pnas_arabidopsis_dataset(plants=PNAS_VAL_PLANTS, **pnas_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )

    # 31. EpiCure (epithelial cell segmentation in fluorescence membrane movies)
    # NOTE: Four model systems (Drosophila notum and histoblasts, zebrafish telencephalon, quail gastrula).
    # The frames are exhaustively labelled and very dense, 250-370 cells per patch. Patch shape is
    # (1, 512, 512) because the movies are stored as time series, so one frame is drawn per sample. The quail
    # frame labels an unannotated corner as one cell, 54x the median cell, which the oversized-label rule drops.
    epicure_kwargs = {
        "path": os.path.join(input_path, "epicure"), "patch_shape": (1, *patch_shape),
        "label_transform2": partial(
            _drop_oversized_label_trafo, max_fraction=0.02,
            label_trafo=label_trafo() if label_trafo is not None else kwargs.get("label_transform2"),
        ),
        **{k: v for k, v in kwargs.items() if k != "label_transform2"},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_epicure_dataset(n_samples=500, **epicure_kwargs), source_ndim=2)
    )

    # 32. CartoCell (3D cell segmentation in confocal epithelial cysts)
    # NOTE: The on-disk layout is 'CartoCell/{train_M1,train_M2,validation,test}/{x,y}', which the torch-em loader
    # does not expect, so the paths are given explicitly. The official folders are used: train_M1 and train_M2
    # train, validation validates and test stays blind. Native volumes are only ~84-128 px in XY, so an 80x80 crop
    # is always resized up to 512x512 rather than zero-padded.
    cartocell_root = os.path.join(input_path, "cartocell", "CartoCell")

    def _cartocell_paths(*folders):
        raw = sorted(p for folder in folders for p in glob(os.path.join(cartocell_root, folder, "x", "*.tif")))
        labels = [p.replace(os.sep + "x" + os.sep, os.sep + "y" + os.sep) for p in raw]
        assert raw and all(os.path.exists(p) for p in labels)
        return raw, labels

    cc_train_r, cc_train_l = _cartocell_paths(*CARTOCELL_TRAIN_FOLDERS)
    cc_val_r, cc_val_l = _cartocell_paths(CARTOCELL_VAL_FOLDER)

    for z in z_slices:
        cartocell_kwargs = {
            "patch_shape": (z, 80, 80),
            "raw_transform": _resize_raw_to_512,
            "label_transform2": (
                partial(_resize_then_em_label_trafo, em_trafo_fn=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
            "n_samples": max(1, 400 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
        }
        for raws, labs, ds_list in [(cc_train_r, cc_train_l, train_ds), (cc_val_r, cc_val_l, val_ds)]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=raws, raw_key=None, label_paths=labs, label_key=None,
                        is_seg_dataset=True, **cartocell_kwargs,
                    ), source_ndim=3, group_key=(3, z),
                )
            )

    # 33. C. elegans atlas (3D nucleus segmentation in confocal fluorescence)
    for z in z_slices:
        celegans_kwargs = {
            "path": os.path.join(input_path, "celegans_atlas"),
            "patch_shape": (z, 128, 128),
            "raw_transform": _resize_raw_to_512,
            "label_transform2": (
                partial(_resize_then_em_label_trafo, em_trafo_fn=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "n_samples": max(1, 300 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2"]},
        }
        train_ds.append(
            UniDataWrapper(
                datasets.get_celegans_atlas_dataset(split="train", **celegans_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_celegans_atlas_dataset(split="val", **celegans_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )

    # 35. U2OS (nucleus segmentation in Hoechst fluorescence images)
    # NOTE: 200 images without an official split, split 80 / 10 / 10 at random (seed 42); the test images are blind.
    u20s_raw, u20s_labels = datasets.u20s.get_u20s_paths(os.path.join(input_path, "u20s"))
    (u20s_train_r, u20s_val_r, _), (u20s_train_l, u20s_val_l, _) = _train_val_test_split(u20s_raw, u20s_labels)
    u20s_kwargs = {
        "patch_shape": patch_shape, "is_seg_dataset": False, "raw_key": None, "label_key": None, "ndim": 2, **kwargs
    }
    for raws, labs, n_samples, ds_list in [
        (u20s_train_r, u20s_train_l, 300, train_ds), (u20s_val_r, u20s_val_l, 50, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **u20s_kwargs
                ), source_ndim=2,
            )
        )

    # 36. IFNuclei (nucleus segmentation in immunofluorescence images)
    # NOTE: A random 10 % of the official train split (seed 42) validates; the official test split is blind.
    ifnuclei_raw, ifnuclei_labels = datasets.ifnuclei.get_ifnuclei_paths(
        os.path.join(input_path, "ifnuclei"), split="train",
    )
    if_train_r, if_val_r, if_train_l, if_val_l = train_test_split(
        ifnuclei_raw, ifnuclei_labels, test_size=0.1, random_state=42
    )
    ifnuclei_kwargs = {
        "patch_shape": patch_shape, "is_seg_dataset": False, "raw_key": None, "label_key": None, "ndim": 2, **kwargs
    }
    for raws, labs, n_samples, ds_list in [(if_train_r, if_train_l, 200, train_ds), (if_val_r, if_val_l, 50, val_ds)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **ifnuclei_kwargs
                ), source_ndim=2,
            )
        )

    # 37. VICAR (cell segmentation in quantitative phase imaging of five cell lines)
    # NOTE: Each cell line is split 80 / 10 / 10 at random (seed 42); the test images are blind.
    vicar_kwargs = {
        "patch_shape": patch_shape, "raw_key": None, "label_key": None, "is_seg_dataset": False, "ndim": 2, **kwargs
    }
    vicar_train_r, vicar_val_r, vicar_train_l, vicar_val_l = [], [], [], []
    for cell_type in datasets.vicar.VALID_CELL_TYPES:
        vicar_raw, vicar_labels = datasets.vicar.get_vicar_paths(
            os.path.join(input_path, "vicar"), cell_types=[cell_type]
        )
        (train_r, val_r, _), (train_l, val_l, _) = _train_val_test_split(vicar_raw, vicar_labels)
        vicar_train_r.extend(train_r)
        vicar_val_r.extend(val_r)
        vicar_train_l.extend(train_l)
        vicar_val_l.extend(val_l)
    for raws, labs, n_samples, ds_list in [
        (vicar_train_r, vicar_train_l, 300, train_ds), (vicar_val_r, vicar_val_l, 50, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **vicar_kwargs
                ), source_ndim=2,
            )
        )

    # 39. microbeSEG (bacteria segmentation in phase contrast images)
    # NOTE: Native images are 320x320, so they are randomly upscaled and padded to the patch shape.
    microbeseg_kwargs = {
        "path": os.path.join(input_path, "microbeseg"), "patch_shape": (320, 320),
        **{**kwargs, "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape)},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_microbeseg_dataset(split="train", n_samples=150, **microbeseg_kwargs),
                       source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_microbeseg_dataset(split="val", n_samples=50, **microbeseg_kwargs), source_ndim=2)
    )

    # 40. OrgLine (organoid segmentation in brightfield images across six organs)
    # NOTE: 59 % of the training images hold one or two organoids, which the default three-instance sampler can
    # never accept, so organoid images sample with a single-instance sampler (also OrganoID below).
    organoid_sampler = MinInstanceSampler(min_num_instances=1, exclude_ids=[0])
    orgline_kwargs = {
        "path": os.path.join(input_path, "orgline"), "patch_shape": patch_shape, "sampler": organoid_sampler,
        **{k: v for k, v in kwargs.items() if k != "sampler"},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_orgline_dataset(split="train", n_samples=500, **orgline_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_orgline_dataset(split="val", n_samples=50, **orgline_kwargs), source_ndim=2)
    )

    # 41. OrganoID (pancreatic organoid segmentation in brightfield culture wells)
    # NOTE: The 'original' (human) and 'mouse' subsets train on their official splits, their test splits are blind.
    # The 'gemcitabine' subset has no split and is left out.
    organoid_kwargs = {
        "path": os.path.join(input_path, "organoid"), "patch_shape": patch_shape, "sampler": organoid_sampler,
        **{k: v for k, v in kwargs.items() if k != "sampler"},
    }
    for source, n_train in ORGANOID_SOURCES.items():
        for split, n_samples, ds_list in [("train", n_train, train_ds), ("val", n_train // 4, val_ds)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_organoid_dataset(split=split, source=source, n_samples=n_samples, **organoid_kwargs),
                    source_ndim=2,
                )
            )

    # 42. EnSeg (enteric neuron segmentation in immunofluorescence whole-mount myenteric plexus)
    # NOTE: Stored RGB but only the green channel carries signal (maxima 41/251/43).
    enseg_kwargs = {
        "path": os.path.join(input_path, "enseg"), "patch_shape": patch_shape,
        "raw_transform": _enseg_green_channel,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"},
    }
    for animal_tags, n_samples, ds_list in [(ENSEG_TRAIN_ANIMALS, 200, train_ds), (ENSEG_VAL_ANIMALS, 50, val_ds)]:
        ds_list.append(
            UniDataWrapper(
                datasets.get_enseg_dataset(animal_tags=list(animal_tags), n_samples=n_samples, **enseg_kwargs),
                source_ndim=2,
            )
        )

    # 43. LPC-NucSeg (nucleus segmentation in DNA fluorescence images)
    # NOTE: Two cell lines, U2OS (gnf) and NIH3T3 (ic100); a random 10 % of each (seed 42) validates, there is no
    # blind test.
    lpc_kwargs = {"patch_shape": patch_shape, "raw_key": "raw", "label_key": "labels", "ndim": 2, **kwargs}
    for source in LPC_NUCSEG_SOURCES:
        lpc_paths = datasets.lpc_nucseg.get_lpc_nucseg_paths(os.path.join(input_path, "lpc_nucseg"), source=source)
        lpc_train, lpc_val = train_test_split(lpc_paths, test_size=0.1, random_state=42)
        for paths, n_samples, ds_list in [(lpc_train, 100, train_ds), (lpc_val, 25, val_ds)]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=paths, label_paths=paths, n_samples=n_samples, **lpc_kwargs
                    ), source_ndim=2,
                )
            )

    # 44. DCIS.COM nuclei (nucleus segmentation in spinning-disk confocal SiR-DNA images)
    # NOTE: The official train split trains and the two official test images validate; there is no blind test.
    dcis_train_r, dcis_train_l = datasets.dcis_com_nuclei.get_dcis_com_nuclei_paths(
        os.path.join(input_path, "dcis_com_nuclei"), split="train",
    )
    dcis_val_r, dcis_val_l = datasets.dcis_com_nuclei.get_dcis_com_nuclei_paths(
        os.path.join(input_path, "dcis_com_nuclei"), split="test",
    )
    dcis_kwargs = {
        "patch_shape": patch_shape, "is_seg_dataset": False, "raw_key": None, "label_key": None, "ndim": 2, **kwargs
    }
    for raws, labs, n_samples, ds_list in [
        (dcis_train_r, dcis_train_l, 200, train_ds), (dcis_val_r, dcis_val_l, 50, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **dcis_kwargs
                ), source_ndim=2,
            )
        )

    # 45. mCellSeg (cell segmentation in DIC and brightfield images of HEK-293T and HUVEC)
    # NOTE: The 200 images are split 80 / 10 / 10 at random (seed 42); the test images are blind.
    mcellseg_raw, mcellseg_labels = datasets.mcellseg.get_mcellseg_paths(os.path.join(input_path, "mcellseg"))
    (mc_train_r, mc_val_r, _), (mc_train_l, mc_val_l, _) = _train_val_test_split(mcellseg_raw, mcellseg_labels)
    mcellseg_kwargs = {
        "patch_shape": patch_shape, "raw_key": None, "label_key": None, "is_seg_dataset": False, "ndim": 2, **kwargs
    }
    for raws, labs, n_samples, ds_list in [(mc_train_r, mc_train_l, 200, train_ds), (mc_val_r, mc_val_l, 50, val_ds)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **mcellseg_kwargs
                ), source_ndim=2,
            )
        )

    # 46. TOIAM (bacteria segmentation in phase contrast time-lapse colonies)
    # NOTE: The colony grows over time, so density is bimodal: a 512 patch holds a median of 39 objects with
    # quartiles at 8 and 305. A 25-instance minimum keeps 56 % of patches and cuts the near-empty early frames.
    # The five movies are split by movie: 03 validates, 04 is blind.
    toiam_raw, toiam_labels = datasets.toiam.get_toiam_paths(os.path.join(input_path, "toiam"))
    toiam_movies = [os.path.basename(os.path.dirname(p)) for p in toiam_raw]
    toiam_kwargs = {
        "patch_shape": patch_shape, "raw_key": None, "label_key": None, "is_seg_dataset": False, "ndim": 2,
        **{**kwargs, "sampler": MinInstanceSampler(min_num_instances=25, exclude_ids=[0])},
    }
    for keep, n_samples, ds_list in [
        ([m in TOIAM_TRAIN_MOVIES for m in toiam_movies], 400, train_ds),
        ([m in TOIAM_VAL_MOVIES for m in toiam_movies], 50, val_ds),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=[p for p, k in zip(toiam_raw, keep) if k],
                    label_paths=[p for p, k in zip(toiam_labels, keep) if k],
                    n_samples=n_samples, **toiam_kwargs,
                ), source_ndim=2,
            )
        )

    # 47. Usiigaci (cell segmentation in phase contrast fibroblast images)
    usiigaci_kwargs = {"path": os.path.join(input_path, "usiigaci"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_usiigaci_dataset(split="train", n_samples=200, **usiigaci_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_usiigaci_dataset(split="val", n_samples=50, **usiigaci_kwargs), source_ndim=2)
    )

    # 50. Pan-multiplex (cell segmentation in MIBI, CODEX and Vectra tissue imaging)
    # NOTE: The loader returns (nuclei, membrane); they are reordered into TissueNet's membrane, nucleus, zeros.
    pan_kwargs = {
        "path": os.path.join(input_path, "pan_multiplex"), "patch_shape": patch_shape,
        "raw_channel": "both", "raw_transform": _pan_multiplex_tissuenet_order,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"},
    }
    # The official train / val / test split is used per subset; the test split is blind.
    for subset in ["codex_colon", "mibi_breast", "mibi_decidua", "vectra_colon", "vectra_pancreas"]:
        for split, n_samples, ds_list in [("train", 150, train_ds), ("val", 30, val_ds)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_pan_multiplex_dataset(subset=subset, split=split, n_samples=n_samples, **pan_kwargs),
                    source_ndim=2,
                )
            )

    # 51. Xenium (nucleus and cell segmentation in whole-slide multi-tissue stain)
    # NOTE: XOA segmented nuclei on DAPI (channel 0) and grew cells from the three morphology stains (channels
    # 1-3), so each target gets the channels it was made from. Whole slides are largely empty, hence the sampler.
    xenium_sampler = MinInstanceSampler(min_num_instances=10, exclude_ids=[0])
    xenium_nuclei_kwargs = {
        "path": os.path.join(input_path, "xenium"), "patch_shape": patch_shape,
        "raw_channel": "dapi", "label_channel": "nuclei",
        **{**kwargs, "sampler": xenium_sampler},
    }
    xenium_cells_kwargs = {
        "path": os.path.join(input_path, "xenium"), "patch_shape": patch_shape,
        "raw_channel": "stack", "label_channel": "cells", "raw_transform": _xenium_cell_channels,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}, "sampler": xenium_sampler,
    }
    # The slides are split for both targets: four train, human_skin validates, human_breast is blind.
    for xenium_kwargs in (xenium_nuclei_kwargs, xenium_cells_kwargs):
        for samples, n_samples, ds_list in [(XENIUM_TRAIN_SAMPLES, 400, train_ds), (XENIUM_VAL_SAMPLES, 50, val_ds)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_xenium_dataset(sample=list(samples), n_samples=n_samples, **xenium_kwargs),
                    source_ndim=2,
                )
            )

    # 52. GoNuclear (3D nucleus segmentation in confocal Arabidopsis ovules)
    # NOTE: Volume 1170 is the blind test volume by convention, 1139 validates.
    for z in z_slices:
        gonuclear_kwargs = {
            "path": os.path.join(input_path, "gonuclear"),
            "patch_shape": (z, *patch_shape),
            "segmentation_task": "nuclei",
            "n_samples": max(1, 400 // n_z),
            **kwargs,
        }
        train_ds.append(
            UniDataWrapper(
                datasets.get_gonuclear_dataset(sample_ids=GONUCLEAR_TRAIN_SAMPLES, **gonuclear_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_gonuclear_dataset(sample_ids=GONUCLEAR_VAL_SAMPLES, **gonuclear_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )

    # 53. NucVerse3D (3D nucleus segmentation in two-photon liver and confocal fly glia)
    # NOTE: Volumes are 320 px or smaller in plane, so a 256 crop is resized up rather than zero-padded. The official
    # test volumes are blind. Liver and liver_hcc give up one train volume for validation; drosophila_glia has only
    # three train volumes, so the first 20 % of slices of one test volume validate instead.
    nucverse_root = os.path.join(input_path, "nucverse3d")
    nucverse_train, nucverse_val, nucverse_val_rois = [], [], []
    for name in ("liver", "liver_hcc", "drosophila_glia"):
        paths = datasets.nucverse3d.get_nucverse3d_paths(nucverse_root, dataset=name, split="train")
        val_volume = NUCVERSE_VAL_VOLUMES.get(name)
        nucverse_train.extend(p for p in paths if os.path.basename(p) != val_volume)
        if val_volume is not None:
            nucverse_val.extend(p for p in paths if os.path.basename(p) == val_volume)
            nucverse_val_rois.append((slice(None),))
    glia_test = datasets.nucverse3d.get_nucverse3d_paths(nucverse_root, dataset="drosophila_glia", split="test")
    nucverse_val.extend(p for p in glia_test if os.path.basename(p) == NUCVERSE_GLIA_VAL_VOLUME)
    nucverse_val_rois.append((NUCVERSE_GLIA_VAL_Z,))
    assert len(nucverse_val) == 3 and len(nucverse_train) == 15

    for z in z_slices:
        nucverse_kwargs = {
            "patch_shape": (z, 256, 256),
            "raw_key": "raw",
            "label_key": "labels",
            "is_seg_dataset": True,
            "raw_transform": _resize_raw_to_512,
            "label_transform2": (
                partial(_resize_then_em_label_trafo, em_trafo_fn=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "n_samples": max(1, 300 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2"]},
        }
        train_ds.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=nucverse_train, label_paths=nucverse_train, **nucverse_kwargs
                ), source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=nucverse_val, label_paths=nucverse_val, rois=nucverse_val_rois, **nucverse_kwargs
                ), source_ndim=3, group_key=(3, z),
            )
        )

    # 54. PhMamm (3D cell segmentation in light-sheet Phallusia embryo membranes)
    # NOTE: Volumes are 256 cubed, so the patch is taken at the native in-plane size and resized up.
    for z in z_slices:
        phmamm_kwargs = {
            "path": os.path.join(input_path, "phmamm"),
            "patch_shape": (z, 256, 256),
            "raw_transform": _resize_raw_to_512,
            "label_transform2": (
                partial(_resize_then_em_label_trafo, em_trafo_fn=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "n_samples": max(1, 400 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2"]},
        }
        train_ds.append(
            UniDataWrapper(
                datasets.get_phmamm_dataset(timepoints=PHMAMM_TRAIN_TIMEPOINTS, **phmamm_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_phmamm_dataset(timepoints=PHMAMM_VAL_TIMEPOINTS, **phmamm_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )

    # 56. BBBC030 (CHO cell segmentation in DIC images)
    # NOTE: 60 images of 1032x1376 with 1-3 cells per 512 crop, hence the 1-instance sampler. The background spans
    # about 4 % of the intensity range, so the raw is min-max normalized instead of percentile normalized. The
    # torch-em split (seed 42) gives 40 / 8 / 12 images; the test split is blind.
    bbbc030_kwargs = {
        "path": os.path.join(input_path, "bbbc030"), "patch_shape": patch_shape, "raw_transform": _minmax_raw_trafo,
        **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "sampler"]},
        "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
    }
    for split, n_samples, ds_list in [("train", 100, train_ds), ("val", 25, val_ds)]:
        ds_list.append(
            UniDataWrapper(
                datasets.get_bbbc030_dataset(split=split, n_samples=n_samples, **bbbc030_kwargs), source_ndim=2,
            )
        )

    # 57. Tsakiroglou (nucleus segmentation in the DAPI channel of multiplex IF lymphoma TMA cores)
    # NOTE: 41 grayscale crops smaller than 256 px on at least one side, so the whole image is drawn and randomly
    # upscaled and padded to the patch shape. A random 10 % of the official train split (seed 42) validates, the
    # three official test images are blind.
    tsakiroglou_raw, tsakiroglou_labels = datasets.histopathology.tsakiroglou.get_tsakiroglou_paths(
        os.path.join(input_path, "tsakiroglou"), split="train",
    )
    ts_train_r, ts_val_r, ts_train_l, ts_val_l = train_test_split(
        tsakiroglou_raw, tsakiroglou_labels, test_size=0.1, random_state=42
    )
    tsakiroglou_kwargs = {
        "patch_shape": None, "raw_key": None, "label_key": None, "is_seg_dataset": False, "ndim": 2,
        "raw_transform": _to_8bit,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"},
        "transform": _with_flips(_random_resize_and_pad_trafo, patch_shape),
    }
    for raws, labs, n_samples, ds_list in [(ts_train_r, ts_train_l, 100, train_ds), (ts_val_r, ts_val_l, 25, val_ds)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **tsakiroglou_kwargs
                ), source_ndim=2,
            )
        )

    # 55. Wing disc (3D cell segmentation in confocal Drosophila wing epithelium)
    # NOTE: Native in-plane size is exactly 512, so no resize or padding is needed.
    for z in z_slices:
        wing_disc_kwargs = {
            "path": os.path.join(input_path, "wing_disc"),
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 300 // n_z),
            **kwargs,
        }
        train_ds.append(
            UniDataWrapper(
                datasets.get_wing_disc_dataset(volumes=WING_DISC_TRAIN_VOLUMES, **wing_disc_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_wing_disc_dataset(
                    volumes=WING_DISC_TEST_VOLUMES, rois=[(WING_DISC_VAL_Z,)] * len(WING_DISC_TEST_VOLUMES),
                    **wing_disc_kwargs,
                ), source_ndim=3, group_key=(3, z),
            )
        )

    # 58. CochleaNet SGN (3D soma segmentation of spiral ganglion neurons, PV stain, light-sheet mouse cochlea)
    # NOTE: Internal data with official train and val folders. The blocks are 256 px in plane and are resized up. The
    # module leaves out the empty blocks and the 'resized' ones, which were interpolated and are only partly labelled.
    cochleanet_root = os.path.join(input_path, "cochleanet_data")
    for z in z_slices:
        sgn_kwargs = {
            "patch_shape": (z, 256, 256),
            "raw_transform": _resize_raw_to_512,
            "label_transform2": (
                partial(_resize_then_em_label_trafo, em_trafo_fn=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=4, exclude_ids=[0]),
            "n_samples": max(1, 300 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
        }
        for split, ds_list in [("train", train_ds), ("val", val_ds)]:
            ds_list.append(
                UniDataWrapper(
                    cochleanet_data.get_cochleanet_dataset(cochleanet_root, name="sgn", split=split, **sgn_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 59. CochleaNet IHC (3D cell segmentation of inner hair cells, Vglut3 stain, light-sheet mouse cochlea)
    # NOTE: Internal data with official train and val folders. The hair cells form one row, so a block is 1-7 %
    # labelled, holds few cells per crop and gets a lower sampler minimum than the SGN blocks. The blocks come in
    # three in-plane sizes and get one leaf each: 256 and 392 are resized up, 512 is native. The module reorients the
    # 512x512x256 blocks so that their short axis is z and leaves out the 'G-LR' crops, which label a few hair cells
    # next to unlabelled rows of cells.
    for z in z_slices:
        for split, ds_list in [("train", train_ds), ("val", val_ds)]:
            paths = cochleanet_data.get_cochleanet_paths(cochleanet_root, name="ihc", split=split)
            groups = {}
            for path in paths:
                groups.setdefault(_ihc_patch_size(_in_plane_size(path)), []).append(path)
            for size, group_paths in sorted(groups.items()):
                resize = size < 512
                ihc_kwargs = {
                    "patch_shape": (z, size, size),
                    "raw_key": "raw",
                    "label_key": "labels",
                    "is_seg_dataset": True,
                    "raw_transform": _resize_raw_to_512 if resize else kwargs["raw_transform"],
                    "label_transform2": (
                        (partial(_resize_then_em_label_trafo, em_trafo_fn=label_trafo(instances=True)) if resize
                         else label_trafo())
                        if label_trafo is not None else kwargs.get("label_transform2")
                    ),
                    "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
                    "n_samples": max(1, (200 // n_z) * len(group_paths) // len(paths)),
                    **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
                }
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=group_paths, label_paths=group_paths, **ihc_kwargs
                        ), source_ndim=3, group_key=(3, z),
                    )
                )

    return train_ds, val_ds


def _in_plane_size(path):
    """The smaller in-plane extent of an h5 volume stored as (z, y, x)."""
    with open_file(path, "r") as f:
        return min(f["raw"].shape[1:])


def _ihc_patch_size(size):
    """The in-plane patch size of a CochleaNet IHC block: 512 native, otherwise the largest of 256 and 392 it holds."""
    return 512 if size >= 512 else 392 if size >= 392 else 256


def _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo, _em_label_trafo=None):
    """Get all electron microscopy (EM) datasets for generalist training.

    Args:
        _em_label_trafo: EM cell label transform function to use. Defaults to
            :func:`_em_cell_label_trafo`. Pass :func:`_joint_em_cell_label_trafo`
            when building joint interactive+automatic datasets.

    Returns:
        Tuple of (train_ds, val_ds) lists of UniDataWrapper instances.
    """
    if _em_label_trafo is None:
        _em_label_trafo = _em_cell_label_trafo

    # Connectomics ids exceed 2^24 and would merge in the float32 flip augmentation.
    kwargs = {**kwargs, "label_transform": partial(_compact_label_ids, keep=(PLATY_IGNORE_LABEL,))}

    train_ds, val_ds = [], []
    n_z = len(z_slices)

    # 1. CREMI (neuron segmentation in vEM)
    # NOTE: Neurons are large - a patch typically contains only 1-2 of them, so min_num_instances=3
    # would reject nearly every sample. Use min_num_instances=1 to require just one foreground object.
    for z in z_slices:
        cremi_kwargs = {
            "path": os.path.join(input_path, "cremi"),
            "patch_shape": (z, *patch_shape),
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(10, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
            "defect_augmentation_kwargs": {
                "p_drop_slice": 0.025,
                "p_low_contrast": 0.0,
                "p_deform_slice": 0.0,
                "deformation_mode": "compress",
            },
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]}
        }

        # Sample C is the blind in-domain test set (shared with NeuronSeg); the last 25 of the 125 sections of
        # A and B validate.
        for rois, ds_list, n_samples in [
            ({"A": np.s_[:100, :, :], "B": np.s_[:100, :, :]}, train_ds, 500),
            ({"A": np.s_[100:, :, :], "B": np.s_[100:, :, :]}, val_ds, 50),
        ]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_cremi_dataset(
                        samples=("A", "B"), rois=rois, n_samples=max(1, n_samples // n_z), **cremi_kwargs
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 2. EMNeuron (neuron segmentation in vEM)
    # NOTE: Large neurons - use min_num_instances=1 (same reasoning as CREMI).
    # J0126-sbem (train: 150×150 or 256×256 XY) and FIB25 (val: 250×250 XY) are too small
    # for the standard 512×512 patch shape - they get their own 128×128 patch group with a
    # resize-to-512 transform applied to both raw and label before the EM label transform.
    from torch_em.data.datasets.electron_microscopy.emneuron import get_emneuron_paths

    emneuron_path = os.path.join(input_path, "emneuron")
    all_train_raw, all_train_lbl = get_emneuron_paths(emneuron_path, "train")
    # The AxonEM, CREMI and FIB-25 folders copy the public releases at full depth, including the sections and
    # volumes the direct loaders hold out for validation and testing, so they train through those loaders only.
    keep = [not any(f"{os.sep}{folder}{os.sep}" in p for folder in EMNEURON_EXCLUDED_FOLDERS) for p in all_train_raw]
    all_train_raw = [p for p, k in zip(all_train_raw, keep) if k]
    all_train_lbl = [p for p, k in zip(all_train_lbl, keep) if k]
    all_val_raw, all_val_lbl = get_emneuron_paths(emneuron_path, "val")
    # Only the in-distribution validation volumes validate. The out-of-distribution folder holds the Harris
    # hippocampus volume, the source of the SynapseWeb OOD test, next to Ionsem and Microns.
    keep = [os.sep + "InDistribution" + os.sep in p for p in all_val_raw]
    all_val_raw = [p for p, k in zip(all_val_raw, keep) if k]
    all_val_lbl = [p for p, k in zip(all_val_lbl, keep) if k]

    def _split(raw_paths, label_paths, small_keys):
        small_r = [r for r in raw_paths if any(k in r for k in small_keys)]
        small_l = [l for r, l in zip(raw_paths, label_paths) if any(k in r for k in small_keys)]
        rest_r = [r for r in raw_paths if not any(k in r for k in small_keys)]
        rest_l = [l for r, l in zip(raw_paths, label_paths) if not any(k in r for k in small_keys)]
        return small_r, small_l, rest_r, rest_l

    sm_train_r, sm_train_l, rest_train_r, rest_train_l = _split(all_train_raw, all_train_lbl, ["J0126"])
    sm_val_r, sm_val_l, rest_val_r, rest_val_l = _split(all_val_raw, all_val_lbl, ["J0126", "FIB25"])

    base_sampler = MinInstanceSampler(min_num_instances=1, exclude_ids=[0])
    base_kwargs = {k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]}
    base_kwargs["label_transform"] = partial(_compact_native_label_ids, keep=(PLATY_IGNORE_LABEL,))

    for z in z_slices:
        em_label_trafo_fn = (
            partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
            if label_trafo is not None else kwargs.get("label_transform2")
        )

        # Normal volumes (XY >= 512)
        rest_kwargs = {
            "patch_shape": (z, *patch_shape),
            "label_transform2": em_label_trafo_fn,
            "sampler": base_sampler,
            **base_kwargs,
        }
        train_ds.append(UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=rest_train_r, raw_key=None, label_paths=rest_train_l, label_key=None,
                is_seg_dataset=True, n_samples=max(1, 500 // n_z), **rest_kwargs,
            ), source_ndim=3, group_key=(3, z),
        ))
        val_ds.append(UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=rest_val_r, raw_key=None, label_paths=rest_val_l, label_key=None,
                is_seg_dataset=True, n_samples=max(1, 450 // n_z), **rest_kwargs,
            ), source_ndim=3, group_key=(3, z),
        ))

        # Small volumes (J0126 train; J0126+FIB25 val): 128×128 patches -> resize to 512×512
        small_kwargs = {
            "patch_shape": (z, 128, 128),
            "raw_transform": _resize_raw_to_512,
            "label_transform2": partial(_resize_then_em_label_trafo, em_trafo_fn=em_label_trafo_fn),
            "sampler": base_sampler,
            **{k: v for k, v in base_kwargs.items() if k != "raw_transform"},
        }
        train_ds.append(UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=sm_train_r, raw_key=None, label_paths=sm_train_l, label_key=None,
                is_seg_dataset=True, n_samples=max(1, 500 // n_z), **small_kwargs,
            ), source_ndim=3, group_key=(3, z),
        ))
        val_ds.append(UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=sm_val_r, raw_key=None, label_paths=sm_val_l, label_key=None,
                is_seg_dataset=True, n_samples=max(1, 450 // n_z), **small_kwargs,
            ), source_ndim=3, group_key=(3, z),
        ))

    # 3. Platynereis (cell segmentation in vEM)
    platy_root = os.path.join(input_path, "platynereis")
    platy_cell_template = "membrane/train_data_membrane_%02i.n5"
    label_key = "volumes/labels/segmentation/s1"

    # Volume 9 is held out: its neuropil id covers 41% of the volume, the largest share of any volume.
    platy_train_ids, platy_val_ids = [1, 2, 3, 4, 5, 6], [7, 8]

    train_rois = _compute_platy_rois(
        platy_root, platy_train_ids, ignore_label=0, file_template=platy_cell_template, label_key=label_key,
    )
    val_rois = _compute_platy_rois(
        platy_root, platy_val_ids, ignore_label=0, file_template=platy_cell_template, label_key=label_key,
    )

    for z in z_slices:
        platynereis_kwargs = {
            "path": os.path.join(input_path, "platynereis"),
            "patch_shape": (z, *patch_shape),
            # sampling=None: ~20nm isotropic
            "label_transform2": (
                partial(
                    _em_label_trafo, label_trafo=label_trafo(instances=True), ignore_label=PLATY_IGNORE_LABEL
                )
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            # The neuropil ignore label is not an instance, so the sampler must not count it.
            "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0, PLATY_IGNORE_LABEL]),
            # Volumes 7 and 8 hold an unannotated tissue band (12 % and 8 % of the roi) next to the labelled
            # cells; it maps to the ignore label instead of being trained as background. Volumes 1-6 are fully
            # labelled, so the transform leaves them unchanged.
            "transform": partial(
                _ignore_unlabelled_blobs_trafo, ignore_label=PLATY_IGNORE_LABEL, min_area=2000,
                transform=get_augmentations(ndim=3),
            ),
            # get_platynereis_cell_dataset concatenates one dataset per volume, so n_samples is per volume.
            "n_samples": max(1, 500 // (n_z * len(platy_train_ids))),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler", "transform"]}
        }

        train_ds.append(
            UniDataWrapper(
                datasets.get_platynereis_cell_dataset(
                    sample_ids=platy_train_ids, rois=train_rois, **platynereis_kwargs
                ),
                source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_platynereis_cell_dataset(
                    sample_ids=platy_val_ids, rois=val_rois, **platynereis_kwargs
                ),
                source_ndim=3, group_key=(3, z),
            )
        )

    # 4. SNEMI (neuron segmentation in vEM)
    # The official test volume has no public labels, so the 100 training sections split into train, val and a
    # blind test slab: z < 60, 60 <= z < 80, z >= 80.
    snemi_train_rois = np.s_[:60, :, :]
    snemi_val_rois = np.s_[60:80, :, :]

    for z in z_slices:
        snemi_kwargs = {
            "path": os.path.join(input_path, "snemi"),
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 500 // n_z),
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(5, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }

        train_ds.append(
            UniDataWrapper(
                datasets.get_snemi_dataset(rois=snemi_train_rois, **snemi_kwargs), source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_snemi_dataset(rois=snemi_val_rois, **snemi_kwargs), source_ndim=3, group_key=(3, z),
            )
        )

    # 5. Igor cells (cell segmentation in vEM)
    # NOTE: This data is used for training only. No validation data is added for it.
    # The volumes are (16, 1024, 1024) uint8 blocks with dense uint32 instance labels.
    igor_cells_root = os.path.join(input_path, "igor_cells")
    all_igor_cells_paths = sorted(glob(os.path.join(igor_cells_root, "data_block_*.tif")))
    igor_cells_raw_paths = [p for p in all_igor_cells_paths if not p.endswith("_seg.tif")]
    igor_cells_label_paths = [p.replace(".tif", "_seg.tif") for p in igor_cells_raw_paths]
    assert igor_cells_raw_paths, f"Did not find any volumes in '{igor_cells_root}'."
    assert all(os.path.exists(p) for p in igor_cells_label_paths)

    for z in z_slices:
        igor_cells_kwargs = {
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 500 // n_z),
            # sampling=None: the volumes are isotropic.
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]}
        }

        train_ds.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=igor_cells_raw_paths, raw_key=None,
                    label_paths=igor_cells_label_paths, label_key=None,
                    is_seg_dataset=True, **igor_cells_kwargs,
                ), source_ndim=3, group_key=(3, z),
            )
        )

    # 6. AxonEM (axon segmentation in vEM of human and mouse cortex)
    # NOTE: Only a central 50x512x512 block per volume is annotated, so each is cropped to its label bounding
    # box. Three mouse blocks lie inside a soma (one or two ids, no neurite boundaries) and are dropped.
    axonem_raw_paths, axonem_label_paths = datasets.axonem.get_axonem_paths(
        path=os.path.join(input_path, "axonem"), samples=("human", "mouse"), download=True,
    )
    axonem_rois = _compute_label_rois(axonem_label_paths, label_key="main", min_ids=AXONEM_MIN_IDS)
    axonem_raw_paths = [p for p, lp in zip(axonem_raw_paths, axonem_label_paths) if lp in axonem_rois]
    axonem_label_paths = [lp for lp in axonem_label_paths if lp in axonem_rois]
    axonem_val = [p for p in axonem_label_paths if os.path.basename(p) in AXONEM_VAL_VOLUMES]
    axonem_test = [p for p in axonem_label_paths if os.path.basename(p) in AXONEM_TEST_VOLUMES]
    axonem_train = [p for p in axonem_label_paths if p not in axonem_val + axonem_test]
    assert len(axonem_val) == len(AXONEM_VAL_VOLUMES) and len(axonem_test) == len(AXONEM_TEST_VOLUMES)

    for z in z_slices:
        axonem_kwargs = {
            "patch_shape": (z, *patch_shape),
            # sampling=None: 30nm isotropic.
            "label_transform2": (
                partial(
                    _em_label_trafo, label_trafo=label_trafo(instances=True), ignore_label=MISSING_RAW_IGNORE_LABEL
                )
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            # Some slices hold missing tiles (exact-zero raw) that still carry labels; they become ignore. The test
            # needs the stored intensities, so the raw is only prepared here and normalized inside the transform.
            "raw_transform": _prepare_identity,
            "transform": partial(
                _ignore_missing_raw_trafo, ignore_label=MISSING_RAW_IGNORE_LABEL, transform=get_augmentations(ndim=3)
            ),
            # uint8 labels cannot hold the ignore label, and torch_em recasts to the loaded dtype.
            "pre_label_transform": _labels_to_uint32,
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{
                k: v for k, v in kwargs.items()
                if k not in ["raw_transform", "label_transform2", "sampler", "transform", "pre_label_transform"]
            },
        }
        for label_paths, ds_list, n_samples in [(axonem_train, train_ds, 500), (axonem_val, val_ds, 50)]:
            raw_paths = [p.replace("seg_", "im_") for p in label_paths]
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=raw_paths, raw_key="main", label_paths=label_paths, label_key="main",
                        rois=[axonem_rois[p] for p in label_paths], is_seg_dataset=True,
                        n_samples=max(1, n_samples // n_z), **axonem_kwargs,
                    ), source_ndim=3, group_key=(3, z),
                )
            )

    # 7. FAFB (neuron segmentation in ssTEM of the full adult fly brain, FlyWire v783 labels)
    # NOTE: Streamed from GCS and cached per box at 16x16x40 nm, the finest level of the segmentation. torch_em's
    # default boxes are tissue-verified 1024x1024x410 crops; one is the validation set.
    for z in z_slices:
        fafb_kwargs = {
            "path": os.path.join(input_path, "fafb"),
            "patch_shape": (z, *patch_shape),
            "download": True,
            # sampling: z is 2.5x coarser than xy.
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(2.5, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for boxes, ds_list, n_samples in [(FAFB_TRAIN_BOXES, train_ds, 500), (FAFB_VAL_BOXES, val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_fafb_dataset(bounding_boxes=boxes, n_samples=max(1, n_samples // n_z), **fafb_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 8. ASTIH (myelinated axon segmentation in SEM and brightfield nerve)
    # NOTE: SEM1, BF1 and BF2 only; TEM is covered by AxonDeepSeg and TEM1 alone would dominate. Labels are
    # semantic (1=myelin, 2=axon): instances are components of the axon class, since touching sheaths would
    # bridge neighbouring fibres. The 22 train images split 80/20, the 4 test images stay blind.
    astih_paths = datasets.astih.get_astih_paths(
        path=os.path.join(input_path, "astih"), name=ASTIH_SUBSETS, split="train", download=True,
    )
    astih_train, astih_val = train_test_split(astih_paths, test_size=0.2, random_state=42)
    astih_kwargs = {
        "patch_shape": patch_shape,
        "raw_transform": _to_8bit,
        "pre_label_transform": _astih_pre_label_transform,
        "label_transform2": (
            partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
            if label_trafo is not None else kwargs.get("label_transform2")
        ),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "ndim": 2,
        **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
    }
    for paths, ds_list, n_samples in [(astih_train, train_ds, 300), (astih_val, val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=paths, raw_key="raw", label_paths=paths, label_key="labels",
                    is_seg_dataset=True, n_samples=n_samples, **astih_kwargs,
                ), source_ndim=2,
            )
        )

    # 9. FIB-25 (neuron segmentation in FIB-SEM of the Drosophila medulla, 8 nm isotropic)
    # NOTE: training_sample2 and validation_sample train, tstvol-520-1 stays blind; FIB-25 adds no validation leaf.
    for z in z_slices:
        fib25_kwargs = {
            "path": os.path.join(input_path, "fib25"),
            "patch_shape": (z, *patch_shape),
            "download": True,
            "ndim": 3,
            # sampling=None: 8 nm isotropic.
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        train_ds.append(
            UniDataWrapper(
                datasets.get_fib25_dataset(samples=FIB25_TRAIN_SAMPLES, n_samples=max(1, 500 // n_z), **fib25_kwargs),
                source_ndim=3, group_key=(3, z),
            )
        )

    # 10. Hemibrain (neuron segmentation in FIB-SEM of the Drosophila central brain, 8 nm isotropic)
    # NOTE: One cached 1024^3 crop (torch_em's default box), ~99% of voxels labelled, proofread. Same pipeline as
    # MANC and MaleCNS.
    for z in z_slices:
        hemibrain_kwargs = {
            "path": os.path.join(input_path, "hemibrain"),
            "patch_shape": (z, *patch_shape),
            "label_choice": "neurons",
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        # Of the 1024 sections, z < 700 train, 700 <= z < 820 validate and z >= 820 stay blind.
        for roi, ds_list, n_samples in [(np.s_[:700, :, :], train_ds, 500), (np.s_[700:820, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_hemibrain_dataset(rois=[roi], n_samples=max(1, n_samples // n_z), **hemibrain_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 11. MANC (neuron segmentation in FIB-SEM of the Drosophila male adult nerve cord, 8 nm isotropic)
    # NOTE: A separate specimen from MaleCNS (bucket flyem-vnc-2-26), so the two do not overlap. One cached
    # 1024^3 crop (torch_em's default box).
    for z in z_slices:
        manc_kwargs = {
            "path": os.path.join(input_path, "manc"),
            "patch_shape": (z, *patch_shape),
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        # Of the 1024 sections, z < 700 train, 700 <= z < 820 validate and z >= 820 stay blind.
        for roi, ds_list, n_samples in [(np.s_[:700, :, :], train_ds, 500), (np.s_[700:820, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_manc_dataset(rois=[roi], n_samples=max(1, n_samples // n_z), **manc_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 12. MaleCNS (neuron segmentation in FIB-SEM of the whole Drosophila male CNS, 8 nm isotropic)
    # NOTE: Six 1024^3 crops streamed from GCS, placed by probing the segmentation density along the
    # brain-neck-VNC axis; four train, one VNC crop validates and the neck connective crop stays blind.
    for z in z_slices:
        malecns_kwargs = {
            "path": os.path.join(input_path, "malecns"),
            "patch_shape": (z, *patch_shape),
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for boxes, ds_list, n_samples in [(MALECNS_TRAIN_BOXES, train_ds, 500), (MALECNS_VAL_BOXES, val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_malecns_dataset(
                        bounding_boxes=boxes, n_samples=max(1, n_samples // n_z), **malecns_kwargs
                    ), source_ndim=3, group_key=(3, z),
                )
            )

    # 13. Wafer4 (neuron segmentation in multi-beam SEM of mouse medial entorhinal cortex, layer 6)
    # NOTE: One 125x1250x1250 volume at 8x8x35 nm. The authors' split is z < 100 train, z >= 100 test; the test
    # sections stay blind and validation is the last fifth of the training sections.
    for z in z_slices:
        wafer4_kwargs = {
            "path": os.path.join(input_path, "wafer4"),
            "patch_shape": (z, *patch_shape),
            "split": "train",
            "download": True,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(4.4, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for roi, ds_list, n_samples in [(np.s_[:80, :, :], train_ds, 300), (np.s_[80:, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_wafer4_dataset(rois=roi, n_samples=max(1, n_samples // n_z), **wafer4_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 14. MICrONS minnie65 (neuron segmentation in the cubic-millimetre mouse visual cortex, 8x8x40 nm)
    # NOTE: torch_em's own split of 14 cached 512x4096x4096 boxes: 8 train, 2 val, 4 test kept blind. At 69 G
    # voxels the train split is the largest EM source by far; n_samples keeps its epoch share in line.
    for z in z_slices:
        minnie_kwargs = {
            "path": os.path.join(input_path, "microns-minnie65"),
            "patch_shape": (z, *patch_shape),
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(5, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for split, ds_list, n_samples in [("train", train_ds, 800), ("val", val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_microns_minnie65_dataset(
                        split=split, n_samples=max(1, n_samples // n_z), **minnie_kwargs
                    ), source_ndim=3, group_key=(3, z),
                )
            )

    # 15. MICrONS pinky (hand-annotated neuron instances in ssEM of mouse visual cortex, 4x4x40 nm)
    # NOTE: Each file is cropped to the bounding box of 'volumes/mask', the annotated region inside padded
    # context. Only the neuropil blocks are used; basil splits nucleus and cytoplasm into separate ids.
    pinky_root = os.path.join(input_path, "microns", "pinky")
    for z in z_slices:
        pinky_kwargs = {
            "patch_shape": (z, *patch_shape),
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(10, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for files, ds_list, n_samples in [(PINKY_TRAIN_FILES, train_ds, 300), (PINKY_VAL_FILES, val_ds, 50)]:
            paths = [os.path.join(pinky_root, name) for name in files]
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=paths, raw_key="volumes/image", label_paths=paths, label_key="volumes/segmentation",
                        rois=[PINKY_MASK_ROIS[name] for name in files], is_seg_dataset=True,
                        n_samples=max(1, n_samples // n_z), **pinky_kwargs,
                    ), source_ndim=3, group_key=(3, z),
                )
            )

    # 16. Zebrafinch j0126 (neuron segmentation in FIB-SEM of zebra finch Area X, 10x10x20 nm, Kornfeld lab)
    # NOTE: Somata, vessels and missing tiles carry no id, so unlabelled blobs above 2 um^2 map to the ignore
    # label. The last fifth of z validates. j0251 (10x10x25 nm) uses boxes placed by a tissue scan, since
    # torch_em's cached boxes sit at the empty volume corner.
    for z in z_slices:
        zebrafinch_kwargs = {
            "path": os.path.join(input_path, "zebrafinch"),
            "patch_shape": (z, *patch_shape),
            "dataset": "j0126",
            "bounding_box": ZEBRAFINCH_J0126_BOX,
            "label_choice": "neurons",
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(
                    _em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(2, 1, 1)),
                    ignore_label=MISSING_RAW_IGNORE_LABEL,
                )
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "raw_transform": _prepare_identity,  # the missing-tile test needs the stored intensities
            "transform": partial(
                _ignore_missing_raw_trafo, ignore_label=MISSING_RAW_IGNORE_LABEL,
                transform=partial(
                    _ignore_unlabelled_blobs_trafo, ignore_label=MISSING_RAW_IGNORE_LABEL, min_area=20000,
                    transform=get_augmentations(ndim=3),
                ),
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0, MISSING_RAW_IGNORE_LABEL]),
            **{
                k: v for k, v in kwargs.items()
                if k not in ["raw_transform", "label_transform2", "sampler", "transform"]
            },
        }
        # Of the 640 cached sections, z < 448 train, 448 <= z < 512 validate and z >= 512 stay blind.
        for roi, ds_list, n_samples in [(np.s_[:448, :, :], train_ds, 500), (np.s_[448:512, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_zebrafinch_dataset(
                        rois=roi, n_samples=max(1, n_samples // n_z), **zebrafinch_kwargs
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )
        j0251_kwargs = {**zebrafinch_kwargs, "dataset": "j0251"}
        j0251_kwargs["label_transform2"] = (
            partial(
                _em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(2.5, 1, 1)),
                ignore_label=MISSING_RAW_IGNORE_LABEL,
            )
            if label_trafo is not None else kwargs.get("label_transform2")
        )
        j0251_splits = [(ZEBRAFINCH_J0251_TRAIN_BOXES, train_ds, 500), (ZEBRAFINCH_J0251_VAL_BOXES, val_ds, 50)]
        for boxes, ds_list, n_samples in j0251_splits:
            for box in boxes:
                j0251_kwargs["bounding_box"] = tuple(v * r for v, r in zip(box, (10, 10, 10, 10, 25, 25)))
                ds_list.append(
                    UniDataWrapper(
                        datasets.get_zebrafinch_dataset(
                            n_samples=max(1, n_samples // (n_z * len(boxes))), **j0251_kwargs
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 17. Wildenberg 2023 (dense automated segmentation of all processes in FIB-SEM of mouse V1 layer 4, 12x12x40 nm)
    # NOTE: The 'saturated' channel labels every process and soma. The box must be given explicitly, the module
    # default would stream the full 120 x 136 x 36 um experiment.
    for z in z_slices:
        wildenberg_kwargs = {
            "path": os.path.join(input_path, "wildenberg2023"),
            "patch_shape": (z, *patch_shape),
            "experiments": ("p105",),
            "label_choice": "saturated",
            "bounding_box": WILDENBERG_P105_BOX,
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(3.3, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        # Of the 150 cached sections, z < 100 train, 100 <= z < 120 validate and z >= 120 stay blind.
        for roi, ds_list, n_samples in [(np.s_[:100, :, :], train_ds, 300), (np.s_[100:120, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_wildenberg_dataset(
                        rois=[roi], n_samples=max(1, n_samples // n_z), **wildenberg_kwargs
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 18. DenseCell (platelet cells in SBF-SEM of human platelet tissue, 10x10x50 nm)
    # NOTE: The source labels are a semantic mask; torch-em derives and caches 3D cell instances
    # (label_choice 'cell_instances'). The train volume splits into z < 35 for training and z >= 35 for validation;
    # the val volume is the blind in-domain test set. The test split is sparsely annotated and not used.
    for z in z_slices:
        densecell_kwargs = {
            "path": os.path.join(input_path, "densecell"),
            "patch_shape": (z, *patch_shape),
            "label_choice": "cell_instances",
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(5, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for roi, ds_list, n_samples in [(np.s_[:35, :, :], train_ds, 200), (np.s_[35:, :, :], val_ds, 40)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_densecell_dataset(
                        split="train", rois=roi, n_samples=max(1, n_samples // n_z), **densecell_kwargs
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 19. Tumor spheroid EM (FaDu tumor spheroid cells in SBF-SEM, 20 manually annotated 2D slices at 50 nm)
    # NOTE: 2D data. The 100 nm set is the same 20 slices downsampled and is not used; the 3D zarr holds automated
    # segmentation only. Our split: 14 slices train, 2 z slices validate, 4 slices stay blind (see the constants).
    spheroid_paths, spheroid_raw_key, spheroid_label_key = datasets.tumor_spheroid_em.get_tumor_spheroid_paths(
        os.path.join(input_path, "tumor_spheroid_em"), source="2d_manual", resolution="50-50-50", target="cells",
        download=True,
    )
    spheroid_val = [p for p in spheroid_paths if os.path.basename(p) in TUMOR_SPHEROID_VAL_SLICES]
    spheroid_test = [p for p in spheroid_paths if os.path.basename(p) in TUMOR_SPHEROID_TEST_SLICES]
    spheroid_train = [p for p in spheroid_paths if p not in spheroid_val + spheroid_test]
    assert len(spheroid_train) == 14 and len(spheroid_val) == 2 and len(spheroid_test) == 4
    spheroid_kwargs = {
        "patch_shape": patch_shape,
        "label_transform2": (
            partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
            if label_trafo is not None else kwargs.get("label_transform2")
        ),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "ndim": 2,
        **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
    }
    for paths, ds_list, n_samples in [(spheroid_train, train_ds, 300), (spheroid_val, val_ds, 40)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=paths, raw_key=spheroid_raw_key, label_paths=paths, label_key=spheroid_label_key,
                    is_seg_dataset=True, n_samples=n_samples, **spheroid_kwargs,
                ), source_ndim=2,
            )
        )

    # 21. LICONN (neurite segmentation in expansion microscopy of mouse CA1, 18x18x24 nm raw, ~16x expansion)
    # NOTE: Light-based connectomics, so it belongs to the EM neurite pool. The proofread segmentation covers
    # only z 64-640 and y < 4608 of the volume (LICONN_ROI); within it z < 512 trains, 512 <= z < 576 validates
    # and z >= 576 stays blind.
    for z in z_slices:
        liconn_kwargs = {
            "path": os.path.join(input_path, "liconn"),
            "patch_shape": (z, *patch_shape),
            "segmentation": "proofread",
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(1.33, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for z_range, ds_list, n_samples in [(slice(64, 512), train_ds, 300), (slice(512, 576), val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_liconn_dataset(
                        roi=(z_range, *LICONN_ROI[1:]), n_samples=max(1, n_samples // n_z), **liconn_kwargs
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 22. XPRESS (myelinated axon segmentation in X-ray holographic nano-tomography of mouse white matter, 33 nm)
    # NOTE: One volume with voxel labels in a 200^3 core (XPRESS_CORE); like ASTIH the target class is myelinated
    # axons only. Patches are sampled at the native 200x200 and resized to 512 like the small EMNeuron volumes.
    # z 128-288 of the core trains, 288-308 validates, 308-328 stays blind.
    for z in z_slices:
        xpress_kwargs = {
            "path": os.path.join(input_path, "xpress"),
            "patch_shape": (z, 200, 200),
            "download": True,
            "ndim": 3,
            "raw_transform": _resize_raw_to_512,
            "label_transform2": (
                partial(
                    _resize_then_em_label_trafo,
                    em_trafo_fn=partial(_em_label_trafo, label_trafo=label_trafo(instances=True)),
                )
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
        }
        for z_range, ds_list, n_samples in [(slice(128, 288), train_ds, 100), (slice(288, 308), val_ds, 20)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_xpress_dataset(
                        rois=[(z_range, *XPRESS_CORE[1:])], n_samples=max(1, n_samples // n_z), **xpress_kwargs
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 23. SynapseNet compartments (compartment segmentation in electron tomograms of three synapse types)
    # NOTE: Internal data, 11 tomograms of 440-495 px in plane. One tomogram per type is the blind test set, the
    # other eight train on their first 80 % of sections and validate on the rest. Patches are drawn at 440 px and
    # resized to 512 like the small EMNeuron volumes; a patch holds one to four compartments, so the sampler asks
    # for one. Ids of a few voxels are annotation specks, which the minimum size drops. The unlabelled voxels are
    # membranes, gaps and a few unannotated processes, so id 0 is the ignore label and not the background.
    synapsenet_paths = [
        p for p in synapsenet_tomograms(os.path.join(input_path, "synapsenet_compartments_data"))
        if os.path.basename(p) not in SYNAPSENET_TEST_TOMOGRAMS
    ]
    for z in z_slices:
        synapsenet_kwargs = {
            "patch_shape": (z, 440, 440),
            "raw_key": "raw",
            "label_key": "labels/compartments",
            "is_seg_dataset": True,
            "ndim": 3,
            "raw_transform": _resize_raw_to_512,
            "label_transform2": (
                partial(
                    _resize_then_em_label_trafo,
                    em_trafo_fn=partial(
                        _em_label_trafo, label_trafo=label_trafo(instances=True, min_size=200), ignore_label=0
                    ),
                )
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
        }
        for split, ds_list, n_samples in [("train", train_ds, 100), ("val", val_ds, 20)]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=synapsenet_paths, label_paths=synapsenet_paths,
                        rois=[synapsenet_roi(p, split) for p in synapsenet_paths],
                        n_samples=max(1, n_samples // n_z), **synapsenet_kwargs,
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    return train_ds, val_ds


# Cached boxes in nm; both modules would otherwise default to far larger regions.
ZEBRAFINCH_J0126_BOX = (0, 51200, 0, 51200, 0, 12800)
# j0251 boxes in mip-0 voxels (x0, x1, y0, y1, z0, z1) at 10x10x25 nm, converted to nm at use.
# Our 3 / 1 / 1 split of the five density-verified j0251 boxes; the mid-depth box (z 7500) is the blind test set.
ZEBRAFINCH_J0251_TRAIN_BOXES = [
    (6656, 8704, 15360, 17408, 3000, 3256),
    (23040, 25088, 1792, 3840, 3000, 3256),
    (12544, 14592, 20992, 23040, 12000, 12256),
]
ZEBRAFINCH_J0251_VAL_BOXES = [(24576, 26624, 3072, 5120, 12000, 12256)]
ZEBRAFINCH_J0251_TEST_BOXES = [(9472, 11520, 15360, 17408, 7500, 7756)]
WILDENBERG_P105_BOX = (576, 24576, 576, 24576, 160, 6160)

# Bounding boxes of 'volumes/mask' (the annotated region) as (z, y, x) slices.
PINKY_MASK_ROIS = {
    "pinky_stitched_vol19-vol34_realigned.h5": np.s_[16:116, 256:2176, 256:2176],
    "pinky_stitched_vol40-vol41.h5": np.s_[0:86, 256:1280, 256:768],
    "pinky_vol401.h5": np.s_[16:116, 256:768, 256:768],
}
PINKY_TRAIN_FILES = ["pinky_stitched_vol19-vol34_realigned.h5", "pinky_stitched_vol40-vol41.h5"]
PINKY_VAL_FILES = ["pinky_vol401.h5"]


# MaleCNS 1024^3 crops in 8 nm voxel coordinates. z runs from the brain through the neck connective into the VNC.
# Our 4 / 1 / 1 split of the six density-verified boxes; the neck connective box is the blind in-domain test set,
# a region no other dataset in the pool covers.
MALECNS_TRAIN_BOXES = [
    (40000, 41024, 40000, 41024, 20000, 21024),  # brain, torch_em default
    (38912, 39936, 19456, 20480, 15000, 16024),  # brain
    (81920, 82944, 33792, 34816, 35000, 36024),  # brain, far lateral
    (63488, 64512, 57344, 58368, 75000, 76024),  # VNC
]
MALECNS_VAL_BOXES = [(40960, 41984, 50176, 51200, 95000, 96024)]  # VNC
MALECNS_TEST_BOXES = [(49152, 50176, 51200, 52224, 55000, 56024)]  # neck connective

# EMNeuron's CREMI folder holds samples A (all 125 sections), B (sections 25-124) and C at 8 nm, its FIB-25 folder
# tstvol-520-1 and validation_sample, verified by cross-correlation against the cached source volumes.
EMNEURON_EXCLUDED_FOLDERS = ("AxonEM[H]-atum", "AxonEM[M]-sstem", "CREMI-sstem", "Fib-25-fib")
# training_sample2 and validation_sample train in full, tstvol-520-1 is the blind in-domain test set. All three are
# at 8 nm; the training sample is only a smaller cube (250^3), not a coarser one.
FIB25_TRAIN_SAMPLES = ("training_sample2", "validation_sample")
FIB25_TEST_SAMPLE = "tstvol-520-1"

PLATY_IGNORE_LABEL = datasets.platynereis.CELL_IGNORE_LABEL

# Mouse volumes 0-0-0, 0-0-3584 and 0-3584-3584 are soma blocks with 1, 10 and 4 ids; this threshold drops them.
AXONEM_MIN_IDS = 50
# Our split of the 15 usable blocks: human 6 / 1 / 2 and mouse 3 / 1 / 2 (train / val / blind test).
AXONEM_VAL_VOLUMES = ("seg_950-3584-3584_pad.h5", "seg_700-3584-3584_pad.h5")
AXONEM_TEST_VOLUMES = (
    "seg_950-0-3584_pad.h5", "seg_950-3584-0_pad.h5",  # human
    "seg_700-0-3584_pad.h5", "seg_700-3584-0_pad.h5",  # mouse
)

# Voxels of missing raw data are mapped to this label and excluded from the loss.
MISSING_RAW_IGNORE_LABEL = datasets.platynereis.CELL_IGNORE_LABEL

# FAFB crops in 16 nm voxel coordinates, 1024x1024x410 voxels each, chosen inside brain tissue with dense
# segmentation at three depths (torch_em's DEFAULT_BOUNDING_BOXES). One mid-depth central crop is the validation set.
# Of torch-em's nine tissue-verified boxes, the left dorsal mid-depth box validates and the midline ventral anterior
# box is the blind in-domain test set (the only ventral-anterior fly brain region in the pool); the other seven train.
FAFB_VAL_BOXES = [(24576, 25600, 11776, 12800, 3500, 3910)]
FAFB_TEST_BOXES = [(32768, 33792, 18944, 19968, 1500, 1910)]
FAFB_TRAIN_BOXES = [
    box for box in datasets.fafb.DEFAULT_BOUNDING_BOXES if box not in FAFB_VAL_BOXES + FAFB_TEST_BOXES
]

ASTIH_SUBSETS = ["SEM1", "BF1", "BF2"]

# Our 14 / 2 / 4 split of the 20 manually annotated tumor spheroid slices: the two z slices below validate,
# the deepest x, y and two z slices are the blind in-domain test set, the other 14 train.
TUMOR_SPHEROID_VAL_SLICES = ("Au_01-vol_01-z_0180.h5", "Au_01-vol_01-z_0192.h5")
TUMOR_SPHEROID_TEST_SLICES = (
    "Au_01-vol_01-x_1300.h5", "Au_01-vol_01-y_1606.h5", "Au_01-vol_01-z_0212.h5", "Au_01-vol_01-z_0274.h5"
)


# The part of the LICONN volume the proofread segmentation covers; z 576-640 of it is the blind test slab.
LICONN_ROI = (slice(64, 640), slice(0, 4608), slice(None))

# The voxel-labelled core of the XPRESS volume; z 308-328 of it is the blind test slab.
XPRESS_CORE = (slice(128, 328), slice(128, 328), slice(128, 328))

# SynapseNet compartments: one tomogram per synapse type (MF, PS, SC) is the blind test set, the other eight train
# on their first 80 % of sections and validate on the rest.
SYNAPSENET_TEST_TOMOGRAMS = (
    "36859_J1_66K_TS_CA3_MF_19_rec_2Kb1dawbp_crop.h5", "36859_J2_66K_TS_R04_PS11_rec_2Kb1dawbp_crop.h5",
    "36859_H2_SP_11_rec_2Kb1dawbp_crop.h5",
)
SYNAPSENET_TRAIN_FRACTION = 0.8


def synapsenet_tomograms(path):
    """The SynapseNet compartment tomograms, h5 files with 'raw' and 'labels/compartments'."""
    paths = sorted(glob(os.path.join(path, "v3", "*", "*.h5")))
    assert paths, f"Did not find any tomograms in '{path}'."
    return paths


def synapsenet_roi(path, split):
    """The z-slab of a training tomogram that trains ('train') or validates ('val'), as a (z, y, x) roi."""
    with open_file(path, "r") as f:
        n_sections = f["raw"].shape[0]
    cut = int(SYNAPSENET_TRAIN_FRACTION * n_sections)
    return (slice(0, cut) if split == "train" else slice(cut, None), slice(None), slice(None))


# PanNuke fold_2 holds 2523 tiles; the first 80 % train, the last 20 % validate. fold_3 is blind.
PANNUKE_FOLD2_TRAIN_TILES = slice(0, 2018)
PANNUKE_FOLD2_VAL_TILES = slice(2018, 2523)

SPATCH_HE_SUBSETS = ["visium_hd_ov", "visium_hd_hcc", "visium_hd_coad", "stereoseq_ov"]

# CartoCell uses the official folders; 'test' is the blind split.
CARTOCELL_TRAIN_FOLDERS = ("train_M1", "train_M2")
CARTOCELL_VAL_FOLDER = "validation"
CARTOCELL_TEST_FOLDER = "test"

GONUCLEAR_TRAIN_SAMPLES = (1135, 1136, 1137)
GONUCLEAR_VAL_SAMPLES = (1139,)
GONUCLEAR_TEST_SAMPLES = (1170,)

BITDEPTH_MAGNIFICATIONS = ("20x", "40x_air", "40x_oil", "63x_oil")
# CellBinDB fluorescence stains train with light microscopy, the H&E tiles with histopathology.
CELLBINDB_STAINS = ("DAPI", "ssDNA", "mIF")
CELLBINDB_HE_STAIN = "HE"
LPC_NUCSEG_SOURCES = ("gnf", "ic100")

# Xenium is split by slide, shared by the nucleus and cell targets.
XENIUM_TRAIN_SAMPLES = ("human_pancreas", "human_lung_cancer", "mouse_colon", "human_prostate")
XENIUM_VAL_SAMPLES = ("human_skin",)
XENIUM_TEST_SAMPLES = ("human_breast",)

# EnSeg is split by animal; the C and TW tags are the two experimental groups.
ENSEG_TRAIN_ANIMALS = ("2C", "4C", "22TW", "23TW")
ENSEG_VAL_ANIMALS = ("5C",)
ENSEG_TEST_ANIMALS = ("28TW",)

# CTC datasets of the v1 generalist; Fluo-N2DH-GOWT1 and Fluo-N2DL-HeLa stay out.
CTC_DATASETS = (
    "BF-C2DL-HSC", "BF-C2DL-MuSC", "DIC-C2DH-HeLa", "Fluo-C2DL-Huh7", "Fluo-C2DL-MSC", "Fluo-N2DH-SIM+",
    "PhC-C2DH-U373", "PhC-C2DL-PSC",
)

# Omnipose subsets and their samples per epoch; a random 10 % of each official train subset validates.
OMNIPOSE_TRAIN_SAMPLES = {"bact_fluor": 150, "bact_phase": 250, "worm": 50, "worm_high_res": 50}

# TOIAM is split by movie (the folder name).
TOIAM_TRAIN_MOVIES = ("00", "01", "02")
TOIAM_VAL_MOVIES = ("03",)
TOIAM_TEST_MOVIES = ("04",)

# OrganoID subsets and their samples per epoch; both use the official splits.
ORGANOID_SOURCES = {"original": 200, "mouse": 60}

# CISD is split by slide (the filename prefix); three slides validate, three are blind, drawn with seed 42.
CISD_VAL_SLIDES = ("0239", "0243", "0248")
CISD_TEST_SLIDES = ("0235", "0251", "0255")

# CELLULAR is split by well.
CELLULAR_TRAIN_WELLS = ("C03", "C05", "I06", "K07")
CELLULAR_VAL_WELLS = ("I04",)
CELLULAR_TEST_WELLS = ("C06",)

# Cell-ACDC is split by movie, named '<experiment folder>/Position_<n>'.
CELL_ACDC_VAL_MOVIES = ("MIA_KC_htb1_mCitrine_flu_control_labeled/Position_1",)
CELL_ACDC_TEST_MOVIES = ("MIA_KC_htb1_mCitrine_labeled/Position_8",)


def cell_acdc_movie(path):
    """The movie a Cell-ACDC frame stack belongs to, as '<experiment folder>/Position_<n>'."""
    parts = os.path.normpath(path).split(os.sep)
    return "/".join(parts[parts.index("TimeLapse_2D") + 1:parts.index("TimeLapse_2D") + 3])


# CVZ Fluo is split by Vectra patient or CODEX / Zeiss slide, shared by the cell and DAPI stains.
CVZ_VAL_GROUPS = ("Vectra:P07", "Vectra:P12", "Zeiss:ZP-10002")
CVZ_TEST_GROUPS = ("Vectra:P01", "Vectra:P11", "Zeiss:Spleen", "CODEX:CODEX_LN")


def cvz_group(path):
    """The patient (Vectra) or slide (CODEX, Zeiss) a CVZ crop belongs to, as 'instrument:name'."""
    parts = os.path.normpath(path).split(os.sep)
    instrument = parts[parts.index("cvz") + 1]
    name = os.path.basename(path)
    match = re.match(r"(P\d+)-", name)
    return f"{instrument}:{match.group(1) if match else name.split('(')[0]}"


# NucVerse3D: one train volume per liver collection validates; the glia validation is a slab of a test volume.
NUCVERSE_VAL_VOLUMES = {
    "liver": "20221006_13_Control_8w_Periportal.h5", "liver_hcc": "20221014_6_control_16w_centro.h5",
}
NUCVERSE_GLIA_VAL_VOLUME = "C2_M01.h5"
NUCVERSE_GLIA_VAL_Z = slice(0, 12)  # of 53; at least the deepest training patch

# Where a blind test volume also validates, a 20 % slab of its slices is the validation region.
NIS3D_VAL_Z = slice(158, 198)  # Drosophila_1, of 198; the first slices hold almost no nuclei in the centre
EMBEDSEG_VAL_Z = {"Mouse-Skull-Nuclei-CBG": slice(100, 125), "Platynereis-ISH-Nuclei-CBG": slice(0, 21)}  # of 125 / 105

# EmbedSeg Platynereis-Nuclei is one time-lapse of 9 volumes; timepoint 300 validates, 350 is blind.
EMBEDSEG_PLATY_NUCLEI_TRAIN_TIMEPOINTS = slice(0, 7)
EMBEDSEG_PLATY_NUCLEI_VAL_TIMEPOINTS = slice(7, 8)
EMBEDSEG_PLATY_NUCLEI_TEST_TIMEPOINTS = slice(8, 9)

# EmbedSeg Mouse-Organoid-Cells is one time-lapse of 108 volumes and is split along time; the last block is blind.
EMBEDSEG_ORGANOID_TRAIN_TIMEPOINTS = slice(0, 86)
EMBEDSEG_ORGANOID_VAL_TIMEPOINTS = slice(86, 97)
EMBEDSEG_ORGANOID_TEST_TIMEPOINTS = slice(97, 108)

# PhMamm is one time-lapse (t001-t101, t008 missing) and is split along time; the last block is blind.
PHMAMM_TRAIN_TIMEPOINTS = range(1, 82)
PHMAMM_VAL_TIMEPOINTS = range(82, 92)
PHMAMM_TEST_TIMEPOINTS = range(92, 102)

# Wing disc has one confocal and one multiphoton volume for training; the other two are the blind test volumes,
# of which the first slices validate (20 % of the 60-slice volume; the 38-slice volume gives up 12 as well).
WING_DISC_TRAIN_VOLUMES = ("WD1_15-02_WT_confocalonly", "WD1.1_17-03_WT_MP")
WING_DISC_TEST_VOLUMES = ("WD2.1_21-02_WT_confocalonly", "WD3.2_21-03_WT_MP")
WING_DISC_VAL_Z = slice(0, 12)

# PNAS Arabidopsis is split by plant, since the timepoints of one plant are near-duplicates.
PNAS_TRAIN_PLANTS = ("plant1", "plant2", "plant13", "plant15")
PNAS_VAL_PLANTS = ("plant4",)
PNAS_TEST_PLANTS = ("plant18",)


def dsb_fluorescence_training_paths(path):
    """The DSB fluorescence images of the 'full' source without the StarDist test images.

    The StarDist test files are named by their Kaggle image id, which is the folder name in the 'full' source, so
    the blind test images are excluded by id.
    """
    test_ids = {
        os.path.splitext(os.path.basename(p))[0]
        for p in datasets.dsb.get_dsb_paths(path, source="reduced", split="test")[0]
    }
    raw, labels = datasets.dsb.get_dsb_paths(path, source="full", domain="fluo")
    keep = [os.path.basename(os.path.dirname(os.path.dirname(p))) not in test_ids for p in raw]
    return [p for p, k in zip(raw, keep) if k], [p for p, k in zip(labels, keep) if k]


def _train_val_test_split(*lists, seed=42):
    """Split parallel path lists 80 / 10 / 10 with a fixed seed; returns (train, val, test) per list."""
    rest_and_test = train_test_split(*lists, test_size=0.1, random_state=seed)
    rest, test = rest_and_test[0::2], rest_and_test[1::2]
    train_and_val = train_test_split(*rest, test_size=1 / 9, random_state=seed)
    train, val = train_and_val[0::2], train_and_val[1::2]
    return [(tr, va, te) for tr, va, te in zip(train, val, test)]


def _compute_label_rois(label_paths, label_key, min_ids=1):
    """Bounding box of the non-zero labels per volume, cached as json next to each file.

    Volumes with fewer than *min_ids* ids inside that box are left out of the result.
    """
    rois = {}
    for path in label_paths:
        cache_path = f"{os.path.splitext(path)[0]}_roi.json"
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
        else:
            with open_file(path, "r") as f:
                labels = f[label_key][:]
            coords = np.where(labels != 0)
            n_ids = int(len(np.unique(labels)) - 1)
            roi = [[int(c.min()), int(c.max()) + 1] for c in coords] if n_ids else []
            cached = {"roi": roi, "n_ids": n_ids}
            tmp_path = f"{cache_path}.tmp{os.getpid()}"
            with open(tmp_path, "w") as f:
                json.dump(cached, f)
            os.replace(tmp_path, cache_path)
        if cached["n_ids"] >= min_ids:
            rois[path] = tuple(slice(start, stop) for start, stop in cached["roi"])
    return rois


def _resize_pad_then_flip(raw, labels, resize, patch_shape, flip):
    raw, labels = resize(raw, labels, patch_shape)
    return flip(raw, labels)


def _with_flips(resize, patch_shape):
    """Compose a resize-and-pad transform with the default 2d flips, which an explicit 'transform' replaces."""
    return partial(_resize_pad_then_flip, resize=resize, patch_shape=patch_shape, flip=get_augmentations(ndim=2))


def _pannuke_random_resize_and_pad_trafo(raw, labels, patch_shape):
    """Randomly upscale a PanNuke 256x256 tile (steps of 64) and zero-pad the rest to patch_shape.

    Runs as the joint 'transform', after normalization, so percentile stats stay on real pixels.
    """
    from skimage.transform import resize

    native = raw.shape[-1]
    target = patch_shape[-1]
    size = random.choice(range(native, target + 1, 64))

    if size != native:
        raw = resize(
            raw, raw.shape[:-2] + (size, size), order=1, anti_aliasing=True, preserve_range=True,
        ).astype(raw.dtype)
        labels = resize(
            labels, labels.shape[:-2] + (size, size), order=0, anti_aliasing=False, preserve_range=True,
        ).astype(labels.dtype)

    pad_total = target - size
    if pad_total <= 0:
        return raw, labels

    pad_width = (0, pad_total)

    def _pad(x):
        return np.pad(x, [(0, 0)] * (x.ndim - 2) + [pad_width, pad_width])

    return _pad(raw), _pad(labels)


def _random_resize_and_pad_trafo(raw, labels, patch_shape):
    """Randomly upscale a whole small image by a factor in [1, target/longest side] and zero-pad to patch_shape.

    A generalization of :func:`_pannuke_random_resize_and_pad_trafo` for datasets whose native images are small
    and not square. Runs as the joint 'transform', after normalization, so percentile stats stay on real pixels.
    """
    from skimage.transform import resize

    target = patch_shape[-1]
    height, width = raw.shape[-2:]
    max_scale = target / max(height, width)
    if max_scale > 1:
        scale = random.uniform(1.0, max_scale)
        new_h, new_w = min(target, int(height * scale)), min(target, int(width * scale))
        raw = resize(
            raw, raw.shape[:-2] + (new_h, new_w), order=1, anti_aliasing=True, preserve_range=True,
        ).astype(raw.dtype)
        labels = resize(
            labels, labels.shape[:-2] + (new_h, new_w), order=0, anti_aliasing=False, preserve_range=True,
        ).astype(labels.dtype)

    def _pad(x):
        pad_h, pad_w = target - x.shape[-2], target - x.shape[-1]
        if pad_h <= 0 and pad_w <= 0:
            return x
        return np.pad(x, [(0, 0)] * (x.ndim - 2) + [(0, max(0, pad_h)), (0, max(0, pad_w))])

    return _pad(raw), _pad(labels)


def _get_hp_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo):
    """Get all histopathology (HP) datasets for generalist training.

    Dataset composition mirrors patho-sam's generalist training set:
    https://github.com/computational-cell-analytics/patho-sam

    Returns:
        Tuple of (train_ds, val_ds) lists of UniDataWrapper instances.
    """
    train_ds, val_ds = [], []

    # 1. CPM15 (nucleus segmentation in H&E histopathology images)
    # NOTE: No official split, and torch-em's own one is an unseeded random pick. All 15 images are pooled and split
    # 80/20 here, so there is no blind set; CPM17's official test split covers that.
    cpm15_raw_paths, cpm15_label_paths = [], []
    for split in ["train", "val", "test"]:
        raws, labels = datasets.cpm.get_cpm_paths(
            path=os.path.join(input_path, "cpm15"), data_choice="cpm15", split=split
        )
        cpm15_raw_paths += raws
        cpm15_label_paths += labels
    cpm15_train_raw, cpm15_val_raw, cpm15_train_labels, cpm15_val_labels = train_test_split(
        cpm15_raw_paths, cpm15_label_paths, test_size=0.2, random_state=42,
    )
    cpm15_kwargs = {"patch_shape": patch_shape, "with_channels": True, "ndim": 2, **kwargs}
    for raws, labels, ds_list in [
        (cpm15_train_raw, cpm15_train_labels, train_ds), (cpm15_val_raw, cpm15_val_labels, val_ds)
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, raw_key=None, label_paths=labels, label_key=None, is_seg_dataset=False,
                    n_samples=50, **cpm15_kwargs,
                ), source_ndim=2,
            )
        )

    # 2. CPM17 (nucleus segmentation in H&E histopathology images)
    # NOTE: No native val split. Split the train image/label paths so train and val get
    # independent dataset instances (a shared random_split Subset would alias raw_transform).
    cpm17_raw_paths, cpm17_label_paths = datasets.cpm.get_cpm_paths(
        path=os.path.join(input_path, "cpm17"), data_choice="cpm17", split="train",
    )
    cpm17_train_raw, cpm17_val_raw, cpm17_train_labels, cpm17_val_labels = train_test_split(
        cpm17_raw_paths, cpm17_label_paths, test_size=0.2, random_state=42,
    )
    cpm17_kwargs = {"patch_shape": patch_shape, "with_channels": True, "ndim": 2, **kwargs}
    train_ds.append(
        UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=cpm17_train_raw, raw_key=None, label_paths=cpm17_train_labels, label_key=None,
                is_seg_dataset=False, n_samples=50, **cpm17_kwargs,
            ), source_ndim=2,
        )
    )
    val_ds.append(
        UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=cpm17_val_raw, raw_key=None, label_paths=cpm17_val_labels, label_key=None,
                is_seg_dataset=False, n_samples=50, **cpm17_kwargs,
            ), source_ndim=2,
        )
    )

    # 3. Lizard (nucleus segmentation in H&E colon, 238 images from six cohorts)
    # NOTE: The train and val splits hold only the crag, dpath and glas cohorts; every PanNuke and CoNSeP source
    # image sits in the test split, so training here never touches the blind PanNuke fold 3. CoNIC is the same
    # material re-tiled and is not used.
    lizard_kwargs = {
        "path": os.path.join(input_path, "lizard"), "patch_shape": patch_shape, "download": True, **kwargs
    }
    train_ds.append(
        UniDataWrapper(datasets.get_lizard_dataset(split="train", n_samples=700, **lizard_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_lizard_dataset(split="val", n_samples=100, **lizard_kwargs), source_ndim=2)
    )

    # 4. Lizard-Mitosis (nucleus segmentation in H&E colon, the dedicated mitosis tile set)
    # NOTE: Only the 'mitosis' subset is used: its tiles are new material, while the 'lizard' subset is CoNIC
    # re-split. Tiles are 256x256 and are resized/padded to 512 like PanNuke.
    lizard_mitosis_kwargs = {
        "path": os.path.join(input_path, "lizard_mitosis"), "patch_shape": (1, 256, 256), "subset": "mitosis",
        "label_choice": "instances", "download": True,
        **{**kwargs, "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape)},
    }
    for split, ds_list, n_samples in [("train", train_ds, 500), ("val", val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                datasets.get_lizard_mitosis_dataset(split=split, n_samples=n_samples, **lizard_mitosis_kwargs),
                source_ndim=2,
            )
        )

    # 5. MoNuSeg (nucleus segmentation in H&E histopathology images)
    # NOTE: No native val split. Split the train image/label paths so train and val get
    # independent dataset instances (a shared random_split Subset would alias raw_transform).
    monuseg_raw_paths, monuseg_label_paths = datasets.monuseg.get_monuseg_paths(
        path=os.path.join(input_path, "monuseg"), split="train", download=True,
    )
    monuseg_train_raw, monuseg_val_raw, monuseg_train_labels, monuseg_val_labels = train_test_split(
        monuseg_raw_paths, monuseg_label_paths, test_size=0.2, random_state=42,
    )
    monuseg_kwargs = {"patch_shape": patch_shape, "is_seg_dataset": False, **kwargs}
    train_ds.append(
        UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=monuseg_train_raw, raw_key=None, label_paths=monuseg_train_labels, label_key=None,
                n_samples=50, **monuseg_kwargs,
            ), source_ndim=2,
        )
    )
    val_ds.append(
        UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=monuseg_val_raw, raw_key=None, label_paths=monuseg_val_labels, label_key=None,
                n_samples=50, **monuseg_kwargs,
            ), source_ndim=2,
        )
    )

    # 6. PanNuke (nucleus segmentation in H&E histopathology images)
    # NOTE: fold_1 and the first 80 % of the fold_2 tiles train, the last 20 % of the fold_2 tiles validate (a fixed
    # tile partition, see PANNUKE_FOLD2_TRAIN_TILES). fold_3 is the blind benchmark split. patch_shape is requested
    # at PanNuke's native 256x256, so torch_em's own padding is a no-op; _pannuke_random_resize_and_pad_trafo does
    # the resize+pad up to 512x512 instead.
    pannuke_kwargs = {
        "path": os.path.join(input_path, "pannuke"), "patch_shape": (1, 256, 256), "download": True, "ndim": 2,
        **{**kwargs, "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape)},
    }
    train_ds.append(
        UniDataWrapper(
            datasets.get_pannuke_dataset(
                folds=["fold_1", "fold_2"], rois={"fold_2": (PANNUKE_FOLD2_TRAIN_TILES, slice(None), slice(None))},
                **pannuke_kwargs,
            ), source_ndim=2,
        )
    )
    val_ds.append(
        UniDataWrapper(
            datasets.get_pannuke_dataset(
                folds=["fold_2"], rois={"fold_2": (PANNUKE_FOLD2_VAL_TILES, slice(None), slice(None))},
                **pannuke_kwargs,
            ), source_ndim=2,
        )
    )

    # 7. PUMA (nucleus segmentation in H&E histopathology images)
    puma_kwargs = {"path": os.path.join(input_path, "puma"), "patch_shape": patch_shape, "download": True, **kwargs}
    train_ds.append(UniDataWrapper(datasets.get_puma_dataset(split="train", **puma_kwargs), source_ndim=2))
    val_ds.append(UniDataWrapper(datasets.get_puma_dataset(split="val", **puma_kwargs), source_ndim=2))

    # 8. TNBC CellType (nucleus segmentation in H&E triple-negative breast cancer plus TCGA brain sections)
    # NOTE: Replaces plain `tnbc`, which it contains with near-identical masks, and adds 18 TCGA brain sections.
    # ndim is explicit: the raw is channels-last (H, W, 3) and auto-detection reads the 3 as a depth axis.
    tnbc_kwargs = {
        "path": os.path.join(input_path, "tnbc_celltype"), "patch_shape": patch_shape, "download": True,
        "ndim": 2, "label_choice": "instances", **kwargs,
    }
    train_ds.append(
        UniDataWrapper(datasets.get_tnbc_celltype_dataset(split="train", n_samples=50, **tnbc_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_tnbc_celltype_dataset(split="val", n_samples=50, **tnbc_kwargs), source_ndim=2)
    )

    # 9. NuInsSeg (nucleus segmentation in H&E histopathology images from 31 human and mouse organs)
    # NOTE: No native split. Split the image/label paths so train and val get independent dataset instances.
    # 'human_liver_31.png' is the one RGBA image of the collection; its constant alpha channel is dropped.
    nuinsseg_raw_paths, nuinsseg_label_paths = datasets.nuinsseg.get_nuinsseg_paths(
        path=os.path.join(input_path, "nuinsseg")
    )
    nuinsseg_train_raw, nuinsseg_val_raw, nuinsseg_train_labels, nuinsseg_val_labels = train_test_split(
        nuinsseg_raw_paths, nuinsseg_label_paths, test_size=0.2, random_state=42,
    )
    nuinsseg_kwargs = {
        "patch_shape": patch_shape, "is_seg_dataset": False, "ndim": 2, "with_channels": True, **kwargs,
        "raw_transform": Compose(_drop_alpha_channel, kwargs["raw_transform"], is_multi_tensor=False),
    }
    train_ds.append(
        UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=nuinsseg_train_raw, raw_key=None, label_paths=nuinsseg_train_labels, label_key=None,
                n_samples=300, **nuinsseg_kwargs,
            ), source_ndim=2,
        )
    )
    val_ds.append(
        UniDataWrapper(
            torch_em.default_segmentation_dataset(
                raw_paths=nuinsseg_val_raw, raw_key=None, label_paths=nuinsseg_val_labels, label_key=None,
                n_samples=50, **nuinsseg_kwargs,
            ), source_ndim=2,
        )
    )

    # 10. LyNSeC (nucleus segmentation in IHC and H&E lymphoma images)
    # NOTE: Both stains are used ('choice' left unset). The raw images are stored as int32 RGB tifs with an 8-bit
    # value range and the labels as int32; the percentile normalization and the label dtype cast handle both.
    # The split files 'lynsec_{ihc,h&e}_split.csv' live with the data.
    lynsec_kwargs = {"path": os.path.join(input_path, "lynsec"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_lynsec_dataset(split="train", n_samples=300, **lynsec_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_lynsec_dataset(split="val", n_samples=50, **lynsec_kwargs), source_ndim=2)
    )

    # 11. SRSA-Net / IHC TMA (nucleus segmentation in IHC tissue microarray images of non-small cell lung cancer)
    # NOTE: Native 256x256 tiles, handled like PanNuke: crop at native size, then randomly upscale and pad to the
    # patch shape in the joint transform. Labels are uint64 connected components of the positive and negative masks.
    # The 35-image test split (fold 3) is kept blind.
    srsanet_kwargs = {
        "path": os.path.join(input_path, "srsanet"), "patch_shape": (256, 256),
        **{**kwargs, "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape)},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_srsanet_dataset(split="train", n_samples=200, **srsanet_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_srsanet_dataset(split="val", n_samples=50, **srsanet_kwargs), source_ndim=2)
    )

    # 12. CryoNuSeg (nucleus segmentation in H&E cryosection images from 10 organs)
    # NOTE: 30 images, rater 'b1'. No official split, and torch-em's own one is an unseeded random pick, so all 30
    # images are pooled and split 80/20 here; there is no blind set.
    cryonuseg_raw_paths, cryonuseg_label_paths = [], []
    for split in ["train", "val", "test"]:
        raws, labels = datasets.cryonuseg.get_cryonuseg_paths(
            path=os.path.join(input_path, "cryonuseg"), split=split, rater_choice="b1"
        )
        cryonuseg_raw_paths += raws
        cryonuseg_label_paths += labels
    cryonuseg_train_raw, cryonuseg_val_raw, cryonuseg_train_labels, cryonuseg_val_labels = train_test_split(
        cryonuseg_raw_paths, cryonuseg_label_paths, test_size=0.2, random_state=42,
    )
    cryonuseg_kwargs = {"patch_shape": patch_shape, "with_channels": True, "ndim": 2, **kwargs}
    for raws, labels, ds_list, n_samples in [
        (cryonuseg_train_raw, cryonuseg_train_labels, train_ds, 50),
        (cryonuseg_val_raw, cryonuseg_val_labels, val_ds, 20),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, raw_key=None, label_paths=labels, label_key=None, is_seg_dataset=False,
                    n_samples=n_samples, **cryonuseg_kwargs,
                ), source_ndim=2,
            )
        )

    # 13. GLySAC (nucleus segmentation in H&E gastric cancer histopathology images)
    # NOTE: Densely annotated (median 124 nuclei per 512x512 area, comparable to MoNuSeg), unlike MoNuSAC.
    # No native val split, so the 34 train tiles are split 80/20. The 25-image test split is kept blind.
    glysac_paths = datasets.glysac.get_glysac_paths(path=os.path.join(input_path, "glysac"), split="train")
    glysac_train, glysac_val = train_test_split(glysac_paths, test_size=0.2, random_state=42)
    glysac_kwargs = {"patch_shape": patch_shape, "with_channels": True, "ndim": 2, **kwargs}
    for paths, ds_list, n_samples in [(glysac_train, train_ds, 200), (glysac_val, val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=paths, raw_key="raw", label_paths=paths, label_key="labels/instances",
                    n_samples=n_samples, **glysac_kwargs,
                ), source_ndim=2,
            )
        )

    # 14. Histo-Miner (nucleus segmentation in H&E cutaneous squamous cell carcinoma)
    # NOTE: Tiles are 256x256, so patches are sampled at 256 and resized/padded to 512 like PanNuke; asking
    # torch_em for 512 directly would give 75 % zero padding. Only train and val are public.
    histo_miner_kwargs = {
        "path": os.path.join(input_path, "histo_miner"), "patch_shape": (256, 256), "download": True,
        "task": "nuclei", "label_choice": "instances",
        **{**kwargs, "transform": _with_flips(_pannuke_random_resize_and_pad_trafo, patch_shape)},
    }
    train_ds.append(
        UniDataWrapper(
            datasets.get_histo_miner_dataset(split="train", n_samples=400, **histo_miner_kwargs), source_ndim=2
        )
    )
    val_ds.append(
        UniDataWrapper(
            datasets.get_histo_miner_dataset(split="val", n_samples=50, **histo_miner_kwargs), source_ndim=2
        )
    )

    # 15. sPATCH (nucleus segmentation in spatial-omics tissue: ovarian cancer, HCC and colon adenocarcinoma)
    # NOTE: The four H&E subsets only; the six DAPI subsets are a light-microscopy evaluation set. No native
    # split, so the 20 tiles are split 80/20 by path.
    spatch_paths = datasets.spatch.get_spatch_paths(
        path=os.path.join(input_path, "spatch"), subset=SPATCH_HE_SUBSETS, download=True,
    )
    spatch_train, spatch_val = train_test_split(spatch_paths, test_size=0.2, random_state=42)
    spatch_kwargs = {"patch_shape": patch_shape, "with_channels": True, "ndim": 2, **kwargs}
    for paths, ds_list, n_samples in [(spatch_train, train_ds, 400), (spatch_val, val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=paths, raw_key="raw/rgb", label_paths=paths, label_key="labels/nuclei",
                    is_seg_dataset=True, n_samples=n_samples, **spatch_kwargs,
                ), source_ndim=2,
            )
        )

    # 16. CoNSeP (nucleus segmentation in H&E colorectal adenocarcinoma, 1000x1000 images)
    # NOTE: CoNSeP is one of Lizard's source cohorts, so it can never serve as a test set; its test split validates.
    consep_kwargs = {
        "path": os.path.join(input_path, "consep"), "patch_shape": patch_shape, "download": True, **kwargs
    }
    for split, ds_list, n_samples in [("train", train_ds, 150), ("test", val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                datasets.get_consep_dataset(split=split, n_samples=n_samples, **consep_kwargs), source_ndim=2
            )
        )

    # 17. DSB histopathology (nucleus segmentation in the 107 H&E images of the Kaggle stage-1 training set)
    # NOTE: A random 10 % (seed 42) validates; there is no blind test. The fluorescence images train with light
    # microscopy, see dsb_fluorescence_training_paths.
    dsb_hp_raw, dsb_hp_labels = datasets.dsb.get_dsb_paths(
        os.path.join(input_path, "dsb"), source="full", domain="histopatho",
    )
    dsb_hp_train_r, dsb_hp_val_r, dsb_hp_train_l, dsb_hp_val_l = train_test_split(
        dsb_hp_raw, dsb_hp_labels, test_size=0.1, random_state=42
    )
    dsb_hp_kwargs = {"patch_shape": patch_shape, "is_seg_dataset": False, "raw_key": None, "label_key": None, **kwargs}
    for raws, labs, ds_list, n_samples in [
        (dsb_hp_train_r, dsb_hp_train_l, train_ds, 100), (dsb_hp_val_r, dsb_hp_val_l, val_ds, 25),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, label_paths=labs, n_samples=n_samples, **dsb_hp_kwargs
                ), source_ndim=2,
            )
        )

    # 18. CellBinDB H&E (nucleus segmentation in H&E tiles of 512x512)
    # NOTE: The H&E tiles of CellBinDB; split 80 / 10 / 10 at random (seed 42), the test tiles are blind.
    cb_he_raw, cb_he_labels = datasets.cellbindb.get_cellbindb_paths(
        path=os.path.join(input_path, "cellbindb"), data_choice=CELLBINDB_HE_STAIN,
    )
    (cb_he_train_r, cb_he_val_r, _), (cb_he_train_l, cb_he_val_l, _) = _train_val_test_split(cb_he_raw, cb_he_labels)
    cb_he_kwargs = {"patch_shape": patch_shape, "is_seg_dataset": False, "ndim": 2, **kwargs}
    for raws, labs, ds_list, n_samples in [
        (cb_he_train_r, cb_he_train_l, train_ds, 150), (cb_he_val_r, cb_he_val_l, val_ds, 30),
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, raw_key=None, label_paths=labs, label_key=None, n_samples=n_samples, **cb_he_kwargs,
                ), source_ndim=2,
            )
        )

    return train_ds, val_ds


# The datasets of the mitochondria EM generalist ('em_mito'). Each name enables one block in
# _get_em_mito_datasets.
EM_MITO_DATASETS = (
    "mitolab", "densecell_mito", "lucchi", "microns_pinky_mito", "mitoem", "mitoemv2", "mitoemv2_mossy", "probtem",
    "uro_cell", "vnc", "mitoverse_xiao18", "mitoverse_haberl18", "mitoverse_microns1",
    "cellmap_mito", "human_organoids_mito", "asem_mito", "brain_pfibsem_mito", "snemi_mito", "u2os_fibsem_mito",
    "hela_stard3_mito", "parlakgul_fasting_liver_mito", "aive_mito", "mito_segem",
)

# Objects below this size (in pixels / voxels of the training patch) are dropped from the mitochondria targets.
EM_MITO_MIN_SIZE = 25
EM_MITO_MIN_SIZE_3D = 50

# The training samples per epoch of the mitochondria datasets (per source; summed over the z groups). A 3D sample is one
# patch and one iteration: 200 x sqrt(training voxels / 250M), clipped to [200, 600], so larger sources get more but
# sublinearly. A 2D sample is one patch: 2800 patches are 200 iterations at the 2D batch of hvit_t (14). The tiny
# sources get about 7 passes per epoch instead, as the small 3D sources: VNC and U2OS (about 64 and 180 patches of
# 512 x 512), haberl18 (20 x 256 x 256, 2 patches) and DenseCell (35 x 800 x 800, 7 patches). CellMap gets more than
# the cap for its 115 crops of many cell types and tissues. CEM-MitoLab draws every training tile once per epoch. The
# brain PFIB-SEM noisy stack repeats the reference stack, so the two share one budget.
EM_MITO_TRAIN_SAMPLES = {
    "densecell_mito": 50, "lucchi": 200, "microns_pinky_mito": 240, "mitoem_human": 600, "mitoem_rat": 600,
    "mitoemv2_beta": 515, "mitoemv2_jurkat": 200, "mitoemv2_macro": 207, "mitoemv2_podo": 207, "mitoemv2_sperm": 254,
    "mitoemv2_stem": 200, "mitoemv2_mossy": 478, "probtem": 2800, "uro_cell": 200, "vnc": 450,
    "mitoverse_xiao18": 200, "mitoverse_haberl18": 14, "mitoverse_microns1": 200, "cellmap_mito": 800,
    "human_organoids_mito": 600, "asem_mito": 225, "asem_mito_small": 50, "brain_pfibsem_reference": 150,
    "brain_pfibsem_noisy": 50, "snemi_mito": 200, "u2os_fibsem_mito": 1260, "hela_stard3_mito": 446,
    "parlakgul_fasting_liver_mito": 205, "aive_mito": 600, "mito_segem": 200,
}
# The CEM-MitoLab validation images per group: one group holds 720 of the 1264 validation tiles otherwise.
MITOLAB_VAL_IMAGES_PER_GROUP = 50

# The sampler counts the connected components of a patch above the min size, without the background (0) and ignore
# labels. A patch needs EM_MITO_MIN_INSTANCES of them; CEM-MitoLab and CellMap need one, since many MitoLab tiles hold
# fewer than three mitochondria (a tile that fails is never drawn) and most 200^3 CellMap crops hold 0-1 per patch.
EM_MITO_MIN_INSTANCES = 3
# The volumes (by file name) where fewer than half of 40 random patches held three mitochondria, with the minimum that
# at least half of them reach (1 if none). The measurement used 14-slice patches.
EM_MITO_SPARSE_VOLUMES = {
    "me2-macro_train01.n5": 1, "me2-jurkat_train01.n5": 2, "pos0.zarr": 2, "fib1-1-0-3.h5": 2,
    "x_0.h5": 2, "x_7.h5": 2, "x_9.h5": 2, "x_13.h5": 2, "x_17.h5": 2,
    "stard3.h5": 1, "1858_mito_80954b8a9027.zarr": 1, "8970_mito_b68e28ee1a3e.zarr": 1,
    "aive_dataset1.h5": 1, "aive_dataset2.h5": 1, "aive_dataset3.h5": 1,
}

# The validation regions of the 3D datasets hold at least 14 slices, the deepest patch of the training configs (z = 14
# for hvit_t and hvit_s); torch-em zero-pads shallower regions in z, which would put blank planes into the patches.

# DenseCell mitochondria use the same split as the DenseCell cell instances in the EM generalist: the train volume
# trains on z < 35 and validates on z >= 35.
DENSECELL_TRAIN_ROI = np.s_[:35, :, :]
DENSECELL_VAL_ROI = np.s_[35:, :, :]

# Lucchi: the train volume trains on z < 140 and validates on z >= 140, the test volume is the evaluation volume.
# The CEM-Mito-Benchmark 'lucchi_pp' is the Lucchi train volume, so it is not evaluated.
LUCCHI_TRAIN_ROI = np.s_[:140, :, :]
LUCCHI_VAL_ROI = np.s_[140:, :, :]

# MICrONS pinky: only the stitched volume vol19-vol34 is densely annotated for mitochondria; the '_realigned' copy is
# the same annotation re-registered, and the other pinky volumes label the mitochondria of a few cells only.
MICRONS_PINKY_MITO_FILE = "pinky_stitched_vol19-vol34.h5"
MICRONS_PINKY_MITO_TRAIN_ROI = np.s_[:86, :, :]
MICRONS_PINKY_MITO_VAL_ROI = np.s_[86:, :, :]

# MitoEM: only the train volumes of both samples train (z < 360) and validate (z >= 360). The val volumes are the
# evaluation volumes, and the test volumes have no labels.
MITOEM_SAMPLES = ("human", "rat")
MITOEM_TRAIN_ROI = np.s_[:360, :, :]
MITOEM_VAL_ROI = np.s_[360:, :, :]

# MitoEM v2: the official train volumes train and the official val volumes validate (the converted n5 copies in
# <data>/mitoemv2/n5_data); the test volumes are the evaluation volumes. The value is the z / xy anisotropy. pyra is
# left out: its crops are regions of MitoEM-H, which trains and evaluates above. The jurkat val volume
# (me2-jurkat_train02) has labels that do not match its raw data, so jurkat validates on the last slices of its train
# volume instead (MITOEMV2_JURKAT_*_ROI).
MITOEMV2_DATASETS = {"beta": 1, "jurkat": 1, "macro": 1, "podo": 1, "sperm": 1, "stem": 30 / 8}
MITOEMV2_JURKAT_TRAIN_ROI = np.s_[:230, :, :]
MITOEMV2_JURKAT_VAL_ROI = np.s_[230:, :, :]

# MitoEM v2 mossy: only the mitochondria inside the mossy fiber boutons are annotated. The bouton mask comes with the
# MitoVerse copies of these volumes (<data>/mitoverse/data/han24), whose labels are identical to the MitoEM v2 ones:
# the MitoEM v2 samples are mapped to the MitoVerse volumes below. Outside the boutons the loss is ignored.
MITOEMV2_MOSSY_TRAIN_VOLUMES = {"me2-mossy_train01": "neg0", "me2-mossy_train02": "pos0"}
MITOEMV2_MOSSY_VAL_VOLUMES = {"me2-mossy_train03": "pos2"}
MITOEMV2_MOSSY_IGNORE_LABEL = np.iinfo("uint32").max

# ProbTEM: the processed h5 files in <data>/probtem_full/processed (read directly: the <data>/probtem copy is
# incomplete). A seeded 3 of the 21 train images validate (the evaluation tunes on them as well); the 6 test images
# are the evaluation images.
PROBTEM_ROOT = os.path.join("probtem_full", "processed")
PROBTEM_N_VAL = 3

# UroCell: the 5 volumes (256^3, 16 nm isotropic) with mitochondria labels; each trains on z < 224 and validates on
# the rest. The other 3 volumes have no mitochondria labels.
UROCELL_MITO_VOLUMES = ["fib1-0-0-0.h5", "fib1-1-0-3.h5", "fib1-3-2-1.h5", "fib1-3-3-0.h5", "fib1-4-3-0.h5"]
UROCELL_TRAIN_ROI = np.s_[:224, :, :]
UROCELL_VAL_ROI = np.s_[224:, :, :]

# VNC: only the train stack (20 slices) is labeled, the test stack has no labels. z < 16 trains and z >= 16 validates;
# the 4-slice validation slab does not fit a 3d patch, so VNC trains in the 2d group on single slices.
VNC_TRAIN_ROI = np.s_[:16, :, :]
VNC_VAL_ROI = np.s_[16:, :, :]

# Parlakgul liver: the 64-slice boxes of the four samples (6461 lean, 6464 / 9430 obese, 1857 obese Climp63), cached by
# torch-em's parlakgul_liver loader. The first one is in the original data folder, the others in parlakgul_liver_boxes
# (their bounding boxes are listed in training_boxes.json there). Each box trains on z < 50 and validates on the rest.
# NOTE: Parlakgul is an evaluation dataset now (it is not in EM_MITO_DATASETS): its semantic masks merge touching
# mitochondria and hold irregular fragments. The evaluation reads PARLAKGUL_LIVER_BOXES.
PARLAKGUL_LIVER_BOXES = [
    os.path.join("parlakgul_liver", "6461_mito_c2fda2f2e9f2.zarr"),
    os.path.join("parlakgul_liver_boxes", "6461_mito_6437b92288e2.zarr"),
    os.path.join("parlakgul_liver_boxes", "6464_mito_f42d478376ae.zarr"),
    os.path.join("parlakgul_liver_boxes", "6464_mito_84cb0665a8d2.zarr"),
    os.path.join("parlakgul_liver_boxes", "9430_mito_d4c0a73009c8.zarr"),
    os.path.join("parlakgul_liver_boxes", "9430_mito_c5f9fef6fb64.zarr"),
    os.path.join("parlakgul_liver_boxes", "1857_mito_0a7959a670ff.zarr"),
    os.path.join("parlakgul_liver_boxes", "1857_mito_f95eb8970213.zarr"),
]
PARLAKGUL_LIVER_TRAIN_ROI = np.s_[:50, :, :]
PARLAKGUL_LIVER_VAL_ROI = np.s_[50:, :, :]

# MitoVerse (<data>/mitoverse/data/<dataset>/<volume>.zarr, arrays 'img' and 'mito'): only the parts that duplicate
# no other training or evaluation data are used; the full collection repeats MitoEM, the CEM benchmark, Kasthuri and
# Lucchi, DenseCell, MitoEM v2, CellMap and others. xiao18 (mouse auditory cortex ssSEM, 8 x 8 x 50 nm) trains on
# z >= 15 and validates on z < 14: section z = 14 is torn (a third of it is blank resin without labels), so no patch
# crosses it. haberl18 (CDeep3M, mouse brain SBF-SEM, 10 x 10 x 24 nm) trains on vol0 and does not validate, since
# vol1 has only 10 slices. The five MICrONS phase 1 volumes of turner22 (ssTEM, 8 x 8 x 40 nm) train on z < 36 and
# validate on the remaining 14 slices, except vol1 and vol2: their last 14 slices hold a missing section (constant
# raw), so every validation patch would be rejected by the blank slice sampler. They train whole.
MITOVERSE_XIAO18_TRAIN_ROI = np.s_[15:, :, :]
MITOVERSE_XIAO18_VAL_ROI = np.s_[:14, :, :]
MITOVERSE_MICRONS1_VOLUMES = [f"microns1_vol{i}" for i in range(5)]
MITOVERSE_MICRONS1_TRAIN_ONLY_VOLUMES = ("microns1_vol1", "microns1_vol2")
MITOVERSE_MICRONS1_TRAIN_ROI = np.s_[:36, :, :]
MITOVERSE_MICRONS1_VAL_ROI = np.s_[36:, :, :]
# Some MICrONS phase 1 slices miss image data (raw == 0) while the labels continue there. Connected raw == 0 regions
# of a slice above this area (in pixels) carry MITOVERSE_IGNORE_LABEL, so the loss ignores them but not single dark
# membrane pixels.
MITOVERSE_MISSING_DATA_MIN_AREA = 2000
MITOVERSE_IGNORE_LABEL = np.iinfo("uint32").max

# Human organoids (<data>/human_organoids/11380/data, FIB-SEM, 2x binned, voxel size undocumented): the mitochondria
# labels of the four volumes, the nuclei and other organelles are background. The raw is int8. Each volume trains on its
# first 90% of z and validates on the rest.
HUMAN_ORGANOIDS_VOLUMES = ["F059_bin2.h5", "F107_A1_bin2.h5", "F107_A2_bin2.h5", "F107_A3_bin2.h5"]
HUMAN_ORGANOIDS_VAL_FRACTION = 0.1
# The mitochondria instance labels of the four volumes (<data>/human_organoids_mito/<volume>_mito_labels.zarr, written
# by finetuning/v2/generalist/em_organelles/preprocess_mito_dataset.py): the connected components of the manual semantic
# masks, split by a neck watershed where touching mitochondria form one component (about 4-12% of the components).
HUMAN_ORGANOIDS_MITO_ROOT = "human_organoids_mito"


# ASEM mitochondria (<data>/asem/mito_crops, zarr crops with 'raw' and semantic 'labels', manual and dense in their
# crop, FIB-SEM): volumes 46 and 58 (5 nm) train on z < 224 and validate on the rest, the small volume 2E (4 nm,
# 200 px in plane) trains whole as 192 x 192 patches resized to 512 x 512. Volume 64 (cell_6, the micro-sam v1 EM test
# volume) and the 61 hold-out block are the in-domain evaluation volumes.
ASEM_MITO_ROOT = os.path.join("asem", "mito_crops")
ASEM_MITO_TRAIN_VOLUMES = ("46_z640-896_y0-1024_x1112-2136.zarr", "58_z152-408_y0-1000_x1480-2504.zarr")
ASEM_MITO_SMALL_VOLUMES = ("2E_z1015-1215_y316-516_x294-494.zarr",)
ASEM_MITO_TRAIN_ROI = np.s_[:224, :, :]
ASEM_MITO_VAL_ROI = np.s_[224:, :, :]
ASEM_MITO_SMALL_SIDE = 192

# Brain PFIB-SEM mitochondria (<data>/brain_pfibsem/brain_pfibsem.h5, mouse brain PFIB-SEM, 5 nm isotropic, 181 x 1300
# x 1300, expert binary masks made on the 2 us reference stack): the noisy 0.5 us stack is registered to the reference
# stack (correlation 0.99 of the smoothed slices without any shift), so it trains with the same labels as a noise
# augmentation. z < 144 trains and the rest (37 slices) validates, as in the torch-em loader.
BRAIN_PFIBSEM_PATH = os.path.join("brain_pfibsem", "brain_pfibsem.h5")
BRAIN_PFIBSEM_TRAIN_ROI = np.s_[:144, :, :]
BRAIN_PFIBSEM_VAL_ROI = np.s_[144:, :, :]

# SNEMI Mito (<data>/snemi_mito_stack/snemi_mito.h5, written by build_snemi_mito_stack.py in mito_nuc_figures/scripts;
# ssSEM of the mouse neocortex, SNEMI3D, 6 x 6 x 30 nm, manual binary masks): the 100 consecutive slices are stacked
# and the official contiguous split is kept: z 0-49 trains, z 50-69 validates, z 70-99 (test) is not used.
SNEMI_MITO_PATH = os.path.join("snemi_mito_stack", "snemi_mito.h5")
SNEMI_MITO_TRAIN_ROI = np.s_[:50, :, :]
SNEMI_MITO_VAL_ROI = np.s_[50:70, :, :]

# U2OS FIB-SEM mitochondria (<data>/u2os_fibsem/1_Mitochondria/1_Train_Mito, FIB-SEM of one U2-OS cell, 45 manually
# annotated single slices of 692 x 1697, binary masks): trained as 2D; every 9th slice validates (5 slices, which are
# 30-250 sections away from their neighbours in the stack).
U2OS_FIBSEM_MITO_ROOT = os.path.join("u2os_fibsem", "1_Mitochondria", "1_Train_Mito")
U2OS_FIBSEM_MITO_VAL_STEP = 9

# HeLa STARD3 mitochondria (<data>/hela_stard3/<volume>.h5, FIB-SEM of HeLa cells, EMPIAR-13156, 8 nm pixels for 'ctrl'
# and 'stard3_fa_ya', 5 nm for 'stard3', treated as isotropic): the binary mitochondria masks of the densely annotated
# ROI of each volume (the torch-em loader crops raw and labels to it). The other two volumes train whole, 'ctrl' trains
# on z < 404 and validates on the last 40 slices.
HELA_STARD3_VOLUMES = ("ctrl", "stard3", "stard3_fa_ya")
HELA_STARD3_VAL_VOLUME = "ctrl"
HELA_STARD3_VAL_START = 404

# Parlakgul fasting liver mitochondria (<data>/parlakgul_fasting_liver/<sample>_mito_<hash>.zarr, FIB-SEM of the mouse
# liver, EMPIAR-12017, 8 nm isotropic, the authors' mitochondria instance segmentation): 64 x 1024 x 1024 tissue
# boxes (9428 lean fasted, 8970 obese fasted, 2679 obese LacZ, 1858 obese RRBP1; 8970 has two boxes that do not
# overlap, the central one and the torch-em default one), cached by torch-em's parlakgul_fasting_liver loader. These
# samples differ from the Parlakgul liver samples of the out-of-domain
# evaluation (parlakgul_liver_eval: 6461, 6464, 9430, 1857). Each box trains on z < 50 and validates on the rest.
PARLAKGUL_FASTING_ROOT = "parlakgul_fasting_liver"
PARLAKGUL_FASTING_TRAIN_ROI = np.s_[:50, :, :]
PARLAKGUL_FASTING_VAL_ROI = np.s_[50:, :, :]

# AIVE mitochondria (<data>/aive/aive_dataset<i>.h5, FIB-SEM of cells and muscle tissue, EMPIAR-12885, 6.5 x 6.5 x
# 10 nm): the mitochondria class (1) of the harmonized semantic organelle labels of torch-em's aive loader. Not every
# mitochondrion is annotated in every volume. Each volume trains on its first ~90% of z and validates on the rest, per
# volume file name the first validation slice.
AIVE_MITO_CLASS = 1
AIVE_MITO_VAL_STARTS = {
    "aive_dataset1.h5": 652, "aive_dataset2.h5": 772, "aive_dataset3.h5": 556, "aive_dataset4.h5": 280,
}


# MitoSegEM mitochondria (<data>/mito_segem, EMPIAR-12535, noisy SBF-SEM, 15 nm pixels, 50 nm sections): one cached
# 64 x 1024 x 1024 box (z, y, x in the full volume) per tissue, network predictions with a manual instance assignment.
# The testis and brainstem boxes train on z < 50 and validate on the rest; the intestine box is an out-of-domain
# evaluation set (finetuning/v2/evaluation/common.py).
MITO_SEGEM_TRAIN_BOXES = {
    "testis": (1268, 1332, 1792, 2816, 7680, 8704),
    "brainstem": (1868, 1932, 7936, 8960, 7168, 8192),
}
MITO_SEGEM_TRAIN_ROI = np.s_[:50, :, :]
MITO_SEGEM_VAL_ROI = np.s_[50:, :, :]


# CellMap mitochondria (<data>/cellmap_mito_crops, one zarr store with 'raw' and 'labels' per challenge crop with a
# mito label array, listed in its index.json): without sum159-4 crop211, whose mito array is empty although
# mitochondria are visible. No crop overlaps the MitoEM v2 jurkat / macro / podo volumes (crops of jrc_jurkat-1,
# jrc_macrophage-2 and jrc_mus-kidney), which a template matching of all their crops against these volumes showed.
# Only crops with labeled mitochondria train, since an empty crop only yields batches without objects, which the
# joint trainer skips. The crops are grouped by in-plane size into the patch sides of CELLMAP_MITO_PATCH_SIDES,
# crops below the smallest side are not used, and whole volumes (cells) validate, because the crops of one volume
# can be near-duplicates or nested.
CELLMAP_MITO_PATCH_SIDES = (128, 192, 256, 384, 512)
CELLMAP_MITO_VAL_FRACTION = 0.1


def probtem_split_paths(input_path, split):
    """The ProbTEM train images of the 'train' or 'val' part."""
    paths = sorted(glob(os.path.join(input_path, PROBTEM_ROOT, "train", "*.h5")))
    train_paths, val_paths = train_test_split(paths, test_size=PROBTEM_N_VAL, random_state=42)
    return sorted(train_paths if split == "train" else val_paths)


def cellmap_mito_split_crops(input_path, split):
    """The CellMap mito crops of the 'train' or 'val' volumes, as {patch side: crop paths}."""
    root = os.path.join(input_path, "cellmap_mito_crops")
    with open(os.path.join(root, "index.json")) as f:
        crops = [
            crop for crop in json.load(f) if crop["n_ids"] > 0 and min(crop["shape"][1:]) >= CELLMAP_MITO_PATCH_SIDES[0]
        ]
    volumes = sorted({crop["volume"] for crop in crops})
    train_volumes, val_volumes = train_test_split(volumes, test_size=CELLMAP_MITO_VAL_FRACTION, random_state=42)
    keep = set(train_volumes if split == "train" else val_volumes)
    groups = {}
    for crop in crops:
        if crop["volume"] in keep:
            side = max(s for s in CELLMAP_MITO_PATCH_SIDES if s <= min(crop["shape"][1:]))
            groups.setdefault(side, []).append(os.path.join(root, f"{crop['name']}.zarr"))
    return {side: sorted(paths) for side, paths in sorted(groups.items())}


# The datasets of the nucleus EM generalist ('em_nucleus'). Each name enables one block in _get_em_nucleus_datasets.
EM_NUCLEUS_DATASETS = (
    "betaseg_nucleus", "cellmap_nucleus", "human_organoids_nucleus", "sxt_nucleus", "fafb_nuclei", "janelia_nucleus",
    "microns_nuclei", "waenet_fibsem_nucleus", "waenet_sstem_nucleus", "waenet_tem_nucleus", "platynereis_nuclei",
    "tumor_spheroid_nuclei", "openorganelle_nucleus", "cardiomyocyte_tem_nucleus", "microsam_em_nuclei",
    "perez_nucleus", "vem_nuclei", "aive_nucleus", "cerebellum_nucleus", "macrophage_a431_nucleus",
)

# Objects below this size (in voxels of the training patch) are dropped from the nucleus targets.
EM_NUCLEUS_MIN_SIZE_3D = 50

# The training samples per epoch of the nucleus datasets (per source; summed over the z groups), by the rules of the
# mitochondria generalist (EM_MITO_TRAIN_SAMPLES): 3D sources 200 x sqrt(training voxels / 250M) clipped to [200, 600],
# 2D sources 2800 patches or about 7 passes over their patches, Cerebellum about 7 passes and the microSAM train crop
# 200. On top, the large and diverse sources get more: OpenOrganelle 1000, CellMap, human organoids, SXT and Janelia
# 800, BetaSeg 400 and AIVE 300 (2026-10-19). Every volume gets at least one sample.
EM_NUCLEUS_TRAIN_SAMPLES = {
    "betaseg_nucleus": 400, "cellmap_nucleus": 800, "human_organoids_nucleus": 800, "sxt_nucleus": 800,
    "fafb_nuclei": 293, "janelia_nucleus": 800, "microns_nuclei": 414, "waenet_fibsem_nucleus": 200,
    "waenet_sstem_nucleus": 510, "waenet_tem_nucleus": 2500, "platynereis_nuclei": 200, "tumor_spheroid_nuclei": 1400,
    "openorganelle_nucleus": 1000, "cardiomyocyte_tem_nucleus": 575, "microsam_em_nuclei": 200, "perez_nucleus": 350,
    "vem_nuclei_2d": 2800, "vem_nuclei_3d": 200, "aive_nucleus": 300, "cerebellum_nucleus": 175,
    "macrophage_a431_nucleus": 424,
}
# The nuclei a patch needs per dataset: at least half of 40 random patches of the dataset (pooled over its volumes,
# connected components, 14-slice patches, 2026-10-19) held this many; the other datasets need one.
EM_NUCLEUS_MIN_INSTANCES = {
    "fafb_nuclei": 3, "microns_nuclei": 3, "platynereis_nuclei": 3, "microsam_em_nuclei": 3, "janelia_nucleus": 3,
    "tumor_spheroid_nuclei": 2, "human_organoids_nucleus": 2, "cellmap_nucleus": 2, "vem_nuclei": 2,
    "openorganelle_nucleus": 2,
}
# The volumes (by their last two path components) where none of the 40 patches held that many nuclei, so their
# sampler would never accept a patch: they need one.
EM_NUCLEUS_SPARSE_VOLUMES = {
    "cellmap_nucleus_crops/jrc_mus-liver-zon-2_crop367.zarr": 1, "janelia_nucleus/jrc_mus-nacc-2.zarr": 1,
    "janelia_nucleus/jrc_mus-nacc-4.zarr": 1, "openorganelle_nucleus_full/jrc_cos7-11.zarr": 1,
    "openorganelle_nucleus_full/jrc_hela-21.zarr": 1, "openorganelle_nucleus_full/aic_desmosome-3.zarr": 1,
    "vem_nuclei_nucleus/id_12.h5": 1,
}

# Nucleus volumes are drawn at the largest of these patch sides that fits in-plane; patches below 512 px are resized
# to 512 x 512.
EM_NUCLEUS_PATCH_SIDES = (128, 192, 256, 320, 384, 512)

# BetaSeg nuclei (<data>/betaseg_nucleus_boxes, 4 nm isotropic FIB-SEM of mouse beta cells): one box per volume
# around the labeled nucleus of the target cell with a margin of 32 voxels. Only the nucleus of the target cell is
# labeled, and unlabeled neighbouring nuclei enter some boxes (high_c4, low_c2), so the labels carry
# BETASEG_NUCLEUS_IGNORE_LABEL outside the target cell mask and on blank raw, and the loss ignores those voxels.
BETASEG_NUCLEUS_TRAIN_VOLUMES = ("high_c1", "high_c2", "high_c3", "high_c4", "low_c1", "low_c2")
BETASEG_NUCLEUS_VAL_VOLUMES = ("low_c3",)
BETASEG_NUCLEUS_IGNORE_LABEL = np.iinfo("uint32").max

# CellMap nuclei (<data>/cellmap_nucleus_crops, one zarr store with 'raw' and 'labels' per crop, listed in its
# index.json): only the large tissue crops at 32 and 64 nm with many complete nuclei, nucleus labels only. The small
# high-resolution crops are not used: most hold one nucleus cut by the crop borders, and 15 lie fully inside a
# nucleus. jrc_mus-liver-zon-2 crop357 (the 'test' crop of the index) is the in-domain evaluation crop, since it
# overlaps no other crop (crop358 and crop367 overlap each other).
CELLMAP_NUCLEUS_VAL_CROPS = ("jrc_zf-cardiac-1_crop379",)

# Human organoids nuclei: the raw of the four volumes (HUMAN_ORGANOIDS_VOLUMES) with cleaned nucleus instance labels
# (<data>/human_organoids_nucleus/<volume>_nucleus_labels.zarr, written by build_human_organoids_nucleus_labels.py in
# mito_nuc_figures/scripts): the speckled manual semantic masks are closed and their holes filled per slice, the
# labels on the black alignment padding are removed, the nuclei are separated by 3D connected components, and label
# fragments below 100k voxels and the entotic cell of F107_A1 (whose nucleus is not labeled) carry
# HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL.
# The z split is the one of the mitochondria block (HUMAN_ORGANOIDS_VAL_FRACTION).
HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL = np.iinfo("uint32").max

# SXT INS-1E nuclei (<data>/sxt_ins1e_mito, one zarr store per cell): soft X-ray tomography (not EM) of INS-1E cells in
# glass capillaries, isotropic 36.8-45.5 nm, semi-automatic semantic labels (0 exterior, 1 cell, 2 nucleus, 5 mito).
# Only the target cell is labeled and some tomograms may hold unlabeled neighbour cells, so the nucleus is the object,
# the rest of the cell is background, and the loss ignores the exterior (SXT_NUCLEUS_IGNORE_LABEL). The ignore label
# is the uint16 maximum, since torch-em casts the labels back to their stored dtype (uint16) after the augmentations.
# A seeded set of SXT_NUCLEUS_N_VAL cells validates.
SXT_NUCLEUS_CLASS = 2
SXT_EXTERIOR_CLASS = 0
SXT_NUCLEUS_IGNORE_LABEL = np.iinfo("uint16").max
SXT_NUCLEUS_N_VAL = 5

# FAFB nuclei (<data>/fafb_nuclei/{train,val,test}, 256 x 512 x 512 blocks of ssTEM of the adult fly brain at
# 32 x 32 x 40 nm): the automatic FlyWire nucleus segmentation (v5), which is clean and near-exhaustive, so it trains
# despite being automatic. The train blocks train, the val blocks validate, and the test blocks are the in-domain
# evaluation blocks.
FAFB_NUCLEI_Z_SCALE = 40 / 32

# Janelia nuclei (<data>/janelia_nucleus, one zarr store with 'raw' and 'labels' per OpenOrganelle volume, 32-128 nm):
# 13 volumes with Cellpose pseudo-labels (near-exhaustive, a few small or very dark nuclei missed) and 3 manual
# jrc_mus-nacc volumes (1-3 nuclei each). The voxels are treated as isotropic (z / xy is 0.74-1.34). jrc_mus-kidney is
# the volume of the CellMap kidney crop179, which trains as well. jrc_mus-granule-neurons-1 is not used, since it is
# only 85 px wide in y. JANELIA_NUCLEUS_VAL_VOLUMES validate.
JANELIA_NUCLEUS_EXCLUDED_VOLUMES = ("jrc_mus-granule-neurons-1",)
JANELIA_NUCLEUS_VAL_VOLUMES = ("jrc_mus-meissner-corpuscle-2",)

# MICrONS nuclei (<data>/microns_nuclei/{train,val,test}, 512^3 blocks of ssTEM of the mouse visual cortex at
# 64 x 64 x 40 nm): the automatic minnie65 nucleus segmentation, clean and near-exhaustive, so it trains despite being
# automatic. As for FAFB, the train blocks train, the val blocks validate, and the test blocks are the in-domain
# evaluation blocks.
MICRONS_NUCLEI_Z_SCALE = 40 / 64

# Platynereis nuclei (<data>/platynereis/nuclei/train_data_nuclei_<id>.h5, SBEM of a 6 dpf larva at about
# 80 x 80 x 100 nm, manual instance labels, exhaustive inside the annotated box). Each volume carries a halo of
# unlabeled voxels (PLATYNEREIS_NUCLEI_IGNORE_LABEL) around the annotated box, so only the valid ROI of each volume
# (_compute_platy_rois) trains. Volume 03 (dense brain region, 108 nuclei) is the in-domain evaluation volume and
# volume 12 validates.
PLATYNEREIS_NUCLEI_TRAIN_IDS = (1, 2, 4, 5, 6, 7, 8, 9, 10, 11)
PLATYNEREIS_NUCLEI_VAL_IDS = (12,)
PLATYNEREIS_NUCLEI_TEST_IDS = (3,)
PLATYNEREIS_NUCLEI_IGNORE_LABEL = -1
PLATYNEREIS_NUCLEI_Z_SCALE = 100 / 80

# Tumor spheroid nuclei (<data>/tumor_spheroid_nucleus, derived from <data>/tumor_spheroid_em/2d_manual/50-50-50 by
# build_tumor_spheroid_nucleus_labels.py in mito_nuc_figures/scripts; SBF-SEM of a tumor spheroid with gold
# nanoparticles, inverted contrast): 20 manually annotated single slices in xy, xz and yz (the 100 nm slices are the
# same slices downsampled). Many annotated cells carry no nucleus label, some of them with nucleus-like content, so
# every cell without a nucleus label carries TUMOR_SPHEROID_NUCLEI_IGNORE_LABEL and the loss ignores it. Some slices
# are only 2-7 sections apart, so the validation slices are ones without a close neighbour.
TUMOR_SPHEROID_NUCLEI_VAL_SLICES = ("Au_01-vol_01-x_1300.h5", "Au_01-vol_01-z_0274.h5")
TUMOR_SPHEROID_NUCLEI_IGNORE_LABEL = np.iinfo("uint32").max

# The new torch-em nucleus datasets of 2026-10-18. The derived data of all of them except microsam_em_nuclei is written
# by build_new_nucleus_data.py in mito_nuc_figures/scripts. The large images and volumes are downscaled there so that a
# nucleus is a few hundred px wide, i.e. a 512 px patch shows nucleus boundaries and context.
# Cardiomyocyte TEM (<data>/cardiomyocyte_tem_nucleus): 9 TEM micrographs of isolated cardiomyocytes, one nucleus
# each (manual masks, the nucleoli are part of the nucleus), downscaled 4x; a seeded pair of images validates.
CARDIOMYOCYTE_NUCLEUS_ROOT = "cardiomyocyte_tem_nucleus"
CARDIOMYOCYTE_NUCLEUS_N_VAL = 2
# microSAM EM nuclei (<data>/microsam_em_nuclei/data): the 'train' crop (32 x 512 x 512) of the microSAM user study,
# the only crop with manual reference labels (48 nuclei); the 'annotation' crop only holds user-study outputs, which
# differ and miss nuclei, so it is not used. The whole train crop trains and there is no validation leaf.
MICROSAM_EM_NUCLEI_PATH = os.path.join("microsam_em_nuclei", "data", "train_data.h5")
# Perez nuclei (<data>/perez_nucleus): the nucleus task tiles of the Perez SBF-SEM data (mouse suprachiasmatic nucleus);
# the 500 px train tiles (resized to 512 px) train and the 1000 px test tiles validate.
PEREZ_NUCLEUS_ROOT = "perez_nucleus"
# vEM nuclei (<data>/vem_nuclei_nucleus, S-BIAD2822): 2D array tomography SEM images of cancer cells and tumor tissue
# (NucleoNet predictions, manually proofread), downscaled so that nuclei are about 250 px wide, with the unlabeled
# border band ignored where the border nuclei were removed (VEM_NUCLEI_IGNORE_LABEL), and two 3D FIB-SEM volumes with
# one nucleus each, downscaled 2x. A seeded 2D image and the last VEM_NUCLEI_VAL_SLICES slices of volume id_12
# validate; the nucleus of id_12 ends at z 399 of 463, so the slab holds its last 37 slices.
VEM_NUCLEI_ROOT = "vem_nuclei_nucleus"
VEM_NUCLEI_3D_IDS = (11, 12)
VEM_NUCLEI_VAL_SLICES = 100
VEM_NUCLEI_IGNORE_LABEL = np.iinfo("uint32").max
# AIVE nuclei (<data>/aive_nucleus, EMPIAR-12885): the nucleus of three FIB-SEM overview volumes (nucleus class with
# the nucleoli, labeled 'chromatin', folded in), downscaled to 30 x 26 x 26 nm; the last AIVE_NUCLEUS_VAL_SLICES slices
# of dataset 3 validate.
AIVE_NUCLEUS_ROOT = "aive_nucleus"
AIVE_NUCLEUS_Z_SCALE = 30 / 26
AIVE_NUCLEUS_VAL_SLICES = 40
# Cerebellum nuclei (<data>/cerebellum_fibsem_nucleus): mouse cerebellum FIB-SEM, nuclei labeled by students (not
# necessarily proofread), downscaled 2x to 20 nm. Two large nuclei (from z = 0); the loss ignores three small labeled
# pieces of a dark object inside a myelinated axon near the end of the volume (CEREBELLUM_NUCLEUS_IGNORE_LABEL).
# z >= CEREBELLUM_NUCLEUS_VAL_Z validates, which holds the last 22 slices of one large nucleus.
CEREBELLUM_NUCLEUS_PATH = os.path.join("cerebellum_fibsem_nucleus", "cerebellum_fibsem.h5")
CEREBELLUM_NUCLEUS_VAL_Z = 200
CEREBELLUM_NUCLEUS_IGNORE_LABEL = np.iinfo("uint32").max
# Macrophage / A431 nuclei (<data>/macrophage_a431_nucleus, EMPIAR-13420): SBF-SEM volumes of macrophages and an A431
# cell with one nucleus each (three in sbf_no_load; quick semantic MIB annotations), downscaled 4x in xy. The voxel
# size is not stored; the nucleus extents suggest a z / xy anisotropy of about 2.5 after downscaling. The slices from
# MACROPHAGE_NUCLEUS_VAL_Z of sbf_a431 validate; its nucleus ends at z 134 of 162, so the slab holds its last 20 slices.
MACROPHAGE_NUCLEUS_ROOT = "macrophage_a431_nucleus"
MACROPHAGE_NUCLEUS_Z_SCALE = 2.5
MACROPHAGE_NUCLEUS_VAL_VOLUME = "sbf_a431"
MACROPHAGE_NUCLEUS_VAL_Z = 115

# WAE-Net nuclei (<data>/waenet_nucleus, written by build_waenet_nucleus.py in mito_nuc_figures/scripts). Dataset 1 is
# a FIB-SEM stack of pancreatic carcinoid cells with all nuclei labeled (dataset_1.h5, the last WAENET_VAL_FRACTION of
# z validates; one alignment seam between slices 180 and 181). Dataset 3 holds unregistered serial ssTEM sections of
# one cell, trained as single images (dataset_3_{train,val}.h5, a contiguous block of sections validates). Datasets
# 4-7 are single TEM images of lymphocytes and B cells in which only the central cell is labeled
# (tem_{train,val}.h5): the loss is restricted to the central cell mask (cytoplasm and nucleus labels) dilated by
# 10 px, all other pixels, including the unlabeled neighbour cells, carry WAENET_NUCLEUS_IGNORE_LABEL. Dataset 2 has no
# nucleus labels. The 2D images are stored as stacks of 512 x 512 images and drawn as (1, 512, 512) patches.
WAENET_VAL_FRACTION = 0.1

# OpenOrganelle nuclei (<data>/openorganelle_nucleus_full/index.json, written by build_openorganelle_nucleus_full.py in
# mito_nuc_figures/scripts): automatic nucleus segmentations of OpenOrganelle FIB-SEM volumes, isotropic 32-128 nm.
# The near-exhaustive tissue volumes as full arrays (the torch-em loader caches only one 128^3 crop of several of
# them) and the single-cell volumes (1-3 nuclei, all labeled) that are at least 128 px wide in-plane; the binary
# single-cell labels are separated by connected components. The volumes with envelope or membrane labels, thin sperm
# nuclei, ragged TEM tiles or a single nucleus in a huge volume are not used. The 'val' volume of the index validates.
WAENET_NUCLEUS_IGNORE_LABEL = np.iinfo("uint32").max


def em_nucleus_patch_side(shape):
    """The largest patch side of EM_NUCLEUS_PATCH_SIDES that fits in-plane into a volume of *shape*."""
    return max(side for side in EM_NUCLEUS_PATCH_SIDES if side <= min(shape[-2:]))


def betaseg_nucleus_split_boxes(input_path, split):
    """The BetaSeg nucleus boxes of the 'train' or 'val' volumes, as {patch side: box paths}."""
    groups = {}
    for volume in BETASEG_NUCLEUS_TRAIN_VOLUMES if split == "train" else BETASEG_NUCLEUS_VAL_VOLUMES:
        path = os.path.join(input_path, "betaseg_nucleus_boxes", f"{volume}.h5")
        with open_file(path, "r") as f:
            groups.setdefault(em_nucleus_patch_side(f["raw"].shape), []).append(path)
    return {side: sorted(paths) for side, paths in sorted(groups.items())}


def sxt_nucleus_labels(labels):
    """The nucleus mask of the SXT INS-1E class labels, with SXT_NUCLEUS_IGNORE_LABEL outside the cell."""
    labels = np.asarray(labels)
    nucleus = (labels == SXT_NUCLEUS_CLASS).astype("uint16")
    nucleus[labels == SXT_EXTERIOR_CLASS] = SXT_NUCLEUS_IGNORE_LABEL
    return nucleus


def sxt_nucleus_instances(labels):
    """The connected components of the SXT INS-1E nucleus mask, as the pre_label_transform (before the sampler)."""
    return _connected_components_pre_label_trafo(sxt_nucleus_labels(labels), ignore_label=SXT_NUCLEUS_IGNORE_LABEL)


def sxt_nucleus_split_paths(input_path, split):
    """The SXT INS-1E cells of the 'train' or 'val' part, as {patch side: zarr paths}."""
    paths = sorted(glob(os.path.join(input_path, "sxt_ins1e_mito", "*.zarr")))
    train_paths, val_paths = train_test_split(paths, test_size=SXT_NUCLEUS_N_VAL, random_state=42)
    groups = {}
    for path in train_paths if split == "train" else val_paths:
        with open_file(path, "r") as f:
            groups.setdefault(em_nucleus_patch_side(f["labels"].shape), []).append(path)
    return {side: sorted(paths) for side, paths in sorted(groups.items())}


def platynereis_nuclei_split_rois(input_path, split):
    """The Platynereis nucleus volumes of the 'train' or 'val' part, as {patch side: (paths, valid rois)}."""
    root = os.path.join(input_path, "platynereis")
    template = "nuclei/train_data_nuclei_%02i.h5"
    sample_ids = PLATYNEREIS_NUCLEI_TRAIN_IDS if split == "train" else PLATYNEREIS_NUCLEI_VAL_IDS
    valid_rois = _compute_platy_rois(
        root, sample_ids, ignore_label=PLATYNEREIS_NUCLEI_IGNORE_LABEL, file_template=template,
        label_key="volumes/labels/nucleus_instance_labels", cache_prefix="_nuclei_roi_cache",
    )
    groups = {}
    for sample_id in sample_ids:
        roi = valid_rois[sample_id]
        side = em_nucleus_patch_side((roi[1].stop - roi[1].start, roi[2].stop - roi[2].start))
        paths, rois = groups.setdefault(side, ([], []))
        paths.append(os.path.join(root, template % sample_id))
        rois.append(roi)
    return dict(sorted(groups.items()))


def janelia_nucleus_split_paths(input_path, split):
    """The Janelia nucleus volumes of the split, as {patch side: zarr paths}, and their voxel counts.

    The voxel counts weight the samples, so that the large tissue volumes are not outweighed by the small volumes with
    one to three nuclei.
    """
    groups, voxels = {}, {}
    for path in sorted(glob(os.path.join(input_path, "janelia_nucleus", "*.zarr"))):
        name = os.path.basename(path)[:-len(".zarr")]
        if name in JANELIA_NUCLEUS_EXCLUDED_VOLUMES or (name in JANELIA_NUCLEUS_VAL_VOLUMES) != (split == "val"):
            continue
        with open_file(path, "r") as f:
            shape = f["labels"].shape
        groups.setdefault(em_nucleus_patch_side(shape), []).append(path)
        voxels[path] = int(np.prod(shape))
    return {side: sorted(paths) for side, paths in sorted(groups.items())}, voxels


def openorganelle_nucleus_split_paths(input_path, split):
    """The OpenOrganelle nucleus volumes of the split, as {patch side: zarr paths}, and their voxel counts.

    The voxel counts weight the samples, so that the large tissue volumes are not outweighed by the many single-cell
    volumes with one to three nuclei.
    """
    with open(os.path.join(input_path, "openorganelle_nucleus_full", "index.json")) as f:
        volumes = [volume for volume in json.load(f) if volume["split"] == split]
    groups, voxels = {}, {}
    for volume in volumes:
        path = os.path.join(input_path, volume["path"])
        groups.setdefault(em_nucleus_patch_side(volume["shape"]), []).append(path)
        voxels[path] = int(np.prod(volume["shape"]))
    return {side: sorted(paths) for side, paths in sorted(groups.items())}, voxels


def cellmap_nucleus_split_crops(input_path, split):
    """The CellMap nucleus crops of the 'train' or 'val' part, as {patch side: crop paths}."""
    root = os.path.join(input_path, "cellmap_nucleus_crops")
    with open(os.path.join(root, "index.json")) as f:
        crops = [crop for crop in json.load(f) if crop["split"] == "train"]
    groups = {}
    for crop in crops:
        if (crop["name"] in CELLMAP_NUCLEUS_VAL_CROPS) == (split == "val"):
            groups.setdefault(em_nucleus_patch_side(crop["shape"]), []).append(
                os.path.join(root, f"{crop['name']}.zarr")
            )
    return {side: sorted(paths) for side, paths in sorted(groups.items())}


# CEM-MitoLab groups (source folders) that never train: every group whose name matches a mitochondria evaluation
# dataset, so that no evaluation data can leak into training through MitoLab. MitoEM (Wei2020_MitoEM-H / -R), MitoEM v2
# jurkat and macro (their OpenOrganelle source cells), DenseCell (Guay2020), Kasthuri (kasthuri2011), MitoSegEM
# (segem2015), Parlakgul (Parlakgul2020), Perez (perez2014) and the hela_cell volume of the CEM-Mito-Benchmark (the
# HeLa groups). Groups that match only training datasets (UroCell) still train.
MITOLAB_EXCLUDED_GROUPS = (
    "Wei2020_MitoEM-H", "Wei2020_MitoEM-R", "jrc_jurkat-1_openorganelle", "jrc_macrophage-2_openorganelle",
    "Guay2020_patient1", "kasthuri2011", "segem2015", "Hennies2020_AMST_HeLaBV2", "jrc_hela-1_openorganelle",
    "jrc_hela-2_openorganelle", "jrc_hela-3_openorganelle", "jrc_hela-4_openorganelle", "jrc_hela-21_openorganelle",
    "jrc_hela-22_openorganelle", "jrc_hela-bfa_openorganelle", "jrc_hela-h89-1_openorganelle",
    "jrc_hela-h89-2_openorganelle", "Parlakgul2020_file03_video", "Parlakgul2020_file04_video",
    "perez2014",
)

# Fraction of the CEM-MitoLab groups that validate. The split is by group, so the tiles of one source volume never
# land in both splits.
MITOLAB_VAL_FRACTION = 0.1


def mitolab_group(path, root):
    """The CEM-MitoLab group (source folder) of an image path: '<root>/<group>/images/<name>.tiff'."""
    return os.path.relpath(path, root).split(os.sep)[0]


def mitolab_split_paths(path, split):
    """The non-empty CEM-MitoLab images and masks of the 'train' or 'val' groups, without the excluded groups."""
    root = datasets.cem.get_mitolab_data(path, download=False)
    raw_paths, label_paths = datasets.cem.get_mitolab_paths(path, split=None, download=False)
    groups = sorted({mitolab_group(p, root) for p in raw_paths} - set(MITOLAB_EXCLUDED_GROUPS))
    train_groups, val_groups = train_test_split(groups, test_size=MITOLAB_VAL_FRACTION, random_state=42)
    keep = set(train_groups if split == "train" else val_groups)
    pairs = [(r, lab) for r, lab in zip(raw_paths, label_paths) if mitolab_group(r, root) in keep]
    if split == "val":
        # A seeded MITOLAB_VAL_IMAGES_PER_GROUP images per group, so that no group dominates the validation.
        by_group = {}
        for pair in pairs:
            by_group.setdefault(mitolab_group(pair[0], root), []).append(pair)
        rng = np.random.default_rng(42)
        pairs = []
        for group in sorted(by_group):
            members = by_group[group]
            if len(members) > MITOLAB_VAL_IMAGES_PER_GROUP:
                members = [members[i] for i in sorted(rng.choice(len(members), MITOLAB_VAL_IMAGES_PER_GROUP, False))]
            pairs.extend(members)
    return [r for r, _ in pairs], [lab for _, lab in pairs]


def mitoemv2_mossy_label_path(input_path, volume):
    """The labels of one MitoVerse han24 volume with MITOEMV2_MOSSY_IGNORE_LABEL outside the annotated boutons.

    They are written once to <input_path>/mitoverse_derived, slab by slab, under a temporary name that is moved in
    place when complete, so that interrupted writes and concurrent ranks never leave a partial array behind.
    """
    import zarr

    out_path = os.path.join(input_path, "mitoverse_derived", f"han24_{volume}_bouton_labels.zarr")
    if os.path.exists(out_path):
        return out_path

    source = zarr.open_group(os.path.join(input_path, "mitoverse", "data", "han24", f"{volume}.zarr"), mode="r")
    tmp_path = f"{out_path}.{os.getpid()}.incomplete"
    labels = zarr.open_group(tmp_path, mode="w").create_array(
        "labels", shape=source["mito"].shape, chunks=(32, 256, 256), dtype="uint32"
    )
    for z in range(0, labels.shape[0], 32):
        slab = source["mito"][z:z + 32].astype("uint32")
        slab[source["mask"][z:z + 32] == 0] = MITOEMV2_MOSSY_IGNORE_LABEL
        labels[z:z + 32] = slab
    try:
        os.replace(tmp_path, out_path)
    except OSError:
        # Another rank moved its copy in place first.
        shutil.rmtree(tmp_path)
    return out_path


def mitoverse_missing_data_label_path(input_path, dataset, volume):
    """The labels of one MitoVerse volume with MITOVERSE_IGNORE_LABEL inside the regions without image data.

    A region without image data is a connected raw == 0 region of a slice above MITOVERSE_MISSING_DATA_MIN_AREA.
    The labels are written once to <input_path>/mitoverse_derived, slice by slice, under a temporary name that is
    moved in place when complete, as in mitoemv2_mossy_label_path.
    """
    import zarr
    from scipy import ndimage

    out_path = os.path.join(input_path, "mitoverse_derived", f"{dataset}_{volume}_missing_data_labels.zarr")
    if os.path.exists(out_path):
        return out_path

    source = zarr.open_group(os.path.join(input_path, "mitoverse", "data", dataset, f"{volume}.zarr"), mode="r")
    tmp_path = f"{out_path}.{os.getpid()}.incomplete"
    labels = zarr.open_group(tmp_path, mode="w").create_array(
        "labels", shape=source["mito"].shape, chunks=(32, 256, 256), dtype="uint32"
    )
    for z in range(labels.shape[0]):
        slice_labels = source["mito"][z].astype("uint32")
        empty, _ = ndimage.label(source["img"][z] == 0)
        areas = np.bincount(empty.ravel())
        missing = np.isin(empty, np.where(areas > MITOVERSE_MISSING_DATA_MIN_AREA)[0]) & (empty > 0)
        slice_labels[missing] = MITOVERSE_IGNORE_LABEL
        labels[z] = slice_labels
    try:
        os.replace(tmp_path, out_path)
    except OSError:
        # Another rank moved its copy in place first.
        shutil.rmtree(tmp_path)
    return out_path


def _set_volume_min_instances(dataset, min_instances, depth=1):
    """Set min_num_instances of the leaves whose raw path ends in a key of *min_instances* ({path end: minimum}).

    The key is the last *depth* components of the raw path.
    """
    if isinstance(dataset, (list, tuple)):
        for ds in dataset:
            _set_volume_min_instances(ds, min_instances, depth)
    elif isinstance(dataset, UniDataWrapper):
        _set_volume_min_instances(dataset.ds, min_instances, depth)
    elif getattr(dataset, "datasets", None) is not None:
        for ds in dataset.datasets:
            _set_volume_min_instances(ds, min_instances, depth)
    elif getattr(dataset, "raw_path", None) is not None:
        key = "/".join(str(dataset.raw_path).split(os.sep)[-depth:])
        if key in min_instances:
            sampler = dataset.sampler
            dataset.sampler = MinInstanceSampler(
                min_num_instances=min_instances[key], min_size=sampler.min_size, exclude_ids=sampler.exclude_ids,
            )


def _get_em_mito_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo):
    """Get the mitochondria EM datasets for the mitochondria generalist.

    Only the mitochondria labels of a dataset are used, all other structures are background. Mitochondria are not dense
    cells, so the targets come from the plain label transform (no outer boundary erosion as in the EM cell datasets).
    The distance transform relabels every extracted patch with connected components (``apply_label=True``): semantic
    labels become instances, and the disconnected pieces of one id inside a patch become separate instances. Objects
    that touch, such as thin bridges across sections, stay merged.

    Returns:
        Tuple of (train_ds, val_ds) lists of UniDataWrapper instances.
    """
    train_ds, val_ds = [], []
    n_z = len(z_slices)

    def mito_label_trafo(min_size=EM_MITO_MIN_SIZE, sampling=None, ignore_label=None):
        trafo = kwargs.get("label_transform2") if label_trafo is None else label_trafo(
            min_size=min_size, sampling=sampling
        )
        if ignore_label is None:
            return trafo
        return partial(_ignore_label_trafo, ignore_label=ignore_label, label_trafo=trafo)

    # 1. CEM-MitoLab (2D mitochondria instance segmentation in EM tiles from about 600 source volumes and images)
    # NOTE: The tiles are 63-512 px, so the whole tile is drawn and randomly upscaled and padded to the patch shape.
    # Only the non-empty tiles are used, split by source group (see mitolab_split_paths).
    if "mitolab" in EM_MITO_DATASETS:
        mitolab_kwargs = {
            "patch_shape": None, "raw_key": None, "label_key": None, "is_seg_dataset": False, "ndim": 2,
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
            "label_transform2": mito_label_trafo(),
            "pre_label_transform": _connected_components_pre_label_trafo,
            "sampler": MinInstanceSampler(min_num_instances=1, min_size=EM_MITO_MIN_SIZE, exclude_ids=[0]),
            "transform": _with_flips(_random_resize_and_pad_trafo, patch_shape),
        }
        for split, ds_list in [("train", train_ds), ("val", val_ds)]:
            raws, labs = mitolab_split_paths(os.path.join(input_path, "mitolab"), split)
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=raws, label_paths=labs, n_samples=len(raws) if split == "train" else 50,
                        **mitolab_kwargs,
                    ), source_ndim=2,
                )
            )

    # 2. DenseCell mitochondria (3D, class 2 of the semantic map in SBF-SEM of human platelets, 10x10x50 nm)
    # NOTE: The mitochondria are small and sparse (13-36 per volume), so one object per patch is enough.
    if "densecell_mito" in EM_MITO_DATASETS:
        for z in z_slices:
            densecell_kwargs = {
                "path": os.path.join(input_path, "densecell"),
                "patch_shape": (z, *patch_shape),
                "label_choice": "mitochondrion",
                "download": True,
                "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(5, 1, 1)),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (DENSECELL_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["densecell_mito"]),
                (DENSECELL_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        datasets.get_densecell_dataset(
                            split="train", rois=roi, n_samples=max(1, n_samples // n_z), **densecell_kwargs
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 3. Lucchi (3D mitochondria in FIB-SEM of the mouse hippocampus, 5 nm isotropic, manual semantic labels)
    # NOTE: The sampler sees the semantic labels before the connected components, i.e. a single id.
    if "lucchi" in EM_MITO_DATASETS:
        for z in z_slices:
            lucchi_kwargs = {
                "path": os.path.join(input_path, "lucchi"),
                "patch_shape": (z, *patch_shape),
                "download": True,
                "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (LUCCHI_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["lucchi"]), (LUCCHI_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        datasets.get_lucchi_dataset(
                            split="train", rois=roi, n_samples=max(1, n_samples // n_z), **lucchi_kwargs
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 4. MICrONS pinky mitochondria (3D, ssTEM of the mouse visual cortex, 4x4x40 nm, manual instance labels)
    # NOTE: The uint8 ids wrap at 255, so distinct mitochondria can share an id; the per-patch connected components
    # of the label transform split the ones that are not connected.
    if "microns_pinky_mito" in EM_MITO_DATASETS:
        pinky_path = os.path.join(input_path, "microns", "pinky", MICRONS_PINKY_MITO_FILE)
        for z in z_slices:
            pinky_kwargs = {
                "raw_paths": pinky_path, "raw_key": "volumes/image",
                "label_paths": pinky_path, "label_key": "volumes/mitochondria",
                "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(10, 1, 1)),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (MICRONS_PINKY_MITO_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["microns_pinky_mito"]),
                (MICRONS_PINKY_MITO_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            rois=roi, n_samples=max(1, n_samples // n_z), **pinky_kwargs
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 5. MitoEM (3D mitochondria in ssTEM of the human and rat cortex, 8x8x30 nm, manual proofread instance labels)
    # NOTE: Reads the train volumes directly; torch-em's mitoem dataset adds its own instance label transform.
    if "mitoem" in EM_MITO_DATASETS:
        for sample in MITOEM_SAMPLES:
            mitoem_path = datasets.mitoem.get_mitoem_paths(
                os.path.join(input_path, "mitoem"), splits="train", samples=sample, download=False
            )[0]
            for z in z_slices:
                mitoem_kwargs = {
                    "raw_paths": mitoem_path, "raw_key": "raw", "label_paths": mitoem_path, "label_key": "labels",
                    "patch_shape": (z, *patch_shape), "ndim": 3,
                    **{
                        k: v for k, v in kwargs.items()
                        if k not in ["raw_transform", "label_transform2", "sampler", "transform"]
                    },
                    "label_transform2": mito_label_trafo(
                        EM_MITO_MIN_SIZE_3D, sampling=(3.75, 1, 1), ignore_label=MISSING_RAW_IGNORE_LABEL
                    ),
                    # Some slices hold regions of one constant value (e.g. saturated 254 in human z = 190) that still
                    # carry labels; they become ignore. The test needs the stored intensities, so the raw is only
                    # prepared here and normalized inside the transform.
                    "raw_transform": _prepare_identity,
                    "transform": partial(
                        _ignore_missing_raw_trafo, ignore_label=MISSING_RAW_IGNORE_LABEL, any_constant=True,
                        transform=get_augmentations(ndim=3),
                    ),
                    "pre_label_transform": _connected_components_pre_label_trafo,
                    "sampler": MinInstanceSampler(
                        min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                    ),
                }
                for roi, ds_list, n_samples in [
                    (MITOEM_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES[f"mitoem_{sample}"]),
                    (MITOEM_VAL_ROI, val_ds, 40),
                ]:
                    ds_list.append(
                        UniDataWrapper(
                            torch_em.default_segmentation_dataset(
                                rois=roi, n_samples=max(1, n_samples // n_z), **mitoem_kwargs
                            ),
                            source_ndim=3, group_key=(3, z),
                        )
                    )

    # 6. MitoEM v2 (3D mitochondria, expert-verified instances; FIB-SEM at 16 nm and SBF-SEM at 8 x 8 x 30 nm)
    # NOTE: The official train / val volumes, see MITOEMV2_DATASETS. jurkat validates on the end of its train volume.
    if "mitoemv2" in EM_MITO_DATASETS:
        mitoemv2_path = os.path.join(input_path, "mitoemv2")
        for name, z_scale in MITOEMV2_DATASETS.items():
            train_paths = datasets.mitoemv2.get_mitoemv2_paths(mitoemv2_path, name, split="train", download=False)
            if name == "jurkat":
                splits = [
                    (train_paths, MITOEMV2_JURKAT_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["mitoemv2_jurkat"]),
                    (train_paths, MITOEMV2_JURKAT_VAL_ROI, val_ds, 40),
                ]
            else:
                val_paths = datasets.mitoemv2.get_mitoemv2_paths(mitoemv2_path, name, split="val", download=False)
                splits = [
                    (train_paths, None, train_ds, EM_MITO_TRAIN_SAMPLES[f"mitoemv2_{name}"]),
                    (val_paths, None, val_ds, 40),
                ]
            for z in z_slices:
                mitoemv2_kwargs = {
                    "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                    **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                    "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(z_scale, 1, 1)),
                    "pre_label_transform": _connected_components_pre_label_trafo,
                    "sampler": MinInstanceSampler(
                        min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                    ),
                }
                for paths, roi, ds_list, n_samples in splits:
                    ds_list.append(
                        UniDataWrapper(
                            torch_em.default_segmentation_dataset(
                                raw_paths=paths, label_paths=paths, rois=None if roi is None else [roi] * len(paths),
                                n_samples=max(1, n_samples // n_z), **mitoemv2_kwargs,
                            ),
                            source_ndim=3, group_key=(3, z),
                        )
                    )

    # 7. MitoEM v2 mossy (3D mitochondria in ssSEM of mossy fiber boutons in the mouse cerebellum, 8 x 8 x 30 nm)
    # NOTE: Only the boutons are annotated: the raw comes from the MitoVerse copies, the labels carry the ignore label
    # outside the bouton mask (see mitoemv2_mossy_label_path), and the loss ignores those voxels.
    if "mitoemv2_mossy" in EM_MITO_DATASETS:
        han24_root = os.path.join(input_path, "mitoverse", "data", "han24")
        for volumes, ds_list, n_samples in [
            (MITOEMV2_MOSSY_TRAIN_VOLUMES, train_ds, EM_MITO_TRAIN_SAMPLES["mitoemv2_mossy"]),
            (MITOEMV2_MOSSY_VAL_VOLUMES, val_ds, 40),
        ]:
            raw_paths = [os.path.join(han24_root, f"{volume}.zarr") for volume in volumes.values()]
            label_paths = [mitoemv2_mossy_label_path(input_path, volume) for volume in volumes.values()]
            for z in z_slices:
                mossy_kwargs = {
                    "raw_key": "img", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                    **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                    "label_transform2": mito_label_trafo(
                        EM_MITO_MIN_SIZE_3D, sampling=(30 / 8, 1, 1), ignore_label=MITOEMV2_MOSSY_IGNORE_LABEL
                    ),
                    "pre_label_transform": partial(
                        _connected_components_pre_label_trafo, ignore_label=MITOEMV2_MOSSY_IGNORE_LABEL
                    ),
                    "sampler": MinInstanceSampler(
                        min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D,
                        exclude_ids=[0, MITOEMV2_MOSSY_IGNORE_LABEL],
                    ),
                }
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=raw_paths, label_paths=label_paths, n_samples=max(1, n_samples // n_z),
                            **mossy_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 8. ProbTEM (2D mitochondria in TEM of mouse skeletal muscle, 2560 x 2560, manual semantic labels)
    # NOTE: The sampler sees the semantic labels before the connected components, i.e. a single id.
    if "probtem" in EM_MITO_DATASETS:
        probtem_kwargs = {
            "raw_key": "raw", "label_key": "labels", "patch_shape": patch_shape, "ndim": 2, "is_seg_dataset": True,
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
            "label_transform2": mito_label_trafo(),
            "pre_label_transform": _connected_components_pre_label_trafo,
            "sampler": MinInstanceSampler(
                min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE, exclude_ids=[0]
            ),
        }
        for split, ds_list, n_samples in [("train", train_ds, EM_MITO_TRAIN_SAMPLES["probtem"]), ("val", val_ds, 40)]:
            paths = probtem_split_paths(input_path, split)
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=paths, label_paths=paths, n_samples=n_samples, **probtem_kwargs
                    ), source_ndim=2,
                )
            )

    # 9. UroCell (3D mitochondria in FIB-SEM of the mouse urinary bladder, 16 nm isotropic, manual instance labels)
    # NOTE: The volumes are 256 px in plane, so 256 x 256 patches are resized to 512 x 512 (the z / xy anisotropy of
    # the resized patch is 2), as for the small EMNeuron volumes. The raw is stored as float64.
    if "uro_cell" in EM_MITO_DATASETS:
        urocell_paths = [os.path.join(input_path, "uro_cell", name) for name in UROCELL_MITO_VOLUMES]
        for z in z_slices:
            urocell_kwargs = {
                "raw_key": "raw", "label_key": "labels/mito", "patch_shape": (z, 256, 256), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
                "raw_transform": _resize_raw_to_512,
                "label_transform2": partial(
                    _resize_then_em_label_trafo,
                    em_trafo_fn=mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(2, 1, 1)),
                ),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (UROCELL_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["uro_cell"]), (UROCELL_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=urocell_paths, label_paths=urocell_paths, rois=[roi] * len(urocell_paths),
                            n_samples=max(1, n_samples // n_z), **urocell_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 10. VNC (mitochondria in ssTEM of the Drosophila ventral nerve cord, 4.6 x 4.6 x 45 nm, manual labels)
    # NOTE: Single slices of the 20-slice train stack (see VNC_TRAIN_ROI), drawn as (1, 512, 512) patches.
    if "vnc" in EM_MITO_DATASETS:
        vnc_path = os.path.join(input_path, "vnc", "vnc_train.h5")
        vnc_kwargs = {
            "raw_key": "raw", "label_key": "labels/mitochondria", "patch_shape": (1, *patch_shape),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
            "label_transform2": mito_label_trafo(),
            "pre_label_transform": _connected_components_pre_label_trafo,
            "sampler": MinInstanceSampler(
                min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE, exclude_ids=[0]
            ),
        }
        for roi, ds_list, n_samples in [
            (VNC_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["vnc"]), (VNC_VAL_ROI, val_ds, 20),
        ]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=vnc_path, label_paths=vnc_path, rois=roi, n_samples=n_samples, **vnc_kwargs
                    ),
                    source_ndim=2,
                )
            )

    # 11. Parlakgul liver (3D mitochondria in FIB-SEM of the mouse liver, 8 nm isotropic, proofread semantic labels)
    # NOTE: The labels are stored as 0 / 255, so the sampler sees a single id before the connected components. The
    # connected components merge a few touching mitochondria (2-7% of the objects).
    if "parlakgul_liver" in EM_MITO_DATASETS:
        parlakgul_paths = [os.path.join(input_path, box) for box in PARLAKGUL_LIVER_BOXES]
        for z in z_slices:
            parlakgul_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (PARLAKGUL_LIVER_TRAIN_ROI, train_ds, 200), (PARLAKGUL_LIVER_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=parlakgul_paths, label_paths=parlakgul_paths, rois=[roi] * len(parlakgul_paths),
                            n_samples=max(1, n_samples // n_z), **parlakgul_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 12. MitoVerse xiao18 (3D mitochondria in ssSEM of the mouse auditory cortex, 8 x 8 x 50 nm, instance labels)
    if "mitoverse_xiao18" in EM_MITO_DATASETS:
        xiao18_path = os.path.join(input_path, "mitoverse", "data", "xiao18", "vol0.zarr")
        for z in z_slices:
            xiao18_kwargs = {
                "raw_paths": xiao18_path, "raw_key": "img", "label_paths": xiao18_path, "label_key": "mito",
                "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(6.25, 1, 1)),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (MITOVERSE_XIAO18_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["mitoverse_xiao18"]),
                (MITOVERSE_XIAO18_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            rois=roi, n_samples=max(1, n_samples // n_z), **xiao18_kwargs
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 13. MitoVerse haberl18 (3D mitochondria in SBF-SEM of the mouse brain, CDeep3M, 10 x 10 x 24 nm, instance labels)
    # NOTE: The volumes are 256 px in plane, so 256 x 256 patches are resized to 512 x 512 (the z / xy anisotropy of
    # the resized patch is 24 / 5), as for UroCell.
    if "mitoverse_haberl18" in EM_MITO_DATASETS:
        haberl18_root = os.path.join(input_path, "mitoverse", "data", "haberl18")
        for z in z_slices:
            haberl18_kwargs = {
                "raw_key": "img", "label_key": "mito", "patch_shape": (z, 256, 256), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2", "sampler"]},
                "raw_transform": _resize_raw_to_512,
                "label_transform2": partial(
                    _resize_then_em_label_trafo,
                    em_trafo_fn=mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(4.8, 1, 1)),
                ),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for volume, ds_list, n_samples in [("vol0", train_ds, EM_MITO_TRAIN_SAMPLES["mitoverse_haberl18"])]:
                haberl18_path = os.path.join(haberl18_root, f"{volume}.zarr")
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=haberl18_path, label_paths=haberl18_path, n_samples=max(1, n_samples // n_z),
                            **haberl18_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 14. MitoVerse MICrONS phase 1 (3D mitochondria in ssTEM of the mouse visual cortex, 8 x 8 x 40 nm, proofread)
    # NOTE: The labels carry the ignore label in the regions without image data (see
    # mitoverse_missing_data_label_path), and the loss ignores those voxels.
    if "mitoverse_microns1" in EM_MITO_DATASETS:
        microns1_root = os.path.join(input_path, "mitoverse", "data", "turner22")
        raw_paths = [os.path.join(microns1_root, f"{volume}.zarr") for volume in MITOVERSE_MICRONS1_VOLUMES]
        label_paths = [
            mitoverse_missing_data_label_path(input_path, "turner22", volume) for volume in MITOVERSE_MICRONS1_VOLUMES
        ]
        for z in z_slices:
            microns1_kwargs = {
                "raw_key": "img", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{
                    k: v for k, v in kwargs.items()
                    if k not in ["raw_transform", "label_transform2", "sampler", "transform"]
                },
                "label_transform2": mito_label_trafo(
                    EM_MITO_MIN_SIZE_3D, sampling=(5, 1, 1), ignore_label=MITOVERSE_IGNORE_LABEL
                ),
                # Missing sections and tiles are also filled with the constant 128, which the raw == 0 test of the
                # derived labels misses; those regions become ignore here. The raw is normalized inside the transform.
                "raw_transform": _prepare_identity,
                "transform": partial(
                    _ignore_missing_raw_trafo, ignore_label=MITOVERSE_IGNORE_LABEL, any_constant=True,
                    transform=get_augmentations(ndim=3),
                ),
                "pre_label_transform": partial(
                    _connected_components_pre_label_trafo, ignore_label=MITOVERSE_IGNORE_LABEL
                ),
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D,
                    exclude_ids=[0, MITOVERSE_IGNORE_LABEL],
                ),
            }
            train_only = [volume in MITOVERSE_MICRONS1_TRAIN_ONLY_VOLUMES for volume in MITOVERSE_MICRONS1_VOLUMES]
            train_rois = [np.s_[:, :, :] if only else MITOVERSE_MICRONS1_TRAIN_ROI for only in train_only]
            val_volumes = [i for i, only in enumerate(train_only) if not only]
            for paths, rois, ds_list, n_samples in [
                ((raw_paths, label_paths), train_rois, train_ds, EM_MITO_TRAIN_SAMPLES["mitoverse_microns1"]),
                (
                    ([raw_paths[i] for i in val_volumes], [label_paths[i] for i in val_volumes]),
                    [MITOVERSE_MICRONS1_VAL_ROI] * len(val_volumes), val_ds, 40,
                ),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=paths[0], label_paths=paths[1], rois=rois,
                            n_samples=max(1, n_samples // n_z), **microns1_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 15. CellMap mitochondria (3D mitochondria in FIB-SEM crops of cells and tissues, 2-32 nm, manual labels)
    # NOTE: Nuclei and all other structures are background. Each patch side group is one dataset; patches below 512 px
    # are resized to 512 x 512, which gives the (isotropic) crops a z / xy anisotropy of 512 / side.
    if "cellmap_mito" in EM_MITO_DATASETS:
        cellmap_groups = {split: cellmap_mito_split_crops(input_path, split) for split in ("train", "val")}
        for z in z_slices:
            for split, ds_list, n_samples in [
                ("train", train_ds, EM_MITO_TRAIN_SAMPLES["cellmap_mito"]), ("val", val_ds, 40),
            ]:
                n_crops = sum(len(paths) for paths in cellmap_groups[split].values())
                for side, paths in cellmap_groups[split].items():
                    resize = side < 512
                    em_trafo_fn = mito_label_trafo(
                        EM_MITO_MIN_SIZE_3D, sampling=(512 / side, 1, 1) if resize else None
                    )
                    excluded = ["raw_transform", "label_transform2", "sampler"]
                    cellmap_kwargs = {
                        "raw_key": "raw", "label_key": "labels", "patch_shape": (z, side, side), "ndim": 3,
                        **{k: v for k, v in kwargs.items() if k not in excluded},
                        "raw_transform": _resize_raw_to_512 if resize else kwargs["raw_transform"],
                        "label_transform2": (
                            partial(_resize_then_em_label_trafo, em_trafo_fn=em_trafo_fn) if resize else em_trafo_fn
                        ),
                        "pre_label_transform": _connected_components_pre_label_trafo,
                        "sampler": MinInstanceSampler(
                            min_num_instances=1, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                        ),
                    }
                    ds_list.append(
                        UniDataWrapper(
                            torch_em.default_segmentation_dataset(
                                raw_paths=paths, label_paths=paths,
                                n_samples=max(1, round(n_samples * len(paths) / n_crops) // n_z), **cellmap_kwargs,
                            ),
                            source_ndim=3, group_key=(3, z),
                        )
                    )

    # 16. Human organoids mitochondria (3D, FIB-SEM of patient-derived colorectal cancer organoids, manual labels)
    # NOTE: The labels are the neck watershed instances (HUMAN_ORGANOIDS_MITO_ROOT); the per-patch connected components
    # keep touching instances apart. The large black alignment padding is handled by the blank slice sampler.
    if "human_organoids_mito" in EM_MITO_DATASETS:
        organoid_paths = [
            os.path.join(input_path, "human_organoids", "11380", "data", name) for name in HUMAN_ORGANOIDS_VOLUMES
        ]
        mito_label_paths = [
            os.path.join(input_path, HUMAN_ORGANOIDS_MITO_ROOT, name.replace("_bin2.h5", "_mito_labels.zarr"))
            for name in HUMAN_ORGANOIDS_VOLUMES
        ]
        organoid_rois = {"train": [], "val": []}
        for path in organoid_paths:
            with open_file(path, "r") as f:
                n_train = int(f["mitos"].shape[0] * (1 - HUMAN_ORGANOIDS_VAL_FRACTION))
            organoid_rois["train"].append(np.s_[:n_train, :, :])
            organoid_rois["val"].append(np.s_[n_train:, :, :])
        for z in z_slices:
            organoid_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for split, ds_list, n_samples in [
                ("train", train_ds, EM_MITO_TRAIN_SAMPLES["human_organoids_mito"]), ("val", val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=organoid_paths, label_paths=mito_label_paths, rois=organoid_rois[split],
                            n_samples=max(1, n_samples // n_z), **organoid_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 17. ASEM mitochondria (3D mitochondria in FIB-SEM of cultured cells, 4-5 nm isotropic, manual semantic labels)
    # NOTE: The sampler sees the semantic labels before the connected components, i.e. a single id.
    if "asem_mito" in EM_MITO_DATASETS:
        asem_paths = [os.path.join(input_path, ASEM_MITO_ROOT, name) for name in ASEM_MITO_TRAIN_VOLUMES]
        small_paths = [os.path.join(input_path, ASEM_MITO_ROOT, name) for name in ASEM_MITO_SMALL_VOLUMES]
        resize = 512 / ASEM_MITO_SMALL_SIDE
        excluded = ["raw_transform", "label_transform2", "sampler"]
        for z in z_slices:
            asem_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (ASEM_MITO_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["asem_mito"]), (ASEM_MITO_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=asem_paths, label_paths=asem_paths, rois=[roi] * len(asem_paths),
                            n_samples=max(1, n_samples // n_z), **asem_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )
            small_kwargs = {
                "raw_key": "raw", "label_key": "labels", "ndim": 3,
                "patch_shape": (z, ASEM_MITO_SMALL_SIDE, ASEM_MITO_SMALL_SIDE),
                **{k: v for k, v in kwargs.items() if k not in excluded},
                "raw_transform": _resize_raw_to_512,
                "label_transform2": partial(
                    _resize_then_em_label_trafo,
                    em_trafo_fn=mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(resize, 1, 1)),
                ),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            train_ds.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=small_paths, label_paths=small_paths,
                        n_samples=max(1, EM_MITO_TRAIN_SAMPLES["asem_mito_small"] // n_z), **small_kwargs,
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 18. Brain PFIB-SEM mitochondria (3D mitochondria in PFIB-SEM of the mouse brain, 5 nm isotropic, expert labels)
    # NOTE: The reference (2 us) and the registered noisy (0.5 us) stack train with the same labels; only the
    # reference stack validates.
    if "brain_pfibsem_mito" in EM_MITO_DATASETS:
        brain_path = os.path.join(input_path, BRAIN_PFIBSEM_PATH)
        for z in z_slices:
            for raw_key, roi, ds_list, n_samples in [
                ("raw/reference", BRAIN_PFIBSEM_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["brain_pfibsem_reference"]),
                ("raw/noisy", BRAIN_PFIBSEM_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["brain_pfibsem_noisy"]),
                ("raw/reference", BRAIN_PFIBSEM_VAL_ROI, val_ds, 40),
            ]:
                brain_kwargs = {
                    "raw_key": raw_key, "label_key": "labels/mitochondria", "patch_shape": (z, *patch_shape),
                    "ndim": 3, **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                    "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D),
                    "pre_label_transform": _connected_components_pre_label_trafo,
                    "sampler": MinInstanceSampler(
                        min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                    ),
                }
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=brain_path, label_paths=brain_path, rois=roi,
                            n_samples=max(1, n_samples // n_z), **brain_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 19. SNEMI Mito (3D mitochondria in ssSEM of the mouse neocortex, SNEMI3D, 6 x 6 x 30 nm, manual binary masks)
    # NOTE: The sampler sees the binary labels before the connected components, i.e. a single id.
    if "snemi_mito" in EM_MITO_DATASETS:
        snemi_path = os.path.join(input_path, SNEMI_MITO_PATH)
        for z in z_slices:
            snemi_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(30 / 6, 1, 1)),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (SNEMI_MITO_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["snemi_mito"]), (SNEMI_MITO_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=snemi_path, label_paths=snemi_path, rois=roi,
                            n_samples=max(1, n_samples // n_z), **snemi_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 20. U2OS FIB-SEM mitochondria (2D mitochondria in single FIB-SEM slices of a U2-OS cell, manual binary masks)
    # NOTE: The sampler sees the binary labels before the connected components, i.e. a single id.
    if "u2os_fibsem_mito" in EM_MITO_DATASETS:
        u2os_images = sorted(glob(os.path.join(input_path, U2OS_FIBSEM_MITO_ROOT, "images_tif", "*.tif")))
        u2os_labels = sorted(glob(os.path.join(input_path, U2OS_FIBSEM_MITO_ROOT, "labels_tif", "*.tif")))
        u2os_kwargs = {
            "raw_key": None, "label_key": None, "patch_shape": patch_shape, "ndim": 2, "is_seg_dataset": False,
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
            "label_transform2": mito_label_trafo(),
            "pre_label_transform": _connected_components_pre_label_trafo,
            "sampler": MinInstanceSampler(
                min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE, exclude_ids=[0]
            ),
        }
        for split, ds_list, n_samples in [
            ("train", train_ds, EM_MITO_TRAIN_SAMPLES["u2os_fibsem_mito"]), ("val", val_ds, 40),
        ]:
            keep = [
                (i % U2OS_FIBSEM_MITO_VAL_STEP == U2OS_FIBSEM_MITO_VAL_STEP // 2) == (split == "val")
                for i in range(len(u2os_images))
            ]
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=[p for p, k in zip(u2os_images, keep) if k],
                        label_paths=[p for p, k in zip(u2os_labels, keep) if k], n_samples=n_samples, **u2os_kwargs,
                    ), source_ndim=2,
                )
            )

    # 21. HeLa STARD3 mitochondria (3D mitochondria in FIB-SEM of HeLa cells, 5-8 nm, manual binary masks in the ROI)
    # NOTE: The sampler sees the binary labels before the connected components, i.e. a single id.
    if "hela_stard3_mito" in EM_MITO_DATASETS:
        hela_paths = [os.path.join(input_path, "hela_stard3", f"{volume}.h5") for volume in HELA_STARD3_VOLUMES]
        train_rois = [
            np.s_[:HELA_STARD3_VAL_START, :, :] if volume == HELA_STARD3_VAL_VOLUME else np.s_[:, :, :]
            for volume in HELA_STARD3_VOLUMES
        ]
        val_path = hela_paths[HELA_STARD3_VOLUMES.index(HELA_STARD3_VAL_VOLUME)]
        for z in z_slices:
            hela_kwargs = {
                "raw_key": "raw", "label_key": "labels/mitochondria", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for paths, rois, ds_list, n_samples in [
                (hela_paths, train_rois, train_ds, EM_MITO_TRAIN_SAMPLES["hela_stard3_mito"]),
                ([val_path], [np.s_[HELA_STARD3_VAL_START:, :, :]], val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=paths, label_paths=paths, rois=rois, n_samples=max(1, n_samples // n_z),
                            **hela_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 22. Parlakgul fasting liver mitochondria (3D mitochondria in FIB-SEM of the mouse liver, 8 nm, instance labels)
    if "parlakgul_fasting_liver_mito" in EM_MITO_DATASETS:
        fasting_paths = sorted(glob(os.path.join(input_path, PARLAKGUL_FASTING_ROOT, "*_mito_*.zarr")))
        for z in z_slices:
            fasting_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (PARLAKGUL_FASTING_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["parlakgul_fasting_liver_mito"]),
                (PARLAKGUL_FASTING_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=fasting_paths, label_paths=fasting_paths, rois=[roi] * len(fasting_paths),
                            n_samples=max(1, n_samples // n_z), **fasting_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 23. AIVE mitochondria (3D mitochondria in FIB-SEM of cells and muscle tissue, 6.5 x 6.5 x 10 nm, semantic labels)
    # NOTE: The mitochondria class is taken before the sampler, i.e. the sampler sees a single id.
    if "aive_mito" in EM_MITO_DATASETS:
        aive_paths = [os.path.join(input_path, "aive", name) for name in AIVE_MITO_VAL_STARTS]
        for z in z_slices:
            aive_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler", "pre_label_transform"]},
                "pre_label_transform": partial(
                    _connected_components_pre_label_trafo, foreground_class=AIVE_MITO_CLASS
                ),
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(10 / 6.5, 1, 1)),
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for rois, ds_list, n_samples in [
                (
                    [np.s_[:start, :, :] for start in AIVE_MITO_VAL_STARTS.values()], train_ds,
                    EM_MITO_TRAIN_SAMPLES["aive_mito"],
                ),
                ([np.s_[start:, :, :] for start in AIVE_MITO_VAL_STARTS.values()], val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=aive_paths, label_paths=aive_paths, rois=rois,
                            n_samples=max(1, n_samples // n_z), **aive_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 24. MitoSegEM (3D mitochondria in SBF-SEM of the mouse testis and brainstem, 15 x 15 x 50 nm, instance labels)
    if "mito_segem" in EM_MITO_DATASETS:
        from torch_em.data.datasets.electron_microscopy.mito_segem import get_mito_segem_paths
        segem_paths = [
            path for tissue, box in MITO_SEGEM_TRAIN_BOXES.items()
            for path in get_mito_segem_paths(os.path.join(input_path, "mito_segem"), [box], tissue, download=True)
        ]
        for z in z_slices:
            segem_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": mito_label_trafo(EM_MITO_MIN_SIZE_3D, sampling=(50 / 15, 1, 1)),
                "pre_label_transform": _connected_components_pre_label_trafo,
                "sampler": MinInstanceSampler(
                    min_num_instances=EM_MITO_MIN_INSTANCES, min_size=EM_MITO_MIN_SIZE_3D, exclude_ids=[0]
                ),
            }
            for roi, ds_list, n_samples in [
                (MITO_SEGEM_TRAIN_ROI, train_ds, EM_MITO_TRAIN_SAMPLES["mito_segem"]), (MITO_SEGEM_VAL_ROI, val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=segem_paths, label_paths=segem_paths, rois=[roi] * len(segem_paths),
                            n_samples=max(1, n_samples // n_z), **segem_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    _set_volume_min_instances([train_ds, val_ds], EM_MITO_SPARSE_VOLUMES)
    return train_ds, val_ds


def _get_em_nucleus_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo):
    """Get the nucleus EM datasets for the nucleus generalist.

    The targets come from the plain label transform, as for the mitochondria generalist, and the distance transform
    relabels every extracted patch with connected components (``apply_label=True``). Each patch side group of a dataset
    (see EM_NUCLEUS_PATCH_SIDES) is one dataset, and patches below 512 px are resized to 512 x 512, which gives the
    isotropic volumes a z / xy anisotropy of 512 / side.

    Returns:
        Tuple of (train_ds, val_ds) lists of UniDataWrapper instances.
    """
    train_ds, val_ds = [], []
    n_z = len(z_slices)

    def nucleus_label_trafo(side, ignore_label=None, z_scale=1):
        resize = side < 512
        z_factor = z_scale * 512 / side
        sampling = None if z_factor == 1 else (z_factor, 1, 1)
        trafo = kwargs.get("label_transform2") if label_trafo is None else label_trafo(
            min_size=EM_NUCLEUS_MIN_SIZE_3D, sampling=sampling
        )
        if ignore_label is not None:
            trafo = partial(_ignore_label_trafo, ignore_label=ignore_label, label_trafo=trafo)
        return partial(_resize_then_em_label_trafo, em_trafo_fn=trafo) if resize else trafo

    def connected_components(ignore_label=None):
        if ignore_label is None:
            return _connected_components_pre_label_trafo
        return partial(_connected_components_pre_label_trafo, ignore_label=ignore_label)

    def nucleus_sampler(ignore_label=None, min_num_instances=1):
        exclude_ids = [0] if ignore_label is None else [0, ignore_label]
        return MinInstanceSampler(
            min_num_instances=min_num_instances, min_size=EM_NUCLEUS_MIN_SIZE_3D, exclude_ids=exclude_ids
        )

    def add_grouped_datasets(
        groups, label_key, train_samples, ignore_label=None, pre_label_transform=None, weights=None,
        min_num_instances=1,
    ):
        # The pre_label_transform (connected components by default) runs before the sampler, so the sampler counts the
        # instances of a patch. The samples of a split are divided across the patch side groups by the summed weights
        # of their volumes (*weights*, {path: weight}), by default one per volume, with at least one per volume.
        excluded = ["raw_transform", "label_transform2", "sampler", "pre_label_transform"]
        if pre_label_transform is None:
            pre_label_transform = connected_components(ignore_label)
        for z in z_slices:
            for split, ds_list, n_samples in [("train", train_ds, train_samples), ("val", val_ds, 40)]:
                group_weights = {
                    side: sum(1 if weights is None else weights[path] for path in paths)
                    for side, paths in groups[split].items()
                }
                total_weight = sum(group_weights.values())
                for side, paths in groups[split].items():
                    nucleus_kwargs = {
                        "raw_key": "raw", "label_key": label_key, "patch_shape": (z, side, side), "ndim": 3,
                        **{k: v for k, v in kwargs.items() if k not in excluded},
                        "raw_transform": _resize_raw_to_512 if side < 512 else kwargs["raw_transform"],
                        "label_transform2": nucleus_label_trafo(side, ignore_label),
                        "pre_label_transform": pre_label_transform,
                        "sampler": nucleus_sampler(ignore_label, min_num_instances),
                    }
                    ds_list.append(
                        UniDataWrapper(
                            torch_em.default_segmentation_dataset(
                                raw_paths=paths, label_paths=paths,
                                n_samples=max(len(paths), round(n_samples * group_weights[side] / total_weight) // n_z),
                                **nucleus_kwargs,
                            ),
                            source_ndim=3, group_key=(3, z),
                        )
                    )

    def add_volume_datasets(volumes, train_samples, ignore_label=None, min_num_instances=1):
        # *volumes* are (split, path, raw_key, label_key, roi, z_scale) tuples; the volumes of a split are grouped by
        # patch side (from the in-plane size of their roi), keys and z_scale, and the samples are divided across the
        # groups by their number of volumes, with at least one per volume.
        excluded = ["raw_transform", "label_transform2", "sampler", "pre_label_transform"]
        groups = {}
        for split, path, raw_key, label_key, roi, z_scale in volumes:
            with open_file(path, "r") as f:
                shape = f[label_key].shape
            in_plane = [len(range(*sl.indices(n))) for sl, n in zip(roi[1:], shape[1:])]
            key = (split, em_nucleus_patch_side(in_plane), raw_key, label_key, z_scale)
            groups.setdefault(key, []).append((path, roi))
        for z in z_slices:
            for split, ds_list, n_samples in [("train", train_ds, train_samples), ("val", val_ds, 40)]:
                n_volumes = sum(len(entries) for key, entries in groups.items() if key[0] == split)
                for (group_split, side, raw_key, label_key, z_scale), entries in groups.items():
                    if group_split != split:
                        continue
                    volume_kwargs = {
                        "raw_key": raw_key, "label_key": label_key, "patch_shape": (z, side, side), "ndim": 3,
                        **{k: v for k, v in kwargs.items() if k not in excluded},
                        "raw_transform": _resize_raw_to_512 if side < 512 else kwargs["raw_transform"],
                        "label_transform2": nucleus_label_trafo(side, ignore_label, z_scale=z_scale),
                        "pre_label_transform": connected_components(ignore_label),
                        "sampler": nucleus_sampler(ignore_label, min_num_instances),
                    }
                    ds_list.append(
                        UniDataWrapper(
                            torch_em.default_segmentation_dataset(
                                raw_paths=[path for path, _ in entries], label_paths=[path for path, _ in entries],
                                rois=[roi for _, roi in entries],
                                n_samples=max(len(entries), round(n_samples * len(entries) / n_volumes) // n_z),
                                **volume_kwargs,
                            ),
                            source_ndim=3, group_key=(3, z),
                        )
                    )

    def add_image_datasets(train_paths, val_paths, train_samples, ignore_label=None, min_num_instances=1):
        # 2D images (h5 files with 'raw' and 'labels') of at least 512 px, drawn as 512 x 512 patches.
        image_kwargs = {
            "raw_key": "raw", "label_key": "labels", "patch_shape": patch_shape, "ndim": 2,
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
            "label_transform2": nucleus_label_trafo(512, ignore_label=ignore_label),
            "pre_label_transform": connected_components(ignore_label),
            "sampler": nucleus_sampler(ignore_label, min_num_instances),
        }
        for paths, ds_list, n_samples in [(train_paths, train_ds, train_samples), (val_paths, val_ds, 40)]:
            if not paths:
                continue
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=paths, label_paths=paths, n_samples=n_samples, **image_kwargs
                    ),
                    source_ndim=2,
                )
            )

    # 1. BetaSeg nuclei (3D nucleus of one target cell per volume in FIB-SEM of mouse beta cells, 4 nm, manual labels)
    # NOTE: The loss ignores the voxels outside the target cell and on blank raw (BETASEG_NUCLEUS_IGNORE_LABEL).
    if "betaseg_nucleus" in EM_NUCLEUS_DATASETS:
        betaseg_groups = {split: betaseg_nucleus_split_boxes(input_path, split) for split in ("train", "val")}
        add_grouped_datasets(
            betaseg_groups, "labels/nucleus", EM_NUCLEUS_TRAIN_SAMPLES["betaseg_nucleus"],
            ignore_label=BETASEG_NUCLEUS_IGNORE_LABEL,
        )

    # 2. CellMap nuclei (3D nuclei in FIB-SEM crops of mouse liver, kidney and heart and zebrafish heart, 32-64 nm,
    # manual instance labels)
    if "cellmap_nucleus" in EM_NUCLEUS_DATASETS:
        cellmap_groups = {split: cellmap_nucleus_split_crops(input_path, split) for split in ("train", "val")}
        add_grouped_datasets(
            cellmap_groups, "labels", EM_NUCLEUS_TRAIN_SAMPLES["cellmap_nucleus"],
            min_num_instances=EM_NUCLEUS_MIN_INSTANCES["cellmap_nucleus"],
        )

    # 3. Human organoids nuclei (3D nuclei in FIB-SEM of a breast cancer spheroid and colorectal cancer organoids,
    # manual labels)
    # NOTE: The loss ignores label fragments and the entotic cell (HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL). The large
    # black alignment padding is handled by the blank slice sampler.
    if "human_organoids_nucleus" in EM_NUCLEUS_DATASETS:
        names = [name.replace("_bin2.h5", "") for name in HUMAN_ORGANOIDS_VOLUMES]
        raw_paths = [os.path.join(input_path, "human_organoids", "11380", "data", f"{name}_bin2.h5") for name in names]
        label_paths = [
            os.path.join(input_path, "human_organoids_nucleus", f"{name}_nucleus_labels.zarr") for name in names
        ]
        organoid_rois = {"train": [], "val": []}
        for path in raw_paths:
            with open_file(path, "r") as f:
                n_train = int(f["nuclei"].shape[0] * (1 - HUMAN_ORGANOIDS_VAL_FRACTION))
            organoid_rois["train"].append(np.s_[:n_train, :, :])
            organoid_rois["val"].append(np.s_[n_train:, :, :])
        for z in z_slices:
            organoid_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": nucleus_label_trafo(512, ignore_label=HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL),
                "pre_label_transform": connected_components(HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL),
                "sampler": nucleus_sampler(
                    HUMAN_ORGANOIDS_NUCLEUS_IGNORE_LABEL, EM_NUCLEUS_MIN_INSTANCES["human_organoids_nucleus"]
                ),
            }
            for split, ds_list, n_samples in [
                ("train", train_ds, EM_NUCLEUS_TRAIN_SAMPLES["human_organoids_nucleus"]), ("val", val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=raw_paths, label_paths=label_paths, rois=organoid_rois[split],
                            n_samples=max(1, n_samples // n_z), **organoid_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 4. SXT INS-1E nuclei (3D nucleus of one cell per tomogram in soft X-ray tomography of INS-1E cells, 36.8-45.5 nm,
    # semi-automatic labels)
    # NOTE: The pre_label_transform maps the class labels to the nucleus and the exterior to SXT_NUCLEUS_IGNORE_LABEL,
    # which the loss ignores, and labels the connected components before the sampler.
    if "sxt_nucleus" in EM_NUCLEUS_DATASETS:
        sxt_groups = {split: sxt_nucleus_split_paths(input_path, split) for split in ("train", "val")}
        add_grouped_datasets(
            sxt_groups, "labels", EM_NUCLEUS_TRAIN_SAMPLES["sxt_nucleus"], ignore_label=SXT_NUCLEUS_IGNORE_LABEL,
            pre_label_transform=sxt_nucleus_instances,
        )

    # 5. FAFB nuclei (3D nuclei in ssTEM of the adult fly brain, 32 x 32 x 40 nm, automatic FlyWire segmentation)
    if "fafb_nuclei" in EM_NUCLEUS_DATASETS:
        for z in z_slices:
            fafb_kwargs = {
                "raw_key": "raw", "label_key": "labels/nuclei", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": nucleus_label_trafo(512, z_scale=FAFB_NUCLEI_Z_SCALE),
                "pre_label_transform": connected_components(),
                "sampler": nucleus_sampler(min_num_instances=EM_NUCLEUS_MIN_INSTANCES["fafb_nuclei"]),
            }
            for split, ds_list, n_samples in [
                ("train", train_ds, EM_NUCLEUS_TRAIN_SAMPLES["fafb_nuclei"]), ("val", val_ds, 40),
            ]:
                paths = sorted(glob(os.path.join(input_path, "fafb_nuclei", split, "*.h5")))
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=paths, label_paths=paths, n_samples=max(1, n_samples // n_z), **fafb_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 6. Janelia nuclei (3D nuclei in FIB-SEM of 15 OpenOrganelle volumes of mouse and fly tissues, 32-128 nm,
    # Cellpose pseudo-labels and 3 manual volumes)
    if "janelia_nucleus" in EM_NUCLEUS_DATASETS:
        janelia_groups, janelia_voxels = {}, {}
        for split in ("train", "val"):
            janelia_groups[split], split_voxels = janelia_nucleus_split_paths(input_path, split)
            janelia_voxels.update(split_voxels)
        add_grouped_datasets(
            janelia_groups, "labels", EM_NUCLEUS_TRAIN_SAMPLES["janelia_nucleus"], weights=janelia_voxels,
            min_num_instances=EM_NUCLEUS_MIN_INSTANCES["janelia_nucleus"],
        )

    # 7. MICrONS nuclei (3D nuclei in ssTEM of the mouse visual cortex, 64 x 64 x 40 nm, automatic minnie65 labels)
    if "microns_nuclei" in EM_NUCLEUS_DATASETS:
        for z in z_slices:
            microns_kwargs = {
                "raw_key": "raw", "label_key": "labels/nuclei", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": nucleus_label_trafo(512, z_scale=MICRONS_NUCLEI_Z_SCALE),
                "pre_label_transform": connected_components(),
                "sampler": nucleus_sampler(min_num_instances=EM_NUCLEUS_MIN_INSTANCES["microns_nuclei"]),
            }
            for split, ds_list, n_samples in [
                ("train", train_ds, EM_NUCLEUS_TRAIN_SAMPLES["microns_nuclei"]), ("val", val_ds, 40),
            ]:
                paths = sorted(glob(os.path.join(input_path, "microns_nuclei", split, "*.h5")))
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=paths, label_paths=paths, n_samples=max(1, n_samples // n_z), **microns_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    waenet_root = os.path.join(input_path, "waenet_nucleus")

    # 8. WAE-Net dataset 1 nuclei (3D nuclei in FIB-SEM of pancreatic carcinoid cells, manual labels)
    if "waenet_fibsem_nucleus" in EM_NUCLEUS_DATASETS:
        waenet_path = os.path.join(waenet_root, "dataset_1.h5")
        with open_file(waenet_path, "r") as f:
            n_train = int(f["raw"].shape[0] * (1 - WAENET_VAL_FRACTION))
        waenet_rois = {"train": np.s_[:n_train, :, :], "val": np.s_[n_train:, :, :]}
        for z in z_slices:
            waenet_kwargs = {
                "raw_key": "raw", "label_key": "labels", "patch_shape": (z, *patch_shape), "ndim": 3,
                **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
                "label_transform2": nucleus_label_trafo(512),
                "pre_label_transform": connected_components(),
                "sampler": nucleus_sampler(),
            }
            for split, ds_list, n_samples in [
                ("train", train_ds, EM_NUCLEUS_TRAIN_SAMPLES["waenet_fibsem_nucleus"]), ("val", val_ds, 40),
            ]:
                ds_list.append(
                    UniDataWrapper(
                        torch_em.default_segmentation_dataset(
                            raw_paths=waenet_path, label_paths=waenet_path, rois=waenet_rois[split],
                            n_samples=max(1, n_samples // n_z), **waenet_kwargs,
                        ),
                        source_ndim=3, group_key=(3, z),
                    )
                )

    # 9. WAE-Net dataset 3 nuclei (2D nucleus in serial ssTEM sections of one interphase BON cell, manual labels) and
    # 10. WAE-Net datasets 4-7 nuclei (2D nucleus of the central cell in TEM images of lymphocytes and B cells, manual
    # labels; the loss ignores the pixels outside the central cell band, WAENET_NUCLEUS_IGNORE_LABEL)
    for name, prefix in [("waenet_sstem_nucleus", "dataset_3"), ("waenet_tem_nucleus", "tem")]:
        if name not in EM_NUCLEUS_DATASETS:
            continue
        waenet_kwargs = {
            "raw_key": "raw", "label_key": "labels", "patch_shape": (1, *patch_shape),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
            "label_transform2": nucleus_label_trafo(512, ignore_label=WAENET_NUCLEUS_IGNORE_LABEL),
            "pre_label_transform": connected_components(WAENET_NUCLEUS_IGNORE_LABEL),
            "sampler": nucleus_sampler(WAENET_NUCLEUS_IGNORE_LABEL),
        }
        for split, ds_list, n_samples in [("train", train_ds, EM_NUCLEUS_TRAIN_SAMPLES[name]), ("val", val_ds, 40)]:
            waenet_path = os.path.join(waenet_root, f"{prefix}_{split}.h5")
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=waenet_path, label_paths=waenet_path, n_samples=n_samples, **waenet_kwargs
                    ),
                    source_ndim=2,
                )
            )

    # 11. Platynereis nuclei (3D nuclei in SBEM of a Platynereis larva, about 80 x 80 x 100 nm, manual instance labels)
    # NOTE: Only the valid ROI of each volume trains, which removes the halo of unlabeled voxels.
    if "platynereis_nuclei" in EM_NUCLEUS_DATASETS:
        platy_groups = {split: platynereis_nuclei_split_rois(input_path, split) for split in ("train", "val")}
        excluded = ["raw_transform", "label_transform2", "sampler"]
        for z in z_slices:
            for split, ds_list, n_samples in [
                ("train", train_ds, EM_NUCLEUS_TRAIN_SAMPLES["platynereis_nuclei"]), ("val", val_ds, 40),
            ]:
                n_volumes = sum(len(paths) for paths, _ in platy_groups[split].values())
                for side, (paths, rois) in platy_groups[split].items():
                    platy_kwargs = {
                        "raw_key": "volumes/raw", "label_key": "volumes/labels/nucleus_instance_labels",
                        "patch_shape": (z, side, side), "ndim": 3,
                        **{k: v for k, v in kwargs.items() if k not in excluded},
                        "raw_transform": _resize_raw_to_512 if side < 512 else kwargs["raw_transform"],
                        "label_transform2": nucleus_label_trafo(side, z_scale=PLATYNEREIS_NUCLEI_Z_SCALE),
                        "pre_label_transform": connected_components(),
                        "sampler": nucleus_sampler(min_num_instances=EM_NUCLEUS_MIN_INSTANCES["platynereis_nuclei"]),
                    }
                    ds_list.append(
                        UniDataWrapper(
                            torch_em.default_segmentation_dataset(
                                raw_paths=paths, label_paths=paths, rois=rois,
                                n_samples=max(len(paths), round(n_samples * len(paths) / n_volumes) // n_z),
                                **platy_kwargs,
                            ),
                            source_ndim=3, group_key=(3, z),
                        )
                    )

    # 12. Tumor spheroid nuclei (2D nuclei in single SBF-SEM slices of a tumor spheroid, 50 nm, manual instance labels)
    # NOTE: The loss ignores the annotated cells without a nucleus label (TUMOR_SPHEROID_NUCLEI_IGNORE_LABEL).
    if "tumor_spheroid_nuclei" in EM_NUCLEUS_DATASETS:
        spheroid_paths = sorted(glob(os.path.join(input_path, "tumor_spheroid_nucleus", "*.h5")))
        spheroid_kwargs = {
            "raw_key": "raw", "label_key": "labels", "patch_shape": patch_shape, "ndim": 2,
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
            "label_transform2": nucleus_label_trafo(512, ignore_label=TUMOR_SPHEROID_NUCLEI_IGNORE_LABEL),
            "pre_label_transform": connected_components(TUMOR_SPHEROID_NUCLEI_IGNORE_LABEL),
            "sampler": nucleus_sampler(
                TUMOR_SPHEROID_NUCLEI_IGNORE_LABEL, EM_NUCLEUS_MIN_INSTANCES["tumor_spheroid_nuclei"]
            ),
        }
        for split, ds_list, n_samples in [
            ("train", train_ds, EM_NUCLEUS_TRAIN_SAMPLES["tumor_spheroid_nuclei"]), ("val", val_ds, 40),
        ]:
            paths = [
                path for path in spheroid_paths
                if (os.path.basename(path) in TUMOR_SPHEROID_NUCLEI_VAL_SLICES) == (split == "val")
            ]
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=paths, label_paths=paths, n_samples=n_samples, **spheroid_kwargs
                    ),
                    source_ndim=2,
                )
            )

    # 13. OpenOrganelle nuclei (3D nuclei in FIB-SEM of 17 OpenOrganelle tissue and single-cell volumes, 32-128 nm,
    # automatic labels)
    if "openorganelle_nucleus" in EM_NUCLEUS_DATASETS:
        openorganelle_groups, openorganelle_voxels = {}, {}
        for split in ("train", "val"):
            openorganelle_groups[split], split_voxels = openorganelle_nucleus_split_paths(input_path, split)
            openorganelle_voxels.update(split_voxels)
        add_grouped_datasets(
            openorganelle_groups, "labels", EM_NUCLEUS_TRAIN_SAMPLES["openorganelle_nucleus"],
            weights=openorganelle_voxels, min_num_instances=EM_NUCLEUS_MIN_INSTANCES["openorganelle_nucleus"],
        )

    full = np.s_[:, :, :]

    # 14. Cardiomyocyte TEM nuclei (2D nucleus in TEM micrographs of isolated cardiomyocytes, manual masks, downscaled)
    if "cardiomyocyte_tem_nucleus" in EM_NUCLEUS_DATASETS:
        cardio_paths = sorted(glob(os.path.join(input_path, CARDIOMYOCYTE_NUCLEUS_ROOT, "*.h5")))
        cardio_train, cardio_val = train_test_split(
            cardio_paths, test_size=CARDIOMYOCYTE_NUCLEUS_N_VAL, random_state=42
        )
        add_image_datasets(
            sorted(cardio_train), sorted(cardio_val), EM_NUCLEUS_TRAIN_SAMPLES["cardiomyocyte_tem_nucleus"]
        )

    # 15. microSAM EM nuclei (3D nuclei in the 32 x 512 x 512 train crop of the microSAM user study, manual labels)
    # NOTE: The whole crop trains (no validation leaf), and the user-study 'annotation' crop is not used.
    if "microsam_em_nuclei" in EM_NUCLEUS_DATASETS:
        microsam_path = os.path.join(input_path, MICROSAM_EM_NUCLEI_PATH)
        add_volume_datasets(
            [("train", microsam_path, "raw", "labels/nuclei", full, 1)], EM_NUCLEUS_TRAIN_SAMPLES["microsam_em_nuclei"],
            min_num_instances=EM_NUCLEUS_MIN_INSTANCES["microsam_em_nuclei"],
        )

    # 16. Perez nuclei (2D nuclei in SBF-SEM tiles of the mouse suprachiasmatic nucleus, manual masks)
    if "perez_nucleus" in EM_NUCLEUS_DATASETS:
        add_image_datasets(
            sorted(glob(os.path.join(input_path, PEREZ_NUCLEUS_ROOT, "train", "*.h5"))),
            sorted(glob(os.path.join(input_path, PEREZ_NUCLEUS_ROOT, "test", "*.h5"))),
            EM_NUCLEUS_TRAIN_SAMPLES["perez_nucleus"],
        )

    # 17. vEM nuclei (2D nuclei in array tomography SEM and 3D nuclei in FIB-SEM of cancer cells and tumor tissue,
    # proofread NucleoNet predictions)
    # NOTE: The loss ignores the unlabeled border band of the 2D images whose border nuclei were removed.
    if "vem_nuclei" in EM_NUCLEUS_DATASETS:
        vem_root = os.path.join(input_path, VEM_NUCLEI_ROOT)
        vem_3d = [os.path.join(vem_root, f"id_{i}.h5") for i in VEM_NUCLEI_3D_IDS]
        vem_2d = sorted(p for p in glob(os.path.join(vem_root, "id_*.h5")) if p not in vem_3d)
        vem_train, vem_val = train_test_split(vem_2d, test_size=1, random_state=42)
        add_image_datasets(
            sorted(vem_train), vem_val, EM_NUCLEUS_TRAIN_SAMPLES["vem_nuclei_2d"], ignore_label=VEM_NUCLEI_IGNORE_LABEL,
            min_num_instances=EM_NUCLEUS_MIN_INSTANCES["vem_nuclei"],
        )
        val_slab = np.s_[-VEM_NUCLEI_VAL_SLICES:, :, :]
        train_slab = np.s_[:-VEM_NUCLEI_VAL_SLICES, :, :]
        add_volume_datasets([
            ("train", vem_3d[0], "raw", "labels", full, 1), ("train", vem_3d[1], "raw", "labels", train_slab, 1),
            ("val", vem_3d[1], "raw", "labels", val_slab, 1),
        ], EM_NUCLEUS_TRAIN_SAMPLES["vem_nuclei_3d"], min_num_instances=EM_NUCLEUS_MIN_INSTANCES["vem_nuclei"])

    # 18. AIVE nuclei (3D nuclei in FIB-SEM overview volumes of cells, downscaled to 30 x 26 x 26 nm, manual labels)
    if "aive_nucleus" in EM_NUCLEUS_DATASETS:
        aive_paths = sorted(glob(os.path.join(input_path, AIVE_NUCLEUS_ROOT, "*.h5")))
        aive_volumes = [("train", path, "raw", "labels", full, AIVE_NUCLEUS_Z_SCALE) for path in aive_paths[:-1]]
        aive_volumes += [
            ("train", aive_paths[-1], "raw", "labels", np.s_[:-AIVE_NUCLEUS_VAL_SLICES, :, :], AIVE_NUCLEUS_Z_SCALE),
            ("val", aive_paths[-1], "raw", "labels", np.s_[-AIVE_NUCLEUS_VAL_SLICES:, :, :], AIVE_NUCLEUS_Z_SCALE),
        ]
        add_volume_datasets(aive_volumes, EM_NUCLEUS_TRAIN_SAMPLES["aive_nucleus"])

    # 19. Cerebellum nuclei (3D nuclei in FIB-SEM of the mouse cerebellum, downscaled to 20 nm, student labels)
    if "cerebellum_nucleus" in EM_NUCLEUS_DATASETS:
        cerebellum_path = os.path.join(input_path, CEREBELLUM_NUCLEUS_PATH)
        add_volume_datasets([
            ("train", cerebellum_path, "raw", "labels", np.s_[:CEREBELLUM_NUCLEUS_VAL_Z, :, :], 1),
            ("val", cerebellum_path, "raw", "labels", np.s_[CEREBELLUM_NUCLEUS_VAL_Z:, :, :], 1),
        ], EM_NUCLEUS_TRAIN_SAMPLES["cerebellum_nucleus"], ignore_label=CEREBELLUM_NUCLEUS_IGNORE_LABEL)

    # 20. Macrophage / A431 nuclei (3D nucleus in SBF-SEM of macrophages and an A431 cell, quick semantic annotations)
    if "macrophage_a431_nucleus" in EM_NUCLEUS_DATASETS:
        macrophage_volumes = []
        for path in sorted(glob(os.path.join(input_path, MACROPHAGE_NUCLEUS_ROOT, "*.h5"))):
            z_scale = MACROPHAGE_NUCLEUS_Z_SCALE
            if os.path.basename(path) == f"{MACROPHAGE_NUCLEUS_VAL_VOLUME}.h5":
                macrophage_volumes += [
                    ("train", path, "raw", "labels", np.s_[:MACROPHAGE_NUCLEUS_VAL_Z, :, :], z_scale),
                    ("val", path, "raw", "labels", np.s_[MACROPHAGE_NUCLEUS_VAL_Z:, :, :], z_scale),
                ]
            else:
                macrophage_volumes.append(("train", path, "raw", "labels", full, z_scale))
        add_volume_datasets(macrophage_volumes, EM_NUCLEUS_TRAIN_SAMPLES["macrophage_a431_nucleus"])

    _set_volume_min_instances([train_ds, val_ds], EM_NUCLEUS_SPARSE_VOLUMES, depth=2)
    return train_ds, val_ds


def get_dataloaders(
    input_path,
    label_trafo=None,
    batch_size=1,
    batch_size_2d=None,
    z_slices=None,
    dataset_choice="all",
    n_workers=32,
    with_boundaries=False,
):
    """Get generalist dataloaders for training UniSAM2.

    Args:
        input_path: Root path to the data.
        label_trafo: Label transform class (instantiated internally).
        batch_size: Default batch size (used for 3D groups).
        batch_size_2d: Optional larger batch size for 2D groups.
            Falls back to *batch_size* when not provided.
        z_slices: List of z-slice counts to use for 3D data (e.g. [2, 4, 6, 8]).
            Each value creates a separate dataset group so that batches have uniform z.
            Defaults to [8] (original behavior).
        dataset_choice: Which dataset domain to include. One of:
            - ``"lm"``: Light microscopy datasets only (2D + 3D LM).
            - ``"em"``: Electron microscopy datasets only.
            - ``"hp"``: Histopathology datasets only.
            - ``"em_mito"``: Mitochondria EM datasets only (not part of ``"all"``).
            - ``"em_nucleus"``: Nucleus EM datasets only (not part of ``"all"``).
            - ``"all"``: LM, EM and HP datasets (default).
        with_boundaries: The flag to append the full object-boundary channel to the automatic targets.
    """
    if dataset_choice not in DATASET_CHOICES:
        raise ValueError(f"Invalid dataset_choice: {dataset_choice!r}. Expected one of {DATASET_CHOICES}.")

    if label_trafo is None:
        from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform
        label_trafo = GeodesicHybridDistanceTransform
    if with_boundaries:
        label_trafo = partial(label_trafo, with_boundaries=True)

    if z_slices is None:
        z_slices = [8]
    if batch_size_2d is None:
        batch_size_2d = batch_size

    # Some common elements for all datasets.
    patch_shape = (512, 512)

    kwargs = {
        "raw_transform": _identity,
        "label_transform2": label_trafo(),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.float32,
    }

    train_ds, val_ds = [], []

    if dataset_choice in ("lm", "all"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "all"):
        em_train, em_val = _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(em_train)
        val_ds.extend(em_val)

    if dataset_choice in ("hp", "all"):
        hp_train, hp_val = _get_hp_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(hp_train)
        val_ds.extend(hp_val)

    if dataset_choice == "em_mito":
        mito_train, mito_val = _get_em_mito_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(mito_train)
        val_ds.extend(mito_val)

    if dataset_choice == "em_nucleus":
        nuc_train, nuc_val = _get_em_nucleus_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(nuc_train)
        val_ds.extend(nuc_val)

    _configure_training_normalization(train_ds, val_ds)

    # Finally, we prepare a 'ConcatDataset' for all the available datasets.
    train_ds = ConcatDataset(*train_ds)
    val_ds = ConcatDataset(*val_ds)

    # Determine per-group batch sizes for the sampler.
    batch_size_per_group = None
    if batch_size_2d != batch_size:
        batch_size_per_group = {2: batch_size_2d}

    # And prepare the dataloaders for them.
    train_loader = _prepare_data_loader(
        train_ds, batch_size=batch_size, shuffle=True,
        batch_size_per_group=batch_size_per_group, num_workers=n_workers,
    )
    val_loader = _prepare_data_loader(
        val_ds, batch_size=batch_size, shuffle=False,
        batch_size_per_group=batch_size_per_group, num_workers=n_workers, deterministic=True,
    )

    return train_loader, val_loader


def get_interactive_dataloaders(
    input_path,
    batch_size=1,
    batch_size_2d=None,
    z_slices=None,
    dataset_choice="all",
    n_workers=32,
):
    """Get generalist dataloaders for SAM2 interactive segmentation training.

    Identical dataset composition to :func:`get_dataloaders` but returns raw
    integer instance labels (``label_dtype=torch.int64``) instead of distance
    transforms. Used with :class:`micro_sam.v2.training.ConvertToSam2VideoBatch`.

    Args:
        input_path: Root path to the generalist training data.
        batch_size: Default batch size (used for 3D groups).
        batch_size_2d: Optional larger batch size for 2D groups.
            Falls back to *batch_size* when not provided.
        z_slices: List of z-slice counts for 3D data (e.g. [8]).
            Defaults to [8].
        dataset_choice: Which dataset domain to include - ``"lm"``, ``"em"``,
            ``"hp"``, ``"em_mito"``, ``"em_nucleus"``, or ``"all"`` (default).
        n_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if dataset_choice not in DATASET_CHOICES:
        raise ValueError(f"Invalid dataset_choice: {dataset_choice!r}. Expected one of {DATASET_CHOICES}.")

    if z_slices is None:
        z_slices = [8]
    if batch_size_2d is None:
        batch_size_2d = batch_size

    train_ds, val_ds = _build_interactive_datasets(input_path, z_slices, dataset_choice)

    batch_size_per_group = None
    if batch_size_2d != batch_size:
        batch_size_per_group = {2: batch_size_2d}

    train_loader = _prepare_data_loader(
        train_ds, batch_size=batch_size, shuffle=True,
        batch_size_per_group=batch_size_per_group, num_workers=n_workers,
    )
    val_loader = _prepare_data_loader(
        val_ds, batch_size=batch_size, shuffle=False,
        batch_size_per_group=batch_size_per_group, num_workers=n_workers, deterministic=True,
    )

    return train_loader, val_loader


def _build_automatic_datasets(input_path, z_slices, dataset_choice, with_boundaries=False):
    """Build train/val ConcatDatasets for automatic UniSAM2 training.

    Separated from :func:`get_dataloaders` so that each DDP rank can
    independently construct its own dataset (required by
    :class:`DistributedUniBatchSampler`).

    Returns:
        Tuple of (train_ds, val_ds) as :class:`ConcatDataset` instances.
    """
    from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform

    patch_shape = (512, 512)
    label_trafo = GeodesicHybridDistanceTransform
    if with_boundaries:
        label_trafo = partial(label_trafo, with_boundaries=True)

    kwargs = {
        "raw_transform": _identity,
        "label_transform2": label_trafo(),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.float32,
    }

    train_ds, val_ds = [], []

    if dataset_choice in ("lm", "all"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "all"):
        em_train, em_val = _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(em_train)
        val_ds.extend(em_val)

    if dataset_choice in ("hp", "all"):
        hp_train, hp_val = _get_hp_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(hp_train)
        val_ds.extend(hp_val)

    if dataset_choice == "em_mito":
        mito_train, mito_val = _get_em_mito_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(mito_train)
        val_ds.extend(mito_val)

    if dataset_choice == "em_nucleus":
        nuc_train, nuc_val = _get_em_nucleus_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(nuc_train)
        val_ds.extend(nuc_val)

    _configure_training_normalization(train_ds, val_ds)
    return ConcatDataset(*train_ds), ConcatDataset(*val_ds)


def _build_interactive_datasets(input_path, z_slices, dataset_choice):
    """Build train/val ConcatDatasets for interactive SAM2 training.

    Separated from :func:`get_interactive_dataloaders` so that each DDP rank
    can independently construct its own dataset (required by
    :class:`DistributedUniBatchSampler`).

    Returns:
        Tuple of (train_ds, val_ds) as :class:`ConcatDataset` instances.
    """
    patch_shape = (512, 512)

    kwargs = {
        "raw_transform": _identity,
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.int64,
        "label_transform2": _instance_labels,
    }

    train_ds, val_ds = [], []

    if dataset_choice in ("lm", "all"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo=None)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "all"):
        em_train, em_val = _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo=None)
        train_ds.extend(em_train)
        val_ds.extend(em_val)

    if dataset_choice in ("hp", "all"):
        hp_train, hp_val = _get_hp_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo=None)
        train_ds.extend(hp_train)
        val_ds.extend(hp_val)

    if dataset_choice == "em_mito":
        mito_train, mito_val = _get_em_mito_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo=None)
        train_ds.extend(mito_train)
        val_ds.extend(mito_val)

    if dataset_choice == "em_nucleus":
        nuc_train, nuc_val = _get_em_nucleus_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo=None)
        train_ds.extend(nuc_train)
        val_ds.extend(nuc_val)

    _configure_training_normalization(train_ds, val_ds)

    # Cap each validation dataset to N_SAMPLES_VAL random samples so the per-epoch
    # validation pass stays cheap (train datasets are left at full size).
    for w in val_ds:
        w.max_samples = N_SAMPLES_VAL

    return ConcatDataset(*train_ds), ConcatDataset(*val_ds)


def _build_joint_datasets(
    input_path, z_slices, dataset_choice, distance_type="geodesic", with_boundaries=False, label_trafo_threads=1,
):
    """Build train/val datasets for joint interactive + automatic SAM2 training.

    Labels have ``[instance_ids, fg, d_z, d_y, d_x, boundaries?]`` channels.

    - Channel 0 (int64): instance IDs -> interactive branch via ``ConvertToSam2VideoBatch``.
    - Channels 1 onward (float32): foreground + directed distances and optional boundary -> automatic branch via
      ``DirectedDistanceLoss``.

    Unlike building two separate datasets, this shares a single data pipeline so both
    branches always see the same image patch.

    Args:
        input_path: Root path to the generalist training data.
        z_slices: Z-slice counts for 3D groups.
        dataset_choice: ``"lm"``, ``"em"``, ``"hp"``, ``"em_mito"``, ``"em_nucleus"``, or ``"all"``.
        distance_type: Which directed distance target the automatic branch regresses.
            ``"geodesic"`` uses :class:`_JointGeodesicLabelTransform`, ``"directed"`` uses
            :class:`_JointLabelTransform`.
        with_boundaries: The flag to append the full object-boundary target for the automatic branch.
        label_trafo_threads: Threads per loader worker that process the objects of one patch in parallel
            in the distance transform. Only pays off when the node has more cores than loader workers.

    Returns:
        Tuple of (train_ds, val_ds) as :class:`ConcatDataset` instances.
    """
    if distance_type not in ("geodesic", "directed"):
        raise ValueError(f"Invalid distance_type: {distance_type!r}. Expected 'geodesic' or 'directed'.")

    patch_shape = (512, 512)
    # The boundary target adds a sixth channel after the instance IDs, foreground, and distances.
    label_trafo = partial(
        _JointGeodesicLabelTransform if distance_type == "geodesic" else _JointLabelTransform,
        with_boundaries=with_boundaries,
        n_threads=label_trafo_threads,
    )

    kwargs = {
        "raw_transform": _identity,
        "label_transform2": label_trafo(),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.float32,
    }

    train_ds, val_ds = [], []

    if dataset_choice in ("lm", "all"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "all"):
        em_train, em_val = _get_em_datasets(
            input_path, patch_shape, z_slices, kwargs, label_trafo,
            _em_label_trafo=_joint_em_cell_label_trafo,
        )
        train_ds.extend(em_train)
        val_ds.extend(em_val)

    if dataset_choice in ("hp", "all"):
        hp_train, hp_val = _get_hp_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(hp_train)
        val_ds.extend(hp_val)

    if dataset_choice == "em_mito":
        mito_train, mito_val = _get_em_mito_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(mito_train)
        val_ds.extend(mito_val)

    if dataset_choice == "em_nucleus":
        nuc_train, nuc_val = _get_em_nucleus_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(nuc_train)
        val_ds.extend(nuc_val)

    _configure_training_normalization(train_ds, val_ds)

    # Cap each validation dataset to N_SAMPLES_VAL random samples so the per-epoch
    # validation pass stays cheap (matches the interactive builder; train datasets are full size).
    for w in val_ds:
        w.max_samples = N_SAMPLES_VAL

    return ConcatDataset(*train_ds), ConcatDataset(*val_ds)
