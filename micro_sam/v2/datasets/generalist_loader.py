import os
import re
import json
import random
from glob import glob
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split

import torch

from elf.io import open_file

import torch_em
from torch_em.transform import get_augmentations
from torch_em.data import datasets, MinInstanceSampler, ConcatDataset

from .wrapper import UniDataWrapper
from .sampler import UniBatchSampler, _build_group_map
from ..transforms.raw import (
    _identity, _cellpose_raw_trafo, _to_8bit, _normalize_percentile, _resize_raw_to_512, _resize_to_512,
    _enseg_green_channel, _xenium_cell_channels, _pan_multiplex_tissuenet_order, _cvz_cell_channels,
    _minmax_raw_trafo,
    get_random_percentile_normalization,
)
from ..transforms.labels import (
    _em_cell_label_trafo, _joint_em_cell_label_trafo, _background_id_label_trafo,
    _plantseg_label_trafo, _astih_pre_label_transform, _instance_labels,
    _ignore_missing_raw_trafo, _ignore_unlabelled_blobs_trafo, _labels_to_uint32, _drop_oversized_label_trafo,
    _JointLabelTransform, _JointGeodesicLabelTransform,
)

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


def _configure_training_normalization(train_datasets, val_datasets):
    """Enable random percentile augmentation for training and deterministic 2nd/98th validation."""
    _set_percentile_normalization(
        train_datasets, lower_percentile_bounds=TRAIN_LOWER_PERCENTILE_BOUNDS,
    )
    _set_percentile_normalization(
        val_datasets, lower_percentile_bounds=VALIDATION_LOWER_PERCENTILE_BOUNDS,
    )
    _set_max_sampling_attempts([train_datasets, val_datasets], MAX_SAMPLING_ATTEMPTS)


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
        (bd_train_r, bd_train_l, train_ds, 200), (bd_val_r, bd_val_l, val_ds, 50)
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
        **{**kwargs, "transform": partial(_random_resize_and_pad_trafo, patch_shape=patch_shape)},
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
        **{**kwargs, "transform": partial(_random_resize_and_pad_trafo, patch_shape=patch_shape)},
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
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
    }
    for stain in CELLBINDB_STAINS:
        cellbindb_raw, cellbindb_labels = datasets.cellbindb.get_cellbindb_paths(
            path=os.path.join(input_path, "cellbindb"), data_choice=stain,
        )
        (cb_train_r, cb_val_r, _), (cb_train_l, cb_val_l, _) = _train_val_test_split(cellbindb_raw, cellbindb_labels)
        for raws, labs, ds_list, n_samples in [
            (cb_train_r, cb_train_l, train_ds, 150), (cb_val_r, cb_val_l, val_ds, 30)
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
            "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape),
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
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_flywing_dataset(split="train", n_samples=300, **flywing_kwargs), source_ndim=2)
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
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
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
        "transform": partial(_random_resize_and_pad_trafo, patch_shape=patch_shape),
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

    return train_ds, val_ds


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
    # The AxonEM and FIB-25 folders copy the complete public releases, including the volumes the direct axonem and
    # fib25 loaders hold out for testing, so they train through those loaders only.
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
    base_kwargs["label_transform"] = _ensure_native_byte_order

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
    def _compute_platy_rois(root, sample_ids, ignore_label, file_template, label_key):
        cache_path = os.path.join(root, f"_roi_cache_{'_'.join(map(str, sample_ids))}.json")
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
            # Some slices hold missing tiles (exact-zero raw) that still carry labels; they become ignore.
            "transform": partial(
                _ignore_missing_raw_trafo, ignore_label=MISSING_RAW_IGNORE_LABEL, transform=get_augmentations(ndim=3)
            ),
            # uint8 labels cannot hold the ignore label, and torch_em recasts to the loaded dtype.
            "pre_label_transform": _labels_to_uint32,
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{
                k: v for k, v in kwargs.items()
                if k not in ["label_transform2", "sampler", "transform", "pre_label_transform"]
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
            "transform": partial(
                _ignore_missing_raw_trafo, ignore_label=MISSING_RAW_IGNORE_LABEL,
                transform=partial(
                    _ignore_unlabelled_blobs_trafo, ignore_label=MISSING_RAW_IGNORE_LABEL, min_area=20000,
                    transform=get_augmentations(ndim=3),
                ),
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0, MISSING_RAW_IGNORE_LABEL]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler", "transform"]},
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

# training_sample2 and validation_sample train in full, tstvol-520-1 is the blind in-domain test set. All three are
# at 8 nm; the training sample is only a smaller cube (250^3), not a coarser one.
EMNEURON_EXCLUDED_FOLDERS = ("AxonEM[H]-atum", "AxonEM[M]-sstem", "Fib-25-fib")
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
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
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
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
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
    nuinsseg_raw_paths, nuinsseg_label_paths = datasets.nuinsseg.get_nuinsseg_paths(
        path=os.path.join(input_path, "nuinsseg")
    )
    nuinsseg_train_raw, nuinsseg_val_raw, nuinsseg_train_labels, nuinsseg_val_labels = train_test_split(
        nuinsseg_raw_paths, nuinsseg_label_paths, test_size=0.2, random_state=42,
    )
    nuinsseg_kwargs = {"patch_shape": patch_shape, "is_seg_dataset": False, "ndim": 2, "with_channels": True, **kwargs}
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
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
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
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
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
        (dsb_hp_train_r, dsb_hp_train_l, train_ds, 100), (dsb_hp_val_r, dsb_hp_val_l, val_ds, 25)
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
        (cb_he_train_r, cb_he_train_l, train_ds, 150), (cb_he_val_r, cb_he_val_l, val_ds, 30)
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, raw_key=None, label_paths=labs, label_key=None, n_samples=n_samples, **cb_he_kwargs,
                ), source_ndim=2,
            )
        )

    return train_ds, val_ds


def get_dataloaders(
    input_path,
    label_trafo=None,
    batch_size=1,
    batch_size_2d=None,
    z_slices=None,
    dataset_choice="all",
    n_workers=32,
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
            - ``"all"``: All datasets (default).
    """
    if dataset_choice not in ("lm", "em", "hp", "all"):
        raise ValueError(f"Invalid dataset_choice: {dataset_choice!r}. Expected 'lm', 'em', 'hp', or 'all'.")

    if label_trafo is None:
        from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform
        label_trafo = GeodesicHybridDistanceTransform

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
            ``"hp"``, or ``"all"`` (default).
        n_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if dataset_choice not in ("lm", "em", "hp", "all"):
        raise ValueError(f"Invalid dataset_choice: {dataset_choice!r}. Expected 'lm', 'em', 'hp', or 'all'.")

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


def _build_automatic_datasets(input_path, z_slices, dataset_choice):
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

    _configure_training_normalization(train_ds, val_ds)

    # Cap each validation dataset to N_SAMPLES_VAL random samples so the per-epoch
    # validation pass stays cheap (train datasets are left at full size).
    for w in val_ds:
        w.max_samples = N_SAMPLES_VAL

    return ConcatDataset(*train_ds), ConcatDataset(*val_ds)


def _build_joint_datasets(input_path, z_slices, dataset_choice, distance_type="geodesic", label_trafo_threads=1):
    """Build train/val datasets for joint interactive + automatic SAM2 training.

    Labels have **5 channels**: ``[instance_ids, fg, d_x, d_y, d_z]``.

    - Channel 0 (int64): instance IDs -> interactive branch via ``ConvertToSam2VideoBatch``.
    - Channels 1-4 (float32): foreground + directed distances -> automatic branch via
      ``DirectedDistanceLoss``.

    Unlike building two separate datasets, this shares a single data pipeline so both
    branches always see the same image patch.

    Args:
        input_path: Root path to the generalist training data.
        z_slices: Z-slice counts for 3D groups.
        dataset_choice: ``"lm"``, ``"em"``, ``"hp"``, or ``"all"``.
        distance_type: Which directed distance target the automatic branch regresses.
            ``"geodesic"`` uses :class:`_JointGeodesicLabelTransform`, ``"directed"`` uses
            :class:`_JointLabelTransform`.
        label_trafo_threads: Threads per loader worker that process the objects of one patch in parallel
            in the distance transform. Only pays off when the node has more cores than loader workers.

    Returns:
        Tuple of (train_ds, val_ds) as :class:`ConcatDataset` instances.
    """
    if distance_type not in ("geodesic", "directed"):
        raise ValueError(f"Invalid distance_type: {distance_type!r}. Expected 'geodesic' or 'directed'.")

    patch_shape = (512, 512)
    # Both default to instances=True -> 5-channel output.
    label_trafo = partial(
        _JointGeodesicLabelTransform if distance_type == "geodesic" else _JointLabelTransform,
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

    _configure_training_normalization(train_ds, val_ds)

    # Cap each validation dataset to N_SAMPLES_VAL random samples so the per-epoch
    # validation pass stays cheap (matches the interactive builder; train datasets are full size).
    for w in val_ds:
        w.max_samples = N_SAMPLES_VAL

    return ConcatDataset(*train_ds), ConcatDataset(*val_ds)
