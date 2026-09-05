import os
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
    _enseg_green_channel, _micro_bench_nuclei_channel, _xenium_cell_channels, _pan_multiplex_tissuenet_order,
    get_random_percentile_normalization,
)
from ..transforms.labels import (
    _em_cell_label_trafo, _joint_em_cell_label_trafo, _background_id_label_trafo,
    _plantseg_label_trafo, _astih_pre_label_transform, _instance_labels,
    _ignore_missing_raw_trafo, _ignore_unlabelled_blobs_trafo, _labels_to_uint32,
    _JointLabelTransform, _JointGeodesicLabelTransform,
)

# Cap on validation samples drawn per dataset, to keep the per-epoch validation pass cheap.
# Each access is a random crop (see UniDataWrapper.max_samples), so this is N random samples.
N_SAMPLES_VAL = 50

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


def _configure_training_normalization(train_datasets, val_datasets):
    """Enable random percentile augmentation for training and deterministic 2nd/98th validation."""
    _set_percentile_normalization(
        train_datasets, lower_percentile_bounds=TRAIN_LOWER_PERCENTILE_BOUNDS,
    )
    _set_percentile_normalization(
        val_datasets, lower_percentile_bounds=VALIDATION_LOWER_PERCENTILE_BOUNDS,
    )


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
    # split, so the validation set is the 68-image 'cyto' test split.
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
    def _get_cvz_dataset(stain_choice, split_choice):
        raw_paths, label_paths = datasets.cvz_fluo.get_cvz_fluo_paths(
            path=os.path.join(input_path, "cvz"), stain_choice=stain_choice,
        )
        train_raw_paths, test_raw_paths, train_label_paths, test_label_paths = train_test_split(
            raw_paths, label_paths, test_size=0.2, random_state=42,
        )
        ds = torch_em.default_segmentation_dataset(
            raw_paths=train_raw_paths if split_choice == "train" else test_raw_paths,
            raw_key=None,
            label_paths=train_label_paths if split_choice == "train" else test_label_paths,
            label_key=None,
            is_seg_dataset=False,
            patch_shape=patch_shape,
            raw_transform=_to_8bit,
            n_samples=200 if split_choice == "train" else 100,
            **{k: v for k, v in kwargs.items() if k != "raw_transform"}
        )
        return ds

    train_ds.append(UniDataWrapper(_get_cvz_dataset("cell", "train"), source_ndim=2))
    train_ds.append(UniDataWrapper(_get_cvz_dataset("dapi", "train"), source_ndim=2))
    val_ds.append(UniDataWrapper(_get_cvz_dataset("cell", "test"), source_ndim=2))
    val_ds.append(UniDataWrapper(_get_cvz_dataset("dapi", "test"), source_ndim=2))

    # 3. DSB dataset (nucleus segmentation in fluorescence images)
    dsb_kwargs = {"path": os.path.join(input_path, "dsb"), "patch_shape": patch_shape, "domain": "fluo", **kwargs}

    train_ds.append(
        UniDataWrapper(datasets.get_dsb_dataset(split="train", n_samples=600, **dsb_kwargs), source_ndim=2)
    )
    val_ds.append(UniDataWrapper(datasets.get_dsb_dataset(split="test", **dsb_kwargs), source_ndim=2))

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

    def _get_embedseg_datasets(split_choice, z):
        if split_choice == "train":
            names = [
                "Mouse-Organoid-Cells-CBG", "Mouse-Skull-Nuclei-CBG",
                "Platynereis-ISH-Nuclei-CBG", "Platynereis-Nuclei-CBG",
            ]
        else:  # Only two datasets have the test split.
            names = ["Mouse-Skull-Nuclei-CBG", "Platynereis-ISH-Nuclei-CBG"]

        all_embedseg_datasets = [
            datasets.get_embedseg_dataset(
                path=os.path.join(input_path, "embedseg"),
                name=name,
                patch_shape=(z, *patch_shape),
                split=split_choice,
                raw_transform=_to_8bit,
                n_samples=max(1, 200 // n_z),
                label_transform2=(
                    label_trafo(sampling=embedseg_sampling[name])
                    if label_trafo is not None else kwargs.get("label_transform2")
                ),
                **{k: v for k, v in kwargs.items() if k not in ["raw_transform", "label_transform2"]}
            ) for name in names
        ]
        return all_embedseg_datasets

    for z in z_slices:
        train_ds.extend(
            [UniDataWrapper(ds, source_ndim=3, group_key=(3, z)) for ds in _get_embedseg_datasets("train", z)]
        )
        val_ds.extend(
            [UniDataWrapper(ds, source_ndim=3, group_key=(3, z)) for ds in _get_embedseg_datasets("test", z)]
        )

    # 5. NIS3D (nucleus segmentation in light-sheet microscopy images)
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
                    **nis3d_kwargs,
                ), source_ndim=3, group_key=(3, z),
            )
        )

    # 6. PlantSeg (cell segmentation in confocal microscopy images)
    for z in z_slices:
        plantseg_kwargs = {
            "path": os.path.join(input_path, "plantseg"),
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 200 // n_z),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[-1, 1]),
            **{k: v for k, v in kwargs.items() if k not in ["sampler", "label_transform2"]}
        }

        # NOTE: Only root trains. In ovules the 'label' key folds 1.2-9.2 % annotator-ignore regions into
        # background, so it is held out for evaluation; nuclei is 98-99 % background and redundant with gonuclear.
        for ds_name in ["root"]:
            _plantseg_trafo = partial(
                _plantseg_label_trafo, data=ds_name,
                label_trafo=label_trafo() if label_trafo is not None else kwargs.get("label_transform2"),
            )
            train_ds.append(
                UniDataWrapper(
                    datasets.get_plantseg_dataset(
                        name=ds_name, split="train",
                        label_transform2=_plantseg_trafo,
                        **plantseg_kwargs
                    ), source_ndim=3, group_key=(3, z),
                )
            )
            val_ds.append(
                UniDataWrapper(
                    datasets.get_plantseg_dataset(
                        name=ds_name, split="val",
                        label_transform2=_plantseg_trafo,
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
    deepbacs_kwargs = {
        "path": os.path.join(input_path, "deepbacs"),
        "patch_shape": patch_shape,
        "bac_type": "mixed",
        "raw_transform": _to_8bit,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    train_ds.append(
        UniDataWrapper(datasets.get_deepbacs_dataset(split="train", n_samples=400, **deepbacs_kwargs), source_ndim=2)
    )

    # The 'mixed' archive pools S. aureus, E. coli and B. subtilis. 'e_coli_stationary' is a separate acquisition
    # of stationary-phase cells and is the only other bac_type torch-em implements. It has no val split.
    deepbacs_stat_kwargs = {**deepbacs_kwargs, "bac_type": "e_coli_stationary"}
    train_ds.append(
        UniDataWrapper(
            datasets.get_deepbacs_dataset(split="train", n_samples=200, **deepbacs_stat_kwargs), source_ndim=2
        )
    )
    val_ds.append(
        UniDataWrapper(datasets.get_deepbacs_dataset(split="test", n_samples=200, **deepbacs_kwargs), source_ndim=2)
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
    omnipose_kwargs = {
        "path": os.path.join(input_path, "omnipose"),
        "patch_shape": patch_shape,
        "raw_transform": _to_8bit,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    train_ds.append(
        UniDataWrapper(datasets.get_omnipose_dataset(split="train", n_samples=500, **omnipose_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_omnipose_dataset(split="test", n_samples=200, **omnipose_kwargs), source_ndim=2)
    )

    # 13. CTC (cell segmentation from Cell Tracking Challenge)
    # NOTE: CTC only supports the train split. No validation data is added for CTC.
    ctc_kwargs = {
        "path": os.path.join(input_path, "ctc"),
        "patch_shape": (1, *patch_shape),
        "raw_transform": _to_8bit,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    for name in datasets.ctc.CTC_CHECKSUMS["train"].keys():
        if name in ["Fluo-N2DH-GOWT1", "Fluo-N2DL-HeLa"]:
            continue

        train_ds.append(
            UniDataWrapper(
                datasets.get_ctc_segmentation_dataset(dataset_name=name, split="train", **ctc_kwargs), source_ndim=2,
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
    # NOTE: No native val split is used: the 146-image test split is kept blind and the 1063 train images are
    # split 80/20 instead. The instance labels are connected components of a binary mask, but the cells are
    # separated cleanly in practice (only 0.8 % of the labelled area sits in objects larger than 2.5x the median).
    bccd_paths = datasets.bccd.get_bccd_paths(path=os.path.join(input_path, "bccd"), split="train")
    bccd_train, bccd_val = train_test_split(bccd_paths, test_size=0.2, random_state=42)
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
    bitdepth_kwargs = {"path": os.path.join(input_path, "bitdepth_nucseg"), "patch_shape": patch_shape, **kwargs}
    bitdepth_raw, bitdepth_labels = datasets.bitdepth_nucseg.get_bitdepth_nucseg_paths(
        path=os.path.join(input_path, "bitdepth_nucseg")
    )
    bd_train_r, bd_val_r, bd_train_l, bd_val_l = train_test_split(
        bitdepth_raw, bitdepth_labels, test_size=0.2, random_state=42,
    )
    del bitdepth_kwargs["path"]
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
    # NOTE: Native images are only 345x382, so they are randomly upscaled and padded to the patch shape.
    bmgd_paths = datasets.bmgd.get_bmgd_paths(path=os.path.join(input_path, "bmgd"))
    bmgd_train, bmgd_val = train_test_split(bmgd_paths, test_size=0.2, random_state=42)
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

    # 19. Cardioblast nuclei (nucleus segmentation in confocal fluorescence time-lapse projections)
    cardio_kwargs = {
        "path": os.path.join(input_path, "cardioblast_nuclei"), "patch_shape": patch_shape, **kwargs
    }
    train_ds.append(
        UniDataWrapper(
            datasets.get_cardioblast_nuclei_dataset(split="train", n_samples=200, **cardio_kwargs), source_ndim=2
        )
    )
    val_ds.append(
        UniDataWrapper(
            datasets.get_cardioblast_nuclei_dataset(split="test", n_samples=50, **cardio_kwargs), source_ndim=2
        )
    )

    # 20. Cell-ACDC (yeast cell segmentation in phase contrast time-lapse)
    # NOTE: Native fields are only ~200x300, so they are randomly upscaled and padded to the patch shape.
    cell_acdc_kwargs = {
        "path": os.path.join(input_path, "cell_acdc"), "patch_shape": (1, 200, 200),
        **{**kwargs, "transform": partial(_random_resize_and_pad_trafo, patch_shape=patch_shape)},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_cell_acdc_dataset(n_samples=200, **cell_acdc_kwargs), source_ndim=2)
    )

    # 21. cellapp (cell segmentation in transmitted light images)
    cellapp_kwargs = {"path": os.path.join(input_path, "cellapp"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_cellapp_dataset(split="train", n_samples=200, **cellapp_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_cellapp_dataset(split="test", n_samples=50, **cellapp_kwargs), source_ndim=2)
    )

    # 22. CellBinDB (nucleus segmentation in DAPI, ssDNA, mIF and H&E images)
    cellbindb_raw, cellbindb_labels = datasets.cellbindb.get_cellbindb_paths(
        path=os.path.join(input_path, "cellbindb")
    )
    cb_train_r, cb_val_r, cb_train_l, cb_val_l = train_test_split(
        cellbindb_raw, cellbindb_labels, test_size=0.2, random_state=42,
    )
    cellbindb_kwargs = {"patch_shape": patch_shape, "is_seg_dataset": False, "ndim": 2, **kwargs}
    for raws, labs, ds_list, n_samples in [
        (cb_train_r, cb_train_l, train_ds, 400), (cb_val_r, cb_val_l, val_ds, 50)
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
    cellular_paths = datasets.cellular.get_cellular_paths(path=os.path.join(input_path, "cellular"))
    cellular_train, cellular_val = train_test_split(cellular_paths, test_size=0.2, random_state=42)
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
    # NOTE: Only 2-3 cells per image, so the shared 3-instance sampler would reject nearly every patch.
    cisd_kwargs = {
        "path": os.path.join(input_path, "cisd"), "patch_shape": patch_shape, "mode": "center_slice",
        **{**kwargs, "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0])},
    }
    train_ds.append(UniDataWrapper(datasets.get_cisd_dataset(n_samples=200, **cisd_kwargs), source_ndim=2))

    # 25. DeepSeas (stem cell segmentation in phase contrast images)
    # NOTE: The source masks are binary and the labelling is partial, so touching cells merge and many
    # visible cells are unlabelled. Kept at a modest sample count for that reason.
    # NOTE: The masks are binary, so the shared 3-instance sampler sees a single foreground id and rejects
    # every patch. The instances only appear after the connected-component label transform, which runs later.
    deepseas_kwargs = {
        "path": os.path.join(input_path, "deepseas"), "patch_shape": patch_shape,
        **{**kwargs, "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0])},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_deepseas_dataset(split="train", n_samples=300, **deepseas_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_deepseas_dataset(split="test", n_samples=50, **deepseas_kwargs), source_ndim=2)
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

    # 27. DIC-HepG2 (cell segmentation in DIC images)
    dic_kwargs = {"path": os.path.join(input_path, "dic_hepg2"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_dic_hepg2_dataset(split="train", n_samples=300, **dic_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_dic_hepg2_dataset(split="val", n_samples=50, **dic_kwargs), source_ndim=2)
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
                datasets.get_pnas_arabidopsis_dataset(**pnas_kwargs), source_ndim=3, group_key=(3, z),
            )
        )

    # 31. EpiCure (epithelial cell segmentation in fluorescence membrane movies)
    # NOTE: Four model systems (Drosophila notum and histoblasts, zebrafish telencephalon, quail gastrula).
    # The frames are exhaustively labelled and very dense, 250-370 cells per patch. Patch shape is
    # (1, 512, 512) because the movies are stored as time series, so one frame is drawn per sample.
    epicure_kwargs = {
        "path": os.path.join(input_path, "epicure"), "patch_shape": (1, *patch_shape), **kwargs
    }
    train_ds.append(
        UniDataWrapper(datasets.get_epicure_dataset(n_samples=500, **epicure_kwargs), source_ndim=2)
    )

    # 32. CartoCell (3D cell segmentation in confocal epithelial cysts)
    # NOTE: The on-disk layout is 'CartoCell/{train_M1,train_M2}/{x,y}', which the torch-em loader does not expect,
    # so the paths are given explicitly. Native volumes are only ~84-128 px in XY, so an 80x80 crop is always
    # resized up to 512x512 rather than zero-padded.
    cartocell_root = os.path.join(input_path, "cartocell", "CartoCell")
    cartocell_raw = sorted(
        glob(os.path.join(cartocell_root, "train_M1", "x", "*.tif"))
        + glob(os.path.join(cartocell_root, "train_M2", "x", "*.tif"))
    )
    cartocell_labels = [p.replace(os.sep + "x" + os.sep, os.sep + "y" + os.sep) for p in cartocell_raw]
    assert cartocell_raw and all(os.path.exists(p) for p in cartocell_labels)
    cc_train_r, cc_val_r, cc_train_l, cc_val_l = train_test_split(
        cartocell_raw, cartocell_labels, test_size=0.2, random_state=42,
    )

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

    # 34. CShaper (3D cell segmentation in confocal C. elegans embryo membranes)
    # NOTE: Volumes are only ~285x131 in XY, so a 128x128 crop is resized up to 512x512.
    for z in z_slices:
        cshaper_kwargs = {
            "path": os.path.join(input_path, "cshaper"),
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
                datasets.get_cshaper_dataset(split="train", **cshaper_kwargs), source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_cshaper_dataset(split="val", **cshaper_kwargs), source_ndim=3, group_key=(3, z),
            )
        )

    # 35. U2OS (nucleus segmentation in Hoechst fluorescence images)
    u20s_kwargs = {"path": os.path.join(input_path, "u20s"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(UniDataWrapper(datasets.get_u20s_dataset(n_samples=300, **u20s_kwargs), source_ndim=2))

    # 36. IFNuclei (nucleus segmentation in immunofluorescence images)
    ifnuclei_kwargs = {"path": os.path.join(input_path, "ifnuclei"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_ifnuclei_dataset(split="train", n_samples=200, **ifnuclei_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_ifnuclei_dataset(split="test", n_samples=50, **ifnuclei_kwargs), source_ndim=2)
    )

    # 37. VICAR (cell segmentation in quantitative phase imaging of five cell lines)
    vicar_kwargs = {"path": os.path.join(input_path, "vicar"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(UniDataWrapper(datasets.get_vicar_dataset(n_samples=300, **vicar_kwargs), source_ndim=2))

    # 38. HeLaCytoNuc (nucleus segmentation in fluorescence images)
    # NOTE: The raw is RGB with red cytoplasm, blue nuclei and an unused green channel. The loader's own
    # raw_channel="nuclei" selects the blue channel, which is the one the nucleus labels correspond to.
    hela_kwargs = {
        "path": os.path.join(input_path, "hela_cytonuc"), "patch_shape": patch_shape,
        "raw_channel": "nuclei", "label_choice": "nuclei", **kwargs,
    }
    train_ds.append(
        UniDataWrapper(datasets.get_hela_cytonuc_dataset(split="train", n_samples=400, **hela_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_hela_cytonuc_dataset(split="val", n_samples=50, **hela_kwargs), source_ndim=2)
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
    orgline_kwargs = {"path": os.path.join(input_path, "orgline"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_orgline_dataset(split="train", n_samples=500, **orgline_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_orgline_dataset(split="val", n_samples=50, **orgline_kwargs), source_ndim=2)
    )

    # 41. OrganoID (pancreatic organoid segmentation in brightfield culture wells)
    organoid_kwargs = {"path": os.path.join(input_path, "organoid"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_organoid_dataset(split="train", source="original", n_samples=200,
                                                     **organoid_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_organoid_dataset(split="val", source="original", n_samples=50,
                                                     **organoid_kwargs), source_ndim=2)
    )

    # 42. EnSeg (enteric neuron segmentation in immunofluorescence whole-mount myenteric plexus)
    # NOTE: Stored RGB but only the green channel carries signal (maxima 41/251/43).
    enseg_kwargs = {
        "path": os.path.join(input_path, "enseg"), "patch_shape": patch_shape,
        "raw_transform": _enseg_green_channel,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"},
    }
    train_ds.append(UniDataWrapper(datasets.get_enseg_dataset(n_samples=200, **enseg_kwargs), source_ndim=2))

    # 43. LPC-NucSeg (nucleus segmentation in DNA fluorescence images)
    lpc_kwargs = {"path": os.path.join(input_path, "lpc_nucseg"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(UniDataWrapper(datasets.get_lpc_nucseg_dataset(n_samples=150, **lpc_kwargs), source_ndim=2))

    # 44. DCIS.COM nuclei (nucleus segmentation in spinning-disk confocal SiR-DNA images)
    dcis_kwargs = {"path": os.path.join(input_path, "dcis_com_nuclei"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_dcis_com_nuclei_dataset(split="train", n_samples=200, **dcis_kwargs),
                       source_ndim=2)
    )

    # 45. mCellSeg (cell segmentation in DIC and brightfield images of HEK-293T and HUVEC)
    mcellseg_kwargs = {"path": os.path.join(input_path, "mcellseg"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_mcellseg_dataset(split="train", val_fraction=0.2, n_samples=200,
                                                     **mcellseg_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_mcellseg_dataset(split="val", val_fraction=0.2, n_samples=50,
                                                     **mcellseg_kwargs), source_ndim=2)
    )

    # 46. TOIAM (bacteria segmentation in phase contrast time-lapse colonies)
    # NOTE: The colony grows over time, so density is bimodal: a 512 patch holds a median of 39 objects with
    # quartiles at 8 and 305. A 25-instance minimum keeps 56 % of patches and cuts the near-empty early frames.
    toiam_kwargs = {
        "path": os.path.join(input_path, "toiam"), "patch_shape": patch_shape,
        **{**kwargs, "sampler": MinInstanceSampler(min_num_instances=25, exclude_ids=[0])},
    }
    train_ds.append(UniDataWrapper(datasets.get_toiam_dataset(n_samples=400, **toiam_kwargs), source_ndim=2))

    # 47. Usiigaci (cell segmentation in phase contrast fibroblast images)
    usiigaci_kwargs = {"path": os.path.join(input_path, "usiigaci"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_usiigaci_dataset(split="train", n_samples=200, **usiigaci_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_usiigaci_dataset(split="val", n_samples=50, **usiigaci_kwargs), source_ndim=2)
    )

    # 48. YeastSAM (budding yeast segmentation in DIC images)
    yeastsam_kwargs = {"path": os.path.join(input_path, "yeastsam"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(UniDataWrapper(datasets.get_yeastsam_dataset(n_samples=100, **yeastsam_kwargs), source_ndim=2))

    # 49. YeastCellSeg (budding yeast segmentation in brightfield images)
    yeastcellseg_kwargs = {
        "path": os.path.join(input_path, "yeastcellseg"), "patch_shape": patch_shape,
        "segmentation_type": "instances", **kwargs,
    }
    train_ds.append(
        UniDataWrapper(datasets.get_yeastcellseg_dataset(n_samples=150, **yeastcellseg_kwargs), source_ndim=2)
    )

    # 50. micro-bench (nucleus segmentation in the OpenCell subset)
    # NOTE: Only 'opencell' is multi-object. Channel 0 is empty in all 1105 files and channel 1 is the varying
    # GFP-tagged protein, so channel 2, the nuclear counterstain, is the one to feed.
    micro_bench_root = os.path.join(input_path, "micro_bench")
    micro_bench_raw = sorted(glob(os.path.join(micro_bench_root, "images", "opencell", "*.tif")))
    micro_bench_labels = [
        os.path.join(micro_bench_root, "labels", "opencell", "instances", os.path.basename(p))
        for p in micro_bench_raw
    ]
    assert micro_bench_raw and all(os.path.exists(p) for p in micro_bench_labels)
    mb_train_r, mb_val_r, mb_train_l, mb_val_l = train_test_split(
        micro_bench_raw, micro_bench_labels, test_size=0.2, random_state=42,
    )
    micro_bench_kwargs = {
        "patch_shape": patch_shape, "is_seg_dataset": False, "ndim": 2, "with_channels": True,
        "raw_transform": _micro_bench_nuclei_channel,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"},
    }
    for raws, labs, ds_list, n_samples in [
        (mb_train_r, mb_train_l, train_ds, 300), (mb_val_r, mb_val_l, val_ds, 50)
    ]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=raws, raw_key=None, label_paths=labs, label_key=None,
                    n_samples=n_samples, **micro_bench_kwargs,
                ), source_ndim=2,
            )
        )

    # 51. Pan-multiplex (cell segmentation in MIBI, CODEX and Vectra tissue imaging)
    # NOTE: The loader returns (nuclei, membrane); they are reordered into TissueNet's membrane, nucleus, zeros.
    pan_kwargs = {
        "path": os.path.join(input_path, "pan_multiplex"), "patch_shape": patch_shape,
        "raw_channel": "both", "raw_transform": _pan_multiplex_tissuenet_order,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"},
    }
    for subset in ["codex_colon", "mibi_breast", "mibi_decidua", "vectra_colon", "vectra_pancreas"]:
        train_ds.append(
            UniDataWrapper(
                datasets.get_pan_multiplex_dataset(subset=subset, split="train", n_samples=150, **pan_kwargs),
                source_ndim=2,
            )
        )

    # 52. Xenium (nucleus and cell segmentation in whole-slide multi-tissue stain)
    # NOTE: XOA segmented nuclei on DAPI (channel 0) and grew cells from the three morphology stains (channels
    # 1-3), so each target gets the channels it was made from. Whole slides are largely empty, hence the sampler.
    xenium_sampler = MinInstanceSampler(min_num_instances=10, exclude_ids=[0])
    xenium_nuclei_kwargs = {
        "path": os.path.join(input_path, "xenium"), "patch_shape": patch_shape,
        "raw_channel": "dapi", "label_channel": "nuclei",
        **{**kwargs, "sampler": xenium_sampler},
    }
    train_ds.append(
        UniDataWrapper(datasets.get_xenium_dataset(n_samples=400, **xenium_nuclei_kwargs), source_ndim=2)
    )
    xenium_cells_kwargs = {
        "path": os.path.join(input_path, "xenium"), "patch_shape": patch_shape,
        "raw_channel": "stack", "label_channel": "cells", "raw_transform": _xenium_cell_channels,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}, "sampler": xenium_sampler,
    }
    train_ds.append(
        UniDataWrapper(datasets.get_xenium_dataset(n_samples=400, **xenium_cells_kwargs), source_ndim=2)
    )

    # 53. GoNuclear (3D nucleus segmentation in confocal Arabidopsis root)
    # NOTE: Volume 1170 is held out by convention and is not used for training.
    for z in z_slices:
        gonuclear_kwargs = {
            "path": os.path.join(input_path, "gonuclear"),
            "patch_shape": (z, *patch_shape),
            "segmentation_task": "nuclei",
            "sample_ids": (1135, 1136, 1137, 1139),
            "n_samples": max(1, 400 // n_z),
            **kwargs,
        }
        train_ds.append(
            UniDataWrapper(datasets.get_gonuclear_dataset(**gonuclear_kwargs), source_ndim=3, group_key=(3, z))
        )

    # 54. NucVerse3D (3D nucleus segmentation in two-photon liver and confocal fly glia)
    # NOTE: Volumes are 320 px or smaller in plane, so a 256 crop is resized up rather than zero-padded.
    for z in z_slices:
        nucverse_kwargs = {
            "path": os.path.join(input_path, "nucverse3d"),
            "patch_shape": (z, 256, 256),
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
                datasets.get_nucverse3d_dataset(split="train", **nucverse_kwargs), source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_nucverse3d_dataset(split="test", **nucverse_kwargs), source_ndim=3, group_key=(3, z),
            )
        )

    # 55. PhMamm (3D cell segmentation in light-sheet Phallusia embryo membranes)
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
            UniDataWrapper(datasets.get_phmamm_dataset(**phmamm_kwargs), source_ndim=3, group_key=(3, z))
        )

    # 56. Wing disc (3D cell segmentation in confocal Drosophila wing epithelium)
    # NOTE: Native in-plane size is exactly 512, so no resize or padding is needed.
    for z in z_slices:
        wing_disc_kwargs = {
            "path": os.path.join(input_path, "wing_disc"),
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 300 // n_z),
            **kwargs,
        }
        train_ds.append(
            UniDataWrapper(datasets.get_wing_disc_dataset(**wing_disc_kwargs), source_ndim=3, group_key=(3, z))
        )

    # 57. Parhyale regeneration (3D nucleus segmentation in light-sheet H2B-EGFP)
    for z in z_slices:
        parhyale_kwargs = {
            "path": os.path.join(input_path, "parhyale_regen"),
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 200 // n_z),
            **kwargs,
        }
        train_ds.append(
            UniDataWrapper(datasets.get_parhyale_regen_dataset(**parhyale_kwargs), source_ndim=3, group_key=(3, z))
        )

    # 58. Vibrio cholerae (3D bacteria segmentation in confocal biofilms)
    for z in z_slices:
        vibrio_kwargs = {
            "path": os.path.join(input_path, "vibrio_cholerae"),
            "patch_shape": (z, *patch_shape),
            "n_samples": max(1, 300 // n_z),
            **kwargs,
        }
        train_ds.append(
            UniDataWrapper(datasets.get_vibrio_cholerae_dataset(**vibrio_kwargs), source_ndim=3, group_key=(3, z))
        )

    # 59. MorphoNet (3D cell and nucleus segmentation across five organisms)
    # NOTE: The Arabidopsis subset numbers its background as id 1, covering about 35 % of the volume, so it is
    # remapped to 0. Phallusia has no prepared volumes on disk and is skipped.
    for z in z_slices:
        for organism, background_id in [
            ("arabidopsis_thaliana", 1), ("caenorhabditis_elegans", None),
            ("patiria_miniata", None), ("tribolium_castaneum", None),
        ]:
            morphonet_trafo = (
                partial(
                    _background_id_label_trafo, background_id=background_id,
                    label_trafo=label_trafo(instances=True) if label_trafo is not None
                    else kwargs.get("label_transform2"),
                )
                if background_id is not None
                else (label_trafo(instances=True) if label_trafo is not None else kwargs.get("label_transform2"))
            )
            morphonet_kwargs = {
                "path": os.path.join(input_path, "morphonet"),
                "patch_shape": (z, *patch_shape),
                "organism": organism,
                "label_transform2": morphonet_trafo,
                "n_samples": max(1, 150 // n_z),
                **{k: v for k, v in kwargs.items() if k != "label_transform2"},
            }
            train_ds.append(
                UniDataWrapper(
                    datasets.get_morphonet_dataset(**morphonet_kwargs), source_ndim=3, group_key=(3, z),
                )
            )

    # 60. LICONN (3D neurite segmentation in expansion-microscopy connectomics)
    # NOTE: Shard coverage is partial. seg_proofread covers only part of the volume, so 14 of 40 random
    # full-volume patches returned all-zero labels. Sampling is restricted to the covered ROI, where 0 of 40 did.
    for z in z_slices:
        liconn_kwargs = {
            "path": os.path.join(input_path, "liconn"),
            "patch_shape": (z, *patch_shape),
            "segmentation": "proofread",
            "roi": (slice(64, 640), slice(0, 4608), slice(None)),
            "n_samples": max(1, 300 // n_z),
            **kwargs,
        }
        train_ds.append(
            UniDataWrapper(datasets.get_liconn_dataset(**liconn_kwargs), source_ndim=3, group_key=(3, z))
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
            "n_samples": max(1, 500 // n_z),
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

        train_ds.append(
            UniDataWrapper(
                datasets.get_cremi_dataset(samples=("A", "B"), **cremi_kwargs), source_ndim=3, group_key=(3, z)
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_cremi_dataset(samples=("C", ), **cremi_kwargs), source_ndim=3, group_key=(3, z)
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
    all_val_raw, all_val_lbl = get_emneuron_paths(emneuron_path, "val")

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
            # get_platynereis_cell_dataset concatenates one dataset per volume, so n_samples is per volume.
            "n_samples": max(1, 500 // (n_z * len(platy_train_ids))),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]}
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
    snemi_train_rois = np.s_[:70, :, :]
    snemi_val_rois = np.s_[70:, :, :]

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
    axonem_train = [p for p in axonem_label_paths if p not in axonem_val]
    assert len(axonem_val) == len(AXONEM_VAL_VOLUMES), axonem_val

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
    # NOTE: All three volumes are used; validation is the top fifth of z of validation_sample, cut out with rois.
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
                datasets.get_fib25_dataset(
                    samples=FIB25_SAMPLES, rois=FIB25_TRAIN_ROIS, n_samples=max(1, 500 // n_z), **fib25_kwargs
                ), source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_fib25_dataset(
                    samples=("validation_sample",), rois=[np.s_[416:, :, :]], n_samples=max(1, 50 // n_z),
                    **fib25_kwargs
                ), source_ndim=3, group_key=(3, z),
            )
        )

    # 10. Hemibrain (neuron segmentation in FIB-SEM of the Drosophila central brain, 8 nm isotropic)
    # NOTE: One cached 1024^3 crop (torch_em's default box), ~99% of voxels labelled, proofread. Same pipeline as
    # MANC and MaleCNS. Validation is the top fifth of z, cut out of training with rois.
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
        for roi, ds_list, n_samples in [(np.s_[:820, :, :], train_ds, 500), (np.s_[820:, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_hemibrain_dataset(rois=[roi], n_samples=max(1, n_samples // n_z), **hemibrain_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 11. MANC (neuron segmentation in FIB-SEM of the Drosophila male adult nerve cord, 8 nm isotropic)
    # NOTE: A separate specimen from MaleCNS (bucket flyem-vnc-2-26), so the two do not overlap. One cached
    # 1024^3 crop (torch_em's default box); validation is the top fifth of z, cut out of training with rois.
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
        for roi, ds_list, n_samples in [(np.s_[:820, :, :], train_ds, 500), (np.s_[820:, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_manc_dataset(rois=[roi], n_samples=max(1, n_samples // n_z), **manc_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 12. MaleCNS (neuron segmentation in FIB-SEM of the whole Drosophila male CNS, 8 nm isotropic)
    # NOTE: Six 1024^3 crops streamed from GCS, placed by probing the segmentation density along the
    # brain-neck-VNC axis (MALECNS_TRAIN_BOXES); one VNC crop is the validation set.
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

    # 15. SynapseWeb hippocampus (axon, dendrite and glia instances in ssTEM of rat CA1, ~2x2x50 nm)
    # NOTE: Only an irregular core of each volume is annotated, so each is cropped to its dense core and the
    # sampler rejects patches with less than half their pixels labelled.
    for z in z_slices:
        synapseweb_kwargs = {
            "path": os.path.join(input_path, "synapseweb_hippocampus"),
            "patch_shape": (z, *patch_shape),
            "download": True,
            "ndim": 3,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(25, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": DenseInstanceSampler(min_num_instances=3, min_fraction=0.5),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        train_ds.append(
            UniDataWrapper(
                datasets.get_synapseweb_hippocampus_dataset(
                    regions=("spine", "apical"), rois=SYNAPSEWEB_CORE_ROIS, n_samples=max(1, 300 // n_z),
                    **synapseweb_kwargs
                ), source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_synapseweb_hippocampus_dataset(
                    regions=("oblique",), rois=SYNAPSEWEB_CORE_ROIS, n_samples=max(1, 50 // n_z), **synapseweb_kwargs
                ), source_ndim=3, group_key=(3, z),
            )
        )

    # 16. MICrONS pinky (hand-annotated neuron instances in ssEM of mouse visual cortex, 4x4x40 nm)
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

    # 17. Zebrafinch j0126 (neuron segmentation in FIB-SEM of zebra finch Area X, 10x10x20 nm, Kornfeld lab)
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
        for roi, ds_list, n_samples in [(np.s_[:512, :, :], train_ds, 500), (np.s_[512:, :, :], val_ds, 50)]:
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

    # 18. Wildenberg 2023 (dense automated segmentation of all processes in FIB-SEM of mouse V1 layer 4, 12x12x40 nm)
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
        for roi, ds_list, n_samples in [(np.s_[:120, :, :], train_ds, 300), (np.s_[120:, :, :], val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_wildenberg_dataset(
                        rois=[roi], n_samples=max(1, n_samples // n_z), **wildenberg_kwargs
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 19. DenseCell (platelet cells in SBF-SEM of human platelet tissue, 10x10x50 nm)
    # NOTE: The source labels are a semantic mask; torch-em derives and caches 3D cell instances
    # (label_choice 'cell_instances'). The test split is sparsely annotated and not used.
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
        for split, ds_list, n_samples in [("train", train_ds, 200), ("val", val_ds, 40)]:
            ds_list.append(
                UniDataWrapper(
                    datasets.get_densecell_dataset(split=split, n_samples=max(1, n_samples // n_z), **densecell_kwargs),
                    source_ndim=3, group_key=(3, z),
                )
            )

    # 20. Tumor spheroid EM (FaDu tumor spheroid cells in SBF-SEM, 20 manually annotated 2D slices at 50 nm)
    # NOTE: 2D data. The 100 nm set is the same 20 slices downsampled and is not used; the 3D zarr holds automated
    # segmentation only. Validation is the two deepest z slices, training the other 18 (5 x, 5 y, 8 z).
    spheroid_paths, spheroid_raw_key, spheroid_label_key = datasets.tumor_spheroid_em.get_tumor_spheroid_paths(
        os.path.join(input_path, "tumor_spheroid_em"), source="2d_manual", resolution="50-50-50", target="cells",
        download=True,
    )
    spheroid_val = [p for p in spheroid_paths if os.path.basename(p) in TUMOR_SPHEROID_VAL_SLICES]
    spheroid_train = [p for p in spheroid_paths if p not in spheroid_val]
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

    # 21. NISB (synthetic neuron instance segmentation benchmark, 27 um cubes at 9x9x20 nm)
    # NOTE: The base setting's five training cubes; the last cube validates, since the official val and test cubes
    # are not cached. Synthetic labels are dense by construction.
    nisb_paths = datasets.nisb.get_nisb_paths(os.path.join(input_path, "nisb"), setting="base", split="train")
    nisb_val = [p for p in nisb_paths if os.path.basename(os.path.dirname(p)) in NISB_VAL_CUBES]
    nisb_train = [p for p in nisb_paths if p not in nisb_val]
    for z in z_slices:
        nisb_kwargs = {
            "patch_shape": (z, *patch_shape),
            "ndim": 3,
            "is_seg_dataset": True,
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True, sampling=(2.2, 1, 1)))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]},
        }
        for paths, ds_list, n_samples in [(nisb_train, train_ds, 300), (nisb_val, val_ds, 50)]:
            ds_list.append(
                UniDataWrapper(
                    torch_em.default_segmentation_dataset(
                        raw_paths=paths, raw_key="img", label_paths=paths, label_key="seg",
                        n_samples=max(1, n_samples // n_z), **nisb_kwargs,
                    ),
                    source_ndim=3, group_key=(3, z),
                )
            )

    return train_ds, val_ds


# Cached boxes in nm; both modules would otherwise default to far larger regions.
ZEBRAFINCH_J0126_BOX = (0, 51200, 0, 51200, 0, 12800)
# j0251 boxes in mip-0 voxels (x0, x1, y0, y1, z0, z1) at 10x10x25 nm, converted to nm at use.
ZEBRAFINCH_J0251_TRAIN_BOXES = [
    (6656, 8704, 15360, 17408, 3000, 3256),
    (23040, 25088, 1792, 3840, 3000, 3256),
    (9472, 11520, 15360, 17408, 7500, 7756),
    (12544, 14592, 20992, 23040, 12000, 12256),
]
ZEBRAFINCH_J0251_VAL_BOXES = [(24576, 26624, 3072, 5120, 12000, 12256)]
WILDENBERG_P105_BOX = (576, 24576, 576, 24576, 160, 6160)

# Bounding boxes of 'volumes/mask' (the annotated region) as (z, y, x) slices.
PINKY_MASK_ROIS = {
    "pinky_stitched_vol19-vol34_realigned.h5": np.s_[16:116, 256:2176, 256:2176],
    "pinky_stitched_vol40-vol41.h5": np.s_[0:86, 256:1280, 256:768],
    "pinky_vol401.h5": np.s_[16:116, 256:768, 256:768],
}
PINKY_TRAIN_FILES = ["pinky_stitched_vol19-vol34_realigned.h5", "pinky_stitched_vol40-vol41.h5"]
PINKY_VAL_FILES = ["pinky_vol401.h5"]


class DenseInstanceSampler:
    """Accept a patch only if it holds enough instances and enough labelled pixels."""

    def __init__(self, min_num_instances, min_fraction):
        self.instances = MinInstanceSampler(min_num_instances=min_num_instances, exclude_ids=[0])
        self.min_fraction = min_fraction

    def __call__(self, x, y):
        return self.instances(x, y) and (y > 0).mean() >= self.min_fraction


# SynapseWeb dense cores as (z, y, x) slices: bounding boxes of the region with coarse labelled density > 0.4.
SYNAPSEWEB_CORE_ROIS = {
    "spine": np.s_[0:42, 768:1984, 1024:1984],
    "oblique": np.s_[5:75, 896:3584, 1344:3328],
    "apical": np.s_[5:111, 192:3776, 320:4032],
}

# MaleCNS 1024^3 crops in 8 nm voxel coordinates. z runs from the brain through the neck connective into the VNC.
MALECNS_TRAIN_BOXES = [
    (40000, 41024, 40000, 41024, 20000, 21024),  # brain, torch_em default
    (38912, 39936, 19456, 20480, 15000, 16024),  # brain
    (81920, 82944, 33792, 34816, 35000, 36024),  # brain, far lateral
    (49152, 50176, 51200, 52224, 55000, 56024),  # neck connective
    (63488, 64512, 57344, 58368, 75000, 76024),  # VNC
]
MALECNS_VAL_BOXES = [(40960, 41984, 50176, 51200, 95000, 96024)]  # VNC

FIB25_SAMPLES = ("training_sample2", "validation_sample", "tstvol-520-1")
FIB25_TRAIN_ROIS = [np.s_[:, :, :], np.s_[:416, :, :], np.s_[:, :, :]]

PLATY_IGNORE_LABEL = datasets.platynereis.CELL_IGNORE_LABEL

# Mouse volumes 0-0-0, 0-0-3584 and 0-3584-3584 are soma blocks with 1, 10 and 4 ids; this threshold drops them.
AXONEM_MIN_IDS = 50
AXONEM_VAL_VOLUMES = ("seg_950-3584-3584_pad.h5", "seg_700-3584-3584_pad.h5")

# Voxels of missing raw data are mapped to this label and excluded from the loss.
MISSING_RAW_IGNORE_LABEL = datasets.platynereis.CELL_IGNORE_LABEL

# FAFB crops in 16 nm voxel coordinates, 1024x1024x410 voxels each, chosen inside brain tissue with dense
# segmentation at three depths (torch_em's DEFAULT_BOUNDING_BOXES). One mid-depth central crop is the validation set.
FAFB_VAL_BOXES = [(24576, 25600, 11776, 12800, 3500, 3910)]
FAFB_TRAIN_BOXES = [box for box in datasets.fafb.DEFAULT_BOUNDING_BOXES if box not in FAFB_VAL_BOXES]

ASTIH_SUBSETS = ["SEM1", "BF1", "BF2"]

# The two deepest z slices of the tumor spheroid volume; x, y and the other z slices train.
TUMOR_SPHEROID_VAL_SLICES = ("Au_01-vol_01-z_0212.h5", "Au_01-vol_01-z_0274.h5")

# The last of the five NISB base training cubes validates.
NISB_VAL_CUBES = ("seed4",)

# CoNIC tiles cut from PanNuke source images could overlap the blind PanNuke fold 3.
CONIC_EXCLUDED_COHORTS = ("pannuke",)

SPATCH_HE_SUBSETS = ["visium_hd_ov", "visium_hd_hcc", "visium_hd_coad", "stereoseq_ov"]


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


def _conic_cohort_roi(path, split, excluded_cohorts):
    """Return the roi over the tiles of a CoNIC split that do not come from the excluded source cohorts.

    torch_em writes the tiles of a split into one h5 stack in index order, and patch_info.csv lists the source
    cohort per index, so the excluded tiles must form the tail of the stack for a single roi to cover the rest.
    """
    import pandas as pd

    data_dir = os.path.join(path, "data")
    cohorts = pd.read_csv(os.path.join(data_dir, "patch_info.csv"))["patch_info"].str.split("_").str[0]
    indices = np.sort(pd.read_csv(os.path.join(data_dir, "split.csv"))[split].dropna().astype(int).to_numpy())
    excluded = cohorts.to_numpy()[indices]
    excluded = np.isin(excluded, list(excluded_cohorts))
    n_kept = int(np.argmax(excluded)) if excluded.any() else len(indices)
    if not excluded[n_kept:].all():
        raise RuntimeError(f"The excluded CoNIC cohorts do not form the tail of the '{split}' split.")
    return np.s_[:n_kept, :, :]


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
    cpm15_kwargs = {
        "path": os.path.join(input_path, "cpm15"), "patch_shape": patch_shape, "data_choice": "cpm15", **kwargs
    }
    train_ds.append(
        UniDataWrapper(datasets.get_cpm_dataset(split="train", n_samples=50, **cpm15_kwargs), source_ndim=2)
    )
    val_ds.append(UniDataWrapper(datasets.get_cpm_dataset(split="val", n_samples=50, **cpm15_kwargs), source_ndim=2))

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

    # 3. Lizard (nucleus segmentation in H&E histopathology images)
    # NOTE: Oversampled to 700 from the 361 patches that tile the 70 train images. CoNIC and lizard-mitosis are the
    # same source images re-tiled, so only Lizard is used here to avoid train/test leakage.
    lizard_kwargs = {
        "path": os.path.join(input_path, "lizard"), "patch_shape": patch_shape, "download": True, **kwargs
    }
    train_ds.append(
        UniDataWrapper(datasets.get_lizard_dataset(split="train", n_samples=700, **lizard_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_lizard_dataset(split="val", n_samples=100, **lizard_kwargs), source_ndim=2)
    )

    # 4. MoNuSeg (nucleus segmentation in H&E histopathology images)
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

    # 5. PanNuke (nucleus segmentation in H&E histopathology images)
    # NOTE: fold_1 + fold_2 for training, split 80/20 for internal val, matching patho-sam's
    # generalist training set. fold_3 is left untouched: it is the held-out benchmark test split
    # used across patho-sam's own evaluation scripts.
    # The full dataset is built twice (independent instances) so train and val get their own
    # raw_transform, rather than a shared random_split Subset that would alias the two.
    # patch_shape is requested at PanNuke's native 256x256, so torch_em's own padding is a no-op;
    # _pannuke_random_resize_and_pad_trafo does the resize+pad up to 512x512 instead.
    pannuke_kwargs = {
        "path": os.path.join(input_path, "pannuke"), "patch_shape": (1, 256, 256),
        "download": True, "ndim": 2, "folds": ["fold_1", "fold_2"],
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
    }
    pannuke_train_full = datasets.get_pannuke_dataset(**pannuke_kwargs)
    pannuke_val_full = datasets.get_pannuke_dataset(**pannuke_kwargs)
    pannuke_train_idx, pannuke_val_idx = train_test_split(
        range(len(pannuke_train_full)), test_size=0.2, random_state=42,
    )
    train_ds.append(
        UniDataWrapper(torch.utils.data.Subset(pannuke_train_full, pannuke_train_idx), source_ndim=2)
    )
    val_ds.append(UniDataWrapper(torch.utils.data.Subset(pannuke_val_full, pannuke_val_idx), source_ndim=2))

    # 6. PUMA (nucleus segmentation in H&E histopathology images)
    puma_kwargs = {"path": os.path.join(input_path, "puma"), "patch_shape": patch_shape, "download": True, **kwargs}
    train_ds.append(UniDataWrapper(datasets.get_puma_dataset(split="train", **puma_kwargs), source_ndim=2))
    val_ds.append(UniDataWrapper(datasets.get_puma_dataset(split="val", **puma_kwargs), source_ndim=2))

    # 7. TNBC CellType (nucleus segmentation in H&E triple-negative breast cancer plus TCGA brain sections)
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

    # 8. NuInsSeg (nucleus segmentation in H&E histopathology images from 31 human and mouse organs)
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

    # 9. LyNSeC (nucleus segmentation in IHC and H&E lymphoma images)
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

    # 10. SRSA-Net / IHC TMA (nucleus segmentation in IHC tissue microarray images of non-small cell lung cancer)
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

    # 11. CryoNuSeg (nucleus segmentation in H&E cryosection images from 10 organs)
    # NOTE: Small dataset (20 train / 4 val images, rater 'b1'). The 6-image test split is kept blind.
    # The split file 'cryonuseg_split.csv' lives with the data.
    cryonuseg_kwargs = {"path": os.path.join(input_path, "cryonuseg"), "patch_shape": patch_shape, **kwargs}
    train_ds.append(
        UniDataWrapper(datasets.get_cryonuseg_dataset(split="train", n_samples=50, **cryonuseg_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_cryonuseg_dataset(split="val", n_samples=20, **cryonuseg_kwargs), source_ndim=2)
    )

    # 12. GLySAC (nucleus segmentation in H&E gastric cancer histopathology images)
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

    # 13. Histo-Miner (nucleus segmentation in H&E cutaneous squamous cell carcinoma)
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

    # 14. sPATCH (nucleus segmentation in spatial-omics tissue: ovarian cancer, HCC and colon adenocarcinoma)
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

    # 15. CoNIC (nucleus segmentation in H&E colon, 256x256 tiles cut from the Lizard images)
    # NOTE: 88 train and 24 test tiles come from PanNuke source images, some of which may sit in the blind fold 3,
    # so the pannuke cohort is cut out by roi. Tiles are 256x256 and are resized/padded to 512 like PanNuke.
    conic_kwargs = {
        "path": os.path.join(input_path, "conic"), "patch_shape": (1, 256, 256), "download": True,
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
    }
    for split, ds_list, n_samples in [("train", train_ds, 700), ("test", val_ds, 50)]:
        roi = _conic_cohort_roi(conic_kwargs["path"], split, excluded_cohorts=CONIC_EXCLUDED_COHORTS)
        ds_list.append(
            UniDataWrapper(
                datasets.get_conic_dataset(split=split, rois=roi, n_samples=n_samples, **conic_kwargs), source_ndim=2
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

    # 17. CytoNuke (nucleus segmentation in H&E head-and-neck squamous cell carcinoma, 256x256 crops)
    # NOTE: Nucleus annotations only; the paired whole-cell polygons cover fewer objects than the nuclei.
    # Tiles are 256x256 and are resized/padded to 512 like PanNuke.
    cytonuke_kwargs = {
        "path": os.path.join(input_path, "cytonuke"), "patch_shape": (256, 256), "annotations": "nuclei",
        "download": True,
        **{**kwargs, "transform": partial(_pannuke_random_resize_and_pad_trafo, patch_shape=patch_shape)},
    }
    for split, ds_list, n_samples in [("train", train_ds, 100), ("val", val_ds, 20)]:
        ds_list.append(
            UniDataWrapper(
                datasets.get_cytonuke_dataset(split=split, n_samples=n_samples, **cytonuke_kwargs), source_ndim=2
            )
        )

    # 18. DeepLIIF (nucleus segmentation in IHC of lung, bladder and Ki67 breast cancer, 512x512 images)
    # NOTE: The IHC modality only; the co-registered mpIF panels are not used.
    deepliif_kwargs = {
        "path": os.path.join(input_path, "deepliif"), "patch_shape": patch_shape, "modality": "ihc",
        "label_choice": "instances", "download": True, **kwargs,
    }
    for split, ds_list, n_samples in [("train", train_ds, 400), ("val", val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                datasets.get_deepliif_dataset(split=split, n_samples=n_samples, **deepliif_kwargs), source_ndim=2
            )
        )

    # 19. PanopTILs (nucleus segmentation in H&E TCGA invasive breast cancer, 1024x1024 ROIs at 0.25 MPP)
    # NOTE: Built from paths, since the torch-em dataset binarizes the instances. No native split, so the 1349
    # ROIs are split 80/20 by path.
    panoptils_raw, panoptils_labels = datasets.panoptils.get_panoptils_paths(
        path=os.path.join(input_path, "panoptils"), label_choice="instances", download=True,
    )
    panoptils_train, panoptils_val = train_test_split(
        list(zip(panoptils_raw, panoptils_labels)), test_size=0.2, random_state=42
    )
    panoptils_kwargs = {"patch_shape": patch_shape, "with_channels": True, "ndim": 2, **kwargs}
    for pairs, ds_list, n_samples in [(panoptils_train, train_ds, 600), (panoptils_val, val_ds, 50)]:
        ds_list.append(
            UniDataWrapper(
                torch_em.default_segmentation_dataset(
                    raw_paths=[r for r, _ in pairs], raw_key=None, label_paths=[l for _, l in pairs], label_key=None,
                    is_seg_dataset=False, n_samples=n_samples, **panoptils_kwargs,
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


def _build_joint_datasets(input_path, z_slices, dataset_choice, distance_type="geodesic"):
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

    Returns:
        Tuple of (train_ds, val_ds) as :class:`ConcatDataset` instances.
    """
    if distance_type not in ("geodesic", "directed"):
        raise ValueError(f"Invalid distance_type: {distance_type!r}. Expected 'geodesic' or 'directed'.")

    patch_shape = (512, 512)
    # Both default to instances=True -> 5-channel output.
    label_trafo = _JointGeodesicLabelTransform if distance_type == "geodesic" else _JointLabelTransform

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
