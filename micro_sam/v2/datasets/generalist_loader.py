import json
import os
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split

import torch

import torch_em
from torch_em.data import datasets, MinInstanceSampler, ConcatDataset

from elf.io import open_file

from .wrapper import UniDataWrapper
from .sampler import UniBatchSampler, _build_group_map
from ..transforms.raw import (
    _identity, _cellpose_raw_trafo, _to_8bit, _normalize_percentile, _resize_raw_to_512, _resize_to_512,
)
from ..transforms.labels import (
    _em_cell_label_trafo, _joint_em_cell_label_trafo,
    _plantseg_label_trafo, _axondeepseg_pre_label_transform, _instance_labels,
    _JointLabelTransform,
)


def _ensure_native_byte_order(y):
    # tifffile.memmap returns big-endian >f4 for some TIFFs; byteswap to native so that
    # Kornia augmentation and skimage/vigra C extensions receive correctly ordered bytes.
    return y.byteswap().newbyteorder() if not y.dtype.isnative else y


def _prepare_data_loader(dataset, batch_size, shuffle, batch_size_per_group=None, num_workers=32):
    if isinstance(dataset, ConcatDataset) and (batch_size > 1 or batch_size_per_group):
        batch_sampler = UniBatchSampler(
            group_per_index=_build_group_map(dataset),
            batch_size=batch_size,
            batch_size_per_group=batch_size_per_group,
            shuffle=shuffle,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers, pin_memory=True,
        )
        # Monkey-patch shuffle attribute for torch_em DefaultTrainer compatibility.
        loader.shuffle = shuffle
    else:
        loader = torch_em.get_data_loader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

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
    cellpose_kwargs = {
        "path": os.path.join(input_path, "cellpose"),
        "patch_shape": patch_shape,
        "raw_transform": _cellpose_raw_trafo,
        "choice": "cyto",  # train on CP1 data.
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    train_ds.append(UniDataWrapper(datasets.get_cellpose_dataset(split="train", **cellpose_kwargs), source_ndim=2))
    val_ds.append(UniDataWrapper(datasets.get_cellpose_dataset(split="test", **cellpose_kwargs), source_ndim=2))

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
            n_samples=100,
            **{k: v for k, v in kwargs.items() if k != "raw_transform"}
        )
        return ds

    train_ds.append(UniDataWrapper(_get_cvz_dataset("cell", "train"), source_ndim=2))
    train_ds.append(UniDataWrapper(_get_cvz_dataset("dapi", "train"), source_ndim=2))
    val_ds.append(UniDataWrapper(_get_cvz_dataset("cell", "test"), source_ndim=2))
    val_ds.append(UniDataWrapper(_get_cvz_dataset("dapi", "test"), source_ndim=2))

    # 3. DSB dataset (nucleus segmentation in fluorescence images)
    dsb_kwargs = {"path": os.path.join(input_path, "dsb"), "patch_shape": patch_shape, "domain": "fluo", **kwargs}

    train_ds.append(UniDataWrapper(datasets.get_dsb_dataset(split="train", **dsb_kwargs), source_ndim=2))
    val_ds.append(UniDataWrapper(datasets.get_dsb_dataset(split="test", **dsb_kwargs), source_ndim=2))

    # 4. EmbedSeg (cell and nucleus segmentation in fluorescence microscopy images)
    # Anisotropy factors (z/xy) from file metadata or EmbedSeg paper (Table 3, arXiv:2101.10033).
    # Mouse-Organoid: z=1.0µm, xy=0.1733µm → ~5.8x → (6, 1, 1)
    # Mouse-Skull: z≈0.5µm, xy≈0.1µm → ~5x → (5, 1, 1)
    # Platynereis-ISH: confirmed isotropic from TIFF metadata (z≈xy≈0.45µm)
    # Platynereis-Nuclei: confirmed from TIFF metadata (z=2.031µm, xy=0.406µm → ~5x)
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
        else:   # Only two datasets have the test split.
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

        for ds_name in ["root", "ovules"]:
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
        "raw_transform": partial(_normalize_percentile, axis=(0, 1)),
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    train_ds.append(
        UniDataWrapper(datasets.get_tissuenet_dataset(split="train", n_samples=400, **tissuenet_kwargs), source_ndim=2)
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
                datasets.get_livecell_dataset(split="train", cell_types=[ctype], n_samples=100, **livecell_kwargs),
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
        "n_samples": 200,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    train_ds.append(UniDataWrapper(datasets.get_deepbacs_dataset(split="train", **deepbacs_kwargs), source_ndim=2))
    val_ds.append(UniDataWrapper(datasets.get_deepbacs_dataset(split="test", **deepbacs_kwargs), source_ndim=2))

    # 10. OrgaSegment (organoid segmentation in bright field images)
    orgasegment_kwargs = {
        "path": os.path.join(input_path, "orgasegment"), "patch_shape": patch_shape, "n_samples": 150, **kwargs
    }

    train_ds.append(
        UniDataWrapper(datasets.get_orgasegment_dataset(split="train", **orgasegment_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_orgasegment_dataset(split="val", **orgasegment_kwargs), source_ndim=2)
    )

    # 11. OrganoidNet (pancreatic organoid segmentation)
    organoidnet_kwargs = {
        "path": os.path.join(input_path, "organoidnet"), "patch_shape": patch_shape, "n_samples": 200, **kwargs
    }

    train_ds.append(
        UniDataWrapper(datasets.get_organoidnet_dataset(split="Training", **organoidnet_kwargs), source_ndim=2)
    )
    val_ds.append(
        UniDataWrapper(datasets.get_organoidnet_dataset(split="Validation", **organoidnet_kwargs), source_ndim=2)
    )

    # 12. Omnipose (bacteria and worm segmentation in mixed modality microscopy images)
    omnipose_kwargs = {
        "path": os.path.join(input_path, "omnipose"),
        "patch_shape": patch_shape,
        "raw_transform": _to_8bit,
        "n_samples": 200,
        **{k: v for k, v in kwargs.items() if k != "raw_transform"}
    }

    train_ds.append(UniDataWrapper(datasets.get_omnipose_dataset(split="train", **omnipose_kwargs), source_ndim=2))
    val_ds.append(UniDataWrapper(datasets.get_omnipose_dataset(split="test", **omnipose_kwargs), source_ndim=2))

    # 13. CTC (cell segmentation from Cell Tracking Challenge)
    # NOTE: CTC only supports the train split; no validation data added for CTC.
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

    return train_ds, val_ds


def _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo, _em_label_trafo=None):
    """Get all electron microscopy (EM) datasets for generalist training.

    Args:
        _em_label_trafo: EM cell label transform function to use.  Defaults to
            :func:`_em_cell_label_trafo`.  Pass :func:`_joint_em_cell_label_trafo`
            when building joint interactive+automatic datasets.

    Returns:
        Tuple of (train_ds, val_ds) lists of UniDataWrapper instances.
    """
    if _em_label_trafo is None:
        _em_label_trafo = _em_cell_label_trafo

    train_ds, val_ds = [], []
    n_z = len(z_slices)

    # 1. CREMI (neuron segmentation in vEM)
    # NOTE: Neurons are large — a patch typically contains only 1-2 of them, so min_num_instances=3
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
    # NOTE: Large neurons — use min_num_instances=1 (same reasoning as CREMI).
    # J0126-sbem (train: 150×150 or 256×256 XY) and FIB25 (val: 250×250 XY) are too small
    # for the standard 512×512 patch shape — they get their own 128×128 patch group with a
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

        # Small volumes (J0126 train; J0126+FIB25 val): 128×128 patches → resize to 512×512
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

    train_rois = _compute_platy_rois(
        platy_root, [1, 2, 3, 4, 5, 6], ignore_label=0, file_template=platy_cell_template, label_key=label_key,
    )
    val_rois = _compute_platy_rois(
        platy_root, [7, 8], ignore_label=0, file_template=platy_cell_template, label_key=label_key,
    )

    for z in z_slices:
        platynereis_kwargs = {
            "path": os.path.join(input_path, "platynereis"),
            "patch_shape": (z, *patch_shape),
            # sampling=None: ~20nm isotropic
            "label_transform2": (
                partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
                if label_trafo is not None else kwargs.get("label_transform2")
            ),
            "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
            "n_samples": max(1, 500 // n_z),
            **{k: v for k, v in kwargs.items() if k not in ["label_transform2", "sampler"]}
        }

        train_ds.append(
            UniDataWrapper(
                datasets.get_platynereis_cell_dataset(
                    sample_ids=[1, 2, 3, 4, 5, 6], rois=train_rois, **platynereis_kwargs
                ),
                source_ndim=3, group_key=(3, z),
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_platynereis_cell_dataset(
                    sample_ids=[7, 8], rois=val_rois, **platynereis_kwargs
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

    # 5. AxonDeepSeg (myelinated axon segmentation in SEM and TEM)
    # NOTE: Labels are semantic (0=bg, 1=myelin, 2=axon). label_transform2 extracts axons only (class 2)
    # so the sampler and distance transform both see axon instances, not myelin.
    axondeepseg_kwargs = {
        "path": os.path.join(input_path, "axondeepseg"),
        "patch_shape": patch_shape,
        "raw_transform": _to_8bit,  # ensures C=3 input for the model (SEM/TEM are grayscale)
        "pre_label_transform": _axondeepseg_pre_label_transform,  # semantic to instances before sampler
        "label_transform2": (
            partial(_em_label_trafo, label_trafo=label_trafo(instances=True))
            if label_trafo is not None else kwargs.get("label_transform2")
        ),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "n_samples": 300,
        **{
            k: v for k, v in kwargs.items()
            if k not in ["raw_transform", "label_transform2", "sampler"]
        },
    }

    for name in ["sem"]:
        train_ds.append(
            UniDataWrapper(
                datasets.get_axondeepseg_dataset(name=name, val_fraction=0.2, split="train", **axondeepseg_kwargs),
                source_ndim=2,
            )
        )
        val_ds.append(
            UniDataWrapper(
                datasets.get_axondeepseg_dataset(name=name, val_fraction=0.2, split="val", **axondeepseg_kwargs),
                source_ndim=2,
            )
        )

    return train_ds, val_ds


def get_dataloaders(
    input_path,
    label_trafo=None,
    batch_size=1,
    batch_size_2d=None,
    z_slices=None,
    dataset_choice="both",
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
            - ``"both"``: All datasets (default).
    """
    if dataset_choice not in ("lm", "em", "both"):
        raise ValueError(f"Invalid dataset_choice: {dataset_choice!r}. Expected 'lm', 'em', or 'both'.")

    if label_trafo is None:
        from micro_sam.v2.transforms.labels import DirectedPerObjectBoundaryDistanceTransform
        label_trafo = DirectedPerObjectBoundaryDistanceTransform

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

    if dataset_choice in ("lm", "both"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "both"):
        em_train, em_val = _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(em_train)
        val_ds.extend(em_val)

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
        batch_size_per_group=batch_size_per_group, num_workers=n_workers,
    )

    return train_loader, val_loader


def get_interactive_dataloaders(
    input_path,
    batch_size=1,
    batch_size_2d=None,
    z_slices=None,
    dataset_choice="both",
    n_workers=32,
):
    """Get generalist dataloaders for SAM2 interactive segmentation training.

    Identical dataset composition to :func:`get_dataloaders` but returns raw
    integer instance labels (``label_dtype=torch.int64``) instead of distance
    transforms.  Used with :class:`micro_sam.v2.training.ConvertToSam2VideoBatch`.

    Args:
        input_path: Root path to the generalist training data.
        batch_size: Default batch size (used for 3D groups).
        batch_size_2d: Optional larger batch size for 2D groups.
            Falls back to *batch_size* when not provided.
        z_slices: List of z-slice counts for 3D data (e.g. [8]).
            Defaults to [8].
        dataset_choice: Which dataset domain to include — ``"lm"``, ``"em"``,
            or ``"both"`` (default).
        n_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if dataset_choice not in ("lm", "em", "both"):
        raise ValueError(f"Invalid dataset_choice: {dataset_choice!r}. Expected 'lm', 'em', or 'both'.")

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
        batch_size_per_group=batch_size_per_group, num_workers=n_workers,
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
    from micro_sam.v2.transforms.labels import DirectedPerObjectBoundaryDistanceTransform

    patch_shape = (512, 512)
    label_trafo = DirectedPerObjectBoundaryDistanceTransform

    kwargs = {
        "raw_transform": _identity,
        "label_transform2": label_trafo(),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.float32,
    }

    train_ds, val_ds = [], []

    if dataset_choice in ("lm", "both"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "both"):
        em_train, em_val = _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(em_train)
        val_ds.extend(em_val)

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

    if dataset_choice in ("lm", "both"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo=None)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "both"):
        em_train, em_val = _get_em_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo=None)
        train_ds.extend(em_train)
        val_ds.extend(em_val)

    return ConcatDataset(*train_ds), ConcatDataset(*val_ds)


def _build_joint_datasets(input_path, z_slices, dataset_choice):
    """Build train/val datasets for joint interactive + automatic SAM2 training.

    Labels have **5 channels**: ``[instance_ids, fg, d_x, d_y, d_z]``.

    - Channel 0 (int64): instance IDs → interactive branch via ``ConvertToSam2VideoBatch``.
    - Channels 1-4 (float32): foreground + directed distances → automatic branch via
      ``DirectedDistanceLoss``.

    Unlike building two separate datasets, this shares a single data pipeline so both
    branches always see the same image patch.

    Returns:
        Tuple of (train_ds, val_ds) as :class:`ConcatDataset` instances.
    """
    patch_shape = (512, 512)
    label_trafo = _JointLabelTransform  # instances=True by default → 5-channel output

    kwargs = {
        "raw_transform": _identity,
        "label_transform2": label_trafo(),
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.float32,
    }

    train_ds, val_ds = [], []

    if dataset_choice in ("lm", "both"):
        lm_train, lm_val = _get_lm_datasets(input_path, patch_shape, z_slices, kwargs, label_trafo)
        train_ds.extend(lm_train)
        val_ds.extend(lm_val)

    if dataset_choice in ("em", "both"):
        em_train, em_val = _get_em_datasets(
            input_path, patch_shape, z_slices, kwargs, label_trafo,
            _em_label_trafo=_joint_em_cell_label_trafo,
        )
        train_ds.extend(em_train)
        val_ds.extend(em_val)

    return ConcatDataset(*train_ds), ConcatDataset(*val_ds)
