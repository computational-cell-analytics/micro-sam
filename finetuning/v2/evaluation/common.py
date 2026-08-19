import os
import ast
import csv
import warnings
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import xxhash
import numpy as np
import imageio.v3 as imageio
from skimage.measure import label as connected_components

import torch

from elf.io import open_file

from torch_em.data import datasets
from torch_em.util.image import load_image
from torch_em.util.segmentation import size_filter

from micro_sam.v1.evaluation.livecell import _get_livecell_paths
from micro_sam.v2.normalization import normalize_raw


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

_MODELS_DIR = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2"

# The pretrained SAM2 backbones. Only SAM2.1 is supported by micro_sam.v2.
CHECKPOINT_PATHS = {
    "hvit_t": os.path.join(_MODELS_DIR, "sam2.1_hiera_tiny.pt"),
    "hvit_s": os.path.join(_MODELS_DIR, "sam2.1_hiera_small.pt"),
    "hvit_b": os.path.join(_MODELS_DIR, "sam2.1_hiera_base_plus.pt"),
    "hvit_l": os.path.join(_MODELS_DIR, "sam2.1_hiera_large.pt"),
}

MODEL_TYPES = list(CHECKPOINT_PATHS)

# The 2d patch shape the models were trained on, see 'generalist_loader'.
TRAINING_PATCH_SHAPE = (512, 512)

# LIVECell test images whose annotation is incomplete: 2 labelled objects in a confluent crop, at
# 348x and 24x the annotated foreground. Both lie outside the stratified subset.
LIVECELL_EXCLUDED_TEST_IMAGES = frozenset({
    "BV2_Phase_A4_2_02d04h00m_3.tif",
    "BV2_Phase_A4_2_00d00h00m_1.tif",
})


def drop_excluded_livecell(raw_paths, label_paths) -> Tuple[List[str], List[str]]:
    """Remove the incompletely annotated LIVECell test images from a path pair list."""
    keep = [
        (raw, label) for raw, label in zip(raw_paths, label_paths)
        if os.path.basename(raw) not in LIVECELL_EXCLUDED_TEST_IMAGES
    ]
    if not keep:
        return [], []
    kept_raw, kept_label = zip(*keep)
    return list(kept_raw), list(kept_label)


# Overridable, so a run can point at another training version. A checkpoint that still trains needs
# a frozen copy, because the trainer overwrites 'best.pt' while jobs queue.
JOINT_CHECKPOINT_ROOT = os.environ.get(
    "MICRO_SAM2_JOINT_CHECKPOINT_ROOT", os.path.join(_MODELS_DIR, "joint", "v2", "checkpoints")
)
# The joint checkpoints are split into loadable weight files here, see 'export_joint_checkpoint'.
JOINT_EXPORT_ROOT = os.environ.get(
    "MICRO_SAM2_JOINT_EXPORT_ROOT", os.path.join(_MODELS_DIR, "exported", "joint", "v2")
)

DATASETS_2D = [
    "livecell",
    "arvidsson", "bitdepth_nucseg", "cellbindb", "cellpose_data",
    "covid_if", "cvz_fluo", "deepbacs", "deepseas", "dic_hepg2", "dsb",
    "dynamicnuclearnet", "hpa", "microbeseg", "neurips_cellseg", "omnipose",
    "segpc", "tissuenet", "usiigaci", "vicar", "yeaz",
]

# Ground-truth size floor that drops the crop-severed slivers relabelling promotes to objects. It
# defines the ground truth, so it is measured, never tuned.
GT_MIN_SIZE_2D = {
    "livecell": 50,
    "cellpose_data": 20, "deepbacs": 50, "dynamicnuclearnet": 50, "tissuenet": 10,
    "u20s": 10, "vicar": 25, "yeaz": 10,
}

DATASETS_3D_LM = [
    "blastospim", "cartocell", "celegans_atlas", "cellseg_3d", "embedseg",
    "gonuclear", "mouse_embryo", "nis3d", "plantseg", "pnas_arabidopsis",
]

DATASETS_3D_EM = ["platynereis_nuclei", "cremi", "snemi", "humanneurons"]

DATASETS_3D = DATASETS_3D_LM + DATASETS_3D_EM

# The split to tune on, or None where the loader has no splits and VAL_Z_RANGE holds out a z-slab.
# A dataset whose 'val' is the evaluated split is absent: tuning there would select on scored samples.
VAL_SPLITS = {
    "livecell": "val",
    "tissuenet": "val",
    "dynamicnuclearnet": "val",
    "deepbacs": "val",
    "dic_hepg2": "val",
    "celegans_atlas": "val",
    "embedseg": "train",
    "gonuclear": None,
    "cremi": None,
    "snemi": None,
}

# The tuning slab for volumes with no splits, disjoint from the slab the evaluation scores. Indices
# count from what load_volume keeps, so snemi starts at slice 70, and gonuclear skips its sparse start.
VAL_Z_RANGE = {
    "cremi": (0, 32),
    "snemi": (0, 8),
    "gonuclear": (32, 96),
}


def _sorted_pairs(raw_paths, label_paths) -> Tuple[List[str], List[str]]:
    """Sort raw and label paths as pairs.

    Sorting the two lists on their own breaks the pairing whenever the label names sort differently,
    which happens when one image name is a prefix of another, e.g. 'x_1.tif' and 'x_11.tif' with
    labels 'x_1_masks.tif' and 'x_11_masks.tif'.
    """
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(
            f"Expect as many raw as label paths, got {len(raw_paths)} and {len(label_paths)}."
        )
    pairs = sorted(zip(raw_paths, label_paths), key=lambda pair: str(pair[0]))
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _get_2d_data_paths(
    dataset_name: str, data_root: str, download: bool = False, split: str = "test"
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name == "livecell":
        img, gt = _get_livecell_paths(input_folder=os.path.join(p, "livecell"), split=split)
        img, gt = drop_excluded_livecell(img, gt)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "arvidsson":
        img, gt = datasets.arvidsson.get_arvidsson_paths(
            path=os.path.join(p, "arvidsson"), split="test", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "bitdepth_nucseg":
        img, gt = datasets.bitdepth_nucseg.get_bitdepth_nucseg_paths(
            path=os.path.join(p, "bitdepth_nucseg"), download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "cellbindb":
        img, gt = datasets.cellbindb.get_cellbindb_paths(
            path=os.path.join(p, "cellbindb"), download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "cellpose_data":
        img, gt = datasets.cellpose.get_cellpose_paths(
            path=os.path.join(p, "cellpose"), split="test", choice="cyto", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "covid_if":
        paths = datasets.covid_if.get_covid_if_paths(
            path=os.path.join(p, "covid_if"), download=download,
        )
        return sorted(paths), sorted(paths), "raw/nuclei/s0", "labels/nuclei/s0"

    if dataset_name == "cvz_fluo":
        img, gt = [], []
        for stain in ("cell", "dapi"):
            i, g = datasets.cvz_fluo.get_cvz_fluo_paths(
                path=os.path.join(p, "cvz"), stain_choice=stain, download=download,
            )
            img.extend(i)
            gt.extend(g)
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "deepbacs":
        img_folder, label_folder = datasets.deepbacs.get_deepbacs_paths(
            path=os.path.join(p, "deepbacs"), bac_type="mixed", split=split, download=download,
        )
        img = sorted(glob(os.path.join(img_folder, "*.tif")))
        gt = sorted(glob(os.path.join(label_folder, "*.tif")))
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "deepseas":
        img, gt = datasets.deepseas.get_deepseas_paths(
            path=os.path.join(p, "deepseas"), split="test", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dic_hepg2":
        img, gt = datasets.dic_hepg2.get_dic_hepg2_paths(
            path=os.path.join(p, "dic_hepg2"), split=split, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dsb":
        img, gt = datasets.dsb.get_dsb_paths(
            path=os.path.join(p, "dsb"), source="full", split=None, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dynamicnuclearnet":
        paths = datasets.dynamicnuclearnet.get_dynamicnuclearnet_paths(
            path=os.path.join(p, "dynamicnuclearnet"), split=split, download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "hpa":
        paths = datasets.hpa.get_hpa_segmentation_paths(
            path=os.path.join(p, "hpa"), split="val", download=download,
        )
        # protein channel for cell body segmentation (peft-sam convention)
        return sorted(paths), sorted(paths), "raw/protein", "labels"

    if dataset_name == "microbeseg":
        img, gt = datasets.microbeseg.get_microbeseg_paths(
            path=os.path.join(p, "microbeseg"), split="test",
            annotation_type="30min-man", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "neurips_cellseg":
        img, gt = datasets.neurips_cell_seg.get_neurips_cellseg_paths(
            root=os.path.join(p, "neurips_cellseg"), split="test", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "omnipose":
        img, gt = [], []
        for choice in ("bact_fluor", "bact_phase", "worm", "worm_high_res"):
            try:
                i, g = datasets.omnipose.get_omnipose_paths(
                    path=os.path.join(p, "omnipose"), split="test",
                    data_choice=choice, download=download,
                )
                img.extend(i)
                gt.extend(g)
            except Exception as e:
                warnings.warn(f"Skipping omnipose choice '{choice}': {e}")
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "segpc":
        # The dataset has no test split, so the evaluation uses the validation split.
        paths = datasets.segpc.get_segpc_paths(
            path=os.path.join(p, "segpc"), split="validation", download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels/cells"

    if dataset_name == "tissuenet":
        paths = datasets.tissuenet.get_tissuenet_paths(
            path=os.path.join(p, "tissuenet"), split=split, download=download,
        )
        # The rgb composite and the cell labels are what the training used.
        return sorted(paths), sorted(paths), "raw/rgb", "labels/cell"

    if dataset_name == "usiigaci":
        # The dataset has no test split, so the evaluation uses the validation split.
        img, gt = datasets.usiigaci.get_usiigaci_paths(
            path=os.path.join(p, "usiigaci"), split="val", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "vicar":
        img, gt = datasets.vicar.get_vicar_paths(
            path=os.path.join(p, "vicar"), download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "yeaz":
        img, gt = [], []
        for choice in ("bf", "phc"):
            i, g = datasets.yeaz.get_yeaz_paths(
                path=os.path.join(p, "yeaz"), choice=choice, split="test", download=download,
            )
            img.extend(i)
            gt.extend(g)
        return (*_sorted_pairs(img, gt), None, None)

    raise ValueError(f"Unknown 2D dataset: {dataset_name!r}")


def _get_3d_lm_data_paths(
    dataset_name: str, data_root: str, download: bool = False, split: str = "test"
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name == "blastospim":
        paths = datasets.blastospim.get_blastospim_paths(
            path=os.path.join(p, "blastospim"), download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "cartocell":
        cartocell_root = os.path.join(p, "cartocell")
        current_data_root = os.path.join(cartocell_root, "CartoCell")
        if os.path.exists(current_data_root):
            img = sorted(glob(os.path.join(current_data_root, "test", "x", "*.tif")))
            gt = [ipath.replace(os.sep + "x" + os.sep, os.sep + "y" + os.sep) for ipath in img]
        else:
            img, gt = [], []
            for name in ("eggChambers", "embryoids", "MDCK-Normoxia", "MDCK-Hypoxia"):
                try:
                    i, g = datasets.cartocell.get_cartocell_paths(
                        path=cartocell_root, split="test", name=name, download=download,
                    )
                    img.extend(i)
                    gt.extend(g)
                except Exception as e:
                    warnings.warn(f"Skipping cartocell name '{name}': {e}")
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "celegans_atlas":
        img, gt = datasets.celegans_atlas.get_celegans_atlas_paths(
            path=os.path.join(p, "celegans_atlas"), split=split, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "cellseg_3d":
        img, gt = datasets.cellseg_3d.get_cellseg_3d_paths(
            path=os.path.join(p, "cellseg_3d"), download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "embedseg":
        img, gt = datasets.embedseg_data.get_embedseg_paths(
            path=os.path.join(p, "embedseg"),
            name="Mouse-Skull-Nuclei-CBG", split=split, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "gonuclear":
        paths = datasets.gonuclear.get_gonuclear_paths(
            path=os.path.join(p, "gonuclear"), download=download,
        )
        return sorted(paths), sorted(paths), "raw/nuclei", "labels/nuclei"

    if dataset_name == "mouse_embryo":
        # The dataset has no test split, so the evaluation uses the validation split.
        paths = datasets.mouse_embryo.get_mouse_embryo_paths(
            path=os.path.join(p, "mouse_embryo"), name="nuclei", split="val", download=download,
        )
        return sorted(paths), sorted(paths), "raw", "label"

    if dataset_name == "nis3d":
        img, gt = datasets.nis3d.get_nis3d_paths(
            path=os.path.join(p, "nis3d"), split="test", split_type="cross-image", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "plantseg":
        all_paths = []
        for name, folder, split in (
            ("nuclei", "plantseg", "train"),
            ("ovules", "plantseg_ovules", "test"),
            ("root", "plantseg_root", "test"),
        ):
            try:
                ps = datasets.plantseg.get_plantseg_paths(
                    path=os.path.join(p, folder), name=name, split=split, download=download,
                )
                all_paths.extend(ps)
            except Exception as e:
                warnings.warn(f"Skipping plantseg name '{name}': {e}")
        return sorted(all_paths), sorted(all_paths), "raw", "label"

    if dataset_name == "pnas_arabidopsis":
        paths = datasets.pnas_arabidopsis.get_pnas_arabidopsis_paths(
            path=os.path.join(p, "pnas_arabidopsis"), download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    raise ValueError(f"Unknown 3D LM dataset: {dataset_name!r}")


def _get_3d_em_data_paths(
    dataset_name: str, data_root: str, download: bool = False
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name == "platynereis_nuclei":
        paths = datasets.platynereis.get_platynereis_paths(
            path=os.path.join(p, "platynereis"), sample_ids=None, name="nuclei", download=download,
        )
        return paths, paths, "volumes/raw", "volumes/labels/nucleus_instance_labels"

    if dataset_name == "cremi":
        # The joint training used samples A and B, so only C is held out.
        paths = datasets.cremi.get_cremi_paths(
            path=os.path.join(p, "cremi"), samples=("C",), download=download,
        )
        return sorted(paths), sorted(paths), "volumes/raw", "volumes/labels/neuron_ids"

    if dataset_name == "snemi":
        # The test file has no labels, so the holdout is the part of the train file that the joint
        # training did not use, see load_volume.
        path = datasets.snemi.get_snemi_paths(
            path=os.path.join(p, "snemi"), sample="train", download=download,
        )
        return [path], [path], "volumes/raw", "volumes/labels/neuron_ids"

    if dataset_name == "humanneurons":
        # Resolved directly: the installed torch-em has no loader for this dataset.
        paths = sorted(glob(os.path.join(p, "humanneurons", "*.h5")))
        return paths, paths, "raw", "labels"

    raise ValueError(f"Unknown 3D EM dataset: {dataset_name!r}")


def get_data_paths(
    dataset_name: str, data_root: str, download: bool = False, split: str = "test"
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """Return (raw_paths, label_paths, raw_key, label_key) for a dataset's evaluation split.

    raw_key / label_key are None for plain image files and non-None for H5 / zarr.

    With split='val' this returns data held out from what the evaluation scores, which is what a
    parameter search has to run on, see VAL_SPLITS. Only the datasets listed there support it.
    """
    all_datasets = DATASETS_2D + DATASETS_3D
    assert dataset_name in all_datasets, (
        f"Unsupported dataset: '{dataset_name}'. Choose from {all_datasets}."
    )

    if split == "val":
        if dataset_name not in VAL_SPLITS:
            raise ValueError(
                f"There is no data held out from the evaluation for '{dataset_name}', so it cannot be "
                f"tuned on a validation split. Datasets that can: {sorted(VAL_SPLITS)}."
            )
        # None means the loader has no split of its own; the holdout is the z-slab in VAL_Z_RANGE,
        # which load_volume applies to the very same volumes.
        split = VAL_SPLITS[dataset_name] or "test"

    if dataset_name in DATASETS_2D:
        return _get_2d_data_paths(dataset_name, data_root, download=download, split=split)
    if dataset_name in DATASETS_3D_LM:
        return _get_3d_lm_data_paths(dataset_name, data_root, download=download, split=split)
    return _get_3d_em_data_paths(dataset_name, data_root, download=download)


def _center_crop_roi(shape, crop_shape):
    """Returns a tuple of slices for a center crop."""
    roi = []
    for s, c in zip(shape, crop_shape):
        c = min(c, s)
        start = (s - c) // 2
        roi.append(slice(start, start + c))
    return tuple(roi)


def load_volume(
    raw_path: str,
    label_path: str,
    raw_key: Optional[str],
    label_key: Optional[str],
    dataset_name: str,
    crop_shape: Tuple[int, ...] = (8, 512, 512),
    ensure_8bit: bool = True,
    ensure_instances: bool = True,
    z_range: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a 3D volume, apply dataset-specific preprocessing, and center-crop.

    valid_roi is a boolean mask that is True where the data is annotated. It is None for every
    dataset except platynereis_nuclei, which is annotated only in part.

    'z_range' restricts the volume to a z-slab before the center crop, which is how a dataset without
    splits holds tuning data out of the evaluated slab. See VAL_Z_RANGE.
    """
    if raw_key is None:
        raw = load_image(raw_path)
    else:
        raw = open_file(raw_path, mode="r")[raw_key][:]

    if label_key is None:
        labels = load_image(label_path)
    else:
        labels = open_file(label_path, mode="r")[label_key][:]

    if dataset_name == "snemi":
        # Training used slices [0:70], so only slices 70+ are held out.
        raw, labels = raw[70:], labels[70:]

    if z_range is not None:
        z_start, z_stop = z_range
        raw, labels = raw[z_start:z_stop], labels[z_start:z_stop]

    valid_roi = None
    if dataset_name == "platynereis_nuclei":
        labels = labels.astype("int64")
        valid_roi = labels != -1
        labels[labels == -1] = 0

    if ensure_8bit:
        raw = normalize_raw(raw) * 255.0

    roi = _center_crop_roi(raw.shape, crop_shape)
    raw, labels = raw[roi], labels[roi]
    if valid_roi is not None:
        valid_roi = valid_roi[roi]

    # Restrict to the annotated z-range. Interior empty slices stay, or the volume is not contiguous.
    annotated = np.any(labels != 0, axis=tuple(range(1, labels.ndim)))
    if annotated.any():
        z_start = int(np.argmax(annotated))
        z_stop = len(annotated) - int(np.argmax(annotated[::-1]))
        raw, labels = raw[z_start:z_stop], labels[z_start:z_stop]
        if valid_roi is not None:
            valid_roi = valid_roi[z_start:z_stop]

    if ensure_instances:
        labels = connected_components(labels)

    assert raw.shape == labels.shape, f"Shape mismatch: raw {raw.shape} vs labels {labels.shape}"
    return raw.astype("float32"), labels.astype("uint32"), valid_roi


_UNISAM2_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/automatic/v1"
UNISAM2_CHECKPOINT = os.path.join(_UNISAM2_ROOT, "checkpoints", "unisam2-both", "best.pt")


def get_joint_checkpoint(model_type: str, checkpoint: str = "best") -> str:
    """Return the joint trainer checkpoint for a model type, e.g. 'hvit_b'."""
    path = os.path.join(JOINT_CHECKPOINT_ROOT, f"joint_sam2_{model_type}_multi_gpu", f"{checkpoint}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"There is no joint '{checkpoint}' checkpoint for '{model_type}' at '{path}'.")
    return path


def _save_atomic(obj, path: str) -> None:
    """Save to a process-unique temporary file first, so concurrent jobs never read a partial file."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _strip_ddp_prefix(state_dict):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def checkpoint_checksum(path: str) -> str:
    """Return the xxh128 checksum of a checkpoint without loading it into memory."""
    checksum = xxhash.xxh128()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def combine_checkpoint_checksums(*checksums: str) -> str:
    """Combine the content checksums of all weights that determine one evaluation run."""
    if len(checksums) == 1:
        return checksums[0]
    combined = xxhash.xxh128()
    for checksum in checksums:
        combined.update(checksum.encode("ascii"))
        combined.update(b"\0")
    return combined.hexdigest()


def export_joint_checkpoint(
    model_type: str, checkpoint: str = "best", export_root: str = JOINT_EXPORT_ROOT,
    source_checksum: Optional[str] = None,
) -> Tuple[str, str]:
    """Split a joint checkpoint into an interactive and an automatic weight file.

    The joint trainer bundles the SAM2 weights ('model_state'), the UniSAM2 decoder weights
    ('unetr_state') and pickled trainer state in a single file. That file cannot be loaded by
    `sam2.build_sam`, which reads `torch.load(...)['model']` with `weights_only=True`. Both
    exported files are plain tensor dicts, mirroring `scripts/model_export/export_sam2_cells_model.py`.

    The checksum in the name records which checkpoint an export came from, so an export is reused
    only for that content. Every training version has a 'best' checkpoint, so a plain
    'joint_sam2_hvit_t_best' would hand back the previous version's export instead.

    Args:
        model_type: The SAM2 backbone the model was finetuned from, e.g. 'hvit_b'.
        checkpoint: Which trainer checkpoint to export, 'best' or 'latest'.
        export_root: The directory the exported weight files are written to.
        source_checksum: A previously computed checksum, to avoid reading the checkpoint twice.

    Returns:
        The paths to the interactive (SAM2) and the automatic (UniSAM2 decoder) weight files.
    """
    checkpoint_path = get_joint_checkpoint(model_type, checkpoint)
    source_checksum = source_checksum or checkpoint_checksum(checkpoint_path)
    name = f"joint_sam2_{model_type}_{checkpoint}_{source_checksum}"
    interactive_path = os.path.join(export_root, f"{name}.pt")
    decoder_path = os.path.join(export_root, f"{name}_decoder.pt")
    if os.path.exists(interactive_path) and os.path.exists(decoder_path):
        return interactive_path, decoder_path

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing = [key for key in ("model_state", "unetr_state") if key not in state]
    if missing:
        raise RuntimeError(f"'{checkpoint_path}' is not a joint checkpoint, it is missing {missing}.")

    os.makedirs(export_root, exist_ok=True)
    _save_atomic({"model": _strip_ddp_prefix(state["model_state"]), "model_type": model_type}, interactive_path)
    _save_atomic(_strip_ddp_prefix(state["unetr_state"]), decoder_path)
    print(f"Exported '{checkpoint_path}' to '{interactive_path}' and '{decoder_path}'.")
    return interactive_path, decoder_path


# Keep a volume's tracking state on the device: the same masks propagate 1.2-1.3x faster, for about
# 17 MB of device memory per slice. That is a batch job's to spend, which is why it is not the default.
VOLUME_SPEED_OPTIONS = {"offload_to_cpu": False}


DATASET_SPACING: dict = {
    # z/xy voxel ratios from published acquisition parameters
    "embedseg": (4, 1, 1),  # Mouse-Skull-Nuclei-CBG: z=1µm, xy=0.25µm
    "blastospim": (10, 1, 1),  # SPIM: z≈2µm, xy≈0.208µm
    "mouse_embryo": (4, 1, 1),  # confocal: z≈1µm, xy≈0.22µm
}


# The parameters `AutomaticPromptGenerator.generate` accepts, so a run can be described by one dict.
GENERATE_PARAM_KEYS = (
    "candidate_threshold", "foreground_threshold", "n_iter", "dt", "sigma", "min_candidate_size",
    "score_threshold", "max_overlap", "min_size", "refine_with_box_prompts", "box_extension",
    "multimasking", "n_objects_per_pass", "early_stop_patience", "batch_size", "n_threads",
)


def resolve_params(overrides=None, ndim=2):
    """The generation parameters for one run, with 'overrides' applied on top of the library defaults.

    The single definition of what a run's parameters are, so that a benchmark, a walk-through and a
    sweep all describe the same run. The result is ready to pass to `generate` as keyword arguments.

    Args:
        overrides: The parameters to change, by the name `generate` gives them. A volume also accepts
            'candidate_threshold_3d', which is the name the defaults give its own threshold.
        ndim: The number of spatial dimensions, 2 or 3.

    Returns:
        The parameters, keyed as `generate` takes them.
    """
    from micro_sam.v2.automatic_prompt_generation import DEFAULT_PROMPT_GENERATION

    overrides = overrides or {}
    params = {key: DEFAULT_PROMPT_GENERATION[key] for key in GENERATE_PARAM_KEYS}
    params.update(overrides)
    if ndim == 3:
        # A candidate's density scales with the object's size, so a volume has its own threshold.
        default_3d = DEFAULT_PROMPT_GENERATION["candidate_threshold_3d"]
        params["candidate_threshold"] = overrides.get("candidate_threshold_3d", default_3d)
    params.pop("candidate_threshold_3d", None)
    return params


def _alias_micro_sam2_modules():
    """Alias the moved 'micro_sam2' modules so checkpoints pickled before the package move load."""
    import sys
    import types
    import micro_sam.v2.datasets.sampler as datasets_sampler
    import micro_sam.v2.datasets.wrapper as datasets_wrapper
    import micro_sam.v2.transforms.labels as transforms_labels
    import micro_sam.v2.transforms.raw as transforms_raw

    root = sys.modules.setdefault("micro_sam2", types.ModuleType("micro_sam2"))
    root.__path__ = []
    datasets_module = sys.modules.setdefault("micro_sam2.datasets", types.ModuleType("micro_sam2.datasets"))
    datasets_module.__path__ = []
    transforms = sys.modules.setdefault("micro_sam2.transforms", types.ModuleType("micro_sam2.transforms"))
    transforms.__path__ = []

    sys.modules["micro_sam2.datasets.sampler"] = datasets_sampler
    sys.modules["micro_sam2.datasets.wrapper"] = datasets_wrapper
    sys.modules["micro_sam2.transforms.labels"] = transforms_labels
    sys.modules["micro_sam2.transforms.raw"] = transforms_raw
    setattr(root, "datasets", datasets_module)
    setattr(root, "transforms", transforms)
    setattr(datasets_module, "sampler", datasets_sampler)
    setattr(datasets_module, "wrapper", datasets_wrapper)
    setattr(transforms, "labels", transforms_labels)
    setattr(transforms, "raw", transforms_raw)


def load_unisam2_model(checkpoint_path, device, encoder="hvit_t"):
    """Load a UniSAM2 model for automatic segmentation.

    Handles the standalone UniSAM2 checkpoints ('model_state'), the joint checkpoints
    ('unetr_state', with the SAM2 encoder wrapped in an adapter) and exported decoder weights.

    Args:
        checkpoint_path: The filepath to the checkpoint.
        device: The torch device.
        encoder: The SAM2 backbone the decoder was trained on, e.g. 'hvit_b'.

    Returns:
        The UniSAM2 model in eval mode.
    """
    from micro_sam.v2.instance_segmentation import get_unisam2_model
    _alias_micro_sam2_modules()
    return get_unisam2_model(checkpoint_path, device=device, encoder=encoder)


def build_apg_segmenter(
    model_type, ndim, device, joint_checkpoint="best", decoder_path=None, joint_checksum=None,
    export_root=None,
):
    """Build the automatic prompt generator from both halves of a joint checkpoint.

    The decoder proposes the candidates and the interactive branch scores them, so a run needs both.
    A volume is propagated by the SAM2 video predictor, which is a different model input type.

    Args:
        model_type: The SAM2 backbone of the joint model, e.g. 'hvit_t'.
        ndim: The number of spatial dimensions, 2 or 3.
        device: The torch device.
        joint_checkpoint: The joint trainer checkpoint, without the '.pt' suffix.
        decoder_path: Decoder weights to use instead of the ones exported from the joint checkpoint.
        export_root: Optional directory for the split checkpoint files. Defaults to JOINT_EXPORT_ROOT.

    Returns:
        The prompt generator, built through the library factory that the CLI and the API use.
    """
    from micro_sam.v2.util import get_sam2_model
    from micro_sam.v2.instance_segmentation import get_instance_segmentation_generator

    export_kwargs = {} if export_root is None else {"export_root": export_root}
    interactive_path, exported_decoder = export_joint_checkpoint(
        model_type, joint_checkpoint, source_checksum=joint_checksum, **export_kwargs
    )
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=interactive_path,
        **({"input_type": "videos"} if ndim == 3 else {}),
    )
    decoder = load_unisam2_model(
        decoder_path or exported_decoder, device, encoder=model.image_encoder
    )
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=ndim,
    )


def predict_unisam2(model, raw, ndim, device, normalization=None):
    from micro_sam.v2.instance_segmentation import get_unisam2_segmentation_generator
    # UniSAM2 takes single-channel input, so a trailing channel axis is averaged away.
    if raw.ndim > ndim:
        raw = raw.mean(axis=-1)

    is_3d = (ndim == 3)
    # Tiling an image that fits the training patch changes the encoder's scale and the normalization.
    is_tiled = is_3d or any(size > TRAINING_PATCH_SHAPE[-1] for size in raw.shape[:2])
    segmenter = get_unisam2_segmentation_generator(model, is_tiled=is_tiled, device=device)
    if is_tiled:
        tile_shape = (4, 384, 384) if is_3d else (384, 384)
        halo = (2, 64, 64) if is_3d else (64, 64)
        segmenter.initialize(raw, ndim=ndim, tile_shape=tile_shape, halo=halo, normalization=normalization)
    else:
        segmenter.initialize(raw, ndim=ndim, normalization=normalization)
    return segmenter.get_state()["prediction"]


def postprocess_unisam2(out, dataset_name, backend="cpp", params=None):
    """Turn a (4, *spatial) prediction into an instance segmentation.

    EM datasets use the dense (multicut) mode, all others the sparse (flow) mode. 'params' overrides
    the postprocessing defaults, e.g. with the best combination found by grid_search_automatic_cells.
    """
    from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut
    params = {} if params is None else params
    fg = out[0]
    if dataset_name in DATASETS_3D_EM:
        boundary_map = fg.max() - fg
        boundary_map /= boundary_map.max()
        distances = np.stack([out[2], out[3]])
        seg = run_multicut(boundary_map, distances, backend=backend, **params)
    else:
        spacing = DATASET_SPACING.get(dataset_name, None)
        seg = flow_instance_segmentation(fg, out[1:], spacing=spacing, backend=backend, **params)
    return seg.astype("uint32")


def run_dataset_evaluation(gt_paths, prediction_paths, dataset_name: str, save_path: str):
    """Score a dataset and write the results to 'save_path'.

    Neuron segmentation in EM is ranked by the CREMI score, not by mSA, so those datasets report the
    VI and adapted-Rand components instead.

    Args:
        gt_paths: The ground-truth label arrays, or the paths to them.
        prediction_paths: The predicted segmentations, or the paths to them.
        dataset_name: The dataset the segmentations belong to.
        save_path: The filepath to write the result CSV to.

    Returns:
        The results as a DataFrame.
    """
    from micro_sam.v1.evaluation.evaluation import run_evaluation

    if dataset_name not in DATASETS_3D_EM:
        return run_evaluation(gt_paths=gt_paths, prediction_paths=prediction_paths, save_path=save_path)

    import pandas as pd
    from elf.evaluation import cremi_score

    rows = []
    for gt, seg in zip(gt_paths, prediction_paths):
        vi_split, vi_merge, adapted_rand, cremi = cremi_score(seg, gt)
        rows.append({
            "cremi": float(cremi),
            "vi_split": float(vi_split),
            "vi_merge": float(vi_merge),
            "adapted_rand": float(adapted_rand),
        })

    results = pd.DataFrame(rows).mean().to_frame().T
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results.to_csv(save_path, index=False)
    return results


def read_tuned_params(
    grid_search_root: str, dataset_name: str, model_type: str, checkpoint_checksum: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the best parameter combination of a grid search as a dict.

    New sweeps are keyed by checkpoint checksum. If no such sweep exists, an old checksum-less sweep
    is still accepted with a warning. The first row is the best combination and values are parsed as
    Python literals, so a tuple-valued 'candidate_threshold' survives the CSV round trip.

    Args:
        grid_search_root: The root the grid search wrote its per-model directories to.
        dataset_name: The dataset whose tuned parameters are read.
        model_type: The SAM2 backbone, which names the subdirectory.
        checkpoint_checksum: The effective weights, for an exact cache lookup. If omitted, read the
            legacy checksum-less location.

    Returns:
        The best combination, ready to be passed to the postprocessing or to 'generate'.
    """
    legacy_path = os.path.join(grid_search_root, model_type, f"{dataset_name}.csv")
    csv_path = (
        legacy_path if checkpoint_checksum is None else
        os.path.join(grid_search_root, model_type, checkpoint_checksum, f"{dataset_name}.csv")
    )
    if not os.path.exists(csv_path) and checkpoint_checksum is not None and os.path.exists(legacy_path):
        warnings.warn(
            f"Using legacy parameter sweep '{legacy_path}'. It has no checkpoint checksum, so its "
            "weights cannot be verified.",
            stacklevel=2,
        )
        csv_path = legacy_path
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"There is no grid search result at '{csv_path}'.")

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"The grid search result at '{csv_path}' is empty.")

    params = {}
    for key, value in rows[0].items():
        if key.endswith(("_mean", "_std")) or key == "n_images":
            continue
        try:
            params[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            params[key] = value

    # These are counts, and a column that ever held a NaN comes back as a float.
    for key in ("min_size", "n_iter", "min_candidate_size", "n_objects_per_pass"):
        if key in params:
            params[key] = int(params[key])
    return params


def _check_key(path: str, key: Optional[str], kind: str) -> None:
    if key is None:
        return
    try:
        with open_file(path, mode="r") as f:
            if key not in f:
                raise RuntimeError(f"Missing {kind} key '{key}' in '{path}'.")
    except Exception as e:
        raise RuntimeError(f"Could not open {kind} data key '{key}' in '{path}': {e}") from e


def check_data_download(dataset_name: str, data_root: str, download: bool = True) -> None:
    """Fail fast if a dataset cannot be resolved from the local data root.

    The check goes through `get_data_paths(..., download=download)`, so it catches a missing download,
    an invalid split and an unavailable cached file before the model loads. It may download a missing
    dataset once. The evaluation itself reads the cached local data afterwards.
    """
    try:
        raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root, download=download)
    except Exception as e:
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}'. "
            "The dataset-specific get_*_paths helper could not resolve local data."
        ) from e

    if not raw_paths:
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            "no raw paths were found. The data is probably missing or all subsets were skipped."
        )
    if not label_paths:
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            "no label paths were found. The data is probably missing or all subsets were skipped."
        )
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            f"found {len(raw_paths)} raw paths but {len(label_paths)} label paths."
        )

    missing = []
    for raw_path, label_path in zip(raw_paths, label_paths):
        if not os.path.exists(raw_path):
            missing.append(raw_path)
        if label_path != raw_path and not os.path.exists(label_path):
            missing.append(label_path)

    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:10])
        suffix = "" if len(missing) <= 10 else f"\n  ... and {len(missing) - 10} more"
        raise RuntimeError(
            f"Data check failed for dataset '{dataset_name}' in '{data_root}': "
            f"{len(missing)} referenced file(s) do not exist:\n{examples}{suffix}"
        )

    _check_key(raw_paths[0], raw_key, "raw")
    _check_key(label_paths[0], label_key, "label")

    print(f"Data check passed for '{dataset_name}': {len(raw_paths)} sample(s).")


CROP_SHAPE_2D = (512, 512)
CROP_SHAPE_3D = (8, 512, 512)


def ensure_8bit_range(raw):
    """Scale raw data into the [0, 255] range the evaluation feeds the models with."""
    if raw.size == 0:
        return raw.astype("float32", copy=False)
    # `read_2d` returns channel-last images. Preserve the contrast of every microscopy channel
    # instead of letting the channel with the largest values determine the shared percentile range.
    spatial_axes = (0, 1) if raw.ndim == 3 and raw.shape[-1] in (1, 2, 3, 4) else None
    return normalize_raw(raw, axis=spatial_axes) * 255.0


def read_2d(path, key):
    """Read a 2d array from an image file, or from an H5 / zarr file using 'key'."""
    if key is not None:
        arr = open_file(path, mode="r")[key][:]
    else:
        arr = np.asarray(imageio.imread(path))
    # Transpose channel-first (C, H, W) to channel-last (H, W, C).
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        arr = arr.transpose(1, 2, 0)
    # Some 2d datasets mix in multi-frame stacks, e.g. yeaz. Evaluate their first frame.
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]
    return arr


def sorted_path_pairs(raw_paths, label_paths):
    """Sort raw and label paths as pairs, so the pairing survives names that sort differently."""
    return sorted(zip(raw_paths, label_paths), key=lambda pair: (str(pair[0]), str(pair[1])))


def interactive_result_name(
    dataset_name, method, model_type, prompt, iteration,
    ndim=2, use_masks=True, mask_threshold=0.0, min_size=0,
):
    """Build the name of the result CSV for one iteration of an interactive run.

    The name encodes every setting that changes the numbers, so one run cannot reuse the results of
    another.
    """
    dim_suffix = "" if ndim == 2 else "_3d"
    tag = interactive_run_tag(ndim, use_masks, mask_threshold, min_size)
    return f"{dataset_name}_{method}_{model_type}{dim_suffix}_{prompt}{tag}_iter{iteration:02d}.csv"


def interactive_run_tag(ndim=2, use_masks=True, mask_threshold=0.0, min_size=0):
    """Build the settings suffix for an interactive run's result names and prediction directory.

    Both use one tag, so a run can never read back the cached predictions of another run.
    """
    # Only the 2d path chooses between mask logits and binarized masks.
    tag = "" if ndim == 3 else ("_with_masks" if use_masks else "_without_masks")
    if ndim == 2 and mask_threshold != 0.0:
        tag += f"_t{mask_threshold:g}"
    if min_size:
        tag += f"_min{min_size}"
    return tag


def apply_min_size(labels, min_size, dataset_name):
    """Drop ground-truth objects below 'min_size' pixels, and warn if that removes too many.

    No single threshold suits every dataset. Gonuclear nuclei have a median of about 3200 pixels per
    object, while cremi neurite cross-sections in a thin crop have a median of about 6.
    """
    if not min_size:
        return labels
    before = len(np.unique(labels)) - 1
    filtered = size_filter(seg=labels, min_size=min_size)
    after = len(np.unique(filtered)) - 1
    if before and (before - after) / before > 0.25:
        warnings.warn(
            f"min_size={min_size} removes {before - after} of {before} ground-truth objects in "
            f"'{dataset_name}'. That is more than a quarter, so the threshold is likely too large "
            f"for this dataset and is discarding real annotations."
        )
    return filtered


def drop_severed_objects(labels, min_size):
    """Drop the objects that a crop face cut down to a sliver, in a ground truth or a prediction.

    Both conditions are needed. A size threshold alone also deletes small interior objects, and
    border contact alone deletes large cells that only reach the edge. The caller filters the ground
    truth and the prediction the same way, so a dropped remnant never becomes a false positive.
    """
    if not min_size:
        return labels
    if labels.ndim == 2:
        edges = (labels[0], labels[-1], labels[:, 0], labels[:, -1])
    else:
        # In-plane faces only, since a thin z-crop cuts almost every object on the first and last slice.
        edges = (labels[:, 0], labels[:, -1], labels[:, :, 0], labels[:, :, -1])
    border_ids = np.unique(np.concatenate([np.unique(edge) for edge in edges]))
    border_ids = border_ids[border_ids != 0]
    if border_ids.size == 0:
        return labels

    ids, sizes = np.unique(labels[labels != 0], return_counts=True)
    severed = np.intersect1d(border_ids, ids[sizes < min_size], assume_unique=True)
    if severed.size == 0:
        return labels
    return np.where(np.isin(labels, severed), 0, labels).astype(labels.dtype)


def severed_objects(gt, max_span=2):
    """The ground-truth objects that occupy no more than 'max_span' slices of a volume.

    A volumetric object is anchored on the slice its density converges on, and that density scales
    with the object's size. An object that the crop reduced to one or two slices never reaches
    'candidate_threshold', so it is never proposed. Separating these says how much of the gap to the
    ground truth is the crop rather than the method.

    Args:
        gt: The ground-truth labels, shape (Z, Y, X).
        max_span: The largest number of slices a severed object may span.

    Returns:
        The labels of the severed objects with everything else zeroed, and their ids.
    """
    ids = np.unique(gt)
    ids = ids[ids != 0]
    if len(ids) == 0:
        return np.zeros_like(gt), np.array([], dtype=gt.dtype)
    spans = np.array([int((gt == index).any(axis=(1, 2)).sum()) for index in ids])
    thin = ids[spans <= max_span]
    return np.where(np.isin(gt, thin), gt, 0).astype(gt.dtype), thin


def unmatched_objects(gt, segmentation, iou_threshold=0.5):
    """The ground-truth objects that no predicted instance matches at the given IoU.

    Matched the way `mean_segmentation_accuracy` matches at its lowest threshold, so these are the
    objects the result genuinely lost rather than segmented imprecisely.

    Args:
        gt: The ground-truth labels.
        segmentation: The predicted instance segmentation.
        iou_threshold: The IoU a prediction must reach to count as a match.

    Returns:
        The labels of the unmatched objects, with everything else zeroed.
    """
    ids = np.unique(gt)
    ids = ids[ids != 0]
    missed = []
    for index in ids:
        mask = gt == index
        overlapping = segmentation[mask]
        overlapping = overlapping[overlapping != 0]
        if overlapping.size == 0:
            missed.append(index)
            continue
        candidates, counts = np.unique(overlapping, return_counts=True)
        best = int(np.argmax(counts))
        intersection = int(counts[best])
        union = int(mask.sum()) + int((segmentation == candidates[best]).sum()) - intersection
        if intersection / union < iou_threshold:
            missed.append(index)
    return np.where(np.isin(gt, missed), gt, 0).astype(gt.dtype)


def genuine_misses(gt, segmentation, iou_threshold=0.5, max_span=2):
    """How many ground-truth objects the result lost that the crop did not sever.

    An aggregate metric hides which objects went missing. A run that recovers objects while costing
    precision elsewhere is not the same as one that does neither, so this counts the losses the
    method is answerable for.

    Args:
        gt: The ground-truth labels, shape (Z, Y, X).
        segmentation: The predicted instance segmentation.
        iou_threshold: The IoU a prediction must reach to count as a match.
        max_span: The largest number of slices a severed object may span.

    Returns:
        The number of unmatched objects, and how many of those the crop did not sever.
    """
    unmatched_ids = np.unique(unmatched_objects(gt, segmentation, iou_threshold))
    unmatched_ids = unmatched_ids[unmatched_ids != 0]
    _, thin = severed_objects(gt, max_span)
    return len(unmatched_ids), int((~np.isin(unmatched_ids, thin)).sum())


def load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset_name):
    """Load one 2d sample the way the evaluation scores it.

    The parameter search and the evaluation both call this function, so both use the same data.
    """
    # Normalize before cropping, so that the percentiles cover the whole image.
    image = ensure_8bit_range(read_2d(raw_path, raw_key))
    roi = _center_crop_roi(image.shape[:2], CROP_SHAPE_2D)
    gt = connected_components(read_2d(label_path, label_key)[roi]).astype("uint32")
    return image[roi], drop_severed_objects(gt, GT_MIN_SIZE_2D.get(dataset_name, 0))


def load_evaluation_sample_3d(
    raw_path, label_path, raw_key, label_key, dataset_name,
    crop_shape=CROP_SHAPE_3D, z_range=None, min_size=0,
):
    """Load one volumetric sample the way the evaluation scores it."""
    raw, labels, valid_roi = load_volume(
        raw_path, label_path, raw_key, label_key, dataset_name, crop_shape, z_range=z_range
    )
    return raw, apply_min_size(labels, min_size, dataset_name), valid_roi


def load_data(dataset_name, data_root, ndim, min_size=0, split="test", crop_shape=None, z_range=None):
    """Yield (image_or_volume, labels, valid_roi) triples for the given dataset.

    valid_roi is a boolean mask that is True where the data is annotated. It is None for every
    dataset except platynereis_nuclei, which is annotated only in part.

    The filtering happens here, in the single source of the labels used for both prompting and
    scoring. If only the prompting copy were filtered, the dropped objects would stay in the scored
    ground truth and count as unmatched.

    Args:
        dataset_name: The dataset to load.
        data_root: The root the data lives in.
        ndim: The number of spatial dimensions, 2 or 3.
        min_size: Drop ground-truth objects below this many pixels (3d only).
        split: The split to load, 'test' or the held-out 'val', see VAL_SPLITS.
        crop_shape: The 3d center crop. Defaults to CROP_SHAPE_3D.
        z_range: Restrict a volume to a z-slab before cropping, see VAL_Z_RANGE.

    Yields:
        One (image_or_volume, labels, valid_roi) triple per sample.
    """
    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset_name, data_root, split=split)
    for raw_path, label_path in sorted_path_pairs(raw_paths, label_paths):
        if ndim == 3:
            yield load_evaluation_sample_3d(
                raw_path, label_path, raw_key, label_key, dataset_name,
                crop_shape=crop_shape or CROP_SHAPE_3D, z_range=z_range, min_size=min_size,
            )
        else:
            image, gt = load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset_name)
            yield image, gt, None


def n_samples(dataset_name, data_root, split="test"):
    """The number of samples of a split, for a progress bar over `load_data`."""
    return len(get_data_paths(dataset_name, data_root, split=split)[0])


def has_val_split(dataset_name: str) -> bool:
    """Whether a dataset holds data out from the samples the evaluation scores.

    Only these datasets can be tuned honestly: everywhere else a sweep would select its parameters
    on the very samples the reported score is measured on. See VAL_SPLITS and VAL_Z_RANGE.
    """
    return dataset_name in VAL_SPLITS
