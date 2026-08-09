import os
import warnings
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
from skimage.measure import label as connected_components

import torch

from elf.io import open_file
from torch_em.data import datasets
from torch_em.util.image import load_image

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

# Test images per LIVECell cell type; eight types fill MAX_EVALUATION_SAMPLES. 0 takes the whole split.
LIVECELL_PER_CELL_TYPE = int(os.environ.get("MICRO_SAM_LIVECELL_PER_CELL_TYPE", "25")) or None

# LIVECell test images whose annotation is incomplete: 2 labelled objects in a confluent crop, with
# predicted-to-annotated foreground ratios of 348x and 24x. Both are outside the stratified subset.
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


# The jointly finetuned (interactive + automatic) SAM2 models for cell segmentation.
JOINT_CHECKPOINT_ROOT = os.path.join(_MODELS_DIR, "joint", "v2", "checkpoints")
# The joint checkpoints are split into loadable weight files here, see 'export_joint_checkpoint'.
JOINT_EXPORT_ROOT = os.path.join(_MODELS_DIR, "exported", "joint", "v2")

# 2D LM datasets
DATASETS_2D = [
    "livecell",
    "arvidsson", "bitdepth_nucseg", "cellbindb", "cellpose_data",
    "covid_if", "cvz_fluo", "deepbacs", "deepseas", "dic_hepg2", "dsb",
    "dynamicnuclearnet", "hpa", "microbeseg", "neurips_cellseg", "omnipose",
    "segpc", "tissuenet", "usiigaci", "vicar", "yeaz",
]

# Ground-truth size floor, applied by `baselines_common._load_data` to drop the crop-severed slivers
# that relabelling promotes to objects. Defines the ground truth, so it is measured, never tuned.
GT_MIN_SIZE_2D = {
    "livecell": 50,
    "cellpose": 20, "deepbacs": 50, "dynamicnuclearnet": 50, "tissuenet": 10,
    "u20s": 10, "vicar": 25, "yeaz": 10,
}

# 3D LM datasets
DATASETS_3D_LM = [
    "blastospim", "cartocell", "celegans_atlas", "cellseg_3d", "embedseg",
    "gonuclear", "mouse_embryo", "nis3d", "plantseg", "pnas_arabidopsis",
]

# 3D EM datasets
DATASETS_3D_EM = ["platynereis_nuclei", "cremi", "snemi", "humanneurons"]

DATASETS_3D = DATASETS_3D_LM + DATASETS_3D_EM


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
    dataset_name: str, data_root: str, download: bool = False
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name == "livecell":
        # Stratified: the test set is sorted by cell type, so heading it covers two of the eight.
        img, gt = _get_livecell_paths(
            input_folder=os.path.join(p, "livecell"), split="test",
            n_val_per_cell_type=LIVECELL_PER_CELL_TYPE,
        )
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
            path=os.path.join(p, "deepbacs"), bac_type="mixed", split="test", download=download,
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
            path=os.path.join(p, "dic_hepg2"), split="test", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dsb":
        img, gt = datasets.dsb.get_dsb_paths(
            path=os.path.join(p, "dsb"), source="full", split=None, download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "dynamicnuclearnet":
        paths = datasets.dynamicnuclearnet.get_dynamicnuclearnet_paths(
            path=os.path.join(p, "dynamicnuclearnet"), split="test", download=download,
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
        # No test split. Use validation.
        paths = datasets.segpc.get_segpc_paths(
            path=os.path.join(p, "segpc"), split="validation", download=download,
        )
        return sorted(paths), sorted(paths), "raw", "labels/cells"

    if dataset_name == "tissuenet":
        paths = datasets.tissuenet.get_tissuenet_paths(
            path=os.path.join(p, "tissuenet"), split="test", download=download,
        )
        # rgb composite + cell labels matches training convention
        return sorted(paths), sorted(paths), "raw/rgb", "labels/cell"

    if dataset_name == "usiigaci":
        # No test split. Use val.
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
    dataset_name: str, data_root: str, download: bool = False
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
            path=os.path.join(p, "celegans_atlas"), split="test", download=download,
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
            name="Mouse-Skull-Nuclei-CBG", split="test", download=download,
        )
        return (*_sorted_pairs(img, gt), None, None)

    if dataset_name == "gonuclear":
        paths = datasets.gonuclear.get_gonuclear_paths(
            path=os.path.join(p, "gonuclear"), download=download,
        )
        return sorted(paths), sorted(paths), "raw/nuclei", "labels/nuclei"

    if dataset_name == "mouse_embryo":
        # No test split. Use val.
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
        paths = datasets.cremi.get_cremi_paths(
            path=os.path.join(p, "cremi"), samples=("A", "B", "C"), download=download,
        )
        return sorted(paths), sorted(paths), "volumes/raw", "volumes/labels/neuron_ids"

    if dataset_name == "snemi":
        # The test file has no labels. Training used train-slices 70+, so slices [0:70] are holdout.
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
    dataset_name: str, data_root: str, download: bool = False
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """Return (raw_paths, label_paths, raw_key, label_key) for a dataset's test split.

    raw_key / label_key are None for plain image files and non-None for H5 / zarr.
    """
    all_datasets = DATASETS_2D + DATASETS_3D
    assert dataset_name in all_datasets, (
        f"Unsupported dataset: '{dataset_name}'. Choose from {all_datasets}."
    )
    if dataset_name in DATASETS_2D:
        return _get_2d_data_paths(dataset_name, data_root, download=download)
    if dataset_name in DATASETS_3D_LM:
        return _get_3d_lm_data_paths(dataset_name, data_root, download=download)
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
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load a 3D volume, apply dataset-specific preprocessing, and center-crop.

    Returns (raw, labels, valid_roi) where valid_roi is a boolean mask (True = annotated)
    for partially annotated datasets (platynereis_nuclei), or None for all others.
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
        # Restrict to holdout slices [0:70]; training used slices 70+.
        raw, labels = raw[:70], labels[:70]

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


# Model helpers shared between the evaluation scripts

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


def export_joint_checkpoint(
    model_type: str, checkpoint: str = "best", export_root: str = JOINT_EXPORT_ROOT
) -> Tuple[str, str]:
    """Split a joint checkpoint into an interactive and an automatic weight file.

    The joint trainer bundles the SAM2 weights ('model_state'), the UniSAM2 decoder weights
    ('unetr_state') and pickled trainer state in a single file. That file cannot be loaded by
    `sam2.build_sam`, which reads `torch.load(...)['model']` with `weights_only=True`. Both
    exported files are plain tensor dicts, mirroring `scripts/model_export/export_sam2_cells_model.py`.
    Existing exports are reused.

    Args:
        model_type: The SAM2 backbone the model was finetuned from, e.g. 'hvit_b'.
        checkpoint: Which trainer checkpoint to export, 'best' or 'latest'.
        export_root: The directory the exported weight files are written to.

    Returns:
        The paths to the interactive (SAM2) and the automatic (UniSAM2 decoder) weight files.
    """
    name = f"joint_sam2_{model_type}_{checkpoint}"
    interactive_path = os.path.join(export_root, f"{name}.pt")
    decoder_path = os.path.join(export_root, f"{name}_decoder.pt")
    if os.path.exists(interactive_path) and os.path.exists(decoder_path):
        return interactive_path, decoder_path

    checkpoint_path = get_joint_checkpoint(model_type, checkpoint)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    missing = [key for key in ("model_state", "unetr_state") if key not in state]
    if missing:
        raise RuntimeError(f"'{checkpoint_path}' is not a joint checkpoint, it is missing {missing}.")

    os.makedirs(export_root, exist_ok=True)
    _save_atomic({"model": _strip_ddp_prefix(state["model_state"]), "model_type": model_type}, interactive_path)
    _save_atomic(_strip_ddp_prefix(state["unetr_state"]), decoder_path)
    print(f"Exported '{checkpoint_path}' to '{interactive_path}' and '{decoder_path}'.")
    return interactive_path, decoder_path


DATASET_SPACING: dict = {
    # z/xy voxel ratios from published acquisition parameters
    "embedseg": (4, 1, 1),  # Mouse-Skull-Nuclei-CBG: z=1µm, xy=0.25µm
    "blastospim": (10, 1, 1),  # SPIM: z≈2µm, xy≈0.208µm
    "mouse_embryo": (4, 1, 1),  # confocal: z≈1µm, xy≈0.22µm
}


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


def predict_unisam2(model, raw, ndim, device, normalization=None):
    from micro_sam.v2.instance_segmentation import get_unisam2_segmentation_generator
    # UniSAM2 expects single-channel input, so a trailing channel axis is averaged away, as in
    # 'read_image_2d'.
    if raw.ndim > ndim:
        raw = raw.mean(axis=-1)

    is_3d = (ndim == 3)
    # Tiling an image that fits the training patch changes the encoder's scale and the normalization
    # window.
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

    This intentionally calls the dataset-specific torch-em `get_*_paths` helpers
    via `get_data_paths(..., download=download)`, so missing downloads, invalid
    splits, and unavailable cached files are caught before model loading. By
    default this check is allowed to download missing datasets once, while the
    actual evaluation code still reads cached local data afterwards.
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
