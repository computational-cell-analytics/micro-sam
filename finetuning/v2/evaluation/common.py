import os
import warnings
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
from skimage.measure import label as connected_components

from elf.io import open_file
from torch_em.data import datasets
from torch_em.transform.raw import normalize
from torch_em.util.image import load_image

from micro_sam.evaluation.livecell import _get_livecell_paths


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

_MODELS_DIR = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2"

CHECKPOINT_PATHS = {
    "sam2.1": {
        "hvit_t": os.path.join(_MODELS_DIR, "sam2.1_hiera_tiny.pt"),
        "hvit_s": os.path.join(_MODELS_DIR, "sam2.1_hiera_small.pt"),
        "hvit_b": os.path.join(_MODELS_DIR, "sam2.1_hiera_base_plus.pt"),
        "hvit_l": os.path.join(_MODELS_DIR, "sam2.1_hiera_large.pt"),
    },
    "sam2.0": {
        "hvit_t": os.path.join(_MODELS_DIR, "sam2_hiera_tiny.pt"),
        "hvit_s": os.path.join(_MODELS_DIR, "sam2_hiera_small.pt"),
        "hvit_b": os.path.join(_MODELS_DIR, "sam2_hiera_base_plus.pt"),
        "hvit_l": os.path.join(_MODELS_DIR, "sam2_hiera_large.pt"),
    },
}

# 2D LM datasets
DATASETS_2D = [
    "livecell",
    "arvidsson", "bitdepth_nucseg", "cellbindb", "cellpose_data",
    "covid_if", "cvz_fluo", "deepbacs", "deepseas", "dic_hepg2", "dsb",
    "dynamicnuclearnet", "hpa", "microbeseg", "neurips_cellseg", "omnipose",
    "segpc", "tissuenet", "usiigaci", "vicar", "yeaz",
]

# 3D LM datasets
DATASETS_3D_LM = [
    "blastospim", "cartocell", "celegans_atlas", "cellseg_3d", "embedseg",
    "gonuclear", "mouse_embryo", "nis3d", "plantseg", "pnas_arabidopsis",
]

# 3D EM datasets
DATASETS_3D_EM = ["lucchi", "platynereis_nuclei", "cremi", "snemi", "humanneurons"]

DATASETS_3D = DATASETS_3D_LM + DATASETS_3D_EM


def _get_2d_data_paths(
    dataset_name: str, data_root: str
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name == "livecell":
        img, gt = _get_livecell_paths(input_folder=os.path.join(p, "livecell"), split="test")
        return sorted(img), sorted(gt), None, None

    if dataset_name == "arvidsson":
        img, gt = datasets.arvidsson.get_arvidsson_paths(
            path=os.path.join(p, "arvidsson"), split="test", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "bitdepth_nucseg":
        img, gt = datasets.bitdepth_nucseg.get_bitdepth_nucseg_paths(
            path=os.path.join(p, "bitdepth_nucseg"), download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "cellbindb":
        img, gt = datasets.cellbindb.get_cellbindb_paths(
            path=os.path.join(p, "cellbindb"), download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "cellpose_data":
        img, gt = [], []
        for choice in ("cyto", "cyto2"):
            i, g = datasets.cellpose.get_cellpose_paths(
                path=os.path.join(p, "cellpose"), split="test", choice=choice, download=False,
            )
            img.extend(i)
            gt.extend(g)
        return sorted(img), sorted(gt), None, None

    if dataset_name == "covid_if":
        paths = datasets.covid_if.get_covid_if_paths(
            path=os.path.join(p, "covid_if"), download=False,
        )
        return sorted(paths), sorted(paths), "raw/nuclei/s0", "labels/nuclei/s0"

    if dataset_name == "cvz_fluo":
        img, gt = [], []
        for stain in ("cell", "dapi"):
            i, g = datasets.cvz_fluo.get_cvz_fluo_paths(
                path=os.path.join(p, "cvz"), stain_choice=stain, download=False,
            )
            img.extend(i)
            gt.extend(g)
        return sorted(img), sorted(gt), None, None

    if dataset_name == "deepbacs":
        img_folder, label_folder = datasets.deepbacs.get_deepbacs_paths(
            path=os.path.join(p, "deepbacs"), bac_type="mixed", split="test", download=False,
        )
        img = sorted(glob(os.path.join(img_folder, "*.tif")))
        gt = sorted(glob(os.path.join(label_folder, "*.tif")))
        return img, gt, None, None

    if dataset_name == "deepseas":
        img, gt = datasets.deepseas.get_deepseas_paths(
            path=os.path.join(p, "deepseas"), split="test", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "dic_hepg2":
        img, gt = datasets.dic_hepg2.get_dic_hepg2_paths(
            path=os.path.join(p, "dic_hepg2"), split="test", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "dsb":
        img, gt = datasets.dsb.get_dsb_paths(
            path=os.path.join(p, "dsb"), source="full", split="test", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "dynamicnuclearnet":
        paths = datasets.dynamicnuclearnet.get_dynamicnuclearnet_paths(
            path=os.path.join(p, "dynamicnuclearnet"), split="test", download=False,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "hpa":
        paths = datasets.hpa.get_hpa_segmentation_paths(
            path=os.path.join(p, "hpa"), split="test", download=False,
        )
        # protein channel for cell body segmentation (peft-sam convention)
        return sorted(paths), sorted(paths), "raw/protein", "labels"

    if dataset_name == "microbeseg":
        img, gt = datasets.microbeseg.get_microbeseg_paths(
            path=os.path.join(p, "microbeseg"), split="test",
            annotation_type="30min-man", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "neurips_cellseg":
        img, gt = datasets.neurips_cell_seg.get_neurips_cellseg_paths(
            root=os.path.join(p, "neurips_cellseg"), split="test", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "omnipose":
        img, gt = [], []
        for choice in ("bact_fluor", "bact_phase", "worm", "worm_high_res"):
            try:
                i, g = datasets.omnipose.get_omnipose_paths(
                    path=os.path.join(p, "omnipose"), split="test",
                    data_choice=choice, download=False,
                )
                img.extend(i)
                gt.extend(g)
            except Exception as e:
                warnings.warn(f"Skipping omnipose choice '{choice}': {e}")
        return sorted(img), sorted(gt), None, None

    if dataset_name == "segpc":
        # No test split; use validation.
        paths = datasets.segpc.get_segpc_paths(
            path=os.path.join(p, "segpc"), split="validation", download=False,
        )
        return sorted(paths), sorted(paths), "raw", "labels/cells"

    if dataset_name == "tissuenet":
        paths = datasets.tissuenet.get_tissuenet_paths(
            path=os.path.join(p, "tissuenet"), split="test", download=False,
        )
        # rgb composite + cell labels matches training convention
        return sorted(paths), sorted(paths), "raw/rgb", "labels/cell"

    if dataset_name == "usiigaci":
        # No test split; use val.
        img, gt = datasets.usiigaci.get_usiigaci_paths(
            path=os.path.join(p, "usiigaci"), split="val", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "vicar":
        img, gt = datasets.vicar.get_vicar_paths(
            path=os.path.join(p, "vicar"), download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "yeaz":
        img, gt = [], []
        for choice in ("bf", "phc"):
            i, g = datasets.yeaz.get_yeaz_paths(
                path=os.path.join(p, "yeaz"), choice=choice, split="test", download=False,
            )
            img.extend(i)
            gt.extend(g)
        return sorted(img), sorted(gt), None, None

    raise ValueError(f"Unknown 2D dataset: {dataset_name!r}")


def _get_3d_lm_data_paths(
    dataset_name: str, data_root: str
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name == "blastospim":
        paths = datasets.blastospim.get_blastospim_paths(
            path=os.path.join(p, "blastospim"), download=False,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    if dataset_name == "cartocell":
        img, gt = [], []
        for name in ("eggChambers", "embryoids", "MDCK-Normoxia", "MDCK-Hypoxia"):
            try:
                i, g = datasets.cartocell.get_cartocell_paths(
                    path=os.path.join(p, "cartocell"), split="test", name=name, download=False,
                )
                img.extend(i)
                gt.extend(g)
            except Exception as e:
                warnings.warn(f"Skipping cartocell name '{name}': {e}")
        return sorted(img), sorted(gt), None, None

    if dataset_name == "celegans_atlas":
        img, gt = datasets.celegans_atlas.get_celegans_atlas_paths(
            path=os.path.join(p, "celegans_atlas"), split="test", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "cellseg_3d":
        img, gt = datasets.cellseg_3d.get_cellseg_3d_paths(
            path=os.path.join(p, "cellseg_3d"), download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "embedseg":
        img, gt = datasets.embedseg_data.get_embedseg_paths(
            path=os.path.join(p, "embedseg"),
            name="Mouse-Skull-Nuclei-CBG", split="test", download=False,
        )
        return list(img), list(gt), None, None

    if dataset_name == "gonuclear":
        paths = datasets.gonuclear.get_gonuclear_paths(
            path=os.path.join(p, "gonuclear"), download=False,
        )
        return sorted(paths), sorted(paths), "raw/nuclei", "labels/nuclei"

    if dataset_name == "mouse_embryo":
        # No test split; use val.
        paths = datasets.mouse_embryo.get_mouse_embryo_paths(
            path=os.path.join(p, "mouse_embryo"), name="nuclei", split="val", download=False,
        )
        return sorted(paths), sorted(paths), "raw", "label"

    if dataset_name == "nis3d":
        img, gt = datasets.nis3d.get_nis3d_paths(
            path=os.path.join(p, "nis3d"), split="test", download=False,
        )
        return sorted(img), sorted(gt), None, None

    if dataset_name == "plantseg":
        all_paths = []
        for name, folder in (
            ("nuclei", "plantseg"),
            ("ovules", "plantseg_ovules"),
            ("root", "plantseg_root"),
        ):
            try:
                ps = datasets.plantseg.get_plantseg_paths(
                    path=os.path.join(p, folder), name=name, split="test", download=False,
                )
                all_paths.extend(ps)
            except Exception as e:
                warnings.warn(f"Skipping plantseg name '{name}': {e}")
        return sorted(all_paths), sorted(all_paths), "raw", "label"

    if dataset_name == "pnas_arabidopsis":
        paths = datasets.pnas_arabidopsis.get_pnas_arabidopsis_paths(
            path=os.path.join(p, "pnas_arabidopsis"), download=False,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    raise ValueError(f"Unknown 3D LM dataset: {dataset_name!r}")


def _get_3d_em_data_paths(
    dataset_name: str, data_root: str
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    p = data_root

    if dataset_name == "lucchi":
        path = datasets.lucchi.get_lucchi_paths(
            path=os.path.join(p, "lucchi"), split="test", download=False,
        )
        return [path], [path], "raw", "labels"

    if dataset_name == "platynereis_nuclei":
        paths = datasets.platynereis.get_platynereis_paths(
            path=os.path.join(p, "platynereis"), sample_ids=None, name="nuclei", download=False,
        )
        return paths, paths, "volumes/raw", "volumes/labels/nucleus_instance_labels"

    if dataset_name == "cremi":
        paths = datasets.cremi.get_cremi_paths(
            path=os.path.join(p, "cremi"), samples=("A", "B", "C"), download=False,
        )
        return sorted(paths), sorted(paths), "volumes/raw", "volumes/labels/neuron_ids"

    if dataset_name == "snemi":
        # The test file has no labels; training used train-slices 70+, so slices [0:70] are holdout.
        path = datasets.snemi.get_snemi_paths(
            path=os.path.join(p, "snemi"), sample="train", download=False,
        )
        return [path], [path], "volumes/raw", "volumes/labels/neuron_ids"

    if dataset_name == "humanneurons":
        paths = datasets.humanneurons.get_humanneurons_paths(
            path=os.path.join(p, "humanneurons"), download=False,
        )
        return sorted(paths), sorted(paths), "raw", "labels"

    raise ValueError(f"Unknown 3D EM dataset: {dataset_name!r}")


def get_data_paths(
    dataset_name: str, data_root: str
) -> Tuple[List[str], List[str], Optional[str], Optional[str]]:
    """Return (raw_paths, label_paths, raw_key, label_key) for a dataset's test split.

    raw_key / label_key are None for plain image files and non-None for H5 / zarr.
    """
    all_datasets = DATASETS_2D + DATASETS_3D
    assert dataset_name in all_datasets, (
        f"Unsupported dataset: '{dataset_name}'. Choose from {all_datasets}."
    )
    if dataset_name in DATASETS_2D:
        return _get_2d_data_paths(dataset_name, data_root)
    if dataset_name in DATASETS_3D_LM:
        return _get_3d_lm_data_paths(dataset_name, data_root)
    return _get_3d_em_data_paths(dataset_name, data_root)


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
) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 3D volume, apply dataset-specific preprocessing, and center-crop."""
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

    if dataset_name == "platynereis_nuclei":
        labels = labels.astype("int64")
        labels[labels == -1] = 0

    if ensure_8bit and raw.max() > 255:
        raw = normalize(raw) * 255

    roi = _center_crop_roi(raw.shape, crop_shape)
    raw, labels = raw[roi], labels[roi]

    if ensure_instances:
        labels = connected_components(labels)

    assert raw.shape == labels.shape, f"Shape mismatch: raw {raw.shape} vs labels {labels.shape}"
    return raw.astype("float32"), labels.astype("uint32")


# UniSAM2 helpers shared between evaluate_2d and evaluate_3d

_UNISAM2_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/automatic/v1"
UNISAM2_CHECKPOINT = os.path.join(_UNISAM2_ROOT, "checkpoints", "unisam2-both", "best.pt")

_EM_DATASETS = {"lucchi", "platynereis_nuclei", "cremi", "snemi", "humanneurons"}

_DATASET_SPACING: dict = {
    # z/xy voxel ratios from published acquisition parameters
    "embedseg": (4, 1, 1),    # Mouse-Skull-Nuclei-CBG: z=1µm, xy=0.25µm
    "blastospim": (10, 1, 1),  # SPIM: z≈2µm, xy≈0.208µm
    "mouse_embryo": (4, 1, 1),  # confocal: z≈1µm, xy≈0.22µm
}


def load_unisam2_model(checkpoint_path, device):
    import torch
    from micro_sam.v2.models.util import UniSAM2
    model = UniSAM2(encoder="hvit_t", output_channels=4)
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(state["model_state"])
    model.to(device)
    model.eval()
    return model


def predict_unisam2(model, raw, ndim, device):
    from torch_em.util.prediction import predict_with_halo

    def _preprocess(crop):
        return np.concatenate([normalize(crop)] * 3, axis=0)

    is_3d = (ndim == 3)
    block_shape = (4, 384, 384) if is_3d else (1, 384, 384)
    halo = (2, 64, 64) if is_3d else (0, 64, 64)
    out = np.zeros((4, *raw.shape), dtype="float32")
    out = predict_with_halo(
        input_=raw[np.newaxis].astype("float32"),
        model=model,
        block_shape=block_shape,
        halo=halo,
        preprocess=_preprocess,
        gpu_ids=[device],
        output=out,
        with_channels=True,
    )
    if not is_3d:
        out = out[:, 0]
    return out


def postprocess_unisam2(out, dataset_name):
    from micro_sam.v2.postprocessing import flow_instance_segmentation, run_multicut
    fg = out[0]
    if dataset_name in _EM_DATASETS:
        boundary_map = fg.max() - fg
        boundary_map /= boundary_map.max()
        distances = np.stack([out[2], out[3]])
        seg = run_multicut(boundary_map, distances)
    else:
        spacing = _DATASET_SPACING.get(dataset_name, None)
        seg = flow_instance_segmentation(fg, out[1:], spacing=spacing)
    return seg.astype("uint32")
