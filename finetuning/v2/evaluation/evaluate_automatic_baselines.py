"""Benchmark evaluation of the automatic segmentation baselines, i.e. everything but micro-sam2.

Supported methods:
  cellpose: CellPose generalist models (cpsam, cpsam_v2, cpdino, cpdino-vitb; cyto3 and nuclei from CellPose 3)
  stardist: StarDist pretrained (2D_versatile_fluo / 3D_demo)
  cellsam: CellSAM pipeline (2d only)
  cellvit: CellViT++ nucleus segmentation (2d histopathology only)
  microsam_ais: micro-sam v1 automatic instance segmentation (vit_*_em_organelles for the EM organelle sets)
  microsam_apg: micro-sam v1 automatic prompt generation (vit_*_em_organelles for the EM organelle sets)
  segneuron: SegNeuron (3d EM only)
  focus3d: FOCUS-3D (3d LM only)
  mitonet: MitoNet from empanada (EM mitochondria, 2d and 3d)
  nucleonet: NucleoNet from empanada (EM nuclei, 2d and 3d)

micro-sam2 itself is evaluated by evaluate_automatic_segmentation.py, which also tunes it.

Usage examples:
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellpose -m cyto3
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method stardist
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellsam
    python evaluate_automatic_baselines.py -d pannuke -e <exp> --method cellvit -m sam_h_x40
    python evaluate_automatic_baselines.py -d embedseg -e <exp> --method microsam_ais -m vit_b
    python evaluate_automatic_baselines.py -d cremi -e <exp> --method segneuron
    python evaluate_automatic_baselines.py -d gonuclear -e <exp> --method focus3d -m nuclei
    python evaluate_automatic_baselines.py -d lucchi -e <exp> --method mitonet
    python evaluate_automatic_baselines.py -d lucchi -e <exp> --method microsam_ais -m vit_b_em_organelles
"""

import os
import json
import warnings
import argparse

import numpy as np

import torch

from elf.segmentation import features, multicut, watershed

from common import (
    CROP_SHAPE_3D, DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_LM, DATASETS_EM, DATASETS_EM_MITO,
    DATASETS_EM_NUCLEUS, DATASETS_HP, EVAL_CROPS, GT_MIN_SIZE_2D, check_data_download, drop_severed_objects,
    ensure_8bit_range, evaluate_samples, load_data, n_samples,
)

LM_DATASETS = set(DATASETS_2D + DATASETS_3D_LM)
EM_DATASETS = set(DATASETS_EM)
HP_DATASETS = set(DATASETS_HP)
EM_ORGANELLE_DATASETS = set(DATASETS_EM_MITO + DATASETS_EM_NUCLEUS)
METHODS = [
    "cellpose", "stardist", "cellsam", "cellvit", "microsam_ais", "microsam_apg", "segneuron", "focus3d", "mitonet",
    "nucleonet",
]

SEGNEURON_ROOT = "/mnt/vast-nhr/home/archit/u12090/SegNeuron"
SEGNEURON_CHECKPOINT = "/mnt/vast-nhr/projects/cidas/cca/models/segneuron/SegNeuronModel.ckpt"

# FOCUS-3D ships a general, a nuclei and a membrane checkpoint; pick one with -m.
FOCUS3D_DEFAULT_MODEL = "general"
# The radius FOCUS-3D is calibrated on, i.e. the value that leaves the volume unscaled in xy.
FOCUS3D_CELL_RADIUS = 15.0
# Throughput knobs, measured worth nothing here: the patch loop is 86% GPU forward. Batch 8 peaks at
# 6.8 GiB, so a job fits the 10 GB slice, where batch 16 needs 11.4 GiB and runs no faster.
FOCUS3D_BATCH_SIZE = 8
FOCUS3D_NUM_WORKERS = 4

# The CellPose 4 generalists. The cyto/nuclei checkpoints need CellPose 3, in the 'cellpose3' environment.
CELLPOSE_MODELS = ("cpsam", "cpsam_v2", "cpdino")
# The CellPose 3 generalists. Neither of them is a histopathology model.
CELLPOSE3_MODELS = ("cyto3", "nuclei")
# The IoU that links the masks of adjacent slices of a volume. It is the default of `cellpose.utils.stitch3D`.
# The default of `eval` is 0, which leaves the slices unlinked.
CELLPOSE_STITCH_THRESHOLD = 0.25

# The channels of the whole-cell and the nuclear marker in the loaded channel-last image, for the baselines that
# name their input channels. The other datasets are passed as loaded.
MARKER_CHANNELS = {
    "tissuenet": {"cell": 0, "nucleus": 1}, "cvz_fluo_cell": {"cell": 0, "nucleus": 1},
    "covid_if_cells": {"cell": 0, "nucleus": 1}, "pan_multiplex": {"cell": 0, "nucleus": 1},
}

# The marker in a grayscale image, for CellSAM. CellSAM returns a nuclear segmentation if only its nuclear slot is
# filled. The other grayscale datasets fill every slot.
GRAYSCALE_MARKER = {
    "dynamicnuclearnet": "nucleus", "covid_if_nuclei": "nucleus", "cvz_fluo_dapi": "nucleus",
    "xenium_nuclei": "nucleus", "bitdepth_nucseg": "nucleus", "bmgd": "nucleus", "cellbindb": "nucleus",
    "u20s": "nucleus", "tsakiroglou": "nucleus", "ifnuclei": "nucleus",
}

# The datasets that CellSAM reads as the mean of their channels. Their channels have no fixed role across images, or
# they are stained brightfield images such as H&E, where the mean measured better than RGB. IHC stays RGB, since
# its brown and blue carry the signal.
CHANNEL_MEAN_DATASETS = {
    "neurips_cellseg_fluorescence", "neurips_cellseg_label_free", "cpm17", "glysac", "histo_miner", "lizard",
    "lizard_mitosis", "lynsec_he", "monuseg", "pannuke", "puma", "tnbc_celltype", "pcns", "bccd", "organoid",
}

STARDIST_2D_MODEL = "2D_versatile_fluo"
# Histopathology uses the model trained on H&E, which reads the RGB image directly.
STARDIST_2D_HP_MODEL = "2D_versatile_he"
STARDIST_3D_MODEL = "3D_demo"

# micro-sam v1 model types
SAM_V1_MODEL_TYPE = "vit_b_lm"
SAM_V1_EM_ORGANELLE_MODEL_TYPE = "vit_b_em_organelles"

# CellViT++ ships a whole-slide pipeline that needs pathopatch, ray and openslide. The patch-level model and its
# post-processor are imported from the checkout instead, which only adds einops to the environment.
CELLVIT_ROOT = os.path.expanduser("~/CellViT-plus-plus")
CELLVIT_CHECKPOINT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/cellvit"
CELLVIT_CHECKPOINTS = {
    "sam_h_x40": "CellViT-SAM-H-x40.pth",
    "sam_h_x20": "CellViT-SAM-H-x20.pth",
    "vit256_x40": "CellViT-256-x40.pth",
    "vit256_x20": "CellViT-256-x20.pth",
}

# MitoNet and NucleoNet run from the empanada-napari checkout, which contains the empanada package.
EMPANADA_ROOT = os.path.expanduser("~/empanada-napari")
EMPANADA_CHECKPOINT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/empanada"
EMPANADA_MODELS = {"mitonet": "MitoNet_v1", "nucleonet": "NucleoNet_base_v2"}
EMPANADA_DATASETS = {"mitonet": set(DATASETS_EM_MITO), "nucleonet": set(DATASETS_EM_NUCLEUS)}
# The defaults of the empanada-napari inference widgets. A volume is segmented in xy only, and the objects are
# linked through z. The ortho-plane mode also segments xz and yz, which are only a few pixels wide in the thin and
# anisotropic test crops.
EMPANADA_PARAMS = {
    "label_divisor": 10000, "nms_threshold": 0.1, "nms_kernel": 3, "confidence_thr": 0.5,
    "fine_boundaries": False,
}
EMPANADA_PARAMS_3D = {"median_kernel_size": 3, "min_size": 500, "min_extent": 5}
# At the full resolution, the models split objects that are larger than in their training data. So the image
# downsampling of the widgets is tuned per dataset on its tuning crops, at most this many of them.
EMPANADA_SCALES = (1, 2, 4, 8)
EMPANADA_MAX_TUNING_SAMPLES = {2: 10, 3: 5}
# A downsampling replaces the default of 1 only if it scores this much higher, since a few crops are a noisy choice.
EMPANADA_MIN_GAIN = 0.02

# The z/xy anisotropy of each dataset (z_voxel / xy_voxel). FOCUS-3D takes it as its z_ratio.
DATASET_ANISOTROPY = {
    "embedseg_mouse_skull": 4.0,
    "embedseg_organoid": 6.0,
    "embedseg_platy_nuclei": 5.0,
    "blastospim": 10.0,
    "mouse_embryo": 4.0,
    "cremi": 10.0,   # z=40nm, xy=4nm
    "snemi": 5.0,    # z=30nm, xy=6nm
}


def _load_cellpose(model_type, device):
    """Load a CellPose model, supporting both the CellPose 3 and the CellPose 4 APIs.

    CellPose 4 dropped the `models.Cellpose` wrapper along with the cyto/nuclei checkpoints, and takes its
    own generalist models ('cpsam', 'cpsam_v2', 'cpdino', 'cpdino-vitb') through `pretrained_model`.
    """
    from cellpose import models
    use_gpu = (device != "cpu") and torch.cuda.is_available()

    if model_type in ("cyto", "cyto2", "cyto3", "nuclei"):
        if not hasattr(models, "Cellpose"):
            raise RuntimeError(
                f"'{model_type}' needs CellPose 3; the installed CellPose only provides "
                f"{getattr(models, 'MODEL_NAMES', ())}."
            )
        return models.Cellpose(gpu=use_gpu, model_type=model_type)

    if model_type in getattr(models, "MODEL_NAMES", ()):
        return models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)

    return models.CellposeModel(gpu=use_gpu, model_type=model_type)


def _load_stardist(ndim, dataset_name):
    from stardist.models import StarDist2D, StarDist3D
    if ndim == 3:
        return StarDist3D.from_pretrained(STARDIST_3D_MODEL)
    return StarDist2D.from_pretrained(STARDIST_2D_HP_MODEL if dataset_name in DATASETS_HP else STARDIST_2D_MODEL)


def _load_segneuron(checkpoint_path, device):
    import sys
    sys.path.insert(0, os.path.join(SEGNEURON_ROOT, "Train_and_Inference"))
    from collections import OrderedDict
    import torch
    from model.Mnet import MNet
    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    weights = state.get("model_weights", state)
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in weights.items())
    model.load_state_dict(clean)
    model.to(device)
    model.eval()
    return model


def _load_focus3d(model_type, checkpoint, device, batch_size, num_workers):
    from focus3d_baseline import Focus3DSegmenter
    return Focus3DSegmenter(
        model_type=model_type, checkpoint=checkpoint, device=device,
        batch_size=batch_size, data_loader_num_workers=num_workers,
    )


def _stub_cupy():
    """Satisfy the module-level cupy import of `cellvit.utils.tools`.

    Only the GPU variants of its helpers use cupy, and the post-processing this runs calls the numpy ones.
    """
    import sys
    import types
    try:
        import cupy  # noqa
    except ImportError:
        pass
    else:
        return

    stubs = {name: types.ModuleType(name) for name in ("cupy", "cupyx", "cupyx.scipy", "cupyx.scipy.ndimage")}
    stubs["cupy"].ndarray = np.ndarray
    stubs["cupyx"].scipy = stubs["cupyx.scipy"]
    stubs["cupyx.scipy"].ndimage = stubs["cupyx.scipy.ndimage"]
    sys.modules.update(stubs)


def _load_cellvit(model_type, checkpoint, magnification, device):
    """Load a CellViT model and the post-processor that turns its branches into instances."""
    import sys
    if CELLVIT_ROOT not in sys.path:
        sys.path.insert(0, CELLVIT_ROOT)
    _stub_cupy()

    from cellvit.models.cell_segmentation.cellvit import CellViT
    from cellvit.models.cell_segmentation.cellvit_256 import CellViT256
    from cellvit.models.cell_segmentation.cellvit_sam import CellViTSAM
    from cellvit.models.cell_segmentation.postprocessing import DetectionCellPostProcessor
    from cellvit.utils.tools import unflatten_dict

    if checkpoint is None:
        if model_type not in CELLVIT_CHECKPOINTS:
            raise ValueError(
                f"Unknown CellViT model type '{model_type}'; expected one of {sorted(CELLVIT_CHECKPOINTS)}."
            )
        checkpoint = os.path.join(CELLVIT_CHECKPOINT_ROOT, CELLVIT_CHECKPOINTS[model_type])

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    run_conf = unflatten_dict(state["config"], ".")
    architecture, model_conf = state["arch"], run_conf["model"]
    if architecture == "CellViTSAM":
        model = CellViTSAM(
            model_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            vit_structure=model_conf["backbone"],
            regression_loss=model_conf.get("regression_loss", False),
        )
    elif architecture == "CellViT256":
        model = CellViT256(
            model256_path=None,
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            regression_loss=model_conf.get("regression_loss", False),
        )
    elif architecture == "CellViT":
        model = CellViT(
            num_nuclei_classes=run_conf["data"]["num_nuclei_classes"],
            num_tissue_classes=run_conf["data"]["num_tissue_classes"],
            embed_dim=model_conf["embed_dim"],
            input_channels=model_conf.get("input_channels", 3),
            depth=model_conf["depth"],
            num_heads=model_conf["num_heads"],
            extract_layers=model_conf["extract_layers"],
            regression_loss=model_conf.get("regression_loss", False),
        )
    else:
        raise ValueError(f"Unsupported CellViT architecture '{architecture}' in '{checkpoint}'.")

    model.load_state_dict(state["model_state_dict"])
    model.eval()
    model.to(device)
    processor = DetectionCellPostProcessor(
        nr_types=run_conf["data"]["num_nuclei_classes"], magnification=magnification
    )
    return model, processor, run_conf


def _stub_cztile():
    """Satisfy the module-level cztile import of `empanada.inference.tile`.

    Only the tiled 2d inference uses cztile, and the images here are segmented whole.
    """
    import sys
    import types
    try:
        import cztile  # noqa
    except ImportError:
        pass
    else:
        return

    stubs = {
        name: types.ModuleType(name)
        for name in ("cztile", "cztile.fixed_total_area_strategy_2d", "cztile.tiling_strategy")
    }
    stubs["cztile.fixed_total_area_strategy_2d"].AlmostEqualBorderFixedTotalAreaStrategy2D = None
    stubs["cztile.tiling_strategy"].Region2D = None
    stubs["cztile.tiling_strategy"].TileInput = None
    sys.modules.update(stubs)


def _load_empanada(method, ndim, device, inference_scale):
    """Load the empanada-napari inference module and the engine of MitoNet or NucleoNet."""
    import sys
    if EMPANADA_ROOT not in sys.path:
        sys.path.insert(0, EMPANADA_ROOT)
    _stub_cztile()

    from empanada.config_loaders import read_yaml
    from empanada_napari import inference

    # The plugin sets the torch hub to the home directory on import. Keep the checkpoints with the other models.
    torch.hub.set_dir(EMPANADA_CHECKPOINT_ROOT)
    config_path = os.path.join(EMPANADA_ROOT, "empanada_napari", "configs", f"{EMPANADA_MODELS[method]}.yaml")
    config = read_yaml(config_path)
    params = {**EMPANADA_PARAMS, "inference_scale": inference_scale, "use_gpu": device != "cpu"}
    if ndim == 2:
        engine = inference.Engine2d(config, **params)
    else:
        engine = inference.Engine3d(config, **params, **EMPANADA_PARAMS_3D)
    return inference, engine, config


def _load_microsam_v1(method, model_type, checkpoint, device):
    from micro_sam.v1.automatic_segmentation import get_predictor_and_segmenter
    mode = {"microsam_ais": "ais", "microsam_apg": "apg"}[method]
    return get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, device=device, segmentation_mode=mode,
    )


def _segment_cellpose(image_or_volume, model, ndim, dataset_name):
    markers = MARKER_CHANNELS.get(dataset_name)
    if type(model).__name__ == "Cellpose":
        # CellPose 3 names the channel to segment and the nuclear one, 1-based with 0 for grayscale.
        channels = [markers["cell"] + 1, markers["nucleus"] + 1] if markers else [0, 0]
        channel_kwargs = {"channels": channels}
    else:
        # CellPose 4 reads the channels in any order and ignores 'channels'.
        channel_kwargs = {"channel_axis": -1} if markers else {}

    if ndim == 3:
        # CellPose 3 cannot estimate the diameter of a volume, so pass the mean diameter, its own fallback.
        diameter = model.diam_mean if type(model).__name__ == "Cellpose" else None
        masks = model.eval(
            image_or_volume, diameter=diameter, stitch_threshold=CELLPOSE_STITCH_THRESHOLD, z_axis=0, **channel_kwargs,
        )[0]
    else:
        masks = model.eval(image_or_volume, diameter=None, **channel_kwargs)[0]
    return masks.astype("uint32")


def _segment_stardist(image_or_volume, model, ndim):
    from csbdeep.utils import normalize as csbdeep_normalize
    if ndim == 2 and image_or_volume.ndim == 3 and model.config.n_channel_in == 1:
        image_or_volume = image_or_volume.mean(axis=-1)
    inp = csbdeep_normalize(image_or_volume, 1.0, 99.8)
    seg, _ = model.predict_instances(inp) if ndim == 3 else model.predict_instances(inp, scale=1)
    return seg.astype("uint32")


def _segment_cellsam(image, dataset_name):
    from cellSAM import cellsam_pipeline
    markers = MARKER_CHANNELS.get(dataset_name)
    if dataset_name in CHANNEL_MEAN_DATASETS and image.ndim == 3:
        image = image.mean(axis=-1)
    if markers:
        # CellSAM reads three channels as (blank, nuclear, whole-cell).
        blank = np.zeros_like(image[..., 0])
        image = np.stack([blank, image[..., markers["nucleus"]], image[..., markers["cell"]]], axis=-1)
    elif GRAYSCALE_MARKER.get(dataset_name) == "nucleus":
        if image.ndim == 3:  # The grayscale image is stored as identical channels.
            image = image.mean(axis=-1)
        blank = np.zeros_like(image)
        image = np.stack([blank, image, blank], axis=-1)
    elif image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    seg = cellsam_pipeline(image, use_wsi=False)
    if seg.ndim == 3:
        seg = seg[0]

    return seg.astype("uint32")


def _segment_cellvit(image, model, processor, run_conf, device):
    """Segment one histopathology patch with CellViT.

    CellViT reads the whole patch at once rather than in tiles of its training size: it was trained across
    magnifications and its own whole-slide pipeline infers on patches four times that size, while tiling costs
    every nucleus that a tile border cuts.
    """
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    image = image[..., :3]

    # The encoder tokenizes the input and then adds a square slice of its positional embedding, so a non-square
    # patch is padded up to a square, and that square to a multiple of the token size.
    extent = max(image.shape[:2])
    extent += -extent % model.patch_size
    padding = [(0, extent - size) for size in image.shape[:2]]
    padded = np.pad(image, padding + [(0, 0)], mode="reflect")

    normalization = run_conf.get("transformations", {}).get("normalize", {})
    mean = torch.tensor(normalization.get("mean", (0.5, 0.5, 0.5))).view(3, 1, 1)
    std = torch.tensor(normalization.get("std", (0.5, 0.5, 0.5))).view(3, 1, 1)
    inputs = torch.from_numpy(np.ascontiguousarray(padded)).float().permute(2, 0, 1) / 255.0
    inputs = ((inputs - mean) / std)[None].to(device)

    with torch.no_grad():
        predictions = model(inputs)

    # The post-processor expects channel-last probabilities, see CellViTInference.apply_softmax_reorder.
    predictions["nuclei_binary_map"] = torch.softmax(predictions["nuclei_binary_map"], dim=1).permute(0, 2, 3, 1)
    predictions["nuclei_type_map"] = torch.softmax(predictions["nuclei_type_map"], dim=1).permute(0, 2, 3, 1)
    predictions["hv_map"] = predictions["hv_map"].permute(0, 2, 3, 1)

    instances, _ = processor.post_process_batch(predictions)
    seg = instances[0].numpy()[:image.shape[0], :image.shape[1]]
    return seg.astype("uint32")


def _segneuron_multicut(affs, beta):
    """Reproduce SegNeuron's 'FRMC_post.post_mc' against the installed elf.

    The installed elf takes the segmentation as the second argument of every rag-consuming function,
    which SegNeuron's own calls predate, so it raises before producing anything.

    Args:
        affs: The affinity map, of shape (3, Z, Y, X).
        beta: The multicut bias towards over- or under-segmentation.

    Returns:
        The instance segmentation.
    """
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    seeds = np.zeros_like(boundary_input, dtype="uint64")
    offset = 0
    for z in range(seeds.shape[0]):
        wsz, max_id = watershed.distance_transform_watershed(boundary_input[z], threshold=0.25, sigma_seeds=2.0)
        seeds[z] = wsz + offset
        offset += max_id

    rag = features.compute_rag(seeds)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = features.compute_affinity_features(rag, seeds, affs, offsets)[:, 0]
    edge_sizes = features.compute_boundary_mean_and_length(rag, seeds, boundary_input)[:, 1]
    costs = multicut.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)
    return features.project_node_labels_to_pixels(rag, seeds, multicut.multicut_kernighan_lin(rag, costs))


def _segment_segneuron(volume, model, device, beta=0.25):
    import torch

    raw = volume.astype("float32") / 255.0 if volume.max() > 1.0 else volume.astype("float32")  # SegNeuron expects /255
    Z, Y, X = raw.shape
    bz, by, bx = 20, 128, 128
    hz, hy, hx = 4, 32, 32
    sz, sy, sx = bz - 2 * hz, by - 2 * hy, bx - 2 * hx

    pz = (-Z % sz) if Z > sz else (bz - Z)
    py = (-Y % sy) if Y > sy else (by - Y)
    px = (-X % sx) if X > sx else (bx - X)
    raw_pad = np.pad(raw, ((hz, hz + pz), (hy, hy + py), (hx, hx + px)), mode="reflect")

    Zp, Yp, Xp = raw_pad.shape
    affs_acc = np.zeros((3, Zp, Yp, Xp), dtype="float32")
    bound_acc = np.zeros((1, Zp, Yp, Xp), dtype="float32")
    count = np.zeros((1, Zp, Yp, Xp), dtype="float32")

    z_starts = list(range(0, Zp - bz + 1, sz)) or [0]
    y_starts = list(range(0, Yp - by + 1, sy)) or [0]
    x_starts = list(range(0, Xp - bx + 1, sx)) or [0]

    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    z1, y1, x1 = z0 + bz, y0 + by, x0 + bx
                    crop = raw_pad[z0:z1, y0:y1, x0:x1]
                    inp = torch.from_numpy(crop[None, None]).to(device)
                    pred_affs, pred_bound = model(inp)
                    affs_acc[:, z0:z1, y0:y1, x0:x1] += pred_affs[0].cpu().numpy()
                    bound_acc[:, z0:z1, y0:y1, x0:x1] += pred_bound[0].cpu().numpy()
                    count[:, z0:z1, y0:y1, x0:x1] += 1.0

    affs = affs_acc[:, hz:hz + Z, hy:hy + Y, hx:hx + X] / count[:, hz:hz + Z, hy:hy + Y, hx:hx + X]
    bound = bound_acc[:, hz:hz + Z, hy:hy + Y, hx:hx + X] / count[:, hz:hz + Z, hy:hy + Y, hx:hx + X]

    combined = np.minimum(np.stack([bound[0]] * 3), affs)
    return _segneuron_multicut(combined, beta=beta).astype("uint32")


def _segment_focus3d(volume, segmenter, dataset_name, cell_radius, z_ratio):
    """Segment one volume with FOCUS-3D, which rescales it to its own scale first.

    `z_ratio` is the z-to-xy spacing ratio. `cell_radius` is the xy radius that the rescaling starts from.
    The two are coupled: at a z_ratio of 1, FOCUS-3D rescales the z axis together with xy.
    """
    if z_ratio is None:
        z_ratio = DATASET_ANISOTROPY.get(dataset_name, 1.0)
    return segmenter(volume, z_ratio=z_ratio, cell_radius=cell_radius)


def _segment_empanada(image_or_volume, inference, engine, config, ndim):
    """Segment one image or volume with MitoNet or NucleoNet."""
    from skimage.segmentation import relabel_sequential
    if ndim == 2:
        image = image_or_volume.mean(axis=-1) if image_or_volume.ndim == 3 else image_or_volume
        # The engine returns the panoptic ids, class * label_divisor + instance.
        seg = engine.infer(image.astype("float32"))
        return relabel_sequential(seg)[0].astype("uint32")

    # The 2d images arrive percentile-stretched to [0, 255], so the volumes get the same.
    volume = ensure_8bit_range(image_or_volume).astype("float32")
    trackers = {"xy": engine.infer_on_axis(volume, "xy")[1]}
    # 'stack_postprocessing' is a napari thread worker. The undecorated generator yields one volume per class.
    stack = inference.stack_postprocessing.__wrapped__(
        trackers, None, config, label_divisor=EMPANADA_PARAMS["label_divisor"],
        min_size=EMPANADA_PARAMS_3D["min_size"], min_extent=EMPANADA_PARAMS_3D["min_extent"],
    )
    seg = next(stack)[0]
    return seg.astype("uint32")


def _segment_microsam_v1(image_or_volume, predictor, segmenter, ndim):
    from micro_sam.v1.automatic_segmentation import automatic_instance_segmentation
    seg = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=image_or_volume, ndim=ndim, verbose=False,
    )
    return seg.astype("uint32") if seg is not None else np.zeros(image_or_volume.shape, dtype="uint32")


def _run_evaluation(
    segment_fn, dataset_name, data_root, save_path, desc, limit, sample_index, crop_shape=None, normalize=True,
    extra_columns=None,
):
    if limit is not None:  # Name the file after the truncation, so it cannot pass for the full evaluation.
        save_path = f"{save_path[:-4]}_n{limit}.csv"
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return
    evaluate_samples(
        segment_fn, dataset_name, data_root, save_path, desc, limit=limit, crop_shape=crop_shape,
        sample_index=sample_index, normalize=normalize, extra_columns=extra_columns,
    )


def run_cellpose_evaluation(dataset_name, data_root, experiment_folder, model_type, device, limit, sample_index):
    if model_type in CELLPOSE3_MODELS and dataset_name in DATASETS_HP:
        warnings.warn(
            f"{model_type} is not a histopathology model and does not run on '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    ndim = 3 if dataset_name in DATASETS_3D else 2
    model = _load_cellpose(model_type, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellpose_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_cellpose(x, model, ndim, dataset_name),
        dataset_name, data_root, save_path, desc=f"cellpose-{model_type}", limit=limit, sample_index=sample_index,
    )


def run_stardist_evaluation(dataset_name, data_root, experiment_folder, limit, sample_index):
    ndim = 3 if dataset_name in DATASETS_3D else 2
    model = _load_stardist(ndim, dataset_name)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_stardist.csv")
    _run_evaluation(
        lambda x: _segment_stardist(x, model, ndim),
        dataset_name, data_root, save_path, desc="stardist", limit=limit, sample_index=sample_index,
    )


def run_cellsam_evaluation(dataset_name, data_root, experiment_folder, limit, sample_index):
    if dataset_name in DATASETS_3D:
        warnings.warn(
            f"CellSAM is 2D-only and does not support 3D dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellsam.csv")
    _run_evaluation(
        lambda x: _segment_cellsam(x, dataset_name),
        dataset_name, data_root, save_path, desc="cellsam", limit=limit, sample_index=sample_index,
    )


def run_cellvit_evaluation(
    dataset_name, data_root, experiment_folder, model_type, checkpoint, device, limit, sample_index,
):
    if dataset_name not in HP_DATASETS:
        warnings.warn(
            f"CellViT is a histopathology nucleus model and does not support dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    # CellViT reads slides in their native stain, which the percentile stretch of the shared loader would rescale
    # channel by channel, so this is the one baseline scored on the stored 8-bit RGB.
    magnification = 20 if model_type.endswith("_x20") else 40
    model, processor, run_conf = _load_cellvit(model_type, checkpoint, magnification, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellvit_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_cellvit(x, model, processor, run_conf, device),
        dataset_name, data_root, save_path, desc=f"cellvit-{model_type}", limit=limit,
        sample_index=sample_index, normalize=False,
    )


def run_segneuron_evaluation(
    dataset_name, data_root, experiment_folder, device, limit, sample_index, checkpoint_path=None,
):
    if dataset_name not in EM_DATASETS:
        warnings.warn(
            f"SegNeuron is 3D EM-only and does not support dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    if checkpoint_path is None:
        checkpoint_path = SEGNEURON_CHECKPOINT

    model = _load_segneuron(checkpoint_path, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_segneuron.csv")
    _run_evaluation(
        lambda x: _segment_segneuron(x, model, device),
        dataset_name, data_root, save_path, desc="segneuron", limit=limit, sample_index=sample_index,
    )


def run_focus3d_evaluation(
    dataset_name, data_root, experiment_folder, model_type, checkpoint, device, cell_radius, limit, sample_index,
    z_ratio=None, crop_shape=None, batch_size=FOCUS3D_BATCH_SIZE, num_workers=FOCUS3D_NUM_WORKERS,
):
    if dataset_name not in DATASETS_3D_LM:
        warnings.warn(
            f"FOCUS-3D is a 3d light microscopy method and does not support dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    segmenter = _load_focus3d(model_type, checkpoint, device, batch_size, num_workers)
    if crop_shape is None and dataset_name not in EVAL_CROPS:
        # The harness default of 8 slices would leave most of the model's 32-deep window as padding.
        crop_shape = (segmenter.patch_size[0],) + tuple(CROP_SHAPE_3D[1:])
    depth = crop_shape[0] if crop_shape else EVAL_CROPS[dataset_name][0][0]
    print(f"FOCUS-3D has the patch size {segmenter.patch_size} and runs on crops of depth {depth}.")

    # The crop depth is in the name: a baseline scored on another crop is not comparable.
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_focus3d_{model_type}_z{depth}.csv")
    _run_evaluation(
        lambda x: _segment_focus3d(x, segmenter, dataset_name, cell_radius, z_ratio),
        dataset_name, data_root, save_path, desc=f"focus3d-{model_type}", limit=limit, sample_index=sample_index,
        crop_shape=crop_shape,
    )


def tune_empanada_scale(method, dataset_name, data_root, experiment_folder, device):
    """Return the downsampling of MitoNet or NucleoNet with the best mSA on the tuning crops of a dataset.

    The scores are stored in the 'tuning' folder of the experiment, so that a dataset is tuned only once.
    """
    from elf.evaluation import mean_segmentation_accuracy
    from parameter_search import TUNING_CROP_3D

    tuning_path = os.path.join(experiment_folder, "tuning", f"{dataset_name}_{method}.json")
    if os.path.exists(tuning_path):
        with open(tuning_path) as f:
            return json.load(f)["inference_scale"]

    ndim = 3 if dataset_name in DATASETS_3D else 2
    total, n_max = n_samples(dataset_name, data_root, split="val"), EMPANADA_MAX_TUNING_SAMPLES[ndim]
    sample_ids = set(range(0, total, max(1, total // n_max))[:n_max])
    samples = [
        sample for sample in load_data(
            dataset_name, data_root, ndim, split="val", crop_shape=TUNING_CROP_3D if ndim == 3 else None,
            sample_ids=sample_ids,
        ) if sample[1].max() > 0
    ]
    scores = {}
    for scale in EMPANADA_SCALES:
        inference, engine, config = _load_empanada(method, ndim, device, scale)
        msa = []
        for raw, labels, valid_roi in samples:
            seg = _segment_empanada(raw, inference, engine, config, ndim)
            if valid_roi is not None:
                seg[~valid_roi] = 0
            if ndim == 2:
                seg = drop_severed_objects(seg, GT_MIN_SIZE_2D.get(dataset_name, 0))
            msa.append(mean_segmentation_accuracy(seg, labels))
        scores[scale] = float(np.mean(msa))
        print(f"{method}, downsampling {scale}, {len(samples)} crops of '{dataset_name}': mSA {scores[scale]:.3f}")

    best = max(scores, key=scores.get)
    if scores[best] - scores[1] <= EMPANADA_MIN_GAIN:
        best = 1
    os.makedirs(os.path.dirname(tuning_path), exist_ok=True)
    with open(tuning_path, "w") as f:
        json.dump({"inference_scale": best, "mSA": scores, "n_crops": len(samples)}, f, indent=2)
    return best


def run_empanada_evaluation(dataset_name, data_root, experiment_folder, method, device, limit, sample_index):
    if dataset_name not in EMPANADA_DATASETS[method]:
        warnings.warn(f"{method} does not support dataset '{dataset_name}'. Skipping.", UserWarning, stacklevel=2)
        return

    ndim = 3 if dataset_name in DATASETS_3D else 2
    scale = tune_empanada_scale(method, dataset_name, data_root, experiment_folder, device)
    inference, engine, config = _load_empanada(method, ndim, device, scale)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_{method}.csv")
    _run_evaluation(
        lambda x: _segment_empanada(x, inference, engine, config, ndim),
        dataset_name, data_root, save_path, desc=method, limit=limit, sample_index=sample_index,
        extra_columns={"inference_scale": scale},
    )


def run_microsam_v1_evaluation(
    dataset_name, data_root, experiment_folder, method, model_type, checkpoint, device, limit, sample_index,
):
    # Of the EM data, the v1 models only cover mitochondria and nuclei, with the organelle models.
    if dataset_name in EM_DATASETS and not (dataset_name in EM_ORGANELLE_DATASETS and "em_organelles" in model_type):
        raise ValueError(
            f"micro-sam v1 runs on EM only with the 'vit_*_em_organelles' models, and only on the mitochondria and "
            f"nucleus datasets. Got '{model_type}' on '{dataset_name}'."
        )
    ndim = 3 if dataset_name in DATASETS_3D else 2
    predictor, segmenter = _load_microsam_v1(method, model_type, checkpoint, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_{method}_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_microsam_v1(x, predictor, segmenter, ndim),
        dataset_name, data_root, save_path, desc=method, limit=limit, sample_index=sample_index,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", required=True, choices=(sorted(LM_DATASETS) + sorted(EM_DATASETS)))
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument(
        "-m", "--model_type", type=str, default=None,
        help="Model type override, e.g. cyto3 for cellpose, sam_h_x40 for cellvit or vit_b_em_organelles for "
        "micro-sam v1."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None,
        help="Checkpoint path for the cellvit, micro-sam v1, segneuron and focus3d methods."
    )
    parser.add_argument(
        "--crop_3d", type=int, nargs=3, default=None,
        help="Override the 3d center crop. FOCUS-3D defaults to its own window depth."
    )
    parser.add_argument("--cell_radius", type=float, default=FOCUS3D_CELL_RADIUS, help="The xy radius to rescale from.")
    parser.add_argument(
        "--z_ratio", type=float, default=None,
        help="The z-to-xy spacing ratio. Defaults to the anisotropy."
    )
    parser.add_argument("--batch_size", type=int, default=FOCUS3D_BATCH_SIZE, help="Patches per forward pass.")
    parser.add_argument("--num_workers", type=int, default=FOCUS3D_NUM_WORKERS, help="Data loader workers.")
    parser.add_argument("--n_samples", type=int, default=None, help="Score only the first N samples, for a check.")
    parser.add_argument("--sample_index", type=int, default=None, help="Score only this sample, as one array task.")
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.method == "cellpose":
        for model_type in ((args.model_type,) if args.model_type else CELLPOSE_MODELS):
            run_cellpose_evaluation(
                args.dataset_name, args.input_path, args.experiment_folder, model_type=model_type, device=device,
                limit=args.n_samples, sample_index=args.sample_index,
            )

    elif args.method == "stardist":
        run_stardist_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, limit=args.n_samples,
            sample_index=args.sample_index,
        )

    elif args.method == "cellsam":
        run_cellsam_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, limit=args.n_samples,
            sample_index=args.sample_index,
        )

    elif args.method == "cellvit":
        for model_type in ((args.model_type,) if args.model_type else tuple(CELLVIT_CHECKPOINTS)):
            run_cellvit_evaluation(
                args.dataset_name, args.input_path, args.experiment_folder, model_type=model_type,
                checkpoint=args.checkpoint, device=device, limit=args.n_samples, sample_index=args.sample_index,
            )

    elif args.method in ("microsam_ais", "microsam_apg"):
        run_microsam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, method=args.method,
            model_type=args.model_type or (
                SAM_V1_EM_ORGANELLE_MODEL_TYPE if args.dataset_name in EM_ORGANELLE_DATASETS else SAM_V1_MODEL_TYPE
            ),
            checkpoint=args.checkpoint, device=device,
            limit=args.n_samples, sample_index=args.sample_index,
        )

    elif args.method == "focus3d":
        run_focus3d_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            model_type=args.model_type or FOCUS3D_DEFAULT_MODEL, checkpoint=args.checkpoint, device=device,
            cell_radius=args.cell_radius, z_ratio=args.z_ratio,
            crop_shape=tuple(args.crop_3d) if args.crop_3d else None,
            batch_size=args.batch_size, num_workers=args.num_workers, limit=args.n_samples,
            sample_index=args.sample_index,
        )

    elif args.method == "segneuron":
        run_segneuron_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint, limit=args.n_samples, sample_index=args.sample_index,
        )

    elif args.method in EMPANADA_MODELS:
        run_empanada_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, method=args.method, device=device,
            limit=args.n_samples, sample_index=args.sample_index,
        )

    else:
        raise ValueError(f"Unknown method: '{args.method}'.")


if __name__ == "__main__":
    main()
