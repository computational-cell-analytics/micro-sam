"""The post-hoc script of Figure 4: object and pixel classification examples for the qualitative plots.

The script reproduces the few-shot classification of the paper with our torch-em loaders and micro-sam. The source is
few-shot-sam by Carolin Teuber (branch 'object-classification-ablation', commit c73079e). The pixel annotations are
zigzag spline strokes, so that the strokes are reproducible and the figure can show them.

Usage: python run_posthoc_classification.py <command> --dataset <name> [options]

The commands:
- prepare: write the 512 x 512 tiles of a dataset with the classes, splits and instances of few-shot-sam. The
  command writes one h5 file per split in DATA_DIR.
- annotate: draw the strokes of each image. Pixels get zigzag splines inside the tissue regions. Objects get one
  stroke along their long axis.
- predict: train the random forest of each encoder on the strokes of the training images, and predict the test
  images. The label budget and the balanced sampling are those of few-shot-sam.
- plot: pick the candidate images with the rules of few-shot-sam, and save one png per dataset.
- download: download a whole slide of GDC_SLIDES (--slide), or the PUMA context images (--dataset puma).
- slide: classify every tissue tile of a whole slide with the forest of one encoder. Use --part or --rows for
  parallel runs, then --merge.
- context: segment the nuclei of the context image of a PUMA ROI with micro-sam2 (--step segment, on a GPU). Then
  classify them with the forest of one encoder (--step classify).

The docstring of 'plots_for_figure4.py' lists the commands for Figure 4.

The script keeps these settings of few-shot-sam:
- 512 tiles with 90% annotated pixels for BCSS and IGNITE, the class groups and fold 0;
- micro-sam box prompts for the BBBC041 objects;
- the encoders and their input preprocessing;
- the features of micro-sam: a 256 grid, AnyUp for pixels, and the area and mean embedding for objects;
- a class pool of 10 x budget and the water-fill sampling;
- the random forest: 150 trees, depth 20, and class weights for objects.
"""
import os
import json
import random
import argparse
from glob import glob
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import imageio.v3 as imageio
from scipy.interpolate import splev, splprep

from skimage.draw import line, polygon
from skimage.morphology import binary_dilation, binary_erosion, disk
from skimage.segmentation import find_boundaries, relabel_sequential
from skimage.measure import label as connected_components, regionprops

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import torch
import torch.nn.functional as F


DATA = "/mnt/vast-nhr/projects/cidas/cca/data"
WORK_DIR = "/mnt/lustre-grete/usr/u12090/few_shot_sam"
DATA_DIR = os.path.join(WORK_DIR, "data")
EMBEDDING_DIR = os.path.join(WORK_DIR, "embeddings")
OUTPUT_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_classification"
MODELS = "/mnt/vast-nhr/projects/cidas/cca/models"
SPLITS = ["train", "val", "test"]
PATCH_SIZE = 512
MIN_ANNOTATED_FRACTION = 0.9
IGNORE = 255

# The task and the class names (ids 1, 2, ...) of each dataset, as in 'get_class_ids' of few-shot-sam.
DATASETS = {
    "pannuke": ("object", ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]),
    # PUMA: the 8 classes of the few-shot-sam runs (Caro). Epithelium (9) and apoptosis (10) are left out: their
    # nuclei are not used for training, and the forest does not predict them.
    "puma": ("object", [
        "Stroma", "Tumor", "Plasma cell", "Histiocyte", "Lymphocyte", "Melanophage", "Neutrophil", "Endothelium",
    ]),
    "bbbc041": ("object", [
        "Red blood cell", "Leukocyte", "Gametocyte", "Ring", "Trophozoite", "Schizont", "Difficult",
    ]),
    "bcss": ("pixel", ["Tumor", "Stroma", "Inflammatory", "Necrosis", "Other"]),
    "puma_tissue": ("pixel", ["Stroma", "Tumor", "Epidermis", "Blood vessel", "Necrosis", "White background"]),
    "ignite": ("pixel", [
        "Background", "Tumor cells", "Stroma and inflammation", "Necrotic tissue", "Macrophages", "Rest",
    ]),
}

# The encoder of each method: (family, model type, checkpoint). micro-sam2 is the joint v5a model of panel a.
ENCODERS = {
    "SAM": ("sam", "vit_b", None),
    "SAM2": ("sam2", "hvit_b", os.path.expanduser("~/.cache/micro_sam/models/hvit_b")),
    "SAM3": ("sam3", "vit_pe", None),
    "DINOv3": ("dinov3", "vit_b", os.path.join(MODELS, "dinov3", "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")),
    "UNIv2": ("univ2", "vit_h", os.path.join(MODELS, "univ2", "pytorch_model.bin")),
    "micro-sam": ("sam", "vit_b_histopathology", None),
    "micro-sam2": ("sam2", "hvit_t_joint_v5a", os.path.join(
        MODELS, "micro_sam2", "exported", "joint", "v5a", "joint_sam2_hvit_t_best_cd26f2555d7a11636c80a92f6eaa3dc0.pt"
    )),
    # The tiny micro-sam2 model of the micro-sam registry, a later training than joint v5a.
    "micro-sam2-cells": ("sam2", "hvit_t_cells", os.path.expanduser("~/.cache/micro_sam/models/hvit_t_cells")),
}
LM_MODEL = "vit_b_lm"
NOT_APPLICABLE = {"bbbc041": {"UNIv2"}}
# The UNIv2 input size of few-shot-sam: 16 * round(patch size * pixel size / 8) for a 20x scan.
UNIV2_IMG_SIZE = {"pannuke": 128, "puma": 256, "puma_tissue": 256, "bcss": 256, "ignite": 512}

# The label budgets of the paper (objects or 256-grid pixels per training set) and the default of the plots.
BUDGETS = {"object": [25, 100, 1000], "pixel": [100, 1000, 10000, 100000]}
DEFAULT_BUDGET = {"object": 25, "pixel": 100}
POOL_FACTOR = 10
GRID_SIZE = 256
# How the pixel labels are sampled from the pool: 'pixels' (random pixels, as few-shot-sam) or 'strokes' (whole
# strokes, see 'sample_strokes'). 'pixels' gives more variety for the same budget.
PIXEL_SAMPLING = "pixels"

# The zigzag strokes of a region: their number and length grow with the square root of the region area.
ZIGZAG_AREA_SCALE = 120.0
ZIGZAG_MAX_STROKES = 3
ZIGZAG_LENGTH_SCALE = 1.5
ZIGZAG_LENGTH = (80, 320)
ZIGZAG_STEP = 27
ZIGZAG_AMPLITUDE = 0.8
# Random factors for the spacing and swing of each turn, and the random walk step of the heading (radians).
ZIGZAG_STEP_JITTER = (0.6, 1.4)
ZIGZAG_SWING_JITTER = (0.35, 1.3)
ZIGZAG_DRIFT = 0.22
# A stroke keeps at least this many pixels from the strokes drawn before it.
ZIGZAG_GAP = 8
ZIGZAG_TRIES = 10
ZIGZAG_INSIDE = 0.9
# The strokes of few-shot-sam: the region is eroded by EROSION and the line has LINE_WIDTH.
EROSION = 5
LINE_WIDTH = 5
SEED = 42


def open_h5(path, mode="a"):
    """Open an h5 file; a file broken by a stopped run is deleted and made again."""
    try:
        return h5py.File(path, mode)
    except BlockingIOError:
        raise
    except OSError:
        print(f"Removing the broken file {path}", flush=True)
        os.remove(path)
        return h5py.File(path, mode)


def data_path(name, split):
    return os.path.join(DATA_DIR, name, f"{split}.h5")


def write_tiles(name, split, tiles):
    """Write the (stem, raw, semantic, instances) tiles of a split into one h5 file."""
    os.makedirs(os.path.join(DATA_DIR, name), exist_ok=True)
    with h5py.File(data_path(name, split), "w") as f:
        for stem, raw, semantic, instances in tiles:
            group = f.create_group(stem)
            group.create_dataset("raw", data=raw, compression="gzip")
            group.create_dataset("semantic", data=semantic.astype("uint8"), compression="gzip")
            group.create_dataset("instances", data=instances.astype("uint32"), compression="gzip")
    print(f"{name} {split}: {len(tiles)} tiles")


def strokes_path(name, split):
    return os.path.join(DATA_DIR, name, f"{split}_strokes.h5")


def read_tile(name, split, stem, keys=("raw", "semantic", "instances")):
    """Read the arrays of a tile. The strokes ('annotation') come from the stroke file of the split.

    A split without a stroke file falls back to strokes stored in the tile file, as in older runs.
    """
    path = strokes_path(name, split)
    with h5py.File(data_path(name, split), "r") as f:
        arrays = [f[stem][key][:] if key in f[stem] and key != "annotation" else None for key in keys]
        if "annotation" in keys and not os.path.exists(path):
            arrays[keys.index("annotation")] = f[stem]["annotation"][:]
    if "annotation" in keys and os.path.exists(path):
        with h5py.File(path, "r") as f:
            arrays[keys.index("annotation")] = f[stem][:]
    return arrays


def stems_of(name, split):
    with h5py.File(data_path(name, split)) as f:
        return sorted(f.keys())


def crop_into_patches(image, patch_size=PATCH_SIZE):
    """Cut an image (H, W) or (C, H, W) into non-overlapping patches, zero-padding the last row and column."""
    height, width = image.shape[-2:]
    patches = []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            patch = image[..., y:y + patch_size, x:x + patch_size]
            pad = [(0, 0)] * (patch.ndim - 2) + [(0, patch_size - patch.shape[-2]), (0, patch_size - patch.shape[-1])]
            patches.append((np.pad(patch, pad), f"{y // patch_size}_{x // patch_size}"))
    return patches


def semantic_to_instances(semantic, min_size=0):
    """Return the connected components of each class as instances, without the components below min_size."""
    instances = np.zeros(semantic.shape, dtype="uint32")
    offset = 0
    for class_id in np.unique(semantic):
        if class_id == 0:
            continue
        components = connected_components(semantic == class_id)
        if min_size > 0:
            sizes = np.bincount(components.ravel())
            components[np.isin(components, np.where(sizes < min_size)[0])] = 0
            components = relabel_sequential(components)[0]
        instances[components > 0] = components[components > 0] + offset
        offset = int(instances.max())
    return relabel_sequential(instances)[0]


def region_tiles(stem, image, semantic, min_size=0):
    """Tile an image (H, W, 3) and its region classes, and keep the tiles with MIN_ANNOTATED_FRACTION labels."""
    tiles = []
    for (raw, tile), (labels, _) in zip(crop_into_patches(image.transpose(2, 0, 1)), crop_into_patches(semantic)):
        if (labels > 0).mean() < MIN_ANNOTATED_FRACTION:
            continue
        tiles.append((f"{stem}_{tile}", raw.transpose(1, 2, 0), labels, semantic_to_instances(labels, min_size)))
    return tiles


def prepare_pannuke():
    """PanNuke: fold 1 for training, fold 2 for validation, fold 3 for testing, the 256 x 256 images as they are."""
    for split, fold in zip(SPLITS, ["fold_1", "fold_2", "fold_3"]):
        with h5py.File(os.path.join(DATA, "pannuke", f"pannuke_{fold}.h5")) as f:
            images, semantic, instances = f["images"][:], f["labels/semantic"][:], f["labels/instances"][:]
        tiles = [
            (f"{fold}_{i:04d}", images[:, i].transpose(1, 2, 0).astype("uint8"), semantic[i], instances[i])
            for i in range(images.shape[1])
        ]
        write_tiles("pannuke", split, tiles)


PUMA_TISSUE = {
    "tissue_stroma": 1, "tissue_tumor": 2, "tissue_epidermis": 3, "tissue_blood_vessel": 4, "tissue_necrosis": 5,
    "tissue_white_background": 6,
}


def rasterize_puma_tissue(roi):
    """Rasterize the tissue polygons of a PUMA ROI as torch-em does, with one instance per polygon.

    torch-em needs geopandas for this, and the environment does not have geopandas.
    """
    path = os.path.join(
        DATA, "puma", "annotations", "tissue", "01_training_dataset_geojson_tissue", f"{roi}_tissue.geojson"
    )
    with open(path) as f:
        features = json.load(f)["features"]
    semantic = np.zeros((1024, 1024), dtype="uint8")
    instances = np.zeros((1024, 1024), dtype="uint32")
    for instance_id, feature in enumerate(features, start=1):
        classification = feature["properties"]["classification"]
        if isinstance(classification, str):
            classification = json.loads(classification)
        geometry = feature["geometry"]
        polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
        for rings in polygons:
            shape = np.zeros(semantic.shape, dtype=bool)
            for k, ring in enumerate(rings):
                ring = np.array(ring)
                rr, cc = polygon(ring[:, 1], ring[:, 0], shape=semantic.shape)
                shape[rr, cc] = k == 0
            semantic[shape] = PUMA_TISSUE[classification["name"]]
            instances[shape] = instance_id
    return semantic, instances


def prepare_puma(name):
    """PUMA: the torch-em split, each 1024 x 1024 ROI as four 512 x 512 tiles, nuclei or tissue labels."""
    for split in SPLITS:
        tiles = []
        for path in sorted(glob(os.path.join(DATA, "puma", split, "preprocessed", "*.h5"))):
            roi = os.path.splitext(os.path.basename(path))[0]
            with h5py.File(path) as f:
                raw = f["raw"][:]
                if name == "puma":
                    semantic, instances = f["labels/semantic/nuclei"][:], f["labels/instances/nuclei"][:]
            if name == "puma_tissue":
                semantic, instances = rasterize_puma_tissue(roi)
            for (raw_tile, tile), (sem_tile, _), (inst_tile, _) in zip(
                crop_into_patches(raw), crop_into_patches(semantic), crop_into_patches(instances)
            ):
                tiles.append((f"{roi}_{tile}", raw_tile.transpose(1, 2, 0), sem_tile, relabel_sequential(inst_tile)[0]))
        write_tiles(name, split, tiles)


# BCSS: the classes of few-shot-sam. 0 and 7 are not labeled, the classes not listed here are 'other' (5).
BCSS_GROUPS = {1: 1, 19: 1, 20: 1, 2: 2, 3: 3, 10: 3, 11: 3, 4: 4}
BCSS_UNLABELED = (0, 7)


def prepare_bcss():
    """BCSS: the torch-em test ROIs, the other ROIs split 80/20 into training and validation (seed 42)."""
    from torch_em.data.datasets.histopathology import bcss
    root = os.path.join(DATA, "bcss")
    pairs = {}
    for split in ["train", "test"]:
        images, labels = bcss.get_bcsss_paths(root, split=split)
        pairs.update({os.path.splitext(os.path.basename(i))[0]: (i, lb) for i, lb in zip(images, labels)})
    test = sorted(stem for stem in pairs if stem in bcss.TEST_LIST)
    train, val = train_test_split(sorted(set(pairs) - set(test)), test_size=0.2, random_state=SEED)
    for split, stems in zip(SPLITS, [train, val, test]):
        tiles = []
        for stem in stems:
            image_path, label_path = pairs[stem]
            labels = imageio.imread(label_path)
            semantic = np.full(labels.shape, 5, dtype="uint8")
            for source, target in BCSS_GROUPS.items():
                semantic[labels == source] = target
            semantic[np.isin(labels, BCSS_UNLABELED)] = 0
            tiles += region_tiles(stem, imageio.imread(image_path)[..., :3], semantic)
        write_tiles("bcss", split, tiles)


# IGNITE: the evaluation classes of the authors ('remap_dict' of the IGNITE data toolkit). 0 is unannotated.
IGNITE_GROUPS = {1: 1, 2: 2, 3: 6, 4: 3, 5: 3, 6: 6, 7: 6, 8: 4, 9: 6, 10: 6, 11: 6, 12: 6, 13: 5, 14: 6, 15: 6, 16: 6}


def decode_ignite(path):
    """Read an IGNITE H&E mask. Some masks store each class c as 256 - c (only 0 and 240-255), they are inverted."""
    labels = imageio.imread(path).astype("int64")
    labels = np.where(labels >= 240, 256 - labels, labels)
    semantic = np.zeros(labels.shape, dtype="uint8")
    for source, target in IGNITE_GROUPS.items():
        semantic[labels == source] = target
    return semantic


def prepare_ignite():
    """IGNITE: the official split, the training ROIs of validation fold 0 for validation, regions of >= 50 pixels."""
    from torch_em.data.datasets.histopathology import ignite
    root = os.path.join(DATA, "ignite")
    overview = pd.read_csv(os.path.join(root, "data_overview.csv"))
    he = overview[overview["task"] == "he_tissue_segmentation"]
    val_names = set(he[he["validation_fold"].astype(str) == "fold0"]["name"])
    paths = {split: list(zip(*ignite.get_ignite_paths(root, split=split))) for split in ["train", "test"]}
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]  # noqa
    split_paths = {
        "train": [p for p in paths["train"] if stem(p[0]) not in val_names],
        "val": [p for p in paths["train"] if stem(p[0]) in val_names],
        "test": paths["test"],
    }
    for split in SPLITS:
        tiles = []
        for image_path, label_path in split_paths[split]:
            tiles += region_tiles(stem(image_path), imageio.imread(image_path)[..., :3], decode_ignite(label_path), 50)
        write_tiles("ignite", split, tiles)


BBBC041_CROP = (1200, 1600)
BBBC041_FRACTIONS = {"train": 0.7, "val": 0.15, "test": 0.15}
MIN_MASK_FRACTION = 0.02


def stratified_split(classes_per_image, fractions, seed=SEED):
    """Split the images so that each split has its fraction of each class (iterative stratification, Sechidis).

    The function places the rarest class first. Each of its images goes to the split that needs the class most. A tie
    goes to the split that needs the most images, then to the first split.
    """
    rng = np.random.default_rng(seed)
    names = list(classes_per_image)
    rng.shuffle(names)
    splits = list(fractions)
    image_need = {s: fractions[s] * len(names) for s in splits}
    class_need = {}
    for name in names:
        for c in classes_per_image[name]:
            class_need.setdefault(c, {s: 0.0 for s in splits})
            for s in splits:
                class_need[c][s] += fractions[s]
    assignment, left = {}, set(names)
    while left:
        counts = {c: sum(c in classes_per_image[n] for n in left) for c in class_need}
        counts = {c: n for c, n in counts.items() if n > 0}
        if not counts:
            for name in sorted(left):
                split = max(splits, key=lambda s: (image_need[s], -splits.index(s)))
                assignment[name], image_need[split] = split, image_need[split] - 1
            break
        rarest = min(counts, key=lambda c: (counts[c], c))
        for name in [n for n in names if n in left and rarest in classes_per_image[n]]:
            split = max(splits, key=lambda s: (class_need[rarest][s], image_need[s], -splits.index(s)))
            assignment[name] = split
            image_need[split] -= 1
            for c in classes_per_image[name]:
                class_need[c][split] -= 1
            left.discard(name)
    return assignment


def prepare_bbbc041():
    """BBBC041: micro-sam (LM) masks from the box of each cell, a stratified 70/15/15 split of all images."""
    from micro_sam.v1.inference import batched_inference
    from micro_sam.v1.util import get_sam_model
    root = os.path.join(DATA, "bbbc041", "malaria")
    annotations = []
    for name in ["training.json", "test.json"]:
        with open(os.path.join(root, name)) as f:
            annotations += json.load(f)
    categories = {n.lower(): i + 1 for i, n in enumerate(DATASETS["bbbc041"][1])}
    by_name = {os.path.splitext(os.path.basename(a["image"]["pathname"]))[0]: a for a in annotations}
    classes = {n: {categories[o["category"]] for o in a["objects"]} for n, a in by_name.items()}
    assignment = stratified_split(classes, BBBC041_FRACTIONS)
    predictor = get_sam_model(model_type=LM_MODEL)
    tiles = {split: [] for split in SPLITS}
    for name in sorted(by_name):
        annotation = by_name[name]
        image = imageio.imread(os.path.join(root, annotation["image"]["pathname"].lstrip("/")))[..., :3]
        objects = annotation["objects"]
        corners = [(o["bounding_box"]["minimum"], o["bounding_box"]["maximum"]) for o in objects]
        boxes = np.array([[low["c"], low["r"], high["c"], high["r"]] for low, high in corners])
        masks = batched_inference(
            predictor, image, batch_size=32, boxes=boxes, multimasking=False, return_instance_segmentation=False,
            verbose_embeddings=False,
        )
        masks = sorted(masks, key=lambda m: m["seg_id"])
        instances = np.zeros(image.shape[:2], dtype="uint32")
        semantic = np.zeros(image.shape[:2], dtype="uint8")
        box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        for index in np.argsort(-box_areas):
            x0, y0, x1, y1 = boxes[index]
            in_box = np.zeros(image.shape[:2], dtype=bool)
            in_box[y0:y1, x0:x1] = True
            mask = masks[index]["segmentation"]
            mask = (mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)).astype(bool) & in_box
            if mask.sum() < max(MIN_MASK_FRACTION * box_areas[index], 1):
                mask = in_box
            instances[mask] = index + 1
            semantic[mask] = categories[objects[index]["category"]]
        height, width = image.shape[:2]
        y0, x0 = (height - BBBC041_CROP[0]) // 2, (width - BBBC041_CROP[1]) // 2
        crop = (slice(y0, y0 + BBBC041_CROP[0]), slice(x0, x0 + BBBC041_CROP[1]))
        tiles[assignment[name]].append((name, image[crop], semantic[crop], relabel_sequential(instances[crop])[0]))
    for split in SPLITS:
        write_tiles("bbbc041", split, tiles[split])


def zigzag_path(shape, center, length, rng):
    """Draw one zigzag spline around 'center': a smooth curve that swings left and right of a slowly curving path.

    The spacing and the swing of each turn vary, and the heading drifts, like a stroke drawn by hand.
    The stroke is drawn on the image padded by 'length' on each side, so that the part outside the image is kept.
    Returns the stroke as a boolean mask of the padded shape (LINE_WIDTH wide).
    """
    pad = int(np.ceil(length))
    shape = (shape[0] + 2 * pad, shape[1] + 2 * pad)
    center = center + pad
    turns = max(4, int(round(length / ZIGZAG_STEP)))
    steps = rng.uniform(*ZIGZAG_STEP_JITTER, size=turns)
    steps *= length / steps.sum()
    headings = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, ZIGZAG_DRIFT, size=turns))
    directions = np.stack([np.sin(headings), np.cos(headings)], axis=1)
    path = np.concatenate([np.zeros((1, 2)), np.cumsum(steps[:, None] * directions, axis=0)])
    normals = np.concatenate([directions[:1], directions])[:, ::-1] * np.array([1.0, -1.0])
    swing = np.where(np.arange(turns + 1) % 2 == 0, 1.0, -1.0) * ZIGZAG_AMPLITUDE * (length / turns) / 2
    swing *= rng.uniform(*ZIGZAG_SWING_JITTER, size=turns + 1)
    points = center + path - path.mean(axis=0) + swing[:, None] * normals
    tck, _ = splprep(points.T, s=0, k=3)
    rr, cc = splev(np.linspace(0, 1, int(4 * length)), tck)
    rr, cc = np.round(rr).astype("int"), np.round(cc).astype("int")
    keep = (rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1])
    stroke = np.zeros(shape, dtype=bool)
    for (r0, c0), (r1, c1) in zip(zip(rr[keep][:-1], cc[keep][:-1]), zip(rr[keep][1:], cc[keep][1:])):
        lr, lc = line(r0, c0, r1, c1)
        stroke[lr, lc] = True
    return binary_dilation(stroke, disk((LINE_WIDTH - 1) // 2))


# The sketched strokes of the training tile of Figure 4: (dataset, tile) -> [(class id, (x, y) points of the path)].
MANUAL_STROKES = {
    ("bcss", "TCGA-AQ-A54N-DX1_xmin32440_ymin26558_MPP-0.2500_0_3"): [
        (1, [(30, 72), (110, 66), (180, 74), (250, 68)]),
        (1, [(50, 296), (150, 302), (260, 300), (350, 286), (410, 262), (445, 246)]),
        (1, [(300, 402), (220, 406), (150, 420), (100, 448), (60, 488)]),
        (2, [(108, 146), (170, 152), (235, 160), (300, 170)]),
        (5, [(306, 96), (330, 108), (352, 102), (368, 120)]),
        (3, [(432, 18), (446, 80), (458, 140), (474, 200)]),
        (3, [(420, 500), (446, 448), (472, 400), (500, 352)]),
    ],
}
# The tiles a run trains on first, in this order, before the shuffled rest (the tiles with manual strokes).
FIRST_TRAINING_TILES = {name: [tile] for name, tile in MANUAL_STROKES}


def chaikin(points, iterations):
    """Round the corners of a polyline by Chaikin corner cutting."""
    for _ in range(iterations):
        q = 0.75 * points[:-1] + 0.25 * points[1:]
        r = 0.25 * points[:-1] + 0.75 * points[1:]
        points = np.concatenate([points[:1], np.stack([q, r], axis=1).reshape(-1, 2), points[-1:]])
    return points


# The ranges of the steps and side swings of a sketched stroke, in pixels.
SCRIBBLE_STEP = (15, 35)
SCRIBBLE_SWING = (4, 11)


def hand_stroke(shape, path, rng):
    """Draw a hand-drawn stroke along 'path' ((x, y) points): uneven bends to both sides, slightly shaky."""
    points = np.array([(y, x) for x, y in path], dtype="float")
    tck, _ = splprep(points.T, s=0, k=min(3, len(points) - 1))
    dense = np.stack(splev(np.linspace(0, 1, 600), tck), axis=1)
    arc = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(dense, axis=0), axis=1))])
    positions = [0.0]
    while positions[-1] < arc[-1]:
        positions.append(positions[-1] + rng.uniform(*SCRIBBLE_STEP))
    positions = np.clip(np.array(positions), 0, arc[-1])
    centers = np.stack([np.interp(positions, arc, dense[:, k]) for k in range(2)], axis=1)
    tangents = np.gradient(centers, axis=0)
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-6)
    normals = tangents[:, ::-1] * np.array([1.0, -1.0])
    sides = np.where(np.arange(len(positions)) % 2 == 0, 1.0, -1.0)
    sides[rng.random(len(positions)) < 0.15] *= -0.3
    offsets = sides * rng.uniform(*SCRIBBLE_SWING, size=len(positions)) + np.cumsum(rng.normal(0, 0.8, len(positions)))
    offsets[[0, -1]] *= 0.3
    stroke_points = chaikin(centers + offsets[:, None] * normals, 1)
    rr, cc = np.round(stroke_points).astype("int").T
    stroke = np.zeros(shape, dtype=bool)
    for (r0, c0), (r1, c1) in zip(zip(rr[:-1], cc[:-1]), zip(rr[1:], cc[1:])):
        lr, lc = line(r0, c0, r1, c1)
        keep = (lr >= 0) & (lr < shape[0]) & (lc >= 0) & (lc < shape[1])
        stroke[lr[keep], lc[keep]] = True
    return binary_dilation(stroke, disk((LINE_WIDTH - 1) // 2))


def manual_annotation(semantic, strokes, rng):
    """Draw the manual strokes of a tile, each cut to the pixels of its class."""
    annotation = np.full(semantic.shape, IGNORE, dtype="uint8")
    for class_id, path in strokes:
        annotation[hand_stroke(semantic.shape, path, rng) & (semantic == class_id)] = class_id
    return annotation


def zigzag_stroke(region, length, rng, taken=None):
    """Draw one zigzag stroke inside 'region', keeping the try that stays most inside it, cut to the region.

    A try that comes closer than ZIGZAG_GAP to the strokes in 'taken' is not used; None is returned when no try fits.
    """
    inside = np.argwhere(region)
    pad = int(np.ceil(length))
    near = None if taken is None or not taken.any() else binary_dilation(taken, disk(ZIGZAG_GAP))
    best, best_fraction = None, -1.0
    for _ in range(ZIGZAG_TRIES):
        center = inside[rng.integers(len(inside))].astype("float")
        padded = zigzag_path(region.shape, center, length, rng)
        stroke = padded[pad:pad + region.shape[0], pad:pad + region.shape[1]]
        if near is not None and (stroke & region & near).any():
            continue
        fraction = (stroke & region).sum() / max(padded.sum(), 1)
        if fraction > best_fraction:
            best, best_fraction = stroke, fraction
        if fraction >= ZIGZAG_INSIDE:
            break
    return None if best is None else best & region


def pixel_annotation(semantic, instances, rng):
    """Zigzag strokes in every region (instance) of the classes, as few-shot-sam annotates each region."""
    annotation = np.full(semantic.shape, IGNORE, dtype="uint8")
    for prop in regionprops(instances):
        region = instances == prop.label
        classes = semantic[region]
        classes = classes[classes > 0]
        if len(classes) == 0:
            continue
        eroded = binary_erosion(region, disk(EROSION))
        region = eroded if eroded.any() else region
        size = np.sqrt(region.sum())
        n_strokes = int(np.clip(round(size / ZIGZAG_AREA_SCALE), 1, ZIGZAG_MAX_STROKES))
        length = float(np.clip(size / ZIGZAG_LENGTH_SCALE, *ZIGZAG_LENGTH))
        for _ in range(n_strokes):
            stroke = zigzag_stroke(region, length, rng, taken=annotation != IGNORE)
            if stroke is not None:
                annotation[stroke] = int(np.bincount(classes).argmax())
    return annotation


def object_annotation(semantic, instances):
    """One stroke through the center of each object along its long axis, as the scribbles of few-shot-sam."""
    annotation = np.full(semantic.shape, IGNORE, dtype="uint8")
    for prop in regionprops(instances, intensity_image=semantic):
        classes = prop.image_intensity[prop.image].astype("int")
        classes = classes[classes > 0]
        if len(classes) == 0:
            continue
        region = instances == prop.label
        eroded = binary_erosion(region, disk(EROSION))
        region = eroded if eroded.any() else region
        cy, cx = prop.centroid
        dy, dx = -np.sin(prop.orientation), np.cos(prop.orientation)
        half = prop.axis_major_length
        start = (int(round(cy - half * dy)), int(round(cx - half * dx)))
        end = (int(round(cy + half * dy)), int(round(cx + half * dx)))
        rr, cc = line(*start, *end)
        keep = (rr >= 0) & (rr < region.shape[0]) & (cc >= 0) & (cc < region.shape[1])
        stroke = np.zeros(region.shape, dtype=bool)
        stroke[rr[keep], cc[keep]] = True
        annotation[binary_dilation(stroke, disk((LINE_WIDTH - 1) // 2)) & region] = int(np.bincount(classes).argmax())
    return annotation


def tile_strokes(args):
    """Draw the strokes of one tile, with a seed of its own, so that tiles can be drawn in parallel."""
    name, split, index, stem = args
    semantic, instances = read_tile(name, split, stem, ("semantic", "instances"))
    if DATASETS[name][0] == "object":
        return stem, object_annotation(semantic, instances)
    rng = np.random.default_rng([SEED, SPLITS.index(split), index])
    if split == "train" and (name, stem) in MANUAL_STROKES:
        return stem, manual_annotation(semantic, MANUAL_STROKES[name, stem], rng)
    return stem, pixel_annotation(semantic, instances, rng)


def annotate(name, splits=SPLITS):
    """Draw and save the strokes of every image, each tile seeded on its own, so that they are the same on every run.

    The strokes go to a temporary file that replaces the stroke file only when it is complete.
    """
    n_workers = len(os.sched_getaffinity(0))
    for split in splits:
        path = strokes_path(name, split)
        jobs = [(name, split, index, stem) for index, stem in enumerate(stems_of(name, split))]
        with Pool(n_workers) as pool, h5py.File(path + ".tmp", "w") as out:
            for stem, annotation in pool.imap_unordered(tile_strokes, jobs, chunksize=8):
                out.create_dataset(stem, data=annotation, compression="gzip")
        os.replace(path + ".tmp", path)
        print(f"{name} {split}: strokes drawn with {n_workers} workers", flush=True)


class CustomEncoder(torch.nn.Module):
    """The encoders of few-shot-sam (few_shot_sam/models/custom_encoder.py), with the same input preprocessing.

    SAM2 and SAM3 get the SAM statistics on the 0-255 range, DINOv3 and UNIv2 the ImageNet statistics on 0-1,
    as in few-shot-sam. The longest side is resized to the input size, the rest is zero-padded.
    """

    def __init__(self, backbone, model_type, checkpoint, img_size=1024):
        super().__init__()
        from torch_em.model.vit import get_vision_transformer
        self.backbone = backbone
        if backbone == "univ2":
            import timm
            kwargs = dict(
                model_name="vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24,
                init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
                mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True,
            )
            self.encoder = timm.create_model(**kwargs)
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(state.get("state_dict", state))
            self.num_prefix_tokens = self.encoder.num_prefix_tokens
            self.encoder.img_size = max(1, round(img_size / 16)) * 14
            return
        architecture = model_type[:6] if backbone == "sam2" else model_type
        self.encoder = get_vision_transformer(img_size=img_size, backbone=backbone, model=architecture)
        if backbone == "sam2":
            from micro_sam.v2.util import get_sam2_model
            state = get_sam2_model(model_type=architecture, checkpoint_path=checkpoint).image_encoder.state_dict()
        elif backbone == "sam3":
            if checkpoint is None:
                from huggingface_hub import hf_hub_download
                checkpoint = hf_hub_download("facebook/sam3", "sam3.pt")
            prefix = "detector.backbone.vision_backbone.trunk."
            state = torch.load(checkpoint, weights_only=False)
            state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
        else:
            state = torch.load(checkpoint, map_location="cpu")
        self.encoder.load_state_dict(state)

    def preprocess(self, x):
        if self.backbone in ("sam2", "sam3"):
            mean, std, scale = (123.675, 116.28, 103.53), (58.395, 57.12, 57.375), 255.0
        else:
            mean, std, scale = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 1.0
        if x.max() <= 1.5 and scale == 255.0:
            x = x * 255.0
        elif x.max() > 1.5 and scale == 1.0:
            x = x / 255.0
        size = self.encoder.img_size
        factor = size / max(x.shape[-2:])
        target = (int(x.shape[-2] * factor + 0.5), int(x.shape[-1] * factor + 0.5))
        x = F.interpolate(x, target, mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        std = torch.tensor(std, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
        x = (x - mean) / std
        return F.pad(x, (0, size - x.shape[-1], 0, size - x.shape[-2]))

    @torch.no_grad()
    def forward(self, image):
        """Return the embedding (C, h, w) of an image (H, W, 3)."""
        param = next(self.encoder.parameters())
        x = torch.from_numpy(image.transpose(2, 0, 1).astype("float32"))[None]
        x = self.preprocess(x.to(device=param.device, dtype=param.dtype))
        if self.backbone == "univ2":
            tokens = self.encoder.forward_features(x)[:, self.num_prefix_tokens:, :]
            side = int(round(tokens.shape[1] ** 0.5))
            return tokens.transpose(1, 2).reshape(1, -1, side, side)[0].float().cpu().numpy()
        return self.encoder(x)[0][0].float().cpu().numpy()


class Embedder:
    """Compute and cache the embeddings of one encoder for the images of a dataset, as float16."""

    def __init__(self, name, method):
        family, model_type, checkpoint = ENCODERS[method]
        if name == "bbbc041" and method == "micro-sam":
            model_type = LM_MODEL
        self.family, self.model_type = family, model_type
        img_size = UNIV2_IMG_SIZE[name] if family == "univ2" else 1024
        tag = f"{method}_{model_type}" + (f"_img{img_size}" if family == "univ2" else "")
        self.folder = os.path.join(EMBEDDING_DIR, name, tag)
        os.makedirs(self.folder, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if family == "sam":
            from micro_sam.v1.util import get_sam_model
            self.predictor = get_sam_model(model_type=model_type, device=device)
        else:
            self.encoder = CustomEncoder(family, model_type, checkpoint, img_size).to(device).eval()

    def __call__(self, split, stem, image):
        path = os.path.join(self.folder, f"{split}.h5")
        # A cached embedding is read without a write lock, so that several runs can share the cache.
        if os.path.exists(path):
            with h5py.File(path, "r", locking=False) as f:
                if stem in f:
                    return f[stem][:].astype("float32")
        with open_h5(path) as f:
            if stem in f:
                return f[stem][:].astype("float32")
            if self.family == "sam":
                from micro_sam.v1.util import precompute_image_embeddings
                embedding = precompute_image_embeddings(
                    self.predictor, image, save_path=None, ndim=2, verbose=False
                )["features"]
                embedding = np.asarray(embedding)[0]
            else:
                embedding = self.encoder(image)
            f.create_dataset(stem, data=embedding.astype("float16"))
            return embedding.astype("float32")


def object_features(embedding, instances, semantic):
    """The object features of micro-sam (area and mean embedding on the 256 grid) and the class of each object."""
    from micro_sam.object_classification import compute_object_features
    image_embeddings = {"features": embedding[None], "input_size": instances.shape, "original_size": instances.shape}
    seg_ids, features = compute_object_features(image_embeddings, instances, verbose=False)
    classes = {}
    for prop in regionprops(instances, intensity_image=semantic):
        values = prop.image_intensity[prop.image].astype("int")
        values = values[values > 0]
        if len(values):
            classes[prop.label] = int(np.bincount(values).argmax())
    labels = np.array([classes.get(int(i), 0) for i in seg_ids])
    return features, labels, np.asarray(seg_ids)


def pixel_features(embedding, image, upsampler):
    """The pixel features of micro-sam on the 256 grid, upsampled with AnyUp."""
    from micro_sam.pixel_classification import compute_pixel_features
    image_embeddings = {"features": embedding[None], "input_size": image.shape[:2]}
    return compute_pixel_features(image_embeddings, image.shape[:2], image=image, upsampler=upsampler, verbose=False)


def grid_labels(labels, grid_shape):
    from micro_sam.pixel_classification import accumulate_pixel_labels
    return accumulate_pixel_labels(labels, grid_shape)


def water_fill(budget, available):
    """Share a budget over the classes: classes with fewer samples than their share get all of them (few-shot-sam)."""
    quotas, remaining = {}, dict(available)
    while remaining:
        share = budget // len(remaining)
        if share == 0:
            break
        small = sorted(c for c, n in remaining.items() if n <= share)
        if not small:
            quotas.update({c: share for c in remaining})
            break
        for c in small:
            quotas[c] = remaining.pop(c)
            budget -= quotas[c]
    return quotas


def balanced_sample(features, labels, budget, seed):
    """Sample a budget of training samples with water-filled class quotas (few-shot-sam '_balanced_sample')."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    quotas = water_fill(budget, dict(zip(classes.tolist(), counts.tolist())))
    chosen = np.concatenate([
        rng.choice(np.where(labels == c)[0], size=min(q, int((labels == c).sum())), replace=False)
        for c, q in quotas.items()
    ])
    return features[chosen], labels[chosen]


def train_forest(features, labels, task, seed=0):
    """The random forest of few-shot-sam, with inverse frequency class weights for objects."""
    class_weight = None
    if task == "object":
        classes, counts = np.unique(labels, return_counts=True)
        weights = (1 / len(classes)) / counts
        class_weight = dict(zip(classes.tolist(), (weights / weights.min()).tolist()))
    rf = RandomForestClassifier(
        n_estimators=150, max_depth=20, class_weight=class_weight, random_state=seed, n_jobs=-1
    )
    return rf.fit(features, labels)


def tile_strokes_on_grid(labels):
    """Split the grid labels of a tile into its strokes: the connected pixels of one class. Returns (class, index)."""
    strokes = []
    flat = labels.reshape(-1)
    for c in np.unique(flat):
        if c == 0 or c == IGNORE:
            continue
        components = connected_components(labels == c)
        for k in range(1, components.max() + 1):
            strokes.append((int(c), np.flatnonzero(components.reshape(-1) == k)))
    return strokes


def sample_strokes(strokes, budget, seed):
    """Sample whole strokes: a water-filled pixel quota per class, filled with strokes in random order.

    'strokes' is a list of (class, features, forced, tile). The function takes all forced strokes, even beyond the
    quota. Then it takes the other strokes of a class in random order until the class has its quota. A tile gives a
    second stroke of a class only after every tile gives one.
    """
    rng = np.random.default_rng(seed)
    available = {}
    for c, features, _, _ in strokes:
        available[c] = available.get(c, 0) + len(features)
    quotas = water_fill(budget, available)
    chosen = []
    for c, quota in quotas.items():
        of_class = [k for k, stroke in enumerate(strokes) if stroke[0] == c]
        forced = [k for k in of_class if strokes[k][2]]
        rest = [of_class[k] for k in rng.permutation(len(of_class)) if not strokes[of_class[k]][2]]
        seen, rounds = {}, []
        for k in rest:
            tile = strokes[k][3]
            rounds.append(seen.get(tile, 0))
            seen[tile] = seen.get(tile, 0) + 1
        rest = [k for _, k in sorted(zip(rounds, rest), key=lambda pair: pair[0])]
        total = 0
        n_before = len(chosen)
        for k in forced:
            chosen.append(k)
            total += len(strokes[k][1])
        for k in rest:
            if total >= quota:
                break
            chosen.append(k)
            total += len(strokes[k][1])
        counts = f"{len(chosen) - n_before} of {len(of_class)} strokes ({len(forced)} forced)"
        print(f"class {c}: {counts}, {total} pixels for a quota of {quota}", flush=True)
    features = np.concatenate([strokes[k][1] for k in chosen])
    labels = np.concatenate([np.full(len(strokes[k][1]), strokes[k][0]) for k in chosen])
    return features, labels


def training_pool(name, embedder, budget, upsampler):
    """Collect the training samples of fold 0 from the shuffled training images, at most POOL_FACTOR x budget per class.

    Objects and pixels are taken one by one, as in few-shot-sam ('strokes' in PIXEL_SAMPLING takes whole strokes).
    The tiles in FIRST_TRAINING_TILES come first, so that the tiles of the figure are always in the pool.
    """
    task, class_names = DATASETS[name]
    class_ids = list(range(1, len(class_names) + 1))
    stems = stems_of(name, "train")
    random.seed(0)
    random.shuffle(stems)
    first = [stem for stem in FIRST_TRAINING_TILES.get(name, []) if stem in stems]
    stems = first + [stem for stem in stems if stem not in first]
    cap = POOL_FACTOR * budget
    pool = {c: [] for c in class_ids}
    stroke_pool, counts = [], {c: 0 for c in class_ids}
    for stem in stems:
        raw, semantic, instances, annotation = read_tile(
            name, "train", stem, ("raw", "semantic", "instances", "annotation")
        )
        embedding = embedder("train", stem, raw)
        if task == "object":
            features, labels, _ = object_features(embedding, instances, semantic)
            for c in class_ids:
                room = cap - len(pool[c])
                if room > 0:
                    pool[c].extend(features[labels == c][:room])
            if all(len(pool[c]) >= cap for c in class_ids):
                break
            continue
        features, grid_shape = pixel_features(embedding, raw, upsampler)
        if PIXEL_SAMPLING == "pixels":
            # Random pixels, as few-shot-sam: the first pixels of each class in row-major order, up to the cap.
            labels = grid_labels(annotation, grid_shape)
            for c in class_ids:
                room = cap - len(pool[c])
                if room > 0:
                    pool[c].extend(features[labels == c][:room])
            if all(len(pool[c]) >= cap for c in class_ids):
                break
            continue
        labels = grid_labels(annotation, grid_shape).reshape(grid_shape)
        for c, index in tile_strokes_on_grid(labels):
            if c in counts and counts[c] < cap:
                stroke_pool.append((c, features[index], stem in first, stem))
                counts[c] += len(index)
        if all(counts[c] >= cap for c in class_ids):
            break
    used = len({stem for stem in stems[:stems.index(stem) + 1]})
    print(f"{name}: training pool from the first {used} of {len(stems)} training tiles", flush=True)
    if task == "pixel" and PIXEL_SAMPLING == "strokes":
        return sample_strokes(stroke_pool, budget, seed=0)
    features = np.concatenate([np.array(pool[c]) for c in class_ids if pool[c]])
    labels = np.concatenate([np.full(len(pool[c]), c) for c in class_ids if pool[c]])
    return balanced_sample(features, labels, budget, seed=0)


def predict_image(rf, task, embedding, raw, instances, semantic, upsampler):
    """Predict the classes of one image at full resolution (per object, or per pixel from the 256 grid)."""
    if task == "object":
        features, _, seg_ids = object_features(embedding, instances, semantic)
        prediction = np.zeros(instances.shape, dtype="uint8")
        if len(seg_ids):
            for seg_id, value in zip(seg_ids, rf.predict(features)):
                prediction[instances == seg_id] = value
        return prediction
    from micro_sam.pixel_classification import project_prediction_to_image
    features, grid_shape = pixel_features(embedding, raw, upsampler)
    return project_prediction_to_image(rf.predict(features), grid_shape, raw.shape[:2]).astype("uint8")


def forest_path(name, budget, method):
    return os.path.join(WORK_DIR, "forests", f"{name}_{method}_budget{budget}.joblib")


def trained_forest(name, embedder, budget, upsampler, method):
    """Train the random forest of a method once and save it; later calls load the saved forest."""
    from joblib import dump, load
    task = DATASETS[name][0]
    path = forest_path(name, budget, method)
    if os.path.exists(path):
        return load(path)
    rf = train_forest(*training_pool(name, embedder, budget, upsampler), task)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dump(rf, path)
    return rf


def prediction_path(name, budget, method):
    return os.path.join(OUTPUT_DIR, "protocol", name, f"budget_{budget}", f"{method}.h5")


def predict(name, budget, methods, max_test):
    """Train each encoder's random forest and predict the test images (protocol) or each annotated test image."""
    from micro_sam.pixel_classification import get_anyup_upsampler
    task = DATASETS[name][0]
    budget = budget or DEFAULT_BUDGET[task]
    stems = stems_of(name, "test")
    if max_test and len(stems) > max_test:
        stems = sorted(np.random.default_rng(SEED).choice(stems, size=max_test, replace=False).tolist())
    upsampler = get_anyup_upsampler(device="cuda" if torch.cuda.is_available() else "cpu") if task == "pixel" else None
    for method in methods:
        if method in NOT_APPLICABLE.get(name, ()):
            continue
        path = prediction_path(name, budget, method)
        # A finished method is skipped, so that a stopped run resumes where it ended.
        if os.path.exists(path):
            with open_h5(path) as f:
                if f.attrs.get("complete", False):
                    print(f"{name} budget {budget}: {method} already done", flush=True)
                    continue
        embedder = Embedder(name, method)
        rf = trained_forest(name, embedder, budget, upsampler, method)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open_h5(path) as f:
            f.attrs.update({"model_type": embedder.model_type, "budget": budget, "complete": False})
            for stem in stems:
                # A tile predicted by a stopped run is kept, so that the run resumes at the next tile.
                if stem in f:
                    continue
                raw, semantic, instances = read_tile(name, "test", stem)
                embedding = embedder("test", stem, raw)
                prediction = predict_image(rf, task, embedding, raw, instances, semantic, upsampler)
                f.create_dataset(stem, data=prediction, compression="gzip")
            f.attrs["complete"] = True
        print(f"{name} budget {budget}: {method} ({embedder.model_type}) done", flush=True)
        del embedder
        torch.cuda.empty_cache()


# The class colors of the candidate plots. The first six colors are for the pixel datasets.
CLASS_COLORS = [
    "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#F0E442", "#7F3C8D", "#56B4E9", "#8C564B", "#7F7F7F", "#11A579",
]
FILL_ALPHA = 0.45
CANDIDATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "figures", "figure4", "qualitative", "candidates"
)
N_CANDIDATES = 4
# Tiles with more zero padding than this (at the image border) are not shown.
MAX_PADDING = 0.01
MIN_OBJECTS = 10


def overlay(image, labels, instances=None, alpha=FILL_ALPHA):
    """Blend the class colors over the image; object outlines are opaque, as in few-shot-sam."""
    from matplotlib import colors
    palette = np.array([[0, 0, 0]] + [colors.to_rgb(c) for c in CLASS_COLORS]) * 255
    out = image.astype("float").copy()
    mask = (labels > 0) & (labels != IGNORE)
    out[mask] = (1 - alpha) * out[mask] + alpha * palette[np.minimum(labels[mask], len(CLASS_COLORS))]
    if instances is not None:
        edges = find_boundaries(instances, mode="inner") & mask
        out[edges] = palette[np.minimum(labels[edges], len(CLASS_COLORS))]
    return out.astype("uint8")


def accuracy(prediction, semantic, instances, task):
    """The accuracy on the labeled pixels, or on the objects (each object counts once)."""
    if task == "pixel":
        valid = semantic > 0
        return float((prediction[valid] == semantic[valid]).mean()) if valid.any() else np.nan
    hits = []
    for prop in regionprops(instances, intensity_image=semantic):
        values = prop.image_intensity[prop.image].astype("int")
        values = values[values > 0]
        if len(values):
            hits.append(np.bincount(values).argmax() == prediction[tuple(prop.coords[0])])
    return float(np.mean(hits)) if hits else np.nan


def select_candidates(name, scores):
    """Pick the candidates: objects by the spread of the methods, pixels around the median accuracy."""
    task = DATASETS[name][0]
    methods = [c for c in scores.columns if c in ENCODERS]
    if task == "object":
        scores = scores[(scores["n_objects"] >= MIN_OBJECTS) & (scores["n_classes"] >= 2)].copy()
        scores["spread"] = scores[methods].std(axis=1)
        return scores.sort_values("spread", ascending=False).head(N_CANDIDATES)
    scores = scores[scores["n_classes"] >= 2].copy()
    scores["mean"] = scores[methods].mean(axis=1)
    scores["distance"] = (scores["mean"] - scores["mean"].median()).abs()
    return scores.sort_values("distance").head(N_CANDIDATES)


def micro_sam2_lead(scores, exclude=()):
    """The lead of micro-sam2 over the best other method of each image (positive where micro-sam2 is best)."""
    others = [c for c in scores.columns if c in ENCODERS and c != "micro-sam2" and c not in exclude]
    return scores["micro-sam2"] - scores[others].max(axis=1)


def select_micro_sam2(name, scores, exclude=()):
    """Pick the images where micro-sam2 leads the other methods (but those in 'exclude') by the most."""
    task = DATASETS[name][0]
    keep = (scores["n_classes"] >= 2) & (scores["padding"] < MAX_PADDING)
    if task == "object":
        keep &= scores["n_objects"] >= MIN_OBJECTS
    scores = scores[keep].copy()
    scores["lead"] = micro_sam2_lead(scores, exclude)
    return scores.sort_values("lead", ascending=False).head(N_CANDIDATES)


def tile_scores(name, budget, files):
    """The accuracy of each method on each test image, cached as a csv in WORK_DIR/scores."""
    task, class_names = DATASETS[name]
    path = os.path.join(WORK_DIR, "scores", f"scores_{name}_budget{budget}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    stems = sorted(set.intersection(*[set(f.keys()) for f in files.values()]))
    rows = []
    for stem in stems:
        raw, semantic, instances = read_tile(name, "test", stem)
        row = {"name": stem, "n_objects": int(len(np.unique(instances)) - 1)}
        # The zero-padded part of the tiles at the image border.
        row["padding"] = float((raw.max(axis=-1) == 0).mean())
        fractions = np.bincount(semantic[semantic > 0], minlength=len(class_names) + 1)[1:] / semantic.size
        row["n_classes"] = int((fractions >= 0.01).sum())
        row.update({m: accuracy(f[stem][:], semantic, instances, task) for m, f in files.items()})
        rows.append(row)
    scores = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scores.to_csv(path, index=False)
    return scores


# The selection rules of the candidate plots: (file suffix, rule).
SELECTIONS = {
    "default": ("", select_candidates),
    "micro-sam2": ("_microsam2_best", select_micro_sam2),
    "micro-sam2-general": ("_microsam2_best_without_univ2", lambda n, s: select_micro_sam2(n, s, ("UNIv2",))),
}


def plot(name, budget, selection="default"):
    """Save the candidate images of a dataset with the reference and the prediction of each method."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    task, class_names = DATASETS[name]
    budget = budget or DEFAULT_BUDGET[task]
    methods = [m for m in ENCODERS if os.path.exists(prediction_path(name, budget, m))]
    files = {m: h5py.File(prediction_path(name, budget, m), "r") for m in methods}
    scores = tile_scores(name, budget, files)
    suffix, rule = SELECTIONS[selection]
    chosen = rule(name, scores)
    columns = ["Image", "Reference"] + methods
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(1.9 * len(columns), 2.0 * len(chosen) + 0.6))
    axes = np.atleast_2d(axes)
    for row, (_, entry) in enumerate(chosen.iterrows()):
        raw, semantic, instances = read_tile(name, "test", entry["name"])
        object_instances = instances if task == "object" else None
        first = raw
        label = entry["name"][:24] + (f"\nlead {entry['lead']:+.2f}" if "lead" in entry else "")
        panels = [(first, label), (overlay(raw, semantic, object_instances), "Reference")]
        for m in methods:
            prediction = files[m][entry["name"]][:]
            panels.append((overlay(raw, prediction, object_instances), f"{m}\nacc {entry[m]:.2f}"))
        for ax, (panel, title) in zip(axes[row], panels):
            ax.imshow(panel)
            ax.set_title(title, fontsize=7)
            ax.axis("off")
    handles = [Patch(color=CLASS_COLORS[i], label=n) for i, n in enumerate(class_names)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), fontsize=7, frameon=False)
    fig.suptitle(f"{name}, {task} classification, budget {budget}", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = os.path.join(CANDIDATE_DIR, f"figure_4_qualitative_candidates_{name}_budget{budget}{suffix}.png")
    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved '{out}'.")
    if selection == "default":
        plot_strokes(name)


def plot_strokes(name, n_images=4):
    """Save the strokes on a few training images, the annotations that the protocol trains on."""
    import matplotlib.pyplot as plt
    stems = stems_of(name, "train")
    rng = np.random.default_rng(SEED)
    stems = rng.choice(stems, size=min(n_images, len(stems)), replace=False)
    fig, axes = plt.subplots(2, len(stems), figsize=(2.6 * len(stems), 5.4))
    for column, stem in enumerate(stems):
        raw, semantic, annotation = read_tile(name, "train", stem, ("raw", "semantic", "annotation"))
        axes[0, column].imshow(overlay(raw, annotation, alpha=1.0))
        axes[0, column].set_title(f"{stem[:24]}: strokes", fontsize=7)
        axes[1, column].imshow(overlay(raw, semantic))
        axes[1, column].set_title("Reference", fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    out = os.path.join(CANDIDATE_DIR, f"figure_4_strokes_{name}.png")
    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved '{out}'.")


SLIDE_DIR = os.path.join(WORK_DIR, "slides")
# The class map and the thumbnail of a slide are saved at this downscaling of the base resolution.
SLIDE_DOWNSCALE = 8
# The tissue fraction of a tile to classify it. The low value keeps the fat, which is mostly white.
SLIDE_MIN_TISSUE = 0.02
# The overlap of the slide windows. Only the center of each window is kept, because its border lacks context.
SLIDE_HALO = 64


def read_slide_level(path, level=0):
    """Open a level of a whole slide image (svs) as a lazy zarr array (H, W, 3)."""
    import tifffile
    import zarr
    tif = tifffile.TiffFile(path)
    data = zarr.open(tif.series[0].aszarr(level=level), mode="r")
    # Some tifffile versions open a level as a group that holds the array.
    return data if hasattr(data, "shape") else data[str(level) if str(level) in data else "0"]


def tissue_fraction(tile):
    """The fraction of a tile (H, W, 3) that is tissue: saturated enough and not black."""
    tile = tile.astype("float")
    saturation = (tile.max(axis=-1) - tile.min(axis=-1)) / np.maximum(tile.max(axis=-1), 1)
    return float(((saturation > 0.08) & (tile.max(axis=-1) > 40)).mean())


def predict_slide(name, budget, method, slide, part=(0, 1), row_range=None):
    """Classify every tissue tile of a slide with the forest of 'method' from 'predict'.

    The windows are PATCH_SIZE pixels at the base level, with SLIDE_HALO pixels of overlap. The BCSS images are 1.215
    times smaller than the base level, so the slide is classified at a slightly higher magnification. On the BCSS
    region of the test crop, both scales give the same map: macro F1 0.67 here and 0.64 at the BCSS scale.

    The function saves a class map and a thumbnail at 1 / SLIDE_DOWNSCALE in an h5 file next to the slide. A stopped
    run resumes at the next tile row. With part = (k, n), the function does the k-th of n blocks of tile rows. With
    row_range = (first, last), it does these rows. 'merge_slide' joins all files of a slide.
    """
    from skimage.transform import resize
    from micro_sam.pixel_classification import get_anyup_upsampler, project_prediction_to_image
    task = DATASETS[name][0]
    assert task == "pixel", "Only pixel classification is supported for slides."
    budget = budget or DEFAULT_BUDGET[task]
    slide_path = os.path.join(SLIDE_DIR, f"{slide}.svs")
    image = read_slide_level(slide_path)
    height, width = image.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    upsampler = get_anyup_upsampler(device=device)
    embedder = Embedder(name, method)
    rf = trained_forest(name, embedder, budget, upsampler, method)
    core = PATCH_SIZE - 2 * SLIDE_HALO
    small, small_halo = core // SLIDE_DOWNSCALE, SLIDE_HALO // SLIDE_DOWNSCALE
    rows, cols = int(np.ceil(height / core)), int(np.ceil(width / core))
    k, n = part
    first, last = (k * rows // n, (k + 1) * rows // n) if row_range is None else row_range
    tag = f"part{k}of{n}" if row_range is None else f"rows{first}-{last}"
    out_path = os.path.join(SLIDE_DIR, f"{slide}_{name}_{method}_budget{budget}_{tag}.h5")
    with open_h5(out_path) as f:
        if "classes" not in f:
            f.create_dataset("classes", shape=(rows * small, cols * small), dtype="uint8", chunks=(small, 8 * small))
            f.create_dataset(
                "thumbnail", shape=(rows * small, cols * small, 3), dtype="uint8", chunks=(small, 8 * small, 3)
            )
            f.attrs.update({
                "first": first, "last": last, "rows_done": first, "slide": slide, "method": method, "budget": budget,
                "downscale": SLIDE_DOWNSCALE, "shape": (height, width), "halo": SLIDE_HALO,
            })
        for row in range(int(f.attrs["rows_done"]), last):
            # The strip of the window row, with the halo above and below, white outside the slide.
            y0 = row * core - SLIDE_HALO
            top, bottom = max(y0, 0), min(y0 + PATCH_SIZE, height)
            strip = np.full((PATCH_SIZE, cols * core + 2 * SLIDE_HALO, 3), 255, dtype="uint8")
            strip[top - y0:bottom - y0, SLIDE_HALO:SLIDE_HALO + width] = np.asarray(image[top:bottom])[..., :3]
            classes = np.zeros((small, cols * small), dtype="uint8")
            thumbnail = np.zeros((small, cols * small, 3), dtype="uint8")
            for col in range(cols):
                window = strip[:, col * core:col * core + PATCH_SIZE]
                center = window[SLIDE_HALO:SLIDE_HALO + core, SLIDE_HALO:SLIDE_HALO + core]
                region = (slice(None), slice(col * small, (col + 1) * small))
                thumbnail[region] = resize(center, (small, small), preserve_range=True, anti_aliasing=True)
                if tissue_fraction(center) < SLIDE_MIN_TISSUE:
                    continue
                if embedder.family == "sam":
                    from micro_sam.v1.util import precompute_image_embeddings
                    embedding = np.asarray(precompute_image_embeddings(
                        embedder.predictor, window, save_path=None, ndim=2, verbose=False
                    )["features"])[0]
                else:
                    embedding = embedder.encoder(window)
                features, grid_shape = pixel_features(embedding.astype("float32"), window, upsampler)
                full = PATCH_SIZE // SLIDE_DOWNSCALE
                prediction = project_prediction_to_image(rf.predict(features), grid_shape, (full, full))
                classes[region] = prediction[small_halo:small_halo + small, small_halo:small_halo + small]
            f["classes"][row * small:(row + 1) * small] = classes
            f["thumbnail"][row * small:(row + 1) * small] = thumbnail
            f.attrs["rows_done"] = row + 1
            # Flush each row, so that a stopped run keeps a readable file with its finished rows.
            f.flush()
            print(f"{slide}: row {row + 1} (block {first}-{last} of {rows})", flush=True)
        f.attrs["complete"] = True
    print(f"Saved part {k} of {n} of the class map of {slide} to {out_path}.", flush=True)


def merge_slide(name, budget, method, slide):
    """Join all files of a slide class map (see 'predict_slide') into one h5 file.

    Each file gives the rows it has done; every row of the slide must be done by one of the files.
    """
    import re
    import h5py
    budget = budget or DEFAULT_BUDGET[DATASETS[name][0]]
    stem = os.path.join(SLIDE_DIR, f"{slide}_{name}_{method}_budget{budget}")
    paths = sorted(glob(f"{stem}_part*of*.h5") + glob(f"{stem}_rows*-*.h5"))
    parts = [h5py.File(path, "r") for path in paths]
    small = (PATCH_SIZE - 2 * int(parts[0].attrs.get("halo", 0))) // SLIDE_DOWNSCALE
    rows = parts[0]["classes"].shape[0] // small
    done = np.zeros(rows, dtype=bool)
    with h5py.File(f"{stem}.h5", "w") as out:
        data = {key: np.zeros(parts[0][key].shape, dtype="uint8") for key in ["classes", "thumbnail"]}
        for path, f in zip(paths, parts):
            if "first" in f.attrs:
                first = int(f.attrs["first"])
            else:
                k, n = (int(v) for v in re.search(r"_part([0-9]+)of([0-9]+)", path).groups())
                first = k * rows // n
            last = int(f.attrs["rows_done"])
            block = slice(first * small, last * small)
            for key in data:
                data[key][block] = f[key][block]
            done[first:last] = True
        if not done.all():
            raise RuntimeError(f"The rows {np.flatnonzero(~done).tolist()} of {slide} are not done.")
        for key, value in data.items():
            out.create_dataset(key, data=value, compression="gzip")
        skip = ("rows_done", "complete", "first", "last")
        out.attrs.update({key: value for key, value in parts[0].attrs.items() if key not in skip})
    print(f"Merged {len(paths)} files into {stem}.h5.", flush=True)


# The whole slides of the 'slide' command, as their file id in the GDC data portal (https://portal.gdc.cancer.gov).
# TCGA-E2-A14X-01Z-00-DX1 is the slide of the BCSS test crop of Figure 4.
GDC_SLIDES = {"TCGA-E2-A14X-01Z-00-DX1": "abf27f46-f39a-4f2d-8e7b-423e5da54419"}
GDC_URL = "https://api.gdc.cancer.gov/data/{}"

# The PUMA context images: 5120 x 5120 pixels around each annotated 1024 x 1024 ROI, which is at their center
# (CONTEXT_ROI_OFFSET). PUMA releases no whole slides; these are its largest images.
CONTEXT_URL = "https://zenodo.org/records/15050523/files/01_training_dataset_tif_context_ROIs.zip"
CONTEXT_DIR = os.path.join(SLIDE_DIR, "puma_context", "01_training_dataset_tif_context_ROIs")
CONTEXT_ROI_OFFSET = 2048
CONTEXT_SEGMENTATION_DIR = os.path.join(SLIDE_DIR, "puma_seg")
# The model (APG) and the untiled color windows of the context segmentation, as in Figure 2. Tiling reads grayscale.
CONTEXT_SEGMENTATION_MODEL = "hvit_l_cells"
CONTEXT_WINDOW, CONTEXT_HALO = 1024, 64


def download(name, slide=None):
    """Download a whole slide from the GDC data portal, or the PUMA context images from Zenodo."""
    import urllib.request
    import zipfile
    if name == "puma":
        folder = os.path.dirname(CONTEXT_DIR)
        if os.path.isdir(CONTEXT_DIR):
            print(f"The PUMA context images are in {CONTEXT_DIR}.")
            return
        os.makedirs(folder, exist_ok=True)
        zip_path = os.path.join(folder, "context_ROIs.zip")
        if not os.path.exists(zip_path):
            urllib.request.urlretrieve(CONTEXT_URL, zip_path + ".tmp")
            os.rename(zip_path + ".tmp", zip_path)
        with zipfile.ZipFile(zip_path) as f:
            f.extractall(folder)
        print(f"Saved the PUMA context images to {CONTEXT_DIR}.")
        return
    path = os.path.join(SLIDE_DIR, f"{slide}.svs")
    if os.path.exists(path):
        print(f"The slide is in {path}.")
        return
    os.makedirs(SLIDE_DIR, exist_ok=True)
    urllib.request.urlretrieve(GDC_URL.format(GDC_SLIDES[slide]), path + ".tmp")
    os.rename(path + ".tmp", path)
    print(f"Saved the slide to {path}.")


def context_segmentation_path(roi):
    return os.path.join(CONTEXT_SEGMENTATION_DIR, f"{roi.replace('training_set_', '')}_{CONTEXT_SEGMENTATION_MODEL}.h5")


def roi_instances(roi):
    """The reference nuclei of a PUMA ROI, from its four test tiles."""
    instances = np.zeros((2 * PATCH_SIZE, 2 * PATCH_SIZE), dtype="uint32")
    for i in range(2):
        for j in range(2):
            tile = read_tile("puma", "test", f"{roi}_{i}_{j}", ("instances",))[0].astype("uint32")
            region = np.s_[i * PATCH_SIZE:(i + 1) * PATCH_SIZE, j * PATCH_SIZE:(j + 1) * PATCH_SIZE]
            instances[region] = np.where(tile > 0, tile + instances.max(), 0)
    return instances


def segment_context(roi):
    """Segment all nuclei of the context image of a PUMA ROI with micro-sam2 APG in untiled overlapping windows.

    Each nucleus is kept from the window whose core holds its centroid. The score on the annotated ROI at the center
    is printed as a check against the PUMA score of Figure 2.
    """
    import tifffile
    from elf.evaluation import mean_segmentation_accuracy
    from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation
    context = tifffile.imread(os.path.join(CONTEXT_DIR, f"{roi}_context.tif"))[..., :3]
    predictor, segmenter = get_predictor_and_segmenter(model_type=CONTEXT_SEGMENTATION_MODEL, segmentation_mode="apg")
    size, halo = CONTEXT_WINDOW, CONTEXT_HALO
    height, width = context.shape[:2]
    segmentation = np.zeros((height, width), dtype="uint32")
    next_id = 1
    starts = list(range(0, height - size, size - 2 * halo)) + [height - size]
    for y in starts:
        for x in starts:
            window = automatic_instance_segmentation(
                predictor, segmenter, context[y:y + size, x:x + size], ndim=2, verbose=False
            )
            low_y, high_y = (0 if y == 0 else halo), (size if y == height - size else size - halo)
            low_x, high_x = (0 if x == 0 else halo), (size if x == width - size else size - halo)
            for prop in regionprops(window):
                cy, cx = prop.centroid
                if not (low_y <= cy < high_y and low_x <= cx < high_x):
                    continue
                rr, cc = prop.coords[:, 0] + y, prop.coords[:, 1] + x
                free = segmentation[rr, cc] == 0
                segmentation[rr[free], cc[free]] = next_id
                next_id += 1
        print(f"{roi}: window row {y} of {height}, {next_id - 1} nuclei", flush=True)
    segmentation = relabel_sequential(segmentation)[0]
    os.makedirs(CONTEXT_SEGMENTATION_DIR, exist_ok=True)
    with open_h5(context_segmentation_path(roi), "w") as f:
        f.create_dataset("segmentation", data=segmentation, compression="gzip")
    roi_region = (slice(CONTEXT_ROI_OFFSET, CONTEXT_ROI_OFFSET + 2 * PATCH_SIZE),) * 2
    msa = mean_segmentation_accuracy(relabel_sequential(segmentation[roi_region])[0], roi_instances(roi))
    print(f"{roi}: {segmentation.max()} nuclei in the context, mSA {msa:.3f} on the annotated ROI", flush=True)


def classify_context(roi, budget, method="micro-sam2"):
    """Classify every segmented nucleus of the context image of a PUMA ROI with the random forest of a method.

    The function cuts the context into windows of PATCH_SIZE with SLIDE_HALO pixels of overlap. It classifies each
    nucleus in the window whose core holds its centroid. It rounds the embedding to float16, as in the cached training
    embeddings. It adds the class map to the segmentation file. As a check, it classifies the test tile 0_0 from the
    context with its reference nuclei, and compares the result with the prediction of 'predict'.
    """
    import tifffile
    from micro_sam.object_classification import compute_object_features
    class_names = DATASETS["puma"][1]
    context = tifffile.imread(os.path.join(CONTEXT_DIR, f"{roi}_context.tif"))[..., :3]
    path = context_segmentation_path(roi)
    with h5py.File(path, "r") as f:
        segmentation = f["segmentation"][:]
    embedder = Embedder("puma", method)
    rf = trained_forest("puma", embedder, budget, None, method)
    size, halo = PATCH_SIZE, SLIDE_HALO
    core = size - 2 * halo
    centroids = {prop.label: prop.centroid for prop in regionprops(segmentation)}
    classes = {}
    height, width = segmentation.shape
    for y in range(-halo, height - halo, core):
        for x in range(-halo, width - halo, core):
            window = np.full((size, size, 3), 255, dtype="uint8")
            labels = np.zeros((size, size), dtype="uint32")
            ys, xs = slice(max(y, 0), min(y + size, height)), slice(max(x, 0), min(x + size, width))
            target = np.s_[ys.start - y:ys.stop - y, xs.start - x:xs.stop - x]
            window[target], labels[target] = context[ys, xs], segmentation[ys, xs]
            low_y, low_x = y + halo, x + halo
            ids = [
                i for i in np.unique(labels)
                if i and low_y <= centroids[i][0] < low_y + core and low_x <= centroids[i][1] < low_x + core
            ]
            if not ids:
                continue
            labels = np.where(np.isin(labels, ids), labels, 0)
            embedding = np.asarray(embedder.encoder(window)).astype("float16").astype("float32")
            image_embeddings = {"features": embedding[None], "input_size": labels.shape, "original_size": labels.shape}
            seg_ids, features = compute_object_features(image_embeddings, labels, verbose=False)
            for i, c in zip(seg_ids, rf.predict(features)):
                classes[int(i)] = int(c)
    lookup = np.zeros(int(segmentation.max()) + 1, dtype="uint8")
    for i, c in classes.items():
        lookup[i] = c
    with open_h5(path) as f:
        if f"classes_budget{budget}" in f:
            del f[f"classes_budget{budget}"]
        f.create_dataset(f"classes_budget{budget}", data=lookup[segmentation], compression="gzip")
    print({class_names[c - 1]: int(np.sum(lookup[1:] == c)) for c in range(1, len(class_names) + 1)}, flush=True)

    raw, semantic, instances = read_tile("puma", "test", f"{roi}_0_0")
    offset = CONTEXT_ROI_OFFSET
    assert np.array_equal(context[offset:offset + size, offset:offset + size], raw), "The ROI is not at the offset."
    with h5py.File(prediction_path("puma", budget, method), "r") as f:
        stored = f[f"{roi}_0_0"][:]
    embedding = np.asarray(embedder.encoder(raw)).astype("float16").astype("float32")
    check = predict_image(rf, "object", embedding, raw, instances, semantic, None)
    agreement = np.mean(check[instances > 0] == stored[instances > 0])
    print(f"{roi}: check against the prediction of the test tile 0_0: pixel agreement {agreement:.3f}", flush=True)


PREPARE = {
    "pannuke": prepare_pannuke, "puma": lambda: prepare_puma("puma"),
    "puma_tissue": lambda: prepare_puma("puma_tissue"), "bcss": prepare_bcss, "ignite": prepare_ignite,
    "bbbc041": prepare_bbbc041,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "annotate", "predict", "plot", "download", "slide", "context"])
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--budget", type=int, default=None, help="The label budget, by default 25 / 100.")
    parser.add_argument("--methods", nargs="+", choices=list(ENCODERS), default=list(ENCODERS))
    parser.add_argument("--max_test", type=int, default=300, help="Predict at most this many test images.")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS, help="The splits to annotate.")
    parser.add_argument("--selection", choices=list(SELECTIONS), default="default", help="The candidate rule.")
    parser.add_argument("--slide", help="The slide name (without .svs) in SLIDE_DIR, for the 'slide' command.")
    parser.add_argument("--part", type=int, nargs=2, default=(0, 1), help="Do block k of n of the slide (k n).")
    parser.add_argument("--merge", action="store_true", help="Merge all files of the slide.")
    parser.add_argument("--rows", type=int, nargs=2, default=None, help="Do these tile rows of a slide (first last).")
    parser.add_argument("--roi", help="The PUMA ROI, for the 'context' command, e.g. training_set_primary_roi_045.")
    parser.add_argument("--step", choices=["segment", "classify"], help="The step of the 'context' command.")
    args = parser.parse_args()
    if args.command == "prepare":
        PREPARE[args.dataset]()
    elif args.command == "annotate":
        annotate(args.dataset, args.splits)
    elif args.command == "predict":
        predict(args.dataset, args.budget, args.methods, args.max_test)
    elif args.command == "download":
        download(args.dataset, args.slide)
    elif args.command == "context":
        if args.step == "segment":
            segment_context(args.roi)
        else:
            classify_context(args.roi, args.budget, args.methods[0])
    elif args.command == "slide":
        if args.merge:
            merge_slide(args.dataset, args.budget, args.methods[0], args.slide)
        else:
            predict_slide(
                args.dataset, args.budget, args.methods[0], args.slide, tuple(args.part),
                None if args.rows is None else tuple(args.rows),
            )
    else:
        plot(args.dataset, args.budget, args.selection)


if __name__ == "__main__":
    main()
