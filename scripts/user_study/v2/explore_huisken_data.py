"""Segmentation with micro-sam2 of the Huisken lab data, see 'huisken_data/README.md' for the layout.

The high-res ExM tile (exm_lsm/high_res) is segmented automatically with AIS and APG and 'hvit_l_cells'.

The volume is a light-sheet stack of nuclei (4871, 2048, 2048) uint16 (ZYX). 'S000_..._P04871.h5' holds it as 'raw',
identical to the headerless 'S000_..._P04871.raw'. The default block lies in the densest nuclei of the specimen.

Usage:
    python explore_huisken_data.py  # AIS and APG on the default block
    python explore_huisken_data.py -e apg  # only APG
    python explore_huisken_data.py --downsample 2  # the same block, averaged 2x in zyx before segmenting
    python explore_huisken_data.py --downsample 2 -e ais --min_size 500  # AIS without the fragments in the nuclei
    python explore_huisken_data.py --view  # the raw block and the saved segmentations in napari
    python explore_huisken_data.py --full --downsample 2 -e ais --min_size 500  # the whole volume, needs ~250 GB RAM

The whole volume in parallel z-slabs, one GPU and < 64 GB RAM each (see 'submit_huisken_slabs.sh'):
    python explore_huisken_data.py --downsample 2 --min_size 500 --slab 0  # AIS on slab 0, see '--n_slabs'
    python explore_huisken_data.py --downsample 2 --min_size 500 --stitch  # merges the slabs across their overlap

The two smaller volumes of the huisken data, with every setting of SMALL_SETTINGS (see there):
- 'exm': the expansion microscopy nuclei (sytox green) of a 6 dpf Platynereis, shared as a downsampled MJPEG video
  (765, 565, 1405) uint8 (ZYX). The nuclei are only ~10-15 px across, the bright ones saturate at 255 and the
  compression adds block artifacts, while the nuclei on the right side of the specimen are dim.
- 'membrane': the fused light-sheet stack (Beke) of the membranes of a 2 dpf Platynereis, (83, 509, 542) uint16,
  without spacing metadata. The specimen is ~430 px wide in xy and ~80 slices deep, so z is ~4-5x coarser than xy.

    python explore_huisken_data.py --sweep exm  # all settings on the crops of the exm volume
    python explore_huisken_data.py --sweep membrane  # all settings on the whole (small) membrane volume
    python explore_huisken_data.py -d exm -s clahe_x2  # one setting on the whole volume
    python explore_huisken_data.py --view -d exm --crop dense  # the raw crop and its segmentations in napari

The two early embryos of Beke (classic_lsm/early_embryo, see EMBRYO_NAMES), with 'hvit_t_cells':
    python explore_huisken_data.py --embryo t0 --view  # the raw volume (and its ground truth) in napari
    python explore_huisken_data.py --embryo t0 --precompute  # the embeddings, over all visible GPUs
    python explore_huisken_data.py --embryo t0  # the annotator
    python explore_huisken_data.py --embryo t0 --slice_norm  # each slice normalized by its own percentiles
    python explore_huisken_data.py --embryo 3.5hpf_crop  # the embryo of 3.5hpf, upsampled 2x by SAM2
    python explore_huisken_data.py --embryo t0_crop --iterative_prompting  # prompted from the ground truth
    python explore_huisken_data.py --embryo t0_crop --apg  # automatic prompt generation, scored on the ground truth
    python explore_huisken_data.py --embryo t0_crop --ais --apg -m hvit_l_cells  # AIS and APG with another model
    python explore_huisken_data.py --embryo t0_crop --figure box  # the figure of the box prompting, iteration 2
    python explore_huisken_data.py --embryo 3.5hpf --cellpose 3d 2d_stitch  # cellposeSAM, scored on the crop
"""

import os
import time
import shutil
import argparse
import tempfile
from concurrent import futures

import cv2
import h5py
import tifffile
import numpy as np
from scipy import ndimage
from skimage.exposure import equalize_adapthist
from skimage.filters import sato
from skimage.measure import block_reduce
from skimage.segmentation import find_boundaries

from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from micro_sam.v2.batched_inference import NORMALIZATION_SAMPLE_SLICES, _volume_normalization_bounds


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data/huisken_data"
TILE_NAME = "S000_t000000_V000_R0009_X000_Y000_C01_I0_D0_P04871"
DATA_PATH = os.path.join(DATA_ROOT, "exm_lsm", "high_res", "tile", f"{TILE_NAME}.h5")
RAW_PATH = os.path.join(DATA_ROOT, "exm_lsm", "high_res", "tile", f"{TILE_NAME}.raw")
RAW_SHAPE = (4871, 2048, 2048)
OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/huisken_data"
MODEL_TYPE = "hvit_l_cells"
ENGINES = ("ais", "apg")
# ZYX start of the block, around the nuclei-dense z range 2700-3100 at the center of the specimen.
BLOCK_START = (2900, 1024, 1280)
BLOCK_SHAPE = (128, 512, 512)
# The 'hvit_l' default of 50 voxels keeps a fragment from a weak second seed in about every nucleus of the 2x
# downsampled block. Removing them floods their voxels into the enclosing nucleus, 500 drops 107 of 108.
AIS_MIN_SIZE = None
N_READ_THREADS = 16
# The threads of the AIS flow post-processing, 16 is 15% faster than the default 8.
N_POSTPROCESSING_THREADS = 16
# The preview of the whole volume is downsampled once more by this factor, to fit into a laptop's memory.
PREVIEW_FACTOR = 2
# The z-slabs of the whole volume, on the downsampled grid. Each slab is segmented with a halo of extra slices on
# both sides, so neighbouring slabs overlap by twice the halo. A nucleus spans ~11 slices of the 2x grid.
SLAB_SIZE = 256
SLAB_HALO = 16
# Two objects of neighbouring slabs are merged if they are each other's best match in the overlap with this IoU.
STITCH_IOU = 0.5
# AIS segments noise and light-sheet stripes in the empty slices as objects. They are removed from the whole volume
# segmentation by their contrast, the mean raw intensity over that of the background ring around them: the noise is
# at 1.00-1.01, the nuclei at 1.4-3.5. A fixed intensity threshold would remove the nuclei at both ends of the
# specimen too, which the light-sheet illuminates more dimly (300 kept 93% of the nuclei in the core, this 98%).
MIN_CONTRAST = 1.1
# The width of the background ring, in voxels of the 2x downsampled grid.
RING_WIDTH = 6
# The two smaller volumes, see the module docstring.
SMALL_DATA_PATHS = {
    "exm": os.path.join(DATA_ROOT, "exm_lsm", "low_res", "whole", "Pdu_6dpf_ExM_downsampled_nuclei-sytox-green.avi"),
    "membrane": os.path.join(DATA_ROOT, "classic_lsm", "membrane", "fused_t000050_R0001_C02_Dfinal_cropped.tif"),
}
# The embeddings are written to the tmp folder of the job and removed after each segmentation.
EMBEDDING_ROOT = os.environ.get("TMPDIR", tempfile.gettempdir())
# The ZYX crops of the exm sweep: the dense, saturated nuclei on the left and the dim nuclei on the right.
EXM_CROPS = {
    "dense": (slice(350, 414), slice(100, 420), slice(40, 420)),
    "dim": (slice(350, 414), slice(200, 460), slice(1000, 1260)),
    # The center of the dense crop, 1/8 of its volume: APG propagates every object through z on its own, which takes
    # > 2h on the whole dense crop.
    "dense_small": (slice(366, 398), slice(180, 340), slice(135, 325)),
}
# The intensity steps run on the whole volume, so that a crop is prepared like the whole volume: 'clahe' equalizes the
# contrast locally per z-slice with this clip limit, which lifts the dim nuclei next to the saturated ones, 'gamma'
# lifts them globally and 'smooth' is the sigma of a gaussian against the compression blocks. 'norm_percentiles' are
# the normalization bounds, also from the whole volume (the library default of the 2nd / 98th percentile of the input
# would differ per crop: the exm upper bound is 255 in the dense crop, 54 in the dim crop and 188 in the volume).
# Then 'scale' resamples the input (z, y, x) linearly and the labels back. 'anisotropy' is the z spacing over the xy
# spacing after the scaling, for the AIS flow post-processing. 'min_size' is in voxels of the original grid.
# 'merge_fragments' merges the objects below this size (original grid) into the object that has at least
# FRAGMENT_MIN_ENCLOSED of their border, the fragments that weak second seeds leave inside the nuclei.
# 'invert' flips the intensities, so that the cells are bright and the membranes dark. 'tile_shape' and 'halo' (y, x)
# tile every slice: each tile with its halo is resized to the 1024 px of the model input, which magnifies the thin
# membranes (the library tiles only slices above 768 px). 'ndim' 2 segments every z-slice on its own, without the
# scaling of z, and normalizes it with the bounds of the volume before the library's own percentiles of the slice.
# 'min_border_intensity' removes the objects whose border is darker than this on average, in raw intensities: the
# membrane cells are bounded by bright membranes (> 5000), the noise objects in the dark interior and the background
# around the specimen are not (~600). Any threshold between 1000 and 3000 removes the same objects.
# 'min_mean_intensity' removes the objects darker than this on average, in raw intensities: in the whole exm volume
# AIS segments the black (0) background, e.g. the empty slices above the specimen, into 582 objects of mean < 5, while
# the dim nuclei of the dim crop are at >= 8. The contrast filter of the hires volume misses them, the background ring
# is black too.
# 'ridge' sharpens the blurred membranes: it blends the volume with this weight of a Sato ridge filter (bright ridges,
# sigmas RIDGE_SIGMAS) per z-slice, before the CLAHE.
SMALL_DEFAULT_SETTINGS = {
    "ridge": None, "clahe": None, "gamma": None, "smooth": None, "invert": False, "norm_percentiles": (2, 98),
    "scale": (1, 1, 1), "anisotropy": 1, "tile_shape": None, "halo": None, "ndim": 3, "min_size": None,
    "merge_fragments": None, "min_border_intensity": None, "min_mean_intensity": None,
}
FRAGMENT_MIN_ENCLOSED = 0.6
# A smaller volume with more z-slices than this is segmented in z-slabs of the original grid, overlapping by twice the
# halo, and stitched like the slabs of the hires volume: the whole exm volume at 2x takes > 256 GB in one piece. Every
# slab is saved, so that a preempted job resumes from the next one.
SMALL_SLAB_SIZE = 128
SMALL_SLAB_HALO = 16
RIDGE_SIGMAS = (1, 2)
SMALL_SETTINGS = {
    # The nuclei are ~10-15 px across. 50% of the voxels are 0 and 1% saturate, the 95th / 98th / 99th percentiles
    # are 71 / 188 / 248, the nuclei on the right side are at ~30-130.
    "exm": {
        "raw": {},
        "x2": {"scale": (2, 2, 2)},
        "x2_p99.9": {"scale": (2, 2, 2), "norm_percentiles": (2, 99.9)},
        "x2_p95": {"scale": (2, 2, 2), "norm_percentiles": (2, 95)},
        "gamma_x2": {"gamma": 0.5, "smooth": 0.7, "scale": (2, 2, 2)},
        "clahe_x2": {"clahe": 0.02, "smooth": 0.7, "scale": (2, 2, 2)},
        "clahe_x2_p99.9": {"clahe": 0.02, "smooth": 0.7, "scale": (2, 2, 2), "norm_percentiles": (2, 99.9)},
        "clahe_x3": {"clahe": 0.02, "smooth": 0.7, "scale": (3, 3, 3)},
        # The 2x AIS results are the cleanest, but half of their objects are fragments < 50 voxels, while the nuclei
        # have ~500 - 1700. 3x splits the nuclei into more fragments, 1x merges neighbours, CLAHE and gamma add objects
        # in the background haze.
        "x2_min100": {"scale": (2, 2, 2), "min_size": 100},
        "x2_p99.9_min100": {"scale": (2, 2, 2), "norm_percentiles": (2, 99.9), "min_size": 100},
        # min100 leaves 10 of the 922 objects of the dense crop as fragments inside a nucleus, this merges them.
        "x2_min100_merge": {"scale": (2, 2, 2), "min_size": 100, "merge_fragments": 400},
        "x2_min100_merge_bright": {"scale": (2, 2, 2), "min_size": 100, "merge_fragments": 400,
                                   "min_mean_intensity": 5},
    },
    # The background is at ~200-1000, the bright membranes on top at 10k-65k, the dim ones below at a few 1000.
    "membrane": {
        "raw": {"anisotropy": 4},
        "z4": {"scale": (4, 1, 1)},
        "z4_p1-99.9": {"scale": (4, 1, 1), "norm_percentiles": (1, 99.9)},
        "z4_p5-99.5": {"scale": (4, 1, 1), "norm_percentiles": (5, 99.5)},
        "z4_p10-95": {"scale": (4, 1, 1), "norm_percentiles": (10, 95)},
        "gamma_z4": {"gamma": 0.5, "scale": (4, 1, 1)},
        "clahe_z4": {"clahe": 0.01, "scale": (4, 1, 1)},
        "clahe_z4_x2": {"clahe": 0.01, "scale": (8, 2, 2)},
        # Upsampling z is needed to separate the cells, the 10th - 95th percentile merges the dim cells and 8 x 2 x 2
        # segments the noise in the dark interior. A third of the z4 objects are fragments < 200 voxels.
        "z4_min200": {"scale": (4, 1, 1), "min_size": 200},
        "z4_p1-99.9_min200": {"scale": (4, 1, 1), "norm_percentiles": (1, 99.9), "min_size": 200},
        # The membranes are only ~1-2 px thin, the cells ~15-25 px wide. Tiles of 256 + 2 * 64 px magnify them 2.7x,
        # tiles of 128 + 2 * 32 px 5.3x, upsampling 2x in xy with tiles of 256 magnifies them 5.3x with more context.
        "z4_t256": {"scale": (4, 1, 1), "tile_shape": (256, 256), "halo": (64, 64)},
        "z4_p1-99.9_t256": {"scale": (4, 1, 1), "norm_percentiles": (1, 99.9), "tile_shape": (256, 256),
                            "halo": (64, 64)},
        "z4_t128": {"scale": (4, 1, 1), "tile_shape": (128, 128), "halo": (32, 32)},
        "z4_x2_t256": {"scale": (4, 2, 2), "anisotropy": 2, "tile_shape": (256, 256), "halo": (64, 64)},
        "inv_z4": {"invert": True, "scale": (4, 1, 1)},
        "inv_z4_t256": {"invert": True, "scale": (4, 1, 1), "tile_shape": (256, 256), "halo": (64, 64)},
        "2d": {"ndim": 2},
        "2d_t256": {"ndim": 2, "tile_shape": (256, 256), "halo": (64, 64)},
        "2d_t128": {"ndim": 2, "tile_shape": (128, 128), "halo": (32, 32)},
        # Tiling traces the small cells, t128 and x2_t256 best. x2_t256 splits them into more fragments (1518 of its
        # 2748 objects are < 200 voxels, 275 of 1101 for t128). Both segment noise in the dark interior and around the
        # specimen, and the tiled runs cut a few cells along the tile borders, which the larger halo should reduce.
        "z4_t128_clean": {"scale": (4, 1, 1), "tile_shape": (128, 128), "halo": (32, 32), "min_size": 200,
                          "min_border_intensity": 1500},
        "z4_t128_h64_clean": {"scale": (4, 1, 1), "tile_shape": (128, 128), "halo": (64, 64), "min_size": 200,
                              "min_border_intensity": 1500},
        "z4_x2_t256_clean": {"scale": (4, 2, 2), "anisotropy": 2, "tile_shape": (256, 256), "halo": (64, 64),
                             "min_size": 200, "min_border_intensity": 1500},
        # t128_clean leaves the dimmer tissue (e.g. the left side) unsegmented, next to the bright top epithelium.
        # Lifting the dim membranes: locally (CLAHE), globally (gamma) or by saturating the bright ones (p1-95).
        "clahe_z4_t128_clean": {"clahe": 0.02, "scale": (4, 1, 1), "tile_shape": (128, 128), "halo": (32, 32),
                                "min_size": 200, "min_border_intensity": 1500},
        "gamma_z4_t128_clean": {"gamma": 0.5, "scale": (4, 1, 1), "tile_shape": (128, 128), "halo": (32, 32),
                                "min_size": 200, "min_border_intensity": 1500},
        "z4_p1-95_t128_clean": {"norm_percentiles": (1, 95), "scale": (4, 1, 1), "tile_shape": (128, 128),
                                "halo": (32, 32), "min_size": 200, "min_border_intensity": 1500},
        # Neither of them segments the blurred tissue, not even in 2d. Of the deblurring tested in 2d (unsharp mask,
        # Richardson-Lucy, difference of gaussians, ridge filter), only the ridge filter recovers it: 490 objects on
        # z 15, 41 and 60 against 220 with CLAHE alone, and the sharp tissue gains cells too.
        "ridge_clahe_z4_t128_clean": {"ridge": 0.5, "clahe": 0.02, "scale": (4, 1, 1), "tile_shape": (128, 128),
                                      "halo": (32, 32), "min_size": 200, "min_border_intensity": 1500},
        "ridge0.7_clahe_z4_t128_clean": {"ridge": 0.7, "clahe": 0.02, "scale": (4, 1, 1), "tile_shape": (128, 128),
                                         "halo": (32, 32), "min_size": 200, "min_border_intensity": 1500},
        "ridge_z4_t128_clean": {"ridge": 0.5, "scale": (4, 1, 1), "tile_shape": (128, 128), "halo": (32, 32),
                                "min_size": 200, "min_border_intensity": 1500},
    },
}


# The two early embryos of Beke, classic light-sheet stacks of the membranes, (Z, 1024, 1024) uint16 (ZYX). 't0' is
# fused t0 (208 slices) and '3.5hpf' the 3.5 hpf composite (79 slices). Each '_crop' is the 512x512 region around the
# embryo, which contains all of the ground truth, as a tif next to the volume.
EMBRYO_ROOT = os.path.join(DATA_ROOT, "classic_lsm", "early_embryo")
EMBRYO_NAMES = {
    "t0": "fused_t000000_R0000_C02_Dfinal",
    "t0_crop": "fused_t000000_R0000_C02_Dfinal_z0-208_y215-727_x253-765",
    "3.5hpf": "C2-Pdu_Composite_3.5hpf_HRas-mScarlet_H2B-GFP",
    "3.5hpf_crop": "C2-Pdu_Composite_3.5hpf_HRas-mScarlet_H2B-GFP_z0-79_y211-723_x255-767",
}
# Only 't0_crop' is evaluated: iterative prompting and APG are scored on its ground truth without the fragments.
EMBRYO_GT_VOLUME = "t0_crop"
EMBRYO_MODEL_TYPE = "hvit_t_cells"
EMBRYO_OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/interactive_visualization"
EMBRYO_N_ITERATIONS = 8
# Drops the specks of the filtered ground truth: its 19 cells have >= 105k voxels, every other object <= 4k.
EMBRYO_MIN_SIZE = 5000
# ZYX in um. The tif has no z spacing, it is estimated from the cell extents.
EMBRYO_SCALE = (0.812, 0.406, 0.406)
# The crop of each full embryo volume and its ROI, so that a segmentation of the full volume is scored on the crop.
EMBRYO_CROPS = {
    "t0": ("t0_crop", np.s_[0:208, 215:727, 253:765]),
    "3.5hpf": ("3.5hpf_crop", np.s_[0:79, 211:723, 255:767]),
}
# The cellposeSAM modes, as in explore_beghin_data.py: '3d' predicts the flows on the yx, zx and zy planes and follows
# their average in 3D, '2d_stitch' segments every z plane in 2D and stitches the masks of neighboring planes with
# IoU >= CELLPOSE_STITCH_THRESHOLD.
CELLPOSE_MODES = ("3d", "2d_stitch")
CELLPOSE_STITCH_THRESHOLD = 0.5
# The figure: the prediction tif suffix and the layer name of each choice of '--figure'.
EMBRYO_FIGURES = {
    "box": (f"prediction_microsam2_{EMBRYO_MODEL_TYPE}_box_iteration2", "iteration 2"),
    "apg": (f"prediction_microsam2_{EMBRYO_MODEL_TYPE}_apg", "apg"),
}
FIGURE_DOWNSAMPLE = 0.25
FIGURE_GAP = 50
FIGURE_BORDER_WIDTH = 2
FIGURE_MIN_FRAGMENT_SIZE = 1000
FIGURE_CLOSING_RADIUS_2D = 3
AXES_LABEL_OFFSET = 0.25
PALETTE = [
    (1.00, 0.10, 0.10), (0.10, 0.60, 1.00), (0.10, 0.85, 0.10), (1.00, 0.85, 0.00), (0.70, 0.10, 1.00),
    (0.00, 0.90, 0.90), (1.00, 0.45, 0.00), (1.00, 0.10, 0.85), (0.45, 1.00, 0.10), (0.00, 0.30, 0.90),
    (1.00, 0.65, 0.75), (0.55, 0.35, 0.10), (0.00, 0.90, 0.50), (0.80, 0.00, 0.30), (0.85, 1.00, 0.10),
    (0.10, 0.10, 0.70), (1.00, 0.55, 0.55), (0.10, 0.70, 0.50), (0.90, 0.70, 0.00), (0.50, 0.00, 0.50),
    (0.00, 0.70, 0.30), (1.00, 0.70, 0.20), (0.30, 0.00, 0.70), (0.00, 1.00, 0.75), (0.75, 0.20, 0.00),
]


def get_roi(start, shape):
    return tuple(slice(begin, begin + size) for begin, size in zip(start, shape))


def get_name(roi, downsample=1):
    name = "hires_volume_" + "_".join(f"{axis}{r.start}-{r.stop}" for axis, r in zip("zyx", roi))
    return name if downsample == 1 else f"{name}_ds{downsample}"


def load_block(roi):
    with h5py.File(DATA_PATH, "r") as f:
        return f["raw"][roi]


def downsample_volume(volume, factor):
    if factor == 1:
        return volume
    return block_reduce(volume, (factor,) * 3, np.mean).astype(volume.dtype)


def upsample_labels(segmentation, factor, shape):
    for axis in range(segmentation.ndim):
        segmentation = np.repeat(segmentation, factor, axis=axis)
    return segmentation[tuple(slice(0, size) for size in shape)]


def segment(volume, engine, embedding_path, min_size, norm_bounds=None):
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine, ndim=3)
    generate_kwargs = {}
    if engine == "ais":
        generate_kwargs["n_threads"] = N_POSTPROCESSING_THREADS
        # Only the AIS post-processing takes the minimum size, None keeps the model default.
        if min_size is not None:
            generate_kwargs["min_size"] = min_size
    start = time.time()
    # 'batch_size=None' picks the batch size per device, and the default 'devices' uses every visible GPU.
    segmentation = automatic_instance_segmentation(
        predictor, segmenter, input_path=volume, embedding_path=embedding_path, ndim=3, batch_size=None,
        norm_bounds=norm_bounds, **generate_kwargs,
    )
    print(f"{engine.upper()}: max label id {segmentation.max()}, segmented in {time.time() - start:.1f} s")
    return segmentation


def get_key(engine, min_size):
    return engine if engine != "ais" or min_size is None else f"ais_min{min_size}"


def run(roi, engines, downsample, min_size):
    name = get_name(roi, downsample)
    full_volume = load_block(roi)
    volume = downsample_volume(full_volume, downsample)
    print(f"Block {name}: {volume.shape} {volume.dtype}, range [{volume.min()}, {volume.max()}]")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    # AIS and APG decode from the same embeddings, so they are cached once per block.
    embedding_path = os.path.join(OUTPUT_ROOT, f"{name}_{MODEL_TYPE}.zarr")
    output_path = os.path.join(OUTPUT_ROOT, f"{name}_{MODEL_TYPE}.h5")
    for engine in engines:
        segmentation = segment(volume, engine, embedding_path, min_size)
        # The labels go back onto the full resolution grid, so that every run is viewed over the same raw.
        segmentation = upsample_labels(segmentation, downsample, full_volume.shape)
        with h5py.File(output_path, "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", data=full_volume, compression="gzip")
            key = get_key(engine, min_size)
            if key in f:
                del f[key]
            f.create_dataset(key, data=segmentation, compression="gzip")
    print(f"Saved the raw block and the segmentations to {output_path}")


def load_downsampled_volume(downsample, z_range=None, n_threads=N_READ_THREADS):
    """Downsample the whole volume, or the z range of it on the downsampled grid, from the uncompressed raw file.

    The headerless raw file is memory mapped, which reads about 8x faster than decompressing the h5.
    The trailing slices and pixels that do not fill a whole downsampling block are dropped.
    """
    data = np.memmap(RAW_PATH, dtype="<u2", mode="r", shape=RAW_SHAPE)
    shape = tuple(size // downsample for size in RAW_SHAPE)
    z_start, z_stop = (0, shape[0]) if z_range is None else z_range
    volume = np.zeros((z_stop - z_start,) + shape[1:], dtype=data.dtype)

    def _downsample_slices(z):
        stop = min(z + downsample, z_stop)
        block = data[z * downsample:stop * downsample, :shape[1] * downsample, :shape[2] * downsample]
        volume[z - z_start:stop - z_start] = downsample_volume(np.asarray(block), downsample)

    with futures.ThreadPoolExecutor(n_threads) as executor:
        list(executor.map(_downsample_slices, range(z_start, z_stop, downsample)))
    return volume


def get_volume_norm_bounds(downsample):
    """The normalization bounds of the whole downsampled volume, from the same z sample the library takes.

    Every slab is normalized with them. A slab of mostly background normalized by its own bounds
    stretches the camera noise to the full range (slab 0 spans 106-126, the whole volume 107-881).
    """
    n_slices = RAW_SHAPE[0] // downsample
    step = max(1, n_slices // NORMALIZATION_SAMPLE_SLICES)
    sample = np.concatenate([load_downsampled_volume(downsample, (z, z + 1)) for z in range(0, n_slices, step)])
    return _volume_normalization_bounds(sample)


def remove_low_contrast_objects(segmentation, volume, min_contrast=MIN_CONTRAST, ring_width=RING_WIDTH):
    """Remove the objects that are not brighter than the background around them, in place."""
    removed, n_objects = [], 0
    bboxes = ndimage.find_objects(segmentation)
    for object_id, bbox in enumerate(bboxes, 1):
        if bbox is None:
            continue
        n_objects += 1
        bbox = tuple(slice(max(axis.start - ring_width, 0), axis.stop + ring_width) for axis in bbox)
        labels = segmentation[bbox]
        mask = labels == object_id
        ring = ndimage.binary_dilation(mask, iterations=ring_width) & (labels == 0)
        # An object enclosed by other objects has no background to compare with, so it is kept.
        if not ring.any():
            continue
        raw = volume[bbox]
        if raw[mask].mean() < min_contrast * raw[ring].mean():
            removed.append(object_id)

    if removed:
        lut = np.arange(len(bboxes) + 1, dtype=segmentation.dtype)
        lut[removed] = 0
        for z in range(0, segmentation.shape[0], 64):
            segmentation[z:z + 64] = lut[segmentation[z:z + 64]]
    print(f"Removed {len(removed)} of {n_objects} objects with a contrast below {min_contrast}", flush=True)
    return segmentation


def save_preview(volume, segmentation, path, factor):
    """Save the raw data and segmentation, downsampled further, small enough to load on a laptop."""
    raw = downsample_volume(volume, factor)
    labels = segmentation[::factor, ::factor, ::factor][tuple(slice(0, size) for size in raw.shape)]
    dtype = "uint16" if labels.max() <= np.iinfo("uint16").max else "uint32"
    with h5py.File(path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip", chunks=True)
        f.create_dataset("segmentation", data=labels.astype(dtype), compression="gzip", chunks=True)
    print(f"Saved the preview {raw.shape} to {path}")


def run_full(engines, downsample, min_size, preview_factor):
    name = f"hires_volume_ds{downsample}"
    start = time.time()
    volume = load_downsampled_volume(downsample)
    print(f"Volume {name}: {volume.shape} {volume.dtype}, read and downsampled in {time.time() - start:.1f} s")

    embedding_path = os.path.join(OUTPUT_ROOT, f"{name}_{MODEL_TYPE}.zarr")
    for engine in engines:
        segmentation = remove_low_contrast_objects(segment(volume, engine, embedding_path, min_size), volume)
        key = get_key(engine, min_size)
        # The segmentation stays on the downsampled grid, at full resolution it would take 80 GB.
        output_path = os.path.join(OUTPUT_ROOT, f"{name}_{MODEL_TYPE}_{key}.h5")
        start = time.time()
        with h5py.File(output_path, "w") as f:
            f.create_dataset("segmentation", data=segmentation, compression="gzip", chunks=(64, 256, 256))
            f.attrs["downsample"] = downsample
        print(f"Saved the segmentation to {output_path} in {time.time() - start:.1f} s")

        preview_path = os.path.join(OUTPUT_ROOT, f"hires_volume_ds{downsample * preview_factor}_{MODEL_TYPE}_{key}.h5")
        save_preview(volume, segmentation, preview_path, preview_factor)


def get_slabs(z_range, slab_size=SLAB_SIZE, halo=SLAB_HALO):
    """The (start, stop, core start, core stop) of every slab. The cores tile the z range, the halos overlap."""
    z_start, z_stop = z_range
    slabs = []
    for core_start in range(z_start, z_stop, slab_size):
        core_stop = min(core_start + slab_size, z_stop)
        slabs.append((max(core_start - halo, z_start), min(core_stop + halo, z_stop), core_start, core_stop))
    return slabs


def get_z_range(downsample, z_range):
    return (0, RAW_SHAPE[0] // downsample) if z_range is None else tuple(z_range)


def get_slab_dir(downsample, min_size, z_range, slab_size):
    name = f"slabs{slab_size}_ds{downsample}_{MODEL_TYPE}_{get_key('ais', min_size)}_z{z_range[0]}-{z_range[1]}"
    return os.path.join(OUTPUT_ROOT, name)


def run_slab(index, downsample, min_size, z_range=None, slab_size=SLAB_SIZE, cache_embeddings=False):
    """Segment one slab with AIS and save it. Every slab runs on its own, so the slabs can go to separate GPUs."""
    z_range = get_z_range(downsample, z_range)
    start, stop, core_start, core_stop = get_slabs(z_range, slab_size)[index]
    begin = time.time()
    volume = load_downsampled_volume(downsample, (start, stop))
    print(f"Slab {index}: slices {start}-{stop} {volume.shape}, read in {time.time() - begin:.1f} s", flush=True)

    # The embeddings of a slab only feed its own decoder. Unless they are cached for later, they are written
    # to the node-local disk and removed afterwards, rather than to the shared file system.
    embedding_root = OUTPUT_ROOT if cache_embeddings else os.environ.get("TMPDIR", OUTPUT_ROOT)
    embedding_path = os.path.join(embedding_root, f"hires_volume_ds{downsample}_z{start}-{stop}_{MODEL_TYPE}.zarr")
    norm_bounds = get_volume_norm_bounds(downsample)
    try:
        segmentation = segment(volume, "ais", embedding_path, min_size, norm_bounds)
    finally:
        if not cache_embeddings:
            shutil.rmtree(embedding_path, ignore_errors=True)

    slab_dir = get_slab_dir(downsample, min_size, z_range, slab_size)
    os.makedirs(slab_dir, exist_ok=True)
    with h5py.File(os.path.join(slab_dir, f"slab_{index:03d}.h5"), "w") as f:
        f.create_dataset("segmentation", data=segmentation, compression="lzf", chunks=(16, 256, 256))
        f.attrs.update({"start": start, "stop": stop, "core_start": core_start, "core_stop": core_stop})
    print(f"Slab {index}: done in {time.time() - begin:.1f} s", flush=True)


def match_overlap(lower, upper, iou_threshold=STITCH_IOU):
    """The (lower id, upper id) pairs of the objects that are each other's best match in the overlap of two slabs."""
    mask = (lower != 0) & (upper != 0)
    pairs, overlaps = np.unique(
        (lower[mask].astype("uint64") << np.uint64(32)) | upper[mask].astype("uint64"), return_counts=True,
    )
    lower_ids, upper_ids = (pairs >> np.uint64(32)).astype("int64"), (pairs & np.uint64(0xFFFFFFFF)).astype("int64")
    lower_sizes = dict(zip(*np.unique(lower, return_counts=True)))
    upper_sizes = dict(zip(*np.unique(upper, return_counts=True)))

    best_lower, best_upper = {}, {}
    for lower_id, upper_id, overlap in zip(lower_ids, upper_ids, overlaps):
        if overlap > best_lower.get(lower_id, (None, 0))[1]:
            best_lower[lower_id] = (upper_id, overlap)
        if overlap > best_upper.get(upper_id, (None, 0))[1]:
            best_upper[upper_id] = (lower_id, overlap)

    matches = []
    for lower_id, (upper_id, overlap) in best_lower.items():
        if best_upper[upper_id][0] != lower_id:
            continue
        iou = overlap / (lower_sizes[lower_id] + upper_sizes[upper_id] - overlap)
        if iou >= iou_threshold:
            matches.append((lower_id, upper_id))
    return matches


def stitch_slabs(downsample, min_size, z_range=None, slab_size=SLAB_SIZE, iou_threshold=STITCH_IOU):
    """Merge the slab segmentations: every slab writes its core, the objects matched in an overlap get one id."""
    z_range = get_z_range(downsample, z_range)
    slabs = get_slabs(z_range, slab_size)
    slab_dir = get_slab_dir(downsample, min_size, z_range, slab_size)
    shape = (z_range[1] - z_range[0],) + tuple(size // downsample for size in RAW_SHAPE[1:])
    segmentation = np.zeros(shape, dtype="uint32")

    begin = time.time()
    offset, matches, previous = 0, [], None
    for index, (start, stop, core_start, core_stop) in enumerate(slabs):
        with h5py.File(os.path.join(slab_dir, f"slab_{index:03d}.h5"), "r") as f:
            assert (f.attrs["start"], f.attrs["stop"]) == (start, stop), f"Slab {index} has other bounds."
            slab = f["segmentation"][:]
        # Every slab numbers its objects from 1, the offset makes the ids unique across the slabs.
        slab[slab != 0] += np.uint32(offset)
        if previous is not None:
            previous_slab, previous_start, previous_stop = previous
            overlap_lower = previous_slab[start - previous_start:]
            matches += match_overlap(overlap_lower, slab[:previous_stop - start], iou_threshold)
        segmentation[core_start - z_range[0]:core_stop - z_range[0]] = slab[core_start - start:core_stop - start]
        offset = max(offset, int(slab.max()))
        previous = (slab, start, stop)
    print(f"Read {len(slabs)} slabs and matched {len(matches)} objects across their overlaps in "
          f"{time.time() - begin:.1f} s", flush=True)

    segmentation, n_objects = merge_matched_ids(segmentation, matches, offset)
    print(f"Stitched {n_objects} objects in {time.time() - begin:.1f} s", flush=True)
    return segmentation


def merge_matched_ids(segmentation, matches, max_id):
    """Give the matched (lower id, upper id) pairs one id by union-find, then make the ids consecutive, in place."""
    parent = np.arange(max_id + 1, dtype="int64")

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for lower_id, upper_id in matches:
        parent[find(upper_id)] = find(lower_id)
    roots = np.array([find(node) for node in range(max_id + 1)])
    present = np.zeros(max_id + 1, dtype=bool)
    for z in range(0, segmentation.shape[0], 64):
        present[np.unique(segmentation[z:z + 64])] = True
    roots[~present] = 0
    _, lut = np.unique(roots, return_inverse=True)
    lut = lut.astype("uint32")
    for z in range(0, segmentation.shape[0], 64):
        segmentation[z:z + 64] = lut[segmentation[z:z + 64]]
    return segmentation, int(lut.max())


def run_stitch(downsample, min_size, preview_factor, z_range=None, slab_size=SLAB_SIZE):
    z_range = get_z_range(downsample, z_range)
    segmentation = stitch_slabs(downsample, min_size, z_range, slab_size)
    # The slabs keep every object, so that the filter can change without segmenting them again.
    volume = load_downsampled_volume(downsample, z_range)
    begin = time.time()
    segmentation = remove_low_contrast_objects(segmentation, volume)
    print(f"Filtered in {time.time() - begin:.1f} s", flush=True)
    key = get_key("ais", min_size)
    name = f"hires_volume_ds{downsample}_z{z_range[0]}-{z_range[1]}_{MODEL_TYPE}_{key}_slabs{slab_size}"
    output_path = os.path.join(OUTPUT_ROOT, f"{name}.h5")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("segmentation", data=segmentation, compression="gzip", chunks=(64, 256, 256))
        f.attrs.update({"downsample": downsample, "z_start": z_range[0], "z_stop": z_range[1]})
    print(f"Saved the segmentation to {output_path}", flush=True)

    preview_name = name.replace(f"_ds{downsample}_", f"_ds{downsample * preview_factor}_")
    save_preview(volume, segmentation, os.path.join(OUTPUT_ROOT, f"{preview_name}.h5"), preview_factor)


def view(roi, downsample):
    import napari

    name = get_name(roi, downsample)
    output_path = os.path.join(OUTPUT_ROOT, f"{name}_{MODEL_TYPE}.h5")
    viewer = napari.Viewer(title=name)
    if os.path.exists(output_path):
        with h5py.File(output_path, "r") as f:
            viewer.add_image(f["raw"][:], name="raw")
            for key in f:
                if key != "raw":
                    viewer.add_labels(f[key][:], name=key)
    else:
        viewer.add_image(load_block(roi), name="raw")
    napari.run()


def load_small_data(dataset):
    path = SMALL_DATA_PATHS[dataset]
    if dataset == "membrane":
        return tifffile.imread(path)
    # The video stores the gray values in all three color channels.
    capture, frames = cv2.VideoCapture(path), []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame[..., 0])
    capture.release()
    return np.stack(frames)


def enhance(raw, settings):
    """The intensity steps of the settings on the whole volume, and its normalization bounds."""
    volume = raw.astype("float32")
    if settings["clahe"] is not None or settings["gamma"] is not None or settings["ridge"] is not None:
        lower, upper = np.percentile(volume, (0.1, 99.9))
        volume = np.clip((volume - lower) / (upper - lower), 0, 1)
    if settings["ridge"] is not None:
        ridges = np.stack([sato(z, sigmas=RIDGE_SIGMAS, black_ridges=False) for z in volume])
        lower, upper = np.percentile(ridges, (0.1, 99.9))
        ridges = np.clip((ridges - lower) / (upper - lower), 0, 1)
        volume = (1 - settings["ridge"]) * volume + settings["ridge"] * ridges
        lower, upper = np.percentile(volume, (0.1, 99.9))
        volume = np.clip((volume - lower) / (upper - lower), 0, 1).astype("float32")
    if settings["clahe"] is not None:
        volume = np.stack([equalize_adapthist(z, clip_limit=settings["clahe"]) for z in volume]).astype("float32")
    if settings["gamma"] is not None:
        volume = volume ** settings["gamma"]
    if settings["smooth"] is not None:
        volume = ndimage.gaussian_filter(volume, settings["smooth"])
    if settings["invert"]:
        volume = volume.max() - volume
    lower, upper = np.percentile(volume, settings["norm_percentiles"])
    return volume, (np.array([lower], dtype="float32"), np.array([upper], dtype="float32"))


def merge_enclosed_fragments(segmentation, max_size, min_enclosed=FRAGMENT_MIN_ENCLOSED):
    """Merge the objects below 'max_size' into the object that covers at least 'min_enclosed' of their border."""
    sizes = np.bincount(segmentation.ravel())
    lut = np.arange(len(sizes), dtype=segmentation.dtype)
    bboxes = ndimage.find_objects(segmentation)
    n_merged = 0
    for object_id in np.argsort(sizes):
        if object_id == 0 or sizes[object_id] == 0 or sizes[object_id] >= max_size:
            continue
        bbox = tuple(slice(max(axis.start - 1, 0), axis.stop + 1) for axis in bboxes[object_id - 1])
        labels = lut[segmentation[bbox]]
        mask = labels == lut[object_id]
        border = labels[ndimage.binary_dilation(mask) & ~mask]
        if border.size == 0:
            continue
        neighbours, counts = np.unique(border[border != 0], return_counts=True)
        if len(neighbours) and counts.max() >= min_enclosed * border.size:
            lut[lut == lut[object_id]] = neighbours[counts.argmax()]
            n_merged += 1
    print(f"Merged {n_merged} fragments below {max_size} voxels into their enclosing objects", flush=True)
    return lut[segmentation]


def remove_dark_border_objects(segmentation, raw, min_intensity):
    """Remove the objects whose border (their outermost voxels) is darker than 'min_intensity' on average."""
    border = find_boundaries(segmentation, mode="inner")
    n_ids = int(segmentation.max()) + 1
    sums = np.bincount(segmentation[border], weights=raw[border].astype("float64"), minlength=n_ids)
    counts = np.bincount(segmentation[border], minlength=n_ids)
    means = sums / np.maximum(counts, 1)
    removed = np.nonzero((means < min_intensity) & (counts > 0))[0]
    removed = removed[removed != 0]
    lut = np.arange(n_ids, dtype=segmentation.dtype)
    lut[removed] = 0
    print(f"Removed {len(removed)} objects with a border darker than {min_intensity}", flush=True)
    return lut[segmentation]


def remove_dark_objects(segmentation, raw, min_intensity):
    """Remove the objects whose mean raw intensity is below 'min_intensity'."""
    n_ids = int(segmentation.max()) + 1
    sizes = np.bincount(segmentation.ravel(), minlength=n_ids)
    means = np.bincount(segmentation.ravel(), weights=raw.ravel().astype("float64"), minlength=n_ids)
    means /= np.maximum(sizes, 1)
    removed = np.nonzero((means < min_intensity) & (sizes > 0))[0]
    removed = removed[removed != 0]
    lut = np.arange(n_ids, dtype=segmentation.dtype)
    lut[removed] = 0
    print(f"Removed {len(removed)} objects with a mean intensity below {min_intensity}", flush=True)
    return lut[segmentation]


def segment_slices(volume, engine, settings, norm_bounds):
    """Segment every z-slice on its own in 2d, with object ids that are unique across the slices."""
    lower, upper = norm_bounds[0][0], norm_bounds[1][0]
    volume = np.clip((volume - lower) / (upper - lower), 0, 1).astype("float32")
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine, ndim=2)
    segmentation, offset = np.zeros(volume.shape, dtype="uint32"), 0
    for z, image in enumerate(volume):
        embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
        try:
            labels = automatic_instance_segmentation(
                predictor, segmenter, input_path=image, embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
                ndim=2, batch_size=None, verbose=False, tile_shape=settings["tile_shape"], halo=settings["halo"],
            ).astype("uint32")
        finally:
            shutil.rmtree(embedding_dir, ignore_errors=True)
        segmentation[z] = np.where(labels > 0, labels + offset, 0)
        offset = max(offset, int(segmentation[z].max()))
    return segmentation


def segment_small(volume, engine, settings, norm_bounds):
    start = time.time()
    if settings["ndim"] == 2:
        segmentation = segment_slices(volume, engine, settings, norm_bounds)
        return segmentation, len(np.unique(segmentation)) - 1, time.time() - start
    scale = tuple(settings["scale"])
    shape = volume.shape
    if scale != (1, 1, 1):
        volume = ndimage.zoom(volume, scale, order=1)
    predictor, segmenter = get_predictor_and_segmenter(model_type=MODEL_TYPE, segmentation_mode=engine, ndim=3)
    generate_kwargs = {}
    if engine == "ais":
        if settings["min_size"] is not None:
            generate_kwargs["min_size"] = int(settings["min_size"] * np.prod(scale))
        if settings["anisotropy"] > 1:
            generate_kwargs["spacing"] = (float(settings["anisotropy"]), 1.0, 1.0)

    embedding_dir = tempfile.mkdtemp(dir=EMBEDDING_ROOT)
    try:
        # 'batch_size=None' picks the batch size per device.
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=volume, embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
            ndim=3, batch_size=None, verbose=False, norm_bounds=norm_bounds, tile_shape=settings["tile_shape"],
            halo=settings["halo"], **generate_kwargs,
        )
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)
    if segmentation.shape != shape:
        # Nearest neighbour sampling at the voxel centers of the original grid.
        coords = [
            ((np.arange(size) + 0.5) * scaled / size).astype(int) for size, scaled in zip(shape, segmentation.shape)
        ]
        segmentation = segmentation[np.ix_(*coords)]
    segmentation = segmentation.astype("uint32")
    if settings["merge_fragments"] is not None:
        segmentation = merge_enclosed_fragments(segmentation, settings["merge_fragments"])
    n_objects = len(np.unique(segmentation)) - 1
    return segmentation, n_objects, time.time() - start


def save_small(path, raw, key, segmentation, attrs):
    with h5py.File(path, "a") as f:
        if "raw" not in f:
            f.create_dataset("raw", data=raw, compression="gzip")
        if key in f:
            del f[key]
        dataset = f.create_dataset(key, data=segmentation, compression="gzip")
        dataset.attrs.update(attrs)


def segment_small_slabs(volume, engine, settings, norm_bounds, slab_dir):
    """Segment the volume in z-slabs with 'segment_small', save every slab to 'slab_dir' and stitch them."""
    os.makedirs(slab_dir, exist_ok=True)
    start_time = time.time()
    slabs = get_slabs((0, volume.shape[0]), SMALL_SLAB_SIZE, SMALL_SLAB_HALO)
    segmentation = np.zeros(volume.shape, dtype="uint32")
    offset, matches, previous = 0, [], None
    for index, (start, stop, core_start, core_stop) in enumerate(slabs):
        slab_path = os.path.join(slab_dir, f"slab_{index:03d}.h5")
        if not os.path.exists(slab_path):
            slab, n_objects, runtime = segment_small(volume[start:stop], engine, settings, norm_bounds)
            # Written under a temporary name first, so that a slab cut off by preemption is not taken as done.
            with h5py.File(slab_path + ".tmp", "w") as f:
                f.create_dataset("segmentation", data=slab, compression="lzf")
            os.replace(slab_path + ".tmp", slab_path)
            print(f"Slab {index + 1} / {len(slabs)}: slices {start}-{stop}, {n_objects} objects in {runtime:.1f} s",
                  flush=True)
        with h5py.File(slab_path, "r") as f:
            slab = f["segmentation"][:]
        slab[slab != 0] += np.uint32(offset)
        if previous is not None:
            previous_slab, previous_start, previous_stop = previous
            matches += match_overlap(previous_slab[start - previous_start:], slab[:previous_stop - start])
        segmentation[core_start:core_stop] = slab[core_start - start:core_stop - start]
        offset = max(offset, int(slab.max()))
        previous = (slab, start, stop)
    segmentation, n_objects = merge_matched_ids(segmentation, matches, offset)
    print(f"Stitched {len(slabs)} slabs into {n_objects} objects", flush=True)
    return segmentation, n_objects, time.time() - start_time


def get_small_path(dataset, roi=None, tag=None):
    suffix = "" if roi is None else "_" + "_".join(f"{axis}{r.start}-{r.stop}" for axis, r in zip("zyx", roi))
    suffix += "" if tag is None else f"_{tag}"
    return os.path.join(OUTPUT_ROOT, f"{dataset}{suffix}_{MODEL_TYPE}.h5")


def run_small(dataset, names, engines, sweep, crops=None, tag=None):
    """Segment the volume with the settings, or for the exm sweep its crops (all or 'crops').

    The results go to a file per volume or crop, with 'tag' in its name if given (for jobs that run in parallel).
    """
    data = load_small_data(dataset)
    print(f"{dataset}: {data.shape} {data.dtype}, range [{data.min()}, {data.max()}]", flush=True)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    # The exm sweep runs on crops, the membrane volume is small enough to sweep as a whole.
    if sweep and dataset == "exm":
        rois = {name: roi for name, roi in EXM_CROPS.items() if crops is None or name in crops}
    else:
        rois = {"full": None}
    for name in names:
        settings = {**SMALL_DEFAULT_SETTINGS, **SMALL_SETTINGS[dataset][name]}
        volume, norm_bounds = enhance(data, settings)
        print(f"{name}: normalization bounds {norm_bounds[0][0]:.3f} - {norm_bounds[1][0]:.3f}", flush=True)
        for crop_name, roi in rois.items():
            path = get_small_path(dataset, roi, tag)
            raw = data if roi is None else data[roi]
            for engine in engines:
                input_volume = volume if roi is None else volume[roi]
                if input_volume.shape[0] > SMALL_SLAB_SIZE:
                    slab_dir = os.path.join(OUTPUT_ROOT, f"{dataset}_slabs{SMALL_SLAB_SIZE}_{engine}_{name}")
                    segmentation, n_objects, runtime = segment_small_slabs(
                        input_volume, engine, settings, norm_bounds, slab_dir
                    )
                else:
                    segmentation, n_objects, runtime = segment_small(input_volume, engine, settings, norm_bounds)
                if settings["min_border_intensity"] is not None:
                    segmentation = remove_dark_border_objects(segmentation, raw, settings["min_border_intensity"])
                    n_objects = len(np.unique(segmentation)) - 1
                if settings["min_mean_intensity"] is not None:
                    segmentation = remove_dark_objects(segmentation, raw, settings["min_mean_intensity"])
                    n_objects = len(np.unique(segmentation)) - 1
                print(f"{dataset} {crop_name} {name} {engine}: {n_objects} objects in {runtime:.1f} s", flush=True)
                attrs = {"settings": str(SMALL_SETTINGS[dataset][name]), "n_objects": n_objects, "runtime": runtime}
                save_small(path, raw, f"{engine}_{name}", segmentation, attrs)
            print(f"Saved to {path}", flush=True)


def view_small(dataset, crop, tag=None):
    import napari

    path = get_small_path(dataset, None if crop is None else EXM_CROPS[crop], tag)
    viewer = napari.Viewer(title=os.path.basename(path))
    with h5py.File(path, "r") as f:
        viewer.add_image(f["raw"][:], name="raw")
        for key in sorted(f):
            if key != "raw":
                viewer.add_labels(f[key][:], name=key)
    napari.run()


def load_embryo(volume):
    name = EMBRYO_NAMES[volume]
    return tifffile.imread(os.path.join(EMBRYO_ROOT, f"{name}.tif")), name


def get_embryo_prediction_path(name, suffix):
    return os.path.join(EMBRYO_ROOT, f"{name}_{suffix}.tif")


def normalize_per_slice(volume):
    from micro_sam.v2.normalization import normalize_raw

    # Every slice keeps >= 2% of its pixels at 0 and 1, so the volume bounds of the embeddings become (0, 1).
    return np.stack([normalize_raw(z_slice) for z_slice in volume])


def view_embryo(raw, name):
    import napari

    viewer = napari.Viewer(title=name)
    viewer.add_image(raw, name="raw")
    if os.path.exists(get_embryo_prediction_path(name, "groundtruth")):
        viewer.add_labels(tifffile.imread(get_embryo_prediction_path(name, "groundtruth")), name="groundtruth")
    napari.run()


def precompute_embryo_embeddings(raw, embedding_path, model_type):
    from micro_sam.v2.util import get_sam2_model, precompute_image_embeddings

    predictor = get_sam2_model(model_type=model_type, input_type="videos")
    # 'devices=None' spreads the slices over every visible GPU.
    embeddings = precompute_image_embeddings(predictor, raw, save_path=embedding_path, ndim=3, lazy_loading=True)
    embeddings.close()


def run_embryo_annotator(raw, embedding_path, model_type):
    import napari
    from micro_sam.sam_annotator import Annotator
    from micro_sam.sam_annotator._state import AnnotatorState
    from micro_sam.sam_annotator._titles import get_dock_title
    from micro_sam.sam_annotator.util import _sync_embedding_widget

    # 'annotator' tiles slices above 768 pixels, so build the tool here to segment over the full XY.
    state = AnnotatorState()
    state.image_shape = raw.shape
    state.initialize_predictor(
        raw, model_type=model_type, ndim=3, save_path=embedding_path, batch_size=None, use_cli=True
    )

    viewer = napari.Viewer()
    viewer.add_image(raw, name="image")
    annotator = Annotator(viewer, ndim=3, reset_state=False)
    annotator._update_image()
    viewer.window.add_dock_widget(annotator, name=get_dock_title("segmentation"))
    _sync_embedding_widget(
        widget=state.widgets["embeddings"], model_type=model_type, save_path=embedding_path,
        checkpoint_path=None, device=None, tile_shape=None, halo=None,
    )
    napari.run()


def load_embryo_ground_truth(name):
    from bioimage_cpp.segmentation import label as connected_components
    from torch_em.util.segmentation import size_filter

    gt = tifffile.imread(get_embryo_prediction_path(name, "groundtruth_filtered"))
    # The interactive inference relabels and filters the ground truth like this, so its predicted ids match these.
    return gt, size_filter(seg=connected_components(gt).astype(gt.dtype), min_size=EMBRYO_MIN_SIZE)


def evaluate_embryo(segmentation, labels):
    from bioimage_py.evaluation import symmetric_best_dice_score
    from elf.evaluation import mean_segmentation_accuracy

    msa, sa = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
    sbd = symmetric_best_dice_score(segmentation, labels)
    return f"mSA {msa:.4f}, SA50 {sa[0]:.4f}, SA75 {sa[5]:.4f}, SBD {sbd:.4f}"


def save_embryo_prediction(name, suffix, segmentation):
    path = get_embryo_prediction_path(name, suffix)
    tifffile.imwrite(path, segmentation.astype("uint16"), compression="zlib")
    print(f"Saved {path}")


def run_embryo_iterative_prompting(raw, name, model_type, prompt):
    from micro_sam.v2.evaluation.inference import run_interactive_segmentation_3d
    from micro_sam.v2.normalization import normalize_raw

    gt, labels = load_embryo_ground_truth(name)
    # Each slice is normalized on its own, as in training and in the interactive evaluation.
    prediction_dir = run_interactive_segmentation_3d(
        raw=np.stack([normalize_raw(z_slice, output_dtype="uint8") for z_slice in raw]),
        labels=gt,
        model_type=model_type,
        checkpoint_path=None,
        start_with_box_prompt=prompt == "box",
        prediction_dir=os.path.join(EMBRYO_OUTPUT_ROOT, "iterative_prompting", f"{name}_{model_type}"),
        prediction_fname=name,
        n_iterations=EMBRYO_N_ITERATIONS,
        min_size=EMBRYO_MIN_SIZE,
    )
    for i in range(EMBRYO_N_ITERATIONS):
        segmentation = tifffile.imread(os.path.join(prediction_dir, f"iteration{i}", f"{name}.tif"))
        print(f"Iteration {i}: {evaluate_embryo(segmentation.astype(labels.dtype), labels)}")
        save_embryo_prediction(name, f"prediction_microsam2_{model_type}_{prompt}_iteration{i}", segmentation)


def run_embryo_automatic(raw, name, model_type, embedding_path, mode):
    _, labels = load_embryo_ground_truth(name)
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode=mode, ndim=3)
    segmentation = automatic_instance_segmentation(
        predictor, segmenter, input_path=raw, embedding_path=embedding_path, ndim=3
    )
    print(f"{mode.upper()}: {evaluate_embryo(segmentation, labels)}")
    save_embryo_prediction(name, f"prediction_microsam2_{model_type}_{mode}", segmentation)


def run_embryo_cellpose(raw, volume, modes):
    """Segment the embryo with cellposeSAM (default settings) and score it on the ground truth of the crop.

    A full volume is segmented as a whole and cropped afterwards. The results are saved next to the crop as
    'prediction_cellposeSAM_<mode>', with the suffix '_full' for the full volume.
    """
    from cellpose import models

    crop_volume, roi = EMBRYO_CROPS.get(volume, (volume, None))
    crop_name = EMBRYO_NAMES[crop_volume]
    _, labels = load_embryo_ground_truth(crop_name)
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    for mode in modes:
        stitch_threshold = CELLPOSE_STITCH_THRESHOLD if mode == "2d_stitch" else 0.0
        start = time.time()
        segmentation = model.eval(raw, z_axis=0, do_3D=mode == "3d", stitch_threshold=stitch_threshold)[0]
        runtime = time.time() - start
        if roi is not None:
            segmentation = segmentation[roi]
        n_objects = len(np.unique(segmentation)) - 1
        print(f"cellposeSAM {mode}: {n_objects} objects, {evaluate_embryo(segmentation, labels)}, {runtime:.1f} s")
        assert segmentation.max() <= np.iinfo("uint16").max
        suffix = f"prediction_cellposeSAM_{mode}" + ("" if roi is None else "_full")
        save_embryo_prediction(crop_name, suffix, segmentation)


def remove_artifacts(segmentation, min_size):
    """Zero the connected pieces of each label with fewer than min_size voxels or on a single z slice."""
    from skimage.measure import label

    components = label(segmentation, background=0, connectivity=segmentation.ndim)
    sizes = np.bincount(components.ravel())
    n_slices = np.bincount(np.concatenate([np.unique(z_slice) for z_slice in components]), minlength=len(sizes))
    segmentation[(sizes[components] < min_size) | (n_slices[components] == 1)] = 0
    return segmentation


def fill_cells_2d(segmentation, cell_ids, radius):
    """Paint only these cells with their gaps closed and holes filled, the largest first so smaller ones stay on top."""
    from skimage.morphology import closing, disk

    out = np.zeros_like(segmentation)
    for cell_id in sorted(cell_ids, key=lambda cell_id: -(segmentation == cell_id).sum()):
        out[ndimage.binary_fill_holes(closing(segmentation == cell_id, disk(radius)))] = cell_id
    return out


def make_color_dict(label_ids):
    colors = {0: np.zeros(4, dtype="float32"), None: np.zeros(4, dtype="float32")}
    for i, label_id in enumerate(label_ids):
        colors[label_id] = np.array([*PALETTE[i % len(PALETTE)], 1.0], dtype="float32")
    return colors


def set_axes_label_offset(viewer):
    try:
        axes_vispy = viewer.window._qt_viewer.canvas._overlay_to_visual[viewer.axes][0]
        axes_vispy.node._text_offsets = AXES_LABEL_OFFSET * np.array([1, 1, 1])
        axes_vispy._on_data_change()
    except Exception:
        pass


def show_scaled(viewer, axis_labels):
    import napari

    viewer.dims.axis_labels = axis_labels
    viewer.axes.visible = True
    set_axes_label_offset(viewer)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"
    napari.run()


def view_embryo_figure(raw, name, figure):
    """The figure: the segmentation in 3d above the raw middle slice, then the cells of that slice in 2d."""
    import napari
    from napari.utils.colormaps.colormap import DirectLabelColormap
    from skimage.transform import rescale

    suffix, layer_name = EMBRYO_FIGURES[figure]
    segmentation = remove_artifacts(tifffile.imread(get_embryo_prediction_path(name, suffix)), FIGURE_MIN_FRAGMENT_SIZE)
    z_2d = raw.shape[0] // 2
    raw_2d_full = raw[z_2d].astype("float32")
    # Every cell on the slice, without the slivers that the 3d fragment filter keeps.
    cell_ids, areas = np.unique(segmentation[z_2d][segmentation[z_2d] > 0], return_counts=True)
    cells_2d = [int(cell_id) for cell_id in cell_ids[areas >= FIGURE_MIN_FRAGMENT_SIZE]]
    segmentation_2d = fill_cells_2d(segmentation[z_2d], cells_2d, FIGURE_CLOSING_RADIUS_2D)

    # z is downsampled less, by the anisotropy, so the displayed voxels are isotropic.
    z_spacing, y_spacing, _ = EMBRYO_SCALE
    factors = (min(1.0, FIGURE_DOWNSAMPLE * z_spacing / y_spacing), FIGURE_DOWNSAMPLE, FIGURE_DOWNSAMPLE)
    scale = tuple(spacing / factor for spacing, factor in zip(EMBRYO_SCALE, factors))
    # A smaller dtype renders much faster in 3d than uint32.
    segmentation_ds = rescale(
        segmentation.astype("float32"), factors, order=0, anti_aliasing=False, channel_axis=None
    ).astype("uint8" if segmentation.max() < 256 else "uint16")
    n_z, n_y, n_x = segmentation_ds.shape
    raw_2d = rescale(raw_2d_full, FIGURE_DOWNSAMPLE, order=1, anti_aliasing=True, channel_axis=None)[:n_y, :n_x]
    seg_start = FIGURE_GAP + 1
    raw_volume = np.zeros((seg_start + n_z, n_y, n_x), dtype=raw_2d.dtype)
    raw_volume[0] = raw_2d

    viewer = napari.Viewer(title=f"{name} 3D {layer_name} (ds={factors})")
    contrast_limits = (float(raw_2d.min()), float(raw_2d.max()) + 1e-6)
    viewer.add_image(raw_volume, name="raw", scale=scale, contrast_limits=contrast_limits)
    layer = viewer.add_labels(np.pad(segmentation_ds, [(seg_start, 0), (0, 0), (0, 0)]), name=layer_name, scale=scale)
    layer.colormap = DirectLabelColormap(color_dict=make_color_dict(np.unique(segmentation_ds)[1:].tolist()))
    viewer.dims.ndisplay = 3
    show_scaled(viewer, ("z", "y", "x"))

    scale_2d = EMBRYO_SCALE[1:]
    contrast_limits = (float(raw_2d_full.min()), float(raw_2d_full.max()) + 1e-6)
    color_dict = make_color_dict(cells_2d)
    viewer = napari.Viewer(title=f"{name} 2D {layer_name} z={z_2d}")
    viewer.add_image(raw_2d_full, name="raw", scale=scale_2d, contrast_limits=contrast_limits)
    fill_layer = viewer.add_labels(segmentation_2d, name=layer_name, scale=scale_2d, opacity=0.5)
    fill_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer = viewer.add_labels(segmentation_2d, name=f"{layer_name} border", scale=scale_2d, opacity=1.0)
    border_layer.colormap = DirectLabelColormap(color_dict=color_dict)
    border_layer.contour = FIGURE_BORDER_WIDTH
    show_scaled(viewer, ("y", "x"))


def run_embryo(args, parser):
    if (args.iterative_prompting or args.ais or args.apg or args.figure) and args.embryo != EMBRYO_GT_VOLUME:
        parser.error(f"Only '{EMBRYO_GT_VOLUME}' has ground truth for iterative prompting, AIS, APG and the figure.")
    raw, name = load_embryo(args.embryo)
    print(f"Loaded {name} with shape {raw.shape} and dtype {raw.dtype}")
    embedding_path = os.path.join(args.embedding_root, f"{name}_{args.model_type}.zarr")
    if args.cellpose:
        run_embryo_cellpose(raw, args.embryo, args.cellpose)
    elif args.iterative_prompting:
        run_embryo_iterative_prompting(raw, name, args.model_type, args.prompt)
    elif args.ais or args.apg:
        for mode in [mode for mode, flag in (("ais", args.ais), ("apg", args.apg)) if flag]:
            run_embryo_automatic(raw, name, args.model_type, embedding_path, mode)
    elif args.figure:
        view_embryo_figure(raw, name, args.figure)
    elif args.view:
        view_embryo(raw, name)
    else:
        if args.slice_norm:
            raw = normalize_per_slice(raw)
            embedding_path = os.path.join(args.embedding_root, f"{name}_slice_norm_{args.model_type}.zarr")
        if args.precompute:
            precompute_embryo_embeddings(raw, embedding_path, args.model_type)
        else:
            run_embryo_annotator(raw, embedding_path, args.model_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--start", nargs=3, type=int, default=BLOCK_START, help="The ZYX start of the block.")
    parser.add_argument("--shape", nargs=3, type=int, default=BLOCK_SHAPE, help="The ZYX shape of the block.")
    parser.add_argument("--downsample", type=int, default=1, help="The factor to average the block by in zyx.")
    parser.add_argument("--min_size", type=int, default=AIS_MIN_SIZE, help="The minimum AIS object size.")
    parser.add_argument("--full", action="store_true", help="Segment the whole volume instead of the block.")
    parser.add_argument("--preview_factor", type=int, default=PREVIEW_FACTOR,
                        help="The extra downsampling of the whole volume preview for a laptop.")
    parser.add_argument("--slab", type=int, help="Segment this z-slab of the whole volume with AIS.")
    parser.add_argument("--stitch", action="store_true", help="Stitch the z-slabs of the whole volume.")
    parser.add_argument("--z_range", nargs=2, type=int, help="Restrict the slabs to this downsampled z range.")
    parser.add_argument("--slab_size", type=int, default=SLAB_SIZE, help="The core slices per slab.")
    parser.add_argument("--n_slabs", action="store_true", help="Print the number of slabs and exit.")
    parser.add_argument("--cache_embeddings", action="store_true", help="Keep the slab embeddings on vast.")
    parser.add_argument("--view", action="store_true", help="Show the block and its segmentations in napari.")
    parser.add_argument("-d", "--dataset", choices=tuple(SMALL_DATA_PATHS),
                        help="Segment one of the smaller volumes with the settings '-s' instead of the block.")
    parser.add_argument("-s", "--settings", nargs="+", help="The names of the SMALL_SETTINGS, by default all.")
    parser.add_argument("--sweep", choices=tuple(SMALL_DATA_PATHS),
                        help="Run every setting on one of the smaller volumes, on the crops for 'exm'.")
    parser.add_argument("--crop", nargs="+", choices=tuple(EXM_CROPS),
                        help="The exm crops to sweep, or the one to view.")
    parser.add_argument("--tag", help="A tag for the name of the output file of the smaller volumes.")
    parser.add_argument("--embryo", choices=tuple(EMBRYO_NAMES), help="Open this early embryo in the annotator.")
    parser.add_argument("-m", "--model_type", default=EMBRYO_MODEL_TYPE, help="The model for the early embryos.")
    parser.add_argument("--embedding_root", default=EMBRYO_OUTPUT_ROOT, help="The folder of the embryo embeddings.")
    parser.add_argument("--precompute", action="store_true", help="Only precompute the embeddings of the embryo.")
    parser.add_argument("--slice_norm", action="store_true", help="Normalize each embryo slice instead of the volume.")
    parser.add_argument("--iterative_prompting", action="store_true", help="Prompt the embryo from the ground truth.")
    parser.add_argument("-p", "--prompt", default="box", choices=["box", "point"], help="The first prompt per object.")
    parser.add_argument("--ais", action="store_true", help="Segment the embryo automatically with AIS.")
    parser.add_argument("--apg", action="store_true", help="Segment the embryo automatically with APG.")
    parser.add_argument("--figure", choices=tuple(EMBRYO_FIGURES), help="Show the figure of this embryo prediction.")
    parser.add_argument("--cellpose", nargs="+", choices=CELLPOSE_MODES, help="Segment the embryo with cellposeSAM.")
    args = parser.parse_args()

    roi = get_roi(args.start, args.shape)
    if args.embryo is not None:
        run_embryo(args, parser)
    elif args.sweep is not None or args.dataset is not None:
        dataset = args.sweep or args.dataset
        if args.view:
            view_small(dataset, None if args.crop is None else args.crop[0], args.tag)
        else:
            names = args.settings or list(SMALL_SETTINGS[dataset])
            run_small(dataset, names, args.engines, args.sweep is not None, args.crop, args.tag)
    elif args.n_slabs:
        print(len(get_slabs(get_z_range(args.downsample, args.z_range), args.slab_size)))
    elif args.slab is not None:
        run_slab(args.slab, args.downsample, args.min_size, args.z_range, args.slab_size, args.cache_embeddings)
    elif args.stitch:
        run_stitch(args.downsample, args.min_size, args.preview_factor, args.z_range, args.slab_size)
    elif args.full:
        run_full(args.engines, args.downsample, args.min_size, args.preview_factor)
    elif args.view:
        view(roi, args.downsample)
    else:
        run(roi, args.engines, args.downsample, args.min_size)


if __name__ == "__main__":
    main()
