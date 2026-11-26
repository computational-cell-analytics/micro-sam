"""Explore the volume EM spheroid of the Beghin lab (PF_Dish_C_pin3new_Spheroid3, 869 x 3068 x 2864 uint8).

convert: write the tif as an OME-Zarr (zarr v3, OME-NGFF 0.5) with sharded, chunked and compressed arrays and a mean
downsampled pyramid s0, s1, ... The layout follows the stitched Expanded_Platynereis.ome.zarr of the Huisken data.
The tif has no voxel size, the OME-Zarr gets VOXEL_SIZE. The first downsampling step is in y and x only, since z is
3x coarser than y and x at s0, the later steps are in z, y and x (DOWNSCALE_FACTORS).

rebuild_pyramid: rewrite s1, s2, ... of the raw and of every label image from their s0 with DOWNSCALE_FACTORS, and
the OME metadata with VOXEL_SIZE.

segment_crops: segment 768 x 768 crops of one scale level in 2d with micro-sam2 (cells and nuclei, AIS and APG), save
one h5 file per crop and plot the objects of all crops.

predictions: save the decoder outputs (foreground, directed distances, boundary) of both models on the crops of one
scale level and plot them.

segment_3d: segment the cells of the whole s2 volume and the nuclei of an s0 block in 3d, save one h5 file per run and
plot orthogonal views.

size_gallery: plot the size histogram of the objects of a 3d run and every object below a size, to choose the minimum
size of the nuclei.

merge_test: merge the objects of a 3d run that are stacked in z (one nucleus cut into pieces along z), remove the
small objects, save the result next to the segmentation and plot every merge.

segment_full: segment the nuclei of the whole s0 volume in overlapping 3d blocks. Several workers can run at the same
time, each block is claimed with a lock file.

stitch_full: stitch the blocks, merge the z stacked objects, remove the small objects and add the result to the
OME-Zarr as 'labels/nuclei'.

clean_full: remove the flat, too wide and tile shaped objects of 'labels/nuclei' and add the result to the OME-Zarr
as 'labels/nuclei_clean'.

watershed_volume: segment the nuclei of the decoder outputs of predict_volume block-wise with bioimage-py: seeds are the
connected components of foreground - boundary > WS_SEED_THRESHOLD, grown back by a seeded watershed on the boundary
within foreground > WS_FOREGROUND_THRESHOLD, then the objects below WS_MIN_SIZE voxels are removed.

segment_inverted_crop: predict the decoder outputs and segment the nuclei with 3d AIS on the inverted raw (255 - raw,
dark membranes as in the training data) of a central crop of INVERTED_CROP_SHAPE at s1, and of the same region at s0.

segment_cell_crop: segment the cells of the inverted central crop of CELL_CROP_SHAPE at s1, and of the same region at
s0, in two ways: 3d AIS of the cells model, and the SAM2 video predictor of the cells model prompted from the nuclei of
watershed_volume, with the nucleus center as positive point, alone or with the centers of the CELL_N_NEGATIVES
nearest nuclei as negative points.

seed_cell_crop: segment the cells of the cell crop from the AIS predictions of segment_cell_crop with the nuclei as
seeds, in two ways: a watershed on the boundary map seeded with the expanded nuclei, and a graph watershed on an
over-segmentation, seeded with the fragments that are mostly nucleus.

watershed_cell_crop: the watershed of the expanded nuclei of seed_cell_crop (the chosen method, at s0), with the
foreground mask closed by CELL_CLOSING_RADIUS first, since the foreground misses the nuclear envelope.

segment_cells_volume: the same as watershed_cell_crop for a whole scale level (s0), block-wise with bioimage-py: the
expanded nuclei of watershed_volume (seeds), the closed foreground mask of the cell predictions of predict_volume
(mask) and the seeded watershed on their boundary prediction (segmentation).

downsample_segmentations: write the pyramids of the cell segmentation of segment_cells_volume (from s0) and of the
nuclei of watershed_volume (from s1) up to s3, block-wise, taking the first voxel of each block as the label pyramids
of the OME-Zarr do. View them with view_beghin_em_volume.py.

segment_nuclei_from_prompts: segment the nuclei that watershed_volume misses with the SAM2 video predictor of the
nuclei model at s1, each prompted with one point of a napari points layer, as a label volume of s1 with new ids.
Add its pyramid with 'downsample_segmentations --names nuclei_prompted'.
The candidates for the partial nuclei from prompts (reprompt_partial_nuclei.py, the label volume 'suggested' at s1)
get their pyramid with 'downsample_segmentations --names nuclei_reprompted'.

merge_prompted_nuclei: the nuclei of watershed_volume without the fragments of the prompted nuclei, plus the prompted
nuclei (with their ids). The prompted nuclei of NUCLEUS_SEED_ONLY_POINTS are not full nuclei, only seeds for their
cells; they are marked in the attributes. 'segment_cells_volume --nuclei merged' uses these nuclei.

join_nucleus_fragments: the nuclei that are still split into several objects in the merged nuclei
(NUCLEUS_FRAGMENT_GROUPS, each of them also splits its cell) become one object each: the nucleus is segmented again
with the video predictor, prompted with the centers of its fragments, and joined with the fragments under the lowest
of their ids. 'segment_cells_volume --nuclei joined' uses these nuclei.

check_border_seed: test the background seed of segment_cells_volume (the mask voxels on the Y and X faces of the
volume seed the resin around the spheroid, their region is set to zero) on slabs of s0, compare the block-wise version
to the watershed of the whole slab and both to a tissue mask from the texture of the raw.

export_segmentations: write the cells and their nuclei as one OME-Zarr with two multiscale label images on the levels
of the raw, next to the data (EXPORT_PATH). By default the updated nuclei and cells of update_partial_nuclei.py (the
joined nuclei with the partial nuclei segmented again from prompts and without small fragments, and their cells), with
the list of the objects to exclude from the analysis in the attributes.

export_contours: write one level (s2) of the inverted raw and the contours of the exported cells and nuclei as a
multichannel ImageJ hyperstack (tif) for Fiji, next to the export.

export_tifs: write one level (s0) of the exported cells and nuclei as label tifs for Fiji (ImageJ format), with the
list of the objects to exclude and a description, in one zip archive next to the export.

check_cells_volume: run the block-wise stages of segment_cells_volume on the inputs of the cell crop and compare them
to the result of watershed_cell_crop.

render: render candidate 3d views of 'labels/nuclei_clean' off screen with pyvista, in the studio style of Figure 1
(on a GPU node, e.g. with srun --overlap in a running job).

render_section: find an xy plane where every nucleus is segmented (2d micro-sam2 nuclei against the 3d labels), render
the nuclei in the studio with the plane and the nuclei of the plane standing on its raw, and plot them side by side.
"""
import os
import csv
import math
import time
import shutil
import zipfile
import argparse
import tempfile
from itertools import product
from concurrent.futures import ThreadPoolExecutor

import h5py
import tifffile
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

import zarr
from zarr.codecs import BloscCodec

from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter

DATA_DIR = "/mnt/vast-nhr/projects/cidas/cca/data/beghin_data/em_spheroids"
INPUT_PATH = os.path.join(DATA_DIR, "PF_Dish_C_pin3new_Spheroid3.tif")
OUTPUT_PATH = os.path.join(DATA_DIR, "PF_Dish_C_pin3new_Spheroid3.ome.zarr")
CROP_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids/crops768"

SHARD_SHAPE = (256, 512, 512)
CHUNK_SHAPE = (64, 128, 128)

# The voxel size of s0 (z, y, x in nanometer). The co-authors give 16.6 nm pixels and 100 nm sections. The tif is
# binned 2x in y and x after the acquisition: at 16.6 nm the cells would be ~6 um and the spheroid twice as deep as
# wide, at 33.2 nm the cells are ~11 um and the spheroid is round (nuclei and spheroid extent at s0).
VOXEL_SIZE = (100.0, 33.2, 33.2)
# The ZYX downsampling factor from each level to the next (s0 -> s1, s1 -> s2, ...). The first step brings y and x
# to 66.4 nm, the closest to the 100 nm of z, the later steps keep the ratio.
DOWNSCALE_FACTORS = ((1, 2, 2), (2, 2, 2), (2, 2, 2))


def get_level_factor(level):
    """The ZYX downsampling factor of a scale level relative to s0."""
    factor = np.ones(3, dtype=int)
    for step in DOWNSCALE_FACTORS[:level]:
        factor *= step
    return tuple(int(f) for f in factor)


def get_level_voxel_size(level):
    """The ZYX voxel size of a scale level in nanometer."""
    return tuple(vs * f for vs, f in zip(VOXEL_SIZE, get_level_factor(level)))


# ZYX start of 768 x 768 crops of s0: center, interior and border of the spheroid at z 250, 434 and 620. The crops of
# a lower scale level have the same center and size, so they hold the same cells and their neighbours.
CROP_SIZE = 768
CROPS = [(250, 1000, 800), (250, 1600, 1600), (434, 700, 1250), (434, 1450, 600), (434, 2000, 1800), (620, 1250, 1150)]
# The cells model trains on LM and EM, including the FaDu tumor spheroid SBF-SEM cells at 50 nm. The nucleus model
# trains on the FIB-SEM human organoids and the same tumor spheroid.
MODELS = {"cells": "hvit_l_cells", "nuclei": "hvit_s_em_nuclei"}
ENGINES = ("ais", "apg")
SEGMENT_3D_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/beghin_data/em_spheroids/segmentation_3d"
# The 3d runs: the scale level, the ZYX bounding box (None for the whole volume), the target and the engines. The cell
# borders are clearest at s2, the nuclear envelopes at s0. The s0 block is at the center of the spheroid.
RUNS_3D = {
    "cells_s2": {"level": 2, "bb": None, "target": "cells", "engines": ("ais", "apg")},
    "nuclei_s0_block": {
        "level": 0, "bb": (slice(370, 498), slice(1000, 2024), slice(900, 1924)), "target": "nuclei",
        "engines": ("apg",),
    },
    # A thicker block with ~5 layers of nuclei in z (a nucleus spans ~70 slices of s0).
    "nuclei_s0_block_z384": {
        "level": 0, "bb": (slice(242, 626), slice(1000, 2024), slice(900, 1924)), "target": "nuclei",
        "engines": ("apg",),
    },
    # The same block with z upsampled 3x, since the z spacing is ~3 - 4x the xy spacing.
    "nuclei_s0_block_z384_zx3": {
        "level": 0, "bb": (slice(242, 626), slice(1000, 2024), slice(900, 1924)), "target": "nuclei",
        "engines": ("apg",), "z_upsample": 3,
    },
}
# The nuclei of the whole s0 volume: 3d APG in blocks of the size of nuclei_s0_block_z384. The overlap is larger than
# the radius of a nucleus (~35 slices in z, ~140 px in y and x), so an object with its center in the core of a block
# lies fully in the block.
FULL_BLOCK_SHAPE = (384, 1024, 1024)
FULL_OVERLAP = (96, 320, 320)
FULL_BLOCK_DIR = os.path.join(SEGMENT_3D_DIR, "nuclei_s0_full_blocks")
# The decoder outputs of a whole scale level (predict_volume): the tiling of the 3d s0 run (the library default for
# blocks above 768 px), predicted in z slabs of VOLUME_SLAB_SIZE slices.
VOLUME_TILE_SHAPE = (4, 512, 512)
VOLUME_HALO = (2, 128, 128)
VOLUME_SLAB_SIZE = 128
# 'em_spheroids' is not writable for every user of the group, the volume predictions get their own folder.
VOLUME_PREDICTION_DIR = os.path.join(
    os.path.dirname(os.path.dirname(SEGMENT_3D_DIR)), "em_spheroids_volume_predictions"
)
# The block-wise watershed of the volume predictions (watershed_volume). A halo of (16, 64, 64) with blocks of
# (64, 256, 256) gives the same result as one whole-array watershed on a 256 x 1024 x 1024 crop of s1.
WS_SEED_THRESHOLD = 0.5
WS_FOREGROUND_THRESHOLD = 0.5
WS_BLOCK_SHAPE = (64, 256, 256)
WS_HALO = (16, 64, 64)
# Most seeds of foreground - boundary are small fragments (~500 of ~800 objects of s1 below 1000 voxels), a nucleus
# at s1 has ~2e5 voxels.
WS_MIN_SIZE = 2500
# The central crop for the test on the inverted raw (segment_inverted_crop), ZYX in voxels of s1.
INVERTED_CROP_SHAPE = (256, 512, 512)
INVERTED_CROP_DIR = os.path.join(VOLUME_PREDICTION_DIR, "inverted_crop")
# The central crop for the comparison of the cell segmentation methods (segment_cell_crop), ZYX in voxels of s1. The
# cells model also gives a much cleaner foreground on the inverted raw, so all cell runs see 255 - raw.
CELL_CROP_SHAPE = (128, 512, 512)
CELL_CROP_DIR = os.path.join(VOLUME_PREDICTION_DIR, "cell_crop")
# The prompts derived from the nuclei: 'positive' prompts each cell with the center of its nucleus, 'negatives' adds
# the centers of the CELL_N_NEGATIVES nearest nuclei (in nanometer) as negative points.
CELL_N_NEGATIVES = 6
CELL_PROMPT_MODES = {"positive": 0, "negatives": CELL_N_NEGATIVES}
# The cell segmentation seeded from the nuclei (seed_cell_crop). The boundary prediction of the cells model also marks
# the nuclear envelope: at s1 its ridge spans from ~200 nm inside to ~300 nm outside the nucleus surface, so the
# nuclei are expanded by CELL_EXPAND_DISTANCE before they seed the watershed on the boundary map.
CELL_FOREGROUND_THRESHOLD = 0.5
CELL_EXPAND_DISTANCE = 300.0
# The foreground of the cells model misses the nuclear envelope: at s0 only ~40 - 70% of the voxels within 100 nm of
# the nucleus surface are foreground, against > 97% inside the nuclei and ~85% 150 - 400 nm outside. The background
# there is thin (half thickness median 66 nm, 90th percentile 120 nm) and the background far from the nuclei thick
# (median 232 nm). A closing with a ball of 150 nm removes ~80% of the background within -300 to 400 nm of the
# nucleus surface (19.6% -> 4.2%) and ~8% of the background more than 1.5 um from it (33.7% -> 30.9%); larger
# radii remove little more at the envelope (300 nm: 2.8%) and more of the rest.
CELL_CLOSING_RADIUS = 150.0
# The block-wise cell segmentation of a whole level (segment_cells_volume). The expansion and the closing are exact
# with a halo of their distance (the closing: twice its radius) plus one voxel. The watershed is exact only where the
# halo reaches the seeds of the cells in a block, a cell spans ~110 slices and ~330 px of s0. On the cell crop of s0
# (check_cells_volume, blocks of (32, 256, 256)) a halo of (8, 64, 64) assigns 3.9% of the mask differently than
# the whole-array watershed, (16, 128, 128) 1.7%, (32, 256, 256) 0.7% (whole regions, cut off from their seed), and
# (64, 384, 384) nothing. The crop has only 128 slices, so the z halo is barely tested there. (32, 256, 256) is the
# chosen trade-off. With it, a block of (128, 1024, 1024) reads ~3.4x its volume. The outputs have smaller chunks,
# the block shape is a multiple of them, so every block writes whole chunks.
CELL_VOLUME_BLOCK_SHAPE = (128, 1024, 1024)
CELL_VOLUME_CHUNK_SHAPE = (64, 256, 256)
CELL_VOLUME_WS_HALO = (32, 256, 256)
# The seed for the empty space around the spheroid (stage 'background' of segment_cells_blockwise). The cells model
# predicts most of the resin as foreground with a low boundary, so without it the watershed gives the resin to
# whichever cell reaches it. No cell touches the Y and X faces of the volume, so the mask voxels of these faces seed
# the resin; the seed gets CELL_BACKGROUND_SEED_ID in the watershed and its region is set to zero afterwards.
# The block-wise watershed only reaches a seed within its halo, and the blocks in the center of the volume do not see
# the faces (on the last 128 slices of s0, 22% of the region of the face seed went to a cell again). So the watershed
# is repeated with the region of the face seed of the previous pass as seed, which moves it one block inwards per
# pass: with two passes for the 3 x 3 blocks of s0 in Y and X, less than 0.001% of the mask differs from the
# watershed of a whole slab. (The region of the face seed in a watershed of the whole volume at a coarse level, eroded
# by 1 um, as seed instead: 0.2% of the mask differed on the central slab, it cut 170 um^3 off a cell.)
CELL_BACKGROUND_SEED_ID = 2 ** 32 - 1
# The check of the background seed (check_border_seed): slabs of s0 over the full Y and X extent, and a tissue mask
# from the texture of the raw at s2, which does not depend on a model: the local standard deviation in a window of
# TISSUE_STD_SIZE px (in the plane), thresholded with Otsu, closed, filled and opened (the opening removes the
# seams of the acquisition tiles).
BORDER_SEED_CHECK_LEVEL = 0
TISSUE_LEVEL = 2
TISSUE_STD_SIZE = 9
TISSUE_CLOSING_RADIUS = 3
TISSUE_OPENING_RADIUS = 7
TISSUE_SURFACE_MARGIN = 8
# The pyramids of the volume segmentations (downsample_segmentations): '<name>/s<level>' for the levels above the one
# a segmentation was computed at. The segmentations themselves stay where they are (SEGMENTATION_SOURCES).
SEGMENTATION_PYRAMID_PATH = os.path.join(VOLUME_PREDICTION_DIR, "segmentation_pyramids.zarr")
# The export of the cells and nuclei (export_segmentations), at the top level of the data folder of the Beghin lab.
EXPORT_PATH = os.path.join(os.path.dirname(DATA_DIR), "PF_Dish_C_pin3new_Spheroid3_segmentation.ome.zarr")
# The label tifs for Fiji (export_tifs): the level and the level that gives the display range (the largest id).
TIF_LEVEL = 0
TIF_RANGE_LEVEL = 2
# The hyperstack for Fiji (export_contours): its level and the channels with their colors (RGB).
CONTOUR_LEVEL = 2
CONTOUR_CHANNELS = {"EM (inverted)": (255, 255, 255), "cell contours": (255, 0, 255), "nucleus contours": (0, 255, 0)}
SEGMENTATION_PYRAMID_CHUNKS = (64, 256, 256)
# The candidates for the partial nuclei from prompts (reprompt_partial_nuclei.py): the label volume 'suggested' at s1
# with the suggested candidate of each partial nucleus under the id of the nucleus.
REPROMPTED_NUCLEI_PATH = os.path.join(VOLUME_PREDICTION_DIR, "partial_nuclei", "reprompted.zarr")
# The nuclei the decoder misses, segmented from one point each (segment_nuclei_from_prompts). The points are a napari
# points layer saved as csv (index, z, y, x in voxels of NUCLEUS_POINTS_LEVEL of the raw). Each nucleus is segmented in
# a crop of NUCLEUS_PROMPT_CROP_SHAPE of s1 around its point (a nucleus spans up to ~110 slices and ~170 px of s1).
# An existing nucleus with more than NUCLEUS_FRAGMENT_FRACTION of its voxels in a prompted nucleus is a fragment of it.
NUCLEUS_POINTS_PATH = os.path.join(VOLUME_PREDICTION_DIR, "nuclei_s3_points.csv")
NUCLEUS_POINTS_LEVEL = 3
NUCLEUS_PROMPT_CROP_SHAPE = (192, 512, 512)
NUCLEUS_FRAGMENT_FRACTION = 0.5
# The points (rows of the points file) without a full nucleus: rounded cells without a clear nucleus in the raw, where
# the prompted mask is only a part of the cell interior and changes with the prompts. They are merged into the nuclei
# (merge_prompted_nuclei) as seeds for their cells, but must not be used as nucleus segmentations.
NUCLEUS_SEED_ONLY_POINTS = (1, 2)
# The nuclei that are split into several objects in the merged nuclei, found by the user in the cells from them
# (2026-11-25): the ids of the fragments of each nucleus. join_nucleus_fragments makes one object of each group.
NUCLEUS_FRAGMENT_GROUPS = ((169, 170, 196), (192, 240), (215, 216, 241), (64, 65))
# The fragments that are not used as a prompt, with the reason. They are still joined with their group.
NUCLEUS_FRAGMENT_NO_PROMPT = {
    169: (
        "Not a part of the nucleus: the cap of the cell above the nucleus (slices 446 - 456 of s1, the nucleus starts "
        "at 461). With its center as a prompt the mask covers the cytoplasm of the cap. Only a seed for the cell."
    ),
}
NUCLEUS_SEED_ONLY_COMMENT = (
    "Not a full nucleus: a rounded cell without a clear nucleus in the raw, the mask is a part of the cell interior "
    "and changes with the prompts. Only a seed for the cell."
)
# The over-segmentation of the graph watershed: 3d AIS with half the default density threshold (more seeds) and the
# boundary prediction added to the height map (boundary_weight), so that the fragments end on the predicted
# boundaries, including the nuclear envelope. min_size is in voxels of s1 and scaled to the voxel size of a level.
# At s1 no fragment holds more than 10% of two nuclei.
CELL_OVERSEGMENTATION = {"density_threshold": 40.0, "boundary_weight": 1.0, "min_size": 500}
# A fragment is a nucleus fragment if more than CELL_NUCLEUS_FRACTION of it belongs to one nucleus. The edges between
# a nucleus fragment and the fragments that are not one cross the nuclear envelope, which the boundary prediction
# marks as well, so their weight is multiplied by CELL_NUCLEUS_EDGE_FACTOR.
CELL_NUCLEUS_FRACTION = 0.5
CELL_NUCLEUS_EDGE_FACTOR = 0.5
MIN_OVERLAP = 0.5
MIN_SIZE = 3e4
# After stitching: a nucleus spans ~70 slices of s0 and ~300 px in y and x. Objects of at most MAX_FLAT_Z slices are
# caps of nuclei that APG followed for a few slices only, or background. Objects wider than MAX_YX_EXTENT or as wide
# as an APG tile (TILE_SIZE, within TILE_TOLERANCE), and objects that touch the y or x border of the volume (the
# spheroid does not), are artifacts in the background. Objects below MIN_CLEAN_SIZE voxels float around the spheroid.
MAX_FLAT_Z = 8
MAX_YX_EXTENT = 700
TILE_SIZE = 512
TILE_TOLERANCE = 12
MIN_CLEAN_SIZE = 1e5
# The 3d renders of 'labels/nuclei_clean', at s2. z is scaled by RENDER_Z_SCALE, the z spacing over the xy spacing of
# that level.
RENDER_LEVEL = 2
RENDER_Z_SCALE = get_level_voxel_size(RENDER_LEVEL)[0] / get_level_voxel_size(RENDER_LEVEL)[2]
RENDER_DIR = "renders"
FOR_PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3, "plotting", "v2", "for_paper")
PREDICTION_CHANNELS = ("foreground", "distance_z", "distance_y", "distance_x", "boundary")


def create_level(group, name, shape, dtype):
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    return group.create_array(
        name, shape=shape, dtype=dtype, shards=SHARD_SHAPE, chunks=CHUNK_SHAPE, compressors=compressor,
        fill_value=0, dimension_names=["z", "y", "x"],
    )


def get_multiscale(name, num_levels, kind):
    """The OME-NGFF multiscale metadata of a pyramid with VOXEL_SIZE and DOWNSCALE_FACTORS.

    kind is 'mean' for the raw, the mean of a block sits at the center of the block, or 'nearest' for the labels,
    which are taken at the first voxel of each block.
    """
    axes = [{"name": ax, "type": "space", "unit": "nanometer"} for ax in "zyx"]
    datasets = []
    for level in range(num_levels):
        transforms = [{"type": "scale", "scale": [round(float(vs), 6) for vs in get_level_voxel_size(level)]}]
        if kind == "mean":
            translation = [round(float(vs * (f - 1) / 2), 6) for vs, f in zip(VOXEL_SIZE, get_level_factor(level))]
            transforms.append({"type": "translation", "translation": translation})
        datasets.append({"path": f"s{level}", "coordinateTransformations": transforms})
    return {"name": name, "axes": axes, "datasets": datasets, "type": kind}


def write_ome_metadata(group, name, num_levels):
    group.attrs["ome"] = {"version": "0.5", "multiscales": [get_multiscale(name, num_levels, "mean")]}


def write_s0(input_path, array, num_workers):
    """Copy the tif into s0, one shard slab along z at a time, so that every shard is written once and complete."""
    n_slices = array.shape[0]
    for z0 in range(0, n_slices, SHARD_SHAPE[0]):
        z1 = min(z0 + SHARD_SHAPE[0], n_slices)
        slab = tifffile.imread(input_path, key=range(z0, z1), maxworkers=num_workers)
        array[z0:z1] = slab
        print(f"s0: wrote z {z0} - {z1} of {n_slices}")


def downsample_mean(block, factor):
    """Downsample a 3d block with the mean of each block of the ZYX factor. An uneven edge is padded with its last
    value."""
    pad = [(0, -s % f) for s, f in zip(block.shape, factor)]
    block = np.pad(block, pad, mode="edge")
    mean = np.zeros([s // f for s, f in zip(block.shape, factor)], dtype="float32")
    for oz, oy, ox in product(*[range(f) for f in factor]):
        mean += block[oz::factor[0], oy::factor[1], ox::factor[2]]
    mean /= np.prod(factor)
    return np.round(mean).astype(block.dtype)


def get_downsampled_shape(shape, factor):
    return tuple((s + f - 1) // f for s, f in zip(shape, factor))


def write_downsampled_level(source, target, factor, num_workers):
    """Write target from source, one target shard per task, so that no two tasks write the same shard."""
    grid = [range(0, s, sh) for s, sh in zip(target.shape, SHARD_SHAPE)]

    def write_shard(start):
        bb = tuple(slice(st, min(st + sh, s)) for st, sh, s in zip(start, SHARD_SHAPE, target.shape))
        source_bb = tuple(slice(f * b.start, min(f * b.stop, s)) for b, f, s in zip(bb, factor, source.shape))
        target[bb] = downsample_mean(source[source_bb], factor)

    with ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(write_shard, product(*grid)))


def write_raw_pyramid(group, num_workers):
    """Write s1, s2, ... of the raw from s0 with DOWNSCALE_FACTORS, replacing the levels that exist already."""
    source = group["s0"]
    for level, factor in enumerate(DOWNSCALE_FACTORS, 1):
        name = f"s{level}"
        if name in group:
            del group[name]
        target = create_level(group, name, get_downsampled_shape(source.shape, factor), source.dtype)
        write_downsampled_level(source, target, factor, num_workers)
        print(f"{name}: wrote shape {target.shape}, voxel size {get_level_voxel_size(level)} nm", flush=True)
        source = target
    level_names = {f"s{level}" for level in range(len(DOWNSCALE_FACTORS) + 1)}
    for name in list(group.array_keys()):
        if name not in level_names:
            print(f"Removed {name}, a level beyond DOWNSCALE_FACTORS")
            del group[name]
    write_ome_metadata(group, group.attrs["ome"]["multiscales"][0]["name"], len(DOWNSCALE_FACTORS) + 1)


def check_s0(input_path, array):
    """Compare the first, middle and last slice of s0 to the tif."""
    for z in (0, array.shape[0] // 2, array.shape[0] - 1):
        expected = tifffile.imread(input_path, key=z)
        assert np.array_equal(array[z], expected), f"s0 differs from the tif at z {z}"
    print("s0 matches the tif at the first, middle and last slice")


def convert_to_ome_zarr(input_path, output_path, num_workers):
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} exists already, delete it first to convert again.")
    zarr.config.set({"threading.max_workers": num_workers, "async.concurrency": 4 * num_workers})

    with tifffile.TiffFile(input_path) as f:
        shape = (len(f.pages),) + f.pages[0].shape
        dtype = f.pages[0].dtype
    print(f"Convert {input_path} with shape {shape} and dtype {dtype}")

    group = zarr.open_group(output_path, mode="w", zarr_format=3)
    name = os.path.basename(output_path).replace(".ome.zarr", "")
    write_ome_metadata(group, name, len(DOWNSCALE_FACTORS) + 1)

    source = create_level(group, "s0", shape, dtype)
    write_s0(input_path, source, num_workers)
    check_s0(input_path, source)
    write_raw_pyramid(group, num_workers)


def rebuild_pyramid(zarr_path, num_workers):
    """Rewrite the levels above s0 of the raw and of every label image, and their metadata."""
    zarr.config.set({"threading.max_workers": num_workers, "async.concurrency": 4 * num_workers})
    root = zarr.open_group(zarr_path, mode="a", zarr_format=3)
    t0 = time.time()
    write_raw_pyramid(root, num_workers)
    print(f"Raw pyramid: {time.time() - t0:.0f} s", flush=True)
    if "labels" in root:
        for name in root["labels"].attrs["ome"]["labels"]:
            t0 = time.time()
            write_label_pyramid(root["labels"][name], name)
            print(f"labels/{name} pyramid: {time.time() - t0:.0f} s", flush=True)


def segment_image(image, model_type, engine, ndim=2, tile_shape=None, halo=None):
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, segmentation_mode=engine, ndim=ndim, is_tiled=tile_shape is not None,
    )
    embedding_dir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    try:
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=image, embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
            ndim=ndim, batch_size=None, verbose=False, tile_shape=tile_shape, halo=halo,
        )
    finally:
        shutil.rmtree(embedding_dir, ignore_errors=True)
    return segmentation.astype("uint32")


def get_crop_name(start):
    return "z{}_y{}_x{}".format(*start)


def get_level_crops(level, shape):
    """Get the ZYX start of each crop at a scale level, shifted inside the volume where it would leave it.

    A crop is cut to the slice where the slice is smaller than the crop, and a crop that repeats another is dropped.
    """
    factor = get_level_factor(level)
    crops = []
    for start in CROPS:
        z = start[0] // factor[0]
        center = [(st + CROP_SIZE // 2) // f for st, f in zip(start[1:], factor[1:])]
        y, x = [min(max(c - CROP_SIZE // 2, 0), max(s - CROP_SIZE, 0)) for c, s in zip(center, shape[1:])]
        if (z, y, x) not in crops:
            crops.append((z, y, x))
    return crops


def segment_crops(zarr_path, output_dir, level):
    """Segment the 768 x 768 crops of a scale level in 2d with every model and engine, one h5 file per crop."""
    os.makedirs(output_dir, exist_ok=True)
    raw = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")
    for start in get_level_crops(level, raw.shape):
        z, y, x = start
        image = raw[z, y:y + CROP_SIZE, x:x + CROP_SIZE]
        path = os.path.join(output_dir, f"{get_crop_name(start)}.h5")
        with h5py.File(path, "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", data=image, chunks=True, compression="gzip")
            for target, model_type in MODELS.items():
                for engine in ENGINES:
                    key = f"segmentation/{target}_{engine}"
                    if key in f:
                        continue
                    t0 = time.time()
                    segmentation = segment_image(image, model_type, engine)
                    f.create_dataset(key, data=segmentation, chunks=True, compression="gzip")
                    n_objects = len(np.unique(segmentation)) - 1
                    print(f"{get_crop_name(start)} {key}: {n_objects} objects in {time.time() - t0:.1f} s")


def plot_crops(zarr_path, output_dir, level, plot_path):
    """Plot every crop with the objects of each model and engine as filled masks with outlines."""
    keys = [f"{target}_{engine}" for target in MODELS for engine in ENGINES]
    shape = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r").shape
    crops = get_level_crops(level, shape)
    fig, axes = plt.subplots(len(crops), len(keys) + 1, figsize=(4 * (len(keys) + 1), 4 * len(crops)), squeeze=False)
    rng = np.random.default_rng(0)
    for row, start in zip(axes, crops):
        with h5py.File(os.path.join(output_dir, f"{get_crop_name(start)}.h5"), "r") as f:
            image = f["raw"][:]
            row[0].imshow(image, cmap="gray")
            row[0].set_title(f"s{level} {get_crop_name(start)}")
            for ax, key in zip(row[1:], keys):
                segmentation = f[f"segmentation/{key}"][:]
                colors = rng.uniform(0.0, 1.0, size=(segmentation.max() + 1, 3))
                overlay = np.stack([image / 255.0] * 3, axis=-1)
                foreground = segmentation > 0
                overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * colors[segmentation[foreground]]
                boundaries = find_boundaries(segmentation, mode="thick")
                overlay[boundaries] = colors[segmentation[boundaries]]
                ax.imshow(overlay)
                ax.set_title(f"{key}: {len(np.unique(segmentation)) - 1} objects")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=100)
    print(f"Saved the plot to {plot_path}")


def predict_image(image, model_type):
    """Get the decoder outputs of a 2d image, shape (5, Y, X)."""
    _, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode="ais", ndim=2)
    segmenter.initialize(image, ndim=2)
    return segmenter.get_state()["prediction"].astype("float32")


def save_predictions(zarr_path, output_dir, level):
    """Save the decoder outputs of each model as 'prediction/<target>' in the h5 file of each crop."""
    os.makedirs(output_dir, exist_ok=True)
    raw = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")
    for start in get_level_crops(level, raw.shape):
        z, y, x = start
        image = raw[z, y:y + CROP_SIZE, x:x + CROP_SIZE]
        with h5py.File(os.path.join(output_dir, f"{get_crop_name(start)}.h5"), "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", data=image, chunks=True, compression="gzip")
            for target, model_type in MODELS.items():
                key = f"prediction/{target}"
                if key not in f:
                    f.create_dataset(key, data=predict_image(image, model_type), chunks=True, compression="gzip")
        print(f"s{level} {get_crop_name(start)}: saved the predictions")


def get_volume_prediction_path(level, target):
    return os.path.join(VOLUME_PREDICTION_DIR, f"{target}_s{level}_predictions.zarr")


def predict_volume(
    zarr_path, level, target, z_range=None, slab_size=VOLUME_SLAB_SIZE, batch_size=1, channels=PREDICTION_CHANNELS,
):
    """Save the 3d decoder outputs of a whole scale level, one z slab at a time, as one zarr array per channel.

    The decoder sees the inverted raw (255 - raw, dark membranes as in the training data), which detects many more
    nuclei than the raw itself (segment_inverted_crop). It runs as in the 3d segmentation (VOLUME_TILE_SHAPE with
    VOLUME_HALO). The volume is too large for the
    memory as float32, so it is predicted in slabs of slab_size slices. Each slab is read with the z halo of the tile
    grid on both sides and starts on the z grid, so its tiles are the tiles of the whole volume and the result is the
    same. A slab that is written is recorded in the attributes, so a run that stops can be continued. Only the
    channels are saved (e.g. foreground and boundary for the cells of s0, see segment_cells_volume).
    """
    raw = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")
    shape = raw.shape
    z_tile, z_halo = VOLUME_TILE_SHAPE[0], VOLUME_HALO[0]
    assert slab_size % z_tile == 0
    # The context of a slab: whole z tiles that cover the halo of its first and last tile.
    context = z_tile * ((z_halo + z_tile - 1) // z_tile)
    out = zarr.open_group(get_volume_prediction_path(level, target), mode="a", zarr_format=3)
    out.attrs.update({
        "model_type": MODELS[target], "scale_level": level, "voxel_size": list(get_level_voxel_size(level)),
        "tile_shape": list(VOLUME_TILE_SHAPE), "halo": list(VOLUME_HALO), "slab_size": slab_size, "inverted": True,
    })
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    arrays = {
        name: out.require_array(
            name, shape=shape, dtype="float16", shards=(slab_size, 512, 512),
            chunks=(math.gcd(32, slab_size), 128, 128),
            compressors=compressor, fill_value=0, dimension_names=["z", "y", "x"],
        ) for name in channels
    }
    _, segmenter = get_predictor_and_segmenter(
        model_type=MODELS[target], segmentation_mode="ais", ndim=3, is_tiled=True,
    )
    z_begin, z_end = (0, shape[0]) if z_range is None else z_range
    assert z_begin % slab_size == 0
    done = set(out.attrs.get("done_slabs", []))
    for z0 in range(z_begin, z_end, slab_size):
        z1 = min(z0 + slab_size, shape[0])
        if z0 in done:
            continue
        t0 = time.time()
        r0, r1 = max(z0 - context, 0), min(z1 + context, shape[0])
        segmenter.initialize(
            255 - raw[r0:r1], ndim=3, tile_shape=VOLUME_TILE_SHAPE, halo=VOLUME_HALO, batch_size=batch_size,
        )
        prediction = segmenter.get_state()["prediction"][:, z0 - r0:z1 - r0]
        t_predict = time.time() - t0
        for name, channel in zip(PREDICTION_CHANNELS, prediction):
            if name in arrays:
                arrays[name][z0:z1] = channel.astype("float16")
        segmenter.clear_state()
        done.add(z0)
        out.attrs["done_slabs"] = sorted(done)
        print(
            f"s{level} {target}: z {z0} - {z1} of {shape[0]}, {r1 - r0} slices predicted in {t_predict:.0f} s, "
            f"written in {time.time() - t0 - t_predict:.0f} s", flush=True,
        )


def get_watershed_path(level, target):
    return os.path.join(VOLUME_PREDICTION_DIR, f"{target}_s{level}_watershed.zarr")


def _subtract(foreground, boundary):
    return foreground.astype("float32") - boundary.astype("float32")


def watershed_volume(
    level, target, num_workers, block_shape=WS_BLOCK_SHAPE, halo=WS_HALO, min_size=WS_MIN_SIZE, filter_only=False,
):
    """Segment the decoder outputs of a whole scale level with a seeded watershed, block-wise (bioimage-py).

    The seeds are the connected components of foreground - boundary > WS_SEED_THRESHOLD (the subtraction and the
    threshold are computed on read). The watershed grows them on the boundary map within the foreground mask, which
    gives back what the subtraction of the boundary took away. The objects below min_size voxels are removed and the
    rest relabeled consecutively. The seeds, the watershed and the filtered segmentation are written as uint64 zarr
    arrays ('seeds', 'watershed', 'segmentation') with chunks of block_shape, so every block writes whole chunks. With
    filter_only, only the size filter runs again on the saved watershed.
    """
    import bioimage_py as bp

    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")

    def create(group, name):
        if name in group:
            del group[name]
        return group.create_array(
            name, shape=shape, dtype="uint64", chunks=block_shape, compressors=compressor, fill_value=0,
            dimension_names=["z", "y", "x"],
        )

    shape = zarr.open_array(os.path.join(get_volume_prediction_path(level, target), "foreground"), mode="r").shape
    if not filter_only:
        _seeded_watershed(level, target, num_workers, block_shape, halo, create)
    out = zarr.open_group(get_watershed_path(level, target), mode="a")
    t0 = time.time()
    bp.segmentation.size_filter(
        out["watershed"], create(out, "segmentation"), min_size=min_size, block_shape=block_shape,
        num_workers=num_workers,
    )
    out.attrs["min_size"] = min_size
    print(f"s{level} {target}: size filter in {time.time() - t0:.0f} s", flush=True)


def _seeded_watershed(level, target, num_workers, block_shape, halo, create):
    import bioimage_py as bp
    from bioimage_py.wrapper import MultiTransformationSource, ThresholdSource

    predictions = zarr.open_group(get_volume_prediction_path(level, target), mode="r")
    foreground, boundary = predictions["foreground"], predictions["boundary"]
    seed_mask = ThresholdSource(
        MultiTransformationSource(_subtract, foreground, boundary, dtype="float32"), WS_SEED_THRESHOLD,
    )
    mask = ThresholdSource(foreground, WS_FOREGROUND_THRESHOLD)

    out = zarr.open_group(get_watershed_path(level, target), mode="w", zarr_format=3)
    out.attrs.update({
        "predictions": get_volume_prediction_path(level, target), "seed_threshold": WS_SEED_THRESHOLD,
        "foreground_threshold": WS_FOREGROUND_THRESHOLD, "block_shape": list(block_shape), "halo": list(halo),
        "voxel_size": list(get_level_voxel_size(level)),
    })
    arrays = {name: create(out, name) for name in ("seeds", "watershed")}
    t0 = time.time()
    bp.segmentation.label(seed_mask, arrays["seeds"], block_shape=block_shape, num_workers=num_workers)
    t_label = time.time() - t0
    bp.segmentation.watershed(
        boundary, arrays["seeds"], arrays["watershed"], halo=halo, mask=mask, block_shape=block_shape,
        num_workers=num_workers,
    )
    print(f"s{level} {target}: seeds in {t_label:.0f} s, watershed in {time.time() - t0 - t_label:.0f} s", flush=True)


def get_inverted_crop(level, shape_s1, crop_shape=INVERTED_CROP_SHAPE):
    """The bounding box of the central crop of crop_shape (voxels of s1) at a scale level up to s1."""
    starts = [(sh - cs) // 2 for sh, cs in zip(shape_s1, crop_shape)]
    factor = [f1 // f for f1, f in zip(get_level_factor(1), get_level_factor(level))]
    return tuple(slice(st * f, (st + cs) * f) for st, cs, f in zip(starts, crop_shape, factor))


def get_inverted_crop_path(level, target="nuclei"):
    return os.path.join(INVERTED_CROP_DIR, f"{target}_s{level}.h5")


def segment_ais_3d(raw, target):
    """The decoder outputs and the 3d AIS segmentation of a volume, with the tiling of predict_volume.

    Returns the prediction (5, Z, Y, X), the segmentation and the times of the prediction and the segmentation.
    """
    t0 = time.time()
    _, segmenter = get_predictor_and_segmenter(
        model_type=MODELS[target], segmentation_mode="ais", ndim=3, is_tiled=True,
    )
    segmenter.initialize(raw, ndim=3, tile_shape=VOLUME_TILE_SHAPE, halo=VOLUME_HALO, batch_size=1)
    prediction = segmenter.get_state()["prediction"]
    t_predict = time.time() - t0
    segmentation = segmenter.generate().astype("uint32")
    t_segment = time.time() - t0 - t_predict
    del segmenter
    return prediction, segmentation, t_predict, t_segment


def segment_inverted_crop(zarr_path, levels, target="nuclei"):
    """Save the decoder outputs and the 3d AIS segmentation of the inverted central crop of each level.

    The decoder runs with the tiling and the normalization of predict_volume, so the inversion is the only
    difference to the predictions of the whole volume.
    """
    os.makedirs(INVERTED_CROP_DIR, exist_ok=True)
    shape_s1 = zarr.open_array(os.path.join(zarr_path, "s1"), mode="r").shape
    for level in levels:
        path = get_inverted_crop_path(level, target)
        if os.path.exists(path):
            print(f"{path} exists, skipping s{level}")
            continue
        bb = get_inverted_crop(level, shape_s1)
        raw = 255 - zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")[bb]
        prediction, segmentation, t_predict, t_segment = segment_ais_3d(raw, target)
        with h5py.File(path + ".tmp", "w") as f:
            f.attrs.update({
                "level": level, "bb_start": [b.start for b in bb], "bb_stop": [b.stop for b in bb],
                "voxel_size": get_level_voxel_size(level), "model_type": MODELS[target], "inverted": True,
                "tile_shape": VOLUME_TILE_SHAPE, "halo": VOLUME_HALO,
                "time_predict": t_predict, "time_segment": t_segment,
            })
            f.create_dataset("raw", data=raw, chunks=True, compression="gzip")
            for name, channel in zip(PREDICTION_CHANNELS, prediction):
                f.create_dataset(f"prediction/{name}", data=channel.astype("float16"), chunks=True, compression="gzip")
            f.create_dataset("segmentation", data=segmentation, chunks=True, compression="gzip")
        os.rename(path + ".tmp", path)
        n_objects = len(np.unique(segmentation)) - 1
        print(
            f"s{level} {target} inverted crop {raw.shape}: predicted in {t_predict:.0f} s, {n_objects} objects "
            f"segmented in {t_segment:.0f} s", flush=True,
        )
        del prediction


def get_cell_crop_path(level, method):
    """The result of a cell segmentation method ('ais' or 'prompts') on the cell crop of a level."""
    return os.path.join(CELL_CROP_DIR, f"cells_{method}_s{level}.h5")


def get_nucleus_prompts_path():
    return os.path.join(CELL_CROP_DIR, "nucleus_prompts.h5")


def get_nucleus_prompts(bb, n_negatives=CELL_N_NEGATIVES, min_size=WS_MIN_SIZE):
    """Derive the cell prompts from the nuclei of watershed_volume in a crop of s1.

    The nuclei are the parts of the watershed nuclei inside the crop of at least min_size voxels, so a nucleus that
    the crop cuts is prompted on its part in the crop. The center of a nucleus is the voxel of it closest to its
    centroid (in nanometer); the centroid itself where it lies inside the nucleus. The prompts of a nucleus are on
    its center slice, which anchors the propagation: the center as positive point and the centers of the n_negatives
    nearest nuclei (3d distance in nanometer), projected onto that slice, as negative points. A projected center that
    falls into the nucleus itself (a neighbor above or below it) is dropped.

    Returns the nuclei (relabeled consecutively), the centers (N, 3) in voxels of s1, and the negative points
    (M, 4) as object id, z, y and x.
    """
    from scipy.spatial import cKDTree
    from skimage.segmentation import relabel_sequential

    nuclei = zarr.open_group(get_watershed_path(1, "nuclei"), mode="r")["segmentation"][bb]
    ids, sizes = np.unique(nuclei, return_counts=True)
    nuclei[~np.isin(nuclei, ids[(ids > 0) & (sizes >= min_size)])] = 0
    nuclei = relabel_sequential(nuclei)[0].astype("uint32")

    voxel_size = np.array(get_level_voxel_size(1))
    centers = []
    for nucleus_id, bb_nucleus in enumerate(ndimage.find_objects(nuclei), 1):
        coords = np.argwhere(nuclei[bb_nucleus] == nucleus_id) + [b.start for b in bb_nucleus]
        distances = (((coords - coords.mean(axis=0)) * voxel_size) ** 2).sum(axis=1)
        centers.append(coords[np.argmin(distances)])
    centers = np.array(centers)

    n_neighbors = min(n_negatives, len(centers) - 1)
    negatives, n_dropped = [], 0
    if n_neighbors > 0:
        _, neighbors = cKDTree(centers * voxel_size).query(centers * voxel_size, k=n_neighbors + 1)
        for nucleus_id, (center, row) in enumerate(zip(centers, neighbors), 1):
            for neighbor in row[1:]:
                y, x = centers[neighbor, 1:]
                if nuclei[center[0], y, x] == nucleus_id:
                    n_dropped += 1
                    continue
                negatives.append((nucleus_id, center[0], y, x))
    negatives = np.array(negatives, dtype="int64").reshape(-1, 4)
    print(
        f"{len(centers)} nuclei in the crop, {len(negatives)} negative points, {n_dropped} dropped in the own nucleus",
        flush=True,
    )
    return nuclei, centers, negatives


def segment_cells_from_prompts(volume, centers, negatives, modes=CELL_PROMPT_MODES):
    """Segment one cell per nucleus with the SAM2 video predictor of the cells model, for each prompt mode.

    Every cell is conditioned on the center slice of its nucleus with all its points in one call, and propagated
    through the whole volume in both directions (no early stop). A single positive point gets the multimask output of
    SAM2 (the mask with the best predicted IoU), several points one mask. The cells overlap, so they are painted
    from the largest to the smallest: a smaller cell stays visible where a larger one covers it as well.

    Args:
        volume: The inverted raw, (Z, Y, X).
        centers: The nucleus centers (N, 3) in voxels of the volume.
        negatives: The negative points (M, 4) as object id, z, y and x in voxels of the volume.
        modes: The prompt modes, the name and whether it uses the negative points.

    Returns:
        Per mode, the segmentation (object id = nucleus id), the size of each object and the fraction of it that
        is visible after painting, and the time of the propagation.
    """
    from micro_sam.v2.util import precompute_image_embeddings
    from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D

    t0 = time.time()
    predictor, _ = get_predictor_and_segmenter(model_type=MODELS["cells"], segmentation_mode="apg", ndim=3)
    embeddings = precompute_image_embeddings(predictor, volume, ndim=3, verbose=True)
    print(f"Embeddings of {volume.shape} in {time.time() - t0:.0f} s", flush=True)

    results = {}
    with embeddings:
        for mode, use_negatives in modes.items():
            t0 = time.time()
            segmenter = PromptableSegmentation3D(predictor, volume, embeddings)
            for nucleus_id, center in enumerate(centers, 1):
                points, labels = [center[1:]], [1]
                if use_negatives:
                    own = negatives[negatives[:, 0] == nucleus_id]
                    points.extend(own[:, 2:])
                    labels.extend([0] * len(own))
                segmenter.add_prompt_set(
                    frame_id=int(center[0]), points=np.array(points, dtype="float32"),
                    point_labels=np.array(labels, dtype="int32"), object_id=nucleus_id,
                )
            video_segments = segmenter.propagate_prompts()
            segmenter.reset_predictor()

            masks = {nucleus_id: np.zeros(volume.shape, dtype=bool) for nucleus_id in range(1, len(centers) + 1)}
            for frame, per_object in video_segments.items():
                for nucleus_id, mask in per_object.items():
                    masks[nucleus_id][frame] = np.asarray(mask).squeeze()[:volume.shape[1], :volume.shape[2]]
            del video_segments
            sizes = np.array([masks[nucleus_id].sum() for nucleus_id in sorted(masks)])
            segmentation = np.zeros(volume.shape, dtype="uint32")
            for nucleus_id in np.argsort(sizes)[::-1] + 1:
                segmentation[masks[nucleus_id]] = nucleus_id
            visible = np.bincount(segmentation.ravel(), minlength=len(sizes) + 1)[1:]
            visible_fraction = np.where(sizes > 0, visible / np.maximum(sizes, 1), 0.0)
            results[mode] = {
                "segmentation": segmentation, "size": sizes, "visible_fraction": visible_fraction,
                "time": time.time() - t0,
            }
            print(
                f"{mode}: {np.sum(sizes > 0)} of {len(sizes)} cells non-empty, median visible fraction "
                f"{np.median(visible_fraction[sizes > 0]):.2f}, in {time.time() - t0:.0f} s", flush=True,
            )
            del masks
    return results


def segment_cell_crop(zarr_path, levels, methods):
    """Segment the cells of the inverted cell crop of each level with 3d AIS and with prompts from the nuclei.

    The nuclei and their prompts come from s1 (watershed_volume) and are saved once (nucleus_prompts.h5). At s0 the
    points are mapped to the center of the 2 x 2 voxels of s0 under each voxel of s1. Each level and method is saved
    in its own h5 file, which a later run skips.
    """
    os.makedirs(CELL_CROP_DIR, exist_ok=True)
    shape_s1 = zarr.open_array(os.path.join(zarr_path, "s1"), mode="r").shape
    bb_s1 = get_inverted_crop(1, shape_s1, CELL_CROP_SHAPE)

    prompts_path = get_nucleus_prompts_path()
    if "prompts" in methods and not os.path.exists(prompts_path):
        nuclei, centers, negatives = get_nucleus_prompts(bb_s1)
        with h5py.File(prompts_path + ".tmp", "w") as f:
            f.attrs.update({
                "bb_start": [b.start for b in bb_s1], "bb_stop": [b.stop for b in bb_s1], "level": 1,
                "watershed": get_watershed_path(1, "nuclei"), "min_size": WS_MIN_SIZE,
                "n_negatives": CELL_N_NEGATIVES,
            })
            f.create_dataset("nuclei", data=nuclei, chunks=True, compression="gzip")
            f.create_dataset("centers", data=centers)
            f.create_dataset("negatives", data=negatives)
        os.rename(prompts_path + ".tmp", prompts_path)

    for level in levels:
        bb = get_inverted_crop(level, shape_s1, CELL_CROP_SHAPE)
        raw = None
        attrs = {
            "level": level, "bb_start": [b.start for b in bb], "bb_stop": [b.stop for b in bb],
            "voxel_size": get_level_voxel_size(level), "model_type": MODELS["cells"], "inverted": True,
        }
        for method in methods:
            path = get_cell_crop_path(level, method)
            if os.path.exists(path):
                print(f"{path} exists, skipping s{level} {method}")
                continue
            if raw is None:
                raw = 255 - zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")[bb]

            if method == "ais":
                prediction, segmentation, t_predict, t_segment = segment_ais_3d(raw, "cells")
                with h5py.File(path + ".tmp", "w") as f:
                    f.attrs.update({
                        **attrs, "tile_shape": VOLUME_TILE_SHAPE, "halo": VOLUME_HALO,
                        "time_predict": t_predict, "time_segment": t_segment,
                    })
                    for name, channel in zip(PREDICTION_CHANNELS, prediction):
                        f.create_dataset(
                            f"prediction/{name}", data=channel.astype("float16"), chunks=True, compression="gzip",
                        )
                    f.create_dataset("segmentation", data=segmentation, chunks=True, compression="gzip")
                os.rename(path + ".tmp", path)
                print(
                    f"s{level} cells AIS {raw.shape}: {len(np.unique(segmentation)) - 1} objects, predicted in "
                    f"{t_predict:.0f} s, segmented in {t_segment:.0f} s", flush=True,
                )
                del prediction, segmentation
                continue

            with h5py.File(prompts_path, "r") as f:
                centers, negatives = f["centers"][:].astype("float32"), f["negatives"][:].astype("float32")
            factor = np.array([f1 // f for f1, f in zip(get_level_factor(1), get_level_factor(level))])
            # The z of the points stays an integer (the slice), y and x go to the center of the voxels of the level.
            centers = centers * factor + (factor - 1) / 2
            negatives[:, 1:] = negatives[:, 1:] * factor + (factor - 1) / 2
            centers[:, 0], negatives[:, 1] = np.round(centers[:, 0]), np.round(negatives[:, 1])
            results = segment_cells_from_prompts(raw, centers, negatives)
            with h5py.File(path + ".tmp", "w") as f:
                f.attrs.update({**attrs, "prompts": prompts_path})
                for mode, result in results.items():
                    f.create_dataset(
                        f"segmentation/{mode}", data=result["segmentation"], chunks=True, compression="gzip",
                    )
                    f.create_dataset(f"size/{mode}", data=result["size"])
                    f.create_dataset(f"visible_fraction/{mode}", data=result["visible_fraction"])
                    f.attrs[f"time_{mode}"] = result["time"]
            os.rename(path + ".tmp", path)
            del results


def get_cell_seeded_path(level):
    return os.path.join(CELL_CROP_DIR, f"cells_seeded_s{level}.h5")


def _distances(mask, voxel_size):
    """The distance (nanometer) of every nonzero voxel to the nearest zero voxel, with bioimage-cpp.

    Without a zero voxel (where bioimage-cpp returns no meaningful distance), every distance is infinite.
    """
    import bioimage_cpp as bic

    if mask.all():
        return np.full(mask.shape, np.inf, dtype="float32")
    return bic.distance.distance_transform(mask, sampling=voxel_size)


def expand_nuclei(nuclei, voxel_size, distance):
    """Grow every nucleus by distance (nanometer) into the background; a voxel goes to its nearest nucleus."""
    import bioimage_cpp as bic

    background = nuclei == 0
    if background.all() or not background.any():
        return nuclei.copy()
    distances, indices = bic.distance.distance_transform(background, sampling=voxel_size, return_indices=True)
    expanded = nuclei[tuple(indices)]
    expanded[distances > distance] = 0
    return expanded


def close_mask(mask, voxel_size, radius):
    """Binary closing with a ball of radius (nanometer), computed exactly with two distance transforms.

    The dilation keeps the voxels within radius of the mask, the erosion the voxels of the dilation that are further
    than radius from its complement. The volume border does not erode the mask.
    """
    mask = mask.astype(bool)
    dilated = _distances(~mask, voxel_size) <= radius
    return (_distances(dilated, voxel_size) > radius) | mask


def get_distance_halo(distance, voxel_size):
    """The halo (voxels per axis) that covers a distance (nanometer), plus one voxel."""
    return tuple(int(math.ceil(distance / vs)) + 1 for vs in voxel_size)


def _get_repeated_source_class():
    """The wrapper that repeats every voxel of a label source factor times along each axis (defined lazily, since
    it registers itself with bioimage-py)."""
    from bioimage_py.wrapper.base import WrapperSource, register_wrapper

    if "RepeatedSource" in globals():
        return globals()["RepeatedSource"]

    @register_wrapper
    class RepeatedSource(WrapperSource):
        """Repeat every voxel of the wrapped source factor times along each axis on read (nearest upsampling).

        Unlike the ResizedSource of bioimage-py with order 0, the voxel i of the source covers exactly the voxels
        factor * i to factor * (i + 1) - 1, as for the labels of s1 on the grid of s0 (np.repeat).
        """

        def __init__(self, source, factor):
            super().__init__(source)
            self._factor = tuple(int(f) for f in factor)
            assert len(self._factor) == len(self._source.shape)

        @property
        def shape(self):
            return tuple(s * f for s, f in zip(self._source.shape, self._factor))

        @property
        def chunks(self):
            chunks = self._source.chunks
            return None if chunks is None else tuple(c * f for c, f in zip(chunks, self._factor))

        @property
        def shards(self):
            return None

        def _getitem(self, roi):
            starts = [0 if sl.start is None else int(sl.start) for sl in roi]
            stops = [sh if sl.stop is None else int(sl.stop) for sl, sh in zip(roi, self.shape)]
            source_bb = tuple(slice(st // f, -(-sp // f)) for st, sp, f in zip(starts, stops, self._factor))
            data = np.asarray(self._source[source_bb])
            for axis, f in enumerate(self._factor):
                if f > 1:
                    data = np.repeat(data, f, axis=axis)
            local = tuple(slice(st - b.start * f, sp - b.start * f)
                          for st, sp, b, f in zip(starts, stops, source_bb, self._factor))
            return data[local]

        def _params(self):
            return {"factor": self._factor}

    globals()["RepeatedSource"] = RepeatedSource
    return RepeatedSource


def _get_subsampled_source_class():
    """The wrapper that takes every factor-th voxel of a label source along each axis (defined lazily, since it
    registers itself with bioimage-py)."""
    from bioimage_py.wrapper.base import WrapperSource, register_wrapper

    if "SubsampledSource" in globals():
        return globals()["SubsampledSource"]

    @register_wrapper
    class SubsampledSource(WrapperSource):
        """Take the first voxel of each block of factor voxels on read (nearest downsampling, as downsample_labels).

        The shape is the ceil of the source shape divided by the factor. Unlike bioimage-py's downsample with
        order 0, which maps by the ratio of the shapes, the voxel i is exactly the source voxel factor * i.
        """

        def __init__(self, source, factor):
            super().__init__(source)
            self._factor = tuple(int(f) for f in factor)
            assert len(self._factor) == len(self._source.shape)

        @property
        def shape(self):
            return tuple(-(-s // f) for s, f in zip(self._source.shape, self._factor))

        @property
        def chunks(self):
            return None

        @property
        def shards(self):
            return None

        def _getitem(self, roi):
            starts = [0 if sl.start is None else int(sl.start) for sl in roi]
            stops = [sh if sl.stop is None else int(sl.stop) for sl, sh in zip(roi, self.shape)]
            source_bb = tuple(
                slice(st * f, min(sp * f, sh)) for st, sp, f, sh in zip(starts, stops, self._factor, self._source.shape)
            )
            return np.asarray(self._source[source_bb])[tuple(slice(None, None, f) for f in self._factor)]

        def _params(self):
            return {"factor": self._factor}

    globals()["SubsampledSource"] = SubsampledSource
    return SubsampledSource


def repeat_labels(labels, factor):
    """A lazy source of labels repeated factor times along each axis (labels of s1 on the grid of s0)."""
    if all(f == 1 for f in factor):
        return labels
    return _get_repeated_source_class()(labels, factor)


def nucleus_watershed(foreground, boundary, nuclei, voxel_size, distance=CELL_EXPAND_DISTANCE, closing_radius=None):
    """Seeded watershed on the boundary prediction within the foreground, seeded with the expanded nuclei.

    With closing_radius (nanometer), the foreground mask is closed first. The parts of the foreground that are not
    connected to a seed stay background. Returns the segmentation, the seeds and the mask.
    """
    import bioimage_cpp as bic

    seeds = expand_nuclei(nuclei, voxel_size, distance)
    mask = foreground > CELL_FOREGROUND_THRESHOLD
    if closing_radius is not None:
        mask = close_mask(mask, voxel_size, closing_radius)
    return bic.segmentation.watershed(np.ascontiguousarray(boundary), seeds, mask=mask), seeds, mask


def graph_watershed(oversegmentation, boundary, nuclei, edge_factor=CELL_NUCLEUS_EDGE_FACTOR):
    """Seeded watershed on the region adjacency graph of an over-segmentation, seeded with its nucleus fragments.

    The edge weight is the mean boundary prediction on the face between two fragments. A fragment is seeded with a
    nucleus if more than CELL_NUCLEUS_FRACTION of its voxels belong to that nucleus (counting the voxels outside the
    nuclei as well), and its edges to the fragments that are not seeded this way are multiplied by edge_factor, so
    that the cell grows across the nuclear envelope. A nucleus without such a fragment lies mostly in a fragment that
    reaches beyond it: it seeds the fragment with the largest part of it, if no other nucleus seeds that one, but
    its edges keep their weight, since that fragment ends beyond the nuclear envelope already. The background
    (label 0 of the over-segmentation) is a seed of its own whose edges come last, so it takes only the fragments no
    nucleus reaches; they become background again.

    Returns the segmentation and statistics of the seeding.
    """
    import bioimage_cpp as bic
    from skimage.segmentation import relabel_sequential

    oversegmentation = relabel_sequential(oversegmentation)[0].astype("uint32")
    rag = bic.graph.region_adjacency_graph(oversegmentation)
    weights = bic.graph.features.edge_map_features(
        rag, oversegmentation, np.ascontiguousarray(boundary, dtype="float32"),
    )[:, 0].astype("float32")

    table = bic.utils.segmentation_overlap(oversegmentation, nuclei).overlap_table(normalize_by="a")
    fragment, nucleus = table["label_a"].astype("int64"), table["label_b"].astype("int64")
    count, fraction = table["count"], table["fraction"]
    seeds = np.zeros(rag.number_of_nodes, dtype="uint32")
    majority = (fragment > 0) & (nucleus > 0) & (fraction > CELL_NUCLEUS_FRACTION)
    seeds[fragment[majority]] = nucleus[majority]
    nucleus_fragment = seeds > 0

    n_nuclei = int(nuclei.max())
    fallback, lost = [], []
    for nucleus_id in sorted(set(range(1, n_nuclei + 1)) - set(nucleus[majority].tolist())):
        rows = np.where((nucleus == nucleus_id) & (fragment > 0))[0]
        if len(rows) == 0:
            lost.append(nucleus_id)
            continue
        best = fragment[rows[np.argmax(count[rows])]]
        if seeds[best] == 0:
            seeds[best] = nucleus_id
            fallback.append(nucleus_id)
        else:
            lost.append(nucleus_id)

    uv = rag.uv_ids()
    downweight = nucleus_fragment[uv[:, 0]] != nucleus_fragment[uv[:, 1]]
    stats = {
        "n_fragments": int(oversegmentation.max()), "n_nucleus_fragments": int(nucleus_fragment.sum()),
        "n_fallback_nuclei": len(fallback), "n_lost_nuclei": len(lost), "n_downweighted_edges": int(downweight.sum()),
        "weight_downweighted_edges": float(np.median(weights[downweight])) if downweight.any() else 0.0,
        "weight_other_edges": float(np.median(weights[~downweight & (uv > 0).all(axis=1)])),
    }
    weights[downweight] *= edge_factor
    background_edge = (uv == 0).any(axis=1)
    weights[background_edge] = weights.max() + 1.0
    background_seed = n_nuclei + 1
    seeds[0] = background_seed

    node_labels = bic.graph.edge_weighted_watershed(rag, weights, seeds)
    node_labels[node_labels == background_seed] = 0
    segmentation = bic.graph.project_node_labels_to_pixels(rag, oversegmentation, node_labels)
    return segmentation.astype("uint32"), oversegmentation, stats


def get_cell_watershed_path(level):
    return os.path.join(CELL_CROP_DIR, f"cells_watershed_s{level}.h5")


def load_cell_crop_nuclei(level):
    """The nuclei of the cell crop (nucleus_prompts.h5, at s1), repeated to the voxel grid of a level."""
    with h5py.File(get_nucleus_prompts_path(), "r") as f:
        nuclei = f["nuclei"][:].astype("uint32")
    factor = [f1 // f for f1, f in zip(get_level_factor(1), get_level_factor(level))]
    for axis, f in enumerate(factor):
        nuclei = np.repeat(nuclei, f, axis=axis)
    return nuclei


def watershed_cell_crop(levels, closing_radius=CELL_CLOSING_RADIUS):
    """The watershed of the expanded nuclei on the AIS predictions of the cell crop, with the closed foreground mask.

    Saves the segmentation, the seeds and the closed mask of each level in one h5 file (cells_watershed_s<level>.h5).
    """
    for level in levels:
        path = get_cell_watershed_path(level)
        if os.path.exists(path):
            print(f"{path} exists, skipping s{level}")
            continue
        nuclei = load_cell_crop_nuclei(level)
        voxel_size = get_level_voxel_size(level)
        with h5py.File(get_cell_crop_path(level, "ais"), "r") as f:
            foreground = f["prediction/foreground"][:].astype("float32")
            boundary = f["prediction/boundary"][:].astype("float32")
            attrs = dict(f.attrs)
        assert nuclei.shape == foreground.shape, (nuclei.shape, foreground.shape)

        t0 = time.time()
        segmentation, seeds, mask = nucleus_watershed(
            foreground, boundary, nuclei, voxel_size, closing_radius=closing_radius,
        )
        t_watershed = time.time() - t0
        n_closed = int(mask.sum() - (foreground > CELL_FOREGROUND_THRESHOLD).sum())

        with h5py.File(path + ".tmp", "w") as f:
            f.attrs.update({
                "level": level, "bb_start": attrs["bb_start"], "bb_stop": attrs["bb_stop"], "voxel_size": voxel_size,
                "predictions": get_cell_crop_path(level, "ais"), "nuclei": get_nucleus_prompts_path(),
                "foreground_threshold": CELL_FOREGROUND_THRESHOLD, "expand_distance": CELL_EXPAND_DISTANCE,
                "closing_radius": closing_radius, "n_closed_voxels": n_closed, "time_watershed": t_watershed,
            })
            f.create_dataset("segmentation", data=segmentation.astype("uint32"), chunks=True, compression="gzip")
            f.create_dataset("seeds", data=seeds.astype("uint32"), chunks=True, compression="gzip")
            f.create_dataset("mask", data=mask.astype("uint8"), chunks=True, compression="gzip")
        os.rename(path + ".tmp", path)
        print(
            f"s{level}: {len(np.unique(segmentation)) - 1} cells in {t_watershed:.0f} s, the closing added "
            f"{n_closed} voxels ({n_closed / mask.size:.3f} of the crop) to the mask", flush=True,
        )


def get_border_faces(shape):
    """The index of each of the Y and X faces of a volume."""
    return (
        (slice(None), 0), (slice(None), shape[1] - 1), (slice(None), slice(None), 0),
        (slice(None), slice(None), shape[2] - 1),
    )


def add_border_seed(seeds, mask, seed_id=CELL_BACKGROUND_SEED_ID):
    """Set the mask voxels without a seed on the Y and X faces of a seed volume to seed_id, in place."""
    for face in get_border_faces(seeds.shape):
        view = seeds[face]
        view[(view == 0) & mask[face]] = seed_id


def _add_background_seed(seeds, background):
    return np.where((seeds == 0) & (background > 0), CELL_BACKGROUND_SEED_ID, seeds).astype("uint64")


def _is_background_seed(segmentation):
    return (segmentation == CELL_BACKGROUND_SEED_ID).astype("uint8")


def _remove_background_seed(segmentation):
    return np.where(segmentation == CELL_BACKGROUND_SEED_ID, 0, segmentation).astype(segmentation.dtype)


def segment_cells_blockwise(
    foreground, boundary, nuclei, outputs, voxel_size, num_workers, block_shape=CELL_VOLUME_BLOCK_SHAPE,
    ws_halo=CELL_VOLUME_WS_HALO, expand_distance=CELL_EXPAND_DISTANCE, closing_radius=CELL_CLOSING_RADIUS,
    stages=None, on_stage_done=None,
):
    """The cell segmentation of watershed_cell_crop, block-wise with bioimage-py and bioimage-cpp.

    The stages, each written into its output before the next one starts:
    - seeds: the nuclei expanded by expand_distance (expand_nuclei on each block with the halo of the distance).
    - mask: foreground > CELL_FOREGROUND_THRESHOLD closed with a ball of closing_radius (close_mask on each block with
      the halo of twice the radius). Both are exact: a voxel of a block only depends on the voxels within the halo.
    - background (only if the outputs have 'background'): the seed for the resin around the spheroid, the mask
      voxels without a seed on the Y and X faces of the volume.
    - segmentation: the seeded watershed of bioimage-py on the boundary prediction within the mask, with ws_halo.
      With the background seed it is repeated with the region of the background seed of the previous pass as seed,
      so that the seed reaches the blocks that do not see the faces (one more block per pass, so half the number of
      blocks in Y or X passes). The region is kept in 'background' and set to zero in the segmentation.

    Args:
        foreground: The foreground prediction (a source, e.g. a zarr array).
        boundary: The boundary prediction.
        nuclei: The nuclei on the same voxel grid (a source, see repeat_labels).
        outputs: The writable output arrays for 'seeds' (uint64), 'mask' (uint8) and 'segmentation' (uint64), and
            optionally 'background' (uint8).
        voxel_size: The ZYX voxel size in nanometer.
        num_workers: The number of threads.
        block_shape: The block shape of all stages; the chunks of the outputs must divide it.
        ws_halo: The halo of the watershed.
        expand_distance: The expansion of the nuclei in nanometer.
        closing_radius: The radius of the closing in nanometer.
        stages: The stages to run, by default all.
        on_stage_done: Called with the name of each stage once it is written.
    """
    from functools import partial
    from bioimage_py.copy import copy
    from bioimage_py.segmentation import watershed
    from bioimage_py.wrapper import (
        MultiTransformationSource, SimpleTransformationSource, SimpleTransformationWithHaloSource, ThresholdSource,
    )

    use_background = "background" in outputs
    if stages is None:
        stages = ("seeds", "mask") + (("background",) if use_background else ()) + ("segmentation",)
    timings = {}
    if "seeds" in stages:
        t0 = time.time()
        seeds = SimpleTransformationWithHaloSource(
            nuclei, partial(expand_nuclei, voxel_size=voxel_size, distance=expand_distance),
            get_distance_halo(expand_distance, voxel_size), dtype="uint64",
        )
        copy(seeds, outputs["seeds"], block_shape=block_shape, num_workers=num_workers)
        timings["seeds"] = time.time() - t0
        if on_stage_done is not None:
            on_stage_done("seeds")

    if "mask" in stages:
        t0 = time.time()
        mask = SimpleTransformationWithHaloSource(
            ThresholdSource(foreground, CELL_FOREGROUND_THRESHOLD),
            partial(_close_mask_uint8, voxel_size=voxel_size, radius=closing_radius),
            get_distance_halo(2 * closing_radius, voxel_size), dtype="uint8",
        )
        copy(mask, outputs["mask"], block_shape=block_shape, num_workers=num_workers)
        timings["mask"] = time.time() - t0
        if on_stage_done is not None:
            on_stage_done("mask")

    if "background" in stages:
        t0 = time.time()
        for face in get_border_faces(outputs["background"].shape):
            free = (np.asarray(outputs["seeds"][face]) == 0) & (np.asarray(outputs["mask"][face]) > 0)
            outputs["background"][face] = np.maximum(np.asarray(outputs["background"][face]), free.astype("uint8"))
        timings["background"] = time.time() - t0
        if on_stage_done is not None:
            on_stage_done("background")

    if "segmentation" in stages:
        t0 = time.time()
        seeds, n_passes = outputs["seeds"], 1
        if use_background:
            seeds = MultiTransformationSource(_add_background_seed, seeds, outputs["background"], dtype="uint64")
            n_passes = math.ceil(max(math.ceil(sh / bs) for sh, bs in zip(seeds.shape[1:], block_shape[1:])) / 2)
        for _ in range(n_passes):
            watershed(
                boundary, seeds, outputs["segmentation"], halo=ws_halo, mask=outputs["mask"],
                block_shape=block_shape, num_workers=num_workers,
            )
            if use_background:
                # The region of the background seed is the seed of the next pass. A pass that stops leaves a
                # part of it in 'background', which is a valid seed for the run that continues.
                copy(
                    SimpleTransformationSource(outputs["segmentation"], _is_background_seed, dtype="uint8"),
                    outputs["background"], block_shape=block_shape, num_workers=num_workers,
                )
        if use_background:
            # In place: every block reads and writes its own voxels only.
            copy(
                SimpleTransformationSource(outputs["segmentation"], _remove_background_seed), outputs["segmentation"],
                block_shape=block_shape, num_workers=num_workers,
            )
        timings["segmentation"] = time.time() - t0
        if on_stage_done is not None:
            on_stage_done("segmentation")
    return timings


def _close_mask_uint8(mask, voxel_size, radius):
    return close_mask(mask, voxel_size, radius).astype("uint8")


def get_merged_nuclei_path():
    return os.path.join(VOLUME_PREDICTION_DIR, "nuclei_s1_merged.zarr")


def get_joined_nuclei_path():
    return os.path.join(VOLUME_PREDICTION_DIR, "nuclei_s1_joined.zarr")


def get_updated_nuclei_path():
    return os.path.join(VOLUME_PREDICTION_DIR, "nuclei_s1_updated.zarr")


def get_nuclei_path(nuclei="watershed"):
    """The group with the nuclei of s1: those of watershed_volume ('watershed'), of merge_prompted_nuclei ('merged'),
    of join_nucleus_fragments ('joined') or of update_partial_nuclei.py ('updated'). Each one builds on the one
    before. The cells of the updated nuclei are not from segment_cells_volume, update_partial_nuclei.py relabels the
    cells of the joined nuclei."""
    paths = {
        "watershed": get_watershed_path(1, "nuclei"), "merged": get_merged_nuclei_path(),
        "joined": get_joined_nuclei_path(), "updated": get_updated_nuclei_path(),
    }
    return paths[nuclei]


def get_cell_volume_path(level, border_seed=False, nuclei="watershed"):
    name = f"cells_s{level}_segmentation{'' if nuclei == 'watershed' else f'_{nuclei}_nuclei'}"
    return os.path.join(VOLUME_PREDICTION_DIR, f"{name}{'_border_seed' if border_seed else ''}.zarr")


def segment_cells_volume(
    level, num_workers, block_shape=CELL_VOLUME_BLOCK_SHAPE, ws_halo=CELL_VOLUME_WS_HALO, border_seed=False,
    nuclei="watershed",
):
    """Segment the cells of a whole scale level, block-wise (segment_cells_blockwise).

    With border_seed, the resin around the spheroid gets its own seed, from the Y and X faces of the volume, and
    its region is set to zero (stage 'background', CELL_BACKGROUND_SEED_ID); the result goes to its own group, with
    the region in 'background'. With nuclei 'merged' or 'joined', the nuclei are those of merge_prompted_nuclei or
    of join_nucleus_fragments (get_nuclei_path); the result also goes to its own group. The cells of the nuclei that
    are only seeds (attribute 'seed_only') are full cells.

    The inputs are the cell predictions of predict_volume (target 'cells') and the nuclei of watershed_volume at s1,
    repeated to the voxel grid of the level. The seeds, the mask and the segmentation are zarr arrays of the group
    get_cell_volume_path(level, border_seed, nuclei) with chunks of CELL_VOLUME_CHUNK_SHAPE, which divide
    block_shape. A stage that is written is recorded in the attributes, so a run that stops continues with the next
    stage.
    """
    zarr.config.set({"threading.max_workers": num_workers})
    predictions = zarr.open_group(get_volume_prediction_path(level, "cells"), mode="r")
    done_slabs = predictions.attrs.get("done_slabs", [])
    n_slabs = math.ceil(predictions["foreground"].shape[0] / predictions.attrs["slab_size"])
    assert len(done_slabs) == n_slabs, f"The cell predictions of s{level} are incomplete: {len(done_slabs)} / {n_slabs}"
    foreground, boundary = predictions["foreground"], predictions["boundary"]
    factor = [f1 // f for f1, f in zip(get_level_factor(1), get_level_factor(level))]
    nuclei_path = get_nuclei_path(nuclei)
    nuclei_source = repeat_labels(zarr.open_group(nuclei_path, mode="r")["segmentation"], factor)
    shape = tuple(foreground.shape)
    assert tuple(nuclei_source.shape) == shape, (nuclei_source.shape, shape)
    voxel_size = get_level_voxel_size(level)

    assert all(b % c == 0 for b, c in zip(block_shape, CELL_VOLUME_CHUNK_SHAPE)), block_shape
    out = zarr.open_group(get_cell_volume_path(level, border_seed, nuclei), mode="a", zarr_format=3)
    params = {
        "predictions": get_volume_prediction_path(level, "cells"), "nuclei": nuclei_path,
        "level": level, "voxel_size": list(voxel_size), "foreground_threshold": CELL_FOREGROUND_THRESHOLD,
        "expand_distance": CELL_EXPAND_DISTANCE, "closing_radius": CELL_CLOSING_RADIUS,
        "block_shape": list(block_shape), "ws_halo": list(ws_halo),
    }
    if border_seed:
        params["border_seed"] = True
    if "params" in out.attrs:
        assert out.attrs["params"] == params, f"{out.store} was written with other parameters."
    out.attrs["params"] = params
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    outputs = {
        name: out.require_array(
            name, shape=shape, dtype=dtype, chunks=CELL_VOLUME_CHUNK_SHAPE, compressors=compressor, fill_value=0,
            dimension_names=["z", "y", "x"],
        ) for name, dtype in (
            ("seeds", "uint64"), ("mask", "uint8"), ("background", "uint8"), ("segmentation", "uint64"),
        ) if border_seed or name != "background"
    }
    done = list(out.attrs.get("done_stages", []))
    all_stages = ("seeds", "mask") + (("background",) if border_seed else ()) + ("segmentation",)
    stages = [stage for stage in all_stages if stage not in done]

    def on_stage_done(stage):
        done.append(stage)
        out.attrs["done_stages"] = done
        print(f"s{level}: {stage} written", flush=True)

    timings = segment_cells_blockwise(
        foreground, boundary, nuclei_source, outputs, voxel_size, num_workers, block_shape=block_shape,
        ws_halo=ws_halo, stages=stages, on_stage_done=on_stage_done,
    )
    print(f"s{level}: cells segmented, {', '.join(f'{k} in {v:.0f} s' for k, v in timings.items())}", flush=True)


def get_segmentation_sources():
    """The volume segmentations: name -> (path of the zarr group, array name, the level it was computed at)."""
    sources = {
        "cells": (get_cell_volume_path(0), "segmentation", 0),
        "nuclei": (get_watershed_path(1, "nuclei"), "segmentation", 1),
    }
    if os.path.exists(get_prompted_nuclei_path()):
        sources["nuclei_prompted"] = (get_prompted_nuclei_path(), "segmentation", 1)
    if os.path.exists(get_border_seed_check_path()):
        sources["cells_border_seed"] = (get_border_seed_check_path(), "segmentation", BORDER_SEED_CHECK_LEVEL)
    for nuclei in ("merged", "joined", "updated"):
        if os.path.exists(get_nuclei_path(nuclei)):
            sources[f"nuclei_{nuclei}"] = (get_nuclei_path(nuclei), "segmentation", 1)
        path = get_cell_volume_path(0, border_seed=True, nuclei=nuclei)
        if os.path.exists(path):
            sources[f"cells_{nuclei}_nuclei_border_seed"] = (path, "segmentation", 0)
    if os.path.exists(os.path.join(REPROMPTED_NUCLEI_PATH, "suggested")):
        sources["nuclei_reprompted"] = (REPROMPTED_NUCLEI_PATH, "suggested", 1)
    return sources


def downsample_segmentations(num_workers, max_level=len(DOWNSCALE_FACTORS), names=None):
    """Write the pyramids of the volume segmentations up to max_level, block-wise (uint16).

    Each level takes the first voxel of each block of DOWNSCALE_FACTORS of the level below (SubsampledSource copied
    with bioimage-py), as write_label_pyramid does for the label images of the OME-Zarr. So the voxel i of a level of
    the cells is the voxel factor * i of s0 (no translation on the grid of s0), the voxel i of a level of the nuclei
    the voxel factor * i of s1 (the translation of s1 of the raw). With names, only these segmentations are written.
    """
    from bioimage_py.copy import copy

    zarr.config.set({"threading.max_workers": num_workers})
    out = zarr.open_group(SEGMENTATION_PYRAMID_PATH, mode="a", zarr_format=3)
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    Subsampled = _get_subsampled_source_class()
    for name, (path, key, source_level) in get_segmentation_sources().items():
        if names is not None and name not in names:
            continue
        source = zarr.open_group(path, mode="r")[key]
        group = out.require_group(name)
        group.attrs.update({"source": os.path.join(path, key), "source_level": source_level})
        previous = source
        for level in range(source_level + 1, max_level + 1):
            t0 = time.time()
            subsampled = Subsampled(previous, DOWNSCALE_FACTORS[level - 1])
            expected = get_downsampled_shape(previous.shape, DOWNSCALE_FACTORS[level - 1])
            assert subsampled.shape == expected, (subsampled.shape, expected)
            if f"s{level}" in group:
                del group[f"s{level}"]
            array = group.create_array(
                f"s{level}", shape=subsampled.shape, dtype="uint16", chunks=SEGMENTATION_PYRAMID_CHUNKS,
                compressors=compressor, fill_value=0, dimension_names=["z", "y", "x"],
            )
            copy(subsampled, array, block_shape=SEGMENTATION_PYRAMID_CHUNKS, num_workers=num_workers)
            print(f"{name} s{level} {array.shape} in {time.time() - t0:.0f} s", flush=True)
            previous = array


def _to_uint16(labels):
    return labels.astype("uint16")


def export_segmentations(output_path=EXPORT_PATH, nuclei="updated", num_workers=1):
    """Export the cells and their nuclei as an OME-Zarr with two multiscale label images, 'cells' and 'nuclei'.

    The cells are those of 'segment_cells_volume --border_seed' for the given nuclei (computed at s0; for the updated
    nuclei the relabeled cells of update_partial_nuclei.py), the nuclei those of get_nuclei_path(nuclei) (computed at
    s1). Both have the levels s0 - s3 of the raw (the same shapes,
    DOWNSCALE_FACTORS), as uint16 with the shards and chunks of the raw: the nuclei at s0 are those of s1 repeated
    twice in y and x, and each level above the one a segmentation was computed at takes the first voxel of each
    block of the level below (as the pyramids of downsample_segmentations and the label images of the raw). So the
    voxel i of a level is the voxel factor * i of s0, in both images, and a cell has the id of its nucleus.

    The attributes of the root list the sources and the notes on the nuclei: the prompted nuclei that are only
    seeds for their cell and not full nuclei ('seed_only'), the nuclei joined from fragments ('joined') and, for the
    updated nuclei, the objects that are not full nuclei or cells and should be excluded from the analysis
    ('excluded': id, what to exclude and the reason) and the changes of update_partial_nuclei.py ('candidates',
    'merged', 'removed').

    Args:
        output_path: The path of the OME-Zarr. It must not exist.
        nuclei: The nuclei, see get_nuclei_path.
        num_workers: The number of threads.
    """
    from bioimage_py.copy import copy
    from bioimage_py.wrapper import SimpleTransformationSource

    assert not os.path.exists(output_path), f"{output_path} exists."
    zarr.config.set({"threading.max_workers": num_workers})
    cells_path, nuclei_path = get_cell_volume_path(0, border_seed=True, nuclei=nuclei), get_nuclei_path(nuclei)
    cells_group, nuclei_group = zarr.open_group(cells_path, mode="r"), zarr.open_group(nuclei_path, mode="r")
    assert "segmentation" in cells_group.attrs["done_stages"], cells_path
    assert cells_group.attrs["params"]["nuclei"] == nuclei_path, cells_group.attrs["params"]["nuclei"]
    nuclei_s1 = nuclei_group["segmentation"]
    # Per image: level -> source, for the levels that are not subsampled from the level below.
    sources = {
        "cells": {0: cells_group["segmentation"]},
        "nuclei": {0: repeat_labels(nuclei_s1, DOWNSCALE_FACTORS[0]), 1: nuclei_s1},
    }
    shape = tuple(cells_group["segmentation"].shape)
    assert tuple(sources["nuclei"][0].shape) == shape, (sources["nuclei"][0].shape, shape)

    root = zarr.open_group(output_path, mode="w", zarr_format=3)
    # Labels compress best with byte shuffle, the ids are piecewise constant.
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    Subsampled = _get_subsampled_source_class()
    num_levels = len(DOWNSCALE_FACTORS) + 1
    for name, level_sources in sources.items():
        group = root.create_group(name)
        previous = None
        for level in range(num_levels):
            t0 = time.time()
            source = level_sources.get(level)
            if source is None:
                source = Subsampled(previous, DOWNSCALE_FACTORS[level - 1])
            level_shape = tuple(source.shape)
            if level > 0:
                expected = get_downsampled_shape(previous.shape, DOWNSCALE_FACTORS[level - 1])
                assert level_shape == expected, (level_shape, expected)
            array = group.create_array(
                f"s{level}", shape=level_shape, dtype="uint16", shards=SHARD_SHAPE, chunks=CHUNK_SHAPE,
                compressors=compressor, fill_value=0, dimension_names=["z", "y", "x"],
            )
            # Every block writes whole shards.
            copy(
                SimpleTransformationSource(source, _to_uint16, dtype="uint16"), array, block_shape=SHARD_SHAPE,
                num_workers=num_workers,
            )
            print(f"{name} s{level} {level_shape} in {time.time() - t0:.0f} s", flush=True)
            previous = array
        multiscale = get_multiscale(name, num_levels, "nearest")
        group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale], "image-label": {"version": "0.5"}}

    nuclei_attrs = dict(nuclei_group.attrs)
    updated = (
        " The partial nuclei were segmented again from prompts, the pieces of one nucleus are one object and small "
        "nucleus fragments are removed, with their cells as part of the neighboring cell (update_partial_nuclei.py). "
        "The objects of 'excluded' are not full nuclei or cells and should be excluded from the analysis."
    ) if "excluded" in nuclei_attrs else ""
    root.attrs.update({
        "raw": OUTPUT_PATH, "voxel_size_s0": list(VOXEL_SIZE), "cells": cells_path, "nuclei": nuclei_path,
        "description": (
            "Cells and nuclei of the whole spheroid, instance labels (0 is background). A cell has the id of its "
            "nucleus. 'nuclei' was segmented at s1 (its s0 is s1 repeated twice in y and x), 'cells' at s0 by a "
            f"watershed seeded with the nuclei.{updated} Written by export_segmentations of "
            "micro-sam/scripts/user_study/v2/beghin_lab/explore_beghin_em_data.py."
        ),
        "seed_only": [
            {"id": obj["id"], "comment": obj["comment"]} for obj in nuclei_attrs.get("objects", []) if obj["seed_only"]
        ],
        "joined": [
            {"id": obj["id"], "fragments": obj["fragments"], "comments": obj["comments"]}
            for obj in nuclei_attrs.get("joined", [])
        ],
        **{name: nuclei_attrs[name] for name in ("excluded", "candidates", "merged", "removed") if updated},
    })
    print(f"Exported to {output_path}", flush=True)


def get_contour_path(level=CONTOUR_LEVEL):
    return os.path.join(os.path.dirname(DATA_DIR), f"PF_Dish_C_pin3new_Spheroid3_s{level}_contours.tif")


def get_contours(labels):
    """The contours of a label volume, per slice: the voxels of an object next to another label in its plane (255)."""
    contours = np.zeros(labels.shape, dtype="uint8")
    for z, plane in enumerate(labels):
        contours[z] = 255 * find_boundaries(plane, mode="inner")
    return contours


def export_contours(zarr_path, export_path=EXPORT_PATH, level=CONTOUR_LEVEL, output_path=None):
    """Export a level of the raw and the contours of the cells and nuclei as an ImageJ hyperstack for Fiji.

    The tif has the axes ZCYX and the channels of CONTOUR_CHANNELS, all uint8: the inverted raw (as the models see
    it, 255 - raw), the contours of the cells and the contours of the nuclei of export_segmentations at this level
    (get_contours, 0 or 255). It is uncompressed and opens in Fiji as a composite with the voxel size in micrometer,
    one color per channel and the channel names as slice labels. The labels of a level are the first voxel of each
    block of s0 and the raw is the mean of the block, so the contours are shifted against the raw by less than half a
    voxel of the level.

    Args:
        zarr_path: The OME-Zarr with the raw data.
        export_path: The OME-Zarr of export_segmentations.
        level: The scale level.
        output_path: The path of the tif, by default get_contour_path(level).
    """
    output_path = get_contour_path(level) if output_path is None else output_path
    raw = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")[:]
    export = zarr.open_group(export_path, mode="r")
    channels = [255 - raw]
    for name in ("cells", "nuclei"):
        labels = export[name][f"s{level}"][:]
        assert labels.shape == raw.shape, (labels.shape, raw.shape)
        channels.append(get_contours(labels))
        print(f"{name}: {len(np.unique(labels)) - 1} objects at s{level}", flush=True)
    data = np.stack(channels, axis=1)

    voxel_size = [vs / 1000 for vs in get_level_voxel_size(level)]
    ramp = np.arange(256) / 255
    luts = [np.round(np.outer(color, ramp)).astype("uint8") for color in CONTOUR_CHANNELS.values()]
    tifffile.imwrite(
        output_path, data, imagej=True, resolution=(1 / voxel_size[2], 1 / voxel_size[1]),
        metadata={
            "axes": "ZCYX", "spacing": voxel_size[0], "unit": "micron", "mode": "composite", "LUTs": luts,
            "Ranges": tuple([0.0, 255.0] * len(luts)), "Labels": list(CONTOUR_CHANNELS) * data.shape[0],
        },
    )
    print(
        f"Saved {output_path}: {data.shape} (ZCYX) {data.dtype}, voxel size {voxel_size} um (ZYX), channels "
        f"{list(CONTOUR_CHANNELS)}, {os.path.getsize(output_path) / 1e6:.0f} MB", flush=True,
    )


def get_tif_archive_path(level=TIF_LEVEL):
    return os.path.join(os.path.dirname(DATA_DIR), f"PF_Dish_C_pin3new_Spheroid3_segmentation_s{level}_tif.zip")


def _iterate_planes(array, step=SHARD_SHAPE[0] // 4):
    for z in range(0, array.shape[0], step):
        yield from array[z:z + step]


TIF_README = """Cells and nuclei of the EM spheroid PF_Dish_C_pin3new_Spheroid3 at the scale level s{level}.

cells_s{level}.tif, nuclei_s{level}.tif: instance labels, {shape} (ZYX) uint16, voxel size {voxel_size} um (ZYX).
0 is the background, a cell has the id of its nucleus. {n_cells} cells and {n_nuclei} nuclei. The voxels are those of
the raw data (PF_Dish_C_pin3new_Spheroid3.tif{raw_note}).

The tifs are uncompressed ImageJ tifs, {size:.1f} GB each. Fiji opens them with File > Open (they are loaded into
memory) or, without loading them, with File > Import > TIFF Virtual Stack. The voxel size is set, the display range
is 0 to the largest id; a LUT such as 'glasbey on dark' (Image > Lookup Tables) shows the objects in colors.
{excluded}
Exported from {export_path} by export_tifs of
micro-sam/scripts/user_study/v2/beghin_lab/explore_beghin_em_data.py.
"""
TIF_README_EXCLUDED = """
excluded.csv: the objects that are not full nuclei or cells and should be excluded from the analysis, with the id,
what to exclude ('nucleus and cell' or only the 'nucleus') and the reason.
"""


def export_tifs(export_path=EXPORT_PATH, level=TIF_LEVEL, output_path=None):
    """Export a level of the cells and nuclei of export_segmentations as label tifs for Fiji, in a zip archive.

    The archive has one folder with 'cells_s<level>.tif' and 'nuclei_s<level>.tif' (instance labels, uint16, ZYX),
    'excluded.csv' (the objects to exclude from the analysis, if the export lists them) and 'README.txt'. The tifs
    are uncompressed ImageJ tifs with the voxel size in micrometer and the ids as display range. Above 4 GB (s0)
    they are written as ImageJ does it: not as BigTIFF but with one image description and all planes one after the
    other, so that Fiji opens them directly or as a virtual stack. The tifs only exist in the archive.

    Args:
        export_path: The OME-Zarr of export_segmentations.
        level: The scale level.
        output_path: The path of the zip archive, by default get_tif_archive_path(level).
    """
    output_path = get_tif_archive_path(level) if output_path is None else output_path
    name = os.path.splitext(os.path.basename(output_path))[0]
    folder = os.path.join(os.path.dirname(output_path), f".{name}")
    assert not os.path.exists(folder), f"{folder} exists."
    os.makedirs(folder)
    export = zarr.open_group(export_path, mode="r")
    voxel_size = [vs / 1000 for vs in get_level_voxel_size(level)]

    files, counts = [], {}
    for image in ("cells", "nuclei"):
        t0 = time.time()
        array = export[image][f"s{level}"]
        ids = np.unique(export[image][f"s{max(level, TIF_RANGE_LEVEL)}"][:])
        counts[image] = int(np.count_nonzero(ids))
        path = os.path.join(folder, f"{image}_s{level}.tif")
        tifffile.imwrite(
            path, _iterate_planes(array), shape=array.shape, dtype=array.dtype, imagej=True, truncate=True,
            resolution=(1 / voxel_size[2], 1 / voxel_size[1]),
            metadata={"axes": "ZYX", "spacing": voxel_size[0], "unit": "micron", "min": 0, "max": int(ids.max())},
        )
        files.append(path)
        print(f"{image}: {array.shape} {array.dtype}, {os.path.getsize(path) / 1e9:.1f} GB in {time.time() - t0:.0f} s",
              flush=True)

    excluded = export.attrs.get("excluded", [])
    if excluded:
        files.append(os.path.join(folder, "excluded.csv"))
        with open(files[-1], "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(excluded[0]))
            writer.writeheader()
            writer.writerows(excluded)
    files.append(os.path.join(folder, "README.txt"))
    with open(files[-1], "w") as f:
        f.write(TIF_README.format(
            level=level, shape=tuple(array.shape), voxel_size=" x ".join(f"{vs:g}" for vs in voxel_size),
            n_cells=counts["cells"], n_nuclei=counts["nuclei"], size=os.path.getsize(files[0]) / 1e9,
            raw_note="" if level == 0 else f", level s{level} of its OME-Zarr",
            excluded=TIF_README_EXCLUDED if excluded else "", export_path=export_path,
        ))

    t0 = time.time()
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, arcname=os.path.join(name, os.path.basename(path)))
    shutil.rmtree(folder)
    print(
        f"Saved {output_path}: {[os.path.basename(path) for path in files]}, "
        f"{os.path.getsize(output_path) / 1e6:.0f} MB in {time.time() - t0:.0f} s", flush=True,
    )


def get_prompted_nuclei_path():
    return os.path.join(VOLUME_PREDICTION_DIR, "nuclei_s1_prompted.zarr")


def convert_points(points, level, target_level):
    """Map points (N, 3) in voxels of a level of the raw to voxels of another level.

    The levels are mean downsampled, so the center of the voxel i of a level is at factor * i + (factor - 1) / 2 in
    voxels of s0 (the translation of the OME metadata).
    """
    factor, target = np.array(get_level_factor(level)), np.array(get_level_factor(target_level))
    return (np.asarray(points) * factor + (factor - 1) / 2 - (target - 1) / 2) / target


def get_label_centers(labels, voxel_size):
    """The center of each object of a label volume: the voxel of it closest to its centroid (in nanometer)."""
    centers = {}
    for label_id, bb in enumerate(ndimage.find_objects(labels), 1):
        if bb is None:
            continue
        coords = np.argwhere(labels[bb] == label_id) + [b.start for b in bb]
        distances = (((coords - coords.mean(axis=0)) * voxel_size) ** 2).sum(axis=1)
        centers[label_id] = coords[np.argmin(distances)]
    return centers


def segment_from_points(predictor, volume, embeddings, prompts):
    """Segment one object with the SAM2 video predictor, conditioned on points in one or several slices.

    Args:
        predictor: The video predictor.
        volume: The volume the embeddings were computed for, (Z, Y, X).
        embeddings: Its embeddings.
        prompts: One (slice, points (N, 2) in y and x, labels (N,)) per conditioned slice.

    Returns:
        The mask of the propagation through the whole volume, in both directions.
    """
    from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D

    segmenter = PromptableSegmentation3D(predictor, volume, embeddings)
    for frame, points, labels in prompts:
        segmenter.add_prompt_set(
            frame_id=int(frame), points=np.array(points, dtype="float32"),
            point_labels=np.array(labels, dtype="int32"), object_id=1,
        )
    video_segments = segmenter.propagate_prompts()
    segmenter.reset_predictor()
    mask = np.zeros(volume.shape, dtype=bool)
    for frame, per_object in video_segments.items():
        mask[frame] = np.asarray(per_object[1]).squeeze()[:volume.shape[1], :volume.shape[2]]
    return mask


def get_component(mask, point):
    """The connected component of a mask that holds a point (empty if the point is not in the mask)."""
    return get_components(mask, [point])


def get_components(mask, points):
    """The connected components of a mask that hold one of the points (empty if no point is in the mask)."""
    labels, _ = ndimage.label(mask)
    label_ids = {int(labels[tuple(point)]) for point in points} - {0}
    return np.isin(labels, list(label_ids)) if label_ids else np.zeros_like(mask)


def segment_nuclei_from_prompts(
    zarr_path, points_path=NUCLEUS_POINTS_PATH, points_level=NUCLEUS_POINTS_LEVEL, z_offsets=(0,), n_negatives=0,
    num_workers=1,
):
    """Segment the nuclei that watershed_volume misses from one point each, with the nuclei model at s1.

    Each point is a positive prompt in its slice of the inverted raw. The SAM2 video predictor propagates it through
    a crop of NUCLEUS_PROMPT_CROP_SHAPE around the point in both directions, and the nucleus is the connected
    component of the result that holds the point: the propagation can jump to another nucleus once the prompted one
    ends, which is not connected to it. The existing nuclei that lie mostly in a prompted nucleus are fragments of
    it (a missed nucleus is often found in a few slices or tiles); they are recorded as 'replaces'.

    The nuclei get the ids after those of watershed_volume, in the order of the points, and are written into a
    label volume of s1 ('segmentation' of get_prompted_nuclei_path()).

    The two variants of the prompts did not improve the result on the first five points (nuclei_s3_points.csv): the
    three nuclei with a faint but visible envelope come out the same in every variant (IoU > 0.87 with the single
    point), the two rounded cells without a clear nucleus stay unstable in all of them.
    - z_offsets: the point is also set in the slices shifted by these offsets (the same y and x).
    - n_negatives: the centers of the nearest existing nuclei that are not fragments of the prompted one are added
      as negative points, projected onto each prompted slice. This needs the result of the single point to find the
      fragments and to drop the negatives that fall into the nucleus itself. They lower the confidence and fragment
      the mask, and with five slices one nucleus leaked into its neighbor.

    Args:
        zarr_path: The OME-Zarr with the raw data.
        points_path: The csv of the napari points layer (index, z, y, x).
        points_level: The scale level of the raw the points are given in.
        z_offsets: The slices to prompt, relative to the slice of the point.
        n_negatives: The number of negative points per prompted slice.
        num_workers: The number of threads for reading the existing nuclei.
    """
    from micro_sam.v2.util import precompute_image_embeddings

    level, target = 1, "nuclei"
    voxel_size = np.array(get_level_voxel_size(level))
    raw = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")
    existing = zarr.open_group(get_watershed_path(level, target), mode="r")["segmentation"]
    shape = raw.shape
    points = np.loadtxt(points_path, delimiter=",", skiprows=1, ndmin=2)[:, 1:]
    points = np.round(convert_points(points, points_level, level)).astype(int)
    assert all((points[:, i] >= 0).all() and (points[:, i] < shape[i]).all() for i in range(3)), points

    slabs = [slice(z, z + WS_BLOCK_SHAPE[0]) for z in range(0, shape[0], WS_BLOCK_SHAPE[0])]
    with ThreadPoolExecutor(num_workers) as pool:
        max_id = int(max(pool.map(lambda slab: existing[slab].max(), slabs)))

    predictor, _ = get_predictor_and_segmenter(model_type=MODELS[target], segmentation_mode="apg", ndim=3)
    out = zarr.open_group(get_prompted_nuclei_path(), mode="w", zarr_format=3)
    segmentation = out.create_array(
        "segmentation", shape=shape, dtype="uint64", chunks=WS_BLOCK_SHAPE, fill_value=0,
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"), dimension_names=["z", "y", "x"],
    )
    # The neighbors of a nucleus at the border of the crop have their center outside of it.
    margin = (64, 128, 128)
    records = []
    for index, point in enumerate(points):
        t0 = time.time()
        nucleus_id = max_id + 1 + index
        point = [int(p) for p in point]
        starts = [min(max(p - cs // 2, 0), sh - cs) for p, cs, sh in zip(point, NUCLEUS_PROMPT_CROP_SHAPE, shape)]
        bb = tuple(slice(start, start + cs) for start, cs in zip(starts, NUCLEUS_PROMPT_CROP_SHAPE))
        local = tuple(int(p - start) for p, start in zip(point, starts))
        outer = tuple(slice(max(b.start - m, 0), min(b.stop + m, sh)) for b, m, sh in zip(bb, margin, shape))
        offset = np.array([b.start - o.start for b, o in zip(bb, outer)])
        neighbors = existing[outer]
        neighbors_crop = neighbors[tuple(slice(o, o + cs) for o, cs in zip(offset, NUCLEUS_PROMPT_CROP_SHAPE))]

        volume = 255 - raw[bb]
        embeddings = precompute_image_embeddings(predictor, volume, ndim=3, verbose=False)
        with embeddings:
            mask = segment_from_points(predictor, volume, embeddings, [(local[0], [local[1:]], [1])])
            mask = get_component(mask, local)
            ids, counts = np.unique(neighbors_crop[mask], return_counts=True)
            sizes = dict(zip(*np.unique(neighbors, return_counts=True)))
            fragments = [
                int(i) for i, count in zip(ids, counts) if i > 0 and count > NUCLEUS_FRAGMENT_FRACTION * sizes[i]
            ]
            if tuple(z_offsets) != (0,) or n_negatives > 0:
                centers = {
                    i: center - offset for i, center in get_label_centers(neighbors, voxel_size).items()
                    if i not in fragments
                }
                nearest = sorted(centers.values(), key=lambda c: (((c - np.array(local)) * voxel_size) ** 2).sum())
                prompts = []
                for z_offset in z_offsets:
                    z = local[0] + z_offset
                    prompt_points, prompt_labels = [local[1:]], [1]
                    for _, y, x in nearest[:n_negatives]:
                        inside = 0 <= y < volume.shape[1] and 0 <= x < volume.shape[2]
                        if inside and not mask[z, y, x]:
                            prompt_points.append((y, x))
                            prompt_labels.append(0)
                    prompts.append((z, prompt_points, prompt_labels))
                mask = get_component(segment_from_points(predictor, volume, embeddings, prompts), local)

        current = segmentation[bb]
        free = mask & (current == 0)
        current[free] = nucleus_id
        segmentation[bb] = current
        coords = np.nonzero(mask.any(axis=(1, 2)))[0]
        z_extent = [int(coords.min()) + starts[0], int(coords.max()) + starts[0]] if len(coords) > 0 else None
        covered = float((neighbors_crop[mask] > 0).mean()) if mask.any() else 0.0
        records.append({
            "id": nucleus_id, "point": point, "size": int(mask.sum()), "z_extent": z_extent,
            "replaces": fragments, "fraction_in_existing": covered, "n_written": int(free.sum()),
        })
        print(
            f"Nucleus {nucleus_id} at {records[-1]['point']} (ZYX of s{level}): {mask.sum()} voxels, slices "
            f"{z_extent}, {covered:.2f} of it in existing nuclei, replaces {fragments}, in {time.time() - t0:.0f} s",
            flush=True,
        )
    out.attrs.update({
        "model_type": MODELS[target], "level": level, "voxel_size": list(voxel_size), "inverted": True,
        "points_path": points_path, "points_level": points_level, "crop_shape": list(NUCLEUS_PROMPT_CROP_SHAPE),
        "z_offsets": list(z_offsets), "n_negatives": n_negatives, "nuclei": get_watershed_path(level, target),
        "max_existing_id": max_id, "objects": records,
    })


def _merge_nuclei(nuclei, prompted, removed):
    nuclei = np.where(np.isin(nuclei, removed), 0, nuclei)
    return np.where(prompted > 0, prompted, nuclei).astype("uint64")


def merge_prompted_nuclei(num_workers):
    """Merge the prompted nuclei of segment_nuclei_from_prompts into the nuclei of watershed_volume (s1).

    The existing nuclei that a prompted nucleus replaces (its fragments, 'replaces' in the attributes of the prompted
    nuclei) are removed and the prompted nuclei are written with their ids, so all other ids stay the same. The
    result is the array 'segmentation' of get_merged_nuclei_path(). Its attributes list the removed ids and the
    prompted nuclei; those of NUCLEUS_SEED_ONLY_POINTS have 'seed_only' and a comment, they are not full nuclei.

    Args:
        num_workers: The number of threads.
    """
    from functools import partial
    from bioimage_py.copy import copy
    from bioimage_py.wrapper import MultiTransformationSource

    zarr.config.set({"threading.max_workers": num_workers})
    nuclei = zarr.open_group(get_watershed_path(1, "nuclei"), mode="r")["segmentation"]
    prompted_group = zarr.open_group(get_prompted_nuclei_path(), mode="r")
    prompted = prompted_group["segmentation"]
    objects = [dict(obj) for obj in prompted_group.attrs["objects"]]
    removed = sorted(nucleus_id for obj in objects for nucleus_id in obj["replaces"])
    for index, obj in enumerate(objects):
        obj["seed_only"] = index in NUCLEUS_SEED_ONLY_POINTS
        if obj["seed_only"]:
            obj["comment"] = NUCLEUS_SEED_ONLY_COMMENT

    out = zarr.open_group(get_merged_nuclei_path(), mode="w", zarr_format=3)
    merged = out.create_array(
        "segmentation", shape=nuclei.shape, dtype="uint64", chunks=nuclei.chunks, fill_value=0,
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"), dimension_names=["z", "y", "x"],
    )
    t0 = time.time()
    source = MultiTransformationSource(partial(_merge_nuclei, removed=removed), nuclei, prompted, dtype="uint64")
    copy(source, merged, block_shape=nuclei.chunks, num_workers=num_workers)
    out.attrs.update({
        "nuclei": get_watershed_path(1, "nuclei"), "prompted": get_prompted_nuclei_path(), "level": 1,
        "voxel_size": list(get_level_voxel_size(1)), "removed": removed, "objects": objects,
        "seed_only": [obj["id"] for obj in objects if obj["seed_only"]],
    })
    print(f"Merged {[obj['id'] for obj in objects]}, removed {removed}, in {time.time() - t0:.0f} s", flush=True)
    for obj in objects:
        if obj["seed_only"]:
            print(f"Nucleus {obj['id']}: {NUCLEUS_SEED_ONLY_COMMENT}", flush=True)


def get_label_boxes(labels, label_ids, num_workers):
    """The bounding boxes (start and stop per axis) of some objects of a label volume, read in slabs."""
    n_slices, step = labels.shape[0], WS_BLOCK_SHAPE[0]
    slabs = [slice(z, min(z + step, n_slices)) for z in range(0, n_slices, step)]

    def get_slab_boxes(slab):
        data = labels[slab]
        boxes = {}
        for label_id in label_ids:
            coords = np.nonzero(data == label_id)
            if len(coords[0]) > 0:
                offset = (slab.start, 0, 0)
                boxes[label_id] = (
                    [int(c.min()) + o for c, o in zip(coords, offset)],
                    [int(c.max()) + 1 + o for c, o in zip(coords, offset)],
                )
        return boxes

    boxes = {}
    with ThreadPoolExecutor(num_workers) as pool:
        for slab_boxes in pool.map(get_slab_boxes, slabs):
            for label_id, (start, stop) in slab_boxes.items():
                previous = boxes.get(label_id, (start, stop))
                boxes[label_id] = (
                    [min(a, b) for a, b in zip(previous[0], start)], [max(a, b) for a, b in zip(previous[1], stop)],
                )
    return boxes


def join_nucleus_fragments(zarr_path, groups=NUCLEUS_FRAGMENT_GROUPS, num_workers=1):
    """Make one object of each nucleus that is split into fragments in the nuclei of merge_prompted_nuclei (s1).

    The nucleus of each group of fragments is segmented with the SAM2 video predictor of the nuclei model: the
    center of each fragment (the voxel of it closest to its centroid) is a positive point in its slice of the
    inverted raw, in a crop of NUCLEUS_PROMPT_CROP_SHAPE around the group, and the nucleus is made of the connected
    components of the result that hold a point. The fragments of NUCLEUS_FRAGMENT_NO_PROMPT are not used as prompts.
    The prompted nucleus and all fragments of the group get the lowest id of the group, so all other ids stay the
    same; voxels of other nuclei are not overwritten. The result is the array 'segmentation' of
    get_joined_nuclei_path(). Its attributes take over the prompted nuclei of the merged nuclei ('objects',
    'seed_only') and list the groups ('joined').

    On the four groups of NUCLEUS_FRAGMENT_GROUPS the prompts of single fragments give the same nucleus as all of
    them together (IoU > 0.95) for three groups. The exception is the fragment 169, see NUCLEUS_FRAGMENT_NO_PROMPT.

    Args:
        zarr_path: The OME-Zarr with the raw data.
        groups: The ids of the fragments of each nucleus.
        num_workers: The number of threads.
    """
    from bioimage_py.copy import copy
    from micro_sam.v2.util import precompute_image_embeddings

    zarr.config.set({"threading.max_workers": num_workers})
    level, target = 1, "nuclei"
    voxel_size = np.array(get_level_voxel_size(level))
    raw = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r")
    merged_group = zarr.open_group(get_merged_nuclei_path(), mode="r")
    nuclei = merged_group["segmentation"]
    shape = raw.shape
    all_ids = [nucleus_id for group in groups for nucleus_id in group]
    assert len(set(all_ids)) == len(all_ids), groups
    boxes = get_label_boxes(nuclei, all_ids, num_workers)
    assert set(boxes) == set(all_ids), f"Not in the nuclei: {sorted(set(all_ids) - set(boxes))}"

    out = zarr.open_group(get_joined_nuclei_path(), mode="w", zarr_format=3)
    joined = out.create_array(
        "segmentation", shape=shape, dtype="uint64", chunks=nuclei.chunks, fill_value=0,
        compressors=BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle"), dimension_names=["z", "y", "x"],
    )
    copy(nuclei, joined, block_shape=nuclei.chunks, num_workers=num_workers)

    predictor, _ = get_predictor_and_segmenter(model_type=MODELS[target], segmentation_mode="apg", ndim=3)
    records = []
    for group in groups:
        t0 = time.time()
        start = np.min([boxes[nucleus_id][0] for nucleus_id in group], axis=0)
        stop = np.max([boxes[nucleus_id][1] for nucleus_id in group], axis=0)
        assert all(stop - start <= NUCLEUS_PROMPT_CROP_SHAPE), f"The group {group} does not fit into the crop."
        starts = [
            int(min(max(center - cs // 2, 0), sh - cs))
            for center, cs, sh in zip((start + stop) // 2, NUCLEUS_PROMPT_CROP_SHAPE, shape)
        ]
        bb = tuple(slice(st, st + cs) for st, cs in zip(starts, NUCLEUS_PROMPT_CROP_SHAPE))
        labels = joined[bb]
        centers = get_label_centers(labels, voxel_size)
        prompt_ids = [nucleus_id for nucleus_id in group if nucleus_id not in NUCLEUS_FRAGMENT_NO_PROMPT]
        prompts = [(centers[nucleus_id][0], [centers[nucleus_id][1:]], [1]) for nucleus_id in prompt_ids]

        volume = 255 - raw[bb]
        embeddings = precompute_image_embeddings(predictor, volume, ndim=3, verbose=False)
        with embeddings:
            mask = segment_from_points(predictor, volume, embeddings, prompts)
        mask = get_components(mask, [centers[nucleus_id] for nucleus_id in prompt_ids])

        fragments = np.isin(labels, group)
        other = int((mask & (labels > 0) & ~fragments).sum())
        fractions = {str(nucleus_id): float(mask[labels == nucleus_id].mean()) for nucleus_id in group}
        nucleus = (mask & (labels == 0)) | fragments
        new_id = min(group)
        labels[nucleus] = new_id
        joined[bb] = labels
        coords = np.nonzero(nucleus.any(axis=(1, 2)))[0]
        record = {
            "id": new_id, "fragments": [int(nucleus_id) for nucleus_id in group],
            "prompts": {str(nucleus_id): [int(c + st) for c, st in zip(centers[nucleus_id], starts)]
                        for nucleus_id in prompt_ids},
            "size": int(nucleus.sum()), "size_prompted": int(mask.sum()), "size_fragments": int(fragments.sum()),
            "fragment_fraction_in_prompted": fractions,
            "n_voxels_in_other_nuclei": other,
            "z_extent": [int(coords.min()) + starts[0], int(coords.max()) + starts[0]],
            "comments": {str(i): NUCLEUS_FRAGMENT_NO_PROMPT[i] for i in group if i in NUCLEUS_FRAGMENT_NO_PROMPT},
        }
        records.append(record)
        print(
            f"Nucleus {new_id} from the fragments {list(group)} (prompts at {prompt_ids}): {record['size']} voxels "
            f"({record['size_fragments']} in the fragments, {record['size_prompted']} prompted), slices "
            f"{record['z_extent']}, fraction of the fragments in the prompted nucleus "
            f"{({k: round(v, 3) for k, v in record['fragment_fraction_in_prompted'].items()})}, {other} prompted "
            f"voxels in other nuclei (left as they are), in {time.time() - t0:.0f} s", flush=True,
        )
        for comment in record["comments"].items():
            print(f"  Fragment {comment[0]}: {comment[1]}", flush=True)
    out.attrs.update({
        "nuclei": get_merged_nuclei_path(), "level": level, "voxel_size": list(voxel_size),
        "model_type": MODELS[target], "inverted": True, "crop_shape": list(NUCLEUS_PROMPT_CROP_SHAPE),
        "objects": merged_group.attrs["objects"], "seed_only": merged_group.attrs["seed_only"],
        "removed": merged_group.attrs["removed"], "joined": records,
    })


def get_border_seed_check_path():
    return os.path.join(VOLUME_PREDICTION_DIR, f"cells_s{BORDER_SEED_CHECK_LEVEL}_border_seed_check.zarr")


def get_tissue_mask(zarr_path, z_range):
    """A tissue mask of the slices z_range of TISSUE_LEVEL from the texture of the raw alone.

    The resin is smooth and the cells are not: the local standard deviation of the raw in a window of
    TISSUE_STD_SIZE px in the plane (the sections differ in brightness) is thresholded with Otsu over the whole
    volume, and each slice is closed, filled (the nuclei are smooth as well) and opened, which removes the seams of
    the acquisition tiles and the cracks of the resin.
    """
    from skimage.filters import threshold_otsu
    from skimage.morphology import disk

    raw = zarr.open_array(os.path.join(zarr_path, f"s{TISSUE_LEVEL}"), mode="r")[:].astype("float32")
    size = (1, TISSUE_STD_SIZE, TISSUE_STD_SIZE)
    mean = ndimage.uniform_filter(raw, size)
    std = np.sqrt(np.maximum(ndimage.uniform_filter(raw * raw, size) - mean * mean, 0))
    textured = std[z_range[0]:z_range[1]] > threshold_otsu(std[::2, ::2, ::2])
    tissue = np.zeros(textured.shape, dtype=bool)
    for z, plane in enumerate(textured):
        plane = ndimage.binary_fill_holes(ndimage.binary_closing(plane, disk(TISSUE_CLOSING_RADIUS)))
        tissue[z] = ndimage.binary_opening(plane, disk(TISSUE_OPENING_RADIUS))
    return tissue


def check_border_seed(zarr_path, z_ranges, num_workers):
    """Test the background seed of the cell segmentation on slabs of s0 over the full Y and X extent.

    The seeds (expanded nuclei) and the closed mask are those of segment_cells_volume, so only the watershed runs:
    - reference: the watershed of the whole slab without a background seed.
    - whole slab: the same with the mask voxels on the Y and X faces as seed, its region set to zero. It shows what
      the seed does: which cells lose volume, and whether a voxel moves from one cell to another.
    - block-wise: the stages 'background' and 'segmentation' of segment_cells_blockwise with the blocks and the halo
      of segment_cells_volume (the face seed propagated over two passes), compared to the whole slab.
    The block-wise result is then compared to the tissue mask from the texture of the raw (get_tissue_mask), on the
    grid of TISSUE_LEVEL (the first voxel of each block): the resin that is still in a cell, and the tissue that
    went to the background, in total and more than TISSUE_SURFACE_MARGIN px inside the tissue. The surface is left
    out there since the window of the texture widens the tissue, and since the cells have protrusions the cells
    model does not mark as foreground. The largest pieces of tissue in the background are listed, those of the
    size of a cell are cells without a nucleus seed.

    The block-wise segmentation and the region of the face seed are written into the slab of two arrays of s0
    (get_border_seed_check_path()), the numbers into its attributes. View them with
    'downsample_segmentations --names cells_border_seed' and view_beghin_em_volume.py.
    """
    import bioimage_cpp as bic
    from skimage.morphology import disk

    level = BORDER_SEED_CHECK_LEVEL
    zarr.config.set({"threading.max_workers": num_workers})
    voxel_size = get_level_voxel_size(level)
    source = zarr.open_group(get_cell_volume_path(level), mode="r")
    boundary_source = zarr.open_group(get_volume_prediction_path(level, "cells"), mode="r")["boundary"]
    shape = tuple(boundary_source.shape)
    out = zarr.open_group(get_border_seed_check_path(), mode="a", zarr_format=3)
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    arrays = {
        name: out.require_array(
            name, shape=shape, dtype=dtype, chunks=CELL_VOLUME_CHUNK_SHAPE, compressors=compressor, fill_value=0,
            dimension_names=["z", "y", "x"],
        ) for name, dtype in (("segmentation", "uint64"), ("background", "uint8"))
    }
    factor = [f2 // f for f2, f in zip(get_level_factor(TISSUE_LEVEL), get_level_factor(level))]
    cubic_um = np.prod(voxel_size) / 1e9

    for z0, z1 in z_ranges:
        t0 = time.time()
        slab = slice(z0, z1)
        seeds, mask = source["seeds"][slab].astype("uint32"), source["mask"][slab].astype(bool)
        boundary = boundary_source[slab].astype("float32")
        print(f"s{level} z {z0}:{z1}: read in {time.time() - t0:.0f} s, mask {mask.mean():.3f} of the slab", flush=True)

        t0 = time.time()
        reference = bic.segmentation.watershed(boundary, seeds, mask=mask)
        face_seeds = seeds.copy()
        add_border_seed(face_seeds, mask)
        whole = bic.segmentation.watershed(boundary, face_seeds, mask=mask)
        del face_seeds
        region = whole == CELL_BACKGROUND_SEED_ID
        whole[region] = 0
        n_ids = int(reference.max()) + 1
        size_before, size_after = np.bincount(reference.ravel(), minlength=n_ids), np.bincount(whole.ravel())
        size_after = np.concatenate([size_after, np.zeros(n_ids - len(size_after), dtype=size_after.dtype)])
        ids = np.nonzero(size_before)[0][1:]
        shrunk = [
            (int(i), round(float(size_before[i] * cubic_um)), round(float(size_after[i] * cubic_um)))
            for i in ids if size_after[i] < 0.99 * size_before[i]
        ]
        n_moved = int(((reference != whole) & (whole > 0)).sum())
        stats = {
            "mask_fraction": float(mask.mean()), "region_fraction_of_mask": float(region.sum() / mask.sum()),
            "n_cells": len(ids), "shrunk_cells_id_before_after_um3": shrunk, "n_voxels_moved_between_cells": n_moved,
            "max_cell_um3_before": float(size_before[ids].max() * cubic_um),
            "max_cell_um3_after": float(size_after[ids].max() * cubic_um),
        }
        print(
            f"  whole slab ({time.time() - t0:.0f} s): the face seed takes {stats['region_fraction_of_mask']:.3f} of "
            f"the mask; {len(shrunk)} of {len(ids)} cells lose more than 1% (id, um^3 before, after): {shrunk}; "
            f"{n_moved} voxels move from one cell to another; largest cell {stats['max_cell_um3_before']:.0f} -> "
            f"{stats['max_cell_um3_after']:.0f} um^3", flush=True,
        )
        del reference

        t0 = time.time()
        outputs = {
            "seeds": seeds, "mask": mask.astype("uint8"), "background": np.zeros(seeds.shape, "uint8"),
            "segmentation": np.zeros(seeds.shape, "uint64"),
        }
        segment_cells_blockwise(
            None, boundary, None, outputs, voxel_size, num_workers, stages=("background", "segmentation"),
        )
        blockwise = outputs["segmentation"]
        stats["blockwise_differs_fraction_of_mask"] = float((blockwise != whole)[mask].mean())
        print(
            f"  block-wise ({time.time() - t0:.0f} s): differs from the whole slab in "
            f"{stats['blockwise_differs_fraction_of_mask']:.6f} of the mask", flush=True,
        )
        arrays["segmentation"][slab] = blockwise
        arrays["background"][slab] = outputs["background"]
        del boundary, seeds, outputs

        # On the grid of the tissue level: the first voxel of each block of the factor.
        z_tissue = range(-(-z0 // factor[0]), -(-z1 // factor[0]))
        sub = (slice(z_tissue.start * factor[0] - z0, None, factor[0]),)
        sub = sub + tuple(slice(None, None, f) for f in factor[1:])
        tissue = get_tissue_mask(zarr_path, (z_tissue.start, z_tissue.stop))
        cells, mask = blockwise[sub] > 0, mask[sub]
        background = mask & ~cells
        del blockwise, whole, region
        cells_before = source["segmentation"][slab][sub] > 0
        core = np.stack([ndimage.binary_erosion(plane, disk(TISSUE_SURFACE_MARGIN)) for plane in tissue])
        far = ~np.stack([ndimage.binary_dilation(plane, disk(TISSUE_SURFACE_MARGIN)) for plane in tissue])
        stats.update({
            "tissue_fraction": float(tissue.mean()),
            "cells_outside_tissue_before": float((cells_before & ~tissue).sum() / cells_before.sum()),
            "cells_outside_tissue_after": float((cells & ~tissue).sum() / cells.sum()),
            "cells_far_outside_tissue_before": float((cells_before & far).sum() / cells_before.sum()),
            "cells_far_outside_tissue_after": float((cells & far).sum() / cells.sum()),
            "resin_mask_to_background": float((background & ~tissue).sum() / (mask & ~tissue).sum()),
            "tissue_in_background": float((background & tissue).sum() / tissue.sum()),
            "tissue_core_in_background": float((background & core).sum() / core.sum()),
        })
        print(
            f"  texture: tissue {stats['tissue_fraction']:.3f} of the slab; cell voxels outside the tissue "
            f"{stats['cells_outside_tissue_before']:.4f} -> {stats['cells_outside_tissue_after']:.4f}, more than "
            f"{TISSUE_SURFACE_MARGIN} px outside {stats['cells_far_outside_tissue_before']:.4f} -> "
            f"{stats['cells_far_outside_tissue_after']:.4f}; {stats['resin_mask_to_background']:.4f} of the mask "
            f"outside the tissue is background now; tissue in the background {stats['tissue_in_background']:.4f}, "
            f"of the tissue more than {TISSUE_SURFACE_MARGIN} px inside {stats['tissue_core_in_background']:.4f}",
            flush=True,
        )
        # The pieces of tissue in the background, as position in voxels of s1 (the level of the nuclei).
        tissue_voxel = np.prod(get_level_voxel_size(TISSUE_LEVEL)) / 1e9
        to_s1 = [f2 // f1 for f2, f1 in zip(get_level_factor(TISSUE_LEVEL), get_level_factor(1))]
        labels, n_labels = ndimage.label(background & core)
        sizes = np.bincount(labels.ravel())[1:]
        pieces = []
        for label_id in np.argsort(sizes)[::-1][:5] + 1 if n_labels > 0 else []:
            center = np.array(ndimage.center_of_mass(labels == label_id)) + [z_tissue.start, 0, 0]
            pieces.append({
                "um3": round(float(sizes[label_id - 1] * tissue_voxel)),
                "center_s1": [int(c * f) for c, f in zip(center, to_s1)],
            })
        stats["tissue_core_in_background_pieces"] = pieces
        print(f"  largest pieces of tissue (more than {TISSUE_SURFACE_MARGIN} px inside) in the background: {pieces}",
              flush=True)
        slabs = dict(out.attrs.get("slabs", {}))
        slabs[f"{z0}:{z1}"] = stats
        out.attrs["slabs"] = slabs
    out.attrs.update({
        "level": level, "seeds_and_mask": get_cell_volume_path(level), "block_shape": list(CELL_VOLUME_BLOCK_SHAPE),
        "ws_halo": list(CELL_VOLUME_WS_HALO),
    })


def check_cells_volume(level=0, num_workers=1, block_shapes=((32, 256, 256),), ws_halos=(CELL_VOLUME_WS_HALO,)):
    """Compare the block-wise stages of segment_cells_volume to watershed_cell_crop on the cell crop.

    The inputs are those of watershed_cell_crop (the AIS predictions of the crop and the nuclei of the prompts),
    so any difference comes from the block-wise processing, or from the distance transform of bioimage-cpp
    instead of scipy for a result that watershed_cell_crop wrote before close_mask and expand_nuclei used it.
    """
    with h5py.File(get_cell_watershed_path(level), "r") as f:
        reference = {"seeds": f["seeds"][:], "mask": f["mask"][:].astype(bool), "segmentation": f["segmentation"][:]}
        assert f.attrs["closing_radius"] == CELL_CLOSING_RADIUS and f.attrs["expand_distance"] == CELL_EXPAND_DISTANCE
    with h5py.File(get_cell_crop_path(level, "ais"), "r") as f:
        foreground = f["prediction/foreground"][:]
        boundary = f["prediction/boundary"][:]
    with h5py.File(get_nucleus_prompts_path(), "r") as f:
        nuclei_s1 = f["nuclei"][:].astype("uint64")
    factor = [f1 // f for f1, f in zip(get_level_factor(1), get_level_factor(level))]
    nuclei = repeat_labels(nuclei_s1, factor)
    assert np.array_equal(nuclei[tuple(slice(None) for _ in factor)], load_cell_crop_nuclei(level))
    voxel_size = get_level_voxel_size(level)

    for block_shape, ws_halo in zip(block_shapes, ws_halos):
        outputs = {
            "seeds": np.zeros(foreground.shape, "uint64"), "mask": np.zeros(foreground.shape, "uint8"),
            "segmentation": np.zeros(foreground.shape, "uint64"),
        }
        timings = segment_cells_blockwise(
            foreground, boundary, nuclei, outputs, voxel_size, num_workers, block_shape=block_shape, ws_halo=ws_halo,
        )
        fg = reference["mask"] | outputs["mask"].astype(bool)
        print(
            f"blocks {block_shape}, watershed halo {ws_halo}: "
            f"seeds differ in {np.sum(outputs['seeds'] != reference['seeds'])} voxels, "
            f"mask in {np.sum(outputs['mask'].astype(bool) != reference['mask'])} voxels, "
            f"segmentation in {np.sum(outputs['segmentation'] != reference['segmentation'])} voxels "
            f"({np.mean(outputs['segmentation'][fg] != reference['segmentation'][fg]):.5f} of the mask), "
            f"{', '.join(f'{k} in {v:.0f} s' for k, v in timings.items())}", flush=True,
        )


def seed_cell_crop(levels):
    """Segment the cells of the cell crop with the nuclei as seeds, from the AIS predictions of segment_cell_crop.

    The nuclei are those of the prompts (nucleus_prompts.h5, at s1), repeated 2 x 2 in y and x for s0. Saves the
    watershed of the expanded nuclei ('watershed', with its 'seeds'), the over-segmentation and the graph watershed
    ('graph_watershed') of each level in one h5 file.
    """
    from micro_sam.v2.postprocessing import flow_instance_segmentation

    for level in levels:
        path = get_cell_seeded_path(level)
        if os.path.exists(path):
            print(f"{path} exists, skipping s{level}")
            continue
        nuclei = load_cell_crop_nuclei(level)
        voxel_size = get_level_voxel_size(level)
        with h5py.File(get_cell_crop_path(level, "ais"), "r") as f:
            prediction = np.stack([f[f"prediction/{name}"][:].astype("float32") for name in PREDICTION_CHANNELS])
            attrs = dict(f.attrs)
        assert nuclei.shape == prediction.shape[1:], (nuclei.shape, prediction.shape)
        foreground, boundary = prediction[0], prediction[4]

        t0 = time.time()
        watershed, seeds, _ = nucleus_watershed(foreground, boundary, nuclei, voxel_size)
        t_watershed = time.time() - t0

        t0 = time.time()
        params = dict(CELL_OVERSEGMENTATION)
        params["min_size"] = int(params["min_size"] * np.prod(get_level_voxel_size(1)) / np.prod(voxel_size))
        oversegmentation = flow_instance_segmentation(
            foreground, prediction[1:4], model_type=MODELS["cells"], boundary=boundary,
            foreground_threshold=CELL_FOREGROUND_THRESHOLD, **params,
        )
        graph_segmentation, oversegmentation, stats = graph_watershed(oversegmentation, boundary, nuclei)
        t_graph = time.time() - t0

        with h5py.File(path + ".tmp", "w") as f:
            f.attrs.update({
                "level": level, "bb_start": attrs["bb_start"], "bb_stop": attrs["bb_stop"], "voxel_size": voxel_size,
                "predictions": get_cell_crop_path(level, "ais"), "nuclei": get_nucleus_prompts_path(),
                "foreground_threshold": CELL_FOREGROUND_THRESHOLD, "expand_distance": CELL_EXPAND_DISTANCE,
                "nucleus_fraction": CELL_NUCLEUS_FRACTION, "nucleus_edge_factor": CELL_NUCLEUS_EDGE_FACTOR,
                **{f"oversegmentation_{key}": value for key, value in params.items()}, **stats,
                "time_watershed": t_watershed, "time_graph_watershed": t_graph,
            })
            for name, data in (
                ("seeds", seeds), ("watershed", watershed), ("oversegmentation", oversegmentation),
                ("graph_watershed", graph_segmentation),
            ):
                f.create_dataset(name, data=data.astype("uint32"), chunks=True, compression="gzip")
        os.rename(path + ".tmp", path)
        n_objects = {
            name: len(np.unique(data)) - 1 for name, data in (("watershed", watershed), ("graph", graph_segmentation))
        }
        print(
            f"s{level}: watershed {n_objects['watershed']} cells in {t_watershed:.0f} s, graph watershed "
            f"{n_objects['graph']} cells in {t_graph:.0f} s, {stats}", flush=True,
        )


def plot_predictions(zarr_path, output_dir, level, target, plot_path):
    """Plot the raw crop and each decoder output channel of one model, one row per crop."""
    n_cols = len(PREDICTION_CHANNELS) + 1
    shape = zarr.open_array(os.path.join(zarr_path, f"s{level}"), mode="r").shape
    crops = get_level_crops(level, shape)
    fig, axes = plt.subplots(len(crops), n_cols, figsize=(4 * n_cols, 4 * len(crops)), squeeze=False)
    for row, start in zip(axes, crops):
        with h5py.File(os.path.join(output_dir, f"{get_crop_name(start)}.h5"), "r") as f:
            row[0].imshow(f["raw"][:], cmap="gray")
            row[0].set_title(f"s{level} {get_crop_name(start)}")
            prediction = f[f"prediction/{target}"][:]
        for ax, channel, name in zip(row[1:], prediction, PREDICTION_CHANNELS):
            im = ax.imshow(channel, cmap="viridis")
            ax.set_title(f"{target} {name}")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=80)
    print(f"Saved the plot to {plot_path}")


def upsample_z(volume, factor):
    """Interpolate the volume linearly in z, slice i of the input is slice factor * i of the output."""
    n_out = factor * (volume.shape[0] - 1) + 1
    upsampled = np.zeros((n_out,) + volume.shape[1:], dtype=volume.dtype)
    for z in range(n_out):
        lower, rest = divmod(z, factor)
        if rest == 0:
            upsampled[z] = volume[lower]
            continue
        weight = rest / factor
        upsampled[z] = np.round((1 - weight) * volume[lower] + weight * volume[lower + 1]).astype(volume.dtype)
    return upsampled


def segment_3d(zarr_path, output_dir, names):
    """Run the 3d segmentations of RUNS_3D, one h5 file per run with the raw data and one segmentation per engine."""
    os.makedirs(output_dir, exist_ok=True)
    for name in names:
        run = RUNS_3D[name]
        raw = zarr.open_array(os.path.join(zarr_path, f"s{run['level']}"), mode="r")
        bb = tuple(slice(None) for _ in raw.shape) if run["bb"] is None else run["bb"]
        volume = raw[bb]
        path = os.path.join(output_dir, f"{name}.h5")
        with h5py.File(path, "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", data=volume, chunks=True, compression="gzip")
                f["raw"].attrs["scale_level"] = run["level"]
                f["raw"].attrs["start"] = [b.start or 0 for b in bb]
            done = set(f["segmentation"]) if "segmentation" in f else set()
        # The file is closed during the segmentation, so the finished results can be read while the next one runs.
        for engine in run["engines"]:
            key = f"{run['target']}_{engine}"
            if key in done:
                continue
            t0 = time.time()
            factor = run.get("z_upsample", 1)
            # The segmentation of the upsampled volume is taken back at the input slices.
            input_volume = volume if factor == 1 else upsample_z(volume, factor)
            segmentation = segment_image(input_volume, MODELS[run["target"]], engine, ndim=3)[::factor]
            with h5py.File(path, "a") as f:
                f.create_dataset(f"segmentation/{key}", data=segmentation, chunks=True, compression="gzip")
            n_objects = len(np.unique(segmentation)) - 1
            print(f"{name} {key}: {n_objects} objects in {time.time() - t0:.1f} s, volume {volume.shape}")


def overlay_labels(image, segmentation, colors):
    image = image.astype("float32")
    image = (image - image.min()) / max(image.max() - image.min(), 1)
    overlay = np.stack([image] * 3, axis=-1)
    foreground = segmentation > 0
    overlay[foreground] = 0.55 * overlay[foreground] + 0.45 * colors[segmentation[foreground]]
    boundaries = find_boundaries(segmentation, mode="thick")
    overlay[boundaries] = colors[segmentation[boundaries]]
    return overlay


def plot_3d(output_dir, name, plot_path):
    """Plot the central xy, xz and yz planes of the raw data and of each segmentation of a 3d run."""
    with h5py.File(os.path.join(output_dir, f"{name}.h5"), "r") as f:
        raw = f["raw"][:]
        segmentations = {key: f[f"segmentation/{key}"][:] for key in f["segmentation"]}
    center = [s // 2 for s in raw.shape]
    planes = {
        f"xy, z={center[0]}": lambda v: v[center[0]],
        f"xz, y={center[1]}": lambda v: v[:, center[1]],
        f"yz, x={center[2]}": lambda v: v[:, :, center[2]].T,
    }
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(segmentations) + 1, len(planes), figsize=(18, 6 * (len(segmentations) + 1)))
    for ax, (plane, get_plane) in zip(axes[0], planes.items()):
        ax.imshow(get_plane(raw), cmap="gray")
        ax.set_title(f"{name} raw {raw.shape}: {plane}")
    for row, (key, segmentation) in zip(axes[1:], segmentations.items()):
        colors = rng.uniform(0.0, 1.0, size=(segmentation.max() + 1, 3))
        n_objects = len(np.unique(segmentation)) - 1
        for ax, (plane, get_plane) in zip(row, planes.items()):
            ax.imshow(overlay_labels(get_plane(raw), get_plane(segmentation), colors))
            ax.set_title(f"{key} ({n_objects} objects): {plane}")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=80)
    print(f"Saved the plot to {plot_path}")


def plot_size_gallery(output_dir, name, key, max_size, plot_path, margin=48, n_cols=8):
    """Plot the size histogram and the xy and xz plane through the center of every object below max_size voxels."""
    with h5py.File(os.path.join(output_dir, f"{name}.h5"), "r") as f:
        raw = f["raw"][:]
        segmentation = f[f"segmentation/{key}"][:]
    ids, sizes = np.unique(segmentation[segmentation > 0], return_counts=True)
    boxes = ndimage.find_objects(segmentation)
    small = [(size, i) for i, size in zip(ids, sizes) if size < max_size]
    small.sort()

    n_rows = 1 + 2 * int(np.ceil(len(small) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.5 * n_rows))
    for ax in axes.flat:
        ax.axis("off")
    hist = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols, fig=fig)
    hist.hist(np.log10(sizes), bins=40)
    hist.axvline(np.log10(max_size), color="k", linestyle="--")
    hist.set_xlabel("log10 object size (voxels)")
    hist.set_title(f"{name} {key}: {len(ids)} objects, {len(small)} below {max_size:.0e} voxels")

    for index, (size, i) in enumerate(small):
        box = boxes[i - 1]
        border = any(b.start == 0 or b.stop == s for b, s in zip(box, segmentation.shape))
        z, y, x = [(b.start + b.stop) // 2 for b in box]
        ys = slice(max(box[1].start - margin, 0), box[1].stop + margin)
        xs = slice(max(box[2].start - margin, 0), box[2].stop + margin)
        zs = slice(max(box[0].start - margin // 4, 0), box[0].stop + margin // 4)
        row, col = 1 + 2 * (index // n_cols), index % n_cols
        for ax, image, mask in (
            (axes[row, col], raw[z, ys, xs], segmentation[z, ys, xs] == i),
            (axes[row + 1, col], raw[zs, y, xs], segmentation[zs, y, xs] == i),
        ):
            overlay = np.stack([image / 255.0] * 3, axis=-1)
            outline = find_boundaries(mask, mode="thick")
            overlay[outline] = (1.0, 0.2, 0.2)
            ax.imshow(overlay)
        axes[row, col].set_title(f"id {i}: {size / 1e3:.0f}k{' border' if border else ''}", fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=70)
    print(f"Saved the plot to {plot_path}")


def get_z_contacts(segmentation):
    """Get the overlap of all object pairs that touch from one slice to the next.

    The overlap of objects a in slice z and b in slice z + 1 is their common xy area divided by the smaller of the two
    areas. The overlap of a pair is the maximum over all slices.
    """
    contacts = {}
    for z in range(segmentation.shape[0] - 1):
        upper, lower = segmentation[z].ravel(), segmentation[z + 1].ravel()
        touching = (upper > 0) & (lower > 0) & (upper != lower)
        if not touching.any():
            continue
        area_upper = np.bincount(upper)
        area_lower = np.bincount(lower)
        pairs, counts = np.unique(np.stack([upper[touching], lower[touching]]), axis=1, return_counts=True)
        for (a, b), count in zip(pairs.T, counts):
            overlap = count / min(area_upper[a], area_lower[b])
            key = (min(a, b), max(a, b))
            contacts[key] = max(contacts.get(key, 0.0), overlap)
    return contacts


def merge_z_stacked(segmentation, min_overlap):
    """Merge the object pairs with a z contact overlap of at least min_overlap. Return the result and the groups."""
    parent = {}

    def find(i):
        while parent.get(i, i) != i:
            i = parent[i]
        return i

    contacts = get_z_contacts(segmentation)
    for (a, b), overlap in contacts.items():
        if overlap >= min_overlap:
            parent[find(b)] = find(a)
    groups = {}
    for i in parent:
        groups.setdefault(find(i), set()).update((i, find(i)))
    mapping = np.arange(segmentation.max() + 1, dtype=segmentation.dtype)
    for root, members in groups.items():
        mapping[list(members)] = root
    return mapping[segmentation], contacts, groups


def remove_small(segmentation, min_size, copy=True):
    sizes = np.bincount(segmentation.ravel())
    small = sizes < min_size
    small[0] = False
    if copy:
        segmentation = segmentation.copy()
    segmentation[small[segmentation]] = 0
    return segmentation


def merge_test(output_dir, name, key, min_overlap, min_size, plot_path, margin=48):
    """Merge the z stacked objects and remove the small objects of a 3d run, save the result and plot the merges."""
    path = os.path.join(output_dir, f"{name}.h5")
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        segmentation = f[f"segmentation/{key}"][:]
    merged, contacts, groups = merge_z_stacked(segmentation, min_overlap)
    cleaned = remove_small(merged, min_size)
    with h5py.File(path, "a") as f:
        out_key = f"segmentation/{key}_merged"
        if out_key in f:
            del f[out_key]
        f.create_dataset(out_key, data=cleaned, chunks=True, compression="gzip")
        f[out_key].attrs["min_overlap"] = min_overlap
        f[out_key].attrs["min_size"] = min_size

    sizes = np.bincount(segmentation.ravel())
    for (a, b), overlap in sorted(contacts.items(), key=lambda item: -item[1]):
        if overlap >= 0.05:
            merge = "merge" if overlap >= min_overlap else "keep"
            print(f"{a} - {b}: overlap {overlap:.2f}, {sizes[a] / 1e3:.0f}k + {sizes[b] / 1e3:.0f}k voxels, {merge}")
    count = {
        "input": len(np.unique(segmentation)) - 1, "merged": len(np.unique(merged)) - 1,
        "cleaned": len(np.unique(cleaned)) - 1
    }
    print(
        f"{name} {key}: {count['input']} objects, {count['merged']} after merging {len(groups)} groups, "
        f"{count['cleaned']} after removing the objects below {min_size:.0e} voxels"
    )

    # One column per merged group: the xz and yz plane through the group center, before and after the merge.
    rng = np.random.default_rng(0)
    colors = rng.uniform(0.2, 1.0, size=(segmentation.max() + 1, 3))
    fig, axes = plt.subplots(4, len(groups), figsize=(3.5 * len(groups), 10), squeeze=False)
    boxes = ndimage.find_objects(merged)
    for col, (root, members) in enumerate(sorted(groups.items())):
        box = boxes[root - 1]
        z, y, x = [(b.start + b.stop) // 2 for b in box]
        zs = slice(max(box[0].start - margin // 2, 0), box[0].stop + margin // 2)
        ys = slice(max(box[1].start - margin, 0), box[1].stop + margin)
        xs = slice(max(box[2].start - margin, 0), box[2].stop + margin)
        planes = (("xz", lambda v: v[zs, y, xs]), ("yz", lambda v: v[zs, ys, x]))
        for row, (plane, get_plane) in enumerate(planes):
            for offset, labels in ((0, segmentation), (2, merged)):
                ax = axes[row + offset, col]
                plane_labels = get_plane(labels) * np.isin(get_plane(labels), list(members))
                ax.imshow(overlay_labels(get_plane(raw), plane_labels, colors), aspect="equal")
                state = "pieces" if offset == 0 else "merged"
                ax.set_title(f"{plane} {state}: {'+'.join(str(m) for m in sorted(members))}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=80)
    print(f"Saved the plot to {plot_path}")


def get_axis_blocks(size, block, overlap):
    """The block starts along one axis and the core of each block, which ends in the middle of the overlap."""
    if size <= block:
        return [(0, (0, size))]
    starts = list(range(0, size - block, block - overlap)) + [size - block]
    bounds = [0] + [(start + previous + block) // 2 for previous, start in zip(starts[:-1], starts[1:])] + [size]
    return [(start, (bounds[k], bounds[k + 1])) for k, start in enumerate(starts)]


def get_full_blocks(shape):
    """The blocks of the s0 volume: their ZYX start and the core as ZYX (start, stop)."""
    axes = [get_axis_blocks(n, b, o) for n, b, o in zip(shape, FULL_BLOCK_SHAPE, FULL_OVERLAP)]
    return [(tuple(a[0] for a in block), tuple(a[1] for a in block)) for block in product(*axes)]


def get_full_block_path(start):
    return os.path.join(FULL_BLOCK_DIR, "block_z{}_y{}_x{}.h5".format(*start))


def segment_full(zarr_path):
    """Segment the blocks that no other worker has claimed yet."""
    os.makedirs(FULL_BLOCK_DIR, exist_ok=True)
    raw = zarr.open_array(os.path.join(zarr_path, "s0"), mode="r")
    blocks = get_full_blocks(raw.shape)
    for start, _ in blocks:
        path = get_full_block_path(start)
        try:
            os.close(os.open(path + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        bb = tuple(slice(s, s + b) for s, b in zip(start, FULL_BLOCK_SHAPE))
        t0 = time.time()
        segmentation = segment_image(raw[bb], MODELS["nuclei"], "apg", ndim=3)
        with h5py.File(path + ".tmp", "w") as f:
            f.create_dataset("segmentation", data=segmentation, chunks=True, compression="gzip")
            f.attrs["start"] = start
        os.rename(path + ".tmp", path)
        n_objects = len(np.unique(segmentation)) - 1
        print(f"{os.path.basename(path)}: {n_objects} objects in {time.time() - t0:.0f} s", flush=True)
    print(f"No unclaimed blocks left of {len(blocks)}")


def stitch_full_blocks(shape):
    """Put the objects of each block, whose bounding box center is in the core of the block, into the volume.

    An object that overlaps the objects already in the volume with more than half of its voxels is a copy of them and
    is dropped. Otherwise it fills the voxels that are still background.
    """
    # uint16 holds the ids of the nuclei and halves the memory, the s0 volume has 2.3e9 voxels.
    stitched = np.zeros(shape, dtype="uint16")
    next_id, n_dropped = 1, 0
    for start, core in get_full_blocks(shape):
        with h5py.File(get_full_block_path(start), "r") as f:
            segmentation = f["segmentation"][:]
        for i, box in enumerate(ndimage.find_objects(segmentation), 1):
            if box is None:
                continue
            center = [s + (b.start + b.stop) // 2 for s, b in zip(start, box)]
            if not all(lo <= c < hi for c, (lo, hi) in zip(center, core)):
                continue
            mask = segmentation[box] == i
            target = stitched[tuple(slice(s + b.start, s + b.stop) for s, b in zip(start, box))]
            if np.mean(target[mask] > 0) > 0.5:
                n_dropped += 1
                continue
            assert next_id < np.iinfo(stitched.dtype).max
            target[mask & (target == 0)] = next_id
            next_id += 1
        print(f"Stitched {os.path.basename(get_full_block_path(start))}, {next_id - 1} objects", flush=True)
    print(f"{next_id - 1} objects in the volume, {n_dropped} copies dropped")
    return stitched


def downsample_labels(labels, factor):
    return labels[::factor[0], ::factor[1], ::factor[2]]


def create_label_level(group, name, data):
    # Labels compress best with byte shuffle, the ids are piecewise constant.
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    array = group.create_array(
        name, shape=data.shape, dtype=data.dtype, shards=SHARD_SHAPE, chunks=CHUNK_SHAPE, compressors=compressor,
        fill_value=0, dimension_names=["z", "y", "x"],
    )
    array[:] = data


def write_label_pyramid(group, name, segmentation=None):
    """Write s1, s2, ... of a label image with DOWNSCALE_FACTORS, from segmentation or from its s0, replacing the
    levels that exist already, and its metadata. The labels are taken at the first voxel of each block."""
    level_data = group["s0"][:] if segmentation is None else segmentation
    for level, factor in enumerate(DOWNSCALE_FACTORS, 1):
        if f"s{level}" in group:
            del group[f"s{level}"]
        level_data = downsample_labels(level_data, factor)
        create_label_level(group, f"s{level}", level_data)
    level_names = {f"s{level}" for level in range(len(DOWNSCALE_FACTORS) + 1)}
    for key in list(group.array_keys()):
        if key not in level_names:
            del group[key]
    multiscale = get_multiscale(name, len(DOWNSCALE_FACTORS) + 1, "nearest")
    image_label = {"version": "0.5", "source": {"image": "../../"}}
    group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale], "image-label": image_label}


def add_nuclei_labels(zarr_path, segmentation, name="nuclei"):
    """Add the segmentation as an OME-Zarr label image with the same pyramid as the raw, under 'labels/<name>'."""
    root = zarr.open_group(zarr_path, mode="a", zarr_format=3)
    assert root["s0"].shape == segmentation.shape
    if segmentation.max() < np.iinfo("uint16").max:
        segmentation = segmentation.astype("uint16")
    labels = root.require_group("labels")
    if name in labels:
        del labels[name]
    group = labels.create_group(name)
    create_label_level(group, "s0", segmentation)
    write_label_pyramid(group, name, segmentation)
    label_names = [label for label in labels.attrs.get("ome", {}).get("labels", []) if label != name] + [name]
    labels.attrs["ome"] = {"version": "0.5", "labels": label_names}
    print(f"Saved {zarr_path}/labels/{name}")


def get_artifact_reason(box, shape):
    """Why an object with this bounding box is an artifact, or None for a nucleus."""
    z_extent, *yx_extents = [b.stop - b.start for b in box]
    if z_extent <= MAX_FLAT_Z:
        return "flat"
    if max(yx_extents) > MAX_YX_EXTENT or any(abs(e - TILE_SIZE) <= TILE_TOLERANCE for e in yx_extents):
        return "too wide or tile shaped"
    if any(b.start == 0 or b.stop == n for b, n in zip(box[1:], shape[1:])):
        return "at the y or x border"
    return None


def remove_artifacts(segmentation):
    """Remove the artifacts in place, keep the largest piece of each nucleus and relabel consecutively."""
    boxes = ndimage.find_objects(segmentation)
    mapping = np.zeros(len(boxes) + 1, dtype=segmentation.dtype)
    removed = {}
    n_pieces = 0
    for i, box in enumerate(boxes, 1):
        if box is None:
            continue
        reason = get_artifact_reason(box, segmentation.shape)
        if reason is None:
            # Small loose pieces of a nucleus show as speckles in the renders.
            mask = segmentation[box] == i
            pieces, n = ndimage.label(mask)
            if n > 1:
                largest = np.argmax(np.bincount(pieces.ravel())[1:]) + 1
                segmentation[box][mask & (pieces != largest)] = 0
                n_pieces += n - 1
            if (pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1).sum() < MIN_CLEAN_SIZE:
                reason = "small"
        if reason is not None:
            removed[reason] = removed.get(reason, 0) + 1
            continue
        mapping[i] = len(np.flatnonzero(mapping)) + 1
    segmentation[:] = mapping[segmentation]
    summary = ", ".join(f"{n} {reason}" for reason, n in removed.items())
    print(f"Removed {summary}; removed {n_pieces} loose pieces; {int(mapping.max())} nuclei left")
    return segmentation


def plot_full(zarr_path, labels, name, plot_path):
    """Plot the central xy, xz and yz planes of the s0 volume with the labels, at half resolution."""
    raw = zarr.open_array(os.path.join(zarr_path, "s0"), mode="r")
    center = [s // 2 for s in labels.shape]
    planes = {
        f"xy, z={center[0]}": (raw[center[0]], labels[center[0]]),
        f"xz, y={center[1]}": (raw[:, center[1]], labels[:, center[1]]),
        f"yz, x={center[2]}": (raw[:, :, center[2]].T, labels[:, :, center[2]].T),
    }
    colors = np.random.default_rng(0).uniform(0.0, 1.0, size=(labels.max() + 1, 3))
    fig, axes = plt.subplots(1, len(planes), figsize=(24, 9))
    for ax, (plane, (image, plane_labels)) in zip(axes, planes.items()):
        ax.imshow(overlay_labels(image[::2, ::2], plane_labels[::2, ::2], colors))
        ax.set_title(f"{name} s0 full: {plane}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=80)
    print(f"Saved the plot to {plot_path}")


def clean_full(zarr_path, plot_path, name="nuclei_clean"):
    """Remove the artifacts from 'labels/nuclei' and add the result as 'labels/<name>'."""
    segmentation = zarr.open_array(os.path.join(zarr_path, "labels", "nuclei", "s0"), mode="r")[:]
    print(f"labels/nuclei: {len(np.unique(segmentation)) - 1} objects")
    segmentation = remove_artifacts(segmentation)
    add_nuclei_labels(zarr_path, segmentation, name=name)
    plot_full(zarr_path, segmentation, name, plot_path)


def stitch_full(zarr_path, plot_path):
    shape = zarr.open_array(os.path.join(zarr_path, "s0"), mode="r").shape
    missing = [start for start, _ in get_full_blocks(shape) if not os.path.exists(get_full_block_path(start))]
    assert not missing, f"{len(missing)} blocks are not segmented yet: {missing}"
    t0 = time.time()
    stitched = stitch_full_blocks(shape)
    merged, _, groups = merge_z_stacked(stitched, MIN_OVERLAP)
    del stitched
    cleaned = remove_small(merged, MIN_SIZE, copy=False)
    print(
        f"Merged {len(groups)} groups, {len(np.unique(cleaned)) - 1} objects of at least {MIN_SIZE:.0e} voxels, "
        f"{time.time() - t0:.0f} s"
    )
    add_nuclei_labels(zarr_path, cleaned)
    plot_full(zarr_path, cleaned, "nuclei", plot_path)


def render_nuclei(zarr_path, out_dir):
    """Render candidate views of the nuclei: the studio from the side and from above, the nuclei above a raw section,
    a low side view that shows the flat shape of the spheroid, and the nuclei on a transparent background."""
    import sys
    sys.path.insert(0, os.path.abspath(FOR_PAPER))
    from PIL import Image
    from plots_for_figure1 import (
        d_add_objects, d_backdrop, d_card, d_colored_mesh, d_distinct_colors, d_near_pairs, d_screenshot, d_separate,
        d_studio, d_view,
    )
    os.makedirs(out_dir, exist_ok=True)
    root = zarr.open_group(zarr_path, mode="r")
    # z points up in the scene, so the slices are flipped: slice 0 of the volume is at the top.
    labels = root[f"labels/nuclei_clean/s{RENDER_LEVEL}"][:][::-1].astype("uint32")
    raw = root[f"s{RENDER_LEVEL}"]
    ids, inverse = np.unique(labels, return_inverse=True)
    labels = inverse.reshape(labels.shape).astype("uint32")
    n_labels = len(ids) - 1
    separated, pairs = d_separate(labels)
    pairs |= d_near_pairs(labels[::2, ::2, ::2])
    colors = d_distinct_colors(n_labels, pairs)
    mesh = d_colored_mesh(separated, colors, RENDER_Z_SCALE)
    print(f"{n_labels} nuclei, mesh with {mesh.n_cells} faces", flush=True)

    nz, ny, nx = labels.shape
    bounds = (0, nx, 0, ny, 0, nz * RENDER_Z_SCALE)
    size = (2000, 2000)

    # The camera distance and the backdrop as for the spheroid of Figure 1, panel d.
    def studio(name, elevation, azimuth, meshes, zoom=1.4, backdrop=True, card=None, transparent=False):
        view = d_view(elevation, azimuth)[0]
        plotter = d_studio(size, bounds, view, 900, zoom)
        if transparent:
            plotter.disable_shadows()
        for m in meshes:
            d_add_objects(plotter, m)
        if card is not None:
            d_card(plotter, *card)
        if backdrop:
            d_backdrop(plotter, bounds, view, 10, 600, 40)
        image = d_screenshot(plotter, size, transparent=transparent)
        plotter.close()
        path = os.path.join(out_dir, f"{name}.png")
        Image.fromarray(image).save(path)
        print(f"Saved {path}", flush=True)

    studio("nuclei_studio_oblique", 30, -60, [mesh])
    studio("nuclei_studio_top", 70, -90, [mesh])
    studio("nuclei_side_low", 8, -90, [mesh])
    studio("nuclei_transparent_oblique", 30, -60, [mesh], backdrop=False, transparent=True)

    # The nuclei above the middle xy plane, cut at the plane, standing on a card with the raw of that plane.
    z_mid = raw.shape[0] // 2
    plane = raw[z_mid].astype("float32")
    low, high = np.percentile(plane, [1, 99.5])
    gray = (np.clip((plane - low) / (high - low), 0, 1) * 255).astype("uint8")
    rgb = np.stack([gray] * 3, axis=-1)
    flipped_mid = nz - 1 - z_mid
    upper = separated.copy()
    upper[:flipped_mid] = 0
    upper_mesh = d_colored_mesh(upper, colors, RENDER_Z_SCALE)
    thickness = 4.0
    card = (rgb[::-1], (0, ny, flipped_mid * RENDER_Z_SCALE - thickness - 0.5), 1.0, thickness)
    studio("nuclei_section_oblique", 35, -60, [upper_mesh], card=card)
    studio("nuclei_section_steep", 55, -75, [upper_mesh], card=card)


def find_section_slice(zarr_path, out_path, z_range=(200, 700), z_step=16, min_area=8000, min_covered=0.5):
    """Find the s0 slices where every nucleus is segmented in 'labels/nuclei_clean'.

    The 2d nucleus model segments each candidate slice at s0 (AIS, in 768 px tiles like the crops it does well on).
    A 2d nucleus of at least min_area pixels counts as missed when the 3d labels cover less than min_covered of it.
    The candidates are on the z grid of RENDER_LEVEL.
    """
    import json
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["s0"], root["labels/nuclei_clean/s0"]
    z_grid = get_level_factor(RENDER_LEVEL)[0]
    results = []
    for z in range(z_range[0], z_range[1] + 1, z_step):
        z = z - z % z_grid
        nuclei_2d = segment_image(raw[z], MODELS["nuclei"], "ais", tile_shape=(768, 768), halo=(128, 128))
        nuclei_3d = labels[z] > 0
        ids, areas = np.unique(nuclei_2d[nuclei_2d > 0], return_counts=True)
        covered = ndimage.sum(nuclei_3d, nuclei_2d, ids) / areas
        large = areas >= min_area
        missed = ids[large & (covered < min_covered)]
        results.append({
            "s0_z": int(z), "nuclei_2d": int(large.sum()), "missed": int(len(missed)),
            "nuclei_3d": int(len(np.unique(labels[z])) - 1),
        })
        print(results[-1], flush=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    complete = [r for r in results if r["missed"] == 0]
    best = max(complete or results, key=lambda r: (-r["missed"], r["nuclei_2d"]))
    print(f"Best slice: {best}")
    return best


def render_section(zarr_path, out_dir, s0_z, height=12):
    """The renders of a section through the xy plane of s0 slice s0_z: the nuclei in the studio with the outline of
    the plane; and every nucleus that crosses the plane, cut at the plane and standing on its raw (s1), up to height
    slices of RENDER_LEVEL above it. Also saves layout.json with the corners of the plane in both renders."""
    import sys
    import json
    sys.path.insert(0, os.path.abspath(FOR_PAPER))
    import pyvista as pv
    from PIL import Image
    from plots_for_figure1 import (
        d_add_objects, d_backdrop, d_colored_mesh, d_distinct_colors, d_near_pairs, d_project, d_px_per_unit,
        d_rectangle, d_screenshot, d_section_shot, d_separate, d_studio, d_view,
    )
    os.makedirs(out_dir, exist_ok=True)
    root = zarr.open_group(zarr_path, mode="r")
    # z points up in the scene, so the slices are flipped: slice 0 of the volume is at the top.
    labels = root[f"labels/nuclei_clean/s{RENDER_LEVEL}"][:][::-1].astype("uint32")
    ids, inverse = np.unique(labels, return_inverse=True)
    labels = inverse.reshape(labels.shape).astype("uint32")
    n_labels = len(ids) - 1
    separated, pairs = d_separate(labels)
    pairs |= d_near_pairs(labels[::2, ::2, ::2])
    colors = d_distinct_colors(n_labels, pairs)
    mesh = d_colored_mesh(separated, colors, RENDER_Z_SCALE)
    nz, ny, nx = labels.shape
    bounds = (0, nx, 0, ny, 0, nz * RENDER_Z_SCALE)

    # The plane in the flipped slices of RENDER_LEVEL, and the nuclei that cross it.
    z_mid = nz - 1 - s0_z // get_level_factor(RENDER_LEVEL)[0]
    plane_ids = np.unique(labels[z_mid])
    plane_ids = plane_ids[plane_ids > 0]
    z_plane = z_mid * RENDER_Z_SCALE
    layout = {"nuclei": n_labels, "plane_nuclei": int(len(plane_ids)), "z_scale": RENDER_Z_SCALE, "s0_z": s0_z}
    # The plane spans the footprint of the spheroid (of all nuclei, with a margin), not the resin around it.
    margin = 24
    rows, cols = np.nonzero(labels.any(axis=0))
    y0, y1 = max(rows.min() - margin, 0), min(rows.max() + margin, ny)
    x0, x1 = max(cols.min() - margin, 0), min(cols.max() + margin, nx)

    # a: the studio with the outline of the plane.
    size = (2000, 2000)
    view = d_view(30, -60)[0]
    plotter = d_studio(size, bounds, view, 900, 1.4)
    d_add_objects(plotter, mesh)
    d_backdrop(plotter, bounds, view, 10, 600, 40)
    Image.fromarray(d_screenshot(plotter, size)).save(os.path.join(out_dir, "section_studio.png"))
    plane_corners = d_rectangle((x0, x1, y0, y1, z_plane, z_plane))
    layout["studio"] = {
        "plane_outline": d_project(plotter, plane_corners),
        "px_per_voxel": d_px_per_unit(plotter, np.array([nx / 2, ny / 2, z_plane])),
    }
    plotter.close()
    print("Saved the studio", flush=True)

    # The raw plane of the level below (twice the resolution in y and x), with the y axis of the labels.
    raw_level = RENDER_LEVEL - 1
    factor = DOWNSCALE_FACTORS[raw_level][1]
    plane = root[f"s{raw_level}"][
        s0_z // get_level_factor(raw_level)[0], y0 * factor:y1 * factor, x0 * factor:x1 * factor
    ].astype("float32")
    low, high = np.percentile(plane, [1, 99.5])
    gray = np.clip((plane - low) / (high - low), 0, 1)
    step = 1.0 / factor

    def add_plane(rgb):
        def add(plotter):
            grid = pv.ImageData(
                dimensions=(rgb.shape[1], rgb.shape[0], 1), spacing=(step, step, 1.0), origin=(x0, y0, z_plane)
            )
            grid.point_data["rgb"] = (rgb.reshape(-1, 3) * 255).astype("uint8")
            plotter.add_mesh(
                grid, scalars="rgb", rgb=True, ambient=1.0, diffuse=0.0, specular=0.0, show_scalar_bar=False
            )
        return add

    # b: the nuclei that cross the plane, cut at the plane, on its raw. The camera is from above, tilted a little
    # like the studio view.
    upper = np.where(np.isin(separated, plane_ids), separated, 0)
    upper[:z_mid] = 0
    upper_mesh = d_colored_mesh(upper, colors, RENDER_Z_SCALE)
    direction = d_view(55, -60)[0]
    section_bounds = (x0, x1, y0, y1, z_plane, z_plane + height * RENDER_Z_SCALE)
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2, z_plane])
    key = np.array([-1500.0, 1500.0, 2500.0])
    raw_rgb = np.stack([gray] * 3, axis=-1)
    image, corners, ppv = d_section_shot(
        lambda p: (add_plane(raw_rgb)(p), d_add_objects(p, upper_mesh)), centre, direction, (0, 0, 1), key,
        section_bounds, size, 1.0, points=plane_corners,
    )
    Image.fromarray(image).save(os.path.join(out_dir, "section_nuclei_on_raw.png"))
    layout["section"] = {"px_per_voxel": ppv, "card_corners": corners}
    with open(os.path.join(out_dir, "layout.json"), "w") as f:
        json.dump(layout, f, indent=1)
    print(f"Saved the section, {len(plane_ids)} nuclei cross the plane", flush=True)


def plot_section_panel(out_dir, plot_path):
    """A preview of the section panel: the studio with the outline of the plane, and the nuclei on the raw of the
    plane. Lines join the two right corners of the outline to the same corners of the section, like a pyramid."""
    import sys
    import json
    sys.path.insert(0, os.path.abspath(FOR_PAPER))
    from plots_for_figure1 import d_crop_white, d_load
    from matplotlib.patches import ConnectionPatch, Polygon
    layout = json.load(open(os.path.join(out_dir, "layout.json")))
    studio = d_load(os.path.join(out_dir, "section_studio.png"))
    section, offset = d_crop_white(d_load(os.path.join(out_dir, "section_nuclei_on_raw.png")), 8)
    corners = np.array(layout["section"]["card_corners"]) - offset
    outline = np.array(layout["studio"]["plane_outline"])
    ratios = [image.shape[1] / image.shape[0] for image in (studio, section)]
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": ratios})
    ax_a.imshow(studio)
    ax_b.imshow(section)
    ax_a.add_patch(Polygon(outline, closed=True, fill=False, edgecolor="#1a1a1a", lw=1.0))
    # The pyramid: the upper and the lower right corner of the outline to the top and the bottom corner of the
    # section, so that the lines frame the section and do not cross it.
    right = outline[np.argsort(outline[:, 0])[-2:]]
    upper, lower = right[np.argmin(right[:, 1])], right[np.argmax(right[:, 1])]
    top, bottom = corners[np.argmin(corners[:, 1])], corners[np.argmax(corners[:, 1])]
    for point_a, point_b in ((upper, top), (lower, bottom)):
        fig.add_artist(ConnectionPatch(
            point_a, point_b, "data", "data", axesA=ax_a, axesB=ax_b, color="#1a1a1a", lw=0.6
        ))
    for ax in (ax_a, ax_b):
        ax.axis("off")
    ax_b.set_title(f"the {layout['plane_nuclei']} nuclei of s0 slice {layout['s0_z']}", fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    print(f"Saved the plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="task", required=True)

    convert = subparsers.add_parser("convert", help="Convert the tif to a multiscale OME-Zarr.")
    convert.add_argument("--input_path", default=INPUT_PATH)
    convert.add_argument("--output_path", default=OUTPUT_PATH)
    convert.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    rebuild = subparsers.add_parser("rebuild_pyramid", help="Rewrite the levels above s0 of the raw and the labels.")
    rebuild.add_argument("--zarr_path", default=OUTPUT_PATH)
    rebuild.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    crops = subparsers.add_parser("segment_crops", help="Segment the 768 x 768 crops in 2d and plot them.")
    crops.add_argument("--zarr_path", default=OUTPUT_PATH)
    crops.add_argument("--level", type=int, default=0, help="The scale level of the crops.")

    predictions = subparsers.add_parser("predictions", help="Save and plot the decoder outputs on the crops.")
    predictions.add_argument("--zarr_path", default=OUTPUT_PATH)
    predictions.add_argument("--level", type=int, default=0, help="The scale level of the crops.")

    volume = subparsers.add_parser("predict_volume", help="Save the 3d decoder outputs of a whole scale level.")
    volume.add_argument("--zarr_path", default=OUTPUT_PATH)
    volume.add_argument("--level", type=int, default=1)
    volume.add_argument("--target", default="nuclei", choices=list(MODELS))
    volume.add_argument("--z_range", type=int, nargs=2, default=None, help="Predict only these slices.")
    volume.add_argument("--slab_size", type=int, default=VOLUME_SLAB_SIZE)
    volume.add_argument("--batch_size", type=int, default=1)
    volume.add_argument("--channels", nargs="+", default=list(PREDICTION_CHANNELS), choices=list(PREDICTION_CHANNELS))

    ws = subparsers.add_parser("watershed_volume", help="Segment the volume predictions with a block-wise watershed.")
    ws.add_argument("--level", type=int, default=1)
    ws.add_argument("--target", default="nuclei", choices=list(MODELS))
    ws.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    ws.add_argument("--min_size", type=int, default=WS_MIN_SIZE, help="Remove the objects below this size.")
    ws.add_argument("--filter_only", action="store_true", help="Only run the size filter on the saved watershed.")

    inverted = subparsers.add_parser(
        "segment_inverted_crop", help="3d AIS on the inverted raw of a central crop at s1 and s0.",
    )
    inverted.add_argument("--zarr_path", default=OUTPUT_PATH)
    inverted.add_argument("--levels", type=int, nargs="+", default=[1, 0])

    cell_crop = subparsers.add_parser(
        "segment_cell_crop", help="Cells of a central crop at s1 and s0: 3d AIS and prompts from the nuclei.",
    )
    cell_crop.add_argument("--zarr_path", default=OUTPUT_PATH)
    cell_crop.add_argument("--levels", type=int, nargs="+", default=[1, 0])
    cell_crop.add_argument("--methods", nargs="+", default=["ais", "prompts"], choices=["ais", "prompts"])

    seeded = subparsers.add_parser(
        "seed_cell_crop", help="Cells of the cell crop from the AIS predictions, seeded with the nuclei.",
    )
    seeded.add_argument("--levels", type=int, nargs="+", default=[1, 0])

    cell_ws = subparsers.add_parser(
        "watershed_cell_crop", help="Cells of the cell crop: watershed of the expanded nuclei, closed foreground mask.",
    )
    cell_ws.add_argument("--levels", type=int, nargs="+", default=[0])
    cell_ws.add_argument("--closing_radius", type=float, default=CELL_CLOSING_RADIUS, help="In nanometer.")

    cells_volume = subparsers.add_parser("segment_cells_volume", help="Segment the cells of a whole level block-wise.")
    cells_volume.add_argument("--level", type=int, default=0)
    cells_volume.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    cells_volume.add_argument("--border_seed", action="store_true", help="Seed the resin from the Y and X faces.")
    cells_volume.add_argument(
        "--nuclei", default="watershed", choices=["watershed", "merged", "joined"],
        help="The nuclei: of watershed_volume, merge_prompted_nuclei or join_nucleus_fragments.",
    )

    merge = subparsers.add_parser(
        "merge_prompted_nuclei", help="Merge the prompted nuclei into the nuclei of s1 and remove their fragments.",
    )
    merge.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    join = subparsers.add_parser(
        "join_nucleus_fragments", help="Make one object of each nucleus that is split into fragments (prompted).",
    )
    join.add_argument("--zarr_path", default=OUTPUT_PATH)
    join.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    border_seed = subparsers.add_parser(
        "check_border_seed", help="Test the background seed of the cells on slabs of s0 and against the raw texture.",
    )
    border_seed.add_argument("--zarr_path", default=OUTPUT_PATH)
    border_seed.add_argument("--z_ranges", type=int, nargs="+", default=[368, 496], help="Start and stop, repeated.")
    border_seed.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    export = subparsers.add_parser(
        "export_segmentations", help="Export the cells and nuclei as an OME-Zarr with two multiscale label images.",
    )
    export.add_argument("--output_path", default=EXPORT_PATH)
    export.add_argument("--nuclei", default="updated", choices=["watershed", "merged", "joined", "updated"])
    export.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    contours = subparsers.add_parser(
        "export_contours", help="Export a level of the inverted raw and the contours of cells and nuclei for Fiji.",
    )
    contours.add_argument("--zarr_path", default=OUTPUT_PATH)
    contours.add_argument("--export_path", default=EXPORT_PATH)
    contours.add_argument("--level", type=int, default=CONTOUR_LEVEL)
    contours.add_argument("--output_path", default=None)

    tifs = subparsers.add_parser(
        "export_tifs", help="Export a level of the exported cells and nuclei as label tifs for Fiji in a zip archive.",
    )
    tifs.add_argument("--export_path", default=EXPORT_PATH)
    tifs.add_argument("--level", type=int, default=TIF_LEVEL)
    tifs.add_argument("--output_path", default=None)

    pyramids = subparsers.add_parser(
        "downsample_segmentations", help="Write the pyramids of the cell and nucleus segmentations up to s3.",
    )
    pyramids.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    pyramids.add_argument("--names", nargs="+", default=None, help="Only these segmentations, by default all.")

    prompted = subparsers.add_parser(
        "segment_nuclei_from_prompts", help="Segment missed nuclei at s1 from the points of a napari points layer.",
    )
    prompted.add_argument("--zarr_path", default=OUTPUT_PATH)
    prompted.add_argument("--points_path", default=NUCLEUS_POINTS_PATH)
    prompted.add_argument("--points_level", type=int, default=NUCLEUS_POINTS_LEVEL)
    prompted.add_argument("--z_offsets", type=int, nargs="+", default=[0], help="Slices to prompt, relative.")
    prompted.add_argument("--n_negatives", type=int, default=0, help="Negative points from the nearest nuclei.")
    prompted.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    check_volume = subparsers.add_parser(
        "check_cells_volume", help="Compare the block-wise cell segmentation to watershed_cell_crop on the cell crop.",
    )
    check_volume.add_argument("--level", type=int, default=0)
    check_volume.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    check_volume.add_argument("--block_shapes", type=int, nargs="+", default=[32, 256, 256], help="ZYX, repeated.")
    check_volume.add_argument(
        "--ws_halos", type=int, nargs="+", default=list(CELL_VOLUME_WS_HALO), help="ZYX, repeated.",
    )

    seg3d = subparsers.add_parser("segment_3d", help="Segment the cells at s2 and the nuclei of an s0 block in 3d.")
    seg3d.add_argument("--zarr_path", default=OUTPUT_PATH)
    seg3d.add_argument("--runs", nargs="+", default=list(RUNS_3D), choices=list(RUNS_3D))

    gallery = subparsers.add_parser("size_gallery", help="Plot the small objects of a 3d run.")
    gallery.add_argument("--run", default="nuclei_s0_block_z384", choices=list(RUNS_3D))
    gallery.add_argument("--key", default="nuclei_apg")
    gallery.add_argument("--max_size", type=float, default=1e6, help="Plot the objects below this size in voxels.")

    merge = subparsers.add_parser("merge_test", help="Merge the z stacked objects of a 3d run and plot the merges.")
    merge.add_argument("--run", default="nuclei_s0_block_z384", choices=list(RUNS_3D))
    merge.add_argument("--key", default="nuclei_apg")
    merge.add_argument("--min_overlap", type=float, default=MIN_OVERLAP)
    merge.add_argument("--min_size", type=float, default=MIN_SIZE)

    full = subparsers.add_parser("segment_full", help="Segment the nuclei of the s0 volume in blocks (one worker).")
    full.add_argument("--zarr_path", default=OUTPUT_PATH)

    stitch = subparsers.add_parser("stitch_full", help="Stitch the blocks and add them to the OME-Zarr.")
    stitch.add_argument("--zarr_path", default=OUTPUT_PATH)

    clean = subparsers.add_parser("clean_full", help="Remove the artifacts of 'labels/nuclei' as a new label image.")
    clean.add_argument("--zarr_path", default=OUTPUT_PATH)

    render = subparsers.add_parser("render", help="Render candidate 3d views of 'labels/nuclei_clean' (GPU node).")
    render.add_argument("--zarr_path", default=OUTPUT_PATH)

    section = subparsers.add_parser("render_section", help="Render the section views (GPU node) and plot them.")
    section.add_argument("--zarr_path", default=OUTPUT_PATH)
    section.add_argument("--plot_only", action="store_true", help="Only plot the panel from the saved renders.")
    section.add_argument("--s0_z", type=int, default=None, help="The slice; by default the best of the search.")

    args = parser.parse_args()
    if args.task == "convert":
        convert_to_ome_zarr(args.input_path, args.output_path, args.num_workers)
    elif args.task == "rebuild_pyramid":
        rebuild_pyramid(args.zarr_path, args.num_workers)
    elif args.task == "segment_crops":
        output_dir = os.path.join(CROP_DIR, f"s{args.level}")
        segment_crops(args.zarr_path, output_dir, args.level)
        plot_crops(args.zarr_path, output_dir, args.level, f"beghin_em_crops768_s{args.level}_segmentation.png")
    elif args.task == "predictions":
        output_dir = os.path.join(CROP_DIR, f"s{args.level}")
        save_predictions(args.zarr_path, output_dir, args.level)
        for target in MODELS:
            plot_path = f"beghin_em_crops768_s{args.level}_{target}_predictions.png"
            plot_predictions(args.zarr_path, output_dir, args.level, target, plot_path)
    elif args.task == "predict_volume":
        predict_volume(
            args.zarr_path, args.level, args.target, args.z_range, args.slab_size, args.batch_size, args.channels,
        )
    elif args.task == "watershed_volume":
        watershed_volume(
            args.level, args.target, args.num_workers, min_size=args.min_size, filter_only=args.filter_only,
        )
    elif args.task == "segment_inverted_crop":
        segment_inverted_crop(args.zarr_path, args.levels)
    elif args.task == "segment_cell_crop":
        segment_cell_crop(args.zarr_path, args.levels, args.methods)
    elif args.task == "seed_cell_crop":
        seed_cell_crop(args.levels)
    elif args.task == "watershed_cell_crop":
        watershed_cell_crop(args.levels, args.closing_radius)
    elif args.task == "segment_cells_volume":
        segment_cells_volume(
            args.level, args.num_workers, border_seed=args.border_seed, nuclei=args.nuclei,
        )
    elif args.task == "join_nucleus_fragments":
        join_nucleus_fragments(args.zarr_path, num_workers=args.num_workers)
    elif args.task == "merge_prompted_nuclei":
        merge_prompted_nuclei(args.num_workers)
    elif args.task == "check_border_seed":
        assert len(args.z_ranges) % 2 == 0, args.z_ranges
        z_ranges = [tuple(args.z_ranges[i:i + 2]) for i in range(0, len(args.z_ranges), 2)]
        check_border_seed(args.zarr_path, z_ranges, args.num_workers)
    elif args.task == "downsample_segmentations":
        downsample_segmentations(args.num_workers, names=args.names)
    elif args.task == "segment_nuclei_from_prompts":
        segment_nuclei_from_prompts(
            args.zarr_path, args.points_path, args.points_level, tuple(args.z_offsets), args.n_negatives,
            args.num_workers,
        )
    elif args.task == "export_segmentations":
        export_segmentations(args.output_path, args.nuclei, args.num_workers)
    elif args.task == "export_contours":
        export_contours(args.zarr_path, export_path=args.export_path, level=args.level, output_path=args.output_path)
    elif args.task == "export_tifs":
        export_tifs(args.export_path, args.level, args.output_path)
    elif args.task == "check_cells_volume":
        assert len(args.block_shapes) % 3 == 0 and len(args.ws_halos) == len(args.block_shapes)
        check_cells_volume(
            args.level, args.num_workers,
            [tuple(args.block_shapes[i:i + 3]) for i in range(0, len(args.block_shapes), 3)],
            [tuple(args.ws_halos[i:i + 3]) for i in range(0, len(args.ws_halos), 3)],
        )
    elif args.task == "segment_3d":
        segment_3d(args.zarr_path, SEGMENT_3D_DIR, args.runs)
        for name in args.runs:
            plot_3d(SEGMENT_3D_DIR, name, f"beghin_em_3d_{name}.png")
    elif args.task == "size_gallery":
        plot_path = f"beghin_em_3d_{args.run}_small_objects.png"
        plot_size_gallery(SEGMENT_3D_DIR, args.run, args.key, args.max_size, plot_path)
    elif args.task == "merge_test":
        plot_path = f"beghin_em_3d_{args.run}_merges.png"
        merge_test(SEGMENT_3D_DIR, args.run, args.key, args.min_overlap, args.min_size, plot_path)
    elif args.task == "segment_full":
        segment_full(args.zarr_path)
    elif args.task == "stitch_full":
        stitch_full(args.zarr_path, "beghin_em_3d_nuclei_s0_full.png")
    elif args.task == "clean_full":
        clean_full(args.zarr_path, "beghin_em_3d_nuclei_s0_full_clean.png")
    elif args.task == "render":
        render_nuclei(args.zarr_path, RENDER_DIR)
    elif args.task == "render_section":
        out_dir = os.path.join(RENDER_DIR, "section")
        if not args.plot_only:
            s0_z = args.s0_z
            if s0_z is None:
                s0_z = find_section_slice(args.zarr_path, os.path.join(out_dir, "slice_search.json"))["s0_z"]
            render_section(args.zarr_path, out_dir, s0_z)
        plot_section_panel(out_dir, "beghin_em_nuclei_section_panel.png")


if __name__ == "__main__":
    main()
