"""Explore the volume EM spheroid of the Beghin lab (PF_Dish_C_pin3new_Spheroid3, 869 x 3068 x 2864 uint8).

convert: write the tif as an OME-Zarr (zarr v3, OME-NGFF 0.5) with sharded, chunked and compressed arrays and a mean
downsampled pyramid s0, s1, ... The layout follows the stitched Expanded_Platynereis.ome.zarr of the Huisken data.
The tif has no voxel size, pass it with '--voxel_size' (z, y, x in nanometer) once it is known.

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

render: render candidate 3d views of 'labels/nuclei_clean' off screen with pyvista, in the studio style of Figure 1
(on a GPU node, e.g. with srun --overlap in a running job).

render_section: find an xy plane where every nucleus is segmented (2d micro-sam2 nuclei against the 3d labels), render
the nuclei in the studio with the plane and the nuclei of the plane standing on its raw, and plot them side by side.
"""
import os
import time
import shutil
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
# The 3d renders of 'labels/nuclei_clean', at s2. The voxel size is unknown: z is scaled by RENDER_Z_SCALE, the z
# spacing over the xy spacing estimated from the nuclei (~70 slices deep, ~280 px wide at s0).
RENDER_LEVEL = 2
RENDER_Z_SCALE = 3.5
RENDER_DIR = "renders"
FOR_PAPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[os.pardir] * 3, "plotting", "v2", "for_paper")
PREDICTION_CHANNELS = ("foreground", "distance_z", "distance_y", "distance_x", "boundary")


def create_level(group, name, shape, dtype):
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    return group.create_array(
        name, shape=shape, dtype=dtype, shards=SHARD_SHAPE, chunks=CHUNK_SHAPE, compressors=compressor,
        fill_value=0, dimension_names=["z", "y", "x"],
    )


def write_ome_metadata(group, name, num_levels, voxel_size):
    unit = None if voxel_size is None else "nanometer"
    voxel_size = (1.0, 1.0, 1.0) if voxel_size is None else voxel_size
    axes = [{"name": ax, "type": "space"} for ax in "zyx"]
    if unit is not None:
        for ax in axes:
            ax["unit"] = unit

    datasets = []
    for level in range(num_levels):
        factor = 2 ** level
        scale = [float(vs * factor) for vs in voxel_size]
        # The mean of a 2x2x2 block sits at the center of the block.
        translation = [float(vs * (factor - 1) / 2) for vs in voxel_size]
        transforms = [{"type": "scale", "scale": scale}, {"type": "translation", "translation": translation}]
        datasets.append({"path": f"s{level}", "coordinateTransformations": transforms})

    multiscale = {"name": name, "axes": axes, "datasets": datasets, "type": "mean"}
    group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale]}


def write_s0(input_path, array, num_workers):
    """Copy the tif into s0, one shard slab along z at a time, so that every shard is written once and complete."""
    n_slices = array.shape[0]
    for z0 in range(0, n_slices, SHARD_SHAPE[0]):
        z1 = min(z0 + SHARD_SHAPE[0], n_slices)
        slab = tifffile.imread(input_path, key=range(z0, z1), maxworkers=num_workers)
        array[z0:z1] = slab
        print(f"s0: wrote z {z0} - {z1} of {n_slices}")


def downsample_mean(block):
    """Downsample a 3d block by 2 with the mean of each 2x2x2 block. An odd edge is padded with its last value."""
    pad = [(0, s % 2) for s in block.shape]
    block = np.pad(block, pad, mode="edge")
    mean = np.zeros([s // 2 for s in block.shape], dtype="float32")
    for oz, oy, ox in product(range(2), repeat=3):
        mean += block[oz::2, oy::2, ox::2]
    mean /= 8
    return np.round(mean).astype(block.dtype)


def write_downsampled_level(source, target, num_workers):
    """Write target from source, one target shard per task, so that no two tasks write the same shard."""
    grid = [range(0, s, sh) for s, sh in zip(target.shape, SHARD_SHAPE)]

    def write_shard(start):
        bb = tuple(slice(st, min(st + sh, s)) for st, sh, s in zip(start, SHARD_SHAPE, target.shape))
        source_bb = tuple(slice(2 * b.start, min(2 * b.stop, s)) for b, s in zip(bb, source.shape))
        target[bb] = downsample_mean(source[source_bb])

    with ThreadPoolExecutor(num_workers) as pool:
        list(pool.map(write_shard, product(*grid)))


def check_s0(input_path, array):
    """Compare the first, middle and last slice of s0 to the tif."""
    for z in (0, array.shape[0] // 2, array.shape[0] - 1):
        expected = tifffile.imread(input_path, key=z)
        assert np.array_equal(array[z], expected), f"s0 differs from the tif at z {z}"
    print("s0 matches the tif at the first, middle and last slice")


def convert_to_ome_zarr(input_path, output_path, voxel_size, num_levels, num_workers):
    if os.path.exists(output_path):
        raise FileExistsError(f"{output_path} exists already, delete it first to convert again.")
    zarr.config.set({"threading.max_workers": num_workers, "async.concurrency": 4 * num_workers})

    with tifffile.TiffFile(input_path) as f:
        shape = (len(f.pages),) + f.pages[0].shape
        dtype = f.pages[0].dtype
    print(f"Convert {input_path} with shape {shape} and dtype {dtype}")

    group = zarr.open_group(output_path, mode="w", zarr_format=3)
    name = os.path.basename(output_path).replace(".ome.zarr", "")
    write_ome_metadata(group, name, num_levels, voxel_size)

    source = create_level(group, "s0", shape, dtype)
    write_s0(input_path, source, num_workers)
    check_s0(input_path, source)

    for level in range(1, num_levels):
        shape = tuple((s + 1) // 2 for s in source.shape)
        target = create_level(group, f"s{level}", shape, dtype)
        write_downsampled_level(source, target, num_workers)
        print(f"s{level}: wrote shape {shape}")
        source = target

    if voxel_size is None:
        print("The voxel size is unknown, the scales are in pixel. Set it with '--voxel_size' once it is known.")


def segment_image(image, model_type, engine, ndim=2, tile_shape=None, halo=None):
    # An all-zero tile shape turns the default tiling off.
    is_tiled = tile_shape is not None and any(tile_shape)
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type, segmentation_mode=engine, ndim=ndim, is_tiled=is_tiled,
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
    factor = 2 ** level
    crops = []
    for start in CROPS:
        z = start[0] // factor
        center = [(st + CROP_SIZE // 2) // factor for st in start[1:]]
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
    count = {"input": len(np.unique(segmentation)) - 1, "merged": len(np.unique(merged)) - 1,
             "cleaned": len(np.unique(cleaned)) - 1}
    print(f"{name} {key}: {count['input']} objects, {count['merged']} after merging {len(groups)} groups, "
          f"{count['cleaned']} after removing the objects below {min_size:.0e} voxels")

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


def get_full_block_dir(untiled=False):
    return FULL_BLOCK_DIR + "_untiled" if untiled else FULL_BLOCK_DIR


def get_full_block_path(start, untiled=False):
    return os.path.join(get_full_block_dir(untiled), "block_z{}_y{}_x{}.h5".format(*start))


def get_full_label_name(untiled=False):
    return "nuclei_untiled" if untiled else "nuclei"


def segment_full(zarr_path, untiled=False):
    """Segment the blocks that no other worker has claimed yet.

    By default each 1024 x 1024 block is tiled in-plane with the default 512 tiles, which cuts nuclei at the tile
    borders. With 'untiled', each block goes to the model in one piece, a block is as large as the encoder input.
    """
    os.makedirs(get_full_block_dir(untiled), exist_ok=True)
    raw = zarr.open_array(os.path.join(zarr_path, "s0"), mode="r")
    blocks = get_full_blocks(raw.shape)
    for start, _ in blocks:
        path = get_full_block_path(start, untiled)
        try:
            os.close(os.open(path + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        bb = tuple(slice(s, s + b) for s, b in zip(start, FULL_BLOCK_SHAPE))
        t0 = time.time()
        tiling = dict(tile_shape=(0, 0, 0), halo=(0, 0, 0)) if untiled else {}
        segmentation = segment_image(raw[bb], MODELS["nuclei"], "apg", ndim=3, **tiling)
        with h5py.File(path + ".tmp", "w") as f:
            f.create_dataset("segmentation", data=segmentation, chunks=True, compression="gzip")
            f.attrs["start"] = start
        os.rename(path + ".tmp", path)
        n_objects = len(np.unique(segmentation)) - 1
        print(f"{os.path.basename(path)}: {n_objects} objects in {time.time() - t0:.0f} s", flush=True)
    print(f"No unclaimed blocks left of {len(blocks)}")


def stitch_full_blocks(shape, untiled=False):
    """Put the objects of each block, whose bounding box center is in the core of the block, into the volume.

    An object that overlaps the objects already in the volume with more than half of its voxels is a copy of them and
    is dropped. Otherwise it fills the voxels that are still background.
    """
    # uint16 holds the ids of the nuclei and halves the memory, the s0 volume has 2.3e9 voxels.
    stitched = np.zeros(shape, dtype="uint16")
    next_id, n_dropped = 1, 0
    for start, core in get_full_blocks(shape):
        with h5py.File(get_full_block_path(start, untiled), "r") as f:
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
        print(f"Stitched {os.path.basename(get_full_block_path(start, untiled))}, {next_id - 1} objects", flush=True)
    print(f"{next_id - 1} objects in the volume, {n_dropped} copies dropped")
    return stitched


def downsample_labels(labels):
    return labels[::2, ::2, ::2]


def add_nuclei_labels(zarr_path, segmentation, name="nuclei"):
    """Add the segmentation as an OME-Zarr label image with the same pyramid as the raw, under 'labels/<name>'."""
    root = zarr.open_group(zarr_path, mode="a", zarr_format=3)
    assert root["s0"].shape == segmentation.shape
    num_levels = len(root.attrs["ome"]["multiscales"][0]["datasets"])
    if segmentation.max() < np.iinfo("uint16").max:
        segmentation = segmentation.astype("uint16")
    labels = root.require_group("labels")
    if name in labels:
        del labels[name]
    group = labels.create_group(name)
    # Labels compress best with byte shuffle, the ids are piecewise constant.
    compressor = BloscCodec(cname="zstd", clevel=5, shuffle="shuffle")
    level_data = segmentation
    for level in range(num_levels):
        if level > 0:
            level_data = downsample_labels(level_data)
        array = group.create_array(
            f"s{level}", shape=level_data.shape, dtype=level_data.dtype, shards=SHARD_SHAPE, chunks=CHUNK_SHAPE,
            compressors=compressor, fill_value=0, dimension_names=["z", "y", "x"],
        )
        array[:] = level_data
    # The raw metadata gives the same scales, the labels are taken at the first voxel of each 2x2x2 block.
    multiscale = dict(root.attrs["ome"]["multiscales"][0], name=name, type="nearest")
    for dataset in multiscale["datasets"]:
        dataset["coordinateTransformations"] = dataset["coordinateTransformations"][:1]
    image_label = {"version": "0.5", "source": {"image": "../../"}}
    group.attrs["ome"] = {"version": "0.5", "multiscales": [multiscale], "image-label": image_label}
    label_names = [label for label in labels.attrs.get("ome", {}).get("labels", []) if label != name] + [name]
    labels.attrs["ome"] = {"version": "0.5", "labels": label_names}
    print(f"Saved {zarr_path}/labels/{name}")


def get_artifact_reason(box, shape, tiled=True):
    """Why an object with this bounding box is an artifact, or None for a nucleus.

    The tile shape rule is only for the tiled segmentation.
    """
    z_extent, *yx_extents = [b.stop - b.start for b in box]
    if z_extent <= MAX_FLAT_Z:
        return "flat"
    tile_shaped = tiled and any(abs(e - TILE_SIZE) <= TILE_TOLERANCE for e in yx_extents)
    if max(yx_extents) > MAX_YX_EXTENT or tile_shaped:
        return "too wide or tile shaped"
    if any(b.start == 0 or b.stop == n for b, n in zip(box[1:], shape[1:])):
        return "at the y or x border"
    return None


def remove_artifacts(segmentation, tiled=True):
    """Remove the artifacts in place, keep the largest piece of each nucleus and relabel consecutively."""
    boxes = ndimage.find_objects(segmentation)
    mapping = np.zeros(len(boxes) + 1, dtype=segmentation.dtype)
    removed = {}
    n_pieces = 0
    for i, box in enumerate(boxes, 1):
        if box is None:
            continue
        reason = get_artifact_reason(box, segmentation.shape, tiled)
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


def clean_full(zarr_path, plot_path, untiled=False):
    """Remove the artifacts from 'labels/<name>' and add the result as 'labels/<name>_clean'."""
    source = get_full_label_name(untiled)
    name = f"{source}_clean"
    segmentation = zarr.open_array(os.path.join(zarr_path, "labels", source, "s0"), mode="r")[:]
    print(f"labels/{source}: {len(np.unique(segmentation)) - 1} objects")
    segmentation = remove_artifacts(segmentation, tiled=not untiled)
    add_nuclei_labels(zarr_path, segmentation, name=name)
    plot_full(zarr_path, segmentation, name, plot_path)


def stitch_full(zarr_path, plot_path, untiled=False):
    shape = zarr.open_array(os.path.join(zarr_path, "s0"), mode="r").shape
    blocks = get_full_blocks(shape)
    missing = [start for start, _ in blocks if not os.path.exists(get_full_block_path(start, untiled))]
    assert not missing, f"{len(missing)} blocks are not segmented yet: {missing}"
    t0 = time.time()
    stitched = stitch_full_blocks(shape, untiled)
    merged, _, groups = merge_z_stacked(stitched, MIN_OVERLAP)
    del stitched
    cleaned = remove_small(merged, MIN_SIZE, copy=False)
    print(f"Merged {len(groups)} groups, {len(np.unique(cleaned)) - 1} objects of at least {MIN_SIZE:.0e} voxels, "
          f"{time.time() - t0:.0f} s")
    name = get_full_label_name(untiled)
    add_nuclei_labels(zarr_path, cleaned, name=name)
    plot_full(zarr_path, cleaned, name, plot_path)


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
    The candidates are multiples of 4 (the s2 grid).
    """
    import json
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["s0"], root["labels/nuclei_clean/s0"]
    results = []
    for z in range(z_range[0], z_range[1] + 1, z_step):
        z = z - z % 4
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
    z_mid = nz - 1 - s0_z // 2 ** RENDER_LEVEL
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

    # The raw plane of s1 (twice the resolution of s2), with the y axis of the labels.
    raw_level = RENDER_LEVEL - 1
    factor = 2 ** (RENDER_LEVEL - raw_level)
    plane = root[f"s{raw_level}"][
        s0_z // 2 ** raw_level, y0 * factor:y1 * factor, x0 * factor:x1 * factor
    ].astype("float32")
    low, high = np.percentile(plane, [1, 99.5])
    gray = np.clip((plane - low) / (high - low), 0, 1)
    step = 2.0 ** (raw_level - RENDER_LEVEL)

    def add_plane(rgb):
        def add(plotter):
            grid = pv.ImageData(
                dimensions=(rgb.shape[1], rgb.shape[0], 1), spacing=(step, step, 1.0), origin=(x0, y0, z_plane)
            )
            grid.point_data["rgb"] = (rgb.reshape(-1, 3) * 255).astype("uint8")
            plotter.add_mesh(grid, scalars="rgb", rgb=True, ambient=1.0, diffuse=0.0, specular=0.0,
                             show_scalar_bar=False)
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
    convert.add_argument("--voxel_size", type=float, nargs=3, default=None, help="z, y, x in nanometer.")
    convert.add_argument("--num_levels", type=int, default=4, help="The number of scale levels, s0 included.")
    convert.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))

    crops = subparsers.add_parser("segment_crops", help="Segment the 768 x 768 crops in 2d and plot them.")
    crops.add_argument("--zarr_path", default=OUTPUT_PATH)
    crops.add_argument("--level", type=int, default=0, help="The scale level of the crops.")

    predictions = subparsers.add_parser("predictions", help="Save and plot the decoder outputs on the crops.")
    predictions.add_argument("--zarr_path", default=OUTPUT_PATH)
    predictions.add_argument("--level", type=int, default=0, help="The scale level of the crops.")

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
    full.add_argument("--untiled", action="store_true", help="Segment each block in one piece.")

    stitch = subparsers.add_parser("stitch_full", help="Stitch the blocks and add them to the OME-Zarr.")
    stitch.add_argument("--zarr_path", default=OUTPUT_PATH)
    stitch.add_argument("--untiled", action="store_true", help="Stitch the untiled blocks to 'labels/nuclei_untiled'.")

    clean = subparsers.add_parser("clean_full", help="Remove the artifacts of 'labels/nuclei' as a new label image.")
    clean.add_argument("--zarr_path", default=OUTPUT_PATH)
    clean.add_argument("--untiled", action="store_true", help="Clean 'labels/nuclei_untiled'.")

    render = subparsers.add_parser("render", help="Render candidate 3d views of 'labels/nuclei_clean' (GPU node).")
    render.add_argument("--zarr_path", default=OUTPUT_PATH)

    section = subparsers.add_parser("render_section", help="Render the section views (GPU node) and plot them.")
    section.add_argument("--zarr_path", default=OUTPUT_PATH)
    section.add_argument("--plot_only", action="store_true", help="Only plot the panel from the saved renders.")
    section.add_argument("--s0_z", type=int, default=None, help="The slice; by default the best of the search.")

    args = parser.parse_args()
    if args.task == "convert":
        convert_to_ome_zarr(args.input_path, args.output_path, args.voxel_size, args.num_levels, args.num_workers)
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
        segment_full(args.zarr_path, args.untiled)
    elif args.task == "stitch_full":
        suffix = "_untiled" if args.untiled else ""
        stitch_full(args.zarr_path, f"beghin_em_3d_nuclei_s0_full{suffix}.png", args.untiled)
    elif args.task == "clean_full":
        suffix = "_untiled" if args.untiled else ""
        clean_full(args.zarr_path, f"beghin_em_3d_nuclei_s0_full{suffix}_clean.png", args.untiled)
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
