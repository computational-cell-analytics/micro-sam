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
}
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


def segment_image(image, model_type, engine, ndim=2):
    predictor, segmenter = get_predictor_and_segmenter(model_type=model_type, segmentation_mode=engine, ndim=ndim)
    embedding_dir = tempfile.mkdtemp(dir=os.environ.get("TMPDIR"))
    try:
        segmentation = automatic_instance_segmentation(
            predictor, segmenter, input_path=image, embedding_path=os.path.join(embedding_dir, "embeddings.zarr"),
            ndim=ndim, batch_size=None, verbose=False,
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
            segmentation = segment_image(volume, MODELS[run["target"]], engine, ndim=3)
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


if __name__ == "__main__":
    main()
