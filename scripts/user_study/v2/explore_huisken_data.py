"""Automatic instance segmentation with AIS and APG and 'hvit_l_cells' on a block of the huisken_data volume.

The volume is a light-sheet stack of nuclei (4871, 2048, 2048) uint16 (ZYX). 'hires_volume.h5' holds it as 'raw',
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
"""

import os
import time
import shutil
import argparse
from concurrent import futures

import h5py
import numpy as np
from scipy import ndimage
from skimage.measure import block_reduce

from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation, get_predictor_and_segmenter
from micro_sam.v2.batched_inference import NORMALIZATION_SAMPLE_SLICES, _volume_normalization_bounds


DATA_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/huisken_data/hires_volume.h5"
RAW_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/huisken_data/S000_t000000_V000_R0009_X000_Y000_C01_I0_D0_P04871.raw"
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

    # Union-find over the matches, then consecutive ids.
    parent = np.arange(offset + 1, dtype="int64")

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for lower_id, upper_id in matches:
        parent[find(upper_id)] = find(lower_id)
    roots = np.array([find(node) for node in range(offset + 1)])
    present = np.zeros(offset + 1, dtype=bool)
    for z in range(0, shape[0], 64):
        present[np.unique(segmentation[z:z + 64])] = True
    roots[~present] = 0
    _, lut = np.unique(roots, return_inverse=True)
    lut = lut.astype("uint32")
    for z in range(0, shape[0], 64):
        segmentation[z:z + 64] = lut[segmentation[z:z + 64]]
    print(f"Stitched {int(lut.max())} objects in {time.time() - begin:.1f} s", flush=True)
    return segmentation


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
    args = parser.parse_args()

    roi = get_roi(args.start, args.shape)
    if args.n_slabs:
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
