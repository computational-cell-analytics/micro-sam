"""Analyze and export the nucleus segmentation of the whole expanded Platynereis (exm_lsm/high_res/stitched).

The analysis computes the unique nucleus ids and their sizes at full resolution (s0) with the block-wise
'bioimage_py.stats.unique', which reads one shard per block and merges the per-block counts. The table holds the id,
the number of voxels and the volume in cubic micrometer of each nucleus.

The export writes the segmentation at the three scales (s0, s1, s2) and the image at s1 and s2 as OME-TIFFs with the
voxel size in micrometer. The arrays are streamed slab by slab (one shard along z), so the 7 GB of s0 never have to fit
into memory at once. The files are BigTIFF, which Fiji opens with Bio-Formats and napari / tifffile read directly.

Usage:
    python export_expanded_platynereis.py  # the analysis and the export
    python export_expanded_platynereis.py --skip_export  # only the ids and sizes
    python export_expanded_platynereis.py --skip_analysis  # only the tiffs
"""

import os
import argparse

import numpy as np
import pandas as pd
import tifffile
import zarr

from bioimage_py.stats import unique

DATA_PATH = "/mnt/vast-nhr/projects/cidas/cca/data/huisken_data/exm_lsm/high_res/stitched/Expanded_Platynereis.ome.zarr"
OUTPUT_DIR = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/huisken_data/expanded_platynereis_export"
LABEL_KEY = "labels/nuclei"
LABEL_SCALES = ("s0", "s1", "s2")
IMAGE_SCALES = ("s1", "s2")


def get_voxel_size(group, scale):
    """The (z, y, x) voxel size of a scale in micrometer, from the OME-Zarr multiscales metadata."""
    datasets = group.attrs["ome"]["multiscales"][0]["datasets"]
    dataset = next(ds for ds in datasets if ds["path"] == scale)
    transform = next(trafo for trafo in dataset["coordinateTransformations"] if trafo["type"] == "scale")
    return tuple(float(s) for s in transform["scale"])


def compute_nucleus_sizes(root, output_dir, num_workers):
    """Compute the nucleus ids and their sizes at full resolution and save them as a csv table."""
    labels = root[f"{LABEL_KEY}/s0"]
    voxel_size = get_voxel_size(root[LABEL_KEY], "s0")
    # One block per shard, so that each shard is read and decompressed exactly once.
    block_shape = labels.shards if labels.shards is not None else labels.chunks
    print(f"Computing the unique ids of {labels.shape} with blocks {block_shape} and {num_workers} workers")
    ids, counts = unique(labels, return_counts=True, num_workers=num_workers, block_shape=block_shape)

    background_voxels = int(counts[ids == 0].sum())
    keep = ids != 0
    ids, counts = ids[keep], counts[keep]
    table = pd.DataFrame({
        "label_id": ids.astype("int64"), "n_voxels": counts, "volume_um3": counts * float(np.prod(voxel_size)),
    })
    table_path = os.path.join(output_dir, "nucleus_sizes_s0.csv")
    table.to_csv(table_path, index=False)

    print(f"Number of nuclei: {len(ids)} (max id: {ids.max() if len(ids) else 0})")
    print(f"Background voxels: {background_voxels} of {int(np.prod(labels.shape))}")
    if len(ids):
        print("Nucleus size in voxels: min {}, median {:.0f}, mean {:.0f}, max {}".format(
            counts.min(), np.median(counts), counts.mean(), counts.max()
        ))
    print(f"Saved the nucleus sizes to {table_path}")
    return table


def export_tiff(array, voxel_size, output_path, compression):
    """Stream a 3d zarr array into an OME-TIFF, one z-slice per page, reading one shard along z at a time."""
    slab = array.shards[0] if array.shards is not None else array.chunks[0]

    def pages():
        for z0 in range(0, array.shape[0], slab):
            data = array[z0:z0 + slab]
            for page in data:
                yield page

    metadata = {
        "axes": "ZYX",
        "PhysicalSizeZ": voxel_size[0], "PhysicalSizeZUnit": "µm",
        "PhysicalSizeY": voxel_size[1], "PhysicalSizeYUnit": "µm",
        "PhysicalSizeX": voxel_size[2], "PhysicalSizeXUnit": "µm",
    }
    tifffile.imwrite(
        output_path, pages(), shape=array.shape, dtype=array.dtype, bigtiff=True, ome=True,
        compression=compression, metadata=metadata, maxworkers=os.cpu_count(),
    )
    print(f"Saved {array.shape} {array.dtype} to {output_path}")


def export_tiffs(root, output_dir, compression, overwrite):
    """Export the segmentation at all scales and the image at s1 and s2."""
    exports = [(LABEL_KEY, scale, f"nuclei_{scale}.ome.tif") for scale in LABEL_SCALES]
    exports += [("", scale, f"image_{scale}.ome.tif") for scale in IMAGE_SCALES]
    for group_key, scale, name in exports:
        output_path = os.path.join(output_dir, name)
        if os.path.exists(output_path) and not overwrite:
            print(f"{output_path} exists already, skipping it")
            continue
        group = root[group_key] if group_key else root
        export_tiff(group[scale], get_voxel_size(group, scale), output_path, compression)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_path", default=DATA_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--num_workers", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--compression", default="zlib", help="The tiff compression, 'none' for no compression.")
    parser.add_argument("--skip_analysis", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the tiffs that exist already.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    root = zarr.open_group(args.input_path, mode="r")
    if not args.skip_analysis:
        compute_nucleus_sizes(root, args.output_dir, args.num_workers)
    if not args.skip_export:
        compression = None if args.compression == "none" else args.compression
        export_tiffs(root, args.output_dir, compression, args.overwrite)


if __name__ == "__main__":
    main()
