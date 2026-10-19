"""Write the cleaned nucleus instance labels of the human organoids volumes for the nucleus EM generalist.

The nucleus labels are manual semantic masks (int8, -128 background, 127 nucleus). They are cleaned as follows:
1. The speckled masks are closed (CLOSING_RADIUS) and the holes inside the nuclei are filled, per z-slice.
2. The labels on the black alignment padding are removed, so that nuclei are only labeled on valid raw. The padding is
   a connected region of the padding value of the volume (-128 or -108) above PADDING_MIN_AREA pixels of a slice.
3. The nuclei are separated into instances by 3D connected components.
4. Components below MIN_NUCLEUS_SIZE voxels are label fragments (speckle clouds and scribbles), not nuclei.
   They get IGNORE_LABEL. The entotic cell of F107_A1 also gets IGNORE_LABEL, because its nucleus is not labeled.

The labels are written to one zarr store per volume, under a temporary name that is moved in place when complete.
"""
import os
import json
import shutil
import argparse

import cc3d
import h5py
import zarr
import numpy as np
from scipy import ndimage
from skimage.morphology import disk

from figure_utils import DATA_ROOT


VOLUMES = {"F059": -128, "F107_A1": -128, "F107_A2": -108, "F107_A3": -108}
IGNORE_LABEL = np.iinfo("uint32").max
PADDING_MIN_AREA = 2000
MIN_NUCLEUS_SIZE = 100000
CLOSING_RADIUS = 4
LABEL_SLAB = 128
CHUNKS = (32, 256, 256)


def padding_mask(raw_slice, value):
    components, _ = ndimage.label(raw_slice == value)
    areas = np.bincount(components.ravel())
    return np.isin(components, np.where(areas > PADDING_MIN_AREA)[0]) & (components > 0)


def close_and_fill(mask):
    """Close the mask with a disk of CLOSING_RADIUS and fill its holes. The padding keeps nuclei at the border."""
    padded = np.pad(mask, CLOSING_RADIUS)
    closed = ndimage.binary_closing(padded, structure=disk(CLOSING_RADIUS))
    closed = closed[CLOSING_RADIUS:-CLOSING_RADIUS, CLOSING_RADIUS:-CLOSING_RADIUS] | mask
    return ndimage.binary_fill_holes(closed)


def label_slabwise(mask):
    """3D connected components (6-connectivity) into uint32, labeled in z-slabs that are merged across their borders.

    For the largest volume, the job memory cannot hold either alternative:
    - scipy's label holds an extra 8-byte array of the volume.
    - The cc3d output for the whole volume is uint64.

    Per slab, cc3d labels into uint32.
    """
    instances = np.zeros(mask.shape, dtype="uint32")
    offset = 0
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        labels, n = cc3d.connected_components(mask[z0:z0 + LABEL_SLAB], connectivity=6, return_N=True)
        labels = labels.astype("uint32", copy=False)
        labels[labels > 0] += offset
        instances[z0:z0 + LABEL_SLAB] = labels
        offset += n
    parent = np.arange(offset + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for z in range(LABEL_SLAB, mask.shape[0], LABEL_SLAB):
        below, above = instances[z - 1], instances[z]
        touching = (below > 0) & (above > 0)
        for a, b in np.unique(np.stack([below[touching], above[touching]], axis=1), axis=0):
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)
    roots = np.array([find(i) for i in range(offset + 1)])
    unique_roots, lut = np.unique(roots, return_inverse=True)
    lut = lut.astype("uint32")
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        instances[z0:z0 + LABEL_SLAB] = lut[instances[z0:z0 + LABEL_SLAB]]
    return instances, len(unique_roots) - 1


def clean_volume(path, padding_value):
    """Clean the nucleus labels of one volume, holding only the boolean mask and the instances in memory."""
    with h5py.File(path, "r") as f:
        raw, labels = f["raw"], f["nuclei"]
        nucleus = np.zeros(labels.shape, dtype=bool)
        n_filled = n_chipped = 0
        for z in range(labels.shape[0]):
            mask = labels[z] > 0
            filled = close_and_fill(mask)
            pad = padding_mask(raw[z], padding_value)
            n_filled += int((filled & ~mask).sum())
            n_chipped += int((filled & pad).sum())
            nucleus[z] = filled & ~pad
    instances, n_ids = label_slabwise(nucleus)
    del nucleus
    # Counted per slice: np.bincount casts its input to int64, a copy too large for the memory for the largest volume.
    sizes = np.zeros(n_ids + 1, dtype="int64")
    for plane in instances:
        sizes += np.bincount(plane.ravel(), minlength=n_ids + 1)
    is_fragment = sizes < MIN_NUCLEUS_SIZE
    is_fragment[0] = False
    n_fragment_voxels = n_entotic = n_nucleus = 0
    with h5py.File(path, "r") as f:
        entotic = f["entotic_cell"] if "entotic_cell" in f else None
        for z in range(instances.shape[0]):
            plane = instances[z]
            fragment = is_fragment[plane]
            n_fragment_voxels += int(fragment.sum())
            plane[fragment] = IGNORE_LABEL
            if entotic is not None:
                cell = entotic[z] > 0
                n_entotic += int(cell.sum())
                plane[cell] = IGNORE_LABEL
            n_nucleus += int(((plane > 0) & (plane != IGNORE_LABEL)).sum())
    info = {
        "n_nuclei": int(n_ids - is_fragment.sum()), "n_fragments_ignored": int(is_fragment.sum()),
        "fragment_voxels": n_fragment_voxels, "voxels_added_by_closing_and_filling": n_filled,
        "padding_voxels_removed": n_chipped, "entotic_voxels_ignored": n_entotic,
        "nucleus_fraction": n_nucleus / instances.size,
    }
    return instances, info


def write_labels(instances, out_path):
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    array = group.create_array("labels", shape=instances.shape, chunks=CHUNKS, dtype="uint32")
    for z in range(0, instances.shape[0], CHUNKS[0]):
        array[z:z + CHUNKS[0]] = instances[z:z + CHUNKS[0]]
    group.attrs["ignore_label"] = int(IGNORE_LABEL)
    # The labels of an earlier run are replaced.
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "human_organoids_nucleus"), help="Output folder.")
    parser.add_argument("--volumes", nargs="+", default=list(VOLUMES), help="The volumes to process.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    infos = {}
    for volume in args.volumes:
        path = os.path.join(DATA_ROOT, "human_organoids", "11380", "data", f"{volume}_bin2.h5")
        instances, info = clean_volume(path, VOLUMES[volume])
        write_labels(instances, os.path.join(args.output, f"{volume}_nucleus_labels.zarr"))
        print(f"{volume}: {info}")
        infos[volume] = info
        with open(os.path.join(args.output, f"{volume}_info.json"), "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
