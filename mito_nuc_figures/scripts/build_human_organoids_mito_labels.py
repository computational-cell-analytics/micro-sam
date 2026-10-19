"""Write the human organoids mitochondria instance labels for the mitochondria EM generalist.

The mitochondria labels are manual semantic masks (int8, -128 background, 127 mitochondria), so touching mitochondria
form one connected component. Each connected component is split by a neck watershed: a seeded watershed on the
distance transform over-segments it, neighbouring segments merge unless the distance saddle on their shared boundary is
at most NECK_RATIO of the smaller of their peaks (a neck between two mitochondria, not the waist of one), and pieces
below MIN_PIECE voxels are absorbed by the neighbour they share most boundary with. The watershed runs per component in
its bounding box, which gives the same result as running it on the whole volume, with little memory.

The labels are written to one zarr store per volume, under a temporary name that is moved in place when complete.
"""
import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import cc3d
import h5py
import numpy as np
import zarr
from scipy import ndimage
from skimage.measure import label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed

from figure_utils import DATA_ROOT


VOLUMES = ("F059", "F107_A1", "F107_A2", "F107_A3")
NECK_RATIO = 0.5
MIN_PIECE = 5000
SEED_HEIGHT = 1.0
CHUNKS = (32, 256, 256)
LABEL_SLAB = 128

# The components of the volume, shared with the worker processes through fork.
COMPONENTS = None


def label_slabwise(mask):
    """3D connected components with 26-connectivity (as the training transform) into uint32, in merged z-slabs."""
    instances = np.zeros(mask.shape, dtype="uint32")
    offset = 0
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        labels, n = cc3d.connected_components(mask[z0:z0 + LABEL_SLAB], connectivity=26, return_N=True)
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
        h, w = below.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                b = below[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
                a = above[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
                touching = (b > 0) & (a > 0)
                for u, v in np.unique(np.stack([b[touching], a[touching]], axis=1), axis=0):
                    ru, rv = find(u), find(v)
                    if ru != rv:
                        parent[max(ru, rv)] = min(ru, rv)
    roots = np.array([find(i) for i in range(offset + 1)])
    unique_roots, lut = np.unique(roots, return_inverse=True)
    lut = lut.astype("uint32")
    for z0 in range(0, mask.shape[0], LABEL_SLAB):
        instances[z0:z0 + LABEL_SLAB] = lut[instances[z0:z0 + LABEL_SLAB]]
    return instances, len(unique_roots) - 1


def oversegment(mask):
    """Seeded watershed on the smoothed distance transform; every connected component gets at least one seed."""
    components = label(mask)
    distance = ndimage.distance_transform_edt(mask).astype("float32")
    smooth = ndimage.gaussian_filter(distance, 1.0)
    markers = label(h_maxima(smooth, SEED_HEIGHT) & mask)
    missing = np.setdiff1d(np.arange(1, components.max() + 1), np.unique(components[markers > 0]))
    rescue = np.isin(components, missing)
    markers[rescue] = components[rescue] + markers.max()
    return distance, watershed(-smooth, markers, mask=mask)


def pair_saddles(segments, distance):
    """Per pair of touching segments: the highest distance on their shared boundary and the boundary size."""
    us, vs, saddles = [], [], []
    for axis in range(segments.ndim):
        seg, dist = np.moveaxis(segments, axis, 0), np.moveaxis(distance, axis, 0)
        left, right = seg[:-1], seg[1:]
        touch = (left != right) & (left > 0) & (right > 0)
        u, v = left[touch].astype("int64"), right[touch].astype("int64")
        us.append(np.minimum(u, v))
        vs.append(np.maximum(u, v))
        saddles.append(np.minimum(dist[:-1], dist[1:])[touch])
    u, v, saddle = np.concatenate(us), np.concatenate(vs), np.concatenate(saddles)
    n = int(segments.max()) + 1
    key = u * n + v
    order = np.argsort(key)
    key, saddle = key[order], saddle[order]
    if len(key) == 0:
        return key, key, saddle, key
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    return key[starts] // n, key[starts] % n, np.maximum.reduceat(saddle, starts), np.diff(np.r_[starts, len(key)])


def merge_without_neck(segments, distance):
    """Merge segments that are not separated by a neck, then absorb pieces below MIN_PIECE voxels."""
    n = int(segments.max()) + 1
    peaks = ndimage.maximum(distance, segments, index=np.arange(n))
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    us, vs, saddles, contacts = pair_saddles(segments, distance)
    for u, v, saddle in zip(us, vs, saddles):
        if saddle > NECK_RATIO * min(peaks[u], peaks[v]):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[max(ru, rv)] = min(ru, rv)
    sizes = np.bincount(segments.ravel(), minlength=n)
    changed = True
    while changed:
        changed = False
        roots = np.array([find(i) for i in range(n)])
        group_size = np.zeros(n, dtype="int64")
        np.add.at(group_size, roots, sizes)
        best = {}
        for u, v, contact in zip(us, vs, contacts):
            ru, rv = roots[u], roots[v]
            if ru == rv:
                continue
            for a, b in ((ru, rv), (rv, ru)):
                if group_size[a] < MIN_PIECE and contact > best.get(a, (None, 0))[1]:
                    best[a] = (b, contact)
        for a, (b, _) in best.items():
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
                changed = True
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1][segments]


def split_component(task):
    """The pieces of one component as a local label array in its padded bounding box, or None if it stays whole."""
    component_id, box = task
    padded = tuple(slice(max(0, s.start - 1), s.stop + 1) for s in box)
    mask = COMPONENTS[padded] == component_id
    if mask.sum() < 2 * MIN_PIECE:
        return component_id, padded, None
    distance, segments = oversegment(mask)
    pieces = merge_without_neck(segments, distance)
    pieces[~mask] = 0
    return component_id, padded, (pieces if pieces.max() > 1 else None)


def split_volume(mask, n_workers):
    global COMPONENTS
    COMPONENTS, n_components = label_slabwise(mask)
    boxes = ndimage.find_objects(COMPONENTS)
    tasks = [(i + 1, box) for i, box in enumerate(boxes) if box is not None]
    next_id, n_split, n_pieces = n_components + 1, 0, 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(split_component, tasks, chunksize=64))
    instances = COMPONENTS
    for component_id, padded, pieces in results:
        if pieces is None:
            continue
        n_split += 1
        n_pieces += int(pieces.max())
        region = instances[padded]
        for piece in range(2, int(pieces.max()) + 1):
            region[pieces == piece] = next_id
            next_id += 1
    COMPONENTS = None
    return instances, {
        "n_components": int(n_components), "n_components_split": n_split, "n_pieces_of_split_components": n_pieces,
        "n_instances": int(n_components - n_split + n_pieces),
    }


def write_labels(instances, out_path):
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    group = zarr.open_group(tmp_path, mode="w")
    array = group.create_array("labels", shape=instances.shape, chunks=CHUNKS, dtype="uint32")
    for z in range(0, instances.shape[0], CHUNKS[0]):
        array[z:z + CHUNKS[0]] = instances[z:z + CHUNKS[0]]
    group.attrs.update({"neck_ratio": NECK_RATIO, "min_piece": MIN_PIECE, "seed_height": SEED_HEIGHT})
    # The labels of an earlier run are replaced.
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.replace(tmp_path, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(DATA_ROOT, "human_organoids_mito"), help="Output folder.")
    parser.add_argument("--volumes", nargs="+", default=list(VOLUMES), help="The volumes to process.")
    parser.add_argument("--n_workers", type=int, default=16, help="Processes for the per-component watershed.")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    for volume in args.volumes:
        path = os.path.join(DATA_ROOT, "human_organoids", "11380", "data", f"{volume}_bin2.h5")
        with h5py.File(path, "r") as f:
            labels = f["mitos"]
            mask = np.zeros(labels.shape, dtype=bool)
            for z in range(0, labels.shape[0], CHUNKS[0]):
                mask[z:z + CHUNKS[0]] = labels[z:z + CHUNKS[0]] > 0
        instances, info = split_volume(mask, args.n_workers)
        del mask
        write_labels(instances, os.path.join(args.output, f"{volume}_mito_labels.zarr"))
        print(f"{volume}: {info}", flush=True)
        with open(os.path.join(args.output, f"{volume}_info.json"), "w") as f:
            json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
