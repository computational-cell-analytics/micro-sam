"""Build the OpenOrganelle nucleus training volumes for the nucleus EM generalist.

The torch-em openorganelle_nucleus loader caches only a single 128^3 crop for several volumes that hold hundreds to
thousands of nuclei. This script downloads the full matched raw and label arrays of these volumes from the
janelia-cosem-datasets S3 bucket (at the loader label level, 64 nm for jrc_choroid-plexus-2), turns the binary labels
of the single-cell volumes into instances by connected components, and writes one zarr store per volume ('raw' and
'labels', uint32 instances) to <data>/openorganelle_nucleus_full, under a temporary name that is moved in place when
complete. The volumes that the loader already caches as full instance arrays are used from its cache. index.json
lists all training and validation volumes (paths relative to the data root), and a figure per new volume checks the
raw / label alignment.
"""
import argparse
import json
import os
import shutil
import time

import numpy as np
import zarr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa
from scipy import ndimage  # noqa
from skimage.segmentation import find_boundaries  # noqa
from torch_em.data.datasets.electron_microscopy.openorganelle_nucleus import (  # noqa
    BUCKET_URL, DATASETS, _multiscale_levels, _open_remote_zarr,
)

from figure_utils import DATA_ROOT, normalize  # noqa


CACHE_ROOT = os.path.join(DATA_ROOT, "openorganelle_nucleus")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "openorganelle_nucleus_full")
FIGURE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nucleus_only", "openorganelle_nucleus", "full_arrays"
)

# The volumes downloaded as full arrays, with the voxel size (nm) of the label level to use.
DOWNLOAD_VOLUMES = {
    "jrc_mus-pancreas-4": 128.0, "jrc_mus-skin-1": 128.0, "jrc_mus-hippocampus-1": 128.0,
    "jrc_mus-sc-zp104a": 96.0, "jrc_mus-sc-zp105a": 96.0, "jrc_choroid-plexus-2": 64.0,
    "jrc_cos7-11": 64.0, "jrc_ctl-id8-3": 128.0, "jrc_ctl-id8-4": 128.0, "jrc_ctl-id8-5": 128.0,
    "jrc_ccl81-covid-1": 64.0,
}
# The binary single-cell volumes of the loader cache that become instances by connected components.
BINARY_CACHED_VOLUMES = {"jrc_hela-21": 64.0, "aic_desmosome-3": 32.0}
# The full instance arrays of the loader cache that are used as they are.
REUSED_VOLUMES = {
    "jrc_mus-thymus-1": 128.0, "jrc_hum-airway-14953vc": 128.0, "jrc_fly-acc-calyx-1": 64.0, "jrc_fly-fsb-1": 64.0,
}
# The single-cell volumes that are thinner than the smallest patch side (128 px) in-plane.
EXCLUDED_THIN_VOLUMES = ("jrc_hela-22", "jrc_hela-bfa", "jrc_hela-h89-1", "jrc_hela-h89-2", "jrc_hela-nz-1")
VAL_VOLUMES = ("jrc_mus-sc-zp105a",)
CHUNKS = (32, 256, 256)
SLAB = 32
RETRIES = 8


def read_with_retries(array, slab=SLAB):
    """Read a remote array in z-slabs, retrying each slab on S3 errors."""
    out = np.empty(array.shape, dtype=array.dtype)
    for z0 in range(0, array.shape[0], slab):
        z1 = min(z0 + slab, array.shape[0])
        for attempt in range(RETRIES):
            try:
                out[z0:z1] = array[z0:z1]
                break
            except Exception as e:
                if attempt == RETRIES - 1:
                    raise
                print(f"Retrying slab {z0}-{z1} after {type(e).__name__}: {e}")
                time.sleep(2 ** attempt)
    return out


def find_level(levels, voxel_size):
    """The pyramid level with the given isotropic voxel size."""
    for level, scale in levels.items():
        if np.allclose(scale, voxel_size):
            return level, scale
    raise ValueError(f"No pyramid level at {voxel_size} nm in {levels}")


def download_volume(name, voxel_size):
    info = DATASETS[name]
    root = f"{name}/{name}.zarr/{info['recon']}"
    label_path = f"labels/inference/{info['label_subpath']}"
    label_level, label_scale = find_level(_multiscale_levels(f"{BUCKET_URL}{root}/{label_path}"), voxel_size)
    em_level, _ = find_level(_multiscale_levels(f"{BUCKET_URL}{root}/em/{info['em_name']}"), voxel_size)
    labels = read_with_retries(_open_remote_zarr(f"s3://janelia-cosem-datasets/{root}/{label_path}/{label_level}"))
    raw = read_with_retries(_open_remote_zarr(f"s3://janelia-cosem-datasets/{root}/em/{info['em_name']}/{em_level}"))
    if info["axis_reverse"] == "label":
        labels = np.ascontiguousarray(labels.transpose(2, 1, 0))
    elif info["axis_reverse"] == "raw":
        raw = np.ascontiguousarray(raw.transpose(2, 1, 0))
    # The independently stored pyramids can differ by a few voxels at the far border.
    common = tuple(min(r, lab) for r, lab in zip(raw.shape, labels.shape))
    if common != raw.shape or common != labels.shape:
        print(f"{name}: cropping raw {raw.shape} and labels {labels.shape} to {common}")
    raw = raw[tuple(slice(0, s) for s in common)]
    labels = labels[tuple(slice(0, s) for s in common)]
    if info["label_kind"] == "binary":
        labels = ndimage.label(labels > 0)[0]
    source = {
        "label_source": f"{root}/{label_path}/{label_level}", "em_source": f"{root}/em/{info['em_name']}/{em_level}"
    }
    return raw, labels, list(label_scale), source


def load_binary_cached(name):
    group = zarr.open_group(os.path.join(CACHE_ROOT, f"{name}.zarr"), mode="r")
    raw, labels = group["raw"][:], group["labels"][:]
    return raw, ndimage.label(labels > 0)[0], {"source": f"loader cache {name}.zarr, connected components"}


def write_volume(name, raw, labels, attrs):
    out_path = os.path.join(OUTPUT_ROOT, f"{name}.zarr")
    tmp_path = f"{out_path}.incomplete"
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    assert labels.max() < np.iinfo("uint32").max
    group = zarr.open_group(tmp_path, mode="w")
    group.create_array("raw", data=raw, chunks=CHUNKS)
    group.create_array("labels", data=labels.astype("uint32"), chunks=CHUNKS)
    group.attrs.update(attrs)
    os.replace(tmp_path, out_path)
    return out_path


def save_figure(name, raw, labels):
    os.makedirs(FIGURE_ROOT, exist_ok=True)
    z, y = raw.shape[0] // 2, raw.shape[1] // 2
    views = [("XY z=%d" % z, raw[z], labels[z]), ("ZX y=%d" % y, raw[:, y], labels[:, y])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (title, image, seg) in zip(axes, views):
        overlay = np.zeros(seg.shape + (4,), dtype="float32")
        overlay[find_boundaries(seg)] = (0.1, 0.6, 1.0, 1.0)
        ax.imshow(normalize(image.astype("float32")), cmap="gray")
        ax.imshow(overlay)
        ax.set_title(f"{title}, {len(np.unique(seg)) - 1} ids", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"openorganelle_nucleus / {name} full array {raw.shape} - nucleus instance outlines", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_ROOT, f"{name}.png"), dpi=70)
    plt.close(fig)


def count_nuclei(labels):
    return int(len(np.unique(labels)) - (1 if (labels == 0).any() else 0))


def build_volume(name):
    if name in DOWNLOAD_VOLUMES:
        raw, labels, voxel_size, attrs = download_volume(name, DOWNLOAD_VOLUMES[name])
    else:
        raw, labels, attrs = load_binary_cached(name)
        voxel_size = [BINARY_CACHED_VOLUMES[name]] * 3
    attrs.update({"dataset": name, "voxel_size_nm": voxel_size})
    path = write_volume(name, raw, labels, attrs)
    save_figure(name, raw, labels)
    entry = {"path": path, "shape": list(raw.shape), "voxel_size_nm": voxel_size, "n_nuclei": count_nuclei(labels)}
    with open(os.path.join(OUTPUT_ROOT, f"{name}.json"), "w") as f:
        json.dump(entry, f, indent=2)
    print(f"{name}: {entry}")


def build_index():
    index = []
    for name in sorted({**DOWNLOAD_VOLUMES, **BINARY_CACHED_VOLUMES}):
        with open(os.path.join(OUTPUT_ROOT, f"{name}.json")) as f:
            entry = json.load(f)
        index.append({"name": name, "raw_key": "raw", "label_key": "labels", **entry})
    for name, voxel_size in REUSED_VOLUMES.items():
        path = os.path.join(CACHE_ROOT, f"{name}.zarr")
        labels = zarr.open_group(path, mode="r")["labels"][:]
        index.append({
            "name": name, "raw_key": "raw", "label_key": "labels", "path": path, "shape": list(labels.shape),
            "voxel_size_nm": [voxel_size] * 3, "n_nuclei": count_nuclei(labels),
        })
    for entry in index:
        entry["path"] = os.path.relpath(entry["path"], DATA_ROOT)
        entry["split"] = "val" if entry["name"] in VAL_VOLUMES else "train"
    with open(os.path.join(OUTPUT_ROOT, "index.json"), "w") as f:
        json.dump(sorted(index, key=lambda e: e["name"]), f, indent=2)
    for entry in index:
        print(entry["name"], entry["split"], entry["shape"], entry["voxel_size_nm"], entry["n_nuclei"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", nargs="+", default=None, help="The volumes to build. Defaults to all.")
    parser.add_argument("--index_only", action="store_true", help="Only write index.json from the built volumes.")
    args = parser.parse_args()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    if not args.index_only:
        for name in args.volumes or sorted({**DOWNLOAD_VOLUMES, **BINARY_CACHED_VOLUMES}):
            build_volume(name)
    if args.index_only or args.volumes is None:
        build_index()


if __name__ == "__main__":
    main()
