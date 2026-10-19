import gzip
import os
import struct
import subprocess

import numpy as np
import requests
import zarr

from cosem_utils import BUCKET_URL, count_ids, get_json, get_levels, group_path
from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "janelia_nucleus"
NUC_SUBPATH = "labels/inference/segmentations/nuc"
DOWNLOAD = (
    "Full matched raw+label pair at the released (coarse) nucleus label level for all 16 volumes, via "
    "get_janelia_nucleus_data(download=True); 1.9 GB on disk in total (largest: guard-hair-follicle 406 MB). "
    "jrc_mus-pancreas-1 cached by the workaround in this script (raw from the n5 container)."
)
NOTES = (
    "Labels exist only at one coarse level: 64 nm (pancreas, granule-neurons; z 54-86 nm), 96 nm (guard-hair, "
    "meissner), 128 nm (kidney, liver-2, fly-mb), 32 nm (nacc, z 24-32 nm). The loader caches exactly this level, "
    "so training happens at 32-128 nm voxels. Loader download path works for training for 15/16 volumes."
)
ISSUES = (
    "Loader bug: jrc_mus-pancreas-1 has no 'em' group in its zarr container (janelia_nucleus.py:92, "
    "_find_em_array_name raises RuntimeError). Raw exists only in jrc_mus-pancreas-1.n5/em/fibsem-uint16 (s4 = "
    "54.4x64x64 nm matches label s0, same translation). Workaround here: read the n5 level over HTTP and write the "
    "cache in the loader format (attrs['workaround'] set), so the loader now returns the cached store. "
    "Minor: scale matching in _find_matching_em_level uses exact float equality (works for all 16 today). "
    "jrc_mus-nacc-2/3/4 are tiny (1-3 nuclei each, mostly cut at the volume border); jrc_mus-nacc-2 raw has very "
    "low contrast at this level."
)
OBSERVATIONS = (
    "All 16 volumes: labels aligned with raw (checked in XY and ZX), filled nucleus instances (not envelopes). "
    "Cellpose volumes look near-exhaustive: pancreas-1/2/3, granule-neurons-1/2/3, kidney, kidney-2, liver-2, "
    "guard-hair-follicle, meissner-1/2, fly-mb; boundaries are smooth and slightly coarse, a few small or very "
    "dark (heterochromatic) nuclei missed (e.g. liver-2), occasional merges/splits not obvious at this scale. "
    "fly-mb: only Kenyon cell body nuclei labeled; large neuropil regions have no nuclei. "
    "nacc-2/3/4 (manual): exhaustive and clean but contain only 1-3 nuclei each. "
    "Object counts: guard-hair 823, kidney 865, fly-mb 751, kidney-2 489, liver-2 223, meissner 141/147, "
    "pancreas 28/25/10, granule-neurons 6/32/17, nacc 1/3/1."
)
TISSUES = {
    "jrc_mus-pancreas": "mouse pancreas", "jrc_mus-granule-neurons": "mouse cerebellar granule neurons",
    "jrc_mus-guard-hair-follicle": "mouse guard hair follicle", "jrc_mus-kidney": "mouse kidney",
    "jrc_mus-liver": "mouse liver", "jrc_mus-meissner-corpuscle": "mouse Meissner corpuscle",
    "jrc_fly-mb": "fly mushroom body", "jrc_mus-nacc": "mouse nucleus accumbens",
}


def get_tissue(name):
    for prefix, tissue in TISSUES.items():
        if name.startswith(prefix):
            return tissue
    return "unknown"


def read_n5_level(path, shape_xyz, block_size_xyz):
    """Read a full N5 level (gzip, big-endian) over HTTP and return it in (z, y, x) order."""
    shape_zyx = tuple(shape_xyz[::-1])
    out = None
    grid = [int(np.ceil(s / b)) for s, b in zip(shape_xyz, block_size_xyz)]
    for bx in range(grid[0]):
        for by in range(grid[1]):
            for bz in range(grid[2]):
                resp = requests.get(f"{BUCKET_URL}{path}/{bx}/{by}/{bz}", timeout=120)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()
                data = resp.content
                mode, ndim = struct.unpack(">HH", data[:4])
                dims = struct.unpack(f">{ndim}I", data[4:4 + 4 * ndim])
                offset = 4 + 4 * ndim + (4 if mode == 1 else 0)
                block = np.frombuffer(gzip.decompress(data[offset:]), dtype=">u2").reshape(dims[::-1])
                if out is None:
                    out = np.zeros(shape_zyx, dtype="uint16")
                z0, y0, x0 = bz * block_size_xyz[2], by * block_size_xyz[1], bx * block_size_xyz[0]
                target = out[z0:z0 + dims[2], y0:y0 + dims[1], x0:x0 + dims[0]]
                target[:] = block[:target.shape[0], :target.shape[1], :target.shape[2]]
    return out


def cache_pancreas_1(zarr_path):
    """Workaround for jrc_mus-pancreas-1: its zarr container has no 'em' group, the raw data is only in the n5."""
    from zarr.codecs import BloscCodec
    from torch_em.data.datasets.electron_microscopy.janelia_nucleus import _read_zarr_array

    name, recon = "jrc_mus-pancreas-1", "recon-1"
    nuc_scale = get_levels(group_path(name, recon, NUC_SUBPATH))["s0"]["scale"]
    n5_em = f"{name}/{name}.n5/em/fibsem-uint16"
    level = None
    for ds in get_json(f"{BUCKET_URL}{n5_em}/attributes.json")["multiscales"][0]["datasets"]:
        if np.allclose(ds["transform"]["scale"], nuc_scale):
            level = ds["path"]
    attrs = get_json(f"{BUCKET_URL}{n5_em}/{level}/attributes.json")
    raw = read_n5_level(f"{n5_em}/{level}", attrs["dimensions"], attrs["blockSize"])
    labels = _read_zarr_array(f"s3://janelia-cosem-datasets/{group_path(name, recon, NUC_SUBPATH)}/s0")
    assert raw.shape == labels.shape, (raw.shape, labels.shape)
    root = zarr.open_group(zarr_path, mode="a")
    for key, data, shuffle in (("raw", raw, "shuffle"), ("labels", labels, "bitshuffle")):
        arr = root.create_array(
            key, shape=data.shape, chunks=tuple(min(128, s) for s in data.shape), dtype=data.dtype,
            compressors=BloscCodec(cname="zstd", clevel=6, shuffle=shuffle),
        )
        arr[:] = data
    root.attrs.update({
        "dataset": name, "recon": recon, "annotation_type": "cellpose", "labels_are_manual": False,
        "em_source": f"s3://janelia-cosem-datasets/{n5_em}/{level}",
        "label_source": f"s3://janelia-cosem-datasets/{group_path(name, recon, NUC_SUBPATH)}/s0",
        "workaround": "raw read from the n5 container, the zarr container has no em group",
    })


def get_volume(name):
    """Get the cached zarr store via the torch-em loader, with the pancreas-1 workaround."""
    from torch_em.data.datasets.electron_microscopy.janelia_nucleus import get_janelia_nucleus_data

    root_dir = os.path.join(DATA_ROOT, DATASET)
    try:
        return get_janelia_nucleus_data(root_dir, name, download=True), ""
    except Exception as e:
        if name != "jrc_mus-pancreas-1":
            raise
        zarr_path = os.path.join(root_dir, f"{name}.zarr")
        cache_pancreas_1(zarr_path)
        return zarr_path, f"loader fails ({e}); raw read from {name}.n5/em/fibsem-uint16 instead"


def get_scale(name, recon):
    return get_levels(group_path(name, recon, NUC_SUBPATH))["s0"]["scale"]


def render_volume(name, out_dir):
    from torch_em.data.datasets.electron_microscopy.janelia_nucleus import DATASETS

    _, recon, annotation = DATASETS[name]
    zarr_path, issue = get_volume(name)
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"]
    issue = issue or root.attrs.get("workaround", "")
    scale = get_scale(name, recon)
    quality = "manual dense (Amira)" if annotation == "manual" else "Cellpose 3.0.9 pseudo-labels"
    voxel = "x".join(f"{s:g}" for s in scale)
    notes = f"FIB-SEM {get_tissue(name)}, {voxel} nm (zyx, released label level), {quality}, instance"
    save_volume_figure(raw, {"nucleus": labels}, os.path.join(out_dir, f"{name}.png"), DATASET, name, notes=notes)
    size = subprocess.run(["du", "-sh", zarr_path], capture_output=True, text=True).stdout.split()[0]
    return {
        "name": name, "shape": raw.shape, "voxel_size_nm": scale, "path": zarr_path, "recon": recon,
        "raw_dtype": str(raw.dtype), "label_dtype": str(labels.dtype), "label_quality": quality,
        "n_nuclei": count_ids(labels), "em_source": root.attrs.get("em_source"), "cache_size": size,
        "issue": issue,
    }


def main():
    from torch_em.data.datasets.electron_microscopy.janelia_nucleus import DATASETS

    out_dir = get_output_dir(DATASET)
    volumes = [render_volume(name, out_dir) for name in DATASETS]
    for vol in volumes:
        print(f"{vol['name']}: {vol['shape']} {vol['voxel_size_nm']} nm, {vol['n_nuclei']} nuclei, {vol['cache_size']}")
    write_info(DATASET, {
        "classes": ["nucleus"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": "13 volumes Cellpose 3.0.9 pseudo-labels (cursory correction), 3 jrc_mus-nacc-* manual dense",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": os.path.join(DATA_ROOT, DATASET),
        "download": DOWNLOAD, "notes": NOTES, "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
