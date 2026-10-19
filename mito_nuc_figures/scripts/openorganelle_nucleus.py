import argparse
import json
import os
import subprocess

import numpy as np
import zarr

from cosem_utils import LazyView, count_ids, get_levels, group_path, index_map, open_remote
from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info


DATASET = "openorganelle_nucleus"
MAX_OVERVIEW_VOXELS = 3e8
N_SUBSET_PLANES = 16
LABEL_QUALITY = (
    "Unverified provenance: all labels live under 'labels/inference' (model predictions or semi-automatic, not "
    "documented as proofread). 22 instance / 8 binary; 26 nucleus masks, jrc_mus-liver-4/5/6 nuclear membranes and "
    "jrc_ctl-id8-2 (declared 'nucleus') is in fact a thick nuclear-envelope band."
)
DOWNLOAD = (
    "Loader download path works for all 30 (get_openorganelle_nucleus_data(download=True)): 13 full matched pairs at "
    "the loader label level + 17 crops (128^3 default, 200-320^3 fixed boxes); 3.2 GB on disk in total. Overview "
    "figures of the 17 cropped volumes are read lazily from S3 at a coarse level (finest label level <= 3e8 voxels, "
    "EM at the same scale), not stored. Full matched pairs of the cropped volumes at the loader level would be "
    "0.02-6 GB uncompressed for 10 of them (cos7-11, ctl-id8-3/4/5, ccl81-covid-1, sc-zp104a/105a, pancreas-4, "
    "skin-1, hippocampus-1: ~17 GB together) and 40-1470 GB for the 7 high-res ones (ctl-id8-2, choroid-plexus-2, "
    "dauer-larva, mosquito-stylet-6, liver-4/5/6), see volumes[*].full_download_size_gb_uncompressed."
)
NOTES = (
    "For full_array volumes the figure shows the loader cache (released label level); for cropped volumes <name>.png "
    "is a coarse whole-volume overview and <name>_loader_crop.png is exactly what the loader stores for training. "
    "<name>_zoom.png: 4x finer zoom for the nuclear-membrane volumes. jrc_mosquito-stylet-6 has a single 32 nm label "
    "level, so its overview shows 16 evenly spaced z planes (no ZX view)."
)
ISSUES = (
    "(1) jrc_ctl-id8-2: label is a ~150-200 nm thick nuclear-envelope band, not a nucleus mask, but the loader "
    "declares label_target='nucleus'. (2) jrc_mus-liver-4/5/6 (nuc_mem): thin membranes; at 128 nm and coarser they "
    "vanish (s5/s6 are all zero), the liver-4 128^3 crop has only 153 fg voxels in 9 sampled planes. (3) The loader "
    "trains crop volumes from a single 128^3 (or fixed box) crop each, i.e. 1-10 nuclei per volume, while e.g. "
    "jrc_mus-hippocampus-1 (1047 nuclei at 512 nm), jrc_mus-pancreas-4 (2214), jrc_mus-skin-1 (2058) hold "
    "thousands and would be only 3-6 GB as full arrays at 128 nm. (4) S3 "
    "data defect: jrc_mus-hippocampus-1 label s1 has corrupt chunks (blosc decompression error -1 in z-slabs 96 and "
    "384); s0 (used by the loader) and s2 read fine. (5) Loader crop shapes are not always the nominal size "
    "(e.g. ccl81-covid-1 (128, 88, 128), ctl-id8-4 (85, 84, 128)) because auto boxes are clipped at the volume "
    "border. (6) jrc_dauer-larva is TEM serial sections with ragged tile borders and resampled raw (40 nm label vs "
    "4*2^k nm EM); jrc_hela-h89-1/2 raw is resampled by the loader from 64 nm to 32 nm in z. (7) Licenses: "
    "jrc_mus-thymus-1 and aic_desmosome-3 unverified (others CC0-1.0). (8) The S3 bucket occasionally drops SSL "
    "connections; the loader has no retry."
)
OBSERVATIONS = (
    "Labels align with raw in all 30 volumes (XY and ZX checked; choroid-plexus-2 needed the em axis reversal, "
    "consistent with the loader). Filled nucleus masks, smooth boundaries, look like good automatic segmentations: "
    "near-exhaustive in thymus-1 (~700 nuclei per plane), hum-airway, skin-1, pancreas-4, hippocampus-1, "
    "choroid-plexus-2, fly-acc-calyx-1 (cell-body layer only), dauer-larva (crop), sc-zp104a/105a (few nuclei, "
    "all labeled), and the single-cell volumes (hela-*, cos7-11, ctl-id8-3/4/5, ccl81-covid-1, aic_desmosome-3, "
    "mosquito-stylet-6) where the 1-3 nuclei present are labeled. fly-fsb-1: sparse nuclei, mostly neuropil. "
    "epididymis-1/2: binary, elongated sperm nuclei, some thin ones only partly covered. ctl-id8-2: envelope band "
    "(see issues). liver-4/5/6: thin nuclear-membrane traces that follow the envelope at 32 nm, incl. lobulated "
    "nuclei; not usable as nucleus masks without filling. Many volumes are very flat in y (single adherent cells, "
    "62-156 px at 64 nm)."
)


def voxel_str(scale):
    return "x".join(f"{s:g}" for s in scale)


def label_description(info):
    target = "nuclear membrane" if info["label_target"] == "nuclear_membrane" else "nucleus"
    return f"{info['label_kind']} {target} ('labels/inference', provenance unverified)"


def modality(info):
    return "TEM" if info["em_name"].startswith("tem") else "FIB-SEM"


def choose_overview_level(levels, exclude=()):
    """The finest label level with at most MAX_OVERVIEW_VOXELS voxels, or None if every level is larger."""
    for level in sorted(levels, key=lambda lv: levels[lv]["scale"]):
        if level not in exclude and np.prod(levels[level]["shape"]) <= MAX_OVERVIEW_VOXELS:
            return level
    return None


def choose_em_level(em_levels, scale):
    """The coarsest EM level that is not coarser than the label scale (exact match if there is one)."""
    candidates = [lv for lv in em_levels if all(e <= s * 1.001 for e, s in zip(em_levels[lv]["scale"], scale))]
    if not candidates:
        return min(em_levels, key=lambda lv: em_levels[lv]["scale"])
    return max(candidates, key=lambda lv: em_levels[lv]["scale"])


def build_overview(name, info, exclude=()):
    """Lazy raw and label views of the whole volume at a coarse label level, on the label grid."""
    recon = info["recon"]
    label_path = group_path(name, recon, f"labels/inference/{info['label_subpath']}")
    em_path = group_path(name, recon, f"em/{info['em_name']}")
    label_levels, em_levels = get_levels(label_path), get_levels(em_path)
    level = choose_overview_level(label_levels, exclude)
    subset = level is None
    if subset:
        level = max(label_levels, key=lambda lv: label_levels[lv]["scale"])
    lab = label_levels[level]
    em_level = choose_em_level(em_levels, lab["scale"])
    em = em_levels[em_level]
    reverse = info["axis_reverse"] != "none"
    em_shape = em["shape"][::-1] if reverse else em["shape"]
    maps = index_map(lab["shape"], lab["scale"], lab["translation"], em_shape, em["scale"], em["translation"])
    label_maps = [np.arange(n) for n in lab["shape"]]
    if subset:
        planes = np.linspace(0, lab["shape"][0] - 1, N_SUBSET_PLANES + 2).astype(int)[1:-1]
        label_maps[0], maps[0] = planes, maps[0][planes]
    raw = LazyView(open_remote(f"{em_path}/{em_level}"), reverse=reverse, maps=maps)
    labels = LazyView(open_remote(f"{label_path}/{level}"), maps=label_maps)
    meta = {
        "overview_label_level": level, "overview_em_level": em_level, "overview_voxel_size_nm": lab["scale"],
        "overview_em_voxel_size_nm": em["scale"], "overview_shape": labels.shape, "em_axes_reversed": reverse,
        "em_resampled": tuple(em["scale"]) != tuple(lab["scale"]), "z_subset_planes": subset,
    }
    return raw, labels, meta, label_levels, em_levels


def download_size_gb(info, label_levels, em_levels):
    """Size of the full matched-resolution pair at the loader's label level, uncompressed."""
    lab = label_levels[info["label_level"]]
    em_level = choose_em_level(em_levels, lab["scale"])
    n_voxels = float(np.prod(lab["shape"]))
    raw_bytes = np.dtype(em_levels[em_level]["dtype"].replace("|", "")).itemsize
    label_bytes = np.dtype(lab["dtype"].replace("|", "")).itemsize
    return round(n_voxels * (raw_bytes + label_bytes) / 1e9, 2)


def du(path):
    return subprocess.run(["du", "-sh", path], capture_output=True, text=True).stdout.split()[0]


def render_cache(name, info, zarr_path, out_path, scale, title_suffix=""):
    root = zarr.open_group(zarr_path, mode="r")
    raw, labels = root["raw"], root["labels"]
    notes = f"{modality(info)}, {voxel_str(scale)} nm (zyx), {label_description(info)}{title_suffix}"
    save_volume_figure(raw, {"nucleus": labels}, out_path, DATASET, name, notes=notes)
    return raw.shape, count_ids(labels), dict(root.attrs)


def render_membrane_zoom(name, info, labels, meta, label_levels, out_dir, factor=4, size=768, z_step=8):
    """Zoom on the most labeled plane at a 4x finer level: thin membrane labels vanish in the coarse overview."""
    overview = np.asarray(labels[:])
    z = int(np.argmax((overview > 0).sum(axis=(1, 2))))
    yx = np.argwhere(overview[z] > 0)
    if len(yx) == 0:
        return {"membrane_zoom": "no labels in the overview level"}
    target_scale = tuple(s / factor for s in meta["overview_voxel_size_nm"])
    level = [lv for lv in label_levels if np.allclose(label_levels[lv]["scale"], target_scale)][0]
    shape = label_levels[level]["shape"]
    center = [z * factor] + [int(c) * factor for c in np.median(yx, axis=0)]
    zs = [min(max(center[0] + dz, 0), shape[0] - 1) for dz in (-z_step, 0, z_step)]
    y0 = min(max(center[1] - size // 2, 0), max(shape[1] - size, 0))
    x0 = min(max(center[2] - size // 2, 0), max(shape[2] - size, 0))
    region = (slice(y0, y0 + size), slice(x0, x0 + size))
    recon = info["recon"]
    lab_arr = open_remote(f"{group_path(name, recon, 'labels/inference/' + info['label_subpath'])}/{level}")
    em_arr = open_remote(f"{group_path(name, recon, 'em/' + info['em_name'])}/{level}")
    lab = np.stack([np.asarray(lab_arr[(zz,) + region]) for zz in zs])
    raw = np.stack([np.asarray(em_arr[(zz,) + region]) for zz in zs])
    notes = (
        f"{modality(info)}, zoom at {voxel_str(target_scale)} nm (label/EM {level}), planes z={zs}, "
        f"y0={y0}, x0={x0}, {label_description(info)}"
    )
    out_path = os.path.join(out_dir, f"{name}_zoom.png")
    save_volume_figure(raw, {"nucleus": lab}, out_path, DATASET, f"{name} zoom", notes=notes, ortho_view=False)
    return {
        "membrane_zoom_level": level, "membrane_zoom_planes": zs, "membrane_zoom_yx_offset": (y0, x0),
        "overview_label_foreground_fraction": float((overview > 0).mean()),
    }


def render_volume(name, info, out_dir):
    from torch_em.data.datasets.electron_microscopy.openorganelle_nucleus import (
        UNVERIFIED_LICENSE, get_openorganelle_nucleus_data
    )

    entry = {"name": name, "label_kind": info["label_kind"], "label_target": info["label_target"]}
    entry["license"] = "unverified" if name in UNVERIFIED_LICENSE else "CC0-1.0"
    raw, labels, meta, label_levels, em_levels = build_overview(name, info)
    loader_scale = label_levels[info["label_level"]]["scale"]
    entry.update({
        "loader_label_level": info["label_level"], "voxel_size_nm": loader_scale,
        "full_volume_shape_at_loader_level": label_levels[info["label_level"]]["shape"],
        "full_download_size_gb_uncompressed": download_size_gb(info, label_levels, em_levels),
        "loader_downloads_full_array": info["full_array"],
    })
    try:
        zarr_path = get_openorganelle_nucleus_data(os.path.join(DATA_ROOT, DATASET), name, download=True)
        entry.update({"path": zarr_path, "loader_cache_size": du(zarr_path), "loader_ok": True})
    except Exception as e:
        zarr_path = None
        entry.update({"path": None, "loader_ok": False, "issue": f"loader failed: {e!r}"})

    if info["full_array"] and zarr_path is not None:
        shape, n_ids, attrs = render_cache(name, info, zarr_path, os.path.join(out_dir, f"{name}.png"), loader_scale)
        entry.update({"shape": shape, "n_label_ids": n_ids, "resampled_by_loader": attrs.get("resampled")})
        return entry

    try:
        render_overview(name, info, raw, labels, meta, out_dir, entry)
    except RuntimeError as e:
        bad_level = meta["overview_label_level"]
        entry["issue"] = f"label level {bad_level} has corrupt chunks on S3 ({e}), overview uses the next coarser level"
        raw, labels, meta, label_levels, em_levels = build_overview(name, info, exclude=(bad_level,))
        render_overview(name, info, raw, labels, meta, out_dir, entry)
    if info["label_target"] == "nuclear_membrane":
        entry.update(render_membrane_zoom(name, info, labels, entry, label_levels, out_dir))
    if zarr_path is not None:
        crop_path = os.path.join(out_dir, f"{name}_loader_crop.png")
        shape, n_ids, attrs = render_cache(name, info, zarr_path, crop_path, loader_scale, ", torch-em loader crop")
        entry.update({"loader_crop_shape": shape, "loader_crop_n_ids": n_ids})
    return entry


def render_overview(name, info, raw, labels, meta, out_dir, entry):
    entry.update(meta)
    suffix = ", z-subset of 16 planes" if meta["z_subset_planes"] else ""
    levels = f"label {meta['overview_label_level']}, EM {meta['overview_em_level']}"
    notes = (
        f"{modality(info)}, overview at {voxel_str(meta['overview_voxel_size_nm'])} nm ({levels}){suffix}, "
        f"{label_description(info)}"
    )
    out_path = os.path.join(out_dir, f"{name}.png")
    save_volume_figure(
        raw, {"nucleus": labels}, out_path, DATASET, name, notes=notes, ortho_view=not meta["z_subset_planes"]
    )
    entry["shape"] = labels.shape
    if not meta["z_subset_planes"]:
        entry["n_label_ids_overview"] = count_ids(labels)


def main():
    from torch_em.data.datasets.electron_microscopy.openorganelle_nucleus import DATASETS

    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=None, help="Only render these volumes, without writing info.")
    parser.add_argument("--force", action="store_true", help="Re-render volumes that already have a cached entry.")
    args = parser.parse_args()
    out_dir = get_output_dir(DATASET)
    volumes = []
    entry_dir = os.path.join(out_dir, "entries")
    os.makedirs(entry_dir, exist_ok=True)
    for name, info in DATASETS.items():
        if args.names is not None and name not in args.names:
            continue
        entry_path = os.path.join(entry_dir, f"{name}.json")
        if os.path.exists(entry_path) and not args.force:
            with open(entry_path) as f:
                volumes.append(json.load(f))
            continue
        entry = render_volume(name, info, out_dir)
        with open(entry_path, "w") as f:
            json.dump(entry, f, default=str)
        print(name, entry.get("shape"), entry.get("voxel_size_nm"), entry.get("loader_cache_size"), flush=True)
        volumes.append(entry)
    if args.names is not None:
        return
    write_info(DATASET, {
        "classes": ["nucleus"], "modality": "FIB-SEM (29) / TEM (jrc_dauer-larva)", "dimensionality": "3d",
        "label_quality": LABEL_QUALITY, "n_volumes": len(volumes), "volumes": volumes,
        "data_path": os.path.join(DATA_ROOT, DATASET), "download": DOWNLOAD, "notes": NOTES, "issues": ISSUES,
        "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
