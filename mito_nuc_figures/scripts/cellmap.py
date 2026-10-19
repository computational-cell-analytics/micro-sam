"""Render the mito / nucleus annotated crops of the CellMap Segmentation Challenge.

The crops are read directly from the zip of the challenge data ('matched_res_no_pad').
This zip holds the EM volumes only for the regions covered by the 289 crops of the challenge manifest.
For each crop, the label scale level that matches an EM scale level is used (the finest one),
following torch-em's '_download_cellmap_data'.
"""
import os
import json
import zipfile
import argparse

import zarr
import numpy as np
import pandas as pd

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "cellmap"
DATA_DIR = os.path.join(DATA_ROOT, "cellmap-segmentation-challenge")
ZIP_PATH = os.path.join(DATA_DIR, "matched_res_no_pad.zip")
ZIP_ROOT = "data_matched_res_no_pad"
CROP_TABLE = os.path.join(get_output_dir(DATASET), "cellmap_mito_nuc.csv")
CLASS_MAP = {"mito": "mito", "nuc": "nucleus"}
SLIVER_FRAC = 2e-3
LABEL_QUALITY = (
    "manual, dense per crop (every voxel of the crop is annotated for every class in the crop), instance ids; "
    "39 of the 147 crops with mito objects hold a single mito id (one object or semantic, cannot tell)"
)
DOWNLOAD = (
    "nothing downloaded; read directly from the existing matched_res_no_pad.zip (42 GB, 1.3M files) with "
    "zarr ZipStore, no extraction, no cache"
)
NOTES = (
    "CellMap Segmentation Challenge FIB-SEM, 22 volumes, 194 crops (one figure per crop) that have mito and/or nuc "
    "objects. Labels at their finest scale level that matches an EM scale level (torch-em logic): 2 nm (cos7), "
    "~3.2-5.2 x 4 x 4 nm (hela/jurkat/macrophage/ctl/sum159-1), 4 nm (fly, nacc), 8 nm (mouse tissue, sum159-4, ut21), "
    "16/32/64 nm (large nucleus and mito crops). Label value 255 = unknown (none found). "
    "Overview sheet per volume: middle slice per crop, or the most labeled slice if the middle one is empty; "
    "'n/a' rows = no label array for that class in the crop."
)
ISSUES = (
    "torch-em cellmap.py:237-238 + 267-269: when a crop reaches beyond the EM volume, the padded EM slice is "
    "clipped to the EM shape but the labels get no negative padding, so raw_crop and label_crop shapes differ "
    "(happens for jrc_cos7-1a crop254, label z-extent 400 vs 399 EM slices; here the labels are clipped to the EM). "
    "cellmap.py:267 casts labels to int16 (jrc_zf-cardiac-1 crop380 mito is uint16 with 271 ids, fine, but "
    "unsafe in general) and keeps 255 (unknown) as a normal id instead of mapping it to -1 like the pad. "
    "'annotated empty' is not always trustworthy: jrc_sum159-4 crop211 (8 nm, 800^3) has an all-zero mito array "
    "although many mitochondria are visible in the cytoplasm below the nucleus, so the large nucleus-oriented "
    "crops should not be used as mito negatives. "
    "Near-duplicate crops (same region, re-annotated, labels differ slightly): cos7-1b 240/291, fly-vnc-1 173/185, "
    "hela-3 101/181, jurkat-1 126/180/182, mus-liver 157/183; split by region, not by crop id, to avoid leakage. "
    "Several fine crops lie inside coarse crops of the same volume (e.g. cos7-1a 248/256/292 in 257, cos7-1b "
    "240/245/258/291 in 238, mus-liver-zon-1 266 in 289 and 273 in 349, sum159-4 209 in 211)."
)
OBSERVATIONS = (
    "All 194 crops render with labels aligned to the EM at every checked scale (2, ~3.4x4x4, 4, 8, 16, 32, 64 nm); "
    "raw/label equality was verified against the torch-em reference crop_234.h5 (identical). "
    "Labels look exhaustive within each crop: tissue crops at 8-32 nm (mus-liver-zon-1/2, mus-kidney, zf-cardiac, "
    "ut21, sum159-4) have 10-270 dense mito instances; nucleus crops at 32 nm (mus-liver-zon-1/2, mus-heart-1, "
    "zf-cardiac-1) label all nuclei incl. small endothelial/sinusoidal ones. "
    "Most cell-line crops are tiny (100-400 voxels per side, often < 1 um^3 at 2-4 nm) and contain just 1-3 "
    "mitochondria, mostly cut by the crop border, so they offer little context per crop. "
    "15 crops are completely inside a nucleus (nucleus fg = 1.0, e.g. hela-2 94/95/96, hela-3 85/86/87, "
    "jurkat 91/92/93, macrophage-2 88/89/90, sum159-1 97/98/99): no nuclear boundary, "
    "useless for nucleus instance training. "
    "Slivers (class fg < 0.2%): mus-kidney crop158 nuc (0.006%), mus-liver crop125 mito (0.04%), hela-3 crop33 nuc, "
    "jurkat-1 crop47 nuc, ctl-id8-1 crop116 mito, mus-kidney crop166 mito, jurkat-1 crop37 mito, "
    "mus-liver-zon-1 crop279 mito. "
    "A few mito-like objects are unlabeled in small cell-line crops (hela-2 crop4 marked mito-empty, jurkat-1 "
    "crop112, sum159-4 crop219); they may be other organelles, but treat mito-empty cell-line crops with care. "
    "Imaging artifacts: vertical curtaining stripes in the 32 nm liver crops (mus-liver-zon-1 349, mus-liver-zon-2 "
    "358/367) and zf-cardiac; strong noise in the 2 nm cos7 crops. "
    "Nucleus arrays exist in 190 of the 194 crops, mito arrays in 177; crops without an array are shown as 'n/a'."
)


def read_json(zip_file, key):
    """Read a json document from the zip file."""
    try:
        return json.loads(zip_file.read(key))
    except KeyError:
        return None


def get_levels(attrs):
    """Get the (path, scale, translation) of all scale levels from OME-Zarr multiscales attributes."""
    levels = []
    for dataset in attrs["multiscales"][0]["datasets"]:
        transforms = {t["type"]: t for t in dataset["coordinateTransformations"]}
        levels.append((dataset["path"], transforms["scale"]["scale"], transforms["translation"]["translation"]))
    return levels


def match_level(label_levels, em_levels):
    """Find the finest pair of label level and EM level with the same scale."""
    matches = []
    for label_path, label_scale, label_translation in label_levels:
        for em_path, em_scale, em_translation in em_levels:
            if np.allclose(label_scale, em_scale, rtol=1e-3, atol=1e-6):
                match = (label_path, em_path, label_scale, label_translation, em_translation)
                matches.append((np.prod(label_scale), *match))
    if not matches:
        return None
    matches.sort(key=lambda m: m[0])
    return matches[0][1:]


def load_crop(store, zip_file, volume, crop_id, em_levels, class_map=CLASS_MAP):
    """Load the EM crop and the labels of the classes in *class_map* of one crop, at the finest matching scale."""
    zarr_root = f"{ZIP_ROOT}/{volume}/{volume}.zarr/recon-1"
    candidates = {}
    for class_name in class_map:
        group = f"{zarr_root}/labels/groundtruth/crop{crop_id}/{class_name}"
        attrs = read_json(zip_file, f"{group}/.zattrs")
        if attrs is None:
            continue
        match = match_level(get_levels(attrs), em_levels)
        if match is None:
            print(f"No EM scale matches the {class_name} labels of crop {crop_id}")
            continue
        encoding = attrs["cellmap"]["annotation"]["annotation_type"]["encoding"]
        candidates[class_name] = (group, encoding, attrs["cellmap"]["annotation"]["annotation_type"]["type"], *match)

    # All classes of a crop are shown at the finest scale among them, coarser labels are upsampled.
    ref_class = min(candidates, key=lambda name: np.prod(candidates[name][5]))
    group, _, _, label_path, em_path, scale, translation, em_translation = candidates[ref_class]
    shape = zarr.open_array(store=store, path=f"{group}/{label_path}", mode="r", zarr_format=2).shape
    em = zarr.open_array(store=store, path=f"{zarr_root}/em/fibsem-uint8/{em_path}", mode="r", zarr_format=2)
    starts = [int(round((t - et) / s)) for t, et, s in zip(translation, em_translation, scale)]
    # Some crops reach beyond the EM volume, the labels are clipped to the EM extent in that case.
    bb = tuple(slice(max(start, 0), min(start + size, n)) for start, size, n in zip(starts, shape, em.shape))
    label_bb = tuple(slice(b.start - start, b.stop - start) for b, start in zip(bb, starts))
    raw = em[bb]
    clipped = tuple(raw.shape) != tuple(shape)

    labels, label_info = {}, {}
    for class_name, (group, encoding, ann_type, label_path, _, class_scale, class_translation, _) in candidates.items():
        data = zarr.open_array(store=store, path=f"{group}/{label_path}", mode="r", zarr_format=2)[:]
        n_unknown = int((data == encoding["unknown"]).sum())
        data = np.where(data == encoding["unknown"], 0, data)
        if not np.allclose(class_scale, scale):
            data = resample_labels(data, class_scale, class_translation, scale, translation, shape)
        data = data[label_bb]
        labels[class_map[class_name]] = data
        ids = np.unique(data)
        label_info[class_map[class_name]] = {
            "label_level": label_path, "label_scale_nm": class_scale, "annotation_type": ann_type,
            "n_ids": int((ids != 0).sum()), "max_id": int(ids.max()), "n_unknown_voxels": n_unknown,
            "fg_frac": float((data > 0).mean()), "upsampled": not np.allclose(class_scale, scale),
        }
    crop_info = {
        "em_level": em_path, "voxel_size_nm": scale, "translation_nm": translation, "em_start_voxel": starts,
        "label_shape": shape, "clipped_to_em": clipped, "raw_zero_frac": float((raw == 0).mean()),
        "labels": label_info,
    }
    return raw, labels, crop_info


def resample_labels(data, scale, translation, target_scale, target_translation, target_shape):
    """Nearest neighbor resampling of labels to the voxel grid of another class."""
    coords = [
        np.clip(np.round((tt + np.arange(n) * ts - t) / s).astype(int), 0, size - 1)
        for t, s, tt, ts, n, size in zip(translation, scale, target_translation, target_scale, target_shape, data.shape)
    ]
    return data[np.ix_(*coords)]


def make_notes(volume, crop_info):
    """Make the one line figure notes."""
    voxel_size = ", ".join(f"{v:g}" for v in crop_info["voxel_size_nm"])
    types = []
    for name, info in crop_info["labels"].items():
        if info["n_ids"] == 0:
            types.append(f"{name} annotated empty")
            continue
        kind = "instance" if info["max_id"] > 1 else "single id"
        types.append(f"{name} {kind} {info['n_ids']} ids")
    return f"FIB-SEM {volume}, voxel ({voxel_size}) nm zyx, manual dense crop labels: " + "; ".join(types)


def pick_overview_slice(labels, n_z):
    """Pick the middle slice, or the slice with most foreground if the middle slice has no labels."""
    mid = n_z // 2
    if any((lab[mid] > 0).any() for lab in labels.values()):
        return mid
    fg = sum((lab > 0).sum(axis=(1, 2)) for lab in labels.values())
    return int(np.argmax(fg))


def load_crop_table():
    """Load the per-crop presence table and group it by crop."""
    table = pd.read_csv(CROP_TABLE)
    present = table[table.status == "present"]
    crops = sorted(set(zip(present.dataset, present.crop)))
    stats = {}
    for volume, crop_id, class_name, frac, status in zip(
        table.dataset, table.crop, table["class"], table.present_frac, table.status
    ):
        stats[(volume, crop_id, class_name)] = {"present_frac": frac, "status": status}
    return crops, stats


def render_crop(store, zip_file, volume, crop_id, em_levels, stats, out_dir):
    """Render the figure of one crop and return its info and overview sample."""
    raw, labels, crop_info = load_crop(store, zip_file, volume, crop_id, em_levels)
    out_path = os.path.join(out_dir, volume, f"crop{crop_id}.png")
    notes = make_notes(volume, crop_info)
    save_volume_figure(raw, labels, out_path, DATASET, f"{volume} crop{crop_id}", notes=notes)
    present = {}
    for class_name, name in CLASS_MAP.items():
        if (volume, crop_id, class_name) in stats:
            present[name] = stats[(volume, crop_id, class_name)]
    info = {
        "name": f"{volume}/crop{crop_id}", "volume": volume, "crop": int(crop_id), "shape": raw.shape,
        "classes_present": [name for name, s in present.items() if s["status"] == "present"],
        "classes_annotated": list(labels), "present_frac": {name: s["present_frac"] for name, s in present.items()},
        "path": f"{ZIP_PATH}::{ZIP_ROOT}/{volume}/{volume}.zarr/recon-1/labels/groundtruth/crop{crop_id}",
        "figure": out_path, **crop_info,
    }
    z = pick_overview_slice(labels, raw.shape[0])
    # Classes without a label array get an empty 'n/a' row, so that the rows of the contact sheet stay aligned.
    empty = np.zeros(raw.shape[1:], dtype="uint8")
    sample_labels = {
        name if name in labels else f"{name} n/a": labels[name][z] if name in labels else empty
        for name in CLASS_MAP.values()
    }
    sample = (f"crop{crop_id} z={z}", raw[z], sample_labels)
    print(f"{volume} crop{crop_id}: shape {raw.shape}, {notes}")
    return info, sample


def render_volume(store, zip_file, volume, crop_ids, stats, out_dir):
    """Render all crops of one volume and the volume overview contact sheet."""
    em_attrs = read_json(zip_file, f"{ZIP_ROOT}/{volume}/{volume}.zarr/recon-1/em/fibsem-uint8/.zattrs")
    em_levels = get_levels(em_attrs)
    infos, samples = [], []
    for crop_id in crop_ids:
        info, sample = render_crop(store, zip_file, volume, crop_id, em_levels, stats, out_dir)
        infos.append(info)
        samples.append(sample)
    voxel_sizes = sorted(set(", ".join(f"{v:g}" for v in info["voxel_size_nm"]) for info in infos))
    notes = f"FIB-SEM, voxel size(s) zyx nm: {' | '.join(voxel_sizes)}; middle slice per crop (most labeled if empty)"
    out_path = os.path.join(out_dir, f"{volume}_overview.png")
    save_image_grid(samples, out_path, DATASET, f"{volume} ({len(samples)} crops)", notes=notes, ncols=6)
    return infos


def render_all(out_dir, volume_names):
    """Render all crops (optionally only of some volumes) and cache the crop infos."""
    crops, stats = load_crop_table()
    by_volume = {}
    for volume, crop_id in crops:
        by_volume.setdefault(volume, []).append(int(crop_id))
    store = zarr.storage.ZipStore(ZIP_PATH, mode="r")
    zip_file = zipfile.ZipFile(ZIP_PATH)
    volumes = []
    for volume, crop_ids in sorted(by_volume.items()):
        if volume_names and volume not in volume_names:
            continue
        volumes.extend(render_volume(store, zip_file, volume, crop_ids, stats, out_dir))
    if not volume_names:
        with open(os.path.join(out_dir, "crop_infos.json"), "w") as f:
            json.dump(volumes, f, indent=2, default=str)


def save_info(out_dir):
    """Write the dataset summary from the cached crop infos."""
    with open(os.path.join(out_dir, "crop_infos.json")) as f:
        volumes = json.load(f)
    for volume in volumes:
        flags = []
        for name, info in volume["labels"].items():
            if 0 < info["fg_frac"] < SLIVER_FRAC:
                flags.append(f"{name} sliver")
            if info["fg_frac"] > 0.999:
                flags.append(f"crop fully inside {name}")
        if volume["clipped_to_em"]:
            flags.append("labels reach beyond the EM volume")
        volume["flags"] = flags
    write_info(DATASET, {
        "classes": ["mito", "nucleus"], "modality": "FIB-SEM", "dimensionality": "3d",
        "label_quality": LABEL_QUALITY, "n_volumes": len(volumes),
        "n_em_volumes": len(set(v["volume"] for v in volumes)),
        "volumes": volumes, "data_path": ZIP_PATH, "download": DOWNLOAD, "notes": NOTES, "issues": ISSUES,
        "observations": OBSERVATIONS,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes", nargs="*", default=None, help="Only render these volumes (for testing).")
    parser.add_argument("--info_only", action="store_true", help="Only write info.json from the cached crop infos.")
    args = parser.parse_args()
    out_dir = get_output_dir(DATASET)
    if not args.info_only:
        render_all(out_dir, args.volumes)
    if not args.volumes:
        save_info(out_dir)


if __name__ == "__main__":
    main()
