"""Contact sheets for the CEM-MitoLab 2D training data (EMPIAR-11037).

The data has one folder per source dataset ("group"). The 20 largest groups get one contact sheet each.
All other groups are covered by one contact sheet per imaging method, with each image from a different group.
"""
import json
import os
import re

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, write_info


DATASET = "cem"
INFO_NAME = "cem_mitolab"
MITOLAB_ROOT = os.path.join(DATA_ROOT, "mitolab", "11037")
IMAGE_ROOT = os.path.join(MITOLAB_ROOT, "cem_mitolab")
METADATA_PATH = os.path.join(MITOLAB_ROOT, "data", "metadata", "cem_mitolab_metadata.xlsx")
N_LARGEST = 20
N_SAMPLES = 12
N_METHOD_SAMPLES = 16
SEED = 42
ISSUES = (
    "3862 of 21871 images have empty masks; the loader drops them by default (non_empty_images.json, 18009 left), "
    "and 54 groups are then dropped completely. The train / val split in get_mitolab_paths is a random image-level "
    "split, so near-identical tiles of the same source volume end up in both. Several groups are tiles of public "
    "benchmarks that may be used for evaluation: Wei2020_MitoEM-H / -R (6583 images, 30% of the data), UroCell, "
    "kasthuri2011, CREMI, OpenOrganelle jrc_* cells, perez2014, Guay2020_patient1 (platelets, likely the "
    "DenseCell source); check for test-set leakage before training. One group ('UNKNOWN') has no metadata."
)
OBSERVATIONS = (
    "Instance labels align with the raw data in all inspected sheets. Most tiles look exhaustively labeled; some "
    "small TEM / SBF-SEM groups label only a subset of the visible mitos (e.g. Courson2019 video, single labels "
    "in large TEM fields), and MitoEM tiles skip some small mitos. Tile size varies from ~80x80 to 512x512, "
    "many 3D-derived tiles are small (<=224 px) and some are upsampled-looking low resolution crops. Huge domain "
    "diversity (FIB-SEM 423 groups, TEM 110, SBF-SEM 29, other 31), but the image count is dominated by MitoEM."
)


def load_metadata():
    table = pd.read_excel(METADATA_PATH, sheet_name="MetaData")
    table = table.set_index(table["Sample UID"].astype(str))
    columns = {
        "Imaging Method": "imaging_method", "Dimension": "source_dimension", "Organism": "organism",
        "Coarse Anatomical Origin": "anatomical_origin", "Cell Type": "cell_type", "Pixel size X": "pixel_size",
        "Units": "units", "CMM Bin XY": "bin_xy", "Datset info": "doi",
    }
    metadata = {}
    for uid, row in table.iterrows():
        entry = {}
        for column, key in columns.items():
            value = row[column]
            entry[key] = None if pd.isna(value) else (value.item() if hasattr(value, "item") else value)
        metadata[uid] = entry
    return metadata


def load_groups():
    with open(os.path.join(IMAGE_ROOT, "non_empty_images.json")) as f:
        non_empty = json.load(f)["images"]
    non_empty_per_group = {}
    for rel_path in non_empty:
        group = rel_path.split(os.sep)[0]
        non_empty_per_group.setdefault(group, []).append(os.path.basename(rel_path))
    groups = {}
    for group in sorted(os.listdir(IMAGE_ROOT)):
        image_dir = os.path.join(IMAGE_ROOT, group, "images")
        if not os.path.isdir(image_dir):
            continue
        images = sorted(name for name in os.listdir(image_dir) if name.endswith(".tiff"))
        groups[group] = {"images": images, "non_empty": sorted(non_empty_per_group.get(group, []))}
    return groups


def short_name(name, length=22):
    return name if len(name) <= length else name[:length - 2] + ".."


def safe_name(name):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def load_sample(group, image_name, title):
    raw = imageio.imread(os.path.join(IMAGE_ROOT, group, "images", image_name))
    labels = imageio.imread(os.path.join(IMAGE_ROOT, group, "masks", image_name))
    return (title, raw, {"mito": labels})


def describe(meta):
    if meta is None:
        return "no metadata"
    pixel_size = meta["pixel_size"]
    bin_xy = meta["bin_xy"] or 1
    size = "pixel size not given" if pixel_size is None else f"{pixel_size * bin_xy:g} nm/px after bin {bin_xy:g}"
    parts = [f"{meta['imaging_method']} {meta['source_dimension']}", meta["organism"], meta["cell_type"], size]
    return ", ".join(str(part) for part in parts if part)


def render_group(group, entry, meta, out_dir, rng):
    candidates = entry["non_empty"] if entry["non_empty"] else entry["images"]
    n = min(N_SAMPLES, len(candidates))
    chosen = sorted(rng.choice(candidates, size=n, replace=False).tolist())
    samples = [load_sample(group, name, f"#{i}") for i, name in enumerate(chosen)]
    notes = f"{describe(meta)}; {len(entry['images'])} images ({len(entry['non_empty'])} non-empty); manual instances"
    out_path = os.path.join(out_dir, f"mitolab_{safe_name(group)}.png")
    save_image_grid(samples, out_path, DATASET, f"mitolab {group}", notes=notes)
    return out_path


def render_methods(groups, metadata, rendered, out_dir, rng):
    by_method = {}
    for group, entry in groups.items():
        if group in rendered or not entry["non_empty"]:
            continue
        meta = metadata.get(group)
        method = meta["imaging_method"] if meta and meta["imaging_method"] else "unknown"
        by_method.setdefault(method, []).append(group)
    figures = {}
    for method, method_groups in sorted(by_method.items()):
        n = min(N_METHOD_SAMPLES, len(method_groups))
        chosen_groups = sorted(rng.choice(method_groups, size=n, replace=False).tolist())
        samples = []
        for group in chosen_groups:
            image_name = rng.choice(groups[group]["non_empty"])
            samples.append(load_sample(group, image_name, short_name(group)))
        n_images = sum(len(groups[group]["images"]) for group in method_groups)
        notes = (
            f"{method}: one random non-empty image each from {n} of the {len(method_groups)} remaining groups "
            f"({n_images} images); manual instances"
        )
        out_path = os.path.join(out_dir, f"mitolab_method_{safe_name(method)}.png")
        save_image_grid(samples, out_path, DATASET, f"mitolab {method} (small groups)", notes=notes)
        figures[method] = {"figure": out_path, "n_groups": len(method_groups), "rendered_groups": chosen_groups}
    return figures


def main():
    out_dir = get_output_dir(DATASET)
    rng = np.random.default_rng(SEED)
    metadata = load_metadata()
    groups = load_groups()
    by_size = sorted(groups, key=lambda group: -len(groups[group]["images"]))
    largest = by_size[:N_LARGEST]
    volumes = []
    for group in by_size:
        entry, meta = groups[group], metadata.get(group)
        figure = render_group(group, entry, meta, out_dir, rng) if group in largest else None
        volumes.append({
            "name": group, "n_images": len(entry["images"]), "n_non_empty": len(entry["non_empty"]),
            "path": os.path.join(IMAGE_ROOT, group), "figure": figure, **(meta or {"metadata": "missing"}),
        })
        if figure:
            print(f"Rendered {figure}")
    method_figures = render_methods(groups, metadata, set(largest), out_dir, rng)
    n_images = sum(len(entry["images"]) for entry in groups.values())
    n_non_empty = sum(len(entry["non_empty"]) for entry in groups.values())
    methods = {}
    for volume in volumes:
        method = volume.get("imaging_method") or "unknown"
        methods.setdefault(method, {"n_groups": 0, "n_images": 0})
        methods[method]["n_groups"] += 1
        methods[method]["n_images"] += volume["n_images"]
    print(f"{len(groups)} groups, {n_images} images, {n_non_empty} non-empty")
    write_info(INFO_NAME, {
        "classes": ["mito"], "modality": "mixed vEM / TEM (FIB-SEM, TEM, SBF-SEM, ssTEM, STEM, SEM, ATUM, ET)",
        "dimensionality": "2d", "label_quality": "manual / proofread instance labels (CEM-MitoLab curation)",
        "n_volumes": len(groups), "n_images": n_images, "n_non_empty_images": n_non_empty,
        "imaging_methods": methods, "method_figures": method_figures, "volumes": volumes,
        "data_path": IMAGE_ROOT, "download": "already present (EMPIAR-11037, 6.3 GB extracted)",
        "notes": "One folder per source dataset (group); 2D tiles of 224x224 to 512x512 px, uint8, instance masks.",
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
