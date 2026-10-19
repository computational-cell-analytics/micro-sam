"""Contact sheets for DeepContact (2D SEM / TEM, LabelMe polygon annotations, mito label choice).

The zips in DATA_ROOT/deepcontact are broken (0 bytes and a 504 error page), and the folder is not writable.
The data was downloaded again with the torch-em loader into DATA_ROOT/deepcontact_redownload.
"""
import os
import json
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, write_info


DATASET = "deepcontact"
DATA_PATH = os.path.join(DATA_ROOT, "deepcontact_redownload")
SOURCES = {
    "cell": {"pixel_size_nm": 5, "modality": "SEM", "description": "SEM U-2 OS cultured cells, 5 nm/px"},
    "sem": {"pixel_size_nm": 10, "modality": "SEM", "description": "SEM Sertoli tissue cells, 10 nm/px"},
    "tem": {"pixel_size_nm": 4.68, "modality": "TEM", "description": "TEM, 4.68 nm/px"},
}
MITO_NAMES = ["Mito", "mito", "Mitochondria", "mitochondria"]
N_SAMPLES = 16
SEED = 42
ISSUES = (
    "DATA_ROOT/deepcontact/deepcontact.zip is 0 bytes and tem_data.zip is a 160 byte '504 Gateway Time-out' HTML "
    "page, so the original download failed; that folder is read-only for u28048, so the data was re-downloaded into "
    "DATA_ROOT/deepcontact_redownload. The loader only checks for zip existence, so with path=DATA_ROOT/deepcontact "
    "it would try to unzip the broken files. The loader rasterizes all mito polygons into one binary mask "
    "(deepcontact.py _rasterize_labelme_json), so the per-polygon instance information is lost. "
    "15 cell and 17 sem images have no mito polygon at all (ER only), and they are not filtered by the loader."
)
OBSERVATIONS = (
    "Polygons align well with the raw data. tem: exhaustive mito labels in all 67 images (single sample series, "
    "1071x1236 px). cell: dense labels in most images but 15/109 images have no mito label although mitos are "
    "visible (would act as false negatives); SEM images have an instrument data bar at the bottom; many near-duplicate "
    "neighbouring sections (e.g. 1-229 / 1-232). sem: most images are consecutive sections of one Sertoli cell series "
    "(104-147 look almost identical), labels cover the cytoplasm region densely; 17/73 images (e.g. 000021, "
    "2016OTO-B14-Semi1_4) have no mito labels despite visible mitos. Large images (up to 4096x6144) at 5 nm."
)


def count_mito_polygons(source):
    """Count the mito polygons per image, keyed by the h5 name that the torch-em preprocessing derives."""
    extract_dir = os.path.join(DATA_PATH, f"{source}_raw")
    counts = {}
    for json_path in sorted(glob(os.path.join(extract_dir, "**", "*.json"), recursive=True)):
        name = os.path.splitext(os.path.relpath(json_path, extract_dir))[0].replace(os.sep, "_")
        with open(json_path) as f:
            shapes = json.load(f).get("shapes", [])
        counts[name] = sum(shape.get("label") in MITO_NAMES for shape in shapes)
    return counts


def load_sample(path, title):
    with h5py.File(path, "r") as f:
        return (title, f["raw"][:], {"mito": f["labels/mito"][:]})


def render_source(source, out_dir, rng):
    paths = sorted(glob(os.path.join(DATA_PATH, "processed", source, "*.h5")))
    polygon_counts = count_mito_polygons(source)
    images = []
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(path, "r") as f:
            shape = f["raw"].shape
        images.append({"name": name, "shape": shape, "n_mito_polygons": polygon_counts.get(name), "path": path})
    n = min(N_SAMPLES, len(paths))
    chosen = sorted(rng.choice(len(paths), size=n, replace=False).tolist())
    samples = [load_sample(paths[i], images[i]["name"].replace(f"{source}_", "")[:24]) for i in chosen]
    n_polygons = sum(image["n_mito_polygons"] or 0 for image in images)
    notes = (
        f"{SOURCES[source]['description']}; {len(paths)} images, {n_polygons} mito polygons; "
        "manual polygons, rasterized to semantic masks by torch-em"
    )
    out_path = os.path.join(out_dir, f"{source}.png")
    save_image_grid(samples, out_path, DATASET, source, notes=notes)
    print(f"Rendered {out_path}")
    return {
        "name": source, "n_images": len(paths), "n_mito_polygons": n_polygons,
        "n_images_without_mito": sum(not image["n_mito_polygons"] for image in images),
        "voxel_size_nm": SOURCES[source]["pixel_size_nm"], "modality": SOURCES[source]["modality"],
        "shapes": sorted({tuple(image["shape"]) for image in images}),
        "path": os.path.join(DATA_PATH, "processed", source),
        "rendered": [images[i]["name"] for i in chosen], "images": images,
    }


def main():
    out_dir = get_output_dir(DATASET)
    rng = np.random.default_rng(SEED)
    volumes = [render_source(source, out_dir, rng) for source in SOURCES]
    write_info(DATASET, {
        "classes": ["mito"], "modality": "SEM (cell, sem) and TEM (tem)", "dimensionality": "2d",
        "label_quality": "manual polygons (LabelMe), rasterized to binary semantic masks by the loader",
        "n_volumes": sum(volume["n_images"] for volume in volumes), "volumes": volumes, "data_path": DATA_PATH,
        "download": (
            "re-downloaded with get_deepcontact_paths(download=True) into deepcontact_redownload: "
            "cell 2.0 GB + sem 0.8 GB + tem 0.1 GB zips, 3.5 GB extracted, 1.5 GB processed h5"
        ),
        "notes": "3 sources (cell / sem / tem); also ER and lipid droplet labels in the same h5 files.",
        "issues": ISSUES, "observations": OBSERVATIONS,
    })


if __name__ == "__main__":
    main()
