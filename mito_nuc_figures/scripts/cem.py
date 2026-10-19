"""Overview figures for the CEM-Mito-Benchmark (EMPIAR-10982): 6 vEM volumes and the 2D TEM benchmark."""
import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
import tifffile

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "cem"
DATA_DIR = os.path.join(DATA_ROOT, "mitolab", "10982", "data")
# Pixel sizes from the EMPIAR-10982 xml (given in Angstrom there).
PIXEL_SIZE_NM = {
    "c_elegans": 24, "fly_brain": 12, "glycolytic_muscle": 18, "hela_cell": 15, "lucchi_pp": 5, "salivary_gland": 15,
}
NOTES = "vEM benchmark ({name}), {size} nm xy pixel, curated instance mito labels"
N_TEM_SAMPLES = 16


def render_volume(name, out_dir):
    folder = os.path.join(DATA_DIR, "mito_benchmarks", name)
    raw_path, label_path = os.path.join(folder, f"{name}_em.tif"), os.path.join(folder, f"{name}_mito.tif")
    raw, labels = tifffile.memmap(raw_path, mode="r"), tifffile.memmap(label_path, mode="r")
    notes = NOTES.format(name=name, size=PIXEL_SIZE_NM[name])
    out_path = os.path.join(out_dir, f"benchmark_{name}.png")
    save_volume_figure(raw, {"mito": labels}, out_path, DATASET, f"benchmark_{name}", notes=notes)
    ids = np.unique(labels)
    return {
        "name": f"benchmark_{name}", "shape": raw.shape, "xy_pixel_size_nm": PIXEL_SIZE_NM[name],
        "path": folder, "label_dtype": str(labels.dtype), "n_mito_ids": int((ids != 0).sum()),
        "mito_fraction": round(float((labels > 0).mean()), 4),
    }


def render_tem(out_dir):
    images = sorted(glob(os.path.join(DATA_DIR, "tem_benchmark", "images", "*.tiff")))
    rng = np.random.default_rng(0)
    chosen = sorted(rng.choice(len(images), size=N_TEM_SAMPLES, replace=False).tolist())
    samples = []
    for idx in chosen:
        path = images[idx]
        label_path = path.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)
        name = os.path.splitext(os.path.basename(path))[0][:8]
        samples.append((name, imageio.imread(path), {"mito": imageio.imread(label_path)}))
    notes = f"2D TEM benchmark, variable low-nm pixel size, instance mito labels; seeded sample of {N_TEM_SAMPLES}"
    save_image_grid(samples, os.path.join(out_dir, "benchmark_tem.png"), DATASET, "benchmark_tem", notes=notes)
    return {"name": "benchmark_tem", "n_images": len(images), "path": os.path.join(DATA_DIR, "tem_benchmark")}


def main():
    out_dir = get_output_dir(DATASET)
    volumes = [render_volume(name, out_dir) for name in PIXEL_SIZE_NM]
    volumes.append(render_tem(out_dir))
    write_info(DATASET, {
        "dataset": "cem (CEM-Mito-Benchmark, EMPIAR-10982)", "classes": ["mito"],
        "modality": "FIB-SEM / SBF-SEM / ssTEM (vEM) + TEM",
        "dimensionality": "3d (6 volumes) + 2d (100 TEM images)",
        "label_quality": "manual / proofread instance labels (benchmark ground truth)",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": DATA_DIR, "download": "none, already local",
        "notes": (
            "mitolab_3d (roi_3d / roi_2d) is a micro-sam v1 benchmark preprocessing of these volumes, not rendered; "
            "the MitoLab training set (EMPIAR-11037) is in cem_mitolab"
        ),
        "issues": "", "observations": (
            "all 6 volumes aligned and look exhaustive (fly_brain, lucchi_pp, glycolytic_muscle, "
            "c_elegans best); hela_cell has a few mito-like objects unlabeled in z > 130; "
            "salivary_gland is a single cell in black padding with only ~1% mito; tem: aligned, mostly "
            "exhaustive, very heterogeneous images"
        ),
    })


if __name__ == "__main__":
    main()
