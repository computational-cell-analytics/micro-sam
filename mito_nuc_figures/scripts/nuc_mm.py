import os
from glob import glob

import h5py
import numpy as np

from figure_utils import DATA_ROOT, get_output_dir, save_image_grid, save_volume_figure, write_info


DATASET = "nuc_mm"
ROOT = os.path.join(DATA_ROOT, DATASET)
ZEBRAFISH_ROOT = os.path.join(ROOT, "Zebrafish (NucMM-Z)")
MOUSE_ROOT = os.path.join(ROOT, "Mouse (NucMM-M)")
EM_NOTES = "EM: NucMM-Z zebrafish whole brain (ssEM), ~0.5 um (downsampled, per paper), instance nucleus labels"
XRAY_NOTES = "X-RAY (not EM): NucMM-M mouse visual cortex micro-CT, ~0.72 um isotropic, instance nucleus labels"


def count_ids(labels):
    return int(len(np.unique(labels)) - 1)


def render_mouse(out_dir):
    """Render the X-ray volumes from the folder the loader uses (nuc_mm/mouse/{train,val})."""
    volumes = []
    for split in ("train", "val"):
        for path in sorted(glob(os.path.join(ROOT, "mouse", split, "*.h5"))):
            name = os.path.splitext(os.path.basename(path))[0]
            with h5py.File(path, "r") as f:
                raw, labels = f["raw"][:], f["labels"][:]
            out_path = os.path.join(out_dir, "xray_mouse", split, f"{name}.png")
            volume_name = f"X-RAY mouse/{split}/{name}"
            save_volume_figure(raw, {"nucleus": labels}, out_path, DATASET, volume_name, notes=XRAY_NOTES)
            volumes.append({
                "name": f"xray_mouse/{split}/{name}", "modality": "X-ray micro-CT", "shape": raw.shape,
                "voxel_size_nm": (720, 720, 720), "path": path, "n_nuclei": count_ids(labels),
            })
            print(f"Rendered X-ray mouse {split}/{name}")

    path = os.path.join(MOUSE_ROOT, "image.h5")
    with h5py.File(path, "r") as f:
        raw = f["main"]
        out_path = os.path.join(out_dir, "xray_mouse", "full_volume_raw_only.png")
        notes = "X-RAY (not EM): full NucMM-M micro-CT volume, no labels released, not used by the loader"
        save_volume_figure(raw, {}, out_path, DATASET, "X-RAY mouse/image.h5", notes=notes)
        volumes.append({
            "name": "xray_mouse/image.h5 (raw only)", "modality": "X-ray micro-CT", "shape": raw.shape,
            "voxel_size_nm": (720, 720, 720), "path": path, "n_nuclei": None,
        })
    print("Rendered X-ray mouse full volume")
    return volumes


def get_label_path(image_path, split):
    name = os.path.basename(image_path).replace("img_", "seg_")
    return os.path.join(ZEBRAFISH_ROOT, "Label", split, name)


def render_zebrafish(out_dir):
    """Render the EM crops from the original download folders, since the loader's zebrafish folder is missing."""
    volumes = []
    for split in ("train", "val"):
        unlabeled = []
        for path in sorted(glob(os.path.join(ZEBRAFISH_ROOT, "Image", split, "*.h5"))):
            name = os.path.splitext(os.path.basename(path))[0]
            label_path = get_label_path(path, split)
            with h5py.File(path, "r") as f:
                raw = f["main"][:]
            volume = {
                "name": f"em_zebrafish/{split}/{name}", "modality": "EM", "shape": raw.shape,
                "voxel_size_nm": "~(480, 510, 510), per paper", "path": path,
            }
            if os.path.exists(label_path):
                with h5py.File(label_path, "r") as f:
                    labels = f["main"][:]
                out_path = os.path.join(out_dir, "em_zebrafish", split, f"{name}.png")
                volume_name = f"EM zebrafish/{split}/{name}"
                save_volume_figure(raw, {"nucleus": labels}, out_path, DATASET, volume_name, notes=EM_NOTES)
                volume.update({"label_path": label_path, "n_nuclei": count_ids(labels)})
                print(f"Rendered EM zebrafish {split}/{name}")
            else:
                unlabeled.append((name.replace("img_", ""), raw[raw.shape[0] // 2], {}))
                volume.update({"label_path": None, "n_nuclei": None})
            volumes.append(volume)
        if unlabeled:
            out_path = os.path.join(out_dir, "em_zebrafish", f"overview_{split}_without_labels.png")
            notes = "EM NucMM-Z crops whose label file is missing locally (middle slices, raw only)"
            save_image_grid(unlabeled, out_path, DATASET, f"EM zebrafish/{split} without labels", notes=notes, ncols=7)
    return volumes


def main():
    out_dir = get_output_dir(DATASET)
    volumes = render_zebrafish(out_dir) + render_mouse(out_dir)
    write_info(DATASET, {
        "classes": ["nucleus"], "modality": "EM (NucMM-Z zebrafish) and X-ray micro-CT (NucMM-M mouse)",
        "dimensionality": "3d", "label_quality": "manual/proofread instance nucleus labels",
        "n_volumes": len(volumes), "volumes": volumes, "data_path": ROOT,
        "download": (
            "Mostly local from an earlier gdown run (428 MB). Re-tried the Google Drive folder on 2026-10-15: added "
            "2 missing zebrafish train labels (seg_0064_0960_0896.h5, seg_0128_0384_0768.h5, ~40 KB); the other 47 "
            "zebrafish label files (20 train, 27 val) are refused by Drive ('too many accesses' quota)."
        ),
        "notes": f"{EM_NOTES}; {XRAY_NOTES}",
        "issues": (
            "Only the X-ray mouse part is usable through the loader: nuc_mm/mouse/{train,val} exists (4+4 volumes of "
            "192^3). The EM zebrafish part (the only EM in this dataset) is incomplete: 27 train + 27 val image crops "
            "(64^3) are local, but only 7/27 train labels and 0/27 val labels; the Drive folder lists all 54 label "
            "files but blocks their download. nuc_mm/zebrafish was never extracted, and get_nuc_mm_data(sample="
            "'zebrafish') would fail at the assert len(image_files) == len(label_files) in _extract_split "
            "(torch_em/data/datasets/electron_microscopy/nuc_mm.py:27) because of the missing labels; it also skips "
            "the download since the nuc_mm folder exists (util.download_source_gdrive returns early). The figures read "
            "the zebrafish crops directly from 'Zebrafish (NucMM-Z)/Image' and '/Label'. Voxel sizes are from the "
            "paper, not stored in the files. Mouse (NucMM-M)/image.h5 is the full unlabeled micro-CT volume."
        ),
        "observations": (
            "EM zebrafish: labels aligned and exhaustive, 70-80 small nuclei (~8-10 px wide) per 64x64 slice, labels "
            "slightly eroded vs. the nucleus edge; the raw has a horizontal seam through each crop and the ZX views "
            "look noisy, with fragmented label profiles, so z-continuity is weaker than in xy. X-ray mouse: nuclei are "
            "dark round blobs in noisy micro-CT; labels align and cover most visible nuclei (14-24 per slice), some "
            "small dark blobs (vessels / other cells) are unlabeled. X-ray is a different modality from EM."
        ),
    })


if __name__ == "__main__":
    main()
