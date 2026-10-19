import os

import numpy as np
import zarr

from denoised_view import DenoisedView
from figure_utils import DATA_ROOT, get_output_dir, save_volume_figure, write_info
from torch_em.data.datasets.electron_microscopy import npc1_mito


DATASET = "npc1_mito"
NOTES = "cryo-ET HEK293T NPC1-KO, 1.5 nm voxels, automated pseudo-label (nnInteractive), semantic; raw 5-slice mean"
OBSERVATIONS = (
    "Aligned: on a z-averaged, smoothed raw the labeled regions match mitochondria with visible cristae (checked at "
    "z=150, 209, 262, 330). Labels are smooth blobs (mcm-cryoET smoothing), boundaries follow the outer membrane "
    "roughly but not tightly. Semantic only: touching mitochondria share one component (1 id per slice). Coverage "
    "looks reasonable for the large mitos; near the lamella surfaces (z>320) only small fragments are labeled and a "
    "few small mitos in the cell center may be missed. Raw at 1.5 nm is very noisy; the figure shows a 5-slice mean "
    "with gaussian sigma 1.5 for display only."
)


def render_tomogram(raw_path, label_path, out_dir):
    raw = zarr.open_array(os.path.join(raw_path, "0"), mode="r")
    labels = zarr.open_array(os.path.join(label_path, "0"), mode="r")
    assert raw.shape == labels.shape, f"{raw.shape} vs {labels.shape}"
    name = npc1_mito.RUN
    out_path = os.path.join(out_dir, f"{name}.png")
    save_volume_figure(DenoisedView(raw), {"mito": labels}, out_path, DATASET, name, notes=NOTES)
    label_values = labels[:]
    ids, counts = np.unique(label_values, return_counts=True)
    return {
        "name": name, "shape": raw.shape, "voxel_size_nm": (1.4985,) * 3, "path": raw_path, "label_path": label_path,
        "label_values": {int(i): int(c) for i, c in zip(ids, counts)},
        "foreground_fraction": round(float((label_values > 0).mean()), 4),
    }


def main():
    data_path = os.path.join(DATA_ROOT, DATASET)
    raw_path, label_path = npc1_mito.get_npc1_mito_paths(data_path, download=True)
    out_dir = get_output_dir(DATASET)
    volume = render_tomogram(raw_path, label_path, out_dir)
    info = {
        "classes": ["mito"], "modality": "cryo-ET (CryoET Data Portal dataset 10456)", "dimensionality": "3d",
        "label_quality": "automated pseudo-label, semantic", "n_volumes": 1, "volumes": [volume],
        "data_path": data_path, "download": "tomogram (float32) + mask (int8), level 0, 1.5 GB, get_npc1_mito_paths",
        "notes": NOTES, "issues": "Single tomogram, unverified pseudo-label.", "observations": OBSERVATIONS,
    }
    write_info(DATASET, info)


if __name__ == "__main__":
    main()
