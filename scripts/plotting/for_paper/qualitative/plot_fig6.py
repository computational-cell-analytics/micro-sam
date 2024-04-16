import os

import h5py
import napari

DATA_ROOT = "./figure_data"


def plot_3d():
    path = os.path.join(DATA_ROOT, "fig_6", "3d", "data_user_study_3d.h5")
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]
        seg_ilastik = f["segmentation/ilastik"][:]
        seg_default = f["segmentation/sam_default"][:]
        seg_finetuned = f["segmentation/sam_finetuned"][:]

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg_ilastik)
    v.add_labels(seg_default)
    v.add_labels(seg_finetuned)
    v.add_labels(labels)
    napari.run()


plot_3d()
