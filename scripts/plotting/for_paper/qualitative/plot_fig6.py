import os

import h5py
import napari

DATA_ROOT = "./figure_data"


# FIXME this is still the wrong raw data.
def plot_3d():
    path = os.path.join(DATA_ROOT, "fig_6", "3d", "data_user_study_3d.h5")
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        seg_sam = f["segmentation/finetuned"][:]
        seg_ilastik = f["segmentation/ilastik"][:].squeeze()

    v = napari.Viewer()
    v.add_image(raw)
    v.add_labels(seg_sam)
    v.add_labels(seg_ilastik)
    napari.run()


plot_3d()
