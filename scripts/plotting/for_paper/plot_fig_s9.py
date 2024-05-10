import os
from glob import glob

import h5py
import imageio.v3 as imageio
import napari

# DATA_ROOT = "./figure_data"
DATA_ROOT = "/media/anwai/ANWAI/figure_data"


def plot_lucchi():
    raw_path = os.path.join(DATA_ROOT, "fig_s9", "lucchi_test.h5")
    with h5py.File(raw_path, "r") as f:
        raw = f["raw"][:]

    seg_paths = glob(os.path.join(DATA_ROOT, "fig_s9", "lucchi*.tif"))
    segmentations = {}
    for path in seg_paths:
        seg = imageio.imread(path)
        name = os.path.basename(path)
        segmentations[name] = seg

    raw_slice = raw.copy()
    z = len(raw_slice) // 2
    raw_slice[:z] = 0
    raw_slice[(z+2):] = 0

    v = napari.Viewer()
    v.axes.visible = True
    v.add_image(raw)
    v.add_image(raw_slice)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name)
    napari.run()


def plot_plantseg_ovules():
    raw_path = os.path.join(DATA_ROOT, "fig_s9", "plantseg_ovules_test_N_435_final_crop_ds2.h5")
    with h5py.File(raw_path, "r") as f:
        raw = f["volume/cropped/raw"][:]

    seg_paths = glob(os.path.join(DATA_ROOT, "fig_s9", "plantseg_ovules*.tif"))
    segmentations = {}
    for path in seg_paths:
        seg = imageio.imread(path)
        name = os.path.basename(path)
        segmentations[name] = seg

    raw_slice = raw.copy()
    z = len(raw_slice) // 2
    raw_slice[:z] = 0
    raw_slice[(z+2):] = 0

    v = napari.Viewer()
    v.axes.visible = True
    v.add_image(raw)
    v.add_image(raw_slice)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name)
    napari.run()


def main():
    # plot_lucchi()
    plot_plantseg_ovules()


main()
