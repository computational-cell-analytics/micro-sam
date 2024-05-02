import os
from glob import glob
from pathlib import Path

import h5py
import napari
import numpy as np
import pandas as pd
import imageio.v3 as imageio
from skimage.measure import regionprops_table

from deepcell_tracking.utils import load_trks


# DATA_ROOT = "./figure_data"
DATA_ROOT = "/media/anwai/ANWAI/figure_data"


def plot_3d():
    path = os.path.join(DATA_ROOT, "fig_6", "3d", "data_user_study_3d.h5")
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]
        seg_ilastik = f["segmentation/ilastik"][:]
        seg_default = f["segmentation/sam_default"][:]
        seg_finetuned = f["segmentation/sam_finetuned"][:]

    raw_slice = raw.copy()
    z = len(raw_slice) // 2
    raw_slice[:z] = 0
    raw_slice[(z+2):] = 0

    v = napari.Viewer()
    v.axes.visible = True
    v.add_image(raw)
    v.add_image(raw_slice)
    v.add_labels(seg_ilastik)
    v.add_labels(seg_default)
    v.add_labels(seg_finetuned)
    v.add_labels(labels)
    napari.run()

    # Exporting the label stack to try out stuff
    # imageio.imwrite(os.path.join(DATA_ROOT, "3d_user_study_seg_finetuned.tif"), seg_finetuned)


def create_data_2d_default():
    from micro_sam.util import get_sam_model
    from micro_sam.instance_segmentation import get_amg, mask_data_to_segmentation

    model = get_sam_model(model_type="vit_b")
    segmenter = get_amg(model, is_tiled=False, decoder=None)

    images = glob("./organoids/images/*.tif")
    assert len(images) > 0
    out_folder = "./organoids/seg_default"
    os.makedirs(out_folder, exist_ok=True)

    for im in images:
        print("Segmenting", im, "...")
        image = imageio.imread(im)
        segmenter.initialize(image, verbose=True)
        seg = segmenter.generate()
        seg = mask_data_to_segmentation(seg, with_background=True)
        imageio.imwrite(os.path.join(out_folder, os.path.basename(im)), seg, compression="zlib")


def create_data_2d_finetuned():
    from micro_sam.instance_segmentation import get_predictor_and_decoder, get_amg, mask_data_to_segmentation

    model, decoder = get_predictor_and_decoder("vit_b", "./organoids/vit_b_organoids.pt")
    segmenter = get_amg(model, is_tiled=False, decoder=decoder)

    images = glob("./organoids/images/*.tif")
    assert len(images) > 0
    out_folder = "./organoids/seg_finetuned"
    os.makedirs(out_folder, exist_ok=True)

    for im in images:
        print("Segmenting", im, "...")
        image = imageio.imread(im)
        segmenter.initialize(image, verbose=True)
        seg = segmenter.generate(image)
        seg = mask_data_to_segmentation(seg, with_background=True)
        imageio.imwrite(os.path.join(out_folder, os.path.basename(im)), seg, compression="zlib")


def plot_2d():
    image_root = os.path.join(DATA_ROOT, "fig_6", "2d", "images")
    seg_finetuned_root = os.path.join(DATA_ROOT, "fig_6", "2d", "seg_finetuned")
    seg_default_root = os.path.join(DATA_ROOT, "fig_6", "2d", "seg_default")

    images = glob(os.path.join(image_root, "*.tif"))

    for im in images:
        image = imageio.imread(im)

        default_path = os.path.join(seg_default_root, os.path.basename(im))
        seg_default = imageio.imread(default_path)

        finetuned_path = os.path.join(seg_finetuned_root, os.path.basename(im))
        seg_finetuned = imageio.imread(finetuned_path)

        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(seg_finetuned)
        v.add_labels(seg_default)
        napari.run()

#
# FOR TRACKING
#


def load_tracking_segmentation(experiment):
    ROOT = r"/home/anwai/results/tracking/MicroSAM testing/"
    if experiment == "vit_l":
        seg_path = glob(os.path.join(ROOT, r"round 2 vit_l", "*.tif"))[0]
    elif experiment == "vit_l_lm":
        seg_path = glob(os.path.join(ROOT, "vit_l_finetuned", "*.tif"))[0]
    elif experiment == "vit_l_specialist":
        seg_path = glob(os.path.join(ROOT, "vit_l_specialist", "*.tif"))[0]
    else:
        print(experiment)
        raise ValueError

    return imageio.imread(seg_path)


def rpros_over_time(labels):
    data_df = []
    for idx in range(labels.shape[0]):
        props = regionprops_table(labels[idx], properties=('label', 'centroid'))
        props['frame'] = np.full(props['label'].shape, idx)
        data_df.append(pd.DataFrame(props))

    data_df = pd.concat(data_df).reset_index(drop=True)
    data_df = data_df.sort_values(['label', 'frame'], ignore_index=True)
    data = data_df.loc[
        :, ['label', 'frame', 'centroid-0', 'centroid-1']
    ].to_numpy()

    return data


def check_tracking_results(raw, labels, curr_lineages, chosen_frames, save=False):
    """
    Total number of objects (times reported per track):
        - true number of objects: 107
        - default vit_l: 102 (0.637 min per track) (38 sec per track)
        - generalist vit_l: 105 (0.4 min per track) (24 sec per track)
        - finetuned vit_l: 104 (0.384 min per track) (23 sec per track)
        - trackmate (stardist): 0.318 min per track; (19 sec per track)
    """
    # path = "/media/anwai/ANWAI/data/for_tracking/DynamicNuclearNet_test_b007.h5"
    # with h5py.File(path, "r") as f:
    #     raw = f["raw"][:]

    # # take every 3rd frame
    # frames = list(range(0, 71, 3))
    # raw = np.stack([raw[frame] for frame in frames])

    seg_default = load_tracking_segmentation("vit_l")
    seg_generalist = load_tracking_segmentation("vit_l_lm")
    seg_specialist = load_tracking_segmentation("vit_l_specialist")

    # let's get the tracks only for the objects present per frame
    for idx in np.unique(labels)[1:]:
        lineage = curr_lineages[idx]
        lineage["frames"] = [frame for frame in lineage["frames"] if frame in chosen_frames]

    # get middle slice
    raw_slice = raw.copy()
    z = len(raw_slice) // 2
    raw_slice[:z] = 0
    raw_slice[(z+2):] = 0

    v = napari.Viewer()
    v.add_image(raw)
    v.add_image(raw_slice)
    # v.add_labels(labels)

    v.axes.visible = True
    v.add_tracks(rpros_over_time(seg_specialist), name="tracklets")

    # v.add_labels(seg_default, visible=False)
    # v.add_labels(seg_generalist, visible=False)
    v.add_labels(seg_specialist, visible=False)

    napari.run()

    if save:
        # let's save the volume
        with h5py.File(
            "/media/anwai/ANWAI/results/micro-sam/tracking/DynamicNuclearNet_results_b007.h5", "a"
        ) as f:
            f.create_dataset("raw", data=raw)
            f.create_dataset("segmentation/default", data=seg_default)
            f.create_dataset("segmentation/generalist", data=seg_generalist)
            f.create_dataset("segmentation/specialist", data=seg_specialist)


def get_tracking_data(view=False, save=False):
    data_dir = "/home/anwai/data/dynamicnuclearnet/DynamicNuclearNet-tracking-v1_0/"
    data_source = np.load(os.path.join(data_dir, "data-source.npz"), allow_pickle=True)

    fname = "test.trks"
    track_file = os.path.join(data_dir, fname)
    split_name = Path(track_file).stem

    data = load_trks(track_file)

    X = data["X"]
    y = data["y"]
    lineages = data["lineages"]

    meta = pd.DataFrame(
        data_source[split_name],
        columns=["filename", "experiment", "pixel_size", "screening_passed", "time_step", "specimen"]
    )
    print(meta)

    # let's convert the data to expected shape
    X = X.squeeze(-1)
    y = y.squeeze(-1)

    # NOTE: chosen slice for the tracking user study.
    _slice = 7
    raw, labels = X[_slice, ...], y[_slice, ...]
    curr_lineages = lineages[_slice]

    # NOTE:
    # let's get every third frame of data and see how it looks
    chosen_frames = list(range(0, raw.shape[0], 3))
    raw = np.stack([raw[frame] for frame in chosen_frames])
    labels = np.stack([labels[frame] for frame in chosen_frames])

    return raw, labels, curr_lineages, chosen_frames


def plot_tracking():
    raw, labels, curr_lineages, chosen_frames = get_tracking_data()
    check_tracking_results(raw, labels, curr_lineages, chosen_frames)


def main():
    # create_data_2d_default()
    # create_data_2d_finetuned()

    plot_3d()
    # plot_2d()
    # plot_tracking()


main()
