import os
from glob import glob

import imageio.v3 as imageio
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


# TODO
def plot_tracking():
    pass


def main():
    # create_data_2d_default()
    # create_data_2d_finetuned()

    # plot_3d()
    plot_2d()


main()
