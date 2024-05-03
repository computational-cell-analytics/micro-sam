import os
from glob import glob
from tqdm import tqdm

import z5py
import numpy as np
import imageio.v3 as imageio

from elf.evaluation import mean_segmentation_accuracy

from micro_sam import util
from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    TiledInstanceSegmentationWithDecoder,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)

import torch


NIPS_ROOT = "/media/anwai/ANWAI/data/neurips-cell-seg/Tuning"
# NIPS_ROOT = "/scratch/projects/nim00007/sam/data/neurips-cell-seg/new/Tuning/"

MODEL_TYPE = "vit_b"

# CHECKPOINT_PATH = "/home/anwai/models/micro-sam/vit_b/lm_generalist/best.pt"
# CHECKPOINT_PATH = "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/lm_generalist_sam/best.pt"
CHECKPOINT_PATH = "/scratch/usr/nimanwai/micro-sam/checkpoints/vit_b/tissuenet_sam/best.pt"


def get_model_for_ais(
    image, model_type, checkpoint_path, tile_shape=(512, 512), halo=(128, 128),
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor, decoder = get_predictor_and_decoder(
        model_type=model_type, checkpoint_path=checkpoint_path, device=device
    )

    tiling_kwargs = {}
    do_tiling = False
    if image.shape > tile_shape:
        tiling_kwargs["tile_shape"] = tile_shape
        tiling_kwargs["halo"] = halo
        do_tiling = True

    image_embeddings = util.precompute_image_embeddings(
        predictor=predictor, input_=image, ndim=2, **tiling_kwargs
    )

    return predictor, decoder, image_embeddings, do_tiling


def ais_with_tiling(image, gt, view=False):
    if isinstance(image, str):
        image = imageio.imread(image)

    if isinstance(gt, str):
        gt = imageio.imread(gt)

    model_type = MODEL_TYPE
    checkpoint_path = CHECKPOINT_PATH

    predictor, decoder, image_embeddings, do_tiling = get_model_for_ais(
        image=image, model_type=model_type, checkpoint_path=checkpoint_path,
    )

    if do_tiling:
        ais = TiledInstanceSegmentationWithDecoder(predictor, decoder)
    else:
        ais = InstanceSegmentationWithDecoder(predictor, decoder)

    ais.initialize(image, image_embeddings=image_embeddings, verbose=True)

    prediction = ais.generate(
        center_distance_threshold=0.3,
        boundary_distance_threshold=0.3,
        distance_smoothing=1.6,
        foreground_smoothing=3,
        min_size=200 if do_tiling else 100
    )
    if len(prediction) == 0:
        prediction = np.zeros_like(gt, dtype="uint8")
    else:
        prediction = mask_data_to_segmentation(prediction, with_background=True)

    if view:
        import napari
        v = napari.Viewer()
        v.add_image(image if image.ndim == 2 else image.transpose(2, 0, 1))  # making channels first
        v.add_labels(prediction)
        v.add_labels(gt, visible=False)
        napari.run()

    msa, sa = mean_segmentation_accuracy(prediction, gt, return_accuracies=True)
    return msa, sa[0]


def for_neurips_tuning_set(view=False):
    image_paths = sorted(glob(os.path.join(NIPS_ROOT, "images", "*")))
    gt_paths = sorted(glob(os.path.join(NIPS_ROOT, "labels", "*")))

    assert len(image_paths) == len(gt_paths)

    msa_list, sa50_list = [], []
    for image_path, gt_path in tqdm(zip(image_paths, gt_paths), total=len(image_paths)):
        msa, sa50 = ais_with_tiling(
            image=image_path, gt=gt_path, view=view
        )
        msa_list.append(msa)
        sa50_list.append(sa50)

    print(f"mSA: {np.mean(msa_list)}, SA50: {np.mean(sa50_list)}")


def for_tissuenet_test_set(data_dir, view=False):
    all_sample_paths = sorted(glob(os.path.join(data_dir, "*.zarr")))

    msa1_list, sa501_list = [], []
    msa2_list, sa502_list = [], []
    msa3_list, sa503_list = [], []
    msa4_list, sa504_list = [], []
    for sample_path in tqdm(all_sample_paths):
        with z5py.File(sample_path, "r") as f:
            raw = f["raw/rgb"][:]
            labels = f["labels/cell"][:]

            # OPTION 1: use the tissuenet inputs as it is (0: zeros, 1: nuclei, 2: cells)
            msa1, sa501 = ais_with_tiling(image=raw.transpose(1, 2, 0), gt=labels, view=view)

            # OPTION 2: use mono-channel image (mean over all channels)
            msa2, sa502 = ais_with_tiling(image=raw.mean(axis=0), gt=labels, view=view)

            # OPTION 3: use mono-channel image (mean over only valid channels, i.e. 1 and 2)
            msa3, sa503 = ais_with_tiling(image=raw[1:].mean(axis=0), gt=labels, view=view)

            # OPTION 4: use 3 channel inputs, but the first chan is replaced by the mean over the other 2 channels
            msa4, sa504 = ais_with_tiling(
                image=np.concatenate([np.mean(raw[1:], axis=0)[None], raw[1:]], axis=0).transpose(1, 2, 0),
                gt=labels, view=view
            )

            msa1_list.append(msa1)
            sa501_list.append(sa501)

            msa2_list.append(msa2)
            sa502_list.append(sa502)

            msa3_list.append(msa3)
            sa503_list.append(sa503)

            msa4_list.append(msa4)
            sa504_list.append(sa504)

            # mSA 1: 0.12127891062264558, SA50 1: 0.28421753449100295
            # mSA 2: 0.09484694513022068, SA50 2: 0.22151359989916541
            # mSA 3: 0.09484766414686455, SA50 3: 0.22152046443548926
            # mSA 4: 0.12875632076886206, SA50 4: 0.2897684912727994

    print(f"mSA 1: {np.mean(msa1_list)}, SA50 1: {np.mean(sa501_list)}")
    print(f"mSA 2: {np.mean(msa2_list)}, SA50 2: {np.mean(sa502_list)}")
    print(f"mSA 3: {np.mean(msa3_list)}, SA50 3: {np.mean(sa503_list)}")
    print(f"mSA 4: {np.mean(msa4_list)}, SA50 4: {np.mean(sa504_list)}")


def main():
    # for_neurips_tuning_set(view=True)

    # tissuenet_dir = "/media/anwai/ANWAI/data/tissuenet/test/"
    tissuenet_dir = "/scratch/projects/nim00007/sam/data/tissuenet/test"
    for_tissuenet_test_set(tissuenet_dir, view=False)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    main()
