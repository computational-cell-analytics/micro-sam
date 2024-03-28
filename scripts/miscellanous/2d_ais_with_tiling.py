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


NIPS_ROOT = "/media/anwai/ANWAI/data/neurips-cell-seg/Tuning"
# NIPS_ROOT = "/scratch/projects/nim00007/sam/data/neurips-cell-seg/new/Tuning/"

MODEL_TYPE = "vit_b"
CHECKPOINT_PATH = "/home/anwai/models/micro-sam/vit_b/lm_generalist/best.pt"


def get_model_for_ais(
    image, model_type, checkpoint_path, tile_shape=(512, 512), halo=(128, 128), device="cpu"
):
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

    msa_list, sa50_list = [], []
    for sample_path in tqdm(all_sample_paths):
        with z5py.File(sample_path, "r") as f:
            raw = f["raw/rgb"][:]
            labels = f["labels/cell"][:]

            msa, sa50 = ais_with_tiling(
                image=raw.transpose(1, 2, 0), gt=labels, view=view
            )
            msa_list.append(msa)
            sa50_list.append(sa50)

    print(f"mSA: {np.mean(msa_list)}, SA50: {np.mean(sa50_list)}")


def main():
    # for_neurips_tuning_set(view=True)
    for_tissuenet_test_set("/media/anwai/ANWAI/data/tissuenet/test/", view=False)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    main()
