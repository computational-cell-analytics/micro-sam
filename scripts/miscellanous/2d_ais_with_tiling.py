import os
from glob import glob

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


ROOT = "/media/anwai/ANWAI/data/neurips-cell-seg/Tuning"


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


def ais_with_tiling(image_path, gt_path, view=False):
    image = imageio.imread(image_path)
    gt = imageio.imread(gt_path)

    model_type = "vit_b"
    checkpoint_path = "/home/anwai/models/micro-sam/vit_b/lm_generalist/best.pt"

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
        v.add_image(image)
        v.add_labels(prediction)
        v.add_labels(gt, visible=False)
        napari.run()

    msa, sa = mean_segmentation_accuracy(prediction, gt, return_accuracies=True)
    return msa, sa[0]


def for_neurips_tuning_set(view=False):
    image_paths = sorted(glob(os.path.join(ROOT, "images", "*")))
    gt_paths = sorted(glob(os.path.join(ROOT, "labels", "*")))

    assert len(image_paths) == len(gt_paths)

    msa_list, sa50_list = [], []
    for image_path, gt_path in zip(image_paths, gt_paths):
        msa, sa50 = ais_with_tiling(
            image_path=image_path, gt_path=gt_path, view=view
        )
        msa_list.append(msa)
        sa50_list.append(sa50)

    msa_score = np.mean(msa_list)
    sa50_score = np.mean(sa50_list)
    print(f"mSA: {msa_score}, SA50: {sa50_score}")


def main():
    for_neurips_tuning_set(view=True)


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    main()
