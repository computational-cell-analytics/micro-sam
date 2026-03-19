import os
from tqdm import tqdm

import numpy as np

from elf.io import open_file
from elf.evaluation import mean_segmentation_accuracy

from torch_em.data.datasets.light_microscopy.mndino import get_mndino_paths

from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def inverse_size_filter(seg: np.ndarray, max_size: int):
    ids, counts = np.unique(seg, return_counts=True)
    out = seg.copy()

    for obj_id, count in zip(ids, counts):
        if obj_id == 0:
            continue  # i.e. skip background
        if count > max_size:
            out[out == obj_id] = 0

    return out


def evaluate_default_model(view=False):
    # Evaluates the OG MicroSAM LM Generalist model.

    # Load the MicroSAM model.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_lm", segmentation_mode="ais", is_tiled=True,
    )

    # Iterate over each image.
    fpaths = get_mndino_paths(path=os.path.join(ROOT, "mndino_data"), split="test")

    running_msa = []
    for curr_fpath in tqdm(fpaths):

        # Get the raw image and corresponding micronuclei annotations.
        f = open_file(curr_fpath, "r")
        raw = f["raw"][:]
        labels = f["labels/micronuclei"][:]

        # If there's no micronuclei in GT, we shouldn't quantify it.
        if len(np.unique(labels)) == 1:
            continue

        # Run the automatic instance segmentation pipeline.
        instances = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=raw,
            ndim=2,
            tile_shape=(384, 384),
            halo=(64, 64),
            verbose=False,
        )

        # Let's filter out big nuclei and only stick to micronuclei.
        area_threshold = 300  # NOTE: As used in the experiments.
        instances = inverse_size_filter(seg=instances, max_size=area_threshold)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(labels)
            v.add_labels(instances)
            napari.run()

        # And evaluate the results.
        running_msa.append(mean_segmentation_accuracy(instances, labels))

    # Calculate the average mSA
    mean_msa = np.mean(running_msa)
    print(mean_msa)


def evaluate_finetuned_model(view=False):
    # Evaluates the finetuned models (starting either 'vit_b' or 'vit_b_lm').

    # Load the MicroSAM model.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b",
        checkpoint="./checkpoints/microsam-mn-vit_b/best.pt"
        segmentation_mode="ais",
        is_tiled=True,
    )

    # Iterate over each image.
    fpaths = get_mndino_paths(path=os.path.join(ROOT, "mndino_data"), split="test")

    running_msa = []
    for curr_fpath in tqdm(fpaths):

        # Get the raw image and corresponding micronuclei annotations.
        f = open_file(curr_fpath, "r")
        raw = f["raw"][:]
        labels = f["labels/micronuclei"][:]

        # If there's no micronuclei in GT, we shouldn't quantify it.
        if len(np.unique(labels)) == 1:
            continue

        # Run the automatic instance segmentation pipeline.
        instances = automatic_instance_segmentation(
            predictor=predictor,
            segmenter=segmenter,
            input_path=raw,
            ndim=2,
            tile_shape=(384, 384),
            halo=(64, 64),
            verbose=False,
        )

        # Let's filter out big nuclei and only stick to micronuclei.
        area_threshold = 300  # NOTE: As used in the experiments.
        instances = inverse_size_filter(seg=instances, max_size=area_threshold)

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(labels)
            v.add_labels(instances)
            napari.run()

        # And evaluate the results.
        running_msa.append(mean_segmentation_accuracy(instances, labels))

    # Calculate the average mSA
    mean_msa = np.mean(running_msa)
    print(mean_msa)


def main():
    evaluate_default_model()


if __name__ == "__main__":
    main()
