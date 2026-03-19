import os
from tqdm import tqdm
from glob import glob
from natsort import natsorted

import h5py
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


def run_default_model(view=False):
    # Evaluates the OG MicroSAM LM Generalist model.

    # Load the MicroSAM model.
    mode = "ais"
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_lm", segmentation_mode=mode, is_tiled=True,
    )

    # Iterate over each image.
    fpaths = get_mndino_paths(path=os.path.join(ROOT, "mndino_data"), split="test", download=True)
    for i, curr_fpath in tqdm(enumerate(fpaths)):
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

        # Store all stuff in an exclusive path.
        with h5py.File(f"mnseg_{i}.h5", "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", raw, compression="gzip")
            if "labels" not in f:
                f.create_dataset("labels", labels, compression="gzip")

            f.create_dataset(f"predictions/{mode}/vit_b_lm", instances, compression="gzip")


def run_finetuned_model(view=False):
    # Evaluates the finetuned models (starting either 'vit_b' or 'vit_b_lm').
    inital_model = "vit_b"

    # Load the MicroSAM model.
    mode = "ais"
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b",
        checkpoint=f"./checkpoints/microsam-mn-{inital_model}/best.pt",
        segmentation_mode=mode,
        is_tiled=True,
    )

    # Iterate over each image.
    fpaths = get_mndino_paths(path=os.path.join(ROOT, "mndino_data"), split="test")
    for i, curr_fpath in tqdm(enumerate(fpaths)):
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

        # Store the new predictions in an exclusive path.
        with h5py.File(f"mnseg_{i}.h5", "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", raw, compression="gzip")
            if "labels" not in f:
                f.create_dataset("labels", labels, compression="gzip")

            f.create_dataset(f"predictions/{mode}/finetuned_{inital_model}", instances, compression="gzip")


def evaluate_predictions():
    fpaths = natsorted(glob("mnseg_*.h5"))

    running_msa = []
    for curr_fpath in tqdm(fpaths):
        # Get the raw image and corresponding micronuclei annotations.
        f = open_file(curr_fpath, "r")
        raw = f["raw"][:]
        labels = f["labels"][:]

        breakpoint()

        # Let's filter out big nuclei and only stick to micronuclei.
        area_threshold = 300  # NOTE: As used in the experiments.
        instances = inverse_size_filter(seg=instances, max_size=area_threshold)

        # And evaluate the results.
        running_msa.append(mean_segmentation_accuracy(instances, labels))

    # Calculate the average mSA
    mean_msa = np.mean(running_msa)
    print(mean_msa)


def main():
    run_default_model(view=False)


if __name__ == "__main__":
    main()
