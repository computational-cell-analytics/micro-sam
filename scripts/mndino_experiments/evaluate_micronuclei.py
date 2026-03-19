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


def run_default_model(mode):
    # Evaluates the OG MicroSAM LM Generalist model.

    # Load the MicroSAM model.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b_lm", segmentation_mode=mode, is_tiled=True,
    )

    # Iterate over each image.
    fpaths = get_mndino_paths(path=os.path.join(ROOT, "mndino_data"), split="test", download=True)
    for i, curr_fpath in tqdm(enumerate(fpaths), total=len(fpaths)):
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
                f.create_dataset("raw", data=raw, compression="gzip")
            if "labels" not in f:
                f.create_dataset("labels", data=labels, compression="gzip")

            f.create_dataset(f"predictions/{mode}/default_vit_b_lm", data=instances, compression="gzip")


def run_finetuned_model(initial_model, mode):
    # Evaluates the finetuned models (starting either 'vit_b' or 'vit_b_lm').

    # Load the MicroSAM model.
    predictor, segmenter = get_predictor_and_segmenter(
        model_type="vit_b",
        checkpoint=f"./checkpoints/microsam-mn-{initial_model}/best.pt",
        segmentation_mode=mode,
        is_tiled=True,
    )

    # Iterate over each image.
    fpaths = get_mndino_paths(path=os.path.join(ROOT, "mndino_data"), split="test")
    for i, curr_fpath in tqdm(enumerate(fpaths), total=len(fpaths)):
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
                f.create_dataset("raw", data=raw, compression="gzip")
            if "labels" not in f:
                f.create_dataset("labels", data=labels, compression="gzip")

            f.create_dataset(f"predictions/{mode}/finetuned_{initial_model}", data=instances, compression="gzip")


def evaluate_predictions(view=False):
    fpaths = natsorted(glob("mnseg_*.h5"))

    running_msa_default, running_msa_finetuned1, running_msa_finetuned2 = [], [], []
    for curr_fpath in tqdm(fpaths):
        # Get the raw image and corresponding micronuclei annotations.
        f = open_file(curr_fpath, "r")
        raw = f["raw"][:]
        labels = f["labels"][:]
        pred_default = f["predictions/ais/default_vit_b_lm"][:]
        pred_finetuned1 = f["predictions/ais/finetuned_vit_b"][:]
        pred_finetuned2 = f["predictions/ais/finetuned_vit_b_lm"][:]

        # Let's filter out big nuclei and only stick to micronuclei (but do this only in the default model).
        area_threshold = 300  # NOTE: As used in the experiments.

        # And evaluate the results.
        running_msa_default.append(
            mean_segmentation_accuracy(inverse_size_filter(seg=pred_default, max_size=area_threshold), labels)
        )
        running_msa_finetuned1.append(mean_segmentation_accuracy(pred_finetuned1, labels))
        running_msa_finetuned2.append(mean_segmentation_accuracy(pred_finetuned2, labels))

        if view:
            import napari
            v = napari.Viewer()
            v.add_image(raw)
            v.add_labels(labels)
            v.add_labels(pred_default, "predictions/default")
            v.add_labels(pred_finetuned1, "predictions/finetuned_vit_b")
            v.add_labels(pred_finetuned2, "predictions/finetuned_vit_b_lm")

    # Calculate the average mSA per setup.
    msa_default = np.mean(running_msa_default)
    msa_finetuned1 = np.mean(running_msa_finetuned1)
    msa_finetuned2 = np.mean(running_msa_finetuned2)
    print(msa_default, msa_finetuned1, msa_finetuned2)


def main():
    # Run the default models
    # run_default_model("ais")

    # Run the finetuned models
    # run_finetuned_model("vit_b", "ais")
    # run_finetuned_model("vit_b_lm", "ais")

    # Evaluate predictions
    evaluate_predictions(view=True)


if __name__ == "__main__":
    main()
