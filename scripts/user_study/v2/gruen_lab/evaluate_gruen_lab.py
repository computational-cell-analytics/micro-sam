"""Evaluate the registry 'hvit_l_cells' and the Gruen lab finetuned checkpoints on all three crops.

Every model runs the same setup on every crop, the two training crops and the validation crop:
- Automatic segmentation with AIS sparse (flow), AIS dense (multicut) and APG, in 2d (slice by slice) and in 3d.
- Interactive segmentation with iterative prompting, starting from a box or a point, in 2d and in 3d.
All segmentations use the default parameters of 'hvit_l_cells', nothing is tuned. The scores are the mean
segmentation accuracy (mSA) and the segmentation accuracy at IoU 0.5 (SA50), averaged over the slices in 2d.
"""

import os
import argparse

import h5py
import tifffile
import numpy as np
import pandas as pd
import imageio.v3 as imageio

import torch

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.v2.normalization import normalize_raw
from micro_sam.v2.models.util import joint_unetr_state
from micro_sam.v2.automatic_segmentation import automatic_instance_segmentation
from micro_sam.v2.instance_segmentation import get_decoder, get_instance_segmentation_generator
from micro_sam.v2.util import get_sam2_model, get_sam2_image_predictor, _download_finetuned_sam2_model
from micro_sam.v2.evaluation.inference import (
    get_prediction_dir_2d, get_prediction_paths_3d, run_interactive_segmentation_2d, run_interactive_segmentation_3d,
)

from train_gruen_lab import MODEL_TYPE, prepare_data, split_crops


BACKBONE = MODEL_TYPE[:6]
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", f"gruen_lab_{MODEL_TYPE}")
MODELS = {
    "registry": None,
    "finetuned_best": os.path.join(CHECKPOINT_DIR, "best.pt"),
    "finetuned_latest": os.path.join(CHECKPOINT_DIR, "latest.pt"),
}
# The automatic methods: (engine, AIS postprocessing mode).
AUTOMATIC_METHODS = {"ais_sparse": ("ais", "sparse"), "ais_dense": ("ais", "dense"), "apg": ("apg", None)}
PROMPTS = ("box", "point")
N_ITERATIONS = 8


def load_crops(data_dir):
    """Load the raw (Z, Y, X, C) uint16 crops and their labels, with the split of every crop."""
    train_paths, val_paths = split_crops(prepare_data(data_dir))
    crops = []
    for path in sorted(train_paths + val_paths):
        with h5py.File(path, "r") as f:
            raw = np.moveaxis(f["raw"][:], 0, -1)  # (C, Z, Y, X) -> (Z, Y, X, C)
            labels = f["labels"][:]
        name = os.path.splitext(os.path.basename(path))[0]
        crops.append((name, "val" if path in val_paths else "train", raw, labels))
    return crops


def get_weights(model_name, weight_dir):
    """Return the interactive (SAM2) and the decoder weight file of a model.

    The joint checkpoints bundle both halves with the trainer state, so they are split into two files, as
    `finetuning/v2/evaluation/common.export_joint_checkpoint` does. The checkpoint's iteration names the export,
    so a newer 'best' or 'latest' checkpoint is exported again.
    """
    if MODELS[model_name] is None:
        interactive_path, _, decoder_path = _download_finetuned_sam2_model(MODEL_TYPE)
        return interactive_path, decoder_path, None

    state = torch.load(MODELS[model_name], map_location="cpu", weights_only=False)
    iteration = state["iteration"]
    interactive_path = os.path.join(weight_dir, f"{model_name}_it{iteration}.pt")
    decoder_path = os.path.join(weight_dir, f"{model_name}_it{iteration}_decoder.pt")
    if not (os.path.exists(interactive_path) and os.path.exists(decoder_path)):
        os.makedirs(weight_dir, exist_ok=True)
        torch.save({"model": state["model_state"], "model_type": BACKBONE}, interactive_path)
        torch.save(joint_unetr_state(state), decoder_path)
    return interactive_path, decoder_path, iteration


def get_segmenter(interactive_path, decoder_path, engine, ndim, device):
    """Build the predictor and the segmenter of an engine, as `get_predictor_and_segmenter` does for a registry model.

    Volumetric APG propagates its prompts through the volume, so it needs the video predictor. The decoder is named
    after the registry model, so every model uses the default postprocessing parameters of 'hvit_l_cells'.
    """
    use_video = engine == "apg" and ndim == 3
    model = get_sam2_model(
        model_type=BACKBONE, input_type="videos" if use_video else "images", device=device,
        checkpoint_path=interactive_path,
    )
    predictor = model if use_video else get_sam2_image_predictor(model)
    decoder = get_decoder(MODEL_TYPE, checkpoint=decoder_path, device=device, encoder=model.image_encoder)
    segmenter = get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode=engine, device=device, ndim=ndim,
    )
    return predictor, segmenter


def score(segmentation, labels):
    """Return mSA and SA50 of one image or volume."""
    msa, accuracies = mean_segmentation_accuracy(segmentation, labels, return_accuracies=True)
    return msa, accuracies[0]


def score_slices(segmentations, labels):
    """Return mSA and SA50 averaged over the slices that hold objects."""
    scores = [score(seg, gt) for seg, gt in zip(segmentations, labels) if gt.max() > 0]
    return tuple(float(np.mean(values)) for values in zip(*scores))


def run_automatic(crops, interactive_path, decoder_path, device, output_dir):
    """Run every automatic method in 2d and 3d on every crop and return one result row per run."""
    rows = []
    for method, (engine, mode) in AUTOMATIC_METHODS.items():
        for ndim in (2, 3):
            predictor, segmenter = get_segmenter(interactive_path, decoder_path, engine, ndim, device)
            kwargs = {} if mode is None else {"mode": mode}
            for name, split, raw, labels in crops:
                if ndim == 2:
                    segmentation = np.stack([
                        automatic_instance_segmentation(
                            predictor, segmenter, input_path=frame, ndim=2, verbose=False, **kwargs
                        ) for frame in raw
                    ])
                    msa, sa50 = score_slices(segmentation, labels)
                else:
                    segmentation = automatic_instance_segmentation(
                        predictor, segmenter, input_path=raw, ndim=3, verbose=False, **kwargs
                    )
                    msa, sa50 = score(segmentation, labels)
                tifffile.imwrite(
                    os.path.join(output_dir, f"{name}_{method}_{ndim}d.tif"), segmentation.astype("uint32"),
                    compression="zlib",
                )
                n_objects = len(np.unique(segmentation)) - 1
                print(f"{name} ({split}) {method} {ndim}d: mSA {msa:.3f}, SA50 {sa50:.3f}, {n_objects} objects")
                rows.append({
                    "crop": name, "split": split, "ndim": ndim, "method": method, "iteration": None,
                    "msa": msa, "sa50": sa50, "n_objects": n_objects,
                })
            del predictor, segmenter
            torch.cuda.empty_cache()
    return rows


def to_uint8(raw, axis):
    """Normalize every channel by its 2nd and 98th percentile over 'axis' to uint8, as the SAM2 inputs expect."""
    return normalize_raw(raw, axis=axis, output_dtype="uint8")


def run_interactive_2d(crops, interactive_path, device, output_dir):
    """Prompt every slice of every crop iteratively and return one result row per crop, prompt and iteration."""
    rows = []
    for name, split, raw, labels in crops:
        input_dir = os.path.join(output_dir, "inputs", name)
        os.makedirs(input_dir, exist_ok=True)
        image_paths, gt_paths = [], []
        for z, (frame, gt) in enumerate(zip(raw, labels)):
            if gt.max() == 0:
                continue
            image_paths.append(os.path.join(input_dir, f"z{z:02d}_image.tif"))
            gt_paths.append(os.path.join(input_dir, f"z{z:02d}_labels.tif"))
            imageio.imwrite(image_paths[-1], to_uint8(frame, axis=(0, 1)))
            imageio.imwrite(gt_paths[-1], gt.astype("uint32"))

        for prompt in PROMPTS:
            prediction_root = os.path.join(output_dir, "interactive_2d", name)
            run_interactive_segmentation_2d(
                image_paths=image_paths, gt_paths=gt_paths, image_key=None, gt_key=None,
                prediction_dir=prediction_root, model_type=BACKBONE, checkpoint_path=interactive_path,
                start_with_box_prompt=prompt == "box", device=device, n_iterations=N_ITERATIONS, use_masks=True,
                ensure_8bit=False,
            )
            prediction_dir = get_prediction_dir_2d(prediction_root, prompt == "box", use_masks=True)
            for iteration in range(N_ITERATIONS):
                predictions = [
                    imageio.imread(os.path.join(prediction_dir, f"iteration{iteration:02d}", os.path.basename(path)))
                    for path in image_paths
                ]
                msa, sa50 = score_slices(predictions, [imageio.imread(path) for path in gt_paths])
                rows.append({
                    "crop": name, "split": split, "ndim": 2, "method": f"interactive_{prompt}",
                    "iteration": iteration, "msa": msa, "sa50": sa50, "n_objects": None,
                })
            print(f"{name} ({split}) interactive {prompt} 2d: mSA {rows[-N_ITERATIONS]['msa']:.3f} -> {msa:.3f}")
    return rows


def run_interactive_3d(crops, interactive_path, device, output_dir):
    """Prompt every crop volume iteratively and return one result row per crop, prompt and iteration."""
    rows = []
    for name, split, raw, labels in crops:
        for prompt in PROMPTS:
            prediction_root = os.path.join(output_dir, "interactive_3d", name)
            run_interactive_segmentation_3d(
                raw=to_uint8(raw, axis=(0, 1, 2)), labels=labels, model_type=BACKBONE,
                checkpoint_path=interactive_path, start_with_box_prompt=prompt == "box",
                prediction_dir=prediction_root, prediction_fname=name, device=device, n_iterations=N_ITERATIONS,
            )
            prediction_paths = get_prediction_paths_3d(prediction_root, name, prompt == "box", N_ITERATIONS)[1]
            for iteration, path in enumerate(prediction_paths):
                msa, sa50 = score(imageio.imread(path), labels)
                rows.append({
                    "crop": name, "split": split, "ndim": 3, "method": f"interactive_{prompt}",
                    "iteration": iteration, "msa": msa, "sa50": sa50, "n_objects": None,
                })
            print(f"{name} ({split}) interactive {prompt} 3d: mSA {rows[-N_ITERATIONS]['msa']:.3f} -> {msa:.3f}")
    return rows


def evaluate_model(model_name, crops, device, output_root):
    """Evaluate one model and write its results to '<output_root>/<model_name>_results.csv'."""
    output_dir = os.path.join(output_root, model_name)
    os.makedirs(output_dir, exist_ok=True)
    interactive_path, decoder_path, iteration = get_weights(model_name, os.path.join(output_root, "weights"))
    print(f"Evaluating '{model_name}' (iteration {iteration}): {interactive_path}, {decoder_path}")

    rows = run_automatic(crops, interactive_path, decoder_path, device, output_dir)
    rows += run_interactive_2d(crops, interactive_path, device, output_dir)
    rows += run_interactive_3d(crops, interactive_path, device, output_dir)

    results = pd.DataFrame(rows)
    results.insert(0, "model", model_name)
    results.insert(1, "checkpoint_iteration", iteration)
    results.to_csv(os.path.join(output_root, f"{model_name}_results.csv"), index=False)
    return results


def summarize(results):
    """Print mSA per model for the automatic methods and the first and last interactive iteration."""
    automatic = results[results["iteration"].isna()]
    interactive = results[results["iteration"].isin([0, N_ITERATIONS - 1])].copy()
    interactive["method"] = interactive["method"] + "_it" + interactive["iteration"].astype(int).astype(str)
    table = pd.concat([automatic, interactive]).pivot_table(
        index=["ndim", "method"], columns=["split", "crop", "model"], values="msa",
    )
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        print(table)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", default=os.path.join(os.getcwd(), "data"), help="Folder of the h5 crops.")
    parser.add_argument("--output_root", default=os.path.join(os.getcwd(), "evaluation"), help="Results folder.")
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    crops = load_crops(args.data_dir)
    results = pd.concat([evaluate_model(model, crops, device, args.output_root) for model in args.models])
    results.to_csv(os.path.join(args.output_root, "results.csv"), index=False)
    summarize(results).to_csv(os.path.join(args.output_root, "summary_msa.csv"))


if __name__ == "__main__":
    main()
