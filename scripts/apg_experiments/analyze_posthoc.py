import argparse
import json
import os
from glob import glob
from inspect import signature
from pathlib import Path

import h5py
import imageio.v3 as imageio
import pandas as pd
import napari
import numpy as np
from tqdm import tqdm

DEFAULT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam/apg_cc"


def _normalize_and_pad(image):
    if image.ndim == 3:  # RGB -> normalize per channel.
        assert image.shape[-1] == 3  # ensure channel last.
        image = image.astype("float32")
        image -= image.min(axis=(0, 1))
        image /= (image.max(axis=(0, 1)) + 1e-7)
        image *= 255
        image = image.astype("uint8")

    min_shape = (512, 512)
    pad_width = [max(0, ms - sh) for sh, ms in zip(image.shape[:2], min_shape)]
    if any(pw > 0 for pw in pad_width):
        pad_width = [(0, pad_width[0]), (0, pad_width[1])]
        if image.ndim == 3:
            pad_width += [(0, 0)]
        image = np.pad(image, pad_width)
    # breakpoint()

    return image


def _run_prediction(image_path, out_path, predictor, segmenter, model_type, settings):
    from micro_sam.instance_segmentation import _derive_point_prompts

    if predictor is None:
        from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder
        predictor, decoder = get_predictor_and_decoder(model_type=model_type)
        segmenter = AutomaticPromptGenerator(predictor, decoder)

    image = imageio.imread(image_path)
    image = _normalize_and_pad(image)
    segmenter.initialize(image)

    # Derive prompts.
    prompt_kwargs = {k: v for k, v in settings.items() if k in signature(_derive_point_prompts).parameters}
    prompts = _derive_point_prompts(
        segmenter._foreground,
        segmenter._center_distances,
        segmenter._boundary_distances,
        **prompt_kwargs,
    )

    with h5py.File(out_path, "w") as f:
        f.create_dataset("foreground", data=segmenter._foreground, compression="gzip")
        f.create_dataset("center_distances", data=segmenter._center_distances, compression="gzip")
        f.create_dataset("boundary_distances", data=segmenter._boundary_distances, compression="gzip")
        if prompts is not None:
            f.create_dataset("prompts", data=prompts["points"])

    return predictor, segmenter


def _require_intermediates(result_info, analysis_folder, model_type, settings):
    os.makedirs(analysis_folder, exist_ok=True)
    paths = []
    predictor, segmenter = None, None
    for image_path in tqdm(result_info["image_paths"], desc=f"Precompute intermediates for {analysis_folder}"):
        fname = f"{Path(image_path).stem}.h5"
        out_path = os.path.join(analysis_folder, fname)
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        predictor, segmenter = _run_prediction(image_path, out_path, predictor, segmenter, model_type, settings)
        paths.append(out_path)
    return paths


def _plot_posthoc(im_path, lab_path, pred_path, intermed, msa):
    image = imageio.imread(im_path)
    labels = imageio.imread(lab_path).astype("uint32")
    seg = imageio.imread(pred_path)

    with h5py.File(intermed, "r") as f:
        foreground = f["foreground"][:]
        boundary_distances = f["boundary_distances"][:]
        center_distances = f["center_distances"][:]
        if "prompts" in f:
            prompts = f["prompts"][:]
        else:
            prompts = None

    fname = os.path.basename(im_path)
    v = napari.Viewer()
    v.add_image(image)
    v.add_image(foreground)
    v.add_image(boundary_distances, visible=False)
    v.add_image(center_distances, visible=False)
    v.add_labels(labels)
    v.add_labels(seg)
    if prompts is not None:
        prompts = prompts.squeeze(1)[:, ::-1]
        v.add_points(prompts)
    v.title = os.path.basename(f"{fname}: mSA: {msa}")
    napari.run()


def analyze_posthoc(dataset_name, skip_visualization, k, gs_root=None):
    result_info_path = os.path.join(f"./figures/{dataset_name}/summary.json")
    assert os.path.exists(result_info_path), result_info_path

    with open(result_info_path, "r") as f:
        result_info = json.load(f)

    # Take care of other histopatho datasets once we have them.
    if dataset_name in ("pannuke",):
        model_type = "vit_b_histopathology"
    else:
        model_type = "vit_b_lm"

    if gs_root is not None:
        settings_path = os.path.join(
            gs_root, dataset_name, "results", "grid_search_params_instance_segmentation_with_decoder.csv"
        )
        assert os.path.exists(settings_path), settings_path
        gs_settings = pd.read_csv(settings_path).drop(columns=["Unnamed: 0",  "best_msa"])
        gs_settings = {k: v for k, v in zip(gs_settings.columns.values, gs_settings.values.squeeze())}
        if "prompt_selection" not in gs_settings:
            gs_settings["prompt_selection"] = "connected_components"
        print("GS Settings:")
        for name, val in gs_settings.items():
            print(name, ":", val)
    else:
        gs_settings = {}

    analysis_folder = f"./analysis/{dataset_name}"
    intermediates = _require_intermediates(result_info, analysis_folder, model_type, gs_settings)

    if skip_visualization:
        return

    i = 0
    for im_path, lab_path, pred_path, intermed, msa in zip(
        result_info["image_paths"],
        result_info["label_paths"],
        result_info["prediction_paths"],
        intermediates,
        result_info["msas"],
    ):
        if i >= k:
            break
        _plot_posthoc(im_path, lab_path, pred_path, intermed, msa)
        i += 1


def run_analyze_posthoc(datasets, skip_visualization, k, gs_root):
    if datasets is None:
        datasets = [os.path.basename(path) for path in glob(os.path.join("figures/*"))]
    for dataset in datasets:
        analyze_posthoc(dataset, skip_visualization, k, gs_root)


# Observations and hypotheses for the datasets I could inspect:
# - Arvidsson: Cytosol is picked up + artifacts from merging the masks together.
# - BlastoSPIM: no prompts in extremely low SNR images.
# - DeepBacs: Artifacts from segmentation merging, slight mismatch of annotations and predictions
#   (pred is better than labels).
# - IFNuclei: Slight mismatch prediction and labels, some over-segmentation, some GT issues
# - Pannuke: Over-segmentation, prediction better than GT?
# - PlantSeg: some artifacts, difficult samples
# - TNBC: Looks very good (min mSA > 0.4)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datasets", nargs="+")
    parser.add_argument("-s", "--skip_visualization", action="store_true")
    parser.add_argument("-k", "--worst_k", type=int, default=10)
    parser.add_argument("-g", "--gs_root", default=DEFAULT_ROOT)
    args = parser.parse_args()
    run_analyze_posthoc(args.datasets, args.skip_visualization, args.worst_k, args.gs_root)


if __name__ == "__main__":
    main()
