import argparse
import json
import os
from pathlib import Path

import h5py
import imageio.v3 as imageio
import napari
from tqdm import tqdm


def _run_prediction(image_path, out_path, predictor, segmenter):
    if predictor is None:
        from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder
        predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
        segmenter = AutomaticPromptGenerator(predictor, decoder)

    image = imageio.imread(image_path)
    segmenter.initialize(image)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("foreground", data=segmenter._foreground, compression="gzip")
        f.create_dataset("center_distances", data=segmenter._center_distances, compression="gzip")
        f.create_dataset("boundary_distances", data=segmenter._boundary_distances, compression="gzip")

    # TODO derive prompts -> we need the prompt settings for this run!
    return predictor, segmenter


def _require_intermediates(result_info, analysis_folder):
    os.makedirs(analysis_folder, exist_ok=True)
    paths = []
    predictor, segmenter = None, None
    for image_path in tqdm(result_info["image_paths"], desc="Precompute intermediate results"):
        fname = f"{Path(image_path).stem}.h5"
        out_path = os.path.join(analysis_folder, fname)
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        predictor, segmenter = _run_prediction(image_path, out_path, predictor, segmenter)
        paths.append(out_path)
    return paths


def _plot_posthoc(im_path, lab_path, pred_path, intermed):
    image = imageio.imread(im_path)
    labels = imageio.imread(lab_path).astype("uint32")
    seg = imageio.imread(pred_path)

    with h5py.File(intermed, "r") as f:
        foreground = f["foreground"][:]
        boundary_distances = f["boundary_distances"][:]
        center_distances = f["center_distances"][:]
        # TODO load and display prompts

    v = napari.Viewer()
    v.add_image(image)
    v.add_image(foreground)
    v.add_image(boundary_distances, visible=False)
    v.add_image(center_distances, visible=False)
    v.add_labels(labels)
    v.add_labels(seg)
    # TODO mSA
    v.title = os.path.basename(im_path)
    napari.run()


def analyze_posthoc(dataset_name):
    result_info_path = os.path.join(f"./figures/{dataset_name}/summary.json")
    assert os.path.exists(result_info_path), result_info_path

    with open(result_info_path, "r") as f:
        result_info = json.load(f)

    analysis_folder = f"./analysis/{dataset_name}"
    intermediates = _require_intermediates(result_info, analysis_folder)

    # TODO also display the mSA in the title.
    for im_path, lab_path, pred_path, intermed in zip(
        result_info["image_paths"],
        result_info["label_paths"],
        result_info["prediction_paths"],
        intermediates,
    ):
        _plot_posthoc(im_path, lab_path, pred_path, intermed)


# Hypotheses for the issues we observe:
# - TissueNet: images with very low contrast in the green channel -> independent normalization of channels?
# - LiveCell: something is off with the predictions (wrong number).
# - PanNuke: model issues + size filter? Are we sure this is PathoSAM?
# - DSB: Predominantly GT issues.
# In general: store prompt settings.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    args = parser.parse_args()
    analyze_posthoc(args.dataset)


if __name__ == "__main__":
    main()
