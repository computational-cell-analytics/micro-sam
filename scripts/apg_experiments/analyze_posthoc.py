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

HISTO_DATASETS = ("cytodark0", "ihc_tma", "lynsec", "monuseg", "pannuke", "nuinsseg", "puma", "tnbc")


def _normalize_and_pad(image):
    if image.ndim == 3:  # RGB -> normalize per channel.
        assert image.shape[-1] == 3  # ensure channel last.
        image = image.astype("float32")
        image -= image.min(axis=(0, 1))
        image /= (image.max(axis=(0, 1)) + 1e-7)
        image *= 255
        image = image.astype("uint8")

    shape = image.shape[:2]
    min_shape = (512, 512)
    pad_width = [max(0, ms - sh) for sh, ms in zip(shape, min_shape)]
    if any(pw > 0 for pw in pad_width):
        pad_width_np = [(0, pad_width[0]), (0, pad_width[1])]
        if image.ndim == 3:
            pad_width_np += [(0, 0)]
        image = np.pad(image, pad_width_np)
        crop = np.s_[0:image.shape[0]-pad_width[0], 0:image.shape[1]-pad_width[1]]
    else:
        crop = None

    return image, crop


def _run_prediction(image_path, label_path, out_path, predictor, segmenter, model_type, settings):
    from micro_sam.inference import batched_inference
    from micro_sam.instance_segmentation import _derive_point_prompts, _derive_box_prompts
    from micro_sam.util import apply_nms
    from elf.evaluation import mean_segmentation_accuracy

    if predictor is None:
        from micro_sam.instance_segmentation import AutomaticPromptGenerator, get_predictor_and_decoder
        predictor, decoder = get_predictor_and_decoder(model_type=model_type)
        segmenter = AutomaticPromptGenerator(predictor, decoder)

    image = imageio.imread(image_path)
    image, crop = _normalize_and_pad(image)
    segmenter.initialize(image)

    # Derive prompts.
    prompt_kwargs = {k: v for k, v in settings.items() if k in signature(_derive_point_prompts).parameters}
    prompts = _derive_point_prompts(
        segmenter._foreground,
        segmenter._center_distances,
        segmenter._boundary_distances,
        **prompt_kwargs,
    )

    if settings.get("refine_with_box_prompts", False) and (prompts is not None):
        box_extension = 0.01  # expose as hyperparam?
        predictions = batched_inference(
            predictor, image=image, batch_size=16, return_instance_segmentation=False,
            verbose_embeddings=False, **prompts,
        )
        box_prompts = _derive_box_prompts(predictions, box_extension)
        # Add the segmentation from points so that we can understand the difference.
        seg_from_points = apply_nms(
            predictions,
            min_size=settings.get("min_size", 25),
            nms_thresh=settings.get("nms_threshold", 0.9),
            intersection_over_min=settings.get("intersection_over_min", False),
        )
        if crop is not None:
            seg_from_points = seg_from_points[crop]
        labels = imageio.imread(label_path)
        assert labels.shape == seg_from_points.shape, f"{labels.shape}, {seg_from_points.shape}"
        msa_points = mean_segmentation_accuracy(seg_from_points, labels)
    else:
        box_prompts = None

    with h5py.File(out_path, "w") as f:
        f.create_dataset("foreground", data=segmenter._foreground, compression="gzip")
        f.create_dataset("center_distances", data=segmenter._center_distances, compression="gzip")
        f.create_dataset("boundary_distances", data=segmenter._boundary_distances, compression="gzip")
        if prompts is not None:
            f.create_dataset("prompts", data=prompts["points"])
        if box_prompts is not None:
            f.create_dataset("box_prompts", data=box_prompts["boxes"])
            ds = f.create_dataset("seg_from_points", data=seg_from_points, compression="gzip")
            ds.attrs["msa_points"] = msa_points

    return predictor, segmenter


def _require_intermediates(result_info, analysis_folder, model_type, settings):
    os.makedirs(analysis_folder, exist_ok=True)
    paths = []
    predictor, segmenter = None, None
    for image_path, label_path in tqdm(
        zip(result_info["image_paths"], result_info["label_paths"]),
        desc=f"Precompute intermediates for {analysis_folder}",
        total=len(result_info["image_paths"]),
    ):
        fname = f"{Path(image_path).stem}.h5"
        out_path = os.path.join(analysis_folder, fname)
        if os.path.exists(out_path):
            paths.append(out_path)
            continue
        predictor, segmenter = _run_prediction(
            image_path, label_path, out_path, predictor, segmenter, model_type, settings
        )
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
        if "box_prompts" in f:
            box_prompts = f["box_prompts"][:]
            seg_from_points = f["seg_from_points"][:]
            msa_points = f["seg_from_points"].attrs["msa_points"]
        else:
            box_prompts = None

    fname = os.path.basename(im_path)
    title = f"{fname}: mSA: {np.round(msa, 4)}"
    v = napari.Viewer()
    v.add_image(image)
    v.add_image(foreground, visible=False)
    v.add_image(boundary_distances, visible=False)
    v.add_image(center_distances, visible=False)
    v.add_labels(labels)
    v.add_labels(seg)

    if prompts is not None:
        prompts = prompts.squeeze(1)[:, ::-1]
        v.add_points(prompts)
    if box_prompts is not None:
        box_prompts = [
            [[p[1], p[0]], [p[3], p[2]]] for p in box_prompts
        ]
        v.add_shapes(box_prompts, shape_type="rectangle", edge_color="orange", edge_width=2, face_color="transparent")
        v.add_labels(seg_from_points)
        title += f", mSA (points): {np.round(msa_points, 4)}"

    v.title = os.path.basename(title)
    napari.run()


def analyze_posthoc(input_folder, dataset_name, skip_visualization, k, gs_root=None):
    result_info_path = os.path.join(f"{input_folder}/figures/{dataset_name}/summary.json")
    assert os.path.exists(result_info_path), result_info_path

    with open(result_info_path, "r") as f:
        result_info = json.load(f)

    # Take care of other histopatho datasets once we have them.
    if dataset_name.startswith(HISTO_DATASETS):
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
        print("GS Settings:")
        for name, val in gs_settings.items():
            print(name, ":", val)
    else:
        gs_settings = {}
        if "with_box" in input_folder:
            gs_settings["refine_with_box_prompts"] = True

    analysis_folder = f"{input_folder}/analysis/{dataset_name}"
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


def run_analyze_posthoc(input_folder, datasets, skip_visualization, k, gs_root):
    if datasets is None:
        datasets = [os.path.basename(path) for path in glob(os.path.join(f"{input_folder}/figures/*"))]
    for dataset in datasets:
        analyze_posthoc(input_folder, dataset, skip_visualization, k, gs_root)


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
    parser.add_argument("-i", "--input_folder", required=True)
    parser.add_argument("-d", "--datasets", nargs="+")
    parser.add_argument("-s", "--skip_visualization", action="store_true")
    parser.add_argument("-k", "--worst_k", type=int, default=10)
    parser.add_argument("-g", "--gs_root")
    args = parser.parse_args()
    run_analyze_posthoc(args.input_folder, args.datasets, args.skip_visualization, args.worst_k, args.gs_root)


if __name__ == "__main__":
    main()
