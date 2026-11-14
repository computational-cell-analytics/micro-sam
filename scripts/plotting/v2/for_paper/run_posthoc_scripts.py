"""The post-hoc scripts of the paper figures: they make and pick the examples that the figure scripts show.

Usage: python run_posthoc_scripts.py <command> <its arguments...>

The commands:
- predict_samples: run one automatic evaluation script on chosen samples only, and save their image, labels and
  prediction (the automatic qualitative plots of Figures 2 and 3).
- keep_predictions: run an interactive evaluation script and keep all of its predictions.
- first_prompt: run an interactive evaluation script for the first prompt only, and keep the mask and the prompt of
  every object (the interactive qualitative plot of Figure 2).
- first_prompt_sam3: give SAM3 the same first prompts as micro-sam2, and keep the mask of every object.
- rank_candidates: the first ranking of the interactive qualitative candidates, by the lead of micro-sam2 over
  micro-sam.
- select_candidates: pick, score and show the crowded images of a dataset as interactive qualitative candidates.
"""
import os
import sys
import glob
import runpy
import shutil
import inspect
import zipfile
import argparse

import tifffile
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image
from skimage import draw
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as patheffects
from skimage.segmentation import find_boundaries

import plots_for_figure2 as fig2
from plot_style import TEXT_COLOR, set_style

matplotlib.use("Agg")

EVALUATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), *[".."] * 4, "finetuning", "v2", "evaluation")
SAM3_ROOT = "/mnt/vast-nhr/home/archit/u12090/SAM3_Experiments"


# The automatic qualitative samples.
def predict_samples(name, script, args):
    """Run one automatic evaluation script on chosen samples only, and save their image, labels and prediction.

    Usage: SAMPLES=3,17 OUT_DIR=<folder> python run_posthoc_scripts.py predict_samples <name> <evaluation script>
    <its arguments...>
    The model is built once; each chosen sample is then segmented and scored as by the evaluation itself. The image goes
    to <OUT_DIR>/image_<sample>.tif (from the first method run on it, so run micro-sam2 first), the labels to
    labels_<sample>.tif and the prediction, after the post-processing of the evaluation, to <name>_<sample>.tif.
    """
    sys.path.insert(0, EVALUATION)
    import common

    out = os.environ["OUT_DIR"]
    chosen = [int(i) for i in os.environ["SAMPLES"].split(",")]
    os.makedirs(out, exist_ok=True)
    current = [None]
    original_evaluate, original_score = common.evaluate_samples, common.run_dataset_evaluation
    parameters = inspect.signature(original_evaluate)

    def evaluate_samples(*a, **k):
        bound = parameters.bind(*a, **k)
        segment_fn = bound.arguments["segment_fn"]

        def segment(raw):
            # The image is kept from the first method that saves it: each method gets the image prepared for its
            # model, e.g. CellViT++ with the channels in another order, so later methods must not overwrite it.
            path = os.path.join(out, f"image_{current[0]}.tif")
            if not os.path.exists(path):
                tifffile.imwrite(path, np.asarray(raw))
            return segment_fn(raw)

        bound.arguments.update(segment_fn=segment, shard=None, limit=None)
        for index in chosen:
            current[0] = index
            bound.arguments["sample_index"] = index
            original_evaluate(*bound.args, **bound.kwargs)

    def run_dataset_evaluation(gts, segs, *a, **k):
        tifffile.imwrite(os.path.join(out, f"labels_{current[0]}.tif"), np.asarray(gts[0]).astype("uint32"))
        tifffile.imwrite(os.path.join(out, f"{name}_{current[0]}.tif"), np.asarray(segs[0]).astype("uint32"))
        return original_score(gts, segs, *a, **k)

    common.evaluate_samples, common.run_dataset_evaluation = evaluate_samples, run_dataset_evaluation
    sys.argv = [script] + args
    runpy.run_path(os.path.join(EVALUATION, script), run_name="__main__")


# The kept predictions.
def keep_predictions(script, args):
    """Run an interactive evaluation script and keep its predictions, for the qualitative figures.

    The interactive evaluation scripts score their predictions and then delete them. This runs one of them with
    'shutil.rmtree' turned off, so that every image, label and prediction per iteration stays in its experiment folder.

    Usage: python run_posthoc_scripts.py keep_predictions <evaluation script> <its arguments...>
    """
    shutil.rmtree = lambda path, *args, **kwargs: print(f"Kept '{path}' for the qualitative figures.")
    sys.path.insert(0, EVALUATION)
    sys.argv = [script] + args
    runpy.run_path(os.path.join(EVALUATION, script), run_name="__main__")


# The first prompt. The interactive evaluation merges the masks of all objects of an image into one segmentation, so
# the mask of a single object is lost. The figures use the masks of single objects.
def first_prompt(script, args):
    """Run an interactive evaluation script for the first prompt only, and keep the mask and the prompt of every object.

    This saves, next to each prediction, an npz file with the mask of every labelled object in the order of its id
    ('masks'), and its point ('points', 'point_labels') or box ('boxes'), as the evaluation computed them.

    Usage: python run_posthoc_scripts.py first_prompt <evaluation script> <its arguments...>  (pass '-iter 1' or
    '--n_iterations 1')

    With the environment variable SINGLE_PROMPT_IMAGES set to a text file of images, one per line, as file names or
    full paths, the dataset is cut to these images, in their order in the test split. The samples are then numbered
    within these images.
    """
    sys.path.insert(0, EVALUATION)
    import common
    import micro_sam.v1.evaluation.inference as v1_inference
    import micro_sam.v2.evaluation.inference as v2_inference

    prompts = {}
    original_prompts, original_save = v1_inference._get_batched_prompts, v1_inference._save_segmentation

    def get_batched_prompts(*args, **kwargs):
        points, point_labels, boxes = original_prompts(*args, **kwargs)
        prompts.update(points=points, point_labels=point_labels, boxes=boxes)
        return points, point_labels, boxes

    def save_segmentation(masks, prediction_path):
        masks_array = np.asarray(masks.detach().cpu() if hasattr(masks, "detach") else masks).astype(bool)
        os.makedirs(os.path.dirname(prediction_path), exist_ok=True)
        np.savez_compressed(
            f"{os.path.splitext(prediction_path)[0]}.npz",
            masks=masks_array.reshape(len(masks_array), *masks_array.shape[-2:]),
            **{key: np.asarray(value) for key, value in prompts.items() if value is not None},
        )
        return original_save(masks, prediction_path)

    for module in (v1_inference, v2_inference):
        module._get_batched_prompts = get_batched_prompts
        module._save_segmentation = save_segmentation
    if os.environ.get("SINGLE_PROMPT_IMAGES"):
        with open(os.environ["SINGLE_PROMPT_IMAGES"]) as f:
            names = {line for line in f.read().split("\n") if line}  # one image per line; paths may have spaces
        original_paths = common.get_data_paths

        def get_data_paths(*args, **kwargs):
            raw_paths, label_paths, *keys = original_paths(*args, **kwargs)
            keep = [path in names or os.path.basename(path) in names for path in raw_paths]
            assert sum(keep) == len(names), f"Found {sum(keep)} of {len(names)} images."
            return (
                [path for path, k in zip(raw_paths, keep) if k], [path for path, k in zip(label_paths, keep) if k],
                *keys
            )

        common.get_data_paths = get_data_paths

    shutil.rmtree = lambda path, *args, **kwargs: print(f"Kept '{path}' for the qualitative figures.")
    sys.argv = [script] + args
    runpy.run_path(os.path.join(EVALUATION, script), run_name="__main__")


def microsam2_folders(dataset, prompt):
    """Return the folders of the inputs and of the first-prompt predictions of micro-sam2. Unlike
    'folders', this does not need the runs of micro-sam, which may not be done yet."""
    pattern = f"{PREDICTIONS}/microsam2/predictions/*hvit_l/{dataset}/{prompt}_with_masks_center"
    folder = glob.glob(pattern)[0]
    return {"inputs": f"{folder}/inputs", "micro-sam2": f"{folder}/start_with_{prompt}/with_masks/iteration00"}


def first_prompt_sam3(args):
    """Run SAM3 with the first prompts of micro-sam2, and keep the mask and the prompt of every object.

    The SAM3 evaluation has its own prompting loop, so 'first_prompt' cannot keep its masks. This gives SAM3 the same
    image and the same point or box per object as micro-sam2 (from its npz files of 'first_prompt'), and saves an npz
    file with the same keys next to the other methods, in '<PREDICTIONS>/sam3/<dataset>/<prompt>'.

    Usage: python run_posthoc_scripts.py first_prompt_sam3 [--root <predictions>] [--dataset <name>]  (all images of
    micro-sam2 for one dataset; by default the chosen images of the figure)
    """
    global PREDICTIONS
    parser = argparse.ArgumentParser(prog="run_posthoc_scripts.py first_prompt_sam3")
    parser.add_argument("--dataset", default=None, help="Run all images of micro-sam2 for this dataset.")
    parser.add_argument("--root", default=PREDICTIONS, help="The folder of the first-prompt runs.")
    args = parser.parse_args(args)
    PREDICTIONS = args.root
    output = f"{args.root}/sam3"
    if args.dataset is None:
        images = [(dataset, image_name) for dataset, image_name, *_ in CHOSEN if image_name is not None]
    else:
        folder = microsam2_folders(args.dataset, "box")["micro-sam2"]
        images = [(args.dataset, os.path.basename(path)[:-4]) for path in sorted(glob.glob(f"{folder}/*.npz"))]

    sys.path.insert(0, SAM3_ROOT)
    from micro_sam3.evaluation.inference import build_sam3_image_predictor

    model, processor = build_sam3_image_predictor()
    for dataset, image_name in images:
        for prompt in PROMPTS:
            paths = microsam2_folders(dataset, prompt)
            out_path = f"{output}/{dataset}/{prompt}/{image_name}.npz"
            if os.path.exists(out_path):
                continue
            image = tifffile.imread(f"{paths['inputs']}/images/{image_name}.tif")
            rgb = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
            state = processor.set_image(Image.fromarray(rgb.astype("uint8")))
            prompts = dict(np.load(f"{paths['micro-sam2']}/{image_name}.npz"))
            masks = []
            for index in range(len(prompts["masks"])):
                if prompt == "box":
                    kwargs = dict(box=prompts["boxes"][index:index + 1].astype("float32"))
                else:
                    kwargs = dict(
                        point_coords=prompts["points"][index].astype("float32"),
                        point_labels=prompts["point_labels"][index]
                    )
                # A single mask for the first prompt, as in the SAM3 evaluation.
                object_masks, _, _ = model.predict_inst(state, multimask_output=False, **kwargs)
                masks.append(object_masks[0] > 0)
            prompts["masks"] = np.stack(masks)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.savez_compressed(out_path, **prompts)
            print(f"Saved '{out_path}'.")


# The first ranking of the interactive qualitative candidates.
PREDICTIONS = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_single_prompt_2d"
RANK_OUTPUT = "figures/figure2/interactive/qualitative/candidates/chosen"
# The datasets and their (domain, modality) folder, in the order of the figure.
DATASETS = {
    "livecell": ("id", "lf_cells"), "orgline": ("id", "lf_cells"), "vicar": ("id", "lf_cells"),
    "tissuenet": ("id", "fl_cells"), "flywing": ("id", "fl_cells"), "neurips_cellseg_fluorescence": ("id", "fl_cells"),
    "bmgd": ("id", "fl_nuc"), "cellbindb": ("id", "fl_nuc"), "ifnuclei": ("id", "fl_nuc"),
    "glysac": ("id", "hp_nuc"), "histo_miner": ("id", "hp_nuc"), "monuseg": ("id", "hp_nuc"),
    "cpm17": ("id", "hp_nuc"), "puma": ("id", "hp_nuc"), "pannuke": ("id", "hp_nuc"),
}
# The datasets of 'chosen.png', chosen by hand; the H&E datasets keep their candidates but are not in it.
CHOSEN_DATASETS = [
    "livecell", "orgline", "vicar", "tissuenet", "flywing", "neurips_cellseg_fluorescence", "bmgd", "cellbindb",
    "ifnuclei",
]
# The images chosen by hand, which replace the top-ranked image of their dataset in 'chosen.png'.
CHOSEN_IMAGES = {"flywing": "00037", "tissuenet": "00030"}
RANK_METHODS = ("micro-sam", "micro-sam2")
PROMPTS = ("point", "box")
# The objects worth showing: away from the image border (border margin, as a fraction of the image), among the
# larger objects of their image (at least a quantile of its object areas), and with micro-sam2 better by at least a
# minimal gap for one prompt; the three objects of an image lie a spacing apart (as a fraction of the image). A
# dataset with fewer than RANK_N_IMAGES images under the strict rules gets the next, looser ones.
RULES = [
    dict(border_margin=0.1, big_quantile=0.6, min_gap=0.15, spacing=0.2),
    dict(border_margin=0.08, big_quantile=0.5, min_gap=0.1, spacing=0.15),
    dict(border_margin=0.05, big_quantile=0.3, min_gap=0.05, spacing=0.12),
]
N_OBJECTS, RANK_N_IMAGES = 3, 3
OBJECT_COLORS = ["#ff4d6d", "#ffd23f", "#3bceac"]  # objects 1, 2 and 3
PROMPT_COLOR = (0.0, 1.0, 1.0)  # cyan, as in the micro-sam (v1) plots
POINT_RADIUS = 3


def folders(dataset, prompt):
    """Return the folders of the inputs and of the first-prompt predictions of both methods, or None."""
    microsam2 = glob.glob(f"{PREDICTIONS}/microsam2/predictions/*hvit_l/{dataset}/{prompt}_with_masks_center")
    microsam = glob.glob(f"{PREDICTIONS}/microsam_v1/predictions/*/{dataset}/{prompt}_without_masks_center")
    if not microsam2 or not microsam:
        return None
    return {
        "inputs": f"{microsam2[0]}/inputs",
        "micro-sam2": f"{microsam2[0]}/start_with_{prompt}/with_masks/iteration00",
        "micro-sam": f"{microsam[0]}/predictions/iteration00",
    }


def score_objects():
    """Return the IoU of every labelled object with both methods and both prompts, and where it lies."""
    rows = []
    for dataset in DATASETS:
        paths = {prompt: folders(dataset, prompt) for prompt in PROMPTS}
        if any(path is None for path in paths.values()):
            print(f"No predictions for '{dataset}'.")
            continue
        for label_path in sorted(glob.glob(f"{paths['point']['inputs']}/labels/*.tif")):
            image = os.path.basename(label_path)[:-4]
            files = {(p, m): f"{paths[p][m]}/{image}.npz" for p in PROMPTS for m in RANK_METHODS}
            if not all(os.path.exists(path) for path in files.values()):
                continue
            labels = tifffile.imread(label_path)
            masks = {key: np.load(path)["masks"] for key, path in files.items()}
            for index, object_id in enumerate(np.setdiff1d(np.unique(labels), [0])):
                reference = labels == object_id
                ys, xs = np.nonzero(reference)
                # The least distance of the object from the image border, as a fraction of the image.
                border = min(
                    ys.min() / labels.shape[0], xs.min() / labels.shape[1],
                    (labels.shape[0] - 1 - ys.max()) / labels.shape[0],
                    (labels.shape[1] - 1 - xs.max()) / labels.shape[1]
                )
                row = dict(
                    dataset=dataset, image=image, object=int(object_id), index=index, area=int(reference.sum()),
                    cy=ys.mean() / labels.shape[0], cx=xs.mean() / labels.shape[1], border=border,
                )
                for (prompt, method), mask in masks.items():
                    mask = mask[index]
                    union = np.logical_or(mask, reference).sum()
                    row[f"{method}_{prompt}"] = np.logical_and(mask, reference).sum() / union if union else 0.0
                rows.append(row)
    objects = pd.DataFrame(rows)
    for prompt in PROMPTS:
        objects[f"gap_{prompt}"] = objects[f"micro-sam2_{prompt}"] - objects[f"micro-sam_{prompt}"]
    objects["gap"] = objects[[f"gap_{prompt}" for prompt in PROMPTS]].mean(axis=1)
    return objects


def pick_objects(objects, border_margin, big_quantile, min_gap, spacing):
    """Return the three objects of each image worth showing under one set of rules, for the images that have them."""
    big = objects.groupby(["dataset", "image"]).area.transform(lambda area: area.quantile(big_quantile))
    good = objects[
        (objects.gap_point > 0) & (objects.gap_box > 0) & (objects.border >= border_margin) & (objects.area >= big)
        & (objects[["gap_point", "gap_box"]].max(axis=1) >= min_gap)
    ]
    picked_rows = []
    for _, group in good.sort_values("gap", ascending=False).groupby(["dataset", "image"], sort=False):
        picked = []
        for obj in group.to_dict("records"):
            if all(np.hypot(obj["cy"] - other["cy"], obj["cx"] - other["cx"]) >= spacing for other in picked):
                picked.append(obj)
            if len(picked) == N_OBJECTS:
                break
        if len(picked) == N_OBJECTS:
            picked_rows += picked
    return pd.DataFrame(picked_rows)


def pick_per_dataset(objects):
    """Return the chosen objects of every dataset under the strictest rules that give it RANK_N_IMAGES images."""
    parts = []
    for dataset, group in objects.groupby("dataset", sort=False):
        for level, rules in enumerate(RULES):
            picked = pick_objects(group, **rules)
            n_images = 0 if picked.empty else picked.image.nunique()
            if n_images >= RANK_N_IMAGES or level == len(RULES) - 1:
                break
        if not picked.empty:
            parts.append(picked.assign(rules=level))
    return pd.concat(parts, ignore_index=True)


def as_rgb(image, shape):
    """Return an image as RGB in [0, 1], with the contrast stretched over the 1 to 99.7 percentiles."""
    image = np.asarray(image, dtype="float32")
    if image.ndim == 3 and image.shape[:2] != shape:
        image = np.moveaxis(image, 0, -1)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 2:
        image = np.stack([np.zeros(image.shape[:2]), image[..., 0], image[..., 1]], axis=-1)
    image = image[..., :3]
    low, high = np.percentile(image, (1, 99.7))
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def overlay_prompt(image, prompt, data, index):
    """Burn the prompt of an object into an RGB image as in the micro-sam (v1) plots: the box as a one-pixel cyan
    rectangle, the point as a cyan disk of radius POINT_RADIUS. SAM stores points as (x, y) and boxes as
    (x0, y0, x1, y1)."""
    out = image.copy()
    if prompt == "point":
        x, y = data["points"][index][0]
        rr, cc = draw.disk((y, x), POINT_RADIUS, shape=out.shape[:2])
    else:
        x0, y0, x1, y1 = data["boxes"][index]
        rr, cc = draw.rectangle_perimeter((y0, x0), end=(y1 - 1, x1 - 1), shape=out.shape[:2])
    out[rr, cc] = PROMPT_COLOR
    return out


def render_row(axes, dataset, image_name, chosen, title):
    """Render one image: the reference and both methods with both prompts, for its three chosen objects."""
    paths = {prompt: folders(dataset, prompt) for prompt in PROMPTS}
    labels = tifffile.imread(f"{paths['point']['inputs']}/labels/{image_name}.tif")
    image = as_rgb(tifffile.imread(f"{paths['point']['inputs']}/images/{image_name}.tif"), labels.shape)
    colors = [np.array(matplotlib.colors.to_rgb(color)) for color in OBJECT_COLORS]
    reference = image.copy()
    for color, obj in zip(colors, chosen):
        reference[find_boundaries(labels == obj["object"], mode="thick")] = color
    axes[0].imshow(reference)
    axes[0].set_title(title, fontsize=8)
    panels = [(prompt, method) for prompt in PROMPTS for method in RANK_METHODS]
    for ax, (prompt, method) in zip(axes[1:], panels):
        data = np.load(f"{paths[prompt][method]}/{image_name}.npz")
        out = image.copy()
        for color, obj in zip(colors, chosen):
            mask = data["masks"][obj["index"]]
            out[mask] = 0.45 * out[mask] + 0.55 * color
        for obj in chosen:
            out = overlay_prompt(out, prompt, data, obj["index"])
        ax.imshow(out)
        name = "μSAM2" if method == "micro-sam2" else "μSAM"
        ious = " / ".join(f"{obj[f'{method}_{prompt}']:.2f}" for obj in chosen)
        ax.set_title(
            f"{name}, {prompt}: IoU {ious}", fontsize=8, fontweight="bold" if method == "micro-sam2" else "normal"
        )
    for ax in axes:
        ax.axis("off")


def rank_candidates(rescore=False):
    """Pick and render the single-prompt candidates of the qualitative interactive figure of Figure 2.

    The predictions come from 'run_posthoc_scripts.py first_prompt': micro-sam2 (hvit_l) and micro-sam (vit_l_lm, or
    PathoSAM on H&E), each with one point or one box per object, with the mask and the prompt of every labelled object
    kept.

    For every labelled object, its gap is its IoU with micro-sam2 minus its IoU with micro-sam, for the point and for
    the box. An object is worth showing when micro-sam2 is better with both prompts and by at least MIN_GAP with one of
    them, when it lies away from the image border and when it is among the larger objects of its image. The three such
    objects of an image with the largest mean gap, at least SPACING apart, make its row; the images of a dataset are
    ranked by the summed gap of their three objects.

    The script writes, under 'figures/figure2/interactive/qualitative/candidates/chosen':
    - '<domain>/<modality>/<dataset>.png': the top RANK_N_IMAGES images of each dataset, one per row;
    - 'chosen.png' and 'chosen.csv': the top image of every dataset, one per row. A row shows the reference with the
      three objects outlined in their colors, then micro-sam and micro-sam2 with the point, and micro-sam and micro-sam2
      with the box. The predicted masks are filled in the colors of their objects, and the prompts are drawn as in the
      micro-sam (v1) plots: a cyan dot for a point, a cyan rectangle for a box.
    """
    # The object scores take long to compute, so they are kept and read back unless 'rescore' is set.
    scores_path = "qualitative_single_prompt_objects_2d.csv"
    if rescore or not os.path.exists(scores_path):
        objects = score_objects()
        objects.to_csv(scores_path, index=False)
    objects = pd.read_csv(scores_path, dtype={"image": str})
    picked = pick_per_dataset(objects)
    images = picked.groupby(["dataset", "image"]).agg(score=("gap", "sum"), rules=("rules", "first")).reset_index()
    images = images.sort_values("score", ascending=False).groupby("dataset").head(RANK_N_IMAGES)
    keep = picked.set_index(["dataset", "image"]).index.isin(images.set_index(["dataset", "image"]).index)
    picked = picked[keep]
    picked.to_csv("qualitative_single_prompt_candidates_2d.csv", index=False)

    tops = []
    for dataset, (domain, modality) in DATASETS.items():
        ranked = images[images.dataset == dataset]
        if ranked.empty:
            print(f"No candidates for '{dataset}'.")
            continue
        fig, axes = plt.subplots(len(ranked), 5, figsize=(17, 3.5 * len(ranked)), squeeze=False)
        for ax_row, image_name in zip(axes, ranked.image):
            chosen = picked[(picked.dataset == dataset) & (picked.image == image_name)].to_dict("records")
            render_row(ax_row, dataset, image_name, chosen, f"{dataset} {image_name}: reference")
        plt.tight_layout()
        folder = os.path.join(RANK_OUTPUT, domain, modality)
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, f"{dataset}.png"), dpi=90)
        plt.close()
        if dataset in CHOSEN_DATASETS:
            tops.append((dataset, CHOSEN_IMAGES.get(dataset, ranked.image.iloc[0])))
        print(f"Rendered '{dataset}' ({len(ranked)} images, rules {ranked.rules.iloc[0]}).")

    fig, axes = plt.subplots(len(tops), 5, figsize=(17, 3.5 * len(tops)), squeeze=False)
    for ax_row, (dataset, image_name) in zip(axes, tops):
        chosen = picked[(picked.dataset == dataset) & (picked.image == image_name)].to_dict("records")
        render_row(
            ax_row, dataset, image_name, chosen,
            f"{dataset} ({DATASETS[dataset][0].upper()}), image {image_name}: reference"
        )
    plt.tight_layout()
    plt.savefig(os.path.join(RANK_OUTPUT, "chosen.png"), dpi=90)
    plt.close()
    pd.DataFrame(tops, columns=["dataset", "image"]).to_csv(os.path.join(RANK_OUTPUT, "chosen.csv"), index=False)
    print("Chosen:", tops)


# The crowded interactive qualitative candidates.
RUNS = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments"
OUTPUT = "figures/figure2/interactive/qualitative/candidates"
# The names of the candidate datasets that the figures do not show.
DATASET_NAMES = {
    "bac_mother": "BacMother", "bbbc010": "BBBC010", "bitdepth_nucseg": "BitDepth NucSeg", "bmgd": "BMGD",
    "cardioblast_nuclei": "Cardioblast", "cell_acdc": "Cell-ACDC", "cellapp": "CellApp", "cellular": "Cellular",
    "cisd": "CISD", "covid_if_cells": "Covid IF", "covid_if_nuclei": "Covid IF", "cvz_fluo_cell": "CVZ Fluo",
    "cvz_fluo_dapi": "CVZ Fluo", "deepbacs_fluorescence": "DeepBacs", "deepbacs_label_free": "DeepBacs",
    "deepliif": "DeepLIIF", "deepseas": "DeepSeas", "dic_hepg2": "DIC HepG2", "dsb": "DSB",
    "dynamicnuclearnet": "DynamicNuclearNet", "ecoli_microcolony_lineage": "E. coli Lineage", "enseg": "EnSeg",
    "flywing": "FlyWing", "fusionx": "FusionX", "glysac": "GlySAC", "hela_cytonuc": "HeLa CytoNuc",
    "ifnuclei": "IFNuclei", "khoshdeli": "Khoshdeli", "livecell": "LIVECell", "lizard_mitosis": "Lizard-Mitosis",
    "lynsec_he": "LyNSeC H&E", "micro_bench": "MicroBench", "microbeseg": "MicrobeSeg", "mif_tonsil": "MIF Tonsil",
    "mif_tonsil_cells": "MIF Tonsil", "monuseg": "MoNuSeg", "neurips_cellseg_fluorescence": "NeurIPS CellSeg",
    "omnipose_bact_fluor": "OmniPose", "omnipose_bact_phase": "OmniPose", "orgline": "OrgLine", "pannuke": "PanNuke",
    "panoptils": "PanopTILs", "pcns": "PCNS", "puma": "PUMA", "toiam": "TOIAM", "tsakiroglou": "Tsakiroglou",
    "u20s": "U2OS", "wing_disc_timelapse": "Wing Disc TL",
    **fig2.DATASET_NAMES,
}
MODALITY_FOLDERS = {
    "label_free_cells": "lf_cells", "fluorescence_cells": "fl_cells", "fluorescence_nuclei": "fl_nuc",
    "histopathology_nuclei": "hp_nuc",
}
# The imaging of the datasets whose candidates were looked at, for the captions of the crops.
IMAGING = {"livecell": "Phase Contrast", "orgline": "Brightfield", "vicar": "QPI"}
# The shortlisted images of a dataset as (image, specimen), instead of the images with the most candidates.
SHORTLISTS = {
    "livecell": [
        ("00005", "BT474 Cells, Day 4"), ("00013", "Huh7 Cells, Day 3"), ("00000", "A172 Cells, Day 2"),
        ("00027", "SKOV3 Cells, Day 3")
    ],
}
OTHERS = ["SAM", "SAM2", "SAM3", "micro-sam"]
OUTLINE_COLOR = np.array([1.0, 0.3, 0.43])  # the objects where micro-sam2 is best
N_IMAGES = 24  # the images of a dataset: the N_CROWDED most crowded, and the rest spread over the others
N_CROWDED = 16
MAX_OBJECTS = 800  # leaves out the images with too many, tiny objects, where there are others
N_COLUMNS = 4
N_SHORTLIST = 4
N_CANDIDATES = 5
TIE = 0.02  # micro-sam2 counts as the best within this of the best other method
BORDER = 40  # pixels between the center of an object and the image border
BORDER_FRACTION = 0.2  # at most this fraction of the image side, for small images
SPACING = 60  # pixels between the centers of the objects of an image
SMALL_FRACTION = 0.3  # the small objects (with '--small') are at most this fraction of the median area


# The chosen images: (dataset, image, modality, domain, imaging, specimen). The image is the index of the test sample
# in the single-prompt run; its three objects are the ones chosen for it in the candidates. A dataset in FINALS shows
# its final objects instead, which may come from several images; their image and specimen are then given per object.
CHOSEN = [
    ("livecell", None, "label_free_cells", "ID", "Phase Contrast", None),
    ("orgline", None, "label_free_cells", "ID", "Brightfield", "Breast Organoids"),
    ("vicar", None, "label_free_cells", "ID", "QPI", None),
    ("cell_acdc", None, "label_free_cells", "ID", "Phase Contrast", None),
    ("neurips_cellseg_label_free", None, "label_free_cells", "ID", "Label-Free", None),
    ("omnipose_bact_phase", None, "label_free_cells", "ID", "Phase Contrast", "Bacteria"),
    ("cvz_fluo_cell", None, "fluorescence_cells", "ID", "Fluorescence", None),
    ("tissuenet", None, "fluorescence_cells", "ID", "Multiplexed IF", None),
    ("flywing", None, "fluorescence_cells", "ID", "Confocal", "Drosophila Wing"),
    ("deepbacs_fluorescence", None, "fluorescence_cells", "ID", "Fluorescence", "Bacteria"),
    ("neurips_cellseg_fluorescence", None, "fluorescence_cells", "ID", "Fluorescence", None),
    ("omnipose_bact_fluor", None, "fluorescence_cells", "ID", "Fluorescence", "Bacteria"),
    ("bmgd", None, "fluorescence_nuclei", "ID", "DAPI", "Mouse Mammary Gland"),
    ("cellbindb", None, "fluorescence_nuclei", "ID", "ssDNA", None),
    ("dsb", None, "fluorescence_nuclei", "ID", "Fluorescence", None),
    ("dynamicnuclearnet", None, "fluorescence_nuclei", "ID", "Fluorescence", None),
    ("ifnuclei", None, "fluorescence_nuclei", "ID", "IF", None),
    ("u20s", None, "fluorescence_nuclei", "ID", "Fluorescence", None),
    ("cpm17", None, "histopathology_nuclei", "ID", "H&E", None),
    ("glysac", None, "histopathology_nuclei", "ID", "H&E", None),
    ("histo_miner", None, "histopathology_nuclei", "ID", "H&E", None),
    ("monuseg", None, "histopathology_nuclei", "ID", "H&E", None),
    ("pannuke", None, "histopathology_nuclei", "ID", "H&E", None),
    ("puma", None, "histopathology_nuclei", "ID", "H&E", None),
    ("covid_if_cells", None, "fluorescence_cells", "OOD", "IF", None),
    ("fusionx", None, "fluorescence_cells", "OOD", "Fluorescence", None),
    ("hpa", None, "fluorescence_cells", "OOD", "IF", None),
    ("medussa", None, "fluorescence_cells", "OOD", "Fluorescence", None),
    ("mif_tonsil_cells", None, "fluorescence_cells", "OOD", "Multiplexed IF", "Tonsil"),
    ("wing_disc_timelapse", None, "fluorescence_cells", "OOD", "Confocal", "Drosophila Wing Disc"),
    ("arvidsson", None, "fluorescence_nuclei", "OOD", "Fluorescence", None),
    ("cardioblast_nuclei", None, "fluorescence_nuclei", "OOD", "Fluorescence", None),
    ("covid_if_nuclei", None, "fluorescence_nuclei", "OOD", "IF", None),
    ("hela_cytonuc", None, "fluorescence_nuclei", "OOD", "Fluorescence", "HeLa Cells"),
    ("micro_bench", None, "fluorescence_nuclei", "OOD", "Fluorescence", None),
    ("mif_tonsil", None, "fluorescence_nuclei", "OOD", "Multiplexed IF", "Tonsil"),
    ("cytodark0", None, "histopathology_nuclei", "OOD", "H&E", None),
    ("khoshdeli", None, "histopathology_nuclei", "OOD", "H&E", None),
    ("lizard_mitosis", None, None, "supp", "H&E", None),
    ("lynsec_he", None, None, "supp", "H&E", None),
    ("pcns", None, "histopathology_nuclei", "OOD", "H&E", None),
    ("srsanet", None, None, "supp", "H&E", None),
    ("bac_mother", None, "label_free_cells", "OOD", "Label-Free", None),
    ("bbbc010", None, "label_free_cells", "OOD", "Brightfield", "C. elegans"),
    ("cellapp", None, "label_free_cells", "OOD", "Label-Free", None),
    ("deepseas", None, "label_free_cells", "OOD", "Label-Free", None),
    ("ecoli_microcolony_lineage", None, "label_free_cells", "OOD", "Label-Free", None),
    ("yeastsam", None, "label_free_cells", "OOD", "Brightfield", "Yeast"),
    ("bbbc030", None, None, "supp", "Label-Free", None),
    ("bccd", None, None, "supp", "Label-Free", None),
    ("bitdepth_nucseg", None, None, "ID", "Fluorescence", None),
    ("cellbindb_he", None, None, "supp", "H&E", None),
    ("cisd", None, None, "supp", "Label-Free", None),
    ("cellular", None, None, "ID", "Label-Free", None),
    ("cvz_fluo_dapi", None, None, "ID", "Fluorescence", None),
    ("deepbacs_label_free", None, None, "ID", "Label-Free", None),
    ("deepliif", None, None, "supp", "IHC", None),
    ("dememseg", None, None, "supp", "Fluorescence", None),
    ("lizard", None, None, "supp", "H&E", None),
    ("lynsec_ihc", None, None, "supp", "IHC", None),
    ("mcellseg", None, None, "supp", "Label-Free", None),
    ("microbeseg", None, None, "ID", "Label-Free", None),
    ("omnipose_worm", None, None, "supp", "Label-Free", None),
    ("organoid", None, None, "supp", "Label-Free", None),
    ("organoidnet", None, None, "supp", "Label-Free", None),
    ("orgasegment", None, None, "supp", "Label-Free", None),
    ("pan_multiplex", None, None, "supp", "Fluorescence", None),
    ("panoptils", None, None, "supp", "H&E", None),
    ("spatch_dapi", None, None, "supp", "DAPI", None),
    ("tnbc_celltype", None, None, "supp", "H&E", None),
    ("toiam", None, None, "supp", "Label-Free", None),
    ("tsakiroglou", None, None, "ID", "Fluorescence", None),
    ("yeaz", None, None, "supp", "Label-Free", None),
    ("mndino", None, None, "supp", "Fluorescence", "Micronuclei"),
]
CANDIDATES = os.path.join(fig2.ROOT, "qualitative_single_prompt_candidates_2d.csv")
# The final objects of a dataset: the folder of its single-prompt runs, and (image, object id, specimen) per object.
# The specimen of an object is None when the caption of the dataset has it.
# LIVECell: crowded images of all cell types ('run_posthoc_scripts.py select_candidates').
FINALS = {
    "livecell": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_single_prompt_livecell",
        [("00005", 42, "BT474 Cells"), ("00005", 24, None), ("00027", 130, "SKOV3 Cells"), ("00027", 22, None)],
    ),
    # OrgLine: crowded images of the organoid types with many organoids per image, as for LIVECell. Both organoids
    # come from one image, so the specimen is in the caption.
    "orgline": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_single_prompt_orgline",
        [("00015", 22, None), ("00015", 76, None)],
    ),
    # VICAR: all test images.
    "vicar": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_single_prompt_vicar",
        [("00018", 8, "PC3 Cells"), ("00000", 115, "A2058 Cells")],
    ),
    # CVZ Fluo, FlyWing and NeurIPS CellSeg: crowded images, as for LIVECell.
    "cvz_fluo_cell": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_single_prompt_cvz_fluo_cell",
        [("00016", 174, None), ("00005", 261, None), ("00019", 72, None), ("00019", 143, None)],
    ),
    "flywing": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_single_prompt_flywing",
        [("00009", 117, None), ("00009", 538, None), ("00023", 202, None)],
    ),
    "neurips_cellseg_fluorescence": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_neurips_cellseg_fluorescence",
        [
            ("00018", 33, None), ("00010", 14, None), ("00010", 30, None), ("00003", 5, None), ("00003", 38, None),
            ("00003", 7, None)
        ],
    ),
    "omnipose_bact_fluor": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_omnipose_bact_fluor",
        [("00014", 43, None), ("00010", 111, None), ("00010", 68, None), ("00013", 124, None), ("00012", 47, None)],
    ),
    "deepbacs_fluorescence": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_deepbacs_fluorescence",
        [("00012", 13, None), ("00014", 7, None), ("00014", 13, None), ("00006", 13, None), ("00007", 7, None)],
    ),
    # TissueNet: crowded images of all tissues.
    "tissuenet": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/qualitative_single_prompt_tissuenet",
        [
            ("00021", 87, "LN Metastasis"), ("00013", 156, "Tonsil"), ("00013", 75, None),
            ("00010", 198, "Colon"), ("00010", 101, None)
        ],
    ),
    # The nuclei datasets of the main figure: crowded images, as for LIVECell.
    "bmgd": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_bmgd",
        [("00001", 11, None), ("00007", 11, None), ("00003", 7, None), ("00021", 7, None)],
    ),
    "cellbindb": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cellbindb",
        [("00007", 15, None), ("00007", 50, None), ("00007", 40, None), ("00009", 40, None), ("00005", 167, None)],
    ),
    "dsb": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_dsb",
        [("00017", 168, None), ("00007", 74, None), ("00015", 105, None)],
    ),
    "dynamicnuclearnet": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_dynamicnuclearnet",
        [("00005", 159, None)],
    ),
    "ifnuclei": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_ifnuclei",
        [("00002", 21, None), ("00020", 125, None), ("00002", 38, None), ("00015", 38, None), ("00015", 32, None)],
    ),
    "u20s": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_u20s",
        [("00000", 20, None), ("00009", 13, None), ("00006", 43, None), ("00006", 51, None)],
    ),
    # The histopathology datasets of the main figure: crowded images, as for LIVECell.
    "cpm17": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cpm17",
        [
            ("00006", 75, None), ("00022", 75, None), ("00014", 109, None), ("00023", 43, None), ("00022", 80, None),
            ("00022", 55, None)
        ],
    ),
    "glysac": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_glysac",
        [("00011", 60, None), ("00018", 42, None)],
    ),
    "histo_miner": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_histo_miner",
        [
            ("00009", 70, None), ("00009", 62, None), ("00008", 62, None), ("00009", 80, None), ("00023", 18, None),
            ("00023", 51, None), ("00008", 51, None), ("00008", 75, None), ("00008", 66, None)
        ],
    ),
    "monuseg": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_monuseg",
        [("00011", 24, None), ("00003", 60, None), ("00003", 97, None), ("00013", 54, None)],
    ),
    "pannuke": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_pannuke",
        [("00003", 64, None), ("00018", 64, None), ("00018", 21, None), ("00020", 99, None)],
    ),
    "puma": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_puma",
        [("00018", 39, None), ("00002", 121, None), ("00020", 37, None), ("00020", 53, None)],
    ),
    # The other label-free datasets of the main figure: crowded images, as for LIVECell.
    "cell_acdc": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cell_acdc",
        [("00010", 2, None), ("00009", 2, None), ("00007", 4, None), ("00010", 5, None)],
    ),
    "neurips_cellseg_label_free": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_neurips_cellseg_label_free",
        [
            ("00006", 70, None), ("00006", 74, None), ("00006", 6, None), ("00000", 19, None), ("00004", 21, None),
            ("00004", 42, None), ("00009", 42, None), ("00009", 32, None)
        ],
    ),
    "omnipose_bact_phase": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_omnipose_bact_phase",
        [("00006", 22, None), ("00006", 96, None), ("00005", 145, None), ("00004", 170, None)],
    ),
    # The out-of-domain fluorescence cell datasets: crowded images, as for LIVECell.
    "covid_if_cells": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_covid_if_cells",
        [("00019", 52, None), ("00019", 96, None), ("00019", 37, None), ("00009", 22, None), ("00007", 73, None)],
    ),
    "fusionx": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_fusionx",
        [("00005", 30, None), ("00002", 92, None), ("00005", 40, None), ("00006", 37, None), ("00006", 36, None)],
    ),
    "hpa": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_hpa",
        [
            ("00003", 2, None), ("00003", 12, None), ("00007", 11, None), ("00001", 11, None), ("00001", 9, None),
            ("00004", 25, None)
        ],
    ),
    "medussa": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_medussa",
        [
            ("00017", 3, None), ("00017", 10, None), ("00017", 6, None), ("00004", 18, None), ("00004", 16, None),
            ("00023", 16, None), ("00017", 12, None), ("00023", 12, None)
        ],
    ),
    # mIF tonsil cells and the wing disc time-lapse: crowded images, as for LIVECell.
    "mif_tonsil_cells": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_mif_tonsil_cells",
        [("00009", 22, None), ("00001", 18, None)],
    ),
    "wing_disc_timelapse": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_wing_disc_timelapse",
        [("00022", 204, None), ("00007", 1615, None), ("00007", 1055, None)],
    ),
    # The out-of-domain fluorescence nucleus datasets: crowded images, as for LIVECell.
    "arvidsson": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_arvidsson",
        [("00003", 5, None), ("00007", 5, None), ("00003", 6, None), ("00002", 2, None), ("00004", 2, None)],
    ),
    "cardioblast_nuclei": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cardioblast_nuclei",
        [
            ("00020", 6, None), ("00018", 6, None), ("00018", 22, None), ("00007", 12, None), ("00018", 12, None),
            ("00006", 9, None), ("00007", 9, None)
        ],
    ),
    "covid_if_nuclei": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_covid_if_nuclei",
        [("00017", 106, None), ("00006", 34, None)],
    ),
    "hela_cytonuc": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_hela_cytonuc",
        [("00005", 1, None), ("00009", 15, None), ("00013", 15, None), ("00004", 47, None)],
    ),
    "spatch_dapi": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_spatch_dapi",
        [("00020", 8, None), ("00008", 81, None), ("00018", 15, None), ("00023", 15, None), ("00023", 43, None)],
    ),
    # mIF tonsil nuclei: small image tiles, with the border scaled to the tile size.
    "mif_tonsil": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_mif_tonsil",
        [("00009", 19, None)],
    ),
    # The out-of-domain histopathology datasets: crowded images, as for LIVECell.
    "cytodark0": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cytodark0",
        [("00012", 10, None), ("00011", 62, None), ("00011", 57, None)],
    ),
    "khoshdeli": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_khoshdeli",
        [("00011", 228, None), ("00002", 223, None), ("00002", 160, None)],
    ),
    "lizard_mitosis": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_lizard_mitosis",
        [("00012", 33, None), ("00016", 163, None), ("00008", 13, None)],
    ),
    "lynsec_he": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_lynsec_he",
        [
            ("00023", 59, None), ("00023", 105, None), ("00023", 136, None), ("00002", 104, None), ("00002", 153, None),
            ("00003", 39, None)
        ],
    ),
    "pcns": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_pcns",
        [("00011", 83, None), ("00002", 28, None), ("00005", 6, None)],
    ),
    "srsanet": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_srsanet",
        [
            ("00007", 23, None), ("00007", 28, None), ("00011", 28, None), ("00011", 17, None), ("00016", 17, None),
            ("00016", 5, None), ("00011", 21, None), ("00016", 21, None)
        ],
    ),
    # The out-of-domain label-free datasets: crowded images, as for LIVECell.
    "bbbc010": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_bbbc010",
        [("00010", 6, None), ("00003", 6, None), ("00011", 4, None), ("00010", 4, None), ("00004", 4, None)],
    ),
    "cisd": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cisd",
        [("00009", 3, None), ("00018", 3, None)],
    ),
    "deepseas": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_deepseas",
        [
            ("00022", 16, None), ("00022", 8, None), ("00022", 1, None), ("00006", 1, None), ("00023", 3, None),
            ("00009", 3, None), ("00006", 3, None)
        ],
    ),
    "mcellseg": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_mcellseg",
        [
            ("00004", 17, None), ("00006", 17, None), ("00013", 17, None), ("00006", 19, None), ("00013", 19, None),
            ("00006", 9, None), ("00010", 9, None), ("00010", 3, None), ("00010", 6, None), ("00013", 7, None)
        ],
    ),
    "toiam": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_toiam",
        [("00006", 40, None), ("00009", 141, None), ("00008", 73, None), ("00009", 73, None)],
    ),
    "yeastsam": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_yeastsam",
        [("00010", 19, None), ("00018", 19, None), ("00010", 16, None)],
    ),
    # The supplementary datasets: crowded images, as for LIVECell.
    "bac_mother": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_bac_mother",
        [("00018", 2, None), ("00015", 5, None)],
    ),
    "bbbc030": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_bbbc030",
        [("00004", 4, None), ("00009", 1, None)],
    ),
    "bccd": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_bccd",
        [("00002", 14, None), ("00013", 14, None), ("00007", 19, None)],
    ),
    "bitdepth_nucseg": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_bitdepth_nucseg",
        [("00004", 2, None), ("00004", 4, None), ("00000", 4, None), ("00000", 8, None)],
    ),
    "cellapp": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cellapp",
        [
            ("00013", 108, None), ("00013", 58, None), ("00014", 41, None), ("00014", 52, None), ("00010", 99, None),
            ("00010", 82, None)
        ],
    ),
    "cellbindb_he": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cellbindb_he",
        [("00001", 46, None), ("00001", 59, None), ("00013", 29, None), ("00003", 34, None), ("00003", 168, None)],
    ),
    "cellular": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cellular",
        [
            ("00000", 16, None), ("00000", 3, None), ("00000", 8, None), ("00001", 8, None), ("00000", 12, None),
            ("00001", 17, None), ("00001", 4, None), ("00001", 11, None)
        ],
    ),
    "cvz_fluo_dapi": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_cvz_fluo_dapi",
        [("00006", 182, None), ("00002", 76, None), ("00002", 170, None), ("00003", 169, None), ("00003", 60, None)],
    ),
    "deepbacs_label_free": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_deepbacs_label_free",
        [("00019", 32, None), ("00019", 13, None), ("00008", 13, None), ("00019", 28, None), ("00017", 3, None)],
    ),
    "deepliif": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_deepliif",
        [("00009", 58, None), ("00005", 19, None)],
    ),
    "dememseg": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_dememseg",
        [("00014", 2, None), ("00000", 3, None), ("00006", 3, None), ("00005", 3, None)],
    ),
    "ecoli_microcolony_lineage": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_ecoli_microcolony_lineage",
        [("00014", 4, None), ("00011", 20, None), ("00004", 41, None)],
    ),
    "lizard": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_lizard",
        [("00004", 73, None), ("00023", 185, None), ("00001", 437, None), ("00001", 68, None)],
    ),
    "lynsec_ihc": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_lynsec_ihc",
        [
            ("00004", 101, None), ("00017", 206, None), ("00019", 246, None), ("00019", 125, None),
            ("00001", 117, None), ("00001", 212, None), ("00001", 109, None)
        ],
    ),
    "micro_bench": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_micro_bench",
        [("00010", 26, None)],
    ),
    "microbeseg": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_microbeseg",
        [("00002", 52, None), ("00000", 14, None)],
    ),
    # More supplementary datasets: crowded images, as for LIVECell (YeaZ: picked by the largest lead).
    "omnipose_worm": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_omnipose_worm",
        [("00019", 6, None), ("00010", 8, None)],
    ),
    "organoid": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_organoid",
        [("00003", 8, None), ("00004", 8, None)],
    ),
    "organoidnet": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_organoidnet",
        [("00006", 51, None), ("00005", 201, None), ("00006", 103, None), ("00005", 103, None), ("00007", 160, None)],
    ),
    "orgasegment": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_orgasegment",
        [("00005", 3, None), ("00009", 19, None)],
    ),
    "pan_multiplex": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_pan_multiplex",
        [("00006", 79, None), ("00006", 21, None), ("00006", 97, None), ("00019", 377, None)],
    ),
    "panoptils": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_panoptils",
        [("00004", 26, None)],
    ),
    "tnbc_celltype": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_tnbc_celltype",
        [("00013", 27, None), ("00002", 28, None), ("00002", 35, None), ("00012", 33, None)],
    ),
    "tsakiroglou": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_tsakiroglou",
        [("00002", 175, None), ("00001", 175, None)],
    ),
    "yeaz": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_yeaz",
        [("00017", 132, None), ("00023", 59, None), ("00022", 38, None)],
    ),
    # mnDINO: micronuclei, the small objects of the images ('run_posthoc_scripts.py select_candidates --small').
    "mndino": (
        "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/"
        "qualitative_single_prompt_mndino",
        [("00017", 78, None), ("00019", 97, None)],
    ),
}
METHODS = ["SAM", "SAM2", "SAM3", "micro-sam", "micro-sam2"]
PROMPT_GROUPS = {"point": "Point", "box": "Box"}
# The crops are drawn as in the interactive qualitative plot of Figure 2.
FILL_ALPHA = fig2.ZOOM_FILL_ALPHA
REFERENCE_COLOR = fig2.REFERENCE_COLOR
REFERENCE_WIDTH = fig2.ZOOM_REFERENCE_WIDTH

# The layout, in inches.
LABEL_WIDTH = 1.0  # the captions of the images left of the crops
CROP_GAP = 0.025  # between the crops of a block
PROMPT_GAP = 0.08  # between the point and the box columns
BLOCK_GAP = 0.1  # between the blocks of the images
HEADER_HEIGHT = 0.3  # the prompt and method names over the crops


def prediction_folder(dataset, prompt, method):
    """Return the folder of the npz files of a method with the first prompt."""
    root = PREDICTIONS
    if method == "SAM":
        return f"{root}/sam/predictions/sam_vit_l/{dataset}/{prompt}_without_masks_center/predictions/iteration00"
    if method == "SAM2":
        folder = glob.glob(f"{root}/sam2/predictions/*hvit_l/{dataset}/{prompt}_with_masks_center")[0]
        return f"{folder}/start_with_{prompt}/with_masks/iteration00"
    if method == "SAM3":
        return f"{root}/sam3/{dataset}/{prompt}"
    return folders(dataset, prompt)[method]


def load(dataset, image_name, root=None):
    """Return the image, the labels and the prompt data of all methods and both prompts of one chosen image.

    'root' is the folder of the single-prompt runs, by default the one of the candidates.
    """
    global PREDICTIONS
    if root is not None:
        default, PREDICTIONS = PREDICTIONS, root
        try:
            return load(dataset, image_name)
        finally:
            PREDICTIONS = default
    inputs = folders(dataset, "point")["inputs"]
    labels = tifffile.imread(f"{inputs}/labels/{image_name}.tif")
    image = as_rgb(tifffile.imread(f"{inputs}/images/{image_name}.tif"), labels.shape)
    # The npz files are read once, as every access to an array of an open npz file decompresses it again.
    data = {
        (prompt, method): dict(np.load(f"{prediction_folder(dataset, prompt, method)}/{image_name}.npz"))
        for prompt in PROMPTS for method in METHODS
    }
    return image, labels, data


def panel(image, mask, color, prompt, data, index):
    """Return a crop panel: the mask of the object filled in its color, and its prompt drawn."""
    out = image.copy()
    out[mask] = (1 - FILL_ALPHA) * out[mask] + FILL_ALPHA * np.array(colors.to_rgb(color))
    return overlay_prompt(out, prompt, data, index)


def columns():
    """Return the (prompt, method) of each column and its left edge, after the captions."""
    lefts, keys, left = [], [], LABEL_WIDTH
    for k, prompt in enumerate(PROMPT_GROUPS):
        left += PROMPT_GAP * (k > 0)
        for method in METHODS:
            keys.append((prompt, method))
            lefts.append(left)
            left += size_of() + CROP_GAP
    return keys, lefts


def size_of():
    """Return the side of a crop, so that the columns fill the figure width."""
    n = len(PROMPT_GROUPS) * len(METHODS)
    return (fig2.FIGURE_WIDTH - LABEL_WIDTH - (n - 1) * CROP_GAP - PROMPT_GAP) / n


def plot_crops(chosen_images=CHOSEN, objects_of=None, tag_objects=False):
    """Plot a block of object crops per chosen image, with the methods and prompts in columns.

    'objects_of' maps (dataset, image) to the objects to show, as records with 'object', and optionally 'image',
    'root' (see 'load') and 'specimen', which is then written on the object's first crop. By default, the objects
    are the final ones of the dataset (FINALS), or else the three objects chosen for the image in the candidates.
    With 'tag_objects', every row carries its object id, to choose from candidates.
    """
    if objects_of is None:
        picked = pd.read_csv(CANDIDATES, dtype={"image": str})
        objects_of = {}
        for dataset, image_name, *_ in chosen_images:
            if dataset in FINALS:
                root, finals = FINALS[dataset]
                objects_of[dataset, image_name] = [
                    dict(image=image, object=object_id, specimen=specimen, root=root)
                    for image, object_id, specimen in finals
                ]
                continue
            chosen = picked[(picked.dataset == dataset) & (picked.image == image_name)]
            objects_of[dataset, image_name] = chosen.sort_values("gap", ascending=False).to_dict("records")
            assert len(objects_of[dataset, image_name]) == N_OBJECTS, (dataset, image_name)
    size, keys, lefts = size_of(), *columns()

    def block_of(n_objects):
        return n_objects * size + (n_objects - 1) * CROP_GAP

    blocks = [block_of(len(objects_of[dataset, image_name])) for dataset, image_name, *_ in chosen_images]
    height = HEADER_HEIGHT + sum(blocks) + (len(chosen_images) - 1) * BLOCK_GAP
    fig = plt.figure(figsize=(fig2.FIGURE_WIDTH, height))

    block_top = height - HEADER_HEIGHT
    for b, (dataset, image_name, modality, domain, imaging, specimen) in enumerate(chosen_images):
        chosen, block = objects_of[dataset, image_name], blocks[b]
        # The data is kept for one block only: the masks of all objects of a dense image take many GB.
        loaded = {}
        for row, obj in enumerate(chosen):
            key = (obj.get("root"), obj.get("image", image_name))
            if key not in loaded:
                loaded[key] = load(dataset, key[1], key[0])
            image, labels, data = loaded[key]
            # The index of the object in the masks: the masks are in the order of the object ids.
            index = int(np.searchsorted(np.setdiff1d(np.unique(labels), [0]), obj["object"]))
            color = OBJECT_COLORS[row % len(OBJECT_COLORS)]
            reference = labels == obj["object"]
            top, left, side = fig2.crop_of(reference, fig2.ZOOM_CROP_SCALE)
            window = np.s_[top:top + side, left:left + side]
            bottom = block_top - row * (size + CROP_GAP) - size
            for (prompt, method), x in zip(keys, lefts):
                mask = data[prompt, method]["masks"][index]
                ax = fig.add_axes([x / fig2.FIGURE_WIDTH, bottom / height, size / fig2.FIGURE_WIDTH, size / height])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(fig2.FRAME_COLOR)
                    spine.set_linewidth(fig2.FRAME_WIDTH)
                ax.imshow(panel(image, mask, color, prompt, data[prompt, method], index)[window], interpolation="none")
                if x == lefts[0] and obj.get("specimen"):
                    text = ax.text(
                        0.04, 0.96, obj["specimen"], transform=ax.transAxes, ha="left", va="top",
                        fontsize=5, color="white"
                    )
                    text.set_path_effects([patheffects.withStroke(linewidth=1.0, foreground="black")])
                contour = ax.contour(
                    reference[window], levels=[0.5], colors=REFERENCE_COLOR, linewidths=REFERENCE_WIDTH
                )
                contour.set(path_effects=[patheffects.withStroke(linewidth=2 * REFERENCE_WIDTH, foreground="black")])
                iou = np.logical_and(mask, reference).sum() / np.logical_or(mask, reference).sum()
                fig2.draw_score(ax, iou, method == "micro-sam2")
            if tag_objects:
                fig.text(
                    (LABEL_WIDTH - 0.04) / fig2.FIGURE_WIDTH, (bottom + size / 2) / height, f"#{obj['object']}",
                    ha="right", va="center", fontsize=5, color=fig2.GROUP_COLOR
                )

        # The caption of the image left of its block, as in the automatic figure: the dataset over the imaging and
        # the specimen, the dataset name in italics in-domain and in bold out-of-domain.
        in_domain = domain == "ID"
        y = (block_top - block / 2) / height
        x = (LABEL_WIDTH - (0.3 if tag_objects else 0.06)) / fig2.FIGURE_WIDTH
        details = imaging if specimen is None else f"{imaging}\n{specimen}"
        fig.text(x, y, details, ha="right", va="top", fontsize=5.5, color=TEXT_COLOR, linespacing=1.15)
        fig.text(
            x, y + 0.02 / height, DATASET_NAMES.get(dataset, dataset), ha="right", va="bottom",
            fontsize=5.5, color=TEXT_COLOR, style="italic" if in_domain else "normal",
            fontweight="normal" if in_domain else "bold"
        )
        block_top -= block + BLOCK_GAP

    # Over the crops: the method names, and the methods of each prompt grouped under a line with the prompt name.
    names = dict(zip(METHODS, fig2.names_of(METHODS, set())))
    for (prompt, method), x in zip(keys, lefts):
        fig.text(
            (x + size / 2) / fig2.FIGURE_WIDTH, (height - HEADER_HEIGHT + 0.03) / height, names[method],
            ha="center", va="bottom", fontsize=6, color=TEXT_COLOR,
            fontweight="bold" if method == "micro-sam2" else "normal"
        )
    line = (height - 0.13) / height
    for prompt, group_name in PROMPT_GROUPS.items():
        xs = [x for (p, _), x in zip(keys, lefts) if p == prompt]
        x0, x1 = xs[0] / fig2.FIGURE_WIDTH, (xs[-1] + size) / fig2.FIGURE_WIDTH
        fig.add_artist(plt.Line2D([x0, x1], [line, line], color=fig2.GROUP_COLOR, linewidth=0.6))
        fig.text(
            (x0 + x1) / 2, line + 0.02 / height, group_name, ha="center", va="bottom", fontsize=6,
            color=fig2.GROUP_COLOR
        )
    return fig


def root_of(dataset):
    """Return the folder of the single-prompt runs of a dataset."""
    return f"{RUNS}/qualitative_single_prompt_{dataset}"


def domain_of(dataset):
    """Return the domain ('ID', 'OOD' or 'supp') and the modality (or None) of a dataset in the main figure."""
    for modality, domains in fig2.PANEL_DATASETS.items():
        for domain, datasets in domains.items():
            if dataset in datasets:
                return domain, modality
    return "supp", None


def output_of(dataset):
    """Return the folder of the candidates of a dataset."""
    domain, modality = domain_of(dataset)
    if modality is None:
        return f"{OUTPUT}/supp/{dataset}"
    return f"{OUTPUT}/{domain.lower()}/{MODALITY_FOLDERS[modality]}/{dataset}"


def tissuenet_groups(raw_paths):
    """Return the tissue of the TissueNet test images, from the metadata of the original test archive."""
    archive = "/mnt/vast-nhr/projects/cidas/cca/data/tissuenet/tissuenet_v1.1.zip"
    with zipfile.ZipFile(archive) as zf, zf.open("tissuenet_v1.1_test.npz") as f:
        meta = np.load(f, allow_pickle=True)["meta"]
    tissues = [str(row[5]) for row in meta if row[0] != "filename"]
    # The test zarrs are named by their index in the archive.
    return [tissues[int(os.path.basename(path).split("_")[1].split(".")[0])] for path in raw_paths]


def select_images(dataset):
    """Pick the crowded images of a dataset and write them, with their groups, to its output folder."""
    sys.path.insert(0, EVALUATION)
    from common import DATA_ROOT, get_data_paths, load_evaluation_sample_2d, sorted_path_pairs

    raw_paths, label_paths, raw_key, label_key = get_data_paths(dataset, DATA_ROOT, split="test")
    rows = []
    for k, (raw_path, label_path) in enumerate(sorted_path_pairs(raw_paths, label_paths)):
        _, labels = load_evaluation_sample_2d(raw_path, label_path, raw_key, label_key, dataset, normalize=False)
        labels = np.asarray(labels)
        n_objects = len(np.unique(labels)) - 1
        if n_objects == 0:
            continue
        # The objects that touch another object: grown by 2 pixels, an object takes the larger id where it touches.
        grown = ndimage.grey_dilation(labels, size=5)
        contact = (grown != labels) & (labels > 0)
        touching = len(np.union1d(np.unique(labels[contact]), np.unique(grown[contact])))
        rows.append(dict(k=k, path=str(raw_path), n_objects=n_objects, touching=touching))
    images = pd.DataFrame(rows)
    images["group"] = tissuenet_groups(images.path) if dataset == "tissuenet" else ""
    if len(images) > N_IMAGES and images.group.nunique() > 1:
        # The most crowded images of each group in turn, so that every group is shown.
        images["rank"] = images.groupby("group").touching.rank(ascending=False, method="first")
        images = images.sort_values(["rank", "touching"], ascending=[True, False]).iloc[:N_IMAGES]
    elif len(images) > N_IMAGES:
        small = images[images.n_objects <= MAX_OBJECTS]
        ranked = (small if len(small) >= N_IMAGES else images).sort_values("touching", ascending=False)
        rest = ranked.iloc[N_CROWDED:]
        spread = rest.iloc[np.linspace(0, len(rest) - 1, N_IMAGES - N_CROWDED).round().astype(int)]
        images = pd.concat([ranked.iloc[:N_CROWDED], spread]).drop_duplicates("k")
    images = images.sort_values("k")
    groups = list(images.group)

    output = output_of(dataset)
    os.makedirs(output, exist_ok=True)
    with open(f"{output}/images.txt", "w") as f:
        f.write("\n".join(images.path) + "\n")
    pd.DataFrame({"name": images.path, "group": groups}).to_csv(f"{output}/groups.csv", index=False)
    print(
        f"Picked {len(images)} images of '{dataset}' with {images.n_objects.median():.0f} objects and "
        f"{images.touching.median():.0f} touching ones in the median."
    )


def groups_of(dataset):
    """Return the image names and the cell type or tissue of each."""
    output = output_of(dataset)
    with open(f"{output}/images.txt") as f:
        names = f.read().split("\n")[:-1]
    path = f"{output}/groups.csv"
    if os.path.exists(path):
        groups = dict(pd.read_csv(path, keep_default_na=False).values)
        return names, [groups[name] for name in names]
    return names, [os.path.basename(name).split("_")[0] for name in names]


def score(dataset):
    """Return the scores of the objects and of the images."""
    rows = []
    for k, (name, group) in enumerate(zip(*groups_of(dataset))):
        image_name = f"{k:05d}"
        _, labels, data = load(dataset, image_name)
        masks = {key: value["masks"] for key, value in data.items()}
        ids = np.setdiff1d(np.unique(labels), [0])
        # The neighbours of an object: the objects that its mask grown by 2 pixels touches.
        grown = ndimage.grey_dilation(labels, size=5)
        for index, object_id in enumerate(ids):
            reference = labels == object_id
            ys, xs = np.nonzero(reference)
            ring = ndimage.binary_dilation(reference, iterations=2) & ~reference
            row = dict(
                image=image_name, name=os.path.basename(name), group=group, object=int(object_id), index=index,
                area=int(reference.sum()), cy=ys.mean(), cx=xs.mean(), height=labels.shape[0], width=labels.shape[1],
                neighbours=len(np.setdiff1d(
                    np.unique(np.concatenate([labels[ring], grown[reference]])), [0, object_id]
                )),
            )
            for (prompt, method), mask in masks.items():
                mask = mask[index]
                union = np.logical_or(mask, reference).sum()
                row[f"{method}_{prompt}"] = np.logical_and(mask, reference).sum() / union if union else 0.0
            rows.append(row)
    objects = pd.DataFrame(rows)
    for prompt in PROMPTS:
        best_other = objects[[f"{method}_{prompt}" for method in OTHERS]].max(axis=1)
        objects[f"lead_{prompt}"] = objects[f"micro-sam2_{prompt}"] - best_other
    objects["best_both"] = (objects.lead_point > 0) & (objects.lead_box > 0)

    columns = [f"{method}_{prompt}" for prompt in PROMPTS for method in METHODS]
    keys = ["image", "name", "group"]
    images = objects.groupby(keys, dropna=False)[columns + ["neighbours", "best_both"]].mean()
    images["n_objects"] = objects.groupby(keys, dropna=False).size()
    for prompt in PROMPTS:
        images[f"lead_{prompt}"] = images[f"micro-sam2_{prompt}"] - images[
            [f"{method}_{prompt}" for method in OTHERS]].max(axis=1)
    images["lead"] = images[["lead_point", "lead_box"]].min(axis=1)
    images = images.reset_index().sort_values("lead", ascending=False)
    return objects, images


def plot_sheet(dataset, objects, images):
    """Plot the images, best first: the objects where micro-sam2 is best with both prompts outlined."""
    names = dict(zip(METHODS, fig2.names_of(METHODS, set())))
    n_rows = int(np.ceil(len(images) / N_COLUMNS))
    fig, axes = plt.subplots(n_rows, N_COLUMNS, figsize=(fig2.FIGURE_WIDTH * 1.4, 2.35 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    inputs = folders(dataset, "point")["inputs"]
    for ax, image_row in zip(axes.flat, images.to_dict("records")):
        labels = tifffile.imread(f"{inputs}/labels/{image_row['image']}.tif")
        image = as_rgb(tifffile.imread(f"{inputs}/images/{image_row['image']}.tif"), labels.shape)
        best = objects[(objects.image == image_row["image"]) & objects.best_both].object.values
        image[find_boundaries(np.isin(labels, best) * labels, mode="inner")] = OUTLINE_COLOR
        ax.imshow(image, interpolation="none")
        scores = "\n".join(
            f"{prompt:>5}: " + "  ".join(
                f"{names[method]} {image_row[f'{method}_{prompt}']:.2f}" for method in METHODS
            )
            for prompt in PROMPTS
        )
        ax.set_title(
            f"{image_row['image']}  {image_row['group']}  {image_row['name'][:30]}\n{image_row['n_objects']} objects, "
            f"{image_row['neighbours']:.1f} neighbours, uSAM2 best for {image_row['best_both']:.0%} "
            f"(lead {image_row['lead_point']:+.2f} / {image_row['lead_box']:+.2f})\n{scores}",
            fontsize=4.5, color=TEXT_COLOR, loc="left", family="monospace",
        )
    fig.tight_layout()
    return fig


def pick_crops(objects, image_names, small=False):
    """Return the candidate objects of each image.

    With 'small', the objects are the small ones of each image (as micronuclei) instead of the large ones.
    """
    objects = objects.copy()
    objects["median_area"] = objects.groupby("image").area.transform("median")
    if small:
        size = objects.area <= SMALL_FRACTION * objects.median_area
    else:
        size = objects.area >= objects.median_area
    border = np.minimum(BORDER, BORDER_FRACTION * np.minimum(objects.height, objects.width))
    good = objects[
        (objects.lead_point >= -TIE) & (objects.lead_box >= -TIE) & size
        & (objects["micro-sam2_point"].round(2) > objects["micro-sam_point"].round(2))
        & (objects["micro-sam2_box"].round(2) > objects["micro-sam_box"].round(2))
        & (objects.cy >= border) & (objects.cy <= objects.height - border)
        & (objects.cx >= border) & (objects.cx <= objects.width - border)
    ].copy()
    # The generalists that were run, or micro-sam alone for the datasets run with '--microsam_only'.
    rivals = [method for method in ["SAM", "SAM2", "SAM3"] if method in METHODS] or ["micro-sam"]
    generalists = good[[f"{method}_point" for method in rivals]].mean(axis=1)
    good["generalist_gap"] = good["micro-sam2_point"] - generalists
    objects_of = {}
    for image_name in image_names:
        picked = []
        for obj in good[good.image == image_name].sort_values("generalist_gap", ascending=False).to_dict("records"):
            if all(np.hypot(obj["cy"] - other["cy"], obj["cx"] - other["cx"]) >= SPACING for other in picked):
                picked.append(obj)
            if len(picked) == N_CANDIDATES:
                break
        objects_of[image_name] = picked
    return objects_of


def select_candidates(argv):
    """Show crowded images of a dataset as candidates for the qualitative interactive figure.

    Usage: python run_posthoc_scripts.py select_candidates --dataset <name> [--select] [--rescore] [--small]
    [--microsam_only]

    With '--select', this picks the images: up to N_IMAGES test images with objects, the most crowded first (the most
    objects that touch another object, among images with at most MAX_OBJECTS objects), and a few more spread over the
    others; for a dataset with groups, the most crowded images of each group in turn. It writes their paths, in the
    order of the test split, to '<output>/images.txt', and the cell type or tissue of each image, where the dataset has
    one, to 'groups.csv' next to it. The output folder is 'candidates/<domain>/<modality>/<dataset>' for the datasets of
    the main figure, and 'candidates/supp/<dataset>' else.

    The predictions of all methods with the first prompt are made by 'run_posthoc_scripts.py first_prompt' with
    SINGLE_PROMPT_IMAGES set to these images, and for SAM3 by 'run_posthoc_scripts.py first_prompt_sam3 --root <root>
    --dataset <dataset>', with the root of 'root_of'.

    Without '--select', this scores every object with every method and prompt, and counts its touching neighbours. It
    writes the scores of the objects ('objects.csv') and of the images ('images.csv'), ranked by how much micro-sam2 is
    ahead of the best other method, and a sheet of the images ('candidates.png'): each image with the objects where
    micro-sam2 is better than all other methods with both prompts outlined, and the mean IoU of every method per prompt.

    For the shortlisted images (SHORTLISTS, or else the N_SHORTLIST images with the most candidates), it then shows up
    to N_CANDIDATES objects each as crops with all methods and both prompts ('crops.png'): large objects away from the
    border, where micro-sam2 is better than micro-sam (also in the two decimals the figure shows) and the best or within
    TIE of the best with both prompts, ranked by how much better micro-sam2 is than SAM, SAM2 and SAM3 with a point. The
    scores are computed once; pass '--rescore' to compute them again. With '--microsam_only', only micro-sam and
    micro-sam2 are run and compared, and the objects are ranked by how much better micro-sam2 is than micro-sam.

    The objects picked by hand from the crops of each dataset are in FINALS. The objects of the interactive qualitative
    plot of Figure 2 come from them, see INTERACTIVE_QUALITATIVE in 'plots_for_figure2.py'.
    """
    parser = argparse.ArgumentParser(prog="run_posthoc_scripts.py select_candidates")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--select", action="store_true", help="Pick the images of the dataset.")
    parser.add_argument("--rescore", action="store_true", help="Score the objects again.")
    parser.add_argument("--small", action="store_true", help="Show small objects ('crops_small.png').")
    parser.add_argument(
        "--microsam_only", action="store_true",
        help="Compare micro-sam2 with micro-sam only, for datasets without SAM, SAM2 and SAM3 runs."
    )
    args = parser.parse_args(argv)
    if args.microsam_only:
        global METHODS, OTHERS
        METHODS, OTHERS = ["micro-sam", "micro-sam2"], ["micro-sam"]
    dataset, output = args.dataset, output_of(args.dataset)
    if args.select:
        select_images(dataset)
        return

    global PREDICTIONS
    set_style()
    PREDICTIONS = root_of(dataset)
    if args.rescore or not os.path.exists(f"{output}/objects.csv"):
        objects, images = score(dataset)
        objects.to_csv(f"{output}/objects.csv", index=False)
        images.to_csv(f"{output}/images.csv", index=False)
        print(
            images[["image", "group", "n_objects", "neighbours", "best_both", "lead_point", "lead_box"]].round(2)
            .to_string(index=False)
        )
        fig = plot_sheet(dataset, objects, images)
        fig.savefig(f"{output}/candidates.png", dpi=250)
        plt.close(fig)
    objects = pd.read_csv(f"{output}/objects.csv", dtype={"image": str}, keep_default_na=False)
    images = pd.read_csv(f"{output}/images.csv", dtype={"image": str}, keep_default_na=False)

    shortlist = SHORTLISTS.get(dataset)
    if shortlist is None:
        # The images with the most candidates, the best images first among those with as many.
        picks = pick_crops(objects, images.image, args.small)
        order = sorted(images.image, key=lambda image_name: -min(len(picks[image_name]), N_CANDIDATES))
        groups = dict(zip(images.image, images.group))
        shortlist = [(image_name, groups[image_name]) for image_name in order[:N_SHORTLIST] if picks[image_name]]
    if not shortlist:
        print(f"No candidates for '{dataset}'.")
        return
    picks = pick_crops(objects, [image_name for image_name, _ in shortlist], args.small)
    domain, modality = domain_of(dataset)
    chosen_images = [(
        dataset, image_name, modality, domain, IMAGING.get(dataset, f"Image {image_name}"), specimen or None
    ) for image_name, specimen in shortlist]
    objects_of = {(dataset, image_name): picks[image_name] for image_name, _ in shortlist}
    fig = plot_crops(chosen_images, objects_of, tag_objects=True)
    fig.savefig(f"{output}/{'crops_small' if args.small else 'crops'}.png", dpi=300)
    print(f"Drew the candidates of '{dataset}' in '{output}'.")


COMMANDS = {
    "predict_samples": lambda args: predict_samples(args[0], args[1], args[2:]),
    "keep_predictions": lambda args: keep_predictions(args[0], args[1:]),
    "first_prompt": lambda args: first_prompt(args[0], args[1:]),
    "first_prompt_sam3": first_prompt_sam3,
    "rank_candidates": lambda args: rank_candidates(rescore="--rescore" in args),
    "select_candidates": select_candidates,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        sys.exit(__doc__)
    COMMANDS[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
