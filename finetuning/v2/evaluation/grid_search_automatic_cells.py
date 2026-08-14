"""Grid search over the automatic-segmentation postprocessing parameters.

Loads a finetuned SAM2 model (the 'hvit_t_cells' registry model, a jointly finetuned model selected
with '-m', or a checkpoint given with '-c'), runs the UniSAM2 decoder once per image, then sweeps the
postprocessing hyperparameters and averages the mean segmentation accuracy (mSA) over the images.

With '-d' the sweep runs on a single dataset, over the same samples the evaluation scores, and writes
'<output_dir>/<dataset>.csv'. EM datasets are tuned in dense (multicut) mode and ranked by the CREMI
score, all others in sparse (flow) mode and ranked by mSA.

Alternatively '--track' sweeps one of the predefined dataset groups:

    lm_cell     LM cell segmentation, sparse (flow) mode, 2d (livecell).
    lm_nucleus  LM nucleus segmentation, sparse (flow) mode, 2d (dsb).
    em_neurons  EM neuron segmentation, dense (multicut) mode, 3d (cremi + snemi + humanneurons).

For each track the best parameter combination is written to '<output_dir>/<track>.csv' together with
the full sweep, and the best row is printed.

'--tune apg' sweeps the automatic prompt generation instead of the postprocessing. Every combination
re-prompts SAM2, so that grid is much smaller and needs both halves of the joint model.

'--split val' tunes on data held out from the samples the evaluation scores, which is what keeps the
evaluation honest. See 'common.VAL_SPLITS' for what counts as held out per dataset.

Usage:
    python grid_search_automatic_cells.py -d livecell -m hvit_b -o /path/to/results
    python grid_search_automatic_cells.py -d livecell -m hvit_b --split val --tune apg -o /path/to/apg
    python grid_search_automatic_cells.py --track lm_cell
    python grid_search_automatic_cells.py --track em_neurons --crop_3d 16 512 512
"""

import os
import sys
import time
import argparse
import warnings
import itertools
import threading
from concurrent import futures

import numpy as np
import pandas as pd
import imageio.v3 as imageio
from tqdm import tqdm

from elf.io import open_file
from elf.evaluation import mean_segmentation_accuracy
from bioimage_cpp.segmentation import label as connected_components, watershed

from micro_sam.v2.postprocessing import (
    flow_instance_segmentation, run_multicut, watershed_heightmap, _compute_flow_density
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_EM, DATASET_SPACING, VAL_Z_RANGE,
    get_data_paths, load_volume, export_joint_checkpoint, drop_excluded_livecell,
)
from baselines_common import MAX_EVALUATION_SAMPLES  # noqa


OUTPUT_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/experiments/grid-search-hvit-t-cells"
DEVICE = "cuda"
MODEL_NAME = "hvit_t_cells"

CROP_SHAPE_2D = (512, 512)

# Sparse (flow) grid for LM data. Keys map to flow_instance_segmentation arguments.
LM_GRID = {
    "foreground_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
    "density_threshold": [5.0, 10.0, 20.0],
    "min_size": [10, 25, 50, 100, 200, 500],
    "sigma": [0.25, 0.5, 1.0, 2.0],
    "n_iter": [50, 100, 200],
    "dt": [0.25, 0.5, 1.0],
    "foreground_weight": [0.0, 0.25, 0.5, 0.65, 0.75, 0.9],
}

# Keys map to AutomaticPromptGenerator.generate. Every combination re-prompts SAM2, unlike the grids
# above which only re-run CPU postprocessing, so this one stays small.
APG_GRID_2D = {
    "candidate_threshold": [1.0, 1.5, 2.5],
    "sigma": [0.5, 1.0, 2.0],
    "score_threshold": [0.5, 0.6, 0.7],
    "min_size": [50, 100],
}

# Drops a level of 'score_threshold', since a volume pays for propagation on top of scoring.
APG_GRID_3D = {
    "candidate_threshold": [(1.5, 10.0), (1.0, 5.0), (2.5, 10.0)],
    "sigma": [1.0, 2.0],
    "score_threshold": [0.5, 0.6],
    "min_size": [50, 100],
}

# Dense (multicut) grid for EM data. Keys map to run_multicut arguments.
EM_GRID = {
    "beta": [0.5, 0.6, 0.7, 0.8],
    "density_threshold": [3.0, 5.0, 10.0],
    "sigma": [0.5, 1.0, 2.0],
    "n_iter": [25, 50],
}

# Sparse-flow parameters that determine the (expensive) convergence-density map. Combos that share
# these reuse a cached density. Only 'density_threshold', 'foreground_weight' and 'min_size' (the cheap
# seed + watershed + size-filter steps) then vary on top. See score_image_sparse_cached.
FLOW_DENSITY_KEYS = ("foreground_threshold", "sigma", "n_iter", "dt")

# Dense-multicut parameters that determine the (expensive) slice-wise oversegmentation and RAG.
# Combos that share these reuse a cached oversegmentation. Only 'beta' (the cheap edge-cost +
# multicut-solve step) then varies on top. See score_image_dense_cached.
OVERSEG_KEYS = ("density_threshold", "sigma", "n_iter", "dt")

# Metric used to rank each track's grid, and whether lower is better. 'msa' (mean segmentation accuracy)
# is maximised; 'cremi' (the CREMI score, a VI + adapted-Rand combination for neuron segmentation) is
# minimised.
CRITERION_ASCENDING = {"msa": False, "cremi": True}

# Each track: the datasets to evaluate, the postprocessing mode, the spatial dimensionality, the grid,
# an optional anisotropic voxel 'spacing' for 3d flow smoothing (matching common._DATASET_SPACING), the
# 3d center-crop 'crop' (None for 2d), and the 'criterion' used to pick the best combo. Crops match the
# micro-sam v2 eval (evaluate_3d uses (8,512,512) for LM, and the EM neuron grid search uses (32,512,512)).
TRACKS = {
    "lm_cell": {
        "datasets": ["livecell"], "mode": "sparse", "ndim": 2, "grid": LM_GRID,
        "spacing": None, "crop": None, "criterion": "msa",
    },
    "lm_nucleus": {
        "datasets": ["dsb"], "mode": "sparse", "ndim": 2, "grid": LM_GRID,
        "spacing": None, "crop": None, "criterion": "msa",
    },
    "skull_nuclei": {
        "datasets": ["embedseg"], "mode": "sparse", "ndim": 3, "grid": LM_GRID,
        "spacing": (4, 1, 1), "crop": (8, 512, 512), "criterion": "msa",
    },
    "em_neurons": {
        "datasets": ["cremi", "snemi", "humanneurons"], "mode": "dense", "ndim": 3, "grid": EM_GRID,
        "spacing": None, "crop": (32, 512, 512), "criterion": "cremi",
    },
}

POSTPROC_THREADS = 4

# The 3d center crop for the per-dataset mode. Matches the crop the evaluation uses, so the tuned
# parameters transfer directly.
CROP_SHAPE_3D = (8, 512, 512)


def dataset_config(dataset_name, criterion=None, split="test", tune="ais"):
    """Build the grid-search config for a single dataset.

    EM datasets are tuned in dense (multicut) mode and ranked by the CREMI score, all others in
    sparse (flow) mode and ranked by mSA. With tune='apg' the postprocessing is replaced by automatic
    prompt generation, which is swept over its own grid; the ranking metric is unchanged, so an APG
    result is directly comparable with the AIS result of the same dataset.

    Args:
        dataset_name: The dataset to tune on.
        criterion: The metric to rank the grid by. Defaults to the mode-specific choice.
        split: The split to tune on. 'val' keeps the tuning off the samples the evaluation scores.
        tune: What is tuned, 'ais' (the postprocessing) or 'apg' (the prompt generation).

    Returns:
        The config dict, in the same shape as an entry of TRACKS.
    """
    is_em = dataset_name in DATASETS_3D_EM
    is_3d = dataset_name in DATASETS_3D
    if criterion is None:
        criterion = "cremi" if is_em else "msa"

    if tune == "apg":
        mode, grid = "apg", (APG_GRID_3D if is_3d else APG_GRID_2D)
    else:
        mode, grid = ("dense", EM_GRID) if is_em else ("sparse", LM_GRID)

    return {
        "datasets": [dataset_name],
        "mode": mode,
        # Which metrics compute_metrics reports. Follows the data, not the mode, so that an APG run on
        # EM is still ranked by the CREMI score.
        "metric_mode": "dense" if is_em else "sparse",
        "ndim": 3 if is_3d else 2,
        "grid": grid,
        "spacing": DATASET_SPACING.get(dataset_name, None),
        "crop": CROP_SHAPE_3D if is_3d else None,
        "criterion": criterion,
        "match_evaluation": split == "test",
        "split": split,
        # Only set for the EM volumes on the val split, which have no split of their own.
        "z_range": VAL_Z_RANGE.get(dataset_name) if split == "val" else None,
    }


def load_model(device, checkpoint_path=None, model_type=None, model_name=MODEL_NAME, joint_checkpoint="best"):
    """Load the UniSAM2 model to tune the postprocessing for.

    Mirrors the annotator's decoder-loading path: the decoder is loaded via get_unisam2_model, which
    rebuilds the matching SAM2 encoder and (re)defines all weights from the checkpoint. The decoder
    comes from an explicit checkpoint, from the jointly finetuned model for 'model_type', or from the
    micro-sam v2 download registry.

    Args:
        device: The device to load the model onto.
        checkpoint_path: Optional path to a custom UniSAM2 / joint checkpoint.
        model_type: The SAM2 backbone, e.g. 'hvit_b'. Selects the jointly finetuned model when no
            checkpoint is given, and defines the encoder a custom checkpoint is built on.
        model_name: The registry model name, e.g. 'hvit_t_cells'.
        joint_checkpoint: Name of the joint trainer checkpoint the decoder is taken from.

    Returns:
        The UniSAM2 model in eval mode.
    """
    from micro_sam.v2.instance_segmentation import get_unisam2_model

    encoder = model_type or model_name[:6]

    if checkpoint_path is None and model_type is not None:
        checkpoint_path = export_joint_checkpoint(model_type, joint_checkpoint)[1]

    if checkpoint_path is not None:
        print(f"Loading the UniSAM2 model from '{checkpoint_path}' with the '{encoder}' encoder.")
        return get_unisam2_model(checkpoint_path, device=device, encoder=encoder)

    from micro_sam.v2.util import FINETUNED_MODELS, _download_finetuned_sam2_model
    assert model_name in FINETUNED_MODELS, f"'{model_name}' is not a registered model: {FINETUNED_MODELS}."
    print(f"Fetching finetuned decoder for '{model_name}' from the micro-sam v2 registry.")
    _, _, decoder_source = _download_finetuned_sam2_model(model_name)
    if decoder_source is None:
        raise RuntimeError(f"The registry model '{model_name}' has no registered decoder.")
    return get_unisam2_model(decoder_source, device=device, encoder=encoder)


def build_apg_segmenter(device, ndim, model_type, joint_checkpoint="best"):
    """Build the automatic prompt generator, which needs both halves of the joint model.

    The decoder proposes the candidates and the SAM2 branch turns each of them into a mask, so unlike
    the postprocessing sweep this cannot be scored off a cached decoder prediction alone. A volume is
    prompted through the video predictor, which propagates the prompts across slices.

    Args:
        device: The device to load the models onto.
        ndim: The number of spatial dimensions, 2 or 3.
        model_type: The SAM2 backbone of the jointly finetuned model, e.g. 'hvit_t'.
        joint_checkpoint: Name of the joint trainer checkpoint both halves are taken from.

    Returns:
        The prompt generator, ready to be initialized on a sample.
    """
    from micro_sam.v2.util import get_sam2_model
    from micro_sam.v2.instance_segmentation import get_unisam2_model, get_instance_segmentation_generator

    if model_type is None:
        raise ValueError("Tuning the prompt generation needs the joint model, so -m/--model_type is required.")

    interactive_path, decoder_path = export_joint_checkpoint(model_type, joint_checkpoint)
    print(f"Loading the prompt generator from '{interactive_path}' and '{decoder_path}'.")
    decoder = get_unisam2_model(decoder_path, device=device, encoder=model_type)
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=interactive_path,
        **({"input_type": "videos"} if ndim == 3 else {}),
    )
    return get_instance_segmentation_generator(
        model=model, decoder=decoder, segmentation_mode="apg", device=device, ndim=ndim,
    )


def score_sample_apg(segmenter, raw, labels, params_list, ndim, metric_mode, spacing=None):
    """Return a per-combo list of metric dicts for one sample, aligned with params_list.

    The encoder and the decoder run once per sample; every combo then only re-derives the prompts and
    re-prompts SAM2, which is what makes sweeping this affordable at all.
    """
    segmenter.clear_state()
    segmenter.initialize(raw, ndim=ndim)
    scores = []
    for params in params_list:
        try:
            seg = segmenter.generate(spacing=spacing, **params)
            scores.append(compute_metrics(seg.astype("uint32"), labels, metric_mode))
        except Exception as e:
            warnings.warn(f"Prompt generation failed for {params}: {e}")
            scores.append(None)
    return scores


NORMALIZATIONS = ("minmax", "percentile_1_99", "percentile_2_98")


def get_normalization(choice):
    """Return the input normalization callable for a choice, or None for the inference default."""
    from micro_sam.v2.normalization import normalize_raw

    if choice == "percentile_2_98":  # The default of the inference path.
        return None

    if choice == "minmax":
        def normalization(crop):
            lower = crop.min(axis=(-2, -1), keepdims=True)
            upper = crop.max(axis=(-2, -1), keepdims=True)
            return (crop - lower) / (upper - lower + 1e-7)
        return normalization

    if choice == "percentile_1_99":
        return lambda crop: normalize_raw(crop, axis=(-2, -1), lower_percentile=1.0, upper_percentile=99.0)

    raise ValueError(f"Unknown normalization: '{choice}'. Choose from {NORMALIZATIONS}.")


def predict(model, raw, ndim, device, normalization=None):
    """Run the UniSAM2 model to predict foreground + directed distances, shape (4, *spatial).

    Uses the same inference path as the micro-sam v2 evaluation (common.predict_unisam2), so the
    tuned parameters transfer directly to the evaluation scripts.
    """
    from common import predict_unisam2
    return predict_unisam2(model, raw, ndim=ndim, device=device, normalization=normalization)


def read_image_2d(path, key):
    """Read a 2d (grayscale) image from a plain image file or an H5/zarr key."""
    if key is not None:
        arr = np.asarray(open_file(path, mode="r")[key][:])
    else:
        arr = np.asarray(imageio.imread(path))
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0] and arr.shape[2] > arr.shape[0]:
        arr = arr.transpose(1, 2, 0)
    # Some 2d datasets mix in multi-frame stacks, e.g. yeaz. Evaluate their first frame.
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        arr = arr[0]
    # The UniSAM2 2d inference path expects single-channel input. Reduce a trailing channel axis.
    if arr.ndim == 3:
        arr = arr.mean(axis=-1)
    return arr


def center_crop_2d(arr, crop_shape):
    """Return a center crop of a 2d array along its first two axes."""
    roi = []
    for size, crop in zip(arr.shape[:2], crop_shape):
        crop = min(crop, size)
        start = (size - crop) // 2
        roi.append(slice(start, start + crop))
    return arr[tuple(roi)]


def resolve_data_paths(dataset_name, livecell_per_celltype=None, match_evaluation=False, split="test"):
    """Return (raw_paths, label_paths, raw_key, label_key) for a dataset's evaluation split.

    Special-cases dsb to use the smaller 'reduced' fluorescence test split (50 images) and livecell to
    use the built-in per-cell-type stratification ('n_val_per_cell_type' in _get_livecell_paths), rather
    than the full sets that common.get_data_paths returns; all other datasets defer to common.
    'match_evaluation' skips both special cases, so the tuned parameters are selected on exactly the
    samples the evaluation scores.

    With split='val' the parameters are tuned on held-out data instead, which keeps the evaluation on
    the test split honest. common.get_data_paths resolves that split; see common.VAL_SPLITS for what
    counts as held out per dataset. livecell keeps its stratification here either way.
    """
    if split == "val":
        if dataset_name == "livecell":
            from micro_sam.v1.evaluation.livecell import _get_livecell_paths
            # 0 means every val image. Passing 0 straight through would match the '>= 0' cap and select none.
            img, gt = _get_livecell_paths(
                input_folder=os.path.join(DATA_ROOT, "livecell"), split="val",
                n_val_per_cell_type=livecell_per_celltype or None,
            )
            return sorted(img), sorted(gt), None, None
        return get_data_paths(dataset_name, DATA_ROOT, split="val")

    if match_evaluation:
        return get_data_paths(dataset_name, DATA_ROOT)
    if dataset_name == "dsb":
        from torch_em.data import datasets
        img, gt = datasets.dsb.get_dsb_paths(
            path=os.path.join(DATA_ROOT, "dsb"), source="reduced", split="test",
        )
        return img, gt, None, None
    if dataset_name == "livecell":
        from micro_sam.v1.evaluation.livecell import _get_livecell_paths
        img, gt = _get_livecell_paths(
            input_folder=os.path.join(DATA_ROOT, "livecell"), split="test",
            n_val_per_cell_type=livecell_per_celltype,
        )
        img, gt = drop_excluded_livecell(img, gt)
        return sorted(img), sorted(gt), None, None
    return get_data_paths(dataset_name, DATA_ROOT)


def build_work_items(track_cfg, n_images, livecell_per_celltype):
    """Resolve a track's datasets into a flat list of (dataset, raw_path, label_path, raw_key, label_key).

    Datasets whose paths cannot be resolved (e.g. a loader missing from the installed torch-em) are
    skipped with a warning rather than aborting the whole track. livecell is stratified to
    'livecell_per_celltype' images per cell type (built into _get_livecell_paths); other 2d datasets use
    every test image (optionally capped to the first 'n_images'); 3d tracks use every available volume.
    A config with 'match_evaluation' instead takes the same sorted samples as the evaluation, capped
    the same way, so the tuned parameters transfer.
    """
    match_evaluation = track_cfg.get("match_evaluation", False)
    split = track_cfg.get("split", "test")
    items = []
    for dataset_name in track_cfg["datasets"]:
        try:
            raw_paths, label_paths, raw_key, label_key = resolve_data_paths(
                dataset_name, livecell_per_celltype, match_evaluation, split,
            )
        except Exception as e:
            warnings.warn(f"Skipping dataset '{dataset_name}': {e}")
            continue

        pairs = list(zip(raw_paths, label_paths))
        if match_evaluation:
            pairs = sorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1])))
            pairs = pairs[:(n_images or MAX_EVALUATION_SAMPLES)]
        elif track_cfg["ndim"] == 2 and dataset_name != "livecell" and n_images is not None:
            pairs = pairs[:n_images]
        for raw_path, label_path in pairs:
            items.append((dataset_name, raw_path, label_path, raw_key, label_key))
    return items


def load_sample(item, ndim, em_crop, z_range=None):
    """Load and preprocess one work item into (raw, labels), matching the micro-sam v2 eval crops.

    'z_range' restricts a volume to a z-slab before cropping, which is how the EM datasets get tuning
    data disjoint from the slab the evaluation scores.
    """
    dataset_name, raw_path, label_path, raw_key, label_key = item
    if ndim == 2:
        raw = center_crop_2d(read_image_2d(raw_path, raw_key), CROP_SHAPE_2D).astype("float32")
        labels = center_crop_2d(read_image_2d(label_path, label_key), CROP_SHAPE_2D)
        labels = connected_components(labels).astype("uint32")
    else:
        raw, labels, _ = load_volume(
            raw_path=raw_path, label_path=label_path, raw_key=raw_key, label_key=label_key,
            dataset_name=dataset_name, crop_shape=tuple(em_crop), z_range=z_range,
        )
    return raw, labels


def compute_metrics(seg, labels, mode):
    """Compute the evaluation metrics for one segmentation.

    Always reports mSA. For dense (EM neuron) mode it additionally reports the CREMI score and its
    VI-split / VI-merge / adapted-Rand components, since neuron segmentation is ranked by CREMI
    (lower is better) rather than mSA.
    """
    metrics = {"msa": float(mean_segmentation_accuracy(seg, labels))}
    if mode == "dense":
        from elf.evaluation import cremi_score
        vi_split, vi_merge, adapted_rand, cremi = cremi_score(seg, labels)
        metrics["cremi"] = float(cremi)
        metrics["vi_split"] = float(vi_split)
        metrics["vi_merge"] = float(vi_merge)
        metrics["adapted_rand"] = float(adapted_rand)
    return metrics


def postprocess(prediction, mode, params, backend="cpp", n_threads=POSTPROC_THREADS, spacing=None):
    """Convert a (4, *spatial) prediction into an instance segmentation for the given mode."""
    if mode == "dense":
        fg = prediction[0]
        boundary_map = fg.max() - fg
        denom = boundary_map.max()
        if denom > 0:
            boundary_map = boundary_map / denom
        distances = np.stack([prediction[2], prediction[3]])
        seg = run_multicut(boundary_map, distances, backend=backend, n_threads=n_threads, **params)
    else:
        seg = flow_instance_segmentation(
            prediction[0], prediction[1:], backend=backend, n_threads=n_threads, spacing=spacing, **params,
        )
    return seg.astype("uint32")


def deduplicate_flow_travel(params_list):
    """Drop sparse combos whose flow travel duplicates a cheaper one.

    'n_iter' and 'dt' act on the segmentation only through their product, the distance a pixel is
    advected: on livecell val, 10 x 0.25 and 50 x 0.05 both scored 0.0287, and 100 x 0.25 and
    50 x 0.50 both scored 0.285. Keeping the smallest 'n_iter' per product therefore covers the same
    travel distances with fewer combos and fewer integration steps. The equivalence is empirical
    rather than exact, since the integrator's discretization error still depends on 'dt'.
    """
    best, order = {}, []
    for params in params_list:
        if "n_iter" not in params or "dt" not in params:
            return params_list
        others = tuple(sorted((k, v) for k, v in params.items() if k not in ("n_iter", "dt")))
        key = (others, round(params["n_iter"] * params["dt"], 6))
        if key not in best:
            best[key] = params
            order.append(key)
        elif params["n_iter"] < best[key]["n_iter"]:
            best[key] = params
    return [best[key] for key in order]


def score_image_sparse_cached(prediction, labels, params_list, backend, n_threads=POSTPROC_THREADS, spacing=None):
    """Score all sparse combos on one image, caching the flow density across shared combos.

    Faithfully reproduces flow_instance_segmentation: the height map and each convergence-density map
    (the expensive flow integration) are computed once and reused; only the cheap seed / watershed /
    min-size steps vary per (density_threshold, min_size). 'spacing' is the anisotropic voxel spacing
    for 3d flow smoothing. Returns a per-combo list of metric dicts (None where a combo failed), aligned
    with params_list.
    """
    foreground = prediction[0]
    directed = prediction[1:]
    ndim = foreground.ndim
    if directed.shape[0] > ndim:
        directed = directed[-ndim:]

    # The convergence densities and the height maps are built up front, so the scoring below only reads them.
    fg_mask_cache, density_cache, hmap_cache = {}, {}, {}
    for params in params_list:
        ft, sigma, n_iter, dt = (params[k] for k in FLOW_DENSITY_KEYS)
        if ft not in fg_mask_cache:
            fg_mask_cache[ft] = foreground > ft
        key = (ft, sigma, n_iter, dt)
        if key not in density_cache:
            density_cache[key] = _compute_flow_density(
                directed, fg_mask_cache[ft], n_iter=int(n_iter), dt=dt, sigma=sigma, spacing=spacing,
                backend=backend, n_threads=n_threads,
            )
        fw = params["foreground_weight"]
        if fw not in hmap_cache:
            hmap_cache[fw] = watershed_heightmap(foreground, directed, fw)

    # The base watershed does not depend on min_size, so all min_size values of a combo reuse it.
    base_cache, base_lock = {}, threading.Lock()

    def base_segmentation(key, fg_mask, density, density_threshold, hmap):
        with base_lock:
            cached = base_cache.get(key)
        if cached is None:
            seeds = connected_components(density > density_threshold)
            cached = watershed(hmap, markers=seeds, mask=fg_mask)
            with base_lock:
                base_cache[key] = cached
        return cached

    def score(params):
        ft, sigma, n_iter, dt = (params[k] for k in FLOW_DENSITY_KEYS)
        fw, density_threshold = params["foreground_weight"], params["density_threshold"]
        fg_mask = fg_mask_cache[ft]
        hmap = hmap_cache[fw]
        try:
            key = (ft, sigma, n_iter, dt, density_threshold, fw)
            seg = base_segmentation(key, fg_mask, density_cache[(ft, sigma, n_iter, dt)], density_threshold, hmap)
            min_size = params["min_size"]
            if min_size > 0:
                seg = seg.copy()
                ids, sizes = np.unique(seg, return_counts=True)
                discard = ids[(sizes < min_size) & (ids > 0)]
                seg[np.isin(seg, discard)] = 0
                seg = watershed(hmap, markers=seg, mask=fg_mask)
            return compute_metrics(seg.astype("uint32"), labels, "sparse")
        except Exception as e:
            warnings.warn(f"Sparse postprocessing failed for {params}: {e}")
            return None

    # Scoring a combination is independent of the others, and dominates the runtime of the sweep.
    with futures.ThreadPoolExecutor(n_threads) as tp:
        return list(tp.map(score, params_list))


def _dense_boundary_and_distances(prediction):
    """Return the (boundary_map, in-plane distances) inputs to run_multicut for a dense prediction."""
    fg = prediction[0]
    boundary_map = fg.max() - fg
    denom = boundary_map.max()
    if denom > 0:
        boundary_map = boundary_map / denom
    distances = np.stack([prediction[2], prediction[3]])
    return boundary_map, distances


def _dense_oversegmentation(boundary_map, distances, density_threshold, sigma, n_iter, dt, backend, n_threads):
    """Slice-wise oversegmentation + RAG + boundary features: the expensive, beta-independent part of run_multicut.

    Replicates run_multicut up to (but not including) the beta-dependent edge costs, so the result can be
    cached and reused across all beta values.
    """
    from elf.segmentation.features import compute_rag, compute_boundary_mean_and_length, compute_z_edge_mask
    n_slices = boundary_map.shape[0]
    overseg = np.zeros(boundary_map.shape, dtype="uint64")

    def run_overseg(z):
        bd = boundary_map[z]
        dists = distances[:, z]
        fg_mask = np.ones(bd.shape, dtype="bool")
        density = _compute_flow_density(
            dists, fg_mask, n_iter=int(n_iter), dt=dt, sigma=sigma, verbose=False, backend=backend, n_threads=1,
        )
        seeds = connected_components(density > density_threshold)
        bd = bd if np.issubdtype(bd.dtype, np.floating) else bd.astype("float32")
        wsz = watershed(bd, markers=seeds)
        overseg[z] = wsz
        return int(wsz.max())

    with futures.ThreadPoolExecutor(n_threads) as tp:
        offsets = list(tp.map(run_overseg, range(n_slices)))

    offsets = np.array(offsets, dtype="uint64")
    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    overseg += np.cumsum(offsets)[:, None, None]

    rag = compute_rag(overseg)
    feats = compute_boundary_mean_and_length(rag, overseg, boundary_map)
    z_edges = None if n_slices == 1 else compute_z_edge_mask(rag, overseg)
    return overseg, rag, feats, z_edges, n_slices


def _dense_solve(rag, feats, z_edges, overseg, n_slices, beta):
    """Run the beta-dependent edge costs + multicut solve on a cached oversegmentation."""
    from elf.segmentation.features import project_node_labels_to_pixels
    from elf.segmentation.multicut import compute_edge_costs, multicut_decomposition
    if n_slices == 1:
        costs = compute_edge_costs(feats[:, 0], edge_sizes=feats[:, 1], weighting_scheme="all", beta=beta)
    else:
        costs = compute_edge_costs(
            feats[:, 0], edge_sizes=feats[:, 1], weighting_scheme="xyz", z_edge_mask=z_edges, beta=beta,
        )
    node_labels = multicut_decomposition(rag, costs)
    seg = project_node_labels_to_pixels(rag, overseg, node_labels)
    return seg.astype("uint32")


def score_image_dense_cached(prediction, labels, params_list, backend, n_threads=POSTPROC_THREADS):
    """Score all dense combos on one image, caching the oversegmentation + RAG across shared combos.

    Faithfully reproduces run_multicut: the slice-wise oversegmentation and region-adjacency graph (the
    expensive part) are computed once per (density_threshold, sigma, n_iter, dt) and reused; only the
    cheap edge-cost + multicut-solve step varies per beta. Returns a per-combo list of metric dicts
    (None where a combo failed), aligned with params_list.
    """
    boundary_map, distances = _dense_boundary_and_distances(prediction)
    overseg_cache = {}
    scores = []
    for params in params_list:
        dt = params.get("dt", 0.5)
        key = tuple(params.get(k, dt if k == "dt" else None) for k in OVERSEG_KEYS)
        if key not in overseg_cache:
            overseg_cache[key] = _dense_oversegmentation(
                boundary_map, distances, params["density_threshold"], params["sigma"], params["n_iter"],
                dt=dt, backend=backend, n_threads=n_threads,
            )
        overseg, rag, feats, z_edges, n_slices = overseg_cache[key]
        try:
            seg = _dense_solve(rag, feats, z_edges, overseg, n_slices, params["beta"])
            scores.append(compute_metrics(seg, labels, "dense"))
        except Exception as e:
            warnings.warn(f"Dense multicut solve failed for {params}: {e}")
            scores.append(None)
    return scores


def score_image(
    prediction, labels, mode, params_list, backend, use_flow_cache=True, n_threads=POSTPROC_THREADS, spacing=None,
):
    """Return a per-combo list of metric dicts for one image (None where a combo failed), aligned with params_list."""
    if use_flow_cache and mode == "sparse":
        return score_image_sparse_cached(
            prediction, labels, params_list, backend, n_threads=n_threads, spacing=spacing,
        )
    if use_flow_cache and mode == "dense":
        return score_image_dense_cached(prediction, labels, params_list, backend, n_threads=n_threads)
    scores = []
    for params in params_list:
        try:
            seg = postprocess(prediction, mode, params, backend=backend, n_threads=n_threads, spacing=spacing)
            scores.append(compute_metrics(seg, labels, mode))
        except Exception as e:
            warnings.warn(f"Postprocessing failed for {params}: {e}")
            scores.append(None)
    return scores


def shard_csv_path(output_dir, track_name, shard, n_shards):
    """Path of one shard's partial results."""
    return os.path.join(output_dir, f"{track_name}.shard{shard}of{n_shards}.csv")


def merge_shards(output_dir, track_name, criterion, n_shards):
    """Combine the per-shard partial results into the final CSV and return it.

    Each shard stores a per-combo sample count plus the sum and the sum of squares of every metric, so
    the pooled mean and (population) standard deviation are exact, not an average of averages.
    """
    paths = [shard_csv_path(output_dir, track_name, i, n_shards) for i in range(n_shards)]
    missing = [os.path.basename(p) for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Cannot merge '{track_name}', these shards are missing: {missing}.")

    frames = [pd.read_csv(p) for p in paths]
    accumulated = [c for c in frames[0].columns if c.endswith(("_sum", "_sumsq")) or c == "n_images"]
    param_keys = [c for c in frames[0].columns if c not in accumulated]

    merged = pd.concat(frames, ignore_index=True).groupby(param_keys, as_index=False)[accumulated].sum()
    metrics = [c[:-len("_sum")] for c in accumulated if c.endswith("_sum")]
    for metric in metrics:
        n = merged["n_images"]
        mean = merged[f"{metric}_sum"] / n
        merged[f"{metric}_mean"] = mean
        merged[f"{metric}_std"] = np.sqrt((merged[f"{metric}_sumsq"] / n - mean ** 2).clip(lower=0.0))
    merged = merged.drop(columns=[f"{m}_{s}" for m in metrics for s in ("sum", "sumsq")])

    ascending = CRITERION_ASCENDING[criterion]
    merged = merged.sort_values(f"{criterion}_mean", ascending=ascending).reset_index(drop=True)
    csv_path = os.path.join(output_dir, f"{track_name}.csv")
    merged.to_csv(csv_path, index=False)
    print(f"Merged {n_shards} shards into {csv_path}.")
    return merged


def run_track(
    track_name, track_cfg, model, n_images, livecell_per_celltype, output_dir, backend, device,
    crop_override=None, use_flow_cache=True, n_threads=POSTPROC_THREADS, normalization=None,
    shard=0, n_shards=1,
):
    """Run the full grid search for one track and save the results CSV.

    Inference is run once per image (streaming, so predictions are never all held in memory at once)
    and every parameter combination is scored on that image before moving on. The per-image mSA is
    then averaged over all images of the track. Datasets or images that fail to load are skipped with
    a warning. 'crop_override' replaces the track's default 3d crop when given.

    With 'n_shards' > 1 this scores only every n_shards-th sample, starting at 'shard', and writes a
    partial CSV of accumulated metrics. The per-image scores are independent, so the shards are
    combined afterwards by merge_shards. Best-parameter reporting is skipped for a partial run.
    """
    ndim = track_cfg["ndim"]
    mode = track_cfg["mode"]
    metric_mode = track_cfg.get("metric_mode", mode)
    spacing = track_cfg.get("spacing")
    z_range = track_cfg.get("z_range")
    crop = crop_override if crop_override is not None else track_cfg.get("crop")
    os.makedirs(output_dir, exist_ok=True)
    is_shard = n_shards > 1
    csv_path = shard_csv_path(output_dir, track_name, shard, n_shards) if is_shard \
        else os.path.join(output_dir, f"{track_name}.csv")

    if os.path.exists(csv_path):
        print(f"Results already exist at '{csv_path}', loading.")
        df = pd.read_csv(csv_path)
    else:
        items = build_work_items(track_cfg, n_images, livecell_per_celltype)
        if not items:
            warnings.warn(f"No usable data for track '{track_name}', skipping.")
            return None
        if is_shard:
            # Striding, not slicing: the samples are sorted by cell type, so every shard sees all of them.
            items = items[shard::n_shards]

        keys = list(track_cfg["grid"].keys())
        combos = list(itertools.product(*[track_cfg["grid"][k] for k in keys]))
        params_list = [dict(zip(keys, combo)) for combo in combos]
        if mode == "sparse":
            n_full = len(params_list)
            params_list = deduplicate_flow_travel(params_list)
            if len(params_list) < n_full:
                print(f"Deduplicated {n_full} combos to {len(params_list)} by flow travel (n_iter * dt).")
        metric_lists = [[] for _ in params_list]  # per combo: a list of per-image metric dicts
        criterion = track_cfg.get("criterion", "msa")
        ascending = CRITERION_ASCENDING[criterion]
        print(f"{track_name}: {len(params_list)} combinations over {len(items)} sample(s), mode='{mode}'.")

        t0 = time.perf_counter()
        for item in tqdm(items, desc=f"{track_name} samples"):
            try:
                raw, labels = load_sample(item, ndim, crop, z_range=z_range)
                if mode == "apg":
                    # 'model' is the prompt generator here, which holds both halves of the joint model.
                    scores = score_sample_apg(model, raw, labels, params_list, ndim, metric_mode, spacing=spacing)
                else:
                    prediction = predict(model, raw, ndim=ndim, device=device, normalization=normalization)
                    scores = score_image(
                        prediction, labels, mode, params_list, backend,
                        use_flow_cache=use_flow_cache, n_threads=n_threads, spacing=spacing,
                    )
            except Exception as e:
                warnings.warn(f"Skipping sample '{item[1]}': {e}")
                continue
            for i, metrics in enumerate(scores):
                if metrics is not None:
                    metric_lists[i].append(metrics)

        rows = []
        for params, per_image in zip(params_list, metric_lists):
            if not per_image:
                continue
            row = {**params, "n_images": len(per_image)}
            for metric_key in per_image[0]:
                values = np.asarray([m[metric_key] for m in per_image], dtype="float64")
                if is_shard:
                    row[f"{metric_key}_sum"] = float(values.sum())
                    row[f"{metric_key}_sumsq"] = float((values ** 2).sum())
                else:
                    row[f"{metric_key}_mean"] = float(values.mean())
                    row[f"{metric_key}_std"] = float(values.std())
            rows.append(row)
        df = pd.DataFrame(rows)
        if not is_shard:
            df = df.sort_values(f"{criterion}_mean", ascending=ascending).reset_index(drop=True)
        df.to_csv(csv_path, index=False)
        print(f"Saved {csv_path} ({time.perf_counter() - t0:.0f}s).")

    if is_shard:
        print(f"Shard {shard}/{n_shards} done. Merge with --merge once every shard has finished.")
        return None

    return report_best(df, track_name, track_cfg)


def report_best(df, track_name, track_cfg):
    """Print and return the best parameter combination of a finished grid."""
    criterion = track_cfg.get("criterion", "msa")
    ascending = CRITERION_ASCENDING[criterion]
    best = df.sort_values(f"{criterion}_mean", ascending=ascending).iloc[0]
    best_params = {k: best[k] for k in track_cfg["grid"].keys()}
    crit_mean, crit_std = best[f"{criterion}_mean"], best[f"{criterion}_std"]
    direction = "lower is better" if ascending else "higher is better"
    print(f"Best params for {track_name} (by {criterion}, {direction}): {best_params}")
    print(f"  {criterion} = {crit_mean:.4f} (+/-{crit_std:.4f})")
    return best_params


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--track", default=None, choices=list(TRACKS) + ["all"], help="Which track to run.")
    parser.add_argument("-d", "--dataset_name", default=None, choices=sorted(DATASETS_2D + DATASETS_3D),
                        help="Tune on a single dataset instead of a track.")
    parser.add_argument("--criterion", default=None, choices=list(CRITERION_ASCENDING),
                        help="Metric the grid is ranked by. Defaults to mSA, or CREMI for EM data.")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="Split to tune on. 'val' keeps the tuning off the evaluated samples.")
    parser.add_argument("--tune", default="ais", choices=["ais", "apg"],
                        help="What to tune: the AIS postprocessing or the automatic prompt generation.")
    parser.add_argument("--joint_checkpoint", default="best",
                        help="Name of the joint trainer checkpoint to tune, without the '.pt' suffix.")
    parser.add_argument("-o", "--output_dir", default=OUTPUT_ROOT, help="Directory to write result CSVs.")
    parser.add_argument("-n", "--n_images", type=int, default=None, help="Cap images per 2d dataset (default: all).")
    parser.add_argument("--livecell_per_celltype", type=int, default=50, help="Images per livecell cell type.")
    parser.add_argument("--crop_3d", type=int, nargs=3, default=None, help="Override the 3d crop (Z Y X).")
    parser.add_argument("-c", "--checkpoint_path", default=None, help="Custom checkpoint instead of registry model.")
    parser.add_argument("-m", "--model_type", default=None, help="Backbone of the jointly finetuned model to tune.")
    parser.add_argument("--backend", default="cpp", choices=["cpp", "python"], help="Flow computation backend.")
    parser.add_argument("--normalization", default="percentile_2_98", choices=NORMALIZATIONS,
                        help="Input normalization used for inference.")
    parser.add_argument("--shard", type=int, default=0, help="Index of this shard, in [0, n_shards).")
    parser.add_argument("--n_shards", type=int, default=1, help="Split the samples over this many jobs.")
    parser.add_argument("--merge", action="store_true", help="Merge finished shards instead of scoring.")
    parser.add_argument("--no_flow_cache", action="store_true", help="Disable the lazy postprocessing caching.")
    parser.add_argument("--n_threads", type=int, default=POSTPROC_THREADS, help="Threads for postprocessing.")
    args = parser.parse_args()

    if not 0 <= args.shard < args.n_shards:
        raise ValueError(f"--shard must be in [0, {args.n_shards}), got {args.shard}.")

    crop_override = tuple(args.crop_3d) if args.crop_3d is not None else None
    if args.dataset_name is not None:
        configs = {args.dataset_name: dataset_config(args.dataset_name, args.criterion, args.split, args.tune)}
    else:
        if args.tune == "apg":
            raise ValueError("Tuning the prompt generation is per dataset, use -d rather than --track.")
        track_names = list(TRACKS) if args.track in (None, "all") else [args.track]
        configs = {name: TRACKS[name] for name in track_names}

    # Merging only reads the partial CSVs, so it needs neither a GPU nor the model.
    if args.merge:
        summary = {}
        for name, config in configs.items():
            df = merge_shards(args.output_dir, name, config.get("criterion", "msa"), args.n_shards)
            summary[name] = report_best(df, name, config)
        print("\nBest parameters:")
        for name, params in summary.items():
            print(f"{name}: {params}")
        return

    import torch
    device = DEVICE if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")

    if args.tune == "apg":
        ndim = next(iter(configs.values()))["ndim"]
        model = build_apg_segmenter(device, ndim, args.model_type, joint_checkpoint=args.joint_checkpoint)
    else:
        model = load_model(
            device, checkpoint_path=args.checkpoint_path, model_type=args.model_type,
            joint_checkpoint=args.joint_checkpoint,
        )

    summary = {}
    for name, config in configs.items():
        summary[name] = run_track(
            name, config, model, args.n_images, args.livecell_per_celltype,
            args.output_dir, args.backend, device, crop_override=crop_override,
            use_flow_cache=(not args.no_flow_cache), n_threads=args.n_threads,
            normalization=get_normalization(args.normalization),
            shard=args.shard, n_shards=args.n_shards,
        )

    print("\nBest parameters:")
    for name, params in summary.items():
        print(f"{name}: {params}")


if __name__ == "__main__":
    main()
