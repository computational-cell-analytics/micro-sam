"""Benchmark evaluation of the automatic segmentation baselines, i.e. everything but micro-sam2.

Supported methods:
  cellpose: CellPose generalist models (cyto3, cpsam)
  stardist: StarDist pretrained (2D_versatile_fluo / 3D_demo)
  cellsam: CellSAM pipeline (2d only)
  microsam_ais: micro-sam v1 automatic instance segmentation
  microsam_apg: micro-sam v1 automatic prompt generation
  segneuron: SegNeuron (3d EM only)
  focus3d: FOCUS-3D (3d LM only)

micro-sam2 itself is evaluated by evaluate_automatic_segmentation.py, which also tunes it.

Usage examples:
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellpose -m cyto3
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method stardist
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellsam
    python evaluate_automatic_baselines.py -d embedseg -e <exp> --method microsam_ais -m vit_b
    python evaluate_automatic_baselines.py -d cremi -e <exp> --method segneuron
    python evaluate_automatic_baselines.py -d gonuclear -e <exp> --method focus3d -m nuclei
"""

import os
import warnings
import argparse
from itertools import islice

import numpy as np
from tqdm import tqdm

import torch

from elf.segmentation import features, multicut, watershed

from common import (
    CROP_SHAPE_3D, DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_LM, DATASETS_EM,
    GT_MIN_SIZE_2D, check_data_download, drop_severed_objects, load_data, n_samples,
    run_dataset_evaluation,
)

LM_DATASETS = set(DATASETS_2D + DATASETS_3D_LM)
EM_DATASETS = set(DATASETS_EM)
METHODS = ["cellpose", "stardist", "cellsam", "microsam_ais", "microsam_apg", "segneuron", "focus3d"]

SEGNEURON_ROOT = "/mnt/vast-nhr/home/archit/u12090/SegNeuron"
SEGNEURON_CHECKPOINT = "/mnt/vast-nhr/projects/cidas/cca/models/segneuron/SegNeuronModel.ckpt"

# FOCUS-3D ships a general, a nuclei and a membrane checkpoint; pick one with -m.
FOCUS3D_DEFAULT_MODEL = "general"
# The radius FOCUS-3D is calibrated on, i.e. the value that leaves the volume unscaled in xy.
FOCUS3D_CELL_RADIUS = 15.0
# Throughput knobs, measured worth nothing here: the patch loop is 86% GPU forward. The batch size
# stays at the plugin value so a job fits the 20 GB slice, which batch 32 overruns at 20.1 GiB.
FOCUS3D_BATCH_SIZE = 16
FOCUS3D_NUM_WORKERS = 4

# The CellPose 4 generalists. The cyto/nuclei checkpoints need CellPose 3, in the 'cellpose3' environment.
CELLPOSE_MODELS = ("cpsam", "cpsam_v2", "cpdino")

STARDIST_2D_MODEL = "2D_versatile_fluo"
STARDIST_3D_MODEL = "3D_demo"

# micro-sam v1 model types
SAM_V1_MODEL_TYPE = "vit_b_lm"

# Per-dataset z/xy anisotropy for CellPose do_3D mode (z_voxel / xy_voxel).
DATASET_ANISOTROPY = {
    "embedseg_mouse_skull": 4.0,
    "embedseg_organoid": 6.0,
    "embedseg_platy_nuclei": 5.0,
    "blastospim": 10.0,
    "mouse_embryo": 4.0,
    "cremi": 10.0,   # z=40nm, xy=4nm
    "snemi": 5.0,    # z=30nm, xy=6nm
}


def _load_cellpose(model_type, device):
    """Load a CellPose model, supporting both the CellPose 3 and the CellPose 4 APIs.

    CellPose 4 dropped the `models.Cellpose` wrapper along with the cyto/nuclei checkpoints, and takes its
    own generalist models ('cpsam', 'cpsam_v2', 'cpdino', 'cpdino-vitb') through `pretrained_model`.
    """
    from cellpose import models
    use_gpu = (device != "cpu") and torch.cuda.is_available()

    if model_type in ("cyto", "cyto2", "cyto3", "nuclei"):
        if not hasattr(models, "Cellpose"):
            raise RuntimeError(
                f"'{model_type}' needs CellPose 3; the installed CellPose only provides "
                f"{getattr(models, 'MODEL_NAMES', ())}."
            )
        return models.Cellpose(gpu=use_gpu, model_type=model_type)

    if model_type in getattr(models, "MODEL_NAMES", ()):
        return models.CellposeModel(gpu=use_gpu, pretrained_model=model_type)

    return models.CellposeModel(gpu=use_gpu, model_type=model_type)


def _load_stardist(ndim):
    from stardist.models import StarDist2D, StarDist3D
    if ndim == 3:
        return StarDist3D.from_pretrained(STARDIST_3D_MODEL)
    return StarDist2D.from_pretrained(STARDIST_2D_MODEL)


def _load_segneuron(checkpoint_path, device):
    import sys
    sys.path.insert(0, os.path.join(SEGNEURON_ROOT, "Train_and_Inference"))
    from collections import OrderedDict
    import torch
    from model.Mnet import MNet
    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU="sub")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    weights = state.get("model_weights", state)
    clean = OrderedDict((k.replace("module.", ""), v) for k, v in weights.items())
    model.load_state_dict(clean)
    model.to(device)
    model.eval()
    return model


def _load_focus3d(model_type, checkpoint, device, batch_size, num_workers):
    from focus3d_baseline import Focus3DSegmenter
    return Focus3DSegmenter(
        model_type=model_type, checkpoint=checkpoint, device=device,
        batch_size=batch_size, data_loader_num_workers=num_workers,
    )


def _load_microsam_v1(method, model_type, checkpoint, device):
    from micro_sam.v1.automatic_segmentation import get_predictor_and_segmenter
    mode = {"microsam_ais": "ais", "microsam_apg": "apg"}[method]
    return get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, device=device, segmentation_mode=mode,
    )


def _segment_cellpose(image_or_volume, model, ndim, dataset_name=None):
    if ndim == 3:
        anisotropy = DATASET_ANISOTROPY.get(dataset_name, None)
        masks = model.eval(
            image_or_volume, diameter=None, channels=[0, 0], do_3D=True, anisotropy=anisotropy, z_axis=0,
        )[0]
    else:
        masks = model.eval(image_or_volume, diameter=None, channels=[0, 0])[0]
    return masks.astype("uint32")


def _segment_stardist(image_or_volume, model, ndim):
    from csbdeep.utils import normalize as csbdeep_normalize
    if ndim == 2 and image_or_volume.ndim == 3:
        image_or_volume = image_or_volume.mean(axis=-1)
    inp = csbdeep_normalize(image_or_volume, 1.0, 99.8)
    seg, _ = model.predict_instances(inp) if ndim == 3 else model.predict_instances(inp, scale=1)
    return seg.astype("uint32")


def _segment_cellsam(image):
    from cellSAM import cellsam_pipeline
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    seg = cellsam_pipeline(image, use_wsi=False)
    if seg.ndim == 3:
        seg = seg[0]

    return seg.astype("uint32")


def _segneuron_multicut(affs, beta):
    """Reproduce SegNeuron's 'FRMC_post.post_mc' against the installed elf.

    The installed elf takes the segmentation as the second argument of every rag-consuming function,
    which SegNeuron's own calls predate, so it raises before producing anything.

    Args:
        affs: The affinity map, of shape (3, Z, Y, X).
        beta: The multicut bias towards over- or under-segmentation.

    Returns:
        The instance segmentation.
    """
    affs = 1 - affs
    boundary_input = np.maximum(affs[1], affs[2])
    seeds = np.zeros_like(boundary_input, dtype="uint64")
    offset = 0
    for z in range(seeds.shape[0]):
        wsz, max_id = watershed.distance_transform_watershed(boundary_input[z], threshold=0.25, sigma_seeds=2.0)
        seeds[z] = wsz + offset
        offset += max_id

    rag = features.compute_rag(seeds)
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
    costs = features.compute_affinity_features(rag, seeds, affs, offsets)[:, 0]
    edge_sizes = features.compute_boundary_mean_and_length(rag, seeds, boundary_input)[:, 1]
    costs = multicut.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)
    return features.project_node_labels_to_pixels(rag, seeds, multicut.multicut_kernighan_lin(rag, costs))


def _segment_segneuron(volume, model, device, beta=0.25):
    import torch

    raw = volume.astype("float32") / 255.0 if volume.max() > 1.0 else volume.astype("float32")  # SegNeuron expects /255
    Z, Y, X = raw.shape
    bz, by, bx = 20, 128, 128
    hz, hy, hx = 4, 32, 32
    sz, sy, sx = bz - 2 * hz, by - 2 * hy, bx - 2 * hx

    pz = (-Z % sz) if Z > sz else (bz - Z)
    py = (-Y % sy) if Y > sy else (by - Y)
    px = (-X % sx) if X > sx else (bx - X)
    raw_pad = np.pad(raw, ((hz, hz + pz), (hy, hy + py), (hx, hx + px)), mode="reflect")

    Zp, Yp, Xp = raw_pad.shape
    affs_acc = np.zeros((3, Zp, Yp, Xp), dtype="float32")
    bound_acc = np.zeros((1, Zp, Yp, Xp), dtype="float32")
    count = np.zeros((1, Zp, Yp, Xp), dtype="float32")

    z_starts = list(range(0, Zp - bz + 1, sz)) or [0]
    y_starts = list(range(0, Yp - by + 1, sy)) or [0]
    x_starts = list(range(0, Xp - bx + 1, sx)) or [0]

    with torch.no_grad():
        for z0 in z_starts:
            for y0 in y_starts:
                for x0 in x_starts:
                    z1, y1, x1 = z0 + bz, y0 + by, x0 + bx
                    crop = raw_pad[z0:z1, y0:y1, x0:x1]
                    inp = torch.from_numpy(crop[None, None]).to(device)
                    pred_affs, pred_bound = model(inp)
                    affs_acc[:, z0:z1, y0:y1, x0:x1] += pred_affs[0].cpu().numpy()
                    bound_acc[:, z0:z1, y0:y1, x0:x1] += pred_bound[0].cpu().numpy()
                    count[:, z0:z1, y0:y1, x0:x1] += 1.0

    affs = affs_acc[:, hz:hz + Z, hy:hy + Y, hx:hx + X] / count[:, hz:hz + Z, hy:hy + Y, hx:hx + X]
    bound = bound_acc[:, hz:hz + Z, hy:hy + Y, hx:hx + X] / count[:, hz:hz + Z, hy:hy + Y, hx:hx + X]

    combined = np.minimum(np.stack([bound[0]] * 3), affs)
    return _segneuron_multicut(combined, beta=beta).astype("uint32")


def _segment_focus3d(volume, segmenter, dataset_name, cell_radius, z_ratio):
    """Segment one volume with FOCUS-3D, which rescales it to its own scale first.

    `z_ratio` is the z-to-xy spacing ratio CellPose gets as its anisotropy, `cell_radius` the xy
    radius to rescale from. They are coupled: at a z_ratio of 1 the z axis is rescaled along with xy.
    """
    if z_ratio is None:
        z_ratio = DATASET_ANISOTROPY.get(dataset_name, 1.0)
    return segmenter(volume, z_ratio=z_ratio, cell_radius=cell_radius)


def _segment_microsam_v1(image_or_volume, predictor, segmenter, ndim):
    from micro_sam.v1.automatic_segmentation import automatic_instance_segmentation
    seg = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=image_or_volume, ndim=ndim, verbose=False,
    )
    return seg.astype("uint32") if seg is not None else np.zeros(image_or_volume.shape, dtype="uint32")


def _run_evaluation(segment_fn, dataset_name, data_root, ndim, save_path, desc, limit, crop_shape=None):
    if limit is not None:  # Name the file after the truncation, so it cannot pass for the full evaluation.
        save_path = f"{save_path[:-4]}_n{limit}.csv"
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return

    total = n_samples(dataset_name, data_root)
    all_gt, all_seg = [], []
    samples = load_data(dataset_name, data_root, ndim, crop_shape=crop_shape)
    if limit is not None:
        total = min(total, limit)
        samples = islice(samples, limit)
    for image_or_volume, labels, valid_roi in tqdm(samples, total=total, desc=desc):
        if labels.max() == 0:  # Nothing to score without ground-truth.
            continue

        seg = segment_fn(image_or_volume)
        if valid_roi is not None:
            seg[~valid_roi] = 0
        if ndim == 2:
            # The ground truth has no severed objects either, so predicting one is not a false positive.
            seg = drop_severed_objects(seg, GT_MIN_SIZE_2D.get(dataset_name, 0))
        all_gt.append(labels)
        all_seg.append(seg)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = run_dataset_evaluation(all_gt, all_seg, dataset_name, save_path)
    print(results)


def run_cellpose_evaluation(dataset_name, data_root, experiment_folder, model_type, device, limit):
    ndim = 3 if dataset_name in DATASETS_3D else 2
    model = _load_cellpose(model_type, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellpose_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_cellpose(x, model, ndim, dataset_name),
        dataset_name, data_root, ndim, save_path, desc=f"cellpose-{model_type}", limit=limit,
    )


def run_stardist_evaluation(dataset_name, data_root, experiment_folder, limit):
    ndim = 3 if dataset_name in DATASETS_3D else 2
    model = _load_stardist(ndim)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_stardist.csv")
    _run_evaluation(
        lambda x: _segment_stardist(x, model, ndim),
        dataset_name, data_root, ndim, save_path, desc="stardist", limit=limit,
    )


def run_cellsam_evaluation(dataset_name, data_root, experiment_folder, limit):
    if dataset_name in DATASETS_3D:
        warnings.warn(
            f"CellSAM is 2D-only and does not support 3D dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellsam.csv")
    _run_evaluation(
        lambda x: _segment_cellsam(x),
        dataset_name, data_root, ndim=2, save_path=save_path, desc="cellsam", limit=limit,
    )


def run_segneuron_evaluation(dataset_name, data_root, experiment_folder, device, limit, checkpoint_path=None):
    if dataset_name not in EM_DATASETS:
        warnings.warn(
            f"SegNeuron is 3D EM-only and does not support dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    if checkpoint_path is None:
        checkpoint_path = SEGNEURON_CHECKPOINT

    model = _load_segneuron(checkpoint_path, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_segneuron.csv")
    _run_evaluation(
        lambda x: _segment_segneuron(x, model, device),
        dataset_name, data_root, ndim=3, save_path=save_path, desc="segneuron", limit=limit,
    )


def run_focus3d_evaluation(
    dataset_name, data_root, experiment_folder, model_type, checkpoint, device, cell_radius, limit,
    z_ratio=None, crop_shape=None, batch_size=FOCUS3D_BATCH_SIZE, num_workers=FOCUS3D_NUM_WORKERS,
):
    if dataset_name not in DATASETS_3D_LM:
        warnings.warn(
            f"FOCUS-3D is a 3d light microscopy method and does not support dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    segmenter = _load_focus3d(model_type, checkpoint, device, batch_size, num_workers)
    if crop_shape is None:
        # The harness default of 8 slices would leave most of the model's 32-deep window as padding.
        crop_shape = (segmenter.patch_size[0],) + tuple(CROP_SHAPE_3D[1:])
    print(f"FOCUS-3D patch size {segmenter.patch_size}, evaluated on the crop {crop_shape}.")

    # The crop depth is in the name: a baseline scored on another crop is not comparable.
    save_path = os.path.join(
        experiment_folder, "results", f"{dataset_name}_focus3d_{model_type}_z{crop_shape[0]}.csv"
    )
    _run_evaluation(
        lambda x: _segment_focus3d(x, segmenter, dataset_name, cell_radius, z_ratio),
        dataset_name, data_root, ndim=3, save_path=save_path, desc=f"focus3d-{model_type}", limit=limit,
        crop_shape=crop_shape,
    )


def run_microsam_v1_evaluation(
    dataset_name, data_root, experiment_folder, method, model_type, checkpoint, device, limit,
):
    if dataset_name in EM_DATASETS:
        raise ValueError(f"micro-sam v1 automatic methods do not support EM datasets; got '{dataset_name}'.")
    ndim = 3 if dataset_name in DATASETS_3D else 2
    predictor, segmenter = _load_microsam_v1(method, model_type, checkpoint, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_{method}_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_microsam_v1(x, predictor, segmenter, ndim),
        dataset_name, data_root, ndim, save_path, desc=method, limit=limit,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--dataset_name", required=True, choices=(sorted(LM_DATASETS) + sorted(EM_DATASETS)))
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT, help="The root the data lives in.")
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=METHODS)
    parser.add_argument(
        "-m", "--model_type", type=str, default=None,
        help="Model type override, e.g. cyto3 for cellpose or vit_b for the micro-sam v1 methods."
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None,
        help="Checkpoint path for the micro-sam v1, segneuron and focus3d methods."
    )
    parser.add_argument(
        "--crop_3d", type=int, nargs=3, default=None,
        help="Override the 3d center crop. FOCUS-3D defaults to its own window depth."
    )
    parser.add_argument("--cell_radius", type=float, default=FOCUS3D_CELL_RADIUS, help="The xy radius to rescale from.")
    parser.add_argument(
        "--z_ratio", type=float, default=None,
        help="The z-to-xy spacing ratio. Defaults to the anisotropy."
    )
    parser.add_argument("--batch_size", type=int, default=FOCUS3D_BATCH_SIZE, help="Patches per forward pass.")
    parser.add_argument("--num_workers", type=int, default=FOCUS3D_NUM_WORKERS, help="Data loader workers.")
    parser.add_argument("--n_samples", type=int, default=None, help="Score only the first N samples, for a check.")
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.method == "cellpose":
        for model_type in ((args.model_type,) if args.model_type else CELLPOSE_MODELS):
            run_cellpose_evaluation(
                args.dataset_name, args.input_path, args.experiment_folder, model_type=model_type, device=device,
                limit=args.n_samples,
            )

    elif args.method == "stardist":
        run_stardist_evaluation(args.dataset_name, args.input_path, args.experiment_folder, limit=args.n_samples)

    elif args.method == "cellsam":
        run_cellsam_evaluation(args.dataset_name, args.input_path, args.experiment_folder, limit=args.n_samples)

    elif args.method in ("microsam_ais", "microsam_apg"):
        run_microsam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, method=args.method,
            model_type=args.model_type or SAM_V1_MODEL_TYPE, checkpoint=args.checkpoint, device=device,
            limit=args.n_samples,
        )

    elif args.method == "focus3d":
        run_focus3d_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            model_type=args.model_type or FOCUS3D_DEFAULT_MODEL, checkpoint=args.checkpoint, device=device,
            cell_radius=args.cell_radius, z_ratio=args.z_ratio,
            crop_shape=tuple(args.crop_3d) if args.crop_3d else None,
            batch_size=args.batch_size, num_workers=args.num_workers, limit=args.n_samples,
        )

    elif args.method == "segneuron":
        run_segneuron_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint, limit=args.n_samples,
        )

    else:
        raise ValueError(f"Unknown method: '{args.method}'.")


if __name__ == "__main__":
    main()
