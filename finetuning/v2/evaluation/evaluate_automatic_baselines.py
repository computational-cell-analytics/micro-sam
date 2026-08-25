"""Benchmark evaluation of the automatic segmentation baselines, i.e. everything but micro-sam2.

Supported methods:
  cellpose: CellPose generalist models (cyto3, cpsam)
  stardist: StarDist pretrained (2D_versatile_fluo / 3D_demo)
  cellsam: CellSAM pipeline (2d only)
  microsam_ais: micro-sam v1 automatic instance segmentation
  microsam_apg: micro-sam v1 automatic prompt generation
  segneuron: SegNeuron (3d EM only)

micro-sam2 itself is evaluated by evaluate_automatic_segmentation.py, which also tunes it.

Usage examples:
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellpose -m cyto3
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method stardist
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellsam
    python evaluate_automatic_baselines.py -d embedseg -e <exp> --method microsam_ais -m vit_b
    python evaluate_automatic_baselines.py -d cremi -e <exp> --method segneuron
"""

import os
import warnings
import argparse

import numpy as np
from tqdm import tqdm

import torch

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_LM, DATASETS_3D_EM,
    GT_MIN_SIZE_2D, check_data_download, drop_severed_objects, load_data, n_samples,
    run_dataset_evaluation,
)

LM_DATASETS = set(DATASETS_2D + DATASETS_3D_LM)
EM_DATASETS = set(DATASETS_3D_EM)
METHODS = ["cellpose", "stardist", "cellsam", "microsam_ais", "microsam_apg", "segneuron"]

SEGNEURON_ROOT = "/mnt/vast-nhr/home/archit/u12090/SegNeuron"
SEGNEURON_CHECKPOINT = "/mnt/vast-nhr/projects/cidas/cca/models/segneuron/SegNeuronModel.ckpt"

STARDIST_2D_MODEL = "2D_versatile_fluo"
STARDIST_3D_MODEL = "3D_demo"

# micro-sam v1 model types
SAM_V1_MODEL_TYPE = "vit_b_lm"

# Per-dataset z/xy anisotropy for CellPose do_3D mode (z_voxel / xy_voxel).
DATASET_ANISOTROPY = {
    "embedseg": 4.0,
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
    sys.path.insert(0, os.path.join(SEGNEURON_ROOT, "Postprocess"))
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


def _segment_segneuron(volume, model, device, beta=0.25):
    import sys
    import torch
    sys.path.insert(0, os.path.join(SEGNEURON_ROOT, "Postprocess"))
    from FRMC_post import post_mc

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
    return post_mc(combined, beta=beta).astype("uint32")


def _segment_microsam_v1(image_or_volume, predictor, segmenter, ndim):
    from micro_sam.v1.automatic_segmentation import automatic_instance_segmentation
    seg = automatic_instance_segmentation(
        predictor=predictor, segmenter=segmenter, input_path=image_or_volume, ndim=ndim, verbose=False,
    )
    return seg.astype("uint32") if seg is not None else np.zeros(image_or_volume.shape, dtype="uint32")


def _run_evaluation(segment_fn, dataset_name, data_root, ndim, save_path, desc):
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return

    total = n_samples(dataset_name, data_root)
    all_gt, all_seg = [], []
    samples = load_data(dataset_name, data_root, ndim)
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


def run_cellpose_evaluation(dataset_name, data_root, experiment_folder, model_type, device):
    ndim = 3 if dataset_name in DATASETS_3D else 2
    model = _load_cellpose(model_type, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellpose_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_cellpose(x, model, ndim, dataset_name),
        dataset_name, data_root, ndim, save_path, desc=f"cellpose-{model_type}",
    )


def run_stardist_evaluation(dataset_name, data_root, experiment_folder):
    ndim = 3 if dataset_name in DATASETS_3D else 2
    model = _load_stardist(ndim)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_stardist.csv")
    _run_evaluation(
        lambda x: _segment_stardist(x, model, ndim),
        dataset_name, data_root, ndim, save_path, desc="stardist",
    )


def run_cellsam_evaluation(dataset_name, data_root, experiment_folder):
    if dataset_name in DATASETS_3D:
        warnings.warn(
            f"CellSAM is 2D-only and does not support 3D dataset '{dataset_name}'. Skipping.",
            UserWarning, stacklevel=2,
        )
        return

    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellsam.csv")
    _run_evaluation(
        lambda x: _segment_cellsam(x),
        dataset_name, data_root, ndim=2, save_path=save_path, desc="cellsam",
    )


def run_segneuron_evaluation(dataset_name, data_root, experiment_folder, device, checkpoint_path=None):
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
        dataset_name, data_root, ndim=3, save_path=save_path, desc="segneuron",
    )


def run_microsam_v1_evaluation(dataset_name, data_root, experiment_folder, method, model_type, checkpoint, device):
    if dataset_name in EM_DATASETS:
        raise ValueError(f"micro-sam v1 automatic methods do not support EM datasets; got '{dataset_name}'.")
    ndim = 3 if dataset_name in DATASETS_3D else 2
    predictor, segmenter = _load_microsam_v1(method, model_type, checkpoint, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_{method}_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_microsam_v1(x, predictor, segmenter, ndim),
        dataset_name, data_root, ndim, save_path, desc=method,
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
        help="Checkpoint path for the micro-sam v1 and segneuron methods."
    )
    args = parser.parse_args()

    check_data_download(args.dataset_name, args.input_path)

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.method == "cellpose":
        for model_type in ((args.model_type,) if args.model_type else ("cyto3", "cpsam")):
            run_cellpose_evaluation(
                args.dataset_name, args.input_path, args.experiment_folder, model_type=model_type, device=device,
            )

    elif args.method == "stardist":
        run_stardist_evaluation(args.dataset_name, args.input_path, args.experiment_folder)

    elif args.method == "cellsam":
        run_cellsam_evaluation(args.dataset_name, args.input_path, args.experiment_folder)

    elif args.method in ("microsam_ais", "microsam_apg"):
        run_microsam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder, method=args.method,
            model_type=args.model_type or SAM_V1_MODEL_TYPE, checkpoint=args.checkpoint, device=device,
        )

    elif args.method == "segneuron":
        run_segneuron_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint,
        )

    else:
        raise ValueError(f"Unknown method: '{args.method}'.")


if __name__ == "__main__":
    main()
