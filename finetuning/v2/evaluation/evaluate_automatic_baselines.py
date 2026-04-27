"""Benchmark evaluation of automatic segmentation baselines.

Supported methods:
  cellpose:    CellPose3 generalist models (cyto3, cpsam)
  stardist:    StarDist pretrained (2D_versatile_fluo / 3D_demo)
  cellsam:     CellSAM pipeline (2D-only)
  sam:         Pretrained SAM v1 with Automatic Mask Generation (AMG)
  sam2:        Pretrained SAM2 with Automatic Mask Generation (AMG)
  micro-sam:   micro-sam v1 finetuned (vit_b_lm for LM, vit_b_em_organelles_v1 for EM)
  micro-samv2: UniSAM2 finetuned (directed distance model)
  microsam_amg: micro-sam v1 Automatic Mask Generation
  microsam_ais: micro-sam v1 Automatic Instance Segmentation
  microsam_apg: micro-sam v1 Automatic Prompt Generation
  segneuron:   SegNeuron (3D EM only)

Usage examples:
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellpose -m cyto3
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method stardist
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method cellsam
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method sam
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method sam2
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method micro-sam
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method micro-samv2
    python evaluate_automatic_baselines.py -d livecell -e <exp> --method microsam_amg -m vit_b
    python evaluate_automatic_baselines.py -d embedseg -e <exp> --method microsam_ais -m vit_b
    python evaluate_automatic_baselines.py -d lucchi   -e <exp> --method segneuron
"""

import os

import numpy as np
from tqdm import tqdm

import torch

from micro_sam.evaluation.evaluation import run_evaluation

from common import (
    DATA_ROOT, DATASETS_2D, DATASETS_3D, DATASETS_3D_LM, DATASETS_3D_EM,
    CHECKPOINT_PATHS, UNISAM2_CHECKPOINT,
    load_unisam2_model, predict_unisam2, postprocess_unisam2,
    get_data_paths,
)
from baselines_common import _load_data

_LM_DATASETS = set(DATASETS_2D + DATASETS_3D_LM)
_EM_DATASETS = set(DATASETS_3D_EM)
_METHODS = [
    "cellpose", "stardist", "cellsam",
    "sam", "sam2", "micro-sam", "micro-samv2",
    "microsam_amg", "microsam_ais", "microsam_apg",
    "segneuron",
]

_SEGNEURON_ROOT = "/mnt/vast-nhr/home/archit/u12090/SegNeuron"
SEGNEURON_CHECKPOINT = "/mnt/vast-nhr/projects/cidas/cca/models/segneuron/SegNeuronModel.ckpt"

_STARDIST_2D_MODEL = "2D_versatile_fluo"
_STARDIST_3D_MODEL = "3D_demo"

# SAM2 defaults (used for sam2 and micro-samv2 interactive backbone)
_SAM2_BACKBONE = "sam2.1"
_SAM2_MODEL_TYPE = "hvit_t"

# micro-sam v1 model types
_SAM_V1_MODEL_TYPE = "vit_b"
_MICROSAM_V1_LM_MODEL = "vit_b_lm"
_MICROSAM_V1_EM_MODEL = "vit_b_em_organelles_v1"

# Per-dataset z/xy anisotropy for CellPose do_3D mode (z_voxel / xy_voxel).
_DATASET_ANISOTROPY = {
    "embedseg": 4.0,
    "blastospim": 10.0,
    "mouse_embryo": 4.0,
    "cremi": 10.0,   # z=40nm, xy=4nm
    "snemi": 5.0,    # z=30nm, xy=6nm
}


def _load_cellpose(model_type, device):
    from cellpose import models
    use_gpu = (device != "cpu") and torch.cuda.is_available()
    if model_type in ("cyto", "cyto2", "cyto3", "nuclei"):
        return models.Cellpose(gpu=use_gpu, model_type=model_type)
    elif model_type == "cpsam":
        return models.CellposeModel(gpu=use_gpu)
    else:
        return models.CellposeModel(gpu=use_gpu, model_type=model_type)


def _load_stardist(ndim):
    from stardist.models import StarDist2D, StarDist3D
    if ndim == 3:
        return StarDist3D.from_pretrained(_STARDIST_3D_MODEL)
    return StarDist2D.from_pretrained(_STARDIST_2D_MODEL)


def _load_segneuron(checkpoint_path, device):
    import sys
    sys.path.insert(0, os.path.join(_SEGNEURON_ROOT, "Train_and_Inference"))
    sys.path.insert(0, os.path.join(_SEGNEURON_ROOT, "Postprocess"))
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
    from micro_sam.automatic_segmentation import get_predictor_and_segmenter
    mode = {"microsam_amg": "amg", "microsam_ais": "ais", "microsam_apg": "apg"}[method]
    return get_predictor_and_segmenter(
        model_type=model_type, checkpoint=checkpoint, device=device, segmentation_mode=mode,
    )


def _load_sam2_amg(model_type, backbone, checkpoint_path, device):
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from micro_sam.v2.util import get_sam2_model
    model = get_sam2_model(
        model_type=model_type, device=device, checkpoint_path=checkpoint_path, backbone=backbone,
    )
    return SAM2AutomaticMaskGenerator(model, pred_iou_thresh=0.6, stability_score_thresh=0.6)


def _segment_cellpose(image_or_volume, model, ndim, dataset_name=None):
    if ndim == 3:
        anisotropy = _DATASET_ANISOTROPY.get(dataset_name, None)
        masks = model.eval(image_or_volume, diameter=None, channels=[0, 0], do_3D=True, anisotropy=anisotropy)[0]
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


def _segment_cellsam(image, ndim):
    if ndim == 3:
        raise ValueError("CellSAM is 2D-only and does not support 3D input.")
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
    sys.path.insert(0, os.path.join(_SEGNEURON_ROOT, "Postprocess"))
    from FRMC_post import post_mc

    raw = volume.astype("float32") / 255.0 if volume.max() > 1.0 else volume.astype("float32")
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
    from micro_sam.automatic_segmentation import automatic_instance_segmentation
    from micro_sam.instance_segmentation import AutomaticPromptGenerator
    if isinstance(segmenter, AutomaticPromptGenerator):
        segmenter.initialize(image_or_volume, ndim=ndim)
        seg = segmenter.generate()
    else:
        seg = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter,
            input_path=image_or_volume, ndim=ndim, verbose=False,
        )
    return seg.astype("uint32") if seg is not None else np.zeros(image_or_volume.shape, dtype="uint32")


def _segment_sam2_amg(image_or_volume, amg, ndim):
    from micro_sam.util import mask_data_to_segmentation
    if ndim == 3:
        seg = np.zeros(image_or_volume.shape, dtype="uint32")
        offset = 0
        for z in range(image_or_volume.shape[0]):
            frame = image_or_volume[z].astype("uint8")
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            outputs = amg.generate(frame)
            if outputs:
                frame_seg = mask_data_to_segmentation(outputs, with_background=True, min_object_size=0)
                frame_seg[frame_seg > 0] += offset
                offset = int(frame_seg.max())
                seg[z] = frame_seg
        return seg
    else:
        image = image_or_volume.astype("uint8")
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        outputs = amg.generate(image)
        if not outputs:
            return np.zeros(image.shape[:2], dtype="uint32")
        return mask_data_to_segmentation(outputs, with_background=True, min_object_size=0)


def _run_evaluation(segment_fn, dataset_name, data_root, ndim, save_path, desc):
    if os.path.exists(save_path):
        print(f"Results already stored at '{save_path}'.")
        return

    n = len(get_data_paths(dataset_name, data_root)[0])
    all_gt, all_seg = [], []
    for image_or_volume, labels in tqdm(_load_data(dataset_name, data_root, ndim), total=n, desc=desc):
        seg = segment_fn(image_or_volume)
        all_gt.append(labels)
        all_seg.append(seg)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    results = run_evaluation(gt_paths=all_gt, prediction_paths=all_seg, save_path=save_path)
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
        raise ValueError(f"CellSAM is 2D-only and does not support 3D dataset '{dataset_name}'.")
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_cellsam.csv")
    _run_evaluation(
        lambda x: _segment_cellsam(x, ndim=2),
        dataset_name, data_root, ndim=2, save_path=save_path, desc="cellsam",
    )


def run_segneuron_evaluation(dataset_name, data_root, experiment_folder, device, checkpoint_path=None):
    if dataset_name not in _EM_DATASETS:
        raise ValueError(f"SegNeuron is EM-only; got '{dataset_name}'.")
    if checkpoint_path is None:
        checkpoint_path = SEGNEURON_CHECKPOINT
    model = _load_segneuron(checkpoint_path, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_segneuron.csv")
    _run_evaluation(
        lambda x: _segment_segneuron(x, model, device),
        dataset_name, data_root, ndim=3, save_path=save_path, desc="segneuron",
    )


def run_microsam_v1_evaluation(dataset_name, data_root, experiment_folder, method, model_type, checkpoint, device):
    ndim = 3 if dataset_name in DATASETS_3D else 2
    predictor, segmenter = _load_microsam_v1(method, model_type, checkpoint, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_{method}_{model_type}.csv")
    _run_evaluation(
        lambda x: _segment_microsam_v1(x, predictor, segmenter, ndim),
        dataset_name, data_root, ndim, save_path, desc=method,
    )


def run_sam2_auto_evaluation(
    dataset_name, data_root, experiment_folder, device,
    model_type=_SAM2_MODEL_TYPE, backbone=_SAM2_BACKBONE, checkpoint_path=None,
):
    ndim = 3 if dataset_name in DATASETS_3D else 2
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINT_PATHS[backbone][model_type]
    amg = _load_sam2_amg(model_type, backbone, checkpoint_path, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_sam2_{model_type}_amg.csv")
    _run_evaluation(
        lambda x: _segment_sam2_amg(x, amg, ndim),
        dataset_name, data_root, ndim, save_path, desc=f"sam2-amg-{model_type}",
    )


def run_microsam2_auto_evaluation(dataset_name, data_root, experiment_folder, device, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_path = UNISAM2_CHECKPOINT
    ndim = 3 if dataset_name in DATASETS_3D else 2
    model = load_unisam2_model(checkpoint_path, device)
    save_path = os.path.join(experiment_folder, "results", f"{dataset_name}_microsam2_auto.csv")
    _run_evaluation(
        lambda x: postprocess_unisam2(predict_unisam2(model, x, ndim, device), dataset_name),
        dataset_name, data_root, ndim, save_path, desc="microsam2-auto",
    )


def main():
    import argparse
    all_datasets = sorted(_LM_DATASETS) + sorted(_EM_DATASETS)
    parser = argparse.ArgumentParser(description="Evaluate automatic segmentation baselines.")
    parser.add_argument("-d", "--dataset_name", required=True, choices=all_datasets)
    parser.add_argument("-i", "--input_path", type=str, default=DATA_ROOT)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("--method", type=str, required=True, choices=_METHODS)
    parser.add_argument("-m", "--model_type", type=str, default=None,
                        help="Model type override (e.g. cyto3 for cellpose, vit_b for sam, hvit_t for sam2).")
    parser.add_argument("-c", "--checkpoint", type=str, default=None,
                        help="Checkpoint path for micro-sam v1 / segneuron / micro-samv2 / sam2 methods.")
    args = parser.parse_args()

    print("Device:", torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.method == "cellpose":
        for mt in ((args.model_type,) if args.model_type else ("cyto3", "cpsam")):
            run_cellpose_evaluation(
                args.dataset_name, args.input_path, args.experiment_folder,
                model_type=mt, device=device,
            )
    elif args.method == "stardist":
        run_stardist_evaluation(args.dataset_name, args.input_path, args.experiment_folder)
    elif args.method == "cellsam":
        run_cellsam_evaluation(args.dataset_name, args.input_path, args.experiment_folder)
    elif args.method == "sam":
        mt = args.model_type or _SAM_V1_MODEL_TYPE
        run_microsam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            method="microsam_amg", model_type=mt, checkpoint=args.checkpoint, device=device,
        )
    elif args.method == "sam2":
        mt = args.model_type or _SAM2_MODEL_TYPE
        run_sam2_auto_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, model_type=mt, backbone=_SAM2_BACKBONE, checkpoint_path=args.checkpoint,
        )
    elif args.method == "micro-sam":
        is_em = args.dataset_name in _EM_DATASETS
        mt = args.model_type or (_MICROSAM_V1_EM_MODEL if is_em else _MICROSAM_V1_LM_MODEL)
        run_microsam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            method="microsam_amg", model_type=mt, checkpoint=args.checkpoint, device=device,
        )
    elif args.method == "micro-samv2":
        run_microsam2_auto_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint,
        )
    elif args.method in ("microsam_amg", "microsam_ais", "microsam_apg"):
        run_microsam_v1_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            method=args.method, model_type=args.model_type or "vit_b",
            checkpoint=args.checkpoint, device=device,
        )
    elif args.method == "segneuron":
        run_segneuron_evaluation(
            args.dataset_name, args.input_path, args.experiment_folder,
            device=device, checkpoint_path=args.checkpoint,
        )


if __name__ == "__main__":
    main()
