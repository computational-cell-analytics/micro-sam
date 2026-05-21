"""Predict UniSAM2 distance outputs for snemi, nis3d, and plantseg root.

Stores raw input, ground-truth labels, and model distance predictions in H5
files for later grid search over postprocessing hyperparameters.

Each output H5 file contains:
    raw: (Z, Y, X) float32 input volume
    distances: (4, Z, Y, X) float32 model output (fg + 3 directed distances)
    labels: (Z, Y, X) uint32 ground-truth instance labels

Output layout:
    predictions/<dataset>/<model>/sample_000.h5

Usage:
    python create_preliminary_figures.py -d snemi -m automatic
    python create_preliminary_figures.py -d nis3d -m joint
    python create_preliminary_figures.py -d plantseg_root -m automatic --output_dir /path/to/out
"""

import argparse
import os
import sys
import types

import h5py
import numpy as np
import torch
from skimage.measure import label as connected_components
from tqdm import tqdm

from elf.io import open_file
from torch_em.data import datasets
from torch_em.transform.raw import normalize
from torch_em.util.image import load_image
from torch_em.util.prediction import predict_with_halo


DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"

CHECKPOINT_PATHS = {
    "automatic": os.path.join(
        "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2",
        "automatic/v1/checkpoints", "unisam2-both", "best.pt",
    ),
    "joint": os.path.join(
        "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2",
        "joint/v1/checkpoints", "joint_sam2_hvit_t_multi_gpu", "best.pt",
    ),
}

DATASETS = [
    "snemi", "nis3d", "plantseg_root", "cremi", "humanneurons",
    "plantseg_ovules", "pnas_arabidopsis", "celegans_atlas", "mitoem",
]

OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictions")


def _setup_module_aliases():
    """Register micro_sam2.* aliases so old checkpoints load without errors."""
    import micro_sam.v2.datasets.sampler as datasets_sampler
    import micro_sam.v2.datasets.wrapper as datasets_wrapper
    import micro_sam.v2.transforms.labels as transforms_labels
    import micro_sam.v2.transforms.raw as transforms_raw

    root = sys.modules.setdefault("micro_sam2", types.ModuleType("micro_sam2"))
    root.__path__ = []
    ds_mod = sys.modules.setdefault("micro_sam2.datasets", types.ModuleType("micro_sam2.datasets"))
    ds_mod.__path__ = []
    tr_mod = sys.modules.setdefault("micro_sam2.transforms", types.ModuleType("micro_sam2.transforms"))
    tr_mod.__path__ = []
    sys.modules["micro_sam2.datasets.sampler"] = datasets_sampler
    sys.modules["micro_sam2.datasets.wrapper"] = datasets_wrapper
    sys.modules["micro_sam2.transforms.labels"] = transforms_labels
    sys.modules["micro_sam2.transforms.raw"] = transforms_raw
    setattr(root, "datasets", ds_mod)
    setattr(root, "transforms", tr_mod)
    setattr(ds_mod, "sampler", datasets_sampler)
    setattr(ds_mod, "wrapper", datasets_wrapper)
    setattr(tr_mod, "labels", transforms_labels)
    setattr(tr_mod, "raw", transforms_raw)


def _build_unisam2(checkpoint_path, state_key, device):
    from micro_sam.v2.models.util import UniSAM2
    _setup_module_aliases()
    model = UniSAM2(encoder="hvit_t", output_channels=4)
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model.load_state_dict(state[state_key])
    model.to(device)
    model.eval()
    return model


def load_automatic_model(checkpoint_path, device):
    """Load UniSAM2 from a single-GPU automatic segmentation checkpoint."""
    return _build_unisam2(checkpoint_path, "model_state", device)


def load_joint_model(checkpoint_path, device):
    """Load the UniSAM2 automatic head from a joint training checkpoint.

    The joint checkpoint stores encoder weights under 'encoder.inner.*'
    (SAM2EncoderAdapter wraps the shared encoder as .inner). Standalone
    UniSAM2(encoder="hvit_t") expects 'encoder.*' directly, so we remap.
    """
    from micro_sam.v2.models.util import UniSAM2
    _setup_module_aliases()
    model = UniSAM2(encoder="hvit_t", output_channels=4)
    state = torch.load(checkpoint_path, weights_only=False, map_location=device)
    raw = state["unetr_state"]
    remapped = {
        k.replace("encoder.inner.", "encoder.", 1) if k.startswith("encoder.inner.") else k: v
        for k, v in raw.items()
    }
    model.load_state_dict(remapped)
    model.to(device)
    model.eval()
    return model


def predict_volume(model, raw, device):
    """Run tiled UniSAM2 distance prediction over a full 3D volume.

    Args:
        model: UniSAM2 instance in eval mode.
        raw: (Z, Y, X) float32 input volume.
        device: torch device string.

    Returns:
        (4, Z, Y, X) float32 - foreground probability and 3 directed distances.
    """
    def _preprocess(crop):
        return np.concatenate([normalize(crop)] * 3, axis=0)

    input_ = raw[np.newaxis].astype("float32")
    out = np.zeros((4, *raw.shape), dtype="float32")
    out = predict_with_halo(
        input_=input_,
        model=model,
        block_shape=(4, 384, 384),
        halo=(2, 64, 64),
        preprocess=_preprocess,
        gpu_ids=[device],
        output=out,
        with_channels=True,
    )
    return out


def _iter_volumes(dataset_name, data_root, max_samples=None):
    """Yield (sample_id, raw, labels) for each volume in the test split.

    Args:
        dataset_name: one of DATASETS.
        data_root: root data directory.
        max_samples: stop after this many volumes; None for all.

    Yields:
        (sample_id, raw, labels): zero-padded string ID, float32 (Z, Y, X) raw,
            uint32 (Z, Y, X) instance labels.
    """
    p = data_root

    n = 0

    if dataset_name == "snemi":
        path = datasets.snemi.get_snemi_paths(
            path=os.path.join(p, "snemi"), sample="train", download=False,
        )
        with open_file(path, mode="r") as f:
            raw = f["volumes/raw"][:70].astype("float32")
            labels = connected_components(f["volumes/labels/neuron_ids"][:70]).astype("uint32")
        yield "sample_000", raw, labels

    elif dataset_name == "nis3d":
        img_paths, gt_paths = datasets.nis3d.get_nis3d_paths(
            path=os.path.join(p, "nis3d"), split="test",
            split_type="cross-image", download=False,
        )
        for idx, (ip, gp) in enumerate(zip(sorted(img_paths), sorted(gt_paths))):
            raw = load_image(ip).astype("float32")
            labels = connected_components(load_image(gp)).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    elif dataset_name == "plantseg_root":
        paths = datasets.plantseg.get_plantseg_paths(
            path=os.path.join(p, "plantseg_root"), name="root",
            split="test", download=False,
        )
        for idx, path in enumerate(sorted(paths)):
            with open_file(path, mode="r") as f:
                raw = f["raw"][:].astype("float32")
                labels = connected_components(f["label"][:]).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    elif dataset_name == "cremi":
        paths = datasets.cremi.get_cremi_paths(
            path=os.path.join(p, "cremi"), samples=("A", "B", "C"), download=False,
        )
        for idx, path in enumerate(sorted(paths)):
            with open_file(path, mode="r") as f:
                raw = f["volumes/raw"][:].astype("float32")
                labels = connected_components(f["volumes/labels/neuron_ids"][:]).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    elif dataset_name == "humanneurons":
        paths = datasets.humanneurons.get_humanneurons_paths(
            path=os.path.join(p, "humanneurons"), download=False,
        )
        for idx, path in enumerate(sorted(paths)):
            with open_file(path, mode="r") as f:
                raw = f["raw"][:].astype("float32")
                labels = connected_components(f["labels"][:]).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    elif dataset_name == "plantseg_ovules":
        paths = datasets.plantseg.get_plantseg_paths(
            path=os.path.join(p, "plantseg_ovules"), name="ovules",
            split="test", download=False,
        )
        # Use N_435 as the standard test volume (matches v1 evaluation holdout).
        target = "N_435_final_crop_ds2.h5"
        paths = [path for path in paths if os.path.basename(path) == target] or sorted(paths)
        for idx, path in enumerate(paths):
            with open_file(path, mode="r") as f:
                raw = f["raw"][:].astype("float32")
                labels = connected_components(f["label"][:]).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    elif dataset_name == "pnas_arabidopsis":
        paths = datasets.pnas_arabidopsis.get_pnas_arabidopsis_paths(
            path=os.path.join(p, "pnas_arabidopsis"), download=False,
        )
        for idx, path in enumerate(sorted(paths)):
            with open_file(path, mode="r") as f:
                raw = f["raw"][:].astype("float32")
                labels = connected_components(f["labels"][:]).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    elif dataset_name == "celegans_atlas":
        img_paths, gt_paths = datasets.celegans_atlas.get_celegans_atlas_paths(
            path=os.path.join(p, "celegans_atlas"), split="test", download=False,
        )
        for idx, (ip, gp) in enumerate(zip(sorted(img_paths), sorted(gt_paths))):
            raw = load_image(ip).astype("float32")
            labels = connected_components(load_image(gp)).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    elif dataset_name == "mitoem":
        paths = datasets.mitoem.get_mitoem_paths(
            path=os.path.join(p, "mitoem"), splits=["val"], download=False,
        )
        for idx, path in enumerate(sorted(paths)):
            with open_file(path, mode="r") as f:
                raw = f["raw"][:].astype("float32")
                labels = connected_components(f["labels"][:]).astype("uint32")
            yield f"sample_{idx:03d}", raw, labels
            n += 1
            if max_samples is not None and n >= max_samples:
                return

    else:
        raise ValueError(f"Unknown dataset: {dataset_name!r}")


def run_prediction(dataset_name, model_name, checkpoint_path, output_dir, device, max_samples=None):
    """Run and cache predictions for one (dataset, model) pair.

    Args:
        dataset_name: dataset to predict on.
        model_name: 'automatic' or 'joint'.
        checkpoint_path: path to model checkpoint.
        output_dir: root directory for H5 output files.
        device: torch device string.
        max_samples: stop after this many volumes; None for all.
    """
    save_dir = os.path.join(output_dir, dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading model '{model_name}' from {checkpoint_path}")
    if model_name == "automatic":
        model = load_automatic_model(checkpoint_path, device)
    else:
        model = load_joint_model(checkpoint_path, device)

    volumes = list(_iter_volumes(dataset_name, DATA_ROOT, max_samples=max_samples))
    for sample_id, raw, labels in tqdm(volumes, desc=f"{dataset_name}/{model_name}"):
        save_path = os.path.join(save_dir, f"{sample_id}.h5")
        if os.path.exists(save_path):
            print(f"  Skipping {save_path} (already exists).")
            continue

        print(f"  Predicting {sample_id} shape={raw.shape} ...")
        distances = predict_volume(model, raw, device)

        with h5py.File(save_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("distances", data=distances, compression="gzip")
            f.create_dataset("labels", data=labels, compression="gzip")

        print(f"  Saved {save_path}.")


def main():
    parser = argparse.ArgumentParser(
        description="Cache UniSAM2 distance predictions on snemi, nis3d, and plantseg root."
    )
    parser.add_argument(
        "-d", "--dataset", required=True, choices=DATASETS,
        help="Dataset to run prediction on.",
    )
    parser.add_argument(
        "-m", "--model", required=True, choices=list(CHECKPOINT_PATHS),
        help="Model variant: 'automatic' (single-GPU) or 'joint' (multi-GPU).",
    )
    parser.add_argument(
        "-c", "--checkpoint", type=str, default=None,
        help="Override the default checkpoint path.",
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default=OUTPUT_ROOT,
        help="Root directory to write H5 prediction files.",
    )
    parser.add_argument(
        "-n", "--max_samples", type=int, default=None,
        help="Stop after this many volumes per dataset (default: all).",
    )
    args = parser.parse_args()

    ckpt = args.checkpoint or CHECKPOINT_PATHS[args.model]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_label = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"Device: {device_label}")
    print(f"Dataset: {args.dataset} | Model: {args.model}")
    print(f"Checkpoint: {ckpt}")

    run_prediction(args.dataset, args.model, ckpt, args.output_dir, device, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
