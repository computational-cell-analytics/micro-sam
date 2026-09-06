"""Building blocks of the AIS decoder training campaign (2026-09).

Decoder-only training of the UniSAM2 automatic branch: the SAM2 image encoder stays frozen at the weights of
the joint/v4 geodesic checkpoint, the UNETR decoder is warm-started from the same checkpoint and trained on
the train splits of the AIS tuning datasets. Everything the trainer pickles into its checkpoints (datasets,
wrappers, the model class) lives in this importable module, so that the staging step can re-open a
checkpoint from another process.

Four variants: 'baseline' (the current four channel target and loss), 'contact' (a fifth output channel
trained on the touching boundaries), 'fgcal' (the foreground trained with Dice plus a boundary-weighted
cross entropy) and 'both'.

Datasets: livecell, tissuenet, dynamicnuclearnet, deepbacs, dic_hepg2, neurips_cellseg, yeaz, puma, tnbc
(train splits). deepseas is excluded (binary masks: connected components merge touching cells, which would
corrupt the contact target and the geodesic field) and so is covid_if (no split; 44 of its 49 files are
production-scored). The trainer's validation set is a deterministic tail of every train file list, so the
evaluation manifests (val splits) and the test splits stay untouched.
"""

import math
import os
from functools import partial
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch_em
from torch_em.data import ConcatDataset, MinInstanceSampler, datasets

from micro_sam.v2.datasets.generalist_loader import _configure_training_normalization, _prepare_data_loader
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.models.util import UniSAM2
from micro_sam.v2.transforms.labels import GeodesicHybridDistanceTransform
from micro_sam.v2.transforms.raw import _identity, _normalize_percentile, _to_8bit

DATA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"
CAMPAIGN_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/micro_sam2/apg_optimization/ais_decoder_training"
V4_CHECKPOINT = (
    "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v4/checkpoints/"
    "joint_sam2_hvit_t_geodesic_multi_gpu/best.pt"
)
MODEL_TYPE = "hvit_t"
INITIAL_FEATURES = 32
PATCH_SHAPE = (512, 512)
VAL_FRACTION = 0.05
MIN_VAL_FILES = 2

# The loss settings of the four variants.
VARIANTS: Dict[str, Dict] = {
    "baseline": {"contact": False, "boundary_weight": None},
    "contact": {"contact": True, "boundary_weight": None},
    "fgcal": {"contact": False, "boundary_weight": 4.0},
    "both": {"contact": True, "boundary_weight": 4.0},
}
BOUNDARY_RADIUS = 2

# Samples per epoch (before scaling) and validation samples per dataset group.
TRAIN_SAMPLES = {
    "livecell": 25,  # per cell type, eight types
    "tissuenet": 200, "dynamicnuclearnet": 200, "neurips_cellseg": 150, "dic_hepg2": 120, "deepbacs": 120,
    "yeaz_bf": 80, "yeaz_phc": 20, "yeaz_phc_stacks": 20, "puma": 100, "tnbc": 60,
}
VAL_SAMPLES = {"livecell": 3, "yeaz_phc": 2, "yeaz_phc_stacks": 2, "tnbc": 2}
DEFAULT_VAL_SAMPLES = 20


def n_output_channels(variant: str) -> int:
    return 4 + int(VARIANTS[variant]["contact"])


# ----------------------------------------------------------------------------------------------
# model


class FrozenEncoderUniSAM2(UniSAM2):
    """UniSAM2 whose image encoder stays in eval mode while the decoder trains.

    The Hiera encoder has neither dropout nor batch norm, so this is hygiene rather than a numerical
    necessity; the freezing itself is done by `freeze_encoder`.
    """

    def train(self, mode: bool = True):
        super().train(mode)
        self.encoder.eval()
        return self


def freeze_encoder(model: torch.nn.Module) -> None:
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    model.encoder.eval()


def decoder_parameters(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """The trainable parameters: everything outside the encoder (the filter `train_joint_sam2` uses)."""
    return [p for name, p in model.named_parameters() if not name.startswith("encoder")]


def _alias_legacy_modules() -> None:
    """Make the module paths pickled into old joint checkpoints importable."""
    import sys
    evaluation_dir = os.path.join(os.path.dirname(__file__), "..", "..", "evaluation")
    sys.path.insert(0, os.path.abspath(evaluation_dir))
    import common  # noqa: F401  (registers the aliases on import when needed)
    if hasattr(common, "_alias_micro_sam2_modules"):
        common._alias_micro_sam2_modules()


def load_lean_v4_states(v4_checkpoint: str = V4_CHECKPOINT, cache_dir: str = CAMPAIGN_ROOT) -> Dict[str, Dict]:
    """The 'model_state' (SAM2) and 'unetr_state' (UniSAM2) of the v4 joint checkpoint, without the pickled
    trainer state. Cached as a lean file, because the full checkpoint takes minutes to unpickle."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "v4_lean_states.pt")
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    try:
        state = torch.load(v4_checkpoint, map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, AttributeError):
        _alias_legacy_modules()
        state = torch.load(v4_checkpoint, map_location="cpu", weights_only=False)

    def strip(state_dict):
        return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}

    lean = {"model_state": strip(state["model_state"]), "unetr_state": strip(state["unetr_state"])}
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    torch.save(lean, tmp_path)
    os.replace(tmp_path, cache_path)
    return lean


def build_model(variant: str, device, unetr_state: Optional[Dict[str, torch.Tensor]]) -> torch.nn.Module:
    """Build the (frozen encoder) UniSAM2 for a variant, warm-started from the v4 decoder state if given.

    A five channel decoder takes the four pretrained output rows and keeps the fresh initialisation of the
    contact row.
    """
    model = FrozenEncoderUniSAM2(
        encoder=MODEL_TYPE, output_channels=n_output_channels(variant), initial_features=INITIAL_FEATURES,
        device=device,
    )
    if unetr_state is not None:
        state = dict(unetr_state)
        if model.out_channels == state["out_conv.weight"].shape[0]:
            model.load_state_dict(state, strict=True)
        else:
            weight, bias = state.pop("out_conv.weight"), state.pop("out_conv.bias")
            missing, unexpected = model.load_state_dict(state, strict=False)
            assert sorted(missing) == ["out_conv.bias", "out_conv.weight"] and not unexpected, (missing, unexpected)
            with torch.no_grad():
                model.out_conv.weight[:weight.shape[0]].copy_(weight)
                model.out_conv.bias[:bias.shape[0]].copy_(bias)
    freeze_encoder(model)
    return model


# ----------------------------------------------------------------------------------------------
# data


class RandomSubsetDataset(torch.utils.data.Dataset):
    """A fixed number of random draws from a dataset.

    torch_em splits 'n_samples' uniformly over the files of a segmentation dataset, so a small sample count
    over many files would only ever read the first files. This wrapper draws a random index per access
    instead. It exposes 'datasets' so the normalization configuration recurses into the wrapped dataset.
    """

    def __init__(self, dataset, n_samples: int):
        self.datasets = (dataset,)
        self.n_samples = int(n_samples)
        self.ndim = getattr(dataset, "ndim", 2)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.datasets[0][np.random.randint(len(self.datasets[0]))]


def _sorted_pairs(raw_paths: Sequence[str], label_paths: Sequence[str]) -> Tuple[List[str], List[str]]:
    if len(raw_paths) != len(label_paths):
        raise RuntimeError(f"Expect as many raw as label paths, got {len(raw_paths)} and {len(label_paths)}.")
    pairs = sorted(zip(raw_paths, label_paths), key=lambda pair: str(pair[0]))
    return [str(p[0]) for p in pairs], [str(p[1]) for p in pairs]


def split_tail(paths: Sequence[str], fraction: float = VAL_FRACTION, minimum: int = MIN_VAL_FILES):
    """Deterministic train / validation split: the tail of the (already sorted) list is the validation set."""
    n_val = max(minimum, int(math.ceil(fraction * len(paths))))
    if n_val >= len(paths):
        raise RuntimeError(f"Cannot hold out {n_val} of {len(paths)} files.")
    return list(paths[:-n_val]), list(paths[-n_val:])


def _common_kwargs(label_transform, sampler=None):
    return {
        "patch_shape": PATCH_SHAPE,
        "label_transform2": label_transform,
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]) if sampler is None else sampler,
        "label_dtype": torch.float32,
        "ndim": 2,
    }


def _image_dataset(raw_paths, label_paths, kwargs, raw_transform, n_samples):
    """Image / label file pairs (tif, png, ...); torch_em draws a random file per sample when n_samples is set."""
    return torch_em.default_segmentation_dataset(
        raw_paths=raw_paths, raw_key=None, label_paths=label_paths, label_key=None, is_seg_dataset=False,
        raw_transform=raw_transform, n_samples=n_samples, **kwargs,
    )


def _container_dataset(paths, raw_key, label_key, kwargs, raw_transform, with_channels, patch_shape=None):
    """zarr / h5 / tif-stack files read with keys; one patch per file, randomised by `RandomSubsetDataset`."""
    kwargs = dict(kwargs)
    if patch_shape is not None:
        kwargs["patch_shape"] = patch_shape
    return torch_em.default_segmentation_dataset(
        raw_paths=paths, raw_key=raw_key, label_paths=paths, label_key=label_key, is_seg_dataset=True,
        with_channels=with_channels, raw_transform=raw_transform, n_samples=None, **kwargs,
    )


def _wrap(dataset, n_samples: Optional[int], is_val: bool, randomise: bool):
    """Training leaves draw 'n_samples' random samples per epoch; validation leaves read their first samples."""
    if is_val:
        return UniDataWrapper(dataset, source_ndim=2, max_samples=n_samples)
    if randomise:
        return UniDataWrapper(RandomSubsetDataset(dataset, n_samples), source_ndim=2)
    return UniDataWrapper(dataset, source_ndim=2)


def _train_count(name: str, scale: float) -> int:
    return max(1, int(round(TRAIN_SAMPLES[name] * scale)))


def _val_count(name: str) -> int:
    return VAL_SAMPLES.get(name, DEFAULT_VAL_SAMPLES)


def _tif_is_stack(path: str) -> bool:
    import tifffile
    with tifffile.TiffFile(path) as f:
        return len(f.series[0].shape) == 3


def build_datasets(
    data_root: str, label_transform, scale: float = 1.0,
) -> Tuple[List[UniDataWrapper], List[UniDataWrapper], Dict[str, Dict[str, List[str]]]]:
    """The training and validation leaves of the nine datasets and the file lists behind them."""
    kwargs = _common_kwargs(label_transform)
    train_leaves, val_leaves, manifest = [], [], {}

    def record(name, train_raw, val_raw):
        manifest[name] = {"train": list(map(str, train_raw)), "val": list(map(str, val_raw))}

    def add_images(name, raw, labels, raw_transform, sampler_kwargs=None, count_name=None):
        count_name = count_name or name
        raw, labels = _sorted_pairs(raw, labels)
        train_raw, val_raw = split_tail(raw)
        train_labels, val_labels = split_tail(labels)
        this_kwargs = kwargs if sampler_kwargs is None else _common_kwargs(label_transform, **sampler_kwargs)
        train_leaves.append(_wrap(
            _image_dataset(train_raw, train_labels, this_kwargs, raw_transform, _train_count(count_name, scale)),
            None, is_val=False, randomise=False,
        ))
        val_leaves.append(_wrap(
            _image_dataset(val_raw, val_labels, this_kwargs, raw_transform, None), _val_count(count_name), is_val=True,
            randomise=False,
        ))
        record(name, train_raw, val_raw)

    def add_containers(
        name, paths, raw_key, label_key, raw_transform, with_channels, patch_shape=None, count_name=None,
    ):
        count_name = count_name or name
        paths = sorted(map(str, paths))
        train_paths, val_paths = split_tail(paths)
        train_leaves.append(_wrap(
            _container_dataset(train_paths, raw_key, label_key, kwargs, raw_transform, with_channels, patch_shape),
            _train_count(count_name, scale), is_val=False, randomise=True,
        ))
        val_leaves.append(_wrap(
            _container_dataset(val_paths, raw_key, label_key, kwargs, raw_transform, with_channels, patch_shape),
            _val_count(count_name), is_val=True, randomise=False,
        ))
        record(name, train_paths, val_paths)

    # 1. LIVECell, one dataset per cell type; images that also appear in the val split are dropped.
    livecell_root = os.path.join(data_root, "livecell")
    for cell_type in datasets.livecell.CELL_TYPES:
        raw, labels = datasets.livecell.get_livecell_paths(livecell_root, split="train", cell_types=[cell_type])
        val_raw, _ = datasets.livecell.get_livecell_paths(livecell_root, split="val", cell_types=[cell_type])
        val_names = {os.path.basename(p) for p in val_raw}
        keep = [i for i, p in enumerate(raw) if os.path.basename(p) not in val_names]
        raw, labels = [raw[i] for i in keep], [labels[i] for i in keep]
        add_images(
            f"livecell_{cell_type}", raw, labels, _identity,
            sampler_kwargs={"sampler": MinInstanceSampler(min_num_instances=6, exclude_ids=[0])}, count_name="livecell",
        )

    # 2. TissueNet: the rgb composite (nucleus, cell, empty) with per-channel normalization, cell labels.
    add_containers(
        "tissuenet", datasets.tissuenet.get_tissuenet_paths(os.path.join(data_root, "tissuenet"), split="train"),
        "raw/rgb", "labels/cell", partial(_normalize_percentile, axis=(1, 2)), with_channels=True,
    )

    # 3. DynamicNuclearNet.
    add_containers(
        "dynamicnuclearnet",
        datasets.dynamicnuclearnet.get_dynamicnuclearnet_paths(
            os.path.join(data_root, "dynamicnuclearnet"), split="train",
        ),
        "raw", "labels", _identity, with_channels=False,
    )

    # 4. DeepBacs (mixed): source / target folders.
    image_folder, label_folder = datasets.deepbacs.get_deepbacs_paths(
        os.path.join(data_root, "deepbacs"), bac_type="mixed", split="train",
    )
    add_images(
        "deepbacs", sorted(glob(os.path.join(image_folder, "*.tif"))),
        sorted(glob(os.path.join(label_folder, "*.tif"))), _to_8bit,
    )

    # 5. DIC HepG2 (rgb png, three distinct channels).
    raw, labels = datasets.dic_hepg2.get_dic_hepg2_paths(os.path.join(data_root, "dic_hepg2"), split="train")
    add_images("dic_hepg2", raw, labels, _identity)

    # 6. NeurIPS CellSeg (mixed formats; `_identity` converts to rgb like the getter's make_rgb).
    raw, labels = datasets.neurips_cell_seg.get_neurips_cellseg_paths(
        os.path.join(data_root, "neurips_cellseg"), split="train",
    )
    add_images("neurips_cellseg", raw, labels, _identity)

    # 7. YeaZ: bright field (2d), phase contrast 2d images and phase contrast frame stacks.
    yeaz_root = os.path.join(data_root, "yeaz")
    bf_raw, bf_labels = datasets.yeaz.get_yeaz_paths(yeaz_root, choice="bf", split="train")
    phc_raw, phc_labels = datasets.yeaz.get_yeaz_paths(yeaz_root, choice="phc", split="train")
    phc_raw, phc_labels = _sorted_pairs(phc_raw, phc_labels)
    is_stack = [_tif_is_stack(p) for p in phc_raw]
    phc_2d = ([p for p, s in zip(phc_raw, is_stack) if not s], [p for p, s in zip(phc_labels, is_stack) if not s])
    phc_stacks = ([p for p, s in zip(phc_raw, is_stack) if s], [p for p, s in zip(phc_labels, is_stack) if s])
    groups = {
        "yeaz_bf": (_sorted_pairs(bf_raw, bf_labels), PATCH_SHAPE),
        "yeaz_phc": (phc_2d, PATCH_SHAPE),
        "yeaz_phc_stacks": (phc_stacks, (1,) + PATCH_SHAPE),
    }
    for name, ((raw, labels), patch_shape) in groups.items():
        train_raw, val_raw = split_tail(raw)
        train_labels, val_labels = split_tail(labels)
        for split_raw, split_labels, is_val in ((train_raw, train_labels, False), (val_raw, val_labels, True)):
            dataset = torch_em.default_segmentation_dataset(
                raw_paths=split_raw, raw_key=None, label_paths=split_labels, label_key=None, is_seg_dataset=True,
                raw_transform=_identity, n_samples=None, **{**kwargs, "patch_shape": patch_shape},
            )
            leaves = val_leaves if is_val else train_leaves
            count = _val_count(name) if is_val else _train_count(name, scale)
            leaves.append(_wrap(dataset, count, is_val, randomise=True))
        record(name, train_raw, val_raw)

    # 8. PUMA nuclei (rgb h5).
    add_containers(
        "puma", datasets.puma.get_puma_paths(os.path.join(data_root, "puma"), split="train", annotations="nuclei"),
        "raw", "labels/instances/nuclei", _identity, with_channels=True,
    )

    # 9. TNBC (rgb h5, channel-first).
    add_containers(
        "tnbc", datasets.tnbc.get_tnbc_paths(os.path.join(data_root, "tnbc"), split="train"),
        "raw", "labels/instances", _identity, with_channels=True,
    )

    _configure_training_normalization(train_leaves, val_leaves)
    return train_leaves, val_leaves, manifest


def build_loaders(
    variant: str, data_root: str, batch_size: int, n_workers: int, val_workers: int, scale: float = 1.0,
):
    """The train and validation loaders of a variant plus the file manifest."""
    label_transform = GeodesicHybridDistanceTransform(contact=VARIANTS[variant]["contact"])
    train_leaves, val_leaves, manifest = build_datasets(data_root, label_transform, scale=scale)
    train_loader = _prepare_data_loader(ConcatDataset(*train_leaves), batch_size, shuffle=True, num_workers=n_workers)
    val_loader = _prepare_data_loader(
        ConcatDataset(*val_leaves), batch_size, shuffle=False, num_workers=val_workers, deterministic=True,
    )
    return train_loader, val_loader, manifest
