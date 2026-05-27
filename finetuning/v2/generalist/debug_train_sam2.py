import os
import argparse

import torch
from torch_em.data import datasets as tem_datasets, MinInstanceSampler

from micro_sam.v2.datasets.generalist_loader import _prepare_data_loader
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.training import train_sam2
from micro_sam.v2.transforms.raw import _to_8bit, VideoAugmentTransform
from micro_sam.v2.transforms.labels import _instance_labels


DATA_PATH = "/mnt/vast-nhr/projects/cidas/cca/data"
SAVE_ROOT = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/interactive/debug"


def get_embedseg_skull_dataloaders(input_path, batch_size=1, n_workers=16):
    """Get dataloaders for Mouse-Skull-Nuclei-CBG (3D nuclei).

    Args:
        input_path: Root data path; EmbedSeg data expected under <input_path>/embedseg/.
        batch_size: Batch size.
        n_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    patch_shape = (8, 512, 512)
    embedseg_path = os.path.join(input_path, "embedseg")
    kwargs = {
        "raw_transform": _to_8bit,
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_dtype": torch.int64,
        "label_transform2": _instance_labels,
        "transform": VideoAugmentTransform(),
    }
    train_ds = UniDataWrapper(
        tem_datasets.get_embedseg_dataset(
            path=embedseg_path, name="Mouse-Skull-Nuclei-CBG", patch_shape=patch_shape,
            split="train", n_samples=500, **kwargs,
        ),
        source_ndim=3,
    )
    val_ds = UniDataWrapper(
        tem_datasets.get_embedseg_dataset(
            path=embedseg_path, name="Mouse-Skull-Nuclei-CBG", patch_shape=patch_shape,
            split="test", n_samples=100, **kwargs,
        ),
        source_ndim=3,
    )
    train_loader = _prepare_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = _prepare_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader


def get_lucchi_dataloaders(input_path, batch_size=1, n_workers=16):
    """Get dataloaders for Lucchi EM mitochondria (3D, binary labels -> instance via CC).

    Args:
        input_path: Root data path; Lucchi data expected under <input_path>/lucchi/.
        batch_size: Batch size.
        n_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    patch_shape = (8, 512, 512)
    lucchi_path = os.path.join(input_path, "lucchi")
    kwargs = {
        "raw_transform": _to_8bit,
        # Labels are binary (0/1), so at most 1 unique non-zero ID before CC is applied.
        # min_num_instances=1 ensures the patch contains at least some foreground.
        "sampler": MinInstanceSampler(min_num_instances=1, exclude_ids=[0]),
        "label_dtype": torch.int64,
        "label_transform2": _instance_labels,
        "transform": VideoAugmentTransform(),
    }
    train_ds = UniDataWrapper(
        tem_datasets.get_lucchi_dataset(
            path=lucchi_path, split="train", patch_shape=patch_shape, n_samples=500, **kwargs,
        ),
        source_ndim=3,
    )
    val_ds = UniDataWrapper(
        tem_datasets.get_lucchi_dataset(
            path=lucchi_path, split="test", patch_shape=patch_shape, n_samples=100, **kwargs,
        ),
        source_ndim=3,
    )
    train_loader = _prepare_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = _prepare_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader


def get_livecell_dataloaders(input_path, batch_size=1, n_workers=16):
    """Get dataloaders for LIVECell 2D phase-contrast cell segmentation.

    Args:
        input_path: Root data path; LIVECell data expected under <input_path>/livecell/.
        batch_size: Batch size.
        n_workers: Number of DataLoader worker processes.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    patch_shape = (512, 512)
    livecell_path = os.path.join(input_path, "livecell")
    # VideoAugmentTransform expects 3D input, so it is not used for 2D LIVECell data.
    # label_dtype is handled by the livecell API directly (default int64).
    kwargs = {
        "raw_transform": _to_8bit,
        "sampler": MinInstanceSampler(min_num_instances=3, exclude_ids=[0]),
        "label_transform2": _instance_labels,
    }
    train_ds = UniDataWrapper(
        tem_datasets.get_livecell_dataset(
            path=livecell_path, split="train", patch_shape=patch_shape, n_samples=500, **kwargs,
        ),
        source_ndim=2,
    )
    val_ds = UniDataWrapper(
        tem_datasets.get_livecell_dataset(
            path=livecell_path, split="val", patch_shape=patch_shape, n_samples=100, **kwargs,
        ),
        source_ndim=2,
    )
    train_loader = _prepare_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = _prepare_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    return train_loader, val_loader


def train_one(dataset, model_type):
    """Run one training for the given dataset.

    Args:
        dataset: One of 'skull', 'lucchi', 'livecell'.
        model_type: SAM2 model type (e.g. 'hvit_t').
    """
    name = f"sam2_interactive_{model_type}_{dataset}"

    if dataset == "skull":
        train_loader, val_loader = get_embedseg_skull_dataloaders(input_path=DATA_PATH)
    elif dataset == "lucchi":
        train_loader, val_loader = get_lucchi_dataloaders(input_path=DATA_PATH)
    elif dataset == "livecell":
        train_loader, val_loader = get_livecell_dataloaders(input_path=DATA_PATH)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # LIVECell is a single-frame video (2D images), so there is no extra conditioning frame to draw from.
    num_init_cond_frames = 1 if dataset == "livecell" else 2

    train_sam2(
        train_loader=train_loader,
        val_loader=val_loader,
        name=name,
        model_type=model_type,
        n_epochs=20,
        lr=1e-5,
        vision_lr=6e-6,
        save_root=os.path.join(SAVE_ROOT, dataset),
        checkpoint_path=None,
        max_num_objects=8,
        num_frames_to_correct=2,
        rand_frames_to_correct=True,
        prob_to_sample_from_gt=0.1,
        add_all_frames_to_correct_as_cond=True,
        num_init_cond_frames=num_init_cond_frames,
        clip_grad_norm=0.1,
        layer_decay=0.9,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", default="hvit_t")
    parser.add_argument(
        "--dataset", choices=["skull", "lucchi", "livecell"], default=None,
        help="Dataset to train on (default: all three sequentially).",
    )
    args = parser.parse_args()

    run_datasets = [args.dataset] if args.dataset else ["skull", "lucchi", "livecell"]
    for ds in run_datasets:
        print(f"\n=== Training on {ds} ===")
        train_one(ds, args.model_type)


if __name__ == "__main__":
    main()
