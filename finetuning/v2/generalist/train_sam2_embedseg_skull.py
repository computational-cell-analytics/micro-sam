import os

import torch
from torch_em.data import datasets, MinInstanceSampler

from micro_sam.v2.datasets.generalist_loader import _prepare_data_loader
from micro_sam.v2.datasets.wrapper import UniDataWrapper
from micro_sam.v2.transforms.raw import _to_8bit, VideoAugmentTransform
from micro_sam.v2.transforms.labels import _instance_labels


def get_embedseg_skull_dataloaders(input_path, batch_size=1, n_workers=16):
    """Get dataloaders for Mouse-Skull-Nuclei-CBG only (interactive SAM2 training).

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
        datasets.get_embedseg_dataset(
            path=embedseg_path,
            name="Mouse-Skull-Nuclei-CBG",
            patch_shape=patch_shape,
            split="train",
            n_samples=500,
            **kwargs,
        ),
        source_ndim=3,
    )
    val_ds = UniDataWrapper(
        datasets.get_embedseg_dataset(
            path=embedseg_path,
            name="Mouse-Skull-Nuclei-CBG",
            patch_shape=patch_shape,
            split="test",
            n_samples=100,
            **kwargs,
        ),
        source_ndim=3,
    )

    train_loader = _prepare_data_loader(train_ds, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    val_loader = _prepare_data_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return train_loader, val_loader


def main():
    model_type = "hvit_t"
    data_path = "/mnt/vast-nhr/projects/cidas/cca/data"
    save_root = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/interactive/debug/embedseg_skull"

    name = f"sam2_interactive_{model_type}_embedseg_skull"

    from micro_sam.v2.training import train_sam2
    train_loader, val_loader = get_embedseg_skull_dataloaders(input_path=data_path, batch_size=1, n_workers=16)
    train_sam2(
        train_loader=train_loader,
        val_loader=val_loader,
        name=name,
        model_type=model_type,
        n_epochs=20,
        lr=1e-5,
        vision_lr=6e-6,
        save_root=save_root,
        checkpoint_path=None,
        max_num_objects=8,
        num_frames_to_correct=2,
        rand_frames_to_correct=True,
        prob_to_sample_from_gt=0.1,
        add_all_frames_to_correct_as_cond=True,
        num_init_cond_frames=2,
        clip_grad_norm=0.1,
        layer_decay=0.9,
    )


if __name__ == "__main__":
    main()
