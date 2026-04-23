import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.transform.raw import normalize
from torch_em.data.datasets import get_covid_if_loader
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training


def covid_if_raw_trafo(raw):
    raw = normalize(raw)
    raw = raw * 255
    return raw


def get_dataloaders(patch_shape, data_path, n_images):
    """This returns the immunofluoroscence data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/covid_if.py
    It will automatically download the immunofluoroscence data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    i.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    num_workers = 8 if torch.cuda.is_available() else 0

    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )
    raw_transform = covid_if_raw_trafo
    sampler = MinInstanceSampler()

    choice_of_images = [1, 2, 5, 10]
    assert n_images in choice_of_images, \
        f"The current choice of experiments explores a limited combination. Choose from {choice_of_images}"

    train_volumes = (None, n_images)
    val_volumes = (10, 13)

    # let's estimate the total number of patches
    train_loader = get_covid_if_loader(
        path=data_path, patch_shape=patch_shape, batch_size=1, target="cells",
        download=True, sampler=sampler, sample_range=train_volumes
    )

    print(
        f"Found {len(train_loader)} samples for training.",
        "Hence, we will use {0} samples for training.".format(50 if len(train_loader) < 50 else len(train_loader))
    )

    # now, let's get the training and validation dataloaders
    train_loader = get_covid_if_loader(
        path=data_path, patch_shape=patch_shape, batch_size=1, target="cells", num_workers=num_workers, shuffle=True,
        raw_transform=raw_transform, sampler=sampler, label_transform=label_transform, label_dtype=torch.float32,
        sample_range=train_volumes, n_samples=50 if len(train_loader) < 50 else None,
    )

    val_loader = get_covid_if_loader(
        path=data_path, patch_shape=patch_shape, batch_size=1, target="cells", download=True, num_workers=num_workers,
        raw_transform=raw_transform, sampler=sampler, label_transform=label_transform, label_dtype=torch.float32,
        sample_range=val_volumes, n_samples=5,
    )

    return train_loader, val_loader


def finetune_covid_if(args):
    """Code for finetuning SAM on Covid-IF"""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path  # override this to start training from a custom checkpoint
    patch_shape = (512, 512)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # the number of objects per batch that will be sampled
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    checkpoint_name = f"{args.model_type}/covid_if_sam"

    # all stuff we need for training
    train_loader, val_loader = get_dataloaders(
        patch_shape=patch_shape, data_path=args.input_path, n_images=args.n_images
    )
    scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 3}

    # Run training
    sam_training.train_sam(
        name=checkpoint_name,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        early_stopping=10,
        n_objects_per_batch=n_objects_per_batch,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts,
        device=device,
        lr=1e-5,
        n_epochs=args.epochs,
        save_root=args.save_root,
        scheduler_kwargs=scheduler_kwargs,
        save_every_kth_epoch=args.save_every_kth_epoch,
        lora_rank=args.lora_rank,
    )


def main():
    print("Available Resource: '{0}'".format(torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"))
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the Covid-IF dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/projects/nim00007/sam/data/covid_if/",
        help="The filepath to the Covid-IF data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument(
        "--save_root", "-s",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "-c", "--checkpoint_path", type=str, default=None,
        help="The path to custom checkpoint for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="For how many epochs should the model be trained? By default 100."
    )
    parser.add_argument(
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--freeze", type=str, nargs="+", default=None, help="Which parts of the model to freeze for finetuning."
    )
    parser.add_argument(
        "--save_every_kth_epoch", type=int, default=None,
        help="To save every kth epoch while fine-tuning. Expects an integer value."
    )
    parser.add_argument(
        "--n_objects", type=int, default=25, help="The number of instances (objects) per batch used for finetuning."
    )
    parser.add_argument(
        "--n_images", type=int, default=None, help="The number of images used for finetuning."
    )
    parser.add_argument(
        "--lora_rank", type=int, default=None, help="The rank used for low rank adaptation."
    )
    args = parser.parse_args()
    finetune_covid_if(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    main()