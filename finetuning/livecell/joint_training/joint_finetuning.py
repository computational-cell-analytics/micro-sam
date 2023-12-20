import os
import argparse

import torch

import torch_em
from torch_em.model import UNETR
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.data.datasets import get_livecell_loader
from torch_em.transform.label import PerObjectDistanceTransform

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def get_dataloaders(patch_shape, data_path, cell_type=None):
    """This returns the livecell data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/livecell.py
    It will automatically download the livecell data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    label_transform = PerObjectDistanceTransform(
        distances=True, boundary_distances=True, directed_distances=False, foreground=True, instances=True, min_size=25
    )
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]
    train_loader = get_livecell_loader(
        path=data_path, patch_shape=patch_shape, split="train", batch_size=2, num_workers=16,
        cell_types=cell_type, download=True, shuffle=True, label_transform=label_transform,
        raw_transform=raw_transform, label_dtype=torch.float32
    )
    val_loader = get_livecell_loader(
        path=data_path, patch_shape=patch_shape, split="val", batch_size=1, num_workers=16,
        cell_types=cell_type, download=True, shuffle=True, label_transform=label_transform,
        raw_transform=raw_transform, label_dtype=torch.float32
    )

    return train_loader, val_loader


def finetune_livecell(args):
    """Example code for finetuning SAM on LiveCELL"""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (520, 704)  # the patch shape for training
    n_objects_per_batch = 25  # this is the number of objects per batch that will be sampled
    freeze_parts = args.freeze  # override this to freeze different parts of the model

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        freeze=freeze_parts
    )
    model.to(device)

    # let's get the UNETR model for automatic instance segmentation pipeline
    unetr = UNETR(
        backbone="sam",
        encoder=model.sam.image_encoder,
        out_channels=3,
        use_sam_stats=True,
        final_activation="Sigmoid",
        use_skip_connection=False
    )
    unetr.to(device)

    # let's get the parameters for SAM and the decoder from UNETR
    joint_model_params = [params for params in model.parameters()]  # sam parameters
    for name, params in unetr.named_parameters():  # unetr's decoder parameters
        if not name.startswith("encoder"):
            joint_model_params.append(params)

    # all the stuff we need for training
    optimizer = torch.optim.Adam(joint_model_params, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs()

    checkpoint_name = "livecell_sam"
    trainer = sam_training.JointSamTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        # currently we compute loss batch-wise, else we pass channelwise True
        loss=torch_em.loss.DiceLoss(channelwise=False),
        metric=torch_em.loss.DiceLoss(),
        device=device,
        lr_scheduler=scheduler,
        logger=sam_training.JointSamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False,
        mask_prob=0.5,  # (optional) overwrite to provide the probability of using mask inputs while training
        unetr=unetr,
        instance_loss=DiceBasedDistanceLoss(mask_distances_in_bg=True),
        instance_metric=DiceBasedDistanceLoss(mask_distances_in_bg=True)
    )
    trainer.fit(args.iterations)
    if args.export_path is not None:
        checkpoint_path = os.path.join(
            "" if args.save_root is None else args.save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=args.export_path,
        )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LiveCELL dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/usr/nimanwai/data/livecell/",
        help="The filepath to the LiveCELL data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_h, vit_b or vit_l."
    )
    parser.add_argument(
        "--save_root", "-s",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e5),
        help="For how many iterations should the model be trained? By default 100k."
    )
    parser.add_argument(
        "--export_path", "-e",
        help="Where to export the finetuned model to. The exported model can be used in the annotation tools."
    )
    parser.add_argument(
        "--freeze", type=str, nargs="+", default=None,
        help="Which parts of the model to freeze for finetuning."
    )
    args = parser.parse_args()
    finetune_livecell(args)


if __name__ == "__main__":
    main()
