import os
import argparse
import numpy as np

from skimage.measure import label

import torch

import torch_em
from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_mitoem_loader

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model


def get_dataloaders(patch_shape, data_path):
    """This returns the mitochondria em data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/mitoem.py
    It will automatically download the mitoem data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    def do_label_trafo(labels):  # to ensure consecutive IDs
        segmentation = label(labels)
        return segmentation

    # training on a small subset of the dataset (~50% train and val dataset)
    train_rois = [np.s_[0:100, :, :], np.s_[0:100, :, :]]
    val_rois = [np.s_[0:50, :, :], np.s_[0:50, :, :]]

    label_transform = do_label_trafo
    sampler = MinInstanceSampler(min_num_instances=5)
    train_loader = get_mitoem_loader(path=data_path, splits="train", patch_shape=patch_shape, batch_size=2,
                                     download=True, shuffle=True, num_workers=16, sampler=sampler, ndim=2,
                                     label_transform=label_transform, rois=train_rois)
    val_loader = get_mitoem_loader(path=data_path, splits="val", patch_shape=patch_shape, batch_size=1,
                                   download=True, shuffle=True, num_workers=16, sampler=sampler, ndim=2,
                                   label_transform=label_transform, rois=val_rois)
    return train_loader, val_loader


def finetune_mitoem(args):
    """Example code for finetuning SAM on MitoEM"""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (1, 512, 512)  # the patch shape for training
    n_objects_per_batch = 25  # this is the number of objects per batch that will be sampled

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(model_type, checkpoint_path, device=device)

    # all the stuff we need for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs()

    checkpoint_name = f"mitoem_sam_{model_type}"
    # the trainer which performs training and validation (implemented using "torch_em")
    trainer = sam_training.SamTrainer(
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
        logger=sam_training.SamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False
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
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the MitoEM dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/usr/nimanwai/data/mitoem/",
        help="The filepath to the MitoEM data. If the data does not exist yet it will be downloaded."
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
    args = parser.parse_args()
    finetune_mitoem(args)


if __name__ == "__main__":
    main()
