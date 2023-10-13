import os
import argparse

import torch

from torch_em.loss import DiceLoss

import micro_sam.training as sam_training
from micro_sam.util import export_custom_sam_model

from .obtain_lm_datasets import get_generalist_lm_loaders


def finetune_lm_generalist(args):
    """Example code for finetuning SAM on multiple light microscopy datasets"""
    # override this (below) if you have some more complex set-up and need to specify the exact gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (512, 512)  # the patch shape for training
    n_objects_per_batch = 25  # this is the number of objects per batch that will be sampled
    freeze_parts = None  # override this to freeze one or more of these backbones

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(model_type, checkpoint_path, freeze_parts)

    # all stuff needed for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)
    train_loader, val_loader = get_generalist_lm_loaders(args.input_path, patch_shape)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs()

    checkpoint_name = "generalist_lm_sam"
    # the trainer which performs training and validation (implemented using "torch_em")
    trainer = sam_training.SamTrainer(
        name=checkpoint_name,
        save_root=args.save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        # currently we compute loss batch-wise, else we pass channelwise True
        loss=DiceLoss(channelwise=False),
        metric=DiceLoss(),
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
    trainer.fit(iterations=args.iterations)
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
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LM datasets.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/usr/nimanwai/data/",
        help="The filepath to all the respective LM datasets. If the data does not exist yet it will be downloaded"
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_h, vit_b or vit_l."
    )
    parser.add_argument(
        "--save_root", "-s",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run from."
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
    finetune_lm_generalist(args)


if __name__ == "__main__":
    main()
