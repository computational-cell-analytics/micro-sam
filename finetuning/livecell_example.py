import argparse

import micro_sam.training as sam_training
import torch
import torch_em
from torch_em.data.datasets import get_livecell_loader


def get_dataloaders(patch_shape, data_path, cell_type=None):
    """This returns the livecell data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/livecell.py
    It will automatically download the livecell data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    """
    train_loader = get_livecell_loader(path=data_path, patch_shape=patch_shape, split="train", batch_size=2,
                                       num_workers=8, cell_types=cell_type, download=True)
    val_loader = get_livecell_loader(path=data_path, patch_shape=patch_shape, split="val", batch_size=1,
                                     num_workers=8, cell_types=cell_type, download=True)
    return train_loader, val_loader


def finetune_livecell(args):
    """Example code for finetuning SAM with ViT-b back-bone on LiveCELL"""

    # training settings:
    model_type = "vit_b"  # change this for fine-tuning a different model
    checkpoint_path = None  # override this to start training from a custom checkpoint
    device = "cuda"  # override this if you have some more complex set-up and need to specify the exact gpu
    patch_shape = ()  # the patch shape for training
    # NOTE: this parameter will most likely change because I don't think it makes sense right now
    n_prompts = [10, 25]  # this is the number of objects per batch that will be sampled (with a random range)

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(model_type, checkpoint_path, device=device)

    # all the stuff we need for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=args.input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    # NOTE: will likely rename this class
    convert_inputs = sam_training.ConvertToSamInputs()

    # the trainer which performs training and validation (implemented using "torch_em")
    trainer = sam_training.SamTrainer(
        name="livecell_sam",
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
        log_image_interval=10,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        mse_loss=torch.nn.MSELoss(),
        _sigmoid=torch.nn.Sigmoid(),
        n_prompts=n_prompts,
        n_sub_iteration=8,
        compile_model=False
    )
    trainer.fit(args.iterations)


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LiveCELL dataset.")
    parser.add_argument(
        "--input_path", "-i", default="",
        help="The filepath to the LiveCELL data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--save_root", "-s",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e5),
        help="For how many iterations should the model be trained? By default 100k."
    )
    args = parser.parse_args()
    finetune_livecell(args)


if __name__ == "__main__":
    finetune_livecell()
