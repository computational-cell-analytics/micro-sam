import os

import micro_sam.training as sam_training
import torch
import torch_em

import torch.utils.data as data_util
from torch_em.data.datasets import get_lizard_dataset
from micro_sam.util import export_custom_sam_model


# TODO use other datasets than lizard
def get_dataloaders(patch_shape, data_path):
    label_transform = torch_em.transform.label.label_consecutive  # to ensure consecutive IDs
    dataset = get_lizard_dataset(path=data_path, patch_shape=patch_shape, label_transform=label_transform)
    train_ds, val_ds = data_util.random_split(dataset, [0.9, 0.1])
    train_loader = torch_em.get_data_loader(train_ds, batch_size=2)
    val_loader = torch_em.get_data_loader(val_ds, batch_size=1)
    return train_loader, val_loader


def finetune_histopatho(input_path, export_path, model_type="vit_h", iterations=int(2.5e4), save_root=None):
    """Example code for finetuning SAM on LiveCELL"""

    # training settings:
    checkpoint_path = None  # override this to start training from a custom checkpoint
    device = "cuda"  # override this if you have some more complex set-up and need to specify the exact gpu
    patch_shape = (512, 512)  # the patch shape for training
    n_objects_per_batch = 25  # this is the number of objects per batch that will be sampled

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(model_type, checkpoint_path, device=device)

    # all the stuff we need for training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=10, verbose=True)
    train_loader, val_loader = get_dataloaders(patch_shape=patch_shape, data_path=input_path)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs()

    checkpoint_name = "sam-histopatho-v1"
    # the trainer which performs training and validation (implemented using "torch_em")
    trainer = sam_training.SamTrainer(
        name=checkpoint_name,
        save_root=save_root,
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
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False
    )
    trainer.fit(iterations)
    if export_path is not None:
        checkpoint_path = os.path.join(
            "" if save_root is None else save_root, "checkpoints", checkpoint_name, "best.pt"
        )
        export_custom_sam_model(
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            save_path=export_path,
        )


def main():
    input_path = ""
    export_path = "./sam-vith-histopatho-v1.pth"
    finetune_histopatho(input_path, export_path)


if __name__ == "__main__":
    main()
