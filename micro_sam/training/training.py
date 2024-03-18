import os
from typing import List, Optional, Union

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch_em.loss import DiceBasedDistanceLoss

from ..util import get_device
from ..instance_segmentation import get_unetr
from .util import get_trainable_sam_model, ConvertToSamInputs
from . import sam_trainer as trainers
from . import joint_sam_trainer as joint_trainers


def train_sam(
    name: str,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 100,
    early_stopping: Optional[int] = 10,
    n_objects_per_batch: Optional[int] = 25,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    with_segmentation_decoder: bool = True,
    freeze: Optional[List[str]] = None,
    device: Optional[Union[str, torch.device]] = None,
    lr: float = 1e-5,
    scheduler: Optional[_LRScheduler] = None,
    n_sub_iteration: int = 8,
    save_root: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Run training for a SAM model.

    Args:
        name: The name of the model to be trained.
            The checkpoint and logs wil have this name.
        model_type: The type of the SAM model.
        train_loader: The dataloader for training.
        val_loader: The dataloader for validation.
        n_epochs: The number of epochs to train for.
        early_stopping: Enable early stopping after this number of epochs
            without improvement.
        n_objects_per_batch: The number of objects per batch used to compute
            the loss for interative segmentation. If None all objects will be used,
            if given objects will be randomly sub-sampled.
        checkpoint_path: Path to checkpoint for initializing the SAM model.
        with_segmentation_decoder: Whether to train additional UNETR decoder
            for automatic instance segmentation.
        freeze: Specify parts of the model that should be frozen, namely: image_encoder, prompt_encoder and mask_decoder
            By default nothing is frozen and the full model is updated.
        device: The device to use for training.
        lr: The learning rate.
        scheduler: The learning rate scheduler. By default ReduceLROnPlateau is used.
        n_sub_iteration: The number of iterative prompts per training iteration.
        save_root: Optional root directory for saving the checkpoints and logs.
            If not given the current working directory is used.
    """
    # TODO check the data loaders!

    device = get_device(device)

    # Get the trainable segment anything model.
    model, state = get_trainable_sam_model(
        model_type=model_type, device=device, freeze=freeze,
        checkpoint_path=checkpoint_path, return_state=True,
    )

    # This class creates all the training data for a batch (inputs, prompts and labels).
    convert_inputs = ConvertToSamInputs(transform=model.transform, box_distortion_factor=0.025)

    # Create the UNETR decoder (if train with it) and the optimizer.
    if with_segmentation_decoder:

        # Get the UNETR.
        unetr = get_unetr(
            image_encoder=model.sam.image_encoder,
            decoder_state=state.get("decoder_state", None),
            device=device,
        )

        # Get the parameters for SAM and the decoder from UNETR.
        joint_model_params = [params for params in model.parameters()]  # sam parameters
        for param_name, params in unetr.named_parameters():  # unetr's decoder parameters
            if not param_name.startswith("encoder"):
                joint_model_params.append(params)

        optimizer = torch.optim.Adam(joint_model_params, lr=lr)

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.9, patience=3, verbose=True
        )

    # The trainer which performs training and validation.
    if with_segmentation_decoder:
        instance_seg_loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
        trainer = joint_trainers.JointSamTrainer(
            name=name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=joint_trainers.JointSamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=n_sub_iteration, compile_model=False, unetr=unetr,
            instance_loss=instance_seg_loss, instance_metric=instance_seg_loss,
            early_stopping=early_stopping, mask_prob=0.5, save_root=save_root,
        )
    else:
        trainer = trainers.SamTrainer(
            name=name, train_loader=train_loader, val_loader=val_loader, model=model,
            optimizer=optimizer, device=device, lr_scheduler=scheduler, logger=trainers.SamLogger,
            log_image_interval=100, mixed_precision=True, convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch, n_sub_iteration=n_sub_iteration, compile_model=False,
            early_stopping=early_stopping, mask_prob=0.5, save_root=save_root,
        )
    trainer.fit(epochs=n_epochs)
