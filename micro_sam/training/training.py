import os
from typing import List, Optional, Union, Any, Dict

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from torch_em.loss import DiceBasedDistanceLoss

from ..util import get_device
from . import sam_trainer as trainers
from ..instance_segmentation import get_unetr
from . import joint_sam_trainer as joint_trainers
from .util import get_trainable_sam_model, ConvertToSamInputs


def _check_loader(loader, with_segmentation_decoder):
    x, y = next(iter(loader))

    # Raw data: check that it is between [0, 255]
    minval, maxval = x.min(), x.max()
    if minval < 0 or minval > 255:
        raise ValueError(
            "Invalid data range for the input data from the data loader. "
            f"The input has to be in range [0, 255], but got minimum value {minval}."
        )
    if maxval < 1 or maxval > 255:
        raise ValueError(
            "Invalid data range for the input data from the data loader. "
            f"The input has to be in range [0, 255], but got maximum value {maxval}."
        )

    # Target data: the check depends on whether we train with or without decoder.

    def check_instance_channel(instance_channel):
        unique_vals = torch.unique(instance_channel)
        if (unique_vals < 0).any():
            raise ValueError(
                "The target channel with the instance segmentation must not have negative values."
            )
        if len(unique_vals) == 1:
            raise ValueError(
                "The target channel with the instance segmentation must have at least one instance."
            )
        if not torch.allclose(unique_vals, unique_vals.round(), atol=1e-7):
            raise ValueError(
                "All values in the target channel with the instance segmentation must be integer."
            )

    n_channels_y = y.shape[1]
    if with_segmentation_decoder:
        if n_channels_y != 4:
            raise ValueError(
                "Invalid number of channels in the target data from the data loader. "
                "Expect 4 channel for training with an instance segmentation decoder, "
                f"but got {n_channels_y} channels."
            )
        check_instance_channel(y[:, 0])

        targets_min, targets_max = y[:, 1:].min(), y[:, 1:].max()
        if targets_min < 0 or targets_min > 1:
            raise ValueError(
                "Invalid value range in the target data from the value loader. "
                "Expect the 3 last target channels (for normalized distances and foreground probabilities)"
                f"to be in range [0.0, 1.0], but got min {targets_min}"
            )
        if targets_max < 0 or targets_max > 1:
            raise ValueError(
                "Invalid value range in the target data from the value loader. "
                "Expect the 3 last target channels (for normalized distances and foreground probabilities)"
                f"to be in range [0.0, 1.0], but got max {targets_max}"
            )

    else:
        if n_channels_y != 1:
            raise ValueError(
                "Invalid number of channels in the target data from the data loader. "
                "Expect 1 channel for training without an instance segmentation decoder,"
                f"but got {n_channels_y} channels."
            )
        check_instance_channel(y)


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
    n_sub_iteration: int = 8,
    save_root: Optional[Union[str, os.PathLike]] = None,
    mask_prob: float = 0.5,
    n_iterations: Optional[int] = None,
    scheduler_class: Optional[_LRScheduler] = ReduceLROnPlateau,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    save_every_kth_epoch: Optional[int] = None,
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
        freeze: Specify parts of the model that should be frozen, namely:
            image_encoder, prompt_encoder and mask_decoder
            By default nothing is frozen and the full model is updated.
        device: The device to use for training.
        lr: The learning rate.
        n_sub_iteration: The number of iterative prompts per training iteration.
        save_root: Optional root directory for saving the checkpoints and logs.
            If not given the current working directory is used.
        mask_prob: The probability for using a mask as input in a given training sub-iteration.
        n_iterations: The number of iterations to use for training. This will over-ride n_epochs if given.
        scheduler_class: The learning rate scheduler to update the learning rate.
            By default, ReduceLROnPlateau is used.
        scheduler_kwargs: The learning rate scheduler parameters.
            If passed None, the chosen default parameters are used in ReduceLROnPlateau.
        save_every_kth_epoch: Save checkpoints after every kth epoch separately.
    """
    _check_loader(train_loader, with_segmentation_decoder)
    _check_loader(val_loader, with_segmentation_decoder)

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

    if scheduler_kwargs is None:
        scheduler_params = {"mode": "min", "factor": 0.9, "patience": 3, "verbose": True}

    scheduler = scheduler_class(optimizer=optimizer, **scheduler_params)

    # The trainer which performs training and validation.
    if with_segmentation_decoder:
        instance_seg_loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
        trainer = joint_trainers.JointSamTrainer(
            name=name,
            save_root=save_root,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            lr_scheduler=scheduler,
            logger=joint_trainers.JointSamLogger,
            log_image_interval=100,
            mixed_precision=True,
            convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch,
            n_sub_iteration=n_sub_iteration,
            compile_model=False,
            unetr=unetr,
            instance_loss=instance_seg_loss,
            instance_metric=instance_seg_loss,
            early_stopping=early_stopping,
            mask_prob=mask_prob,
        )
    else:
        trainer = trainers.SamTrainer(
            name=name,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            device=device,
            lr_scheduler=scheduler,
            logger=trainers.SamLogger,
            log_image_interval=100,
            mixed_precision=True,
            convert_inputs=convert_inputs,
            n_objects_per_batch=n_objects_per_batch,
            n_sub_iteration=n_sub_iteration,
            compile_model=False,
            early_stopping=early_stopping,
            mask_prob=mask_prob,
            save_root=save_root,
        )

    if n_iterations is None:
        trainer_fit_params = {"epochs": n_epochs}
    else:
        trainer_fit_params = {"iterations": n_iterations}

    if save_every_kth_epoch is not None:
        trainer_fit_params["save_every_kth_epoch"] = save_every_kth_epoch

    trainer.fit(**trainer_fit_params)
