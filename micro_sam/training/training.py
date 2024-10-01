import os
import time
import warnings
from glob import glob
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio.v3 as imageio

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, Dataset

import torch_em
from torch_em.data.datasets.util import split_kwargs

from elf.io import open_file

try:
    from qtpy.QtCore import QObject
except Exception:
    QObject = Any

from ..util import get_device
from . import sam_trainer as trainers
from ..instance_segmentation import get_unetr
from . import joint_sam_trainer as joint_trainers
from .util import get_trainable_sam_model, ConvertToSamInputs, require_8bit


FilePath = Union[str, os.PathLike]


def _check_loader(loader, with_segmentation_decoder, name=None, verify_n_labels_in_loader=None):
    x, _ = next(iter(loader))

    # Raw data: check that we have 1 or 3 channels.
    n_channels = x.shape[1]
    if n_channels not in (1, 3):
        raise ValueError(
            "Invalid number of channels for the input data from the data loader. "
            f"Expect 1 or 3 channels, got {n_channels}."
        )

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
    # NOTE: Verification step to check whether all labels from dataloader are valid (i.e. have atleast one instance).

    def _check_instance_channel(instance_channel):
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

    counter = 0
    name = "" if name is None else f"'{name}'"
    for x, y in tqdm(
        loader,
        desc=f"Verifying labels in {name} dataloader",
        total=verify_n_labels_in_loader if verify_n_labels_in_loader is not None else None,
    ):
        n_channels_y = y.shape[1]
        if with_segmentation_decoder:
            if n_channels_y != 4:
                raise ValueError(
                    "Invalid number of channels in the target data from the data loader. "
                    "Expect 4 channel for training with an instance segmentation decoder, "
                    f"but got {n_channels_y} channels."
                )
            # Check instance channel per sample in a batch
            for per_y_sample in y:
                _check_instance_channel(per_y_sample[0])

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
            # Check instance channel per sample in a batch
            for per_y_sample in y:
                _check_instance_channel(per_y_sample)

        counter += 1
        if verify_n_labels_in_loader is not None and counter > verify_n_labels_in_loader:
            break


# Make the progress bar callbacks compatible with a tqdm progress bar interface.
class _ProgressBarWrapper:
    def __init__(self, signals):
        self._signals = signals
        self._total = None

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        self._signals.pbar_total.emit(value)
        self._total = value

    def update(self, steps):
        self._signals.pbar_update.emit(steps)

    def set_description(self, desc, **kwargs):
        self._signals.pbar_description.emit(desc)


def _count_parameters(model_parameters):
    params = sum(p.numel() for p in model_parameters if p.requires_grad)
    params = params / 1e6
    print(f"The number of trainable parameters for the provided model is {round(params, 2)}M")


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
    scheduler_class: Optional[_LRScheduler] = torch.optim.lr_scheduler.ReduceLROnPlateau,
    scheduler_kwargs: Optional[Dict[str, Any]] = None,
    save_every_kth_epoch: Optional[int] = None,
    pbar_signals: Optional[QObject] = None,
    optimizer_class: Optional[Optimizer] = torch.optim.AdamW,
    peft_kwargs: Optional[Dict] = None,
    verify_n_labels_in_loader: Optional[int] = 50,
    **model_kwargs,
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
            By default, torch.optim.lr_scheduler.ReduceLROnPlateau is used.
        scheduler_kwargs: The learning rate scheduler parameters.
            If passed None, the chosen default parameters are used in ReduceLROnPlateau.
        save_every_kth_epoch: Save checkpoints after every kth epoch separately.
        pbar_signals: Controls for napari progress bar.
        optimizer_class: The optimizer class.
            By default, torch.optim.AdamW is used.
        peft_kwargs: Keyword arguments for the PEFT wrapper class.
        verify_n_labels_in_loader: The number of labels to verify out of the train and validation dataloaders.
            By default, 50 batches of labels are verified from the dataloaders.
        model_kwargs: Additional keyword arguments for the `util.get_sam_model`.
    """
    t_start = time.time()

    _check_loader(train_loader, with_segmentation_decoder, "train", verify_n_labels_in_loader)
    _check_loader(val_loader, with_segmentation_decoder, "validation", verify_n_labels_in_loader)

    device = get_device(device)
    # Get the trainable segment anything model.
    model, state = get_trainable_sam_model(
        model_type=model_type,
        device=device,
        freeze=freeze,
        checkpoint_path=checkpoint_path,
        return_state=True,
        peft_kwargs=peft_kwargs,
        **model_kwargs
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

        optimizer = optimizer_class(joint_model_params, lr=lr)

    else:
        optimizer = optimizer_class(model.parameters(), lr=lr)

    if scheduler_kwargs is None:
        scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 3, "verbose": True}

    scheduler = scheduler_class(optimizer=optimizer, **scheduler_kwargs)

    # The trainer which performs training and validation.
    if with_segmentation_decoder:
        instance_seg_loss = torch_em.loss.DiceBasedDistanceLoss(mask_distances_in_bg=True)
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

    if pbar_signals is not None:
        progress_bar_wrapper = _ProgressBarWrapper(pbar_signals)
        trainer_fit_params["progress"] = progress_bar_wrapper

    trainer.fit(**trainer_fit_params)

    t_run = time.time() - t_start
    hours = int(t_run // 3600)
    minutes = int(t_run // 60)
    seconds = int(round(t_run % 60, 0))
    print("Training took", t_run, f"seconds (= {hours:02}:{minutes:02}:{seconds:02} hours)")


def _update_patch_shape(patch_shape, raw_paths, raw_key, with_channels):
    if isinstance(raw_paths, (str, os.PathLike)):
        path = raw_paths
    else:
        path = raw_paths[0]
    assert isinstance(path, (str, os.PathLike))

    # Check the underlying data dimensionality.
    if raw_key is None:  # If no key is given then we assume it's an image file.
        ndim = imageio.imread(path).ndim
    else:  # Otherwise we try to open the file from key.
        try:  # First try to open it with elf.
            with open_file(path, "r") as f:
                ndim = f[raw_key].ndim
        except ValueError:  # This may fail for images in a folder with different sizes.
            # In that case we read one of the images.
            image_path = glob(os.path.join(path, raw_key))[0]
            ndim = imageio.imread(image_path).ndim

    if ndim == 2:
        assert len(patch_shape) == 2
        return patch_shape
    elif ndim == 3 and len(patch_shape) == 2 and not with_channels:
        return (1,) + patch_shape
    elif ndim == 4 and len(patch_shape) == 2 and with_channels:
        return (1,) + patch_shape
    else:
        return patch_shape


def default_sam_dataset(
    raw_paths: Union[List[FilePath], FilePath],
    raw_key: Optional[str],
    label_paths: Union[List[FilePath], FilePath],
    label_key: Optional[str],
    patch_shape: Tuple[int],
    with_segmentation_decoder: bool,
    with_channels: bool = False,
    sampler: Optional[Callable] = None,
    raw_transform: Optional[Callable] = None,
    n_samples: Optional[int] = None,
    is_train: bool = True,
    min_size: int = 25,
    max_sampling_attempts: Optional[int] = None,
    **kwargs,
) -> Dataset:
    """Create a PyTorch Dataset for training a SAM model.

    Args:
        raw_paths: The path(s) to the image data used for training.
            Can either be multiple 2D images or volumetric data.
        raw_key: The key for accessing the image data. Internal filepath for hdf5-like input
            or a glob pattern for selecting multiple files.
        label_paths: The path(s) to the label data used for training.
            Can either be multiple 2D images or volumetric data.
        label_key: The key for accessing the label data. Internal filepath for hdf5-like input
            or a glob pattern for selecting multiple files.
        patch_shape: The shape for training patches.
        with_segmentation_decoder: Whether to train with additional segmentation decoder.
        with_channels: Whether the image data has RGB channels.
        sampler: A sampler to reject batches according to a given criterion.
        raw_transform: Transformation applied to the image data.
            If not given the data will be cast to 8bit.
        n_samples: The number of samples for this dataset.
        is_train: Whether this dataset is used for training or validation.
        min_size: Minimal object size. Smaller objects will be filtered.
        max_sampling_attempts: Number of sampling attempts to make from a dataset.

    Returns:
        The dataset.
    """

    # Set the data transformations.
    if raw_transform is None:
        raw_transform = require_8bit

    if with_segmentation_decoder:
        label_transform = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True, boundary_distances=True, directed_distances=False,
            foreground=True, instances=True, min_size=min_size,
        )
    else:
        label_transform = torch_em.transform.label.MinSizeLabelTransform(
            min_size=min_size
        )

    # Set a default sampler if none was passed.
    if sampler is None:
        sampler = torch_em.data.sampler.MinInstanceSampler(3, min_size=min_size)

    # Check the patch shape to add a singleton if required.
    patch_shape = _update_patch_shape(
        patch_shape, raw_paths, raw_key, with_channels
    )

    # Set a minimum number of samples per epoch.
    if n_samples is None:
        loader = torch_em.default_segmentation_loader(
            raw_paths, raw_key, label_paths, label_key,
            batch_size=1, patch_shape=patch_shape, ndim=2
        )
        n_samples = max(len(loader), 100 if is_train else 5)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths, raw_key, label_paths, label_key,
        patch_shape=patch_shape,
        raw_transform=raw_transform, label_transform=label_transform,
        with_channels=with_channels, ndim=2,
        sampler=sampler, n_samples=n_samples,
        **kwargs,
    )

    if max_sampling_attempts is not None:
        if isinstance(dataset, torch_em.data.concat_dataset.ConcatDataset):
            for ds in dataset.datasets:
                ds.max_sampling_attempts = max_sampling_attempts
        else:
            dataset.max_sampling_attempts = max_sampling_attempts

    return dataset


def default_sam_loader(**kwargs) -> DataLoader:
    ds_kwargs, loader_kwargs = split_kwargs(default_sam_dataset, **kwargs)
    ds = default_sam_dataset(**ds_kwargs)
    loader = torch_em.segmentation.get_data_loader(ds, **loader_kwargs)
    return loader


CONFIGURATIONS = {
    "Minimal": {"model_type": "vit_t", "n_objects_per_batch": 4, "n_sub_iteration": 4},
    "CPU": {"model_type": "vit_b", "n_objects_per_batch": 10},
    "gtx1080": {"model_type": "vit_t", "n_objects_per_batch": 5},
    "rtx5000": {"model_type": "vit_b", "n_objects_per_batch": 10},
    "V100": {"model_type": "vit_b"},
    "A100": {"model_type": "vit_h"},
}
"""Best training configurations for given hardware resources.
"""


def train_sam_for_configuration(
    name: str,
    configuration: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    with_segmentation_decoder: bool = True,
    model_type: Optional[str] = None,
    **kwargs,
) -> None:
    """Run training for a SAM model with the configuration for a given hardware resource.

    Selects the best training settings for the given configuration.
    The available configurations are listed in `CONFIGURATIONS`.

    Args:
        name: The name of the model to be trained.
            The checkpoint and logs wil have this name.
        configuration: The configuration (= name of hardware resource).
        train_loader: The dataloader for training.
        val_loader: The dataloader for validation.
        checkpoint_path: Path to checkpoint for initializing the SAM model.
        with_segmentation_decoder: Whether to train additional UNETR decoder
            for automatic instance segmentation.
        model_type: Over-ride the default model type.
            This can be used to use one of the micro_sam models as starting point
            instead of a default sam model.
        kwargs: Additional keyword parameterts that will be passed to `train_sam`.
    """
    if configuration in CONFIGURATIONS:
        train_kwargs = CONFIGURATIONS[configuration]
    else:
        raise ValueError(f"Invalid configuration {configuration} expect one of {list(CONFIGURATIONS.keys())}")

    if model_type is None:
        model_type = train_kwargs.pop("model_type")
    else:
        expected_model_type = train_kwargs.pop("model_type")
        if model_type[:5] != expected_model_type:
            warnings.warn("You have specified a different model type.")

    train_kwargs.update(**kwargs)
    train_sam(
        name=name, train_loader=train_loader, val_loader=val_loader,
        checkpoint_path=checkpoint_path, with_segmentation_decoder=with_segmentation_decoder,
        model_type=model_type, **train_kwargs
    )
