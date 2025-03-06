import os
import time
import warnings
from glob import glob
from tqdm import tqdm
from contextlib import contextmanager, nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import imageio.v3 as imageio

import torch
from torch.optim import Optimizer
from torch.utils.data import random_split
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import _LRScheduler

import torch_em
from torch_em.util import load_data
from torch_em.data.datasets.util import split_kwargs

from elf.io import open_file

try:
    from qtpy.QtCore import QObject
except Exception:
    QObject = Any

from . import sam_trainer as trainers
from ..instance_segmentation import get_unetr
from . import joint_sam_trainer as joint_trainers
from ..util import get_device, get_model_names, export_custom_sam_model
from .util import get_trainable_sam_model, ConvertToSamInputs, require_8bit, get_raw_transform


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
                    "Expect the 3 last target channels (for normalized distances and foreground probabilities) "
                    f"to be in range [0.0, 1.0], but got min {targets_min}"
                )
            if targets_max < 0 or targets_max > 1:
                raise ValueError(
                    "Invalid value range in the target data from the value loader. "
                    "Expect the 3 last target channels (for normalized distances and foreground probabilities) "
                    f"to be in range [0.0, 1.0], but got max {targets_max}"
                )

        else:
            if n_channels_y != 1:
                raise ValueError(
                    "Invalid number of channels in the target data from the data loader. "
                    "Expect 1 channel for training without an instance segmentation decoder, "
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


@contextmanager
def _filter_warnings(ignore_warnings):
    if ignore_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield
    else:
        with nullcontext():
            yield


def _count_parameters(model_parameters):
    params = sum(p.numel() for p in model_parameters if p.requires_grad)
    params = params / 1e6
    print(f"The number of trainable parameters for the provided model is {params} (~{round(params, 2)}M)")


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
    ignore_warnings: bool = True,
    verify_n_labels_in_loader: Optional[int] = 50,
    box_distortion_factor: Optional[float] = 0.025,
    overwrite_training: bool = True,
    **model_kwargs,
) -> None:
    """Run training for a SAM model.

    Args:
        name: The name of the model to be trained. The checkpoint and logs will have this name.
        model_type: The type of the SAM model.
        train_loader: The dataloader for training.
        val_loader: The dataloader for validation.
        n_epochs: The number of epochs to train for.
        early_stopping: Enable early stopping after this number of epochs without improvement.
        n_objects_per_batch: The number of objects per batch used to compute
            the loss for interative segmentation. If None all objects will be used,
            if given objects will be randomly sub-sampled.
        checkpoint_path: Path to checkpoint for initializing the SAM model.
        with_segmentation_decoder: Whether to train additional UNETR decoder for automatic instance segmentation.
        freeze: Specify parts of the model that should be frozen, namely: image_encoder, prompt_encoder and mask_decoder
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
        ignore_warnings: Whether to ignore raised warnings.
        verify_n_labels_in_loader: The number of labels to verify out of the train and validation dataloaders.
            By default, 50 batches of labels are verified from the dataloaders.
        box_distortion_factor: The factor for distorting the box annotations derived from the ground-truth masks.
        model_kwargs: Additional keyword arguments for the `util.get_sam_model`.
    """
    with _filter_warnings(ignore_warnings):

        t_start = time.time()

        _check_loader(train_loader, with_segmentation_decoder, "train", verify_n_labels_in_loader)
        _check_loader(val_loader, with_segmentation_decoder, "val", verify_n_labels_in_loader)

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
        convert_inputs = ConvertToSamInputs(transform=model.transform, box_distortion_factor=box_distortion_factor)

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

            model_params = joint_model_params
        else:
            model_params = model.parameters()

        optimizer = optimizer_class(model_params, lr=lr)

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

        # Avoid overwriting a trained model, if desired by the user.
        trainer_fit_params["overwrite_training"] = overwrite_training

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

    if not isinstance(patch_shape, tuple):
        patch_shape = tuple(patch_shape)

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
    with_channels: Optional[bool] = None,
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
        with_channels: Whether the image data has channels. By default, it makes the decision based on inputs.
        sampler: A sampler to reject batches according to a given criterion.
        raw_transform: Transformation applied to the image data.
            If not given the data will be cast to 8bit.
        n_samples: The number of samples for this dataset.
        is_train: Whether this dataset is used for training or validation.
        min_size: Minimal object size. Smaller objects will be filtered.
        max_sampling_attempts: Number of sampling attempts to make from a dataset.
        kwargs: Additional keyword arguments for `torch_em.default_segmentation_dataset`.

    Returns:
        The segmentation dataset.
    """

    # By default, let the 'default_segmentation_dataset' heuristic decide for itself.
    is_seg_dataset = kwargs.pop("is_seg_dataset", None)

    # Check if the raw inputs are RGB or not. If yes, use 'ImageCollectionDataset'.
    # Get valid raw paths to make checks possible.
    if raw_key and "*" in raw_key:  # Use the wildcard pattern to find the filepath to only one image.
        rpath = glob(os.path.join(raw_paths if isinstance(raw_paths, str) else raw_paths[0], raw_key))[0]
    else:  # Otherwise, either 'raw_key' is None or container format, supported by 'elf', then we load 1 filepath.
        rpath = raw_paths if isinstance(raw_paths, str) else raw_paths[0]

    # Load one of the raw inputs to validate whether it is RGB or not.
    test_raw_inputs = load_data(path=rpath, key=raw_key if raw_key and "*" not in raw_key else None)
    if test_raw_inputs.ndim == 3:
        if test_raw_inputs.shape[-1] == 3:  # i.e. if it is an RGB image and has channels last.
            is_seg_dataset = False  # we use 'ImageCollectionDataset' in this case.
            # We need to provide a list of inputs to 'ImageCollectionDataset'.
            raw_paths = [raw_paths] if isinstance(raw_paths, str) else raw_paths
            label_paths = [label_paths] if isinstance(label_paths, str) else label_paths

            # This is not relevant for 'ImageCollectionDataset'. Hence, we set 'with_channels' to 'False'.
            with_channels = False if with_channels is None else with_channels

        elif test_raw_inputs.shape[0] == 3:  # i.e. if it is a RGB image and has 3 channels first.
            # This is relevant for 'SegmentationDataset'. If not provided by the user, we set this to 'True'.
            with_channels = True if with_channels is None else with_channels

    # Set 'with_channels' to 'False', i.e. the default behavior of 'default_segmentation_dataset'
    # Otherwise, let the user make the choice as priority, else set this to our suggested default.
    with_channels = False if with_channels is None else with_channels

    # Set the data transformations.
    if raw_transform is None:
        raw_transform = require_8bit

    if with_segmentation_decoder:
        label_transform = torch_em.transform.label.PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            instances=True,
            min_size=min_size,
        )
    else:
        label_transform = torch_em.transform.label.MinSizeLabelTransform(min_size=min_size)

    # Set a default sampler if none was passed.
    if sampler is None:
        sampler = torch_em.data.sampler.MinInstanceSampler(3, min_size=min_size)

    # Check the patch shape to add a singleton if required.
    patch_shape = _update_patch_shape(
        patch_shape=patch_shape, raw_paths=raw_paths, raw_key=raw_key, with_channels=with_channels,
    )

    # Set a minimum number of samples per epoch.
    if n_samples is None:
        loader = torch_em.default_segmentation_loader(
            raw_paths=raw_paths,
            raw_key=raw_key,
            label_paths=label_paths,
            label_key=label_key,
            batch_size=1,
            patch_shape=patch_shape,
            with_channels=with_channels,
            ndim=2,
            is_seg_dataset=is_seg_dataset,
            raw_transform=raw_transform,
            **kwargs
        )
        n_samples = max(len(loader), 100 if is_train else 5)

    dataset = torch_em.default_segmentation_dataset(
        raw_paths=raw_paths,
        raw_key=raw_key,
        label_paths=label_paths,
        label_key=label_key,
        patch_shape=patch_shape,
        raw_transform=raw_transform,
        label_transform=label_transform,
        with_channels=with_channels,
        ndim=2,
        sampler=sampler,
        n_samples=n_samples,
        is_seg_dataset=is_seg_dataset,
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
    """Create a PyTorch DataLoader for training a SAM model.

    Args:
        kwargs: Keyword arguments for `micro_sam.training.default_sam_dataset` or for the PyTorch DataLoader.

    Returns:
        The DataLoader.
    """
    sam_ds_kwargs, extra_kwargs = split_kwargs(default_sam_dataset, **kwargs)

    # There might be additional parameters supported by `torch_em.default_segmentation_dataset`,
    # which the users can provide to get their desired segmentation dataset.
    extra_ds_kwargs, loader_kwargs = split_kwargs(torch_em.default_segmentation_dataset, **extra_kwargs)
    ds_kwargs = {**sam_ds_kwargs, **extra_ds_kwargs}

    ds = default_sam_dataset(**ds_kwargs)
    return torch_em.segmentation.get_data_loader(ds, **loader_kwargs)


CONFIGURATIONS = {
    "Minimal": {"model_type": "vit_t", "n_objects_per_batch": 4, "n_sub_iteration": 4},
    "CPU": {"model_type": "vit_b", "n_objects_per_batch": 10},
    "gtx1080": {"model_type": "vit_t", "n_objects_per_batch": 5},
    "rtx5000": {"model_type": "vit_b", "n_objects_per_batch": 10},
    "V100": {"model_type": "vit_b"},
    "A100": {"model_type": "vit_h"},
}


def _find_best_configuration():
    if torch.cuda.is_available():

        # Check how much memory we have and select the best matching GPU
        # for the available VRAM size.
        _, vram = torch.cuda.mem_get_info()
        vram = vram / 1e9  # in GB

        # Maybe we can get more configurations in the future.
        if vram > 80:  # More than 80 GB: use the A100 configurations.
            return "A100"
        elif vram > 30:  # More than 30 GB: use the V100 configurations.
            return "V100"
        elif vram > 14:  # More than 14 GB: use the RTX5000 configurations.
            return "rtx5000"
        else:  # Otherwise: not enough memory to train on the GPU, use CPU instead.
            return "CPU"
    else:
        return "CPU"


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
        kwargs: Additional keyword parameters that will be passed to `train_sam`.
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
        name=name,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        with_segmentation_decoder=with_segmentation_decoder,
        model_type=model_type,
        **train_kwargs
    )


def _export_helper(save_root, checkpoint_name, output_path, model_type, with_segmentation_decoder, val_loader):

    # Whether the model is stored in the current working directory or in another location.
    if save_root is None:
        save_root = os.getcwd()  # Map this to current working directory, if not specified by the user.

    # Get the 'best' model checkpoint ready for export.
    best_checkpoint = os.path.join(save_root, "checkpoints", checkpoint_name, "best.pt")
    if not os.path.exists(best_checkpoint):
        raise FileNotFoundError(f"The trained model not found at the expected location: '{best_checkpoint}'.")

    # Export the model if an output path has been given.
    if output_path:

        # If the filepath has a pytorch-specific ending, then we just export the checkpoint.
        if os.path.splitext(output_path)[1] in (".pt", ".pth"):
            export_custom_sam_model(
                checkpoint_path=best_checkpoint,
                model_type=model_type[:5],
                save_path=output_path,
                with_segmentation_decoder=with_segmentation_decoder,
            )

        # Otherwise we export it as bioimage.io model.
        else:
            from micro_sam.bioimageio import export_sam_model

            # Load image and corresponding labels from the val loader.
            with torch.no_grad():
                image_data, label_data = next(iter(val_loader))
                image_data, label_data = image_data.numpy().squeeze(), label_data.numpy().squeeze()

                # Select the first channel of the label image if we have a channel axis, i.e. contains the labels
                if label_data.ndim == 3:
                    label_data = label_data[0]  # Gets the channel with instances.
                assert image_data.shape == label_data.shape
                label_data = label_data.astype("uint32")

                export_sam_model(
                    image=image_data,
                    label_image=label_data,
                    model_type=model_type[:5],
                    name=checkpoint_name,
                    output_path=output_path,
                    checkpoint_path=best_checkpoint,
                )

        # The final path where the model has been stored.
        final_path = output_path

    else:  # If no exports have been made, inform the user about the best checkpoint.
        final_path = best_checkpoint

    return final_path


def main():
    """@private"""
    import argparse

    available_models = list(get_model_names())
    available_models = ", ".join(available_models)

    available_configurations = list(CONFIGURATIONS.keys())
    available_configurations = ", ".join(available_configurations)

    parser = argparse.ArgumentParser(description="Finetune Segment Anything Models on custom data.")

    # Images and labels for training.
    parser.add_argument(
        "--images", required=True, type=str, nargs="*",
        help="Filepath to images or the directory where the image data is stored."
    )
    parser.add_argument(
        "--labels", required=True, type=str, nargs="*",
        help="Filepath to ground-truth labels or the directory where the label data is stored."
    )
    parser.add_argument(
        "--image_key", type=str, default=None,
        help="The key for accessing image data, either a pattern / wildcard or with elf.io.open_file. "
    )
    parser.add_argument(
        "--label_key", type=str, default=None,
        help="The key for accessing label data, either a pattern / wildcard or with elf.io.open_file. "
    )

    # Images and labels for validation.
    # NOTE: This isn't required, i.e. we create a val-split on-the-fly from the training data if not provided.
    # Users can choose to have their explicit validation set via this feature as well.
    parser.add_argument(
        "--val_images", type=str, nargs="*",
        help="Filepath to images for validation or the directory where the image data is stored."
    )
    parser.add_argument(
        "--val_labels", type=str, nargs="*",
        help="Filepath to ground-truth labels for validation or the directory where the label data is stored."
    )
    parser.add_argument(
        "--val_image_key", type=str, default=None,
        help="The key for accessing image data for validation, either a pattern / wildcard or with elf.io.open_file."
    )
    parser.add_argument(
        "--val_label_key", type=str, default=None,
        help="The key for accessing label data for validation, either a pattern / wildcard or with elf.io.open_file."
    )

    # Other necessary stuff for training.
    parser.add_argument(
        "--configuration", type=str, default=_find_best_configuration(),
        help=f"The configuration for finetuning the Segment Anything Model, one of {available_configurations}."
    )
    parser.add_argument(
        "--segmentation_decoder", type=str, default="instances",  # TODO: in future, we can extend this to semantic seg.
        help="Whether to finetune Segment Anything Model with additional segmentation decoder for desired targets. "
        "By default, it trains with the additional segmentation decoder for instance segmentation."
    )

    # Optional advanced settings a user can opt to change the values for.
    parser.add_argument(
        "-d", "--device", type=str, default=None,
        help="The device to use for finetuning. Can be one of 'cuda', 'cpu' or 'mps' (only MAC). "
        "By default the most performant available device will be selected."
    )
    parser.add_argument(
        "--patch_shape", type=int, nargs="*", default=(512, 512),
        help="The choice of patch shape for training Segment Anything."
    )
    parser.add_argument(
        "-m", "--model_type", type=str, default=None,
        help=f"The Segment Anything Model that will be used for finetuning, one of {available_models}."
    )
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Checkpoint from which the SAM model will be loaded for finetuning."
    )
    parser.add_argument(
        "-s", "--save_root", type=str, default=None,
        help="The directory where the trained models and corresponding logs will be stored. "
        "By default, there are stored in your current working directory."
    )
    parser.add_argument(
        "--trained_model_name", type=str, default="sam_model",
        help="The custom name of trained model. Allows users to have several trained models under the same 'save_root'."
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="The directory (eg. '/path/to/folder') or filepath (eg. '/path/to/model.pt') to export the trained model."
    )
    parser.add_argument(
        "--n_epochs", type=int, default=100,
        help="The total number of epochs to train the Segment Anything Model. By default, trains for 100 epochs."
    )
    parser.add_argument(
        "--num_workers", type=int, default=1, help="The number of workers for processing data with dataloaders."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="The choice of batch size for training the Segment Anything Model. By default, trains on batch size 1."
    )
    parser.add_argument(
        "--preprocess", type=str, default=None, choices=("normalize_minmax", "normalize_percentile"),
        help="Whether to normalize the raw inputs. By default, does not perform any preprocessing of input images "
        "Otherwise, choose from either 'normalize_percentile' or 'normalize_minmax'."
    )

    args = parser.parse_args()

    # 1. Get all necessary stuff for training.
    checkpoint_name = args.trained_model_name
    config = args.configuration
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    batch_size = args.batch_size
    patch_shape = args.patch_shape
    epochs = args.n_epochs
    num_workers = args.num_workers
    device = args.device
    save_root = args.save_root
    output_path = args.output_path
    with_segmentation_decoder = (args.segmentation_decoder == "instances")

    # Get image paths and corresponding keys.
    train_images, train_gt, train_image_key, train_gt_key = args.images, args.labels, args.image_key, args.label_key
    val_images, val_gt, val_image_key, val_gt_key = args.val_images, args.val_labels, args.val_image_key, args.val_label_key  # noqa

    # 2. Prepare the dataloaders.

    # If the user wants to preprocess the inputs, we allow the possibility to do so.
    _raw_transform = get_raw_transform(args.preprocess)

    # Get the dataset with files for training.
    dataset = default_sam_dataset(
        raw_paths=train_images,
        raw_key=train_image_key,
        label_paths=train_gt,
        label_key=train_gt_key,
        patch_shape=patch_shape,
        with_segmentation_decoder=with_segmentation_decoder,
        raw_transform=_raw_transform,
    )

    # If val images are not exclusively provided, we create a val split from the training data.
    if val_images is None:
        assert val_gt is None and val_image_key is None and val_gt_key is None
        # Use 10% of the dataset for validation - at least one image - for validation.
        n_val = max(1, int(0.1 * len(dataset)))
        train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - n_val, n_val])

    else:  # If val images provided, we create a new dataset for it.
        train_dataset = dataset
        val_dataset = default_sam_dataset(
            raw_paths=val_images,
            raw_key=val_image_key,
            label_paths=val_gt,
            label_key=val_gt_key,
            patch_shape=patch_shape,
            with_segmentation_decoder=with_segmentation_decoder,
            raw_transform=_raw_transform,
        )

    # Get the dataloaders from the datasets.
    train_loader = torch_em.get_data_loader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch_em.get_data_loader(val_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    # 3. Train the Segment Anything Model.

    # Get a valid model and other necessary parameters for training.
    if model_type is not None and model_type not in available_models:
        raise ValueError(f"'{model_type}' is not a valid choice of model.")
    if config is not None and config not in available_configurations:
        raise ValueError(f"'{config}' is not a valid choice of configuration.")

    if model_type is None:  # If user does not specify the model, we use the default model corresponding to the config.
        model_type = CONFIGURATIONS[config]["model_type"]

    train_sam_for_configuration(
        name=checkpoint_name,
        configuration=config,
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=epochs,
        checkpoint_path=checkpoint_path,
        with_segmentation_decoder=with_segmentation_decoder,
        freeze=None,  # TODO: Allow for PEFT.
        device=device,
        save_root=save_root,
        peft_kwargs=None,  # TODO: Allow for PEFT.
    )

    # 4. Export the model, if desired by the user
    final_path = _export_helper(
        save_root, checkpoint_name, output_path, model_type, with_segmentation_decoder, val_loader
    )

    print(f"Training has finished. The trained model is saved at {final_path}.")
