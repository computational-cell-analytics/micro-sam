"""Training of hybrid 2D/3D segmentation decoders on top of a frozen SAM2 image encoder.

The decoders are implemented in `torch_em.model.hybrid_unetr`, e.g. the Hiera U-Net decoder ('hiera_u').
The image encoder is taken from a (finetuned) SAM2 model and is not updated; only the decoder is trained.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from micro_sam.util import get_device, training_autocast_dtype
from micro_sam.v2.loss.directed_distance_based import DirectedDistanceLoss
from micro_sam.v2.models.util import CustomActivation
from micro_sam.v2.util import get_sam2_model

from .sam2_trainer import UniSAM2Logger, UniSAM2Trainer

DECODER_PREFIX = "decoder."


class SAM2MultiScaleEncoder(nn.Module):
    """Wrap the SAM2 image encoder so that it returns the features at stride 4, 8 and 16.

    The output follows the convention of `torch_em.model.vit.ViT_Sam2`, which `torch_em.model.HybridUNETR` expects.
    All feature levels have 256 channels; the projections of the stride-4 and stride-8 levels
    (`conv_s0` and `conv_s1` of the SAM2 mask decoder) are part of the decoder.

    Args:
        sam2_image_encoder: The SAM2 image encoder.
        img_size: The input size of the image encoder.
        max_batch_size: The maximal number of images that are passed to the image encoder at once. Larger batches
            are encoded in chunks. The windowed attention of Hiera folds the windows into the batch axis
            (1024 windows per image in the first stage), and the attention kernel fails for 65536 or more
            windows, i.e. for 64 or more images ('CUDA error: invalid configuration argument').
            Chunking is only valid for an encoder that is not trained.
    """
    def __init__(self, sam2_image_encoder: nn.Module, img_size: int = 1024, max_batch_size: Optional[int] = 32):
        super().__init__()
        self.inner = sam2_image_encoder
        self.img_size = img_size
        self.max_batch_size = max_batch_size

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        levels = list(self.inner(x)["backbone_fpn"])
        if len(levels) != 3:
            raise RuntimeError(f"Expected three feature levels from the SAM2 image encoder, got {len(levels)}.")
        return levels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode a batch of images.

        Args:
            x: The preprocessed images with shape (N, 3, H, W).

        Returns:
            The features at stride 16.
            The list of features at stride 4, 8 and 16.
        """
        if self.max_batch_size is None or x.shape[0] <= self.max_batch_size:
            levels = self._encode(x)
        else:
            chunks = [self._encode(chunk) for chunk in torch.split(x, self.max_batch_size)]
            levels = [torch.cat(level, dim=0) for level in zip(*chunks)]
        return levels[-1], levels


def _hybrid_unetr_class():
    try:
        from torch_em.model.hybrid_unetr import HybridUNETR
    except ImportError as e:
        raise ImportError(
            "The decoder training requires 'torch_em.model.hybrid_unetr'. Please update torch_em."
        ) from e
    return HybridUNETR


class FrozenEncoderHybridUNETR(_hybrid_unetr_class()):
    """A hybrid UNETR with the frozen image encoder of a SAM2 model and a trainable segmentation decoder.

    The model predicts the targets for directed distance based instance segmentation for images with
    shape (B, C, Y, X) and volumes with shape (B, C, Z, Y, X).

    The encoder is always in evaluation mode, its parameters do not require gradients and it is applied
    without building the autograd graph. The projections of the stride-4 and stride-8 features in the decoder
    are initialized from the SAM2 model and are trained with the decoder.

    Args:
        model_type: The SAM2 model that provides the image encoder, e.g. 'hvit_t_cells' or 'hvit_t'.
        decoder: The name of the decoder in `torch_em.model.hybrid_unetr.HYBRID_DECODERS`.
        decoder_kwargs: The keyword arguments for the decoder.
        out_channels: The number of output channels. The default corresponds to foreground,
            three directed distances and boundaries.
        checkpoint_path: Optional checkpoint with the weights of the SAM2 model.
            By default the weights that belong to `model_type` are used.
        perform_range_checks: Whether to validate the input value range on each forward pass.
    """
    def __init__(
        self,
        model_type: str = "hvit_t_cells",
        decoder: str = "hiera_u",
        decoder_kwargs: Optional[Dict] = None,
        out_channels: int = 5,
        checkpoint_path: Optional[Union[str, os.PathLike]] = None,
        perform_range_checks: bool = True,
    ):
        sam2 = get_sam2_model(
            model_type=model_type, device="cpu", checkpoint_path=checkpoint_path, input_type="images",
        )
        super().__init__(
            backbone="sam2",
            encoder=SAM2MultiScaleEncoder(sam2.image_encoder, img_size=sam2.image_size),
            decoder=decoder,
            decoder_kwargs=decoder_kwargs,
            out_channels=out_channels,
            use_sam_stats=True,
            final_activation=CustomActivation(),
            perform_range_checks=perform_range_checks,
        )
        self.init_skip_projections(sam2.sam_mask_decoder.conv_s0, sam2.sam_mask_decoder.conv_s1)

        # Only values that can be serialized, the trainer stores them in the checkpoints.
        self.init_kwargs = {
            "model_type": model_type, "decoder": decoder, "decoder_kwargs": decoder_kwargs,
            "out_channels": out_channels,
            "checkpoint_path": None if checkpoint_path is None else str(checkpoint_path),
            "perform_range_checks": perform_range_checks,
        }
        self.model_type = model_type

        self.encoder.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True):
        """Set the training mode of the decoder. The encoder stays in evaluation mode."""
        super().train(mode)
        self.encoder.eval()
        return self

    def encode(self, x: torch.Tensor):
        """Preprocess the input and run the frozen image encoder on all slices, see `HybridUNETR.encode`."""
        with torch.no_grad():
            return super().encode(x)

    def decoder_parameters(self) -> List[nn.Parameter]:
        """The trainable parameters, which all belong to the decoder."""
        return list(self.decoder.parameters())

    def decoder_state(self) -> Dict[str, torch.Tensor]:
        """The state of the decoder, with the prefix 'decoder.' of the full model state."""
        return {DECODER_PREFIX + key: value for key, value in self.decoder.state_dict().items()}

    def load_decoder_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Load the state of the decoder.

        Args:
            state: The decoder state, as returned by `decoder_state`. Other entries are not allowed.
        """
        unexpected = [key for key in state if not key.startswith(DECODER_PREFIX)]
        if unexpected:
            raise RuntimeError(f"The decoder state contains entries that do not belong to the decoder: {unexpected}.")
        self.decoder.load_state_dict({key[len(DECODER_PREFIX):]: value for key, value in state.items()})


class DecoderLoss(DirectedDistanceLoss):
    """The directed distance loss for targets with or without the instance ids.

    The datasets of the joint training (`micro_sam.v2.datasets.generalist_loader._build_joint_datasets`) yield
    the instance ids as first target channel, followed by the targets for automatic segmentation.
    This loss leaves out the first channel if the target has one more channel than the prediction.
    """
    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.shape[1] == input_.shape[1] + 1:
            target = target[:, 1:]
        return super().forward(input_, target)


class _DecoderStateView:
    """Stand-in for the model while checkpoints are saved or loaded, so that they hold only the decoder weights."""
    def __init__(self, model: FrozenEncoderHybridUNETR):
        self.model = model

    def state_dict(self):
        return self.model.decoder_state()

    def load_state_dict(self, state):
        self.model.load_decoder_state(state)

    def to(self, device):
        self.model.to(device)
        return self


class DecoderTrainer(UniSAM2Trainer):
    """Trainer for a segmentation decoder on top of a frozen image encoder.

    The checkpoints contain only the weights of the decoder, in 'model_state' and 'decoder_state' (identical),
    the 'decoder_config' to rebuild the decoder, and the 'encoder_model_type' of the SAM2 model
    that provides the frozen encoder.
    """

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        model = self.model
        extra_save_dict.setdefault("decoder_state", model.decoder_state())
        extra_save_dict.setdefault("decoder_config", model.decoder_config)
        extra_save_dict.setdefault("encoder_model_type", model.model_type)
        self.model = _DecoderStateView(model)
        try:
            super().save_checkpoint(name, current_metric, best_metric, **extra_save_dict)
        finally:
            self.model = model

    def load_checkpoint(self, checkpoint="best"):
        model = self.model
        self.model = _DecoderStateView(model)
        try:
            return super().load_checkpoint(checkpoint)
        finally:
            self.model = model


def get_decoder_train_model(
    model_type: str = "hvit_t_cells",
    decoder: str = "hiera_u",
    decoder_kwargs: Optional[Dict] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    device: Optional[Union[str, torch.device]] = None,
    out_channels: int = 5,
) -> FrozenEncoderHybridUNETR:
    """Get the model for training a segmentation decoder on top of a frozen SAM2 image encoder.

    Args:
        model_type: The SAM2 model that provides the image encoder, e.g. 'hvit_t_cells'.
        decoder: The name of the decoder, 'hiera_u' or 'c3f2'.
        decoder_kwargs: The keyword arguments for the decoder.
        checkpoint_path: Optional checkpoint with the weights of the SAM2 model.
        device: The device for the model.
        out_channels: The number of output channels.

    Returns:
        The model.
    """
    model = FrozenEncoderHybridUNETR(
        model_type=model_type, decoder=decoder, decoder_kwargs=decoder_kwargs, out_channels=out_channels,
        checkpoint_path=checkpoint_path,
    )
    return model.to(get_device(device))


def get_decoder_trainer(
    name: str,
    model: FrozenEncoderHybridUNETR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
    save_root: Optional[Union[str, os.PathLike]] = None,
    boundary_dice_weight: float = 0.5,
    scheduler_kwargs: Optional[Dict] = None,
    log_image_interval: int = 500,
) -> DecoderTrainer:
    """Get the trainer for a segmentation decoder on top of a frozen SAM2 image encoder.

    The settings follow the automatic segmentation branch of the joint SAM2 training
    (`micro_sam.v2.training.train_joint_sam2`): the AdamW optimizer with its default weight decay,
    a 'ReduceLROnPlateau' scheduler on the validation loss, the directed distance loss with boundaries
    and bfloat16 autocast where the device supports it.

    Args:
        name: The name of the checkpoint and log folder.
        model: The model, see `get_decoder_train_model`.
        train_loader: The loader for training. It yields the inputs and the directed distance targets with boundaries,
            optionally with the instance ids as additional first channel, see `DecoderLoss`.
        val_loader: The loader for validation, with the same format.
        lr: The learning rate.
        device: The device for training.
        save_root: The root directory for checkpoints and logs.
        boundary_dice_weight: The Dice weight in the boundary loss.
        scheduler_kwargs: The arguments for the 'ReduceLROnPlateau' scheduler.
        log_image_interval: The interval for logging images to tensorboard, in iterations.

    Returns:
        The trainer.
    """
    device = torch.device(get_device(device))
    if scheduler_kwargs is None:
        scheduler_kwargs = {"mode": "min", "factor": 0.9, "patience": 10}

    trainable = [name_ for name_, param in model.named_parameters() if param.requires_grad]
    not_decoder = [name_ for name_ in trainable if not name_.startswith(DECODER_PREFIX)]
    if not_decoder:
        raise RuntimeError(f"Parameters outside of the decoder require gradients: {not_decoder[:5]}.")

    optimizer = torch.optim.AdamW(model.decoder_parameters(), lr=lr, fused=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **scheduler_kwargs)
    loss = DecoderLoss(
        mask_distances_in_bg=True, with_boundaries=model.out_channels == 5, boundary_dice_weight=boundary_dice_weight,
    )

    return DecoderTrainer(
        name=name,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        loss=loss,
        optimizer=optimizer,
        metric=loss,
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=log_image_interval,
        # The hardware decides the precision. SAM2 runs in bfloat16 only.
        mixed_precision=training_autocast_dtype(device) is not None,
        mixed_precision_dtype="bfloat16",
        early_stopping=None,
        logger=UniSAM2Logger,
        save_root=None if save_root is None else str(save_root),
        compile_model=False,
    )


def _finished_epochs(save_root, name, checkpoint):
    path = os.path.join("." if save_root is None else save_root, "checkpoints", name, f"{checkpoint}.pt")
    if not os.path.exists(path):
        return None
    # The stored value is the index of the last finished epoch.
    return torch.load(path, map_location="cpu", weights_only=False)["epoch"] + 1


def train_decoder(
    name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_type: str = "hvit_t_cells",
    decoder: str = "hiera_u",
    decoder_kwargs: Optional[Dict] = None,
    n_epochs: int = 100,
    lr: float = 1e-4,
    device: Optional[Union[str, torch.device]] = None,
    save_root: Optional[Union[str, os.PathLike]] = None,
    save_every_kth_epoch: Optional[int] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    resume: bool = False,
    boundary_dice_weight: float = 0.5,
    scheduler_kwargs: Optional[Dict] = None,
) -> None:
    """Train a hybrid 2D/3D segmentation decoder on top of the frozen image encoder of a SAM2 model.

    Pass loaders with the targets for directed distances with boundaries, for example for the datasets of
    `micro_sam.v2.datasets.generalist_loader._build_joint_datasets` with `with_boundaries=True`.

    Args:
        name: The name of the checkpoint and log folder.
        train_loader: The loader for training.
        val_loader: The loader for validation.
        model_type: The SAM2 model that provides the image encoder, e.g. 'hvit_t_cells'.
        decoder: The name of the decoder, 'hiera_u' or 'c3f2'.
        decoder_kwargs: The keyword arguments for the decoder.
        n_epochs: The total number of epochs to train for.
        lr: The learning rate.
        device: The device for training.
        save_root: The root directory for checkpoints and logs.
        save_every_kth_epoch: Save a separate checkpoint every k-th epoch.
        checkpoint_path: Optional checkpoint with the weights of the SAM2 model.
        resume: Whether to continue from the latest checkpoint of this training, if it exists.
            The training then runs for the epochs that are missing to reach `n_epochs`.
        boundary_dice_weight: The Dice weight in the boundary loss.
        scheduler_kwargs: The arguments for the 'ReduceLROnPlateau' scheduler.
            By default the mode 'min', a factor of 0.9 and a patience of 10 epochs.
    """
    device = get_device(device)
    model = get_decoder_train_model(
        model_type=model_type, decoder=decoder, decoder_kwargs=decoder_kwargs, checkpoint_path=checkpoint_path,
        device=device,
    )
    trainer = get_decoder_trainer(
        name=name, model=model, train_loader=train_loader, val_loader=val_loader, lr=lr, device=device,
        save_root=save_root, boundary_dice_weight=boundary_dice_weight, scheduler_kwargs=scheduler_kwargs,
    )

    fit_kwargs = {"epochs": n_epochs, "overwrite_training": True}
    finished = _finished_epochs(save_root, name, "latest") if resume else None
    if finished is not None:
        if finished >= n_epochs:
            print(f"The training '{name}' has already finished {finished} epochs.")
            return
        fit_kwargs.update(epochs=n_epochs - finished, load_from_checkpoint="latest")
    if save_every_kth_epoch is not None:
        fit_kwargs["save_every_kth_epoch"] = save_every_kth_epoch
    trainer.fit(**fit_kwargs)
