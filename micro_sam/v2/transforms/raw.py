import random
from functools import partial
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter
from torch_em.transform.raw import RandomPercentileNormalization, RawTransform

from micro_sam.v2.normalization import normalize_raw

from .labels import _em_cell_label_trafo  # noqa
from .labels import _axondeepseg_pre_label_transform  # noqa
from .labels import _plantseg_label_trafo  # noqa


# NOTE: This is a legacy function: we will keep this for now for unpickling checkpoints saved before the refactor.
def _axondeepseg_label_transform(y):  # noqa
    from bioimage_cpp.segmentation import label as connected_components
    return connected_components(y == 2).astype("uint32")


def to_rgb(image):
    if image.ndim == 2:  # Simple triplication with channels first.
        image = np.concatenate([image[None]] * 3, axis=0)

    if image.ndim == 3 and image.shape[-1] == 3:  # Make channels first for RGB images.
        image = image.transpose(2, 0, 1)

    assert image.ndim == 3
    return image


def _prepare_to_8bit(raw):
    """Prepare raw inputs for ``_to_8bit`` without changing their dynamic range."""
    if raw.ndim == 3 and raw.shape[0] == 1:  # If the inputs have 1 channel, we triplicate it.
        raw = np.concatenate([raw] * 3, axis=0)

    return to_rgb(raw)  # Ensure channel-first RGB without rescaling the original intensities.


def _to_8bit(raw):
    "Ensures three channels for inputs and percentile-normalizes them to [0, 1]."
    raw = _prepare_to_8bit(raw)
    raw = normalize_raw(raw, axis=(1, 2))
    return raw


def _prepare_identity(x):
    """Prepare raw inputs for ``_identity`` without changing their dynamic range."""
    return to_rgb(x)


def _identity(x):
    "Ensures three channels for inputs and percentile-normalizes them to [0, 1]."
    x = _prepare_identity(x)
    x = normalize_raw(x, axis=(1, 2))
    return x


def _prepare_cellpose_raw(x):
    """Prepare CellPose inputs without changing their dynamic range.

    NOTE: The input channel logic is arranged a bit strangely in `cyto` dataset.
    This function takes care of it here.
    """
    r, g, b = x

    assert g.max() != 0
    if r.max() == 0:
        # The image is 1 channel and exists in green channel only.
        assert b.max() == 0
        x = np.concatenate([g[None]] * 3, axis=0)

    elif r.max() != 0 and g.max() != 0:
        # The image is 2 channels and we sort the channels such that - 0: cell, 1: nucleus
        x = np.stack([g, r, np.zeros_like(b)], axis=0)

    return to_rgb(x)  # Ensures three channels for inputs and avoids rescaling inputs.


def _cellpose_raw_trafo(x):
    """Prepare and percentile-normalize CellPose inputs."""
    x = _prepare_cellpose_raw(x)
    x = normalize_raw(x, axis=(1, 2))
    return x


def _resize_to_512(x, is_label=False):
    """Resize trailing spatial dimensions to longest side 512 and pad bottom/right."""
    from micro_sam.v2.transforms.resize import resize_longest_side_and_pad_spatial_numpy
    return resize_longest_side_and_pad_spatial_numpy(x, target_length=512, is_label=is_label)[0]


def _resize_raw_to_512(x):
    """Resize small raw volume patch to 512×512 and normalize."""
    x = _prepare_resize_raw_to_512(x)
    return normalize_raw(x, axis=(1, 2))


def _prepare_resize_raw_to_512(x):
    """Resize a raw patch without normalizing it."""
    return _prepare_identity(_resize_to_512(x, is_label=False))


class VideoAugment:
    """Consistent spatial + color augmentation for (B, C, [Z,] H, W) tensors in [0, 1].

    Mirrors the SAM2 MOSE finetune augmentation pipeline. Spatial transforms and
    the first ColorJitter are applied with the same random parameters across all Z-frames
    (consistent_transform=True in the original YAML). The second ColorJitter is applied
    independently per frame (consistent_transform=False).

    Args:
        p_hflip: Probability of horizontal flip.
        degrees: Max rotation magnitude for RandomAffine.
        shear: Max shear magnitude for RandomAffine.
        brightness: Per-video brightness jitter bound.
        contrast: Per-video contrast jitter bound.
        saturation: Per-video saturation jitter bound.
        p_grayscale: Probability of converting the video to grayscale (consistent).
        per_frame_brightness: Per-frame brightness jitter bound.
        per_frame_contrast: Per-frame contrast jitter bound.
        per_frame_saturation: Per-frame saturation jitter bound.
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        degrees: float = 25.0,
        shear: float = 20.0,
        brightness: float = 0.1,
        contrast: float = 0.03,
        saturation: float = 0.03,
        p_grayscale: float = 0.05,
        per_frame_brightness: float = 0.1,
        per_frame_contrast: float = 0.05,
        per_frame_saturation: float = 0.05,
    ):
        self.p_hflip = p_hflip
        self.degrees = degrees
        self.shear = shear
        self.p_grayscale = p_grayscale
        self._cj_vol = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        self._cj_frame = ColorJitter(
            brightness=per_frame_brightness, contrast=per_frame_contrast, saturation=per_frame_saturation,
        )

    @staticmethod
    def _cj_apply(frame, fn_idx, b_f, c_f, s_f):
        for fn_id in fn_idx:
            if fn_id == 0 and b_f is not None:
                frame = TF.adjust_brightness(frame, b_f)
            elif fn_id == 1 and c_f is not None:
                frame = TF.adjust_contrast(frame, c_f)
            elif fn_id == 2 and s_f is not None:
                frame = TF.adjust_saturation(frame, s_f)
        return frame

    def _augment_frames(self, frames):
        """Apply consistent-then-per-frame augmentation to a list of (C, H, W) tensors."""
        do_flip = random.random() < self.p_hflip
        angle = random.uniform(-self.degrees, self.degrees)
        shear_x = random.uniform(-self.shear, self.shear)
        fn_idx_v, b_f_v, c_f_v, s_f_v, _ = self._cj_vol.get_params(
            self._cj_vol.brightness, self._cj_vol.contrast, self._cj_vol.saturation, self._cj_vol.hue,
        )
        do_gray = random.random() < self.p_grayscale

        out = []
        for frame in frames:
            if do_flip:
                frame = TF.hflip(frame)
            frame = TF.affine(
                frame, angle=angle, translate=[0, 0], scale=1.0, shear=[shear_x, 0.0],
                interpolation=TF.InterpolationMode.BILINEAR,
            )
            frame = self._cj_apply(frame, fn_idx_v, b_f_v, c_f_v, s_f_v)
            if do_gray:
                frame = TF.rgb_to_grayscale(frame, num_output_channels=frame.shape[0])
            fn_idx_f, b_f_f, c_f_f, s_f_f, _ = self._cj_frame.get_params(
                self._cj_frame.brightness, self._cj_frame.contrast, self._cj_frame.saturation, self._cj_frame.hue,
            )
            frame = self._cj_apply(frame, fn_idx_f, b_f_f, c_f_f, s_f_f)
            out.append(frame.clamp(0.0, 1.0))
        return out

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) or (B, C, Z, H, W) float tensor in [0, 1].
        Returns:
            Augmented tensor of the same shape in [0, 1].
        """
        is_3d = x.ndim == 5
        B = x.shape[0]
        out = []
        for b in range(B):
            if is_3d:
                Z = x.shape[2]
                frames = [x[b, :, z] for z in range(Z)]
                aug = self._augment_frames(frames)
                out.append(torch.stack(aug, dim=1))  # (C, Z, H, W)
            else:
                aug = self._augment_frames([x[b]])
                out.append(aug[0])  # (C, H, W)
        return torch.stack(out)


class VideoAugmentTransform:
    """torch-em dataset transform with MOSE-style augmentation for 3D microscopy.

    Unlike VideoAugment (which runs in the training loop on raw only), this class
    is designed for the dataloader transform argument and therefore:
    - Runs in DataLoader workers, in parallel with the GPU forward pass.
    - Applies spatial transforms to both raw and labels consistently.
    - Applies color transforms only to raw.

    Designed for (C, Z, H, W) raw arrays and (1, Z, H, W) label arrays
    from torch-em 3D segmentation datasets.

    Args:
        p_hflip: Horizontal flip probability.
        p_vflip: Vertical flip probability.
        p_zflip: Z-axis (depth) flip probability. Reverses frame order, teaching the model
            to propagate in both directions. Default 0.0; set to 0.5 for bidirectional training.
        brightness: Volume-consistent brightness jitter bound.
        contrast: Volume-consistent contrast jitter bound.
        saturation: Volume-consistent saturation jitter bound.
        per_frame_brightness: Per-frame brightness jitter bound.
        per_frame_contrast: Per-frame contrast jitter bound.
        per_frame_saturation: Per-frame saturation jitter bound.
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        p_zflip: float = 0.0,
        brightness: float = 0.1,
        contrast: float = 0.03,
        saturation: float = 0.03,
        per_frame_brightness: float = 0.1,
        per_frame_contrast: float = 0.05,
        per_frame_saturation: float = 0.05,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self.p_zflip = p_zflip
        self._cj_vol = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        self._cj_frame = ColorJitter(
            brightness=per_frame_brightness, contrast=per_frame_contrast, saturation=per_frame_saturation,
        )

    def __call__(self, raw, labels):
        """Apply augmentation to a single 3D sample.

        Args:
            raw: (Z, H, W) or (C, Z, H, W) float array in [0, 1].
            labels: integer array with the same spatial shape as raw.

        Returns:
            Tuple of (augmented raw, augmented labels).
        """
        # ascontiguousarray with explicit dtype converts byte order (TIFF/HDF5 sources
        # may use big-endian, which torch.from_numpy cannot handle without conversion).
        raw_t = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))
        labels_t = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

        if random.random() < self.p_hflip:
            raw_t = TF.hflip(raw_t)
            labels_t = TF.hflip(labels_t)
        if random.random() < self.p_vflip:
            raw_t = TF.vflip(raw_t)
            labels_t = TF.vflip(labels_t)
        if random.random() < self.p_zflip:
            z_dim = 1 if raw_t.ndim == 4 else 0
            raw_t = torch.flip(raw_t, dims=[z_dim])
            labels_t = torch.flip(labels_t, dims=[z_dim])

        # Color jitter on raw only. TF functions expect (C, H, W), so we iterate over Z.
        # For (Z, H, W) input, treat each Z-slice as a (1, H, W) single-channel frame.
        is_4d = raw_t.ndim == 4
        frames = [raw_t[:, z] for z in range(raw_t.shape[1])] if is_4d else [
            raw_t[z].unsqueeze(0) for z in range(raw_t.shape[0])
        ]
        fn_idx_v, b_f_v, c_f_v, s_f_v, _ = self._cj_vol.get_params(
            self._cj_vol.brightness, self._cj_vol.contrast, self._cj_vol.saturation, self._cj_vol.hue,
        )
        out = []
        for frame in frames:
            frame = VideoAugment._cj_apply(frame, fn_idx_v, b_f_v, c_f_v, s_f_v)
            fn_idx_f, b_f_f, c_f_f, s_f_f, _ = self._cj_frame.get_params(
                self._cj_frame.brightness, self._cj_frame.contrast, self._cj_frame.saturation, self._cj_frame.hue,
            )
            frame = VideoAugment._cj_apply(frame, fn_idx_f, b_f_f, c_f_f, s_f_f)
            out.append(frame.clamp(0.0, 1.0))

        if is_4d:
            raw_t = torch.stack(out, dim=1)  # (C, Z, H, W)
        else:
            raw_t = torch.stack([f.squeeze(0) for f in out])  # (Z, H, W)

        return raw_t.numpy(), labels_t.numpy()


class ImageAugmentTransform:
    """torch-em dataset transform with flips + ColorJitter for 2D microscopy images.

    The 2D analogue of VideoAugmentTransform: horizontal/vertical flips are applied
    to both raw and labels, and two-stage ColorJitter (matching the MOSE finetune
    bounds) to the raw image only. Expects (C, H, W) raw in [0, 1] - i.e. it runs
    after the raw_transform - and an integer label of matching spatial shape.

    Args:
        p_hflip: Horizontal flip probability.
        p_vflip: Vertical flip probability.
        brightness: First-stage brightness jitter bound.
        contrast: First-stage contrast jitter bound.
        saturation: First-stage saturation jitter bound.
        per_frame_brightness: Second-stage brightness jitter bound.
        per_frame_contrast: Second-stage contrast jitter bound.
        per_frame_saturation: Second-stage saturation jitter bound.
    """

    def __init__(
        self,
        p_hflip: float = 0.5,
        p_vflip: float = 0.5,
        brightness: float = 0.1,
        contrast: float = 0.03,
        saturation: float = 0.03,
        per_frame_brightness: float = 0.1,
        per_frame_contrast: float = 0.05,
        per_frame_saturation: float = 0.05,
    ):
        self.p_hflip = p_hflip
        self.p_vflip = p_vflip
        self._cj_1 = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation)
        self._cj_2 = ColorJitter(
            brightness=per_frame_brightness, contrast=per_frame_contrast, saturation=per_frame_saturation,
        )

    def __call__(self, raw, labels):
        """Apply augmentation to a single 2D sample.

        Args:
            raw: (C, H, W) float array in [0, 1].
            labels: integer array with the same spatial shape as raw.

        Returns:
            Tuple of (augmented raw, augmented labels).
        """
        raw_t = torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32))
        labels_t = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

        if random.random() < self.p_hflip:
            raw_t = TF.hflip(raw_t)
            labels_t = TF.hflip(labels_t)
        if random.random() < self.p_vflip:
            raw_t = TF.vflip(raw_t)
            labels_t = TF.vflip(labels_t)

        for cj in (self._cj_1, self._cj_2):
            fn_idx, b, c, s, _ = cj.get_params(cj.brightness, cj.contrast, cj.saturation, cj.hue)
            raw_t = VideoAugment._cj_apply(raw_t, fn_idx, b, c, s)
        raw_t = raw_t.clamp(0.0, 1.0)

        return raw_t.numpy(), labels_t.numpy()


def _normalize_percentile(x, axis=None):
    """Transforms input images with percentile normalization.

    NOTE: For example, this is a specific input transformation for
    'rgb' format of TissueNet image data for the expected axes.
    """
    return normalize_raw(x, axis=axis)


def get_random_percentile_normalization(
    raw_transform: Callable,
    lower_percentile_bounds: Tuple[float, float] = (0.0, 5.0),
    distribution: str = "uniform",
    distribution_kwargs: Optional[Dict[str, float]] = None,
) -> RawTransform:
    """Convert a generalist raw transform to random percentile normalization.

    The data-specific shape/channel preparation is retained as ``RawTransform.augmentation1``, so the
    normalizer receives data in its original intensity range. An existing ``augmentation2`` is preserved.
    """
    augmentation2 = None
    if isinstance(raw_transform, RawTransform):
        augmentation1, augmentation2 = raw_transform.augmentation1, raw_transform.augmentation2
        if isinstance(raw_transform.normalizer, RandomPercentileNormalization):
            axis = raw_transform.normalizer.axis
        elif raw_transform.normalizer is _identity:
            # Defect-augmented EM datasets (CREMI): the defect pipeline lives in augmentation1 with an
            # identity normalizer. Preserve it and apply per-channel percentile normalization.
            axis = (1, 2)
        else:
            raise ValueError(f"Unsupported normalizer in generalist raw transform: {raw_transform.normalizer!r}")
    elif isinstance(raw_transform, RandomPercentileNormalization):
        augmentation1, axis = None, raw_transform.axis
    elif raw_transform is _identity:
        augmentation1, axis = _prepare_identity, (1, 2)
    elif raw_transform is _to_8bit:
        augmentation1, axis = _prepare_to_8bit, (1, 2)
    elif raw_transform is _cellpose_raw_trafo:
        augmentation1, axis = _prepare_cellpose_raw, (1, 2)
    elif raw_transform is _resize_raw_to_512:
        augmentation1, axis = _prepare_resize_raw_to_512, (1, 2)
    elif raw_transform is _normalize_percentile:
        augmentation1, axis = None, None
    elif isinstance(raw_transform, partial) and raw_transform.func is _normalize_percentile:
        if raw_transform.args:
            raise ValueError("Positional arguments are not supported for _normalize_percentile transforms.")
        unexpected_kwargs = set(raw_transform.keywords) - {"axis"}
        if unexpected_kwargs:
            raise ValueError(f"Unsupported _normalize_percentile arguments: {sorted(unexpected_kwargs)}")
        augmentation1, axis = None, raw_transform.keywords.get("axis")
    else:
        raise ValueError(f"Unsupported generalist raw transform: {raw_transform!r}")

    normalizer = RandomPercentileNormalization(
        lower_percentile_bounds=lower_percentile_bounds,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        rounding_decimals=1,
        axis=axis,
    )
    return RawTransform(normalizer=normalizer, augmentation1=augmentation1, augmentation2=augmentation2)
