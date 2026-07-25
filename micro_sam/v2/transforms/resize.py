"""Shared resize-longest-side transforms for SAM2 training and inference."""

from typing import Iterable, Tuple

import numpy as np

import torch
import torch.nn.functional as F


def get_preprocess_shape(old_h: int, old_w: int, target_length: int) -> Tuple[int, int]:
    """Return the aspect-preserving shape whose longest side is ``target_length``."""
    scale = float(target_length) / max(old_h, old_w)
    return int(old_h * scale + 0.5), int(old_w * scale + 0.5)


def resize_longest_side_and_pad_tensor(
    x: torch.Tensor, target_length: int, mode: str = "bilinear", antialias: bool = True,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Resize trailing YX dimensions isotropically and zero-pad bottom/right.

    Supports BCHW and BCZYX tensors. The Z dimension is never resized.
    Padding is applied in image space, before any caller-side normalization.
    """
    old_h, old_w = x.shape[-2:]
    new_h, new_w = get_preprocess_shape(old_h, old_w, target_length)

    kwargs = {}
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = False
    if mode == "bilinear":
        kwargs["antialias"] = antialias

    if x.ndim == 4:
        x = F.interpolate(x, size=(new_h, new_w), mode=mode, **kwargs)
    elif x.ndim == 5:
        batch, channels, depth, _, _ = x.shape
        planes = x.permute(0, 2, 1, 3, 4).reshape(batch * depth, channels, old_h, old_w)
        planes = F.interpolate(planes, size=(new_h, new_w), mode=mode, **kwargs)
        x = planes.reshape(batch, depth, channels, new_h, new_w).permute(0, 2, 1, 3, 4)
    else:
        raise ValueError(f"Expected BCHW or BCZYX input, got shape {tuple(x.shape)}.")

    return F.pad(x, (0, target_length - new_w, 0, target_length - new_h)), (new_h, new_w)


def resize_longest_side_and_pad_numpy(image: np.ndarray, target_length: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize a channels-last image isotropically and zero-pad bottom/right."""
    from skimage.transform import resize

    old_h, old_w = image.shape[:2]
    new_h, new_w = get_preprocess_shape(old_h, old_w, target_length)
    output_shape = (new_h, new_w) + image.shape[2:]
    image = resize(image, output_shape=output_shape, order=1, anti_aliasing=True, preserve_range=True)
    pad_width = ((0, target_length - new_h), (0, target_length - new_w)) + ((0, 0),) * (image.ndim - 2)
    return np.pad(image, pad_width), (new_h, new_w)


def resize_longest_side_and_pad_spatial_numpy(
    data: np.ndarray, target_length: int, is_label: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Resize trailing YX dimensions and zero-pad bottom/right."""
    from skimage.transform import resize

    old_h, old_w = data.shape[-2:]
    new_h, new_w = get_preprocess_shape(old_h, old_w, target_length)
    kwargs = {"order": 0, "anti_aliasing": False} if is_label else {}
    output_shape = data.shape[:-2] + (new_h, new_w)
    data = resize(data, output_shape=output_shape, preserve_range=True, **kwargs).astype(data.dtype)
    pad_width = ((0, 0),) * (data.ndim - 2) + (
        (0, target_length - new_h), (0, target_length - new_w),
    )
    return np.pad(data, pad_width), (new_h, new_w)


class ResizeLongestSideAndPad(torch.nn.Module):
    """Resize a CHW tensor and pad it to a square."""

    def __init__(self, resolution: int):
        super().__init__()
        self.resolution = resolution

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batched = image.ndim == 4
        if not batched:
            image = image.unsqueeze(0)
        image = resize_longest_side_and_pad_tensor(image, self.resolution)[0]
        return image if batched else image[0]


class ResizeLongestSideTransforms(torch.nn.Module):
    """Drop-in replacement for ``SAM2Transforms`` using resize-longest + padding."""

    def __init__(self, resolution, mask_threshold, max_hole_area=0.0, max_sprinkle_area=0.0):
        super().__init__()
        from torchvision.transforms import Normalize, ToTensor

        self.resolution = resolution
        self.mask_threshold = mask_threshold
        self.max_hole_area = max_hole_area
        self.max_sprinkle_area = max_sprinkle_area
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.to_tensor = ToTensor()
        self.transforms = torch.nn.Sequential(
            ResizeLongestSideAndPad(resolution),
            Normalize(self.mean, self.std),
        )

    def _transform_image(self, image):
        return self.transforms(self.to_tensor(image))

    def __call__(self, image):
        return self._transform_image(image)

    def forward_batch(self, image_list: Iterable):
        return torch.stack([self._transform_image(image) for image in image_list], dim=0)

    def transform_coords(self, coords: torch.Tensor, normalize=False, orig_hw=None) -> torch.Tensor:
        coords = coords.clone()
        if normalize:
            if orig_hw is None:
                raise ValueError("orig_hw is required when normalizing prompt coordinates.")
            scale = float(self.resolution) / max(orig_hw)
            coords[..., 0] *= scale
            coords[..., 1] *= scale
        else:
            coords *= self.resolution
        return coords

    def transform_boxes(self, boxes: torch.Tensor, normalize=False, orig_hw=None) -> torch.Tensor:
        return self.transform_coords(boxes.reshape(-1, 2, 2), normalize=normalize, orig_hw=orig_hw)

    def postprocess_masks(self, masks: torch.Tensor, orig_hw) -> torch.Tensor:
        """Upsample, remove bottom/right padding, and restore the original shape."""
        from sam2.utils.misc import get_connected_components

        masks = masks.float()
        flat = masks.flatten(0, 1).unsqueeze(1)
        if self.max_hole_area > 0:
            labels, areas = get_connected_components(flat <= self.mask_threshold)
            masks = torch.where(
                ((labels > 0) & (areas <= self.max_hole_area)).reshape_as(masks),
                self.mask_threshold + 10.0,
                masks,
            )
        if self.max_sprinkle_area > 0:
            labels, areas = get_connected_components(flat > self.mask_threshold)
            masks = torch.where(
                ((labels > 0) & (areas <= self.max_sprinkle_area)).reshape_as(masks),
                self.mask_threshold - 10.0,
                masks,
            )

        new_h, new_w = get_preprocess_shape(orig_hw[0], orig_hw[1], self.resolution)
        masks = F.interpolate(
            masks.float(), (self.resolution, self.resolution), mode="bilinear", align_corners=False,
        )
        masks = masks[..., :new_h, :new_w]
        return F.interpolate(masks, orig_hw, mode="bilinear", align_corners=False)


class ResizeLongestSideAndPadAPI:
    """SAM2 training transform that resizes longest side and pads bottom/right."""

    def __init__(self, target_length: int, consistent_transform: bool = True, v2: bool = False):
        self.target_length = target_length
        self.consistent_transform = consistent_transform
        self.v2 = v2

    def __call__(self, datapoint, **kwargs):
        from training.dataset.transforms import pad, resize

        indices = range(len(datapoint.frames))
        for index in indices:
            frame = datapoint.frames[index]
            old_h, old_w = frame.data.shape[-2:] if self.v2 else (frame.data.height, frame.data.width)
            new_h, new_w = get_preprocess_shape(old_h, old_w, self.target_length)
            datapoint = resize(datapoint, index, (new_w, new_h), square=False, v2=self.v2)
            datapoint = pad(
                datapoint, index, (self.target_length - new_w, self.target_length - new_h), v2=self.v2,
            )
        return datapoint
