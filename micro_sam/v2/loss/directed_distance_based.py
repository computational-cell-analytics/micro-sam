from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_em.loss import DiceLoss


def _masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over the masked elements only, normalized per sample.

    Deliberately differs from torch_em's DistanceLoss, which averages over the whole patch and
    so scales the distance gradient by the foreground fraction.
    """
    error = (prediction - target).square() * mask
    dims = tuple(range(1, error.ndim))
    return (error.sum(dims) / mask.sum(dims).clamp_min(1.0)).mean()


def _weighted_bce(
    prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
) -> torch.Tensor:
    """Binary cross entropy on probabilities, weighted per pixel and normalized per sample by the weight sum.

    The probabilities are cast to float32 before the logarithm: in bfloat16 a value close to one rounds to
    exactly one and the log of its complement would be infinite.
    """
    prediction = prediction.float().clamp(eps, 1.0 - eps)
    target = target.float()
    error = -(target * torch.log(prediction) + (1.0 - target) * torch.log1p(-prediction)) * weight
    dims = tuple(range(1, error.ndim))
    return (error.sum(dims) / weight.sum(dims).clamp_min(1.0)).mean()


def boundary_band(foreground: torch.Tensor, radius: int) -> torch.Tensor:
    """The pixels within 'radius' of a transition between foreground and background.

    Computed in-plane with a max and a min pooling of the binary foreground, so the band has the same width
    on either side of every object boundary.

    Args:
        foreground: The binary foreground target, shape (B, 1, Z, Y, X).
        radius: The half width of the band in pixels.

    Returns:
        The band as a float tensor of the foreground's shape and dtype (1 inside the band).
    """
    kernel, padding = (1, 2 * radius + 1, 2 * radius + 1), (0, radius, radius)
    upper = F.max_pool3d(foreground, kernel, stride=1, padding=padding)
    lower = -F.max_pool3d(-foreground, kernel, stride=1, padding=padding)
    return (upper != lower).to(foreground.dtype)


class DirectedDistanceLoss(nn.Module):
    """Loss for directed distance based instance segmentation.

    Expects input and targets with four channels, foreground and three distance channels (in z, y and x),
    plus a fifth contact channel when ``contact=True``. The foreground is trained with ``foreground_loss``
    (Dice by default); ``boundary_weight`` adds a per-pixel binary cross entropy whose weight rises to
    ``1 + boundary_weight`` within ``boundary_radius`` pixels of every object boundary, which calibrates the
    predicted extent to the annotated boundary instead of rewarding a wide, soft foreground. The distances are
    trained with a masked mean squared error, the contact channel with Dice plus binary cross entropy.

    Args:
        mask_distances_in_bg: Whether to mask the loss for distance predictions in the background.
        foreground_loss: The loss for comparing foreground predictions and target. Dice by default.
        contact: Whether the fifth channel holds the contact (touching boundary) probability.
        contact_weight: The weight of the contact term.
        boundary_weight: The extra weight of the foreground cross entropy in the boundary band. None disables
            the cross entropy term altogether (the default, Dice only).
        boundary_radius: The half width of the boundary band in pixels.
    """
    def __init__(
        self,
        mask_distances_in_bg: bool = True,
        foreground_loss: Optional[nn.Module] = None,
        contact: bool = False,
        contact_weight: float = 1.0,
        boundary_weight: Optional[float] = None,
        boundary_radius: int = 2,
    ) -> None:
        super().__init__()

        self.foreground_loss = DiceLoss() if foreground_loss is None else foreground_loss
        self.mask_distances_in_bg = mask_distances_in_bg
        self.contact = contact
        self.contact_weight = contact_weight
        self.boundary_weight = boundary_weight
        self.boundary_radius = boundary_radius
        self.contact_loss = DiceLoss() if contact else None

        self.init_kwargs = {
            "mask_distances_in_bg": mask_distances_in_bg, "contact": contact, "contact_weight": contact_weight,
            "boundary_weight": boundary_weight, "boundary_radius": boundary_radius,
        }

    @property
    def n_channels(self) -> int:
        """The number of prediction and target channels the loss expects."""
        return 4 + int(self.contact)

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input_.shape == target.shape, (input_.shape, target.shape)
        assert input_.shape[1] == self.n_channels, (input_.shape, self.n_channels)

        # IMPORTANT: preserve the channels!
        # Otherwise the Dice Loss will do all kinds of shennanigans.
        # Because it always interprets the first axis as channel,
        # and treats it differently (sums over it independently).
        # This will lead to a very large dice loss that dominates over everything else.
        fg_input, fg_target = input_[:, 0:1], target[:, 0:1]
        fg_loss = self.foreground_loss(fg_input, fg_target)
        if self.boundary_weight is not None:
            weight = 1.0 + self.boundary_weight * boundary_band(fg_target, self.boundary_radius)
            fg_loss = fg_loss + _weighted_bce(fg_input, fg_target, weight)

        # Check whether the input is 2d or not.
        # For 2d inputs, we avoid computing gradients for masked (pseudo) z-distances.
        is_3d = (target.shape[2] != 1)

        if self.mask_distances_in_bg:
            # The all-zero mask zeroes out the z-term (and its gradient) for 2d inputs.
            z_mask = fg_target if is_3d else torch.zeros_like(fg_target)
            yx_mask = fg_target
        else:
            z_mask = yx_mask = torch.ones_like(fg_target)

        zdist_loss = _masked_mse(input_[:, 1:2], target[:, 1:2], z_mask)
        ydist_loss = _masked_mse(input_[:, 2:3], target[:, 2:3], yx_mask)
        xdist_loss = _masked_mse(input_[:, 3:4], target[:, 3:4], yx_mask)

        overall_loss = fg_loss + zdist_loss + ydist_loss + xdist_loss

        if self.contact:
            contact_input, contact_target = input_[:, 4:5], target[:, 4:5]
            contact_loss = self.contact_loss(contact_input, contact_target)
            contact_loss = contact_loss + _weighted_bce(contact_input, contact_target, torch.ones_like(contact_target))
            overall_loss = overall_loss + self.contact_weight * contact_loss

        return overall_loss
