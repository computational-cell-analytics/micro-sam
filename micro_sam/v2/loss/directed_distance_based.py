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


class DirectedDistanceLoss(nn.Module):
    """Loss for directed distance based instance segmentation.

    Expects foreground and three directed-distance channels, plus an optional fifth boundary channel. The
    boundary loss interpolates between Dice and binary cross entropy according to ``boundary_dice_weight``.

    Args:
        mask_distances_in_bg: Whether to mask distance predictions in the background.
        foreground_loss: Loss for comparing foreground predictions and target.
        with_boundaries: Whether input and target contain the fifth boundary channel.
        boundary_dice_weight: Relative Dice weight in the boundary loss. One selects Dice only, zero selects
            BCE only, and intermediate values compute their convex combination.
    """
    def __init__(
        self,
        mask_distances_in_bg: bool = True,
        foreground_loss: nn.Module = DiceLoss(),
        with_boundaries: bool = False,
        boundary_dice_weight: float = 1.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= boundary_dice_weight <= 1.0:
            raise ValueError(
                f"boundary_dice_weight must be between zero and one, got {boundary_dice_weight}."
            )

        self.foreground_loss = foreground_loss
        self.mask_distances_in_bg = mask_distances_in_bg
        self.with_boundaries = with_boundaries
        self.boundary_dice_weight = boundary_dice_weight
        self.boundary_loss = DiceLoss() if with_boundaries else None

        self.init_kwargs = {
            "mask_distances_in_bg": mask_distances_in_bg,
            "with_boundaries": with_boundaries,
            "boundary_dice_weight": boundary_dice_weight,
        }

    @property
    def n_channels(self) -> int:
        """Number of prediction and target channels expected by the loss."""
        return 4 + int(self.with_boundaries)

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
        if self.with_boundaries:
            boundary_input, boundary_target = input_[:, 4:5], target[:, 4:5]
            dice_loss = self.boundary_loss(boundary_input, boundary_target)
            if self.boundary_dice_weight == 1.0:
                boundary_loss = dice_loss
            else:
                # The decoder returns sigmoid probabilities. Compute BCE in float32 for stable mixed-precision
                # training; clamping prevents exact zero or one after a bfloat16 sigmoid from producing infinities.
                probability = boundary_input.float().clamp(1e-6, 1.0 - 1e-6)
                bce_loss = F.binary_cross_entropy(probability, boundary_target.float())
                boundary_loss = (
                    self.boundary_dice_weight * dice_loss
                    + (1.0 - self.boundary_dice_weight) * bce_loss
                )
            overall_loss = overall_loss + boundary_loss
        return overall_loss
