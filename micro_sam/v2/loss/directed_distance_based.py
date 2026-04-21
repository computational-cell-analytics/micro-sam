import torch
import torch.nn as nn

from torch_em.loss import DiceLoss


class DirectedDistanceLoss(nn.Module):
    """Loss for directed distance based instance segmentation.

    Expects input and targets with three channels: foreground and three distance channels.
    Typically the distance channels are in three directions, i.e. x, y and z.

    Args:
        mask_distances_in_bg: whether to mask the loss for distance predictions in the background.
        foreground_loss: the loss for comparing foreground predictions and target.
        distance_loss: the loss for comparing distance predictions and target.
    """
    def __init__(
        self,
        mask_distances_in_bg: bool = True,
        foreground_loss: nn.Module = DiceLoss(),
        distance_loss: nn.Module = nn.MSELoss(reduction="mean")
    ) -> None:
        super().__init__()

        self.foreground_loss = foreground_loss
        self.distance_loss = distance_loss
        self.mask_distances_in_bg = mask_distances_in_bg

        self.init_kwargs = {"mask_distances_in_bg": mask_distances_in_bg}

    def forward(self, input_: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert input_.shape == target.shape, (input_.shape, target.shape)
        assert input_.shape[1] == 4, input_.shape

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

        zdist_input, zdist_target = input_[:, 1:2], target[:, 1:2]
        if self.mask_distances_in_bg:
            mask = fg_target if is_3d else torch.zeros_like(fg_target)  # We do this to avoid any gradients in z-chan.
            zdist_loss = self.distance_loss(zdist_input * mask, zdist_target * mask)
        else:
            zdist_loss = self.distance_loss(zdist_input, zdist_target)

        ydist_input, ydist_target = input_[:, 2:3], target[:, 2:3]
        if self.mask_distances_in_bg:
            mask = fg_target
            ydist_loss = self.distance_loss(ydist_input * mask, ydist_target * mask)
        else:
            ydist_loss = self.distance_loss(ydist_input, ydist_target)

        xdist_input, xdist_target = input_[:, 3:4], target[:, 3:4]
        if self.mask_distances_in_bg:
            mask = fg_target
            xdist_loss = self.distance_loss(xdist_input * mask, xdist_target * mask)
        else:
            xdist_loss = self.distance_loss(xdist_input, xdist_target)

        overall_loss = fg_loss + zdist_loss + ydist_loss + xdist_loss
        return overall_loss
