import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch_em.loss import DiceLoss

# Key under which the combined, backpropagated loss is returned. Must match
# CORE_LOSS_KEY = "core_loss" in the SAM2 training repo, which Sam2Trainer indexes.
CORE_LOSS_KEY = "core_loss"


def dice_loss(logits, targets, eps=1.0):
    """Soft Dice loss per object and mask candidate.

    Args:
        logits: Predicted mask logits of shape (N, M, H, W).
        targets: Binary target masks of shape (N, M, H, W).
        eps: Laplace smoothing added to the numerator and denominator.

    Returns:
        Dice loss of shape (N, M), one value per object and mask candidate.
    """
    probs = logits.sigmoid().flatten(2)
    targets = targets.flatten(2)
    numerator = 2 * (probs * targets).sum(-1)
    denominator = probs.sum(-1) + targets.sum(-1)
    return 1 - (numerator + eps) / (denominator + eps)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss per object and mask candidate, averaged over pixels.

    Args:
        logits: Predicted mask logits of shape (N, M, H, W).
        targets: Binary target masks of shape (N, M, H, W).
        alpha: Class-balancing weight in (0, 1); set < 0 to disable balancing.
        gamma: Focusing parameter that down-weights easy pixels.

    Returns:
        Focal loss of shape (N, M).
    """
    prob = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * (1 - p_t) ** gamma
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.flatten(2).mean(-1)


def iou_regression_loss(logits, targets, pred_ious, use_l1_loss=False):
    """Regression loss between the predicted IoU and the true mask IoU.

    Args:
        logits: Predicted mask logits of shape (N, M, H, W).
        targets: Binary target masks of shape (N, M, H, W).
        pred_ious: IoU scores predicted by the model, of shape (N, M).
        use_l1_loss: Use L1 loss instead of MSE.

    Returns:
        IoU regression loss of shape (N, M).
    """
    pred_mask = logits.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    intersection = (pred_mask & gt_mask).sum(-1).float()
    union = (pred_mask | gt_mask).sum(-1).float()
    true_iou = intersection / union.clamp(min=1.0)
    if use_l1_loss:
        return F.l1_loss(pred_ious, true_iou, reduction="none")
    return F.mse_loss(pred_ious, true_iou, reduction="none")


def _num_objects(targets_batch):
    """Number of objects per frame, averaged across distributed ranks (min 1)."""
    num_objects = torch.tensor(float(targets_batch.shape[1]), device=targets_batch.device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(num_objects)
        num_objects = num_objects / dist.get_world_size()
    return num_objects.clamp(min=1).item()


class CustomSAM2Loss(nn.Module):
    """Readable interactive loss for SAM2 training, modeled on micro_sam's v1 SamTrainer.

    Consumes the output structure of SAM2Train (a list of per-frame dicts holding the
    ``multistep_*`` keys) and returns a dict with the combined loss under ``"core_loss"``
    plus the individual components for logging.

    The core supervision mirrors :class:`micro_sam.training.sam_trainer.SamTrainer`: a
    best-mask Dice loss plus an IoU regression loss, accumulated over all correction
    steps and frames. The two SAM2-specific terms - the focal mask loss and the
    object-presence (object-score) loss - are optional and toggled with booleans.

    For each correction step the best mask candidate per object is selected by the
    smallest active mask loss (Dice alone, or Dice + focal when ``use_focal_loss`` is
    set). The Dice and focal terms use this best candidate; the IoU term is averaged over
    all candidates (matching both v1 and SAM2's ``supervise_all_iou=True``).

    Args:
        dice_weight: Weight of the Dice mask loss.
        iou_weight: Weight of the IoU regression loss.
        iou_use_l1_loss: Use L1 instead of MSE for the IoU regression loss.
        use_focal_loss: Add SAM2's sigmoid focal mask loss alongside the Dice loss.
        focal_weight: Weight of the focal mask loss when enabled.
        focal_alpha: Alpha for the focal mask loss.
        focal_gamma: Gamma for the focal mask loss.
        use_object_score_loss: Supervise the object-presence score with a BCE loss.
        object_score_weight: Weight of the object-score loss when enabled.
        average_over_frames: Average the loss over frames instead of summing them. This
            puts 2D (1 frame) and 3D (many frames) batches on the same scale, so neither
            dominates when training on mixed 2D + 3D data. Summing is the SAM2 convention.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        iou_weight: float = 1.0,
        iou_use_l1_loss: bool = False,
        use_focal_loss: bool = False,
        focal_weight: float = 20.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_object_score_loss: bool = False,
        object_score_weight: float = 1.0,
        average_over_frames: bool = False,
    ):
        super().__init__()
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.iou_use_l1_loss = iou_use_l1_loss
        self.use_focal_loss = use_focal_loss
        self.focal_weight = focal_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_object_score_loss = use_object_score_loss
        self.object_score_weight = object_score_weight
        self.average_over_frames = average_over_frames
        self.init_kwargs = {
            "dice_weight": dice_weight, "iou_weight": iou_weight, "iou_use_l1_loss": iou_use_l1_loss,
            "use_focal_loss": use_focal_loss, "focal_weight": focal_weight, "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma, "use_object_score_loss": use_object_score_loss,
            "object_score_weight": object_score_weight, "average_over_frames": average_over_frames,
        }

    def _loss_per_frame(self, outputs, targets, num_objects):
        """Accumulate the loss components over all correction steps for one frame."""
        target_masks = targets.unsqueeze(1).float()  # (N, 1, H, W)
        masks_list = outputs["multistep_pred_multimasks_high_res"]  # steps of (N, M, H, W)
        ious_list = outputs["multistep_pred_ious"]  # steps of (N, M)
        object_score_list = outputs["multistep_object_score_logits"]  # steps of (N, 1)

        components = {"loss_dice": 0.0, "loss_iou": 0.0, "loss_focal": 0.0, "loss_object_score": 0.0}
        for src_masks, pred_ious, object_score_logits in zip(masks_list, ious_list, object_score_list):
            target = target_masks.expand_as(src_masks)  # (N, M, H, W)

            dice = dice_loss(src_masks, target)  # (N, M)
            iou = iou_regression_loss(src_masks, target, pred_ious, self.iou_use_l1_loss)  # (N, M)
            focal = focal_loss(src_masks, target, self.focal_alpha, self.focal_gamma) if self.use_focal_loss else None

            # Pick the best mask candidate per object by the active mask loss.
            select = dice if focal is None else self.dice_weight * dice + self.focal_weight * focal
            best = select.argmin(dim=-1)  # (N,)
            rows = torch.arange(select.shape[0], device=select.device)

            # Only supervise the mask terms for objects that are present in this frame.
            target_obj = (target_masks[:, 0].flatten(1) > 0).any(-1).float()  # (N,)

            components["loss_dice"] += (dice[rows, best] * target_obj).sum() / num_objects
            components["loss_iou"] += (iou.mean(-1) * target_obj).sum() / num_objects
            if focal is not None:
                components["loss_focal"] += (focal[rows, best] * target_obj).sum() / num_objects
            if self.use_object_score_loss:
                # BCE drives the presence score for both present and absent objects.
                object_score_bce = F.binary_cross_entropy_with_logits(
                    object_score_logits, target_obj.unsqueeze(-1), reduction="none",
                )
                components["loss_object_score"] += object_score_bce.mean(-1).sum() / num_objects

        return components

    def forward(self, outs_batch, targets_batch):
        """Compute the loss over a batch of per-frame SAM2 outputs.

        Args:
            outs_batch: List of per-frame output dicts from SAM2Train.
            targets_batch: Target instance masks of shape (T, N, H, W).

        Returns:
            Dict with the per-component losses and the combined loss under "core_loss".
        """
        assert len(outs_batch) == len(targets_batch)
        num_objects = _num_objects(targets_batch)

        losses = {"loss_dice": 0.0, "loss_iou": 0.0, "loss_focal": 0.0, "loss_object_score": 0.0}
        for outputs, targets in zip(outs_batch, targets_batch):
            for key, value in self._loss_per_frame(outputs, targets, num_objects).items():
                losses[key] += value

        # Average over frames so 2D (1 frame) and 3D (many frames) batches share a scale.
        if self.average_over_frames:
            n_frames = max(len(outs_batch), 1)
            losses = {key: value / n_frames for key, value in losses.items()}

        core_loss = self.dice_weight * losses["loss_dice"] + self.iou_weight * losses["loss_iou"]
        if self.use_focal_loss:
            core_loss = core_loss + self.focal_weight * losses["loss_focal"]
        if self.use_object_score_loss:
            core_loss = core_loss + self.object_score_weight * losses["loss_object_score"]

        losses[CORE_LOSS_KEY] = core_loss
        return losses


class CustomSAM2Metric(nn.Module):
    """Best-mask Dice metric for SAM2 interactive training, mirroring micro_sam's v1 SamTrainer.

    Consumes SAM2Train's per-frame output dicts and scores only the initial (step-0)
    prompt response - a single forward like v1's validation, which does not iterate over
    correction steps. The later correction steps are oracle-driven toward the ground truth
    and would inflate the score, so they are excluded. The oracle best mask candidate per
    object (smallest Dice loss) is used.

    The Dice loss is computed with :class:`torch_em.loss.DiceLoss` (the object axis placed
    on the channel axis, ``reduce_channel=None``) so the values match v1's metric exactly.
    Returns the mean Dice loss over present objects and frames (a [0, 1] value, lower is
    better, like the metric returned by v1's SamTrainer). Report ``1 - value`` as the Dice
    score.
    """

    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss(reduce_channel=None)
        self.init_kwargs = {}

    def forward(self, outs_batch, targets_batch):
        dice_sum = 0.0
        dice_count = 0.0
        for outputs, targets in zip(outs_batch, targets_batch):
            target_masks = targets.unsqueeze(1).float()  # (N, 1, H, W)
            probs = outputs["multistep_pred_multimasks_high_res"][0].sigmoid().float()  # step 0: (N, M, H, W)

            # Per-object Dice loss for each mask candidate, with the object axis as the
            # channel axis (matching v1's DiceLoss(reduce_channel=None) usage).
            target = target_masks.swapaxes(0, 1)  # (1, N, H, W)
            dice = torch.stack([
                self.dice_loss(probs[:, i:i + 1].swapaxes(0, 1), target) for i in range(probs.shape[1])
            ])  # (M, N)
            best_dice = dice.min(dim=0).values  # (N,) oracle best of M candidates

            # Only score objects that are present in this frame.
            target_obj = (target_masks[:, 0].flatten(1) > 0).any(-1).float()  # (N,)
            dice_sum = dice_sum + (best_dice * target_obj).sum()
            dice_count = dice_count + target_obj.sum()

        if torch.is_tensor(dice_count):
            return dice_sum / dice_count.clamp(min=1.0)
        return torch.zeros((), device=targets_batch.device)
