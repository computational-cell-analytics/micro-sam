import os
import time
import warnings
from typing import Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch_em
from torch_em.trainer.logger_base import TorchEmLogger

from training.trainer import CORE_LOSS_KEY  # SAM2 repo


def _get_cmap():
    from matplotlib import colormaps
    return colormaps["tab20"]


def _colorize_instance_map(label_hw):
    """(H,W) int64 numpy → (3,H,W) float32 [0,1], one tab20 color per instance ID."""
    cmap = _get_cmap()
    H, W = label_hw.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i, uid in enumerate(np.unique(label_hw)):
        if uid == 0:
            continue
        rgb[label_hw == uid] = np.array(cmap(i % 20)[:3], dtype=np.float32)
    return rgb.transpose(2, 0, 1)


def _overlay_binary_masks(masks_ohw, target_hw=None):
    """(O,H,W) bool tensor → (3,H,W) float32 [0,1] overlay, one tab20 color per object."""
    cmap = _get_cmap()
    O, H, W = masks_ohw.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for i in range(O):
        rgb[masks_ohw[i].cpu().numpy().astype(bool)] = np.array(cmap(i % 20)[:3], dtype=np.float32)
    result = rgb.transpose(2, 0, 1)
    if target_hw is not None and (H, W) != target_hw:
        t = torch.from_numpy(result).unsqueeze(0)
        result = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False).squeeze(0).numpy()
    return result


class Sam2Trainer(torch_em.trainer.DefaultTrainer):
    """torch-em style trainer for interactive segmentation with SAM2Train.

    Uses SAM2Train (full model with video memory) and SAM2's native prompting
    strategy, which handles 2D (T=1) and 3D/video (T>1) batches uniformly:
    - T=1: forces point-input mode, one correction round.
    - T>1: mixes point/box/mask prompts across frames with iterative correction.

    The prompting logic (initial point/box/mask selection, iterative correction
    from error regions) is fully embedded in SAM2Train.forward().  No manual
    iterative loop is needed here.

    Args:
        convert_inputs: Callable that converts (x, y) torch-em batches to
            BatchedVideoDatapoint.  Use ConvertToSam2VideoBatch.
        loss: Loss module compatible with SAM2Train outputs.  Defaults to
            MultiStepMultiMasksAndIous when constructed via train_sam2().
        kwargs: Forwarded to torch_em.trainer.DefaultTrainer (model,
            train_loader, val_loader, optimizer, device, lr_scheduler,
            logger, save_root, etc.).
    """

    def __init__(
        self,
        convert_inputs: Callable,
        loss: torch.nn.Module,
        clip_grad_norm: Optional[float] = 0.1,
        **kwargs,
    ):
        super().__init__(loss=loss, metric=loss, **kwargs)
        self.convert_inputs = convert_inputs
        self.clip_grad_norm = clip_grad_norm
        self.interactive_loss = loss
        self._kwargs = kwargs

    def _check_input_normalization(self, x, input_check_done):
        if not input_check_done:
            data_min, data_max = x.min(), x.max()
            if (data_min < 0) or (data_max < 1):
                warnings.warn(
                    "It looks like you are normalizing the training data. "
                    "SAM2 takes care of normalization internally, so it is better not to do this. "
                    "We recommend removing data normalization and providing inputs in the range [0, 255]."
                )
            input_check_done = True
        return input_check_done

    def _sam2_backprop(self, loss: torch.Tensor) -> None:
        """Backward + optional gradient clipping + optimizer step."""
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

    def _interactive_step(self, x, y):
        batch = self.convert_inputs(x, y)
        batch = batch.to(self.device, non_blocking=True)
        outputs = self.model(batch)
        loss = self.interactive_loss(outputs, batch.masks)[CORE_LOSS_KEY]
        return loss, batch, outputs

    def _train_epoch_impl(self, progress, forward_context, backprop):
        # Advance the distributed sampler seed so each epoch has a different shuffle.
        if hasattr(self.train_loader, "batch_sampler") and \
                hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(self._epoch)

        self.model.train()
        n_iter = 0
        t_per_iter = time.time()
        input_check_done = False

        for x, y in self.train_loader:
            input_check_done = self._check_input_normalization(x, input_check_done)
            self.optimizer.zero_grad()

            with forward_context():
                loss, batch, outputs = self._interactive_step(x, y)

            self._sam2_backprop(loss)

            if self.logger is not None:
                log_imgs = (self._iteration % self.log_image_interval == 0)
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_train(
                    self._iteration, loss.item(), lr,
                    x=x if log_imgs else None,
                    y=y if log_imgs else None,
                    batch=batch if log_imgs else None,
                    outputs=outputs if log_imgs else None,
                )

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        return (time.time() - t_per_iter) / n_iter

    def _validate_impl(self, forward_context):
        self.model.eval()
        val_loss = 0.0
        n_iter = 0
        input_check_done = False
        last_x = last_y = last_batch = last_outputs = None

        with torch.no_grad():
            for x, y in self.val_loader:
                input_check_done = self._check_input_normalization(x, input_check_done)
                with forward_context():
                    loss, batch, outputs = self._interactive_step(x, y)
                    val_loss += loss.item()
                n_iter += 1
                last_x, last_y, last_batch, last_outputs = x, y, batch, outputs

        val_loss /= max(n_iter, 1)

        # Synchronize val_loss across DDP ranks so every rank makes the same
        # early-stopping decision.  Without this, ranks can desync and deadlock.
        if dist.is_available() and dist.is_initialized():
            val_tensor = torch.tensor(val_loss, device=self.device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            val_loss = val_tensor.item()

        if self.logger is not None:
            self.logger.log_validation(
                self._iteration, val_loss, last_x, last_y, last_batch, last_outputs,
            )

        return val_loss

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        # Unwrap DDP before saving so checkpoints load directly into non-DDP models.
        original_model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.module
        try:
            super().save_checkpoint(name, current_metric, best_metric, **extra_save_dict)
        finally:
            self.model = original_model


class Sam2Logger(TorchEmLogger):
    """TensorBoard logger for Sam2Trainer.

    Logs scalars every step; logs four image panels at log_image_interval:
      raw          — input image (first batch item, first frame)
      gt_all       — colorized instance map (all objects in the patch)
      gt_chosen    — overlay of the objects sampled for this training step
      predictions  — model's predicted masks for the chosen objects
    Only rank 0 writes; all other ranks are no-ops.
    """

    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        rank = getattr(trainer, "rank", None)
        if rank is not None and rank != 0:
            self.tb = None
            return
        self.log_dir = (
            f"./logs/{trainer.name}"
            if save_root is None
            else os.path.join(save_root, "logs", trainer.name)
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def _masks_for_batch_item(self, batch, b=0, t=0):
        """Return binary masks from batch.masks for a single batch item at frame t.

        batch.masks is (T, O_total, H, W) where O_total spans all batch items.
        batch.obj_to_frame_idx is (T, O_total, 2): [:, :, 1] gives the batch index
        for each object slot.  Filter to only the objects belonging to batch item b.
        """
        b_indices = batch.obj_to_frame_idx[t, :, 1]  # (O_total,)
        return batch.masks[t][b_indices == b]         # (O_b, H, W)

    def _log_images(self, step, x, y, batch, outputs, prefix):
        is_3d = (x.ndim == 5)

        # Raw: first batch item, first frame, first 3 channels — already [0, 1]
        raw = x[0, :3, 0].cpu().float() if is_3d else x[0, :3].cpu().float()
        H, W = raw.shape[-2:]
        self.tb.add_image(f"{prefix}/raw", raw, step)

        # GT all: colorized instance map for first batch item (all objects in the patch)
        gt_lbl = y[0, 0, 0].cpu().numpy() if is_3d else y[0, 0].cpu().numpy()
        self.tb.add_image(f"{prefix}/gt_all", _colorize_instance_map(gt_lbl), step)

        # GT chosen: objects sampled for batch item 0 at frame 0 (subset of gt_all)
        gt_chosen = self._masks_for_batch_item(batch, b=0, t=0)
        self.tb.add_image(f"{prefix}/gt_chosen", _overlay_binary_masks(gt_chosen, target_hw=(H, W)), step)

        # Predictions: show step-0 prediction (initial response to the first prompt, before
        # oracle corrections).  Final corrected output is near-identical to GT by design,
        # since SAM2Train uses GT-derived clicks for iterative correction.
        # multistep_pred_masks_high_res: (O_total, num_steps, 1024, 1024)
        b_indices = batch.obj_to_frame_idx[0, :, 1]
        pred_step0 = outputs[0]["multistep_pred_masks_high_res"][b_indices == 0, 0].detach()
        self.tb.add_image(f"{prefix}/predictions", _overlay_binary_masks(pred_step0 > 0, target_hw=(H, W)), step)

    def log_train(self, step, loss, lr, x=None, y=None, batch=None, outputs=None):
        if self.tb is None:
            return
        self.tb.add_scalar("train/loss", loss, step)
        self.tb.add_scalar("train/learning_rate", lr, step)
        if x is not None:
            self._log_images(step, x, y, batch, outputs, "train")

    def log_validation(self, step, loss, x=None, y=None, batch=None, outputs=None):
        if self.tb is None:
            return
        self.tb.add_scalar("validation/loss", loss, step)
        if x is not None:
            self._log_images(step, x, y, batch, outputs, "validation")


class UniSAM2Trainer(torch_em.trainer.DefaultTrainer):
    """DefaultTrainer subclass for UniSAM2 automatic segmentation.

    Adds two DDP-compatible overrides on top of DefaultTrainer:
    - :meth:`save_checkpoint` unwraps the DDP wrapper before saving so
      checkpoints load directly into non-DDP models.
    - :meth:`_validate_impl` all_reduces the validation loss across ranks
      so every rank makes the same early-stopping decision, and passes
      the last batch to the logger for image visualization.
    """

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        original_model = self.model
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            self.model = self.model.module
        try:
            super().save_checkpoint(name, current_metric, best_metric, **extra_save_dict)
        finally:
            self.model = original_model

    def _validate_impl(self, forward_context):
        self.model.eval()
        metric_val = 0.0
        n_iter = 0
        last_x = last_y = last_pred = None

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                with forward_context():
                    pred, loss = self._forward_and_loss(x, y)
                metric_val += loss.item()
                n_iter += 1
                last_x, last_y, last_pred = x, y, pred

        metric_val /= max(n_iter, 1)

        if dist.is_available() and dist.is_initialized():
            val_tensor = torch.tensor(metric_val, device=self.device)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            metric_val = val_tensor.item()

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, last_x, last_y, last_pred)

        return metric_val


class UniSAM2Logger(TorchEmLogger):
    """TensorBoard logger for UniSAM2Trainer.

    Only rank 0 writes; all other ranks get a no-op logger.
    Logs scalars every step and per-channel images of input, target, and
    prediction at the trainer's log_image_interval.
    """

    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        rank = getattr(trainer, "rank", None)
        if rank is not None and rank != 0:
            self.tb = None
            return
        self.log_dir = (
            f"./logs/{trainer.name}"
            if save_root is None
            else os.path.join(save_root, "logs", trainer.name)
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def _log_images(self, step, x, y, pred, prefix, gradients=None):
        from torch_em.trainer.tensorboard_logger import make_grid_image, normalize_im

        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]
        image = normalize_im(x[selection].cpu())
        self.tb.add_image(f"{prefix}/input", image, step)
        im, im_name = make_grid_image(image, y, pred, selection, gradients)
        self.tb.add_image(f"{prefix}/{im_name}", im, step)

    def log_train(self, step, loss, lr, x, y, pred, log_gradients=False):
        if self.tb is None:
            return
        self.tb.add_scalar("train/loss", loss.item() if hasattr(loss, "item") else loss, step)
        self.tb.add_scalar("train/learning_rate", lr, step)
        if step % self.log_image_interval == 0:
            gradients = pred.grad if log_gradients else None
            self._log_images(step, x, y, pred, "train", gradients=gradients)

    def log_validation(self, step, metric, x, y, pred):
        if self.tb is None:
            return
        self.tb.add_scalar("validation/loss", metric, step)
        if x is not None:
            self._log_images(step, x, y, pred, "validation")
