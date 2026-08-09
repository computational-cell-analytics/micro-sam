import os
import time
import random
from typing import Callable, Optional

import numpy as np

import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from torch_em.trainer.logger_base import TorchEmLogger

from .sam2_trainer import (
    Sam2Trainer, _colorize_instance_map, _overlay_binary_masks, _pinned_validation_rng,
    VALIDATION_SEED,
)


class JointSam2Trainer(Sam2Trainer):
    """Trainer for joint interactive + automatic segmentation with SAM2 and UniSAM2.

    Uses a single **5-channel label tensor** per batch built by
    :func:`~micro_sam.v2.datasets.generalist_loader._build_joint_datasets`:

    - Channel 0 (int64): instance IDs -> interactive branch.
    - Channels 1-4 (float32): foreground mask + directed distances -> automatic branch.

    Both branches see the **same image patch**, which halves the number of data
    loaders compared to using separate interactive and automatic datasets.

    Each branch is trained with a separate forward-backward-step cycle per
    iteration (matching micro-sam v1), so their loss scales never compete.
    In DDP, only ``model`` (sam2_train_model) is DDP-wrapped; every gradient the
    automatic backward produces - the ``unetr`` decoder *and* the shared image
    encoder - is manually all_reduced via :meth:`_sync_automatic_grads`.
    The shared encoder therefore takes two optimizer steps per batch, one per branch.

    Args:
        unetr: UniSAM2 model constructed with ``encoder=sam2_train_model.image_encoder``
            so the backbone is shared with the interactive model.
        convert_inputs: Converts (x, y_instances) torch-em tuples from the
            loader into ``BatchedVideoDatapoint`` expected by SAM2Train.
        interactive_loss: Loss for the interactive branch (e.g. ``CustomSAM2Loss``).
        automatic_loss: Loss for the automatic branch
            (e.g. ``DirectedDistanceLoss``).
        automatic_metric: Validation metric for the automatic branch. Defaults to
            ``automatic_loss`` (matching train_automatic's ``metric=loss``).
        automatic_metric_weight: Weight of the automatic metric in the combined validation
            metric used for checkpointing and early stopping. Defaults to 1/4, putting
            ``DirectedDistanceLoss``'s four summed terms on the scale of the interactive Dice
            metric. Set to 0 to select purely on the interactive task.
        clip_grad_norm: Max gradient norm for clipping. Disabled by default; SAM2's own
            finetuning uses 0.1.
        kwargs: Forwarded to ``Sam2Trainer``. ``model`` should be the
            sam2_train_model (or its DDP wrapper); ``train_loader`` and
            ``val_loader`` must yield ``(x, y)`` where ``y`` has 5 channels as
            described above.
    """

    def __init__(
        self,
        unetr: torch.nn.Module,
        convert_inputs: Callable,
        interactive_loss: torch.nn.Module,
        automatic_loss: torch.nn.Module,
        automatic_metric: Optional[torch.nn.Module] = None,
        automatic_metric_weight: float = 0.25,
        clip_grad_norm: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            convert_inputs=convert_inputs, loss=interactive_loss, clip_grad_norm=clip_grad_norm, **kwargs,
        )
        self.unetr = unetr
        self.automatic_loss = automatic_loss
        # Automatic metric mirrors train_automatic's metric=loss convention when not given.
        self.automatic_metric = automatic_metric if automatic_metric is not None else automatic_loss
        self.automatic_metric_weight = automatic_metric_weight

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        super().save_checkpoint(
            name, current_metric, best_metric, unetr_state=self.unetr.state_dict(), **extra_save_dict,
        )

    def load_checkpoint(self, checkpoint="best"):
        save_dict = super().load_checkpoint(checkpoint)
        if save_dict is not None and "unetr_state" in save_dict:
            self.unetr.load_state_dict(save_dict["unetr_state"])
            self.unetr.to(self.device)
        return save_dict

    def _interactive_step(self, x, y):
        # Slice channel 0 (instance IDs) from the 5-channel joint label tensor.
        return super()._interactive_step(x, y[:, 0:1].to(torch.int64))

    def _automatic_step(self, x, y, retain_grad=False):
        x = x.to(self.device, non_blocking=True)
        # Channels 1-4: foreground mask + 3 directed-distance channels.
        y_dist = y[:, 1:].to(self.device, non_blocking=True)
        pred = self.unetr(x)
        if retain_grad:
            pred.retain_grad()
        return self.automatic_loss(pred, y_dist), y_dist, pred

    def _sync_automatic_grads(self):
        """All-reduce every gradient produced by the automatic backward pass.

        The automatic branch bypasses the DDP wrapper, so nothing here has been reduced
        already - including the encoder it shares with the interactive model.
        """
        if not (dist.is_available() and dist.is_initialized()):
            return
        for p in self.unetr.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

    def _clip_automatic_grads(self):
        """Clip the automatic branch to the same norm as the interactive branch."""
        if self.clip_grad_norm is None:
            return None
        grad_norm = torch.nn.utils.clip_grad_norm_(self.unetr.parameters(), self.clip_grad_norm)
        return grad_norm.item()

    def _backprop_automatic(self, loss):
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.clip_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
            self._sync_automatic_grads()
            grad_norm = self._clip_automatic_grads()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self._sync_automatic_grads()
            grad_norm = self._clip_automatic_grads()
            self.optimizer.step()
        return grad_norm

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()
        self.unetr.train()

        if hasattr(self.train_loader, "batch_sampler") and \
                hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(self._epoch)

        n_iter = 0
        t_per_iter = time.time()
        input_check_done = False

        for x, y in self.train_loader:
            input_check_done = self._check_input_normalization(x, input_check_done)

            self.optimizer.zero_grad()
            with forward_context():
                try:
                    inter_loss, batch, outputs = self._interactive_step(x, y)
                except RuntimeError as e:
                    if "no objects found" in str(e):
                        continue
                    raise
            self._sam2_backprop(inter_loss)

            log_imgs = (self._iteration % self.log_image_interval == 0)
            self.optimizer.zero_grad()
            with forward_context():
                auto_loss, y_dist, pred = self._automatic_step(x, y, retain_grad=log_imgs)
            self._backprop_automatic(auto_loss)

            if self.logger is not None:
                lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_train(
                    self._iteration, inter_loss.item() + auto_loss.item(), lr,
                    inter_loss.item(), auto_loss.item(),
                    x=x if log_imgs else None,
                    y=y if log_imgs else None,
                    batch=batch if log_imgs else None,
                    outputs=outputs if log_imgs else None,
                    y_dist=y_dist if log_imgs else None,
                    pred=pred if log_imgs else None,
                )

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        return (time.time() - t_per_iter) / n_iter

    def _validate_impl(self, forward_context):
        self.model.eval()
        self.unetr.eval()

        inter_loss_val = 0.0
        inter_metric_val = 0.0
        auto_loss_val = 0.0
        auto_metric_val = 0.0
        n_iter = 0
        input_check_done = False
        log_x = log_y = log_batch = log_outputs = log_y_dist = log_pred = None

        # Pick one validation sample to log via reservoir sampling, seeded per epoch.
        log_rng = random.Random(VALIDATION_SEED + self._epoch)

        # Pin the RNGs so the sampled prompts are identical every epoch.
        with _pinned_validation_rng(self.model):
            with torch.no_grad():
                for i, (x, y) in enumerate(self.val_loader):
                    input_check_done = self._check_input_normalization(x, input_check_done)
                    with forward_context():
                        inter_loss, batch, outputs = self._interactive_step(x, y)
                        inter_loss_val += inter_loss.item()
                        inter_metric_val += self.metric(outputs, batch.masks).item()
                        auto_loss, y_dist, pred = self._automatic_step(x, y)
                        auto_loss_val += auto_loss.item()
                        auto_metric_val += self.automatic_metric(pred, y_dist).item()
                    n_iter += 1
                    if log_rng.random() < 1.0 / (i + 1):
                        log_x, log_y, log_batch, log_outputs = x, y, batch, outputs
                        log_y_dist, log_pred = y_dist, pred

        n_iter = max(n_iter, 1)
        inter_loss_val /= n_iter
        inter_metric_val /= n_iter
        auto_loss_val /= n_iter
        auto_metric_val /= n_iter
        combined_metric = inter_metric_val + self.automatic_metric_weight * auto_metric_val

        # Synchronize across DDP ranks so every rank makes the same early-stopping decision.
        if dist.is_available() and dist.is_initialized():
            stats = torch.tensor(
                [inter_loss_val, inter_metric_val, auto_loss_val, auto_metric_val, combined_metric],
                device=self.device,
            )
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            inter_loss_val, inter_metric_val, auto_loss_val, auto_metric_val, combined_metric = (
                stats[0].item(), stats[1].item(), stats[2].item(), stats[3].item(), stats[4].item(),
            )

        if self.logger is not None:
            self.logger.log_validation(
                self._iteration, inter_loss_val, inter_metric_val, auto_loss_val, auto_metric_val,
                combined_metric, log_x, log_y, log_batch, log_outputs, log_y_dist, log_pred,
            )

        return combined_metric


class JointSam2Logger(TorchEmLogger):
    """TensorBoard logger for JointSam2Trainer. Only rank 0 writes in DDP."""

    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        rank = getattr(trainer, "rank", None)
        if rank is not None and rank != 0:
            self.tb = None
            return
        self.log_dir = (
            f"./logs/{trainer.name}" if save_root is None
            else os.path.join(save_root, "logs", trainer.name)
        )
        os.makedirs(self.log_dir, exist_ok=True)
        self.tb = SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def _masks_for_batch_item(self, batch, b=0, t=0):
        b_indices = batch.obj_to_frame_idx[t, :, 1]
        return batch.masks[t][b_indices == b]

    def _log_interactive_images(self, step, x, y, batch, outputs, prefix):
        is_3d = (x.ndim == 5)
        raw = x[0, :3, 0].cpu().float() if is_3d else x[0, :3].cpu().float()
        H, W = raw.shape[-2:]
        self.tb.add_image(f"{prefix}/interactive/raw", raw, step)

        gt_lbl = y[0, 0, 0].cpu().numpy() if is_3d else y[0, 0].cpu().numpy()
        self.tb.add_image(f"{prefix}/interactive/gt_all", _colorize_instance_map(gt_lbl), step)

        gt_chosen = self._masks_for_batch_item(batch, b=0, t=0)
        self.tb.add_image(
            f"{prefix}/interactive/gt_chosen", _overlay_binary_masks(gt_chosen, target_hw=(H, W)), step
        )

        b_indices = batch.obj_to_frame_idx[0, :, 1]
        pred_step0 = outputs[0]["multistep_pred_masks_high_res"][b_indices == 0, 0].detach()
        self.tb.add_image(
            f"{prefix}/interactive/predictions", _overlay_binary_masks(pred_step0 > 0, target_hw=(H, W)), step
        )

    def _log_automatic_images(self, step, x, y_dist, pred, prefix):
        from torch_em.trainer.tensorboard_logger import make_grid_image, normalize_im

        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]
        image = normalize_im(x[selection].cpu())
        self.tb.add_image(f"{prefix}/automatic/input", image, step)
        gradients = pred.grad if pred.grad is not None else None
        im, im_name = make_grid_image(image, y_dist, pred, selection, gradients)
        self.tb.add_image(f"{prefix}/automatic/{im_name}", im, step)

    def log_train(self, step, total_loss, lr, interactive_loss, auto_loss,
                  x=None, y=None, batch=None, outputs=None, y_dist=None, pred=None):
        if self.tb is None:
            return
        self.tb.add_scalar("train/total_loss", total_loss, global_step=step)
        self.tb.add_scalar("train/interactive_loss", interactive_loss, global_step=step)
        self.tb.add_scalar("train/automatic_loss", auto_loss, global_step=step)
        self.tb.add_scalar("train/learning_rate", lr, global_step=step)
        if x is not None:
            self._log_interactive_images(step, x, y, batch, outputs, "train")
            self._log_automatic_images(step, x, y_dist, pred, "train")

    def log_validation(
        self, step, interactive_loss, interactive_metric, auto_loss, auto_metric, combined_metric,
        x=None, y=None, batch=None, outputs=None, y_dist=None, pred=None,
    ):
        if self.tb is None:
            return
        self.tb.add_scalar("validation/interactive_loss", interactive_loss, global_step=step)
        self.tb.add_scalar("validation/interactive_metric", interactive_metric, global_step=step)
        self.tb.add_scalar("validation/automatic_loss", auto_loss, global_step=step)
        self.tb.add_scalar("validation/automatic_metric", auto_metric, global_step=step)
        self.tb.add_scalar("validation/combined_metric", combined_metric, global_step=step)
        if x is not None:
            self._log_interactive_images(step, x, y, batch, outputs, "validation")
            self._log_automatic_images(step, x, y_dist, pred, "validation")
