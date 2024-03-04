import os
import time
import numpy as np
from collections import OrderedDict

import torch
from torchvision.utils import make_grid

from .sam_trainer import SamTrainer

from torch_em.trainer.logger_base import TorchEmLogger
from torch_em.trainer.tensorboard_logger import normalize_im


class JointSamTrainer(SamTrainer):
    def __init__(
        self,
        unetr: torch.nn.Module,
        instance_loss: torch.nn.Module,
        instance_metric: torch.nn.Module,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.unetr = unetr
        self.instance_loss = instance_loss
        self.instance_metric = instance_metric

    def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
        current_unetr_state = self.unetr.state_dict()
        decoder_state = []
        for k, v in current_unetr_state.items():
            if not k.startswith("encoder"):
                decoder_state.append((k, v))
        decoder_state = OrderedDict(decoder_state)

        super().save_checkpoint(
            name, current_metric=current_metric, best_metric=best_metric, decoder_state=decoder_state, **extra_save_dict
        )

    def load_checkpoint(self, checkpoint="best"):
        save_dict = super().load_checkpoint(checkpoint)

        # let's get the image encoder params from sam
        sam_state = save_dict["model_state"]
        encoder_state = []
        prune_prefix = "sam.image_"
        for k, v in sam_state.items():
            if k.startswith(prune_prefix):
                encoder_state.append((k[len(prune_prefix):], v))
        encoder_state = OrderedDict(encoder_state)

        # let's get the decoder params from unetr
        decoder_state = save_dict["decoder_state"]

        # now let's merge the two to get the params for the unetr
        unetr_state = OrderedDict(list(encoder_state.items()) + list(decoder_state.items()))

        self.unetr.load_state_dict(unetr_state)
        self.unetr.to(self.device)
        return save_dict

    def _instance_iteration(self, x, y, metric_for_val=False):
        outputs = self.unetr(x.to(self.device))
        loss = self.instance_loss(outputs, y.to(self.device))
        if metric_for_val:
            metric = self.instance_metric(outputs, y.to(self.device))
            return loss, metric
        else:
            return loss

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        input_check_done = False

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            labels_instances = y[:, 0, ...].unsqueeze(1)
            labels_for_unetr = y[:, 1:, ...]

            input_check_done = self._check_input_normalization(x, input_check_done)

            self.optimizer.zero_grad()

            with forward_context():
                # 1. train for the interactive segmentation
                (loss, mask_loss, iou_regression_loss, model_iou,
                 sampled_binary_y) = self._interactive_train_iteration(x, labels_instances)

            backprop(loss)

            self.optimizer.zero_grad()

            with forward_context():
                # 2. train for the automatic instance segmentation
                unetr_loss = self._instance_iteration(x, labels_for_unetr)

            backprop(unetr_loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = sampled_binary_y if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(
                    self._iteration, loss, lr, x, labels_instances, samples,
                    mask_loss, iou_regression_loss, model_iou, unetr_loss
                )

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_impl(self, forward_context):
        self.model.eval()

        input_check_done = False

        val_iteration = 0
        metric_val, loss_val, model_iou_val = 0.0, 0.0, 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                labels_instances = y[:, 0, ...].unsqueeze(1)
                labels_for_unetr = y[:, 1:, ...]

                input_check_done = self._check_input_normalization(x, input_check_done)

                with forward_context():
                    # 1. validate for the interactive segmentation
                    (loss, mask_loss, iou_regression_loss, model_iou,
                     sampled_binary_y, metric) = self._interactive_val_iteration(x, labels_instances, val_iteration)

                with forward_context():
                    # 2. validate for the automatic instance segmentation
                    unetr_loss, unetr_metric = self._instance_iteration(x, labels_for_unetr, metric_for_val=True)

                loss_val += loss.item()
                metric_val += metric.item() + (unetr_metric.item() / 3)
                model_iou_val += model_iou.item()
                val_iteration += 1

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        model_iou_val /= len(self.val_loader)

        if self.logger is not None:
            self.logger.log_validation(
                self._iteration, metric_val, loss_val, x, labels_instances, sampled_binary_y,
                mask_loss, iou_regression_loss, model_iou_val, unetr_loss
            )

        return metric_val


class JointSamLogger(TorchEmLogger):
    """@private"""
    def __init__(self, trainer, save_root, **unused_kwargs):
        super().__init__(trainer, save_root)
        self.log_dir = f"./logs/{trainer.name}" if save_root is None else\
            os.path.join(save_root, "logs", trainer.name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.tb = torch.utils.tensorboard.SummaryWriter(self.log_dir)
        self.log_image_interval = trainer.log_image_interval

    def add_image(self, x, y, samples, name, step):
        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        image = normalize_im(x[selection].cpu())

        self.tb.add_image(tag=f"{name}/input", img_tensor=image, global_step=step)
        self.tb.add_image(tag=f"{name}/target", img_tensor=y[selection], global_step=step)
        sample_grid = make_grid([sample[0] for sample in samples], nrow=4, padding=4)
        self.tb.add_image(tag=f"{name}/samples", img_tensor=sample_grid, global_step=step)

    def log_train(
            self, step, loss, lr, x, y, samples, mask_loss, iou_regression_loss, model_iou, instance_loss
    ):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="train/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="train/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="train/instance_loss", scalar_value=instance_loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, samples, "train", step)

    def log_validation(
            self, step, metric, loss, x, y, samples, mask_loss, iou_regression_loss, model_iou, instance_loss
    ):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="validation/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="validation/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="train/instance_loss", scalar_value=instance_loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, samples, "validation", step)
