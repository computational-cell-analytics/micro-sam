import os
import time
from typing import Optional, Union

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from .sam_trainer import SamTrainer

from torch_em.model import UNETR
from torch_em.trainer.logger_base import TorchEmLogger
from torch_em.model.unet import Decoder, ConvBlock2d, Upsampler2d


class CustomDecoder(Decoder):
    "To make use of the `V-Net` level logic - as we can't make use of the skip connections"
    def forward(self, x):
        for block, sampler in zip(self.blocks, self.samplers):
            x = sampler(x)
            x = block(x)

        return x


class UNETRForJointTraining(UNETR):
    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        out_channels: int = 1,
        final_activation: Optional[Union[str, nn.Module]] = None,
        **kwargs
    ) -> None:
        super().__init__(encoder, out_channels, **kwargs)

        self.encoder = encoder

        # parameters for the decoder network
        depth = 3
        initial_features = 64
        gain = 2
        features_decoder = [initial_features * gain ** i for i in range(depth + 1)][::-1]
        scale_factors = depth * [2]
        self.out_channels = out_channels

        self.decoder = Decoder(
            features=features_decoder,
            scale_factors=scale_factors[::-1],
            conv_block_impl=ConvBlock2d,
            sampler_impl=Upsampler2d
        )

        self.out_conv = nn.Conv2d(features_decoder[-1], out_channels, 1)
        self.final_activation = self._get_activation(final_activation)

    def forward(self, x):
        org_shape = x.shape[-2:]

        x = torch.stack([self.preprocess(e) for e in x], dim=0)

        x = self.encoder(x)
        x = self.decoder(x)

        x = self.out_conv(x)
        if self.final_activation is not None:
            x = self.final_activation(x)

        x = self.postprocess_masks(x, org_shape, org_shape)

        return x


class JointSamTrainer(SamTrainer):
    def __init__(
            self, **kwargs
    ):
        super().__init__(**kwargs)
        self.unetr = UNETRForJointTraining(
            img_size=self.model.img_size,
            backbone="sam",
            encoder=self.model.encoder,
            out_channels=self.model.out_channels,
            use_sam_stats=True,
            final_activation="Sigmoid"
        )

    def _instance_train_iteration(self, x, y):
        # we pass the inputs to the unetr model
        # get the outputs, calculate the dice loss
        # return the dice loss
        instance_loss = ...
        return instance_loss

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        input_check_done = False

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            input_check_done = self._check_input_normalization(x, input_check_done)

            self.optimizer.zero_grad()

            with forward_context():
                # 1. train for the interactive segmentation
                (loss, mask_loss, iou_regression_loss, model_iou,
                 sampled_binary_y) = self._interactive_train_iteration(x, y, self._iteration)

            backprop(loss)

            # let's get the unetr decoder for doing the instance segmentation
            # TODO: we need to ship the weights from the encoder of SAM to UNETR

            with forward_context():
                # 2. train for the automatic instance segmentation
                instance_loss = self._instance_train_iteration(x, y)

            backprop(...)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = sampled_binary_y if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(
                    self._iteration, loss, lr, x, y, samples, mask_loss, iou_regression_loss, model_iou, instance_loss
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
                input_check_done = self._check_input_normalization(x, input_check_done)

                with forward_context():
                    (loss, mask_loss, iou_regression_loss, model_iou,
                     sampled_binary_y, metric) = self._interactive_val_iteration(x, y, val_iteration)

                # TODO: instance segmentation for validation

                loss_val += loss.item()
                metric_val += metric.item()
                model_iou_val += model_iou.item()
                val_iteration += 1

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        model_iou_val /= len(self.val_loader)
        print()
        print(...)  # provide a message for the respective metric score

        if self.logger is not None:
            self.logger.log_validation(
                self._iteration, metric_val, loss_val, x, y, sampled_binary_y,
                mask_loss, iou_regression_loss, model_iou_val, instance_loss
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
        self.tb.add_image(tag=f"{name}/input", img_tensor=x[0], global_step=step)
        self.tb.add_image(tag=f"{name}/target", img_tensor=y[0], global_step=step)
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
