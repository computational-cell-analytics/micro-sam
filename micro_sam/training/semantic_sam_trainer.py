import time

import numpy as np

import torch
import torch.nn as nn

from torch_em.loss import DiceLoss
from torch_em.trainer import DefaultTrainer
from torch_em.trainer.tensorboard_logger import TensorboardLogger, normalize_im


class SemanticSamTrainer(DefaultTrainer):
    """
    """
    def __init__(
        self,
        convert_inputs,
        num_classes: int = 1,
        **kwargs
    ):
        loss = DiceLoss()
        metric = DiceLoss()
        super().__init__(loss=loss, metric=metric, **kwargs)

        self.convert_inputs = convert_inputs
        self.num_classes = num_classes
        self.compute_ce_loss = nn.BCELoss() if num_classes == 1 else nn.CrossEntropyLoss()
        self._kwargs = kwargs

    def _compute_loss(self, y, masks):
        target = y.to(self.device, non_blocking=True)
        # Compute dice loss for the predictions
        dice_loss = self.loss(masks, target)
        breakpoint()

        # Compute cross entropy loss for the predictions
        ce_loss = self.compute_ce_loss(masks, target)

        net_loss = dice_loss + ce_loss
        return net_loss

    def _get_model_outputs(self, batched_inputs):
        image_embeddings, batched_inputs = self.model.image_embeddings_oft(batched_inputs)
        batched_outputs = self.model(batched_inputs, image_embeddings, multimask_output=(self.num_classes > 1))
        masks = torch.stack([output["masks"].squeeze(0) for output in batched_outputs])
        return masks

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        t_per_iter = time.time()
        for x, y in self.train_loader:
            self.optimizer.zero_grad()

            batched_inputs = self.convert_inputs(x, y)

            with forward_context():
                masks = self._get_model_outputs(batched_inputs)
                net_loss = self._compute_loss(y, masks)

            backprop(net_loss)

            self._iteration += 1

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_train(self._iteration, net_loss, lr, x, y, masks, log_gradients=False)

            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter)
        return t_per_iter

    def _validate_impl(self, forward_context):
        self.model.eval()

        metric_val, loss_val = 0.0, 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                batched_inputs = self.convert_inputs(x, y)

                with forward_context():
                    masks = self._get_model_outputs(batched_inputs)
                    net_loss = self._compute_loss(y, masks)

                loss_val += net_loss.item()
                metric_val += net_loss.item()

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        print()
        print(f"The Average Validation Metric Score for the Current Epoch is {1 - metric_val}")

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, masks)

        return metric_val


class SemanticSamTrainer3D(SemanticSamTrainer):
    def _get_model_outputs(self, batched_inputs):
        model_input = torch.stack([inp["image"] for inp in batched_inputs]).to(self.device)
        image_size = batched_inputs[0]["original_size"][-1]
        batched_outputs = self.model(
            model_input,
            multimask_output=(self.num_classes > 1),
            image_size=image_size
        )
        # masks = torch.stack([output["masks"].squeeze(0) for output in batched_outputs])
        masks = batched_outputs["masks"]
        return masks


class SemanticSamLogger3D(TensorboardLogger):
    def log_images(self, step, x, y, prediction, name, gradients=None):

        selection_image = np.s_[0] if x.ndim == 4 else np.s_[0, x.shape[2] // 2, :]
        selection = np.s_[0] if x.ndim == 4 else np.s_[0, :, x.shape[2] // 2]

        image = normalize_im(x[selection_image].cpu())
        self.tb.add_image(tag=f"{name}/input",
                          img_tensor=image,
                          global_step=step)

        im, im_name = self.make_image(image, y, prediction, selection, gradients)
        im_name = f"{name}/{im_name}"
        self.tb.add_image(tag=im_name, img_tensor=im, global_step=step)

    def log_train(self, step, loss, lr, x, y, prediction, log_gradients=False):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)

        # the embedding visualisation function currently doesn't support gradients,
        # so we can't log them even if log_gradients is true
        log_grads = log_gradients
        if self.have_embeddings:
            log_grads = False

        if step % self.log_image_interval == 0:
            gradients = prediction.grad if log_grads else None
            self.log_images(step, x, y, prediction, "train", gradients=gradients)

    def log_validation(self, step, metric, loss, x, y, prediction):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.log_images(step, x, y, prediction, "validation")
