import time
from typing import Optional

import torch
import torch.nn as nn

from torch_em.loss import DiceLoss
from torch_em.trainer import DefaultTrainer


class CustomDiceLoss(nn.Module):
    def __init__(self, num_classes: int, softmax: bool = True) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_loss = DiceLoss()
        self.softmax = softmax

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def __call__(self, pred, target):
        if self.softmax:
            pred = torch.softmax(pred, dim=1)
        target = self._one_hot_encoder(target)
        loss = self.dice_loss(pred, target)
        return loss


class SemanticSamTrainer(DefaultTrainer):
    """
    """
    def __init__(
        self,
        convert_inputs,
        num_classes: int,
        dice_weight: Optional[float] = None,
        **kwargs
    ):
        assert num_classes > 1

        loss = CustomDiceLoss(num_classes=num_classes)
        metric = CustomDiceLoss(num_classes=num_classes)
        super().__init__(loss=loss, metric=metric, **kwargs)

        self.convert_inputs = convert_inputs
        self.num_classes = num_classes
        self.compute_ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self._kwargs = kwargs

    def _compute_loss(self, y, masks):
        target = y.to(self.device, non_blocking=True)
        # Compute dice loss for the predictions
        dice_loss = self.loss(masks, target)

        # Compute cross entropy loss for the predictions
        ce_loss = self.compute_ce_loss(masks, target.squeeze(1).long())

        if self.dice_weight is None:
            net_loss = dice_loss + ce_loss
        else:
            net_loss = self.dice_weight * dice_loss + (1 - self.dice_weight) * ce_loss

        return net_loss

    def _get_model_outputs(self, batched_inputs):
        # Precompute the image embeddings if the model exposes it as functionality.
        if hasattr(self.model, "image_embeddings_oft"):
            image_embeddings, batched_inputs = self.model.image_embeddings_oft(batched_inputs)
            batched_outputs = self.model(batched_inputs, image_embeddings, multimask_output=True)
        else:  # Otherwise we assume that the embeddings are computed internally as part of the forward pass.
            # We need to take care of sending things to the device here.
            batched_inputs = [
                {"image": inp["image"].to(self.device, non_blocking=True), "original_size": inp["original_size"]}
                for inp in batched_inputs
            ]
            batched_outputs = self.model(batched_inputs, multimask_output=True)

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
                self.logger.log_train(
                    self._iteration, net_loss, lr, x, y, torch.softmax(masks, dim=1), log_gradients=False
                )

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
        dice_metric = 1 - (metric_val / self.num_classes)
        print()
        print(f"The Average Validation Metric Score for the Current Epoch is {dice_metric}")

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, torch.softmax(masks, dim=1))

        return metric_val
