import time

import torch
import torch.nn as nn

from torch_em.trainer import DefaultTrainer


class SemanticSamTrainer(DefaultTrainer):
    """
    """
    def __init__(
        self,
        convert_inputs,
        num_classes: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.convert_inputs = convert_inputs
        self.num_classes = num_classes
        self.compute_ce_loss = nn.BCELoss() if num_classes == 1 else nn.CrossEntropyLoss()
        self._kwargs = kwargs

    def _compute_loss(self, y, downsized_gt, masks, mask_logits):
        # Compute dice loss for the predictions
        dice_loss = self.loss(masks, y.to(self.device, non_blocking=True))

        # Compute cross entropy loss for the logits
        ce_loss = self.compute_ce_loss(mask_logits, downsized_gt.to(self.device, non_blocking=True))

        net_loss = dice_loss + ce_loss
        return net_loss

    def _get_model_outputs(self, batched_inputs):
        image_embeddings, batched_inputs = self.model.image_embeddings_oft(batched_inputs)
        batched_outputs = self.model(batched_inputs, image_embeddings, multimask_output=(self.num_classes > 1))
        masks = torch.stack([output["masks"].squeeze(0) for output in batched_outputs])
        mask_logits = torch.stack([output["low_res_masks"].squeeze(0) for output in batched_outputs])
        return masks, mask_logits

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        t_per_iter = time.time()
        for x, y in self.train_loader:
            self.optimizer.zero_grad()

            batched_inputs, downsized_gt = self.convert_inputs(x, y)

            with forward_context():
                masks, mask_logits = self._get_model_outputs(batched_inputs)
                net_loss = self._compute_loss(y, downsized_gt, masks, mask_logits)

            backprop(net_loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_train(self._iteration, net_loss, lr, x, y, masks, log_gradients=True)

            self._iteration += 1
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
                batched_inputs, downsized_gt = self.convert_inputs(x, y)

                with forward_context():
                    masks, mask_logits = self._get_model_outputs(batched_inputs)
                    net_loss = self._compute_loss(y, downsized_gt, masks, mask_logits)

                loss_val += net_loss.item()
                metric_val += net_loss.item()

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        print()
        print(f"The Average Validation Metric Score for the Current Epoch is {1 - metric_val}")

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, masks)

        return metric_val
