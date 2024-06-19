import time

import torch

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
        self._kwargs = kwargs

    def _compute_loss(self, y, masks):
        # Compute dice loss for the predictions
        dice_loss = self.loss(masks, y.to(self.device, non_blocking=True))
        return dice_loss

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
                loss = self._compute_loss(y, masks)

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                self.logger.log_train(self._iteration, loss, lr, x, y, masks, log_gradients=True)

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
                batched_inputs = self.convert_inputs(x, y)

                with forward_context():
                    masks = self._get_model_outputs(batched_inputs)
                    loss = self._compute_loss(y, masks)

                loss_val += loss.item()
                metric_val += loss.item()

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        print()
        print(f"The Average Validation Metric Score for the Current Epoch is {1 - metric_val}")

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, masks)

        return metric_val
