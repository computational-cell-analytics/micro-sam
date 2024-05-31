import os
import time

import torch
from torchvision.utils import make_grid

import torch_em
from torch_em.trainer.logger_base import TorchEmLogger


class MedSAMTrainer(torch_em.trainer.DefaultTrainer):
    """Trainer class for replicating the training of MedSAM (https://arxiv.org/abs/2304.12306)
    Reference: https://github.com/bowang-lab/MedSAM

    This class is derived from `torch_em.trainer.DefaultTrainer`.
    Check out https://github.com/constantinpape/torch-em/blob/main/torch_em/trainer/default_trainer.py
    for details on its usage and implementation

    Args:
        convert_inputs: The class that converts outputs of the dataloader to the expected input format of SAM.
            The class `micro_sam.training.util.ConvertToSamInputs` can be used here.
        **kwargs: The keyword arguments of the DefaultTrainer super class.
    """
    def __init__(
        self,
        convert_inputs,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.convert_inputs = convert_inputs
        self._kwargs = kwargs

    def _seg_loss(self, masks, y_one_hot, get_metric=False):
        predicted_labels = torch.stack(
            [torch.stack([torch.sigmoid(m) for m in mask]) for mask in masks]
        )

        loss = self.loss(predicted_labels, y_one_hot)

        if get_metric:
            metric = self.metric(predicted_labels, y_one_hot)
            return loss, metric
        else:
            return loss

    def _preprocess_batch(self, batched_inputs, y, sampled_ids):
        assert len(y) == len(sampled_ids)

        n_objects = min(len(ids) for ids in sampled_ids)

        y = y.to(self.device)
        # Compute the one hot targets for the seg-id.
        y_one_hot = torch.stack([
            torch.stack([target == seg_id for seg_id in ids[:n_objects]])
            for target, ids in zip(y, sampled_ids)
        ]).float()

        # Also restrict the prompts to the number of objects.
        batched_inputs = [
            {k: (v[:n_objects] if k in ("point_coords", "point_labels", "boxes") else v) for k, v in inp.items()}
            for inp in batched_inputs
        ]
        return batched_inputs, y_one_hot

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            self.optimizer.zero_grad()

            with forward_context():
                batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos=0, n_neg=0, get_boxes=True)
                batched_inputs, y_one_hot = self._preprocess_batch(batched_inputs, y, sampled_ids)

                image_embeddings, batched_inputs = self.model.image_embeddings_oft(batched_inputs)

                batched_outputs = self.model(
                    batched_inputs, image_embeddings=image_embeddings, multimask_output=False
                )

                masks = [m["masks"] for m in batched_outputs]
                loss = self._seg_loss(masks, y_one_hot)

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = y_one_hot if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, x, y, samples)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _validate_impl(self, forward_context):
        self.model.eval()

        metric_val, loss_val = 0.0, 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                with forward_context():
                    batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos=0, n_neg=0, get_boxes=True)

                    image_embeddings, batched_inputs = self.model.image_embeddings_oft(batched_inputs)

                    batched_outputs = self.model(
                        batched_inputs, image_embeddings=image_embeddings, multimask_output=False
                    )

                    masks = [m["masks"] for m in batched_outputs]
                    sampled_binary_y = torch.stack(
                        [torch.isin(y[i], torch.tensor(sampled_ids[i])) for i in range(len(y))]
                    ).to(torch.float32)
                    loss, metric = self._seg_loss(masks, sampled_binary_y, get_metric=True)

                loss_val += loss.item()
                metric_val += metric.item()

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        print()
        print(f"The Average Dice Score for the Current Epoch is {1 - metric_val}")

        if self.logger is not None:
            self.logger.log_validation(self._iteration, metric_val, loss_val, x, y, sampled_binary_y)

        return metric_val


class MedSAMLogger(TorchEmLogger):
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

    def log_train(self, step, loss, lr, x, y, samples):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, samples, "train", step)

    def log_validation(self, step, metric, loss, x, y, samples):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, samples, "validation", step)
