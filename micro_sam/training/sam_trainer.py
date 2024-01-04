import os
import time
import random
import warnings
from typing import Optional

import numpy as np
import torch
import torch_em

from torchvision.utils import make_grid
from torch_em.trainer.logger_base import TorchEmLogger

from ..prompt_generators import PromptGeneratorBase, IterativePromptGenerator


class SamTrainer(torch_em.trainer.DefaultTrainer):
    """Trainer class for training the Segment Anything model.

    This class is derived from `torch_em.trainer.DefaultTrainer`.
    Check out https://github.com/constantinpape/torch-em/blob/main/torch_em/trainer/default_trainer.py
    for details on its usage and implementation.

    Args:
        convert_inputs: The class that converts outputs of the dataloader to the expected input format of SAM.
            The class `micro_sam.training.util.ConvertToSamInputs` can be used here.
        n_sub_iteration: The number of iteration steps for which the masks predicted for one object are updated.
            In each sub-iteration new point prompts are sampled where the model was wrong.
        n_objects_per_batch: If not given, we compute the loss for all objects in a sample.
            Otherwise the loss computation is limited to n_objects_per_batch, and the objects are randomly sampled.
        mse_loss: The regression loss to compare the IoU predicted by the model with the true IoU.
        prompt_generator: The iterative prompt generator which takes care of the iterative prompting logic for training
        mask_prob: The probability of using the mask inputs in the iterative prompting (per `n_sub_iteration`)
        **kwargs: The keyword arguments of the DefaultTrainer super class.
    """

    def __init__(
        self,
        convert_inputs,
        n_sub_iteration: int,
        n_objects_per_batch: Optional[int] = None,
        mse_loss: torch.nn.Module = torch.nn.MSELoss(),
        prompt_generator: PromptGeneratorBase = IterativePromptGenerator(),
        mask_prob: float = 0.5,
        **kwargs
    ):
        # We have to use the Dice Loss with reduce channel set to None.
        # Hence we hard-code it here to avoid issues by passsing wrong options for the loss.
        dice_loss = torch_em.loss.DiceLoss(reduce_channel=None)
        super().__init__(loss=dice_loss, metric=dice_loss, **kwargs)
        self.convert_inputs = convert_inputs
        self.mse_loss = mse_loss
        self.n_objects_per_batch = n_objects_per_batch
        self.n_sub_iteration = n_sub_iteration
        self.prompt_generator = prompt_generator
        self.mask_prob = mask_prob
        self._kwargs = kwargs

    def _get_prompt_and_multimasking_choices(self, current_iteration):
        """Choose the type of prompts we sample for training, and then we call
        'convert_inputs' with the correct prompting from here.
        """
        if current_iteration % 2 == 0:  # sample only a single point per object
            n_pos, n_neg = 1, 0
            get_boxes = False
            multimask_output = True

        else:  # sample only a single box per object
            n_pos, n_neg = 0, 0
            get_boxes = True
            multimask_output = False

        return n_pos, n_neg, get_boxes, multimask_output

    def _get_prompt_and_multimasking_choices_for_val(self, current_iteration):
        """Choose the type of prompts we sample for validation, and then we call
        'convert_inputs' with the correct prompting from here.
        """
        if current_iteration % 4 == 0:  # sample only a single point per object
            n_pos, n_neg = 1, 0
            get_boxes = False
            multimask_output = True

        elif current_iteration % 4 == 1:  # sample only a single box per object
            n_pos, n_neg = 0, 0
            get_boxes = True
            multimask_output = False

        elif current_iteration % 4 == 2:  # sample a random no. of points
            pos_range, neg_range = 4, 4

            n_pos = np.random.randint(1, pos_range + 1)
            if n_pos == 1:  # to avoid (1, 0) combination for redundancy but still have (n_pos, 0)
                n_neg = np.random.randint(1, neg_range + 1)
            else:
                n_neg = np.random.randint(0, neg_range + 1)
            get_boxes = False
            multimask_output = False

        else:  # sample boxes AND random no. of points
            # here we can have (1, 0) because we also have box
            pos_range, neg_range = 4, 4

            n_pos = np.random.randint(1, pos_range + 1)
            n_neg = np.random.randint(0, neg_range + 1)
            get_boxes = True
            multimask_output = False

        return n_pos, n_neg, get_boxes, multimask_output

    def _compute_iou(self, pred, true, eps=1e-7):
        """Compute the IoU score between the prediction and target.
        """
        pred_mask = pred > 0.5  # binarizing the output predictions
        overlap = pred_mask.logical_and(true).sum(dim=(1, 2, 3))
        union = pred_mask.logical_or(true).sum(dim=(1, 2, 3))
        iou = overlap / (union + eps)
        return iou

    def _compute_loss(self, batched_outputs, y_one_hot):
        """Compute the loss for one iteration. The loss is made up of two components:
        - The mask loss: dice score between the predicted masks and targets.
        - The IOU loss: L2 loss between the predicted IOU and the actual IOU of prediction and target.
        """
        mask_loss, iou_regression_loss = 0.0, 0.0

        # Loop over the batch.
        for batch_output, targets in zip(batched_outputs, y_one_hot):

            predicted_objects = torch.sigmoid(batch_output["masks"])
            # Compute the dice scores for the 1 or 3 predicted masks per true object (outer loop).
            # We swap the axes that go into the dice loss so that the object axis
            # corresponds to the channel axes. This ensures that the dice is computed
            # independetly per channel. We do not reduce the channel axis in the dice,
            # so that we can take the minimum (best score) of the 1/3 predicted masks per object.
            dice_scores = torch.stack([
                self.loss(predicted_objects[:, i:i+1].swapaxes(0, 1), targets.swapaxes(0, 1))
                for i in range(predicted_objects.shape[1])
            ])
            dice_scores, _ = torch.min(dice_scores, dim=0)

            # Compute the actual IOU between the predicted and true objects.
            # The outer loop is for the 1 or 3 predicted masks per true object.
            with torch.no_grad():
                true_iou = torch.stack([
                    self._compute_iou(predicted_objects[:, i:i+1], targets) for i in range(predicted_objects.shape[1])
                ])
            # Compute the L2 loss between true and predicted IOU. We need to swap the axes so that
            # the object axis is back in the first dimension.
            iou_score = self.mse_loss(true_iou.swapaxes(0, 1), batch_output["iou_predictions"])

            mask_loss = mask_loss + torch.mean(dice_scores)
            iou_regression_loss = iou_regression_loss + iou_score

        loss = mask_loss + iou_regression_loss

        return loss, mask_loss, iou_regression_loss

    #
    # Functionality for iterative prompting loss
    #

    def _get_best_masks(self, batched_outputs, batched_iou_predictions):
        # Batched mask and logit (low-res mask) predictions.
        masks = torch.stack([m["masks"] for m in batched_outputs])
        logits = torch.stack([m["low_res_masks"] for m in batched_outputs])

        # Determine the best IOU across the multi-object prediction axis
        # and turn this into a mask we can use for indexing.
        # See https://stackoverflow.com/questions/72628000/pytorch-indexing-by-argmax
        # for details on the indexing logic.
        best_iou_idx = torch.argmax(batched_iou_predictions, dim=2, keepdim=True)
        best_iou_idx = torch.zeros_like(batched_iou_predictions).scatter(2, best_iou_idx, value=1).bool()

        # Index the mask and logits with the best iou indices.
        # Note that we squash the first two axes (batch x objects) into one when indexing.
        # That's why we need to reshape bax into (batch x objects) using a view.
        # We also keep the multi object axis as a singleton, that's why the view has (batch_size, n_objects, 1, ...)
        batch_size, n_objects = masks.shape[:2]
        h, w = masks.shape[-2:]
        masks = masks[best_iou_idx].view(batch_size, n_objects, 1, h, w)

        h, w = logits.shape[-2:]
        logits = logits[best_iou_idx].view(batch_size, n_objects, 1, h, w)

        # Binarize the mask. Note that the mask here also contains logits, so we use 0.0
        # as threshold instead of using 0.5. (Hence we don't need to apply a sigmoid)
        masks = (masks > 0.0).float()
        return masks, logits

    def _compute_iterative_loss(self, batched_inputs, y_one_hot, num_subiter, multimask_output):
        """Compute the loss for several (sub-)iterations of iterative prompting.
        In each iterations the prompts are updated based on the previous predictions.
        """
        image_embeddings, batched_inputs = self.model.image_embeddings_oft(batched_inputs)

        loss, mask_loss, iou_regression_loss, mean_model_iou = 0.0, 0.0, 0.0, 0.0

        for i in range(0, num_subiter):
            # We do multimasking only in the first sub-iteration as we then pass single prompt
            # after the first sub-iteration, we don't do multimasking because we get multiple prompts.
            batched_outputs = self.model(batched_inputs,
                                         image_embeddings=image_embeddings,
                                         multimask_output=multimask_output if i == 0 else False)

            # Compute loss for tis sub-iteration.
            net_loss, net_mask_loss, net_iou_regression_loss = self._compute_loss(batched_outputs, y_one_hot)

            # Compute the mean IOU predicted by the model. We keep track of this in the logger.
            batched_iou_predictions = torch.stack([m["iou_predictions"] for m in batched_outputs])
            with torch.no_grad():
                net_mean_model_iou = torch.mean(batched_iou_predictions)

            loss += net_loss
            mask_loss += net_mask_loss
            iou_regression_loss += net_iou_regression_loss
            mean_model_iou += net_mean_model_iou

            # Determine the next prompts based on current predictions.
            with torch.no_grad():
                # Get the mask and logit predictions corresponding to the predicted object
                # (per actual object) with the best IOU.
                masks, logits = self._get_best_masks(batched_outputs, batched_iou_predictions)
                batched_inputs = self._update_prompts(batched_inputs, y_one_hot, masks, logits)

        loss = loss / num_subiter
        mask_loss = mask_loss / num_subiter
        iou_regression_loss = iou_regression_loss / num_subiter
        mean_model_iou = mean_model_iou / num_subiter

        return loss, mask_loss, iou_regression_loss, mean_model_iou

    def _update_prompts(self, batched_inputs, y_one_hot, masks, logits_masks):
        # here, we get the pair-per-batch of predicted and true elements (and also the "batched_inputs")
        for x1, x2, _inp, logits in zip(masks, y_one_hot, batched_inputs, logits_masks):
            # here, we get each object in the pairs and do the point choices per-object
            net_coords, net_labels, _, _ = self.prompt_generator(x2, x1)

            # convert the point coordinates to the expected resolution for iterative prompting
            # NOTE:
            #   - "only" need to transform the point prompts from the iterative prompting
            #   - the `logits` are the low res masks (256, 256), hence do not need the transform
            net_coords = self.model.transform.apply_coords_torch(net_coords, y_one_hot.shape[-2:])

            updated_point_coords = torch.cat([_inp["point_coords"], net_coords], dim=1) \
                if "point_coords" in _inp.keys() else net_coords
            updated_point_labels = torch.cat([_inp["point_labels"], net_labels], dim=1) \
                if "point_labels" in _inp.keys() else net_labels

            _inp["point_coords"] = updated_point_coords
            _inp["point_labels"] = updated_point_labels

            if self.mask_prob > 0:
                # using mask inputs for iterative prompting while training, with a probability
                use_mask_inputs = (random.random() < self.mask_prob)
                if use_mask_inputs:
                    _inp["mask_inputs"] = logits
                else:  # remove  previously existing mask inputs to avoid using them in next sub-iteration
                    _inp.pop("mask_inputs", None)

        return batched_inputs

    #
    # Training Loop
    #

    def _preprocess_batch(self, batched_inputs, y, sampled_ids):
        """Compute one hot target (one mask per channel) for the sampled ids
        and restrict the number of sampled objects to the minimal number in the batch.
        """
        assert len(y) == len(sampled_ids)

        # Get the minimal number of objects in this batch.
        # The number of objects in a patch might be < n_objects_per_batch.
        # This is why we need to restrict it here to ensure the same
        # number of objects across the batch.
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

    def _interactive_train_iteration(self, x, y):
        n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices(self._iteration)

        batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos, n_neg, get_boxes, self.n_objects_per_batch)
        batched_inputs, y_one_hot = self._preprocess_batch(batched_inputs, y, sampled_ids)

        loss, mask_loss, iou_regression_loss, model_iou = self._compute_iterative_loss(
            batched_inputs, y_one_hot,
            num_subiter=self.n_sub_iteration, multimask_output=multimask_output
        )
        return loss, mask_loss, iou_regression_loss, model_iou, y_one_hot

    def _check_input_normalization(self, x, input_check_done):
        # The expected data range of the SAM model is 8bit (0-255).
        # It can easily happen that data is normalized beforehand in training.
        # For some reasons we don't fully understand this still works, but it
        # should still be avoided and is very detrimental in some settings
        # (e.g. when freezing the image encoder)
        # We check once per epoch if the data seems to be normalized already and
        # raise a warning if this is the case.
        if not input_check_done:
            data_min, data_max = x.min(), x.max()
            if (data_min < 0) or (data_max < 1):
                warnings.warn(
                    "It looks like you are normalizing the training data."
                    "The SAM model takes care of normalization, so it is better to not do this."
                    "We recommend to remove data normalization and input data in the range [0, 255]."
                )
            input_check_done = True

        return input_check_done

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        input_check_done = False

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:
            input_check_done = self._check_input_normalization(x, input_check_done)

            self.optimizer.zero_grad()

            with forward_context():
                (loss, mask_loss, iou_regression_loss, model_iou,
                 sampled_binary_y) = self._interactive_train_iteration(x, y)

            backprop(loss)

            if self.logger is not None:
                lr = [pm["lr"] for pm in self.optimizer.param_groups][0]
                samples = sampled_binary_y if self._iteration % self.log_image_interval == 0 else None
                self.logger.log_train(self._iteration, loss, lr, x, y, samples,
                                      mask_loss, iou_regression_loss, model_iou)

            self._iteration += 1
            n_iter += 1
            if self._iteration >= self.max_iteration:
                break
            progress.update(1)

        t_per_iter = (time.time() - t_per_iter) / n_iter
        return t_per_iter

    def _interactive_val_iteration(self, x, y, val_iteration):
        n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices_for_val(val_iteration)

        batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos, n_neg, get_boxes, self.n_objects_per_batch)
        batched_inputs, y_one_hot = self._preprocess_batch(batched_inputs, y, sampled_ids)

        image_embeddings, batched_inputs = self.model.image_embeddings_oft(batched_inputs)

        batched_outputs = self.model(
            batched_inputs,
            image_embeddings=image_embeddings,
            multimask_output=multimask_output,
        )

        loss, mask_loss, iou_regression_loss = self._compute_loss(batched_outputs, y_one_hot)
        # We use the dice loss over the masks as metric.
        metric = mask_loss
        model_iou = torch.mean(torch.stack([m["iou_predictions"] for m in batched_outputs]))

        return loss, mask_loss, iou_regression_loss, model_iou, y_one_hot, metric

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

                loss_val += loss.item()
                metric_val += metric.item()
                model_iou_val += model_iou.item()
                val_iteration += 1

        loss_val /= len(self.val_loader)
        metric_val /= len(self.val_loader)
        model_iou_val /= len(self.val_loader)
        print()
        print(f"The Average Dice Score for the Current Epoch is {1 - metric_val}")

        if self.logger is not None:
            self.logger.log_validation(
                self._iteration, metric_val, loss_val, x, y,
                sampled_binary_y, mask_loss, iou_regression_loss, model_iou_val
            )

        return metric_val


class SamLogger(TorchEmLogger):
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

    def log_train(self, step, loss, lr, x, y, samples, mask_loss, iou_regression_loss, model_iou):
        self.tb.add_scalar(tag="train/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="train/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="train/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="train/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="train/learning_rate", scalar_value=lr, global_step=step)
        if step % self.log_image_interval == 0:
            self.add_image(x, y, samples, "train", step)

    def log_validation(self, step, metric, loss, x, y, samples, mask_loss, iou_regression_loss, model_iou):
        self.tb.add_scalar(tag="validation/loss", scalar_value=loss, global_step=step)
        self.tb.add_scalar(tag="validation/mask_loss", scalar_value=mask_loss, global_step=step)
        self.tb.add_scalar(tag="validation/iou_loss", scalar_value=iou_regression_loss, global_step=step)
        self.tb.add_scalar(tag="validation/model_iou", scalar_value=model_iou, global_step=step)
        self.tb.add_scalar(tag="validation/metric", scalar_value=metric, global_step=step)
        self.add_image(x, y, samples, "validation", step)
