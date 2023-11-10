import os
import time
import random
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
        sigmoid: The activation function for normalizing the model output.
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
        _sigmoid: torch.nn.Module = torch.nn.Sigmoid(),
        prompt_generator: PromptGeneratorBase = IterativePromptGenerator(),
        mask_prob: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.convert_inputs = convert_inputs
        self.mse_loss = mse_loss
        self._sigmoid = _sigmoid
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

    def _get_dice(self, input_, target):
        """Using the default "DiceLoss" called by the trainer from "torch_em"
        """
        dice_loss = self.loss(input_, target)
        return dice_loss

    def _get_iou(self, pred, true, eps=1e-7):
        """Getting the IoU score for the predicted and true labels
        """
        pred_mask = pred > 0.5  # binarizing the output predictions
        overlap = pred_mask.logical_and(true).sum()
        union = pred_mask.logical_or(true).sum()
        iou = overlap / (union + eps)
        return iou

    def _get_net_loss(self, batched_outputs, y, sampled_ids):
        """What do we do here? two **separate** things
        1. compute the mask loss: loss between the predicted and ground-truth masks
            for this we just use the dice of the prediction vs. the gt (binary) mask
        2. compute the mask for the "IOU Regression Head": so we want the iou output from the decoder to
            match the actual IOU between predicted and (binary) ground-truth mask. And we use L2Loss / MSE for this.
        """
        masks = [m["masks"] for m in batched_outputs]
        predicted_iou_values = [m["iou_predictions"] for m in batched_outputs]
        with torch.no_grad():
            mean_model_iou = torch.mean(torch.stack([p.mean() for p in predicted_iou_values]))

        mask_loss = 0.0  # this is the loss term for 1.
        iou_regression_loss = 0.0  # this is the loss term for 2.

        # outer loop is over the batch (different image/patch predictions)
        for m_, y_, ids_, predicted_iou_ in zip(masks, y, sampled_ids, predicted_iou_values):
            per_object_dice_scores, per_object_iou_scores = [], []

            # inner loop is over the channels, this corresponds to the different predicted objects
            for i, (predicted_obj, predicted_iou) in enumerate(zip(m_, predicted_iou_)):
                predicted_obj = self._sigmoid(predicted_obj).to(self.device)
                true_obj = (y_ == ids_[i]).to(self.device)

                # this is computing the LOSS for 1.)
                _dice_score = min([self._get_dice(p[None], true_obj) for p in predicted_obj])
                per_object_dice_scores.append(_dice_score)

                # now we need to compute the loss for 2.)
                with torch.no_grad():
                    true_iou = torch.stack([self._get_iou(p[None], true_obj) for p in predicted_obj])
                _iou_score = self.mse_loss(true_iou, predicted_iou)
                per_object_iou_scores.append(_iou_score)

            mask_loss = mask_loss + torch.mean(torch.stack(per_object_dice_scores))
            iou_regression_loss = iou_regression_loss + torch.mean(torch.stack(per_object_iou_scores))

        loss = mask_loss + iou_regression_loss

        return loss, mask_loss, iou_regression_loss, mean_model_iou

    def _postprocess_outputs(self, masks):
        """ "masks" look like -> (B, 1, X, Y)
        where, B is the number of objects, (X, Y) is the input image shape
        """
        instance_labels = []
        for m in masks:
            instance_list = [self._sigmoid(_val) for _val in m.squeeze(1)]
            instance_label = torch.stack(instance_list, dim=0).sum(dim=0).clip(0, 1)
            instance_labels.append(instance_label)
        instance_labels = torch.stack(instance_labels).unsqueeze(1)
        return instance_labels

    def _get_val_metric(self, batched_outputs, sampled_binary_y):
        """ Tracking the validation metric based on the DiceLoss
        """
        masks = [m["masks"] for m in batched_outputs]
        pred_labels = self._postprocess_outputs(masks)

        # we do the condition below to adapt w.r.t. the multimask output to select the "objectively" best response
        if pred_labels.dim() == 5:
            metric = min([self.metric(pred_labels[:, :, i, :, :], sampled_binary_y.to(self.device))
                          for i in range(pred_labels.shape[2])])
        else:
            metric = self.metric(pred_labels, sampled_binary_y.to(self.device))

        return metric

    #
    # Update Masks Iteratively while Training
    #
    def _update_masks(self, batched_inputs, y, sampled_binary_y, sampled_ids, num_subiter, multimask_output):
        # estimating the image inputs to make the computations faster for the decoder
        input_images = torch.stack([self.model.preprocess(x=x["image"].to(self.device)) for x in batched_inputs], dim=0)
        image_embeddings = self.model.image_embeddings_oft(input_images)

        loss, mask_loss, iou_regression_loss, mean_model_iou = 0.0, 0.0, 0.0, 0.0

        # this loop takes care of the idea of sub-iterations, i.e. the number of times we iterate over each batch
        for i in range(0, num_subiter):
            # we do multimasking only in the first sub-iteration as we then pass single prompt
            # after the first sub-iteration, we don't do multimasking because we get multiple prompts
            batched_outputs = self.model(batched_inputs,
                                         multimask_output=multimask_output if i == 0 else False,
                                         image_embeddings=image_embeddings)

            # we want to average the loss and then backprop over the net sub-iterations
            net_loss, net_mask_loss, net_iou_regression_loss, net_mean_model_iou = self._get_net_loss(batched_outputs,
                                                                                                      y, sampled_ids)
            loss += net_loss
            mask_loss += net_mask_loss
            iou_regression_loss += net_iou_regression_loss
            mean_model_iou += net_mean_model_iou

            masks, logits_masks = [], []
            # the loop below gets us the masks and logits from the batch-level outputs
            for m in batched_outputs:
                mask, l_mask = [], []
                for _m, _l, _iou in zip(m["masks"], m["low_res_masks"], m["iou_predictions"]):
                    best_iou_idx = torch.argmax(_iou)
                    best_mask, best_logits = _m[best_iou_idx][None], _l[best_iou_idx][None]
                    mask.append(self._sigmoid(best_mask))
                    l_mask.append(best_logits)

                mask, l_mask = torch.stack(mask), torch.stack(l_mask)
                masks.append(mask)
                logits_masks.append(l_mask)

            masks, logits_masks = torch.stack(masks), torch.stack(logits_masks)
            masks = (masks > 0.5).to(torch.float32)

            self._get_updated_points_per_mask_per_subiter(masks, sampled_binary_y, batched_inputs, logits_masks)

        loss = loss / num_subiter
        mask_loss = mask_loss / num_subiter
        iou_regression_loss = iou_regression_loss / num_subiter
        mean_model_iou = mean_model_iou / num_subiter

        return loss, mask_loss, iou_regression_loss, mean_model_iou

    def _get_updated_points_per_mask_per_subiter(self, masks, sampled_binary_y, batched_inputs, logits_masks):
        # here, we get the pair-per-batch of predicted and true elements (and also the "batched_inputs")
        for x1, x2, _inp, logits in zip(masks, sampled_binary_y, batched_inputs, logits_masks):
            # here, we get each object in the pairs and do the point choices per-object
            net_coords, net_labels, _, _ = self.prompt_generator(x2, x1)

            updated_point_coords = torch.cat([_inp["point_coords"], net_coords], dim=1) \
                if "point_coords" in _inp.keys() else net_coords
            updated_point_labels = torch.cat([_inp["point_labels"], net_labels], dim=1) \
                if "point_labels" in _inp.keys() else net_labels

            _inp["point_coords"] = updated_point_coords
            _inp["point_labels"] = updated_point_labels

            if self.mask_prob > 0:
                # using mask inputs for iterative prompting while training, with a probability
                use_mask_inputs = True if random.random() < self.mask_prob else False
                if use_mask_inputs:
                    _inp["mask_inputs"] = logits
                else:  # remove  previously existing mask inputs to avoid using them in next sub-iteration
                    _inp.pop("mask_inputs", None)

    #
    # Training Loop
    #

    def _update_samples_for_gt_instances(self, y, n_samples):
        num_instances_gt = torch.amax(y, dim=(1, 2, 3))
        num_instances_gt = num_instances_gt.numpy().astype(int)
        n_samples = min(num_instances_gt) if n_samples > min(num_instances_gt) else n_samples
        return n_samples

    def _train_epoch_impl(self, progress, forward_context, backprop):
        self.model.train()

        n_iter = 0
        t_per_iter = time.time()
        for x, y in self.train_loader:

            self.optimizer.zero_grad()

            with forward_context():
                n_samples = self._update_samples_for_gt_instances(y, self.n_objects_per_batch)

                n_pos, n_neg, get_boxes, multimask_output = self._get_prompt_and_multimasking_choices(self._iteration)

                batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos, n_neg, get_boxes, n_samples)

                assert len(y) == len(sampled_ids)
                sampled_binary_y = []
                for i in range(len(y)):
                    _sampled = [torch.isin(y[i], torch.tensor(idx)) for idx in sampled_ids[i]]
                    sampled_binary_y.append(_sampled)

                # the steps below are done for one reason in a gist:
                # to handle images where there aren't enough instances as expected
                # (e.g. where one image has only one instance)
                obj_lengths = [len(s) for s in sampled_binary_y]
                sampled_binary_y = [s[:min(obj_lengths)] for s in sampled_binary_y]
                sampled_binary_y = [torch.stack(s).to(torch.float32) for s in sampled_binary_y]
                sampled_binary_y = torch.stack(sampled_binary_y)

                # gist for below - while we find the mismatch, we need to update the batched inputs
                # else it would still generate masks using mismatching prompts, and it doesn't help us
                # with the subiterations again. hence we clip the number of input points as well
                f_objs = sampled_binary_y.shape[1]
                batched_inputs = [
                    {k: (v[:f_objs] if k in ("point_coords", "point_labels", "boxes") else v) for k, v in inp.items()}
                    for inp in batched_inputs
                ]

                loss, mask_loss, iou_regression_loss, model_iou = self._update_masks(batched_inputs, y,
                                                                                     sampled_binary_y, sampled_ids,
                                                                                     num_subiter=self.n_sub_iteration,
                                                                                     multimask_output=multimask_output)

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

    def _validate_impl(self, forward_context):
        self.model.eval()

        val_iteration = 0
        metric_val, loss_val, model_iou_val = 0.0, 0.0, 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                with forward_context():
                    n_samples = self._update_samples_for_gt_instances(y, self.n_objects_per_batch)

                    (n_pos, n_neg,
                     get_boxes, multimask_output) = self._get_prompt_and_multimasking_choices_for_val(val_iteration)

                    batched_inputs, sampled_ids = self.convert_inputs(x, y, n_pos, n_neg, get_boxes, n_samples)

                    batched_outputs = self.model(batched_inputs, multimask_output=multimask_output)

                    assert len(y) == len(sampled_ids)
                    sampled_binary_y = torch.stack(
                        [torch.isin(y[i], torch.tensor(sampled_ids[i])) for i in range(len(y))]
                    ).to(torch.float32)

                    loss, mask_loss, iou_regression_loss, model_iou = self._get_net_loss(batched_outputs,
                                                                                         y, sampled_ids)

                    metric = self._get_val_metric(batched_outputs, sampled_binary_y)

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
