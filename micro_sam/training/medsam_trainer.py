# import os
# import time

# import torch
# from torchvision.utils import make_grid

# import torch_em
# from torch_em.trainer.logger_base import TorchEmLogger

from . import SamTrainer


class MedSAMTrainer(SamTrainer):
    """Trainer class for replicating the trainer of MedSAM (https://arxiv.org/abs/2304.12306)
    """
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def _get_prompt_and_multimasking_choices(self, current_iteration):
        n_pos, n_neg = 0, 0
        get_boxes = True
        multimask_output = False
        return n_pos, n_neg, get_boxes, multimask_output

    def _get_prompt_and_multimasking_choices_for_val(self, current_iteration):
        return self._get_prompt_and_multimasking_choices(current_iteration=current_iteration)
