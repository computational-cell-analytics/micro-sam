import random

from . import SamTrainer


class SimpleSamTrainer(SamTrainer):
    """Trainer class for creating a simple SAM trainer for limited prompt-based segmentation.

    This class is inherited from `SamTrainer`.
    Check out https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/training/sam_trainer.py
    for details on its implementation.

    Args:
        use_points: Whether to use point prompts for interactive segmentation.
        use_box: Whether to use box prompts for interactive segmentation.
        kwargs: The keyword arguments of the `SamTrainer` (and `DefaultTrainer`) class.
    """
    def __init__(
        self,
        use_points: bool = True,
        use_box: bool = True,
        **kwargs
    ):
        super().__init__(
            n_sub_iteration=1,
            mask_prob=0,
            **kwargs
        )
        self.use_points = use_points
        self.use_box = use_box

        if self.use_points and self.use_box:
            self.random_prompt_choice = True
        else:
            self.random_prompt_choice = False

        assert (self.use_points + self.use_box) != 0, "Please choose at least one of the prompt-based method."

    def _choose_one_positive_point(self):
        """Samples only a single positive point per object
        """
        n_pos, n_neg = 1, 0
        multimask_output = True
        return n_pos, n_neg, None, multimask_output

    def _choose_box(self):
        """Samples only a single box per object
        """
        n_pos, n_neg = 0, 0
        multimask_output = False
        get_boxes = True
        return n_pos, n_neg, get_boxes, multimask_output

    def _get_prompt_and_multimasking_choices(self, current_iteration):
        if self.random_prompt_choice:  # both "use_points" and "use_box" are True
            available_choices = [self._choose_one_positive_point(), self._choose_box()]
            return random.choice(available_choices)
        else:  # either of "use_points" or "use_box" are True
            if self.use_points:
                return self._choose_one_positive_point()
            else:
                return self._choose_box()

    def _get_prompt_and_multimasking_choices_for_val(self, current_iteration):
        return self._get_prompt_and_multimasking_choices(current_iteration)


class MedSAMTrainer(SimpleSamTrainer):
    """Trainer class for replicating the trainer of MedSAM (https://arxiv.org/abs/2304.12306).

    This class is inherited from `SimpleSamTrainer`.
    Check out
    https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/training/simple_sam_trainer.py
    for details on its implementation.
    """
    def __init__(self, **kwargs):
        super().__init__(
            use_points=False,
            use_box=True,
            **kwargs
        )
