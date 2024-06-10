import random

from . import SamTrainer


class SimpleSamTrainer(SamTrainer):
    """Trainer class for creating a simple SAM trainer for limited prompt-based segmentation.
    """
    def __init__(
        self,
        use_points: bool = False,
        use_box: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_points = use_points
        self.use_box = use_box

        if not self.use_points and not self.use_box:  # if user doesn't specify, we randomly choose box / point prompts
            self.random_prompt_choice = True
        else:
            self.random_prompt_choice = False

        assert (self.use_points + self.use_box) < 2, "Please choose either of the prompt-based segmentation."

    def _choose_one_positive_point(self):
        "samples only a single positive point per object"
        n_pos, n_neg = 1, 0
        multimask_output = True
        return n_pos, n_neg, None, multimask_output

    def _choose_box(self):
        "samples only a single box per object"
        n_pos, n_neg = 0, 0
        multimask_output = False
        get_boxes = True
        return n_pos, n_neg, get_boxes, multimask_output

    def _get_prompt_and_multimasking_choices(self, current_iteration):

        if self.random_prompt_choice:
            available_choices = [self._choose_one_positive_point(), self._choose_box()]
            return random.choice(available_choices)
        else:
            # this condition supports using either "point" / "box" / "points + box"
            if self.use_points:
                return self._choose_one_positive_point()
            else:
                return self._choose_box()

    def _get_prompt_and_multimasking_choices_for_val(self, current_iteration):
        return self._get_prompt_and_multimasking_choices(current_iteration)
