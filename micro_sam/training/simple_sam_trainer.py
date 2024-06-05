from . import SamTrainer


class SimpleSamTrainer(SamTrainer):
    """Trainer class for creating a simple SAM trainer for limited prompt-based segmentation.
    """
    def __init__(
        self,
        use_points: bool = True,
        use_box: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.use_points = use_points
        self.use_box = use_box

        if (self.use_points + self.use_box) != 1:
            raise ValueError("You must choose either 'use_points' / 'use_box'.")

    def _get_prompt_and_multimasking_choices(self, current_iteration):
        if self.use_points:
            n_pos, n_neg = 1, 0  # samples only a single positive point per object
            multimask_output = True
        else:
            n_pos, n_neg = 0, 0
            multimask_output = False

        get_boxes = self.use_box  # samples only a single box per object

        return n_pos, n_neg, get_boxes, multimask_output

    def _get_prompt_and_multimasking_choices_for_val(self, current_iteration):
        return self._get_prompt_and_multimasking_choices(current_iteration)
