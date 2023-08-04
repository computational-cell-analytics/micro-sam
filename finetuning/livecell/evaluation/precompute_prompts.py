import os

from micro_sam.evaluation import precompute_all_prompts
from util import get_paths, EXPERIMENT_ROOT


def main():
    prompt_save_dir = os.path.join(EXPERIMENT_ROOT, "prompts")
    _, gt_paths = get_paths()

    # TODO get all relevant prompt settings
    prompt_settings = [
        {"use_points": True, "use_boxes": False, "n_positives": 2, "n_negatives": 4},
    ]

    precompute_all_prompts(gt_paths, prompt_save_dir, prompt_settings)


if __name__ == "__main__":
    main()
