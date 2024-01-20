import os
import argparse
from glob import glob

from micro_sam.evaluation import get_predictor
from micro_sam.evaluation.livecell import _get_livecell_paths

# FIXME make sure this uses the corrected ground-truth!!!
DATA_ROOT = "/scratch/projects/nim00007/data/LiveCELL"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/livecell"
PROMPT_FOLDER = "/scratch/projects/nim00007/sam/experiments/prompts/livecell"

VANILLA_MODELS = {
    "vit_b": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_b_01ec64.pth",
    "vit_l": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_l_0b3195.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/models/new_models/vanilla/sam_vit_h_4b8939.pth"
}


def get_paths(split="test"):
    return _get_livecell_paths(DATA_ROOT, split=split)


def get_model(model_type=None, ckpt=None):
    if ckpt is None:
        ckpt = VANILLA_MODELS[model_type]
    predictor = get_predictor(ckpt, model_type)
    return predictor


def get_experiment_folder(name):
    return os.path.join(EXPERIMENT_ROOT, name)


def get_pred_and_gt_paths(prediction_folder):
    pred_paths = sorted(glob(os.path.join(prediction_folder, "*.tif")))
    names = [os.path.split(path)[1] for path in pred_paths]
    gt_root = os.path.join(DATA_ROOT, "annotations_corrected/livecell_test_images")
    gt_paths = [
        os.path.join(gt_root, name.split("_")[0], name) for name in names
    ]
    assert all(os.path.exists(pp) for pp in gt_paths)
    return pred_paths, gt_paths


def download_livecell():
    from torch_em.data.datasets import get_livecell_loader
    get_livecell_loader(DATA_ROOT, "train", (512, 512), 1, download=True)
    get_livecell_loader(DATA_ROOT, "val", (512, 512), 1, download=True)
    get_livecell_loader(DATA_ROOT, "test", (512, 512), 1, download=True)


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", type=str, required=True, help="Provide the model type to initialize the predictor"
    )
    parser.add_argument("-c", "--checkpoint", type=none_or_str, required=True)
    parser.add_argument("-e", "--experiment_folder", type=str, required=True)
    parser.add_argument("--box", action="store_true", help="If passed, starts with first prompt as box")
    args = parser.parse_args()
    return args


def none_or_str(value):
    if value == 'None':
        return None
    return value


if __name__ == "__main__":
    download_livecell()
