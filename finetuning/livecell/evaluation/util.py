import os

from micro_sam.evaluation import get_predictor
from micro_sam.evaluation.livecell import _get_livecell_paths

DATA_ROOT = "/scratch/projects/nim00007/data/LiveCELL"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments"
MODELS = {
    "vit_b": "/scratch-grete/projects/nim00007/sam/vanilla/sam_vit_b_01ec64.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/vanilla/sam_vit_h_4b8939.pth",
    "vit_b_specialist": "/scratch-grete/projects/nim00007/sam/LM/LiveCELL/vit_b/best.pt",
    "vit_h_specialist": "/scratch-grete/projects/nim00007/sam/LM/LiveCELL/vit_h/best.pt",
    "vit_b_generalist": "/scratch-grete/projects/nim00007/sam/LM/generalist/vit_b/best.pt",
    "vit_h_generalist": "/scratch-grete/projects/nim00007/sam/LM/generalist/vit_h/best.pt",
}


def get_paths():
    return _get_livecell_paths(DATA_ROOT)


def get_checkpoint(name):
    assert name in MODELS, name
    ckpt = MODELS[name]
    assert os.path.exists(ckpt), ckpt
    model_type = name[:5]
    assert model_type in ("vit_b", "vit_h"), model_type
    return ckpt, model_type


def get_model(name):
    ckpt, model_type = get_checkpoint(name)
    predictor = get_predictor(ckpt, model_type)
    return predictor


def get_experiment_folder(name):
    return os.path.join(EXPERIMENT_ROOT, name)


def download_livecell():
    from torch_em.data.datasets import get_livecell_loader
    get_livecell_loader(DATA_ROOT, "train", (512, 512), 1, download=True)
    get_livecell_loader(DATA_ROOT, "val", (512, 512), 1, download=True)
    get_livecell_loader(DATA_ROOT, "test", (512, 512), 1, download=True)


if __name__ == "__main__":
    download_livecell()
