import os

DATA_ROOT = "/scratch/projects/nim00007/data/LiveCELL"
EXPERIMENT_ROOT = "/scratch/projects/nim00007/sam/experiments/livecell"
MODELS = {
    "vit_b": "/scratch-grete/projects/nim00007/sam/vanilla/sam_vit_b_01ec64.pth",
    "vit_h": "/scratch-grete/projects/nim00007/sam/vanilla/sam_vit_h_4b8939.pth",
    "vit_b_specialist": "/scratch-grete/projects/nim00007/sam/LM/LiveCELL/vit_b/best.pt",
    "vit_h_specialist": "/scratch-grete/projects/nim00007/sam/LM/LiveCELL/vit_h/best.pt",
    "vit_b_generalist": "/scratch-grete/projects/nim00007/sam/LM/generalist/vit_b/best.pt",
    "vit_h_generalist": "/scratch-grete/projects/nim00007/sam/LM/generalist/vit_h/best.pt",
}


def get_checkpoint(name):
    assert name in MODELS, name
    ckpt = MODELS[name]
    assert os.path.exists(ckpt), ckpt
    model_type = name[:5]
    assert model_type in ("vit_b", "vit_h"), model_type
    return ckpt, model_type


def get_experiment_folder(name):
    return os.path.join(EXPERIMENT_ROOT, name)
