import os
from micro_sam.util import export_custom_sam_model
from evaluate_generalist import CHECKPOINTS, EXPERIMENT_ROOT

OUT_ROOT = os.path.join(EXPERIMENT_ROOT, "exported")
os.makedirs(OUT_ROOT, exist_ok=True)


def export_generalist(model):
    checkpoint_path = CHECKPOINTS[model]
    model_type = model[:5]
    save_path = os.path.join(OUT_ROOT, f"{model}.pth")
    export_custom_sam_model(checkpoint_path, model_type, save_path)


def main():
    export_generalist("vit_b_em")
    export_generalist("vit_h_em")
    export_generalist("vit_b_lm")
    export_generalist("vit_h_lm")


if __name__ == "__main__":
    main()
