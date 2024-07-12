import os

from common import get_default_arguments, run_training_for_livecell, run_inference_for_livecell

import torch

from torch_em.model import UNETR


def main(args):
    for_sam = args.sam
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "/scratch-grete/share/cidas/cca/models/sam/sam_vit_l_0b3195.pth" if for_sam else None

    model = UNETR(
        encoder="vit_l",
        out_channels=3,
        final_activation="Sigmoid",
        use_skip_connection=False,
        use_sam_stats=for_sam,
        encoder_checkpoint=checkpoint_path,
    )
    model.to(device)

    if args.phase == "train":
        run_training_for_livecell(
            name="livecell-unetr-sam" if for_sam else "livecell-unetr",
            path=args.input_path,
            save_root=args.save_root,
            iterations=args.iterations,
            model=model,
            device=device,
            for_sam=for_sam,
        )

    if args.phase == "predict":
        checkpoint_path = os.path.join(
            args.save_root, "checkpoints", "livecell-unetr-sam" if for_sam else "livecell-unetr", "best.pt"
        )
        result_path = "livecell-unetr-sam" if for_sam else "livecell-unetr"
        run_inference_for_livecell(
            path=args.input_path,
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
            result_path=result_path,
            for_sam=for_sam,
        )


if __name__ == "__main__":
    args = get_default_arguments()
    main(args)
