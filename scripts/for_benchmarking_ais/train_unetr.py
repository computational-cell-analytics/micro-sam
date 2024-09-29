import os

from common import get_default_arguments, run_training, run_inference

import torch

from torch_em.model import UNETR


SAM_PRETRAINED = "/scratch-grete/share/cidas/cca/models/sam/sam_vit_l_0b3195.pth"


def main(args):
    dataset = args.dataset
    for_sam = args.sam
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = SAM_PRETRAINED if for_sam and args.phase == "train" else None

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
        run_training(
            name=f"{dataset}-unetr-sam" if for_sam else f"{dataset}-unetr",
            path=args.input_path,
            save_root=args.save_root,
            iterations=args.iterations,
            model=model,
            device=device,
            for_sam=for_sam,
            dataset=dataset,
        )

    if args.phase == "predict":
        ckpt_path = os.path.join(
            "./" if args.save_root is None else args.save_root,
            "checkpoints", f"{dataset}-unetr-sam" if for_sam else f"{dataset}-unetr", "best.pt"
        )
        result_path = "results/" + f"{dataset}-unetr-sam" if for_sam else f"{dataset}-unetr"
        run_inference(
            path=args.input_path,
            checkpoint_path=ckpt_path,
            model=model,
            device=device,
            result_path=result_path,
            for_sam=for_sam,
            dataset=dataset,
        )


if __name__ == "__main__":
    args = get_default_arguments()
    main(args)
