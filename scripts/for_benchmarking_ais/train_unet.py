import os

from common import get_default_arguments, run_inference, run_training

import torch

from torch_em.model import UNet2d
from torch_em.model.unetr import SingleDeconv2DBlock


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet2d(
        in_channels=1,
        out_channels=3,
        initial_features=64,
        final_activation="Sigmoid",
        sampler_impl=SingleDeconv2DBlock,
    )
    model.to(device)

    if args.phase == "train":
        run_training(
            name=f"{args.dataset}-unet",
            path=args.input_path,
            save_root=args.save_root,
            iterations=args.iterations,
            model=model,
            device=device,
        )

    if args.phase == "predict":
        checkpoint_path = os.path.join(
            "./" if args.save_root is None else args.save_root, "checkpoints", f"{args.dataset}-unet", "best.pt"
        )
        result_path = f"results/{args.dataset}_unet/"
        run_inference(
            path=args.input_path,
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
            result_path=result_path,
        )


if __name__ == "__main__":
    args = get_default_arguments()
    main(args)
