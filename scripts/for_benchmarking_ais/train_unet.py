from common import get_loaders, get_default_arguments

import torch

from torch_em.model import UNet2d
from torch_em.loss import DiceBasedDistanceLoss
from torch_em import default_segmentation_trainer
from torch_em.model.unetr import SingleDeconv2DBlock


def run_training_for_livecell(path, save_root, iterations):
    # all the necessary stuff for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_shape = (512, 512)
    train_loader, val_loader = get_loaders(path=path, patch_shape=patch_shape)
    loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    model = UNet2d(
        in_channels=1,
        out_channels=3,
        initial_features=64,
        final_activation="Sigmoid",
        sampler_impl=SingleDeconv2DBlock,
    )
    model.to(device)

    trainer = default_segmentation_trainer(
        name="livecell-unet",
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        mixed_precision=True,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 5}
    )

    trainer.fit(int(iterations))


def main(args):
    if args.phase == "train":
        run_training_for_livecell(path=args.input_path, save_root=args.save_root, iterations=args.iterations)

    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = get_default_arguments()
    main(args)
