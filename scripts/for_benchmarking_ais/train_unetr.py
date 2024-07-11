from common import get_loaders, get_default_arguments

import torch

from torch_em.model import UNETR
from torch_em.loss import DiceBasedDistanceLoss
from torch_em import default_segmentation_trainer


def run_training_for_livecell(path, save_root, iterations, for_sam):
    # all the necessary stuff for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patch_shape = (512, 512)
    train_loader, val_loader = get_loaders(path=path, patch_shape=patch_shape, for_sam=for_sam)
    loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)
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

    trainer = default_segmentation_trainer(
        name="livecell-unetr-sam" if for_sam else "livecell-unetr",
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
        run_training_for_livecell(
            path=args.input_path,
            save_root=args.save_root,
            iterations=args.iterations,
            for_sam=args.sam,
        )

    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = get_default_arguments()
    main(args)
