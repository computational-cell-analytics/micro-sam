import os

import torch

from torch_em.loss import DiceBasedDistanceLoss

import micro_sam.training as sam_training
from micro_sam.training.util import ConvertToSemanticSamInputs

from common import get_default_arguments, get_loaders, run_inference


def run_semantic_training(path, save_root, iterations, model, device, model_type, num_classes, dataset):
    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5, verbose=True)

    patch_shape = (512, 512)
    train_loader, val_loader = get_loaders(path=path, patch_shape=patch_shape, dataset=dataset, for_sam=True)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    checkpoint_name = f"{model_type}/{dataset}_semanticsam"

    # the trainer which performs the semantic segmentation training and validation (implemented using "torch_em")
    trainer = sam_training.SemanticMapsSamTrainer(
        name=checkpoint_name,
        save_root=save_root,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        lr_scheduler=scheduler,
        log_image_interval=50,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        num_classes=num_classes,
        compile_model=False,
        loss=DiceBasedDistanceLoss(mask_distances_in_bg=True),
        metric=DiceBasedDistanceLoss(mask_distances_in_bg=True),
    )
    trainer.fit(int(iterations))


def main(args):
    # training settings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_type = "vit_l"
    num_classes = 3
    checkpoint_path = None

    # get the trainable segment anything model
    model = sam_training.get_trainable_sam_model(
        model_type=model_type,
        device=device,
        checkpoint_path=checkpoint_path,
        flexible_load_checkpoint=True,
        num_multimask_outputs=num_classes,
    )
    model.to(device)

    if args.phase == "train":
        run_semantic_training(
            path=args.input_path,
            save_root=args.save_root,
            iterations=args.iterations,
            model=model,
            device=device,
            model_type=model_type,
            num_classes=num_classes,
        )

    if args.phase == "predict":
        checkpoint_path = os.path.join(
            "./" if args.save_root is None else args.save_root,
            "checkpoints", f"{model_type}/{args.dataset}_semanticsam", "best.pt"
        )
        result_path = f"results/{args.dataset}-semanticsam"
        run_inference(
            path=args.input_path,
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
            result_path=result_path,
            for_sam=True,
            with_semantic_sam=True,
        )


if __name__ == "__main__":
    args = get_default_arguments()
    main(args)
