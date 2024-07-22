import os

import torch

from torch_em.loss import DiceBasedDistanceLoss

import micro_sam.training as sam_training
from micro_sam.training.trainable_sam import TrainableSAM
from micro_sam.training.util import ConvertToSemanticSamInputs

from segment_anything import sam_model_registry

from common import get_default_arguments, get_loaders, run_inference


def run_semantic_training(path, save_root, iterations, model, device, for_sam, num_classes, dataset):
    # all the stuff we need for training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=5, verbose=True)

    patch_shape = (512, 512)
    train_loader, val_loader = get_loaders(path=path, patch_shape=patch_shape, dataset=dataset, for_sam=True)

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = ConvertToSemanticSamInputs()

    checkpoint_name = f"{dataset}_semanticsam" + ("-sam" if for_sam else "-scratch")

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
    for_sam = args.sam
    dataset = args.dataset
    model_type = "vit_l"
    num_classes = 3
    checkpoint_path = None

    if for_sam:
        # This model is always initializes with pretrained SAM weights.
        model = sam_training.get_trainable_sam_model(
            model_type=model_type,
            device=device,
            checkpoint_path=checkpoint_path,
            flexible_load_checkpoint=True,
            num_multimask_outputs=num_classes,
        )
    else:
        # This model is initialized without the pretrained SAM weights.
        sam = sam_model_registry[model_type]()
        model = TrainableSAM(sam)

    model.to(device)

    if args.phase == "train":
        run_semantic_training(
            path=args.input_path,
            save_root=args.save_root,
            iterations=args.iterations,
            model=model,
            device=device,
            for_sam=for_sam,
            num_classes=num_classes,
            dataset=dataset,
        )

    if args.phase == "predict":
        checkpoint_path = os.path.join(
            "./" if args.save_root is None else args.save_root,
            "checkpoints", f"{dataset}_semanticsam-sam" if for_sam else f"{dataset}_semanticsam-scratch", "best.pt"
        )
        result_path = f"results/{dataset}-semanticsam"
        run_inference(
            path=args.input_path,
            checkpoint_path=checkpoint_path,
            model=model,
            device=device,
            result_path=result_path,
            for_sam=True,
            with_semantic_sam=True,
            dataset=dataset,
        )


if __name__ == "__main__":
    args = get_default_arguments()
    main(args)
