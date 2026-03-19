import os

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_mndino_loader
from torch_em.transform.label import PerObjectDistanceTransform

from micro_sam.util import get_device
import micro_sam.training as sam_training
from micro_sam.training.util import get_raw_transform


ROOT = "/mnt/vast-nhr/projects/cidas/cca/data"


def get_dataloader(split):
    # Other important stuff
    label_transform = PerObjectDistanceTransform(
        distances=True,
        boundary_distances=True,
        directed_distances=False,
        foreground=True,
        instances=True,
    )

    # Prepare the dataloaders for micronuclei segmentation.
    loader = get_mndino_loader(
        path=os.path.join(ROOT, "mndino_data"),
        batch_size=1,
        patch_shape=(512, 512),
        num_workers=16,
        shuffle=True,
        split=split,
        label_choice="micronuclei",
        raw_transform=get_raw_transform("normalize_percentile"),
        download=True,
        label_transform=label_transform,
        sampler=MinInstanceSampler(),
    )

    return loader


def run_mndino_training(model_type):
    # All training hyperparameters and important stuff too.
    n_objects_per_batch = 25
    device = get_device()

    # Get the dataloaders.
    train_loader = get_dataloader("train")
    val_loader = get_dataloader("val")

    # Run training.
    sam_training.train_sam(
        name=f"microsam-mn-{model_type}",
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=100,
        early_stopping=None,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=True,
        device=device,
    )


def main():
    # One could choose to finetune either `vit_b` / `vit_b_lm`.
    model_type = "vit_b_lm"

    run_mndino_training(model_type)


if __name__ == "__main__":
    main()
