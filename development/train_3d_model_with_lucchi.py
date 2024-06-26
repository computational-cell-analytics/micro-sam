import os
import argparse

import torch

from torch_em.data.datasets import get_lucchi_loader
import torch_em

from micro_sam.sam_3d_wrapper import get_3d_sam_model
from micro_sam.training.semantic_sam_trainer import SemanticSamTrainer3D
import micro_sam.training as sam_training

def get_dataloaders(patch_shape, data_path, batch_size=1, num_workers=4):
    """This returns the livecell data loaders implemented in torch_em:
    https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/livecell.py
    It will automatically download the livecell data.

    Note: to replace this with another data loader you need to return a torch data loader
    that retuns `x, y` tensors, where `x` is the image data and `y` are the labels.
    The labels have to be in a label mask instance segmentation format.
    I.e. a tensor of the same spatial shape as `x`, with each object mask having its own ID.
    Important: the ID 0 is reseved for background, and the IDs must be consecutive
    """
    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
    
    train_loader = get_lucchi_loader(
        path=data_path, patch_shape=patch_shape, split="train", batch_size=batch_size, num_workers=num_workers,
        download=True, shuffle=True, label_transform=label_transform, label_dtype=torch.float32
    )
    val_loader = get_lucchi_loader(
        path=data_path, patch_shape=patch_shape, split="test", batch_size=batch_size, num_workers=num_workers,
        download=True, shuffle=True, label_transform=label_transform, label_dtype=torch.float32
    )

    return train_loader, val_loader


def train_on_lucchi(input_path, patch_shape, model_type, n_classes, n_iterations):
    from micro_sam.training.util import ConvertToSemanticSamInputs
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_3d = get_3d_sam_model(
        device, n_classes=n_classes, image_size=patch_shape[1],
        model_type=model_type)
    train_loader, val_loader = get_dataloaders(patch_shape, input_path)
    optimizer = torch.optim.AdamW(sam_3d.parameters(), lr=5e-5)
    
    trainer = SemanticSamTrainer3D(
        name="test-3d-sam",
        model=sam_3d,
        convert_inputs=ConvertToSemanticSamInputs(),
        num_classes=n_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        compile_model=False,
    )
    trainer.fit(n_iterations)
    

def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LiveCELL dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/scratch/projects/nim00007/sam/data/lucchi/",
        help="The filepath to the LiveCELL data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(64, 256, 256), help="Patch shape for data loading (3D tuple)")
    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of training iterations")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of classes to predict")
    args = parser.parse_args()
    args = parser.parse_args()
    train_on_lucchi(
        args.input_path, args.patch_shape, args.model_type,
        args.n_classes, args.n_iterations)


if __name__ == "__main__":
    main()
