import os
import argparse
import numpy as np

import torch

from torch_em.data.datasets import get_lucchi_loader, get_lucchi_dataset
from torch_em.segmentation import SegmentationDataset
import torch_em

from micro_sam.sam_3d_wrapper import get_3d_sam_model
from micro_sam.training.semantic_sam_trainer import SemanticSamTrainer3D
import micro_sam.training as sam_training


class LucchiSegmentationDataset(SegmentationDataset):
    def __init__(self, patch_shape, num_classes, label_transform=None, **kwargs):
        super().__init__(patch_shape=patch_shape, label_transform=label_transform, **kwargs)  # Call parent class constructor
        self.num_classes = num_classes

    def __getitem__(self, index):
        raw, label = super().__getitem__(index)
        # raw shape: (z, color channels, x, y) channels is fixed to 3
        image_shape = (self.patch_shape[0], 1) + self.patch_shape[1:]
        raw = raw.unsqueeze(2)
        raw = raw.view(image_shape)
        raw = raw.squeeze(0)
        raw = raw.repeat(1, 3, 1, 1)  
        # label shape: (classes, z, x, y)
        label_shape = (self.num_classes,) + self.patch_shape
        label = label.view(label_shape)
        return raw, label


def get_loader(path, split, patch_shape, n_classes, batch_size, label_transform, num_workers=1):
    assert split in ("train", "test")
    data_path = os.path.join(path, f"lucchi_{split}.h5")
    raw_key, label_key = "raw", "labels"
    ds = LucchiSegmentationDataset(
        raw_path=data_path, label_path=data_path, raw_key=raw_key, 
        label_key=label_key, patch_shape=patch_shape, 
        num_classes=n_classes, label_transform=label_transform)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers)
    loader.shuffle = True
    return loader


def train_on_lucchi(args):
    from micro_sam.training.util import ConvertToSemanticSamInputs
    input_path = args.input_path
    patch_shape = args.patch_shape
    batch_size = args.batch_size
    num_workers = args.num_workers
    n_classes = args.n_classes
    model_type = args.model_type
    n_iterations = args.n_iterations
    save_root = args.save_root
    
    label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_3d = get_3d_sam_model(
        device, n_classes=n_classes, image_size=patch_shape[1],
        model_type=model_type)
    #get_dataloaders(patch_shape, input_path)
    train_loader = get_loader(
        input_path, split="train", patch_shape=patch_shape,
        n_classes=n_classes, batch_size=batch_size, num_workers=num_workers,
        label_transform=label_transform)
    val_loader = get_loader(
        input_path, split="test", patch_shape=patch_shape,
        n_classes=n_classes, batch_size=batch_size, num_workers=num_workers,
        label_transform=label_transform)
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
        save_root=save_root,
        logger=None
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
    parser.add_argument("--n_iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes to predict")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="num_workers")
    parser.add_argument(
        "--save_root", "-s", default="/scratch-grete/usr/nimlufre/micro-sam3d",
        help="The filepath to where the logs and the checkpoints will be saved."
    )

    args = parser.parse_args()
    train_on_lucchi(args)


if __name__ == "__main__":
    main()
