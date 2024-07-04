import os
import argparse
import numpy as np
from math import ceil, floor
import torch

from torch_em.data.datasets import get_lucchi_loader, get_lucchi_dataset
from torch_em.segmentation import SegmentationDataset
import torch_em
from torch_em.util.debug import check_loader
from torch_em.transform.raw import normalize

from micro_sam.models.sam_3d_wrapper import get_sam_3d_model
from micro_sam.training.semantic_sam_trainer import SemanticSamTrainer
import micro_sam.training as sam_training


class RawTrafoFor3dInputs:
    def _normalize_inputs(self, raw):
        raw = normalize(raw)
        raw = raw * 255
        return raw

    def _set_channels_for_inputs(self, raw):
        raw = np.stack([raw] * 3, axis=0)
        return raw

    def __call__(self, raw):
        raw = self._normalize_inputs(raw)
        raw = self._set_channels_for_inputs(raw)
        return raw


# for sega
class RawResizeTrafoFor3dInputs(RawTrafoFor3dInputs):
    def __init__(self, desired_shape, padding="constant"):
        super().__init__()
        self.desired_shape = desired_shape
        self.padding = padding

    def __call__(self, raw):
        raw = self._normalize_inputs(raw)

        # let's pad the inputs
        tmp_ddim = (
           self.desired_shape[0] - raw.shape[0],
           self.desired_shape[1] - raw.shape[1],
           self.desired_shape[2] - raw.shape[2]
        )
        ddim = (tmp_ddim[0] / 2, tmp_ddim[1] / 2, tmp_ddim[2] / 2)
        raw = np.pad(
            raw,
            pad_width=(
                (ceil(ddim[0]), floor(ddim[0])), (ceil(ddim[1]), floor(ddim[1])), (ceil(ddim[2]), floor(ddim[2]))
            ),
            mode=self.padding
        )

        raw = self._set_channels_for_inputs(raw)

        return raw


class LucchiSegmentationDataset(SegmentationDataset):
    def __init__(self, patch_shape, label_transform=None, **kwargs):
        super().__init__(patch_shape=patch_shape, label_transform=label_transform, **kwargs)  # Call parent class constructor

    def __getitem__(self, index):
        raw, label = super().__getitem__(index)
        # raw shape: (z, color channels, y, x) channels is fixed to 3
        image_shape = (self.patch_shape[0], 1) + self.patch_shape[1:]
        raw = raw.unsqueeze(2)
        raw = raw.view(image_shape)
        raw = raw.squeeze(0)
        raw = raw.repeat(1, 3, 1, 1)
        # print("raw shape", raw.shape)
        # wanted label shape: (1, z, y, x)
        label = (label != 0).to(torch.float)
        # print("label shape", label.shape)
        return raw, label


def transform_labels(y):
    return (y > 0).astype("float32")


def get_loaders(input_path, patch_shape):
    train_loader = get_lucchi_loader(
        input_path, split="train", patch_shape=patch_shape, batch_size=1, download=True,
        raw_transform=RawTrafoFor3dInputs(), label_transform=transform_labels,
        n_samples=100
    )
    val_loader = get_lucchi_loader(
        input_path, split="test", patch_shape=patch_shape, batch_size=1,
        raw_transform=RawTrafoFor3dInputs(), label_transform=transform_labels
    )
    return train_loader, val_loader
# def get_loader(path, split, patch_shape, n_classes, batch_size, label_transform, num_workers=1):
#     assert split in ("train", "test")
#     data_path = os.path.join(path, f"lucchi_{split}.h5")
#     raw_key, label_key = "raw", "labels"
#     ds = LucchiSegmentationDataset(
#         raw_path=data_path, label_path=data_path, raw_key=raw_key, 
#         label_key=label_key, patch_shape=patch_shape, label_transform=label_transform)
#     loader = torch.utils.data.DataLoader(
#         ds, batch_size=batch_size, shuffle=True, 
#         num_workers=num_workers)
#     loader.shuffle = True
#     return loader


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
    
    # label_transform = torch_em.transform.label.BoundaryTransform(add_binary_target=True)
    # label_transform = None
    raw_data = np.random.rand(64, 256, 256)  # Shape (z, y, x)
    raw_data2, label = next(iter(get_lucchi_loader(input_path, split="train", patch_shape=patch_shape, batch_size=1, download=True)))

    # Create an instance of RawTrafoFor3dInputs
    transformer = RawTrafoFor3dInputs()

    # Apply transformations
    processed_data = transformer(raw_data)
    processed_data2 = transformer(raw_data2)
    print("input (64,256,256)", processed_data.shape)
    print("input", raw_data2.shape, processed_data2.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_3d = get_sam_3d_model(
        device, n_classes=n_classes, image_size=patch_shape[1],
        model_type=model_type, lora_rank=4)
    train_loader, val_loader = get_loaders(input_path=input_path, patch_shape=patch_shape)
    optimizer = torch.optim.AdamW(sam_3d.parameters(), lr=5e-5)
    
    trainer = SemanticSamTrainer(
        name="3d-sam-lucchi-train",
        model=sam_3d,
        convert_inputs=ConvertToSemanticSamInputs(),
        num_classes=n_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        compile_model=False,
        save_root=save_root,
        #logger=None
    )
    # check_loader(train_loader, n_samples=10)
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
    parser.add_argument("--patch_shape", type=int, nargs=3, default=(32, 512, 512), help="Patch shape for data loading (3D tuple)")
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
