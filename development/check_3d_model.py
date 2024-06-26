import numpy as np
import torch
import micro_sam.util as util

from micro_sam.sam_3d_wrapper import get_3d_sam_model
from micro_sam.training.semantic_sam_trainer import SemanticSamTrainer3D


def predict_3d_model():
    d_size = 8
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_3d = get_3d_sam_model(device, d_size)

    input_ = 255 * np.random.rand(1, d_size, 3, 1024, 1024).astype("float32")
    with torch.no_grad():
        input_ = torch.from_numpy(input_).to(device)
        out = sam_3d(input_, multimask_output=False, image_size=1024)
        print(out["masks"].shape)


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, patch_shape, n_classes):
        self.patch_shape = patch_shape
        self.n_classes = n_classes

    def __len__(self):
        return 5

    def __getitem__(self, index):
        image_shape = (self.patch_shape[0], 3) + self.patch_shape[1:]
        x = np.random.rand(*image_shape).astype("float32")
        label_shape = (self.n_classes,) + self.patch_shape
        y = (np.random.rand(*label_shape) > 0.5).astype("float32")
        return x, y


def get_loader(patch_shape, n_classes, batch_size):
    ds = DummyDataset(patch_shape, n_classes)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    loader.shuffle = True
    return loader


# TODO: we are missing the resizing in the model, so currently this only supports 1024x1024
def train_3d_model():
    from micro_sam.training.util import ConvertToSemanticSamInputs

    d_size = 4
    n_classes = 5
    batch_size = 2
    image_size = 512

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_3d = get_3d_sam_model(device, n_classes=n_classes, image_size=image_size)

    train_loader = get_loader((d_size, image_size, image_size), n_classes, batch_size)
    val_loader = get_loader((d_size, image_size, image_size), n_classes, batch_size)

    optimizer = torch.optim.AdamW(sam_3d.parameters(), lr=5e-5)

    trainer = SemanticSamTrainer3D(
        name="test-sam",
        model=sam_3d,
        convert_inputs=ConvertToSemanticSamInputs(),
        num_classes=n_classes,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        compile_model=False,
    )
    trainer.fit(10)


def main():
    # predict_3d_model()
    train_3d_model()


if __name__ == "__main__":
    main()
