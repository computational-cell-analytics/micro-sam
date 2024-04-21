import torch_em
from micro_sam.training import default_sam_dataset, train_sam_for_setting
from torch.utils.data import random_split

raw_path = "/home/pape/Work/my_projects/micro-sam/scripts/user_study/organoids/data/user_study_v2/user_study_data"
raw_key = "*.tif"

label_path = "/home/pape/Work/my_projects/micro-sam/scripts/user_study/organoids/data/user_study_v2/result-sam"
label_key = "*.tif"

patch_shape = (1, 512, 512)

device = "cpu"
with_segmentation_decoder = True

dataset = default_sam_dataset(
    str(raw_path), raw_key, str(label_path), label_key,
    patch_shape=patch_shape, with_segmentation_decoder=with_segmentation_decoder,
)
train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - 1, 1])

num_workers = 0
batch_size = 1
train_loader = torch_em.segmentation.get_data_loader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
)
val_loader = torch_em.segmentation.get_data_loader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
)

name = "test"
checkpoint_path = None
train_sam_for_setting(
    name=name, setting="CPU",
    train_loader=train_loader, val_loader=val_loader,
    checkpoint_path=checkpoint_path,
    with_segmentation_decoder=with_segmentation_decoder,
    model_type="vit_b",
    device=device
)
