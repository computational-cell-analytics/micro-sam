from pathlib import Path
from typing import Optional, Tuple

import torch_em
from magicgui import magic_factory
from torch.utils.data import random_split

import micro_sam.util as util
from micro_sam.training import default_sam_dataset, train_sam_for_setting


# TODO rethink some of the names
# TODO set and get checkpoint path
# TODO add optional val paths
@magic_factory(call_button="Start Training")
def sam_training(
    # TODO make the setting a choice
    setting: str,
    raw_path: Path,
    label_path: Path,
    raw_key: Optional[str] = None,
    label_key: Optional[str] = None,
    train_instance_segmentation: bool = True,
    name: Optional[str] = None,
    patch_shape: Tuple[int, int] = (512, 512),
    initial_model_name: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
) -> None:

    # TODO set this depending on the settings.
    batch_size = 1
    num_workers = 1

    # TODO check if the data is 3dim and add a singleton to the patch shape

    # TODO these should become optional params
    raw_path_val, label_path_val = None, None
    if raw_path_val is None:
        dataset = default_sam_dataset(
            raw_path, raw_key, label_path, label_key,
            patch_shape=patch_shape, batch_size=batch_size,
            with_segmentation_decoder=train_instance_segmentation,
        )
        # TODO better heuristic for the split?
        train_dataset, val_dataset = random_split(dataset, lengths=[len(dataset) - 1, 1])
    else:
        if label_path_val is None:
            raise ValueError
        # TODO
        train_dataset, val_dataset = "", ""

    train_loader = torch_em.segmentation.get_data_loader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = torch_em.segmentation.get_data_loader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    # TODO consolidate initial model name, the checkpoint path and the model type according to the settings.
    if checkpoint_path is None:
        model_registry = util.models()
        checkpoint_path = model_registry.fetch()

    # TODO set napari pbar in torch_em
    train_sam_for_setting(
        name=name, setting=setting,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        with_segmentation_decoder=train_instance_segmentation,
    )
