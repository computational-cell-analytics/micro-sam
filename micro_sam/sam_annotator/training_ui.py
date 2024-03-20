import warnings
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch_em
from magicgui import magic_factory
from torch.utils.data import random_split

import micro_sam.util as util
from micro_sam.training import default_sam_dataset, train_sam_for_setting, SETTINGS


def _find_best_setting():
    if torch.cuda.is_available():
        # can we check the GPU type and use it to match the setting?
        return "rtx5000"
    else:
        return "CPU"


# TODO rethink some of the names
# TODO add optional val paths
@magic_factory(
    call_button="Start Training",
    setting={"choices": list(SETTINGS.keys())},
    initial_model_name={"choices": list(util.models().urls.keys())}
)
def sam_training(
    raw_path: Path,
    label_path: Path,
    raw_key: Optional[str] = None,
    label_key: Optional[str] = None,
    setting: str = _find_best_setting(),
    train_instance_segmentation: bool = True,
    name: Optional[str] = None,
    patch_shape: Tuple[int, int] = (512, 512),
    initial_model_name: Optional[str] = None,
    checkpoint_path: Optional[Path] = None,
) -> None:
    batch_size = 1
    num_workers = 1 if setting == "CPU" else 4

    # TODO these should become optional params
    raw_path_val, label_path_val = None, None
    if raw_path_val is None:
        dataset = default_sam_dataset(
            str(raw_path), raw_key, str(label_path), label_key,
            patch_shape=patch_shape, with_segmentation_decoder=train_instance_segmentation,
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

    # Consolidate initial model name, the checkpoint path and the model type according to the settings.
    if initial_model_name is None or initial_model_name == "None" or initial_model_name == "":
        model_type = SETTINGS[setting]["model_type"]
    else:
        model_type = initial_model_name[:5]
        if model_type != SETTINGS[setting]["model_type"]:
            warnings.warn(
                f"You have changed the model type for your chosen setting {setting} "
                f"from {SETTINGS[setting]['model_type']} to {model_type}. "
                "The training may be very slow or not work at all."
            )
    assert model_type is not None

    if checkpoint_path is None:
        model_registry = util.models()
        checkpoint_path = model_registry.fetch(model_type)

    # TODO set napari pbar in torch_em
    train_sam_for_setting(
        name=name, setting=setting,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_path=checkpoint_path,
        with_segmentation_decoder=train_instance_segmentation,
        model_type=model_type,
    )
