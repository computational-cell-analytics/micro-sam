import torch

from micro_sam.v2.datasets.generalist_loader import get_dataloaders
from micro_sam.v2.training import train_automatic, train_automatic_multi_gpu


def main():
    model_type = "hvit_t"
    data_path = "/mnt/vast-nhr/projects/cidas/cca/data"
    save_root = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/automatic/v2"

    n_gpus = torch.cuda.device_count()
    name = f"unisam2_automatic_{model_type}_{'multi' if n_gpus > 1 else 'single'}_gpu"

    loader_kwargs = dict(
        batch_size=2,
        batch_size_2d=8,
        z_slices=[8],
        dataset_choice="all",
        n_workers=16,
    )

    # Set 'peft_kwargs' to finetune with a parameter efficient method instead of full finetuning: the
    # SAM2 image encoder is frozen and the method is applied on top of it, while the UniSAM2 decoder is
    # trained. Examples:
    #   from micro_sam.v2.models.peft_sam2 import LoRASurgery, ClassicalSurgery
    #   peft_kwargs = {"rank": 4, "peft_module": LoRASurgery}  # LoRA on all Hiera blocks
    #   peft_kwargs = {"peft_module": ClassicalSurgery, "attention_layers_to_update": [11]}  # late finetuning
    peft_kwargs = None

    trainer_kwargs = dict(
        name=name,
        model_type=model_type,
        n_iterations=int(2e5),
        lr=1e-5,
        save_root=save_root,
        peft_kwargs=peft_kwargs,  # None = full finetuning; set above to use LoRA / late finetuning
    )

    if n_gpus > 1:  # i.e. for multiple GPUs
        train_automatic_multi_gpu(input_path=data_path, **loader_kwargs, **trainer_kwargs)
    else:  # i.e. for single GPU
        train_loader, val_loader = get_dataloaders(input_path=data_path, **loader_kwargs)
        train_automatic(train_loader=train_loader, val_loader=val_loader, **trainer_kwargs)


if __name__ == "__main__":
    main()
