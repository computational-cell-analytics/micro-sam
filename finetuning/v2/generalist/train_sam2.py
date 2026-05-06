import torch


def main():
    model_type = "hvit_t"
    data_path = "/mnt/vast-nhr/projects/cidas/cca/data"
    save_root = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/interactive/v2"

    n_gpus = torch.cuda.device_count()
    name = f"sam2_interactive_{model_type}_{'multi' if n_gpus > 1 else 'single'}_gpu"

    loader_kwargs = dict(
        batch_size=1,
        batch_size_2d=8,
        z_slices=[8],
        dataset_choice="both",
        n_workers=16,
    )
    trainer_kwargs = dict(
        name=name,
        model_type=model_type,
        n_iterations=int(2e5),
        lr=1e-5,
        save_root=save_root,
        checkpoint_path=None,  # downloads default SAM2 weights if None
        max_num_objects=8,  # max objects sampled per image/volume per step
        num_frames_to_correct=5,  # max frames per volume receiving correction clicks
        rand_frames_to_correct=True,  # randomly sample 1..num_frames_to_correct each step
        prob_to_sample_from_gt=0.1,  # prob of clicking GT mask instead of error region
        add_all_frames_to_correct_as_cond=True,  # treat corrected frames as memory cond frames
        clip_grad_norm=0.1,  # max gradient norm; None to disable
    )

    if n_gpus > 1:
        from micro_sam.v2.training import train_sam2_multi_gpu
        train_sam2_multi_gpu(input_path=data_path, n_gpus=n_gpus, **loader_kwargs, **trainer_kwargs)
    else:
        from micro_sam.v2.datasets.generalist_loader import get_interactive_dataloaders
        from micro_sam.v2.training import train_sam2
        train_loader, val_loader = get_interactive_dataloaders(input_path=data_path, **loader_kwargs)
        train_sam2(train_loader=train_loader, val_loader=val_loader, **trainer_kwargs)


if __name__ == "__main__":
    main()
