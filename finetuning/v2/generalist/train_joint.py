import torch


def main():
    model_type = "hvit_t"
    data_path = "/mnt/vast-nhr/projects/cidas/cca/data"
    # save_root = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v2"
    save_root = None

    n_gpus = torch.cuda.device_count()
    name = f"joint_sam2_{model_type}_{'multi' if n_gpus > 1 else 'single'}_gpu"

    common = dict(
        name=name,
        model_type=model_type,
        input_path=data_path,
        batch_size=1,
        batch_size_2d=8,
        z_slices=[8],
        dataset_choice="both",
        n_workers=16,
        n_iterations=int(2e5),
        lr=1e-5,
        save_root=save_root,
        checkpoint_path=None,  # downloads default SAM2 weights if None
        max_num_objects=7,
        num_frames_to_correct=4,
        rand_frames_to_correct=True,
        prob_to_sample_from_gt=0.1,
        add_all_frames_to_correct_as_cond=True,
        clip_grad_norm=0.1,
    )

    if n_gpus > 1:
        from micro_sam.v2.training import train_joint_sam2_multi_gpu
        train_joint_sam2_multi_gpu(n_gpus=n_gpus, **common)
    else:
        from micro_sam.v2.training import train_joint_sam2
        train_joint_sam2(**common)


if __name__ == "__main__":
    main()
