import os


def main():
    model_type = "hvit_t"
    data_path = "/mnt/vast-nhr/projects/cidas/cca/data"

    # NOTE:
    # -> v2 - best working model with OG loss (focal weight 20, cosine LR, vision_lr, grad clip 0.1).
    # -> v3 - simplified CustomSAM2Loss, 1x weighting, average over frames; trails v2 on 3D.
    # -> v4 - v3 but sum over frames is restored (average_over_frames=False) to test the 3D deficit.
    save_root = "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/interactive/v4"

    is_multi_gpu = "LOCAL_RANK" in os.environ
    name = f"sam2_interactive_{model_type}_{'multi' if is_multi_gpu else 'single'}_gpu"

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
        early_stopping=None,
        lr=1e-5,  # single LR for all parameters (no separate vision_lr)
        save_root=save_root,
        checkpoint_path=None,  # downloads default SAM2 weights if None
        max_num_objects=8,  # max objects sampled per image/volume per step
        largest_first=True,  # sample the biggest objects first, then fill remaining slots randomly
        prob_to_use_pt_input=1.0,  # always point/box prompts, never the GT mask
        num_frames_to_correct=2,  # max frames per volume receiving correction clicks
        rand_frames_to_correct=True,  # randomly sample 1..num_frames_to_correct each step
        prob_to_sample_from_gt=0.1,  # prob of clicking GT mask instead of error region
        add_all_frames_to_correct_as_cond=True,  # treat corrected frames as memory cond frames
        num_correction_pt_per_frame=7,  # correction clicks per frame per round
        num_init_cond_frames=2,  # initial conditioning frames (2D is forced to 1 internally)
        clip_grad_norm=None,  # no gradient clipping
        bidirectional=True,  # bidirectional propagation for 3D z-stacks
        use_focal_loss=True,  # add SAM2's focal mask loss on top of dice
        focal_weight=1.0,  # keep focal on equal footing with dice (SAM2 uses 20)
        use_object_score_loss=True,  # supervise object presence (needed for 3D propagation)
        average_over_frames=False,  # sum over frames so 3D keeps its per-slice weight (like v2)
    )

    if is_multi_gpu:
        from micro_sam.v2.training import train_sam2_multi_gpu
        train_sam2_multi_gpu(input_path=data_path, **loader_kwargs, **trainer_kwargs)
    else:
        from micro_sam.v2.datasets.generalist_loader import get_interactive_dataloaders
        from micro_sam.v2.training import train_sam2
        train_loader, val_loader = get_interactive_dataloaders(input_path=data_path, **loader_kwargs)
        train_sam2(train_loader=train_loader, val_loader=val_loader, **trainer_kwargs)


if __name__ == "__main__":
    main()
