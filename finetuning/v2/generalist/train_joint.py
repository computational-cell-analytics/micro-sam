import os
import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_iterations", type=int, default=None)
    args = parser.parse_args()

    model_type = "hvit_t"
    data_path = "/mnt/vast-nhr/projects/cidas/cca/data"
    save_root = os.environ.get("SAVE_ROOT", "/mnt/vast-nhr/projects/cidas/cca/models/micro_sam2/joint/v0")

    n_gpus = torch.cuda.device_count()
    name = f"joint_sam2_{model_type}_{'multi' if n_gpus > 1 else 'single'}_gpu"

    # Set 'peft_kwargs' to jointly finetune with a parameter efficient method instead of full
    # finetuning (the SAM2 image encoder is frozen and the method is applied on top of it). Examples:
    #   from micro_sam.v2.models.peft_sam2 import LoRASurgery, ClassicalSurgery
    #   peft_kwargs = {"rank": 4, "peft_module": LoRASurgery}  # LoRA on all Hiera blocks
    #   peft_kwargs = {"peft_module": ClassicalSurgery, "attention_layers_to_update": [11]}  # late finetuning
    peft_kwargs = None

    # Interactive config mirrors train_sam2.py (v4): CustomSAM2Loss with summed-frame
    # weighting, point/box prompts only, bidirectional 3D propagation, no grad clipping.
    common = dict(
        name=name,
        model_type=model_type,
        input_path=data_path,
        batch_size=1,
        batch_size_2d=8,
        z_slices=[8],
        dataset_choice="both",
        n_workers=8,
        n_epochs=args.n_epochs,
        n_iterations=args.n_iterations,
        lr=1e-5,  # single LR for all parameters
        save_root=save_root,
        checkpoint_path=None,  # downloads default SAM2 weights if None
        max_num_objects=5,  # lower than interactive-only (8): joint also holds the UNETR decoder
        prob_to_use_pt_input=1.0,  # always point/box prompts, never the GT mask
        prob_to_use_box_input=0.5,  # conditional prob of a box instead of a click
        num_frames_to_correct=2,  # max frames per volume receiving correction clicks
        rand_frames_to_correct=True,
        prob_to_sample_from_gt=0.1,
        add_all_frames_to_correct_as_cond=True,
        num_correction_pt_per_frame=7,  # correction clicks per frame per round
        num_init_cond_frames=2,  # initial conditioning frames (2D forced to 1 internally)
        clip_grad_norm=None,  # no gradient clipping
        largest_first=True,
        bidirectional=True,  # bidirectional propagation for 3D z-stacks
        use_focal_loss=True,  # add SAM2's focal mask loss on top of dice
        focal_weight=1.0,  # keep focal on equal footing with dice (SAM2 uses 20)
        use_object_score_loss=True,  # supervise object presence (needed for 3D propagation)
        average_over_frames=False,  # sum over frames so 3D keeps its per-slice weight
        peft_kwargs=peft_kwargs,  # None = full finetuning; set above to use LoRA / late finetuning
    )

    if n_gpus > 1:
        from micro_sam.v2.training import train_joint_sam2_multi_gpu
        train_joint_sam2_multi_gpu(**common)
    else:
        from micro_sam.v2.training import train_joint_sam2
        train_joint_sam2(**common)


if __name__ == "__main__":
    main()
