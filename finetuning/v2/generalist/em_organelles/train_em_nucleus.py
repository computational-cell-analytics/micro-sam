import os
import argparse

import torch


# The training domain of the nucleus EM generalist (see EM_NUCLEUS_DATASETS in the generalist loader).
DATASET_CHOICE = "em_nucleus"

CHOSEN_PARAMETERS = {
    "hvit_t": (14, 14, 4),
    "hvit_s": (12, 14, 4),
    "hvit_b": (10, 10, 4),
    "hvit_l": (8, 8, 4),
}


def build_common(model_type, n_epochs, n_iterations, batch_size, use_compile=False, boundary_dice_weight=0.5):
    """Build the keyword arguments that the single-GPU and the multi-GPU entry points share.

    The per-model config (batch_size_2d, z_slices, max_num_objects) is fixed and not a CLI option.
    The env vars BATCH_SIZE_2D, Z_SLICE and MAX_NUM_OBJECTS override it for debug jobs on smaller GPUs.
    """
    batch_size_2d, z_slice, max_num_objects = CHOSEN_PARAMETERS[model_type]
    batch_size_2d = int(os.environ.get("BATCH_SIZE_2D", batch_size_2d))
    z_slice = int(os.environ.get("Z_SLICE", z_slice))
    max_num_objects = int(os.environ.get("MAX_NUM_OBJECTS", max_num_objects))
    data_path = "/mnt/vast-nhr/projects/cidas/cca/data"
    save_root = os.environ.get("SAVE_ROOT", os.getcwd())

    is_multi_gpu = "RANK" in os.environ
    name = f"{DATASET_CHOICE}_sam2_{model_type}_{'multi' if is_multi_gpu else 'single'}_gpu"
    if os.environ.get("RUN_TAG"):
        name = f"{name}_{os.environ['RUN_TAG']}"

    # The training config is the one of the joint cell generalist (v6).
    return dict(
        name=name,
        model_type=model_type,
        input_path=data_path,
        batch_size=batch_size,
        batch_size_2d=batch_size_2d,
        z_slices=[z_slice],
        dataset_choice=DATASET_CHOICE,
        n_workers=int(os.environ.get("N_WORKERS", 8)),  # loader workers per GPU
        n_epochs=n_epochs,
        n_iterations=n_iterations,
        lr=1e-5,
        save_root=save_root,
        checkpoint_path=None,  # downloads default SAM2 weights if None
        max_num_objects=max_num_objects,
        prob_to_use_pt_input=1.0,  # always point/box prompts, never the GT mask
        prob_to_use_box_input=0.5,  # conditional prob of a box instead of a click
        num_frames_to_correct=2,  # max frames per volume receiving correction clicks
        rand_frames_to_correct=True,
        prob_to_sample_from_gt=0.1,
        add_all_frames_to_correct_as_cond=True,
        num_correction_pt_per_frame=7,  # correction clicks per frame per round
        num_init_cond_frames=2,  # initial conditioning frames (2D forced to 1 internally)
        clip_grad_norm=None,
        largest_first=True,
        bidirectional=True,  # bidirectional propagation for 3D z-stacks
        use_focal_loss=True,
        focal_weight=1.0,
        use_object_score_loss=True,  # supervise object presence (needed for 3D propagation)
        average_over_frames=False,  # sum over frames so 3D keeps its per-slice weight
        peft_kwargs=None,  # full finetuning
        initial_features=32,  # decoder bottleneck matches the hvit_t embed_dim
        distance_type="geodesic",  # regression target of the automatic branch
        with_boundaries=True,  # the object-boundary channel of the automatic branch
        boundary_dice_weight=boundary_dice_weight,
        save_every_kth_epoch=5,  # keeps fallback checkpoints next to best.pt and latest.pt
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 10},
        label_trafo_threads=int(os.environ.get("LABEL_TRAFO_THREADS", 1)),  # threads per worker in the label transform
        compile=["encoder", "decoder", "loss"] if use_compile else None,
    )


def run_training(common):
    """Run the multi-GPU entry point under torchrun, otherwise the single-GPU one."""
    if "RANK" in os.environ:
        from micro_sam.v2.training import train_joint_sam2_multi_gpu
        train_joint_sam2_multi_gpu(**common)
    else:
        from micro_sam.v2.training import train_joint_sam2
        train_joint_sam2(**common)


def report_peak_memory(common):
    rank = int(os.environ.get("RANK", "0"))
    if torch.cuda.is_available() and rank == 0:
        peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        print(
            f"[peak-memory] model_type={common['model_type']} batch_size={common['batch_size']} "
            f"batch_size_2d={common['batch_size_2d']} z_slices={common['z_slices']} "
            f"max_num_objects={common['max_num_objects']} "
            f"allocated={peak_alloc:.2f}GiB reserved={peak_reserved:.2f}GiB", flush=True
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--n_iterations", type=int, default=None)
    parser.add_argument("--model_type", default="hvit_t", choices=list(CHOSEN_PARAMETERS))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Compile the encoder, the decoder and the loss.")
    parser.add_argument(
        "--boundary_dice_weight", type=float, default=0.5,
        help="Relative Dice weight for the boundary channel; zero selects BCE only.",
    )
    args = parser.parse_args()

    common = build_common(
        args.model_type, args.n_epochs, args.n_iterations, args.batch_size,
        use_compile=args.compile, boundary_dice_weight=args.boundary_dice_weight,
    )
    run_training(common)
    report_peak_memory(common)


if __name__ == "__main__":
    main()
