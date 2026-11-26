"""Train a hybrid 2D/3D segmentation decoder on top of the frozen image encoder of a uSAM2 model.

The data and the training settings are the ones of the automatic segmentation branch in the joint training
of the cells model (finetuning/v2/generalist/train_joint.py). Only the decoder is updated.

    python train_decoder.py --model_type hvit_t_cells --batch_size_2d <B> --z_slice <Z>
    python train_decoder.py ... --resume    # continue from the latest checkpoint
"""

import argparse
import os

import common


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_type", default="hvit_t_cells", help="The SAM2 model that provides the encoder.")
    parser.add_argument("--decoder", default="hiera_u", choices=["hiera_u", "c3f2"])
    parser.add_argument("--name", default=None, help="The name of the run. By default '<decoder>_<model_type>'.")
    parser.add_argument("--batch_size_2d", type=int, default=None, help="By default the value of the joint training.")
    parser.add_argument("--z_slice", type=int, default=None, help="By default the value of the joint training.")
    parser.add_argument("--batch_size_3d", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=None, help="By default the epochs of the joint training.")
    parser.add_argument("--lr", type=float, default=common.LEARNING_RATE)
    parser.add_argument("--dataset_choice", default="all")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--label_trafo_threads", type=int, default=6)
    parser.add_argument("--save_root", default=common.SAVE_ROOT)
    parser.add_argument("--save_every_kth_epoch", type=int, default=5)
    parser.add_argument("--resume", action="store_true", help="Continue from the latest checkpoint, if it exists.")
    args = parser.parse_args()

    common.set_start_method()
    from micro_sam.v2.training import train_decoder

    backbone = args.model_type[:6]
    joint_batch_size_2d, joint_z_slice = common.JOINT_BATCH_SETTINGS[backbone]
    batch_size_2d = joint_batch_size_2d if args.batch_size_2d is None else args.batch_size_2d
    z_slice = joint_z_slice if args.z_slice is None else args.z_slice
    n_epochs = common.N_EPOCHS[backbone] if args.n_epochs is None else args.n_epochs
    name = common.run_name(args.model_type, args.decoder) if args.name is None else args.name

    common.log_host_memory()
    train_loader, val_loader = common.build_loaders(
        batch_size_2d=batch_size_2d, z_slice=z_slice, batch_size_3d=args.batch_size_3d,
        dataset_choice=args.dataset_choice, n_workers=args.n_workers, label_trafo_threads=args.label_trafo_threads,
    )

    run_info = {
        **vars(args), "name": name, "batch_size_2d": batch_size_2d, "z_slice": z_slice, "n_epochs": n_epochs,
        "boundary_dice_weight": common.BOUNDARY_DICE_WEIGHT, "code": common.code_versions(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "train_loader": common.describe_loader(train_loader), "val_loader": common.describe_loader(val_loader),
    }
    job_tag = os.environ.get("SLURM_JOB_ID", "local")
    common.write_json(os.path.join(args.save_root, "checkpoints", name, f"run_info_{job_tag}.json"), run_info)
    print(f"Training '{name}': {len(train_loader)} iterations per epoch, {n_epochs} epochs.")

    train_decoder(
        name=name, train_loader=train_loader, val_loader=val_loader, model_type=args.model_type,
        decoder=args.decoder, n_epochs=n_epochs, lr=args.lr, save_root=args.save_root,
        save_every_kth_epoch=args.save_every_kth_epoch, resume=args.resume,
        boundary_dice_weight=common.BOUNDARY_DICE_WEIGHT,
    )


if __name__ == "__main__":
    main()
