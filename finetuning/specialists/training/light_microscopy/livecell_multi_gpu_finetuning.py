import argparse

import torch

from torch_em.data import MinInstanceSampler
from torch_em.data.datasets import get_livecell_dataset
from torch_em.multi_gpu_training import train_multi_gpu

import micro_sam.training as sam_training

from segment_anything.utils.transforms import ResizeLongestSide


def finetune_livecell(args):
    """Example code for finetuning SAM on LIVECell"""

    # training settings:
    model_type = args.model_type
    checkpoint_path = None  # override this to start training from a custom checkpoint
    patch_shape = (520, 704)  # the patch shape for training
    n_objects_per_batch = args.n_objects  # the number of objects per batch that will be sampled (default: 25)
    freeze_parts = args.freeze  # override this to freeze different parts of the model
    checkpoint_name = f"{args.model_type}/livecell_sam_multi_gpu"

    # this class creates all the training data for a batch (inputs, prompts and labels)
    convert_inputs = sam_training.ConvertToSamInputs(
        transform=ResizeLongestSide(target_length=1024), box_distortion_factor=0.025
    )

    # let's get the transforms, dataset class and respective kwargs.
    raw_transform = sam_training.identity  # the current workflow avoids rescaling the inputs to [-1, 1]

    train_dataset_class = get_livecell_dataset
    val_dataset_class = get_livecell_dataset
    dataset_kwargs = {
        "path": args.input_path,
        "patch_shape": patch_shape,
        "split": "train",
        "raw_transform": raw_transform,
        "sampler": MinInstanceSampler(),
        "label_dtype": torch.float32,
    }
    train_dataset_kwargs = {"split": "train", **dataset_kwargs}
    val_dataset_kwargs = {"split": "val", **dataset_kwargs}

    loader_kwargs = {
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 16,
        "pin_memory": True
    }

    # Run training on multiple GPUs.
    train_multi_gpu(
        model_callable=sam_training.get_trainable_sam_model,
        model_kwargs={"model_type": model_type, "checkpoint_path": checkpoint_path, "freeze": freeze_parts},
        train_dataset_callable=train_dataset_class,
        train_dataset_kwargs=train_dataset_kwargs,
        val_dataset_callable=val_dataset_class,
        val_dataset_kwargs=val_dataset_kwargs,
        loader_kwargs=loader_kwargs,
        iterations=int(args.iterations),
        find_unused_parameters=True,
        optimizer_callable=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-5},
        lr_scheduler_callable=torch.optim.lr_scheduler.StepLR,
        lr_scheduler_kwargs={"step_size": 1, "gamma": 0.9, "verbose": True},
        # trainer params
        trainer_callable=sam_training.SamTrainer,
        name=checkpoint_name,
        save_root=args.save_root,
        logger=sam_training.SamLogger,
        log_image_interval=100,
        mixed_precision=True,
        convert_inputs=convert_inputs,
        n_objects_per_batch=n_objects_per_batch,
        n_sub_iteration=8,
        compile_model=False,
        mask_prob=0.5,  # (optional) overwrite to provide the probability of using mask inputs while training
    )


def main():
    parser = argparse.ArgumentParser(description="Finetune Segment Anything for the LIVECell dataset.")
    parser.add_argument(
        "--input_path", "-i", default="/mnt/vast-nhr/projects/cidas/cca/data/livecell/",
        help="The filepath to the LIVECell data. If the data does not exist yet it will be downloaded."
    )
    parser.add_argument(
        "--model_type", "-m", default="vit_b",
        help="The model type to use for fine-tuning. Either vit_t, vit_b, vit_l or vit_h."
    )
    parser.add_argument(
        "--save_root", "-s",
        help="Where to save the checkpoint and logs. By default they will be saved where this script is run."
    )
    parser.add_argument(
        "--iterations", type=int, default=int(1e5),
        help="For how many iterations should the model be trained? By default 100k."
    )
    parser.add_argument(
        "--freeze", type=str, nargs="+", default=None,
        help="Which parts of the model to freeze for finetuning."
    )
    parser.add_argument(
        "--n_objects", type=int, default=25, help="The number of instances (objects) per batch used for finetuning."
    )
    args = parser.parse_args()
    finetune_livecell(args)


if __name__ == "__main__":
    main()
