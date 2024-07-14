import os
import argparse
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.transform.raw import standardize
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.util import segmentation, prediction
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets.light_microscopy import get_livecell_loader

import micro_sam.training as sam_training

from elf.evaluation import mean_segmentation_accuracy


#
# DATALOADERS
#


def get_loaders(path, patch_shape, for_sam=False):
    kwargs = {
        "label_transform": PerObjectDistanceTransform(
            distances=True,
            boundary_distances=True,
            directed_distances=False,
            foreground=True,
            min_size=25,
        ),
        "label_dtype": torch.float32,
        "num_workers": 16,
        "patch_shape": patch_shape
    }

    if for_sam:
        kwargs["raw_transform"] = sam_training.identity

    train_loader = get_livecell_loader(path=path, split="train", batch_size=2, **kwargs)
    val_loader = get_livecell_loader(path=path, split="val", batch_size=1, **kwargs)

    return train_loader, val_loader


#
# TRAINING SCRIPTS
#


def run_training_for_livecell(name, path, save_root, iterations, model, device, for_sam=False):
    # all the necessary stuff for training
    patch_shape = (512, 512)
    train_loader, val_loader = get_loaders(path=path, patch_shape=patch_shape, for_sam=for_sam)
    loss = DiceBasedDistanceLoss(mask_distances_in_bg=True)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=1e-4,
        loss=loss,
        metric=loss,
        log_image_interval=50,
        save_root=save_root,
        compile_model=False,
        mixed_precision=True,
        scheduler_kwargs={"mode": "min", "factor": 0.9, "patience": 5}
    )

    trainer.fit(int(iterations))


#
# INFERENCE SCRIPTS
#

def run_inference_for_livecell(path, checkpoint_path, model, device, result_path, for_sam=False):
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state"])
    model.to(device)
    model.eval()

    # the splits are provided with the livecell dataset to reproduce the results:
    # run the inference on the entire dataset as it is.
    test_image_dir = os.path.join(path, "images", "livecell_test_images")
    all_test_labels = glob(os.path.join(path, "annotations", "livecell_test_images", "*", "*"))[:10]

    msa_list, sa50_list, sa75_list = [], [], []
    for label_path in tqdm(all_test_labels):
        labels = imageio.imread(label_path)
        image_id = os.path.split(label_path)[-1]

        image = imageio.imread(os.path.join(test_image_dir, image_id))

        if for_sam:
            image = image.astype("float32")  # functional interpolate does not like uint
            per_tile_pp = None
        else:
            per_tile_pp = standardize

        predictions = prediction.predict_with_halo(
            input_=image,
            model=model,
            gpu_ids=[device],
            block_shape=(384, 384),
            halo=(64, 64),
            preprocess=per_tile_pp,
            disable_tqdm=True,
            output=np.zeros(image.shape)
        )
        breakpoint()
        predictions = predictions.squeeze()

        fg, cdist, bdist = predictions
        instances = segmentation.watershed_from_center_and_boundary_distances(
            cdist, bdist, fg, min_size=50,
            center_distance_threshold=0.5,
            boundary_distance_threshold=0.6,
            distance_smoothing=1.0
        )

        msa, sa_acc = mean_segmentation_accuracy(instances, labels, return_accuracies=True)
        msa_list.append(msa)
        sa50_list.append(sa_acc[0])
        sa75_list.append(sa_acc[5])

    res = {
        "LIVECell": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list),
        "SA75": np.mean(sa75_list)
    }

    os.makedirs(result_path, exist_ok=True)
    res_path = os.path.join(result_path, "results.csv")
    df = pd.DataFrame.from_dict([res])
    df.to_csv(res_path)
    print(df)
    print(f"The result is saved at {res_path}")


#
# MISCELLANOUS
#


def get_default_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", type=str, default="/scratch/projects/nim00007/sam/data/livecell")
    parser.add_argument("-s", "--save_root", type=str, default=None)
    parser.add_argument("-p", "--phase", type=str, default=None, choices=["train", "predict"])
    parser.add_argument("--iterations", type=str, default=1e5)
    parser.add_argument("--sam", action="store_true")
    args = parser.parse_args()
    return args
