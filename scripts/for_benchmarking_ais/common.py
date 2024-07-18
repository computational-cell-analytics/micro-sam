import os
import argparse
from glob import glob
from tqdm import tqdm

import h5py
import numpy as np
import pandas as pd
import imageio.v3 as imageio

import torch

import torch_em
from torch_em.transform.raw import normalize
from torch_em.transform.raw import standardize
from torch_em.loss import DiceBasedDistanceLoss
from torch_em.util import segmentation, prediction
from torch_em.transform.label import PerObjectDistanceTransform
from torch_em.data.datasets.light_microscopy import get_livecell_loader, get_covid_if_loader

import micro_sam.training as sam_training
from micro_sam.training.util import ConvertToSemanticSamInputs

from elf.evaluation import mean_segmentation_accuracy


#
# DATALOADERS
#


def covid_if_raw_trafo(raw):
    raw = normalize(raw)
    raw = raw * 255
    return raw


def get_loaders(path, patch_shape, dataset, for_sam=False):
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
        "patch_shape": patch_shape,
        "shuffle": True,
    }

    if for_sam:
        kwargs["raw_transform"] = sam_training.identity if dataset == "livecell" else covid_if_raw_trafo

    if dataset == "livecell":
        train_loader = get_livecell_loader(path=os.path.join(path, "livecell"), split="train", batch_size=2, **kwargs)
        val_loader = get_livecell_loader(path=os.path.join(path, "livecell"), split="val", batch_size=1, **kwargs)

    elif dataset.startswith("covid_if"):
        data_path = os.path.join(path, "covid_if")

        # Let's get the number of images to train on
        n_images = int(dataset.split("-")[-1])
        assert n_images in [1, 2, 5, 10], f"Please choose number of images from 1, 2, 5, or 10; instead of {n_images}."

        train_volumes = (None, n_images)
        val_volumes = (10, 13)

        # Let's get the number of samples extracted, to set the "n_samples" value
        # This is done to avoid the time taken to save checkpoints over fewer training images.
        _loader = get_covid_if_loader(
            path=data_path, patch_shape=patch_shape, batch_size=1, sample_range=train_volumes
        )

        print(
            f"Found {len(_loader)} samples for training.",
            "Hence, we will use {0} samples for training.".format(50 if len(_loader) < 50 else len(_loader))
        )

        # Finally, let's get the dataloaders
        train_loader = get_covid_if_loader(
            path=data_path,
            batch_size=1,
            sample_range=train_volumes,
            n_samples=50 if len(_loader) < 50 else None,
            **kwargs
        )
        val_loader = get_covid_if_loader(
            path=data_path,
            batch_size=1,
            sample_range=val_volumes,
            **kwargs
        )

    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name.")

    return train_loader, val_loader


#
# TRAINING SCRIPTS
#


def run_training(name, path, save_root, iterations, model, device, dataset, for_sam=False):
    # all the necessary stuff for training
    patch_shape = (512, 512)
    train_loader, val_loader = get_loaders(path=path, patch_shape=patch_shape, dataset=dataset, for_sam=for_sam)
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

def run_inference(
    path, checkpoint_path, model, device, result_path, dataset, for_sam=False, with_semantic_sam=False,
):
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state"])
    model.to(device)
    model.eval()

    if dataset == "livecell":
        # the splits are provided with the livecell dataset to reproduce the results:
        # run the inference on the entire dataset as it is.
        test_image_dir = os.path.join(path, "livecell", "images", "livecell_test_images")
        all_test_labels = glob(os.path.join(path, "livecell", "annotations", "livecell_test_images", "*", "*"))

    elif dataset.startswith("covid_if"):
        # we create our own splits for this dataset.
        # - the first 10 images are dedicated for training.
        # - the next 3 images are dedicated for validation.
        # - the remaining images are used for testing
        all_test_labels = glob(os.path.join(path, "covid_if", "*.h5"))[13:]

    else:
        raise ValueError(f"'{dataset}' is not a valid dataset name.")

    def prediction_fn(net, inp):
        convert_inputs = ConvertToSemanticSamInputs()
        batched_inputs = convert_inputs(inp, torch.zeros_like(inp))
        image_embeddings, batched_inputs = net.image_embeddings_oft(batched_inputs)
        batched_outputs = net(batched_inputs, image_embeddings, multimask_output=True)
        masks = torch.stack([output["masks"] for output in batched_outputs]).squeeze()
        masks = masks[None]
        return masks

    msa_list, sa50_list, sa75_list = [], [], []
    for label_path in tqdm(all_test_labels):
        image_id = os.path.split(label_path)[-1]

        if dataset == "livecell":
            image = imageio.imread(os.path.join(test_image_dir, image_id))
            labels = imageio.imread(label_path)
        else:
            with h5py.File(label_path) as f:
                image = f["raw/serum_IgG/s0"][:]
                labels = f["labels/cells/s0"][:]

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
            output=np.zeros((3, *image.shape)) if with_semantic_sam else None,
            prediction_function=prediction_fn if with_semantic_sam else None,
        )
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
        "LIVECell" if dataset == "livecell" else "Covid IF": "Metrics",
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
    parser.add_argument("-d", "--dataset", type=str, required=True)
    parser.add_argument("-i", "--input_path", type=str, default="/scratch/projects/nim00007/sam/data")
    parser.add_argument("-s", "--save_root", type=str, default=None)
    parser.add_argument("-p", "--phase", type=str, default=None, choices=["train", "predict"])
    parser.add_argument("--iterations", type=str, default=1e5)
    parser.add_argument("--sam", action="store_true")
    args = parser.parse_args()
    return args
