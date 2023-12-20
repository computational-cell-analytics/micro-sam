import os
import h5py
import argparse
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from pathlib import Path
import imageio.v3 as imageio
from collections import OrderedDict

import torch

from torch_em.model import UNETR
from torch_em.util import segmentation
from torch_em.util.prediction import predict_with_padding

from elf.evaluation import mean_segmentation_accuracy

from micro_sam.util import get_sam_model


def get_unetr_model(model_type, checkpoint, device):
    # let's get the sam finetuned model
    predictor = get_sam_model(
        model_type=model_type
    )

    # load the model with the respective unetr model state
    model = UNETR(
        encoder=predictor.model.image_encoder,
        out_channels=3,
        use_sam_stats=True,
        final_activation="Sigmoid",
        use_skip_connection=False
    )

    sam_state = torch.load(checkpoint, map_location="cpu")["model_state"]
    # let's get the vit parameters from sam
    encoder_state = []
    prune_prefix = "sam.image_"
    for k, v in sam_state.items():
        if k.startswith(prune_prefix):
            encoder_state.append((k[len(prune_prefix):], v))
    encoder_state = OrderedDict(encoder_state)

    decoder_state = torch.load(checkpoint, map_location="cpu")["decoder_state"]

    unetr_state = OrderedDict(list(encoder_state.items()) + list(decoder_state.items()))
    model.load_state_dict(unetr_state)
    model.to(device)
    model.eval()

    return model


def predict_for_unetr(inputs, save_dir, model, device):
    save_dir = os.path.join(save_dir, "results")
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(glob(os.path.join(inputs, "images", "livecell_test_images", "*")),
                             desc="Run unetr inference"):
            fname = Path(img_path).stem
            save_path = os.path.join(save_dir, f"{fname}.h5")
            if os.path.exists(save_path):
                continue

            input_ = imageio.imread(img_path)

            outputs = predict_with_padding(model, input_, device=device, min_divisible=(16, 16))
            fg, cdist, bdist = outputs.squeeze()
            dm_seg = segmentation.watershed_from_center_and_boundary_distances(
                cdist, bdist, fg, min_size=50,
                center_distance_threshold=0.5,
                boundary_distance_threshold=0.6,
                distance_smoothing=1.0
            )

            with h5py.File(save_path, "a") as f:
                ds = f.require_dataset("segmentation", shape=dm_seg.shape, compression="gzip", dtype=dm_seg.dtype)
                ds[:] = dm_seg


def evaluation_for_unetr(inputs, save_dir, csv_path):
    if os.path.exists(csv_path):
        return

    msa_list, sa50_list = [], []
    for gt_path in tqdm(glob(os.path.join(inputs, "annotations", "livecell_test_images", "*", "*")),
                        desc="Run unetr evaluation"):
        gt = imageio.imread(gt_path)
        fname = Path(gt_path).stem

        output_file = os.path.join(save_dir, "results", f"{fname}.h5")
        with h5py.File(output_file, "r") as f:
            instances = f["segmentation"][:]

        msa, sa_acc = mean_segmentation_accuracy(instances, gt, return_accuracies=True)
        msa_list.append(msa)
        sa50_list.append(sa_acc[0])

    res_dict = {
        "LiveCELL": "Metrics",
        "mSA": np.mean(msa_list),
        "SA50": np.mean(sa50_list)
    }
    df = pd.DataFrame.from_dict([res_dict])
    df.to_csv(csv_path)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # let's get the unetr model (initialized with the joint training setup)
    model = get_unetr_model(model_type=args.model_type, checkpoint=args.checkpoint, device=device)

    # let's get the predictions
    predict_for_unetr(inputs=args.inputs, save_dir=args.save_dir, model=model, device=device)

    # let's evaluate the predictions
    evaluation_for_unetr(inputs=args.inputs, save_dir=args.save_dir, csv_path=args.csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", default="/scratch/usr/nimanwai/data/livecell/")
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-m", "--model_type", type=str, default="vit_b")
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default="livecell_joint_training.csv")
    args = parser.parse_args()
    main(args)
