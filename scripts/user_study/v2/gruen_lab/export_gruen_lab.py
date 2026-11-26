"""Export the gruen lab finetuned models and normalized crops for checking them in the annotator.

The models are single files with the SAM2 weights and the UniSAM2 decoder, so the annotator loads both from
'-m hvit_l -c <file>'. The crops are (Z, Y, X, 3) uint8 tifs in the training channel order (membrane, nuclei,
cell body), normalized per channel over the volume with the 2/98 percentiles used for validation.
"""

import os
import argparse
from glob import glob

import h5py
import torch
import tifffile
import numpy as np


CHECKPOINT_DIR = "checkpoints/gruen_lab_hvit_l_cells"
LOWER_PERCENTILE, UPPER_PERCENTILE = 2.0, 98.0


def export_models(out_dir):
    for name in ("best", "latest"):
        state = torch.load(os.path.join(CHECKPOINT_DIR, f"{name}.pt"), map_location="cpu", weights_only=False)
        model_state = state["model_state"]
        out_path = os.path.join(out_dir, f"gruen_lab_hvit_l_{name}_it{state['iteration']}.pt")
        export = {
            "model": model_state, "model_state": model_state, "decoder_state": state["decoder_state"],
            "model_type": "hvit_l",
        }
        torch.save(export, out_path)
        print(f"Model: {out_path}")


def normalize_channel(channel):
    lower, upper = np.percentile(channel, (LOWER_PERCENTILE, UPPER_PERCENTILE))
    channel = (channel.astype("float32") - lower) / max(upper - lower, 1e-7)
    return (np.clip(channel, 0, 1) * 255).round().astype("uint8")


def export_crops(data_dir, out_dir):
    for path in sorted(glob(os.path.join(data_dir, "crop*.h5"))):
        name = os.path.splitext(os.path.basename(path))[0]
        with h5py.File(path, "r") as f:
            raw, labels = f["raw"][:], f["labels"][:]
        image = np.stack([normalize_channel(channel) for channel in raw], axis=-1)
        image_path = os.path.join(out_dir, f"{name}_image.tif")
        tifffile.imwrite(image_path, image, compression="zlib", photometric="rgb")
        tifffile.imwrite(os.path.join(out_dir, f"{name}_labels.tif"), labels, compression="zlib")
        print(f"Crop: {image_path} {image.shape} {image.dtype}")


def main():
    parser = argparse.ArgumentParser(description="Export the gruen lab models and normalized crops.")
    parser.add_argument("--data_dir", default="data", help="The folder with the crop h5 files.")
    parser.add_argument("--out_dir", default="annotator_export", help="The output folder.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_models(args.out_dir)
    export_crops(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
