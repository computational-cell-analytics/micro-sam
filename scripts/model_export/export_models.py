"""Helper scripts to export models for upload to bioimageio/zenodo.
"""

import os
import xxhash
import warnings
import argparse
from glob import glob

import h5py
from skimage.measure import label
import imageio.v3 as imageio

import bioimageio.spec.model.v0_5 as spec

from micro_sam.bioimageio import export_sam_model

from models import get_id_and_emoji, MODEL_TO_NAME


BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

INPUT_FOLDER = "/media/anwai/ANWAI/models/micro_sam"
OUTPUT_FOLDER = "./exported_models"
BIOIMAGEIO_VERSION = 1.2  # version marked for v4 LM and EM-Organelles Generalist Models


def create_doc(model_type, modality, version):
    if modality not in ("lm", "em_organelles"):
        raise ValueError(f"Invalid modality template {modality}")

    template_file = os.path.join(
        os.path.split(__file__)[0], "../../doc/bioimageio", f"{modality}_v{version}.md"
    )
    assert os.path.exists(template_file), template_file
    with open(template_file, "r") as f:
        template = f.read()

    doc = template % (model_type, version)
    return doc


def get_data(modality):
    if modality == "lm":
        image_path = os.path.join(
            "/media/anwai/ANWAI/data/livecell/"
            "images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif"
        )
        label_path = os.path.join(
            "/media/anwai/ANWAI/data/livecell/"
            "annotations/livecell_train_val_images/A172/A172_Phase_A7_1_00d00h00m_1.tif"
        )
        image = imageio.imread(image_path)
        label_image = imageio.imread(label_path)
    else:
        path = "/media/anwai/ANWAI/data/kasthuri/kasthuri_train.h5"
        with h5py.File(path, "r") as f:
            image = f["raw"][0]
            label_image = f["labels"][0]
            label_image = label(label_image == 1)

    assert image.shape == label_image.shape
    return image, label_image


def get_covers(modality):
    if modality == "lm":
        return ["./covers/cover_lm.png"]
    else:
        return ["./covers/cover_em.png"]


def compute_checksum(path):
    xxh_checksum = xxhash.xxh128()
    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            xxh_checksum.update(data)
    return xxh_checksum.hexdigest()


def export_model(model_path, model_type, modality, version, email):
    output_folder = os.path.join(OUTPUT_FOLDER, modality)
    os.makedirs(output_folder, exist_ok=True)

    model_name = f"{model_type}_{modality}"
    output_path = os.path.join(output_folder, model_name)
    if os.path.exists(output_path):
        print("The model", model_name, "has already been exported.")
        return

    image, label_image = get_data(modality)
    covers = get_covers(modality)
    doc = create_doc(model_type, modality, version)

    model_id, emoji = get_id_and_emoji(model_name)
    uploader = spec.Uploader(email=email)

    export_name = MODEL_TO_NAME[model_name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_sam_model(
            image=image,
            label_image=label_image,
            name=export_name,
            model_type=model_type,
            checkpoint_path=model_path,
            output_path=output_path,
            documentation=doc,
            covers=covers,
            id=model_id,
            id_emoji=emoji,
            uploader=uploader,
            version=BIOIMAGEIO_VERSION,
        )

    # Unzip files
    from torch_em.data.datasets.util import unzip
    unzip(zip_path=output_path, dst=(output_path + ".unzip"), remove=False)

    print("Exported model", model_id)
    encoder_path = os.path.join(output_path + ".unzip", f"{model_type}.pt")
    encoder_checksum = compute_checksum(encoder_path)
    print("Encoder:")
    print(model_name, f"xxh128:{encoder_checksum}")

    decoder_path = os.path.join(output_path + ".unzip", f"{model_type}_decoder.pt")
    decoder_checksum = compute_checksum(decoder_path)
    print("Decoder:")
    print(f"{model_name}_decoder", f"xxh128:{decoder_checksum}")

    breakpoint()


def export_all_models(email, version, model_type):
    if model_type is None:
        model_type = "vit*"

    models = glob(os.path.join(INPUT_FOLDER, f"v{version}/**/{model_type}"), recursive=True)
    for path in models:
        modality, _, model_type = path.split("/")[-3:]  # current expected structure: v4/lm/generalist/vit_b/best.pt
        model_path = os.path.join(path, "best.pt")
        print(model_path, modality, model_type)
        assert os.path.exists(model_path), model_path
        export_model(model_path, model_type, modality, version=version, email=email)


# For testing.
def export_vit_t_lm(email):
    model_type = "vit_t"
    model_path = os.path.join(INPUT_FOLDER, "lm", "generalist", model_type, "best.pt")
    export_model(model_path, model_type, "lm", version=4, email=email)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--email", required=True)
    parser.add_argument("-v", "--version", default=4, type=int)
    parser.add_argument("-m", "--model_type", type=str, default=None)
    args = parser.parse_args()

    export_all_models(args.email, args.version, args.model_type)


if __name__ == "__main__":
    main()
