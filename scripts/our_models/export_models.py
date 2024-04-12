"""Helper scripts to export models for upload to bioimageio/zenodo.
"""

import os
import warnings
from glob import glob

import imageio.v3 as imageio
# import numpy as np
import xxhash
from micro_sam.bioimageio import export_sam_model

BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

INPUT_FOLDER = "./v2"
OUTPUT_FOLDER = "./exported_models"


def create_doc(model_type, modality, version):
    if modality not in ("lm", "em_mito"):
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
        image_path = "/home/pape/Work/data/incu_cyte/livecell/images/livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif"
        label_path = "/home/pape/Work/data/incu_cyte/livecell/annotations/livecell_train_val_images/A172/A172_Phase_A7_1_00d00h00m_1.tif"
        image = imageio.imread(image_path)
        label_image = imageio.imread(label_path)
        assert image.shape == label_image.shape
    else:
        raise RuntimeError(modality)

    return image, label_image


# TODO
def get_covers(modality):
    return None


def compute_checksum(path):
    xxh_checksum = xxhash.xxh128()
    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            xxh_checksum.update(data)
    return xxh_checksum.hexdigest()


def export_model(model_path, model_type, modality, version):
    output_folder = os.path.join(OUTPUT_FOLDER, modality)
    os.makedirs(output_folder, exist_ok=True)

    export_name = f"{model_type}_{modality}"
    output_path = os.path.join(output_folder, export_name)
    if os.path.exists(output_path):
        print("The model", export_name, "has already been exported.")
        return

    image, label_image = get_data(modality)
    covers = get_covers(modality)
    doc = create_doc(model_type, modality, version)

    # TODO ids and uploader stuff
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_sam_model(
            image, label_image,
            name=export_name,
            model_type=model_type,
            checkpoint_path=model_path,
            output_path=output_path,
            documentation=doc,
            covers=covers,
        )

    encoder_path = os.path.join(output_path + ".unzip", f"{model_type}.pt")
    encoder_checksum = compute_checksum(encoder_path)
    print("Encoder:")
    print(export_name, f"xxh128:{encoder_checksum}")

    decoder_path = os.path.join(output_path + ".unzip", f"{model_type}_decoder.pt")
    decoder_checksum = compute_checksum(decoder_path)
    print("Decoder:")
    print(export_name, f"xxh128:{decoder_checksum}")


# TODO
def export_all_models():
    models = glob(os.path.join("./new_models/*.pt"))
    model_type = "vit_b"
    for model_path in models:
        export_name = os.path.basename(model_path).replace(".pt", ".pth")
        export_model(model_path, model_type, export_name)


def export_lm_vitt():
    model_type = "vit_t"
    model_path = os.path.join(INPUT_FOLDER, "lm", "generalist", model_type, "best.pt")
    export_model(model_path, model_type, "lm", version=2)


def main():
    export_lm_vitt()
    # export_all_models()


if __name__ == "__main__":
    main()
