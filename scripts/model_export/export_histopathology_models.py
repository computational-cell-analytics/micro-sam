import os
import xxhash
import argparse
import warnings
from glob import glob

import h5py

import bioimageio.spec.model.v0_5 as spec

from micro_sam.bioimageio import export_sam_model

from models import get_id_and_emoji


MODEL_TO_NAME = {
    "vit_b_histopathology": "SAM Histopathology Generalist (ViT-B)",
    "vit_l_histopathology": "SAM Histopathology Generalist (ViT-L)",
    "vit_h_histopathology": "SAM Histopathology Generalist (ViT-H)",
}

BUF_SIZE = 65536  # lets read stuff in 64kb chunks!
OUTPUT_FOLDER = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/exported_models/"
PUMA_ROOT = "/mnt/vast-nhr/projects/cidas/cca/experiments/patho_sam/data/puma"


def create_doc(model_type, version):
    template_file = os.path.join(
        os.path.split(__file__)[0], "../../doc/bioimageio", f"histopathology_v{version}.md"
    )
    assert os.path.exists(template_file), template_file
    with open(template_file, "r") as f:
        template = f.read()

    doc = template % (model_type, version)
    return doc


def get_data():
    input_paths = glob(os.path.join(PUMA_ROOT, "test", "preprocessed", "training_set_*.h5"))
    # Choose the first input path
    input_path = input_paths[0]

    with h5py.File(input_path, "r") as f:
        image = f["raw"][:]
        label_image = f["labels/nuclei"][:]

    # Convert to channels first.
    image = image.transpose(1, 2, 0)

    return image, label_image


def compute_checksum(path):
    xxh_checksum = xxhash.xxh128()
    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            xxh_checksum.update(data)
    return xxh_checksum.hexdigest()


def export_model(model_path, model_type, version, email):
    output_folder = os.path.join(OUTPUT_FOLDER, "histopathology")
    os.makedirs(output_folder, exist_ok=True)

    model_name = f"{model_type}_histopathology"

    output_path = os.path.join(output_folder, model_name)
    if os.path.exists(output_path):
        print("The model", model_name, "has already been exported.")
        return

    image, label_image = get_data()
    covers = ["./covers/cover_lm.png"]  # HACK: We use existing covers.
    doc = create_doc(model_type, version)

    model_id, emoji = get_id_and_emoji(model_name)
    uploader = spec.Uploader(email=email)

    export_name = MODEL_TO_NAME[model_name]
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
            id=model_id,
            id_emoji=emoji,
            uploader=uploader,
        )

    # NOTE: I needed to unzip the files myself. Not sure how this worked before. Maybe something changed in spec?
    from torch_em.data.datasets.util import unzip
    unzip(zip_path=output_path, dst=(output_path + ".unzip"))

    print("Exported model", model_id)
    encoder_path = os.path.join(output_path + ".unzip", f"{model_type}.pt")
    encoder_checksum = compute_checksum(encoder_path)
    print("Encoder:")
    print(model_name, f"xxh128:{encoder_checksum}")

    decoder_path = os.path.join(output_path + ".unzip", f"{model_type}_decoder.pt")
    decoder_checksum = compute_checksum(decoder_path)
    print("Decoder:")
    print(f"{model_name}_decoder", f"xxh128:{decoder_checksum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--email", required=True)
    parser.add_argument("-v", "--version", default=1, type=int)
    parser.add_argument("-c", "--checkpoint", required=True, type=str)
    parser.add_argument("-m", "--model_type", required=True, type=str)
    args = parser.parse_args()

    export_model(
        model_path=args.checkpoint,
        model_type=args.model_type,
        version=1,
        email=args.email,
    )


if __name__ == "__main__":
    main()
