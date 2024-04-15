"""Helper scripts to export models for upload to bioimageio/zenodo.
"""

import argparse
import os
import json
import warnings
from glob import glob

import bioimageio.spec.model.v0_5 as spec
import imageio.v3 as imageio
import numpy as np
import requests
import xxhash
import yaml

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


def download_file(url, filename):
    if os.path.exists(filename):
        return

    # Send HTTP GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file in write-text mode
        with open(filename, "w", encoding=response.encoding or "utf-8") as file:
            file.write(response.text)  # Using .text instead of .content
        print(f"File '{filename}' has been downloaded successfully.")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")


def get_id_and_emoji():
    addjective_url = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/main/adjectives.txt"
    animal_url = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/main/animals.yaml"
    collection_url = "https://raw.githubusercontent.com/bioimage-io/collection-bioimage-io/gh-pages/collection.json"

    adjective_file = "adjectives.txt"
    download_file(addjective_url, adjective_file)
    adjectives = []
    with open(adjective_file) as f:
        for adj in f.readlines():
            adjectives.append(adj.rstrip("\n"))

    animal_file = "animals.yaml"
    download_file(animal_url, animal_file)
    with open(animal_file) as f:
        animal_dict = yaml.safe_load(f)
    animal_names = list(animal_dict.keys())

    collection_file = "collection.json"
    download_file(collection_url, collection_file)
    with open(collection_file) as f:
        collection = json.load(f)["collection"]

    existing_ids = []
    for entry in collection:
        this_id = entry.get("nickname", None)
        if this_id is None:
            continue
        existing_ids.append(this_id)

    adj, name = np.random.choice(adjectives), np.random.choice(animal_names)
    model_id = f"{adj}-{name}"
    while model_id in existing_ids:
        adj, name = np.random.choice(adjectives), np.random.choice(animal_names)
        model_id = f"{adj}-{name}"

    return model_id, animal_dict[name]


def get_data(modality):
    if modality == "lm":
        image_path = os.path.join(
            "/home/pape/Work/data/incu_cyte/livecell/images",
            "livecell_train_val_images/A172_Phase_A7_1_00d00h00m_1.tif"
        )
        label_path = os.path.join(
            "/home/pape/Work/data/incu_cyte/livecell/annotations",
            "livecell_train_val_images/A172/A172_Phase_A7_1_00d00h00m_1.tif"
        )
        image = imageio.imread(image_path)
        label_image = imageio.imread(label_path)
        assert image.shape == label_image.shape
    else:
        raise RuntimeError(modality)

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

    export_name = f"{model_type}_{modality}"
    output_path = os.path.join(output_folder, export_name)
    if os.path.exists(output_path):
        print("The model", export_name, "has already been exported.")
        return

    image, label_image = get_data(modality)
    covers = get_covers(modality)
    doc = create_doc(model_type, modality, version)

    model_id, emoji = get_id_and_emoji()
    uploader = spec.Uploader(email=email)

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

    print("Exported model", model_id)
    encoder_path = os.path.join(output_path + ".unzip", f"{model_type}.pt")
    encoder_checksum = compute_checksum(encoder_path)
    print("Encoder:")
    print(export_name, f"xxh128:{encoder_checksum}")

    decoder_path = os.path.join(output_path + ".unzip", f"{model_type}_decoder.pt")
    decoder_checksum = compute_checksum(decoder_path)
    print("Decoder:")
    print(f"{export_name}_decoder", f"xxh128:{decoder_checksum}")


def export_all_models():
    models = glob(os.path.join("./new_models/*.pt"))
    model_type = "vit_b"
    for model_path in models:
        export_name = os.path.basename(model_path).replace(".pt", ".pth")
        export_model(model_path, model_type, export_name)


def export_vit_t_lm(email):
    model_type = "vit_t"
    model_path = os.path.join(INPUT_FOLDER, "lm", "generalist", model_type, "best.pt")
    export_model(model_path, model_type, "lm", version=2, email=email)


# Update this to automate model exports more.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--email", required=True)
    args = parser.parse_args()

    export_vit_t_lm(args.email)
    # export_all_models()


if __name__ == "__main__":
    main()
