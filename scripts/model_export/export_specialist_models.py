"""Helper scripts to export models for upload to bioimageio/zenodo.
"""

import os
import argparse
import warnings

import h5py
import xxhash
import imageio.v3 as imageio
from skimage.measure import label

import bioimageio.spec.model.v0_5 as spec

from micro_sam.bioimageio import export_sam_model

from models import get_id_and_emoji


MODEL_TO_NAME = {
    # light microscopy specialists
    "vit_t_livecell_lm": "SAM LIVECell LM Specialist (ViT-T)",
    "vit_b_livecell_lm": "SAM LIVECell LM Specialist (ViT-B)",
    "vit_l_livecell_lm": "SAM LIVECell LM Specialist (ViT-L)",
    "vit_h_livecell_lm": "SAM LIVECell LM Specialist (ViT-H)",
    "vit_t_deepbacs_lm": "SAM DeepBacs LM Specialist (ViT-T)",
    "vit_b_deepbacs_lm": "SAM DeepBacs LM Specialist (ViT-B)",
    "vit_l_deepbacs_lm": "SAM DeepBacs LM Specialist (ViT-L)",
    "vit_h_deepbacs_lm": "SAM DeepBacs LM Specialist (ViT-H)",
    "vit_t_tissuenet_lm": "SAM TissueNet LM Specialist (ViT-T)",
    "vit_b_tissuenet_lm": "SAM TissueNet LM Specialist (ViT-B)",
    "vit_l_tissuenet_lm": "SAM TissueNet LM Specialist (ViT-L)",
    "vit_h_tissuenet_lm": "SAM TissueNet LM Specialist (ViT-H)",
    "vit_t_neurips_cellseg_lm": "SAM NeurIPS CellSeg LM Specialist (ViT-T)",
    "vit_b_neurips_cellseg_lm": "SAM NeurIPS CellSeg LM Specialist (ViT-B)",
    "vit_l_neurips_cellseg_lm": "SAM NeurIPS CellSeg LM Specialist (ViT-L)",
    "vit_h_neurips_cellseg_lm": "SAM NeurIPS CellSeg LM Specialist (ViT-H)",
    "vit_t_plantseg_root_lm": "SAM PlantSeg (Root) LM Specialist (ViT-T)",
    "vit_b_plantseg_root_lm": "SAM PlantSeg (Root) LM Specialist (ViT-B)",
    "vit_l_plantseg_root_lm": "SAM PlantSeg (Root) LM Specialist (ViT-L)",
    "vit_h_plantseg_root_lm": "SAM PlantSeg (Root) LM Specialist (ViT-H)",
    # electron microscopy specialists
    "vit_t_asem_er_em_organelles": "SAM ASEM (ER) EM Specialist (ViT-T)",
    "vit_b_asem_er_em_organelles": "SAM ASEM (ER) EM Specialist (ViT-B)",
    "vit_l_asem_er_em_organelles": "SAM ASEM (ER) EM Specialist (ViT-L)",
    "vit_h_asem_er_em_organelles": "SAM ASEM (ER) EM Specialist (ViT-H)",
    "vit_t_cremi_em_boundaries": "SAM CREMI EM Specialist (ViT-T)",
    "vit_b_cremi_em_boundaries": "SAM CREMI EM Specialist (ViT-B)",
    "vit_l_cremi_em_boundaries": "SAM CREMI EM Specialist (ViT-L)",
    "vit_h_cremi_em_boundaries": "SAM CREMI EM Specialist (ViT-H)",
    # vit_h models for the generalist models
    "vit_h_lm": "SAM LM Generalist (ViT-H)",
    "vit_h_em_organelles": "SAM EM Organelles Generalist (ViT-H)",
    # user study models
    "vit_b_2d_user_study_lm": "SAM 2d User Study Finetuned (ViT-B)",
    "vit_b_3d_user_study_em_organelles": "SAM 3d User Study Finetuned (ViT-B)",
    "vit_l_tracking_user_study_lm": "SAM Tracking User Study Finetuned (ViT-L)",
}

BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

INPUT_FOLDER = "/scratch/usr/nimanwai/micro-sam/checkpoints/"
OUTPUT_FOLDER = "/scratch/usr/nimanwai/exported_models/"

LIVECELL_ROOT = "/scratch/usr/nimanwai/data/livecell/"
KASTHURI_ROOT = "/scratch/usr/nimanwai/data/em/kasthuri"


class FilterObjectsLabelTrafo:
    """Dummy interface for the namespace to dodge errors with ASEM (ER) specialist.
    """
    def __init__(self):
        pass


def create_doc(model_type, modality, version):
    if modality not in ("lm", "em_organelles", "em_boundaries"):
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
            LIVECELL_ROOT, "images", "livecell_train_val_images", "A172_Phase_A7_1_00d00h00m_1.tif"
        )
        label_path = os.path.join(
            LIVECELL_ROOT, "annotations", "livecell_train_val_images", "A172/A172_Phase_A7_1_00d00h00m_1.tif"
        )
        image = imageio.imread(image_path)
        label_image = imageio.imread(label_path)
    else:
        path = os.path.join(KASTHURI_ROOT, "kasthuri_train.h5")
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


def export_model(model_path, model_type, modality, version, email, dataset=None):
    output_folder = os.path.join(OUTPUT_FOLDER, modality)
    os.makedirs(output_folder, exist_ok=True)

    if dataset is None:
        model_name = f"{model_type}_{modality}"
    else:
        model_name = f"{model_type}_{dataset}_{modality}"

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
    print(model_name, f"xxh128:{encoder_checksum}")

    decoder_path = os.path.join(output_path + ".unzip", f"{model_type}_decoder.pt")
    decoder_checksum = compute_checksum(decoder_path)
    print("Decoder:")
    print(f"{model_name}_decoder", f"xxh128:{decoder_checksum}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--email", required=True)
    parser.add_argument("-v", "--version", default=2, type=int)
    parser.add_argument("-c", "--checkpoint", required=True, type=str)
    parser.add_argument("-m", "--model_type", required=True, type=str)
    parser.add_argument("-d", "--dataset", default=None, type=str)
    parser.add_argument("--modality", required=True, type=str)
    args = parser.parse_args()

    export_model(
        model_path=args.checkpoint,
        model_type=args.model_type,
        modality=args.modality,
        version=2,
        email=args.email,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
