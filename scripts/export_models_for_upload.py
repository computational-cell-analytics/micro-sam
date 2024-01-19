"""Helper scripts to export models for upload to zenodo.
"""

import hashlib
import os
import warnings
from glob import glob

import xxhash
from micro_sam.util import export_custom_sam_model

BUF_SIZE = 65536  # lets read stuff in 64kb chunks!


def export_model(model_path, model_type, export_name):
    output_folder = "./exported_models"
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, export_name)
    if os.path.exists(output_path):
        print("The model", export_name, "has already been exported.")
        return

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_custom_sam_model(
            checkpoint_path=model_path,
            model_type=model_type,
            save_path=output_path,
        )

    print("Exported", export_name)

    sha_checksum = hashlib.sha256()
    xxh_checksum = xxhash.xxh128()

    with open(output_path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha_checksum.update(data)
            xxh_checksum.update(data)

    print("sha256:", f"sha256:{sha_checksum.hexdigest()}")
    print("xxh128:", f"xxh128:{xxh_checksum.hexdigest()}")


def export_all_models():
    models = glob(os.path.join("./new_models/*.pt"))
    model_type = "vit_b"
    for model_path in models:
        export_name = os.path.basename(model_path).replace(".pt", ".pth")
        export_model(model_path, model_type, export_name)


def main():
    export_all_models()


if __name__ == "__main__":
    main()
