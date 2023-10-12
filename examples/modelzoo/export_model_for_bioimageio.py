from micro_sam.modelzoo import export_bioimageio_model
from micro_sam.sample_data import synthetic_data


def export_model_with_synthetic_data():
    image, labels = synthetic_data(shape=(1024, 1024))

    export_bioimageio_model(
        image, labels,
        model_type="vit_b", model_name="sam-test-vit-b",
        output_path="./test_export.zip",
    )


def main():
    export_model_with_synthetic_data()


if __name__ == "__main__":
    main()
