from micro_sam.bioimageio import export_sam_model
from micro_sam.sample_data import synthetic_data


def export_model_with_synthetic_data():
    image, labels = synthetic_data(shape=(1024, 1022))

    # checkpoint_path = None
    checkpoint_path = "/home/pape/Work/my_projects/micro-sam/v2/lm/generalist/vit_t/best.pt"

    export_sam_model(
        image, labels,
        model_type="vit_t", name="sam-test-vit-t",
        output_path="./test_export.zip",
        checkpoint_path=checkpoint_path
    )


def main():
    export_model_with_synthetic_data()


if __name__ == "__main__":
    main()
