from micro_sam.util import microsam_cachedir
from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter


def main():
    get_predictor_and_segmenter(model_type="hvit_t_cells", segmentation_mode="ais")
    print(f"The models for the workshop have been downloaded to {microsam_cachedir()}")


if __name__ == "__main__":
    main()
