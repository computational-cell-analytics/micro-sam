from micro_sam.util import get_model_cache_directory
from micro_sam.v2.automatic_segmentation import get_predictor_and_segmenter


def main():
    get_predictor_and_segmenter(model_type="hvit_t_cells", segmentation_mode="ais")
    print(f"The models for the workshop have been downloaded to {get_model_cache_directory()}")


if __name__ == "__main__":
    main()
