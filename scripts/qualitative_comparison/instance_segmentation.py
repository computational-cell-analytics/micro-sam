import time
import imageio.v3 as imageio
import napari
from elf.io import open_file

from micro_sam.util import get_sam_model
from micro_sam.instance_segmentation import (
    mask_data_to_segmentation, get_amg, get_predictor_and_decoder
)


def compare_instance_segmentation(image, segmenters, segmenter_kwargs):
    segmentations = {}
    times = {}

    for name, segmenter in segmenters.items():
        kwargs = segmenter_kwargs[name]
        t0 = time.time()
        segmenter.initialize(image)
        seg = segmenter.generate(**kwargs)
        seg = mask_data_to_segmentation(seg, with_background=True)
        segmentations[name] = seg
        times[name] = time.time() - t0

    v = napari.Viewer()
    v.add_image(image)
    for name, seg in segmentations.items():
        v.add_labels(seg, name=name)
    v.add_image(segmenter._foreground, name="foreground-pred")
    napari.run()

    print("Runtimes:")
    for name, rt in times.items():
        print(name, ":", rt, "s")


def load_amg(model_type):
    predictor = get_sam_model(model_type=model_type, device="cpu")
    return get_amg(predictor, is_tiled=False)


def load_instance_segmentation_with_decoder_from_checkpoint(model, checkpoint):
    predictor, decoder = get_predictor_and_decoder(
        model_type=model, checkpoint_path=checkpoint, device="cpu"
    )
    segmenter = get_amg(predictor=predictor, is_tiled=False, decoder=decoder)
    return segmenter


def instance_segmentation_lucchi():
    data_path = "/media/anwai/ANWAI/data/lucchi/Lucchi++/Test_In"
    with open_file(data_path, "r") as f:
        image = f["*.png"][0]

    segmenters = {
        "vit_b_amg": load_amg("vit_b"),
        "vit_b_em_amg": load_amg("vit_b_em_organelles"),
        "vit_b_em_ais": load_instance_segmentation_with_decoder_from_checkpoint(
            "vit_b", "/home/anwai/models/micro-sam/vit_b/em_organelles/best.pt"
        ),
    }
    segmenter_kwargs = {
        "vit_b_amg": {},
        "vit_b_em_amg": {
            "pred_iou_thresh": 0.83, "stability_score_thresh": 0.8
        },
        "vit_b_em_ais": {
            "center_distance_threshold": 0.3,
            "boundary_distance_threshold": 0.4,
            "distance_smoothing": 2.2,
            "min_size": 200,
        },
    }

    compare_instance_segmentation(image, segmenters, segmenter_kwargs)


def instance_segmentation_livecell():
    data_path = "/home/anwai/.cache/micro_sam/sample_data/hela-2d-image.png"
    image = imageio.imread(data_path)

    segmenters = {
        "vit_b_amg": load_amg("vit_b"),
        "vit_b_em_amg": load_amg("vit_b_lm"),
        "vit_b_em_ais": load_instance_segmentation_with_decoder_from_checkpoint(
            "vit_b", "/home/anwai/models/micro-sam/vit_b/lm_generalist/best.pt"
        ),
    }
    # TODO update the params!!
    segmenter_kwargs = {
        "vit_b_amg": {},
        "vit_b_em_amg": {
            "pred_iou_thresh": 0.83, "stability_score_thresh": 0.8
        },
        "vit_b_em_ais": {
            "center_distance_threshold": 0.3,
            "boundary_distance_threshold": 0.4,
            "distance_smoothing": 2.2,
            "min_size": 200,
        },
    }

    compare_instance_segmentation(image, segmenters, segmenter_kwargs)


# Times Lucchi: @ CP
# vit_b_amg : 84.40151119232178 s
# vit_b_em_amg : 82.40972113609314 s
# vit_b_em_ais : 13.019227027893066 s

# Times Lucchi: @ AA
# vit_b_amg : 48.53724002838135 s
# vit_b_em_amg : 49.28683686256409 s
# vit_b_em_ais : 8.000006437301636 s

# Times CTC DIC-HeLa: @ AA
# vit_b_amg : 42.097354888916016 s
# vit_b_em_amg : 47.52872157096863 s
# vit_b_em_ais : 8.633260250091553 s


def main():
    instance_segmentation_lucchi()
    # instance_segmentation_livecell()


if __name__ == "__main__":
    main()
