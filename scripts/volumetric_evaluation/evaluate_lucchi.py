import os

import h5py
from skimage.measure import label

from micro_sam import instance_segmentation
from micro_sam.multi_dimensional_segmentation import automatic_3d_segmentation
from micro_sam.evaluation.multi_dimensional_segmentation import run_multi_dimensional_segmentation_grid_search


def get_raw_and_label_volumes(volume_path):
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]

    return raw, labels


def _interactive_segmentation(args):
    test_volume_path = os.path.join(args.input_path, "lucchi_test.h5")
    volume, labels = get_raw_and_label_volumes(test_volume_path)

    # applying connected components to get instances
    labels = label(labels)

    run_multi_dimensional_segmentation_grid_search(
        volume=volume,
        ground_truth=labels,
        model_type=args.model_type,
        checkpoint_path=args.checkpoint,
        embedding_path=args.embedding_path,
        result_dir="./results_v2_em_organelles/",
        interactive_seg_mode="box",
        verbose=False
    )


def _instance_segmentation_with_decoder(args):
    test_volume_path = os.path.join(args.input_path, "lucchi_test.h5")
    volume, labels = get_raw_and_label_volumes(test_volume_path)

    # applying connected components to get instances
    labels = label(labels)

    model_type = "vit_b"
    checkpoint_path = "/home/anwai/models/micro-sam/vit_b/em_organelles/best.pt"
    embedding_path = "/home/anwai/embeddings/lucchi_r2_embeddings/embeddings_vit_b_finetuned_v2"

    predictor, decoder = instance_segmentation.get_predictor_and_decoder(model_type, checkpoint_path)
    segmentor = instance_segmentation.InstanceSegmentationWithDecoder(predictor, decoder)

    instances = automatic_3d_segmentation(
        volume=volume,
        predictor=predictor,
        segmentor=segmentor,
        embedding_path=embedding_path,
        center_distance_threshold=0.3,
        boundary_distance_threshold=0.4,
        distance_smoothing=2.2,
        min_size=200,
        gap_closing=2,
    )

    import numpy as np
    from elf.evaluation import mean_segmentation_accuracy

    msa, sa = mean_segmentation_accuracy(instances, labels, return_accuracies=True)
    print("mSA for 3d volume is:", msa)
    print("SA50 for 3d volume is:", sa[0])
    print()

    per_slice_msa, per_slice_sa50 = [], []
    for _instance, _label in zip(instances, labels):
        msa, sa = mean_segmentation_accuracy(_instance, _label, return_accuracies=True)
        per_slice_msa.append(msa)
        per_slice_sa50.append(sa[0])

    print("mSA for mean over each 2d slice is:", np.mean(per_slice_msa))
    print("SA50 for mean over each 2d slice is:", np.mean(per_slice_sa50))
    print()

    import napari
    v = napari.Viewer()
    v.add_image(volume)
    v.add_labels(labels, visible=False)
    v.add_labels(instances)
    napari.run()


def main(args):
    # _interactive_segmentation(args)
    _instance_segmentation_with_decoder(args)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input_path", type=str, default="/scratch/projects/nim00007/sam/data/lucchi", help="Path to volume"
    )
    parser.add_argument("-m", "--model_type", type=str, default="vit_b", help="Name of the image encoder")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="The custom checkpoint path.")
    parser.add_argument("-e", "--embedding_path", type=str, default=None, help="Path to save embeddings")
    args = parser.parse_args()
    main(args)
