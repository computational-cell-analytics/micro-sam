import os

import h5py
from math import ceil

from micro_sam.sam_annotator import image_series_annotator, annotator_3d


DATA_ROOT = "/home/anwai/data/lucchi/lucchi_test.h5"
EMBEDDING_ROOT = "/home/anwai/embeddiings/test/"
OUTPUT_ROOT = "/home/anwai/data/lucchi/outputs/"


def _get_volume(volume_path):
    """Getting the lucchi test volume"""
    with h5py.File(volume_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels"][:]

    return raw, labels


def segment_volume(input_volume, embedding_path):
    """Load the entire volume in the tool for segmentation.
    """
    assert input_volume.ndim == 3
    annotator_3d(
        image=input_volume,
        embedding_path=embedding_path,
        model_type="vit_b_em_organelles",
        tile_shape=None,
        halo=None,
    )


def segment_each_slice(input_volume, embedding_dir, output_folder):
    """Load each slice from the volume one-by-one in the tool for segmentation.
    """
    assert input_volume.ndim == 3

    all_slices = [each_slice for each_slice in input_volume]
    image_series_annotator(
        images=all_slices,
        output_folder=output_folder,
        model_type="vit_b_em_organelles",
        embedding_path=embedding_dir,
        tile_shape=None,
        halo=None,
    )


def segment_each_n_slices(z_batch, input_volume, embedding_dir, output_folder):
    """Load n slices from the volume one-by-one in the tool for segmentation.
    """
    assert input_volume.ndim == 3

    n_z_slices = input_volume.shape[0]
    all_z_idxx = int(ceil(n_z_slices / z_batch))

    all_per_n_slices_volumes = []
    for z_id in range(all_z_idxx):
        z_start = z_id * z_batch
        z_stop = min((z_id + 1) * z_batch, n_z_slices)

        batch_volume = input_volume[z_start: z_stop]
        all_per_n_slices_volumes.append(batch_volume)

    print(f"We split the volume into {len(all_per_n_slices_volumes)} sub-volumes.")
    image_series_annotator(
        images=all_per_n_slices_volumes,
        output_folder=output_folder,
        model_type="vit_b_em_organelles",
        embedding_path=embedding_dir,
        tile_shape=None,
        halo=None,
        is_volumetric=True,
    )


def main():
    raw, _ = _get_volume(DATA_ROOT)

    # segment_volume(
    #     input_volume=raw,
    #     embedding_path=os.path.join(EMBEDDING_ROOT, "lucchi_3d_volume")
    # )

    # segment_each_slice(
    #     input_volume=raw,
    #     embedding_dir=os.path.join(EMBEDDING_ROOT, "lucchi_2d_per_slice"),
    #     output_folder=os.path.join(OUTPUT_ROOT, "per_slice_segmentation")
    # )

    segment_each_n_slices(
        z_batch=15,
        input_volume=raw,
        embedding_dir=os.path.join(EMBEDDING_ROOT, "lucchi_3d_per_n_slices"),
        output_folder=os.path.join(OUTPUT_ROOT, "per_n_slices")
    )


if __name__ == "__main__":
    main()
