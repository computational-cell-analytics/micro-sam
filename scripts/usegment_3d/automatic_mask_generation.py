"""Automatic mask generation (AMG) with SAM2 for 2d images and 3d volumes.

Runs the grid-based automatic mask generation of SAM2 via
`micro_sam.v2.instance_segmentation.AutomaticMaskGenerationSegmenter`. For 3d data the slices are
segmented individually and stitched across z with the multi-dimensional segmentation stitching.
"""

import argparse
import os

import imageio.v3 as imageio

from micro_sam.util import get_cache_directory
from micro_sam.sample_data import fetch_nucleus_3d_example_data, fetch_hela_2d_example_data
from micro_sam.v2.util import get_sam2_model
from micro_sam.v2.instance_segmentation import get_amg_segmenter, amg_3d_segmentation


DATA_CACHE = os.path.join(get_cache_directory(), "sample_data")


def run_2d_amg(model_type, tile_shape, halo, embedding_path, view, generate_kwargs):
    """Run SAM2 AMG for an example 2d image from the Cell Tracking Challenge (HeLa) dataset."""
    image = imageio.imread(fetch_hela_2d_example_data(DATA_CACHE))

    model = get_sam2_model(model_type=model_type)
    segmenter = get_amg_segmenter(model, is_tiled=tile_shape is not None, model_type=model_type)

    if tile_shape is None:
        init_kwargs = {"save_path": embedding_path}
    else:
        init_kwargs = {"tile_shape": tile_shape, "halo": halo}
    segmenter.initialize(image, **init_kwargs)
    segmentation = segmenter.generate(**generate_kwargs)

    print(f"2d AMG found {int(segmentation.max())} objects.")
    if view:
        import napari
        v = napari.Viewer()
        v.add_image(image)
        v.add_labels(segmentation)
        napari.run()


def _crop_center(volume, crop):
    """Crop a central (crop, crop) region in xy from a (Z, Y, X) volume."""
    _, y, x = volume.shape
    y0, x0 = (y - crop) // 2, (x - crop) // 2
    return volume[:, y0:y0 + crop, x0:x0 + crop]


def run_3d_amg(model_type, tile_shape, halo, crop, view, generate_kwargs):
    """Run SAM2 AMG for an example 3d nucleus volume, stitching the slices across z."""
    volume = imageio.imread(fetch_nucleus_3d_example_data(DATA_CACHE))
    if crop is not None:
        volume = _crop_center(volume, crop)

    model = get_sam2_model(model_type=model_type)
    segmenter = get_amg_segmenter(model, is_tiled=tile_shape is not None, model_type=model_type)

    segmentation = amg_3d_segmentation(
        volume=volume, segmenter=segmenter, tile_shape=tile_shape, halo=halo, **generate_kwargs
    )

    print(f"3d AMG found {int(segmentation.max())} objects.")
    if view:
        import napari
        v = napari.Viewer()
        v.add_image(volume)
        v.add_labels(segmentation)
        napari.run()


def main():
    parser = argparse.ArgumentParser(description="Run SAM2 automatic mask generation (AMG).")
    parser.add_argument("--ndim", type=int, default=3, choices=(2, 3), help="Run 2d or 3d AMG.")
    parser.add_argument("--model_type", default="hvit_t", help="The SAM2 model type, e.g. 'hvit_t'.")
    parser.add_argument("--min_object_size", type=int, default=50, help="Minimal object size in pixels.")
    parser.add_argument("--tile_shape", type=int, nargs=2, default=None, help="Tile shape (y, x) for tiling.")
    parser.add_argument("--halo", type=int, nargs=2, default=None, help="Halo (y, x) for tiling.")
    parser.add_argument("--crop", type=int, default=None, help="Run 3d AMG on a central xy crop of this size.")
    parser.add_argument("--embedding_path", default=None, help="Path to cache the 2d image embeddings (zarr).")
    parser.add_argument("--view", action="store_true", help="Display the result in napari.")
    args = parser.parse_args()

    tile_shape = tuple(args.tile_shape) if args.tile_shape is not None else None
    halo = tuple(args.halo) if args.halo is not None else None
    if tile_shape is not None and halo is None:
        halo = (64, 64)

    generate_kwargs = {"min_object_size": args.min_object_size}
    if args.ndim == 2:
        run_2d_amg(args.model_type, tile_shape, halo, args.embedding_path, args.view, generate_kwargs)
    else:
        run_3d_amg(args.model_type, tile_shape, halo, args.crop, args.view, generate_kwargs)


if __name__ == "__main__":
    main()
