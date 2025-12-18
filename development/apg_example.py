import os
import time

import h5py
import napari

from micro_sam.sample_data import sample_data_hela_2d
from micro_sam.instance_segmentation import (
    TiledAutomaticPromptGenerator, AutomaticPromptGenerator, get_predictor_and_decoder
)
from micro_sam.util import precompute_image_embeddings
from elf.wrapper.resized_volume import ResizedVolume


# TODO example with a custom prompt function
def example_script():
    im = sample_data_hela_2d()[0][0]

    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    image_embeddings = precompute_image_embeddings(predictor, im, save_path="x.zarr")
    generator = AutomaticPromptGenerator(predictor, decoder)
    generator.initialize(im, image_embeddings=image_embeddings)
    segmentation = generator.generate(intersection_over_min=True)

    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(segmentation)
    napari.run()


def example_script_tiled():
    im = sample_data_hela_2d()[0][0]

    tile_shape, halo = (256, 256), (64, 64)
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_lm")
    image_embeddings = precompute_image_embeddings(predictor, im, tile_shape=tile_shape, halo=halo, save_path="y.zarr")
    generator = TiledAutomaticPromptGenerator(predictor, decoder)
    generator.initialize(im, image_embeddings=image_embeddings, tile_shape=tile_shape, halo=halo, verbose=True)
    segmentation = generator.generate(intersection_over_min=False)

    v = napari.Viewer()
    v.add_image(im)
    v.add_labels(segmentation)
    napari.run()


def _require_wsi_data():
    out_path = "./data/wsi.h5"
    if os.path.exists(out_path):
        return out_path

    from micro_sam.sample_data import fetch_wholeslide_histopathology_example_data
    from patho_sam.io.util import read_wsi

    example_data = fetch_wholeslide_histopathology_example_data("./data")
    data = read_wsi(example_data)
    shape = data.shape[:2]

    with h5py.File(out_path, "w") as f:
        f.create_dataset("data/s0", data=data, compression="gzip")
        for level in range(1, 5):
            ds_shape = tuple(sh // (2 ** level) for sh in shape)
            print(level, ds_shape)
            data = read_wsi(example_data, scale=ds_shape)
            f.create_dataset(f"data/s{level}", data=data, compression="gzip")

    os.remove(example_data)
    return out_path


def _require_mask(path, level=4, bg_threshold=240, window=15, majority_threshold=0.3):
    mask_key = f"mask/s{level}"
    with h5py.File(path, "a") as f:
        full_shape = f["data/s0"].shape[:2]
        if mask_key in f:
            mask = f[mask_key][:]
        else:
            from scipy.ndimage import uniform_filter
            image = f[f"data/s{level}"][:]
            mask = (image > bg_threshold).all(axis=-1)
            mask = uniform_filter(mask.astype("float"), size=window)
            mask = ~(mask >= majority_threshold)
            f.create_dataset(mask_key, data=mask, compression="gzip")

    resized_mask = ResizedVolume(mask, shape=full_shape, order=0)
    return resized_mask


def example_script_wsi():
    data_path = _require_wsi_data()
    mask = _require_mask(data_path)

    tile_shape, halo = (768, 768), (64, 64)
    predictor, decoder = get_predictor_and_decoder(model_type="vit_b_histopathology")

    with h5py.File(data_path, "r") as f:
        data = f["data/s0"][:]
        print("Run prediction for WSI of shape:", data.shape)

        # Processing time: 10:34 min (batch size 24 on an A100 with 80 GB)
        # WITH MASK: 3:33 min (+ some further optimizartions)
        embed_path = "./data/embeds.zarr"
        image_embeddings = precompute_image_embeddings(
            predictor, data, tile_shape=tile_shape, halo=halo, save_path=embed_path, batch_size=24, ndim=2, mask=mask,
        )

        # Processing time: 03:14 min (batch size 24 on an A100 with 80 GB)
        # WITH MASK: 34 seconds
        generator = TiledAutomaticPromptGenerator(predictor, decoder)
        generator.initialize(
            data, image_embeddings=image_embeddings, tile_shape=tile_shape, halo=halo, verbose=True, batch_size=24
        )

        # Processing time: 21:12 min
        # Out of this 18:09 for the batched prediction, the rest for pre/post-processing.
        # WITH MASK: 19:59 min (total time).
        print("Start generate ...")
        t0 = time.time()
        seg = generator.generate(batch_size=32, optimize_memory=True)
        print("Generate took:", time.time() - t0, "s")
        print(seg.shape)

        # Save the segmentation to check the result
        with h5py.File("./data/seg.h5", "w") as f:
            f.create_dataset("seg", data=seg, compression="gzip")


def debug_wsi():
    from micro_sam.inference import _stitch_segmentation
    from nifty.tools import blocking
    from tqdm import tqdm

    print("Load data for debugging ....")
    masks = []
    with h5py.File("./debug.h5", mode="r") as f:
        tile_ids = f["tile_ids"][:]
        g = f["masks"]
        for tile_id in tqdm(tile_ids, desc="Load masks"):
            masks.append(g[str(tile_id)][:])

        halo = f.attrs["halo"]
        shape = f.attrs["shape"]
        tile_shape = f.attrs["tile_shape"]

    tiling = blocking([0, 0], shape, tile_shape)
    print("Start stitching ...")
    seg = _stitch_segmentation(masks, tile_ids, tiling, halo, output_shape=shape)
    print(seg.shape)


def main():
    # example_script()
    # example_script_tiled()
    example_script_wsi()
    # debug_wsi()


if __name__ == "__main__":
    main()
