import numpy as np
from tqdm import tqdm
import imageio.v3 as imageio

from bioimage_cpp.transformation import resample

from micro_sam.automatic_segmentation import get_predictor_and_segmenter


IMAGE_PATH = "/home/anwai/data/tehmina_data/Anwai/image_1066422_z900.tif"
CHECKPOINT_PATH = "/home/anwai/data/tehmina_data/Anwai/downsample/best.pt"
OUTPUT_PATH = "/home/anwai/data/tehmina_data/Anwai/image_1066422_z900_prediction.tif"

# The model was finetuned on images downsampled by 2, e.g. (7813, 6823) -> (3906, 3411).
DOWNSCALE_FACTOR = 2
# Tile plus halo on both sides is 1024, the finetuning patch shape.
TILE_SHAPE, HALO = (512, 512), (256, 256)
STATE_KEYS = ("foreground", "center_distances", "boundary_distances")


def resize(data, shape, order):
    # Same pixel center mapping as skimage.transform.resize.
    scales = np.array(data.shape) / np.array(shape)
    matrix = np.concatenate([np.diag(scales), (0.5 * scales - 0.5)[:, None]], axis=1)
    return resample(data, matrix, bounding_box=tuple(slice(0, s) for s in shape), order=order)


def feather_weights(shape):
    # Linear ramps over the overlap of neighboring tiles, which is two times the halo.
    profiles = []
    for size, halo in zip(shape, HALO):
        distance = np.minimum(np.arange(size), np.arange(size)[::-1]) + 0.5
        profiles.append(np.minimum(distance / (2 * halo), 1))
    return np.outer(*profiles)


def get_tiles(shape):
    (ty, tx), (hy, hx) = TILE_SHAPE, HALO
    return [
        (slice(max(y - hy, 0), min(y + ty + hy, shape[0])), slice(max(x - hx, 0), min(x + tx + hx, shape[1])))
        for y in range(0, shape[0], ty) for x in range(0, shape[1], tx)
    ]


def segment(image):
    image = image.astype(np.float32)
    image = 255 * (image - image.min()) / (image.max() - image.min() + 1e-8)
    _, segmenter = get_predictor_and_segmenter(model_type="vit_b", checkpoint=CHECKPOINT_PATH, segmentation_mode="ais")

    # Feathered stitching: average overlapping tile outputs to avoid cuts at the tile borders.
    maps, weight_sum = np.zeros((3,) + image.shape, dtype="float32"), np.zeros(image.shape, dtype="float32")
    for tile in tqdm(get_tiles(image.shape), desc="Predict tiles"):
        segmenter.initialize(image[tile])
        state = segmenter.get_state()
        weights = feather_weights(image[tile].shape)
        maps[(slice(None),) + tile] += weights * np.stack([state[key] for key in STATE_KEYS])
        weight_sum[tile] += weights

    segmenter.set_state(dict(zip(STATE_KEYS, maps / weight_sum)))
    return segmenter.generate()


def main():
    image = imageio.imread(IMAGE_PATH)
    shape = tuple(s // DOWNSCALE_FACTOR for s in image.shape)
    # Resize in float and truncate, like the finetuning data.
    small_image = resize(image.astype(np.float32), shape, order=1).astype(image.dtype)
    print(f"Image shape: {image.shape}, downscaled by {DOWNSCALE_FACTOR} to: {small_image.shape}")

    segmentation = segment(small_image)
    print(f"Instances: {len(np.unique(segmentation)) - 1}")

    segmentation = resize(segmentation.astype(np.uint32), image.shape, order=0)
    imageio.imwrite(OUTPUT_PATH, segmentation, compression="zlib")
    print(f"Saved segmentation with shape {segmentation.shape} to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
