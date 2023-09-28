"""
Functionality for visualizing image embeddings.
"""

from typing import Tuple

import numpy as np

from elf.segmentation.embeddings import embedding_pca
from nifty.tools import blocking
from skimage.transform import resize

from .util import ImageEmbeddings


#
# PCA visualization for the image embeddings
#

def compute_pca(embeddings: np.ndarray) -> np.ndarray:
    """Compute the pca projection of the embeddings to visualize them as RGB image.

    Args:
        embeddings: The embeddings. For example predicted by the SAM image encoder.

    Returns:
        PCA of the embeddings, mapped to the pixels.
    """
    if embeddings.ndim == 4:
        pca = embedding_pca(embeddings.squeeze()).transpose((1, 2, 0))
    elif embeddings.ndim == 5:
        pca = []
        for embed in embeddings:
            vis = embedding_pca(embed.squeeze()).transpose((1, 2, 0))
            pca.append(vis)
        pca = np.stack(pca)
    else:
        raise ValueError(f"Expect input of ndim 4 or 5, got {embeddings.ndim}")
    return pca


def _get_crop(embed_shape, shape):
    if shape[0] == shape[1]:  # square image, we don't need to do anything
        crop = np.s_[:, :, :]
    elif shape[0] > shape[1]:
        aspect_ratio = float(shape[1] / shape[0])
        crop = np.s_[:, :int(aspect_ratio * embed_shape[1])]
    elif shape[1] > shape[0]:
        aspect_ratio = float(shape[0] / shape[1])
        crop = np.s_[:int(aspect_ratio * embed_shape[0]), :]
    return crop


def _project_embeddings(embeddings, shape, apply_crop=True):
    assert embeddings.ndim == len(shape) + 2, f"{embeddings.shape}, {shape}"

    embedding_vis = compute_pca(embeddings)
    if not apply_crop:
        pass
    elif len(shape) == 2:
        crop = _get_crop(embedding_vis.shape, shape)
        embedding_vis = embedding_vis[crop]
    elif len(shape) == 3:
        crop = _get_crop(embedding_vis.shape[1:], shape[1:])
        crop = (slice(None),) + crop
        embedding_vis = embedding_vis[crop]
    else:
        raise ValueError(f"Expect 2d or 3d data, got {len(shape)}")

    scale = tuple(float(sh) / vsh for sh, vsh in zip(shape, embedding_vis.shape))
    return embedding_vis, scale


def _project_embeddings_to_tile(tile, tile_embeds):
    outer_tile = tile.outerBlock
    inner_tile_local = tile.innerBlockLocal

    embed_shape = tile_embeds.shape[-2:]
    outer_tile_shape = tuple(end - beg for beg, end in zip(outer_tile.begin, outer_tile.end))

    crop = _get_crop(embed_shape, outer_tile_shape)
    crop = (tile_embeds.ndim - len(crop)) * (slice(None),) + crop
    this_embeds = tile_embeds[crop]

    tile_scale = tuple(esh / float(fsh) for esh, fsh in zip(this_embeds.shape[-2:], outer_tile_shape))
    tile_bb = tuple(
        slice(int(np.round(beg * scale)), int(np.round(end * scale)))
        for beg, end, scale in zip(inner_tile_local.begin, inner_tile_local.end, tile_scale)
    )
    tile_bb = (tile_embeds.ndim - len(outer_tile_shape)) * (slice(None),) + tile_bb

    this_embeds = this_embeds[tile_bb]
    return this_embeds


def _resize_and_cocatenate(arrays, axis):
    assert axis in (-1, -2)
    resize_axis = -1 if axis == -2 else -2
    resize_len = max([arr.shape[resize_axis] for arr in arrays])

    def resize_shape(shape):
        axis_ = arrays[0].ndim + resize_axis
        return tuple(resize_len if i == axis_ else sh for i, sh in enumerate(shape))

    return np.concatenate(
        [resize(arr, resize_shape(arr.shape)) for arr in arrays],
        axis=axis
    )


def _project_tiled_embeddings(image_embeddings):
    features = image_embeddings["features"]
    tile_shape, halo, shape = features.attrs["tile_shape"], features.attrs["halo"], features.attrs["shape"]
    tiling = blocking([0, 0], shape, tile_shape)

    tile_grid = tiling.blocksPerAxis

    embeds = {
        i: {j: None for j in range(tile_grid[1])} for i in range(tile_grid[0])
    }

    for tile_id in range(tiling.numberOfBlocks):
        tile_embeds = features[tile_id][:]
        assert tile_embeds.ndim in (4, 5)

        # extract the embeddings corresponding to the inner tile
        tile = tiling.getBlockWithHalo(tile_id, list(halo))
        tile_coords = tiling.blockGridPosition(tile_id)
        this_embeds = _project_embeddings_to_tile(tile, tile_embeds)

        i, j = tile_coords
        embeds[i][j] = this_embeds

    embeds = _resize_and_cocatenate(
        [
            _resize_and_cocatenate(
                [embeds[i][j] for j in range(tile_grid[1])], axis=-1
            )
            for i in range(tile_grid[0])
        ], axis=-2
    )

    if features[0].ndim == 5:
        shape = (features[0].shape[0],) + tuple(shape)
    embedding_vis, scale = _project_embeddings(embeds, shape, apply_crop=False)
    return embedding_vis, scale


def project_embeddings_for_visualization(
    image_embeddings: ImageEmbeddings
) -> Tuple[np.ndarray, Tuple[float, ...]]:
    """Project image embeddings to pixel-wise PCA.

    Args:
        image_embeddings: The image embeddings.

    Returns:
        The PCA of the embeddings.
        The scale factor for resizing to the original image size.
    """
    is_tiled = image_embeddings["input_size"] is None
    if is_tiled:
        embedding_vis, scale = _project_tiled_embeddings(image_embeddings)
    else:
        embeddings = image_embeddings["features"]
        shape = tuple(image_embeddings["original_size"])
        if embeddings.ndim == 5:
            shape = (embeddings.shape[0],) + shape
        embedding_vis, scale = _project_embeddings(embeddings, shape)
    return embedding_vis, scale
