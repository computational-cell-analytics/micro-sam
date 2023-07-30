"""
Helper functions for downloading Segment Anything models and predicting image embeddings.
"""

import hashlib
import os
import warnings
from shutil import copyfileobj
from typing import Any, Callable, Dict, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import requests
import torch
import vigra
import zarr

from elf.io import open_file
from nifty.tools import blocking
from skimage.measure import regionprops

from segment_anything import sam_model_registry, SamPredictor

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    # preliminary finetuned models
    "vit_h_lm": "https://owncloud.gwdg.de/index.php/s/CnxBvsdGPN0TD3A/download",
    "vit_b_lm": "https://owncloud.gwdg.de/index.php/s/gGlR1LFsav0eQ2k/download",
}
_CHECKPOINT_FOLDER = os.environ.get("SAM_MODELS", os.path.expanduser("~/.sam_models"))
_CHECKSUMS = {
    "vit_h": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
    "vit_l": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
    "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
    # preliminary finetuned models
    "vit_h_lm": "c30a580e6ccaff2f4f0fbaf9cad10cee615a915cdd8c7bc4cb50ea9bdba3fc09",
    "vit_b_lm": "f2b8676f92a123f6f8ac998818118bd7269a559381ec60af4ac4be5c86024a1b",
}


# TODO define the proper type for image embeddings
ImageEmbeddings = Dict[str, Any]


def _download(url, path, model_type):
    with requests.get(url, stream=True, verify=True) as r:
        if r.status_code != 200:
            r.raise_for_status()
            raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
        file_size = int(r.headers.get("Content-Length", 0))
        desc = f"Download {url} to {path}"
        if file_size == 0:
            desc += " (unknown file size)"
        with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw, open(path, "wb") as f:
            copyfileobj(r_raw, f)

    # validate the checksum
    expected_checksum = _CHECKSUMS[model_type]
    if expected_checksum is None:
        return
    with open(path, "rb") as f:
        file_ = f.read()
        checksum = hashlib.sha256(file_).hexdigest()
    if checksum != expected_checksum:
        raise RuntimeError(
            "The checksum of the download does not match the expected checksum."
            f"Expected: {expected_checksum}, got: {checksum}"
        )
    print("Download successful and checksums agree.")


def _get_checkpoint(model_type, checkpoint_path=None):
    if checkpoint_path is None:
        checkpoint_url = _MODEL_URLS[model_type]
        checkpoint_name = checkpoint_url.split("/")[-1]
        checkpoint_path = os.path.join(_CHECKPOINT_FOLDER, checkpoint_name)

        # download the checkpoint if necessary
        if not os.path.exists(checkpoint_path):
            os.makedirs(_CHECKPOINT_FOLDER, exist_ok=True)
            _download(checkpoint_url, checkpoint_path, model_type)
    elif not os.path.exists(checkpoint_path):
        raise ValueError(f"The checkpoint path {checkpoint_path} that was passed does not exist.")

    return checkpoint_path


def get_sam_model(
    device: Optional[str] = None,
    model_type: str = "vit_h",
    checkpoint_path: Optional[str] = None,
    return_sam: bool = False
) -> SamPredictor:
    """Get the SegmentAnything Predictor.

    This function will download the required model checkpoint or load it from file if it
    was already downloaded. By default the models are downloaded to '~/.sam_models'.
    This location can be changed by setting the environment variable SAM_MODELS.

    Args:
        device: The device for the model. If none is given will use GPU if available.
        model_type: The SegmentAnything model to use.
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        return_sam: Return the sam model object as well as the predictor.

    Returns:
        The segment anything predictor.
    """
    checkpoint = _get_checkpoint(model_type, checkpoint_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Our custom model types have a suffix "_...". This suffix needs to be stripped
    # before calling sam_model_registry.
    model_type_ = model_type[:5]
    assert model_type_ in ("vit_h", "vit_b", "vit_l")

    sam = sam_model_registry[model_type_](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    if return_sam:
        return predictor, sam
    return predictor


def _to_image(input_):
    # we require the input to be uint8
    if input_.dtype != np.dtype("uint8"):
        # first normalize the input to [0, 1]
        input_ = input_.astype("float32") - input_.min()
        input_ = input_ / input_.max()
        # then bring to [0, 255] and cast to uint8
        input_ = (input_ * 255).astype("uint8")
    if input_.ndim == 2:
        image = np.concatenate([input_[..., None]] * 3, axis=-1)
    elif input_.ndim == 3 and input_.shape[-1] == 3:
        image = input_
    else:
        raise ValueError(f"Invalid input image of shape {input_.shape}. Expect either 2D grayscale or 3D RGB image.")
    return image


def _precompute_tiled_2d(predictor, input_, tile_shape, halo, f, verbose=True):
    tiling = blocking([0, 0], input_.shape[:2], tile_shape)
    n_tiles = tiling.numberOfBlocks

    f.attrs["input_size"] = None
    f.attrs["original_size"] = None

    features = f.require_group("features")
    features.attrs["shape"] = input_.shape[:2]
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    for tile_id in tqdm(range(n_tiles), total=n_tiles, desc="Predict image embeddings for tiles", disable=not verbose):
        tile = tiling.getBlockWithHalo(tile_id, list(halo))
        outer_tile = tuple(slice(beg, end) for beg, end in zip(tile.outerBlock.begin, tile.outerBlock.end))

        predictor.reset_image()
        tile_input = _to_image(input_[outer_tile])
        predictor.set_image(tile_input)
        tile_features = predictor.get_image_embedding()
        original_size = predictor.original_size
        input_size = predictor.input_size

        ds = features.create_dataset(
            str(tile_id), data=tile_features.cpu().numpy(), compression="gzip", chunks=tile_features.shape
        )
        ds.attrs["original_size"] = original_size
        ds.attrs["input_size"] = input_size

    return features


def _precompute_tiled_3d(predictor, input_, tile_shape, halo, f, verbose=True):
    assert input_.ndim == 3

    shape = input_.shape[1:]
    tiling = blocking([0, 0], shape, tile_shape)
    n_tiles = tiling.numberOfBlocks

    f.attrs["input_size"] = None
    f.attrs["original_size"] = None

    features = f.require_group("features")
    features.attrs["shape"] = shape
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    n_slices = input_.shape[0]
    pbar = tqdm(total=n_tiles * n_slices, desc="Predict image embeddings for tiles and slices", disable=not verbose)

    for tile_id in range(n_tiles):
        tile = tiling.getBlockWithHalo(tile_id, list(halo))
        outer_tile = tuple(slice(beg, end) for beg, end in zip(tile.outerBlock.begin, tile.outerBlock.end))

        ds = None
        for z in range(n_slices):
            predictor.reset_image()
            tile_input = _to_image(input_[z][outer_tile])
            predictor.set_image(tile_input)
            tile_features = predictor.get_image_embedding()

            if ds is None:
                shape = (input_.shape[0],) + tile_features.shape
                chunks = (1,) + tile_features.shape
                ds = features.create_dataset(
                    str(tile_id), shape=shape, dtype="float32", compression="gzip", chunks=chunks
                )

            ds[z] = tile_features.cpu().numpy()
            pbar.update(1)

        original_size = predictor.original_size
        input_size = predictor.input_size

        ds.attrs["original_size"] = original_size
        ds.attrs["input_size"] = input_size

    return features


def _compute_2d(input_, predictor):
    image = _to_image(input_)
    predictor.set_image(image)
    features = predictor.get_image_embedding()
    original_size = predictor.original_size
    input_size = predictor.input_size
    image_embeddings = {
        "features": features.cpu().numpy(), "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _precompute_2d(input_, predictor, save_path, tile_shape, halo):
    f = zarr.open(save_path, "a")

    use_tiled_prediction = tile_shape is not None
    if "input_size" in f.attrs:  # the embeddings have already been precomputed
        features = f["features"][:] if tile_shape is None else f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]

    elif use_tiled_prediction:  # the embeddings have not been computed yet and we use tiled prediction
        features = _precompute_tiled_2d(predictor, input_, tile_shape, halo, f)
        original_size, input_size = None, None

    else:  # the embeddings have not been computed yet and we use normal prediction
        image = _to_image(input_)
        predictor.set_image(image)
        features = predictor.get_image_embedding()
        original_size, input_size = predictor.original_size, predictor.input_size
        f.create_dataset("features", data=features.cpu().numpy(), chunks=features.shape)
        f.attrs["input_size"] = input_size
        f.attrs["original_size"] = original_size

    image_embeddings = {
        "features": features, "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _compute_3d(input_, predictor):
    features = []
    original_size, input_size = None, None

    for z_slice in tqdm(input_, desc="Precompute Image Embeddings"):
        predictor.reset_image()

        image = _to_image(z_slice)
        predictor.set_image(image)
        embedding = predictor.get_image_embedding()
        features.append(embedding[None])

        if original_size is None:
            original_size = predictor.original_size
        if input_size is None:
            input_size = predictor.input_size

    # concatenate across the z axis
    features = torch.cat(features)

    image_embeddings = {
        "features": features.cpu().numpy(), "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _precompute_3d(input_, predictor, save_path, lazy_loading, tile_shape=None, halo=None):
    f = zarr.open(save_path, "a")

    use_tiled_prediction = tile_shape is not None
    if "input_size" in f.attrs:  # the embeddings have already been precomputed
        features = f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]

    elif use_tiled_prediction:  # the embeddings have not been computed yet and we use tiled prediction
        features = _precompute_tiled_3d(predictor, input_, tile_shape, halo, f)
        original_size, input_size = None, None

    else:  # the embeddings have not been computed yet and we use normal prediction
        features = f["features"] if "features" in f else None
        original_size, input_size = None, None

        for z, z_slice in tqdm(enumerate(input_), total=input_.shape[0], desc="Precompute Image Embeddings"):
            if features is not None:
                emb = features[z]
                if np.count_nonzero(emb) != 0:
                    continue

            predictor.reset_image()
            image = _to_image(z_slice)
            predictor.set_image(image)
            embedding = predictor.get_image_embedding()

            original_size, input_size = predictor.original_size, predictor.input_size
            if features is None:
                shape = (input_.shape[0],) + embedding.shape
                chunks = (1,) + embedding.shape
                features = f.create_dataset("features", shape=shape, chunks=chunks, dtype="float32")
            features[z] = embedding.cpu().numpy()

        f.attrs["input_size"] = input_size
        f.attrs["original_size"] = original_size

    # we load the data into memory if lazy loading was not specified
    # and if we do not use tiled prediction (we cannot load the full tiled data structure into memory)
    if not lazy_loading and not use_tiled_prediction:
        features = features[:]

    image_embeddings = {
        "features": features, "input_size": input_size, "original_size": original_size,
    }
    return image_embeddings


def _compute_data_signature(input_):
    data_signature = hashlib.sha1(np.asarray(input_).tobytes()).hexdigest()
    return data_signature


def precompute_image_embeddings(
    predictor: SamPredictor,
    input_: np.ndarray,
    save_path: Optional[str] = None,
    lazy_loading: bool = False,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    wrong_file_callback: Optional[Callable] = None,
) -> ImageEmbeddings:
    """Compute the image embeddings (output of the encoder) for the input.

    If 'save_path' is given the embeddings will be loaded/saved in a zarr container.

    Args:
        predictor: The SegmentAnything predictor
        input_: The input data. Can be 2 or 3 dimensional, corresponding to an image, volume or timeseries.
        save_path: Path to save the embeddings in a zarr container.
        lazy_loading: Whether to load all embeddings into memory or return an
            object to load them on demand when required. This only has an effect if 'save_path' is given
            and if the input is 3 dimensional.
        ndim: The dimensionality of the data. If not given will be deduced from the input data.
        tile_shape: Shape of tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction.
        wrong_file_callback [callable]: Function to call when an embedding file with wrong file signature
            is passed. If none is given a wrong file signature will cause a warning.
            The callback ,ust have the signature 'def callback(save_path: str) -> str',
            where the return value is the (potentially updated) embedding save path.
    """
    ndim = input_.ndim if ndim is None else ndim
    if tile_shape is not None:
        assert save_path is not None, "Tiled prediction is only supported when the embeddings are saved to file."

    if save_path is not None:
        data_signature = _compute_data_signature(input_)

        f = zarr.open(save_path, "a")
        if "input_size" in f.attrs:  # we have computed the embeddings already

            # data signature does not match or is not in the file
            if "data_signature" not in f.attrs or f.attrs["data_signature"] != data_signature:
                warnings.warn("Embeddings file is invalid. Please recompute embeddings in a new file.")
                if wrong_file_callback is not None:
                    save_path = wrong_file_callback(save_path)
                f = zarr.open(save_path, "a")
                if "data_signature" not in f.attrs:
                    f.attrs["data_signature"] = data_signature

        else:  # embeddings have not yet been computed
            f.attrs["data_signature"] = data_signature

    if ndim == 2:
        image_embeddings = _compute_2d(input_, predictor) if save_path is None else\
            _precompute_2d(input_, predictor, save_path, tile_shape, halo)

    elif ndim == 3:
        image_embeddings = _compute_3d(input_, predictor) if save_path is None else\
            _precompute_3d(input_, predictor, save_path, lazy_loading, tile_shape, halo)

    else:
        raise ValueError(f"Invalid dimesionality {input_.ndim}, expect 2 or 3 dim data.")

    return image_embeddings


def set_precomputed(
    predictor: SamPredictor,
    image_embeddings: ImageEmbeddings,
    i: Optional[int] = None
):
    """Set the precomputed image embeddings for a predictor.

    Arguments:
        predictor: The SegmentAnything predictor.
        image_embeddings: The precomputed image embeddings computed by `precompute_image_embeddings`.
        i: Index for the image data. Required if `image` has three spatial dimensions
            or a time dimension and two spatial dimensions.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = image_embeddings["features"]

    assert features.ndim in (4, 5)
    if features.ndim == 5 and i is None:
        raise ValueError("The data is 3D so an index i is needed.")
    elif features.ndim == 4 and i is not None:
        raise ValueError("The data is 2D so an index is not needed.")

    if i is None:
        predictor.features = features.to(device) if torch.is_tensor(features) else \
            torch.from_numpy(features[:]).to(device)
    else:
        predictor.features = features[i].to(device) if torch.is_tensor(features) else \
            torch.from_numpy(features[i]).to(device)
    predictor.original_size = image_embeddings["original_size"]
    predictor.input_size = image_embeddings["input_size"]
    predictor.is_image_set = True

    return predictor


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the intersection over union of two masks.

    Args:
        mask1: The first mask.
        mask2: The second mask.

    Returns:
        The intersection over union of the two masks.
    """
    overlap = np.logical_and(mask1 == 1, mask2 == 1).sum()
    union = np.logical_or(mask1 == 1, mask2 == 1).sum()
    eps = 1e-7
    iou = float(overlap) / (float(union) + eps)
    return iou


def get_centers_and_bounding_boxes(
    segmentation: np.ndarray,
    mode: str = "v"
) -> Tuple[Dict[int, np.ndarray], Dict[int, tuple]]:
    """Returns the center coordinates of the foreground instances in the ground-truth.

    Args:
        segmentation: The segmentation.
        mode: Determines the functionality used for computing the centers.
        If 'v', the object's eccentricity centers computed by vigra are used.
        If 'p' the object's centroids computed by skimage are used.

    Returns:
        A dictionary that maps object ids to the corresponding centroid.
        A dictionary that maps object_ids to the corresponding bounding box.
    """
    assert mode in ["p", "v"], "Choose either 'p' for regionprops or 'v' for vigra"

    properties = regionprops(segmentation)

    if mode == "p":
        center_coordinates = {prop.label: prop.centroid for prop in properties}
    elif mode == "v":
        center_coordinates = vigra.filters.eccentricityCenters(segmentation.astype('float32'))
        center_coordinates = {i: coord for i, coord in enumerate(center_coordinates) if i > 0}

    bbox_coordinates = {prop.label: prop.bbox for prop in properties}

    assert len(bbox_coordinates) == len(center_coordinates)
    return center_coordinates, bbox_coordinates


def load_image_data(
    path: str,
    key: Optional[str] = None,
    lazy_loading: bool = False
) -> np.ndarray:
    """Helper function to load image data from file.

    Args:
        path: The filepath to the image data.
        key: The internal filepath for complex data formats like hdf5.
        lazy_loading: Whether to lazyly load data. Only supported for n5 and zarr data.

    Returns:
        The image data.
    """
    if key is None:
        image_data = imageio.imread(path)
    else:
        with open_file(path, mode="r") as f:
            image_data = f[key]
            if not lazy_loading:
                image_data = image_data[:]
    return image_data


def main():
    """@private"""
    import argparse

    parser = argparse.ArgumentParser(description="Compute the embeddings for an image.")
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-m", "--model_type", default="vit_h")
    parser.add_argument("-c", "--checkpoint_path", default=None)
    parser.add_argument("-k", "--key")
    args = parser.parse_args()

    predictor = get_sam_model(model_type=args.model_type, checkpoint_path=args.checkpoint_path)
    with open_file(args.input_path, mode="r") as f:
        data = f[args.key]
        precompute_image_embeddings(predictor, data, save_path=args.output_path)


if __name__ == "__main__":
    main()
