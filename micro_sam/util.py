"""
Helper functions for downloading Segment Anything models and predicting image embeddings.
"""

import hashlib
import os
from pathlib import Path
import pickle
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union

import imageio.v3 as imageio
import numpy as np
import pooch
import torch
import vigra
import zarr

from elf.io import open_file
from nifty.tools import blocking
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

try:
    from mobile_sam import sam_model_registry, SamPredictor
    VIT_T_SUPPORT = True
except ImportError:
    from segment_anything import sam_model_registry, SamPredictor
    VIT_T_SUPPORT = False

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm


# this is the default model used in micro_sam
# currently set to the default vit_h
_DEFAULT_MODEL = "vit_h"

# The valid model types. Each type corresponds to the architecture of the
# vision transformer used within SAM.
_MODEL_TYPES = ("vit_h", "vit_b", "vit_l", "vit_t")


# TODO define the proper type for image embeddings
ImageEmbeddings = Dict[str, Any]
"""@private"""


def get_cache_directory() -> None:
    """Get micro-sam cache directory location.

    Users can set the MICROSAM_CACHEDIR environment variable for a custom cache directory.
    """
    default_cache_directory = os.path.expanduser(pooch.os_cache('micro_sam'))
    cache_directory = Path(os.environ.get('MICROSAM_CACHEDIR', default_cache_directory))
    return cache_directory


#
# Functionality for model download and export
#

def microsam_cachedir():
    """Return the micro-sam cache directory.

    Returns the top level cache directory for micro-sam models and sample data.

    Every time this function is called, we check for any user updates made to
    the MICROSAM_CACHEDIR os environment variable since the last time.
    """
    cache_directory = os.environ.get('MICROSAM_CACHEDIR') or pooch.os_cache('micro_sam')
    return cache_directory


def models():
    """Return the segmentation models registry.

    We recreate the model registry every time this function is called,
    so any user changes to the default micro-sam cache directory location
    are respected.
    """
    models = pooch.create(
        path=os.path.join(microsam_cachedir(), "models"),
        base_url="",
        registry={
            # the default segment anything models
            "vit_h": "a7bf3b02f3ebf1267aba913ff637d9a2d5c33d3173bb679e46d9f338c26f262e",
            "vit_l": "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622",
            "vit_b": "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912",
            # the model with vit tiny backend fom https://github.com/ChaoningZhang/MobileSAM
            "vit_t": "6dbb90523a35330fedd7f1d3dfc66f995213d81b29a5ca8108dbcdd4e37d6c2f",
            # first version of finetuned models on zenodo
            "vit_h_lm": "9a65ee0cddc05a98d60469a12a058859c89dc3ea3ba39fed9b90d786253fbf26",
            "vit_b_lm": "5a59cc4064092d54cd4d92cd967e39168f3760905431e868e474d60fe5464ecd",
            "vit_h_em": "ae3798a0646c8df1d4db147998a2d37e402ff57d3aa4e571792fbb911d8a979c",
            "vit_b_em": "c04a714a4e14a110f0eec055a65f7409d54e6bf733164d2933a0ce556f7d6f81",
        },
        # Now specify custom URLs for some of the files in the registry.
        urls={
            # the default segment anything models
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            # the model with vit tiny backend fom https://github.com/ChaoningZhang/MobileSAM
            "vit_t": "https://owncloud.gwdg.de/index.php/s/TuDzuwVDHd1ZDnQ/download",
            # first version of finetuned models on zenodo
            "vit_h_lm": "https://zenodo.org/record/8250299/files/vit_h_lm.pth?download=1",
            "vit_b_lm": "https://zenodo.org/record/8250281/files/vit_b_lm.pth?download=1",
            "vit_h_em": "https://zenodo.org/record/8250291/files/vit_h_em.pth?download=1",
            "vit_b_em": "https://zenodo.org/record/8250260/files/vit_b_em.pth?download=1",
        },
    )
    return models


def _get_default_device():
    # check that we're in CI and use the CPU if we are
    # otherwise the tests may run out of memory on MAC if MPS is used.
    if os.getenv("GITHUB_ACTIONS") == "true":
        return "cpu"
    # Use cuda enabled gpu if it's available.
    if torch.cuda.is_available():
        device = "cuda"
    # As second priority use mps.
    # See https://pytorch.org/docs/stable/notes/mps.html for details
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("Using apple MPS device.")
        device = "mps"
    # Use the CPU as fallback.
    else:
        device = "cpu"
    return device


def get_device(device: Optional[Union[str, torch.device]] = None) -> Union[str, torch.device]:
    """Get the torch device.

    If no device is passed the default device for your system is used.
    Else it will be checked if the device you have passed is supported.

    Args:
        device: The input device.

    Returns:
        The device.
    """
    if device is None or device == "auto":
        device = _get_default_device()
    else:
        device_type = device if isinstance(device, str) else device.type
        if device_type.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA backend is not available.")
        elif device_type.lower() == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("PyTorch MPS backend is not available or is not built correctly.")
        elif device_type.lower() == "cpu":
            pass  # cpu is always available
        else:
            raise RuntimeError(f"Unsupported device: {device}\n"
                               "Please choose from 'cpu', 'cuda', or 'mps'.")
    return device


def _available_devices():
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = get_device(i)
        except RuntimeError:
            pass
        else:
            available_devices.append(device)
    return available_devices


def get_sam_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    return_sam: bool = False,
) -> SamPredictor:
    r"""Get the SegmentAnything Predictor.

    This function will download the required model or load it from the cached weight file.
    This location of the cache can be changed by setting the environment variable: MICROSAM_CACHEDIR.
    The name of the requested model can be set via `model_type`.
    See https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#finetuned-models
    for an overview of the available models

    Alternatively this function can also load a model from weights stored in a local filepath.
    The corresponding file path is given via `checkpoint_path`. In this case `model_type`
    must be given as the matching encoder architecture, e.g. "vit_b" if the weights are for
    a SAM model with vit_b encoder.

    By default the models are downloaded to a folder named 'micro_sam/models'
    inside your default cache directory, eg:
    * Mac: ~/Library/Caches/<AppName>
    * Unix: ~/.cache/<AppName> or the value of the XDG_CACHE_HOME environment variable, if defined.
    * Windows: C:\Users\<user>\AppData\Local\<AppAuthor>\<AppName>\Cache
    See the pooch.os_cache() documentation for more details:
    https://www.fatiando.org/pooch/latest/api/generated/pooch.os_cache.html

    Args:
        model_type: The SegmentAnything model to use. Will use the standard vit_h model by default.
            To get a list of all available model names you can call `get_model_names`.
        device: The device for the model. If none is given will use GPU if available.
        checkpoint_path: The path to a file with weights that should be used instead of using the
            weights corresponding to `model_type`. If given, `model_type` must match the architecture
            corresponding to the weight file. E.g. if you use weights for SAM with vit_b encoder
            then `model_type` must be given as "vit_b".
        return_sam: Return the sam model object as well as the predictor.

    Returns:
        The segment anything predictor.
    """
    device = get_device(device)

    # We support passing a local filepath to a checkpoint.
    # In this case we do not download any weights but just use the local weight file,
    # as it is, without copying it over anywhere or checking it's hashes.

    # checkpoint_path has not been passed, we download a known model and derive the correct
    # URL from the model_type. If the model_type is invalid pooch will raise an error.
    if checkpoint_path is None:
        model_registry = models()
        checkpoint = model_registry.fetch(model_type)
    # checkpoint_path has been passed, we use it instead of downloading a model.
    else:
        # Check if the file exists and raise an error otherwise.
        # We can't check any hashes here, and we don't check if the file is actually a valid weight file.
        # (If it isn't the model creation will fail below.)
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint at {checkpoint_path} could not be found.")
        checkpoint = checkpoint_path

    # Our fine-tuned model types have a suffix "_...". This suffix needs to be stripped
    # before calling sam_model_registry.
    abbreviated_model_type = model_type[:5]
    if abbreviated_model_type not in _MODEL_TYPES:
        raise ValueError(f"Invalid model_type: {abbreviated_model_type}. Expect one of {_MODEL_TYPES}")
    if abbreviated_model_type == "vit_t" and not VIT_T_SUPPORT:
        raise RuntimeError(
            "mobile_sam is required for the vit-tiny."
            "You can install it via 'pip install git+https://github.com/ChaoningZhang/MobileSAM.git'"
        )

    sam = sam_model_registry[abbreviated_model_type](checkpoint=checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.model_type = abbreviated_model_type
    if return_sam:
        return predictor, sam
    return predictor


# We write a custom unpickler that skips objects that cannot be found instead of
# throwing an AttributeError or ModueNotFoundError.
# NOTE: since we just want to unpickle the model to load its weights these errors don't matter.
# See also https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules
class _CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError) as e:
            warnings.warn(f"Did not find {module}:{name} and will skip it, due to error {e}")
            return None


def get_custom_sam_model(
    checkpoint_path: Union[str, os.PathLike],
    model_type: str = "vit_h",
    device: Optional[Union[str, torch.device]] = None,
    return_sam: bool = False,
    return_state: bool = False,
) -> SamPredictor:
    """Load a SAM model from a torch_em checkpoint.

    This function enables loading from the checkpoints saved by
    the functionality in `micro_sam.training`.

    Args:
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        model_type: The SegmentAnything model_type for the given checkpoint.
        device: The device for the model. If none is given will use GPU if available.
        return_sam: Return the sam model object as well as the predictor.
        return_state: Return the full state of the checkpoint in addition to the predictor.

    Returns:
        The segment anything predictor.
    """
    assert not (return_sam and return_state)

    # over-ride the unpickler with our custom one
    custom_pickle = pickle
    custom_pickle.Unpickler = _CustomUnpickler

    device = get_device(device)
    sam = sam_model_registry[model_type]()

    # load the model state, ignoring any attributes that can't be found by pickle
    state = torch.load(checkpoint_path, map_location=device, pickle_module=custom_pickle)
    model_state = state["model_state"]

    # copy the model weights from torch_em's training format
    sam_prefix = "sam."
    model_state = OrderedDict(
        [(k[len(sam_prefix):] if k.startswith(sam_prefix) else k, v) for k, v in model_state.items()]
    )
    sam.load_state_dict(model_state)
    sam.to(device)

    predictor = SamPredictor(sam)
    predictor.model_type = model_type

    if return_sam:
        return predictor, sam
    if return_state:
        return predictor, state
    return predictor


def export_custom_sam_model(
    checkpoint_path: Union[str, os.PathLike],
    model_type: str,
    save_path: Union[str, os.PathLike],
) -> None:
    """Export a finetuned segment anything model to the standard model format.

    The exported model can be used by the interactive annotation tools in `micro_sam.annotator`.

    Args:
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        model_type: The SegmentAnything model type corresponding to the checkpoint (vit_h, vit_b, vit_l or vit_t).
        save_path: Where to save the exported model.
    """
    _, state = get_custom_sam_model(
        checkpoint_path, model_type=model_type, return_state=True, device="cpu",
    )
    model_state = state["model_state"]
    prefix = "sam."
    model_state = OrderedDict(
        [(k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in model_state.items()]
    )
    torch.save(model_state, save_path)


def get_model_names() -> Iterable:
    model_registry = models()
    model_names = model_registry.registry.keys()
    return model_names


#
# Functionality for precomputing embeddings and other state
#


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
        predictor: The SegmentAnything predictor.
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
        save_path = str(save_path)
        data_signature = _compute_data_signature(input_)

        f = zarr.open(save_path, "a")
        key_vals = [
            ("data_signature", data_signature),
            ("tile_shape", tile_shape if tile_shape is None else list(tile_shape)),
            ("halo", halo if halo is None else list(halo)),
            ("model_type", predictor.model_type)
        ]
        if "input_size" in f.attrs:  # we have computed the embeddings already and perform checks
            for key, val in key_vals:
                if val is None:
                    continue
                # check whether the key signature does not match or is not in the file
                if key not in f.attrs or f.attrs[key] != val:
                    raise RuntimeError(
                        f"Embeddings file {save_path} is invalid due to unmatching {key}: "
                        f"{f.attrs.get(key)} != {val}.Please recompute embeddings in a new file."
                    )
                    if wrong_file_callback is not None:
                        save_path = wrong_file_callback(save_path)
                        f = zarr.open(save_path, "a")
                    break

        for key, val in key_vals:
            if key not in f.attrs:
                f.attrs[key] = val

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
    device = predictor.device
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


#
# Misc functionality
#


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

    assert len(bbox_coordinates) == len(center_coordinates), f"{len(bbox_coordinates)}, {len(center_coordinates)}"
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


def segmentation_to_one_hot(
    segmentation: np.ndarray,
    segmentation_ids: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """Convert the segmentation to one-hot encoded masks.

    Args:
        segmentation: The segmentation.
        segmentation_ids: Optional subset of ids that will be used to subsample the masks.

    Returns:
        The one-hot encoded masks.
    """
    masks = segmentation.copy()
    if segmentation_ids is None:
        n_ids = int(segmentation.max())

    else:
        assert segmentation_ids[0] != 0

        # the segmentation ids have to be sorted
        segmentation_ids = np.sort(segmentation_ids)

        # set the non selected objects to zero and relabel sequentially
        masks[~np.isin(masks, segmentation_ids)] = 0
        masks = relabel_sequential(masks)[0]
        n_ids = len(segmentation_ids)

    masks = torch.from_numpy(masks)

    one_hot_shape = (n_ids + 1,) + masks.shape
    masks = masks.unsqueeze(0)  # add dimension to scatter
    masks = torch.zeros(one_hot_shape).scatter_(0, masks, 1)[1:]

    # add the extra singleton dimenion to get shape NUM_OBJECTS x 1 x H x W
    masks = masks.unsqueeze(1)
    return masks
