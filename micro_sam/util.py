"""Helper functions for downloading Segment Anything models and predicting image embeddings.
"""

import os
import json
import uuid
import pooch
import atexit
import pickle
import shutil
import xxhash
import hashlib
import warnings
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import z5py
import zarr
import numpy as np
import imageio.v3 as imageio
import segment_anything.utils.amg as amg_utils

from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

import torch
from torchvision.ops.boxes import batched_nms

import elf.parallel as parallel_impl
from elf.io import open_file

from bioimage_cpp.distance import distance_transform
from bioimage_cpp.segmentation import relabel_sequential

from .__version__ import __version__

try:
    # Avoid import warnigns from mobile_sam
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from mobile_sam import sam_model_registry, SamPredictor  # noqa
    VIT_T_SUPPORT = True
except ImportError:
    from segment_anything import sam_model_registry, SamPredictor  # noqa
    VIT_T_SUPPORT = False

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm

# This is the default model used in micro_sam
# Currently it is set to vit_b_lm
_DEFAULT_MODEL = "vit_b_lm"

# The valid model types. Each type corresponds to the architecture of the
# vision transformer used within SAM.
_MODEL_TYPES = ("vit_l", "vit_b", "vit_h", "vit_t")


ImageEmbeddings = Dict[str, Any]
"""@private"""


def get_cache_directory() -> None:
    """Get micro-sam cache directory location.

    Users can set the MICROSAM_CACHEDIR environment variable for a custom cache directory.
    """
    default_cache_directory = os.path.expanduser(pooch.os_cache("micro_sam"))
    cache_directory = Path(os.environ.get("MICROSAM_CACHEDIR", default_cache_directory))
    return cache_directory


def make_temp_embedding_path() -> str:
    """Create a fresh ephemeral on-disk zarr path for streaming image embeddings.

    Used when no explicit embedding save path is given: caching to disk (under the micro-sam cache
    directory, honoring MICROSAM_CACHEDIR) instead of holding the whole volume in RAM keeps memory
    bounded on large volumes / tiled images. The cache is disk-backed rather than in /tmp, which is
    tmpfs (RAM) on many systems. The caller owns eager cleanup; the returned path is also removed on
    process exit.
    """
    parent = get_cache_directory() / "tmp_embeddings"
    parent.mkdir(parents=True, exist_ok=True)
    path = str(parent / f"{uuid.uuid4().hex}.zarr")
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


#
# Functionality for model download and export
#


def microsam_cachedir() -> None:
    """Return the micro-sam cache directory.

    Returns the top level cache directory for micro-sam models and sample data.

    Every time this function is called, we check for any user updates made to
    the MICROSAM_CACHEDIR os environment variable since the last time.
    """
    cache_directory = os.environ.get("MICROSAM_CACHEDIR") or pooch.os_cache("micro_sam")
    return cache_directory


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


def device_type(device: Union[str, torch.device]) -> str:
    """Get the device type ('cpu', 'cuda' or 'mps'), ignoring any device index.

    Torch reports accelerators with an index (e.g. 'mps:0'), so comparing 'str(device)' against
    'mps' or 'cuda' silently fails. Compare against this instead.

    Args:
        device: The device, as a string or torch.device.

    Returns:
        The device type.
    """
    return torch.device(device).type


def _configure_mps_memory(device: Union[str, torch.device]) -> None:
    """Disable the MPS memory watermark so 3d automatic segmentation does not hit a premature OOM.

    MPS's default watermark rejects allocations that would still fit in unified memory. '0.0' disables
    it. We set it only when unset (so a user-provided value is kept) and it must run before the first
    MPS allocation to apply.
    """
    try:
        is_mps = device_type(device) == "mps"
    except (RuntimeError, TypeError):
        is_mps = False
    if is_mps and "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ:
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        print("Lifted the MPS memory limit for large 3d segmentation. This can use swap on low-memory Macs.")


def get_device(device: Optional[Union[str, torch.device]] = None) -> Union[str, torch.device]:
    """Get the torch device.

    If no device is passed the default device for your system is used.
    Else it will be checked if the device you have passed is supported.

    Args:
        device: The input device. By default, selects the best available device supports.

    Returns:
        The device.
    """
    if device is None or device == "auto":
        device = _get_default_device()
    else:
        try:
            requested_type = device_type(device)
        except (RuntimeError, TypeError) as e:
            raise RuntimeError(
                f"Unsupported device: '{device}'. Please choose from 'cpu', 'cuda', or 'mps'."
            ) from e

        if requested_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA backend is not available.")
        elif requested_type == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("PyTorch MPS backend is not available or is not built correctly.")
        elif requested_type == "cpu":
            pass  # cpu is always available
        else:
            raise RuntimeError(f"Unsupported device: '{device}'. Please choose from 'cpu', 'cuda', or 'mps'.")

    _configure_mps_memory(device)
    return device


def get_embedding_function(model_type: str) -> callable:
    """Get the precompute-embeddings function for the model family of `model_type`.

    Dispatches across the three families: VFM (DINO / UNI) encoders, SAM2 ('hvit_*') and SAM1 ('vit_*').
    All returned functions share the interface (predictor, input_, save_path, ndim, tile_shape, halo,
    verbose, lazy_loading, pbar_init, pbar_update).

    Args:
        model_type: The model name, e.g. 'vit_b_lm', 'hvit_t' or 'vit_b_dinov2'.

    Returns:
        The matching `precompute_image_embeddings` / `precompute_vfm_embeddings` function.

    Raises:
        ValueError: If `model_type` does not belong to any supported model family.
    """
    from .models.vfm import is_vfm_model, get_vfm_model_names
    from .v2.util import SUPPORTED_MODELS as sam2_backbones

    if is_vfm_model(model_type):
        from .models.vfm import precompute_vfm_embeddings
        return precompute_vfm_embeddings

    # Finetuned names keep their backbone prefix ('vit_b_lm' -> 'vit_b', 'hvit_t_cells' -> 'hvit_t').
    if isinstance(model_type, str) and model_type[:6] in sam2_backbones:
        from .v2.util import precompute_image_embeddings
        return precompute_image_embeddings
    if isinstance(model_type, str) and model_type[:5] in _MODEL_TYPES:
        from .v1.util import precompute_image_embeddings
        return precompute_image_embeddings

    raise ValueError(
        f"Invalid model_type: '{model_type}'. Expected a SAM1 model (backbone one of {list(_MODEL_TYPES)}), "
        f"a SAM2 model (backbone one of {sam2_backbones}) or a VFM encoder (one of {list(get_vfm_model_names())}). "
        "Finetuned models keep their backbone prefix, e.g. 'vit_b_lm' or 'hvit_t_cells'. "
        "Run 'micro_sam info' to list all available models."
    )


# TODO: refactor this once we decide which models to support.
# (Likely only SAM2 models)
def _get_sam_model(model_type, ndim, device, checkpoint_path, decoder_path, use_cli):
    """Build the predictor for a model name, dispatching across the VFM, SAM2 and SAM1 families.

    This lives here rather than next to the annotator state that first needed it, so that the CLI and
    the automatic-segmentation entry points can load a model without importing napari and the Qt
    widgets. Every family is imported inside its own branch, so loading a SAM2 model does not pull in
    SAM1 or the training stack either.
    """
    from .models.vfm import is_vfm_model, get_vfm_model
    if is_vfm_model(model_type):  # VFM encoders (DINO / UNI) for the classification tools.
        encoder = get_vfm_model(model_type, device=device, checkpoint_path=checkpoint_path)
        return encoder, {}

    if model_type.startswith("hvit"):  # i.e. SAM2 models.
        from .v2.util import get_sam2_image_predictor, get_sam2_model

        # 'device=None' lets 'get_sam2_model' auto-detect the best device (cuda > mps > cpu);
        # an explicit device (e.g. from the '--device' CLI argument) is forwarded and honored.
        if ndim == 2:  # Get the SAM2 model and prepare the image predictor.
            model = get_sam2_model(model_type=model_type, input_type="images", device=device)
            # Use the shared resize-longest predictor.
            predictor = get_sam2_image_predictor(model)
            # 'get_sam2_model' sets these on the video predictor. Set them here on the image
            # predictor too, so the tool can write the embedding signature when it caches embeddings.
            predictor.model_type = model_type
            predictor.model_name = model_type
        elif ndim == 3:  # Get SAM2 video predictor
            predictor = get_sam2_model(model_type=model_type, input_type="videos", device=device)
        else:
            raise ValueError
        state = {}

    else:
        from .v1.util import get_sam_model

        def progress_bar_factory(model_type):
            pbar = tqdm(desc=f"Downloading '{model_type}'. This may take a while")
            return pbar

        predictor, state = get_sam_model(
            device=device, model_type=model_type,
            checkpoint_path=checkpoint_path, decoder_path=decoder_path, return_state=True,
            progress_bar_factory=None if use_cli else progress_bar_factory,
        )

    return predictor, state


def _available_devices():
    """List the devices that can be selected explicitly, e.g. in the annotator's device dropdown.

    Every visible GPU is listed by its index when there is more than one, so that a multi-GPU user can
    choose which GPU to run on. Using all of them is what the annotator's 'auto' entry does.
    """
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = get_device(i)
        except RuntimeError:
            continue

        if device == "cuda" and torch.cuda.device_count() > 1:
            available_devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
        else:
            available_devices.append(device)
    return available_devices


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


def _compute_hash(path, chunk_size=8192):
    hash_obj = xxhash.xxh128()
    with open(path, "rb") as f:
        chunk = f.read(chunk_size)
        while chunk:
            hash_obj.update(chunk)
            chunk = f.read(chunk_size)
    hash_val = hash_obj.hexdigest()
    return f"xxh128:{hash_val}"


# Load the state from a checkpoint.
# The checkpoint can either contain a sam encoder state
# or it can be a checkpoint for model finetuning.
def _load_checkpoint(checkpoint_path):
    # Over-ride the unpickler with our custom one.
    # This enables imports from torch_em checkpoints even if it cannot be fully unpickled.
    custom_pickle = pickle
    custom_pickle.Unpickler = _CustomUnpickler

    state = torch.load(checkpoint_path, map_location="cpu", pickle_module=custom_pickle)
    if "model_state" in state:
        # Copy the model weights from torch_em's training format.
        model_state = state["model_state"]
        sam_prefix = "sam."
        model_state = OrderedDict(
            [(k[len(sam_prefix):] if k.startswith(sam_prefix) else k, v) for k, v in model_state.items()]
        )
    else:
        model_state = state

    return state, model_state


#
# Functionality for precomputing image embeddings.
#


def _ensure_rgb(image):
    """Map a 2D or channel-last image to a 3-channel (H, W, 3) array without normalizing."""
    ndim = image.ndim
    n_channels = 1 if ndim == 2 else image.shape[-1]

    if ndim == 2:  # Grayscale image -> replicate channels.
        image = np.concatenate([image[..., None]] * 3, axis=-1)
    elif ndim == 3 and n_channels == 1:  # Grayscale image -> replicate channels.
        image = np.concatenate([image] * 3, axis=-1)
    elif ndim == 3 and n_channels == 2:  # Two channels -> add a zero channel.
        zero_channel = np.zeros(image.shape[:2] + (1,), dtype=image.dtype)
        image = np.concatenate([image, zero_channel], axis=-1)
    elif ndim == 3 and n_channels == 3:  # RGB input -> do nothing.
        pass
    elif ndim == 3 and n_channels > 3:  # More than three channels -> select first three.
        warnings.warn(f"You provided an input with {n_channels} channels. Only the first three will be used.")
        image = image[..., :3]
    else:
        raise ValueError(
            f"Invalid input dimensionality {ndim}. Expect either a 2D input (=grayscale image) "
            "or a 3D input (= image with channels)."
        )

    assert image.ndim == 3 and image.shape[-1] == 3
    return image


def _to_image(image):
    # Map to three channels, then normalize per channel and bring it to uint8.
    input_ = _ensure_rgb(image).astype("float32")
    input_ -= input_.min(axis=(0, 1))[None, None]
    input_ /= (input_.max(axis=(0, 1))[None, None] + 1e-7)
    input_ = (input_ * 255).astype("uint8")

    # Explicitly return a numpy array for compatibility with torchvision
    # because the input_ array could be something like dask array.
    return np.array(input_)


# The zarr format used when writing new on-disk embedding caches. Reading auto-detects the format,
# so existing caches (v2 or v3) still load. This only controls newly created containers.
EMBEDDING_ZARR_FORMAT = 3
# Compression codec for on-disk embeddings. blosc (byte-shuffle + lz4) is the fastest to read and write
# and the smallest for float32 features. It is also z5py's default, so this just pins it explicitly.
EMBEDDING_COMPRESSION = "blosc"
# Upper bound on how much of the input is materialized at once when hashing it for the data signature.
DATA_SIGNATURE_BLOCK_SIZE = 64 * 1024**2


def _open_embeddings(save_path, mode="a"):
    """Open the container for image embeddings.

    On-disk embeddings use z5py, whose C++ core reads and writes zarr much faster than zarr-python.
    Reading auto-detects the zarr format, so existing caches (v2 and v3) still load; new caches use
    the format given by ``EMBEDDING_ZARR_FORMAT``. In-memory embeddings (``save_path=None``) use an
    in-memory zarr group, which z5py does not support.
    """
    if save_path is None:
        return zarr.group()
    save_path = str(save_path)
    # Unlike zarr-python, z5py cannot open an existing directory that has no zarr metadata
    # (e.g. an empty folder left by a previous run). When appending, write the group metadata
    # for the pinned format first so the container opens as an empty group, as zarr-python did.
    if mode == "a" and os.path.isdir(save_path):
        has_metadata = any(
            os.path.exists(os.path.join(save_path, name)) for name in (".zgroup", ".zarray", "zarr.json")
        )
        if not has_metadata:
            if EMBEDDING_ZARR_FORMAT == 2:
                meta_name, meta = ".zgroup", {"zarr_format": 2}
            else:
                meta_name, meta = "zarr.json", {"zarr_format": 3, "node_type": "group", "attributes": {}}
            with open(os.path.join(save_path, meta_name), "w") as f:
                json.dump(meta, f)
    return z5py.ZarrFile(save_path, mode=mode, zarr_format=EMBEDDING_ZARR_FORMAT)


def _create_dataset_with_data(group, name, data, chunks=None):
    if chunks is None:
        chunks = data.shape
    # z5py exposes the h5py-style create_dataset for both zarr v2 and v3.
    if isinstance(group, z5py.Group):
        return group.create_dataset(name, data=data, shape=data.shape, chunks=chunks, compression=EMBEDDING_COMPRESSION)
    # In-memory zarr group (only used when no save_path is given).
    zarr_major_version = int(zarr.__version__.split(".")[0])
    if zarr_major_version == 2:
        ds = group.create_dataset(name, data=data, shape=data.shape, chunks=chunks)
    elif zarr_major_version == 3:
        ds = group.create_array(name, shape=data.shape, chunks=chunks, dtype=data.dtype)
        ds[:] = data
    else:
        raise RuntimeError(f"Unsupported zarr version: {zarr_major_version}")
    return ds


def _create_dataset_without_data(group, name, shape, dtype, chunks):
    if isinstance(group, z5py.Group):
        return group.create_dataset(
            name, shape=shape, dtype=dtype, chunks=chunks, compression=EMBEDDING_COMPRESSION
        )
    # In-memory zarr group (only used when no save_path is given).
    zarr_major_version = int(zarr.__version__.split(".")[0])
    if zarr_major_version == 2:
        ds = group.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks)
    elif zarr_major_version == 3:
        ds = group.create_array(name, shape=shape, chunks=chunks, dtype=dtype)
    else:
        raise RuntimeError(f"Unsupported zarr version: {zarr_major_version}")
    return ds


def _compute_data_signature(input_):
    # Hash the input in blocks along the leading axis, so a lazy input (dask / zarr / h5py) is never
    # materialized as a whole. The digest is unchanged: it is the same byte stream, fed in parts.
    shape = tuple(getattr(input_, "shape", ()))
    itemsize = getattr(getattr(input_, "dtype", None), "itemsize", None)
    signature = hashlib.sha1()
    if len(shape) == 0 or itemsize is None:
        signature.update(np.asarray(input_).tobytes())
        return signature.hexdigest()

    bytes_per_slice = itemsize * int(np.prod(shape[1:], dtype="int64"))
    block = max(1, DATA_SIGNATURE_BLOCK_SIZE // max(bytes_per_slice, 1))
    for start in range(0, shape[0], block):
        signature.update(np.asarray(input_[start:start + block]).tobytes())
    return signature.hexdigest()


# Create all metadata that is stored along with the embeddings.
def _get_embedding_signature(input_, predictor, tile_shape, halo, data_signature=None):
    if data_signature is None:
        data_signature = _compute_data_signature(input_)

    signature = {
        "data_signature": data_signature,
        "tile_shape": tile_shape if tile_shape is None else list(tile_shape),
        "halo": halo if halo is None else list(halo),
        "model_type": predictor.model_type,
        "model_name": predictor.model_name,
        "micro_sam_version": __version__,
        "model_hash": getattr(predictor, "_hash", None),
    }
    return signature


# Note: the input size and orginal size are different if embeddings are tiled or not.
# That's why we do not include them in the main signature that is being checked
# (_get_embedding_signature), but just add it for serialization here.
def _write_embedding_signature(f, input_, predictor, tile_shape, halo, input_size, original_size):
    signature = _get_embedding_signature(input_, predictor, tile_shape, halo)
    signature.update({"input_size": input_size, "original_size": original_size})
    for key, val in signature.items():
        f.attrs[key] = val


# Helper function for optional external progress bars.
def handle_pbar(verbose, pbar_init, pbar_update):
    """@private"""

    # Noop to provide dummy functions.
    def noop(*args):
        pass

    if verbose and pbar_init is None:  # we are verbose and don't have an external progress bar.
        assert pbar_update is None  # avoid inconsistent state of callbacks

        # Create our own progress bar and callbacks
        pbar = tqdm()

        def pbar_init(total, description):
            pbar.total = total
            pbar.set_description(description)

        def pbar_update(update):
            pbar.update(update)

        def pbar_close():
            pbar.close()

    elif verbose and pbar_init is not None:  # external pbar -> we don't have to do anything
        assert pbar_update is not None
        pbar = None
        pbar_close = noop

    else:  # we are not verbose, do nothing
        pbar = None
        pbar_init, pbar_update, pbar_close = noop, noop, noop

    return pbar, pbar_init, pbar_update, pbar_close


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
    segmentation: np.ndarray, mode: str = "v"
) -> Tuple[Dict[int, np.ndarray], Dict[int, tuple]]:
    """Returns the center coordinates of the foreground instances in the ground-truth.

    Args:
        segmentation: The segmentation.
        mode: Determines the functionality used for computing the centers.
            If 'v', the point of maximal distance to the object boundary is used as center.
            This center is guaranteed to lie inside the object, also for concave shapes.
            If 'p' the object's centroids computed by skimage are used.

    Returns:
        A dictionary that maps object ids to the corresponding centroid.
        A dictionary that maps object_ids to the corresponding bounding box.
    """
    assert mode in ["p", "v"], "Choose either 'p' for regionprops centroids or 'v' for distance-based centers"

    properties = regionprops(segmentation)

    if mode == "p":
        center_coordinates = {prop.label: prop.centroid for prop in properties}
    elif mode == "v":
        # Use the point of maximal distance to the object boundary as the center.
        # In contrast to the centroid, this point is guaranteed to lie inside the object,
        # also for concave shapes. This replaces vigra.filters.eccentricityCenters.
        # Compute the boundaries and a single distance transform for the whole
        # segmentation, instead of one distance transform per object.
        ndim = segmentation.ndim
        # Pad so objects touching the image border also get a boundary there,
        # matching a per-object padded distance transform.
        padded = np.pad(segmentation, 1)
        boundaries = find_boundaries(padded, mode="inner")
        distances = distance_transform(boundaries == 0)

        center_coordinates = {}
        for prop in properties:
            bbox = prop.bbox
            # Slice the global distance field to this object's bbox (shifted by the
            # pad of 1) and restrict the argmax to the object's own pixels.
            region = distances[tuple(slice(b + 1, e + 1) for b, e in zip(bbox[:ndim], bbox[ndim:]))]
            masked = np.where(prop.image, region, -1.0)
            center_local = np.unravel_index(int(np.argmax(masked)), masked.shape)
            center_coordinates[prop.label] = tuple(int(c + o) for c, o in zip(center_local, bbox[:ndim]))

    bbox_coordinates = {prop.label: prop.bbox for prop in properties}

    assert len(bbox_coordinates) == len(center_coordinates), f"{len(bbox_coordinates)}, {len(center_coordinates)}"
    return center_coordinates, bbox_coordinates


def load_image_data(path: str, key: Optional[str] = None, lazy_loading: bool = False) -> np.ndarray:
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


def segmentation_to_one_hot(segmentation: np.ndarray, segmentation_ids: Optional[np.ndarray] = None) -> torch.Tensor:
    """Convert the segmentation to one-hot encoded masks.

    Args:
        segmentation: The segmentation.
        segmentation_ids: Optional subset of ids that will be used to subsample the masks.
            By default, computes the number of ids from the provided `segmentation` masks.

    Returns:
        The one-hot encoded masks.
    """
    masks = segmentation.copy()
    if segmentation_ids is None:
        n_ids = int(segmentation.max())

    else:
        msg = "No foreground objects were found."
        if len(segmentation_ids) == 0:  # The list should not be completely empty.
            raise RuntimeError(msg)

        if 0 in segmentation_ids:  # The list should not have 'zero' as a value.
            raise RuntimeError(msg)

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

    # add the extra singleton dimension to get shape NUM_OBJECTS x 1 x H x W
    masks = masks.unsqueeze(1)
    return masks


def get_block_shape(shape: Tuple[int]) -> Tuple[int]:
    """Get a suitable block shape for chunking a given shape.

    The primary use for this is determining chunk sizes for
    zarr arrays or block shapes for parallelization.

    Args:
        shape: The image or volume shape.

    Returns:
        The block shape.
    """
    ndim = len(shape)
    if ndim == 2:
        block_shape = tuple(min(bs, sh) for bs, sh in zip((1024, 1024), shape))
    elif ndim == 3:
        block_shape = tuple(min(bs, sh) for bs, sh in zip((32, 256, 256), shape))
    else:
        raise ValueError(f"Only 2 or 3 dimensional shapes are supported, got {ndim}D.")

    return block_shape


def micro_sam_info(download: Optional[List[str]] = None) -> None:
    """Display μSAM information using a rich console.

    Args:
        download: Optional list of pretrained models to download by name (SAM1, SAM2 or their finetuned
            variants). E.g. ['vit_b_lm', 'hvit_t'] downloads the listed models; ['all'] downloads every
            available model.
    """
    import psutil
    import platform
    from .v1.util import models, _download_sam_model
    from .v2.util import SUPPORTED_MODELS, get_model_names, _get_checkpoint, _download_finetuned_sam2_model
    from rich import progress
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console

    import torch
    import micro_sam

    # Open up a new console.
    console = Console()

    # The header for information CLI.
    console.print("[bold #0072B2]μSAM Information Booth[/bold #0072B2]", justify="center")
    console.print("-" * console.width)

    # μSAM version panel.
    console.print(
        Panel(f"[bold #F0E442]Version:[/bold #F0E442] {micro_sam.__version__}", title="μSAM Version", expand=True)
    )

    # The documentation link panel.
    console.print(
        Panel(
            "[bold #CC79A7]Tools documented at:[/bold #CC79A7]\n"
            "https://computational-cell-analytics.github.io/micro-sam", title="Documentation"
        )
    )

    # The publication panel.
    console.print(
        Panel(
            "[bold #E69F00]Published in Nature Methods:[/bold #E69F00]\n"
            "https://www.nature.com/articles/s41592-024-02580-4", title="Publication"
        )
    )

    # Creating a cache directory when users' run `micro_sam.info`.
    cache_dir = get_cache_directory()
    os.makedirs(cache_dir, exist_ok=True)

    # The cache directory panel.
    console.print(
        Panel(f"[bold #009E73]Cache Directory:[/bold #009E73]\n{cache_dir}", title="Cache Directory")
    )

    # SAM1 models. 'sam1_display' holds the labeled names shown in the panel (the '(v2/v3/v4)' suffixes
    # refer to the BioImageIO ModelZoo upload version, not the SAM version); 'sam1_names' holds the bare
    # names accepted by the downloader.
    sam1_display, sam1_names = [], []
    for model_name, model_path in models().urls.items():  # We filter out the decoder models.
        if model_name.endswith("decoder"):
            continue
        sam1_names.append(model_name)

        if "https://dl.fbaipublicfiles.com/segment_anything/" in model_path:  # Valid v1 SAM models.
            sam1_display.append(model_name)

        if "https://owncloud.gwdg.de/" in model_path:  # Our own hosted models (in their v1 mode quite often)
            if model_name == "vit_t":  # MobileSAM model.
                sam1_display.append(model_name)
            else:
                sam1_display.append(f"{model_name} (v1)")

        # Now for our models, the BioImageIO ModelZoo upload structure is such that:
        # '/1/files' corresponds to v2 models.
        # '/1.1/files' corresponds to v3 models.
        # '/1.2/files' corresponds to v4 models.
        if "/1/files" in model_path:
            sam1_display.append(f"{model_name} (v2)")
        if "/1.1/files" in model_path:
            sam1_display.append(f"{model_name} (v3)")
        if "/1.2/files" in model_path:
            sam1_display.append(f"{model_name} (v4)")

    # SAM2 models: the base backbones plus the finetuned micro-sam models (with a registered decoder).
    sam2_base = list(SUPPORTED_MODELS)
    sam2_finetuned = list(get_model_names())
    sam2_names = sam2_base + sam2_finetuned
    sam2_display = sam2_base + [f"{name} (DEV)" for name in sam2_finetuned]

    # The available models panels (SAM1 and SAM2 shown separately to avoid confusing the BioImageIO
    # version suffixes above with the SAM version).
    console.print(
        Panel(f"[bold #D55E00]{chr(10).join(sam1_display)}[/bold #D55E00]", title="SAM1 Models")
    )
    console.print(
        Panel(f"[bold #D55E00]{chr(10).join(sam2_display)}[/bold #D55E00]", title="SAM2 Models")
    )

    # The system information table.
    total_memory = psutil.virtual_memory().total / (1024 ** 3)
    table = Table(title="System Information", show_header=True, header_style="bold #0072B2", expand=True)
    table.add_column("Property")
    table.add_column("Value", style="bold #56B4E9")
    table.add_row("System", platform.system())
    table.add_row("Node Name", platform.node())
    table.add_row("Release", platform.release())
    table.add_row("Version", platform.version())
    table.add_row("Machine", platform.machine())
    table.add_row("Processor", platform.processor())
    table.add_row("Platform", platform.platform())
    table.add_row("Python", platform.python_version())
    table.add_row("CPU Count", str(psutil.cpu_count(logical=True)))
    table.add_row("Total RAM (GB)", f"{total_memory:.2f}")
    console.print(table)

    # Accelerator / device information. This identifies the exact backend, since PyTorch exposes several
    # (NVIDIA CUDA, AMD ROCm - which also reports through the CUDA API but sets 'torch.version.hip' -,
    # Apple MPS and Intel XPU) and knowing which one (and the device name / memory) is key for debugging.
    device_lines = [f"[bold]PyTorch:[/bold] {torch.__version__}"]
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        is_rocm = getattr(torch.version, "hip", None) is not None
        device_lines.append(f"[bold]Backend:[/bold] {'ROCm (AMD)' if is_rocm else 'CUDA (NVIDIA)'}")
        device_lines.append(f"[bold]Device:[/bold] {props.name}")
        device_lines.append(f"[bold]Device count:[/bold] {torch.cuda.device_count()}")
        device_lines.append(f"[bold]GPU memory (GB):[/bold] {props.total_memory / (1024 ** 3):.2f}")
        if is_rocm:
            device_lines.append(f"[bold]HIP version:[/bold] {torch.version.hip}")
        else:
            device_lines.append(f"[bold]CUDA (torch build):[/bold] {torch.version.cuda}")
            device_lines.append(f"[bold]cuDNN:[/bold] {torch.backends.cudnn.version()}")
    elif torch.backends.mps.is_available():
        # On Apple Silicon the GPU is the SoC. Report the exact chip on a best-effort basis.
        chip = platform.processor()
        if platform.system() == "Darwin":
            try:
                import subprocess
                chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip() or chip
            except Exception:
                pass
        device_lines.append("[bold]Backend:[/bold] MPS (Apple Silicon)")
        device_lines.append(f"[bold]Device:[/bold] {chip}")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        idx = torch.xpu.current_device()
        device_lines.append("[bold]Backend:[/bold] XPU (Intel)")
        device_lines.append(f"[bold]Device:[/bold] {torch.xpu.get_device_name(idx)}")
        device_lines.append(f"[bold]Device count:[/bold] {torch.xpu.device_count()}")
    else:
        device_lines.append("[bold]Backend:[/bold] CPU (no GPU acceleration detected)")
    console.print(Panel("\n".join(device_lines), title="Accelerator Information"))

    # The section allowing to download models.
    # NOTE: In future, can be extended to download sample data.
    if download:
        all_names = sam1_names + sam2_names
        if any(t.lower() == "all" for t in download):  # Download every available model.
            download_list = all_names
            # Guard the bulk download (all SAM1 + SAM2 models, several GB) behind a confirmation, but
            # only when interactive so scripted '--download all' still runs unattended.
            import sys
            from rich.prompt import Confirm
            if sys.stdin.isatty() and not Confirm.ask(
                f"[yellow]This downloads all {len(download_list)} models (SAM1 + SAM2), several GB. Continue?[/]",
                console=console, default=False,
            ):
                console.print("[red]Download aborted.[/]")
                return
        else:
            download_list = list(download)
            incorrect_models = [m for m in download_list if m not in all_names]
            if incorrect_models:
                console.print(Panel("[red]Unknown model(s):[/] " + ", ".join(incorrect_models), title="Download Error"))
                return

        def _download(name):  # Dispatch to the downloader for the model's family.
            if name in sam2_finetuned:
                _download_finetuned_sam2_model(name)
            elif name in sam2_base:
                _get_checkpoint(name)
            else:
                _download_sam_model(model_type=name)

        with progress.Progress(
            progress.SpinnerColumn(),
            progress.TextColumn("[progress.description]{task.description}"),
            progress.BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            progress.TimeRemainingColumn(),
            console=console,
        ) as prog:
            task = prog.add_task("[green]Downloading μSAM models…", total=len(download_list))
            for model_type in download_list:
                prog.update(task, description=f"Downloading [cyan]{model_type}[/]…")
                _download(model_type)
                prog.advance(task)

        console.print(Panel("[bold green] Downloads complete![/]", title="Finished"))


#
# Functionality to convert mask predictions to an instance segmentation via non-maximum suppression.
# The functionality for computing NMS for masks is taken from CellSeg1:
# https://github.com/Nuisal/cellseg1/blob/1c027c2568b83494d2662d1fbecec9aafb478ee0/mask_nms.py
#


def _overlap_matrix(boxes):
    x1 = torch.max(boxes[:, None, 0], boxes[:, 0])
    y1 = torch.max(boxes[:, None, 1], boxes[:, 1])
    x2 = torch.min(boxes[:, None, 2], boxes[:, 2])
    y2 = torch.min(boxes[:, None, 3], boxes[:, 3])

    w = torch.clamp(x2 - x1, min=0)
    h = torch.clamp(y2 - y1, min=0)

    return (w * h) > 0


def _calculate_ious_between_pred_masks(masks, boxes, diagonal_value=1):
    n_points = masks.shape[0]
    m = torch.zeros((n_points, n_points))

    overlap_m = _overlap_matrix(boxes)

    for i in range(n_points):
        js = torch.where(overlap_m[i])[0]
        js_half = js[js > i].to(masks.device)

        if len(js_half) > 0:
            intersection = torch.logical_and(masks[i], masks[js_half]).sum(dim=(1, 2))
            union = torch.logical_or(masks[i], masks[js_half]).sum(dim=(1, 2))
            iou = intersection / union
            m[i, js_half] = iou

    m = m + m.T
    m.fill_diagonal_(diagonal_value)
    return m


def _calculate_iomin_between_pred_masks(masks, boxes, eps=1e-6):
    overlap_m = _overlap_matrix(boxes)

    # Flatten spatial dimensions: (N, H*W) or (N, D*H*W)
    N = masks.shape[0]
    masks_flat = masks.reshape(N, -1).float()

    # Per-mask area
    areas = masks_flat.sum(dim=1)  # (N,)

    # Pairwise intersections via matrix multiplication
    # inter[i, j] = sum_k masks_flat[i, k] * masks_flat[j, k]
    inter = masks_flat @ masks_flat.t()  # (N, N)

    # Denominator: min area of the two masks
    min_areas = torch.minimum(areas[:, None], areas[None, :])  # (N, N)

    # IoMin = intersection / min(area_i, area_j)
    iomin = inter / (min_areas + eps)

    # Set elements without any overlap explicitly to zero.
    iomin[~overlap_m] = 0
    return iomin


def _batched_mask_nms(masks, boxes, scores, nms_thresh, intersection_over_min):
    boxes = (
        boxes.detach() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes)
    ).cpu()
    scores = (
        scores.detach() if isinstance(scores, torch.Tensor) else torch.tensor(scores)
    ).cpu()
    masks = (
        masks.detach() if isinstance(masks, torch.Tensor) else torch.tensor(masks)
    ).cpu()

    if intersection_over_min:
        iou_matrix = _calculate_iomin_between_pred_masks(masks, boxes)
    else:
        iou_matrix = _calculate_ious_between_pred_masks(masks, boxes)
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        keep.append(i)

        if len(sorted_indices) == 1:
            break

        iou_values = iou_matrix[i, sorted_indices[1:]]
        mask = iou_values <= nms_thresh
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep)


def _xywh_to_xyxy(boxes):
    boxes = boxes.clone() if isinstance(boxes, torch.Tensor) else torch.tensor(boxes)
    boxes = boxes.to(torch.float32)
    boxes[:, 2] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 1]
    return boxes


def _infer_tiled_shape(predictions):
    shape = [0, 0]
    for pred in predictions:
        bbox, global_bbox = pred["bbox"], pred["global_bbox"]
        offset = (global_bbox[0] - bbox[0], global_bbox[1] - bbox[1])
        mask_shape = pred["segmentation"].shape
        shape[0] = max(shape[0], offset[1] + mask_shape[0])
        shape[1] = max(shape[1], offset[0] + mask_shape[1])
    return tuple(shape)


def _calculate_tiled_mask_overlap_matrix(masks, boxes, global_boxes, intersection_over_min):
    n_masks = len(masks)
    overlap_scores = torch.zeros((n_masks, n_masks))
    overlap_scores.fill_diagonal_(1)

    boxes = (
        boxes.detach().cpu().to(dtype=torch.long)
        if isinstance(boxes, torch.Tensor) else torch.tensor(boxes, dtype=torch.long)
    )
    global_boxes = (
        global_boxes.detach().cpu().to(dtype=torch.long)
        if isinstance(global_boxes, torch.Tensor) else torch.tensor(global_boxes, dtype=torch.long)
    )
    global_boxes_xyxy = _xywh_to_xyxy(global_boxes).to(torch.long)
    overlap_m = _overlap_matrix(global_boxes_xyxy)
    masks = [mask.detach().cpu() if isinstance(mask, torch.Tensor) else torch.tensor(mask) for mask in masks]
    areas = torch.tensor([mask.sum() for mask in masks], dtype=torch.float32)

    for i in range(n_masks):
        js = torch.where(overlap_m[i])[0]
        js_half = js[js > i]
        if len(js_half) == 0:
            continue

        offset_i = global_boxes[i, :2] - boxes[i, :2]
        for j in js_half:
            offset_j = global_boxes[j, :2] - boxes[j, :2]
            overlap = [
                max(global_boxes_xyxy[i, 0], global_boxes_xyxy[j, 0]),
                max(global_boxes_xyxy[i, 1], global_boxes_xyxy[j, 1]),
                min(global_boxes_xyxy[i, 2], global_boxes_xyxy[j, 2]),
                min(global_boxes_xyxy[i, 3], global_boxes_xyxy[j, 3]),
            ]

            mask_i = masks[i][
                overlap[1] - offset_i[1]:overlap[3] - offset_i[1],
                overlap[0] - offset_i[0]:overlap[2] - offset_i[0],
            ]
            mask_j = masks[j][
                overlap[1] - offset_j[1]:overlap[3] - offset_j[1],
                overlap[0] - offset_j[0]:overlap[2] - offset_j[0],
            ]
            intersection = torch.logical_and(mask_i, mask_j).sum()
            min_area = torch.minimum(areas[i], areas[j])
            denominator = min_area if intersection_over_min else areas[i] + areas[j] - intersection
            overlap_scores[i, j] = intersection / denominator

    overlap_scores = overlap_scores + overlap_scores.T
    overlap_scores.fill_diagonal_(1)
    return overlap_scores


def _batched_tiled_mask_nms(masks, boxes, global_boxes, scores, nms_thresh, intersection_over_min):
    scores = (
        scores.detach() if isinstance(scores, torch.Tensor) else torch.tensor(scores)
    ).cpu()

    iou_matrix = _calculate_tiled_mask_overlap_matrix(masks, boxes, global_boxes, intersection_over_min)
    sorted_indices = torch.argsort(scores, descending=True)

    keep = []
    while len(sorted_indices) > 0:
        i = sorted_indices[0]
        keep.append(i)

        if len(sorted_indices) == 1:
            break

        iou_values = iou_matrix[i, sorted_indices[1:]]
        mask = iou_values <= nms_thresh
        sorted_indices = sorted_indices[1:][mask]

    return torch.tensor(keep)


def mask_data_to_segmentation(
    masks: List[Dict[str, Any]],
    shape: Optional[Tuple[int, int]] = None,
    min_object_size: int = 0,
    max_object_size: Optional[int] = None,
    label_masks: bool = True,
    with_background: bool = False,
    merge_exclusively: bool = True,
) -> np.ndarray:
    """Convert the output of the automatic mask generation to an instance segmentation.

    Args:
        masks: The outputs generated by `AutomaticMaskGenerator`, other classes from
            `micro_sam.v1.instance_segmentation`, or from `micro_sam.v1.inference` functions. Only
            supported for output_mode=binary_mask.
        shape: The shape of the output segmentation. If None, it will be derived from the mask input.
            If the mask where predicted with tiling then the shape must be given.
        min_object_size: The minimal size of an object in pixels. By default, set to '0'.
        max_object_size: The maximal size of an object in pixels.
        label_masks: Whether to apply connected components to the result before removing small objects.
            By default, set to 'True'.
        with_background: Whether to remove the largest object, which often covers the background for AMG.
        merge_exclusively: Whether to exclude previous merged masks from merging.

    Returns:
        The instance segmentation.
    """
    masks = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    if shape is None:
        shape = next(iter(masks))["segmentation"].shape
    segmentation = np.zeros(shape, dtype="uint32")

    def require_numpy(mask):
        return mask.cpu().numpy() if torch.is_tensor(mask) else mask

    seg_id = 1
    for mask_data in masks:
        area = mask_data["area"]
        if (area < min_object_size) or (max_object_size is not None and area > max_object_size):
            continue

        this_mask = require_numpy(mask_data["segmentation"])
        this_seg_id = mask_data.get("seg_id", seg_id)
        if "global_bbox" in mask_data:
            bb = mask_data["bbox"]
            bb = np.s_[bb[1]:bb[1] + bb[3], bb[0]:bb[0] + bb[2]]
            global_bb = mask_data["global_bbox"]
            global_bb = np.s_[global_bb[1]:global_bb[1] + global_bb[3], global_bb[0]:global_bb[0] + global_bb[2]]
            if merge_exclusively:
                this_mask = np.logical_and(this_mask[bb], segmentation[global_bb] == 0)
            else:
                this_mask = this_mask[bb]
            segmentation[global_bb][this_mask] = this_seg_id
        else:
            if merge_exclusively:
                this_mask = np.logical_and(this_mask, segmentation == 0)
            segmentation[this_mask] = this_seg_id
        seg_id = this_seg_id + 1

    block_shape = (512, 512)
    if label_masks:
        segmentation_cc = np.zeros_like(segmentation, dtype=segmentation.dtype)
        segmentation_cc = parallel_impl.label(segmentation, out=segmentation_cc, block_shape=block_shape)
        segmentation = segmentation_cc

    seg_ids, sizes = parallel_impl.unique(segmentation, return_counts=True, block_shape=block_shape)
    filter_ids = seg_ids[sizes < min_object_size]
    if with_background:
        bg_id = seg_ids[np.argmax(sizes)]
        filter_ids = np.concatenate([filter_ids, [bg_id]])

    filter_mask = np.zeros(segmentation.shape, dtype="bool")
    filter_mask = parallel_impl.isin(segmentation, filter_ids, out=filter_mask, block_shape=block_shape)
    segmentation[filter_mask] = 0
    parallel_impl.relabel_consecutive(segmentation, block_shape=block_shape)[0]

    return segmentation


def apply_nms(
    predictions: List[Dict[str, Any]],
    min_size: int,
    shape: Optional[Tuple[int, int]] = None,
    perform_box_nms: bool = False,
    nms_thresh: float = 0.9,
    max_size: Optional[int] = None,
    intersection_over_min: bool = False,
) -> np.ndarray:
    """Apply non-maximum suppression to mask predictions from a segment anything model.

    Args:
        predictions: The mask predictions from SAM.
        min_size: The minimum mask size to keep in the output.
        shape: The shape of the output segmentation.
            For tiled predictions this is inferred from the tile-local mask shapes if it is not passed.
        perform_box_nms: Whether to perform NMS on the box coordinates or on the masks.
        nms_thresh: The threshold for filtering out objects in NMS.
        max_size: The maximum mask size to keep in the output.
        intersection_over_min: Whether to perform intersection over the minimum overlap shape
            or to perform intersection over union.

    Returns:
        The segmentation obtained from merging the masks left after NMS.
    """
    # Check if the input comes with a 'global_bbox' attribute. If it does, then the predictions are from
    # a tiled prediction. In this case, we have to take the coordinates w.r.t. the tiling into account.
    is_tiled = "global_bbox" in predictions[0]
    if is_tiled and shape is None:
        shape = _infer_tiled_shape(predictions)

    masks = [pred["segmentation"] for pred in predictions]
    nms_masks = None if is_tiled else torch.cat([mask[None] for mask in masks], dim=0)
    data = amg_utils.MaskData(masks=masks, iou_preds=torch.tensor([pred["predicted_iou"] for pred in predictions]))
    data["boxes"] = torch.tensor(np.array([pred["bbox"] for pred in predictions]))
    data["area"] = [int(mask.sum()) for mask in data["masks"]]
    data["stability_scores"] = torch.tensor([pred["stability_score"] for pred in predictions])
    if is_tiled:
        data["global_boxes"] = torch.tensor(np.array([pred["global_bbox"] for pred in predictions]))

    if min_size > 0:
        keep_by_size = torch.tensor(
            [i for i, area in enumerate(data["area"]) if area > min_size], dtype=torch.long,
        )
        data.filter(keep_by_size)
        if nms_masks is not None:
            nms_masks = nms_masks[keep_by_size]

    if max_size is not None:
        keep_by_size = torch.tensor([i for i, area in enumerate(data["area"]) if area < max_size])
        data.filter(keep_by_size)
        if nms_masks is not None:
            nms_masks = nms_masks[keep_by_size]

    if len(data["masks"]) == 0:
        if shape is None:
            shape = predictions[0]["segmentation"].shape
        return np.zeros(shape, dtype="uint32")

    scores = data["iou_preds"] * data["stability_scores"]
    boxes = _xywh_to_xyxy(data["global_boxes"] if is_tiled else data["boxes"])
    if perform_box_nms:
        assert not intersection_over_min  # not implemented
        keep_by_nms = batched_nms(
            boxes,
            scores,
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=nms_thresh,
        )
    elif is_tiled:
        keep_by_nms = _batched_tiled_mask_nms(
            masks=data["masks"],
            boxes=data["boxes"],
            global_boxes=data["global_boxes"],
            scores=scores,
            nms_thresh=nms_thresh,
            intersection_over_min=intersection_over_min,
        )
    else:
        keep_by_nms = _batched_mask_nms(
            masks=nms_masks,
            boxes=boxes,
            scores=scores,
            nms_thresh=nms_thresh,
            intersection_over_min=intersection_over_min,
        )
    data.filter(keep_by_nms)

    if is_tiled:
        mask_data = [
            {"segmentation": mask, "area": area, "bbox": box, "global_bbox": global_box}
            for mask, area, box, global_box in zip(data["masks"], data["area"], data["boxes"], data["global_boxes"])
        ]
    else:
        mask_data = [
            {"segmentation": mask, "area": area, "bbox": box}
            for mask, area, box in zip(data["masks"], data["area"], data["boxes"])
        ]

    if shape is None:
        shape = predictions[0]["segmentation"].shape
    if mask_data:
        segmentation = mask_data_to_segmentation(mask_data, shape=shape, min_object_size=min_size)
    else:  # In case all objects have been filtered out due to size filtering.
        segmentation = np.zeros(shape, dtype="uint32")

    return segmentation
