"""
Helper functions for downloading Segment Anything models and predicting image embeddings.
"""

import os
import pickle
import hashlib
import warnings
from pathlib import Path
from collections import OrderedDict
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import zarr
import vigra
import torch
import pooch
import xxhash
import numpy as np
import imageio.v3 as imageio
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential

from elf.io import open_file

from nifty.tools import blocking

from .__version__ import __version__
from . import models as custom_models

try:
    # Avoid import warnigns from mobile_sam
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
# currently set to the default vit_l
_DEFAULT_MODEL = "vit_l"

# The valid model types. Each type corresponds to the architecture of the
# vision transformer used within SAM.
_MODEL_TYPES = ("vit_l", "vit_b", "vit_h", "vit_t")


# TODO define the proper type for image embeddings
ImageEmbeddings = Dict[str, Any]
"""@private"""


def get_cache_directory() -> None:
    """Get micro-sam cache directory location.

    Users can set the MICROSAM_CACHEDIR environment variable for a custom cache directory.
    """
    default_cache_directory = os.path.expanduser(pooch.os_cache("micro_sam"))
    cache_directory = Path(os.environ.get("MICROSAM_CACHEDIR", default_cache_directory))
    return cache_directory


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


def models():
    """Return the segmentation models registry.

    We recreate the model registry every time this function is called,
    so any user changes to the default micro-sam cache directory location
    are respected.
    """

    # We use xxhash to compute the hash of the models, see
    # https://github.com/computational-cell-analytics/micro-sam/issues/283
    # (It is now a dependency, so we don't provide the sha256 fallback anymore.)
    # To generate the xxh128 hash:
    #     xxh128sum filename
    encoder_registry = {
        # The default segment anything models:
        "vit_l": "xxh128:a82beb3c660661e3dd38d999cc860e9a",
        "vit_h": "xxh128:97698fac30bd929c2e6d8d8cc15933c2",
        "vit_b": "xxh128:6923c33df3637b6a922d7682bfc9a86b",
        # The model with vit tiny backend fom https://github.com/ChaoningZhang/MobileSAM.
        "vit_t": "xxh128:8eadbc88aeb9d8c7e0b4b60c3db48bd0",
        # The current version of our models in the modelzoo.
        # LM generalist models:
        "vit_l_lm": "xxh128:ad3afe783b0d05a788eaf3cc24b308d2",
        "vit_b_lm": "xxh128:61ce01ea731d89ae41a252480368f886",
        "vit_t_lm": "xxh128:f90e2ba3dd3d5b935aa870cf2e48f689",
        # EM models:
        "vit_l_em_organelles": "xxh128:096c9695966803ca6fde24f4c1e3c3fb",
        "vit_b_em_organelles": "xxh128:f6f6593aeecd0e15a07bdac86360b6cc",
        "vit_t_em_organelles": "xxh128:253474720c497cce605e57c9b1d18fd9",
    }
    # Additional decoders for instance segmentation.
    decoder_registry = {
        # LM generalist models:
        "vit_l_lm_decoder": "xxh128:40c1ae378cfdce24008b9be24889a5b1",
        "vit_b_lm_decoder": "xxh128:1bac305195777ba7375634ca15a3c370",
        "vit_t_lm_decoder": "xxh128:82d3604e64f289bb66ec46a5643da169",
        # EM models:
        "vit_l_em_organelles_decoder": "xxh128:d60fd96bd6060856f6430f29e42568fb",
        "vit_b_em_organelles_decoder": "xxh128:b2d4dcffb99f76d83497d39ee500088f",
        "vit_t_em_organelles_decoder": "xxh128:8f897c7bb93174a4d1638827c4dd6f44",
    }
    registry = {**encoder_registry, **decoder_registry}

    encoder_urls = {
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_t": "https://owncloud.gwdg.de/index.php/s/TuDzuwVDHd1ZDnQ/download",
        "vit_l_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1/files/vit_l.pt",
        "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b.pt",
        "vit_t_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/faithful-chicken/1/files/vit_t.pt",
        "vit_l_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/humorous-crab/1/files/vit_l.pt",  # noqa
        "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b.pt",
        "vit_t_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/greedy-whale/1/files/vit_t.pt",  # noqa
    }

    decoder_urls = {
        "vit_l_lm_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1/files/vit_l_decoder.pt",  # noqa
        "vit_b_lm_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1/files/vit_b_decoder.pt",  # noqa
        "vit_t_lm_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/faithful-chicken/1/files/vit_t_decoder.pt",  # noqa
        "vit_l_em_organelles_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/humorous-crab/1/files/vit_l_decoder.pt",  # noqa
        "vit_b_em_organelles_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1/files/vit_b_decoder.pt",  # noqa
        "vit_t_em_organelles_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/greedy-whale/1/files/vit_t_decoder.pt",  # noqa
    }
    urls = {**encoder_urls, **decoder_urls}

    models = pooch.create(
        path=os.path.join(microsam_cachedir(), "models"),
        base_url="",
        registry=registry,
        urls=urls,
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
            raise RuntimeError(
                f"Unsupported device: {device}\n"
                "Please choose from 'cpu', 'cuda', or 'mps'."
            )

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


def get_sam_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    return_sam: bool = False,
    return_state: bool = False,
    peft_kwargs: Optional[Dict] = None,
    flexible_load_checkpoint: bool = False,
    **model_kwargs,
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
        return_state: Return the unpickled checkpoint state.
        peft_kwargs: Keyword arguments for th PEFT wrapper class.
        flexible_load_checkpoint: Whether to adjust mismatching params while loading pretrained checkpoints.
        model_kwargs: Additional parameters necessary to initialize the Segment Anything model.

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
        checkpoint_path = model_registry.fetch(model_type, progressbar=True)
        model_hash = model_registry.registry[model_type]

        # If we have a custom model then we may also have a decoder checkpoint.
        # Download it here, so that we can add it to the state.
        decoder_name = f"{model_type}_decoder"
        decoder_path = model_registry.fetch(
            decoder_name, progressbar=True
        ) if decoder_name in model_registry.registry else None

    # checkpoint_path has been passed, we use it instead of downloading a model.
    else:
        # Check if the file exists and raise an error otherwise.
        # We can't check any hashes here, and we don't check if the file is actually a valid weight file.
        # (If it isn't the model creation will fail below.)
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint at {checkpoint_path} could not be found.")
        model_hash = _compute_hash(checkpoint_path)
        decoder_path = None

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

    state, model_state = _load_checkpoint(checkpoint_path)

    # Whether to update parameters necessary to initialize the model
    if model_kwargs:  # Checks whether model_kwargs have been provided or not
        if abbreviated_model_type == "vit_t":
            raise ValueError("'micro-sam' does not support changing the model parameters for 'mobile-sam'.")
        sam = custom_models.sam_model_registry[abbreviated_model_type](**model_kwargs)

    else:
        sam = sam_model_registry[abbreviated_model_type]()

    # Whether to use Parameter Efficient Finetuning methods to wrap around Segment Anything.
    # Overwrites the SAM model by freezing the backbone and allow PEFT.
    if peft_kwargs and isinstance(peft_kwargs, dict):
        _quantize = peft_kwargs.pop("quantize", False)
        if _quantize:
            # get default sam model and put lora wrapper on top of it
            _, sam = get_sam_model(
                model_type=model_type,
                checkpoint_path=None,
                device=device,
                return_state=False,
                return_sam=True
            )
        if abbreviated_model_type == "vit_t":
            raise ValueError("'micro-sam' does not support parameter efficient finetuning for 'mobile-sam'.")

        sam = custom_models.peft_sam.PEFT_Sam(sam, **peft_kwargs).sam

        # update the model state to take the lora weights from the qlora checkpoint and the sam weights for everything else
        if _quantize:
            updated_model_state = {}
            for k, v in sam.state_dict().items():
                if k.find("w_b_linear") != -1 or k.find("w_a_linear") != -1:
                    updated_model_state[k] = model_state[k]
                else:
                    updated_model_state[k] = v
            model_state = updated_model_state

    # In case the model checkpoints have some issues when it is initialized with different parameters than default.
    if flexible_load_checkpoint:
        sam = _handle_checkpoint_loading(sam, model_state)
    else:
        sam.load_state_dict(model_state)

    sam.to(device=device)

    predictor = SamPredictor(sam)
    predictor.model_type = abbreviated_model_type
    predictor._hash = model_hash
    predictor.model_name = model_type

    # Add the decoder to the state if we have one and if the state is returned.
    if decoder_path is not None and return_state:
        state["decoder_state"] = torch.load(decoder_path, map_location=device, weights_only=False)

    if return_sam and return_state:
        return predictor, sam, state
    if return_sam:
        return predictor, sam
    if return_state:
        return predictor, state
    return predictor


def _handle_checkpoint_loading(sam, model_state, peft_kwargs):
    # Whether to handle the mismatch issues in a bit more elegant way.
    # eg. while training for multi-class semantic segmentation in the mask encoder,
    # parameters are updated - leading to "size mismatch" errors

    new_state_dict = {}  # for loading matching parameters
    mismatched_layers = []  # for tracking mismatching parameters

    reference_state = sam.state_dict()

    for k, v in model_state.items():
        if k in reference_state:  # This is done to get rid of unwanted layers from pretrained SAM.
            if reference_state[k].size() == v.size():
                new_state_dict[k] = v
            else:
                mismatched_layers.append(k)

    reference_state.update(new_state_dict)

    if len(mismatched_layers) > 0:
        warnings.warn(f"The layers with size mismatch: {mismatched_layers}")

    for mlayer in mismatched_layers:
        if 'weight' in mlayer:
            torch.nn.init.kaiming_uniform_(reference_state[mlayer])
        elif 'bias' in mlayer:
            reference_state[mlayer].zero_()

    sam.load_state_dict(reference_state)

    return sam


def export_custom_sam_model(
    checkpoint_path: Union[str, os.PathLike], model_type: str, save_path: Union[str, os.PathLike],
) -> None:
    """Export a finetuned segment anything model to the standard model format.

    The exported model can be used by the interactive annotation tools in `micro_sam.annotator`.

    Args:
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        model_type: The SegmentAnything model type corresponding to the checkpoint (vit_h, vit_b, vit_l or vit_t).
        save_path: Where to save the exported model.
    """
    _, state = get_sam_model(
        model_type=model_type, checkpoint_path=checkpoint_path, return_state=True, device="cpu",
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
# Functionality for precomputing image embeddings.
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


def _compute_tiled_features_2d(predictor, input_, tile_shape, halo, f, pbar_init, pbar_update):
    tiling = blocking([0, 0], input_.shape[:2], tile_shape)
    n_tiles = tiling.numberOfBlocks

    features = f.require_group("features")
    features.attrs["shape"] = input_.shape[:2]
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    pbar_init(n_tiles, "Compute Image Embeddings 2D tiled.")
    for tile_id in range(n_tiles):
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
        pbar_update(1)

    _write_embedding_signature(f, input_, predictor, tile_shape, halo, input_size=None, original_size=None)

    return features


def _compute_tiled_features_3d(predictor, input_, tile_shape, halo, f, pbar_init, pbar_update):
    assert input_.ndim == 3

    shape = input_.shape[1:]
    tiling = blocking([0, 0], shape, tile_shape)
    n_tiles = tiling.numberOfBlocks

    features = f.require_group("features")
    features.attrs["shape"] = shape
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    n_slices = input_.shape[0]
    pbar_init(n_tiles * n_slices, "Compute Image Embeddings 3D tiled.")

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
            pbar_update(1)

        original_size = predictor.original_size
        input_size = predictor.input_size

        ds.attrs["original_size"] = original_size
        ds.attrs["input_size"] = input_size

    _write_embedding_signature(f, input_, predictor, tile_shape, halo, input_size=None, original_size=None)

    return features


def _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update):
    # Check if the embeddings are already cached.
    if save_path is not None and "input_size" in f.attrs:
        # In this case we load the embeddings.
        features = f["features"][:]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]
        image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
        # Also set the embeddings.
        set_precomputed(predictor, image_embeddings)
        return image_embeddings

    pbar_init(1, "Compute Image Embeddings 2D.")
    # Otherwise we have to compute the embeddings.
    predictor.reset_image()
    predictor.set_image(_to_image(input_))
    features = predictor.get_image_embedding().cpu().numpy()
    original_size = predictor.original_size
    input_size = predictor.input_size
    pbar_update(1)

    # Save the embeddings if we have a save_path.
    if save_path is not None:
        f.create_dataset("features", data=features, compression="gzip", chunks=features.shape)
        _write_embedding_signature(
            f, input_, predictor, tile_shape=None, halo=None, input_size=input_size, original_size=original_size,
        )

    image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
    return image_embeddings


def _compute_tiled_2d(input_, predictor, tile_shape, halo, f, pbar_init, pbar_update):
    # Check if the features are already computed.
    if "input_size" in f.attrs:
        features = f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]
        image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
        return image_embeddings

    # Otherwise compute them. Note: saving happens automatically because we
    # always write the features to zarr. If no save path is given we use an in-memory zarr.
    features = _compute_tiled_features_2d(predictor, input_, tile_shape, halo, f, pbar_init, pbar_update)
    image_embeddings = {"features": features, "input_size": None, "original_size": None}
    return image_embeddings


def _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update):
    # Check if the embeddings are already fully cached.
    if save_path is not None and "input_size" in f.attrs:
        # In this case we load the embeddings.
        features = f["features"] if lazy_loading else f["features"][:]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]
        image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
        return image_embeddings

    # Otherwise we have to compute the embeddings.

    # First check if we have a save path or not and set things up accordingly.
    if save_path is None:
        features = []
        save_features = False
        partial_features = False
    else:
        save_features = True
        embed_shape = (1, 256, 64, 64)
        shape = (input_.shape[0],) + embed_shape
        chunks = (1,) + embed_shape
        if "features" in f:
            partial_features = True
            features = f["features"]
            if features.shape != shape or features.chunks != chunks:
                raise RuntimeError("Invalid partial features")
        else:
            partial_features = False
            features = f.create_dataset("features", shape=shape, chunks=chunks, dtype="float32")

    # Initialize the pbar.
    pbar_init(input_.shape[0], "Compute Image Embeddings 3D")

    # Compute the embeddings for each slice.
    for z, z_slice in enumerate(input_):
        # Skip feature computation in case of partial features in non-zero slice.
        if partial_features and np.count_nonzero(features[z]) != 0:
            continue

        predictor.reset_image()
        predictor.set_image(_to_image(z_slice))
        embedding = predictor.get_image_embedding()
        original_size, input_size = predictor.original_size, predictor.input_size

        if save_features:
            features[z] = embedding.cpu().numpy()
        else:
            features.append(embedding[None])
        pbar_update(1)

    if save_features:
        _write_embedding_signature(
            f, input_, predictor, tile_shape=None, halo=None, input_size=input_size, original_size=original_size,
        )
    else:
        # Concatenate across the z axis.
        features = torch.cat(features).cpu().numpy()

    image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
    return image_embeddings


def _compute_tiled_3d(input_, predictor, tile_shape, halo, f, pbar_init, pbar_update):
    # Check if the features are already computed.
    if "input_size" in f.attrs:
        features = f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]
        image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
        return image_embeddings

    # Otherwise compute them. Note: saving happens automatically because we
    # always write the features to zarr. If no save path is given we use an in-memory zarr.
    features = _compute_tiled_features_3d(predictor, input_, tile_shape, halo, f, pbar_init, pbar_update)
    image_embeddings = {"features": features, "input_size": None, "original_size": None}
    return image_embeddings


def _compute_data_signature(input_):
    data_signature = hashlib.sha1(np.asarray(input_).tobytes()).hexdigest()
    return data_signature


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


def _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo):
    # We may have an empty zarr file that was already created to save the embeddings in.
    # In this case the embeddings will be computed and we don't need to perform any checks.
    if "input_size" not in f.attrs:
        return

    signature = _get_embedding_signature(input_, predictor, tile_shape, halo)
    for key, val in signature.items():
        # Check whether the key is missing from the attrs or if the value is not matching.
        if key not in f.attrs or f.attrs[key] != val:
            # These keys were recently added, so we don't want to fail yet if they don't
            # match in order to not invalidate previous embedding files.
            # Instead we just raise a warning. (For the version we probably also don't want to fail
            # i the future since it should not invalidate the embeddings).
            if key in ("micro_sam_version", "model_hash", "model_name"):
                warnings.warn(
                    f"The signature for {key} in embeddings file {save_path} has a mismatch: "
                    f"{f.attrs.get(key)} != {val}. This key was recently added, so your embeddings are likely correct. "
                    "But please recompute them if model predictions don't look as expected."
                )
            else:
                raise RuntimeError(
                    f"Embeddings file {save_path} is invalid due to mismatch in {key}: "
                    f"{f.attrs.get(key)} != {val}. Please recompute embeddings in a new file."
                )


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


def precompute_image_embeddings(
    predictor: SamPredictor,
    input_: np.ndarray,
    save_path: Optional[Union[str, os.PathLike]] = None,
    lazy_loading: bool = False,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
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
        verbose: Whether to be verbose in the computation.
        pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
            Can be used together with pbar_update to handle napari progress bar in other thread.
            To enables using this function within a threadworker.
        pbar_update: Callback to update an external progress bar.

    Returns:
        The image embeddings.
    """
    ndim = input_.ndim if ndim is None else ndim

    # Handle the embedding save_path.
    # We don't have a save path, open in memory zarr file to hold tiled embeddings.
    if save_path is None:
        f = zarr.group()

    # We have a save path and it already exists. Embeddings will be loaded from it,
    # check that the saved embeddings in there match the parameters of the function call.
    elif os.path.exists(save_path):
        f = zarr.open(save_path, "a")
        _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo)

    # We have a save path and it does not exist yet. Create the zarr file to which the
    # embeddings will then be saved.
    else:
        f = zarr.open(save_path, "a")

    _, pbar_init, pbar_update, pbar_close = handle_pbar(verbose, pbar_init, pbar_update)

    if ndim == 2 and tile_shape is None:
        embeddings = _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update)
    elif ndim == 2 and tile_shape is not None:
        embeddings = _compute_tiled_2d(input_, predictor, tile_shape, halo, f, pbar_init, pbar_update)
    elif ndim == 3 and tile_shape is None:
        embeddings = _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update)
    elif ndim == 3 and tile_shape is not None:
        embeddings = _compute_tiled_3d(input_, predictor, tile_shape, halo, f, pbar_init, pbar_update)
    else:
        raise ValueError(f"Invalid dimesionality {input_.ndim}, expect 2 or 3 dim data.")

    pbar_close()
    return embeddings


def set_precomputed(
    predictor: SamPredictor, image_embeddings: ImageEmbeddings, i: Optional[int] = None, tile_id: Optional[int] = None,
) -> SamPredictor:
    """Set the precomputed image embeddings for a predictor.

    Args:
        predictor: The SegmentAnything predictor.
        image_embeddings: The precomputed image embeddings computed by `precompute_image_embeddings`.
        i: Index for the image data. Required if `image` has three spatial dimensions
            or a time dimension and two spatial dimensions.
        tile_id: Index for the tile. This is required if the embeddings are tiled.

    Returns:
        The predictor with set features.
    """
    if tile_id is not None:
        tile_features = image_embeddings["features"][tile_id]
        tile_image_embeddings = {
            "features": tile_features,
            "input_size": tile_features.attrs["input_size"],
            "original_size": tile_features.attrs["original_size"]
        }
        return set_precomputed(predictor, tile_image_embeddings, i=i)

    device = predictor.device
    features = image_embeddings["features"]
    assert features.ndim in (4, 5), f"{features.ndim}"
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
    segmentation: np.ndarray, mode: str = "v"
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

    Returns:
        The one-hot encoded masks.
    """
    masks = segmentation.copy()
    if segmentation_ids is None:
        n_ids = int(segmentation.max())

    else:
        assert segmentation_ids[0] != 0, "No objects were found."

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
