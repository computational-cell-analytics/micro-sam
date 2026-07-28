"""SAM1 (Segment Anything v1) model loading and embedding-computation utilities.
"""

import os
import pooch
import warnings
import multiprocessing as mp
from concurrent import futures
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Callable  # noqa

import numpy as np

import torch

from bioimage_cpp.utils import Blocking

from . import models as custom_models  # noqa
from ..util import (  # noqa
    ImageEmbeddings, SamPredictor, sam_model_registry, VIT_T_SUPPORT,
    _DEFAULT_MODEL, _MODEL_TYPES,
    get_device, _available_devices, _to_image, _load_checkpoint, _compute_hash,
    _CustomUnpickler, get_cache_directory, microsam_cachedir,
    _create_dataset_with_data, _create_dataset_without_data, _open_embeddings,
    _compute_data_signature, _get_embedding_signature, _write_embedding_signature,
    handle_pbar,
)


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
        "vit_l_lm": "xxh128:017f20677997d628426dec80a8018f9d",
        "vit_b_lm": "xxh128:fe9252a29f3f4ea53c15a06de471e186",
        "vit_t_lm": "xxh128:72ec5074774761a6e5c05a08942f981e",
        # EM models:
        "vit_l_em_organelles": "xxh128:810b084b6e51acdbf760a993d8619f2d",
        "vit_b_em_organelles": "xxh128:f3bf2ed83d691456bae2c3f9a05fb438",
        "vit_t_em_organelles": "xxh128:253474720c497cce605e57c9b1d18fd9",
        # Histopathology models:
        "vit_b_histopathology": "xxh128:ffd1a2cd84570458b257bd95fdd8f974",
        "vit_l_histopathology": "xxh128:b591833c89754271023e901281dee3f2",
        "vit_h_histopathology": "xxh128:bd1856dafc156a43fb3aa705f1a6e92e",
        # Medical Imaging models:
        "vit_b_medical_imaging": "xxh128:40169f1e3c03a4b67bff58249c176d92",
    }
    # Additional decoders for instance segmentation.
    decoder_registry = {
        # LM generalist models:
        "vit_l_lm_decoder": "xxh128:2faeafa03819dfe03e7c46a44aaac64a",
        "vit_b_lm_decoder": "xxh128:708b15ac620e235f90bb38612c4929ba",
        "vit_t_lm_decoder": "xxh128:3e914a5f397b0312cdd36813031f8823",
        # EM models:
        "vit_l_em_organelles_decoder": "xxh128:334877640bfdaaabce533e3252a17294",
        "vit_b_em_organelles_decoder": "xxh128:bb6398956a6b0132c26b631c14f95ce2",
        "vit_t_em_organelles_decoder": "xxh128:8f897c7bb93174a4d1638827c4dd6f44",
        # Histopathology models:
        "vit_b_histopathology_decoder": "xxh128:6a66194dcb6e36199cbee2214ecf7213",
        "vit_l_histopathology_decoder": "xxh128:46aab7765d4400e039772d5a50b55c04",
        "vit_h_histopathology_decoder": "xxh128:3ed9f87e46ad5e16935bd8d722c8dc47",
        # Medical Imaging models:
        "vit_b_medical_imaging_decoder": "xxh128:9e498b12f526f119b96c88be76e3b2ed",
    }
    registry = {**encoder_registry, **decoder_registry}

    encoder_urls = {
        "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_t": "https://owncloud.gwdg.de/index.php/s/TuDzuwVDHd1ZDnQ/download",
        "vit_l_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1.2/files/vit_l.pt",
        "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt",
        "vit_t_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/faithful-chicken/1.1/files/vit_t.pt",
        "vit_l_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/humorous-crab/1.2/files/vit_l.pt",  # noqa
        "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1.2/files/vit_b.pt",  # noqa
        "vit_t_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/greedy-whale/1/files/vit_t.pt",  # noqa
        "vit_b_histopathology": "https://owncloud.gwdg.de/index.php/s/sBB4H8CTmIoBZsQ/download",
        "vit_l_histopathology": "https://owncloud.gwdg.de/index.php/s/IZgnn1cpBq2PHod/download",
        "vit_h_histopathology": "https://owncloud.gwdg.de/index.php/s/L7AcvVz7DoWJ2RZ/download",
        "vit_b_medical_imaging": "https://owncloud.gwdg.de/index.php/s/f5Ol4FrjPQWfjUF/download",
    }

    decoder_urls = {
        "vit_l_lm_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1.2/files/vit_l_decoder.pt",  # noqa
        "vit_b_lm_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b_decoder.pt",  # noqa
        "vit_t_lm_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/faithful-chicken/1.1/files/vit_t_decoder.pt",  # noqa
        "vit_l_em_organelles_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/humorous-crab/1.2/files/vit_l_decoder.pt",  # noqa
        "vit_b_em_organelles_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1.2/files/vit_b_decoder.pt",  # noqa
        "vit_t_em_organelles_decoder": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/greedy-whale/1/files/vit_t_decoder.pt",  # noqa
        "vit_b_histopathology_decoder": "https://owncloud.gwdg.de/index.php/s/KO9AWqynI7SFOBj/download",
        "vit_l_histopathology_decoder": "https://owncloud.gwdg.de/index.php/s/oIs6VSmkOp7XrKF/download",
        "vit_h_histopathology_decoder": "https://owncloud.gwdg.de/index.php/s/1qAKxy5H0jgwZvM/download",
        "vit_b_medical_imaging_decoder": "https://owncloud.gwdg.de/index.php/s/ahd3ZhZl2e0RIwz/download",
    }
    urls = {**encoder_urls, **decoder_urls}

    models = pooch.create(
        path=os.path.join(microsam_cachedir(), "models"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return models


def _download_sam_model(model_type, progress_bar_factory=None):
    model_registry = models()

    progress_bar = True
    # Check if we have to download the model.
    # If we do and have a progress bar factory, then we over-write the progress bar.
    if not os.path.exists(os.path.join(get_cache_directory(), model_type)) and progress_bar_factory is not None:
        progress_bar = progress_bar_factory(model_type)

    checkpoint_path = model_registry.fetch(model_type, progressbar=progress_bar)
    if not isinstance(progress_bar, bool):  # Close the progress bar when the task finishes.
        progress_bar.close()

    model_hash = model_registry.registry[model_type]

    # If we have a custom model then we can also have a decoder checkpoint.
    # Download it here, so that we can add it to the state.
    decoder_name = f"{model_type}_decoder"
    decoder_path = model_registry.fetch(
        decoder_name, progressbar=True
    ) if decoder_name in model_registry.registry else None

    return checkpoint_path, model_hash, decoder_path


def get_sam_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[str, torch.device]] = None,
    checkpoint_path: Optional[Union[str, os.PathLike]] = None,
    return_sam: bool = False,
    return_state: bool = False,
    peft_kwargs: Optional[Dict] = None,
    flexible_load_checkpoint: bool = False,
    progress_bar_factory: Optional[Callable] = None,
    decoder_path: Optional[Union[str, os.PathLike]] = None,
    **model_kwargs,
) -> SamPredictor:
    r"""Get the Segment Anything Predictor.

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
        model_type: The Segment Anything model to use. Will use the 'vit_b_lm' model by default.
            To get a list of all available model names you can call `micro_sam.v1.util.get_model_names`.
        device: The device for the model. If 'None' is provided, will use GPU if available.
        checkpoint_path: The path to a file with weights that should be used instead of using the
            weights corresponding to `model_type`. If given, `model_type` must match the architecture
            corresponding to the weight file. e.g. if you use weights for SAM with `vit_b` encoder
            then `model_type` must be given as 'vit_b'.
        return_sam: Return the sam model object as well as the predictor. By default, set to 'False'.
        return_state: Return the unpickled checkpoint state. By default, set to 'False'.
        peft_kwargs: Keyword arguments for th PEFT wrapper class.
            If passed 'None', it does not initialize any parameter efficient finetuning.
        flexible_load_checkpoint: Whether to adjust mismatching params while loading pretrained checkpoints.
            By default, set to 'False'.
        progress_bar_factory: A function to create a progress bar for the model download.
        decoder_path: Optional path to weights for a segmentation decoder. If given and
            `return_state=True`, the decoder state is added to the returned state as
            'decoder_state'. This can be used to provide decoder-only weights that are
            separate from the encoder checkpoint.
        model_kwargs: Additional parameters necessary to initialize the Segment Anything model.

    Returns:
        The Segment Anything predictor.
    """
    device = get_device(device)

    # We support passing a local filepath to a checkpoint.
    # In this case we do not download any weights but just use the local weight file,
    # as it is, without copying it over anywhere or checking it's hashes.

    # checkpoint_path has not been passed, we download a known model and derive the correct
    # URL from the model_type. If the model_type is invalid pooch will raise an error.
    _provided_checkpoint_path = checkpoint_path is not None
    if checkpoint_path is None:
        checkpoint_path, model_hash, downloaded_decoder_path = _download_sam_model(model_type, progress_bar_factory)
        if decoder_path is None:
            decoder_path = downloaded_decoder_path

    # If checkpoint_path was passed, we use it instead of downloading a model.
    else:
        # Check if the file exists and raise an error otherwise.
        # We can't check any hashes here, and we don't check if the file is actually a valid weight file.
        # (If it isn't the model creation will fail below.)
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint at '{checkpoint_path}' could not be found.")
        model_hash = _compute_hash(checkpoint_path)

    if decoder_path is not None and not os.path.exists(decoder_path):
        raise ValueError(f"Decoder checkpoint at '{decoder_path}' could not be found.")

    # Our fine-tuned model types have a suffix "_...". This suffix needs to be stripped
    # before calling sam_model_registry.
    abbreviated_model_type = model_type[:5]
    if abbreviated_model_type not in _MODEL_TYPES:
        raise ValueError(f"Invalid model_type: {abbreviated_model_type}. Expect one of {_MODEL_TYPES}")
    if abbreviated_model_type == "vit_t" and not VIT_T_SUPPORT:
        raise RuntimeError(
            "'mobile_sam' is required for the vit-tiny. "
            "You can install it via 'pip install git+https://github.com/ChaoningZhang/MobileSAM.git'"
        )

    state, model_state = _load_checkpoint(checkpoint_path)

    if _provided_checkpoint_path:
        # To get the model weights, we prioritize having the correct 'checkpoint_path' over 'model_type'
        # It is done to avoid strange parameter mismatch issues while incompatible model type and weights combination.
        from micro_sam.v1.models.build_sam import _validate_model_type
        _provided_model_type = _validate_model_type(model_state)

        # Verify whether the 'abbreviated_model_type' matches the '_provided_model_type'
        # Otherwise replace 'abbreviated_model_type' with the later.
        if abbreviated_model_type != _provided_model_type:
            # Printing the message below to avoid any filtering of warnings on user's end.
            print(
                f"CRITICAL WARNING: The chosen 'model_type' is '{abbreviated_model_type}', "
                f"however the model checkpoint provided correspond to '{_provided_model_type}', which does not match. "
                f"We internally switch the model type to the expected value, i.e. '{_provided_model_type}'. "
                "However, please avoid mismatching combination of 'model_type' and 'checkpoint_path' in future."
            )

        # Replace the extracted 'abbreviated_model_type' subjected to the model weights.
        abbreviated_model_type = _provided_model_type

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
        # NOTE: We bump out 'quantize' parameter, if found, as we do not quantize in inference.
        peft_kwargs.pop("quantize", None)

        if abbreviated_model_type == "vit_t":
            raise ValueError("'micro-sam' does not support parameter efficient finetuning for 'mobile-sam'.")

        sam = custom_models.peft_sam.PEFT_Sam(sam, **peft_kwargs).sam
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
    predictor.checkpoint_path = checkpoint_path

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


def _handle_checkpoint_loading(sam, model_state):
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
    checkpoint_path: Union[str, os.PathLike],
    model_type: str,
    save_path: Union[str, os.PathLike],
    with_segmentation_decoder: bool = False,
    prefix: str = "sam.",
) -> None:
    """Export a finetuned Segment Anything Model to the standard model format.

    The exported model can be used by the interactive annotation tools in `micro_sam.annotator`.

    Args:
        checkpoint_path: The path to the corresponding checkpoint if not in the default model folder.
        model_type: The Segment Anything Model type corresponding to the checkpoint (vit_h, vit_b, vit_l or vit_t).
        save_path: Where to save the exported model.
        with_segmentation_decoder: Whether to store the decoder state in the model checkpoint as well.
            If set to 'True', the model checkpoint will not be compatible with other tools besides 'micro-sam'.
        prefix: The prefix to remove from the model parameter keys.
    """
    state, model_state = _load_checkpoint(checkpoint_path=checkpoint_path)
    model_state = OrderedDict([(k[len(prefix):] if k.startswith(prefix) else k, v) for k, v in model_state.items()])

    # Store the 'decoder_state' as well, if desired.
    if with_segmentation_decoder:
        if "decoder_state" not in state:
            raise RuntimeError(f"'decoder_state' is not found in the model at '{checkpoint_path}'.")
        decoder_state = state["decoder_state"]
        save_state = {"model_state": model_state, "decoder_state": decoder_state}
    else:
        save_state = model_state

    torch.save(save_state, save_path)


def export_custom_qlora_model(
    checkpoint_path: Optional[Union[str, os.PathLike]],
    finetuned_path: Union[str, os.PathLike],
    model_type: str,
    save_path: Union[str, os.PathLike],
) -> None:
    """Export a finetuned Segment Anything Model, in QLoRA style, to LoRA-style checkpoint format.

    The exported model can be used with the LoRA backbone by passing the relevant `peft_kwargs` to `get_sam_model`.

    Args:
        checkpoint_path: The path to the base foundation model from which the new model has been finetuned.
        finetuned_path: The path to the new finetuned model, using QLoRA.
        model_type: The Segment Anything Model type corresponding to the checkpoint.
        save_path: Where to save the exported model.
    """
    # Step 1: Get the base SAM model: used to start finetuning from.
    _, sam = get_sam_model(
        model_type=model_type, checkpoint_path=checkpoint_path, return_sam=True,
    )

    # Step 2: Load the QLoRA-style finetuned model.
    ft_state, ft_model_state = _load_checkpoint(finetuned_path)

    # Step 3: Identify LoRA layers from QLoRA model.
    # - differentiate between LoRA applied to the attention matrices and LoRA applied to the MLP layers.
    # - then copy the LoRA layers from the QLoRA model to the new state dict
    updated_model_state = {}

    modified_attn_layers = set()
    modified_mlp_layers = set()

    for k, v in ft_model_state.items():
        if "blocks." in k:
            layer_id = int(k.split("blocks.")[1].split(".")[0])
        if k.find("qkv.w_a_linear") != -1 or k.find("qkv.w_b_linear") != -1:
            modified_attn_layers.add(layer_id)
            updated_model_state[k] = v
        if k.find("mlp.w_a_linear") != -1 or k.find("mlp.w_b_linear") != -1:
            modified_mlp_layers.add(layer_id)
            updated_model_state[k] = v

    # Step 4: Next, we get all the remaining parameters from the base SAM model.
    for k, v in sam.state_dict().items():
        if "blocks." in k:
            layer_id = int(k.split("blocks.")[1].split(".")[0])
        if k.find("attn.qkv.") != -1:
            if layer_id in modified_attn_layers:  # We have LoRA in QKV layers, so we need to modify the key
                k = k.replace("qkv", "qkv.qkv_proj")
        elif k.find("mlp") != -1 and k.find("image_encoder") != -1:
            if layer_id in modified_mlp_layers:  # We have LoRA in MLP layers, so we need to modify the key
                k = k.replace("mlp.", "mlp.mlp_layer.")
        updated_model_state[k] = v

    # Step 5: Finally, we replace the old model state with the new one (to retain other relevant stuff)
    ft_state['model_state'] = updated_model_state

    # Step 6: Store the new "state" to "save_path"
    torch.save(ft_state, save_path)


def get_model_names() -> Iterable:
    model_registry = models()
    model_names = model_registry.registry.keys()
    return model_names


@torch.no_grad
def _compute_embeddings_batched(predictor, batched_images):
    predictor.reset_image()
    batched_tensors, original_sizes, input_sizes = [], [], []

    # Apply proeprocessing to all images in the batch, and then stack them.
    # Note: after the transformation the images are all of the same size,
    # so they can be stacked and processed as a batch, even if the input images were of different size.
    for image in batched_images:
        tensor = predictor.transform.apply_image(image)
        tensor = torch.as_tensor(tensor, device=predictor.device)
        tensor = tensor.permute(2, 0, 1).contiguous()[None, :, :, :]

        original_sizes.append(image.shape[:2])
        input_sizes.append(tensor.shape[-2:])

        tensor = predictor.model.preprocess(tensor)
        batched_tensors.append(tensor)

    batched_tensors = torch.cat(batched_tensors)
    features = predictor.model.image_encoder(batched_tensors)

    predictor.original_size = original_sizes[-1]
    predictor.input_size = input_sizes[-1]
    predictor.features = features[-1]
    predictor.is_image_set = True

    return features, original_sizes, input_sizes


def _write_batch(features, tile_ids, batched_embeddings, original_sizes, input_sizes, slices=None, n_slices=None):

    # Pre-create / pre-fetch the datasets if we have slices.
    # (Dataset creation is not thread-safe)
    if slices is not None:
        datasets = {}
        for tile_id, tile_embeddings, original_size, input_size in zip(
            tile_ids, batched_embeddings, original_sizes, input_sizes
        ):
            ds_name = str(tile_id)
            if ds_name in datasets:
                continue
            if ds_name in features:
                datasets[ds_name] = features[ds_name]
                continue
            shape = (n_slices, 1) + tile_embeddings.shape
            chunks = (1, 1) + tile_embeddings.shape
            ds = _create_dataset_without_data(features, ds_name, shape=shape, dtype="float32", chunks=chunks)
            ds.attrs["original_size"] = original_size
            ds.attrs["input_size"] = input_size
            datasets[ds_name] = ds

    def _write_embed(i):
        ds_name = str(tile_ids[i])
        tile_embeddings = batched_embeddings[i].unsqueeze(0)
        if slices is None:
            ds = _create_dataset_with_data(features, ds_name, data=tile_embeddings.cpu().numpy())
            ds.attrs["original_size"] = original_sizes[i]
            ds.attrs["input_size"] = input_sizes[i]
        elif ds_name in features:
            ds = datasets[ds_name]
            z = slices[i]
            ds[z] = tile_embeddings.cpu().numpy()

    n_tiles = len(tile_ids)
    n_workers = min(mp.cpu_count(), n_tiles)
    with futures.ThreadPoolExecutor(n_workers) as tp:
        list(tp.map(_write_embed, range(n_tiles)))


def _get_tiles_in_mask(mask, tiling, halo, z=None):
    def _check_mask(tile_id):
        tile = tiling.get_block_with_halo(tile_id, list(halo))
        outer_tile = tuple(slice(beg, end) for beg, end in zip(tile.outer_block.begin, tile.outer_block.end))
        if z is not None:
            outer_tile = (z,) + outer_tile
        tile_mask = mask[outer_tile].astype("bool")
        return None if tile_mask.sum() == 0 else tile_id

    n_threads = mp.cpu_count()
    with futures.ThreadPoolExecutor(n_threads) as tp:
        tiles_in_mask = tp.map(_check_mask, range(tiling.number_of_blocks))
    return sorted([tile_id for tile_id in tiles_in_mask if tile_id is not None])


def _compute_tiled_features_2d(predictor, input_, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask):
    tiling = Blocking([0, 0], input_.shape[:2], tile_shape)
    n_tiles = tiling.number_of_blocks

    features = f.require_group("features")
    features.attrs["shape"] = input_.shape[:2]
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    n_batches = int(np.ceil(n_tiles / batch_size))
    if mask is None:
        tile_ids_for_batches = [
            range(batch_id * batch_size, min((batch_id + 1) * batch_size, n_tiles))
            for batch_id in range(n_batches)
        ]
        pbar_init(n_tiles, "Compute Image Embeddings 2D tiled")
    else:
        tiles_in_mask = _get_tiles_in_mask(mask, tiling, halo)
        pbar_init(len(tiles_in_mask), "Compute Image Embeddings 2D tiled with mask")
        tile_ids_for_batches = np.array_split(tiles_in_mask, n_batches)
        assert len(tile_ids_for_batches) == n_batches

    for tile_ids in tile_ids_for_batches:
        batched_images = []
        for tile_id in tile_ids:
            tile = tiling.get_block_with_halo(tile_id, list(halo))
            outer_tile = tuple(slice(beg, end) for beg, end in zip(tile.outer_block.begin, tile.outer_block.end))
            tile_input = _to_image(input_[outer_tile])
            batched_images.append(tile_input)

        batched_embeddings, original_sizes, input_sizes = _compute_embeddings_batched(predictor, batched_images)
        _write_batch(features, tile_ids, batched_embeddings, original_sizes, input_sizes)
        pbar_update(len(tile_ids))

    _write_embedding_signature(f, input_, predictor, tile_shape, halo, input_size=None, original_size=None)
    if mask is not None:
        features.attrs["tiles_in_mask"] = tiles_in_mask

    return features


class _BatchProvider:
    def __init__(self, n_slices, n_tiles_per_plane, tiles_in_mask_per_slice, batch_size):
        if tiles_in_mask_per_slice is None:
            self.n_tiles_total = n_slices * n_tiles_per_plane
        else:
            self.n_tiles_total = sum(len(val) for val in tiles_in_mask_per_slice.values())

        self.n_batches = int(np.ceil(self.n_tiles_total / batch_size))
        self.n_slices = n_slices
        self.n_tiles_per_plane = n_tiles_per_plane
        self.tiles_in_mask_per_slice = tiles_in_mask_per_slice
        self.batch_size = batch_size

        # Iter variables.
        self.batch_id = 0
        self.z = 0
        self.tile_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_id >= self.n_batches:
            raise StopIteration

        z_list = list(range(self.n_tiles_per_plane))
        z_tiles = z_list if self.tiles_in_mask_per_slice is None else self.tiles_in_mask_per_slice[self.z]

        slices, tile_ids = [], []
        this_batch_size = 0
        while this_batch_size < self.batch_size:
            if self.tile_id == len(z_tiles):
                self.z += 1
                self.tile_id = 0
                if self.z >= self.n_slices:
                    break
                z_tiles = z_list if self.tiles_in_mask_per_slice is None else self.tiles_in_mask_per_slice[self.z]
                continue

            slices.append(self.z), tile_ids.append(z_tiles[self.tile_id])
            self.tile_id += 1
            this_batch_size += 1

        self.batch_id += 1
        return slices, tile_ids


def _compute_tiled_features_3d(predictor, input_, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask):
    assert input_.ndim == 3

    shape = input_.shape[1:]
    tiling = Blocking([0, 0], shape, tile_shape)
    features = f.require_group("features")
    features.attrs["shape"] = shape
    features.attrs["tile_shape"] = tile_shape
    features.attrs["halo"] = halo

    n_tiles_per_plane = tiling.number_of_blocks
    n_slices = input_.shape[0]

    msg = "Compute Image Embeddings 3D tiled"
    if mask is None:
        n_tiles_total = n_slices * n_tiles_per_plane
        tiles_in_mask_per_slice = None
    else:
        tiles_in_mask_per_slice = {}
        for z in range(n_slices):
            tiles_in_mask_per_slice[z] = _get_tiles_in_mask(mask, tiling, halo, z=z)
        n_tiles_total = sum(len(val) for val in tiles_in_mask_per_slice.values())
        msg += " masked"
    pbar_init(n_tiles_total, msg)

    batch_provider = _BatchProvider(n_slices, n_tiles_per_plane, tiles_in_mask_per_slice, batch_size)
    for slices, tile_ids in batch_provider:
        batched_images = []
        for z, tile_id in zip(slices, tile_ids):
            tile = tiling.get_block_with_halo(tile_id, list(halo))
            outer_tile = (z,) + tuple(
                slice(beg, end) for beg, end in zip(tile.outer_block.begin, tile.outer_block.end)
            )
            tile_input = _to_image(input_[outer_tile])
            batched_images.append(tile_input)

        batched_embeddings, original_sizes, input_sizes = _compute_embeddings_batched(predictor, batched_images)
        _write_batch(
            features, tile_ids, batched_embeddings, original_sizes, input_sizes, slices=slices, n_slices=n_slices
        )
        pbar_update(len(tile_ids))

    if mask is not None:
        features.attrs["tiles_in_mask"] = {str(z): per_slice for z, per_slice in tiles_in_mask_per_slice.items()}

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

    pbar_init(1, "Compute Image Embeddings 2D")
    # Otherwise we have to compute the embeddings.
    predictor.reset_image()
    predictor.set_image(_to_image(input_))
    features = predictor.get_image_embedding().cpu().numpy()
    original_size = predictor.original_size
    input_size = predictor.input_size
    pbar_update(1)

    # Save the embeddings if we have a save_path.
    if save_path is not None:
        _create_dataset_with_data(f, "features", data=features)
        _write_embedding_signature(
            f, input_, predictor, tile_shape=None, halo=None, input_size=input_size, original_size=original_size,
        )

    image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
    return image_embeddings


def _compute_tiled_2d(input_, predictor, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask):
    # Check if the features are already computed.
    if "input_size" in f.attrs:
        features = f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]
        image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
        return image_embeddings

    # Otherwise compute them. Note: saving happens automatically because we
    # always write the features to zarr. If no save path is given we use an in-memory zarr.
    features = _compute_tiled_features_2d(
        predictor, input_, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask=mask
    )
    image_embeddings = {"features": features, "input_size": None, "original_size": None}
    return image_embeddings


def _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update, batch_size):
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
            features = _create_dataset_without_data(f, "features", shape=shape, chunks=chunks, dtype="float32")

    # Initialize the pbar and batches.
    n_slices = input_.shape[0]
    pbar_init(n_slices, "Compute Image Embeddings 3D")
    n_batches = int(np.ceil(n_slices / batch_size))

    for batch_id in range(n_batches):
        z_start = batch_id * batch_size
        z_stop = min(z_start + batch_size, n_slices)

        batched_images, batched_z = [], []
        for z in range(z_start, z_stop):
            # Skip feature computation in case of partial features in non-zero slice.
            if partial_features and np.count_nonzero(features[z]) != 0:
                continue
            tile_input = _to_image(input_[z])
            batched_images.append(tile_input)
            batched_z.append(z)

        batched_embeddings, original_sizes, input_sizes = _compute_embeddings_batched(predictor, batched_images)

        for z, embedding in zip(batched_z, batched_embeddings):
            embedding = embedding.unsqueeze(0)
            if save_features:
                features[z] = embedding.cpu().numpy()
            else:
                features.append(embedding.unsqueeze(0))
            pbar_update(1)

    if save_features:
        _write_embedding_signature(
            f, input_, predictor, tile_shape=None, halo=None,
            input_size=input_sizes[-1], original_size=original_sizes[-1],
        )
    else:
        # Concatenate across the z axis.
        features = torch.cat(features).cpu().numpy()

    image_embeddings = {"features": features, "input_size": input_sizes[-1], "original_size": original_sizes[-1]}
    return image_embeddings


def _compute_tiled_3d(input_, predictor, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask):
    # Check if the features are already computed.
    if "input_size" in f.attrs:
        features = f["features"]
        original_size, input_size = f.attrs["original_size"], f.attrs["input_size"]
        image_embeddings = {"features": features, "input_size": input_size, "original_size": original_size}
        return image_embeddings

    # Otherwise compute them. Note: saving happens automatically because we
    # always write the features to zarr. If no save path is given we use an in-memory zarr.
    features = _compute_tiled_features_3d(
        predictor, input_, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask
    )
    image_embeddings = {"features": features, "input_size": None, "original_size": None}
    return image_embeddings


def _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo):
    """Validate saved embeddings against the requested configuration.

    Returns True if the saved embeddings are stale and should be recomputed (the model or tiling
    configuration changed), False if they can be loaded. Raises if they belong to different image
    data (data signature mismatch).
    """
    # We can have an empty zarr file that was already created to save the embeddings in.
    # In this case the embeddings will be computed and we don't need to perform any checks.
    if "input_size" not in f.attrs:
        return False

    signature = _get_embedding_signature(input_, predictor, tile_shape, halo)
    stale = False
    for key, val in signature.items():
        # A key absent from an older file must not invalidate it (it predates that key).
        if key not in f.attrs or f.attrs[key] == val:
            continue
        # Different image data: surface as an error rather than silently overwriting it.
        if key == "data_signature":
            raise RuntimeError(
                f"Embeddings file {save_path} is invalid due to mismatch in {key}: "
                f"{f.attrs.get(key)} != {val}. Please recompute embeddings in a new file."
            )
        # A version bump alone does not invalidate the embeddings.
        if key == "micro_sam_version":
            warnings.warn(
                f"The signature for {key} in embeddings file {save_path} has a mismatch: "
                f"{f.attrs.get(key)} != {val}. This key was recently added, so your embeddings are likely correct. "
                "But please recompute them if model predictions don't look as expected."
            )
            continue
        # Model or tiling changed: the saved embeddings are stale and must be recomputed.
        stale = True
    return stale


def precompute_image_embeddings(
    predictor: SamPredictor,
    input_: np.ndarray,
    save_path: Optional[Union[str, os.PathLike]] = None,
    lazy_loading: bool = False,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    batch_size: Optional[int] = 1,
    mask: Optional[np.typing.ArrayLike] = None,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
) -> ImageEmbeddings:
    """Compute the image embeddings (output of the encoder) for the input.

    If 'save_path' is given the embeddings will be loaded/saved in a zarr container.

    Args:
        predictor: The Segment Anything predictor.
        input_: The input data. Can be 2 or 3 dimensional, corresponding to an image, volume or timeseries.
        save_path: Path to save the embeddings in a zarr container.
            By default, set to 'None', i.e. the computed embeddings will not be stored locally.
        lazy_loading: Whether to load all embeddings into memory or return an
            object to load them on demand when required. This only has an effect if 'save_path' is given
            and if the input is 3 dimensional. By default, set to 'False'.
        ndim: The dimensionality of the data. If not given will be deduced from the input data.
            By default, set to 'None', i.e. will be computed from the provided `input_`.
        tile_shape: Shape of tiles for tiled prediction. By default prediction is run without tiling.
        halo: Overlap of the tiles for tiled prediction. By default prediction is run without tiling.
        verbose: Whether to be verbose in the computation. By default, set to 'True'.
        batch_size: The batch size for precomputing image embeddings over tiles (or planes). By default, set to '1'.
            Pass None to leave the choice to the backend, which is a single tile / plane for SAM1.
        mask: An optional mask to define areas that are ignored in the computation.
            The mask will be used within tiled embedding computation and tiles that don't contain any foreground
            in the mask will be excluded from the computation. It does not have any effect for non-tiled embeddings.
        pbar_init: Callback to initialize an external progress bar. Must accept number of steps and description.
            Can be used together with pbar_update to handle napari progress bar in other thread.
            To enable using this function within a threadworker.
        pbar_update: Callback to update an external progress bar.

    Returns:
        The image embeddings.
    """
    ndim = input_.ndim if ndim is None else ndim
    # SAM1 has no per-device batch-size lookup, so the automatic choice is the single tile / slice.
    batch_size = 1 if batch_size is None else batch_size

    # Handle the embedding save_path.
    # We don't have a save path, open in memory zarr file to hold tiled embeddings.
    if save_path is None:
        f = _open_embeddings(None)

    # We have a save path and it already exists. Embeddings will be loaded from it,
    # check that the saved embeddings in there match the parameters of the function call.
    elif os.path.exists(save_path):
        f = _open_embeddings(save_path, mode="a")
        if _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo):
            # Stale embeddings (model or tiling changed): truncate and recompute, overwriting them.
            f = _open_embeddings(save_path, mode="w")

    # We have a save path and it does not exist yet. Create the zarr file to which the
    # embeddings will then be saved.
    else:
        f = _open_embeddings(save_path, mode="a")

    _, pbar_init, pbar_update, pbar_close = handle_pbar(verbose, pbar_init, pbar_update)

    if ndim == 2 and tile_shape is None:
        embeddings = _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update)
    elif ndim == 2 and tile_shape is not None:
        embeddings = _compute_tiled_2d(
            input_, predictor, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask=mask
        )
    elif ndim == 3 and tile_shape is None:
        embeddings = _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update, batch_size)
    elif ndim == 3 and tile_shape is not None:
        embeddings = _compute_tiled_3d(
            input_, predictor, tile_shape, halo, f, pbar_init, pbar_update, batch_size, mask=mask
        )
    else:
        raise ValueError(f"Invalid dimesionality {input_.ndim}, expect 2 or 3 dim data.")

    pbar_close()
    return embeddings


def set_precomputed(
    predictor: SamPredictor, image_embeddings: ImageEmbeddings, i: Optional[int] = None, tile_id: Optional[int] = None,
) -> SamPredictor:
    """Set the precomputed image embeddings for a predictor.

    Args:
        predictor: The Segment Anything predictor.
        image_embeddings: The precomputed image embeddings computed by `precompute_image_embeddings`.
        i: Index for the image data. Required if `image` has three spatial dimensions
            or a time dimension and two spatial dimensions.
        tile_id: Index for the tile. This is required if the embeddings are tiled.

    Returns:
        The predictor with set features.
    """
    if tile_id is not None:
        tile_features = image_embeddings["features"][str(tile_id)]
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
