import os
import sys
import pooch
from pathlib import Path
from typing import Union, Literal, Optional, Tuple

import zarr
import numpy as np

import torch

from micro_sam.util import get_device
from micro_sam.v2.models._video_predictor import _build_sam2_video_predictor

import sam2
from sam2.build_sam import build_sam2


# NOTE: The model config is expected to be fetched from the module's relative path location.
sys.path.append(Path(sam2.__file__).parents[0])


_DEFAULT_MODEL = "hvit_t"

BACKBONE = "sam2.1"

CFG_PATHS = {
    "sam2.0": {
        "hvit_t": "configs/sam2/sam2_hiera_t.yaml",
        "hvit_s": "configs/sam2/sam2_hiera_s.yaml",
        "hvit_b": "configs/sam2/sam2_hiera_b+.yaml",
        "hvit_l": "configs/sam2/sam2_hiera_l.yaml",
    },
    "sam2.1": {
        "hvit_t": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "hvit_s": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "hvit_b": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "hvit_l": "configs/sam2.1/sam2.1_hiera_l.yaml",
    }
}

SUPPORTED_MODELS = ["hvit_t", "hvit_s", "hvit_b", "hvit_l"]

URLS = {
    "sam2.0": {
        "hvit_t": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
        "hvit_s": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "hvit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "hvit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
    },
    "sam2.1": {
        "hvit_t": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "hvit_s": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "hvit_b": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "hvit_l": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
    },
}

HASHES = {
    "sam2.0": {
        "hvit_t": "65b50056e05bcb13694174f51bb6da89c894b57b75ccdf0ba6352c597c5d1125",
        "hvit_s": "95949964d4e548409021d47b22712d5f1abf2564cc0c3c765ba599a24ac7dce3",
        "hvit_b": "d0bb7f236400a49669ffdd1be617959a8b1d1065081789d7bbff88eded3a8071",
        "hvit_l": "7442e4e9b732a508f80e141e7c2913437a3610ee0c77381a66658c3a445df87b",
    },
    "sam2.1": {
        "hvit_t": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
        "hvit_s": "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38",
        "hvit_b": "a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5",
        "hvit_l": "2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
    },
}


def _get_device(device=None):
    if device is None:
        device = get_device()

    if device == "cuda":
        # NOTE: Adapt global variables to work with flash attentions.
        sam2.modeling.sam.transformer.OLD_GPU = True
        sam2.modeling.sam.transformer.USE_FLASH_ATTN = True
        sam2.modeling.sam.transformer.MATH_KERNEL_ON = True
    elif device == "mps":
        raise ValueError("The scripts have not been tested on MPS device.")

    return device


def _get_checkpoint(model_type=_DEFAULT_MODEL, backbone=BACKBONE):
    # Let's first create a cache directory.
    save_directory = os.path.expanduser(pooch.os_cache("micro_sam2/models"))

    # Download the checkpoint paths if the user does not provide them.
    fname = f"{model_type}_{backbone}"
    pooch.retrieve(
        url=URLS[backbone][model_type],
        known_hash=HASHES[backbone][model_type],
        fname=fname,
        path=save_directory,
        progressbar=True
    )

    # Finally, get the filepath to the cached checkpoint.
    checkpoint_path = os.path.join(save_directory, fname)

    return checkpoint_path


def get_sam2_model(
    model_type: str = _DEFAULT_MODEL,
    device: Optional[Union[torch.device, str]] = None,
    checkpoint_path: Optional[Union[os.PathLike, str]] = None,
    input_type: Literal["images", "videos"] = "images",
    backbone: Literal["sam2.0", "sam2.1"] = BACKBONE,
):
    """Get the Segment Anything 2 (SAM2) model for interactive segmentation of images and videos.

    Args:
        model_type: The choice of size for the vision transformer, eg. `hvit_t`. The default is `hvit_t` model.
        device: The pytorch device.
        checkpoint_path: Filepath to the pretrained model weights.
        input_type: Whether the inputs are images or videos.
        backbone: Whether the SAM2 backbone is initialized with `sam2.0` or `sam2.1` model configuration.
            The default is `sam2.1`.

    Returns:
        The SAM2 model.
    """
    model_cfg = CFG_PATHS[backbone][model_type[:6]]

    device = _get_device(device)

    if input_type == "images":
        _build_segment_anything_2 = build_sam2
    elif input_type == "videos":
        _build_segment_anything_2 = _build_sam2_video_predictor
    else:
        raise ValueError(f"'{input_type}' is not a valid input type.")

    if checkpoint_path is None:
        checkpoint_path = _get_checkpoint(model_type=model_type, backbone=backbone)

    model = _build_segment_anything_2(
        config_file=model_cfg,
        ckpt_path=checkpoint_path,
        device=device,
        mode="eval",
        apply_postprocessing=False,
    )

    return model


def _check_saved_embeddings():
    raise NotImplementedError


def _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update):
    # Check if the embeddings are already cached.
    if save_path is not None and "original_size" in f.attrs:
        # In this case we load the embeddings.
        features = f["features"][:]
        original_size = f.attrs["original_size"]
        image_embeddings = {"features": features, "original_size": original_size}
        # Also set the embeddings.
        set_precomputed(predictor, image_embeddings)
        return image_embeddings

    pbar_init(1, "Compute Image Embeddings 2D")
    # Otherwise we have to compute the embeddings.
    predictor.reset_predictor()

    from micro_sam.util import _to_image
    predictor.set_image(_to_image(input_))
    features = predictor.get_image_embedding().cpu().numpy()
    high_res_features = predictor._features.get("high_res_feats")
    original_size = predictor._orig_hw
    pbar_update(1)

    # Save the embeddings if we have a save_path.
    if save_path is not None:
        from micro_sam.util import _create_dataset_with_data
        _create_dataset_with_data(f, "features", data=features)
        # TODO: Write the embedding signature.

    image_embeddings = {"features": features, "high_res_feats": high_res_features, "original_size": original_size}
    return image_embeddings


def _create_list_dataset_without_data(group, prefix_name, tensors, dtype, z_slices):
    zarr_major_version = int(zarr.__version__.split(".")[0])
    subgroup = group.require_group(prefix_name)

    ds_list = []
    for i, curr_tensor in enumerate(tensors):
        curr_shape = tuple(curr_tensor.shape)
        shape = (z_slices,) + curr_shape
        chunks = (1,) + curr_shape
        name = str(i)

        if name in subgroup:
            ds = subgroup[name]
            if ds.shape != shape:
                raise RuntimeError(f"Invalid shape for {prefix_name}/{name}: expected {shape}, got {ds.shape}")
            if getattr(ds, "chunks", None) is not None and ds.chunks != chunks:
                raise RuntimeError(f"Invalid chunks for {prefix_name}/{name}: expected {chunks}, got {ds.chunks}")
        else:
            if zarr_major_version == 2:
                ds = subgroup.create_dataset(name, shape=shape, dtype=dtype, chunks=chunks, overwrite=False)
            elif zarr_major_version == 3:
                ds = subgroup.create_array(name, shape=shape, chunks=chunks, dtype=dtype, overwrite=False)
            else:
                raise RuntimeError(f"Unsupported zarr version: {zarr_major_version}")

        ds_list.append(ds)

    return ds_list


def _load_list_datasets(group, prefix_name, lazy_loading):
    if prefix_name not in group:
        return []

    subgroup = group[prefix_name]
    out = []
    i = 0
    while str(i) in subgroup:
        ds = subgroup[str(i)]
        out.append(ds if lazy_loading else ds[:])
        i += 1
    return out


@torch.no_grad
def _compute_embeddings_batched_3d(inference_state, predictor, batched_z, batched_images):
    batched_vision_features, batched_pos_enc, batched_backbone_fpn, original_sizes = [], [], [], []

    for image, z_id in zip(batched_images, batched_z):
        # Run the image encoder to extract relevant features
        predictor._get_image_feature(inference_state, frame_idx=z_id, batch_size=1)

        # Let's extract the current 'cached_features' outputs
        _, curr_backbone_out = inference_state["cached_features"][z_id]

        # Store the vision transformer outputs and other stuff.
        batched_vision_features.append(curr_backbone_out["vision_features"])
        batched_pos_enc.append(curr_backbone_out["vision_pos_enc"])
        batched_backbone_fpn.append(curr_backbone_out["backbone_fpn"])
        original_sizes.append(image.shape[:2])

    return batched_vision_features, batched_pos_enc, batched_backbone_fpn, original_sizes


def _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update, batch_size):
    # Check if the embeddings are already fully cached.
    if save_path is not None and "original_size" in f.attrs:
        # In this case we load the embeddings.
        features = f["features"] if lazy_loading else f["features"][:]
        pos_enc = _load_list_datasets(f, "pos_enc", lazy_loading)
        fpn = _load_list_datasets(f, "fpn", lazy_loading)
        original_size = f.attrs["original_size"]
        image_embeddings = {"features": features, "pos_enc": pos_enc, "fpn": fpn, "original_size": original_size}
        return image_embeddings

    # Otherwise we have to compute the embeddings.
    # First check if we have a save path or not and set things up accordingly.
    if save_path is None:
        features, pos_encs, fpns = [], [], []
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
            from micro_sam.util import _create_dataset_without_data
            features = _create_dataset_without_data(f, "features", shape=shape, chunks=chunks, dtype="float32")

    # We create the 'inference_state' object which keeps all important components in memory.
    inference_state = predictor.init_state(volume=input_, ignore_caching_features=True)

    # Initialize the pbar and batches.
    n_slices = input_.shape[0]
    pbar_init(n_slices, "Compute Image Embeddings 3D")
    n_batches = int(np.ceil(n_slices / batch_size))

    pos_enc_dsets, fpn_dsets = None, None
    for batch_id in range(n_batches):
        z_start = batch_id * batch_size
        z_stop = min(z_start + batch_size, n_slices)

        batched_images, batched_z = [], []
        for z in range(z_start, z_stop):
            # Skip feature computation in case of partial features in non-zero slice.
            if partial_features and np.count_nonzero(features[z]) != 0:
                continue

            from micro_sam.util import _to_image
            tile_input = _to_image(input_[z])
            batched_images.append(tile_input)
            batched_z.append(z)

        (
            batched_vision_features, batched_pos_enc, batched_backbone_fpn, original_sizes
        ) = _compute_embeddings_batched_3d(inference_state, predictor, batched_z, batched_images)

        for z, curr_vision_feats, curr_pos_enc, curr_back_fpn in zip(
            batched_z, batched_vision_features, batched_pos_enc, batched_backbone_fpn
        ):
            if curr_vision_feats.ndim == 3:
                curr_vision_feats = curr_vision_feats.unsqueeze(0)

            if save_features:
                features[z] = curr_vision_feats.detach().cpu().numpy()
                if pos_enc_dsets is None:
                    pos_enc_dsets = _create_list_dataset_without_data(
                        f, "pos_enc", curr_pos_enc, dtype="float32", z_slices=n_slices
                    )
                for i, t in enumerate(curr_pos_enc):
                    arr = t.detach().cpu().numpy()
                    if arr.shape != pos_enc_dsets[i][z].shape:
                        breakpoint()
                    pos_enc_dsets[i][z] = t.detach().cpu().numpy()

                if fpn_dsets is None:
                    fpn_dsets = _create_list_dataset_without_data(
                        f, "fpn", curr_back_fpn, dtype="float32", z_slices=n_slices
                    )
                for i, t in enumerate(curr_back_fpn):
                    fpn_dsets[i][z] = t.detach().cpu().numpy()

            else:
                features.append(curr_vision_feats)
                pos_encs.append(curr_pos_enc)
                fpns.append(curr_back_fpn)

            pbar_update(1)

    if save_features:
        pass  # TODO: Write the embedding signature?
    else:
        # Concatenate across the z axis for 'vision_features'.
        features = torch.cat(features).cpu().numpy()

        # Concatenate across the z axis for other features too.
        depth = 3  # Corresponds to the depth of both FPN and Positional Embeddings.
        pos_encs = [torch.stack([p[i] for p in pos_encs]) for i in range(depth)]
        fpns = [torch.stack([p[i] for p in fpns]) for i in range(depth)]

    pos_enc = _load_list_datasets(f, "pos_enc", lazy_loading) if save_features else pos_encs
    fpn = _load_list_datasets(f, "fpn", lazy_loading) if save_features else fpns

    # HACK: I'll store 'original_size' at f.attrs just like that.
    f.attrs["original_size"] = original_sizes[-1]

    image_embeddings = {"features": features, "pos_enc": pos_enc, "fpn": fpn, "original_size": original_sizes[-1]}
    return image_embeddings


def precompute_image_embeddings(
    predictor,
    input_: np.ndarray,
    save_path: Optional[Union[str, os.PathLike]] = None,
    lazy_loading: bool = False,
    ndim: Optional[int] = None,
    tile_shape: Optional[Tuple[int, int]] = None,
    halo: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
    batch_size: int = 1,
    pbar_init: Optional[callable] = None,
    pbar_update: Optional[callable] = None,
):
    """Compute the image embeddings (output of the encoder) for the input.

    If 'save_path' is given the embeddings will be loaded/saved in a zarr container.

    Args:
        ...

    Returns:
        The image embeddings.
    """
    ndim = input_.ndim if ndim is None else ndim

    # Handle the embedding save_path.
    # We don't have a save path, open in memory zarr file to hold tiled embeddings.
    if save_path is None:
        f = zarr.group()

    # We have a save path and it already exists. Embeddings will be loaded from it,
    # check tha tthe saved embeedidng in there match the parameters of the function call.abs
    elif os.path.exists(save_path):
        f = zarr.open(save_path, mode="a")
        # _check_saved_embeddings(input_, predictor, f, save_path, tile_shape, halo)  # TODO: Update this.

    # We have a save path and it does not exist yet. Create the zarr file to which the
    # embeddings will then be saved.
    else:
        f = zarr.open(save_path, mode="a")

    from micro_sam.util import handle_pbar
    _, pbar_init, pbar_update, pbar_close = handle_pbar(verbose, pbar_init, pbar_update)

    if ndim == 2 and tile_shape is None:
        embeddings = _compute_2d(input_, predictor, f, save_path, pbar_init, pbar_update)
    elif ndim == 2 and tile_shape is not None:
        raise NotImplementedError
    elif ndim == 3 and tile_shape is None:
        embeddings = _compute_3d(input_, predictor, f, save_path, lazy_loading, pbar_init, pbar_update, batch_size)
    elif ndim == 3 and tile_shape is not None:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid dimensionality {input_.ndim}, expect 2 or 3 dim data.")

    pbar_close()
    return embeddings


def set_precomputed(
    predictor,
    image_embeddings,
    i=None,
    tile_id=None,
):
    """Set the precomputed image embeddings for a predictor.

    Args:
        ...

    Returns:
        ...
    """
    if tile_id is not None:
        raise NotImplementedError

    try:
        device = predictor.device()  # Works for video predictor.
    except TypeError:
        device = predictor.device  # Otherwise, for image predictor.

    features = image_embeddings["features"]
    assert features.ndim in (4, 5), f"{features.ndim}"
    if features.ndim == 5:
        if i is None:
            raise ValueError("The data is 3D so an index i is needed.")

        # NOTE: The assumption is that 'predictor' is a tuple of the
        # predictor object and the pre-initialized 'inference_state'.
        _predictor, inference_state = predictor

        # TODO: I need to puzzle this together. I can't find an elegant way atm to initialize stuff.
        # We need to figure out the "backbone_out' from 'prepare_features'.

        return _predictor, inference_state

    elif features.ndim == 4:
        if i is not None:
            raise ValueError("The data is 2D so an index is not needed.")

        predictor._features = {"image_embed": features, "high_res_feats": image_embeddings["high_res_feats"]}
        predictor._is_image_set = True
        predictor._orig_hw = image_embeddings["original_size"]
        return predictor
