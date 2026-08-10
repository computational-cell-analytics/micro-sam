"""Front-end and CLI helpers for automatic instance segmentation and tracking with SAM2.

This module mirrors the micro-sam v1 `automatic_segmentation` module: it provides only the
CLI / front-end entry points and delegates all inference to the backend engines in
`micro_sam.v2.instance_segmentation` (grid-based AMG and decoder-based AIS with the UniSAM2 model).

- `get_predictor_and_segmenter`: load the SAM2 predictor and the automatic segmentation generator.
- `automatic_instance_segmentation`: run automatic segmentation for a single 2d image or 3d volume.
- `automatic_tracking`: run automatic tracking for a timeseries.
"""

import os
from typing import Optional, Tuple, Union

import numpy as np

import torch

from .util import DEFAULT_MODEL, Devices


def get_predictor_and_segmenter(
    model_type: str = DEFAULT_MODEL,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    device: Optional[Union[str, torch.device]] = None,
    segmentation_mode: Optional[str] = None,
    is_tiled: bool = False,
    ndim: int = 2,
    **kwargs,
) -> Tuple[object, object]:
    """Get the SAM2 predictor and the generator for automatic instance segmentation.

    Automatic segmentation with SAM2 uses one of three engines, selected via `segmentation_mode`:
    the decoder-based AIS (with a UniSAM2 decoder from a finetuned model, e.g. 'hvit_t_cells', or a
    `checkpoint`), the grid-based AMG (no decoder), or APG, which derives candidates from the decoder
    and prompts the interactive branch with them. By default AIS is used when a decoder is available
    and AMG otherwise. APG costs several forward passes per image, so it is opt-in.

    Args:
        model_type: The SAM2 model. Either a finetuned model with a registered decoder (see
            `micro_sam.v2.util.get_model_names`) or a base backbone combined with `checkpoint`.
        checkpoint: Optional path to a decoder checkpoint to build the UniSAM2 decoder from.
        device: The torch device. By default the best available device is selected.
        segmentation_mode: The segmentation engine, one of 'amg', 'ais' or 'apg'. By default 'ais' is
            used if a decoder is available, otherwise 'amg'.
        is_tiled: Whether to return a segmenter for in-plane (xy) tiled segmentation.
        ndim: The number of spatial dimensions the segmenter is built for. Only APG needs to know:
            it propagates its prompts through a volume, which needs the video predictor.
        kwargs: Keyword arguments for the automatic mask generation (AMG) class.

    Returns:
        The SAM2 predictor (used to precompute embeddings) and the automatic segmentation generator.
    """
    from ..util import get_device, _get_sam_model
    from .instance_segmentation import get_decoder, get_instance_segmentation_generator

    # Keep the un-resolved request (None = 'auto') separate from the concrete model placement, so the
    # segmenter can tell 'use all visible GPUs' apart from an explicitly selected single device.
    requested_device = device
    model_device = get_device(device)

    # Load a SAM2 image predictor, used to precompute the image embeddings that the decoder / grid
    # prediction reuses. Volumetric APG propagates its prompts, so it loads the video predictor
    # instead - which also encodes the volume, so this stays one model on the device.
    predictor, _ = _get_sam_model(
        model_type=model_type, ndim=3 if (segmentation_mode == "apg" and ndim == 3) else 2,
        device=model_device, checkpoint_path=None, decoder_path=None, use_cli=True,
    )

    # Resolve the UniSAM2 decoder if the caller requests one or one is available. 'ais' and 'apg'
    # require one. 'amg' never uses one. 'auto' (None) prefers a decoder and uses AMG when none is found.
    decoder = None
    if segmentation_mode != "amg":
        try:
            # Reuse the predictor's already-built image encoder for the decoder (its weights are
            # redefined by the checkpoint's strict load) to avoid building a second SAM2 backbone.
            encoder = getattr(getattr(predictor, "model", predictor), "image_encoder", None)
            decoder = get_decoder(model_type, checkpoint=checkpoint, device=model_device, encoder=encoder)
        except Exception as e:
            if segmentation_mode in ("ais", "apg"):
                raise
            print(f"Could not load a UniSAM2 decoder for '{model_type}', falling back to AMG: {e}")

    # An explicit mode is honored; only the automatic choice falls back to what the decoder allows.
    if segmentation_mode is None:
        engine = "ais" if decoder is not None else "amg"
    else:
        engine = segmentation_mode
    if engine == "amg":  # tag cached embeddings with the model so the AMG state is not reused across models.
        kwargs.setdefault("model_type", model_type)
    segmenter = get_instance_segmentation_generator(
        # The video predictor is the SAM2 model itself, an image predictor wraps one.
        model=getattr(predictor, "model", predictor), decoder=decoder, is_tiled=is_tiled,
        segmentation_mode=engine, device=model_device, inference_device=requested_device, ndim=ndim, **kwargs,
    )
    return predictor, segmenter


def automatic_instance_segmentation(
    predictor,
    segmenter,
    input_path: Union[str, os.PathLike, np.ndarray],
    output_path: Optional[Union[str, os.PathLike]] = None,
    embedding_path: Optional[Union[str, os.PathLike]] = None,
    model_type: str = DEFAULT_MODEL,
    checkpoint: Optional[Union[str, os.PathLike]] = None,
    key: Optional[str] = None,
    ndim: Optional[int] = None,
    tile_shape: Optional[tuple] = None,
    halo: Optional[tuple] = None,
    mode: str = "sparse",
    device: Optional[Union[str, torch.device]] = None,
    verbose: bool = True,
    batch_size: Optional[int] = 1,
    devices: Devices = None,
    num_prefetch_workers: int = 4,
    num_write_workers: int = 2,
    **generate_kwargs,
) -> np.ndarray:
    """Run automatic instance segmentation for a single input and save the result.

    Args:
        predictor: The SAM2 predictor (see `get_predictor_and_segmenter`), used to precompute image
            embeddings for 3d AIS or when `embedding_path` is given.
        segmenter: The automatic instance segmentation generator (see `get_predictor_and_segmenter`).
        input_path: The input image, either a filepath (e.g. tif or a container with `key`) or an array.
        output_path: Optional path to save the segmentation as a tif file.
        embedding_path: Optional path to cache the image embeddings. When given, embeddings are
            persisted and reused. Decoder-based 3d inference always precomputes embeddings first;
            without this path it uses an ephemeral cache.
        model_type: Retained for API compatibility; the loaded predictor determines the embedding model.
        checkpoint: Retained for API compatibility; the loaded predictor already contains its weights.
        key: The key for opening `input_path` with `elf.io.open_file` (container files or image stacks).
        ndim: The number of spatial dimensions (2 or 3). By default inferred from the data.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction runs without tiling.
        halo: Overlap of the tiles for tiled prediction.
        mode: The AIS post-processing mode, 'sparse' (flow) or 'dense' (multicut). Ignored for AMG.
        device: The device to run inference on.
        verbose: Whether to print progress.
        batch_size: The batch size used when running inference for multiple slices and / or tiles.
            Defaults to one. Pass None to select it per device: from the free VRAM for the encoder,
            and benchmarked for the (3d) decoder, which needs the headroom to probe.
        devices: Inference device or devices. None uses all visible GPUs when the model is on CUDA.
        num_prefetch_workers: Number of input reading and preprocessing threads.
        num_write_workers: Number of output writing threads.
        generate_kwargs: Additional post-processing parameters forwarded to the segmenter's `generate`.

    Returns:
        The instance segmentation, uint32 array.
    """
    import shutil

    from ..util import load_image_data, make_temp_embedding_path
    from .util import precompute_image_embeddings
    from .instance_segmentation import UniSAM2InstanceSegmentation, amg_3d_segmentation

    raw = input_path if isinstance(input_path, np.ndarray) else load_image_data(input_path, key=key)
    if ndim is None:
        ndim = raw.ndim

    # AIS and APG share the decoder path but differ in what makes the instances: AIS post-processes the
    # prediction and 'mode' picks how, APG prompts instead. Asked of the segmenter to avoid importing it.
    is_decoder_based = isinstance(segmenter, UniSAM2InstanceSegmentation)
    takes_mode = getattr(segmenter, "_has_postprocessing_mode", True)

    if is_decoder_based:
        # Resolve one device selection for the whole staged workflow. Explicit `devices` takes
        # precedence over the per-call `device`. When the caller omits both, preserve the intent from
        # `get_predictor_and_segmenter` (None means fan out, an explicit device means stay pinned).
        requested_devices = devices if devices is not None else device
        inference_devices = segmenter._inference_devices(requested_devices)

        # Decoder-based 3d segmentation always stages encoder and decoder inference. This avoids
        # re-encoding overlapping z-halo slices and keeps the decoder's peak separate from the encoder.
        # Two-dimensional inference keeps the fused path unless persistent embeddings were requested.
        image_embeddings = None
        temp_embedding_path = None
        try:
            if embedding_path is not None or ndim == 3:
                # The tool streams volumes and tiled images from the zarr. Only small 2d stays in memory.
                is_streamed = ndim == 3 or tile_shape is not None
                # Own the ephemeral store here so it is removed after this input, rather than only at
                # process exit (which piles up one store per input in a multi-input loop).
                effective_path = embedding_path
                if effective_path is None and is_streamed:
                    temp_embedding_path = make_temp_embedding_path()
                    effective_path = temp_embedding_path
                # Reuse the predictor's underlying SAM2 model to avoid a second accelerator-resident backbone.
                emb_predictor = getattr(predictor, "model", predictor) if ndim == 3 else predictor
                image_embeddings = precompute_image_embeddings(
                    emb_predictor,
                    raw,
                    save_path=effective_path,
                    ndim=ndim,
                    tile_shape=tile_shape,
                    halo=halo,
                    verbose=verbose,
                    lazy_loading=is_streamed,
                    batch_size=batch_size,
                    devices=inference_devices,
                    num_prefetch_workers=num_prefetch_workers,
                    num_write_workers=num_write_workers,
                )
            segmenter.initialize(
                raw,
                ndim=ndim,
                image_embeddings=image_embeddings,
                tile_shape=tile_shape,
                halo=halo,
                batch_size=batch_size,
                devices=inference_devices,
                num_prefetch_workers=num_prefetch_workers,
                num_write_workers=num_write_workers,
            )
            if takes_mode:
                segmentation = segmenter.generate(mode=mode, **generate_kwargs)
            else:
                segmentation = segmenter.generate(**generate_kwargs)
        finally:
            # Close all handles. Remove only a store created implicitly for this call.
            if image_embeddings is not None:
                image_embeddings.close()
            image_embeddings = None
            if temp_embedding_path is not None:
                shutil.rmtree(temp_embedding_path, ignore_errors=True)

    elif ndim == 3:
        # Grid-based AMG on a volume: segment slice-by-slice and stitch across z.
        segmentation = amg_3d_segmentation(
            raw, segmenter, tile_shape=tile_shape, halo=halo, state_save_path=embedding_path,
            verbose=verbose, **generate_kwargs,
        )
    else:
        # Grid-based AMG on a single 2d image. The segmenter computes and caches its own embeddings.
        init_kwargs = {"tile_shape": tile_shape, "halo": halo} if tile_shape is not None else {}
        segmenter.initialize(raw, save_path=embedding_path, verbose=verbose, **init_kwargs)
        segmentation = segmenter.generate(**generate_kwargs)

    if output_path is not None:
        import imageio.v3 as imageio
        imageio.imwrite(output_path, segmentation, compression="zlib")
        if verbose:
            print(f"The automatic segmentation results are stored at '{os.path.abspath(str(output_path))}'.")

    return segmentation


def automatic_tracking(
    predictor,
    segmenter,
    input_path: Union[str, os.PathLike, np.ndarray],
    output_path: Optional[Union[str, os.PathLike]] = None,
    key: Optional[str] = None,
    tile_shape: Optional[tuple] = None,
    halo: Optional[tuple] = None,
    mode: str = "sparse",
    device: Optional[Union[str, torch.device]] = None,
    gap_closing: Optional[int] = None,
    min_time_extent: Optional[int] = None,
    verbose: bool = True,
    **generate_kwargs,
):
    """Run automatic tracking for a timeseries.

    Each frame is segmented independently with the segmenter (`automatic_instance_segmentation`), the
    per-frame results are relabeled to globally-unique ids, and the objects are linked across frames
    with Trackastra (see `micro_sam.v1.multi_dimensional_segmentation.track_across_frames`).

    Args:
        predictor: The SAM2 predictor (see `get_predictor_and_segmenter`).
        segmenter: The automatic instance segmentation generator (see `get_predictor_and_segmenter`).
        input_path: The input timeseries, a filepath (tif / container with `key`) or a (T, Y, X) array.
        output_path: Optional folder to save the tracking result in CTC format.
        key: The key for opening `input_path` with `elf.io.open_file` (container files or image stacks).
        tile_shape: Shape of the tiles for tiled per-frame prediction. By default runs without tiling.
        halo: Overlap of the tiles for tiled per-frame prediction.
        mode: The AIS post-processing mode, 'sparse' (flow) or 'dense' (multicut). Ignored for AMG.
        device: The device to run inference on.
        gap_closing: If given, close gaps in the tracks over this many frames.
        min_time_extent: If given, require tracks to span at least this many frames.
        verbose: Whether to print progress.
        generate_kwargs: Additional post-processing parameters forwarded to the segmenter's `generate`.

    Returns:
        The tracking result, a (T, Y, X) array where each object is labeled by its track id.
        The lineages, encoding cell divisions.
    """
    from tqdm import trange

    from ..util import load_image_data
    from ..v1.multi_dimensional_segmentation import track_across_frames

    timeseries = input_path if isinstance(input_path, np.ndarray) else load_image_data(input_path, key=key)
    if timeseries.ndim != 3:
        raise ValueError(f"Automatic tracking expects a (T, Y, X) timeseries, got shape {timeseries.shape}.")

    # Segment every frame independently and relabel so ids do not overlap across frames.
    segmentation = np.zeros(timeseries.shape, dtype="uint32")
    offset = 0
    for t in trange(timeseries.shape[0], desc="Segment frames", disable=not verbose):
        frame_seg = automatic_instance_segmentation(
            predictor=predictor, segmenter=segmenter, input_path=timeseries[t], ndim=2,
            tile_shape=tile_shape, halo=halo, mode=mode, device=device, verbose=False, **generate_kwargs,
        )
        max_id = int(frame_seg.max())
        if max_id == 0:
            continue
        frame_seg[frame_seg != 0] += offset
        offset += max_id
        segmentation[t] = frame_seg

    segmentation, lineage = track_across_frames(
        timeseries=timeseries, segmentation=segmentation, gap_closing=gap_closing,
        min_time_extent=min_time_extent, verbose=verbose, output_folder=output_path,
    )
    return segmentation, lineage
