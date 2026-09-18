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

    # Kept apart from the placement below, so 'use all visible GPUs' stays distinct from one device.
    requested_device = device
    model_device = get_device(device)

    # Volumetric APG propagates its prompts, so it needs the video predictor, which also encodes.
    predictor, _ = _get_sam_model(
        model_type=model_type, ndim=3 if (segmentation_mode == "apg" and ndim == 3) else 2,
        device=model_device, checkpoint_path=None, decoder_path=None, use_cli=True,
    )

    # 'ais' and 'apg' require a decoder, 'amg' never uses one, and 'auto' prefers one if it loads.
    decoder = None
    if segmentation_mode != "amg":
        try:
            # Reuse the predictor's encoder, so no second SAM2 backbone is built.
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
    if engine == "amg":  # Tags the cached embeddings, so the AMG state is not reused across models.
        kwargs.setdefault("model_type", model_type)
    segmenter = get_instance_segmentation_generator(
        # The video predictor is the SAM2 model itself, an image predictor wraps one.
        model=getattr(predictor, "model", predictor), decoder=decoder, is_tiled=is_tiled,
        segmentation_mode=engine, device=model_device, inference_device=requested_device, ndim=ndim, **kwargs,
    )
    return predictor, segmenter


def _resolve_tiling(spatial_shape, tile_shape, halo, with_z_axis, embedding_path=None):
    """Resolve the tiling for the headless entry points, in the axes their segmenter expects.

    Tiling is supported by every engine, so it is applied by default for images whose in-plane size
    exceeds the cutoff - the model resizes whatever it is given to its encoder patch, so running a
    large image in one piece shrinks the objects far below the scale the model was trained on. This
    shares `micro_sam.v2.util.resolve_default_tiling` with the annotation tools, so the CLI, the
    Python API and the GUI all tile the same image the same way, and all of them reuse the tiling of
    already cached embeddings rather than recomputing them with different settings. Pass an all-zero
    `tile_shape` and `halo` to run untiled regardless of the size.

    Args:
        spatial_shape: The image shape without any channel axis, (y, x) or (z, y, x).
        tile_shape: The requested tile shape, in the axes described by `with_z_axis`.
        halo: The requested tile overlap, matching `tile_shape`'s axes.
        with_z_axis: Whether the segmenter takes the full (z, y, x) rather than just the in-plane
            (y, x). True for decoder-based inference on a volume, which chunks z as well.
        embedding_path: Optional filepath of the cached embeddings, whose tiling is reused.

    Returns:
        The resolved tile shape and halo, None if the run stays untiled. Plus whether the result
        tiles in-plane, which decides the tiled or non-tiled segmenter.

    Raises:
        ValueError: If tiling is turned off with an all-zero tile shape but a real halo is given.
    """
    from .util import resolve_default_tiling, DEFAULT_TILE_Z, DEFAULT_HALO_Z

    # Decide on the in-plane axes alone, then put the z entry back for the volumetric decoder.
    in_plane_tile = None if tile_shape is None else tuple(tile_shape[-2:])
    in_plane_halo = None if halo is None else tuple(halo[-2:])
    in_plane_tile, in_plane_halo = resolve_default_tiling(
        spatial_shape, in_plane_tile, in_plane_halo, embedding_path,
    )
    is_tiled = in_plane_tile is not None

    if not with_z_axis:
        return in_plane_tile, in_plane_halo, is_tiled

    # Without in-plane tiling, keep passing None so z keeps chunking itself (see '_resolve_z_chunk').
    if not is_tiled:
        return None, None, False

    if tile_shape is not None and len(tile_shape) == 3:  # The caller set the z chunk itself.
        z_tile = int(tile_shape[0])
        z_halo = 0 if (halo is None or len(halo) != 3) else int(halo[0])
    else:  # The same z chunk the untiled 3d path picks for itself.
        n_slices = spatial_shape[0]
        z_tile = min(DEFAULT_TILE_Z, n_slices)
        z_halo = DEFAULT_HALO_Z if z_tile < n_slices else 0
    return (z_tile, *in_plane_tile), (z_halo, *in_plane_halo), True


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
        predictor: The SAM2 predictor (see `get_predictor_and_segmenter`), used to precompute the
            image embeddings for the decoder-based engines.
        segmenter: The automatic instance segmentation generator (see `get_predictor_and_segmenter`).
        input_path: The input image, either a filepath (e.g. tif or a container with `key`) or an array.
        output_path: Optional path to save the segmentation as a tif file.
        embedding_path: Optional path to cache the image embeddings. When given, embeddings are
            persisted and reused. The decoder-based engines always precompute embeddings first;
            without this path they are kept in memory (untiled 2d) or in an ephemeral cache. Tiled
            APG without this path encodes each of its tiles / blocks itself.
        model_type: Retained for API compatibility; the loaded predictor determines the embedding model.
        checkpoint: Retained for API compatibility; the loaded predictor already contains its weights.
        key: The key for opening `input_path` with `elf.io.open_file` (container files or image stacks).
        ndim: The number of spatial dimensions (2 or 3). By default inferred from the data: a
            trailing axis of size 2, 3 or 4 is read as channels, anything else as a volume. Pass 2
            to read a 3d array as a multi-channel image (channels-first or channels-last).
        tile_shape: Shape of the tiles for tiled prediction, (y, x). For a 3d AIS or APG volume,
            always the full (z, y, x) instead - it is chunked along z regardless of tiling, so the z
            entry sets that chunk too (its own default when tile_shape is None). AMG segments a
            volume slice by slice, so it stays (y, x) even for a 3d volume. By default the tiling of
            already cached embeddings is reused, else images exceeding the in-plane size threshold
            are tiled with the default tile shape and smaller ones run untiled.
        halo: Overlap of the tiles, matching `tile_shape`'s axes. By default the overlap of already
            cached embeddings is reused, else the default overlap whenever tiling is active.
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
    import warnings

    from ..util import load_image_data, make_temp_embedding_path, prepare_annotation_image
    from .util import precompute_image_embeddings
    from .instance_segmentation import amg_3d_segmentation, retile_instance_segmentation_generator

    raw = input_path if isinstance(input_path, np.ndarray) else load_image_data(input_path, key=key)
    # Normalize the input the same way the annotation tools do: squeeze singleton axes and move a
    # channel axis to the trailing RGB position, so a channels-first (C, Y, X) array is read as an
    # image rather than a volume. The spatial shape below is then the real (y, x) / (z, y, x), which
    # is what the tiling decision and the segmenters need. 'ndim' disambiguates a 3d array; without
    # it a trailing axis of size 2, 3 or 4 is taken as channels and anything else as a volume.
    raw, ndim, is_rgb = prepare_annotation_image(raw, ndim=ndim)
    spatial_shape = raw.shape[:-1] if is_rgb else raw.shape

    # Decoder-based segmenters use this staged path. APG prompts instead of post-processing.
    # Both tiling variants of an engine share this, so it is safe to read before the swap below.
    is_decoder_based = getattr(segmenter, "_is_decoder_based", False)

    # Tiling is decided from the image, which the caller has usually not read yet when it builds the
    # segmenter, so swap in the tiling variant that matches. Only decoder-based inference on a volume
    # takes the z axis too; AMG segments a volume slice by slice and stays in-plane.
    requested_tile_shape = tile_shape
    with_z_axis = ndim == 3 and is_decoder_based
    tile_shape, halo, is_tiled = _resolve_tiling(
        spatial_shape, tile_shape, halo, with_z_axis=with_z_axis, embedding_path=embedding_path,
    )
    swapped = retile_instance_segmentation_generator(segmenter, is_tiled=is_tiled)

    # A generator built by hand (not through the factory) records no build config, so it cannot be
    # swapped. Handing a non-tiled one the tiling we picked ourselves would silently drop it - the
    # in-plane entries are ignored there - and quietly produce the untiled result the default exists
    # to avoid. Run untiled instead, and say why, rather than claiming a tiling that never happens.
    if is_tiled and requested_tile_shape is None and getattr(swapped, "_generator_config", None) is None:
        warnings.warn(
            f"The image is {spatial_shape}, which would be tiled by default, but this segmenter was "
            "built by hand and cannot be swapped for its tiled variant, so the image is segmented in "
            "one piece. Build it with 'get_predictor_and_segmenter' (or pass 'tile_shape' yourself) "
            "to tile."
        )
        tile_shape, halo, is_tiled = None, None, False
    segmenter = swapped

    # Tiled APG encodes every block itself, unless there are cached embeddings to read them from.
    precompute_embeddings = getattr(segmenter, "_precompute_embeddings_in_frontend", True)
    takes_mode = getattr(segmenter, "_has_postprocessing_mode", True)

    if is_decoder_based:
        # One selection for the whole staged workflow; without either the segmenter's intent stands.
        requested_devices = devices if devices is not None else device
        inference_devices = segmenter._inference_devices(requested_devices)

        # The encoder and the decoder are staged: 3d encodes its z-halo once and the peaks do not add up,
        # and every run decodes from embeddings, so the result does not depend on whether they are
        # cached and matches the annotation tools (a joint encoder-decoder pass would pad border tiles).
        image_embeddings = None
        temp_embedding_path = None
        try:
            if embedding_path is not None or precompute_embeddings:
                # The tool streams volumes and tiled images from the zarr. Only small 2d stays in memory.
                is_streamed = ndim == 3 or tile_shape is not None
                # Owned here, so a multi-input loop does not pile up one store per input.
                effective_path = embedding_path
                if effective_path is None and is_streamed:
                    temp_embedding_path = make_temp_embedding_path()
                    effective_path = temp_embedding_path
                # The underlying SAM2 model, so no second backbone lands on the device.
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
            if precompute_embeddings:
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
            else:
                # The blocks are tiled in-plane like the embeddings, so each reads its tile's embeddings.
                segmenter.initialize(
                    raw, ndim=ndim, tile_shape=tile_shape, halo=halo, verbose=verbose,
                    image_embeddings=image_embeddings,
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
        tile_shape: Shape of the tiles for tiled per-frame prediction, (y, x). By default frames
            exceeding the in-plane size threshold are tiled with the default tile shape.
        halo: Overlap of the tiles for tiled per-frame prediction. By default the default overlap
            is used whenever tiling is active.
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
    from .instance_segmentation import retile_instance_segmentation_generator

    timeseries = input_path if isinstance(input_path, np.ndarray) else load_image_data(input_path, key=key)
    if timeseries.ndim != 3:
        raise ValueError(f"Automatic tracking expects a (T, Y, X) timeseries, got shape {timeseries.shape}.")

    # Every frame has the same shape, so resolve the tiling and swap the segmenter once here rather
    # than per frame, then pass the resolved settings down.
    tile_shape, halo, is_tiled = _resolve_tiling(timeseries.shape[1:], tile_shape, halo, with_z_axis=False)
    segmenter = retile_instance_segmentation_generator(segmenter, is_tiled=is_tiled)

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
