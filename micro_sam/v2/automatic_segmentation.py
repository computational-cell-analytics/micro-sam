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
    **kwargs,
) -> Tuple[object, object]:
    """Get the SAM2 predictor and the generator for automatic instance segmentation.

    Automatic segmentation with SAM2 uses one of two engines, selected via `segmentation_mode`:
    the decoder-based AIS (with a UniSAM2 decoder from a finetuned model, e.g. 'hvit_t_cells', or a
    `checkpoint`) or the grid-based AMG (no decoder). By default AIS is used when a decoder is
    available and AMG otherwise.

    Args:
        model_type: The SAM2 model. Either a finetuned model with a registered decoder (see
            `micro_sam.v2.util.get_model_names`) or a base backbone combined with `checkpoint`.
        checkpoint: Optional path to a decoder checkpoint to build the UniSAM2 decoder from.
        device: The torch device. By default the best available device is selected.
        segmentation_mode: The segmentation engine, one of 'amg' or 'ais'. By default 'ais' is used
            if a decoder is available, otherwise 'amg'.
        is_tiled: Whether to return a segmenter for in-plane (xy) tiled segmentation.
        kwargs: Keyword arguments for the automatic mask generation (AMG) class.

    Returns:
        The SAM2 predictor (used to precompute embeddings) and the automatic segmentation generator.
    """
    from ..util import get_device
    from ..sam_annotator._state import _get_sam_model
    from .instance_segmentation import get_decoder, get_instance_segmentation_generator

    device = get_device(device)

    # Load a SAM2 image predictor, used to precompute the image embeddings that the decoder / grid
    # prediction reuses. The video predictor for 3d embeddings is built on demand in the front-end.
    predictor, _ = _get_sam_model(
        model_type=model_type, ndim=2, device=device, checkpoint_path=None, decoder_path=None, use_cli=True,
    )

    # Resolve the UniSAM2 decoder if one is requested / available. 'ais' requires a decoder; 'amg'
    # never uses one; 'auto' (None) prefers a decoder and falls back to AMG when none is found.
    decoder = None
    if segmentation_mode != "amg":
        try:
            # Reuse the predictor's already-built image encoder for the decoder (its weights are
            # redefined by the checkpoint's strict load) to avoid building a second SAM2 backbone.
            encoder = getattr(getattr(predictor, "model", predictor), "image_encoder", None)
            decoder = get_decoder(model_type, checkpoint=checkpoint, device=device, encoder=encoder)
        except Exception as e:
            if segmentation_mode == "ais":
                raise
            print(f"Could not load a UniSAM2 decoder for '{model_type}', falling back to AMG: {e}")

    engine = "ais" if decoder is not None else "amg"
    if engine == "amg":  # tag cached embeddings with the model so the AMG state is not reused across models.
        kwargs.setdefault("model_type", model_type)
    segmenter = get_instance_segmentation_generator(
        model=predictor.model, decoder=decoder, is_tiled=is_tiled,
        segmentation_mode=engine, device=device, **kwargs,
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
    num_write_workers: int = 1,
    **generate_kwargs,
) -> np.ndarray:
    """Run automatic instance segmentation for a single input and save the result.

    Args:
        predictor: The SAM2 predictor (see `get_predictor_and_segmenter`), used to precompute the
            image embeddings when `embedding_path` is given.
        segmenter: The automatic instance segmentation generator (see `get_predictor_and_segmenter`).
        input_path: The input image, either a filepath (e.g. tif or a container with `key`) or an array.
        output_path: Optional path to save the segmentation as a tif file.
        embedding_path: Optional path to cache the image embeddings. If given, the embeddings are
            precomputed with the predictor and only the decoder / grid prediction is run on them.
        model_type: The SAM2 model. Used to build the 3d video predictor for embedding precomputation.
        checkpoint: Optional checkpoint for the embedding predictor.
        key: The key for opening `input_path` with `elf.io.open_file` (container files or image stacks).
        ndim: The number of spatial dimensions (2 or 3). By default inferred from the data.
        tile_shape: Shape of the tiles for tiled prediction. By default prediction runs without tiling.
        halo: Overlap of the tiles for tiled prediction.
        mode: The AIS post-processing mode, 'sparse' (flow) or 'dense' (multicut). Ignored for AMG.
        device: The device to run inference on.
        verbose: Whether to print progress.
        batch_size: Explicit tile or slice batch size. Defaults to one; pass None for
            throughput-based automatic selection.
        devices: Inference device or devices. None uses all visible GPUs when the model is on CUDA.
        num_prefetch_workers: Number of input reading and preprocessing threads.
        num_write_workers: Number of output writing threads for full tiled inference.
        generate_kwargs: Additional post-processing parameters forwarded to the segmenter's `generate`.

    Returns:
        The instance segmentation, uint32 array.
    """
    from ..util import load_image_data
    from .util import precompute_image_embeddings
    from .instance_segmentation import UniSAM2InstanceSegmentation, amg_3d_segmentation

    raw = input_path if isinstance(input_path, np.ndarray) else load_image_data(input_path, key=key)
    if ndim is None:
        ndim = raw.ndim

    is_ais = isinstance(segmenter, UniSAM2InstanceSegmentation)

    if is_ais:
        # Decoder-based segmentation: optionally precompute embeddings (reusing the predictor) and run
        # only the decoder on them, otherwise run the full model. A 3d volume needs the video predictor.
        image_embeddings = None
        if embedding_path is not None:
            if ndim == 3:
                from ..sam_annotator._state import _get_sam_model
                emb_predictor, _ = _get_sam_model(
                    model_type=model_type, ndim=3, device=device,
                    checkpoint_path=checkpoint, decoder_path=None, use_cli=True,
                )
            else:
                emb_predictor = predictor
            image_embeddings = precompute_image_embeddings(
                emb_predictor,
                raw,
                save_path=embedding_path,
                ndim=ndim,
                tile_shape=tile_shape,
                halo=halo,
                verbose=verbose,
                lazy_loading=(ndim == 3),
                batch_size=batch_size,
                devices=devices,
                num_prefetch_workers=num_prefetch_workers,
            )
        segmenter.initialize(
            raw,
            ndim=ndim,
            image_embeddings=image_embeddings,
            tile_shape=tile_shape,
            halo=halo,
            batch_size=batch_size,
            devices=devices,
            num_prefetch_workers=num_prefetch_workers,
            num_write_workers=num_write_workers,
        )
        segmentation = segmenter.generate(mode=mode, **generate_kwargs)

    elif ndim == 3:
        # Grid-based AMG on a volume: segment slice-by-slice and stitch across z.
        segmentation = amg_3d_segmentation(
            raw, segmenter, tile_shape=tile_shape, halo=halo, state_save_path=embedding_path,
            verbose=verbose, **generate_kwargs,
        )
    else:
        # Grid-based AMG on a single 2d image; the segmenter computes / caches its own embeddings.
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
