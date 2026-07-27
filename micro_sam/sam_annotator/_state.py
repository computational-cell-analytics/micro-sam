"""Implements a singleton class for the state of the annotation tools.
The singleton is implemented following the metaclass design described here:
https://itnext.io/deciding-the-best-singleton-approach-in-python-65c61e90cdc4
"""

import inspect
from functools import partial
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from segment_anything import SamPredictor

import torch.nn as nn

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm
from napari.layers import Image

from qtpy.QtWidgets import QWidget

import micro_sam
import micro_sam.util as util
from micro_sam.util import _get_sam_model
from micro_sam.v1.instance_segmentation import AutoSegBase, get_decoder
from micro_sam.precompute_state import (
    cache_amg_state, cache_is_state, cache_autoseg_state, _cache_amg_volume_state
)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


@dataclass
class AnnotatorState(metaclass=Singleton):

    # predictor, image_embeddings and image_shape:
    # This needs to be initialized for the interactive segmentation fucntionality.
    image_embeddings: Optional[util.ImageEmbeddings] = None
    predictor: Optional[SamPredictor] = None
    image_shape: Optional[Tuple[int, int]] = None
    image_scale: Optional[Tuple[float, ...]] = None
    ndim: Optional[int] = None
    image_name: Optional[str] = None
    embedding_path: Optional[str] = None
    # Path to an ephemeral on-disk embedding cache created when no save path is given (SAM2 volumes /
    # tiled images), so embeddings stream from disk instead of filling RAM. Removed on reset / exit.
    embedding_tmpdir: Optional[str] = None
    data_signature: Optional[str] = None
    skip_recomputing_embeddings: Optional[bool] = None
    # The un-resolved device request, forwarded to every batched backend so inference stays on the
    # selected device. None means 'auto', i.e. fan out over all visible GPUs.
    inference_devices: Optional[Union[str, List[str]]] = None
    # Whether the tool showed the one-time CPU info popup this session (not reset on recompute).
    cpu_info_shown: Optional[bool] = None

    # The segmenter and its cached state for automatic segmentation. The state contains grid masks
    # for AMG or decoder predictions for AIS and is loaded on demand for SAM2.
    # decoder: for direct prediction of instance segmentation
    automatic_segmenter: Optional[AutoSegBase] = None
    autoseg_state: Optional[Dict] = None
    decoder: Optional[nn.Module] = None

    # current_track_id, lineage, committed_lineages:
    # State for the tracking annotator to keep track of lineage information.
    current_track_id: Optional[int] = None
    lineage: Optional[Dict] = None
    committed_lineages: Optional[List[Dict]] = None

    # Dict to keep track of all widgets, so that we can update their states.
    widgets: Dict[str, QWidget] = field(default_factory=dict)

    # z-range to limit the data being committed in 3d / tracking.
    z_range: Optional[Tuple[int, int]] = None

    # annotator_class
    annotator: Optional["micro_sam.sam_annotator._annotator._AnnotatorBase"] = None

    # Extra options for object classification.
    object_features: Optional[np.ndarray] = None
    seg_ids: Optional[np.ndarray] = None
    # TODO use RF class
    object_rf: Optional[Any] = None
    # TODO use proper class
    segmentation_selection: Optional[Any] = None
    # For batch_object_classifier
    previous_features: Optional[np.ndarray] = None
    previous_labels: Optional[np.ndarray] = None

    # Extra options for pixel classification.
    pixel_features: Optional[np.ndarray] = None
    pixel_grid_shape: Optional[Tuple[int, ...]] = None
    # TODO use RF class
    pixel_rf: Optional[Any] = None

    # Cached AnyUp upsampler, shared by the classification tools when upsampling is enabled.
    anyup_upsampler: Optional[Any] = None

    # Interactive segmentation class for 'micro-sam2'.
    interactive_segmenter: Optional[Any] = None  # TODO: Create a base class and add it here.
    is_sam2: Optional[bool] = None  # Whether this is a SAM1 or SAM2 model.
    is_vfm: Optional[bool] = None  # Whether this is a VFM (DINO / UNI) encoder (classification tools only).

    def initialize_predictor(
        self,
        image_data,
        model_type,
        ndim,
        save_path=None,
        device=None,
        predictor=None,
        decoder=None,
        checkpoint_path=None,
        decoder_path=None,
        tile_shape=None,
        halo=None,
        batch_size=1,
        precompute_autoseg_state=False,
        prefer_decoder=True,
        pbar_init=None,
        pbar_update=None,
        skip_load=True,
        use_cli=False,
    ):
        assert ndim in (2, 3)

        # GUI path inputs are optional. Treat empty and whitespace-only strings like ``None`` so a
        # cleared custom-weights field cannot override a registered model with an invalid path.
        # Keep this normalization at the backend boundary as well as in the widget, since this
        # method is also called directly by the Python API.
        if isinstance(checkpoint_path, str) and not checkpoint_path.strip():
            checkpoint_path = None
        if isinstance(decoder_path, str) and not decoder_path.strip():
            decoder_path = None

        from micro_sam.models.vfm import is_vfm_model
        self.is_sam2 = model_type.startswith("hvit")
        self.is_vfm = is_vfm_model(model_type)

        # Initialize the model if necessary.
        if predictor is None:
            self.predictor, state = _get_sam_model(model_type, ndim, device, checkpoint_path, decoder_path, use_cli)
            if prefer_decoder and "decoder_state" in state and model_type != "vit_b_medical_imaging":
                self.decoder = get_decoder(
                    image_encoder=self.predictor.model.image_encoder,
                    decoder_state=state["decoder_state"],
                    device=device,
                )

        else:
            self.predictor = predictor
            self.decoder = decoder

        # For SAM2, load a UniSAM2 decoder so the automatic segmentation widget can run in decoder
        # mode. This mirrors the v1 decoder-from-checkpoint behavior. Two cases are handled: a custom
        # finetuned checkpoint passed via 'checkpoint_path', or a finetuned model from the download
        # console (e.g. 'hvit_t_cells') whose decoder is downloaded from the SAM2 model registry.
        if self.is_sam2 and prefer_decoder and self.decoder is None:
            from micro_sam.v2.instance_segmentation import get_unisam2_model
            from micro_sam.v2.util import FINETUNED_MODELS, _download_finetuned_sam2_model

            # The decoder is built on the base SAM2 backbone, i.e. the first 6 characters of the
            # name ('hvit_t_cells' -> 'hvit_t'). Resolve where to load the decoder weights from and
            # which encoder ('encoder') the decoder is built on.
            decoder_source, encoder = None, model_type[:6]
            if checkpoint_path is not None:
                decoder_source = checkpoint_path
            elif model_type in FINETUNED_MODELS:
                _, _, decoder_source = _download_finetuned_sam2_model(model_type)
                # Reuse the interactive predictor's already-loaded (finetuned) image encoder as the
                # decoder's encoder instead of rebuilding it from the base backbone. This avoids a
                # redundant download and build of the base backbone. The strict load inside
                # 'get_unisam2_model' still fully redefines these encoder weights from the decoder
                # checkpoint. The 2d image predictor holds the SAM2 model under '.model'. The 3d
                # video predictor is itself a SAM2 model. Use the backbone name if the model
                # exposes no encoder.
                sam2_model = getattr(self.predictor, "model", self.predictor)
                encoder = getattr(sam2_model, "image_encoder", encoder)

            if decoder_source is not None:
                try:
                    # Resolve 'auto'/None to a concrete device so the model is loaded and placed on
                    # the same device as the predictor (torch.load does not accept 'auto').
                    self.decoder = get_unisam2_model(
                        decoder_source, device=util.get_device(device), encoder=encoder,
                    )
                except Exception as e:
                    print(f"Could not load a UniSAM2 decoder from '{decoder_source}': {e}")
                    self.decoder = None

        # The inference devices follow the un-resolved request, not the resolved model placement:
        # None / 'auto' fans out over every visible GPU, an explicit device stays pinned to it.
        inference_devices = None if device in (None, "auto") else device
        self.inference_devices = inference_devices

        # Compute the image embeddings.
        if isinstance(save_path, dict) and "features" in save_path:  # i.e. embeddings are precomputed
            self.image_embeddings = save_path
            self.embedding_path = None  # setting this to 'None' as we do not have embeddings cached.

        else:  # Otherwise, compute the image embeddings.
            _comp_embed_fn = util.get_embedding_function(model_type)

            # The SAM1 embedding function has no 'devices' parameter.
            device_kwargs = {}
            if "devices" in inspect.signature(_comp_embed_fn).parameters:
                device_kwargs["devices"] = inference_devices

            # When no save path is given for a SAM2 volume or a tiled image, cache the embeddings to
            # an ephemeral on-disk zarr instead of holding the whole volume in RAM. All slices at once
            # cost about 200 MB per slice and run out of memory on large volumes. The disk cache lets
            # the consumers stream slices or tiles one at a time. It is removed on 'reset_state' and at
            # process exit. Small non-tiled 2d stays in memory (a single image is cheap).
            needs_disk_cache = self.is_sam2 and (ndim == 3 or tile_shape is not None)
            if needs_disk_cache and not isinstance(save_path, str):
                self._cleanup_embedding_tmpdir()
                save_path = util.make_temp_embedding_path()
                self.embedding_tmpdir = save_path

            # For SAM2 volumes and tiled images, load the embeddings lazily from the zarr so the
            # high-resolution features stay on disk and are streamed one slice / tile at a time.
            # This keeps memory bounded for large volumes (materialising all slices costs
            # ~200 MB/slice and OOMs); it only applies when the embeddings are cached on disk.
            lazy_loading = needs_disk_cache

            self.image_embeddings = _comp_embed_fn(
                predictor=self.predictor,
                input_=image_data,
                save_path=save_path,
                ndim=ndim,
                tile_shape=tile_shape,
                halo=halo,
                batch_size=batch_size,
                verbose=True,
                lazy_loading=lazy_loading,
                pbar_init=pbar_init,
                pbar_update=pbar_update,
                **device_kwargs,
            )
            self.embedding_path = save_path

        # Let's prepare the interactive segmentation class. When the embeddings are tiled (their
        # top-level 'input_size' is None, per-tile sizes live in the tile attrs) we use the tiled
        # variant, which routes prompts to the matching tile-column and stitches the results.
        if self.is_sam2 and ndim == 3:
            is_tiled = self.image_embeddings.get("input_size") is None
            if is_tiled:
                from micro_sam.v2.prompt_based_segmentation import TiledPromptableSegmentation3D
                self.interactive_segmenter = TiledPromptableSegmentation3D(
                    predictor=self.predictor, volume=image_data,
                    volume_embeddings=self.image_embeddings, devices=inference_devices,
                )
            else:
                from micro_sam.v2.prompt_based_segmentation import PromptableSegmentation3D
                self.interactive_segmenter = PromptableSegmentation3D(
                    predictor=self.predictor, volume=image_data,
                    volume_embeddings=self.image_embeddings, device=device,
                )

        # If we have an embedding path the data signature has already been computed,
        # and we can read it from there.
        if save_path is not None and isinstance(save_path, str):
            f = util._open_embeddings(save_path, mode="r")
            self.data_signature = f.attrs["data_signature"]

        # Otherwise we compute it here.
        else:
            self.data_signature = util._compute_data_signature(image_data)

        # Precompute the automatic-segmentation state (if specified). Decoder present -> AIS
        # (decoder predictions), otherwise -> AMG (grid masks); this mirrors the v1 dispatch.
        if precompute_autoseg_state:
            if save_path is None:
                raise RuntimeError("Require a save path to precompute the automatic segmentation state")

            if self.is_sam2:
                self._precompute_autoseg_state_sam2(
                    image_data, ndim, save_path, model_type, pbar_init=pbar_init, pbar_update=pbar_update,
                )
            else:
                self._precompute_autoseg_state_sam1(image_data, ndim, save_path, skip_load)

    def _precompute_autoseg_state_sam1(self, image_data, ndim, save_path, skip_load):
        cache_state = cache_amg_state if self.decoder is None else partial(
            cache_is_state, decoder=self.decoder, skip_load=skip_load,
        )
        if ndim == 2:
            self.automatic_segmenter = cache_state(
                predictor=self.predictor, raw=image_data,
                image_embeddings=self.image_embeddings, save_path=save_path,
            )
        else:
            n_slices = image_data.shape[0] if image_data.ndim == 3 else image_data.shape[1]
            for i in tqdm(range(n_slices), desc="Precompute automatic segmentation state"):
                slice_ = np.s_[i] if image_data.ndim == 3 else np.s_[:, i]
                cache_state(
                    predictor=self.predictor, raw=image_data[slice_],
                    image_embeddings=self.image_embeddings, save_path=save_path, i=i, verbose=False,
                )

    def _precompute_autoseg_state_sam2(
        self, image_data, ndim, save_path, model_type, pbar_init=None, pbar_update=None,
    ):
        model = getattr(self.predictor, "model", self.predictor)
        resolved_model_type = getattr(self.predictor, "model_type", model_type)

        # The decoder / AMG functions drive the bar per tile / slice / z-block, but label it as if it
        # were the actual run. Relabel it with a single clear description while keeping the real unit
        # total, so the precompute phase (after the embeddings) reads meaningfully.
        def relabel_pbar_init(total, _description):
            pbar_init(total, "Precompute automatic segmentation state")
        init_cb = relabel_pbar_init if pbar_init is not None else None

        if self.decoder is not None:  # AIS: decoder over the whole image / volume (per tile / z-block).
            device = next(self.decoder.parameters()).device
            cache_autoseg_state(
                "ais", self.decoder, image_data, self.image_embeddings, save_path, ndim=ndim,
                model_type=resolved_model_type, device=device, devices=self.inference_devices,
                pbar_init=init_cb, pbar_update=pbar_update,
            )
        elif ndim == 2:  # AMG on a single 2d image.
            if pbar_init is not None:
                pbar_init(1, "Precompute automatic segmentation state")
            cache_autoseg_state(
                "amg", model, image_data, self.image_embeddings, save_path, model_type=resolved_model_type,
            )
            if pbar_update is not None:
                pbar_update(1)
        else:  # AMG on a volume: cache the grid-prediction state per slice, reusing the 3d embeddings.
            n_slices = image_data.shape[0] if image_data.ndim == 3 else image_data.shape[1]

            def get_slice(i):
                return image_data[np.s_[i] if image_data.ndim == 3 else np.s_[:, i]]

            _cache_amg_volume_state(
                model, get_slice, n_slices, self.image_embeddings, save_path,
                model_type=resolved_model_type, pbar_init=pbar_init, pbar_update=pbar_update,
            )

    # Get the name of the image layer used to compute the embeddings.
    # If the 'image_name' attribute exists we can just use it.
    # Otherwise, we use the first image layer in the viewer.
    # Note that this case might happen if we load pre-computed embeddings.
    def get_image_name(self, viewer=None):
        if self.image_name is not None:
            return self.image_name
        if viewer is None:
            raise RuntimeError("Did not find the 'image_name' attribute and the viewer was not passed.")
        image_name = None
        for layer in viewer.layers:
            if isinstance(layer, Image):
                image_name = layer.name
                break
        if image_name is None:
            raise RuntimeError("Did not find the 'image_name' attribute and the viewer did not contain an image layer.")
        return image_name

    def initialized_for_interactive_segmentation(self):
        have_image_embeddings = self.image_embeddings is not None
        have_predictor = self.predictor is not None
        have_image_shape = self.image_shape is not None
        init_sum = sum((have_image_embeddings, have_predictor, have_image_shape))
        if init_sum == 3:
            return True
        elif init_sum == 0:
            return False
        else:
            miss_vars = [
                name for name, have_name in zip(
                    ["image_embeddings", "predictor", "image_shape"],
                    [have_image_embeddings, have_predictor, have_image_shape]
                )
                if not have_name
            ]
            miss_vars = ", ".join(miss_vars)
            raise RuntimeError(
                f"Invalid state: the variables {miss_vars} have to be initialized for interactive segmentation."
            )

    def initialized_for_tracking(self):
        have_current_track_id = self.current_track_id is not None
        have_lineage = self.lineage is not None
        have_committed_lineages = self.committed_lineages is not None
        have_tracking_widget = "tracking" in self.widgets
        init_sum = sum((have_current_track_id, have_lineage, have_committed_lineages, have_tracking_widget))
        if init_sum == 4:
            return True
        elif init_sum == 0:
            return False
        else:
            miss_vars = [
                name for name, have_name in zip(
                    ["current_track_id", "lineage", "committed_lineages", "widgets['tracking']"],
                    [have_current_track_id, have_lineage, have_committed_lineages, have_tracking_widget]
                )
                if not have_name
            ]
            miss_vars = ", ".join(miss_vars)
            raise RuntimeError(f"Invalid state: the variables {miss_vars} have to be initialized for tracking.")

    def _cleanup_embedding_tmpdir(self):
        """Remove the ephemeral on-disk embedding cache, if one was created for this image."""
        if self.embedding_tmpdir is not None:
            import shutil
            shutil.rmtree(self.embedding_tmpdir, ignore_errors=True)
            self.embedding_tmpdir = None

    def reset_state(self):
        """Reset state, clear all attributes."""
        self._cleanup_embedding_tmpdir()
        self.image_embeddings = None
        self.predictor = None
        self.image_shape = None
        self.image_scale = None
        self.ndim = None
        self.image_name = None
        self.embedding_path = None
        self.inference_devices = None
        self.automatic_segmenter = None
        self.autoseg_state = None
        self.decoder = None
        self.current_track_id = None
        self.lineage = None
        self.committed_lineages = None
        self.z_range = None
        self.data_signature = None
        self.interactive_segmenter = None
        self.is_sam2 = None
        self.is_vfm = None
        self.anyup_upsampler = None
        # Note: we don't clear the widgets here, because they are fixed for a viewer session.
