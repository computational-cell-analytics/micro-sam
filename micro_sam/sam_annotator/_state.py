"""Implements a singleton class for the state of the annotation tools.
The singleton is implemented following the metaclass design described here:
https://itnext.io/deciding-the-best-singleton-approach-in-python-65c61e90cdc4
"""

from functools import partial
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import zarr
import numpy as np

from napari.layers import Image
from qtpy.QtWidgets import QWidget

import torch.nn as nn

import micro_sam
import micro_sam.util as util
from micro_sam.v1.util import get_sam_model, precompute_image_embeddings
from micro_sam.v1.instance_segmentation import AMGBase, get_decoder
from micro_sam.precompute_state import cache_amg_state, cache_is_state

from segment_anything import SamPredictor

try:
    from napari.utils import progress as tqdm
except ImportError:
    from tqdm import tqdm


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# TODO: this should be refactored once we have decided on which models to support.
# (Likely only SAM2 models)
def _get_sam_model(model_type, ndim, device, checkpoint_path, decoder_path, use_cli):
    from micro_sam.v1.models.vfm import is_vfm_model, get_vfm_model
    if is_vfm_model(model_type):  # VFM encoders (DINO / UNI) for the classification tools.
        encoder = get_vfm_model(model_type, device=device, checkpoint_path=checkpoint_path)
        return encoder, {}

    if model_type.startswith("h"):  # i.e. SAM2 models.
        from micro_sam.v2.util import get_sam2_model

        # 'device=None' lets 'get_sam2_model' auto-detect the best device (cuda > mps > cpu);
        # an explicit device (e.g. from the '--device' CLI argument) is forwarded and honored.
        if ndim == 2:  # Get the SAM2 model and prepare the image predictor.
            model = get_sam2_model(model_type=model_type, input_type="images", device=device)
            # Prepare the SAM2 predictor.
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            predictor = SAM2ImagePredictor(model)
            # The video predictor gets these set in 'get_sam2_model'; set them here on the image
            # predictor too, so the embedding signature can be written when caching embeddings.
            predictor.model_type = model_type
            predictor.model_name = model_type
        elif ndim == 3:  # Get SAM2 video predictor
            predictor = get_sam2_model(model_type=model_type, input_type="videos", device=device)
        else:
            raise ValueError
        state = {}

    else:
        def progress_bar_factory(model_type):
            pbar = tqdm(desc=f"Downloading '{model_type}'. This may take a while")
            return pbar

        predictor, state = get_sam_model(
            device=device, model_type=model_type,
            checkpoint_path=checkpoint_path, decoder_path=decoder_path, return_state=True,
            progress_bar_factory=None if use_cli else progress_bar_factory,
        )

    return predictor, state


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
    data_signature: Optional[str] = None
    skip_recomputing_embeddings: Optional[bool] = None

    # amg: needs to be initialized for the automatic segmentation functionality.
    # amg_state: for storing the instance segmentation state for the 3d segmentation tool.
    # decoder: for direct prediction of instance segmentation
    amg: Optional[AMGBase] = None
    amg_state: Optional[Dict] = None
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
        precompute_amg_state=False,
        prefer_decoder=True,
        pbar_init=None,
        pbar_update=None,
        skip_load=True,
        use_cli=False,
    ):
        assert ndim in (2, 3)
        from micro_sam.v1.models.vfm import is_vfm_model
        self.is_sam2 = model_type.startswith("h")
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
            from micro_sam.v2.automatic_segmentation import get_unisam2_model
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
                # redundant base-backbone download/build; the strict load inside 'get_unisam2_model'
                # still fully (re)defines these encoder weights from the decoder checkpoint. The 2d
                # image predictor holds the SAM2 model under '.model'; the 3d video predictor is
                # itself a SAM2 model. Fall back to the backbone name if no encoder is exposed.
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

        # Compute the image embeddings.
        if isinstance(save_path, dict) and "features" in save_path:  # i.e. embeddings are precomputed
            self.image_embeddings = save_path
            self.embedding_path = None  # setting this to 'None' as we do not have embeddings cached.

        else:  # Otherwise, compute the image embeddings.
            if self.is_vfm:
                from micro_sam.v1.models.vfm import precompute_vfm_embeddings as _comp_embed_fn
            elif self.is_sam2:
                from micro_sam.v2.util import precompute_image_embeddings as _comp_embed_fn
            else:
                _comp_embed_fn = precompute_image_embeddings

            # For SAM2 volumes, load the embeddings lazily from the zarr so the high-resolution
            # per-slice features stay on disk and are streamed one slice at a time during tracking.
            # This keeps memory bounded for large volumes (materialising all slices costs
            # ~200 MB/slice and OOMs); it only applies when the embeddings are cached on disk.
            lazy_loading = self.is_sam2 and ndim == 3 and isinstance(save_path, str)

            self.image_embeddings = _comp_embed_fn(
                predictor=self.predictor,
                input_=image_data,
                save_path=save_path,
                ndim=ndim,
                tile_shape=tile_shape,
                halo=halo,
                verbose=True,
                lazy_loading=lazy_loading,
                pbar_init=pbar_init,
                pbar_update=pbar_update,
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
                    volume_embeddings=self.image_embeddings, device=device,
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
            f = zarr.open(save_path, mode="r")
            self.data_signature = f.attrs["data_signature"]

        # Otherwise we compute it here.
        else:
            self.data_signature = util._compute_data_signature(image_data)

        # Precompute the amg state (if specified).
        if precompute_amg_state:
            if save_path is None:
                raise RuntimeError("Require a save path to precompute the amg state")

            cache_state = cache_amg_state if self.decoder is None else partial(
                cache_is_state, decoder=self.decoder, skip_load=skip_load,
            )

            if ndim == 2:
                self.amg = cache_state(
                    predictor=self.predictor,
                    raw=image_data,
                    image_embeddings=self.image_embeddings,
                    save_path=save_path
                )
            else:
                n_slices = image_data.shape[0] if image_data.ndim == 3 else image_data.shape[1]
                for i in tqdm(range(n_slices), desc="Precompute amg state"):
                    slice_ = np.s_[i] if image_data.ndim == 3 else np.s_[:, i]
                    cache_state(
                        predictor=self.predictor,
                        raw=image_data[slice_],
                        image_embeddings=self.image_embeddings,
                        save_path=save_path, i=i, verbose=False,
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

    def reset_state(self):
        """Reset state, clear all attributes."""
        self.image_embeddings = None
        self.predictor = None
        self.image_shape = None
        self.image_scale = None
        self.ndim = None
        self.image_name = None
        self.embedding_path = None
        self.amg = None
        self.amg_state = None
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
