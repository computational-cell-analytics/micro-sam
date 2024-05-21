"""Implements a singleton class for the state of the annotation tools.
The singleton is implemented following the metaclass design described here:
https://itnext.io/deciding-the-best-singleton-approach-in-python-65c61e90cdc4
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch.nn as nn
import zarr

import micro_sam.util as util
from micro_sam.instance_segmentation import AMGBase, get_decoder
from micro_sam.precompute_state import cache_amg_state, cache_is_state
from qtpy.QtWidgets import QWidget

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


@dataclass
class AnnotatorState(metaclass=Singleton):

    # predictor, image_embeddings and image_shape:
    # This needs to be initialized for the interactive segmentation fucntionality.
    image_embeddings: Optional[util.ImageEmbeddings] = None
    predictor: Optional[SamPredictor] = None
    image_shape: Optional[Tuple[int, int]] = None
    embedding_path: Optional[str] = None
    data_signature: Optional[str] = None

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
        tile_shape=None,
        halo=None,
        precompute_amg_state=False,
        prefer_decoder=True,
        pbar_init=None,
        pbar_update=None,
    ):
        assert ndim in (2, 3)

        # Initialize the model if necessary.
        if predictor is None:
            self.predictor, state = util.get_sam_model(
                device=device, model_type=model_type,
                checkpoint_path=checkpoint_path, return_state=True
            )
            if prefer_decoder and "decoder_state" in state:
                self.decoder = get_decoder(
                    image_encoder=self.predictor.model.image_encoder,
                    decoder_state=state["decoder_state"],
                    device=device,
                )

        else:
            self.predictor = predictor
            self.decoder = decoder

        # Compute the image embeddings.
        self.image_embeddings = util.precompute_image_embeddings(
            predictor=self.predictor,
            input_=image_data,
            save_path=save_path,
            ndim=ndim,
            tile_shape=tile_shape,
            halo=halo,
            verbose=True,
            pbar_init=pbar_init,
            pbar_update=pbar_update,
        )
        self.embedding_path = save_path
        # If we have an embedding path the data signature has already been computed,
        # and we can read it from there.
        if save_path is not None:
            with zarr.open(save_path, "r") as f:
                self.data_signature = f.attrs["data_signature"]
        # Otherwise we compute it here.
        else:
            self.data_signature = util._compute_data_signature(image_data)

        # Precompute the amg state (if specified).
        if precompute_amg_state:
            if save_path is None:
                raise RuntimeError("Require a save path to precompute the amg state")

            cache_state = cache_amg_state if self.decoder is None else partial(
                cache_is_state, decoder=self.decoder, skip_load=True,
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
        self.embedding_path = None
        self.amg = None
        self.amg_state = None
        self.decoder = None
        self.current_track_id = None
        self.lineage = None
        self.committed_lineages = None
        self.z_range = None
        self.data_signature = None
        # Note: we don't clear the widgets here, because they are fixed for a viewer session.
