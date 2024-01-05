"""Implements a singleton class for the state of the annotation tools.
The singleton is implemented following the metaclass design described here:
https://itnext.io/deciding-the-best-singleton-approach-in-python-65c61e90cdc4
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from micro_sam.instance_segmentation import AMGBase
from micro_sam.util import ImageEmbeddings
from segment_anything import SamPredictor
from magicgui.widgets import Container


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
    image_embeddings: Optional[ImageEmbeddings] = None
    predictor: Optional[SamPredictor] = None
    image_shape: Optional[Tuple[int, int]] = None

    # amg: needs to be initialized for the automatic segmentation functionality.
    # amg_state: for storing the instance segmentation state for the 3d segmentation tool.
    amg: Optional[AMGBase] = None
    amg_state: Optional[Dict] = None

    # current_track_id, lineage, committed_lineages, tracking_widget:
    # State for the tracking annotator to keep track of lineage information.
    current_track_id: Optional[int] = None
    lineage: Optional[Dict] = None
    committed_lineages: Optional[List[Dict]] = None
    tracking_widget: Optional[Container] = None

    # TODO
    # TODO precompute amg state?
    def initialize_predictor(
        self,
        image,
        model_type,
        device=None,
        predictor=None,
        tile_shape=None,
        halo=None,
    ):
        pass

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
            raise RuntimeError(
                f"Invalid AnnotatorState: {init_sum} / 3 parts of the state "
                "needed for interactive segmentation are initialized."
            )

    def initialized_for_tracking(self):
        have_current_track_id = self.current_track_id is not None
        have_lineage = self.lineage is not None
        have_committed_lineages = self.committed_lineages is not None
        have_tracking_widget = self.tracking_widget is not None
        init_sum = sum((have_current_track_id, have_lineage, have_committed_lineages, have_tracking_widget))
        if init_sum == 4:
            return True
        elif init_sum == 0:
            return False
        else:
            raise RuntimeError(
                f"Invalid AnnotatorState: {init_sum} / 4 parts of the state "
                "needed for tracking are initialized."
            )

    def reset_state(self):
        """Reset state, clear all attributes."""
        self.image_embeddings = None
        self.predictor = None
        self.image_shape = None
        self.amg = None
        self.amg_state = None
        self.current_track_id = None
        self.lineage = None
        self.committed_lineages = None
