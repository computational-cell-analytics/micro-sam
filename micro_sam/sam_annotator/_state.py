from dataclasses import dataclass
from typing import Optional

from micro_sam.instance_segmentation import AMGBase
from micro_sam.util import ImageEmbeddings
from segment_anything import SamPredictor


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# TODO the shape should also go in here
@dataclass
class AnnotatorState(metaclass=Singleton):
    image_embeddings: Optional[ImageEmbeddings] = None
    predictor: Optional[SamPredictor] = None
    amg: Optional[AMGBase] = None
