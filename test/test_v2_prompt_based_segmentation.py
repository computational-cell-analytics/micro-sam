import numpy as np
from bioimage_cpp.utils import Blocking

from micro_sam.v2.prompt_based_segmentation import (
    PromptableSegmentation3D,
    TiledPromptableSegmentation3D,
)


class FakeTileSegmenter:
    def add_point_prompts(self, **kwargs):
        pass

    def add_box_prompts(self, **kwargs):
        pass


def make_tiled_segmenter():
    segmenter_cls = TiledPromptableSegmentation3D
    segmenter = segmenter_cls.__new__(segmenter_cls)
    segmenter.shape = (8, 16, 16)
    segmenter.halo = (0, 0)
    segmenter.tiling = Blocking([0, 0], [16, 16], [8, 8])
    segmenter._segmenters = {}

    def get_segmenter(tile_id):
        return segmenter._segmenters.setdefault(tile_id, FakeTileSegmenter())

    segmenter._get_segmenter = get_segmenter
    return segmenter


def test_promptable_segmentation_3d_progress_total():
    segmenter = PromptableSegmentation3D.__new__(PromptableSegmentation3D)
    segmenter.volume = np.zeros((8, 16, 16), dtype="uint8")

    assert segmenter.get_progress_total() == 8
    assert segmenter.get_progress_total((2, 5)) == 4


def test_tiled_promptable_segmentation_3d_progress_total():
    segmenter = make_tiled_segmenter()

    assert segmenter.get_progress_total() == 0

    segmenter.add_point_prompts(frame_ids=0, points=[[1, 1]], point_labels=[1])
    assert segmenter.get_progress_total() == 8
    assert segmenter.get_progress_total((2, 5)) == 4

    segmenter.add_point_prompts(frame_ids=0, points=[[1, 9]], point_labels=[1])
    assert segmenter.get_progress_total() == 16
    assert segmenter.get_progress_total((2, 5)) == 8


def test_tiled_promptable_segmentation_3d_box_progress_total():
    segmenter = make_tiled_segmenter()

    segmenter.add_box_prompts(frame_ids=0, boxes=[np.array([1, 1, 7, 9])])

    assert len(segmenter._segmenters) == 2
    assert segmenter.get_progress_total() == 16
