import sys
from pathlib import Path

import numpy as np
import pytest


OPTIMIZATION_ROOT = Path(__file__).parents[1] / "finetuning/v2/evaluation/optimization"
sys.path.insert(0, str(OPTIMIZATION_ROOT))

hybrid = pytest.importorskip("screen_apg_3d_hybrid")


def _cylinder_stack(depth=6, shape=(32, 32)):
    """Two objects that persist through every slice, each labeled 1 in its own slice."""
    stack = np.zeros((depth, *shape), dtype="uint32")
    stack[:, 4:12, 4:12] = 1
    stack[:, 4:12, 20:28] = 2
    return stack


def test_relabel_stack_makes_ids_unique_and_keeps_the_mapping():
    stack = _cylinder_stack(depth=3)
    unique, maps = hybrid.relabel_stack(stack)
    assert unique.max() == 6
    assert len(set(np.unique(unique)) - {0}) == 6
    assert maps[0] == {1: 1, 2: 2} and maps[2] == {5: 1, 6: 2}
    # Every slice keeps its two objects, only renamed.
    for z in range(3):
        assert set(np.unique(unique[z])) - {0} == {2 * z + 1, 2 * z + 2}


@pytest.mark.parametrize("linker", ["greedy", "multicut"])
def test_linking_recovers_two_separated_cylinders(linker):
    stack = _cylinder_stack()
    linked = hybrid.link_slices(stack, linker, beta=0.5, iou_threshold=0.5, min_z_extent=1)
    assert set(np.unique(linked)) == {0, 1, 2}
    # Each object is one id through the whole depth, and the two never share an id.
    for z in range(stack.shape[0]):
        assert len(np.unique(linked[z][4:12, 4:12])) == 1
        assert len(np.unique(linked[z][4:12, 20:28])) == 1
        assert linked[z, 6, 6] != linked[z, 6, 24]
    assert len(set(linked[:, 6, 6])) == 1 and len(set(linked[:, 6, 24])) == 1


def test_greedy_linking_splits_an_object_whose_overlap_falls_below_the_threshold():
    stack = np.zeros((4, 32, 32), dtype="uint32")
    stack[:2, 4:12, 4:12] = 1
    stack[2:, 4:12, 14:22] = 1  # jumps sideways: IoU with the slice before is 0
    linked = hybrid.link_slices(stack, "greedy", beta=0.5, iou_threshold=0.5, min_z_extent=1)
    assert set(np.unique(linked)) == {0, 1, 2}
    assert linked[0, 6, 6] != linked[3, 6, 16]


def test_min_z_extent_drops_short_chains():
    stack = _cylinder_stack(depth=5)
    stack[1:, 4:12, 20:28] = 0  # the second object exists on one slice only
    linked = hybrid.link_slices(stack, "greedy", beta=0.5, iou_threshold=0.5, min_z_extent=2)
    assert set(np.unique(linked)) == {0, 1}
    assert linked[0, 6, 24] == 0


def test_chains_to_prompts_picks_the_slice_of_highest_learned_score():
    stack = _cylinder_stack(depth=3)
    linked = hybrid.link_slices(stack, "greedy", beta=0.5, iou_threshold=0.5, min_z_extent=1)
    instances = []
    for z in range(3):
        instances.append({"z": z, "instance_id": 1, "selection_score": [0.4, 0.9, 0.5][z], "predicted_iou": 0.7,
                          "point": (7.0, 7.0)})
        instances.append({"z": z, "instance_id": 2, "selection_score": [0.8, 0.3, 0.2][z], "predicted_iou": 0.7,
                          "point": (23.0, 7.0)})
    prompts = hybrid.chains_to_prompts(linked, stack, instances, with_masks=True)
    assert prompts["points"].shape == (2, 1, 2) and prompts["point_labels"].shape == (2, 1)
    frames = dict(zip(map(tuple, prompts["points"][:, 0].tolist()), prompts["frames"].tolist()))
    assert frames == {(7.0, 7.0): 1, (23.0, 7.0): 0}
    assert len(prompts["conditioning"]) == 2
    assert all(conditioning["mask"].shape == (32, 32) and conditioning["mask"].sum() == 64
               for conditioning in prompts["conditioning"])


def test_union_prompts_adds_only_uncovered_hybrid_anchors():
    stack = _cylinder_stack(depth=2)
    density = {
        "points": np.array([[[6.0, 6.0]]], dtype="float32"), "point_labels": np.ones((1, 1), dtype="int32"),
        "frames": np.array([0], dtype="int64"),
    }
    hybrid_prompts = {
        "points": np.array([[[7.0, 7.0]], [[23.0, 7.0]]], dtype="float32"),
        "point_labels": np.ones((2, 1), dtype="int32"), "frames": np.array([0, 1], dtype="int64"),
    }
    union = hybrid.union_prompts(density, hybrid_prompts, stack)
    # The first hybrid anchor sits in the instance the density anchor already covers; the second is new.
    assert union["points"].shape == (2, 1, 2)
    assert union["frames"].tolist() == [0, 1]
    assert union["points"][1, 0].tolist() == [23.0, 7.0]
