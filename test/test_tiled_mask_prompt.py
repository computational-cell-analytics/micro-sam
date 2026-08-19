import numpy as np

from micro_sam.v1 import prompt_based_segmentation


class _Features:
    def __init__(self, shape, tile_shape, halo):
        self.attrs = {"shape": shape, "tile_shape": tile_shape, "halo": halo}


def _inner_origin(job):
    return tuple(job["block"].inner_block.begin)


def test_prepare_tiled_mask_prompt_jobs():
    shape, tile_shape, halo = (8, 12), (4, 4), (1, 1)
    mask = np.zeros(shape, dtype=bool)
    mask[1:7, 3:6] = True
    box = np.array([1, 3, 7, 6])
    points = np.array([
        [2, 7],   # A negative correction in the halo of a mask tile.
        [1, 10],  # A negative correction that must not activate its tile.
        [6, 10],  # A positive correction that expands into a new tile.
    ])
    labels = np.array([0, 0, 1])

    _, jobs = prompt_based_segmentation._prepare_tiled_mask_prompt_jobs(
        mask, shape, tile_shape, halo, box=box, points=points, labels=labels
    )

    assert {_inner_origin(job) for job in jobs.values()} == {
        (0, 0), (0, 4), (4, 0), (4, 4), (4, 8)
    }

    seen_points = []
    for job in jobs.values():
        outer = job["block"].outer_block
        outer_begin = np.asarray(outer.begin)
        outer_end = np.asarray(outer.end)

        expected_mask = mask[tuple(slice(beg, end) for beg, end in zip(outer_begin, outer_end))]
        np.testing.assert_array_equal(job["mask"], expected_mask)

        if job["box"] is not None:
            global_box = job["box"] + np.tile(outer_begin, 2)
            expected_box = np.array([
                max(box[0], outer_begin[0]), max(box[1], outer_begin[1]),
                min(box[2], outer_end[0]), min(box[3], outer_end[1]),
            ])
            np.testing.assert_array_equal(global_box, expected_box)

        if job["points"] is not None:
            global_points = job["points"] + outer_begin
            seen_points.extend(map(tuple, global_points.tolist()))

    assert (6, 10) in seen_points
    assert (2, 7) in seen_points
    assert (1, 10) not in seen_points

    expansion_job = next(job for job in jobs.values() if _inner_origin(job) == (4, 8))
    assert not expansion_job["mask"].any()
    assert expansion_job["box"] is None
    np.testing.assert_array_equal(expansion_job["points"], [[3, 3]])
    np.testing.assert_array_equal(expansion_job["labels"], [1])


def test_segment_from_mask_tiled_routes_and_stitches(monkeypatch):
    shape, tile_shape, halo = (8, 12), (4, 4), (1, 1)
    mask = np.zeros(shape, dtype=bool)
    mask[1:7, 3:6] = True
    box = np.array([1, 3, 7, 6])
    points = np.array([[2, 7], [6, 10]])
    labels = np.array([0, 1])
    embeddings = {
        "features": _Features(shape, tile_shape, halo),
        "input_size": None,
        "original_size": None,
    }

    predictor = object()
    calls = []
    current_tile = {}

    def fake_set_precomputed(this_predictor, image_embeddings, i=None, tile_id=None):
        assert this_predictor is predictor
        assert image_embeddings is embeddings
        assert i == 3
        current_tile["id"] = tile_id

    def fake_segment_from_mask(this_predictor, local_mask, **kwargs):
        assert this_predictor is predictor
        calls.append((current_tile["id"], local_mask.copy(), kwargs))
        return np.ones((1,) + local_mask.shape, dtype=bool)

    monkeypatch.setattr(prompt_based_segmentation, "set_precomputed", fake_set_precomputed)
    monkeypatch.setattr(prompt_based_segmentation, "segment_from_mask", fake_segment_from_mask)

    result = prompt_based_segmentation.segment_from_mask_tiled(
        predictor, mask, embeddings, i=3, box=box, points=points, labels=labels
    )

    _, jobs = prompt_based_segmentation._prepare_tiled_mask_prompt_jobs(
        mask, shape, tile_shape, halo, box=box, points=points, labels=labels
    )
    assert [tile_id for tile_id, _, _ in calls] == list(jobs)

    expected = np.zeros((1,) + shape, dtype=bool)
    for job in jobs.values():
        inner = job["block"].inner_block
        glob = tuple(slice(beg, end) for beg, end in zip(inner.begin, inner.end))
        expected[(slice(None),) + glob] = True
    np.testing.assert_array_equal(result, expected)

    for tile_id, local_mask, kwargs in calls:
        job = jobs[tile_id]
        np.testing.assert_array_equal(local_mask, job["mask"])
        np.testing.assert_array_equal(kwargs["box"], job["box"])
        expected_points = None if job["points"] is None else job["points"][:, ::-1]
        np.testing.assert_array_equal(kwargs["points"], expected_points)
        np.testing.assert_array_equal(kwargs["labels"], job["labels"])
        assert kwargs["use_mask"]
        assert not kwargs["use_box"]


def test_tiled_mask_inner_stitching_is_order_independent():
    shape, tile_shape, halo = (8, 8), (4, 4), (1, 1)
    mask = np.ones(shape, dtype=bool)
    _, jobs = prompt_based_segmentation._prepare_tiled_mask_prompt_jobs(
        mask, shape, tile_shape, halo
    )

    predictions = []
    for index, job in enumerate(jobs.values()):
        prediction = np.zeros((1,) + job["mask"].shape, dtype=bool)
        prediction[:, index % 2::2, (index // 2) % 2::2] = True
        predictions.append((job["block"], prediction))

    forward = prompt_based_segmentation._stitch_tiled_mask_predictions(predictions, shape)
    reverse = prompt_based_segmentation._stitch_tiled_mask_predictions(list(reversed(predictions)), shape)
    np.testing.assert_array_equal(forward, reverse)
