import unittest

import numpy as np
from skimage.data import binary_blobs
from skimage.measure import label

try:
    from trackastra.model import Trackastra
except ImportError:
    Trackastra = None


class TestMultiDimensionalSegmentation(unittest.TestCase):

    def test_merge_instance_segmentation_3d(self):
        from micro_sam.v1.multi_dimensional_segmentation import merge_instance_segmentation_3d

        n_slices = 5
        data = np.stack(n_slices * binary_blobs(512))
        seg = label(data)

        stacked_seg = []
        offset = 0
        for _ in range(n_slices):
            stack_seg = seg.copy()
            stack_seg[stack_seg != 0] += offset
            offset = stack_seg.max()
            stacked_seg.append(stack_seg)
        stacked_seg = np.stack(stacked_seg)

        merged_seg = merge_instance_segmentation_3d(stacked_seg)

        # Make sure that we don't have any new objects in z + 1.
        # Every object should be merged, since we have full overlap due to stacking.
        ids0 = np.unique(merged_seg[0])
        for z in range(1, n_slices):
            self.assertTrue(np.array_equal(ids0, np.unique(merged_seg[z])))

    def test_merge_instance_segmentation_3d_with_closing(self):
        from micro_sam.v1.multi_dimensional_segmentation import merge_instance_segmentation_3d

        n_slices = 5
        data = np.stack(n_slices * binary_blobs(512))
        seg = label(data)

        stacked_seg = []
        offset = 0
        for z in range(n_slices):
            # Leave the middle slice blank, so that we can check that it
            # gets merged via closing.
            if z == 2:
                stack_seg = np.zeros_like(seg)
            else:
                stack_seg = seg.copy()
                stack_seg[stack_seg != 0] += offset
                offset = stack_seg.max()
            stacked_seg.append(stack_seg)
        stacked_seg = np.stack(stacked_seg)

        merged_seg = merge_instance_segmentation_3d(stacked_seg, gap_closing=1)

        # Make sure that we don't have any new objects in z + 1.
        # Every object should be merged, since we have full overlap due to stacking.
        ids0 = np.unique(merged_seg[0])
        for z in range(1, n_slices):
            self.assertTrue(np.array_equal(ids0, np.unique(merged_seg[z])))

    @unittest.skipIf(Trackastra is None, "Requires trackastra")
    def test_track_across_frames(self):
        from micro_sam.v1.multi_dimensional_segmentation import track_across_frames, get_napari_track_data

        n_slices = 5
        data = binary_blobs(512).astype("uint8")
        seg = label(data)

        stacked_data, stacked_seg = [], []
        offset = 0
        for _ in range(n_slices):
            stack_seg = seg.copy()
            stack_seg[stack_seg != 0] += offset
            offset = stack_seg.max()
            stacked_data.append(data)
            stacked_seg.append(stack_seg)

        stacked_data = np.stack(stacked_data)
        stacked_seg = np.stack(stacked_seg)

        tracks, lineages = track_across_frames(stacked_data, stacked_seg)

        self.assertEqual(tracks.shape, stacked_seg.shape)
        track_ids = set(np.unique(tracks)) - {0}
        lineage_roots = set([next(iter(lin.keys())) for lin in lineages])
        self.assertEqual(track_ids, lineage_roots)

        get_napari_track_data(tracks, lineages)

    def test_extract_tracks_and_lineages_orientation(self):
        # A division where the mother (track 1) appears in frames 0 and 1 and divides into
        # daughters (tracks 2 and 3) in frame 2. The lineage must be oriented by time, i.e. the
        # temporally earliest track is the parent, regardless of the (undirected) parent graph order.
        from micro_sam.v1.multi_dimensional_segmentation import _extract_tracks_and_lineages

        # track_data columns: track_id, timepoint, y, x.
        track_data = np.array([
            [1, 0, 2, 2], [1, 1, 2, 2], [2, 2, 1, 4], [3, 2, 4, 4],
        ], dtype="float64")

        seg = np.zeros((3, 8, 8), dtype="uint16")
        seg[0, 2, 2] = 1
        seg[1, 2, 2] = 1
        seg[2, 1, 4] = 2
        seg[2, 4, 4] = 3

        # The napari parent graph maps children to their parent.
        parent_graph = {2: 1, 3: 1}

        _, lineages = _extract_tracks_and_lineages(seg, track_data, parent_graph)

        division = [lineage for lineage in lineages if any(children for children in lineage.values())]
        self.assertEqual(len(division), 1)
        self.assertEqual(sorted(division[0][1]), [2, 3])


if __name__ == "__main__":
    unittest.main()
