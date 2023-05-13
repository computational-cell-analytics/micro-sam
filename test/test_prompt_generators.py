import unittest
import numpy as np

from skimage.data import binary_blobs
from skimage.measure import label


class TestPromptGenerators(unittest.TestCase):

    def _get_test_data(self):
        data = binary_blobs(length=256)
        labels = label(data)
        return labels

    def test_point_prompt_generator(self):
        from micro_sam.prompt_generators import PointPromptGenerator
        from micro_sam.util import get_cell_center_coordinates

        labels = self._get_test_data()
        label_ids = np.unique(labels)[1:]
        centers, boxes = get_cell_center_coordinates(labels)

        test_point_pairs = [(1, 0), (1, 1), (2, 4), (3, 9)]
        for (n_pos, n_neg) in test_point_pairs:
            generator = PointPromptGenerator(n_pos, n_neg, dilation_strength=4)
            for label_id in label_ids:
                center, box = centers[label_id], boxes[label_id]
                coords, point_labels, _, _ = generator(labels, label_id, center, box)
                coords_ = (np.array([int(coo[0]) for coo in coords]),
                           np.array([int(coo[1]) for coo in coords]))
                mask = labels == label_id
                expected_labels = mask[coords_]
                agree = (point_labels == expected_labels)
                # DEBUG: check the points in napari if they don't match
                if not agree.all():
                    print(n_pos, n_neg)
                    # import napari
                    # v = napari.Viewer()
                    # v.add_image(mask)
                    # v.add_points(coords)
                    # napari.run()
                self.assertTrue(agree.all())


if __name__ == "__main__":
    unittest.main()
