import unittest
import numpy as np

from skimage.data import binary_blobs
from skimage.measure import label


class TestPromptGenerators(unittest.TestCase):

    def _get_test_data(self):
        data = binary_blobs(length=256)
        labels = label(data)
        return labels

    def _debug(self, mask, center, box, coords, point_labels):
        import napari

        v = napari.Viewer()
        v.add_image(mask)
        v.add_points([center], name="center")
        v.add_shapes(
            [np.array(
                [[box[0], box[1]], [box[2], box[3]]]
            )],
            shape_type="rectangle"
        )
        prompts = v.add_points(
            data=np.array(coords),
            name="prompts",
            properties={"label": point_labels},
            edge_color="label",
            edge_color_cycle=["#00FF00", "#FF0000"],
            symbol="o",
            face_color="transparent",
            edge_width=0.5,
            size=5,
            ndim=2
        )  # this function helps to view the (colored) background/foreground points
        prompts.edge_color_mode = "cycle"
        napari.run()

    def test_point_prompt_generator(self):
        from micro_sam.prompt_generators import PointAndBoxPromptGenerator
        from micro_sam.util import get_cell_center_coordinates

        labels = self._get_test_data()
        label_ids = np.unique(labels)[1:]

        centers, boxes = get_cell_center_coordinates(labels)

        test_point_pairs = [(1, 0), (1, 1), (4, 3), (2, 4), (3, 9), (13, 27)]
        for (n_pos, n_neg) in test_point_pairs:
            generator = PointAndBoxPromptGenerator(n_pos, n_neg, dilation_strength=4)
            for label_id in label_ids:
                center, box = centers.get(label_id), boxes.get(label_id)
                coords, point_labels, _, _ = generator(labels, label_id, center, box)
                coords_ = (np.array([int(coo[0]) for coo in coords]),
                           np.array([int(coo[1]) for coo in coords]))
                mask = labels == label_id
                expected_labels = mask[coords_]
                agree = (point_labels == expected_labels)

                # DEBUG: check the points in napari if they don't match
                debug = False
                if not agree.all() and debug:
                    print(n_pos, n_neg)
                    self._debug(mask, center, box, coords, point_labels)

                self.assertTrue(agree.all())


if __name__ == "__main__":
    unittest.main()
