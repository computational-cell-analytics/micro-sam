import unittest
import numpy as np

from micro_sam.sample_data import synthetic_data


class TestPromptGenerators(unittest.TestCase):

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
        from micro_sam.util import get_centers_and_bounding_boxes

        _, labels = synthetic_data(shape=(256, 256))
        label_ids = np.unique(labels)[1:]

        centers, boxes = get_centers_and_bounding_boxes(labels)

        test_point_pairs = [(1, 0), (1, 1), (4, 3), (2, 4), (3, 9), (13, 27)]
        for (n_pos, n_neg) in test_point_pairs:
            generator = PointAndBoxPromptGenerator(n_pos, n_neg, dilation_strength=4)
            for label_id in label_ids:
                center, box = centers.get(label_id), boxes.get(label_id)
                _label = (labels == label_id)
                coords, point_labels, _, _ = generator(_label, box, center)
                coords_ = (np.array([int(coo[0]) for coo in coords]),
                           np.array([int(coo[1]) for coo in coords]))
                expected_labels = _label[coords_]
                agree = (point_labels == expected_labels)

                # DEBUG: check the points in napari if they don't match
                debug = False
                if not agree.all() and debug:
                    print(n_pos, n_neg)
                    self._debug(_label, center, box, coords, point_labels)

                self.assertTrue(agree.all())

    def test_box_prompt_generator(self):
        from micro_sam.prompt_generators import PointAndBoxPromptGenerator
        from micro_sam.util import get_centers_and_bounding_boxes

        _, labels = synthetic_data(shape=(256, 256))
        label_ids = np.unique(labels)[1:]

        centers, boxes = get_centers_and_bounding_boxes(labels)
        generator = PointAndBoxPromptGenerator(0, 0, dilation_strength=0, get_point_prompts=False, get_box_prompts=True)

        for label_id in label_ids:
            center, box_ = centers.get(label_id), boxes.get(label_id)
            _label = (labels == label_id)
            _, _, box, _ = generator(_label, box_, center)
            coords = np.where(_label)
            expected_box = [coo.min() for coo in coords] + [coo.max() + 1 for coo in coords]
            self.assertEqual(expected_box, list(box[0]))


if __name__ == "__main__":
    unittest.main()
