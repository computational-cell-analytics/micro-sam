import unittest
import numpy as np

from skimage.data import binary_blobs
from skimage.measure import label

COLOR_CYCLE = ["#00FF00", "#FF0000"]


class TestPromptGenerators(unittest.TestCase):

    def _get_test_data(self):
        data = binary_blobs(length=256)
        labels = label(data)
        return labels

    def test_point_prompt_generator(self):
        from micro_sam.prompt_generators import PointPromptGenerator
        from micro_sam.util import get_cell_center_coordinates

        labels = self._get_test_data()
        label_ids = np.unique(labels)

        mode = "v"  # this gets center points using vigra - regionprops throws error for asymmetrical blobs
        centers, boxes = get_cell_center_coordinates(labels, mode=mode)
        if mode == "v":
            centers.pop(0)   # to avoid the background element

        test_point_pairs = [(1, 0), (1, 1), (2, 4), (3, 9)]
        for (n_pos, n_neg) in test_point_pairs:
            generator = PointPromptGenerator(n_pos, n_neg, dilation_strength=4)
            for label_id in label_ids[:-1]:  # we dodge the last element due to indexing issues for last val
                center, box = centers[label_id], boxes[label_id]  # we start @id0 hence first mask prompts are accessed
                coords, point_labels, _, _ = generator(labels, label_id, center, box)
                coords_ = (np.array([int(coo[0]) for coo in coords]),
                           np.array([int(coo[1]) for coo in coords]))
                mask = labels == label_id + 1  # here we start @id0 hence adding 1 (to ignore background)
                expected_labels = mask[coords_]
                agree = (point_labels == expected_labels)
                # DEBUG: check the points in napari if they don't match
                if not agree.all():
                    print(n_pos, n_neg)
                    # import napari
                    # v = napari.Viewer()
                    # v.add_image(mask)
                    # prompts = v.add_points(
                    #     data=np.array(coords),
                    #     name="prompts",
                    #     properties={"label": point_labels},
                    #     edge_color="label",
                    #     edge_color_cycle=COLOR_CYCLE,
                    #     symbol="o",
                    #     face_color="transparent",
                    #     edge_width=0.5,
                    #     size=5,
                    #     ndim=2
                    # )  # this function helps to view the (colored) background/foreground points
                    # prompts.edge_color_mode = "cycle"
                    # napari.run()

                self.assertTrue(agree.all())


if __name__ == "__main__":
    unittest.main()
