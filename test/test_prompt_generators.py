import unittest

import numpy as np
import torch

from micro_sam.sample_data import synthetic_data
from skimage.data import binary_blobs
from skimage.measure import label
from skimage.transform import AffineTransform, warp


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

    def test_point_prompt_generator_for_single_object(self):
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
                mask = (labels == label_id)[None]
                coords, point_labels, _, _ = generator(mask, [box], [center])
                coords_ = (
                    np.array([int(coo[0]) for coo in coords[0]]),
                    np.array([int(coo[1]) for coo in coords[0]])
                )
                expected_labels = mask[0][coords_]
                agree = (point_labels == expected_labels)

                # DEBUG: check the points in napari if they don't match
                debug = False
                if not agree.all() and debug:
                    print(n_pos, n_neg)
                    self._debug(mask, center, box, coords, point_labels)

                self.assertTrue(agree.all())

    def test_box_prompt_generator_for_single_object(self):
        from micro_sam.prompt_generators import PointAndBoxPromptGenerator
        from micro_sam.util import get_centers_and_bounding_boxes

        _, labels = synthetic_data(shape=(256, 256))
        label_ids = np.unique(labels)[1:]

        centers, boxes = get_centers_and_bounding_boxes(labels)
        generator = PointAndBoxPromptGenerator(0, 0, dilation_strength=0, get_point_prompts=False, get_box_prompts=True)

        for label_id in label_ids:
            center, box_ = centers.get(label_id), boxes.get(label_id)
            mask = (labels == label_id)[None]
            _, _, box, _ = generator(label, [box_], [center])
            coords = np.where(mask[0])
            expected_box = [coo.min() for coo in coords] + [coo.max() + 1 for coo in coords]
            self.assertEqual(expected_box, list(box[0][0]))

    def test_iterative_prompt_generator(self):
        from micro_sam.prompt_generators import IterativePromptGenerator

        def _get_labels(n_objects=5):
            labels = label(binary_blobs(256))

            ids, sizes = np.unique(labels, return_counts=True)
            ids, sizes = ids[1:], sizes[1:]
            keep_ids = ids[np.argsort(sizes)[::-1][:n_objects]]

            return labels, keep_ids

        def _to_one_hot(labels, keep_ids):
            mask = np.zeros((len(keep_ids),) + labels.shape, dtype="float32")
            for idx, label_id in enumerate(keep_ids):
                mask[idx, labels == label_id] = 1
            return mask

        def _deform_labels(labels):
            scale = np.random.uniform(low=0.9, high=1.1, size=2)
            translation = np.random.rand(2) * 5
            trafo = AffineTransform(scale=scale, translation=translation)
            deformed_labels = warp(labels, trafo.inverse, order=0, preserve_range=True).astype(labels.dtype)
            return deformed_labels

        prompt_gen = IterativePromptGenerator()
        n = 5
        for _ in range(n):
            labels, keep_ids = _get_labels()
            mask = _to_one_hot(labels, keep_ids)

            for n in range(n):
                deformed_labels = _deform_labels(labels)
                obj = _to_one_hot(deformed_labels, keep_ids)

                # import napari
                # v = napari.Viewer()
                # v.add_image(mask)
                # v.add_labels(obj.astype("uint8"))
                # napari.run()

                prompt_mask = torch.from_numpy(mask[:, None]).to(torch.float32)
                prompt_pred = torch.from_numpy(obj[:, None]).to(torch.float32)
                points, point_labels, _, _ = prompt_gen(prompt_mask, prompt_pred)


if __name__ == "__main__":
    unittest.main()
