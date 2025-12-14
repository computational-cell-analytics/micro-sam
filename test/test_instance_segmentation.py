import unittest
from copy import deepcopy

import micro_sam.util as util
import numpy as np
from micro_sam.instance_segmentation import get_predictor_and_decoder

from elf.evaluation.matching import matching
from skimage.draw import disk
from skimage.measure import label


class TestInstanceSegmentation(unittest.TestCase):
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"
    model_type_ais = "vit_t_lm"
    tile_shape = (512, 512)
    halo = (96, 96)

    # create an input image with three objects
    @staticmethod
    def _get_input(shape=(256, 256)):
        mask = np.zeros(shape, dtype="uint8")

        def write_object(center, radius):
            circle = disk(center, radius, shape=shape)
            mask[circle] = 1

        center = tuple(sh // 4 for sh in shape)
        write_object(center, radius=29)

        center = tuple(sh // 2 for sh in shape)
        write_object(center, radius=33)

        center = tuple(3 * sh // 4 for sh in shape)
        write_object(center, radius=35)

        image = mask * 255
        mask = label(mask)
        return mask, image

    @classmethod
    def setUpClass(cls):
        # Input data for normal and tiled segmentation.
        cls.mask, cls.image = cls._get_input()
        cls.large_mask, cls.large_image = cls._get_input(shape=(1024, 1024))

    def _get_model(self, image, model_type, with_tiling=False, checkpoint=None, with_decoder=None):
        if with_decoder:
            predictor, decoder = get_predictor_and_decoder(model_type=model_type, checkpoint_path=checkpoint)
        else:
            predictor = util.get_sam_model(model_type=model_type, checkpoint_path=checkpoint)
        if with_tiling:
            image_embeddings = util.precompute_image_embeddings(
                predictor, image, tile_shape=self.tile_shape, halo=self.halo, verbose=False,
            )
        else:
            image_embeddings = util.precompute_image_embeddings(predictor, image, verbose=False)

        if with_decoder:
            return predictor, decoder, image_embeddings
        else:
            return predictor, image_embeddings

    def _clear_gpu_memory(self):
        device = util.get_device(None)
        if device == "cuda":
            import torch.cuda
            torch.cuda.empty_cache()
        elif device == "mps":
            import torch.mps
            torch.mps.empty_cache()

    def _test_instance_segmentation(
        self, amg_class, create_kwargs={}, generate_kwargs={}, with_decoder=False, with_tiling=False, check_init=True,
    ):
        from micro_sam.instance_segmentation import mask_data_to_segmentation

        if with_tiling:
            mask, image = self.large_mask, self.large_image
        else:
            mask, image = self.mask, self.image

        if with_decoder:
            predictor, decoder, image_embeddings = self._get_model(
                image, self.model_type_ais, with_decoder=True, with_tiling=with_tiling
            )
            create_kwargs = deepcopy(create_kwargs)
            create_kwargs.update({"decoder": decoder})
        else:
            predictor, image_embeddings = self._get_model(image, self.model_type, with_tiling=with_tiling)

        amg = amg_class(predictor, **create_kwargs)
        amg.initialize(image, image_embeddings=image_embeddings, verbose=False)

        predicted = amg.generate(**generate_kwargs)
        predicted = mask_data_to_segmentation(predicted, with_background=True)
        self.assertGreater(matching(predicted, mask, threshold=0.75)["segmentation_accuracy"], 0.99)

        # Check that regenerating the segmentation works.
        predicted2 = amg.generate(**generate_kwargs)
        predicted2 = mask_data_to_segmentation(predicted2, with_background=True)
        self.assertTrue(np.array_equal(predicted, predicted2))

        # Check that serializing and reserializing the state works.
        if check_init:
            state = amg.get_state()
            amg = amg_class(predictor, **create_kwargs)
            amg.set_state(state)
            predicted3 = amg.generate(**generate_kwargs)
            predicted3 = mask_data_to_segmentation(predicted3, with_background=True)
            self.assertTrue(np.array_equal(predicted, predicted3))

    def test_automatic_mask_generator(self):
        from micro_sam.instance_segmentation import AutomaticMaskGenerator
        create_kwargs = dict(points_per_side=10, points_per_batch=16)
        self._test_instance_segmentation(AutomaticMaskGenerator, create_kwargs=create_kwargs)

    def test_tiled_automatic_mask_generator(self):
        from micro_sam.instance_segmentation import TiledAutomaticMaskGenerator
        self._clear_gpu_memory()  # Release all unoccupied cached memory, tiling requires a lot of memory.
        create_kwargs = dict(points_per_side=8)
        generate_kwargs = dict(pred_iou_thresh=0.75)
        self._test_instance_segmentation(
            TiledAutomaticMaskGenerator, create_kwargs=create_kwargs, generate_kwargs=generate_kwargs, with_tiling=True
        )

    def test_instance_segmentation_with_decoder(self):
        from micro_sam.instance_segmentation import InstanceSegmentationWithDecoder
        # VIT_T behaves a bit weirdly, that's why we need these specific settings
        generate_kwargs = dict(foreground_threshold=0.8, min_size=100)
        self._test_instance_segmentation(
            InstanceSegmentationWithDecoder, generate_kwargs=generate_kwargs, with_decoder=True
        )

    def test_tiled_instance_segmentation_with_decoder(self):
        from micro_sam.instance_segmentation import TiledInstanceSegmentationWithDecoder
        generate_kwargs = dict(foreground_threshold=0.8, min_size=100)
        self._test_instance_segmentation(
            TiledInstanceSegmentationWithDecoder, generate_kwargs=generate_kwargs, with_decoder=True, with_tiling=True,
        )

    def test_automatic_prompt_generator(self):
        from micro_sam.instance_segmentation import AutomaticPromptGenerator
        generate_kwargs = dict(foreground_threshold=0.8, min_size=100)
        self._test_instance_segmentation(AutomaticPromptGenerator, generate_kwargs=generate_kwargs, with_decoder=True)

    def test_tiled_automatic_prompt_generator(self):
        from micro_sam.instance_segmentation import TiledAutomaticPromptGenerator
        generate_kwargs = dict(foreground_threshold=0.8, min_size=100)
        self._test_instance_segmentation(
            TiledAutomaticPromptGenerator, generate_kwargs=generate_kwargs,
            with_decoder=True, with_tiling=True, check_init=False,
        )


if __name__ == "__main__":
    unittest.main()
