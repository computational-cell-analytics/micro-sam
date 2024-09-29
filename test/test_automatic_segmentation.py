import unittest

import numpy as np
from skimage.draw import disk
from skimage.measure import label as connected_components

import micro_sam.util as util


class TestAutomaticSegmentation(unittest.TestCase):
    model_type = "vit_t" if util.VIT_T_SUPPORT else "vit_b"
    model_type_ais = "vit_t_lm" if util.VIT_T_SUPPORT else "vit_b_lm"
    tile_shape = (384, 768)
    halo = (96, 96)

    # create an input 2d image with three objects
    @staticmethod
    def _get_2d_inputs(shape):
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
        mask = connected_components(mask)
        return mask, image

    # create an input 2d image with three objects and stack them together
    @classmethod
    def _get_3d_inputs(cls, shape):
        mask, image = cls._get_2d_inputs(shape[-2:])

        # Create volumes by stacking the input image and respective mask.
        volume = np.stack([image] * shape[0])
        labels = np.stack([mask] * shape[0])
        return labels, volume

    @classmethod
    def setUpClass(cls):
        # Input 2d data for normal and tiled segmentation.
        cls.mask, cls.image = cls._get_2d_inputs(shape=(256, 256))
        cls.large_mask, cls.large_image = cls._get_2d_inputs(shape=(768, 768))

        # Input 3d data for normal and tiled segmentation.
        cls.labels, cls.volume = cls._get_3d_inputs(shape=(3, 256, 256))
        cls.large_labels, cls.large_volume = cls._get_3d_inputs(shape=(3, 768, 768))

    def tearDown(self):
        # Release all unoccupied cached memory (eg. tiling requires a lot of memory)
        device = util.get_device(None)
        if device == "cuda":
            import torch.cuda
            torch.cuda.empty_cache()
        elif device == "mps":
            import torch.mps
            torch.mps.empty_cache()

    def test_automatic_mask_generator_2d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        mask, image = self.mask, self.image
        instances = automatic_instance_segmentation(
            input_path=image, model_type=self.model_type, ndim=2, use_amg=True,
            amg_kwargs={"points_per_side": 4}
        )
        self.assertEqual(mask.shape, instances.shape)

    def test_tiled_automatic_mask_generator_2d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        mask, image = self.large_mask, self.large_image
        instances = automatic_instance_segmentation(
            input_path=image,
            model_type=self.model_type,
            ndim=2,
            tile_shape=self.tile_shape,
            halo=self.halo,
            use_amg=True,
            amg_kwargs={"points_per_side": 4}
        )
        self.assertEqual(mask.shape, instances.shape)

    def test_instance_segmentation_with_decoder_2d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        mask, image = self.mask, self.image
        instances = automatic_instance_segmentation(
            input_path=image, model_type=self.model_type_ais, ndim=2
        )
        self.assertEqual(mask.shape, instances.shape)

    def test_tiled_instance_segmentation_with_decoder_2d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        mask, image = self.large_mask, self.large_image
        instances = automatic_instance_segmentation(
            input_path=image, model_type=self.model_type_ais,
            ndim=2, tile_shape=self.tile_shape, halo=self.halo,
        )
        self.assertEqual(mask.shape, instances.shape)

    @unittest.skip("Skipping long running tests by default.")
    def test_automatic_mask_generator_3d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        labels, volume = self.labels, self.volume
        instances = automatic_instance_segmentation(
            input_path=volume, model_type=self.model_type, ndim=3, use_amg=True
        )
        self.assertEqual(labels.shape, instances.shape)

    @unittest.skip("Skipping long running tests by default.")
    def test_tiled_automatic_mask_generator_3d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        labels, volume = self.large_labels, self.large_volume
        instances = automatic_instance_segmentation(
            input_path=volume,
            model_type=self.model_type,
            ndim=3,
            tile_shape=self.tile_shape,
            halo=self.halo,
            use_amg=True,
        )
        self.assertEqual(labels.shape, instances.shape)

    def test_instance_segmentation_with_decoder_3d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        labels, volume = self.labels, self.volume
        instances = automatic_instance_segmentation(
            input_path=volume, model_type=self.model_type_ais, ndim=3,
        )
        self.assertEqual(labels.shape, instances.shape)

    def test_tiled_instance_segmentation_with_decoder_3d(self):
        from micro_sam.automatic_segmentation import automatic_instance_segmentation

        labels, volume = self.large_labels, self.large_volume
        instances = automatic_instance_segmentation(
            input_path=volume, model_type=self.model_type_ais, ndim=3, tile_shape=self.tile_shape, halo=self.halo,
        )
        self.assertEqual(labels.shape, instances.shape)


if __name__ == "__main__":
    unittest.main()
