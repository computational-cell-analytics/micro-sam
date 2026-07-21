import unittest

import numpy as np
import torch
import torch.nn.functional as F
from bioimage_cpp.utils import Blocking

from micro_sam.v2.batched_inference import (
    decode_tiled_3d_embeddings, decode_volume_embeddings, run_batched_pipeline,
)
from micro_sam.v2.prompt_based_segmentation import TiledPromptableSegmentation3D


class ArrayWithAttrs:
    def __init__(self, data, **attrs):
        self.data = np.asarray(data)
        self.attrs = attrs
        self.shape = self.data.shape

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)

    def __getitem__(self, index):
        return self.data[index]


class Group(dict):
    def __init__(self, *args, **attrs):
        super().__init__(*args)
        self.attrs = attrs


class FakeEncoder(torch.nn.Module):
    def __init__(self, img_size=8):
        super().__init__()
        self.img_size = img_size


class FakeUNETR(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FakeEncoder()
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.batch_sizes = []

    def forward(self, x):
        self.batch_sizes.append(int(x.shape[0]))
        features = torch.stack(
            [self.encoder(x[:, :, index])[0] for index in range(x.shape[2])],
            dim=2,
        )
        prediction = features.mean(dim=1, keepdim=True)
        prediction = F.interpolate(
            prediction,
            size=(x.shape[2], *x.shape[-2:]),
            mode="trilinear",
            align_corners=False,
        )
        return prediction.repeat(1, 4, 1, 1, 1) * self.scale


class TestBatchedPipeline(unittest.TestCase):
    def test_pipeline_batches_and_writes_all_jobs(self):
        outputs = {}
        progress = []

        def predict(model, items, device):
            self.assertEqual(device, torch.device("cpu"))
            return [value + model for value in items]

        run_batched_pipeline(
            jobs=range(7),
            model_devices=[(3, torch.device("cpu"))],
            batch_sizes=[3],
            load_fn=lambda value: 2 * value,
            predict_fn=predict,
            write_fn=outputs.__setitem__,
            num_prefetch_workers=2,
            update_progress=progress.append,
        )

        self.assertEqual(outputs, {index: 2 * index + 3 for index in range(7)})
        self.assertEqual(sum(progress), 7)


class TestBatchedDecoder(unittest.TestCase):
    def test_volume_decoder_batches_z_blocks(self):
        features = np.arange(8 * 2 * 2 * 2, dtype="float32").reshape(8, 2, 2, 2)
        model = FakeUNETR()

        output = decode_volume_embeddings(
            model,
            {"features": features, "original_size": (8, 8)},
            device="cpu",
            z_block=2,
            z_halo=0,
            batch_size=2,
            num_prefetch_workers=2,
        )

        self.assertEqual(output.shape, (4, 8, 8, 8))
        self.assertEqual(model.batch_sizes, [2, 2])

    def test_tiled_volume_decoder_batches_tiles_and_z_blocks(self):
        features = Group(
            {
                str(tile_id): ArrayWithAttrs(
                    np.full((4, 1, 2, 2, 2), tile_id + 1, dtype="float32"),
                    original_size=(4, 4),
                )
                for tile_id in range(4)
            },
            shape=(4, 8, 8),
            tile_shape=(4, 4),
            halo=(0, 0),
        )
        model = FakeUNETR()

        output = decode_tiled_3d_embeddings(
            model,
            {"features": features},
            device="cpu",
            z_block=2,
            z_halo=0,
            batch_size=2,
            num_prefetch_workers=2,
        )

        self.assertEqual(output.shape, (4, 4, 8, 8))
        self.assertTrue(np.allclose(output[:, :, :4, :4], 1))
        self.assertTrue(np.allclose(output[:, :, :4, 4:], 2))
        self.assertTrue(np.allclose(output[:, :, 4:, :4], 3))
        self.assertTrue(np.allclose(output[:, :, 4:, 4:], 4))
        self.assertEqual(model.batch_sizes, [2, 2, 2, 2])


class FakeInteractiveTile:
    def __init__(self, value):
        self.value = value

    def predict(self, update_progress=None, **kwargs):
        if update_progress is not None:
            update_progress(2)
        return np.full((2, 4, 4), self.value, dtype="uint64")


class TestInteractiveTileScheduling(unittest.TestCase):
    def test_active_tile_columns_are_stitched_after_device_jobs(self):
        segmenter = TiledPromptableSegmentation3D.__new__(TiledPromptableSegmentation3D)
        segmenter.shape = (2, 8, 8)
        segmenter.halo = (0, 0)
        segmenter.tiling = Blocking([0, 0], [8, 8], [4, 4])
        segmenter._predictor_devices = [(None, torch.device("cpu")), (None, torch.device("cpu"))]
        segmenter._segmenters = {
            tile_id: FakeInteractiveTile(tile_id + 1)
            for tile_id in range(4)
        }
        progress = []

        output = segmenter.predict(update_progress=progress.append)

        self.assertTrue(np.all(output[:, :4, :4] == 1))
        self.assertTrue(np.all(output[:, :4, 4:] == 2))
        self.assertTrue(np.all(output[:, 4:, :4] == 3))
        self.assertTrue(np.all(output[:, 4:, 4:] == 4))
        self.assertEqual(sum(progress), 8)


if __name__ == "__main__":
    unittest.main()
