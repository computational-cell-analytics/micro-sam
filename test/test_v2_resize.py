import types

import numpy as np
import torch

from micro_sam.v2.instance_segmentation import ResizeLongestSideWrapper
from micro_sam.v2.prompt_based_segmentation import _crop_to_original_shape
from micro_sam.v2.transforms.resize import (
    ResizeLongestSideTransforms,
    get_preprocess_shape,
    resize_longest_side_and_pad_spatial_numpy,
    resize_longest_side_and_pad_tensor,
)


def test_get_preprocess_shape():
    assert get_preprocess_shape(4, 8, 8) == (4, 8)
    assert get_preprocess_shape(8, 4, 8) == (8, 4)
    assert get_preprocess_shape(3, 5, 10) == (6, 10)


def test_resize_longest_side_tensor_keeps_z_and_pads():
    x = torch.ones((1, 3, 2, 4, 8))
    resized, content_shape = resize_longest_side_and_pad_tensor(x, target_length=8)

    assert resized.shape == (1, 3, 2, 8, 8)
    assert content_shape == (4, 8)
    assert torch.all(resized[..., :4, :] == 1)
    assert torch.all(resized[..., 4:, :] == 0)


def test_resize_longest_side_numpy_keeps_z_and_pads():
    data = np.ones((2, 4, 8), dtype="float32")
    resized, content_shape = resize_longest_side_and_pad_spatial_numpy(data, target_length=8)

    assert resized.shape == (2, 8, 8)
    assert content_shape == (4, 8)
    assert np.all(resized[..., :4, :] == 1)
    assert np.all(resized[..., 4:, :] == 0)

    padded_mask = np.ones((8, 8), dtype=bool)
    assert _crop_to_original_shape(padded_mask, (4, 8)).shape == (4, 8)


def test_video_frame_uses_shared_normalization():
    from micro_sam.v2.normalization import normalize_raw
    from micro_sam.v2.models._video_predictor import _load_img_as_tensor

    raw = np.arange(32, dtype="uint16").reshape(4, 8)
    image, video_height, video_width = _load_img_as_tensor(raw, image_size=8)

    rgb = np.stack([raw] * 3, axis=-1)
    expected = normalize_raw(rgb, axis=(0, 1))
    expected = torch.from_numpy(expected).permute(2, 0, 1)
    expected, _ = resize_longest_side_and_pad_tensor(expected[None], target_length=8)

    assert image.shape == (3, 8, 8)
    assert torch.equal(image, expected[0])
    assert video_height == video_width == 8


def test_image_transform_coordinates_and_mask_crop():
    transform = ResizeLongestSideTransforms(resolution=8, mask_threshold=0.0)
    image = np.full((4, 8, 3), 255, dtype="uint8")
    transformed = transform(image)

    assert transformed.shape == (3, 8, 8)
    assert transform.mean == [0.485, 0.456, 0.406]
    assert transform.std == [0.229, 0.224, 0.225]
    assert torch.equal(transform.transforms(transform.to_tensor(image)), transformed)
    assert torch.all(transformed[:, :4] > 0)
    assert torch.all(transformed[:, 4:] < 0)

    coords = torch.tensor([[4.0, 2.0]])
    transformed_coords = transform.transform_coords(coords, normalize=True, orig_hw=(4, 8))
    assert torch.equal(transformed_coords, coords)

    masks = torch.zeros((1, 1, 8, 8))
    masks[..., :4, :] = 1
    restored = transform.postprocess_masks(masks, orig_hw=(4, 8))
    assert restored.shape == (1, 1, 4, 8)
    assert torch.allclose(restored, torch.ones_like(restored))


class EchoModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = types.SimpleNamespace(img_size=8)
        self.seen = None

    def forward(self, x):
        self.seen = x
        return x[:, :1].repeat(1, 4, 1, 1, 1)


def test_automatic_wrapper_uses_resize_longest():
    model = EchoModel()
    wrapper = ResizeLongestSideWrapper(model, img_size=8)
    x = torch.ones((1, 3, 2, 4, 8))
    output = wrapper(x)

    assert model.seen.shape == (1, 3, 2, 8, 8)
    assert torch.all(model.seen[..., :4, :] == 1)
    assert torch.all(model.seen[..., 4:, :] == 0)
    assert output.shape == (1, 4, 2, 4, 8)
    assert torch.allclose(output, torch.ones_like(output))
