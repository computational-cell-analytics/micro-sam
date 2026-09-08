import unittest.mock

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from torch_em.loss import DiceLoss

from micro_sam.v2.loss import DirectedDistanceLoss
from micro_sam.v2.models.util import CustomActivation
from micro_sam.v2.transforms.labels import (
    FOREGROUND_IGNORE_VALUE,
    _JointLabelTransform,
    DirectedPerObjectBoundaryDistanceTransform,
    GeodesicHybridDistanceTransform,
    object_boundaries,
)


def _labels():
    labels = np.zeros((24, 24), dtype="uint32")
    labels[3:12, 3:12] = 1
    labels[13:21, 14:22] = 2
    return labels


def _loss_tensors():
    target = torch.zeros(2, 5, 1, 12, 12)
    target[:, 0, :, 2:10, 2:10] = 1.0
    target[:, 4, :, 1:11, 1] = 1.0
    prediction = torch.full_like(target, 0.25)
    return prediction, target


def test_boundary_transform_is_optional_and_uses_all_object_boundaries():
    labels = _labels()
    expected = object_boundaries(labels).astype("float32")

    plain = DirectedPerObjectBoundaryDistanceTransform(apply_label=False)(labels)
    with_boundaries = DirectedPerObjectBoundaryDistanceTransform(
        apply_label=False, with_boundaries=True,
    )(labels)

    assert plain.shape == (4, *labels.shape)
    assert with_boundaries.shape == (5, *labels.shape)
    np.testing.assert_array_equal(with_boundaries[:4], plain)
    np.testing.assert_array_equal(with_boundaries[4], expected)
    # Both isolated objects must contribute, unlike a touching-objects contact target.
    assert with_boundaries[4, 3:12, 3:12].any()
    assert with_boundaries[4, 13:21, 14:22].any()


def test_joint_transform_appends_boundary_after_automatic_targets():
    labels = _labels()
    target = _JointLabelTransform(apply_label=False, with_boundaries=True)(labels)

    assert target.shape == (6, *labels.shape)
    np.testing.assert_array_equal(target[0], labels)
    np.testing.assert_array_equal(target[-1], object_boundaries(labels))


@pytest.mark.parametrize("dice_weight", [0.0, 0.25, 1.0])
def test_boundary_loss_interpolates_dice_and_bce(dice_weight):
    prediction, target = _loss_tensors()
    base = DirectedDistanceLoss()(prediction[:, :4], target[:, :4])
    actual = DirectedDistanceLoss(
        with_boundaries=True, boundary_dice_weight=dice_weight,
    )(prediction, target)

    dice = DiceLoss()(prediction[:, 4:5], target[:, 4:5])
    bce = F.binary_cross_entropy(prediction[:, 4:5].float(), target[:, 4:5].float())
    expected = base + dice_weight * dice + (1.0 - dice_weight) * bce
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dice_weight", [0.0, 0.5, 1.0])
def test_boundary_loss_excludes_ignored_values_and_gradients(dice_weight):
    prediction, target = _loss_tensors()
    target[0, 0, :, :6] = FOREGROUND_IGNORE_VALUE
    target[1, 0, :, :3] = FOREGROUND_IGNORE_VALUE
    ignored = target[:, 0] == FOREGROUND_IGNORE_VALUE
    changed_prediction, changed_target = prediction.clone(), target.clone()
    changed_prediction[:, 4][ignored] = 0.9
    changed_target[:, 4][ignored] = 1.0 - changed_target[:, 4][ignored]
    loss_function = DirectedDistanceLoss(with_boundaries=True, boundary_dice_weight=dice_weight)

    prediction.requires_grad_()
    changed_prediction.requires_grad_()
    loss = loss_function(prediction, target)
    changed_loss = loss_function(changed_prediction, changed_target)
    loss.backward()
    changed_loss.backward()

    torch.testing.assert_close(loss, changed_loss)
    torch.testing.assert_close(prediction.grad, changed_prediction.grad)
    assert torch.count_nonzero(prediction.grad[:, 4][ignored]) == 0
    assert prediction.grad[:, 4][~ignored].abs().sum() > 0


def test_boundary_bce_normalizes_each_sample_by_valid_voxel_count():
    prediction, target = _loss_tensors()
    target[:, 4] = 0.0
    target[0, 0, :, :9] = FOREGROUND_IGNORE_VALUE
    target[1, 0, :, :3] = FOREGROUND_IGNORE_VALUE
    prediction[0, 4] = 0.2
    prediction[1, 4] = 0.8
    base = DirectedDistanceLoss()(prediction[:, :4], target[:, :4])
    loss = DirectedDistanceLoss(with_boundaries=True, boundary_dice_weight=0.0)(prediction, target)

    # Each sample has equal weight despite having different numbers of valid voxels.
    expected_bce = -torch.log(torch.tensor([0.8, 0.2])).mean()
    torch.testing.assert_close(loss - base, expected_bce)


@pytest.mark.parametrize("dice_weight", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("fully_ignored_batch", [False, True])
def test_boundary_loss_handles_fully_ignored_samples(dice_weight, fully_ignored_batch):
    prediction, target = _loss_tensors()
    target[0, 0] = FOREGROUND_IGNORE_VALUE
    if fully_ignored_batch:
        target[1, 0] = FOREGROUND_IGNORE_VALUE
    prediction.requires_grad_()
    loss = DirectedDistanceLoss(with_boundaries=True, boundary_dice_weight=dice_weight)(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[0]) == 0
    if fully_ignored_batch:
        assert torch.count_nonzero(prediction.grad) == 0
    else:
        assert prediction.grad[1, 4].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast regression")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dice_weight", [0.0, 0.5])
def test_boundary_bce_under_cuda_autocast(dtype, dice_weight):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")

    prediction, target = _loss_tensors()
    prediction = prediction.to(device="cuda", dtype=dtype).requires_grad_()
    target = target.cuda()
    loss_function = DirectedDistanceLoss(with_boundaries=True, boundary_dice_weight=dice_weight)

    with torch.autocast(device_type="cuda", dtype=dtype):
        loss = loss_function(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[:, 4].abs().sum() > 0


def test_boundary_loss_rejects_invalid_weight():
    with pytest.raises(ValueError, match="between zero and one"):
        DirectedDistanceLoss(with_boundaries=True, boundary_dice_weight=1.1)


def test_custom_activation_supports_four_and_five_channels():
    activation = CustomActivation()
    for n_channels in (4, 5):
        logits = torch.linspace(-2, 2, n_channels)[None, :, None, None]
        prediction = activation(logits)

        torch.testing.assert_close(prediction[:, :1], logits[:, :1].sigmoid())
        torch.testing.assert_close(prediction[:, 1:4], logits[:, 1:4].tanh())
        if n_channels == 5:
            torch.testing.assert_close(prediction[:, 4:], logits[:, 4:].sigmoid())


def test_boundary_training_target_activation_and_backward():
    target = GeodesicHybridDistanceTransform(
        apply_label=False, with_boundaries=True,
    )(_labels())
    target = torch.from_numpy(target)[None, :, None]
    logits = torch.randn_like(target, requires_grad=True)
    prediction = CustomActivation()(logits)

    loss = DirectedDistanceLoss(
        with_boundaries=True, boundary_dice_weight=0.5,
    )(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert logits.grad[:, 4].abs().sum() > 0
    assert prediction[:, 4].min() >= 0 and prediction[:, 4].max() <= 1
    assert torch.isfinite(DirectedDistanceLoss(
        with_boundaries=True, boundary_dice_weight=0.5,
    )(prediction.bfloat16(), target))


@pytest.mark.parametrize("with_boundaries, expected_channels", [(False, 4), (True, 5)])
def test_train_automatic_wires_boundary_configuration(monkeypatch, with_boundaries, expected_channels):
    from micro_sam.v2.training import training

    captured = {}

    def build_model(*args, **kwargs):
        captured["output_channels"] = kwargs["output_channels"]
        return torch.nn.Identity()

    class Trainer:
        def fit(self, **kwargs):
            captured["fit_kwargs"] = kwargs

    def build_trainer(**kwargs):
        captured["loss"] = kwargs["loss"]
        return Trainer()

    monkeypatch.setattr(training, "get_device", lambda device: torch.device("cpu"))
    monkeypatch.setattr(training, "_build_unisam2_model", build_model)
    with unittest.mock.patch("torch_em.default_segmentation_trainer", side_effect=build_trainer):
        training.train_automatic(
            name="boundary-wiring-test",
            model_type="hvit_t",
            train_loader=object(),
            val_loader=object(),
            n_iterations=1,
            with_boundaries=with_boundaries,
            boundary_dice_weight=0.25,
        )

    assert captured["output_channels"] == expected_channels
    assert captured["loss"].with_boundaries is with_boundaries
    assert captured["loss"].boundary_dice_weight == 0.25
    assert captured["fit_kwargs"]["iterations"] == 1


def test_joint_multi_gpu_forwards_boundary_configuration(monkeypatch):
    from micro_sam.v2.training import training

    captured = {}
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(training, "_train_joint_rank", lambda **kwargs: captured.update(kwargs))

    training.train_joint_sam2_multi_gpu(
        name="boundary-wiring-test",
        model_type="hvit_t",
        input_path="unused",
        with_boundaries=True,
        boundary_dice_weight=0.75,
    )

    assert captured["with_boundaries"] is True
    assert captured["boundary_dice_weight"] == 0.75
