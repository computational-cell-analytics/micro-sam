import os
import types
import random
import socket
import tempfile
import unittest
import contextlib
import unittest.mock

import pytest
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from micro_sam.v2.transforms.raw import VideoAugment
from micro_sam.v2.loss.directed_distance_based import _masked_mse, DirectedDistanceLoss


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resume_worker(rank, world_size, port, checkpoint_path, result_dir):
    """Resume a checkpoint saved by a single-GPU (non-DDP) run into a DDP-wrapped trainer."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from micro_sam.v2.training.sam2_trainer import CheckpointAdapter

        class Base:
            def load_checkpoint(self, checkpoint="best"):
                self.model.load_state_dict(self.saved["model_state"])
                return self.saved

        class Trainer(CheckpointAdapter, Base):
            def __init__(self, model):
                self.model = model

        wrapped = torch.nn.parallel.DistributedDataParallel(nn.Linear(4, 4, bias=False))
        trainer = Trainer(wrapped)
        trainer.saved = torch.load(checkpoint_path, weights_only=False)
        trainer.load_checkpoint()

        torch.save(wrapped.module.weight.detach().clone(), os.path.join(result_dir, f"resumed{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _sync_grads_worker(rank, world_size, port, result_dir):
    """Run JointSam2Trainer._sync_automatic_grads on one rank and dump the synced gradients."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        from micro_sam.v2.training.joint_sam2_trainer import JointSam2Trainer

        unetr = nn.Module()
        unetr.encoder = nn.Linear(2, 2, bias=False)
        unetr.decoder = nn.Linear(2, 2, bias=False)
        # Rank 0 gets gradient 1.0, rank 1 gets 2.0, so a correct all-reduce yields 1.5 on both.
        for module in (unetr.encoder, unetr.decoder):
            module.weight.grad = torch.full((2, 2), float(rank + 1))

        JointSam2Trainer._sync_automatic_grads(types.SimpleNamespace(unetr=unetr))

        torch.save(
            {
                "encoder": unetr.encoder.weight.grad.clone(),
                "decoder": unetr.decoder.weight.grad.clone(),
            },
            os.path.join(result_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class TestPairedVideoAugment(unittest.TestCase):
    """VideoAugment must move the labels with the image.

    Applying a spatial transform to the raw only trains the model to predict a mask where the
    object is not, which is silent, structured label noise.
    """

    def _spatial_only_augment(self, **kwargs):
        # Disable the color path so only the geometry is exercised.
        return VideoAugment(
            brightness=0.0, contrast=0.0, saturation=0.0, p_grayscale=0.0,
            per_frame_brightness=0.0, per_frame_contrast=0.0, per_frame_saturation=0.0,
            **kwargs,
        )

    def test_hflip_moves_image_and_label_together_2d(self):
        augment = self._spatial_only_augment(p_hflip=1.0, degrees=0.0, shear=0.0)

        x = torch.zeros(1, 1, 4, 4)
        y = torch.zeros(1, 1, 4, 4, dtype=torch.int64)
        x[0, 0, :, 0] = 1.0
        y[0, 0, :, 0] = 7

        x_aug, y_aug = augment(x, y)

        # The object must end up in the last column in both.
        self.assertTrue(torch.all(x_aug[0, 0, :, -1] > 0.5))
        self.assertTrue(torch.all(y_aug[0, 0, :, -1] == 7))
        self.assertEqual(int((y_aug > 0).sum()), 4)
        self.assertEqual(y_aug.dtype, torch.int64)

    def test_hflip_moves_image_and_label_together_3d(self):
        augment = self._spatial_only_augment(p_hflip=1.0, degrees=0.0, shear=0.0)

        x = torch.zeros(1, 1, 3, 4, 4)
        y = torch.zeros(1, 1, 3, 4, 4, dtype=torch.int64)
        x[0, 0, :, :, 0] = 1.0
        y[0, 0, :, :, 0] = 3

        x_aug, y_aug = augment(x, y)

        self.assertEqual(x_aug.shape, x.shape)
        self.assertEqual(y_aug.shape, y.shape)
        for z in range(3):
            self.assertTrue(torch.all(x_aug[0, 0, z, :, -1] > 0.5))
            self.assertTrue(torch.all(y_aug[0, 0, z, :, -1] == 3))

    def test_rotation_keeps_image_and_label_registered(self):
        random.seed(0)
        augment = self._spatial_only_augment(p_hflip=0.5, degrees=25.0, shear=20.0)

        # An off-center block makes any un-applied transform obvious.
        x = torch.zeros(1, 1, 32, 32)
        y = torch.zeros(1, 1, 32, 32, dtype=torch.int64)
        x[0, 0, 4:14, 6:16] = 1.0
        y[0, 0, 4:14, 6:16] = 1

        x_aug, y_aug = augment(x, y)

        image_fg = x_aug[0, 0] > 0.5
        label_fg = y_aug[0, 0] > 0
        intersection = (image_fg & label_fg).sum().item()
        union = (image_fg | label_fg).sum().item()
        self.assertGreater(union, 0)
        # Bilinear raw vs nearest label differ slightly at the boundary, so this is not exact.
        self.assertGreater(intersection / union, 0.7)


class TestMaskedMse(unittest.TestCase):
    """The masked distance loss must not be diluted by background.

    Averaging over the whole patch scales the distance gradient by the foreground fraction,
    so supervision for a small object is orders of magnitude weaker than for a large one.
    """

    def test_normalizes_by_mask_size(self):
        prediction = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)
        # A single foreground pixel with a squared error of 4.
        target[0, 0, 0, 0] = 2.0
        mask[0, 0, 0, 0] = 1.0

        loss = _masked_mse(prediction, target, mask)
        self.assertAlmostEqual(loss.item(), 4.0, places=6)

    def test_is_invariant_to_background_size(self):
        def loss_for_size(size):
            prediction = torch.zeros(1, 1, size, size)
            target = torch.zeros(1, 1, size, size)
            mask = torch.zeros(1, 1, size, size)
            target[0, 0, :2, :2] = 3.0
            mask[0, 0, :2, :2] = 1.0
            return _masked_mse(prediction, target, mask).item()

        # The same object in a 4x larger patch must give the same distance loss.
        self.assertAlmostEqual(loss_for_size(8), loss_for_size(16), places=6)

    def test_empty_mask_gives_zero_without_nan(self):
        prediction = torch.ones(1, 1, 4, 4, requires_grad=True)
        target = torch.zeros(1, 1, 4, 4)
        mask = torch.zeros(1, 1, 4, 4)

        loss = _masked_mse(prediction, target, mask)
        self.assertEqual(loss.item(), 0.0)

        loss.backward()
        self.assertFalse(torch.isnan(prediction.grad).any())

    def test_2d_input_has_no_z_distance_gradient(self):
        # For 2d inputs the z channel is masked out entirely, so it must not contribute.
        loss_fn = DirectedDistanceLoss(mask_distances_in_bg=True)
        prediction = torch.rand(1, 4, 1, 8, 8, requires_grad=True)
        target = torch.zeros(1, 4, 1, 8, 8)
        target[:, 0, :, 2:6, 2:6] = 1.0

        loss_fn(prediction, target).backward()
        self.assertAlmostEqual(prediction.grad[:, 1].abs().sum().item(), 0.0, places=6)


class TestConvertToSam2VideoBatch(unittest.TestCase):
    """The vectorized conversion must match the per-frame, per-object loop it replaced.

    The converter feeds SAM2's prompt sampling, so a shifted object slot or a mis-shaped mask
    would silently corrupt every prompt derived from it.
    """

    def _fake_sam2_data_utils(self):
        """Stub SAM2's datapoint types so the converter runs without the tensordict dependency."""
        module = types.ModuleType("training.utils.data_utils")

        class Record(dict):
            def __init__(self, **kwargs):
                super().__init__(kwargs)

        module.BatchedVideoDatapoint = Record
        module.BatchedVideoMetaData = Record
        return {"training.utils.data_utils": module}

    def _convert(self, converter, x, y):
        with unittest.mock.patch.dict("sys.modules", self._fake_sam2_data_utils()):
            return converter(x, y)

    def _reference_masks(self, converter, y, is_3d, T, B, obj_ids_per_b):
        """The original per-(frame, batch item), per-object mask construction."""
        step_masks = []
        for t in range(T):
            masks_t = []
            for b in range(B):
                ids = obj_ids_per_b[b]
                if len(ids) == 0:
                    continue
                lbl = y[b, t] if is_3d else y[b]
                raw = torch.stack([lbl == oid for oid in ids])
                obj_masks = converter._resize_masks(raw)
                for o_i in range(len(ids)):
                    masks_t.append(obj_masks[o_i])
            step_masks.append(torch.stack(masks_t))
        return torch.stack(step_masks)

    def _check(self, x, y, is_3d, T, B):
        from micro_sam.v2.training.util import ConvertToSam2VideoBatch

        converter = ConvertToSam2VideoBatch(max_num_objects=20)
        out = self._convert(converter, x, y)

        size = converter._SAM2_SIZE
        y_squeezed = y.squeeze(1)
        obj_ids_per_b = [
            converter._sample_obj_ids(y_squeezed[b].flatten() if is_3d else y_squeezed[b]) for b in range(B)
        ]
        n_objects = sum(len(ids) for ids in obj_ids_per_b)

        self.assertEqual(tuple(out["img_batch"].shape), (T, B, 3, size, size))
        self.assertEqual(tuple(out["masks"].shape), (T, n_objects, size, size))
        self.assertEqual(out["masks"].dtype, torch.bool)
        self.assertEqual(tuple(out["obj_to_frame_idx"].shape), (T, n_objects, 2))

        # Images: identical to resizing each frame on its own.
        for t in range(T):
            frame = x[:, :, t] if is_3d else x
            expected = converter._to_sam2_image(frame)
            self.assertTrue(torch.allclose(out["img_batch"][t], expected))

        expected_masks = self._reference_masks(converter, y_squeezed, is_3d, T, B, obj_ids_per_b)
        self.assertTrue(torch.equal(out["masks"], expected_masks))

        # Object slots must stay aligned with their batch item and instance ID.
        flat_b = [b for b in range(B) for _ in range(len(obj_ids_per_b[b]))]
        flat_ids = [int(oid) for b in range(B) for oid in obj_ids_per_b[b].tolist()]
        for t in range(T):
            self.assertEqual(out["obj_to_frame_idx"][t][:, 0].tolist(), [t] * n_objects)
            self.assertEqual(out["obj_to_frame_idx"][t][:, 1].tolist(), flat_b)
            identifier = out["metadata"]["unique_objects_identifier"][t]
            self.assertEqual(identifier[:, 0].tolist(), flat_b)
            self.assertEqual(identifier[:, 1].tolist(), flat_ids)
            self.assertEqual(identifier[:, 2].tolist(), [t] * n_objects)

        h, w = (x.shape[-2], x.shape[-1])
        self.assertTrue(torch.equal(
            out["metadata"]["frame_orig_size"], torch.tensor([h, w]).expand(T, n_objects, 2)
        ))

    def test_matches_reference_2d(self):
        torch.manual_seed(0)
        x = torch.rand(2, 1, 24, 32)
        y = torch.zeros(2, 1, 24, 32, dtype=torch.int64)
        y[0, 0, 2:8, 3:9] = 1
        y[0, 0, 12:18, 14:20] = 4
        y[1, 0, 5:11, 5:11] = 2
        self._check(x, y, is_3d=False, T=1, B=2)

    def test_matches_reference_3d(self):
        torch.manual_seed(0)
        x = torch.rand(2, 1, 3, 24, 32)
        y = torch.zeros(2, 1, 3, 24, 32, dtype=torch.int64)
        y[0, 0, :, 2:8, 3:9] = 1
        # An object that is absent on the first frame must still get a slot on every frame.
        y[0, 0, 2, 12:18, 14:20] = 5
        y[1, 0, :2, 5:11, 5:11] = 3
        self._check(x, y, is_3d=True, T=3, B=2)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
    def test_gpu_conversion_matches_cpu(self):
        """Converting on the device must give the same batch as converting on the host.

        The resize dominates the conversion and is much faster on GPU, so _interactive_step
        moves the inputs first. Object sampling is kept below max_num_objects here because
        the CPU and CUDA RNGs draw different permutations for the subsampling path.
        """
        from micro_sam.v2.training.util import ConvertToSam2VideoBatch

        converter = ConvertToSam2VideoBatch(max_num_objects=20)
        torch.manual_seed(0)
        x = torch.rand(2, 1, 3, 24, 32)
        y = torch.zeros(2, 1, 3, 24, 32, dtype=torch.int64)
        y[0, 0, :, 2:8, 3:9] = 1
        y[0, 0, 2, 12:18, 14:20] = 5
        y[1, 0, :2, 5:11, 5:11] = 3

        on_cpu = self._convert(converter, x, y)
        on_gpu = self._convert(converter, x.cuda(), y.cuda())

        self.assertTrue(torch.allclose(on_cpu["img_batch"], on_gpu["img_batch"].cpu(), atol=1e-5))
        self.assertTrue(torch.equal(on_cpu["masks"], on_gpu["masks"].cpu()))
        self.assertTrue(torch.equal(on_cpu["obj_to_frame_idx"], on_gpu["obj_to_frame_idx"].cpu()))
        for key in ("unique_objects_identifier", "frame_orig_size"):
            self.assertTrue(torch.equal(on_cpu["metadata"][key], on_gpu["metadata"][key].cpu()))

    def test_raises_without_objects(self):
        from micro_sam.v2.training.util import ConvertToSam2VideoBatch

        converter = ConvertToSam2VideoBatch(max_num_objects=20)
        x = torch.rand(1, 1, 16, 16)
        y = torch.zeros(1, 1, 16, 16, dtype=torch.int64)
        with self.assertRaisesRegex(RuntimeError, "no objects found"):
            self._convert(converter, x, y)


class TestDdpCheckpointRoundTrip(unittest.TestCase):
    """Checkpoints must transfer between DDP and non-DDP trainers.

    Saving unwraps the DDP wrapper, so the stored keys have no "module." prefix. Loading has
    to unwrap too, otherwise resume matches those keys against the wrapper's prefixed ones
    and fails - which made every DDP resume unusable.
    """

    class FakeBase:
        """Minimal stand-in for DefaultTrainer's checkpoint handling."""

        def save_checkpoint(self, name, current_metric, best_metric, **extra_save_dict):
            # state_dict returns references; clone so this behaves like serializing to disk.
            state = {k: v.clone() for k, v in self.model.state_dict().items()}
            self.saved = {"model_state": state, **extra_save_dict}

        def load_checkpoint(self, checkpoint="best"):
            self.model.load_state_dict(self.saved["model_state"])
            return self.saved

    def _trainer_cls(self):
        from micro_sam.v2.training.sam2_trainer import CheckpointAdapter

        class Trainer(CheckpointAdapter, TestDdpCheckpointRoundTrip.FakeBase):
            def __init__(self, model):
                self.model = model

        return Trainer

    def test_ddp_checkpoint_loads_into_plain_model_and_back(self):
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(_free_port())
        dist.init_process_group("gloo", rank=0, world_size=1)
        try:
            Trainer = self._trainer_cls()

            torch.manual_seed(0)
            plain = nn.Linear(4, 4, bias=False)
            wrapped = torch.nn.parallel.DistributedDataParallel(nn.Linear(4, 4, bias=False))
            with torch.no_grad():
                wrapped.module.weight.copy_(torch.full((4, 4), 3.0))

            ddp_trainer = Trainer(wrapped)
            ddp_trainer.save_checkpoint("best", 0.0, 0.0)

            # Saved keys must be unwrapped so a non-DDP model can consume them.
            self.assertTrue(all(not k.startswith("module.") for k in ddp_trainer.saved["model_state"]))

            # DDP -> non-DDP.
            plain_trainer = Trainer(plain)
            plain_trainer.saved = ddp_trainer.saved
            plain_trainer.load_checkpoint()
            self.assertTrue(torch.allclose(plain.weight, torch.full((4, 4), 3.0)))

            # DDP -> DDP, the case that used to fail.
            with torch.no_grad():
                wrapped.module.weight.zero_()
            ddp_trainer.load_checkpoint()
            self.assertTrue(torch.allclose(wrapped.module.weight, torch.full((4, 4), 3.0)))

            # Single-GPU -> DDP: a checkpoint written by a non-DDP trainer must resume under DDP.
            single_gpu = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                single_gpu.weight.copy_(torch.full((4, 4), 7.0))
            single_trainer = Trainer(single_gpu)
            single_trainer.save_checkpoint("best", 0.0, 0.0)

            with torch.no_grad():
                wrapped.module.weight.zero_()
            ddp_trainer.saved = single_trainer.saved
            ddp_trainer.load_checkpoint()
            self.assertTrue(torch.allclose(wrapped.module.weight, torch.full((4, 4), 7.0)))

            # Without unwrapping, the wrapper rejects the unprefixed keys.
            with self.assertRaises(RuntimeError):
                wrapped.load_state_dict(ddp_trainer.saved["model_state"])
        finally:
            dist.destroy_process_group()

    @pytest.mark.slow
    def test_single_gpu_checkpoint_resumes_on_two_ranks(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as result_dir:
            # A checkpoint as a single-GPU run would write it: no DDP anywhere in sight.
            single_gpu = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                single_gpu.weight.copy_(torch.full((4, 4), 5.0))
            checkpoint_path = os.path.join(result_dir, "single.pt")
            torch.save({"model_state": single_gpu.state_dict()}, checkpoint_path)

            mp.spawn(
                _resume_worker,
                args=(world_size, _free_port(), checkpoint_path, result_dir),
                nprocs=world_size,
                join=True,
            )
            for rank in range(world_size):
                with self.subTest(rank=rank):
                    resumed = torch.load(os.path.join(result_dir, f"resumed{rank}.pt"), weights_only=False)
                    self.assertTrue(torch.allclose(resumed, torch.full((4, 4), 5.0)))


class TestAutomaticValidationRng(unittest.TestCase):
    """Automatic validation must also be deterministic when data loading is synchronous."""

    def test_zero_worker_validation_is_deterministic_and_restores_rng(self):
        from micro_sam.v2.training.sam2_trainer import UniSAM2Trainer

        class RandomValidationDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 6

            def __getitem__(self, index):
                # With num_workers=0 all three draws happen in the trainer process.
                sample = torch.tensor(
                    [random.random(), np.random.random(), torch.rand(()).item()],
                    dtype=torch.float32,
                )
                return sample, torch.zeros_like(sample)

        trainer = types.SimpleNamespace(
            model=nn.Identity(),
            val_loader=torch.utils.data.DataLoader(
                RandomValidationDataset(), batch_size=2, num_workers=0,
            ),
            device=torch.device("cpu"),
            logger=None,
            _iteration=0,
            _forward_and_loss=lambda x, y: (x, x.mean()),
        )

        # Validation must leave the training RNG streams exactly where it found them.
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        first_metric = UniSAM2Trainer._validate_impl(trainer, contextlib.nullcontext)
        actual_next = (random.random(), np.random.random(), torch.rand(()).item())

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        expected_next = (random.random(), np.random.random(), torch.rand(()).item())
        self.assertEqual(actual_next, expected_next)

        # The validation samples and metric must not depend on the surrounding training RNG.
        random.seed(987)
        np.random.seed(987)
        torch.manual_seed(987)
        second_metric = UniSAM2Trainer._validate_impl(trainer, contextlib.nullcontext)
        self.assertEqual(first_metric, second_metric)


class TestInstanceLabels(unittest.TestCase):
    """Labels must survive non-native byte order.

    13 of the EmbedSeg mask TIFFs are stored big-endian, and connected_components rejects
    those dtypes outright, so 3d interactive and joint training crash when one is sampled.
    """

    def test_handles_big_endian_labels(self):
        import numpy as np

        from micro_sam.v2.transforms.labels import _instance_labels

        labels = np.zeros((2, 16, 16), dtype=">u2")
        labels[:, 2:6, 2:6] = 1
        labels[:, 10:14, 10:14] = 2

        out = _instance_labels(labels)
        self.assertEqual(out.dtype, np.dtype("int64"))
        self.assertEqual(out.shape, labels.shape)
        self.assertEqual(len(np.unique(out)) - 1, 2)

    def test_matches_native_byte_order_result(self):
        import numpy as np

        from micro_sam.v2.transforms.labels import _instance_labels

        native = np.zeros((2, 16, 16), dtype="uint16")
        native[:, 2:6, 2:6] = 1
        native[:, 10:14, 10:14] = 2

        np.testing.assert_array_equal(_instance_labels(native.astype(">u2")), _instance_labels(native))


class TestJointDdpGradientSync(unittest.TestCase):
    """The shared image encoder must be all-reduced after the automatic backward.

    The automatic branch bypasses the DDP wrapper, so DDP never reduces the gradients it
    produces. Skipping the encoder leaves every rank applying its own local gradient, which
    makes the ranks diverge permanently from the first automatic optimizer step.
    """

    @pytest.mark.slow
    def test_encoder_and_decoder_grads_are_averaged_across_ranks(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as result_dir:
            mp.spawn(
                _sync_grads_worker,
                args=(world_size, _free_port(), result_dir),
                nprocs=world_size,
                join=True,
            )
            for rank in range(world_size):
                grads = torch.load(os.path.join(result_dir, f"rank{rank}.pt"), weights_only=False)
                with self.subTest(rank=rank, param="encoder"):
                    self.assertTrue(torch.allclose(grads["encoder"], torch.full((2, 2), 1.5)))
                with self.subTest(rank=rank, param="decoder"):
                    self.assertTrue(torch.allclose(grads["decoder"], torch.full((2, 2), 1.5)))


if __name__ == "__main__":
    unittest.main()
