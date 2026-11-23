import os
import unittest
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from micro_sam.util import get_device

try:
    from torch_em.model.hybrid_unetr import HybridUNETR  # noqa
    HAVE_HYBRID_UNETR = True
except ImportError:
    HAVE_HYBRID_UNETR = False

MODEL_TYPE = "hvit_t"


class _RandomTargets(Dataset):
    """Random images with the targets for directed distances with boundaries."""
    def __init__(self, n_z, n_samples=4, size=128):
        self.n_z, self.n_samples, self.size = n_z, n_samples, size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(index)
        shape = (self.n_z, self.size, self.size)
        x = torch.rand((3, *shape), generator=generator)
        foreground = (torch.rand((1, *shape), generator=generator) > 0.5).float()
        distances = torch.rand((3, *shape), generator=generator) * 2 - 1
        boundaries = (torch.rand((1, *shape), generator=generator) > 0.8).float()
        return x, torch.cat([foreground, distances, boundaries], dim=0)


@unittest.skipUnless(HAVE_HYBRID_UNETR, "Needs torch_em.model.hybrid_unetr")
class TestDecoderTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from micro_sam.v2.training import get_decoder_train_model
        cls.device = get_device()
        cls.model = get_decoder_train_model(MODEL_TYPE, device=cls.device)

    def _encoder_state(self):
        return {key: value.detach().clone() for key, value in self.model.encoder.state_dict().items()}

    def test_encoder_is_frozen(self):
        self.model.train()
        self.assertFalse(self.model.encoder.training)
        self.assertTrue(self.model.decoder.training)
        self.assertFalse(any(param.requires_grad for param in self.model.encoder.parameters()))
        self.assertTrue(all(param.requires_grad for param in self.model.decoder.parameters()))

    def test_features_match_sam2(self):
        from micro_sam.v2.util import get_sam2_model

        sam2 = get_sam2_model(MODEL_TYPE, device=self.device, input_type="images")
        image = torch.rand(2, 3, 1024, 1024, device=self.device)
        with torch.no_grad():
            expected = sam2.forward_image(image)["backbone_fpn"]
            _, levels = self.model.encoder(image)
            levels = [self.model.decoder.proj_s4(levels[0]), self.model.decoder.proj_s8(levels[1]), levels[2]]
        # The projections are trained, so this only holds for a model that was not trained yet.
        fresh = self.model.decoder.proj_s4.weight.equal(sam2.sam_mask_decoder.conv_s0.weight)
        self.assertTrue(fresh)
        for level, expected_level in zip(levels, expected):
            self.assertTrue(torch.allclose(level, expected_level, atol=1e-5))

    def test_chunked_encoding(self):
        # Large batches are encoded in chunks, see SAM2MultiScaleEncoder.
        encoder = self.model.encoder
        image = torch.rand(3, 3, 1024, 1024, device=self.device)
        max_batch_size = encoder.max_batch_size
        try:
            with torch.no_grad():
                encoder.max_batch_size = None
                expected = encoder(image)[1]
                encoder.max_batch_size = 2
                levels = encoder(image)[1]
        finally:
            encoder.max_batch_size = max_batch_size
        for level, expected_level in zip(levels, expected):
            self.assertEqual(level.shape, expected_level.shape)
            self.assertTrue(torch.allclose(level, expected_level, atol=1e-5))

    def test_forward(self):
        self.model.eval()
        for shape in [(2, 3, 128, 128), (1, 3, 1, 128, 128), (1, 3, 2, 128, 128)]:
            x = torch.rand(shape, device=self.device)
            with torch.no_grad():
                y = self.model(x)
            self.assertEqual(tuple(y.shape), (shape[0], 5, *shape[2:]))
            y = y.cpu().numpy()
            if len(shape) == 4:
                y = y[:, :, None]
            self.assertTrue(np.all((y[:, 0] >= 0) & (y[:, 0] <= 1)))
            self.assertTrue(np.all((y[:, 1:4] >= -1) & (y[:, 1:4] <= 1)))
            self.assertTrue(np.all((y[:, 4] >= 0) & (y[:, 4] <= 1)))

    def test_encoder_gets_no_gradients(self):
        self.model.train()
        y = self.model(torch.rand(1, 3, 2, 128, 128, device=self.device))
        y.mean().backward()
        self.assertTrue(all(param.grad is None for param in self.model.encoder.parameters()))
        self.assertTrue(any(param.grad is not None for param in self.model.decoder.parameters()))
        self.model.zero_grad(set_to_none=True)

    def test_training_and_checkpoints(self):
        from micro_sam.v2.training import get_decoder_train_model, get_decoder_trainer

        model = get_decoder_train_model(MODEL_TYPE, device=self.device)
        encoder_before = {key: value.detach().clone() for key, value in model.encoder.state_dict().items()}
        decoder_before = {key: value.detach().clone() for key, value in model.decoder.state_dict().items()}

        def get_trainer(model, save_root):
            # The loaders mix 2D batches (one slice) and 3D batches, as the generalist loader does.
            train_loader = DataLoader(_RandomTargets(n_z=2), batch_size=1)
            val_loader = DataLoader(_RandomTargets(n_z=1, n_samples=2), batch_size=2)
            for loader in (train_loader, val_loader):
                loader.shuffle = False
            return get_decoder_trainer(
                "test-decoder", model, train_loader, val_loader, device=self.device, save_root=save_root,
            )

        with TemporaryDirectory() as tmp:
            trainer = get_trainer(model, tmp)
            trainer.fit(epochs=1, overwrite_training=True)

            for key, value in model.encoder.state_dict().items():
                self.assertTrue(torch.equal(value, encoder_before[key]), key)
            changed = [not torch.equal(value, decoder_before[key]) for key, value in model.decoder.state_dict().items()]
            self.assertTrue(any(changed))

            checkpoint_path = os.path.join(tmp, "checkpoints", "test-decoder", "latest.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertTrue(all(key.startswith("decoder.") for key in checkpoint["model_state"]))
            self.assertEqual(set(checkpoint["model_state"]), set(checkpoint["decoder_state"]))
            self.assertEqual(checkpoint["decoder_config"]["decoder"], "hiera_u")
            self.assertEqual(checkpoint["encoder_model_type"], MODEL_TYPE)
            self.assertEqual(checkpoint["iteration"], 4)

            # A new model reproduces the trained one after loading the decoder state.
            x = torch.rand(1, 3, 2, 128, 128, device=self.device)
            model.eval()
            with torch.no_grad():
                expected = model(x)
            new_model = get_decoder_train_model(MODEL_TYPE, device=self.device)
            new_model.load_decoder_state(checkpoint["decoder_state"])
            new_model.eval()
            with torch.no_grad():
                self.assertTrue(torch.allclose(new_model(x), expected, atol=1e-5))

            # Resuming continues from the stored iteration.
            resumed = get_decoder_train_model(MODEL_TYPE, device=self.device)
            trainer = get_trainer(resumed, tmp)
            trainer.fit(epochs=1, overwrite_training=True, load_from_checkpoint="latest")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["iteration"], 8)
            self.assertEqual(checkpoint["epoch"], 1)

    def test_resume_with_changed_parameter_order(self):
        # The optimizer state has to follow the parameters by name if the decoder registers them in another order.
        from micro_sam.v2.training import get_decoder_train_model, get_decoder_trainer

        def get_trainer(save_root):
            model = get_decoder_train_model(MODEL_TYPE, device=self.device)
            train_loader = DataLoader(_RandomTargets(n_z=1, n_samples=2), batch_size=1)
            val_loader = DataLoader(_RandomTargets(n_z=1, n_samples=1), batch_size=1)
            for loader in (train_loader, val_loader):
                loader.shuffle = False
            return get_decoder_trainer(
                "test-order", model, train_loader, val_loader, device=self.device, save_root=save_root,
            )

        with TemporaryDirectory() as tmp:
            trainer = get_trainer(tmp)
            trainer.fit(epochs=1, overwrite_training=True)
            path = os.path.join(tmp, "checkpoints", "test-order", "latest.pt")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            names = checkpoint["optimizer_param_names"]
            self.assertEqual(names, trainer.model.decoder_parameter_names())
            state = checkpoint["optimizer_state"]["state"]
            # Parameters that did not get gradients (the 3D branches for 2D data) have no optimizer state.
            expected = {name: state[i]["exp_avg"] for i, name in enumerate(names) if i in state}
            self.assertGreater(len(expected), 4)

            # Move the first four parameters to the end, as if their modules were registered last.
            order = list(range(4, len(names))) + list(range(4))
            moved_names = [names[i] for i in order]
            self.assertNotEqual(moved_names, names)
            for with_names in (True, False):
                moved = dict(checkpoint)
                moved["optimizer_state"] = dict(checkpoint["optimizer_state"])
                moved["optimizer_state"]["state"] = {
                    new: state[old] for new, old in enumerate(order) if old in state
                }
                if with_names:
                    moved["optimizer_param_names"] = moved_names
                else:  # Older checkpoints: the order is the one of the stored model state.
                    moved.pop("optimizer_param_names")
                    moved["model_state"] = {name: checkpoint["model_state"][name] for name in moved_names}

                trainer = get_trainer(tmp)
                trainer.load_checkpoint(moved)
                parameters = dict(zip(names, trainer.model.decoder_parameters()))
                self.assertEqual(len(trainer.optimizer.state), len(expected))
                for name in expected:
                    param = parameters[name]
                    exp_avg = trainer.optimizer.state[param]["exp_avg"]
                    self.assertEqual(exp_avg.shape, param.shape)
                    self.assertTrue(torch.equal(exp_avg.cpu(), expected[name]), name)

    def test_loss_ignores_instance_channel(self):
        from micro_sam.v2.training.decoder_training import DecoderLoss

        loss = DecoderLoss(with_boundaries=True, boundary_dice_weight=0.5)
        _, target = _RandomTargets(n_z=2, size=32)[0]
        target = target[None]
        prediction = torch.rand(1, 5, 2, 32, 32)
        with_instances = torch.cat([torch.randint(0, 5, (1, 1, 2, 32, 32)).float(), target], dim=1)
        self.assertTrue(torch.equal(loss(prediction, target), loss(prediction, with_instances)))

    def test_load_decoder_state_rejects_encoder_weights(self):
        state = self.model.decoder_state()
        state["encoder.inner.weight"] = torch.zeros(1)
        with self.assertRaises(RuntimeError):
            self.model.load_decoder_state(state)


if __name__ == "__main__":
    unittest.main()
