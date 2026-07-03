import os
import tempfile
import unittest

import torch

from micro_sam.v2.util import get_sam2_model


class TestPEFTSam2(unittest.TestCase):
    model_type = "hvit_t"

    def _get_model(self):
        return get_sam2_model(model_type=self.model_type, device="cpu")

    def _run_encoder(self, sam, x=None):
        """Run the (PEFT-wrapped) image encoder and return the primary feature tensor."""
        if x is None:
            x = torch.rand(1, 3, 1024, 1024)
        return sam.image_encoder(x)["vision_features"]

    def _trainable_encoder_params(self, sam):
        return [name for name, p in sam.image_encoder.named_parameters() if p.requires_grad]

    def _check_output(self, sam):
        with torch.no_grad():
            features = self._run_encoder(sam)
        self.assertEqual(tuple(features.shape), (1, 256, 64, 64))

    def _check_no_frozen_gradients(self, sam):
        features = self._run_encoder(sam)
        features.float().pow(2).mean().backward()
        trainable_with_grad = 0
        for _, param in sam.image_encoder.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    trainable_with_grad += 1
            else:
                self.assertIsNone(param.grad)
        # The encoder forward must produce gradients for at least some trainable parameters.
        self.assertGreater(trainable_with_grad, 0)

    def test_lora_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, LoRASurgery

        model = self._get_model()
        # At initialization LoRA must be a no-op, i.e. the encoder output on a given input matches
        # the base model. Reuse the same input for both passes so only the LoRA weights differ.
        x = torch.rand(1, 3, 1024, 1024)
        with torch.no_grad():
            reference = self._run_encoder(model, x).clone()

        sam = PEFT_Sam2(model, rank=2, peft_module=LoRASurgery).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(len(names) > 0)
        self.assertTrue(all("w_a_linear" in n or "w_b_linear" in n for n in names))
        with torch.no_grad():
            self.assertTrue(torch.allclose(self._run_encoder(sam, x), reference, atol=1e-5))
        self._check_no_frozen_gradients(sam)

    def test_lora_with_mlp_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, LoRASurgery

        model = self._get_model()
        sam = PEFT_Sam2(model, rank=2, peft_module=LoRASurgery, update_matrices=["q", "k", "v", "mlp"]).sam
        self._check_output(sam)
        self._check_no_frozen_gradients(sam)

    def test_classical_surgery_late_finetuning_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, ClassicalSurgery

        model = self._get_model()
        last = len(model.image_encoder.trunk.blocks) - 1
        sam = PEFT_Sam2(model, peft_module=ClassicalSurgery, attention_layers_to_update=[last]).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(len(names) > 0)
        self.assertTrue(all(n.startswith(f"trunk.blocks.{last}.") for n in names))
        self._check_no_frozen_gradients(sam)

    def test_attention_layer_peft_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, AttentionSurgery

        model = self._get_model()
        sam = PEFT_Sam2(model, peft_module=AttentionSurgery).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(all(".attn." in n for n in names))
        self._check_no_frozen_gradients(sam)

    def test_norm_layer_peft_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, LayerNormSurgery

        model = self._get_model()
        sam = PEFT_Sam2(model, peft_module=LayerNormSurgery).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(all("norm1" in n or "norm2" in n for n in names))
        self._check_no_frozen_gradients(sam)

    def test_bias_layer_peft_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, BiasSurgery

        model = self._get_model()
        sam = PEFT_Sam2(model, peft_module=BiasSurgery).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(all(n.endswith("bias") for n in names))
        self._check_no_frozen_gradients(sam)

    def test_get_sam2_train_model_lora(self):
        from micro_sam.v2.training.util import get_sam2_train_model
        from micro_sam.v2.models.peft_sam2 import LoRASurgery

        model = get_sam2_train_model(
            model_type=self.model_type, device="cpu", peft_kwargs={"rank": 2, "peft_module": LoRASurgery},
        )
        names = self._trainable_encoder_params(model)
        self.assertTrue(len(names) > 0)
        self.assertTrue(all("w_a_linear" in n or "w_b_linear" in n for n in names))
        # The PEFT config must be recorded on the model so the trainer can persist it.
        self.assertEqual(model.peft_config, {"rank": 2, "peft_module": "LoRASurgery"})

    def test_peft_freeze_conflict_guard(self):
        from micro_sam.v2.training.util import get_sam2_train_model
        from micro_sam.v2.models.peft_sam2 import LoRASurgery

        with self.assertRaises(ValueError):
            get_sam2_train_model(
                model_type=self.model_type, device="cpu", freeze=["image_encoder"],
                peft_kwargs={"rank": 2, "peft_module": LoRASurgery},
            )

    def _make_lora_checkpoint(self, tmpdir, include_config):
        """Build a base model, apply LoRA, randomize the LoRA weights, and save a checkpoint."""
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, LoRASurgery
        from micro_sam.models.peft import serialize_peft_kwargs

        peft_kwargs = {"rank": 2, "peft_module": LoRASurgery}
        source = PEFT_Sam2(self._get_model(), **peft_kwargs).sam
        with torch.no_grad():
            for name, p in source.image_encoder.named_parameters():
                if "w_b_linear" in name:  # zero-initialized; make them non-trivial.
                    p.normal_(0.0, 0.02)

        ckpt = os.path.join(tmpdir, "peft.pt")
        save_dict = {"model": source.state_dict()}
        if include_config:
            save_dict["peft_kwargs"] = serialize_peft_kwargs(peft_kwargs)
        torch.save(save_dict, ckpt)
        return source, ckpt, peft_kwargs

    def test_lora_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source, ckpt, peft_kwargs = self._make_lora_checkpoint(tmpdir, include_config=False)
            loaded = get_sam2_model(
                model_type=self.model_type, device="cpu", checkpoint_path=ckpt, peft_kwargs=peft_kwargs,
            )
            src_sd, load_sd = source.state_dict(), loaded.state_dict()
            lora_keys = [k for k in src_sd if "w_a_linear" in k or "w_b_linear" in k]
            self.assertTrue(len(lora_keys) > 0)
            self.assertTrue(all(torch.equal(src_sd[k], load_sd[k]) for k in lora_keys))
            x = torch.rand(1, 3, 1024, 1024)
            with torch.no_grad():
                out_src = source.image_encoder(x)["vision_features"]
                out_load = loaded.image_encoder(x)["vision_features"]
            self.assertTrue(torch.allclose(out_src, out_load, atol=1e-6))

    def test_lora_load_auto_detect(self):
        # With a saved PEFT config, loading needs no explicit peft_kwargs.
        with tempfile.TemporaryDirectory() as tmpdir:
            source, ckpt, _ = self._make_lora_checkpoint(tmpdir, include_config=True)
            loaded = get_sam2_model(model_type=self.model_type, device="cpu", checkpoint_path=ckpt)
            names = self._trainable_encoder_params(loaded)
            self.assertTrue(any("w_a_linear" in n or "w_b_linear" in n for n in names))
            src_sd, load_sd = source.state_dict(), loaded.state_dict()
            lora_keys = [k for k in src_sd if "w_a_linear" in k or "w_b_linear" in k]
            self.assertTrue(all(torch.equal(src_sd[k], load_sd[k]) for k in lora_keys))


if __name__ == "__main__":
    unittest.main()
