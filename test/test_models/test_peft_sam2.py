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
        self.assertGreater(trainable_with_grad, 0)

    def test_lora_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, LoRASurgery

        model = self._get_model()
        # LoRA must not change the encoder output before training.
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

    def test_fact_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, FacTSurgery

        model = self._get_model()
        sam = PEFT_Sam2(model, rank=2, peft_module=FacTSurgery).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(len(names) > 0)
        self.assertTrue(all("FacT" in n for n in names))
        self._check_no_frozen_gradients(sam)

    def test_ssf_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, SSFSurgery

        model = self._get_model()
        sam = PEFT_Sam2(model, peft_module=SSFSurgery).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(len(names) > 0)
        self.assertTrue(all("scale" in n or "shift" in n for n in names))
        self._check_no_frozen_gradients(sam)

    def test_adaptformer_sam2(self):
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, AdaptFormer

        model = self._get_model()
        sam = PEFT_Sam2(model, rank=2, peft_module=AdaptFormer, projection_size=64, alpha=2.0, dropout=0.5).sam
        self._check_output(sam)

        names = self._trainable_encoder_params(sam)
        self.assertTrue(len(names) > 0)
        self.assertTrue(all("down_proj" in n or "up_proj" in n or "alpha" in n for n in names))
        self._check_no_frozen_gradients(sam)

    @unittest.skip("Training tests are not run in CI.")
    def test_get_sam2_train_model_lora(self):
        from micro_sam.v2.models.peft_sam2 import LoRASurgery
        from micro_sam.v2.training.util import get_sam2_train_model

        model = get_sam2_train_model(
            model_type=self.model_type, device="cpu", peft_kwargs={"rank": 2, "peft_module": LoRASurgery},
        )
        names = self._trainable_encoder_params(model)
        self.assertTrue(len(names) > 0)
        self.assertTrue(all("w_a_linear" in n or "w_b_linear" in n for n in names))
        self.assertEqual(model.peft_config, {"rank": 2, "peft_module": "LoRASurgery"})

    @unittest.skip("Training tests are not run in CI.")
    def test_peft_freeze_conflict_guard(self):
        from micro_sam.v2.models.peft_sam2 import LoRASurgery
        from micro_sam.v2.training.util import get_sam2_train_model

        with self.assertRaises(ValueError):
            get_sam2_train_model(
                model_type=self.model_type, device="cpu", freeze=["image_encoder"],
                peft_kwargs={"rank": 2, "peft_module": LoRASurgery},
            )

    def _make_lora_checkpoint(self, tmpdir, include_config):
        """Build a LoRA model with random weights and save its checkpoint."""
        from micro_sam.models.peft import serialize_peft_kwargs
        from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, LoRASurgery

        peft_kwargs = {"rank": 2, "peft_module": LoRASurgery}
        source = PEFT_Sam2(self._get_model(), **peft_kwargs).sam
        with torch.no_grad():
            for name, p in source.image_encoder.named_parameters():
                if "w_b_linear" in name:
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
        with tempfile.TemporaryDirectory() as tmpdir:
            source, ckpt, _ = self._make_lora_checkpoint(tmpdir, include_config=True)
            loaded = get_sam2_model(model_type=self.model_type, device="cpu", checkpoint_path=ckpt)
            names = self._trainable_encoder_params(loaded)
            self.assertTrue(any("w_a_linear" in n or "w_b_linear" in n for n in names))
            src_sd, load_sd = source.state_dict(), loaded.state_dict()
            lora_keys = [k for k in src_sd if "w_a_linear" in k or "w_b_linear" in k]
            self.assertTrue(all(torch.equal(src_sd[k], load_sd[k]) for k in lora_keys))


def test_adaptformer_starts_from_the_original_mlp_output():
    from micro_sam.models.peft import AdaptFormer

    class HieraMLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 8), torch.nn.Linear(8, 4)])

        def forward(self, x):
            return self.layers[1](torch.relu(self.layers[0](x)))

    block = torch.nn.Module()
    block.mlp = HieraMLP()
    x = torch.randn(2, 4)
    expected = block.mlp(x)

    AdaptFormer(rank=2, block=block, projection_size=2, alpha=1.0)

    assert torch.allclose(block.mlp(x), expected)


def test_peft_load_restores_eval_mode(tmp_path, monkeypatch):
    from micro_sam.v2 import util as v2_util
    from micro_sam.v2.models.peft_sam2 import PEFT_Sam2, FacTSurgery

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(4, 12)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attention()

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([Block()])

    class ImageEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = Trunk()

    class Sam(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = ImageEncoder()

        def forward(self, x):
            return self.image_encoder.trunk.blocks[0].attn.qkv(x)

    peft_kwargs = {"rank": 2, "peft_module": FacTSurgery}
    source = PEFT_Sam2(Sam().eval(), **peft_kwargs).sam
    checkpoint = tmp_path / "fact.pt"
    torch.save({"model": source.state_dict()}, checkpoint)

    monkeypatch.setattr(v2_util, "_get_checkpoint", lambda **kwargs: "base.pt")
    monkeypatch.setattr(v2_util, "_build_sam2_backbone", lambda *args, **kwargs: Sam().eval())

    loaded = v2_util._load_peft_finetuned_sam2("config", "hvit_t", "images", checkpoint, "cpu", peft_kwargs)
    fact = loaded.image_encoder.trunk.blocks[0].attn.qkv
    assert not loaded.training
    assert not fact.dp_q.training
    assert not fact.dp_v.training

    x = torch.randn(256, 4)
    with torch.no_grad():
        assert torch.equal(loaded(x), loaded(x))


def test_qlora_export_preserves_finetuned_non_encoder_weights(tmp_path, monkeypatch):
    from micro_sam.v2 import util as v2_util

    base_state = {
        "image_encoder.trunk.blocks.0.attn.qkv.weight": torch.tensor([1.0]),
        "sam_prompt_encoder.weight": torch.tensor([2.0]),
        "sam_mask_decoder.weight": torch.tensor([3.0]),
        "memory_attention.weight": torch.tensor([4.0]),
    }
    finetuned_state = {
        "image_encoder.trunk.blocks.0.attn.qkv.qkv_proj.weight": torch.tensor([10.0]),
        "image_encoder.trunk.blocks.0.attn.qkv.qkv_proj.quant_state.bitsandbytes__nf4": torch.tensor([11.0]),
        "image_encoder.trunk.blocks.0.attn.qkv.w_a_linear.weight": torch.tensor([12.0]),
        "image_encoder.trunk.blocks.0.attn.qkv.w_b_linear.weight": torch.tensor([13.0]),
        "sam_prompt_encoder.weight": torch.tensor([20.0]),
        "sam_mask_decoder.weight": torch.tensor([30.0]),
        "memory_attention.weight": torch.tensor([40.0]),
    }

    class BaseModel:
        def state_dict(self):
            return base_state

    monkeypatch.setattr(v2_util, "get_sam2_model", lambda **kwargs: BaseModel())
    finetuned_path = tmp_path / "qlora.pt"
    exported_path = tmp_path / "lora.pt"
    torch.save({"model_state": finetuned_state, "peft_kwargs": {"quantize": True}}, finetuned_path)

    v2_util.export_custom_qlora_sam2_model(None, finetuned_path, "hvit_t", exported_path)

    exported = torch.load(exported_path, weights_only=False)
    exported_state = exported["model_state"]
    assert torch.equal(
        exported_state["image_encoder.trunk.blocks.0.attn.qkv.qkv_proj.weight"], torch.tensor([1.0])
    )
    assert torch.equal(
        exported_state["image_encoder.trunk.blocks.0.attn.qkv.w_a_linear.weight"], torch.tensor([12.0])
    )
    assert torch.equal(exported_state["sam_prompt_encoder.weight"], torch.tensor([20.0]))
    assert torch.equal(exported_state["sam_mask_decoder.weight"], torch.tensor([30.0]))
    assert torch.equal(exported_state["memory_attention.weight"], torch.tensor([40.0]))
    assert all("quant_state" not in key for key in exported_state)
    assert exported["peft_kwargs"] == {"quantize": True}


if __name__ == "__main__":
    unittest.main()
